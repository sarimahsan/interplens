"""Logit Lens Analysis Engine.

Performs vectorized unembedding projections across intermediate Transformer residual stream
layers to uncover internal model token predictions layer-by-layer.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F

from interplens.adapters.base import BaseModelAdapter
from interplens.schema import (
    LogitLensToken,
    LogitLensLayerResult,
    LogitLensResponse,
    PositionLogitLensData,
    LogitLensMatrixResponse,
)


def compute_logit_lens(
    adapter: BaseModelAdapter,
    cache: Dict[str, torch.Tensor],
    tokens: List[str],
    session_id: str = "default_session",
    prompt: str = "",
    top_k: int = 5,
    apply_ln: bool = True,
    position: Optional[int] = None,
) -> LogitLensMatrixResponse:
    """Computes Logit Lens predictions across positions and layers.

    Args:
        adapter: BaseModelAdapter instance.
        cache: Dictionary mapping hook names to activation tensors (batch, pos, d_model).
        tokens: List of string tokens for the prompt.
        session_id: Session identifier.
        prompt: Raw prompt text.
        top_k: Number of top token predictions to retrieve per layer.
        apply_ln: Whether to apply final LayerNorm before unembedding (if available).
        position: Optional single position filter.

    Returns:
        LogitLensMatrixResponse containing predictions for all positions and layers.
    """
    model = getattr(adapter, "model", getattr(adapter, "_model_instance", None))
    device = adapter.device
    num_layers = getattr(adapter, "num_layers", 0)
    if not num_layers and hasattr(adapter, "get_model_info"):
        num_layers = adapter.get_model_info().get("num_layers", 0)

    # Identify unembedding weights W_U and final LayerNorm
    w_u = None
    if hasattr(adapter, "get_unembedding_weight"):
        try:
            w_u = adapter.get_unembedding_weight()
        except Exception:
            w_u = None

    if w_u is None and model is not None:
        if hasattr(model, "W_U"):
            w_u = model.W_U
        elif hasattr(model, "lm_head"):
            w_u = model.lm_head.weight.T
        elif hasattr(model, "get_output_embeddings"):
            embeds = model.get_output_embeddings()
            if embeds is not None and hasattr(embeds, "weight"):
                w_u = embeds.weight.T

    ln_final = None
    if apply_ln and model is not None:
        if hasattr(model, "ln_final"):
            ln_final = model.ln_final
        elif hasattr(model, "final_layernorm"):
            ln_final = model.final_layernorm
        elif hasattr(model, "norm"):
            ln_final = model.norm

    # Collect residual stream tensors per layer
    residual_tensors = []
    layer_names = []

    # Check for embedding hook
    embed_hook = None
    for name in cache.keys():
        if "embed" in name.lower() and "hook" in name.lower():
            embed_hook = name
            break

    if embed_hook and embed_hook in cache:
        residual_tensors.append(cache[embed_hook])
        layer_names.append("Embed")

    # Collect block residual stream hooks
    for layer in range(num_layers):
        hook_name = adapter.get_resid_post_hook_name(layer)
        if hook_name in cache:
            residual_tensors.append(cache[hook_name])
            layer_names.append(f"L{layer}")
        else:
            found = False
            for k, v in cache.items():
                if f"blocks.{layer}" in k or f"layers.{layer}" in k or f"block_{layer}" in k or f"block.{layer}" in k:
                    residual_tensors.append(v)
                    layer_names.append(f"L{layer}")
                    found = True
                    break
            if not found and len(residual_tensors) > 0:
                residual_tensors.append(residual_tensors[-1])
                layer_names.append(f"L{layer}")

    if not residual_tensors:
        raise ValueError("No residual stream tensors found in activation cache for Logit Lens.")

    # Convert to single batch tensor if 3D (1, pos, d_model)
    processed_tensors = []
    for t in residual_tensors:
        if t.ndim == 3:
            t = t[0]  # (pos, d_model)
        processed_tensors.append(t)

    # Stack: (num_layers, pos, d_model)
    stacked_resid = torch.stack(processed_tensors, dim=0)

    @torch.inference_mode()
    def _project_unembed(resid_stack: torch.Tensor) -> torch.Tensor:
        # Apply final LayerNorm if requested and available
        if ln_final is not None and apply_ln:
            try:
                normed_resid = ln_final(resid_stack)
            except Exception:
                normed_resid = resid_stack
        else:
            normed_resid = resid_stack

        # Compute logits: (num_layers, pos, d_model) x (d_model, d_vocab) -> (num_layers, pos, d_vocab)
        if w_u is not None:
            w_u_dev = w_u.to(device=normed_resid.device, dtype=normed_resid.dtype)
            logits = torch.matmul(normed_resid, w_u_dev)
        elif hasattr(model, "unembed"):
            logits = model.unembed(normed_resid)
        else:
            raise RuntimeError("Model does not expose unembedding weights W_U or lm_head.")

        return logits

    logits = _project_unembed(stacked_resid)  # (L_count, P, V)
    probs = F.softmax(logits, dim=-1)         # (L_count, P, V)

    # Extract top-K per position and layer
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    top_logits, _ = torch.topk(logits, k=top_k, dim=-1)

    num_positions = len(tokens)
    positions_data: List[PositionLogitLensData] = []

    target_positions = range(num_positions) if position is None else [position]

    for p in target_positions:
        if p >= num_positions:
            continue

        token_str = tokens[p]
        layer_results: List[LogitLensLayerResult] = []

        for idx, l_name in enumerate(layer_names):
            top_tokens_list: List[LogitLensToken] = []

            for k_idx in range(top_k):
                t_id = int(top_indices[idx, p, k_idx].item())
                prob_val = float(top_probs[idx, p, k_idx].item())
                logit_val = float(top_logits[idx, p, k_idx].item())
                
                # Decode token id to string string
                t_str = adapter.decode([t_id]) if hasattr(adapter, "decode") else str(t_id)

                top_tokens_list.append(
                    LogitLensToken(
                        token=t_str,
                        token_id=t_id,
                        probability=prob_val,
                        logit=logit_val,
                        rank=k_idx + 1,
                    )
                )

            layer_results.append(
                LogitLensLayerResult(
                    layer=idx,
                    top_tokens=top_tokens_list,
                )
            )

        positions_data.append(
            PositionLogitLensData(
                position=p,
                token=token_str,
                layers=layer_results,
            )
        )

    return LogitLensMatrixResponse(
        session_id=session_id,
        prompt=prompt,
        tokens=tokens,
        num_layers=len(layer_names),
        positions=positions_data,
    )
