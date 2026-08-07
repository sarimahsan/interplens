"""Logit Lens Analysis Engine.

Performs vectorized unembedding projections across intermediate Transformer residual stream
layers to uncover internal model token predictions layer-by-layer.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F

try:
    from interplens.exceptions import UnembeddingNotFoundError
    from interplens.adapters.base import BaseModelAdapter
    from interplens.schema import (
        LogitLensToken,
        LogitLensLayerResult,
        LogitLensResponse,
        PositionLogitLensData,
        LogitLensMatrixResponse,
    )
except ImportError:
    from exceptions import UnembeddingNotFoundError
    from adapters.base import BaseModelAdapter
    from schema import (
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
        except UnembeddingNotFoundError:
            w_u = None
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
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            ln_final = model.model.norm
        elif hasattr(model, "ln_final"):
            ln_final = model.ln_final
        elif hasattr(model, "final_layernorm"):
            ln_final = model.final_layernorm
        elif hasattr(model, "norm"):
            ln_final = model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            ln_final = model.transformer.ln_f

    # Collect residual stream tensors per layer
    residual_tensors = []
    layer_names = []

    # Check for embedding hook
    embed_hook = None
    for name in cache.keys():
        n_lower = name.lower()
        if ("embed" in n_lower or "wte" in n_lower) and "unembed" not in n_lower and "lm_head" not in n_lower:
            embed_hook = name
            break

    if embed_hook and embed_hook in cache:
        residual_tensors.append(cache[embed_hook])
        layer_names.append("Embedding Stream")

    # Collect block residual stream hooks
    for layer in range(num_layers):
        hook_name = adapter.get_resid_post_hook_name(layer)
        found_tensor = None

        if hook_name in cache and cache[hook_name].ndim in (2, 3):
            found_tensor = cache[hook_name]
        else:
            for k, v in cache.items():
                k_lower = k.lower()
                # Exclude attention maps, MLP post-activations, unembedding layers, and 4D tensors
                if "pattern" in k_lower or "attn" in k_lower or "mlp" in k_lower or "unembed" in k_lower or "lm_head" in k_lower or v.ndim not in (2, 3):
                    continue
                if f"blocks.{layer}" in k_lower or f"layers.{layer}" in k_lower or f"block_{layer}" in k_lower or f"h.{layer}" in k_lower:
                    found_tensor = v
                    break

        if found_tensor is not None:
            residual_tensors.append(found_tensor)
            layer_names.append(f"Resid L{layer}")
        elif len(residual_tensors) > 0:
            # Avoid duplicate predictions: use zero tensor matching shape of active residual stream
            ref_tensor = residual_tensors[-1]
            zero_tensor = torch.zeros_like(ref_tensor)
            residual_tensors.append(zero_tensor)
            layer_names.append(f"Resid L{layer} (Missing)")

    if not residual_tensors:
        raise ValueError("No residual stream tensors found in activation cache for Logit Lens.")

    # Convert to single batch tensor if 3D (1, pos, d_model)
    processed_tensors = []
    for t in residual_tensors:
        if t.ndim == 3:
            t = t[0]  # (pos, d_model)
        if t.ndim == 2:
            processed_tensors.append(t)

    # Stack: (num_layers, pos, d_model)
    stacked_resid = torch.stack(processed_tensors, dim=0)

    import logging as _logging
    _ll_logger = _logging.getLogger(__name__)

    @torch.inference_mode()
    def _project_unembed(resid_stack: torch.Tensor) -> torch.Tensor:
        """Projects residual stream activations through final LayerNorm + unembedding to get logits.
        
        Handles device/dtype alignment between hook-captured tensors and model modules.
        Falls back to per-layer processing if batched stacking through norm/lm_head fails.
        """
        # Determine target device and dtype from model parameters
        target_device = resid_stack.device
        target_dtype = resid_stack.dtype
        if ln_final is not None:
            ln_params = list(ln_final.parameters())
            if ln_params:
                target_device = ln_params[0].device
                target_dtype = ln_params[0].dtype
        elif model is not None:
            try:
                p = next(model.parameters())
                target_device = p.device
                target_dtype = p.dtype
            except StopIteration:
                pass

        # Move stacked residuals to model's device and dtype
        resid_aligned = resid_stack.to(device=target_device, dtype=target_dtype)

        # Step 1: Apply final LayerNorm/RMSNorm
        normed_resid = resid_aligned
        if ln_final is not None and apply_ln:
            try:
                # Try batched application: RMSNorm handles arbitrary leading dims
                normed_resid = ln_final(resid_aligned)
            except Exception as e_batch:
                _ll_logger.warning(f"Batched LayerNorm failed ({e_batch}), processing layers individually...")
                # Process each layer individually through the norm
                normed_layers = []
                for layer_idx in range(resid_aligned.shape[0]):
                    single_layer = resid_aligned[layer_idx].unsqueeze(0)  # (1, pos, d_model)
                    try:
                        normed_layers.append(ln_final(single_layer).squeeze(0))
                    except Exception as e_single:
                        _ll_logger.warning(f"LayerNorm failed for layer {layer_idx}: {e_single}")
                        normed_layers.append(resid_aligned[layer_idx])
                normed_resid = torch.stack(normed_layers, dim=0)

        # Step 2: Project through unembedding (lm_head or W_U)
        if hasattr(model, "lm_head") and callable(getattr(model, "lm_head")):
            try:
                logits = model.lm_head(normed_resid)
            except Exception as e_lm:
                _ll_logger.warning(f"model.lm_head() failed ({e_lm}), falling back to W_U matmul...")
                if w_u is not None:
                    w_u_dev = w_u.to(device=normed_resid.device, dtype=normed_resid.dtype)
                    logits = torch.matmul(normed_resid, w_u_dev)
                else:
                    raise UnembeddingNotFoundError("Unembedding failed.")
        elif w_u is not None:
            w_u_dev = w_u.to(device=normed_resid.device, dtype=normed_resid.dtype)
            logits = torch.matmul(normed_resid, w_u_dev)
        elif hasattr(model, "unembed"):
            logits = model.unembed(normed_resid)
        else:
            raise UnembeddingNotFoundError(
                "Model does not expose unembedding weights W_U or lm_head required for Logit Lens analysis."
            )

        return logits

    logits = _project_unembed(stacked_resid)  # (L_count, P, V)
    probs = F.softmax(logits.float(), dim=-1)  # Cast to float32 for softmax stability

    # Cast to float32 for numerical stability (prevents float16 log2(0) NaN/null underflow)
    probs_f32 = probs.to(torch.float32)
    entropy_tensor = -torch.sum(probs_f32 * torch.log2(probs_f32 + 1e-7), dim=-1)  # (L_count, P)

    # Compute KL divergence KL(P_l || P_{l-1}) between consecutive layers
    kl_tensor = torch.zeros((probs.shape[0], probs.shape[1]), device=probs.device, dtype=torch.float32)
    if probs.shape[0] > 1:
        log_p_f32 = torch.log2(probs_f32 + 1e-7)
        kl_steps = torch.sum(probs_f32[1:] * (log_p_f32[1:] - log_p_f32[:-1]), dim=-1) # (L_count-1, P)
        kl_tensor[1:] = torch.clamp(kl_steps, min=0.0)

    # Extract top-K per position and layer
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    top_logits, _ = torch.topk(logits, k=top_k, dim=-1)

    # Offload tensors to CPU numpy arrays once to eliminate CUDA synchronization overhead inside loops
    probs_cpu = probs.detach().cpu().numpy()
    logits_cpu = logits.detach().cpu().numpy()
    top_probs_cpu = top_probs.detach().cpu().numpy()
    top_indices_cpu = top_indices.detach().cpu().numpy()
    top_logits_cpu = top_logits.detach().cpu().numpy()
    entropy_cpu = entropy_tensor.detach().cpu().numpy()
    kl_cpu = kl_tensor.detach().cpu().numpy()

    # Pre-collect unique token IDs and batch decode / map to strings
    unique_token_ids = set(top_indices_cpu.flatten().tolist())
    token_str_map: Dict[int, str] = {}
    if hasattr(adapter, "decode"):
        for uid in unique_token_ids:
            try:
                token_str_map[uid] = adapter.decode([uid])
            except Exception:
                token_str_map[uid] = str(uid)
    else:
        for uid in unique_token_ids:
            token_str_map[uid] = str(uid)

    num_positions = len(tokens)
    positions_data: List[PositionLogitLensData] = []
    target_positions = range(num_positions) if position is None else [position]

    for p in target_positions:
        if p >= num_positions:
            continue

        token_str = tokens[p]
        layer_results: List[LogitLensLayerResult] = []

        # Target token for rank tracking across layers is top prediction at final layer
        target_token_id = int(top_indices_cpu[-1, p, 0])

        # Vectorized rank calculation across all layers for position p (no CUDA sync loops)
        target_logits_p = logits_cpu[:, p, target_token_id][:, None]
        target_ranks_p = (logits_cpu[:, p] > target_logits_p).sum(axis=-1) + 1
        target_ranks = target_ranks_p.tolist()

        # Collect top-5 candidate token IDs at final layer for competition ribbon tracking
        final_top5_ids = [int(top_indices_cpu[-1, p, k]) for k in range(min(5, top_k))]
        top5_trajectories_dict: Dict[str, List[float]] = {}
        for candidate_id in final_top5_ids:
            cand_str = token_str_map.get(candidate_id, str(candidate_id))
            top5_trajectories_dict[cand_str] = (probs_cpu[:, p, candidate_id] * 100.0).tolist()

        for idx, l_name in enumerate(layer_names):
            top_tokens_list: List[LogitLensToken] = []

            layer_entropy = round(float(entropy_cpu[idx, p]), 3)
            layer_kl = round(float(kl_cpu[idx, p]), 3)

            for k_idx in range(top_k):
                t_id = int(top_indices_cpu[idx, p, k_idx])
                prob_val = float(top_probs_cpu[idx, p, k_idx])
                logit_val = float(top_logits_cpu[idx, p, k_idx])
                t_str = token_str_map.get(t_id, str(t_id))

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
                    entropy=layer_entropy,
                    kl_divergence=layer_kl,
                    top_tokens=top_tokens_list,
                )
            )

        positions_data.append(
            PositionLogitLensData(
                position=p,
                token=token_str,
                layers=layer_results,
                target_token_ranks=target_ranks,
                top5_trajectories=top5_trajectories_dict,
            )
        )

    return LogitLensMatrixResponse(
        session_id=session_id,
        prompt=prompt,
        tokens=tokens,
        num_layers=len(layer_names),
        positions=positions_data,
    )
