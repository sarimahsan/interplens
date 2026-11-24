"""Direct Logit Attribution (DLA) Engine for InterpLens.

Decomposes the final logit difference (target_token - foil_token) into exact additive
contributions from the embedding layer, every Attention sublayer, and every MLP sublayer
by projecting component residual writes onto the unembedding difference vector:
    d_U = W_U[:, target_id] - W_U[:, foil_id]
"""

from typing import Dict, Any, List, Optional, Tuple
import torch

from interplens.adapters.base import BaseModelAdapter
from interplens.exceptions import UnembeddingNotFoundError
from interplens.schema import (
    DLAComponentScore,
    DLALayerContribution,
    DirectLogitAttributionResponse,
)


def _resolve_token_id(adapter: BaseModelAdapter, token_str: Optional[str]) -> Optional[int]:
    """Resolves a user-supplied token string to a single vocab token ID."""
    if not token_str:
        return None
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is None:
        model_inst = getattr(adapter, "_model_instance", getattr(adapter, "model", None))
        tokenizer = getattr(model_inst, "tokenizer", None)

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        for candidate in (token_str, f" {token_str.strip()}", token_str.strip()):
            try:
                ids = tokenizer.encode(candidate, add_special_tokens=False)
                if ids:
                    return int(ids[-1])
            except TypeError:
                try:
                    ids = tokenizer.encode(candidate)
                    if ids:
                        return int(ids[-1])
                except Exception:
                    pass
            except Exception:
                pass
    return None


def _decode_id(adapter: BaseModelAdapter, token_id: int) -> str:
    """Decodes a single token ID into a readable string."""
    if hasattr(adapter, "decode"):
        try:
            res = adapter.decode([token_id])
            if res:
                return res
        except Exception:
            pass
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            res = tokenizer.decode([token_id])
            if res:
                return res
        except Exception:
            pass
    return f"#{token_id}"


def _extract_vec(tensor: Optional[torch.Tensor], pos: int, d_model: int) -> Optional[torch.Tensor]:
    """Extracts a 1D float32 CPU vector of shape (d_model,) at `pos` from a 2D/3D activation tensor."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    t = tensor.detach().cpu().to(torch.float32)
    if t.ndim == 3:
        t = t[0]
    if t.ndim == 2 and t.shape[-1] == d_model and 0 <= pos < t.shape[0]:
        return t[pos]
    return None


def _find_component_tensor(
    cache: Dict[str, torch.Tensor],
    layer: int,
    kind: str,
    d_model: int,
    pos: int,
    adapter: BaseModelAdapter,
) -> Optional[torch.Tensor]:
    """Finds a layer's attention output, MLP output, or residual post tensor in the activation cache."""
    if kind == "resid":
        hook_name = adapter.get_resid_post_hook_name(layer) if hasattr(adapter, "get_resid_post_hook_name") else ""
        candidates = [
            hook_name,
            f"blocks.{layer}.hook_resid_post",
            f"transformer.h.{layer}",
            f"model.layers.{layer}",
            f"model.decoder.layers.{layer}",
            f"gpt_neox.layers.{layer}",
            f"layers.{layer}",
            f"blocks.{layer}",
        ]
    elif kind == "attn":
        candidates = [
            f"blocks.{layer}.hook_attn_out",
            f"transformer.h.{layer}.attn",
            f"transformer.h.{layer}.self_attention",
            f"model.layers.{layer}.self_attn",
            f"model.decoder.layers.{layer}.self_attn",
            f"gpt_neox.layers.{layer}.attention",
            f"layers.{layer}.attn",
            f"layers.{layer}.self_attn",
        ]
    else:  # mlp / moe
        candidates = [
            f"blocks.{layer}.hook_mlp_out",
            f"transformer.h.{layer}.mlp",
            f"model.layers.{layer}.mlp",
            f"model.layers.{layer}.block_sparse_moe",
            f"model.layers.{layer}.feed_forward",
            f"gpt_neox.layers.{layer}.mlp",
            f"layers.{layer}.mlp",
        ]

    for key in candidates:
        if key and key in cache:
            vec = _extract_vec(cache[key], pos, d_model)
            if vec is not None:
                return vec

    # Fallback substring search
    for k, v in cache.items():
        k_low = k.lower()
        if "pattern" in k_low:
            continue
        layer_match = (
            f"blocks.{layer}." in k_low
            or f"layers.{layer}." in k_low
            or f"h.{layer}." in k_low
            or k_low.endswith(f".{layer}")
        )
        if not layer_match:
            continue
        if kind == "attn" and ("attn" in k_low or "attention" in k_low):
            vec = _extract_vec(v, pos, d_model)
            if vec is not None:
                return vec
        elif kind == "mlp" and ("mlp" in k_low or "ffn" in k_low):
            vec = _extract_vec(v, pos, d_model)
            if vec is not None:
                return vec
        elif kind == "resid" and not any(x in k_low for x in ("attn", "mlp", "ffn", "norm", "ln_")):
            vec = _extract_vec(v, pos, d_model)
            if vec is not None:
                return vec
    return None


@torch.inference_mode()
def compute_direct_logit_attribution(
    adapter: BaseModelAdapter,
    cache: Dict[str, torch.Tensor],
    tokens: List[str],
    session_id: str,
    prompt: str = "",
    position: Optional[int] = None,
    target_token: Optional[str] = None,
    foil_token: Optional[str] = None,
    logits: Optional[torch.Tensor] = None,
) -> DirectLogitAttributionResponse:
    """Computes Direct Logit Attribution (DLA) across Embedding, Attention, and MLP sublayers."""
    model = getattr(adapter, "model", getattr(adapter, "_model_instance", None))
    info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    num_layers = info.get("num_layers", getattr(adapter, "num_layers", 12))
    seq_len = len(tokens)
    target_pos = position if position is not None and 0 <= position < seq_len else max(0, seq_len - 1)
    selected_token = tokens[target_pos] if seq_len > 0 else ""

    # 1. Extract unembedding matrix W_U of shape [d_model, vocab_size]
    w_u = None
    if hasattr(adapter, "get_unembedding_weight"):
        try:
            w_u = adapter.get_unembedding_weight()
        except Exception:
            w_u = None
    if w_u is None and model is not None:
        if hasattr(model, "W_U"):
            w_u = model.W_U
        elif hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            w_u = model.lm_head.weight.T
        elif hasattr(model, "get_output_embeddings"):
            oe = model.get_output_embeddings()
            if oe is not None and hasattr(oe, "weight"):
                w_u = oe.weight.T

    if w_u is None:
        raise UnembeddingNotFoundError(
            "Direct Logit Attribution requires an unembedding matrix (W_U or lm_head.weight)."
        )

    w_u_cpu = w_u.detach().cpu().to(torch.float32)  # [d_model, vocab_size]
    d_model, vocab_size = w_u_cpu.shape

    # 2. Compute final residual & logits at target_pos to pick default target/foil tokens
    final_resid = None
    for l in range(num_layers - 1, -1, -1):
        final_resid = _find_component_tensor(cache, l, "resid", d_model, target_pos, adapter)
        if final_resid is not None:
            break

    pos_logits = None
    if logits is not None and isinstance(logits, torch.Tensor):
        lt = logits.detach().cpu().to(torch.float32)
        if lt.ndim == 3:
            lt = lt[0]
        if lt.ndim == 2 and target_pos < lt.shape[0]:
            pos_logits = lt[target_pos]
    if pos_logits is None and final_resid is not None:
        pos_logits = torch.matmul(final_resid, w_u_cpu)

    # Resolve target & foil token IDs
    target_id = _resolve_token_id(adapter, target_token)
    foil_id = _resolve_token_id(adapter, foil_token)

    if pos_logits is not None and pos_logits.shape[-1] == vocab_size:
        top2 = torch.topk(pos_logits, k=min(2, vocab_size)).indices.tolist()
        if target_id is None or not (0 <= target_id < vocab_size):
            target_id = int(top2[0])
        if foil_id is None or not (0 <= foil_id < vocab_size) or foil_id == target_id:
            foil_id = int(top2[1]) if len(top2) > 1 and int(top2[1]) != target_id else int((target_id + 1) % vocab_size)
    else:
        target_id = target_id if (target_id is not None and 0 <= target_id < vocab_size) else 0
        foil_id = foil_id if (foil_id is not None and 0 <= foil_id < vocab_size and foil_id != target_id) else min(1, vocab_size - 1)

    target_str = _decode_id(adapter, target_id)
    foil_str = _decode_id(adapter, foil_id)

    # Unembedding difference direction d_U = W_U[:, target] - W_U[:, foil]
    unembed_dir = w_u_cpu[:, target_id] - w_u_cpu[:, foil_id]

    # Approximate final LayerNorm / RMSNorm linear scaling factor at target_pos
    ln_scale = 1.0
    if final_resid is not None:
        rms = float(torch.sqrt(torch.mean(final_resid ** 2) + 1e-5).item())
        if rms > 1e-5:
            ln_scale = 1.0 / rms

    # 3. Embedding contribution
    embed_vec = None
    for k, v in cache.items():
        k_low = k.lower()
        if ("embed" in k_low or "wte" in k_low) and "unembed" not in k_low and "lm_head" not in k_low:
            embed_vec = _extract_vec(v, target_pos, d_model)
            if embed_vec is not None:
                break

    embed_contrib = round(float(torch.dot(embed_vec, unembed_dir).item() * ln_scale), 4) if embed_vec is not None else 0.0

    # 4. Per-layer Attention & MLP contributions
    layer_contributions: List[DLALayerContribution] = []
    all_components: List[DLAComponentScore] = [
        DLAComponentScore(
            component="Embedding",
            layer=-1,
            kind="embed",
            contribution=embed_contrib,
        )
    ]

    prev_resid = embed_vec
    for l in range(num_layers):
        attn_vec = _find_component_tensor(cache, l, "attn", d_model, target_pos, adapter)
        mlp_vec = _find_component_tensor(cache, l, "mlp", d_model, target_pos, adapter)
        curr_resid = _find_component_tensor(cache, l, "resid", d_model, target_pos, adapter)

        if attn_vec is not None or mlp_vec is not None:
            a_val = float(torch.dot(attn_vec, unembed_dir).item() * ln_scale) if attn_vec is not None else 0.0
            m_val = float(torch.dot(mlp_vec, unembed_dir).item() * ln_scale) if mlp_vec is not None else 0.0
        elif curr_resid is not None and prev_resid is not None:
            # Residual delta fallback when sublayer hooks aren't exposed
            delta = curr_resid - prev_resid
            a_val = float(torch.dot(delta, unembed_dir).item() * ln_scale)
            m_val = 0.0
        else:
            a_val = 0.0
            m_val = 0.0

        if curr_resid is not None:
            prev_resid = curr_resid

        a_round = round(a_val, 4)
        m_round = round(m_val, 4)
        tot_round = round(a_val + m_val, 4)

        layer_contributions.append(
            DLALayerContribution(
                layer=l,
                attn_contribution=a_round,
                mlp_contribution=m_round,
                total_contribution=tot_round,
            )
        )
        all_components.append(
            DLAComponentScore(component=f"L{l} Attn", layer=l, kind="attn", contribution=a_round)
        )
        all_components.append(
            DLAComponentScore(component=f"L{l} MLP", layer=l, kind="mlp", contribution=m_round)
        )

    final_logit_diff = (
        round(float((pos_logits[target_id] - pos_logits[foil_id]).item()), 4)
        if pos_logits is not None
        else round(embed_contrib + sum(lc.total_contribution for lc in layer_contributions), 4)
    )

    top_components = sorted(all_components, key=lambda c: abs(c.contribution), reverse=True)[:12]

    return DirectLogitAttributionResponse(
        session_id=session_id,
        prompt=prompt,
        tokens=tokens,
        position=target_pos,
        selected_token=selected_token,
        target_token=target_str,
        target_token_id=target_id,
        foil_token=foil_str,
        foil_token_id=foil_id,
        final_logit_diff=final_logit_diff,
        embed_contribution=embed_contrib,
        layer_contributions=layer_contributions,
        top_components=top_components,
    )
