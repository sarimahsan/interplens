"""Residual Stream vector analysis and activation steering engine for InterpLens."""

import math
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F

try:
    from ..adapters.base import BaseModelAdapter
    from ..schema import ModelInfo
except ImportError:
    from interplens.adapters.base import BaseModelAdapter
    from interplens.schema import ModelInfo


def compute_residual_metrics(
    adapter: BaseModelAdapter,
    cache: Dict[str, torch.Tensor],
    tokens: List[str],
    session_id: str,
    position: Optional[int] = None,
) -> Dict[str, Any]:
    """Computes L2 norms, layer-by-layer cosine similarity matrices, and drift metrics across residual streams."""
    model_info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    num_layers = model_info.get("num_layers", 12)

    # Collect residual stream tensors per layer
    residual_tensors: List[torch.Tensor] = []
    layer_labels: List[str] = []

    # Check for embedding hook
    embed_hook = None
    for name in cache.keys():
        if "embed" in name.lower() and "hook" in name.lower():
            embed_hook = name
            break

    if embed_hook and embed_hook in cache:
        residual_tensors.append(cache[embed_hook])
        layer_labels.append("Embedding")

    # Collect layer residual streams
    for l in range(num_layers):
        hook_name = adapter.get_resid_post_hook_name(l)
        if hook_name in cache:
            residual_tensors.append(cache[hook_name])
            layer_labels.append(f"L{l}")
        else:
            for k, v in cache.items():
                if f"blocks.{l}" in k or f"layers.{l}" in k or f"block_{l}" in k:
                    residual_tensors.append(v)
                    layer_labels.append(f"L{l}")
                    break

    if not residual_tensors:
        return {"session_id": session_id, "tokens": tokens, "layers": [], "error": "No residual stream tensors in cache."}

    # Format tensors to (num_layers, num_pos, d_model)
    processed = []
    for t in residual_tensors:
        if t.ndim == 3:
            t = t[0]
        processed.append(t.to(torch.float32))

    stacked = torch.stack(processed, dim=0)  # (L_count, P, D)
    L_count, P_count, D_dim = stacked.shape

    # 1. Compute L2 Vector Norms: ||x_{l, p}||
    norms_tensor = torch.norm(stacked, p=2, dim=-1)  # (L_count, P)

    # 2. Compute Cosine Similarity between consecutive layers: cos(x_{l, p}, x_{l+1, p})
    # Normalize vectors
    normed_stacked = F.normalize(stacked, p=2, dim=-1)  # (L_count, P, D)
    
    layer_cosine_sim = []
    if L_count > 1:
        cos_steps = torch.sum(normed_stacked[1:] * normed_stacked[:-1], dim=-1)  # (L_count-1, P)
        for l_idx in range(L_count - 1):
            layer_cosine_sim.append({
                "from_layer": layer_labels[l_idx],
                "to_layer": layer_labels[l_idx + 1],
                "similarities": [round(float(cos_steps[l_idx, p].item()), 4) for p in range(P_count)],
            })

    # 3. Position inspection metrics
    target_pos = position if position is not None and 0 <= position < P_count else (P_count - 1)
    
    pos_vector_history = []
    for l_idx in range(L_count):
        l_norm = round(float(norms_tensor[l_idx, target_pos].item()), 2)
        pos_vector_history.append({
            "layer_index": l_idx,
            "layer_name": layer_labels[l_idx],
            "norm": l_norm,
        })

    # 4. Full Layer-by-Layer Cosine Similarity Matrix for selected position
    pos_vecs = normed_stacked[:, target_pos, :]  # (L_count, D)
    cos_matrix_tensor = torch.matmul(pos_vecs, pos_vecs.T)  # (L_count, L_count)
    
    cos_matrix = []
    for r in range(L_count):
        row = [round(float(cos_matrix_tensor[r, c].item()), 4) for c in range(L_count)]
        cos_matrix.append(row)

    return {
        "session_id": session_id,
        "tokens": tokens,
        "selected_position": target_pos,
        "selected_token": tokens[target_pos],
        "layer_labels": layer_labels,
        "vector_norms": [
            {
                "layer": layer_labels[l_idx],
                "norms": [round(float(norms_tensor[l_idx, p].item()), 2) for p in range(P_count)],
            }
            for l_idx in range(L_count)
        ],
        "layer_cosine_transitions": layer_cosine_sim,
        "position_history": pos_vector_history,
        "cosine_matrix": cos_matrix,
    }


def apply_activation_steering(
    adapter: BaseModelAdapter,
    prompt: str,
    target_layer: int,
    steering_vector: List[float],
    multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Applies activation steering to a target layer residual stream during forward pass."""
    # Ensure vector matches d_model
    model_info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    hidden_dim = model_info.get("hidden_dim", 768)

    steer_tensor = torch.tensor(steering_vector, dtype=torch.float32)
    if steer_tensor.shape[0] != hidden_dim:
        # Pad or truncate to hidden_dim
        if steer_tensor.shape[0] < hidden_dim:
            steer_tensor = F.pad(steer_tensor, (0, hidden_dim - steer_tensor.shape[0]))
        else:
            steer_tensor = steer_tensor[:hidden_dim]

    steer_delta = (steer_tensor * multiplier).to(device=adapter.device)

    # Hook function to inject delta vector into target layer
    def steering_hook(tensor: torch.Tensor, hook: Any = None) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor + steer_delta.view(1, 1, -1)
        elif tensor.ndim == 2:
            return tensor + steer_delta.view(1, -1)
        return tensor + steer_delta

    hook_name = adapter.get_resid_post_hook_name(target_layer)
    
    # Run forward pass with hook
    try:
        out_logits, cache = adapter.run_with_cache(prompt, fwd_hooks=[(hook_name, steering_hook)])
        tokens = adapter.to_str_tokens(prompt) if hasattr(adapter, "to_str_tokens") else [str(i) for i in range(out_logits.shape[1])]
        
        # Extract top predicted token under steering
        last_logits = out_logits[0, -1] if out_logits.ndim == 3 else out_logits[-1]
        probs = F.softmax(last_logits, dim=-1)
        top_prob, top_id = torch.max(probs, dim=-1)
        top_tok = adapter.decode([int(top_id.item())]) if hasattr(adapter, "decode") else str(top_id.item())

        return {
            "prompt": prompt,
            "target_layer": target_layer,
            "multiplier": multiplier,
            "top_steered_token": top_tok,
            "top_steered_prob": round(float(top_prob.item()) * 100, 2),
            "status": "success",
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "target_layer": target_layer,
            "multiplier": multiplier,
            "status": "error",
            "error": str(e),
        }
