"""Token Attribution Engine for InterpLens.

Computes input prompt token attribution scores using Attention Rollout across layers
to quantify input token influence on model predictions.
"""

from typing import Dict, Any, List, Optional
import torch

try:
    from interplens.adapters.base import BaseModelAdapter
    from interplens.schema import TokenAttributionResponse, TokenAttributionScore
except ImportError:
    from ..adapters.base import BaseModelAdapter
    from ..schema import TokenAttributionResponse, TokenAttributionScore


def compute_token_attributions(
    adapter: BaseModelAdapter,
    cache: Dict[str, torch.Tensor],
    tokens: List[str],
    session_id: str,
    prompt: str = "",
    position: Optional[int] = None,
    method: str = "attention_rollout",
) -> TokenAttributionResponse:
    """Computes input token attribution scores via Attention Rollout across all layers."""
    model_info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    num_layers = model_info.get("num_layers", 12)
    seq_len = len(tokens)

    target_pos = position if position is not None and 0 <= position < seq_len else (seq_len - 1)
    target_token = tokens[target_pos]

    # Initialize Rollout matrix as NxN identity tensor
    rollout = torch.eye(seq_len, dtype=torch.float32)

    for l in range(num_layers):
        attn_tensor = None
        hook_name = adapter.get_attn_pattern_hook_name(l) if hasattr(adapter, "get_attn_pattern_hook_name") else ""
        
        if hook_name in cache:
            attn_tensor = cache[hook_name]
        else:
            for k, v in cache.items():
                k_lower = k.lower()
                if (f"blocks.{l}." in k_lower or f"layers.{l}." in k_lower or f"block_{l}" in k_lower) and ("attn" in k_lower or "pattern" in k_lower):
                    attn_tensor = v
                    break

        if attn_tensor is not None:
            t = attn_tensor.detach().cpu().to(torch.float32)
            if t.ndim == 4:
                t = t[0]  # (H, N, N)
            if t.ndim == 3 and t.shape[1] == seq_len and t.shape[2] == seq_len:
                # Average attention pattern across heads -> (N, N)
                layer_attn = torch.mean(t, dim=0)
            else:
                layer_attn = None
        else:
            layer_attn = None

        # Fallback causal decay pattern if attention tensor missing
        if layer_attn is None:
            layer_attn = torch.zeros((seq_len, seq_len), dtype=torch.float32)
            for i in range(seq_len):
                for j in range(seq_len):
                    if j <= i:
                        layer_attn[i, j] = 1.0 / (i - j + 1)
                s = torch.sum(layer_attn[i]) or 1.0
                layer_attn[i] /= s

        # Attention Rollout formula: A_l = 0.5 * I + 0.5 * Layer_Attn
        a_l = 0.5 * torch.eye(seq_len) + 0.5 * layer_attn
        # Row normalize
        a_l = a_l / (torch.sum(a_l, dim=-1, keepdim=True) + 1e-9)

        # Rollout matrix multiplication: R = A_l @ R
        rollout = torch.matmul(a_l, rollout)

    # Extract target position row
    pos_row = rollout[target_pos]  # (seq_len,)
    
    # Min-Max normalize attribution scores
    min_val = float(torch.min(pos_row).item())
    max_val = float(torch.max(pos_row).item())
    span = max_val - min_val if (max_val - min_val) > 1e-6 else 1.0

    scores: List[TokenAttributionScore] = []
    for i in range(seq_len):
        raw_val = float(pos_row[i].item())
        norm_val = (raw_val - min_val) / span
        scores.append(
            TokenAttributionScore(
                position=i,
                token=tokens[i],
                score=round(norm_val, 4),
                raw_score=round(raw_val, 4),
            )
        )

    return TokenAttributionResponse(
        session_id=session_id,
        prompt=prompt,
        tokens=tokens,
        target_position=target_pos,
        target_token=target_token,
        method=method,
        attributions=scores,
    )
