"""Induction Head Auto-Detector Engine for InterpLens.

Automatically constructs repeated token sequences (S_1 S_2), runs forward pass activation sweeps,
and computes induction scores across all (layer, head) combinations to identify and rank active induction heads.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F

try:
    from interplens.adapters.base import BaseModelAdapter
    from interplens.schema import InductionHeadScore, InductionDetectorResponse
except ImportError:
    from ..adapters.base import BaseModelAdapter
    from ..schema import InductionHeadScore, InductionDetectorResponse

# Standard random word pool for constructing clean repeated sequences S_1 S_2
DEFAULT_WORD_POOL = [
    "apple", "banana", "cherry", "dragon", "elephant", "falcon", "guitar", "harbor",
    "island", "jungle", "kingdom", "lemon", "mountain", "needle", "ocean", "python",
    "queen", "robot", "silver", "tiger", "umbrella", "volcano", "wizard", "xenon"
]


def detect_induction_heads(
    adapter: BaseModelAdapter,
    sequence_length: int = 20,
    threshold: float = 0.15,
    top_k: int = 10,
    custom_words: Optional[List[str]] = None,
) -> InductionDetectorResponse:
    """Runs repeated sequence test (S_1 S_2) to auto-detect and rank induction heads."""
    model_info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    num_layers = model_info.get("num_layers", 12)
    num_heads = model_info.get("num_heads", 12)

    # 1. Prepare repeated word sequence S_1 S_2
    words = custom_words if custom_words and len(custom_words) >= 10 else DEFAULT_WORD_POOL
    n_words = min(sequence_length, len(words))
    sub_seq = words[:n_words]
    
    # Prompt string: "apple banana cherry ... apple banana cherry ..."
    prompt_str = " ".join(sub_seq + sub_seq)

    # 2. Run model forward pass and capture attention weights
    logits, cache = adapter.run_with_cache(prompt_str)
    tokens = adapter.tokenize(prompt_str)
    seq_len = len(tokens)

    # 3. Locate repeat transition boundary
    # Halfway point in token space
    half_len = seq_len // 2
    if half_len < 4:
        # Fallback dummy matrix if sequence too short
        matrix_scores = [[0.0] * num_heads for _ in range(num_layers)]
        return InductionDetectorResponse(
            total_heads_scanned=num_layers * num_heads,
            flagged_count=0,
            num_layers=num_layers,
            num_heads=num_heads,
            top_induction_heads=[],
            matrix_scores=matrix_scores,
            tokens_used=tokens,
            prompt_used=prompt_str
        )

    # Compute induction scores: query at (half_len + i) attending to key (i + 1)
    matrix_scores = [[0.0] * num_heads for _ in range(num_layers)]
    all_scores: List[InductionHeadScore] = []

    for l in range(num_layers):
        attn_map = None
        hook_name = adapter.get_attn_pattern_hook_name(l) if hasattr(adapter, "get_attn_pattern_hook_name") else f"layers.{l}.attn.hook_pattern"

        if hook_name in cache and cache[hook_name].ndim in (3, 4) and cache[hook_name].shape[-1] == cache[hook_name].shape[-2]:
            attn_map = cache[hook_name]
        else:
            # Fallback search for square attention pattern tensors (N_seq x N_seq)
            for k, v in cache.items():
                k_lower = k.lower()
                if (f"blocks.{l}." in k_lower or f"layers.{l}." in k_lower or f"block_{l}" in k_lower or f"h.{l}" in k_lower):
                    if ("pattern" in k_lower or "attn" in k_lower) and v.ndim in (3, 4) and v.shape[-1] == v.shape[-2]:
                        attn_map = v
                        break

        if attn_map is not None:
            t = attn_map.detach().cpu().to(torch.float32)
            if t.ndim == 4:
                t = t[0]  # (heads, N, N)
            
            heads_in_map, n_q, n_k = t.shape
            actual_heads = min(num_heads, heads_in_map)

            for h in range(actual_heads):
                head_attn = t[h]  # (N, N)
                
                # Sum attention weights from second-half query tokens to first-half next-token positions
                accum_score = 0.0
                eval_count = 0

                for i in range(1, half_len - 1):
                    q_idx = half_len + i
                    k_idx = i + 1  # position of token following t_i in first half
                    if q_idx < n_q and k_idx < n_k:
                        accum_score += float(head_attn[q_idx, k_idx].item())
                        eval_count += 1

                avg_score = round(accum_score / max(1, eval_count), 4)
                matrix_scores[l][h] = avg_score

                is_flagged = avg_score >= threshold
                all_scores.append(InductionHeadScore(
                    layer=l,
                    head=h,
                    score=avg_score,
                    is_induction_head=is_flagged
                ))

    # Sort heads descending by score
    all_scores.sort(key=lambda x: x.score, reverse=True)
    top_heads = all_scores[:top_k]
    flagged_count = sum(1 for s in all_scores if s.is_induction_head)

    return InductionDetectorResponse(
        total_heads_scanned=num_layers * num_heads,
        flagged_count=flagged_count,
        num_layers=num_layers,
        num_heads=num_heads,
        top_induction_heads=top_heads,
        matrix_scores=matrix_scores,
        tokens_used=tokens,
        prompt_used=prompt_str
    )
