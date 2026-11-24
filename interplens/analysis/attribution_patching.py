"""Gradient-Based Attribution Patching (AtP) Engine for InterpLens.

Approximates the full Layer x Position causal patching heatmap AND per-component
(Attention vs MLP) causal importance in just 2 forward passes + 1 backward pass:

    AtP(a) = (a_clean - a_corrupt) . grad_{a_corrupt}(M(x_corrupt))

where M = logit(target_token) - logit(foil_token).
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from interplens.adapters.base import BaseModelAdapter
from interplens.analysis.dla import _resolve_token_id, _decode_id
from interplens.schema import (
    AttributionPatchingComponent,
    AttributionPatchingResponse,
)


def _find_layer_modules(model: nn.Module, num_layers: int) -> Tuple[Dict[int, nn.Module], Dict[int, nn.Module], Dict[int, nn.Module]]:
    """Finds residual block, attention, and MLP modules per layer for hooking."""
    resid_mods: Dict[int, nn.Module] = {}
    attn_mods: Dict[int, nn.Module] = {}
    mlp_mods: Dict[int, nn.Module] = {}

    named = dict(model.named_modules())
    for l in range(num_layers):
        for cand in (
            f"blocks.{l}",
            f"transformer.h.{l}",
            f"model.layers.{l}",
            f"model.decoder.layers.{l}",
            f"gpt_neox.layers.{l}",
            f"layers.{l}",
        ):
            if cand in named:
                resid_mods[l] = named[cand]
                break

        for cand in (
            f"blocks.{l}.attn",
            f"transformer.h.{l}.attn",
            f"transformer.h.{l}.self_attention",
            f"model.layers.{l}.self_attn",
            f"model.decoder.layers.{l}.self_attn",
            f"gpt_neox.layers.{l}.attention",
            f"layers.{l}.attn",
            f"layers.{l}.self_attn",
        ):
            if cand in named:
                attn_mods[l] = named[cand]
                break

        for cand in (
            f"blocks.{l}.mlp",
            f"transformer.h.{l}.mlp",
            f"model.layers.{l}.mlp",
            f"model.layers.{l}.block_sparse_moe",
            f"model.layers.{l}.feed_forward",
            f"gpt_neox.layers.{l}.mlp",
            f"layers.{l}.mlp",
        ):
            if cand in named:
                mlp_mods[l] = named[cand]
                break

    return resid_mods, attn_mods, mlp_mods


def _prepare_model_input(adapter: BaseModelAdapter, prompt: str, device: Any) -> Any:
    """Encodes a prompt into input_ids tensor suitable for direct forward pass with autograd."""
    tokenizer = getattr(adapter, "tokenizer", None)
    model = getattr(adapter, "model", getattr(adapter, "_model_instance", None))
    if tokenizer is None and model is not None:
        tokenizer = getattr(model, "tokenizer", None)

    if model is not None and hasattr(model, "to_tokens"):
        try:
            return model.to_tokens(prompt).to(device)
        except Exception:
            pass

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(prompt)
        if isinstance(ids, torch.Tensor):
            return ids.unsqueeze(0).to(device) if ids.ndim == 1 else ids.to(device)
        return torch.tensor([ids], dtype=torch.long, device=device)

    vocab_size = max(1, getattr(adapter, "vocab_size", 1000))
    raw_ids = [ord(ch) % vocab_size for ch in prompt] or [0]
    return torch.tensor([raw_ids], dtype=torch.long, device=device)


def run_attribution_patching(
    adapter: BaseModelAdapter,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token: Optional[str] = None,
    foil_token: Optional[str] = None,
) -> AttributionPatchingResponse:
    """Runs fast gradient-based Attribution Patching (2 forward + 1 backward pass)."""
    t0 = time.perf_counter()
    model = getattr(adapter, "model", getattr(adapter, "_model_instance", None))
    device = getattr(adapter, "device", "cpu")
    info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    num_layers = info.get("num_layers", getattr(adapter, "num_layers", 12))

    clean_tokens = adapter.tokenize(clean_prompt)
    corrupt_tokens = adapter.tokenize(corrupt_prompt)
    seq_len = min(len(clean_tokens), len(corrupt_tokens))
    clean_tokens = clean_tokens[:seq_len]
    corrupt_tokens = corrupt_tokens[:seq_len]

    if model is None or not isinstance(model, nn.Module):
        raise ValueError("Attribution Patching requires an underlying PyTorch nn.Module.")

    resid_mods, attn_mods, mlp_mods = _find_layer_modules(model, num_layers)
    clean_input = _prepare_model_input(adapter, clean_prompt, device)
    corrupt_input = _prepare_model_input(adapter, corrupt_prompt, device)

    # Ensure matching sequence lengths on tensors if tokenizer lengths differed slightly
    if isinstance(clean_input, torch.Tensor) and isinstance(corrupt_input, torch.Tensor):
        if clean_input.ndim == 2 and corrupt_input.ndim == 2:
            min_t = min(clean_input.shape[1], corrupt_input.shape[1])
            clean_input = clean_input[:, :min_t]
            corrupt_input = corrupt_input[:, :min_t]
            if min_t < seq_len:
                seq_len = min_t
                clean_tokens = clean_tokens[:seq_len]
                corrupt_tokens = corrupt_tokens[:seq_len]

    # 1. Clean forward pass (no grad) -> cache clean activations on CPU
    clean_resid: Dict[int, torch.Tensor] = {}
    clean_attn: Dict[int, torch.Tensor] = {}
    clean_mlp: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    def _make_capture(store: Dict[int, torch.Tensor], idx: int):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                store[idx] = t.detach().cpu().to(torch.float32)
        return _hook

    for l, mod in resid_mods.items():
        handles.append(mod.register_forward_hook(_make_capture(clean_resid, l)))
    for l, mod in attn_mods.items():
        handles.append(mod.register_forward_hook(_make_capture(clean_attn, l)))
    for l, mod in mlp_mods.items():
        handles.append(mod.register_forward_hook(_make_capture(clean_mlp, l)))

    try:
        with torch.no_grad():
            try:
                clean_out = model(clean_input, use_cache=False)
            except TypeError:
                clean_out = model(clean_input)
            clean_logits = clean_out.logits if hasattr(clean_out, "logits") else (clean_out[0] if isinstance(clean_out, tuple) else clean_out)
    finally:
        for h in handles:
            h.remove()
        handles.clear()

    clean_last = clean_logits[0, -1].detach().cpu().to(torch.float32) if clean_logits.ndim == 3 else clean_logits[-1].detach().cpu().to(torch.float32)
    vocab_size = clean_last.shape[-1]

    # 2. Corrupt forward pass WITH gradients enabled + retain_grad on hooked outputs
    corrupt_resid: Dict[int, torch.Tensor] = {}
    corrupt_attn: Dict[int, torch.Tensor] = {}
    corrupt_mlp: Dict[int, torch.Tensor] = {}

    def _make_grad_capture(store: Dict[int, torch.Tensor], idx: int):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.retain_grad()
                store[idx] = t
        return _hook

    # Temporarily ensure parameters require grad so intermediate activations have grad_fn
    saved_req_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(True)

    for l, mod in resid_mods.items():
        handles.append(mod.register_forward_hook(_make_grad_capture(corrupt_resid, l)))
    for l, mod in attn_mods.items():
        handles.append(mod.register_forward_hook(_make_grad_capture(corrupt_attn, l)))
    for l, mod in mlp_mods.items():
        handles.append(mod.register_forward_hook(_make_grad_capture(corrupt_mlp, l)))

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            try:
                corrupt_out = model(corrupt_input, use_cache=False)
            except TypeError:
                corrupt_out = model(corrupt_input)
            corrupt_logits = corrupt_out.logits if hasattr(corrupt_out, "logits") else (corrupt_out[0] if isinstance(corrupt_out, tuple) else corrupt_out)
            corrupt_last = corrupt_logits[0, -1] if corrupt_logits.ndim == 3 else corrupt_logits[-1]

            # Resolve target & foil token IDs
            target_id = _resolve_token_id(adapter, target_token)
            foil_id = _resolve_token_id(adapter, foil_token)

            if target_id is None or not (0 <= target_id < vocab_size):
                target_id = int(torch.argmax(clean_last).item())
            if foil_id is None or not (0 <= foil_id < vocab_size) or foil_id == target_id:
                corrupt_top = int(torch.argmax(corrupt_last.detach()).item())
                if corrupt_top != target_id:
                    foil_id = corrupt_top
                else:
                    top2 = torch.topk(clean_last, k=min(2, vocab_size)).indices.tolist()
                    foil_id = int(top2[1]) if len(top2) > 1 and int(top2[1]) != target_id else int((target_id + 1) % vocab_size)

            metric = corrupt_last[target_id] - corrupt_last[foil_id]
            metric.backward()

        corrupt_logit_diff = float(metric.detach().cpu().item())
        clean_logit_diff = float((clean_last[target_id] - clean_last[foil_id]).item())
        baseline_span = clean_logit_diff - corrupt_logit_diff
        denom = baseline_span if abs(baseline_span) > 1e-6 else 1.0

        # 3. Compute Layer x Position Attribution Patching heatmap
        heatmap_matrix: List[List[float]] = []
        max_score = -1e9
        max_cell = (0, 0)

        for l in range(num_layers):
            row: List[float] = []
            c_res = clean_resid.get(l)
            r_res = corrupt_resid.get(l)

            if c_res is not None and r_res is not None and r_res.grad is not None:
                c_t = c_res[0] if c_res.ndim == 3 else c_res
                r_t = (r_res[0] if r_res.ndim == 3 else r_res).detach().cpu().to(torch.float32)
                g_t = (r_res.grad[0] if r_res.grad.ndim == 3 else r_res.grad).detach().cpu().to(torch.float32)
                n_pos = min(seq_len, c_t.shape[0], r_t.shape[0], g_t.shape[0])
                # AtP(l, p) = sum_d (clean - corrupt) * grad
                atp_pos = torch.sum((c_t[:n_pos] - r_t[:n_pos]) * g_t[:n_pos], dim=-1)
                for p in range(seq_len):
                    raw_val = float(atp_pos[p].item()) if p < n_pos else 0.0
                    pct = round((raw_val / denom) * 100.0, 2)
                    row.append(pct)
                    if pct > max_score:
                        max_score = pct
                        max_cell = (l, p)
            else:
                for p in range(seq_len):
                    row.append(0.0)
            heatmap_matrix.append(row)

        # 4. Compute per-component (Attn & MLP) attribution scores
        component_scores: List[AttributionPatchingComponent] = []
        for l in range(num_layers):
            for kind, c_map, r_map in (
                ("attn", clean_attn, corrupt_attn),
                ("mlp", clean_mlp, corrupt_mlp),
            ):
                c_t = c_map.get(l)
                r_t = r_map.get(l)
                if c_t is not None and r_t is not None and r_t.grad is not None:
                    c2 = c_t[0] if c_t.ndim == 3 else c_t
                    r2 = (r_t[0] if r_t.ndim == 3 else r_t).detach().cpu().to(torch.float32)
                    g2 = (r_t.grad[0] if r_t.grad.ndim == 3 else r_t.grad).detach().cpu().to(torch.float32)
                    n_pos = min(c2.shape[0], r2.shape[0], g2.shape[0])
                    raw_score = float(torch.sum((c2[:n_pos] - r2[:n_pos]) * g2[:n_pos]).item())
                    pct_score = round((raw_score / denom) * 100.0, 2)
                    component_scores.append(
                        AttributionPatchingComponent(
                            component=f"L{l} {'Attn' if kind == 'attn' else 'MLP'}",
                            layer=l,
                            kind=kind,
                            score=round(raw_score, 4),
                            recovery_pct=pct_score,
                        )
                    )

        component_scores.sort(key=lambda c: abs(c.recovery_pct), reverse=True)
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        for p, orig_req in zip(model.parameters(), saved_req_grad):
            p.requires_grad_(orig_req)

    runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return AttributionPatchingResponse(
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        clean_tokens=clean_tokens,
        corrupt_tokens=corrupt_tokens,
        target_token=_decode_id(adapter, target_id),
        foil_token=_decode_id(adapter, foil_id),
        baseline_clean_logit_diff=round(clean_logit_diff, 4),
        baseline_corrupt_logit_diff=round(corrupt_logit_diff, 4),
        num_layers=num_layers,
        seq_len=seq_len,
        max_attribution_layer=max_cell[0],
        max_attribution_position=max_cell[1],
        max_attribution_pct=round(max_score, 2) if max_score > -1e8 else 0.0,
        heatmap_matrix=heatmap_matrix,
        top_components=component_scores[:15],
        runtime_ms=runtime_ms,
    )
