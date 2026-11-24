"""Contrastive Activation Steering (CAA) & Side-by-Side Generation Engine for InterpLens.

Computes Difference-in-Means steering vectors from Positive vs. Negative prompt pairs
at any transformer layer and performs multi-token Baseline vs. Steered generation
across GPT-2, Llama, Qwen, Mistral, Gemma, Mixtral, and custom PyTorch models.
"""

import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from interplens.adapters.base import BaseModelAdapter
from interplens.analysis.attribution_patching import _prepare_model_input
from interplens.analysis.dla import _decode_id
from interplens.schema import (
    SteeringVectorInfo,
    GeneratedTokenProb,
    SteeringGenerationResponse,
)


class SteeringVectorStore:
    """Thread-safe in-memory registry of computed contrastive steering vectors."""

    def __init__(self, max_vectors: int = 32):
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._max_vectors = max_vectors

    def save(self, vector: torch.Tensor, meta: Dict[str, Any]) -> str:
        vid = meta.get("vector_id") or uuid.uuid4().hex[:8]
        with self._lock:
            if len(self._store) >= self._max_vectors and vid not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[vid] = {"vector": vector.detach().cpu().to(torch.float32), "meta": {**meta, "vector_id": vid}}
        return vid

    def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.get(vector_id)

    def list_metadata(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [entry["meta"] for entry in self._store.values()]


global_steering_store = SteeringVectorStore()


def _find_resid_module(model: nn.Module, layer: int, adapter: BaseModelAdapter) -> Optional[nn.Module]:
    """Finds the target transformer layer module across all supported model families."""
    named = dict(model.named_modules())
    hook_name = adapter.get_resid_post_hook_name(layer) if hasattr(adapter, "get_resid_post_hook_name") else ""
    for cand in (
        hook_name,
        f"blocks.{layer}",
        f"transformer.h.{layer}",
        f"model.layers.{layer}",
        f"model.decoder.layers.{layer}",
        f"gpt_neox.layers.{layer}",
        f"layers.{layer}",
    ):
        if cand and cand in named:
            return named[cand]

    for container_attr in ("blocks", "layers", "h"):
        blocks = getattr(model, container_attr, None)
        if blocks is None and hasattr(model, "model"):
            blocks = getattr(model.model, container_attr, None)
        if blocks is None and hasattr(model, "transformer"):
            blocks = getattr(model.transformer, container_attr, None)
        if blocks is not None and hasattr(blocks, "__len__") and 0 <= layer < len(blocks):
            return blocks[layer]
    return None


def _extract_last_resid(cache: Dict[str, torch.Tensor], layer: int, adapter: BaseModelAdapter) -> Optional[torch.Tensor]:
    """Extracts the final token position's 1D residual vector at `layer` from an activation cache."""
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
    for key in candidates:
        if key and key in cache and isinstance(cache[key], torch.Tensor):
            t = cache[key].detach().cpu().to(torch.float32)
            if t.ndim == 3:
                return t[0, -1]
            if t.ndim == 2:
                return t[-1]

    for k, v in cache.items():
        k_low = k.lower()
        if any(x in k_low for x in ("pattern", "attn", "mlp", "ffn", "norm", "ln_")):
            continue
        if (f"blocks.{layer}" in k_low or f"layers.{layer}" in k_low or f"h.{layer}" in k_low) and isinstance(v, torch.Tensor) and v.ndim in (2, 3):
            t = v.detach().cpu().to(torch.float32)
            return t[0, -1] if t.ndim == 3 else t[-1]
    return None


@torch.inference_mode()
def compute_contrastive_steering_vector(
    adapter: BaseModelAdapter,
    positive_prompts: List[str],
    negative_prompts: List[str],
    target_layer: int,
    name: str = "Custom Steering Vector",
    normalize: bool = False,
) -> SteeringVectorInfo:
    """Computes a Difference-in-Means steering vector v = mean(h_pos) - mean(h_neg) at `target_layer`."""
    pos_clean = [p.strip() for p in positive_prompts if p and p.strip()]
    neg_clean = [p.strip() for p in negative_prompts if p and p.strip()]
    if not pos_clean or not neg_clean:
        raise ValueError("Provide at least one positive prompt and one negative prompt.")

    info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}
    num_layers = info.get("num_layers", getattr(adapter, "num_layers", 12))
    layer = max(0, min(target_layer, max(0, num_layers - 1)))

    pos_vecs: List[torch.Tensor] = []
    for p in pos_clean:
        _, cache = adapter.run_with_cache(p)
        v = _extract_last_resid(cache, layer, adapter)
        if v is not None:
            pos_vecs.append(v)

    neg_vecs: List[torch.Tensor] = []
    for p in neg_clean:
        _, cache = adapter.run_with_cache(p)
        v = _extract_last_resid(cache, layer, adapter)
        if v is not None:
            neg_vecs.append(v)

    if not pos_vecs or not neg_vecs:
        raise ValueError(f"Could not extract residual stream vectors at layer {layer} for contrastive steering.")

    mean_pos = torch.stack(pos_vecs, dim=0).mean(dim=0)
    mean_neg = torch.stack(neg_vecs, dim=0).mean(dim=0)
    diff_vec = mean_pos - mean_neg

    raw_norm = float(torch.norm(diff_vec, p=2).item())
    pos_neg_cos = float(F.cosine_similarity(mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)).item())

    if normalize and raw_norm > 1e-8:
        diff_vec = diff_vec / raw_norm

    meta = {
        "name": name.strip() or "Contrastive Vector",
        "target_layer": layer,
        "hidden_dim": int(diff_vec.shape[0]),
        "num_positive": len(pos_vecs),
        "num_negative": len(neg_vecs),
        "l2_norm": round(raw_norm, 4),
        "pos_neg_cosine": round(pos_neg_cos, 4),
        "normalized": normalize,
        "model_name": getattr(adapter, "model_name", "model"),
    }
    vector_id = global_steering_store.save(diff_vec, meta)

    return SteeringVectorInfo(vector_id=vector_id, **meta)


@torch.inference_mode()
def _autoregressive_generate(
    model: nn.Module,
    adapter: BaseModelAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Tuple[str, List[GeneratedTokenProb]]:
    """Lightweight architecture-agnostic autoregressive loop returning text + token probabilities."""
    curr_ids = input_ids.clone()
    eos_id = getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None)
    generated_ids: List[int] = []
    token_probs: List[GeneratedTokenProb] = []

    for _ in range(max_new_tokens):
        try:
            out = model(curr_ids, use_cache=False)
        except TypeError:
            out = model(curr_ids)
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
        last_logits = logits[0, -1].float() if logits.ndim == 3 else logits[-1].float()

        if temperature and temperature > 1e-4:
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            next_prob = float(probs[next_id].item())
        else:
            probs = F.softmax(last_logits, dim=-1)
            next_prob_t, next_id_t = torch.max(probs, dim=-1)
            next_id = int(next_id_t.item())
            next_prob = float(next_prob_t.item())

        generated_ids.append(next_id)
        tok_str = _decode_id(adapter, next_id)
        token_probs.append(
            GeneratedTokenProb(
                token=tok_str,
                token_id=next_id,
                probability=round(next_prob * 100.0, 2),
            )
        )

        next_tensor = torch.tensor([[next_id]], dtype=curr_ids.dtype, device=curr_ids.device)
        curr_ids = torch.cat([curr_ids, next_tensor], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception:
            gen_text = "".join(tp.token for tp in token_probs)
    else:
        gen_text = "".join(tp.token for tp in token_probs)

    return gen_text, token_probs


@torch.inference_mode()
def generate_steered_comparison(
    adapter: BaseModelAdapter,
    prompt: str,
    vector_id: str,
    target_layer: Optional[int] = None,
    multiplier: float = 2.0,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
) -> SteeringGenerationResponse:
    """Runs side-by-side Baseline vs. Steered multi-token generation using a stored CAA vector."""
    t0 = time.perf_counter()
    entry = global_steering_store.get(vector_id)
    if entry is None:
        raise ValueError(f"Steering vector '{vector_id}' not found. Compute a contrastive vector first.")

    model = getattr(adapter, "model", getattr(adapter, "_model_instance", None))
    if model is None or not isinstance(model, nn.Module):
        raise ValueError("Side-by-side steered generation requires an underlying PyTorch nn.Module.")

    layer = target_layer if target_layer is not None else int(entry["meta"]["target_layer"])
    steer_vec = entry["vector"]  # CPU float32 [d_model]
    device = getattr(adapter, "device", "cpu")
    input_ids = _prepare_model_input(adapter, prompt, device)
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("Could not encode prompt into token IDs for generation.")

    max_tokens = max(1, min(int(max_new_tokens), 64))

    # 1. Baseline unsteered generation
    baseline_text, baseline_tokens = _autoregressive_generate(
        model=model,
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    # 2. Steered generation with forward hook at target_layer
    target_mod = _find_resid_module(model, layer, adapter)
    if target_mod is None:
        raise ValueError(f"Could not locate transformer layer {layer} for steering hook injection.")

    steer_delta = steer_vec * float(multiplier)

    def _steer_hook(_mod, _inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if not isinstance(tensor, torch.Tensor):
            return out
        delta = steer_delta.to(device=tensor.device, dtype=tensor.dtype)
        if tensor.ndim == 3 and tensor.shape[-1] == delta.shape[0]:
            steered = tensor + delta.view(1, 1, -1)
        elif tensor.ndim == 2 and tensor.shape[-1] == delta.shape[0]:
            steered = tensor + delta.view(1, -1)
        else:
            steered = tensor
        return (steered,) + out[1:] if isinstance(out, tuple) else steered

    handle = target_mod.register_forward_hook(_steer_hook)
    try:
        steered_text, steered_tokens = _autoregressive_generate(
            model=model,
            adapter=adapter,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    finally:
        handle.remove()

    runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return SteeringGenerationResponse(
        prompt=prompt,
        vector_id=vector_id,
        vector_name=entry["meta"]["name"],
        target_layer=layer,
        multiplier=multiplier,
        max_new_tokens=max_tokens,
        baseline_completion=baseline_text,
        steered_completion=steered_text,
        baseline_tokens=baseline_tokens,
        steered_tokens=steered_tokens,
        runtime_ms=runtime_ms,
    )
