"""InPlaceModelAdapter for wrapping pre-loaded GPU model instances with zero-copy VRAM execution."""

from typing import List, Dict, Any, Tuple
import torch
from interplens.adapters.base import BaseModelAdapter


class InPlaceModelAdapter(BaseModelAdapter):
    """Adapter wrapping an already instantiated TransformerLens HookedTransformer in GPU memory."""

    def __init__(self, model_instance: Any, model_name: str = "InPlace-Model"):
        # Infer device from model parameter
        try:
            device = str(next(model_instance.parameters()).device)
        except Exception:
            device = "cpu"
            
        super().__init__(model_name=model_name, device=device)
        self._model_instance = model_instance
        self._extract_metadata()

    def _extract_metadata(self):
        """Extracts layer counts, head counts, and dimensions from HookedTransformer metadata."""
        cfg = getattr(self._model_instance, "cfg", None)
        if cfg is not None:
            self.num_layers = getattr(cfg, "n_layers", 0)
            self.num_heads = getattr(cfg, "n_heads", 0)
            self.hidden_dim = getattr(cfg, "d_model", 0)
            self.vocab_size = getattr(cfg, "d_vocab", 0)
        else:
            # Fallback introspection for non-HookedTransformer objects
            self.num_layers = len(getattr(self._model_instance, "blocks", []))
            self.num_heads = 12
            self.hidden_dim = 768
            self.vocab_size = 50257

    def load(self) -> None:
        """No-op since model is already loaded in memory."""
        pass

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text using model's tokenizer and formats token strings."""
        if hasattr(self._model_instance, "to_str_tokens"):
            return self._model_instance.to_str_tokens(text)
        elif hasattr(self._model_instance, "tokenizer"):
            tokens = self._model_instance.tokenizer.encode(text)
            return [self._model_instance.tokenizer.decode([t]) for t in tokens]
        return [c for c in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to human-readable string token labels (e.g. ' Paris')."""
        if not token_ids:
            return ""
        
        # 1. Try TransformerLens single token string decoder
        if hasattr(self._model_instance, "to_single_str_token"):
            try:
                res = self._model_instance.to_single_str_token(token_ids[0])
                if res:
                    return res
            except Exception:
                pass

        # 2. Try HuggingFace / TransformerLens tokenizer decode
        tokenizer = getattr(self, "tokenizer", None) or getattr(self._model_instance, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                res = tokenizer.decode(token_ids if isinstance(token_ids, list) else [token_ids])
                if res:
                    return res
            except Exception:
                pass

        # 3. Try to_string with tensor conversion
        if hasattr(self._model_instance, "to_string"):
            try:
                res = self._model_instance.to_string(torch.tensor(token_ids, device=self.device))
                if res and isinstance(res, str):
                    return res
            except Exception:
                pass

        return str(token_ids[0]) if isinstance(token_ids, list) else str(token_ids)

    @torch.inference_mode()
    def run_with_cache(self, prompt: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs forward pass on existing model instance and returns (logits, activation_cache)."""
        if hasattr(self._model_instance, "run_with_cache"):
            logits, cache = self._model_instance.run_with_cache(prompt)
            # Ensure cache is dict-like
            cache_dict = dict(cache) if hasattr(cache, "items") else cache
            return logits, cache_dict
        else:
            raise NotImplementedError("InPlaceModelAdapter requires model with run_with_cache or custom hooker.")

    def get_unembedding_weight(self) -> torch.Tensor:
        """Extracts W_U unembedding tensor [hidden_dim, vocab_size]."""
        if hasattr(self._model_instance, "W_U"):
            return self._model_instance.W_U
        elif hasattr(self._model_instance, "unembed") and hasattr(self._model_instance.unembed, "W_U"):
            return self._model_instance.unembed.W_U
        elif hasattr(self._model_instance, "lm_head") and hasattr(self._model_instance.lm_head, "weight"):
            return self._model_instance.lm_head.weight.T
        raise AttributeError("Unembedding weight W_U or lm_head not found on model instance.")

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.hook_resid_post"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.attn.hook_pattern"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.mlp.hook_post"
