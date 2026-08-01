"""GPT-2 model family adapter implementation for InterpLens."""

from typing import List, Dict, Any, Tuple
import torch
try:
    from transformer_lens import HookedTransformer
    HAS_TRANSFORMER_LENS = True
except ImportError:
    HookedTransformer = Any
    HAS_TRANSFORMER_LENS = False

from interplens.adapters.base import BaseModelAdapter


class GPT2Adapter(BaseModelAdapter):
    """Adapter for TransformerLens HookedTransformer GPT-2 model family."""

    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        if device == "auto":
            from interplens.utils.device import get_optimal_device
            device = str(get_optimal_device())
        super().__init__(model_name=model_name, device=device)

    def load(self) -> None:
        """Loads HookedTransformer model weights into specified device."""
        if not HAS_TRANSFORMER_LENS:
            raise ImportError(
                "transformer_lens package is required to load pretrained TransformerLens models. "
                "Install it via `pip install transformer-lens`."
            )
        if self._model_instance is None:
            self._model_instance = HookedTransformer.from_pretrained(
                self.model_name,
                device=str(self.device),
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
            )
            self.num_layers = self._model_instance.cfg.n_layers
            self.num_heads = self._model_instance.cfg.n_heads
            self.hidden_dim = self._model_instance.cfg.d_model
            self.vocab_size = self._model_instance.cfg.d_vocab

    def tokenize(self, text: str) -> List[str]:
        """Converts string prompt into GPT-2 string tokens."""
        if self._model_instance is None:
            self.load()
        return self._model_instance.to_str_tokens(text)

    @torch.inference_mode()
    def run_with_cache(self, prompt: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs forward pass under @torch.inference_mode() and returns activation cache."""
        if self._model_instance is None:
            self.load()
        logits, cache = self._model_instance.run_with_cache(prompt)
        return logits, dict(cache)

    def get_unembedding_weight(self) -> torch.Tensor:
        """Returns GPT-2 unembedding matrix W_U [d_model, d_vocab]."""
        if self._model_instance is None:
            self.load()
        return self._model_instance.W_U

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.hook_resid_post"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.attn.hook_pattern"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.mlp.hook_post"
