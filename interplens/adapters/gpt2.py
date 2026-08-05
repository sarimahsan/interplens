"""GPT-2 model family adapter implementation for InterpLens."""

from typing import List, Dict, Any, Tuple, Union
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

            from interplens.adapters.fingerprint import StaticFingerprint, RuntimeFingerprint
            from interplens.adapters.capabilities import evaluate_engine_capabilities
            from interplens.adapters.discovery import HookDiscovery
            from interplens.adapters.report import generate_model_report

            discovery = HookDiscovery(self._model_instance, model_name=self.model_name)
            self.graph, self.capabilities, self.strategy, self.discovery_confidence = discovery.discover()

            self.static_fingerprint = StaticFingerprint(
                architecture=self.strategy.architecture_id,
                family=self.strategy.family.value,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                vocab_size=self.vocab_size,
            )
            self.runtime_fingerprint = RuntimeFingerprint(
                device=str(self.device),
                dtype="float32",
            )
            self.engine_capabilities = evaluate_engine_capabilities(self.capabilities, self.runtime_fingerprint, self.discovery_confidence)
            self.report = generate_model_report(
                model_name=self.model_name,
                strategy_id=self.strategy.architecture_id,
                family=self.strategy.family.value,
                confidence=self.discovery_confidence,
                static_fp=self.static_fingerprint,
                runtime_fp=self.runtime_fingerprint,
                model_cap=self.capabilities,
                engine_matrix=self.engine_capabilities,
            )

    def tokenize(self, text: str) -> List[str]:
        """Converts string prompt into GPT-2 string tokens."""
        if self._model_instance is None:
            self.load()
        return self._model_instance.to_str_tokens(text)

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        """Decodes token IDs back to human-readable string token labels (e.g. ' Paris')."""
        if token_ids is None:
            return ""
        if self._model_instance is None:
            self.load()
            
        if isinstance(token_ids, int):
            t_id = token_ids
            ids_list = [token_ids]
        elif isinstance(token_ids, (list, tuple)) and len(token_ids) > 0:
            t_id = token_ids[0]
            ids_list = list(token_ids)
        else:
            return ""

        if hasattr(self._model_instance, "to_single_str_token"):
            try:
                res = self._model_instance.to_single_str_token(t_id)
                if res:
                    return res
            except Exception:
                pass

        if hasattr(self._model_instance, "tokenizer") and hasattr(self._model_instance.tokenizer, "decode"):
            try:
                res = self._model_instance.tokenizer.decode(ids_list)
                if res:
                    return res
            except Exception:
                pass

        return str(t_id)

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
