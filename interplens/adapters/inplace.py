"""InPlaceModelAdapter for wrapping pre-loaded GPU TransformerLens model instances."""

import logging
from typing import List, Dict, Any, Tuple, Union
import torch
from interplens.exceptions import UnembeddingNotFoundError
from interplens.adapters.base import BaseModelAdapter, resolve_tokenizer
from interplens.adapters.fingerprint import StaticFingerprint, RuntimeFingerprint
from interplens.adapters.capabilities import evaluate_engine_capabilities, ModelCapability, CapabilityLevel
from interplens.adapters.discovery import HookDiscovery
from interplens.adapters.report import generate_model_report, ModelReport

logger = logging.getLogger(__name__)


class InPlaceModelAdapter(BaseModelAdapter):
    """Adapter wrapping an already instantiated TransformerLens HookedTransformer in GPU memory."""

    def __init__(self, model_instance: Any, model_name: str = "InPlace-Model"):
        try:
            device = str(next(model_instance.parameters()).device)
        except Exception:
            device = "cpu"
            
        super().__init__(model_name=model_name, device=device)
        self._model_instance = model_instance
        self.tokenizer = resolve_tokenizer(model=model_instance, model_name=model_name)

        # 1. Run automatic Hook Discovery
        discovery = HookDiscovery(model_instance, model_name=model_name)
        self.graph, self.capabilities, self.strategy, self.discovery_confidence = discovery.discover()

        # 2. Extract Metadata & Fingerprints
        self._extract_metadata()

        # 3. Evaluate Engine Capability Matrix & Build Report
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

        from interplens.config import settings
        if settings.debug:
            logger.debug(self.report.format_text_report())

    def _extract_metadata(self):
        """Extracts layer counts, head counts, and dimensions from HookedTransformer metadata."""
        cfg = getattr(self._model_instance, "cfg", None)
        if cfg is not None:
            self.num_layers = getattr(cfg, "n_layers", 0)
            self.num_heads = getattr(cfg, "n_heads", 0)
            self.hidden_dim = getattr(cfg, "d_model", 0)
            self.vocab_size = getattr(cfg, "d_vocab", 0)
        else:
            self.num_layers = len(getattr(self._model_instance, "blocks", []))
            self.num_heads = 12
            self.hidden_dim = 768
            self.vocab_size = 50257

        self.static_fingerprint = StaticFingerprint(
            architecture=self.strategy.architecture_id,
            family=self.strategy.family.value,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
        )

        p_dtype = "float32"
        try:
            p_list = list(self._model_instance.parameters())
            if p_list:
                p_dtype = str(p_list[0].dtype).replace("torch.", "")
        except Exception:
            pass

        self.runtime_fingerprint = RuntimeFingerprint(
            device=str(self.device),
            dtype=p_dtype,
        )

    def load(self) -> None:
        pass

    def tokenize(self, text: str) -> List[str]:
        if hasattr(self._model_instance, "to_str_tokens"):
            return self._model_instance.to_str_tokens(text)
        elif hasattr(self._model_instance, "tokenizer"):
            tokens = self._model_instance.tokenizer.encode(text)
            return [self._model_instance.tokenizer.decode([t]) for t in tokens]
        return text.split() if text else []

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        if token_ids is None:
            return ""
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

        tokenizer = getattr(self, "tokenizer", None) or getattr(self._model_instance, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                res = tokenizer.decode(ids_list)
                if res:
                    return res
            except Exception:
                pass

        return str(t_id)

    @torch.inference_mode()
    def run_with_cache(self, inputs: Union[str, Dict[str, Any], Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prompt_str = inputs if isinstance(inputs, str) else (inputs.get("prompt", "") if isinstance(inputs, dict) else str(inputs))
        if hasattr(self._model_instance, "run_with_cache"):
            logits, cache = self._model_instance.run_with_cache(prompt_str)
            cache_dict = dict(cache) if hasattr(cache, "items") else cache
            return logits, cache_dict
        else:
            raise NotImplementedError("InPlaceModelAdapter requires model with run_with_cache.")

    def get_unembedding_weight(self) -> torch.Tensor:
        unembed = self.strategy.extract_unembedding(self._model_instance)
        if unembed is not None:
            return unembed
        if hasattr(self._model_instance, "W_U"):
            return self._model_instance.W_U
        raise UnembeddingNotFoundError(
            f"Unembedding matrix W_U could not be extracted from model instance '{self.model_name}' "
            f"under strategy '{self.strategy.architecture_id}'."
        )

    def get_resid_post_hook_name(self, layer: int) -> str:
        return self.strategy.get_resid_post_hook_name(layer)

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return self.strategy.get_attn_pattern_hook_name(layer)

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return self.strategy.get_mlp_post_hook_name(layer)
