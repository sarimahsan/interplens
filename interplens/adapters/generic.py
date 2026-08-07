"""GenericAdapter providing progressive interpretability fallbacks (Levels 0-5) for arbitrary PyTorch nn.Module objects."""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn

from interplens.exceptions import UnembeddingNotFoundError
from interplens.adapters.base import BaseModelAdapter, resolve_tokenizer
from interplens.adapters.fingerprint import StaticFingerprint, RuntimeFingerprint
from interplens.adapters.capabilities import evaluate_engine_capabilities, CapabilityLevel
from interplens.adapters.discovery import HookDiscovery
from interplens.adapters.report import generate_model_report, ModelReport

logger = logging.getLogger(__name__)


class GenericAdapter(BaseModelAdapter):
    """Progressive fallback adapter for arbitrary research PyTorch nn.Module objects."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "Generic-PyTorch-Model",
        tokenizer: Optional[Any] = None,
    ):
        device = str(next(model.parameters()).device) if len(list(model.parameters())) > 0 else "cpu"
        super().__init__(model_name=model_name, device=device)
        self._model_instance = model
        self.tokenizer = resolve_tokenizer(model=model, model_name=model_name, tokenizer=tokenizer)

        # 1. Run automatic Hook Discovery
        discovery = HookDiscovery(model, model_name=model_name)
        self.graph, self.capabilities, self.strategy, self.discovery_confidence = discovery.discover()

        # 2. Build Static & Runtime Fingerprints
        self._extract_fingerprints()

        # 3. Evaluate Engine Capabilities Matrix & Build Report
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
            print(self.report.format_text_report())

    def _extract_fingerprints(self):
        """Extracts structural static fingerprint and execution runtime fingerprint."""
        cfg = getattr(self._model_instance, "config", None)
        h_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", getattr(self._model_instance, "hidden_dim", getattr(self._model_instance, "d_model", None))))
        v_size = getattr(cfg, "vocab_size", getattr(cfg, "d_vocab", getattr(self._model_instance, "vocab_size", getattr(self._model_instance, "d_vocab", None))))
        n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", getattr(self._model_instance, "num_heads", getattr(self._model_instance, "n_heads", 12))))

        if h_size is None or v_size is None:
            for mod in self._model_instance.modules():
                if isinstance(mod, nn.Embedding):
                    if h_size is None: h_size = mod.embedding_dim
                    if v_size is None: v_size = mod.num_embeddings
                elif isinstance(mod, nn.Linear):
                    if h_size is None: h_size = mod.in_features

        if h_size is None:
            h_size = 768
            logger.warning(f"Could not introspect hidden_dim for '{self.model_name}', falling back to default {h_size}.")
        if v_size is None:
            v_size = 50257
            logger.warning(f"Could not introspect vocab_size for '{self.model_name}', falling back to default {v_size}.")

        n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", getattr(self._model_instance, "num_layers", getattr(self._model_instance, "n_layers", None))))
        if n_layers is None:
            blocks = getattr(self._model_instance, "blocks", getattr(self._model_instance, "layers", getattr(self._model_instance, "h", None)))
            if blocks is not None and hasattr(blocks, "__len__"):
                n_layers = len(blocks)
            elif len(self.graph.nodes) > 0:
                n_layers = len(self.graph.nodes)
            else:
                n_layers = 12
                logger.warning(f"Could not introspect num_layers for '{self.model_name}', falling back to default {n_layers}.")

        self.num_layers = n_layers
        self.num_heads = n_heads
        self.hidden_dim = h_size
        self.vocab_size = v_size

        self.static_fingerprint = StaticFingerprint(
            architecture=self.strategy.architecture_id,
            family=self.strategy.family.value,
            hidden_size=h_size,
            num_layers=n_layers,
            num_heads=n_heads,
            vocab_size=v_size,
            has_rope=getattr(cfg, "rope_theta", None) is not None,
            is_moe=getattr(cfg, "num_local_experts", None) is not None,
        )

        p_dtype = "float32"
        p_list = list(self._model_instance.parameters())
        if p_list:
            p_dtype = str(p_list[0].dtype).replace("torch.", "")

        self.runtime_fingerprint = RuntimeFingerprint(
            device=str(self.device),
            dtype=p_dtype,
        )

    def load(self) -> None:
        pass

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                ids = self.tokenizer.encode(text)
                return [self.tokenizer.decode([i]) for i in ids]
            except Exception as e:
                logger.debug(f"Tokenizer encoding failed in GenericAdapter: {e}")
        return text.split() if text else []

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        """Decodes token IDs back to human-readable token string."""
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

        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "decode"):
                try:
                    res = self.tokenizer.decode(ids_list, skip_special_tokens=False)
                    if res:
                        return res
                except Exception as e:
                    logger.debug(f"Tokenizer decoding failed: {e}")
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                try:
                    res = self.tokenizer.convert_ids_to_tokens(t_id)
                    if res:
                        return str(res)
                except Exception as e:
                    logger.debug(f"Tokenizer convert_ids_to_tokens failed: {e}")
        return str(t_id)

    @torch.inference_mode()
    def run_with_cache(self, inputs: Union[str, Dict[str, Any], Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs forward pass with hook discovery active, supporting string prompts or multimodal dict inputs."""
        captured_acts: Dict[str, torch.Tensor] = {}
        handles = []

        # Attach PyTorch forward hooks based on discovered graph modules
        for name, mod in self._model_instance.named_modules():
            if name == "":
                continue

            def create_hook(mod_name: str):
                def hook(m, i_tensor, o_tensor):
                    out = o_tensor[0] if isinstance(o_tensor, tuple) else o_tensor
                    if isinstance(out, torch.Tensor):
                        captured_acts[mod_name] = out.detach()
                return hook

            handle = mod.register_forward_hook(create_hook(name))
            handles.append(handle)

        try:
            if isinstance(inputs, torch.Tensor):
                out = self._model_instance(inputs.to(self.device))
            elif isinstance(inputs, str):
                if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                    input_ids = torch.tensor([self.tokenizer.encode(inputs)], device=self.device)
                    out = self._model_instance(input_ids)
                else:
                    out = self._model_instance(inputs)
            elif isinstance(inputs, dict):
                out = self._model_instance(**inputs)
            else:
                out = self._model_instance(inputs)

            logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
            return logits, captured_acts
        finally:
            for h in handles:
                h.remove()

    def get_unembedding_weight(self) -> torch.Tensor:
        unembed = self.strategy.extract_unembedding(self._model_instance)
        if unembed is not None:
            return unembed
        raise UnembeddingNotFoundError(
            f"Unembedding matrix W_U could not be extracted from model architecture '{self.model_name}' "
            f"under strategy '{self.strategy.architecture_id}'."
        )

    def get_resid_post_hook_name(self, layer: int) -> str:
        return self.strategy.get_resid_post_hook_name(layer)

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return self.strategy.get_attn_pattern_hook_name(layer)

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return self.strategy.get_mlp_post_hook_name(layer)
