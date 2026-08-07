"""CustomModelAdapter wrapping HuggingFace and PyTorch nn.Module architectures with HookDiscovery and Strategy Registry."""

import logging
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import torch
import torch.nn as nn

from interplens.exceptions import UnembeddingNotFoundError
from interplens.adapters.base import BaseModelAdapter, resolve_tokenizer
from interplens.adapters.fingerprint import StaticFingerprint, RuntimeFingerprint
from interplens.adapters.capabilities import evaluate_engine_capabilities
from interplens.adapters.discovery import HookDiscovery
from interplens.adapters.report import generate_model_report, ModelReport

logger = logging.getLogger(__name__)


class PyTorchAutoHooker:
    """Intercepts activations across submodules using native PyTorch forward hooks."""

    def __init__(self, model: nn.Module, target_module_names: Optional[List[str]] = None):
        self.model = model
        self.target_module_names = set(target_module_names) if target_module_names else None
        self.captured_activations: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[Any] = []

    def set_target_module_names(self, names: List[str]):
        """Sets target module names for selective hooking."""
        self.target_module_names = set(names) if names else None

    def attach_hooks(self):
        """Attaches PyTorch forward hooks to target submodules or all named submodules."""
        self.captured_activations.clear()

        for name, module in self.model.named_modules():
            if name == "":
                continue

            # If target list is set and non-empty, only hook specified modules
            if self.target_module_names:
                if name not in self.target_module_names:
                    continue

            def create_hook(module_name: str):
                def hook(mod, input_tensor, output_tensor):
                    out = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                    if isinstance(out, torch.Tensor):
                        self.captured_activations[module_name] = out.detach()
                return hook

            handle = module.register_forward_hook(create_hook(name))
            self._hook_handles.append(handle)

    def remove_hooks(self):
        """Removes all active PyTorch forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


class CustomModelAdapter(BaseModelAdapter):
    """Adapter for HuggingFace AutoModelForCausalLM and novel PyTorch architectures."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        model_name: str = "Custom-PyTorch-Model",
        tokenize_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        device = str(next(model.parameters()).device) if len(list(model.parameters())) > 0 else "cpu"
        super().__init__(model_name=model_name, device=device)
        self._model_instance = model
        self.tokenizer = resolve_tokenizer(model=model, model_name=model_name, tokenizer=tokenizer)
        self.tokenize_fn = tokenize_fn
        if hasattr(model, "config"):
            try:
                model.config.output_attentions = True
            except Exception:
                pass

        # 1. Run automatic Hook Discovery first
        discovery = HookDiscovery(model, model_name=model_name)
        self.graph, self.capabilities, self.strategy, self.discovery_confidence = discovery.discover()

        # 2. PyTorchAutoHooker hooks named submodules
        self.auto_hooker = PyTorchAutoHooker(model)

        # 2. Extract Structural Fingerprints & Dimensions
        self._extract_fingerprints()

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

    def _extract_fingerprints(self):
        """Introspects module structure to build fingerprints and geometry metadata."""
        cfg = getattr(self._model_instance, "config", None)
        if cfg is not None:
            h_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 2048))
            n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", 36))
            n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 16))
            v_size = getattr(cfg, "vocab_size", getattr(cfg, "d_vocab", 151936))
        else:
            # Fallback dynamic introspection for custom models without HuggingFace config
            h_size = getattr(self._model_instance, "hidden_dim", getattr(self._model_instance, "d_model", getattr(self._model_instance, "hidden_size", 768)))
            v_size = getattr(self._model_instance, "vocab_size", getattr(self._model_instance, "d_vocab", 50257))
            n_heads = getattr(self._model_instance, "num_heads", getattr(self._model_instance, "n_heads", getattr(self._model_instance, "num_attention_heads", 12)))
            
            # Count actual layer blocks or inspect num_layers attribute
            n_layers = getattr(self._model_instance, "num_layers", getattr(self._model_instance, "n_layers", None))
            if n_layers is None:
                blocks = getattr(self._model_instance, "blocks", getattr(self._model_instance, "layers", getattr(self._model_instance, "h", None)))
                if blocks is not None and hasattr(blocks, "__len__"):
                    n_layers = len(blocks)
                elif len(self.graph.nodes) > 0:
                    n_layers = len(self.graph.nodes)
                else:
                    n_layers = 12

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
        if self.tokenize_fn is not None:
            return self.tokenize_fn(text)

        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "encode") and hasattr(self.tokenizer, "decode"):
                try:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if ids:
                        tokens = []
                        for i in ids:
                            tok_str = self.tokenizer.decode([i])
                            tokens.append(tok_str if tok_str else f"[{i}]")
                        return tokens
                except Exception:
                    pass

            if hasattr(self.tokenizer, "tokenize"):
                try:
                    return self.tokenizer.tokenize(text)
                except Exception:
                    pass

        return text.split()

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

        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "decode"):
                try:
                    res = self.tokenizer.decode(ids_list, skip_special_tokens=False)
                    if res:
                        return res
                except Exception:
                    pass
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                try:
                    res = self.tokenizer.convert_ids_to_tokens(t_id)
                    if res:
                        return str(res)
                except Exception:
                    pass
        return str(t_id)

    @torch.inference_mode()
    def run_with_cache(self, inputs: Union[str, Dict[str, Any], Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs forward pass with PyTorchAutoHooker active and collects activations."""
        self.auto_hooker.attach_hooks()
        try:
            prompt_str = inputs if isinstance(inputs, str) else (inputs.get("prompt", "") if isinstance(inputs, dict) else str(inputs))

            if hasattr(self.tokenizer, "encode"):
                try:
                    input_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
                except TypeError:
                    input_ids = self.tokenizer.encode(prompt_str)

                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor([input_ids], device=self.device)
                elif input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0).to(self.device)
                try:
                    out = self._model_instance(input_ids, output_attentions=True)
                except Exception:
                    out = self._model_instance(input_ids)
            else:
                try:
                    out = self._model_instance(prompt_str, output_attentions=True)
                except Exception:
                    out = self._model_instance(prompt_str)
                
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
                
            cache = dict(self.auto_hooker.captured_activations)

            attentions = getattr(out, "attentions", None)
            if attentions is not None and isinstance(attentions, (tuple, list)):
                for l_idx, attn_map in enumerate(attentions):
                    if isinstance(attn_map, torch.Tensor):
                        cache[f"layers.{l_idx}.attn.hook_pattern"] = attn_map
                        cache[f"blocks.{l_idx}.attn.hook_pattern"] = attn_map

            return logits, cache
        finally:
            self.auto_hooker.remove_hooks()

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

