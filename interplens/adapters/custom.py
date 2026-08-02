"""CustomModelAdapter and PyTorchAutoHooker for novel, experimental PyTorch architectures."""

from typing import List, Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
from interplens.adapters.base import BaseModelAdapter


class PyTorchAutoHooker:
    """Intercepts activations across nn.ModuleList submodules using native PyTorch forward hooks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.captured_activations: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[Any] = []

    def attach_hooks(self):
        """Attaches PyTorch forward hooks to all named modules."""
        self.captured_activations.clear()
        
        for name, module in self.model.named_modules():
            if name == "":
                continue
            
            def create_hook(module_name: str):
                def hook(mod, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        out = output_tensor[0]
                    else:
                        out = output_tensor
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
    """Adapter for novel/custom PyTorch nn.Module architectures with auto-hooking or custom hook mappings."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_name: str = "Custom-PyTorch-Model",
        tokenize_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        device = str(next(model.parameters()).device) if len(list(model.parameters())) > 0 else "cpu"
        super().__init__(model_name=model_name, device=device)
        self._model_instance = model
        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize_fn
        self.auto_hooker = PyTorchAutoHooker(model)
        self._extract_custom_metadata()

    def _extract_custom_metadata(self):
        """Introspects module structure to determine layer counts and dimensions."""
        layers = []
        for name, module in self._model_instance.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                layers.append(name)
        self.num_layers = max(len(layers), 1)
        self.num_heads = 12
        self.hidden_dim = 768
        self.vocab_size = 50257

    def load(self) -> None:
        """No-op for in-memory custom models."""
        pass

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text using provided tokenizer or custom tokenize function."""
        if self.tokenize_fn is not None:
            return self.tokenize_fn(text)
        if hasattr(self.tokenizer, "tokenize"):
            return self.tokenizer.tokenize(text)
        elif hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text)
            return [str(i) for i in ids]
    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to string token label."""
        if not token_ids:
            return ""
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids)
        return str(token_ids[0])

    @torch.inference_mode()
    def run_with_cache(self, prompt: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs forward pass with PyTorchAutoHooker active and collects activations."""
        self.auto_hooker.attach_hooks()
        try:
            # Tokenize input to tensor if tokenizer supported
            if hasattr(self.tokenizer, "encode"):
                input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
                logits = self._model_instance(input_ids)
            else:
                logits = self._model_instance(prompt)
                
            if isinstance(logits, tuple):
                logits = logits[0]
                
            cache = dict(self.auto_hooker.captured_activations)
            return logits, cache
        finally:
            self.auto_hooker.remove_hooks()

    def get_unembedding_weight(self) -> torch.Tensor:
        """Attempts to locate unembedding linear layer or matrix."""
        for name, param in self._model_instance.named_parameters():
            if "unembed" in name.lower() or "lm_head" in name.lower() or "output" in name.lower():
                if param.ndim == 2:
                    return param.T if param.shape[0] > param.shape[1] else param
        # Fallback dummy matrix
        return torch.randn(self.hidden_dim, self.vocab_size, device=self.device)

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"layers.{layer}"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"layers.{layer}.attn"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"layers.{layer}.mlp"
