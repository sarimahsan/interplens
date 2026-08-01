"""Model Adapters package for InterpLens."""

from typing import Any, Optional, Type
from interplens.adapters.base import BaseModelAdapter
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.adapters.custom import CustomModelAdapter, PyTorchAutoHooker
from interplens.adapters.gpt2 import GPT2Adapter

# Registry mapping model names to adapter classes
ADAPTER_REGISTRY = {
    "gpt2": GPT2Adapter,
    "gpt2-medium": GPT2Adapter,
    "gpt2-large": GPT2Adapter,
    "gpt2-xl": GPT2Adapter,
}


def register_adapter(name: str, adapter_cls: Type[BaseModelAdapter]):
    """Registers a new model adapter class dynamically."""
    ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter_for_model(
    model_or_name: Any,
    device: str = "auto",
    auto_hook: bool = False,
    tokenizer: Optional[Any] = None,
) -> BaseModelAdapter:
    """Returns appropriate BaseModelAdapter instance for a given string model name or loaded object."""
    if isinstance(model_or_name, str):
        adapter_cls = ADAPTER_REGISTRY.get(model_or_name.lower(), GPT2Adapter)
        return adapter_cls(model_name=model_or_name, device=device)
    
    # If model is already a BaseModelAdapter instance
    if isinstance(model_or_name, BaseModelAdapter):
        return model_or_name
        
    # If custom PyTorch model and auto_hook is requested
    if auto_hook and hasattr(model_or_name, "named_modules"):
        return CustomModelAdapter(model=model_or_name, tokenizer=tokenizer)
        
    # Default in-place adapter for loaded TransformerLens models
    return InPlaceModelAdapter(model_instance=model_or_name)


__all__ = [
    "BaseModelAdapter",
    "InPlaceModelAdapter",
    "CustomModelAdapter",
    "PyTorchAutoHooker",
    "GPT2Adapter",
    "get_adapter_for_model",
    "register_adapter",
]
