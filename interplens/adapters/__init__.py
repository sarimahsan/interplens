"""Model Adapters package for InterpLens."""

from typing import Any, Optional, Type, Tuple
from interplens.adapters.base import BaseModelAdapter
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.adapters.custom import CustomModelAdapter, PyTorchAutoHooker
from interplens.adapters.generic import GenericAdapter
from interplens.adapters.gpt2 import GPT2Adapter
from interplens.adapters.strategy import BaseArchitectureStrategy, ArchitectureFamily
from interplens.adapters.registry import (
    ArchitectureRegistry,
    global_architecture_registry,
    register_architecture_strategy,
    get_strategy_for_model,
)
from interplens.adapters.fingerprint import StaticFingerprint, RuntimeFingerprint
from interplens.adapters.capabilities import (
    CapabilityLevel,
    ModelCapability,
    EngineStatus,
    EngineCapability,
    EngineCapabilityMatrix,
    evaluate_engine_capabilities,
)
from interplens.adapters.graph import ModelGraph, GraphNode, DiscoveredModule
from interplens.adapters.discovery import HookDiscovery
from interplens.adapters.report import ModelReport, generate_model_report


def get_adapter_for_model(
    model_or_name: Any,
    device: str = "auto",
    auto_hook: bool = False,
    tokenizer: Optional[Any] = None,
) -> BaseModelAdapter:
    """Returns appropriate BaseModelAdapter instance for a given string model name or loaded object."""
    if isinstance(model_or_name, BaseModelAdapter):
        return model_or_name

    if isinstance(model_or_name, str):
        if model_or_name.lower().startswith("gpt2"):
            return GPT2Adapter(model_name=model_or_name, device=device)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tok = AutoTokenizer.from_pretrained(model_or_name, trust_remote_code=True)
            mod = AutoModelForCausalLM.from_pretrained(
                model_or_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device if device != "auto" else "auto",
                attn_implementation="eager",
                trust_remote_code=True,
            )
            return CustomModelAdapter(model=mod, tokenizer=tok, model_name=model_or_name)
        except Exception:
            return GPT2Adapter(model_name=model_or_name, device=device)
        
    if hasattr(model_or_name, "named_modules"):
        m_name = getattr(model_or_name, "name_or_path", getattr(getattr(model_or_name, "config", None), "_name_or_path", "PyTorch-Model"))
        return CustomModelAdapter(model=model_or_name, tokenizer=tokenizer, model_name=str(m_name))

    return InPlaceModelAdapter(model_instance=model_or_name)


def register_adapter(name: str, adapter_cls: Type[BaseModelAdapter]):
    """Registers adapter class dynamically (legacy compatibility)."""
    pass


__all__ = [
    "BaseModelAdapter",
    "InPlaceModelAdapter",
    "CustomModelAdapter",
    "GenericAdapter",
    "GPT2Adapter",
    "PyTorchAutoHooker",
    "BaseArchitectureStrategy",
    "ArchitectureFamily",
    "ArchitectureRegistry",
    "global_architecture_registry",
    "register_architecture_strategy",
    "register_adapter",
    "get_strategy_for_model",
    "StaticFingerprint",
    "RuntimeFingerprint",
    "CapabilityLevel",
    "ModelCapability",
    "EngineStatus",
    "EngineCapability",
    "EngineCapabilityMatrix",
    "evaluate_engine_capabilities",
    "ModelGraph",
    "GraphNode",
    "DiscoveredModule",
    "HookDiscovery",
    "ModelReport",
    "generate_model_report",
    "get_adapter_for_model",
]
