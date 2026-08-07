"""Model Adapters package for InterpLens."""

import logging
from typing import Any, Optional, Type, Tuple

from interplens.exceptions import ModelLoadError
from interplens.adapters.base import BaseModelAdapter
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.adapters.custom import CustomModelAdapter, PyTorchAutoHooker
from interplens.adapters.generic import GenericAdapter
from interplens.adapters.gpt2 import GPT2Adapter
from interplens.adapters.strategy import BaseArchitectureStrategy, ArchitectureFamily
from interplens.adapters.registry import (
    ArchitectureRegistry,
    AdapterRegistry,
    global_architecture_registry,
    global_adapter_registry,
    register_architecture_strategy,
    register_adapter_class,
    get_registered_adapter_class,
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

logger = logging.getLogger(__name__)


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
        # 1. Check custom registered adapters first
        reg_cls = get_registered_adapter_class(model_or_name)
        if reg_cls is not None:
            return reg_cls(model_name=model_or_name, device=device)

        # 2. Built-in GPT2 shortcut
        if model_or_name.lower().startswith("gpt2"):
            return GPT2Adapter(model_name=model_or_name, device=device)

        # 3. Attempt HuggingFace AutoModel loading
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
        except Exception as err:
            raise ModelLoadError(
                f"Failed to load model '{model_or_name}' via HuggingFace AutoModel: {err}. "
                "Ensure model ID is valid, network/auth is accessible, or pass a pre-loaded model instance directly."
            ) from err

    if hasattr(model_or_name, "named_modules"):
        m_name = getattr(model_or_name, "name_or_path", getattr(getattr(model_or_name, "config", None), "_name_or_path", "PyTorch-Model"))
        return CustomModelAdapter(model=model_or_name, tokenizer=tokenizer, model_name=str(m_name))

    return InPlaceModelAdapter(model_instance=model_or_name)


def register_adapter(name: str, adapter_cls: Type[BaseModelAdapter]):
    """Registers custom adapter class dynamically in the adapter registry."""
    register_adapter_class(name, adapter_cls)


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
    "AdapterRegistry",
    "global_architecture_registry",
    "global_adapter_registry",
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

