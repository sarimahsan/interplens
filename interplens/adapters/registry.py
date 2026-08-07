"""Plugin-based Architecture Strategy & Adapter Registry for InterpLens."""

import logging
from typing import Dict, List, Optional, Any, Type, Tuple
import torch.nn as nn

from interplens.adapters.strategy import (
    BaseArchitectureStrategy,
    LlamaStrategy,
    QwenStrategy,
    MistralStrategy,
    GemmaStrategy,
    PhiStrategy,
    GPT2Strategy,
)

logger = logging.getLogger(__name__)


class ArchitectureRegistry:
    """Central registry storing architecture strategy objects for model discovery."""

    def __init__(self):
        self._strategies: Dict[str, BaseArchitectureStrategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Registers built-in architecture strategies."""
        self.register(LlamaStrategy())
        self.register(QwenStrategy())
        self.register(MistralStrategy())
        self.register(GemmaStrategy())
        self.register(PhiStrategy())
        self.register(GPT2Strategy())

    def register(self, strategy: BaseArchitectureStrategy) -> None:
        """Registers a strategy object under its architecture_id."""
        self._strategies[strategy.architecture_id] = strategy
        logger.debug(f"Registered Architecture Strategy: '{strategy.architecture_id}' (family: {strategy.family.value})")

    def get_strategy(self, architecture_id: str) -> Optional[BaseArchitectureStrategy]:
        return self._strategies.get(architecture_id)

    def resolve_strategy(self, model_or_config: Any, model_name: str = "") -> Tuple[BaseArchitectureStrategy, float]:
        """Finds the best matching architecture strategy and returns (strategy, confidence)."""
        best_strategy = None
        best_score = 0.0

        for strat in self._strategies.values():
            score = strat.matches(model_or_config, model_name)
            if score > best_score:
                best_score = score
                best_strategy = strat

        if best_strategy is not None and best_score > 0.0:
            return best_strategy, best_score

        # Default fallback to GPT2Strategy / generic strategy
        fallback = self._strategies.get("gpt2", GPT2Strategy())
        return fallback, 0.4


class AdapterRegistry:
    """Registry mapping string model keys/names to custom BaseModelAdapter subclasses."""

    def __init__(self):
        self._adapters: Dict[str, Type[Any]] = {}

    def register(self, name: str, adapter_cls: Type[Any]) -> None:
        """Registers a custom adapter class under a name."""
        key = name.lower().strip()
        self._adapters[key] = adapter_cls
        logger.info(f"Registered custom adapter class '{adapter_cls.__name__}' for '{key}'.")

    def get_adapter_class(self, name: str) -> Optional[Type[Any]]:
        """Retrieves registered adapter class by name."""
        return self._adapters.get(name.lower().strip())


# Global singletons
global_architecture_registry = ArchitectureRegistry()
global_adapter_registry = AdapterRegistry()


def register_architecture_strategy(strategy: BaseArchitectureStrategy) -> None:
    """Public plugin API for registering custom architecture strategies (e.g. RWKV, Mamba, Hyena)."""
    global_architecture_registry.register(strategy)


def get_strategy_for_model(model_or_config: Any, model_name: str = "") -> Tuple[BaseArchitectureStrategy, float]:
    """Resolves best matching architecture strategy for model instance/config."""
    return global_architecture_registry.resolve_strategy(model_or_config, model_name)


def register_adapter_class(name: str, adapter_cls: Type[Any]) -> None:
    """Registers a custom adapter class in the global adapter registry."""
    global_adapter_registry.register(name, adapter_cls)


def get_registered_adapter_class(name: str) -> Optional[Type[Any]]:
    """Fetches a registered adapter class by name if available."""
    return global_adapter_registry.get_adapter_class(name)

