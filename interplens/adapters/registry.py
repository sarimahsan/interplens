"""Plugin-based Architecture Strategy Registry for InterpLens."""

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
        print(f"📦 Registered Architecture Strategy: '{strategy.architecture_id}' (family: {strategy.family.value})")

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


# Global architecture strategy registry singleton
global_architecture_registry = ArchitectureRegistry()


def register_architecture_strategy(strategy: BaseArchitectureStrategy) -> None:
    """Public plugin API for registering custom architecture strategies (e.g. RWKV, Mamba, Hyena)."""
    global_architecture_registry.register(strategy)


def get_strategy_for_model(model_or_config: Any, model_name: str = "") -> Tuple[BaseArchitectureStrategy, float]:
    """Resolves best matching architecture strategy for model instance/config."""
    return global_architecture_registry.resolve_strategy(model_or_config, model_name)
