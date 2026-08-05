"""Abstract base class standardizing model interactions across architectures for InterpLens."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import torch


class BaseModelAdapter(ABC):
    """Abstract base class standardizing model loading, tokenization, and hook resolution."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.hidden_dim: int = 0
        self.vocab_size: int = 0
        self._model_instance: Any = None

        # Framework properties
        self.static_fingerprint: Any = None
        self.runtime_fingerprint: Any = None
        self.capabilities: Any = None
        self.engine_capabilities: Any = None
        self.graph: Any = None
        self.report: Any = None
        self.discovery_confidence: float = 1.0

    @abstractmethod
    def load(self) -> None:
        """Loads model into target memory/device."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes input prompt string into formatted list of string token labels."""
        pass

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to human-readable token string."""
        return str(token_ids[0]) if token_ids else ""

    @abstractmethod
    def run_with_cache(self, inputs: Union[str, Dict[str, Any], Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs forward pass under @torch.inference_mode() and returns (logits, activation_cache_dict)."""
        pass

    @abstractmethod
    def get_unembedding_weight(self) -> torch.Tensor:
        """Returns the unembedding matrix tensor W_U [hidden_dim, vocab_size]."""
        pass

    @abstractmethod
    def get_resid_post_hook_name(self, layer: int) -> str:
        """Returns standard hook name for post-layer residual stream vector."""
        pass

    @abstractmethod
    def get_attn_pattern_hook_name(self, layer: int) -> str:
        """Returns standard hook name for layer attention pattern matrix."""
        pass

    @abstractmethod
    def get_mlp_post_hook_name(self, layer: int) -> str:
        """Returns standard hook name for post-MLP activation vector."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Returns summary metadata dictionary for model parameters, layer counts, and capability indicators."""
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "device": str(self.device),
            "discovery_confidence": self.discovery_confidence,
            "static_fingerprint": self.static_fingerprint.to_dict() if self.static_fingerprint else {},
            "runtime_fingerprint": self.runtime_fingerprint.to_dict() if self.runtime_fingerprint else {},
            "capabilities": self.capabilities.to_dict() if self.capabilities else {},
            "engine_capabilities": self.engine_capabilities.to_dict() if self.engine_capabilities else {},
        }
