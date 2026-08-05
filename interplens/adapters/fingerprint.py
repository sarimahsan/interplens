"""Fingerprint data structures for static model architecture and runtime execution environment."""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any


@dataclass
class StaticFingerprint:
    """Invariant static parameters describing model architecture geometry."""
    architecture: str = "unknown"
    family: str = "transformer"  # transformer, state_space, moe, rnn, cnn, etc.
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: Optional[int] = None
    vocab_size: int = 0
    has_rope: bool = False
    is_moe: bool = False
    multimodal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeFingerprint:
    """Dynamic execution environment parameters for active loaded model instance."""
    device: str = "cpu"
    dtype: str = "float32"
    quantization: str = "none"  # none, int8, int4, fp8
    flash_attention: bool = False
    is_compiled: bool = False
    kv_cache_enabled: bool = False
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
