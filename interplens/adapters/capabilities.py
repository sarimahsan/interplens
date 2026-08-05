"""Model and Interpretability Engine capability inspection data structures."""

from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time


class CapabilityLevel(IntEnum):
    """Progressive support levels for model adapters."""
    LEVEL_0_LOADED_ONLY = 0      # Model loaded into memory, no internal hooks
    LEVEL_1_EMBEDDINGS = 1       # Input token embeddings / unembeddings resolved
    LEVEL_2_RESIDUAL_HOOKS = 2   # Residual stream vector hooks resolved per layer
    LEVEL_3_ATTENTION_MAPS = 3   # Head attention weight matrices captured
    LEVEL_4_NEURON_ACTS = 4      # MLP neuron activation tensors captured
    LEVEL_5_FULL_SUPPORT = 5     # All internal representations & interventions supported


class EngineStatus(str, Enum):
    SUPPORTED = "supported"        # 100% operational
    PARTIAL = "partial"            # Operates with limitations
    UNAVAILABLE = "unavailable"    # Feature unsupported for this model setup


@dataclass
class ModelCapability:
    """Raw model internal feature exposure capabilities."""
    has_unembedding: bool = False
    has_residual_stream: bool = False
    has_attention_maps: bool = False
    has_mlp_activations: bool = False
    is_weight_tied: bool = False
    capability_level: CapabilityLevel = CapabilityLevel.LEVEL_0_LOADED_ONLY

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["capability_level"] = int(self.capability_level)
        return d


@dataclass
class EngineCapability:
    """Readiness status for an individual interpretability engine."""
    engine_id: str
    engine_name: str
    status: EngineStatus = EngineStatus.UNAVAILABLE
    reason: str = "Not evaluated"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "status": str(self.status.value),
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class EngineCapabilityMatrix:
    """Matrix of capability readiness across all InterpLens engines."""
    schema_version: int = 1
    generated_by: str = "InterpLens.HookDiscovery"
    timestamp: float = field(default_factory=time.time)
    engines: Dict[str, EngineCapability] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "timestamp": self.timestamp,
            "engines": {k: v.to_dict() for k, v in self.engines.items()},
        }


def evaluate_engine_capabilities(
    model_cap: ModelCapability,
    runtime_fp: Any = None,
    discovery_confidence: float = 1.0,
) -> EngineCapabilityMatrix:
    """Evaluates readiness of all InterpLens interpretability engines based on model capabilities."""
    matrix = EngineCapabilityMatrix()

    # 1. Logit Lens Engine
    if model_cap.has_unembedding and model_cap.has_residual_stream:
        matrix.engines["logit_lens"] = EngineCapability(
            engine_id="logit_lens",
            engine_name="Logit Lens",
            status=EngineStatus.SUPPORTED,
            reason="Unembedding weight matrix W_U and residual stream hooks resolved.",
            confidence=discovery_confidence,
        )
    elif model_cap.has_residual_stream:
        matrix.engines["logit_lens"] = EngineCapability(
            engine_id="logit_lens",
            engine_name="Logit Lens",
            status=EngineStatus.PARTIAL,
            reason="Residual stream hooks resolved, but unembedding W_U matrix inferred via fallback.",
            confidence=0.5,
        )
    else:
        matrix.engines["logit_lens"] = EngineCapability(
            engine_id="logit_lens",
            engine_name="Logit Lens",
            status=EngineStatus.UNAVAILABLE,
            reason="Residual stream or unembedding hooks unavailable.",
            confidence=0.0,
        )

    # 2. Residual Stream Inspector
    if model_cap.has_residual_stream:
        matrix.engines["residual_stream"] = EngineCapability(
            engine_id="residual_stream",
            engine_name="Residual Stream Inspector",
            status=EngineStatus.SUPPORTED,
            reason="Layer hidden state vectors captured across blocks.",
            confidence=discovery_confidence,
        )
    else:
        matrix.engines["residual_stream"] = EngineCapability(
            engine_id="residual_stream",
            engine_name="Residual Stream Inspector",
            status=EngineStatus.UNAVAILABLE,
            reason="Residual stream vector hooks not detected.",
            confidence=0.0,
        )

    # 3. Attention Head Explorer
    if model_cap.has_attention_maps:
        matrix.engines["attention"] = EngineCapability(
            engine_id="attention",
            engine_name="Attention Map Explorer",
            status=EngineStatus.SUPPORTED,
            reason="Per-head query-key attention weight matrices captured.",
            confidence=discovery_confidence,
        )
    else:
        matrix.engines["attention"] = EngineCapability(
            engine_id="attention",
            engine_name="Attention Map Explorer",
            status=EngineStatus.UNAVAILABLE,
            reason="Attention weight matrices not exposed by model architecture (e.g. non-eager attention kernel).",
            confidence=0.0,
        )

    # 4. Neuron Activations Engine
    if model_cap.has_mlp_activations:
        matrix.engines["neurons"] = EngineCapability(
            engine_id="neurons",
            engine_name="Neuron Activation Explorer",
            status=EngineStatus.SUPPORTED,
            reason="Intermediate MLP feed-forward activation tensors captured.",
            confidence=discovery_confidence,
        )
    else:
        matrix.engines["neurons"] = EngineCapability(
            engine_id="neurons",
            engine_name="Neuron Activation Explorer",
            status=EngineStatus.UNAVAILABLE,
            reason="MLP activation tensor hooks not detected.",
            confidence=0.0,
        )

    # 5. Causal Patching Engine
    if model_cap.has_residual_stream:
        matrix.engines["causal_patching"] = EngineCapability(
            engine_id="causal_patching",
            engine_name="Causal Patching Sweep",
            status=EngineStatus.SUPPORTED if model_cap.has_unembedding else EngineStatus.PARTIAL,
            reason="Activation intervention supported on residual stream highway.",
            confidence=discovery_confidence,
        )
    else:
        matrix.engines["causal_patching"] = EngineCapability(
            engine_id="causal_patching",
            engine_name="Causal Patching Sweep",
            status=EngineStatus.UNAVAILABLE,
            reason="Residual stream activation hooks required for activation swapping.",
            confidence=0.0,
        )

    # 6. Induction Head Detector
    if model_cap.has_attention_maps and model_cap.has_residual_stream:
        matrix.engines["induction"] = EngineCapability(
            engine_id="induction",
            engine_name="Induction Head Detector",
            status=EngineStatus.SUPPORTED,
            reason="Full attention pattern matrices captured across layers.",
            confidence=discovery_confidence,
        )
    else:
        matrix.engines["induction"] = EngineCapability(
            engine_id="induction",
            engine_name="Induction Head Detector",
            status=EngineStatus.UNAVAILABLE,
            reason="Requires attention weight matrix maps across model layers.",
            confidence=0.0,
        )

    # 7. GPU Telemetry Profiler
    matrix.engines["gpu_profiler"] = EngineCapability(
        engine_id="gpu_profiler",
        engine_name="GPU Hardware Telemetry",
        status=EngineStatus.SUPPORTED,
        reason="CUDA memory and execution profiler active.",
        confidence=1.0,
    )

    return matrix
