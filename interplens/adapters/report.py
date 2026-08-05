"""Model Report generator for automated model inspection summaries."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time

from interplens.adapters.fingerprint import StaticFingerprint, RuntimeFingerprint
from interplens.adapters.capabilities import ModelCapability, EngineCapabilityMatrix, CapabilityLevel


@dataclass
class ModelReport:
    """Diagnostic model inspection report generated on model loading."""
    model_name: str
    architecture_id: str
    family: str
    discovery_confidence: float
    capability_level: int
    capability_level_name: str
    static_fingerprint: Dict[str, Any]
    runtime_fingerprint: Dict[str, Any]
    model_capabilities: Dict[str, Any]
    engine_capabilities: Dict[str, Any]
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_text_report(self) -> str:
        """Returns clean human-readable ASCII terminal report."""
        lines = []
        lines.append("=" * 65)
        lines.append(f"🔍 InterpLens Model Discovery Report: {self.model_name}")
        lines.append("=" * 65)
        lines.append(f"Architecture Strategy : {self.architecture_id.upper()} ({self.family})")
        lines.append(f"Discovery Confidence  : {self.discovery_confidence * 100:.1f}%")
        lines.append(f"Progressive Level     : Level {self.capability_level} ({self.capability_level_name})")
        lines.append("-" * 65)
        lines.append("📐 Geometry & Parameters:")
        sf = self.static_fingerprint
        lines.append(f"  • Layers        : {sf.get('num_layers', 0)}")
        lines.append(f"  • Heads         : {sf.get('num_heads', 0)} (KV Heads: {sf.get('num_kv_heads') or sf.get('num_heads', 0)})")
        lines.append(f"  • Hidden Dim    : {sf.get('hidden_size', 0)}")
        lines.append(f"  • Vocab Size    : {sf.get('vocab_size', 0)}")
        lines.append(f"  • RoPE / MoE    : RoPE: {sf.get('has_rope')}, MoE: {sf.get('is_moe')}")
        lines.append("-" * 65)
        lines.append("⚡ Engine Capability Matrix:")
        ec = self.engine_capabilities.get("engines", {})
        for eng_id, eng in ec.items():
            status = eng.get("status", "unavailable").upper()
            icon = "✓" if status == "SUPPORTED" else ("⚠" if status == "PARTIAL" else "✗")
            lines.append(f"  {icon} {eng.get('engine_name', eng_id):<26} : [{status}] — {eng.get('reason')}")
        lines.append("=" * 65)
        return "\n".join(lines)


def generate_model_report(
    model_name: str,
    strategy_id: str,
    family: str,
    confidence: float,
    static_fp: StaticFingerprint,
    runtime_fp: RuntimeFingerprint,
    model_cap: ModelCapability,
    engine_matrix: EngineCapabilityMatrix,
) -> ModelReport:
    """Generates a ModelReport instance."""
    level_names = {
        0: "Loaded Only",
        1: "Embeddings",
        2: "Residual Hooks",
        3: "Attention Maps",
        4: "Neuron Activations",
        5: "Full Support",
    }
    cap_level = int(model_cap.capability_level)

    return ModelReport(
        model_name=model_name,
        architecture_id=strategy_id,
        family=family,
        discovery_confidence=confidence,
        capability_level=cap_level,
        capability_level_name=level_names.get(cap_level, "Unknown"),
        static_fingerprint=static_fp.to_dict(),
        runtime_fingerprint=runtime_fp.to_dict(),
        model_capabilities=model_cap.to_dict(),
        engine_capabilities=engine_matrix.to_dict(),
    )
