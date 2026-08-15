"""Analysis package for InterpLens interpretability engines."""

from interplens.analysis.logit_lens import compute_logit_lens
from interplens.analysis.residual_stream import compute_residual_metrics, apply_activation_steering
from interplens.analysis.attention_heads import compute_attention_metrics
from interplens.analysis.neurons import compute_neuron_activations
from interplens.analysis.attribution import compute_token_attributions
from interplens.analysis.causal_patching import run_causal_patching_sweep
from interplens.analysis.induction_heads import detect_induction_heads
from interplens.analysis.topology import inspect_model_topology

__all__ = [
    "compute_logit_lens",
    "compute_residual_metrics",
    "apply_activation_steering",
    "compute_attention_metrics",
    "compute_neuron_activations",
    "compute_token_attributions",
    "run_causal_patching_sweep",
    "detect_induction_heads",
    "inspect_model_topology",
]
