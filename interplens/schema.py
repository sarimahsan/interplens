"""Pydantic schemas and DTOs for InterpLens API and backend data structures."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata describing a loaded or supported model."""
    model_name: str
    num_layers: int
    num_heads: int
    hidden_dim: int
    vocab_size: int
    device: str
    is_custom: bool = False


class RunRequest(BaseModel):
    """Payload to run model forward pass and cache activations."""
    prompt: str
    model_name: Optional[str] = None
    corrupted_prompt: Optional[str] = None


class RunResponse(BaseModel):
    """Response returned after executing forward pass."""
    session_id: str
    prompt: str
    tokens: List[str]
    model_info: ModelInfo
    vram_usage: Dict[str, Any]


class LogitLensToken(BaseModel):
    """Top token prediction at a specific layer."""
    token: str
    token_id: int
    probability: float
    rank: int
    logit: Optional[float] = None


class LogitLensLayerResult(BaseModel):
    """Logit lens predictions for one layer at one token position."""
    layer: int
    entropy: float = 0.0
    kl_divergence: float = 0.0
    top_tokens: List[LogitLensToken]


class LogitLensResponse(BaseModel):
    """Full Logit Lens analysis payload for a specific position across layers."""
    session_id: str
    position: int
    selected_token: str
    layers: List[LogitLensLayerResult]


class PositionLogitLensData(BaseModel):
    """Logit lens predictions for one token position across all layers."""
    position: int
    token: str
    layers: List[LogitLensLayerResult]
    target_token_ranks: Optional[List[int]] = None
    top5_trajectories: Optional[Dict[str, List[float]]] = None


class LogitLensMatrixResponse(BaseModel):
    """Full 2D grid matrix of Logit Lens predictions across all positions and layers."""
    session_id: str
    prompt: str
    tokens: List[str]
    num_layers: int
    positions: List[PositionLogitLensData]


class SteeringRequest(BaseModel):
    """Payload to inject a steering vector into a target layer residual stream."""
    prompt: str
    target_layer: int = 0
    multiplier: float = 1.0
    steering_vector: Optional[List[float]] = None


class AttentionLink(BaseModel):
    """Represents a directional query-to-key connection link for arc diagram visualizer."""
    source: int
    target: int
    source_token: str
    target_token: str
    weight: float


class AttentionHeadResponse(BaseModel):
    """Full response payload for Attention Head Explorer (matrix, grid, arc links)."""
    session_id: str
    prompt: str
    tokens: List[str]
    layer: int
    head: int
    num_heads: int
    num_layers: int
    matrix: List[List[float]]  # NxN matrix for target head
    grid: Optional[List[List[List[float]]]] = None  # H x N x N for all heads in layer
    arc_links: List[AttentionLink]


class NeuronInfo(BaseModel):
    """Metadata describing a top firing MLP neuron."""
    neuron_idx: int
    layer: int
    activation: float


class NeuronLightingToken(BaseModel):
    """Single token activation score for single neuron prompt lighting strip visualizer."""
    position: int
    token: str
    activation: float


class NeuronAnalysisResponse(BaseModel):
    """Full payload for Neuron Activation Explorer (top-K firing neurons & text lighting strip)."""
    session_id: str
    prompt: str
    tokens: List[str]
    layer: int
    position: int
    selected_token: str
    d_mlp: int
    top_neurons: List[NeuronInfo]
    selected_neuron_idx: int
    selected_neuron_activation: float
    lighting_strip: List[NeuronLightingToken]


class TokenAttributionScore(BaseModel):
    """Attribution score for input prompt token."""
    position: int
    token: str
    score: float
    raw_score: float


class TokenAttributionResponse(BaseModel):
    """Full payload for Token Attribution Engine (Attention Rollout)."""
    session_id: str
    prompt: str
    tokens: List[str]
    target_position: int
    target_token: str
    method: str
    attributions: List[TokenAttributionScore]


class CausalPatchingRequest(BaseModel):
    """Payload request for clean vs corrupted activation patching sweep."""
    clean_prompt: str
    corrupt_prompt: str
    target_token: Optional[str] = None


class CausalTracingCell(BaseModel):
    """Single cell metadata in layer x position causal patching matrix."""
    layer: int
    position: int
    clean_token: str
    corrupt_token: str
    logit_diff_recovery: float
    patched_logit_diff: float


class CausalTracingResponse(BaseModel):
    """Full payload for Automated Causal Interventions & ROME Causal Tracing Sweep."""
    clean_prompt: str
    corrupt_prompt: str
    clean_tokens: List[str]
    corrupt_tokens: List[str]
    target_token: str
    baseline_clean_logit_diff: float
    baseline_corrupt_logit_diff: float
    num_layers: int
    seq_len: int
    max_recovery_layer: int
    max_recovery_position: int
    max_recovery_percentage: float
    heatmap_matrix: List[List[float]]  # num_layers x seq_len recovery matrix
    cells: List[CausalTracingCell]


class InductionHeadScore(BaseModel):
    """Single attention head induction score metadata."""
    layer: int
    head: int
    score: float
    is_induction_head: bool


class InductionDetectorResponse(BaseModel):
    """Full payload for Induction Head Auto-Detector sweep."""
    total_heads_scanned: int
    flagged_count: int
    num_layers: int
    num_heads: int
    top_induction_heads: List[InductionHeadScore]
    matrix_scores: List[List[float]]  # num_layers x num_heads matrix
    tokens_used: List[str]
    prompt_used: str


