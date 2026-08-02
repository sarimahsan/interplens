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

