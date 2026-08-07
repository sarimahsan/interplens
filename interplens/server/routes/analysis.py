"""FastAPI Router for Interpretability Analysis Endpoints."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from interplens.schema import LogitLensMatrixResponse, SteeringRequest, CausalPatchingRequest
from interplens.server.session import global_session_store
from interplens.server.state import get_active_adapter, get_adapter_model_info
from interplens.analysis.logit_lens import compute_logit_lens
from interplens.analysis.residual_stream import compute_residual_metrics, apply_activation_steering
from interplens.analysis.attention_heads import compute_attention_metrics
from interplens.analysis.neurons import compute_neuron_activations
from interplens.analysis.attribution import compute_token_attributions
from interplens.analysis.topology import inspect_model_topology
from interplens.analysis.causal_patching import run_causal_patching_sweep
from interplens.analysis.induction_heads import detect_induction_heads

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


@router.get("/logit-lens", response_model=LogitLensMatrixResponse)
def get_logit_lens(
    session_id: str = Query(..., description="Active session ID"),
    top_k: int = Query(5, ge=1, le=20, description="Top-K token predictions per layer"),
    apply_ln: bool = Query(True, description="Whether to apply final LayerNorm"),
    position: Optional[int] = Query(None, description="Optional position filter"),
):
    """Computes Logit Lens predictions across all positions and layers for a cached session."""
    session = global_session_store.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or evicted from session store."
        )

    adapter = session.adapter or get_active_adapter()

    try:
        res = compute_logit_lens(
            adapter=adapter,
            cache=session.cache,
            tokens=session.tokens,
            session_id=session.session_id,
            prompt=session.prompt,
            top_k=top_k,
            apply_ln=apply_ln,
            position=position,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing Logit Lens: {str(e)}"
        )


@router.get("/residual-stream")
def get_residual_stream_metrics(
    session_id: str = Query(..., description="Active session ID"),
    position: Optional[int] = Query(None, description="Optional token position index"),
):
    """Computes residual stream L2 norms and layer-by-layer cosine similarity matrices for a session."""
    session = global_session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    adapter = session.adapter or get_active_adapter()
    return compute_residual_metrics(
        adapter=adapter,
        cache=session.cache,
        tokens=session.tokens,
        session_id=session.session_id,
        position=position,
    )


@router.post("/residual-stream/steer")
def steer_residual_stream(req: SteeringRequest):
    """Injects activation steering vector into target layer during forward pass."""
    adapter = get_active_adapter()
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")

    info = get_adapter_model_info(adapter)
    vec = req.steering_vector
    if vec is None:
        vec = [0.1] * info.hidden_dim

    return apply_activation_steering(
        adapter=adapter,
        prompt=req.prompt,
        target_layer=req.target_layer,
        steering_vector=vec,
        multiplier=req.multiplier,
    )


@router.get("/attention")
def get_attention_heads(
    session_id: str = Query(..., description="Active session ID"),
    layer: int = Query(0, ge=0, description="Target layer index"),
    head: int = Query(0, ge=0, description="Target head index"),
    threshold: float = Query(0.02, ge=0.0, le=1.0, description="Arc connection link weight threshold"),
):
    """Computes N x N attention matrix, multi-head grid, and arc diagram links for a target layer and head."""
    session = global_session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    adapter = session.adapter or get_active_adapter()
    return compute_attention_metrics(
        adapter=adapter,
        cache=session.cache,
        tokens=session.tokens,
        session_id=session.session_id,
        prompt=session.prompt,
        layer=layer,
        head=head,
        threshold=threshold,
    )


@router.get("/neurons")
def get_neuron_activations(
    session_id: str = Query(..., description="Active session ID"),
    layer: int = Query(0, ge=0, description="Target layer index"),
    position: Optional[int] = Query(None, description="Optional target token position index"),
    top_k: int = Query(10, ge=1, le=50, description="Top-K highest firing neurons"),
    neuron_idx: Optional[int] = Query(None, ge=0, description="Optional target neuron index for lighting strip"),
):
    """Computes top-K firing MLP neurons for token position and prompt text activation lighting strip."""
    session = global_session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    adapter = session.adapter or get_active_adapter()
    return compute_neuron_activations(
        adapter=adapter,
        cache=session.cache,
        tokens=session.tokens,
        session_id=session.session_id,
        prompt=session.prompt,
        layer=layer,
        position=position,
        top_k=top_k,
        neuron_idx=neuron_idx,
    )


@router.get("/attribution")
def get_token_attributions(
    session_id: str = Query(..., description="Active session ID"),
    position: Optional[int] = Query(None, description="Optional target token position index"),
    method: str = Query("attention_rollout", description="Attribution calculation method"),
):
    """Computes input token attribution scores using Attention Rollout across layers."""
    session = global_session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    adapter = session.adapter or get_active_adapter()
    return compute_token_attributions(
        adapter=adapter,
        cache=session.cache,
        tokens=session.tokens,
        session_id=session.session_id,
        prompt=session.prompt,
        position=position,
        method=method,
    )


@router.get("/topology")
def get_model_topology_endpoint():
    """Inspects active model parameters and builds a node diagram specification for the UI."""
    adapter = get_active_adapter()
    return inspect_model_topology(adapter)


@router.post("/causal-patching")
def run_causal_patching(req: CausalPatchingRequest):
    """Runs a full layer x position activation patching sweep between clean and corrupted prompts."""
    if not req.clean_prompt or not req.corrupt_prompt:
        raise HTTPException(status_code=400, detail="Both 'clean_prompt' and 'corrupt_prompt' must be provided.")

    adapter = get_active_adapter()
    return run_causal_patching_sweep(
        adapter=adapter,
        clean_prompt=req.clean_prompt,
        corrupt_prompt=req.corrupt_prompt,
        target_token_str=req.target_token,
    )


@router.get("/induction-heads")
def run_induction_detector(
    sequence_length: int = Query(20, description="Length of random word sequence to duplicate S_1 S_2"),
    threshold: float = Query(0.15, description="Induction score threshold for flagging active induction heads"),
    top_k: int = Query(10, description="Top-K ranked induction heads to return"),
):
    """Runs automated repeated sequence test (S_1 S_2) and returns induction scores for all heads."""
    adapter = get_active_adapter()
    return detect_induction_heads(
        adapter=adapter,
        sequence_length=sequence_length,
        threshold=threshold,
        top_k=top_k,
    )
