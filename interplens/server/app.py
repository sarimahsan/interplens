"""FastAPI web server for InterpLens UI and REST API.

Provides endpoints for running hooked model forward passes, retrieving activation caches,
and performing real-time interpretability analysis (Logit Lens, etc.).
"""

import os
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

try:
    from interplens.config import settings
    from interplens.schema import RunRequest, RunResponse, LogitLensMatrixResponse, ModelInfo, SteeringRequest, CausalPatchingRequest
    from interplens.utils.device import get_vram_usage, get_optimal_device
    from interplens.server.session import global_session_store
    from interplens.adapters.inplace import InPlaceModelAdapter
    from interplens.analysis.logit_lens import compute_logit_lens
except ImportError:
    # Fallback if imported from within the interplens package directory directly
    from config import settings
    from schema import RunRequest, RunResponse, LogitLensMatrixResponse, ModelInfo, SteeringRequest, CausalPatchingRequest
    from utils.device import get_vram_usage, get_optimal_device
    from server.session import global_session_store
    from adapters.inplace import InPlaceModelAdapter
    from analysis.logit_lens import compute_logit_lens

app = FastAPI(
    title="InterpLens Debugger API",
    description="Interactive Mechanistic Interpretability API and Web Debugger",
    version="0.1.0",
)

# Enable CORS for local web dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_active_adapter = None
_model_loading_status = {"status": "idle", "model_name": "none", "error": None}


# Global active model adapter instance
def init_model(model_name: str = "gpt2", device: Optional[Any] = None, hf_token: Optional[str] = None):
    """Loads target model into GPU VRAM."""
    global _active_adapter, _model_loading_status
    if device is None:
        device = get_optimal_device()

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    # 0. Check if model is ALREADY loaded in GPU VRAM
    if _active_adapter is not None:
        curr_name = getattr(_active_adapter, "model_name", "")
        if curr_name.lower() == model_name.lower():
            print(f"ℹ️ Model '{model_name}' is already loaded in GPU VRAM. Skipping reload.")
            _model_loading_status["status"] = "online"
            _model_loading_status["model_name"] = model_name
            return _active_adapter
        else:
            # Free previous model from GPU VRAM before loading new model
            print(f"🧹 Clearing previous model '{curr_name}' from VRAM...")
            del _active_adapter
            _active_adapter = None
            from interplens.utils.device import free_gpu_memory
            free_gpu_memory()

    _model_loading_status["status"] = "loading"
    _model_loading_status["model_name"] = model_name
    _model_loading_status["error"] = None

    print(f"⚡ Loading model '{model_name}' onto {device}...")

    # Known native TransformerLens models
    tl_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "stanford-gpt2-small-a"]
    is_tl_native = any(m == model_name.lower() or model_name.lower().startswith("pythia") for m in tl_models)

    if is_tl_native:
        try:
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained(model_name, device=device)
            _active_adapter = InPlaceModelAdapter(model, model_name=model_name)
            _model_loading_status["status"] = "online"
            print(f"✅ Loaded '{model_name}' via TransformerLens on {device}")
            return _active_adapter
        except Exception as e1:
            print(f"Notice: TransformerLens could not load '{model_name}' ({e1}). Falling back to HuggingFace...")

    # Direct HuggingFace AutoModelForCausalLM loader for Qwen, Llama, Mistral, Gemma, etc.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from interplens.adapters.custom import CustomModelAdapter
        
        print(f"📥 Loading HuggingFace AutoModel '{model_name}' directly onto GPU (fp16, eager attention)...")
        token_kwargs = {"token": token} if token else {}
        
        tokenizer = None
        tokenizer_warning = None

        # 1. Primary Tokenizer Resolution
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
        except Exception as t_err1:
            print(f"⚠️ Primary tokenizer for '{model_name}' could not be loaded directly ({t_err1}).")
            m_lower = model_name.lower()
            fallback_repo = None
            if "llama" in m_lower:
                fallback_repo = "meta-llama/Llama-3.2-1B"
            elif "qwen" in m_lower:
                fallback_repo = "Qwen/Qwen2.5-0.5B"
            elif "gemma" in m_lower:
                fallback_repo = "google/gemma-2b"
            elif "phi" in m_lower:
                fallback_repo = "microsoft/phi-2"

            if fallback_repo:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(fallback_repo, trust_remote_code=True, **token_kwargs)
                    tokenizer_warning = f"Primary tokenizer unavailable. Automatically resolved compatible tokenizer '{fallback_repo}'."
                    print(f"✅ Automatically resolved compatible tokenizer '{fallback_repo}'!")
                except Exception as t_err2:
                    print(f"Notice: Fallback tokenizer '{fallback_repo}' also unavailable ({t_err2}).")

            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    tokenizer_warning = f"Using generic fallback tokenizer for '{model_name}'."
                except Exception:
                    tokenizer_warning = f"No tokenizer found for '{model_name}'. Operating with raw token ID indexing."

        # 2. Model Weight Loading
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
                device_map="cuda" if str(device).startswith("cuda") else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",
                **token_kwargs
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
                device_map="cuda" if str(device).startswith("cuda") else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **token_kwargs
            )

        if hasattr(model, "config"):
            model.config.output_attentions = True

        _active_adapter = CustomModelAdapter(model=model, tokenizer=tokenizer, model_name=model_name)
        _model_loading_status["status"] = "online"
        _model_loading_status["warning"] = tokenizer_warning
        print(f"✅ Loaded '{model_name}' directly into CUDA GPU VRAM!")
        if tokenizer_warning:
            print(f"💡 Notice: {tokenizer_warning}")
        return _active_adapter
        return _active_adapter
    except Exception as e2:
        err_msg = f"Failed to load model '{model_name}': {e2}"
        _model_loading_status["status"] = "error"
        _model_loading_status["error"] = err_msg
        print(f"❌ Error: {err_msg}")
        return None


def get_active_adapter():
    """Returns the loaded model adapter for the server session."""
    global _active_adapter, _model_loading_status
    if _active_adapter is not None:
        return _active_adapter

    # If a model is currently downloading/loading in background thread, wait for completion
    if _model_loading_status.get("status") == "loading":
        model_name = _model_loading_status.get("model_name", "target model")
        import time
        for _ in range(120):
            time.sleep(0.5)
            if _active_adapter is not None:
                return _active_adapter
            if _model_loading_status.get("status") == "error":
                err = _model_loading_status.get("error") or "Failed to load model."
                raise HTTPException(status_code=500, detail=f"Model loading error: {err}")

        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' is currently downloading/loading into GPU VRAM. Please wait a few seconds."
        )

    target_name = _model_loading_status.get("model_name")
    if not target_name or target_name == "none":
        target_name = "gpt2"

    init_model(target_name)

    if _active_adapter is None:
        err = _model_loading_status.get("error") or "No model loaded."
        raise HTTPException(status_code=500, detail=f"Model loading error: {err}")
    return _active_adapter


def set_active_adapter(adapter: InPlaceModelAdapter):
    """Sets a custom loaded model adapter."""
    global _active_adapter, _model_loading_status
    _active_adapter = adapter
    _model_loading_status["status"] = "online"
    _model_loading_status["model_name"] = getattr(adapter, "model_name", "custom")


def get_adapter_model_info(adapter) -> ModelInfo:
    if hasattr(adapter, "model_info") and isinstance(adapter.model_info, ModelInfo):
        return adapter.model_info
    info_dict = adapter.get_model_info()
    return ModelInfo(
        model_name=info_dict.get("model_name", "custom"),
        num_layers=info_dict.get("num_layers", 0),
        num_heads=info_dict.get("num_heads", 0),
        hidden_dim=info_dict.get("hidden_dim", 0),
        vocab_size=info_dict.get("vocab_size", 0),
        device=str(adapter.device),
        is_custom=info_dict.get("is_custom", False),
    )


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Returns system hardware info, device allocation, and VRAM status."""
    device = get_optimal_device()
    vram = get_vram_usage(device)
    
    adapter = _active_adapter
    model_name = get_adapter_model_info(adapter).model_name if adapter else _model_loading_status.get("model_name", "None")

    return {
        "status": _model_loading_status["status"] if _active_adapter is None else "online",
        "device": str(device),
        "active_model": model_name,
        "vram_usage": vram,
        "warning": _model_loading_status.get("warning"),
        "error": _model_loading_status.get("error"),
        "sessions_cached": len(global_session_store._sessions),
    }


@app.get("/api/hardware/gpu-status")
def get_gpu_status() -> Dict[str, Any]:
    """Returns 32-block VRAM memory grid allocation and CUDA compute metrics."""
    from interplens.utils.device import get_gpu_grid_status
    device = get_optimal_device()
    return get_gpu_grid_status(device)


@app.get("/api/hardware/gpu-profiler")
def get_gpu_profiler(session_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Returns granular GPU hardware specs, 64-block memory topology, and per-layer activation memory breakdown."""
    from interplens.utils.device import get_detailed_gpu_profiler
    adapter = _active_adapter
    cache = None
    if session_id:
        sess = global_session_store.get_session(session_id)
        if sess:
            cache = sess.cache
    prof = get_detailed_gpu_profiler(adapter, cache)
    prof["sessions"] = global_session_store.get_sessions_metadata()
    prof["max_sessions"] = global_session_store.max_sessions
    prof["request_history"] = global_session_store.request_history
    return prof


@app.get("/api/sessions")
def get_active_sessions():
    """Returns metadata for all cached activation sessions in LRU memory."""
    return {"sessions": global_session_store.get_sessions_metadata(), "max_sessions": global_session_store.max_sessions}


@app.delete("/api/sessions/{session_id}")
def evict_session(session_id: str):
    """Manually evicts a session from LRU memory cache and clears GPU VRAM."""
    success = global_session_store.evict_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or already evicted.")
    return {"status": "evicted", "session_id": session_id}


@app.post("/api/run", response_model=RunResponse)
def run_prompt(req: RunRequest):
    """Runs forward pass on prompt, caches activation tensors, and returns session ID."""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")

    adapter = get_active_adapter()

    # Create session and store in LRU session store
    session = global_session_store.create_session(adapter, req.prompt)
    model_info = get_adapter_model_info(adapter)
    vram = get_vram_usage(adapter.device)

    return RunResponse(
        session_id=session.session_id,
        prompt=session.prompt,
        tokens=session.tokens,
        model_info=model_info,
        vram_usage=vram,
    )


@app.get("/api/analysis/logit-lens", response_model=LogitLensMatrixResponse)
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


@app.get("/api/analysis/residual-stream")
def get_residual_stream_metrics(
    session_id: str = Query(..., description="Active session ID"),
    position: Optional[int] = Query(None, description="Optional token position index"),
):
    """Computes residual stream L2 norms and layer-by-layer cosine similarity matrices for a session."""
    session = global_session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    adapter = session.adapter or get_active_adapter()
    from interplens.analysis.residual_stream import compute_residual_metrics
    return compute_residual_metrics(
        adapter=adapter,
        cache=session.cache,
        tokens=session.tokens,
        session_id=session.session_id,
        position=position,
    )


@app.post("/api/analysis/residual-stream/steer")
def steer_residual_stream(req: SteeringRequest):
    """Injects activation steering vector into target layer during forward pass."""
    adapter = get_active_adapter()
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")

    info = get_adapter_model_info(adapter)
    vec = req.steering_vector
    if vec is None:
        vec = [0.1] * info.hidden_dim

    from interplens.analysis.residual_stream import apply_activation_steering
    return apply_activation_steering(
        adapter=adapter,
        prompt=req.prompt,
        target_layer=req.target_layer,
        steering_vector=vec,
        multiplier=req.multiplier,
    )


@app.get("/api/analysis/attention")
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
    try:
        from interplens.analysis.attention_heads import compute_attention_metrics
    except ImportError:
        from analysis.attention_heads import compute_attention_metrics

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


@app.get("/api/analysis/neurons")
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
    try:
        from interplens.analysis.neurons import compute_neuron_activations
    except ImportError:
        from analysis.neurons import compute_neuron_activations

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


@app.get("/api/analysis/attribution")
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
    try:
        from interplens.analysis.attribution import compute_token_attributions
    except ImportError:
        from analysis.attribution import compute_token_attributions

    return compute_token_attributions(
        adapter=adapter,
        cache=session.cache,
        tokens=session.tokens,
        session_id=session.session_id,
        prompt=session.prompt,
        position=position,
        method=method,
    )


@app.get("/api/model/topology")
def get_model_topology():
    """Inspects active model parameters and builds a node diagram specification for the UI."""
    adapter = get_active_adapter()
    from interplens.analysis.topology import inspect_model_topology
    return inspect_model_topology(adapter)


@app.post("/api/analysis/causal-patching")
def run_causal_patching(req: CausalPatchingRequest):
    """Runs a full layer x position activation patching sweep between clean and corrupted prompts."""
    if not req.clean_prompt or not req.corrupt_prompt:
        raise HTTPException(status_code=400, detail="Both 'clean_prompt' and 'corrupt_prompt' must be provided.")

    adapter = get_active_adapter()
    try:
        from interplens.analysis.causal_patching import run_causal_patching_sweep
    except ImportError:
        from analysis.causal_patching import run_causal_patching_sweep

    return run_causal_patching_sweep(
        adapter=adapter,
        clean_prompt=req.clean_prompt,
        corrupt_prompt=req.corrupt_prompt,
        target_token_str=req.target_token,
    )


@app.get("/api/analysis/induction-heads")
def run_induction_detector(
    sequence_length: int = Query(20, description="Length of random word sequence to duplicate S_1 S_2"),
    threshold: float = Query(0.15, description="Induction score threshold for flagging active induction heads"),
    top_k: int = Query(10, description="Top-K ranked induction heads to return"),
):
    """Runs automated repeated sequence test (S_1 S_2) and returns induction scores for all heads."""
    adapter = get_active_adapter()
    try:
        from interplens.analysis.induction_heads import detect_induction_heads
    except ImportError:
        from analysis.induction_heads import detect_induction_heads

    return detect_induction_heads(
        adapter=adapter,
        sequence_length=sequence_length,
        threshold=threshold,
        top_k=top_k,
    )


# Mount UI static files if directory exists
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui_alt")
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui_root")
