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

from interplens.config import settings
from interplens.schema import RunRequest, RunResponse, LogitLensMatrixResponse, ModelInfo
from interplens.utils.device import get_vram_usage, get_optimal_device
from interplens.server.session import global_session_store
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.analysis.logit_lens import compute_logit_lens

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

# Global active model adapter instance
def init_model(model_name: str = "gpt2", device: Optional[Any] = None):
    """Loads target pretrained model into adapter at CLI launch time."""
    global _active_adapter
    try:
        from transformer_lens import HookedTransformer
        if device is None:
            device = get_optimal_device()
        print(f"Loading model '{model_name}' onto {device}...")
        model = HookedTransformer.from_pretrained(model_name, device=device)
        _active_adapter = InPlaceModelAdapter(model, model_name=model_name)
        return _active_adapter
    except Exception as e:
        print(f"Warning: Could not load model '{model_name}': {e}")
        return None


def get_active_adapter():
    """Returns the loaded model adapter for the server session."""
    global _active_adapter
    if _active_adapter is None:
        init_model("gpt2")
    if _active_adapter is None:
        raise HTTPException(
            status_code=500,
            detail="No model is currently loaded. Launch CLI with '--model <name>' (e.g. gpt2, gpt2-medium)."
        )
    return _active_adapter


def set_active_adapter(adapter: InPlaceModelAdapter):
    """Sets a custom loaded model adapter."""
    global _active_adapter
    _active_adapter = adapter


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
    model_name = get_adapter_model_info(adapter).model_name if adapter else "None"

    return {
        "status": "online",
        "device": str(device),
        "active_model": model_name,
        "vram_usage": vram,
        "sessions_cached": len(global_session_store._sessions),
    }


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


# Mount UI static files if directory exists
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui_alt")
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui_root")
