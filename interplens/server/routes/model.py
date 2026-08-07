"""FastAPI Router for Model & Hardware Status Endpoints."""

import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from interplens.schema import RunRequest, RunResponse
from interplens.utils.device import get_optimal_device, get_vram_usage, get_gpu_grid_status, get_detailed_gpu_profiler
from interplens.utils.pdf_report import generate_model_report_pdf
from interplens.server.session import global_session_store
from interplens.server.state import (
    state_manager,
    get_active_adapter,
    get_adapter_model_info,
)
from interplens.analysis.topology import inspect_model_topology

router = APIRouter(tags=["Model"])


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint for live VRAM and model telemetry updates."""
    await websocket.accept()
    try:
        while True:
            device = get_optimal_device()
            vram = get_vram_usage(device)
            adapter = state_manager.active_adapter
            status_info = state_manager.status
            model_info = get_adapter_model_info(adapter) if adapter else None
            model_name = model_info.model_name if model_info else status_info.get("model_name", "None")

            data = {
                "status": status_info.get("status", "idle") if adapter is None else "online",
                "device": str(device),
                "active_model": model_name,
                "vram_usage": vram,
                "warning": status_info.get("warning"),
                "error": status_info.get("error"),
                "sessions_cached": len(global_session_store._sessions),
                "timestamp": time.time(),
            }
            await websocket.send_json(data)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@router.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Returns system hardware info, device allocation, and VRAM status."""
    device = get_optimal_device()
    vram = get_vram_usage(device)
    
    adapter = state_manager.active_adapter
    status_info = state_manager.status
    model_info = get_adapter_model_info(adapter) if adapter else None
    model_name = model_info.model_name if model_info else status_info.get("model_name", "None")

    return {
        "status": status_info.get("status", "idle") if adapter is None else "online",
        "device": str(device),
        "active_model": model_name,
        "vram_usage": vram,
        "warning": status_info.get("warning"),
        "error": status_info.get("error"),
        "sessions_cached": len(global_session_store._sessions),
        "engine_capabilities": model_info.engine_capabilities if model_info else None,
        "discovery_confidence": model_info.discovery_confidence if model_info else 1.0,
    }


@router.get("/api/model/report")
def get_model_report():
    """Returns automated model inspection discovery report for active adapter."""
    adapter = get_active_adapter()
    report = getattr(adapter, "report", None)
    if report is not None:
        res = report.to_dict()
        res["text_report"] = report.format_text_report()
        return res
    raise HTTPException(status_code=404, detail="Model discovery report unavailable.")


@router.get("/api/model/report/pdf")
def get_model_report_pdf():
    """Generates and downloads production-grade PDF inspection report for active model."""
    adapter = get_active_adapter()
    report = getattr(adapter, "report", None)
    if report is not None:
        report_dict = report.to_dict()
        try:
            pdf_bytes = generate_model_report_pdf(report_dict)
        except RuntimeError as err:
            raise HTTPException(status_code=501, detail=str(err))
        filename = f"InterpLens_Model_Report_{report.model_name}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    raise HTTPException(status_code=404, detail="Model discovery report unavailable.")


@router.get("/api/model/topology")
def get_model_topology() -> Dict[str, Any]:
    """Inspects active model parameters and builds a node diagram specification for the UI."""
    adapter = get_active_adapter()
    return inspect_model_topology(adapter)


@router.get("/api/hardware/gpu-status")
def get_gpu_status() -> Dict[str, Any]:
    """Returns 32-block VRAM memory grid allocation and CUDA compute metrics."""
    device = get_optimal_device()
    return get_gpu_grid_status(device)


@router.get("/api/hardware/gpu-profiler")
def get_gpu_profiler(session_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Returns granular GPU hardware specs, memory topology, and per-layer activation breakdown."""
    adapter = state_manager.active_adapter
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


@router.post("/api/run", response_model=RunResponse)
def run_prompt(req: RunRequest):
    """Runs forward pass on prompt, caches activation tensors, and returns session ID."""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")

    adapter = get_active_adapter()

    session = global_session_store.create_session(adapter, req.prompt, corrupted_prompt=req.corrupted_prompt)
    model_info = get_adapter_model_info(adapter)
    vram = get_vram_usage(adapter.device)

    return RunResponse(
        session_id=session.session_id,
        prompt=session.prompt,
        tokens=session.tokens,
        model_info=model_info,
        vram_usage=vram,
    )
