"""FastAPI Router for Activation Session Management Endpoints."""

from fastapi import APIRouter, HTTPException
from interplens.server.session import global_session_store

router = APIRouter(prefix="/api", tags=["Sessions"])


@router.get("/sessions")
def get_active_sessions():
    """Returns metadata for all cached activation sessions in LRU memory."""
    return {
        "sessions": global_session_store.get_sessions_metadata(),
        "max_sessions": global_session_store.max_sessions,
    }


@router.delete("/sessions/{session_id}")
def evict_session(session_id: str):
    """Manually evicts a session from LRU memory cache and clears GPU VRAM."""
    success = global_session_store.evict_session(session_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already evicted."
        )
    return {"status": "evicted", "session_id": session_id}
