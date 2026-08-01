"""Global configuration and environment defaults for InterpLens."""

import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global configuration settings for InterpLens engine and server."""
    
    app_name: str = "InterpLens Debugger"
    version: str = "0.1.0"
    debug: bool = Field(default_factory=lambda: os.getenv("INTERPLENS_DEBUG", "false").lower() == "true")
    
    # Server settings
    host: str = Field(default_factory=lambda: os.getenv("INTERPLENS_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("INTERPLENS_PORT", "8501")))
    
    # Compute & Memory settings
    default_device: str = Field(default_factory=lambda: os.getenv("INTERPLENS_DEVICE", "auto"))
    use_half_precision: bool = Field(default_factory=lambda: os.getenv("INTERPLENS_FP16", "true").lower() == "true")
    max_cached_sessions: int = Field(default_factory=lambda: int(os.getenv("INTERPLENS_MAX_SESSIONS", "3")))
    
    # UI static assets path
    ui_dist_dir: str = os.path.join(os.path.dirname(__file__), "ui", "dist")


settings = Settings()
