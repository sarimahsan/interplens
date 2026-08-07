"""FastAPI web server entrypoint for InterpLens UI and REST API.

Provides modular router registration and mounts static visual debugger UI.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from interplens.server.state import (
    state_manager,
    init_model,
    set_active_adapter,
    get_active_adapter,
    get_adapter_model_info,
)
from interplens.server.routes.model import router as model_router
from interplens.server.routes.analysis import router as analysis_router
from interplens.server.routes.session import router as session_router

logger = logging.getLogger("interplens.server")

app = FastAPI(
    title="InterpLens Debugger API",
    description="Interactive Mechanistic Interpretability API and Web Debugger",
    version="0.1.0",
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular sub-routers
app.include_router(model_router)
app.include_router(analysis_router)
app.include_router(session_router)

# Mount UI static files if UI directory exists
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui_alt")
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui_root")
