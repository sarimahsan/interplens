"""InterpLens: An interactive mechanistic interpretability package and visual debugger for LLM internals."""

import os
import sys
import logging
from typing import Any, Optional

from .config import settings
from .exceptions import (
    InterpLensError,
    ModelLoadError,
    AdapterNotFoundError,
    CapabilityError,
    UnembeddingNotFoundError,
    ServerExecutionError,
)
from .adapters import (
    BaseModelAdapter,
    InPlaceModelAdapter,
    CustomModelAdapter,
    PyTorchAutoHooker,
    GPT2Adapter,
    get_adapter_for_model,
    register_adapter,
)
from .server.session import SessionStore, global_session_store
from .utils.device import get_optimal_device, resolve_device, free_gpu_memory, get_vram_usage

__version__ = settings.version

logger = logging.getLogger("interplens")

_server_thread = None
_uvicorn_server = None


def launch(
    model: Optional[Any] = None,
    model_name: str = "gpt2",
    port: int = 8000,
    host: str = "127.0.0.1",
    device: str = "auto",
    auto_hook: bool = False,
    tokenizer: Optional[Any] = None,
    block: bool = False,
):
    """Launches InterpLens visual debugger backend server and prints UI link.
    
    Args:
        model: Optional pre-loaded model object (HookedTransformer or PyTorch nn.Module).
        model_name: String name of model if loading from pretrained (e.g. 'gpt2').
        port: HTTP port for local Web UI server (default: 8000).
        host: Host binding IP address.
        device: Target compute device ('cuda', 'cpu', 'mps', 'auto').
        auto_hook: If True, uses PyTorchAutoHooker for novel nn.Module models.
        tokenizer: Optional tokenizer for custom PyTorch models.
        block: If True, blocks current process; if False, runs web server in background thread.
    """
    global _server_thread, _uvicorn_server
    target_device = resolve_device(device)
    if model is not None:
        adapter = get_adapter_for_model(model, device=str(target_device), auto_hook=auto_hook, tokenizer=tokenizer)
    else:
        adapter = get_adapter_for_model(model_name, device=str(target_device), tokenizer=tokenizer)

    if hasattr(adapter, "load"):
        try:
            adapter.load()
        except Exception as e:
            logger.warning(f"Notice during model load: {e}")
        
    from .server.app import set_active_adapter, app
    set_active_adapter(adapter)

    # Detect Google Colab environment
    is_colab = "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None
    if is_colab and host == "127.0.0.1":
        host = "0.0.0.0"

    logger.info(f"InterpLens v{__version__} attached to model: {adapter.model_name}")
    logger.info(f"Device: {adapter.device} | Layers: {adapter.num_layers} | Hidden Dim: {adapter.hidden_dim}")
    logger.info(f"Debugger Web UI active at http://{host}:{port}")

    if is_colab:
        try:
            from google.colab import output
            output.serve_kernel_port(port)
            logger.info(f"Colab Proxy Enabled for Port {port}")
        except Exception as e:
            logger.debug(f"Colab proxy initialization skipped: {e}")

    if block:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    else:
        if _server_thread is None or not _server_thread.is_alive():
            import uvicorn
            import threading
            import time
            config = uvicorn.Config(app=app, host=host, port=port, log_level="error")
            _uvicorn_server = uvicorn.Server(config)
            _server_thread = threading.Thread(target=_uvicorn_server.run, daemon=True)
            _server_thread.start()
            time.sleep(0.5)

    return adapter


def stop_server():
    """Stops the active background web debugger server if running."""
    global _uvicorn_server, _server_thread
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
        _uvicorn_server = None
        _server_thread = None
        logger.info("InterpLens debugger web server stopped.")


def inspect(
    prompt: str,
    model: Optional[Any] = None,
    model_name: str = "gpt2",
    device: str = "auto",
    corrupted_prompt: Optional[str] = None,
):
    """Runs forward pass under @torch.inference_mode() and returns ActivationSession for programmatic inspection."""
    adapter = get_adapter_for_model(model if model is not None else model_name, device=device)
    session = global_session_store.create_session(adapter=adapter, prompt=prompt, corrupted_prompt=corrupted_prompt)
    return session


__all__ = [
    "launch",
    "stop_server",
    "inspect",
    "InterpLensError",
    "ModelLoadError",
    "AdapterNotFoundError",
    "CapabilityError",
    "UnembeddingNotFoundError",
    "ServerExecutionError",
    "BaseModelAdapter",
    "InPlaceModelAdapter",
    "CustomModelAdapter",
    "PyTorchAutoHooker",
    "GPT2Adapter",
    "SessionStore",
    "get_adapter_for_model",
    "register_adapter",
    "get_optimal_device",
    "resolve_device",
    "free_gpu_memory",
    "get_vram_usage",
]

