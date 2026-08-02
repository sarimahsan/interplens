"""InterpLens: An interactive mechanistic interpretability package and visual debugger for LLM internals."""

from typing import Any, Optional
from .config import settings
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


def launch(
    model: Optional[Any] = None,
    model_name: str = "gpt2",
    port: int = 8501,
    host: str = "127.0.0.1",
    device: str = "auto",
    auto_hook: bool = False,
    tokenizer: Optional[Any] = None,
):
    """Launches InterpLens visual debugger backend server and prints UI link.
    
    Args:
        model: Optional pre-loaded model object (HookedTransformer or PyTorch nn.Module).
        model_name: String name of model if loading from pretrained (e.g. 'gpt2').
        port: HTTP port for local Web UI server.
        host: Host binding IP address.
        device: Target compute device ('cuda', 'cpu', 'mps', 'auto').
        auto_hook: If True, uses PyTorchAutoHooker for novel nn.Module models.
        tokenizer: Optional tokenizer for custom PyTorch models.
    """
    target_device = resolve_device(device)
    if model is not None:
        adapter = get_adapter_for_model(model, device=str(target_device), auto_hook=auto_hook, tokenizer=tokenizer)
    else:
        adapter = get_adapter_for_model(model_name, device=str(target_device))
        
    print(f"🚀 InterpLens v{__version__} attached to model: {adapter.model_name}")
    print(f"📍 Device: {adapter.device} | Layers: {adapter.num_layers} | Hidden Dim: {adapter.hidden_dim}")
    print(f"🌐 Debugger Web UI active at http://{host}:{port}")
    return adapter


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
    "inspect",
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
