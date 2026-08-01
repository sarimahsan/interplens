"""CUDA and PyTorch device memory management utilities for InterpLens."""

import gc
from typing import Dict, Any
import torch


def get_optimal_device() -> torch.device:
    """Detects best available compute device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device_str: str) -> torch.device:
    """Resolves string device name or 'auto' to torch.device."""
    if device_str.lower() == "auto":
        return get_optimal_device()
    return torch.device(device_str)


def get_vram_usage(device: torch.device) -> Dict[str, Any]:
    """Returns allocated and reserved VRAM in MB for CUDA devices."""
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        return {
            "device": str(device),
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "total_mb": round(total, 2),
            "free_mb": round(total - reserved, 2),
        }
    return {
        "device": str(device),
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "total_mb": 0.0,
        "free_mb": 0.0,
    }


def free_gpu_memory():
    """Forces garbage collection and clears CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_torch_dtype(use_half: bool = True, device: torch.device = None) -> torch.dtype:
    """Selects optimal PyTorch dtype (bfloat16/float16/float32) for device."""
    if not use_half:
        return torch.float32
    if device is None:
        device = get_optimal_device()
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif device.type == "cuda":
        return torch.float16
    return torch.float32
