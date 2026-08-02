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


def get_gpu_grid_status(device: torch.device = None) -> Dict[str, Any]:
    """Returns granular GPU VRAM 32-block memory grid and CUDA stream execution metrics."""
    if device is None:
        device = get_optimal_device()

    if device.type == "cuda" and torch.cuda.is_available():
        dev_idx = device.index if device.index is not None else 0
        total = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)
        free = total - reserved

        total_blocks = 32
        alloc_blocks = min(total_blocks, int(round((allocated / total) * total_blocks)))
        cache_blocks = min(total_blocks - alloc_blocks, max(0, int(round(((reserved - allocated) / total) * total_blocks))))
        free_blocks = max(0, total_blocks - alloc_blocks - cache_blocks)

        blocks = []
        for i in range(alloc_blocks):
            blocks.append({"id": i, "type": "weights", "label": "Model Weights"})
        for i in range(cache_blocks):
            blocks.append({"id": alloc_blocks + i, "type": "cache", "label": "Activation Cache"})
        for i in range(free_blocks):
            blocks.append({"id": alloc_blocks + cache_blocks + i, "type": "free", "label": "Free VRAM"})

        return {
            "has_gpu": True,
            "device_name": torch.cuda.get_device_name(dev_idx),
            "total_mb": round(total, 1),
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "max_allocated_mb": round(max_allocated, 1),
            "free_mb": round(free, 1),
            "utilization_pct": round((allocated / total) * 100, 1),
            "blocks": blocks,
        }

    return {
        "has_gpu": False,
        "device_name": "CPU",
        "total_mb": 0.0,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "free_mb": 0.0,
        "utilization_pct": 0.0,
        "blocks": [{"id": i, "type": "free", "label": "CPU RAM"} for i in range(32)],
    }


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
