"""Device management and reporting utility functions for InterpLens."""

from interplens.utils.device import (
    get_optimal_device,
    resolve_device,
    get_vram_usage,
    free_gpu_memory,
    get_gpu_grid_status,
    get_detailed_gpu_profiler,
    get_torch_dtype,
)
from interplens.utils.pdf_report import generate_model_report_pdf

__all__ = [
    "get_optimal_device",
    "resolve_device",
    "get_vram_usage",
    "free_gpu_memory",
    "get_gpu_grid_status",
    "get_detailed_gpu_profiler",
    "get_torch_dtype",
    "generate_model_report_pdf",
]
