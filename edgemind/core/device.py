"""
Device detection and hardware utilities for EdgeMind AI.

Design Philosophy:
    Edge AI research is fundamentally about hardware constraints. This
    module provides utilities to detect available compute devices and
    report their capabilities (memory, compute type).

    In later phases, we'll use this for:
    - Automatic device selection during training
    - Memory-aware batch size selection
    - Edge device constraint simulation

Usage:
    >>> from edgemind.core.device import get_device, get_device_info
    >>> device = get_device()          # Returns "cuda" or "cpu"
    >>> info = get_device_info()       # Returns detailed hardware info
    >>> print(info)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DeviceInfo:
    """Structured information about the compute device.

    Attributes:
        device: The PyTorch device string ("cpu", "cuda", "mps").
        device_name: Human-readable name (e.g., "NVIDIA RTX 3090").
        total_memory_mb: Total GPU memory in MB (None for CPU).
        cuda_version: CUDA version string (None if not available).
        pytorch_version: Installed PyTorch version.
        cpu_count: Number of CPU cores available.
    """
    device: str
    device_name: str
    total_memory_mb: Optional[float] = None
    cuda_version: Optional[str] = None
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    cpu_count: int = field(default_factory=lambda: torch.get_num_threads())

    def __str__(self) -> str:
        lines = [
            "╔══════════════════════════════════════════╗",
            "║         EdgeMind AI — Device Info        ║",
            "╠══════════════════════════════════════════╣",
            f"║  Device:          {self.device:<22s} ║",
            f"║  Device Name:     {self.device_name:<22s} ║",
        ]

        if self.total_memory_mb is not None:
            mem_str = f"{self.total_memory_mb:.0f} MB"
            lines.append(f"║  GPU Memory:      {mem_str:<22s} ║")

        if self.cuda_version is not None:
            lines.append(f"║  CUDA Version:    {self.cuda_version:<22s} ║")

        lines.extend([
            f"║  PyTorch Version: {self.pytorch_version:<22s} ║",
            f"║  CPU Threads:     {str(self.cpu_count):<22s} ║",
            "╚══════════════════════════════════════════╝",
        ])
        return "\n".join(lines)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Detect and return the best available compute device.

    Priority order: CUDA → MPS (Apple Silicon) → CPU.
    Can be overridden with the ``preferred`` argument.

    Args:
        preferred: Force a specific device ("cpu", "cuda", "mps").
            If None, auto-detects the best available.

    Returns:
        A ``torch.device`` instance.
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_info(preferred: Optional[str] = None) -> DeviceInfo:
    """Get detailed information about the compute device.

    Args:
        preferred: Force a specific device. If None, auto-detects.

    Returns:
        A ``DeviceInfo`` dataclass with hardware details.
    """
    device = get_device(preferred)

    if device.type == "cuda":
        return DeviceInfo(
            device="cuda",
            device_name=torch.cuda.get_device_name(0),
            total_memory_mb=torch.cuda.get_device_properties(0).total_mem / (1024 ** 2),
            cuda_version=torch.version.cuda,
        )

    if device.type == "mps":
        return DeviceInfo(
            device="mps",
            device_name="Apple Silicon (MPS)",
        )

    return DeviceInfo(
        device="cpu",
        device_name="CPU",
    )
