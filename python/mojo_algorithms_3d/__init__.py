"""NumPy access to mojo-algorithms-3d's zero-copy batch kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = ctypes.CDLL(str(_ROOT / "build" / "libalgorithms3d.so"))
_QUALITY = _LIBRARY.a3d_triangle_soup_quality
_QUALITY.argtypes = [ctypes.c_ssize_t] * 5
_QUALITY.restype = None
_SIMD_WIDTH = _LIBRARY.a3d_simd_width
_SIMD_WIDTH.restype = ctypes.c_ssize_t
_PARALLEL_THRESHOLD = _LIBRARY.a3d_quality_parallel_threshold
_PARALLEL_THRESHOLD.restype = ctypes.c_ssize_t

SIMD_WIDTH = int(_SIMD_WIDTH())
QUALITY_PARALLEL_THRESHOLD = int(_PARALLEL_THRESHOLD())


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    mean_edge_length: float
    edge_coefficient_of_variation: float
    mean_triangle_quality: float
    min_triangle_quality: float
    objective: float
    worker_tasks: int


def triangle_soup_quality(
    coordinates: NDArray[np.float32],
    *,
    result: NDArray[np.float32] | None = None,
    scratch: NDArray[np.float32] | None = None,
) -> QualityMetrics:
    """Measure a ``(9, n)`` SoA triangle soup without copying its input.

    Rows are ``ax, ay, az, bx, by, bz, cx, cy, cz``. The array must already
    be C-contiguous ``float32`` so an accidental conversion cannot hide in a
    timed or latency-sensitive call.
    """
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("coordinates must be a numpy.ndarray")
    if coordinates.dtype != np.float32:
        raise TypeError("coordinates must have dtype float32")
    if coordinates.ndim != 2 or coordinates.shape[0] != 9:
        raise ValueError("coordinates must have shape (9, triangle_count)")
    if not coordinates.flags.c_contiguous:
        raise ValueError("coordinates must be C-contiguous")

    if result is None:
        result = np.empty(6, dtype=np.float32)
    elif result.dtype != np.float32 or result.shape != (6,) or not result.flags.c_contiguous:
        raise ValueError("result must be a C-contiguous float32 array with shape (6,)")

    if scratch is None:
        scratch = np.empty((16, 4), dtype=np.float32)
    elif (
        scratch.dtype != np.float32
        or scratch.ndim != 2
        or scratch.shape[1] != 4
        or not scratch.flags.c_contiguous
    ):
        raise ValueError("scratch must be a C-contiguous float32 array with shape (n, 4)")

    _QUALITY(
        coordinates.ctypes.data,
        coordinates.shape[1],
        result.ctypes.data,
        scratch.ctypes.data,
        scratch.shape[0],
    )
    return QualityMetrics(
        mean_edge_length=float(result[0]),
        edge_coefficient_of_variation=float(result[1]),
        mean_triangle_quality=float(result[2]),
        min_triangle_quality=float(result[3]),
        objective=float(result[4]),
        worker_tasks=int(result[5]),
    )


__all__ = [
    "QUALITY_PARALLEL_THRESHOLD",
    "SIMD_WIDTH",
    "QualityMetrics",
    "triangle_soup_quality",
]
