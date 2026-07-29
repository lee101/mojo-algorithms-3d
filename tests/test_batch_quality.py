from __future__ import annotations

import numpy as np

from mojo_algorithms_3d import (
    QUALITY_PARALLEL_THRESHOLD,
    SIMD_WIDTH,
    triangle_soup_quality,
)


def numpy_quality(coordinates: np.ndarray) -> np.ndarray:
    a, b, c = coordinates[0:3], coordinates[3:6], coordinates[6:9]
    ab = np.sqrt(np.sum((a - b) ** 2, axis=0, dtype=np.float32))
    bc = np.sqrt(np.sum((b - c) ** 2, axis=0, dtype=np.float32))
    ca = np.sqrt(np.sum((c - a) ** 2, axis=0, dtype=np.float32))
    edge_sum = np.sum(ab + bc + ca, dtype=np.float64)
    edge_square_sum = np.sum(ab * ab + bc * bc + ca * ca, dtype=np.float64)
    denominator = ab * ab + bc * bc + ca * ca
    twice_area = np.sqrt(
        np.sum(np.cross((b - a).T, (c - a).T).T ** 2, axis=0, dtype=np.float32)
    )
    quality = np.divide(
        np.float32(2 * np.sqrt(3)) * twice_area,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.float32(1e-12),
    )
    edge_count = 3 * coordinates.shape[1]
    mean_edge = edge_sum / edge_count
    variance = max(0.0, edge_square_sum / edge_count - mean_edge * mean_edge)
    cv = np.sqrt(variance) / mean_edge if mean_edge > 1e-12 else 0.0
    mean_quality = float(np.mean(quality, dtype=np.float64))
    return np.array(
        [
            mean_edge,
            cv,
            mean_quality,
            float(np.min(quality, initial=1.0)),
            cv + 0.35 * (1.0 - mean_quality),
        ]
    )


def test_simd_tail_matches_numpy() -> None:
    count = SIMD_WIDTH * 2 + 3
    rng = np.random.default_rng(17)
    coordinates = rng.random((9, count), dtype=np.float32)
    coordinates[3:6] += np.float32(0.25)
    coordinates[6:9] += np.float32(0.5)

    actual = triangle_soup_quality(coordinates)
    expected = numpy_quality(coordinates)

    np.testing.assert_allclose(
        [
            actual.mean_edge_length,
            actual.edge_coefficient_of_variation,
            actual.mean_triangle_quality,
            actual.min_triangle_quality,
            actual.objective,
        ],
        expected,
        rtol=2e-5,
        atol=2e-6,
    )
    assert actual.worker_tasks == 1


def test_parallel_threshold_executes_parallel_path() -> None:
    count = QUALITY_PARALLEL_THRESHOLD + 3
    coordinates = np.empty((9, count), dtype=np.float32)
    coordinates[0:3] = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    coordinates[3:6] = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
    coordinates[6:9] = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)

    parallel = triangle_soup_quality(coordinates)
    assert parallel.worker_tasks > 1
    mean_edge = (2.0 + np.sqrt(2.0)) / 3.0
    cv = np.sqrt(4.0 / 3.0 - mean_edge * mean_edge) / mean_edge
    quality = np.sqrt(3.0) / 2.0
    np.testing.assert_allclose(
        [
            parallel.mean_edge_length,
            parallel.edge_coefficient_of_variation,
            parallel.mean_triangle_quality,
            parallel.min_triangle_quality,
            parallel.objective,
        ],
        [
            mean_edge,
            cv,
            quality,
            quality,
            cv + 0.35 * (1.0 - quality),
        ],
        rtol=2e-3,
        atol=2e-4,
    )
