"""Warm, repeated, median benchmarks against NumPy and pure Python."""

from __future__ import annotations

import collections
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from mojo_algorithms_3d import triangle_soup_quality

ROOT = Path(__file__).resolve().parents[1]


def median_time(function, repeats: int = 7) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) * 1e-9)
    return statistics.median(samples)


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_grid(side: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:side, :side]
    index = y * side + x
    boundary = (x == 0) | (y == 0) | (x == side - 1) | (y == side - 1)
    jitter_x = np.where(boundary, 0.0, np.sin(index * 2.17) * 0.095)
    jitter_y = np.where(boundary, 0.0, np.cos(index * 1.31) * 0.075)
    vertices = np.column_stack(
        (
            -1.0 + 2.0 * x.ravel() / (side - 1) + jitter_x.ravel(),
            -1.0 + 2.0 * y.ravel() / (side - 1) + jitter_y.ravel(),
            np.zeros(side * side),
        )
    ).astype(np.float32)
    cells = np.arange((side - 1) * (side - 1), dtype=np.int64)
    cy, cx = np.divmod(cells, side - 1)
    a = cy * side + cx
    b = a + 1
    c = a + side
    d = c + 1
    faces = np.empty((2 * len(cells), 3), dtype=np.int64)
    faces[0::2] = np.column_stack((a, b, d))
    faces[1::2] = np.column_stack((a, d, c))
    return vertices, faces


def numpy_quality(vertices: np.ndarray, faces: np.ndarray) -> tuple[float, ...]:
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab = np.linalg.norm(a - b, axis=1)
    bc = np.linalg.norm(b - c, axis=1)
    ca = np.linalg.norm(c - a, axis=1)
    edge_sum = np.sum(ab + bc + ca, dtype=np.float64)
    edge_square_sum = np.sum(ab * ab + bc * bc + ca * ca, dtype=np.float64)
    denominator = ab * ab + bc * bc + ca * ca
    twice_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    quality = np.divide(
        np.float32(2 * np.sqrt(3)) * twice_area,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.float32(1e-12),
    )
    mean_edge = edge_sum / (3 * len(faces))
    variance = max(
        0.0,
        edge_square_sum / (3 * len(faces)) - mean_edge * mean_edge,
    )
    cv = math.sqrt(variance) / mean_edge if mean_edge > 1e-12 else 0.0
    mean_quality = float(np.mean(quality, dtype=np.float64))
    return (
        mean_edge,
        cv,
        mean_quality,
        float(np.min(quality, initial=1.0)),
        cv + 0.35 * (1.0 - mean_quality),
    )


def numpy_soup_quality(coordinates: np.ndarray) -> tuple[float, ...]:
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
    mean_edge = edge_sum / (3 * coordinates.shape[1])
    variance = max(
        0.0,
        edge_square_sum / (3 * coordinates.shape[1]) - mean_edge * mean_edge,
    )
    cv = math.sqrt(variance) / mean_edge if mean_edge > 1e-12 else 0.0
    mean_quality = float(np.mean(quality, dtype=np.float64))
    return (
        mean_edge,
        cv,
        mean_quality,
        float(np.min(quality, initial=1.0)),
        cv + 0.35 * (1.0 - mean_quality),
    )


def python_repair(vertices: np.ndarray, faces: np.ndarray) -> int:
    seen: set[tuple[int, int, int]] = set()
    kept = 0
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        if a < 0 or b < 0 or c < 0 or max(a, b, c) >= len(vertices):
            continue
        if a == b or b == c or c == a:
            continue
        if np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])) <= 1e-12:
            continue
        key = tuple(sorted((a, b, c)))
        if key in seen:
            continue
        seen.add(key)
        kept += 1
    return kept


def closest_point_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + ab * (d1 / (d1 - d3))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    denominator = 1.0 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)


def python_surface_relax(
    input_vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 2,
) -> float:
    vertices = input_vertices.copy()
    source = input_vertices
    incident: list[list[int]] = [[] for _ in vertices]
    edges: collections.Counter[tuple[int, int]] = collections.Counter()
    for face_index, face in enumerate(faces):
        for vertex in face:
            incident[int(vertex)].append(face_index)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[tuple(sorted((int(a), int(b))))] += 1
    boundaries = np.zeros(len(vertices), dtype=bool)
    for (a, b), count in edges.items():
        if count == 1:
            boundaries[a] = boundaries[b] = True

    def face_data(points: np.ndarray):
        triangles = points[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        normals = np.divide(
            cross,
            lengths[:, None],
            out=np.zeros_like(cross),
            where=lengths[:, None] > 1e-12,
        )
        return normals, lengths * 0.5, np.mean(triangles, axis=1)

    source_normals, source_areas, _ = face_data(source)
    quadrics = np.zeros((len(vertices), 3, 3), dtype=np.float32)
    linear = np.zeros((len(vertices), 3), dtype=np.float32)
    areas = np.zeros(len(vertices), dtype=np.float32)
    for vertex, face_ids in enumerate(incident):
        for face_id in face_ids:
            normal = source_normals[face_id]
            area = source_areas[face_id]
            d = -float(normal @ source[faces[face_id, 0]])
            quadrics[vertex] += np.outer(normal, normal) * area
            linear[vertex] += normal * d * area
            areas[vertex] += area

    def project(point: np.ndarray, vertex: int) -> np.ndarray:
        best = source[vertex]
        best_distance = math.inf
        for face_id in incident[vertex]:
            a, b, c = source[faces[face_id]]
            candidate = closest_point_triangle(point, a, b, c)
            distance = float((candidate - point) @ (candidate - point))
            if distance < best_distance:
                best, best_distance = candidate, distance
        return best

    objective = numpy_quality(vertices, faces)[4]
    for _ in range(iterations):
        normals, face_areas, centroids = face_data(vertices)
        proposals = vertices.copy()
        for vertex, face_ids in enumerate(incident):
            if boundaries[vertex]:
                continue
            ids = np.asarray(face_ids)
            total_area = float(np.sum(face_areas[ids]))
            if total_area <= 1e-12:
                continue
            centroid = np.sum(centroids[ids] * face_areas[ids, None], axis=0) / total_area
            normal = np.sum(normals[ids] * face_areas[ids, None], axis=0)
            magnitude = float(np.linalg.norm(normal))
            if magnitude > 1e-12:
                normal /= magnitude
            raw = centroid - vertices[vertex]
            target = vertices[vertex] + (raw - normal * float(raw @ normal)) * 0.7
            weight = 0.8 / areas[vertex] if areas[vertex] > 1e-12 else 0.0
            matrix = np.eye(3, dtype=np.float32) + weight * quadrics[vertex]
            rhs = target - weight * linear[vertex]
            try:
                regularized = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                regularized = target
            proposals[vertex] = project(regularized, vertex)

        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125):
            candidate = vertices + (proposals - vertices) * step
            for vertex in range(len(vertices)):
                candidate[vertex] = project(candidate[vertex], vertex)
            candidate_objective = numpy_quality(candidate, faces)[4]
            if candidate_objective <= objective + 1e-7:
                vertices = candidate
                objective = candidate_objective
                accepted = True
                break
        if not accepted:
            break
    return objective


def owned_mojo_times() -> dict[str, float]:
    output = subprocess.run(
        [str(ROOT / "build" / "owned_bench")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    samples: dict[str, list[float]] = collections.defaultdict(list)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].endswith("_ns"):
            samples[fields[0][:-3]].append(int(fields[1]) * 1e-9)
    expected = {"quality", "repair", "relax"}
    if set(samples) != expected:
        raise RuntimeError(f"unexpected owned benchmark output:\n{output}")
    return {name: statistics.median(values) for name, values in samples.items()}


def main() -> None:
    rows = []

    rng = np.random.default_rng(42)
    triangle_count = 1_000_003
    coordinates = rng.random((9, triangle_count), dtype=np.float32)
    coordinates[3:6] += np.float32(0.25)
    coordinates[6:9] += np.float32(0.5)
    result = np.empty(6, dtype=np.float32)
    scratch = np.empty((16, 4), dtype=np.float32)
    ours = median_time(
        lambda: triangle_soup_quality(
            coordinates,
            result=result,
            scratch=scratch,
        )
    )
    baseline = median_time(lambda: numpy_soup_quality(coordinates))
    rows.append(("triangle-soup quality", "1,000,003 triangles", ours, baseline, "NumPy"))

    owned = owned_mojo_times()
    quality_vertices, quality_faces = make_grid(200)
    quality_python = median_time(lambda: numpy_quality(quality_vertices, quality_faces))
    rows.append(("owned-mesh quality", "79,202 triangles", owned["quality"], quality_python, "NumPy"))

    repair_vertices = np.empty((15_000, 3), dtype=np.float32)
    face_number = np.arange(5_000, dtype=np.float32)
    repair_vertices[0::3] = np.column_stack((face_number, np.zeros(5_000), np.zeros(5_000)))
    repair_vertices[1::3] = np.column_stack((face_number + 0.5, np.ones(5_000), np.zeros(5_000)))
    repair_vertices[2::3] = np.column_stack((face_number + 1.0, np.zeros(5_000), np.zeros(5_000)))
    unique = np.arange(15_000, dtype=np.int64).reshape(-1, 3)
    repair_faces = np.empty((10_000, 3), dtype=np.int64)
    repair_faces[0::2] = unique
    repair_faces[1::2] = unique[:, ::-1]
    repair_python = median_time(lambda: python_repair(repair_vertices, repair_faces))
    rows.append(("duplicate-face repair", "10,000 input faces", owned["repair"], repair_python, "pure Python"))

    relax_vertices, relax_faces = make_grid(12)
    relax_python = median_time(
        lambda: python_surface_relax(relax_vertices, relax_faces),
        repeats=5,
    )
    rows.append(("surface relaxation", "242 triangles, 2 iterations", owned["relax"], relax_python, "NumPy/Python"))

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print("Warmup: 1 run; reported statistic: median of 7 repeats (5 for relaxation)")
    print()
    print("| Operation | Input size | mojo-algorithms-3d | Baseline | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for operation, size, mojo_time, baseline_time, baseline_name in rows:
        speedup = baseline_time / mojo_time
        print(
            f"| {operation} | {size} | {mojo_time * 1e3:.3f} ms | "
            f"{baseline_time * 1e3:.3f} ms {baseline_name} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
