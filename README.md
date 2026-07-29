# mojo-algorithms-3d

[![CI](https://github.com/lee101/mojo-algorithms-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/lee101/mojo-algorithms-3d/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Open, reusable 3D geometry algorithms written in
[Mojo](https://www.modular.com/mojo).

The library provides a compact owned triangle mesh, validation and quality
metrics, conservative structural repair, plus topology-preserving surface
relaxation that combines Voronoi-style centroid movement with quadric error
constraints.

## Why this exists

SimplexGen needed geometry processing that could run beside its `mo3d`
renderer without routing every operation through Python or C++. This project
extracts that work behind a small public API so other Mojo applications can
use and improve it.

This is not a port or derivative of MeshLab or VCGlib. Those GPL projects are
useful references for the breadth of a mature mesh-processing toolkit, but no
source from either project is included here.

## Install and test

Install [Pixi](https://pixi.sh), then run:

```bash
git clone https://github.com/lee101/mojo-algorithms-3d.git
cd mojo-algorithms-3d
pixi run test
pixi run example
```

Build the importable package:

```bash
pixi run package
```

The compiler-version-specific artifact is written to
`dist/algorithms3d.mojoc`. Source imports are recommended for libraries that
need to support more than one Mojo compiler release.

Build the optional zero-copy NumPy bridge:

```bash
pixi run build
PYTHONPATH=python python -c "import mojo_algorithms_3d"
```

Its `triangle_soup_quality` function accepts an existing C-contiguous
`float32` array with shape `(9, triangle_count)`, ordered as
`ax, ay, az, bx, by, bz, cx, cy, cz`. The input buffer is passed directly to
the compiled Mojo shared library. Wrong dtypes and non-contiguous arrays are
rejected instead of being copied implicitly.

## Use from source

```mojo
from algorithms3d import RemeshOptions, TriangleMesh, Vec3, surface_relax

fn main() raises:
    var mesh = TriangleMesh()
    # Add vertices and counter-clockwise triangles.
    var options = RemeshOptions()
    var stats = surface_relax(mesh, options)
    print(stats.quality_improvement())
```

Run consumers with this repository on the Mojo import path:

```bash
mojo run -I /path/to/mojo-algorithms-3d app.mojo
```

## Repair damaged meshes

```mojo
from algorithms3d import RepairOptions, TriangleMesh, repair_mesh

fn clean(mut mesh: TriangleMesh) raises:
    var options = RepairOptions()
    var stats = repair_mesh(mesh, options)
    print("removed duplicate faces:", stats.removed_duplicate_faces)
```

The repair pass preserves the first valid occurrence of each face and reports
invalid faces, degenerate faces, duplicates, and compacted vertices separately.

## Benchmarks

Run the reproducible harness with:

```bash
pixi run bench
```

The Pixi task builds the native artifacts, then runs the timed command under
`flock /tmp/mojo-bench.lock`. Every case gets one warmup; the table reports the
median of seven repeats, except surface relaxation, which uses five. Inputs are
identical between implementations, setup is outside the timed region, and the
Python relaxation reference mirrors the same centroid, QEM, projection, and
line-search steps.

Measured on 2026-07-29 with Mojo 1.0.0b2 and an Intel Xeon E5-2697 v4:

| Operation | Input size | mojo-algorithms-3d | Baseline | Speedup |
|---|---:|---:|---:|---:|
| triangle-soup quality | 1,000,003 triangles | 4.770 ms | 112.461 ms NumPy | 23.57x |
| owned-mesh quality | 79,202 triangles | 6.562 ms | 21.667 ms NumPy | 3.30x |
| duplicate-face repair | 10,000 input faces | 0.899 ms | 669.132 ms pure Python | 744.34x |
| surface relaxation | 242 triangles, 2 iterations | 0.385 ms | 136.296 ms NumPy/Python | 353.81x |

These are measurements from this machine, not performance guarantees. The
triangle-soup quality kernel uses eight-lane Float32 SIMD here, including a
scalar tail. It parallelizes only at 4,000,000 triangles or more and caps
execution at 16 physical-core workers. Repair uses hashed canonical face keys.
Surface relaxation uses cached vertex-face adjacency and per-iteration face
data; neither of those two kernels starts worker threads.

## API stability

`algorithms3d` is the only public import root. The `remesh` implementation
modules are internal details. Before `1.0`, breaking API changes are allowed
only in minor releases and will be documented in `CHANGELOG.md`.

## Roadmap

- QEM edge-collapse simplification with manifold and attribute guards
- vertex and corner normal generation
- connected components and topology inspection
- spatial queries and BVH acceleration
- OBJ, STL, PLY, and glTF adapters
- isotropic remeshing and subdivision

The order is intentionally “kernel first, editor second”: algorithms should
remain useful without a GUI or a particular renderer.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
