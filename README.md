# mojo-algorithms-3d

[![CI](https://github.com/lee101/mojo-algorithms-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/lee101/mojo-algorithms-3d/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
