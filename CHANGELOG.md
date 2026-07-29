# Changelog

All notable changes are documented here.

## Unreleased

### Added

- Locked, warmed, median benchmarks against NumPy and pure Python references.
- Zero-copy NumPy access to a SIMD triangle-soup quality kernel.

### Changed

- Replaced quadratic duplicate-face lookup with hashed canonical face keys.
- Reused vertex-face adjacency and per-iteration face data in surface relaxation.
- Relicensed the project under the MIT License.

## 0.2.0 - 2026-07-25

### Added

- Conservative mesh repair for invalid, degenerate, and duplicate faces.
- Optional unreferenced-vertex compaction with detailed repair statistics.

## 0.1.0 - 2026-07-25

### Added

- Owned `TriangleMesh` and `Vec3` primitives.
- Structural and geometric validation.
- Scale-independent mesh quality metrics.
- Deterministic, topology-preserving surface relaxation with QEM constraints.
- Boundary and feature preservation controls.
- Pixi development environment, package build, tests, and example.
