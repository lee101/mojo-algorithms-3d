# Contributing

Thank you for helping build a dependable 3D toolkit for Mojo.

1. Open an issue for API changes or new algorithms before writing a large
   patch.
2. Keep algorithms deterministic unless randomness is explicit and seeded.
3. Add contract tests for invalid inputs, boundaries, non-manifold topology,
   and degenerate geometry as applicable.
4. Run `pixi run test`.
5. Do not copy code from MeshLab, VCGlib, or another project unless its
   license is compatible, attribution is complete, and the dependency has
   been discussed in an issue.

Contributions are accepted under the Apache-2.0 license.
