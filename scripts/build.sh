#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
mojo build --emit shared-lib -I . src/capi.mojo -o build/libalgorithms3d.so -Xlinker -lm
mojo build -I . bench/owned_bench.mojo -o build/owned_bench -Xlinker -lm
