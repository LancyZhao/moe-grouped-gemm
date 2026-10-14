#!/usr/bin/env bash
# Nsight Compute profile of the tiled grouped-GEMM kernel.
# Requires `ncu` (NVIDIA Nsight Compute) and a GPU. Builds first if needed.
#
#   bash scripts/profile.sh            # default problem
#   bash scripts/profile.sh --m 8192   # pass-through args to ./build/bench
set -euo pipefail

[ -x ./build/bench ] || { echo "building..."; cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-80}"; cmake --build build -j; }

# -k regex limits profiling to the tiled kernel; sections cover the metrics you
# actually reason about: occupancy, compute/memory throughput, memory workload.
ncu --set basic \
    -k "regex:sgemm_tiled" \
    --section SpeedOfLight \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    -o tiled_profile \
    ./build/bench "$@"

echo "wrote tiled_profile.ncu-rep — open in Nsight Compute UI or: ncu -i tiled_profile.ncu-rep --page details"
