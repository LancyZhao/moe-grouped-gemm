# moe-grouped-gemm

[![CI](https://github.com/LancyZhao/moe-grouped-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/LancyZhao/moe-grouped-gemm/actions/workflows/ci.yml)
![CUDA](https://img.shields.io/badge/CUDA-11%2B-76b900)
![License](https://img.shields.io/badge/license-MIT-green)

> CUDA **grouped GEMM** kernels for the MoE expert FFN — the batched
> variable-M matmul you hit right after tokens are routed and sorted by expert.
> A naive kernel → a shared-memory **tiled** kernel → optional **fused SiLU**,
> each checked against a CPU reference and benchmarked against a per-group
> **cuBLAS** loop, with Nsight Compute profiling.

Why this problem: in an MoE FFN, after the permute/sort step each expert `g`
multiplies its token rows `A_g [Mₘ×K]` by its weight `W_g [K×N]`. The group
sizes vary with routing, which is exactly what makes it more interesting than a
single big GEMM.

## Build & run (on a GPU box)

```bash
make build                       # cmake + nvcc; set ARCH=89 for 4090, 90 for H100
make bench                       # default M=4096 K=1024 N=1024 groups=8
make bench ARGS="--m 8192 --groups 16 --fuse"
make profile                     # Nsight Compute on the tiled kernel
```

No GPU handy? The CPU correctness path needs neither nvcc nor a GPU:

```bash
make test                        # builds the CPU reference + runs the correctness test
```

## What the benchmark prints

```
MoE grouped GEMM  M=4096 K=1024 N=1024 groups=8 fuse_silu=0 iters=20

impl              ms       GFLOP/s     max_abs_err
naive          <fill from your GPU>
tiled          <fill from your GPU>
cublas         <fill from your GPU>
```

Every implementation's output is diffed against the CPU reference
(`max_abs_err` ~1e-4 for fp32 is expected from reorder of the K-reduction).

## The kernels (the optimization story)

1. **naive** (`sgemm_naive`) — one thread per output element, streams the whole
   K dimension from global memory. Memory-bound; the baseline to beat.
2. **tiled** (`sgemm_tiled<16>`) — stages 16×16 tiles of A and W in shared
   memory so each loaded element is reused across the tile. This is the real
   win: fewer global loads, more arithmetic per byte.
3. **fused SiLU** (`--fuse`) — applies `SiLU(x)=x·σ(x)` in-kernel on the GEMM
   output, saving a separate activation pass + a round-trip to HBM (the kind of
   fusion that matters on the MoE FFN critical path).

Baseline: **cuBLAS** `cublasSgemm` in a per-group loop (the column-major/row-major
operand swap is handled in `cublas_grouped`). The point isn't to beat cuBLAS —
it's to see how close a readable tiled kernel gets and to read *why* in Nsight.

## Profiling

`make profile` runs Nsight Compute on `sgemm_tiled` and captures the sections
worth reasoning about:

- **SpeedOfLight** — compute vs memory throughput (which roofline you're on);
- **Occupancy** — achieved vs theoretical (shared-mem/register limited?);
- **MemoryWorkloadAnalysis** — global load efficiency / coalescing.

Record a before/after for at least one optimization (e.g. tile size, or
transposing the W tile load for coalescing) in your write-up.

## Status (honest)

- ✅ **CPU reference + CMake CPU build + correctness test**: verified locally
  (clang++) and in CI with no GPU.
- ✅ **CUDA kernels + cuBLAS harness**: compiled & linked in CI inside a CUDA
  devel container (no GPU execution there).
- ⏳ **GPU benchmark numbers + Nsight analysis**: run `make bench` / `make
  profile` on your GPU and paste the table above. Until then the perf numbers
  are intentionally left as placeholders — not fabricated.

This is a learning/portfolio project (fp32, square-ish tiles, one baseline), not
a production kernel.

## Roadmap

- bf16 + Tensor Cores (`wmma`/`mma`), then compare against cuBLASLt
- int8 grouped GEMM (ties to MoE quantized inference)
- register blocking / double buffering; a CUTLASS grouped-GEMM comparison
- true variable-M grouped launch (one grid over all groups) vs the per-group loop

## 对应岗位能力（JD → 实现）

| 岗位要求 | 落点 |
| --- | --- |
| CUDA 算子 / kernel 开发 | `src/kernels.cu`：naive → shared-memory tiled → fused SiLU |
| 算子性能优化 / Nsight | `scripts/profile.sh` + README 的 SpeedOfLight/Occupancy/访存分析 |
| MoE 推理链路理解 | grouped GEMM = MoE FFN permute 后的核心 matmul（接实习 offload 经历） |
| C++ 工程 / 正确性保证 | CPU 参考 + max_abs_err 校验 + CI 编译 + ctest |
| 库对比 / 选型 | 对照 cuBLAS 每组 Sgemm 基线 |

## Layout

```
include/  grouped_gemm.h (problem + CPU reference)   kernels.cuh (launchers)
src/      reference.cpp (CPU, host-compilable)        kernels.cu  main.cu (harness)
tests/    test_reference.cpp (CPU correctness oracle)
scripts/  profile.sh (Nsight Compute)
```

## License

MIT © LancyZhao

