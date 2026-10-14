// CUDA kernel launchers for grouped GEMM. Host code includes this; the
// definitions live in kernels.cu. Launchers loop over experts (groups) and run
// a per-group SGEMM A_g[Mg x K] * W_g[K x N] -> C_g[Mg x N], optionally fusing
// SiLU on the output.
#pragma once

#include <cuda_runtime.h>

// row_offset is a host array of size num_groups+1 (prefix sum of group sizes).
void launch_grouped_gemm_naive(const float* dA, const float* dW, float* dC,
                               const int* row_offset, int num_groups,
                               int K, int N, bool fuse_silu, cudaStream_t stream);

void launch_grouped_gemm_tiled(const float* dA, const float* dW, float* dC,
                               const int* row_offset, int num_groups,
                               int K, int N, bool fuse_silu, cudaStream_t stream);
