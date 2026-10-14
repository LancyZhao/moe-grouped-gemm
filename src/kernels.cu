// CUDA kernels: naive and shared-memory-tiled grouped GEMM for MoE FFN.
#include "kernels.cuh"

#include <math.h>

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

// --- naive: one thread per output element, streams K from global memory ------
__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ W,
                            float* __restrict__ C, int M, int N, int K, bool fuse) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * W[k * N + col];
  C[row * N + col] = fuse ? silu(acc) : acc;
}

// --- tiled: stage TILE x TILE blocks of A and W in shared memory -------------
template <int TILE>
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ W,
                            float* __restrict__ C, int M, int N, int K, bool fuse) {
  __shared__ float As[TILE][TILE];
  __shared__ float Ws[TILE][TILE];
  int ty = threadIdx.y, tx = threadIdx.x;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  float acc = 0.0f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int a_k = t * TILE + tx;
    int w_k = t * TILE + ty;
    As[ty][tx] = (row < M && a_k < K) ? A[row * K + a_k] : 0.0f;
    Ws[ty][tx] = (w_k < K && col < N) ? W[w_k * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TILE; ++i) acc += As[ty][i] * Ws[i][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = fuse ? silu(acc) : acc;
}

static constexpr int TILE = 16;

void launch_grouped_gemm_naive(const float* dA, const float* dW, float* dC,
                               const int* row_offset, int num_groups,
                               int K, int N, bool fuse_silu, cudaStream_t stream) {
  dim3 block(16, 16);
  for (int g = 0; g < num_groups; ++g) {
    int m0 = row_offset[g], Mg = row_offset[g + 1] - m0;
    if (Mg <= 0) continue;
    dim3 grid((N + block.x - 1) / block.x, (Mg + block.y - 1) / block.y);
    sgemm_naive<<<grid, block, 0, stream>>>(dA + (size_t)m0 * K,
                                            dW + (size_t)g * K * N,
                                            dC + (size_t)m0 * N, Mg, N, K, fuse_silu);
  }
}

void launch_grouped_gemm_tiled(const float* dA, const float* dW, float* dC,
                               const int* row_offset, int num_groups,
                               int K, int N, bool fuse_silu, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  for (int g = 0; g < num_groups; ++g) {
    int m0 = row_offset[g], Mg = row_offset[g + 1] - m0;
    if (Mg <= 0) continue;
    dim3 grid((N + TILE - 1) / TILE, (Mg + TILE - 1) / TILE);
    sgemm_tiled<TILE><<<grid, block, 0, stream>>>(dA + (size_t)m0 * K,
                                                  dW + (size_t)g * K * N,
                                                  dC + (size_t)m0 * N, Mg, N, K, fuse_silu);
  }
}
