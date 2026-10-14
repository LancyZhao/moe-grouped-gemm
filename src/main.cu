// Benchmark harness: runs naive / tiled / cuBLAS grouped GEMM, checks each
// against the CPU reference, times them, and prints GFLOP/s. Build with CUDA
// (see CMakeLists.txt); run on a GPU box.
#include "grouped_gemm.h"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),  \
                   __FILE__, __LINE__);                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t s = (x);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__,     \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Time a device operation averaged over `iters` (plus one warmup run).
static float time_ms(int iters, const std::function<void()>& fn) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  fn();  // warmup (JIT, caches)
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// cuBLAS baseline: a per-group Sgemm loop. cuBLAS is column-major, so to get
// row-major C[Mg x N] = A[Mg x K] * W[K x N] we compute it as the transposed
// problem with swapped operands (no explicit transpose needed).
static void cublas_grouped(cublasHandle_t h, const float* dA, const float* dW,
                           float* dC, const std::vector<int>& row_offset,
                           int K, int N) {
  const float alpha = 1.0f, beta = 0.0f;
  int groups = (int)row_offset.size() - 1;
  for (int g = 0; g < groups; ++g) {
    int m0 = row_offset[g], Mg = row_offset[g + 1] - m0;
    if (Mg <= 0) continue;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, Mg, K, &alpha,
                             dW + (size_t)g * K * N, N, dA + (size_t)m0 * K, K,
                             &beta, dC + (size_t)m0 * N, N));
  }
}

static int arg_int(int argc, char** argv, const char* flag, int fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
  return fallback;
}

static bool arg_flag(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return true;
  return false;
}

int main(int argc, char** argv) {
  int M = arg_int(argc, argv, "--m", 4096);
  int K = arg_int(argc, argv, "--k", 1024);
  int N = arg_int(argc, argv, "--n", 1024);
  int G = arg_int(argc, argv, "--groups", 8);
  int iters = arg_int(argc, argv, "--iters", 20);
  bool fuse = arg_flag(argc, argv, "--fuse");

  GroupedGemmProblem p = make_problem(M, K, N, G);
  std::vector<float> ref_plain, ref_fused;
  cpu_reference(p, ref_plain, false);
  cpu_reference(p, ref_fused, true);
  const std::vector<float>& ref_kernel = fuse ? ref_fused : ref_plain;

  size_t aN = (size_t)M * K, wN = (size_t)G * K * N, cN = (size_t)M * N;
  float *dA, *dW, *dC;
  CUDA_CHECK(cudaMalloc(&dA, aN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dW, wN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, cN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, p.A.data(), aN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, p.W.data(), wN * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> hC(cN);
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::printf("MoE grouped GEMM  M=%d K=%d N=%d groups=%d fuse_silu=%d iters=%d\n\n",
              M, K, N, G, (int)fuse, iters);
  std::printf("%-10s%12s%14s%16s\n", "impl", "ms", "GFLOP/s", "max_abs_err");

  auto bench = [&](const char* name, const std::function<void()>& fn,
                   const std::vector<float>& ref) {
    CUDA_CHECK(cudaMemset(dC, 0, cN * sizeof(float)));
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, cN * sizeof(float), cudaMemcpyDeviceToHost));
    float err = max_abs_err(hC, ref);
    float ms = time_ms(iters, fn);
    double gflops = p.flops() / (ms * 1e6);
    std::printf("%-10s%12.3f%14.1f%16.2e\n", name, ms, gflops, err);
  };

  bench("naive", [&] { launch_grouped_gemm_naive(dA, dW, dC, p.row_offset.data(), G, K, N, fuse, 0); }, ref_kernel);
  bench("tiled", [&] { launch_grouped_gemm_tiled(dA, dW, dC, p.row_offset.data(), G, K, N, fuse, 0); }, ref_kernel);
  bench("cublas", [&] { cublas_grouped(handle, dA, dW, dC, p.row_offset, K, N); }, ref_plain);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}

