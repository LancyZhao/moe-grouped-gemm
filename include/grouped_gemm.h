// Grouped GEMM for MoE expert FFNs (host-side problem definition + CPU reference).
// Pure C++ (no CUDA includes) so it compiles and is testable on any host,
// including machines without a GPU. The CUDA kernels live in kernels.cuh/.cu.
//
// Problem: tokens are routed and sorted by expert into `num_groups` contiguous
// groups. Group g holds `group_sizes[g]` token rows and multiplies them by that
// expert's weight W_g [K x N]:
//
//     C[row, :] = A[row, :] (1 x K)  *  W_g [K x N]      for row in group g
//
// This is exactly the first GEMM of an MoE FFN after the permute/sort step.
#pragma once

#include <cstdint>
#include <vector>

struct GroupedGemmProblem {
  int K = 0;                       // shared contraction dim
  int N = 0;                       // shared output dim (per expert)
  int num_groups = 0;              // number of experts
  int total_m = 0;                 // sum of group_sizes
  std::vector<int> group_sizes;    // tokens per expert
  std::vector<int> row_offset;     // prefix sum, size num_groups+1
  std::vector<float> A;            // [total_m x K], row-major
  std::vector<float> W;            // [num_groups x K x N], row-major per group
  std::vector<float> C;            // [total_m x N], row-major (output)

  int64_t flops() const {          // 2*M*N*K multiply-adds across all groups
    return 2ll * total_m * N * K;
  }
};

// Build a deterministic problem (seeded LCG, no <random> dependency) with
// group sizes as even as possible.
GroupedGemmProblem make_problem(int total_m, int K, int N, int num_groups,
                                uint32_t seed = 1234);

// CPU reference: fills `out` (sized total_m*N). If fuse_silu, applies
// SiLU(x)=x*sigmoid(x) elementwise to the GEMM result (the MoE gate activation).
void cpu_reference(const GroupedGemmProblem& p, std::vector<float>& out,
                   bool fuse_silu = false);

// Max absolute error between two equally-sized buffers (for correctness checks).
float max_abs_err(const std::vector<float>& a, const std::vector<float>& b);
