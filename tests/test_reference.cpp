// CPU-only test of the reference + problem generator. Compiles and runs with a
// plain host compiler (clang++/g++), no CUDA — this is the correctness oracle
// the GPU kernels are checked against.
#include "grouped_gemm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int failures = 0;
#define CHECK(cond, msg)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::printf("FAIL: %s\n", msg);                           \
      ++failures;                                               \
    }                                                           \
  } while (0)

int main() {
  // 5 rows across 2 groups -> sizes {3, 2}, offsets {0,3,5}.
  GroupedGemmProblem p = make_problem(/*total_m=*/5, /*K=*/3, /*N=*/4,
                                      /*num_groups=*/2, /*seed=*/7);
  CHECK(p.group_sizes.size() == 2, "num groups");
  CHECK(p.group_sizes[0] == 3 && p.group_sizes[1] == 2, "even partition");
  CHECK(p.row_offset[0] == 0 && p.row_offset[2] == 5, "row offsets");

  std::vector<float> out;
  cpu_reference(p, out, /*fuse_silu=*/false);
  CHECK(out.size() == (size_t)p.total_m * p.N, "output size");

  // Recompute C[row=4, n=1] by hand. Row 4 is in group 1 (rows 3..4).
  int row = 4, n = 1, g = 1;
  const float* Wg = p.W.data() + (size_t)g * p.K * p.N;
  float expect = 0.0f;
  for (int k = 0; k < p.K; ++k)
    expect += p.A[(size_t)row * p.K + k] * Wg[(size_t)k * p.N + n];
  float got = out[(size_t)row * p.N + n];
  CHECK(std::fabs(got - expect) < 1e-4f, "manual dot product matches");

  // SiLU fusion: silu(x) = x * sigmoid(x).
  std::vector<float> fused;
  cpu_reference(p, fused, /*fuse_silu=*/true);
  float x = expect;
  float silu = x / (1.0f + std::exp(-x));
  CHECK(std::fabs(fused[(size_t)row * p.N + n] - silu) < 1e-4f, "silu fusion");

  // max_abs_err sanity.
  CHECK(max_abs_err(out, out) == 0.0f, "zero self-error");

  if (failures == 0) std::printf("all reference tests passed\n");
  return failures == 0 ? 0 : 1;
}
