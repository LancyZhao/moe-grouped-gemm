// CPU reference implementation + problem generator (pure C++, no CUDA).
#include "grouped_gemm.h"

#include <cmath>

namespace {
// Small, fast, fully deterministic PRNG so problems are reproducible without
// pulling in <random> (whose engines vary across stdlibs).
struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed ? seed : 1u) {}
  float next() {  // uniform in [-1, 1)
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 8) / (float)(1u << 23) - 1.0f;
  }
};
}  // namespace

GroupedGemmProblem make_problem(int total_m, int K, int N, int num_groups,
                                uint32_t seed) {
  GroupedGemmProblem p;
  p.K = K;
  p.N = N;
  p.num_groups = num_groups;
  p.total_m = total_m;

  // Partition total_m rows across groups as evenly as possible.
  p.group_sizes.assign(num_groups, total_m / num_groups);
  for (int g = 0; g < total_m % num_groups; ++g) p.group_sizes[g] += 1;
  p.row_offset.assign(num_groups + 1, 0);
  for (int g = 0; g < num_groups; ++g)
    p.row_offset[g + 1] = p.row_offset[g] + p.group_sizes[g];

  Lcg rng(seed);
  p.A.resize((size_t)total_m * K);
  for (auto& x : p.A) x = rng.next();
  p.W.resize((size_t)num_groups * K * N);
  for (auto& x : p.W) x = rng.next();
  p.C.assign((size_t)total_m * N, 0.0f);
  return p;
}

void cpu_reference(const GroupedGemmProblem& p, std::vector<float>& out,
                   bool fuse_silu) {
  out.assign((size_t)p.total_m * p.N, 0.0f);
  for (int g = 0; g < p.num_groups; ++g) {
    const float* Wg = p.W.data() + (size_t)g * p.K * p.N;
    for (int row = p.row_offset[g]; row < p.row_offset[g + 1]; ++row) {
      const float* Arow = p.A.data() + (size_t)row * p.K;
      float* Crow = out.data() + (size_t)row * p.N;
      for (int n = 0; n < p.N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < p.K; ++k) acc += Arow[k] * Wg[(size_t)k * p.N + n];
        if (fuse_silu) acc = acc / (1.0f + std::exp(-acc));
        Crow[n] = acc;
      }
    }
  }
}

float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}
