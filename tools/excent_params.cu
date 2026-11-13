// excent_params.cu the given constants of a graph, computed once before any timed run:
//   lambda1   largest eigenvalue of A (power iteration), needed by eigenvector centrality
//   alpha     Katz attenuation, 0.85 / lambda1 (beta = 1)
//   eps       convergence thresholds, 0.01 x each measure's fixed-point mean
//             (PageRank mean is 1, so eps_pr = 0.01)
// Writes them as a shell env file that run.sh reads with --params.
//
// Loads the whole graph onto the GPU: run it on a machine where the graph fits, or once on a
// larger GPU. The graph must be symmetric (every edge stored in both directions).
//
// Build: nvcc -O3 -std=c++17 -arch=native tools/excent_params.cu -o bin/excent_params
// Usage: excent_params <graph.egr> <out.env>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

typedef unsigned long long ull;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

__device__ __forceinline__ double warp_sum(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}
// out[0] += sum x^2, out[1] += sum x
__global__ void k_norms(int n, const double* x, double* out) {
  double a = 0, b = 0;
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n;
       v += (long long)gridDim.x * blockDim.x) { a += x[v] * x[v]; b += x[v]; }
  a = warp_sum(a); b = warp_sum(b);
  if ((threadIdx.x & 31) == 0) { atomicAdd(&out[0], a); atomicAdd(&out[1], b); }
}
// y[v] = sum_{u in N(v)} x[u]     (one warp per vertex)
__global__ void k_spmv(int n, const ull* idx, const int* adj, const double* x, double* y) {
  long long w = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (w >= n) return;
  double acc = 0;
  for (ull e = idx[w] + lane; e < idx[w + 1]; e += 32) acc += x[adj[e]];
  acc = warp_sum(acc);
  if (lane == 0) y[w] = acc;
}
__global__ void k_scale(int n, double* x, double f) {
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n;
       v += (long long)gridDim.x * blockDim.x) x[v] *= f;
}
__global__ void k_fill(int n, double* x, double f) {
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n;
       v += (long long)gridDim.x * blockDim.x) x[v] = f;
}
__global__ void k_addc(int n, double* x, double c) {
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n;
       v += (long long)gridDim.x * blockDim.x) x[v] += c;
}

static int G = 0, n_ = 0;
static double* d_red;
static double norm2(const double* x, double* sum = nullptr) {
  CK(cudaMemset(d_red, 0, 2 * sizeof(double)));
  k_norms<<<G, 256>>>(n_, x, d_red);
  double h[2]; CK(cudaMemcpy(h, d_red, sizeof(h), cudaMemcpyDeviceToHost));
  if (sum) *sum = h[1];
  return sqrt(h[0]);
}
static unsigned warp_blocks(int n) { return (unsigned)(((long long)n * 32 + 255) / 256); }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s graph.egr out.env\n", argv[0]); return 1; }
  auto T0 = std::chrono::steady_clock::now();

  // .egr: int32 nodes, int32 edges (-1 => ull edges follows), nindex[n+1] (ull if wide else int32), int32 nlist[m]
  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  int n, e32; ull m;
  if (fread(&n, 4, 1, f) != 1 || fread(&e32, 4, 1, f) != 1) return 1;
  std::vector<ull> idx((size_t)n + 1);
  if (e32 == -1) {
    if (fread(&m, 8, 1, f) != 1) return 1;
    if (fread(idx.data(), 8, (size_t)n + 1, f) != (size_t)n + 1) return 1;
  } else {
    m = (ull)e32;
    std::vector<int> i32((size_t)n + 1);
    if (fread(i32.data(), 4, (size_t)n + 1, f) != (size_t)n + 1) return 1;
    for (int i = 0; i <= n; i++) idx[i] = (ull)(unsigned)i32[i];
  }
  int* adj = nullptr;
  CK(cudaMallocHost(&adj, m * sizeof(int)));
  for (ull off = 0; off < m;) {
    ull want = std::min<ull>(m - off, 1ull << 28);
    size_t got = fread(adj + off, 4, want, f);
    if (got == 0) { fprintf(stderr, "short read\n"); return 1; }
    off += got;
  }
  fclose(f);
  n_ = n;
  printf("graph: n=%d m=%llu  load %.1fs\n", n, m,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count());

  // symmetry check on a random edge sample
  {
    std::mt19937_64 rng(42);
    int bad = 0, S = 200000;
    for (int i = 0; i < S; i++) {
      ull e = rng() % m;
      int v = (int)(std::upper_bound(idx.begin(), idx.end(), e) - idx.begin()) - 1;
      int u = adj[e];
      const int* b = adj + idx[u]; const int* en = adj + idx[u + 1];
      bool found = std::is_sorted(b, en) ? std::binary_search(b, en, v) : std::find(b, en, v) != en;
      if (!found) bad++;
    }
    if (bad) { fprintf(stderr, "graph is not symmetric (%d/%d sampled edges lack a reverse edge)\n", bad, S); return 1; }
  }

  ull* d_idx; int* d_adj;
  CK(cudaMalloc(&d_idx, ((size_t)n + 1) * sizeof(ull)));
  CK(cudaMalloc(&d_adj, m * sizeof(int)));
  CK(cudaMemcpy(d_idx, idx.data(), ((size_t)n + 1) * sizeof(ull), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_adj, adj, m * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaFreeHost(adj));
  CK(cudaMalloc(&d_red, 2 * sizeof(double)));
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  G = sms * 8;
  double* V[2];
  for (auto& p : V) CK(cudaMalloc(&p, (size_t)n * sizeof(double)));

  // lambda1 and the eigenvector's fixed-point mean
  double lam = 0, lamPrev = 0;
  k_fill<<<G, 256>>>(n, V[0], 1.0);
  k_scale<<<G, 256>>>(n, V[0], 1.0 / norm2(V[0]));
  int pit = 0;
  for (; pit < 5000; pit++) {
    k_spmv<<<warp_blocks(n), 256>>>(n, d_idx, d_adj, V[0], V[1]);
    lam = norm2(V[1]);
    k_scale<<<G, 256>>>(n, V[1], 1.0 / lam);
    std::swap(V[0], V[1]);
    if (pit > 10 && fabs(lam - lamPrev) / lam < 1e-12) break;
    lamPrev = lam;
  }
  double v1sum; norm2(V[0], &v1sum);
  double meanEV = v1sum * v1sum / n;   // (A/lam)^k 1 -> (v1.1) v1, mean = (v1.1)^2 / n
  printf("lambda1 = %.10f after %d power iterations; eigenvector fixed-point mean = %.6g\n", lam, pit, meanEV);

  // Katz fixed point x = beta + alpha A x, for its mean
  const double alpha = 0.85 / lam, beta = 1.0;
  k_fill<<<G, 256>>>(n, V[0], beta);
  double meanK = 0;
  for (int it = 0; it < 5000; it++) {
    k_spmv<<<warp_blocks(n), 256>>>(n, d_idx, d_adj, V[0], V[1]);
    k_scale<<<G, 256>>>(n, V[1], alpha);
    k_addc<<<G, 256>>>(n, V[1], beta);
    std::swap(V[0], V[1]);
    double sum; norm2(V[0], &sum);
    double newMean = sum / n;
    if (fabs(newMean - meanK) / newMean < 1e-12) { meanK = newMean; break; }
    meanK = newMean;
  }
  printf("Katz: alpha = 0.85/lambda1 = %.6g, beta = 1, fixed-point mean = %.6g\n", alpha, meanK);

  FILE* o = fopen(argv[2], "w");
  if (!o) { perror(argv[2]); return 1; }
  fprintf(o, "# Given constants for %s (tools/excent_params.cu; lambda1 after %d power iterations).\n", argv[1], pit);
  fprintf(o, "# Katz alpha = 0.85 / lambda1, beta = 1. Thresholds: 0.01 x each measure's fixed-point mean.\n");
  fprintf(o, "LAMBDA=%.10f\nALPHA=%.6g\nBETA=1\nEPS_PR=0.01\nEPS_KATZ=%.6g\nEPS_EV=%.6g\n",
          lam, alpha, 0.01 * meanK, 0.01 * meanEV);
  fclose(o);
  printf("wrote %s\n", argv[2]);
  return 0;
}
