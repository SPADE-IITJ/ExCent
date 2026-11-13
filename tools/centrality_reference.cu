// centrality_reference.cu exact references for the three centralities and
// comparison of result dumps against them. Verification only: loads the whole
// graph onto the GPU and is never part of a timed run. Assumes a symmetric
// .egr, so CSR lists are also in-neighbour lists.
//
//   pr   : x = (1-d) + d * sum_{q in N(p)} x(q)/L(q)            (d = 0.85, unnormalised)
//   katz : x = beta + alpha * sum_{q in N(p)} x(q)
//   ev   : principal eigenvector of A (power iteration, unit L2 norm)
// pr and katz: double-precision Jacobi until max |change| < 1e-9 * max |x|.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_89 centrality_reference.cu -o ../bin/centrality_reference
// Usage: centrality_reference <graph.egr> pr|katz|ev <ref.bin> [--alpha A] [--beta B]   (writes float64[n])
//        centrality_reference --compare pr|katz|ev <ref.bin> <dump.bin>               (dump: float32[n])
//        centrality_reference --compare-dumps pr|katz|ev <a.bin> <b.bin>              (both float32[n])
//        centrality_reference --rank pr|katz|ev <ref.bin> <dump.bin>                  (precision@k, Kendall tau@1000)
// For ev both vectors are scaled to unit L2 norm (sign fixed by their sum) before comparing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

typedef unsigned long long ull;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// y[w] = c + s * sum_{u in N(w)} x[u] * (divdeg ? 1/L(u) : 1); maxabs[0] = max |y - x|, maxabs[1] = max |y|
__global__ void k_step(int n, const ull* idx, const int* adj, const double* x, double* y,
                       double c, double s, int divdeg, unsigned long long* mx) {
    long long w = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= n) return;
    double acc = 0;
    for (ull e = idx[w] + lane; e < idx[w + 1]; e += 32) {
        int u = adj[e];
        acc += divdeg ? x[u] / (double)(idx[u + 1] - idx[u]) : x[u];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) {
        double nv = c + s * acc;
        y[w] = nv;
        atomicMax(&mx[0], (unsigned long long)__double_as_longlong(fabs(nv - x[w])));
        atomicMax(&mx[1], (unsigned long long)__double_as_longlong(fabs(nv)));
    }
}

__global__ void k_sumsq(int n, const double* x, double* out) {
    double a = 0;
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n; v += (long long)gridDim.x * blockDim.x) a += x[v] * x[v];
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, a);
}
__global__ void k_scale(int n, double* x, double f) {
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n; v += (long long)gridDim.x * blockDim.x) x[v] *= f;
}

static std::vector<char> slurp(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> b(sz);
    if (fread(b.data(), 1, sz, f) != (size_t)sz) { perror("read"); exit(1); }
    fclose(f);
    return b;
}

static void unit(std::vector<double>& v) {
    double s = 0, t = 0;
    for (double a : v) { s += a * a; t += a; }
    double f = (t < 0 ? -1.0 : 1.0) / std::sqrt(s);
    for (double& a : v) a *= f;
}

static int compare(const std::string& measure, const char* refp, const char* dumpp, bool ref_f32 = false) {
    auto rb = slurp(refp), xb = slurp(dumpp);
    size_t n = rb.size() / (ref_f32 ? sizeof(float) : sizeof(double));
    if (xb.size() != n * sizeof(float)) { fprintf(stderr, "size mismatch: ref n=%zu, dump %zu floats\n", n, xb.size() / 4); return 1; }
    std::vector<double> r(n);
    if (ref_f32) { const float* rf = (const float*)rb.data(); for (size_t i = 0; i < n; i++) r[i] = rf[i]; }
    else r.assign((const double*)rb.data(), (const double*)rb.data() + n);
    std::vector<double> x(n);
    const float* xf = (const float*)xb.data();
    for (size_t i = 0; i < n; i++) x[i] = xf[i];
    if (measure == "ev") { unit(r); unit(x); }
    double l1 = 0, ref1 = 0, l2 = 0, maxrel = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(x[i] - r[i]);
        l1 += d; ref1 += fabs(r[i]); l2 += d * d;
        if (fabs(r[i]) > 1e-12) maxrel = std::max(maxrel, d / fabs(r[i]));
    }
    const int K = 100;
    std::vector<int> ir(n), ix(n);
    std::iota(ir.begin(), ir.end(), 0); std::iota(ix.begin(), ix.end(), 0);
    std::partial_sort(ir.begin(), ir.begin() + K, ir.end(), [&](int a, int b) { return r[a] > r[b]; });
    std::partial_sort(ix.begin(), ix.begin() + K, ix.end(), [&](int a, int b) { return x[a] > x[b]; });
    std::sort(ir.begin(), ir.begin() + K); std::sort(ix.begin(), ix.begin() + K);
    std::vector<int> common;
    std::set_intersection(ir.begin(), ir.begin() + K, ix.begin(), ix.begin() + K, std::back_inserter(common));
    printf("%s: n=%zu  L1 rel err %.3e  L2 err %.3e%s  top-%d overlap %zu/%d\n", measure.c_str(), n, l1 / ref1,
           std::sqrt(l2), measure == "ev" ? " (unit vectors)" : "", K, common.size(), K);
    return 0;
}


static int rank_quality(const std::string& measure, const char* refp, const char* dumpp) {
    auto rb = slurp(refp), xb = slurp(dumpp);
    size_t n = rb.size() / sizeof(double);
    if (xb.size() != n * sizeof(float)) { fprintf(stderr, "size mismatch\n"); return 1; }
    std::vector<double> r((const double*)rb.data(), (const double*)rb.data() + n), x(n);
    const float* xf = (const float*)xb.data();
    for (size_t i = 0; i < n; i++) x[i] = xf[i];
    if (measure == "ev") { unit(r); unit(x); }
    const size_t KMAX = std::min<size_t>(100000, n);
    std::vector<int> ir(n), ix(n);
    std::iota(ir.begin(), ir.end(), 0); std::iota(ix.begin(), ix.end(), 0);
    std::partial_sort(ir.begin(), ir.begin() + KMAX, ir.end(), [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
    std::partial_sort(ix.begin(), ix.begin() + KMAX, ix.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    printf("%s:", measure.c_str());
    for (size_t k : {10, 100, 1000, 10000, 100000}) {
        if (k > KMAX) break;
        std::vector<int> a(ir.begin(), ir.begin() + k), b(ix.begin(), ix.begin() + k), c;
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c));
        printf("  P@%zu %.4f", k, (double)c.size() / k);
    }
    const int T = (int)std::min<size_t>(1000, n);
    long long conc = 0, disc = 0;
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++) {
            double dr = r[ir[i]] - r[ir[j]], dx = x[ir[i]] - x[ir[j]];
            if (dr * dx > 0) conc++; else if (dr * dx < 0) disc++;
        }
    printf("  tau@1000 %.4f\n", (double)(conc - disc) / ((double)T * (T - 1) / 2));
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && !strcmp(argv[1], "--compare")) return compare(argv[2], argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "--rank")) return rank_quality(argv[2], argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "--compare-dumps")) return compare(argv[2], argv[3], argv[4], true);
    if (argc < 4) {
        fprintf(stderr, "usage: %s graph.egr pr|katz|ev ref.bin [--alpha A] [--beta B]\n"
                        "       %s --compare pr|katz|ev ref.bin dump.bin\n", argv[0], argv[0]);
        return 1;
    }
    std::string measure = argv[2];
    double alpha = -1, beta = 1;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--alpha")) alpha = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--beta")) beta = atof(argv[i + 1]);
    }
    if (measure == "katz" && alpha <= 0) { fprintf(stderr, "katz needs --alpha\n"); return 1; }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    int n, e32; ull m;
    if (fread(&n, 4, 1, f) != 1 || fread(&e32, 4, 1, f) != 1) return 1;
    std::vector<ull> idx((size_t)n + 1);
    if (e32 == -1) {
        if (fread(&m, 8, 1, f) != 1 || fread(idx.data(), 8, idx.size(), f) != idx.size()) return 1;
    } else {
        m = (ull)e32;
        std::vector<int> t(idx.size());
        if (fread(t.data(), 4, t.size(), f) != t.size()) return 1;
        for (size_t i = 0; i < t.size(); i++) idx[i] = (ull)(unsigned)t[i];
    }
    std::vector<int> adj(m);
    if (fread(adj.data(), 4, m, f) != m) return 1;
    fclose(f);

    ull* d_idx; int* d_adj; double *x, *y, *d_red; unsigned long long* d_mx;
    CK(cudaMalloc(&d_idx, idx.size() * sizeof(ull)));
    CK(cudaMalloc(&d_adj, m * sizeof(int)));
    CK(cudaMalloc(&x, (size_t)n * sizeof(double)));
    CK(cudaMalloc(&y, (size_t)n * sizeof(double)));
    CK(cudaMalloc(&d_mx, 2 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_red, sizeof(double)));
    CK(cudaMemcpy(d_idx, idx.data(), idx.size() * sizeof(ull), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_adj, adj.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<double> h(n, measure == "pr" ? 0.15 : 1.0);
    CK(cudaMemcpy(x, h.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned blocks = (unsigned)(((long long)n * 32 + 255) / 256);

    double c = 0, s = 1; int divdeg = 0;
    if (measure == "pr") { c = 0.15; s = 0.85; divdeg = 1; }
    else if (measure == "katz") { c = beta; s = alpha; }

    int it = 0; double rel = 1, lam = 0;
    for (; it < 20000 && rel > 1e-9; it++) {
        CK(cudaMemset(d_mx, 0, 2 * sizeof(unsigned long long)));
        k_step<<<blocks, 256>>>(n, d_idx, d_adj, x, y, c, s, divdeg, d_mx);
        if (measure == "ev") {
            CK(cudaMemset(d_red, 0, sizeof(double)));
            k_sumsq<<<1024, 256>>>(n, y, d_red);
            double ss; CK(cudaMemcpy(&ss, d_red, sizeof(double), cudaMemcpyDeviceToHost));
            lam = std::sqrt(ss);
            k_scale<<<1024, 256>>>(n, y, 1.0 / lam);
            std::vector<double> a(n), b(n);
            CK(cudaMemcpy(a.data(), x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), y, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
            double mxd = 0; for (int i = 0; i < n; i++) mxd = std::max(mxd, fabs(a[i] - b[i]));
            rel = mxd;
            if (it == 0) { unit(b); CK(cudaMemcpy(y, b.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice)); rel = 1; }
        } else {
            unsigned long long hm[2]; CK(cudaMemcpy(hm, d_mx, sizeof hm, cudaMemcpyDeviceToHost));
            double d, mxv; memcpy(&d, &hm[0], 8); memcpy(&mxv, &hm[1], 8);
            rel = d / mxv;
        }
        std::swap(x, y);
    }
    CK(cudaMemcpy(h.data(), x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    double sum = 0; for (double v : h) sum += v;
    printf("reference %s: %d iterations, final rel change %.2e, sum %.6e%s\n", measure.c_str(), it, rel, sum,
           measure == "ev" ? (" (unit L2; lambda1 ~ " + std::to_string(lam) + ")").c_str() : "");
    FILE* o = fopen(argv[3], "wb");
    fwrite(h.data(), sizeof(double), n, o);
    fclose(o);
    return 0;
}
