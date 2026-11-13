/*
ExCent: PageRank, Katz and eigenvector centrality on a static graph that
stays on storage, computed on a GPU with bounded host memory.
*/

#include "StaticGraph.h"

#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static constexpr float DAMP = 0.85f;
static constexpr unsigned FULL = 0xffffffffu;
#define MAXM 3
enum { M_PR = 0, M_KATZ = 1, M_EV = 2 };
static const char* MNAME[MAXM] = {"pr", "katz", "ev"};

typedef std::chrono::steady_clock Clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// Per-run measure table, passed by value to every kernel.
struct Measures {
    int nm;                       // number of measures in this run
    float* x[MAXM];               // values
    float* r[MAXM];               // residuals
    float eps[MAXM];              // pending iff |r| > eps
    float scale[MAXM];            // push weight numerator
    int divdeg[MAXM];             // 1: weight = scale / L(v) (PageRank)
    ull* pushed;                  // total adjacency entries pushed (--count-pushes; nullptr otherwise)
};

// --------------------------------------------------------------------- kernels

__global__ void k_init(int n, const ull* idx, float* x, float* r, int kind, float beta, float lambda) {
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < n; v += (long long)gridDim.x * blockDim.x) {
        if (kind == M_PR)        { x[v] = 0.f; r[v] = 1.f - DAMP; }
        else if (kind == M_KATZ) { x[v] = 0.f; r[v] = beta; }
        else                     { x[v] = 1.f; r[v] = (float)((double)(idx[v + 1] - idx[v]) / lambda - 1.0); }
    }
}

__device__ __forceinline__ int chunk_of(int v, const int* cv0, int nc) {
    int lo = 0, hi = nc - 1;   // last chunk with cv0[c] <= v
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (cv0[mid] <= v) lo = mid; else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ bool pending(const Measures& M, long long v) {
    bool p = false;
    #pragma unroll
    for (int k = 0; k < MAXM; k++)
        if (k < M.nm) p |= fabsf(M.r[k][v]) > M.eps[k];
    return p;
}

__device__ __forceinline__ bool hot_bit(const unsigned* hot, long long v) {
    return hot && ((hot[v >> 5] >> (v & 31)) & 1u);
}
__global__ void k_snapshot(int n, Measures M, const ull* idx, const int* cv0, int nc,
                           unsigned* bitmap, ull* cnt,
                           const unsigned* hot, ull* hotpend, unsigned short* trace) {
    long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool in = v < n;
    bool pend = in && pending(M, v);
    if (trace && pend) trace[v]++;   // --trace-pending: passes in which v was pending
    if (hot) {
        bool ph = pend && hot_bit(hot, v);
        unsigned hb = __ballot_sync(FULL, ph);
        if (lane == 0 && hb) atomicAdd(hotpend, (ull)__popc(hb));
        pend = pend && !ph;
    }
    unsigned b = __ballot_sync(FULL, pend);
    if (lane == 0 && (v >> 5) < ((n + 31) >> 5)) bitmap[v >> 5] = b;
    if (!b) return;   // warp-uniform
    int c = in ? chunk_of((int)v, cv0, nc) : -1;
    int c0 = __shfl_sync(FULL, c, 0);
    ull d = pend ? idx[v + 1] - idx[v] : 0;
    bool same = __all_sync(FULL, !in || c == c0);
    if (same) {
        ull nv = __popc(b), ne = d;
        for (int o = 16; o > 0; o >>= 1) ne += __shfl_down_sync(FULL, ne, o);
        if (lane == 0) { atomicAdd(&cnt[2 * c0], nv); atomicAdd(&cnt[2 * c0 + 1], ne); }
    } else if (pend) {
        atomicAdd(&cnt[2 * c], 1ull);
        atomicAdd(&cnt[2 * c + 1], d);
    }
}

template <int G>
__device__ __forceinline__ void push_vertex_g(bool valid, int v, const int* nbrs, ull deg, const Measures& M, int gl) {
    float p[MAXM];
    bool any = false;
    #pragma unroll
    for (int k = 0; k < MAXM; k++) {
        p[k] = 0.f;
        if (k >= M.nm) continue;
        float d = 0.f;
        if (valid && gl == 0 && fabsf(M.r[k][v]) > M.eps[k]) {
            d = atomicExch(&M.r[k][v], 0.f);
            M.x[k][v] += d;   // a vertex lives in exactly one chunk -> one writer
        }
        d = __shfl_sync(FULL, d, 0, G);
        if (d != 0.f) { any = true; p[k] = M.divdeg[k] ? M.scale[k] * d / (float)deg : M.scale[k] * d; }
    }
    if (!any || deg == 0) return;   // group-uniform; no shuffles below
    if (M.pushed && gl == 0) atomicAdd(M.pushed, deg);   // --count-pushes only: one shared counter
    for (ull i = gl; i < deg; i += G) {
        int u = nbrs[i];
        #pragma unroll
        for (int k = 0; k < MAXM; k++)
            if (p[k] != 0.f) atomicAdd(&M.r[k][u], p[k]);
    }
}

// Stream / resident chunk: vertices [v0, v1), edges laid out from global edge e0.
template <int G>
__global__ void k_push_range(int v0, int v1, ull e0, const int* edges, const ull* idx, Measures M,
                             const unsigned* hot) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (v0 + (t >> 5) * (32 / G) >= v1) return;   // whole warp past the range (warp-uniform)
    int gl = (int)(t % G);
    long long v = v0 + t / G;
    bool valid = v < v1 && !hot_bit(hot, v);       // a cached high-degree vertex is pushed by the hot list
    ull b = valid ? idx[v] : 0, d = valid ? idx[v + 1] - b : 0;
    push_vertex_g<G>(valid, (int)(valid ? v : 0), edges + (b - (valid ? e0 : 0)), d, M, gl);
}

// Gather chunk: k pending vertices, list i at edges[off[i] .. off[i+1]).
template <int G>
__global__ void k_push_list(int k, const int* vids, const ull* off, const int* edges, Measures M) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if ((t >> 5) * (32 / G) >= k) return;
    int gl = (int)(t % G);
    long long i = t / G;
    bool valid = i < k;
    push_vertex_g<G>(valid, valid ? vids[i] : 0, edges + (valid ? off[i] : 0), valid ? off[i + 1] - off[i] : 0, M, gl);
}

__device__ __forceinline__ void warp_append(bool p, int value, int* out, unsigned* count) {
    const int lane = threadIdx.x & 31;
    const unsigned b = __ballot_sync(FULL, p);
    if (!b) return;   // warp-uniform
    unsigned base = 0;
    if (lane == 0) base = atomicAdd(count, (unsigned)__popc(b));
    base = __shfl_sync(FULL, base, 0);
    if (p) out[base + __popc(b & ((1u << lane) - 1u))] = value;
}
__global__ void k_compact(int v0, int v1, const unsigned* bitmap, int* out, unsigned* count) {
    long long v = v0 + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (v0 + (long long)blockIdx.x * blockDim.x + (threadIdx.x & ~31) >= v1) return;   // whole warp past v1
    bool p = v < v1 && ((bitmap[v >> 5] >> (v & 31)) & 1u);
    warp_append(p, (int)v, out, count);
}

template <int G>
__global__ void k_push_ids(int k, const int* ids, const ull* idx, const int* edges, ull e0, Measures M) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if ((t >> 5) * (32 / G) >= k) return;
    int gl = (int)(t % G);
    long long i = t / G;
    bool valid = i < k;
    int v = valid ? ids[i] : 0;
    ull b = valid ? idx[v] : 0, d = valid ? idx[v + 1] - b : 0;
    push_vertex_g<G>(valid, v, edges + (b - (valid ? e0 : 0)), d, M, gl);
}

__global__ void k_compact_hot(int lo, int hi, const int* hids, Measures M, int* out, unsigned* count) {
    long long i = lo + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (lo + (long long)blockIdx.x * blockDim.x + (threadIdx.x & ~31) >= hi) return;   // whole warp past hi
    warp_append(i < hi && pending(M, hids[i]), (int)i, out, count);
}

template <int G>
__global__ void k_push_hot_list(const int* list, const unsigned* count, const int* hids, const ull* hoff,
                                const int* hedges, Measures M) {
    const long long k = *count;
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int gl = (int)(t % G), grp = (int)((t & 31) / G);
    const long long per = 32 / G, step = ((long long)gridDim.x * blockDim.x >> 5) * per;
    for (long long base = (t >> 5) * per; base < k; base += step) {
        long long e = base + grp;
        bool valid = e < k;
        int i = valid ? list[e] : 0;
        push_vertex_g<G>(valid, valid ? hids[i] : 0, hedges + (valid ? hoff[i] : 0), valid ? hoff[i + 1] - hoff[i] : 0, M, gl);
    }
}

// Push k listed cached vertices, one group of G threads each (k known on the host).
template <int G>
__global__ void k_push_hot_ids(int k, const int* list, const int* hids, const ull* hoff, const int* hedges, Measures M) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if ((t >> 5) * (32 / G) >= k) return;
    int gl = (int)(t % G);
    long long e = t / G;
    bool valid = e < k;
    int i = valid ? list[e] : 0;
    push_vertex_g<G>(valid, valid ? hids[i] : 0, hedges + (valid ? hoff[i] : 0), valid ? hoff[i + 1] - hoff[i] : 0, M, gl);
}

// Pass 0: copy the lists of cached vertices [lo, hi) from a chunk staged at src (edges from e0).
__global__ void k_fill_hot(int lo, int hi, const int* hids, const ull* hoff, const ull* idx, ull e0,
                           const int* src, int* hedges) {
    long long w = lo + ((blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (w >= hi) return;
    const int* from = src + (idx[hids[w]] - e0);
    int* to = hedges + hoff[w];
    for (ull i = lane, d = hoff[w + 1] - hoff[w]; i < d; i += 32) to[i] = from[i];
}

// ------------------------------------------------------------------------ host

struct Slot {
    int* h_edges; int* d_edges;
    int* h_vids;  int* d_vids;
    ull* h_off;   ull* d_off;
    unsigned* d_count;   // k_compact counter
    int* d_list;         // k_compact output (pending vertices of a chunk launched from GPU memory)
    cudaStream_t stream;
    cudaEvent_t done;
    bool busy = false;
};

static unsigned warp_grid(long long items) { return (unsigned)((items * 32 + 255) / 256); }
static unsigned group_grid(long long items, int G) { return (unsigned)((items * G + 255) / 256); }
// threads per vertex for a launch whose vertices have average degree avg: next power of two, 1..32
static int pick_group(double avg) { int g = 1; while (g < 32 && g < avg) g <<= 1; return g; }
#define GROUP_SWITCH(G, LAUNCH) switch (G) { case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; \
    case 4: LAUNCH(4); break; case 8: LAUNCH(8); break; case 16: LAUNCH(16); break; default: LAUNCH(32); }

int main(int argc, char** argv) {
    auto T0 = Clock::now();
    if (argc < 2) {
        fprintf(stderr, "usage: %s graph.egr --measure pr|katz|ev|all [--alpha A] [--beta 1] [--lambda L] "
                        "[--eps-pr 0.01] [--eps-katz E] [--eps-ev E] [--chunk-medges auto] [--target-chunks 16] "
                        "[--min-chunk-medges 4] [--list-frac 0.5] [--stream-ratio auto|R] "
                        "[--reserve-mb 256] [--cache chunks|hot|none] [--max-passes N] [--dump PREFIX]\n", argv[0]);
        return 1;
    }
    double chunk_medges = 0, stream_ratio = 2.0, reserve_mb = 256;
    double target_chunks = 16, min_chunk_medges = 4;
    double list_frac = 0.5;
    bool use_cache = true;
    std::string cache_mode = "auto";
    std::string hot_when = "end";
    std::string hot_order = "sweep";
    std::string trace_file;
    bool stream_auto = true;
    bool recalibrate = false;
    bool dedup = true;
    std::string hot_rank;
    int group_opt = 0;
    bool count_pushes = false;
    std::string zero_copy = "auto";
    int max_passes = 100000;
    std::string dump, measure;
    double alpha = -1, beta = 1.0, lambda = -1, eps_pr = 0.01, eps_katz = -1, eps_ev = -1;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "--chunk-medges") chunk_medges = atof(next());
        else if (a == "--target-chunks") target_chunks = atof(next());
        else if (a == "--min-chunk-medges") min_chunk_medges = atof(next());
        else if (a == "--list-frac") list_frac = atof(next());
        else if (a == "--stream-ratio") { std::string v = next(); stream_auto = v == "auto"; if (!stream_auto) stream_ratio = atof(v.c_str()); }
        else if (a == "--dedup") { std::string v = next(); if (v != "on" && v != "off") { fprintf(stderr, "--dedup must be on or off\n"); return 1; } dedup = v == "on"; }
        else if (a == "--hot-rank") hot_rank = next();
        else if (a == "--recalibrate") recalibrate = true;
        else if (a == "--reserve-mb") reserve_mb = atof(next());
        else if (a == "--no-cache") use_cache = false;
        else if (a == "--cache") cache_mode = next();
        else if (a == "--hot-when") hot_when = next();
        else if (a == "--hot-order") hot_order = next();
        else if (a == "--trace-pending") trace_file = next();
        else if (a == "--group") group_opt = atoi(next());
        else if (a == "--count-pushes") count_pushes = true;
        else if (a == "--zero-copy") zero_copy = next();
        else if (a == "--max-passes") max_passes = atoi(next());
        else if (a == "--dump") dump = next();
        else if (a == "--measure") measure = next();
        else if (a == "--alpha") alpha = atof(next());
        else if (a == "--beta") beta = atof(next());
        else if (a == "--lambda") lambda = atof(next());
        else if (a == "--eps-pr") eps_pr = atof(next());
        else if (a == "--eps-katz") eps_katz = atof(next());
        else if (a == "--eps-ev") eps_ev = atof(next());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }

    if (!use_cache) cache_mode = "none";
    if (cache_mode != "chunks" && cache_mode != "hot" && cache_mode != "none" && cache_mode != "auto") { fprintf(stderr, "--cache must be auto, chunks, hot or none\n"); return 1; }
    // auto is decided once the remaining GPU memory is known (below)
    use_cache = cache_mode == "chunks";
    bool hot_mode = cache_mode == "hot";

    std::vector<int> kinds;
    if (measure == "pr") kinds = {M_PR};
    else if (measure == "katz") kinds = {M_KATZ};
    else if (measure == "ev") kinds = {M_EV};
    else if (measure == "all") kinds = {M_PR, M_KATZ, M_EV};
    else { fprintf(stderr, "--measure must be pr, katz, ev or all\n"); return 1; }
    for (int k : kinds) {
        if (k == M_KATZ && (alpha <= 0 || eps_katz <= 0)) { fprintf(stderr, "katz needs --alpha and --eps-katz\n"); return 1; }
        if (k == M_EV && (lambda <= 0 || eps_ev <= 0)) { fprintf(stderr, "ev needs --lambda and --eps-ev\n"); return 1; }
    }

    StaticGraph g(argv[1], 16, /*offsets_in_ram=*/false);
    const int n = g.n;
    const ull m = g.m;
    ull chunk_cap;
    if (chunk_medges > 0) chunk_cap = (ull)(chunk_medges * 1e6);
    else {
        const ull lo = (ull)(min_chunk_medges * 1e6), hi = 64000000ull;
        chunk_cap = std::min(hi, std::max(lo, (ull)std::ceil((double)m / target_chunks)));
    }
    std::vector<EdgeChunk> chunks = g.plan_chunks(chunk_cap);
    const int nc = (int)chunks.size();
    ull max_chunk_edges = 0; int max_chunk_verts = 0;
    for (auto& c : chunks) { max_chunk_edges = std::max(max_chunk_edges, c.edges()); max_chunk_verts = std::max(max_chunk_verts, c.v1 - c.v0); }
    auto T_open = Clock::now();
    printf("graph: n=%d m=%llu chunks=%d (cap %llu, max %llu edges, %d vertices)  open %.2fs\n",
           n, m, nc, chunk_cap, max_chunk_edges, max_chunk_verts, secs(T0, T_open));

    CK(cudaSetDevice(0));
    CK(cudaFree(0));
    ull* d_idx; unsigned* d_bitmap; ull* d_cnt; int* d_cv0;
    const size_t words = ((size_t)n + 31) / 32;
    CK(cudaMalloc(&d_idx, ((size_t)n + 1) * sizeof(ull)));
    Measures M{};
    M.nm = (int)kinds.size();
    M.pushed = nullptr;
    if (count_pushes) { CK(cudaMalloc(&M.pushed, sizeof(ull))); CK(cudaMemset(M.pushed, 0, sizeof(ull))); }
    for (int i = 0; i < M.nm; i++) {
        CK(cudaMalloc(&M.x[i], (size_t)n * sizeof(float)));
        CK(cudaMalloc(&M.r[i], (size_t)n * sizeof(float)));
        int k = kinds[i];
        M.eps[i] = (float)(k == M_PR ? eps_pr : k == M_KATZ ? eps_katz : eps_ev);
        M.scale[i] = (float)(k == M_PR ? DAMP : k == M_KATZ ? alpha : 1.0 / lambda);
        M.divdeg[i] = k == M_PR;
    }
    CK(cudaMalloc(&d_bitmap, words * sizeof(unsigned)));
    CK(cudaMalloc(&d_cnt, 2 * (size_t)nc * sizeof(ull)));
    CK(cudaMalloc(&d_cv0, (size_t)nc * sizeof(int)));
    {
        std::vector<int> cv0(nc);
        for (int c = 0; c < nc; c++) cv0[c] = chunks[c].v0;
        CK(cudaMemcpy(d_cv0, cv0.data(), nc * sizeof(int), cudaMemcpyHostToDevice));
    }
    {   
        const size_t B = (size_t)1 << 24;
        ull* stage; CK(cudaMallocHost(&stage, B * sizeof(ull)));
        for (size_t v0 = 0; v0 < (size_t)n + 1; v0 += B) {
            size_t cnt = std::min(B, (size_t)n + 1 - v0);
            g.read_offsets(v0, cnt, stage);
            CK(cudaMemcpy(d_idx + v0, stage, cnt * sizeof(ull), cudaMemcpyHostToDevice));
        }
        CK(cudaFreeHost(stage));
    }
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int G = sms * 8;
    for (int i = 0; i < M.nm; i++) k_init<<<G, 256>>>(n, d_idx, M.x[i], M.r[i], kinds[i], (float)beta, (float)lambda);

    int integrated = 0;
    CK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0));
    const bool zc = zero_copy == "on" || (zero_copy == "auto" && integrated);
    auto stage_alloc = [&](void** h, void** d, size_t bytes) {
        if (zc) { CK(cudaHostAlloc(h, bytes, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer(d, *h, 0)); }
        else { CK(cudaMallocHost(h, bytes)); CK(cudaMalloc(d, bytes)); }
    };
    Slot slot[2];
    for (auto& s : slot) {
        stage_alloc((void**)&s.h_edges, (void**)&s.d_edges, max_chunk_edges * sizeof(int));
        stage_alloc((void**)&s.h_vids, (void**)&s.d_vids, (size_t)max_chunk_verts * sizeof(int));
        stage_alloc((void**)&s.h_off, (void**)&s.d_off, ((size_t)max_chunk_verts + 1) * sizeof(ull));
        CK(cudaMalloc(&s.d_list, (size_t)max_chunk_verts * sizeof(int)));
        CK(cudaMalloc(&s.d_count, sizeof(unsigned)));
        CK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    }
    cudaStream_t rstream;
    CK(cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking));
    int* d_rlist; unsigned* d_rcount;
    CK(cudaMalloc(&d_rlist, (size_t)max_chunk_verts * sizeof(int)));
    CK(cudaMalloc(&d_rcount, sizeof(unsigned)));
    unsigned* h_bitmap; ull* h_cnt;
    CK(cudaMallocHost(&h_bitmap, words * sizeof(unsigned)));
    CK(cudaMallocHost(&h_cnt, 2 * (size_t)nc * sizeof(ull)));

    std::vector<int*> resident(nc, nullptr);
    int n_resident = 0; ull resident_edges = 0;
    if (cache_mode == "auto") {
        size_t freeB, totalB;
        CK(cudaMemGetInfo(&freeB, &totalB));
        const size_t reserve = (size_t)(reserve_mb * (1 << 20));
        const bool fits = freeB > reserve && m * sizeof(int) <= freeB - reserve;
        use_cache = fits; hot_mode = !fits;
        printf("cache: auto -> %s\n", fits ? "chunks (the whole graph fits)" : "high-degree lists");
    }
    if (use_cache) {
        size_t freeB, totalB;
        CK(cudaMemGetInfo(&freeB, &totalB));
        size_t reserve = (size_t)(reserve_mb * (1 << 20));
        size_t avail = freeB > reserve ? freeB - reserve : 0;
        for (int c = 0; c < nc; c++) {
            size_t need = chunks[c].edges() * sizeof(int);
            if (need > avail) break;
            if (cudaMalloc(&resident[c], need) != cudaSuccess) { resident[c] = nullptr; cudaGetLastError(); break; }
            avail -= need; n_resident++; resident_edges += chunks[c].edges();
        }
    }

    int nh = 0; ull hot_edges = 0, hot_deg_min = 0;
    int* d_hids = nullptr; ull* d_hoff = nullptr; int* d_hedges = nullptr; unsigned* d_hot = nullptr;
    int* d_hlist = nullptr; unsigned* d_hcount = nullptr;
    CK(cudaMalloc(&d_hcount, sizeof(unsigned)));
    std::vector<int> hlo(nc, 0), hhi(nc, 0);
    if (hot_mode) {
        size_t freeB, totalB;
        CK(cudaMemGetInfo(&freeB, &totalB));
        const size_t reserve = (size_t)(reserve_mb * (1 << 20)), bits = words * sizeof(unsigned);
        const size_t budget = freeB > reserve + bits ? freeB - reserve - bits : 0;
        auto cost = [](ull d) { return d * sizeof(int) + 2 * sizeof(int) + sizeof(ull); }; 
        const ull CAP = 1ull << 20;
        std::vector<ull> hist(CAP, 0), big;
        #pragma omp parallel num_threads(16)
        {
            std::vector<ull> lh(CAP, 0), lb;
            #pragma omp for schedule(static) nowait
            for (int v = 0; v < n; v++) { ull d = g.degree(v); if (d >= CAP) lb.push_back(d); else lh[d]++; }
            #pragma omp critical
            {
                for (ull d = 0; d < CAP; d++) hist[d] += lh[d];
                big.insert(big.end(), lb.begin(), lb.end());
            }
        }
        std::sort(big.rbegin(), big.rend());
        size_t used = 0; ull T = 0, quota = 0; bool done = false;
        for (ull d : big) { if (used + cost(d) > budget) { T = d; quota = 0; done = true; break; } used += cost(d); T = d; quota++; }
        if (!done) { quota = 0; T = CAP - 1; }
        for (ull d = CAP - 1; !done && d >= 1; d--) {
            if (!hist[d]) continue;
            ull take = std::min<ull>(hist[d], (budget - used) / cost(d));
            used += take * cost(d);
            T = d; quota = take;
            if (take < hist[d]) done = true;
        }
        
        std::vector<unsigned short> score;
        int S = 0; size_t s_left = 0;
        if (!hot_rank.empty()) {
            score.resize(n);
            FILE* f = fopen(hot_rank.c_str(), "rb");
            if (!f || fread(score.data(), sizeof(unsigned short), n, f) != (size_t)n) { fprintf(stderr, "--hot-rank: cannot read %d entries from %s\n", n, hot_rank.c_str()); return 1; }
            fclose(f);
            std::vector<double> cs(65536, 0.0);
            for (int v = 0; v < n; v++) { ull d = g.degree(v); if (d && score[v]) cs[score[v]] += (double)cost(d); }
            double u = 0; S = 0; s_left = 0;
            for (int s = 65535; s >= 1; s--) {
                if (u + cs[s] <= (double)budget) { u += cs[s]; continue; }
                S = s; s_left = budget - (size_t)u; break;
            }
        }

        std::vector<int> ids; std::vector<ull> off{0};
        std::vector<unsigned> hb(words, 0);
        for (int v = 0; v < n; v++) {
            ull d = g.degree(v);
            if (d == 0) continue;
            if (!score.empty()) {
                if (score[v] == 0 || score[v] < S) continue;
                if (score[v] == S) { if (cost(d) > s_left) continue; s_left -= cost(d); }
            } else {
                bool take = d > T || (d == T && quota > 0);
                if (!take) continue;
                if (d == T) quota--;
            }
            ids.push_back(v); off.push_back(off.back() + d);
            hb[v >> 5] |= 1u << (v & 31);
        }
        nh = (int)ids.size(); hot_edges = off.back(); hot_deg_min = score.empty() ? T : 0;
        bool ok = nh > 0;
        ok = ok && cudaMalloc(&d_hedges, std::max<ull>(hot_edges, 1) * sizeof(int)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hids, (size_t)nh * sizeof(int)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hoff, ((size_t)nh + 1) * sizeof(ull)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hot, bits) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hlist, (size_t)nh * sizeof(int)) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            fprintf(stderr, "degree cache: could not allocate %d lists (%.2f GB); running without a cache\n", nh, hot_edges * 4.0 / 1e9);
            if (d_hedges) cudaFree(d_hedges); if (d_hids) cudaFree(d_hids); if (d_hoff) cudaFree(d_hoff); if (d_hot) cudaFree(d_hot);
            if (d_hlist) cudaFree(d_hlist);
            d_hedges = nullptr; d_hids = nullptr; d_hoff = nullptr; d_hot = nullptr; d_hlist = nullptr; nh = 0; hot_edges = 0;
        } else {
            CK(cudaMemcpy(d_hids, ids.data(), (size_t)nh * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_hoff, off.data(), ((size_t)nh + 1) * sizeof(ull), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_hot, hb.data(), bits, cudaMemcpyHostToDevice));
            for (int c = 0; c < nc; c++) {
                hlo[c] = (int)(std::lower_bound(ids.begin(), ids.end(), chunks[c].v0) - ids.begin());
                hhi[c] = (int)(std::lower_bound(ids.begin(), ids.end(), chunks[c].v1) - ids.begin());
            }
        }
    }
   
    double t_calib = 0;

    std::string tau_file;
    if (stream_auto) {
        const char* home = getenv("HOME");
        std::string base = argv[1];
        base = base.substr(base.find_last_of('/') + 1);
        std::string dir = std::string(home ? home : ".") + "/.cache";
        mkdir(dir.c_str(), 0755); dir += "/excent"; mkdir(dir.c_str(), 0755);
        tau_file = dir + "/" + base + "." + std::to_string(n) + "." + std::to_string(m) + ".tau";
        FILE* f = recalibrate ? nullptr : fopen(tau_file.c_str(), "r");
        if (f) {
            if (fscanf(f, "%lf %lf", &stream_ratio, &t_calib) == 2) {
                printf("fetch rule: tau = %.3f (measured earlier on this machine in %.2fs, %s)\n", stream_ratio, t_calib, tau_file.c_str());
                stream_auto = false;
            }
            fclose(f);
        }
    }
    if (stream_auto) {
        auto c0 = Clock::now();
        const EdgeChunk& ch = chunks[nc / 2];
        Slot& s = slot[0];
        g.drop_cache(ch.e0, ch.e1, ch.v0, ch.v1);
        auto a = Clock::now();
        g.read_edges(ch.e0, ch.e1, s.h_edges);
        const double t_stream = secs(a, Clock::now());
        const double fr[3] = {0.05, 0.15, 0.4};
        double f_got[3], t_g[3];

        std::vector<std::pair<ull, int>> byd;
        for (int v = ch.v0; v < ch.v1; v++) byd.push_back({g.degree(v), v});
        std::sort(byd.begin(), byd.end(), [](auto& x, auto& y) { return x.first > y.first; });
        for (int j = 0; j < 3; j++) {
            std::vector<int> pick;
            for (ull acc = 0; acc < fr[j] * ch.edges() && pick.size() < byd.size(); ) { acc += byd[pick.size()].first; pick.push_back(byd[pick.size()].second); }
            std::sort(pick.begin(), pick.end());
            g.drop_cache(ch.e0, ch.e1, ch.v0, ch.v1);
            a = Clock::now();
            int k = 0;
            for (int v : pick) s.h_vids[k++] = v;
            s.h_off[0] = 0;
            for (int i = 0; i < k; i++) s.h_off[i + 1] = s.h_off[i] + g.degree(s.h_vids[i]);
            ull pg, rn;
            if (dedup) g.prefetch_lists(s.h_vids, k, &pg, &rn);
            #pragma omp parallel for schedule(dynamic, 256) num_threads(16)
            for (int i = 0; i < k; i++) memcpy(s.h_edges + s.h_off[i], g.edges(s.h_vids[i]), g.degree(s.h_vids[i]) * sizeof(int));
            t_g[j] = secs(a, Clock::now());
            f_got[j] = (double)s.h_off[k] / ch.edges();
        }

        if (t_g[0] >= t_stream) stream_ratio = f_got[0] * t_stream / t_g[0];
        else if (t_g[2] < t_stream) stream_ratio = f_got[2] * t_stream / t_g[2];
        else {
            int j = t_g[1] >= t_stream ? 1 : 2;
            stream_ratio = f_got[j - 1] + (f_got[j] - f_got[j - 1]) * (t_stream - t_g[j - 1]) / (t_g[j] - t_g[j - 1]);
        }
        stream_ratio = std::min(1.0, std::max(0.005, stream_ratio));
        t_calib = secs(c0, Clock::now());
        printf("fetch rule: tau = %.3f (stream %.3fs; gather %.3fs / %.3fs / %.3fs at demand %.3f / %.3f / %.3f)  calibrate %.2fs\n",
               stream_ratio, t_stream, t_g[0], t_g[1], t_g[2], f_got[0], f_got[1], f_got[2], t_calib);
        if (FILE* f = fopen(tau_file.c_str(), "w")) { fprintf(f, "%.6f %.4f\n", stream_ratio, t_calib); fclose(f); }
    } else if (tau_file.empty()) {
        printf("fetch rule: tau = %.3f (given)\n", stream_ratio);
    }
    CK(cudaDeviceSynchronize());
    auto T_setup = Clock::now();
    printf("staging: %s\n", zc ? "zero copy (pinned, GPU-mapped buffers read in place)" : "pinned host buffers copied to the GPU");
    if (hot_mode && !hot_rank.empty())
        printf("GPU cache ranked by passes pending (%s): %d vertices (%.1f%% of edges, %.2f GB)  setup %.2fs\n",
               hot_rank.c_str(), nh, 100.0 * hot_edges / m, hot_edges * 4.0 / 1e9, secs(T_open, T_setup));
    else if (hot_mode)
        printf("GPU degree cache: %d vertices with degree >= %llu (%.1f%% of edges, %.2f GB)  setup %.2fs\n",
               nh, hot_deg_min, 100.0 * hot_edges / m, hot_edges * 4.0 / 1e9, secs(T_open, T_setup));
    else
        printf("GPU edge cache: %d/%d chunks resident (%.1f%% of edges)  setup %.2fs\n",
               n_resident, nc, 100.0 * resident_edges / m, secs(T_open, T_setup));

    // ---- passes
    double t_io = 0, t_wait = 0, t_snap = 0;
    ull bytes_stream = 0, bytes_gather = 0;
    int pass = 0;
    std::vector<bool> loaded(nc, false);
    int cur = 0;
    auto acquire = [&](Slot& s) {
        if (s.busy) { auto a = Clock::now(); CK(cudaEventSynchronize(s.done)); t_wait += secs(a, Clock::now()); s.busy = false; }
    };

    ull n_list = 0, n_range = 0;
    const unsigned* hot_skip = nullptr;
    ull* d_hotpend; CK(cudaMalloc(&d_hotpend, sizeof(ull)));
    unsigned short* d_trace = nullptr;
    if (!trace_file.empty()) { CK(cudaMalloc(&d_trace, (size_t)n * sizeof(unsigned short))); CK(cudaMemset(d_trace, 0, (size_t)n * sizeof(unsigned short))); }
    ull h_hotpend = 0;
    const int G_hot = group_opt ? group_opt : pick_group(nh ? (double)hot_edges / nh : 32.0);
    unsigned* h_hcount; CK(cudaMallocHost(&h_hcount, sizeof(unsigned)));
   
    std::vector<std::pair<int, int>> hot_ranges;
    if (hot_order == "ranges") { for (int c = 0; c < nc; c++) if (hhi[c] > hlo[c]) hot_ranges.push_back({hlo[c], hhi[c]}); }
    else if (nh) hot_ranges.push_back({0, nh});
    auto launch_hot = [&](cudaStream_t st) {
        for (auto [lo, hi] : hot_ranges) {
            CK(cudaMemsetAsync(d_hcount, 0, sizeof(unsigned), st));
            k_compact_hot<<<(unsigned)((hi - lo + 255) / 256), 256, 0, st>>>(lo, hi, d_hids, M, d_hlist, d_hcount);

            CK(cudaMemcpyAsync(h_hcount, d_hcount, sizeof(unsigned), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            const int k = (int)*h_hcount;
            if (!k) continue;
#define L_HOT(GG) k_push_hot_ids<GG><<<group_grid(k, GG), 256, 0, st>>>(k, d_hlist, d_hids, d_hoff, d_hedges, M)
            GROUP_SWITCH(G_hot, L_HOT)
#undef L_HOT
        }
    };
    auto group_of = [&](int c) {
        if (group_opt) return group_opt;
        const ull k = h_cnt[2 * c];
        return pick_group(k ? (double)h_cnt[2 * c + 1] / k : 1.0);
    };
    auto launch_chunk = [&](int c, const int* dst, int* list, unsigned* count, cudaStream_t st) {
        const EdgeChunk& ch = chunks[c];
        const ull k = h_cnt[2 * c];
        const int nv = ch.v1 - ch.v0;
        const int G = group_of(c);
        if ((double)k < list_frac * nv) {
            n_list++;
            CK(cudaMemsetAsync(count, 0, sizeof(unsigned), st));
            k_compact<<<(unsigned)((nv + 255) / 256), 256, 0, st>>>(ch.v0, ch.v1, d_bitmap, list, count);
#define L_IDS(GG) k_push_ids<GG><<<group_grid((long long)k, GG), 256, 0, st>>>((int)k, list, d_idx, dst, ch.e0, M)
            GROUP_SWITCH(G, L_IDS)
#undef L_IDS
        } else {
            n_range++;
#define L_RANGE(GG) k_push_range<GG><<<group_grid(nv, GG), 256, 0, st>>>(ch.v0, ch.v1, ch.e0, dst, d_idx, M, hot_skip)
            GROUP_SWITCH(G, L_RANGE)
#undef L_RANGE
        }
    };

    auto faults = []() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return std::make_pair((long)ru.ru_majflt, (long)ru.ru_minflt); };
    long tot_maj = 0, tot_min = 0; ull tot_pages = 0, tot_runs = 0;
    for (; pass < max_passes; pass++) {
        auto a = Clock::now();
        const auto f0 = faults();
        ull pass_pages = 0, pass_runs = 0;
        CK(cudaMemsetAsync(d_cnt, 0, 2 * (size_t)nc * sizeof(ull), 0));
        CK(cudaMemsetAsync(d_hotpend, 0, sizeof(ull), 0));
        hot_skip = (hot_mode && nh > 0 && pass > 0) ? d_hot : nullptr;
        k_snapshot<<<(unsigned)(((size_t)n + 255) / 256), 256>>>(n, M, d_idx, d_cv0, nc, d_bitmap, d_cnt, hot_skip, d_hotpend, d_trace);
        CK(cudaMemcpy(&h_hotpend, d_hotpend, sizeof(ull), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_cnt, d_cnt, 2 * (size_t)nc * sizeof(ull), cudaMemcpyDeviceToHost));
        ull pend_v = 0, demand = 0;
        bool need_bitmap = false;
        std::vector<char> mode(nc, 0);   // 0 skip, 1 resident, 2 stream, 3 gather
        for (int c = 0; c < nc; c++) {
            ull cv = h_cnt[2 * c], ce = h_cnt[2 * c + 1];
            pend_v += cv; demand += ce;
            if (cv == 0) continue;
            if (resident[c] && loaded[c]) mode[c] = 1;
            else if (resident[c] || (double)ce >= stream_ratio * chunks[c].edges()) mode[c] = 2;
            else if (hot_mode && pass == 0 && hhi[c] > hlo[c]) mode[c] = 2;
            else { mode[c] = 3; need_bitmap = true; }
        }
        if (need_bitmap) CK(cudaMemcpy(h_bitmap, d_bitmap, words * sizeof(unsigned), cudaMemcpyDeviceToHost));
        t_snap += secs(a, Clock::now());
        if (pend_v == 0 && h_hotpend == 0) break;

        int nres = 0, nstream = 0, ngather = 0;
        if (hot_skip && h_hotpend && hot_when != "end") launch_hot(rstream);
        for (int c = 0; c < nc; c++) {
            if (mode[c] != 1) continue;
            nres++;
            launch_chunk(c, resident[c], d_rlist, d_rcount, rstream);
        }
        for (int c = 0; c < nc; c++) {
            if (mode[c] < 2) continue;
            const EdgeChunk& ch = chunks[c];
            Slot& s = slot[cur];
            cur ^= 1;
            acquire(s);
            auto io0 = Clock::now();
            if (mode[c] == 2) {
                nstream++;
                g.read_edges(ch.e0, ch.e1, s.h_edges);
                bytes_stream += ch.edges() * sizeof(int);
                t_io += secs(io0, Clock::now());
                int* dst = resident[c] ? resident[c] : s.d_edges;
                if (!zc) CK(cudaMemcpyAsync(dst, s.h_edges, ch.edges() * sizeof(int), cudaMemcpyHostToDevice, s.stream));
                else if (resident[c]) CK(cudaMemcpyAsync(dst, s.d_edges, ch.edges() * sizeof(int), cudaMemcpyDeviceToDevice, s.stream));   // cache fill, pass 0 only
                if (hot_mode && pass == 0 && hhi[c] > hlo[c])
                    k_fill_hot<<<warp_grid(hhi[c] - hlo[c]), 256, 0, s.stream>>>(hlo[c], hhi[c], d_hids, d_hoff, d_idx, ch.e0, dst, d_hedges);
                launch_chunk(c, dst, s.d_list, s.d_count, s.stream);
                if (resident[c]) loaded[c] = true;
            } else {
                ngather++;

                int k = 0;
                for (int v = ch.v0; v < ch.v1;) {
                    unsigned w = h_bitmap[v >> 5] >> (v & 31);
                    if (!w) { v = (v | 31) + 1; continue; }
                    int t = __builtin_ctz(w);
                    v += t;
                    if (v >= ch.v1) break;
                    s.h_vids[k++] = v;
                    v++;
                }
                s.h_off[0] = 0;
                for (int i = 0; i < k; i++) s.h_off[i + 1] = s.h_off[i] + g.degree(s.h_vids[i]);
                const ull ne = s.h_off[k];
                if (dedup) { ull pg, rn; g.prefetch_lists(s.h_vids, k, &pg, &rn); pass_pages += pg; pass_runs += rn; }
                #pragma omp parallel for schedule(dynamic, 256) num_threads(16)
                for (int i = 0; i < k; i++) {
                    int v = s.h_vids[i];
                    memcpy(s.h_edges + s.h_off[i], g.edges(v), g.degree(v) * sizeof(int));
                }
                bytes_gather += ne * sizeof(int) + (size_t)k * (sizeof(int) + sizeof(ull));
                t_io += secs(io0, Clock::now());
                if (!zc) {
                    CK(cudaMemcpyAsync(s.d_vids, s.h_vids, (size_t)k * sizeof(int), cudaMemcpyHostToDevice, s.stream));
                    CK(cudaMemcpyAsync(s.d_off, s.h_off, ((size_t)k + 1) * sizeof(ull), cudaMemcpyHostToDevice, s.stream));
                    if (ne) CK(cudaMemcpyAsync(s.d_edges, s.h_edges, ne * sizeof(int), cudaMemcpyHostToDevice, s.stream));
                }
                const int G = group_opt ? group_opt : pick_group(k ? (double)ne / k : 1.0);
#define L_LIST(GG) k_push_list<GG><<<group_grid(k, GG), 256, 0, s.stream>>>(k, s.d_vids, s.d_off, s.d_edges, M)
                if (k) { GROUP_SWITCH(G, L_LIST) }
#undef L_LIST
            }
            CK(cudaEventRecord(s.done, s.stream));
            s.busy = true;
        }
        if (hot_skip && hot_when != "start") {
            for (auto& s : slot) if (s.busy) CK(cudaStreamWaitEvent(rstream, s.done, 0));
            launch_hot(rstream);
        }
        auto w0 = Clock::now();
        CK(cudaDeviceSynchronize());
        for (auto& s : slot) s.busy = false;
        t_wait += secs(w0, Clock::now());
        CK(cudaGetLastError());
        const auto f1 = faults();
        tot_maj += f1.first - f0.first; tot_min += f1.second - f0.second; tot_pages += pass_pages; tot_runs += pass_runs;
        printf("pass %3d: pending %llu v, demand ratio %.4f | resident %d stream %d gather %d | cached high-degree pending %llu"
               " | faults major %ld minor %ld | gather pages %llu runs %llu\n",
               pass, pend_v, (double)demand / m, nres, nstream, ngather, h_hotpend,
               f1.first - f0.first, f1.second - f0.second, pass_pages, pass_runs);
    }
    auto T_iter = Clock::now();

    std::vector<double> sum(M.nm, 0.0);
    {
        const size_t B = (size_t)1 << 24;
        std::vector<float> slice(std::min(B, (size_t)n));
        for (int i = 0; i < M.nm; i++) {
            FILE* df = nullptr;
            if (!dump.empty()) { std::string p = dump + "." + MNAME[kinds[i]] + ".bin"; df = fopen(p.c_str(), "wb"); }
            for (size_t v0 = 0; v0 < (size_t)n; v0 += B) {
                size_t cnt = std::min(B, (size_t)n - v0);
                CK(cudaMemcpy(slice.data(), M.x[i] + v0, cnt * sizeof(float), cudaMemcpyDeviceToHost));
                for (size_t j = 0; j < cnt; j++) sum[i] += slice[j];
                if (df) fwrite(slice.data(), sizeof(float), cnt, df);
            }
            if (df) fclose(df);
        }
    }
    if (d_trace) {
        FILE* tf = fopen(trace_file.c_str(), "wb");
        const size_t B = (size_t)1 << 24;
        std::vector<unsigned short> sl(std::min(B, (size_t)n));
        for (size_t v0 = 0; v0 < (size_t)n; v0 += B) {
            size_t cnt = std::min(B, (size_t)n - v0);
            CK(cudaMemcpy(sl.data(), d_trace + v0, cnt * sizeof(unsigned short), cudaMemcpyDeviceToHost));
            fwrite(sl.data(), sizeof(unsigned short), cnt, tf);
        }
        fclose(tf);
    }
    auto T_end = Clock::now();

    printf("passes %d  measure %s (n=%d):", pass, measure.c_str(), n);
    for (int i = 0; i < M.nm; i++) printf("  %s sum %.6e", MNAME[kinds[i]], sum[i]);
    printf("\n");
    if (M.pushed) { ull pushed = 0; CK(cudaMemcpy(&pushed, M.pushed, sizeof(ull), cudaMemcpyDeviceToHost));
      printf("edges pushed: %.2f x |E|\n", (double)pushed / m); }
    printf("launches from GPU memory: %llu pending-only, %llu whole-range\n", n_list, n_range);
    printf("bytes shipped: stream %.2f GB, gather %.2f GB (graph edges %.2f GB)\n",
           bytes_stream / 1e9, bytes_gather / 1e9, m * 4.0 / 1e9);
    printf("page faults in passes: major %ld, minor %ld | gather dedup %s: %llu pages in %llu runs\n",
           tot_maj, tot_min, dedup ? "on" : "off", tot_pages, tot_runs);
    printf("time: open %.2fs setup %.2fs passes %.2fs [cpu io %.2fs, gpu wait %.2fs, snapshot %.2fs] result %.2fs | internal total %.2fs\n",
           secs(T0, T_open), secs(T_open, T_setup), secs(T_setup, T_iter), t_io, t_wait, t_snap,
           secs(T_iter, T_end), secs(T0, T_end));
    return 0;
}
