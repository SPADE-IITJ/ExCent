#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

typedef unsigned long long ull;

struct EdgeChunk {
    int v0, v1;
    ull e0, e1;
    ull edges() const { return e1 - e0; }
};

class StaticGraph {
public:
    int n = 0;
    ull m = 0;
    std::vector<ull> nindex;

    explicit StaticGraph(const char* path, int io_threads = 16, bool offsets_in_ram = true) : io_threads_(io_threads) {
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) { perror(path); exit(1); }
        int hdr[2];
        read_exact(hdr, sizeof hdr, 0);
        n = hdr[0];
        size_t off = sizeof hdr;
        bool wide = hdr[1] == -1;
        if (wide) { read_exact(&m, sizeof m, off); off += sizeof m; }
        else m = (ull)hdr[1];
        if (n < 1) { fprintf(stderr, "StaticGraph: bad header in %s\n", path); exit(1); }

        idx_off_ = off;
        if (!wide) offsets_in_ram = true;
        if (wide && !offsets_in_ram) {
            off += ((size_t)n + 1) * sizeof(ull);
        } else if (wide) {
            nindex.resize((size_t)n + 1);
            read_parallel(nindex.data(), nindex.size() * sizeof(ull), off);
            off += nindex.size() * sizeof(ull);
        } else {
            nindex.resize((size_t)n + 1);
            std::vector<int32_t> tmp((size_t)n + 1);
            read_parallel(tmp.data(), tmp.size() * sizeof(int32_t), off);
            off += tmp.size() * sizeof(int32_t);
            for (size_t i = 0; i < tmp.size(); i++) nindex[i] = (ull)(uint32_t)tmp[i];
        }
        nlist_off_ = off;

        struct stat st;
        fstat(fd_, &st);
        map_size_ = (size_t)st.st_size;
        map_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) { perror("mmap"); exit(1); }
       
        madvise(map_, map_size_, MADV_RANDOM);
        posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
        nlist_ = (const int*)((const char*)map_ + nlist_off_);
        off_ = offsets_in_ram ? nindex.data() : (const ull*)((const char*)map_ + idx_off_);
    }

    ~StaticGraph() {
        if (map_ && map_ != MAP_FAILED) munmap(map_, map_size_);
        if (fd_ >= 0) ::close(fd_);
    }
    StaticGraph(const StaticGraph&) = delete;
    StaticGraph& operator=(const StaticGraph&) = delete;

    ull offset(size_t v) const { return off_[v]; }
    ull degree(int v) const { return off_[v + 1] - off_[v]; }
    const int* edges(int v) const { return nlist_ + off_[v]; }

    void read_offsets(size_t v0, size_t cnt, ull* dst) const {
        if (!nindex.empty()) memcpy(dst, nindex.data() + v0, cnt * sizeof(ull));
        else read_parallel(dst, cnt * sizeof(ull), idx_off_ + v0 * sizeof(ull));
    }

    void read_edges(ull e0, ull e1, int* dst) const {
        read_parallel(dst, (e1 - e0) * sizeof(int), nlist_off_ + e0 * sizeof(int));
    }

    void prefetch_lists(const int* vids, int k, ull* pages, ull* runs) const {
        const size_t P = (size_t)sysconf(_SC_PAGESIZE);
        ull np = 0, nr = 0;
        const int parts = std::max(1, std::min(io_threads_, k / 1024));
        #pragma omp parallel for num_threads(parts) reduction(+ : np, nr) schedule(static, 1)
        for (int t = 0; t < parts; t++) {
            const int i0 = (int)((long long)k * t / parts), i1 = (int)((long long)k * (t + 1) / parts);
            size_t r0 = 0, r1 = 0; bool open = false;
            auto flush = [&]() {
                if (!open) return;
                madvise((char*)map_ + r0 * P, (r1 - r0) * P, MADV_WILLNEED);
                np += r1 - r0; nr++;
            };
            for (int i = i0; i < i1; i++) {
                const ull d = degree(vids[i]);
                if (!d) continue;
                const size_t b0 = nlist_off_ + off_[vids[i]] * sizeof(int), b1 = b0 + d * sizeof(int);
                const size_t p0 = b0 / P, p1 = (b1 + P - 1) / P;
                if (open && p0 <= r1) r1 = std::max(r1, p1);
                else { flush(); r0 = p0; r1 = p1; open = true; }
            }
            flush();
        }
        *pages = np; *runs = nr;
    }

    void drop_cache(ull e0, ull e1, size_t v0, size_t v1) const {
        const size_t P = (size_t)sysconf(_SC_PAGESIZE);
        auto drop = [&](size_t b0, size_t b1) {
            const size_t a0 = b0 / P * P, a1 = std::min(map_size_, (b1 + P - 1) / P * P);
            if (a1 <= a0) return;
            madvise((char*)map_ + a0, a1 - a0, MADV_DONTNEED);   // unmap our pages, or fadvise keeps them
            posix_fadvise(fd_, (off_t)a0, (off_t)(a1 - a0), POSIX_FADV_DONTNEED);
        };
        drop(nlist_off_ + e0 * sizeof(int), nlist_off_ + e1 * sizeof(int));
        drop(idx_off_ + v0 * sizeof(ull), idx_off_ + (v1 + 1) * sizeof(ull));
    }

    std::vector<EdgeChunk> plan_chunks(ull max_edges) const {
        std::vector<EdgeChunk> out;
        int v = 0;
        while (v < n) {
            ull limit = off_[v] + max_edges;
            int v1 = (int)(std::upper_bound(off_ + v + 1, off_ + n + 1, limit) - off_) - 1;
            if (v1 <= v) v1 = v + 1;
            out.push_back({v, v1, off_[v], off_[v1]});
            v = v1;
        }
        return out;
    }

private:
    int fd_ = -1;
    int io_threads_;
    size_t nlist_off_ = 0;
    size_t idx_off_ = 0;
    const ull* off_ = nullptr;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    const int* nlist_ = nullptr;

    void read_exact(void* dst, size_t bytes, size_t off) const {
        char* p = (char*)dst;
        while (bytes) {
            ssize_t r = pread(fd_, p, bytes, (off_t)off);
            if (r <= 0) { perror("pread"); exit(1); }
            p += r; off += (size_t)r; bytes -= (size_t)r;
        }
    }

    void read_parallel(void* dst, size_t bytes, size_t off) const {
        const size_t piece = 8ull << 20;
        const long long pieces = (long long)((bytes + piece - 1) / piece);
        int threads = (int)std::min<long long>(io_threads_, std::max<long long>(1, pieces));
        #pragma omp parallel for num_threads(threads) schedule(static, 1)
        for (long long i = 0; i < pieces; i++) {
            size_t b = (size_t)i * piece;
            size_t len = std::min(piece, bytes - b);
            read_exact((char*)dst + b, len, off + b);
        }
    }
};
