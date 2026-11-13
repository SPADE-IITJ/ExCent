#include <cstdio>
#include <cstdint>
#include <random>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
typedef unsigned long long ull;
int main(int c, char** v) {
  int fd = open(v[1], O_RDONLY); struct stat st; fstat(fd, &st);
  const char* b = (const char*)mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  int n = *(const int*)b; int e32 = *(const int*)(b + 4);
  bool wide = e32 == -1; ull m = wide ? *(const ull*)(b + 8) : (ull)e32;
  size_t off = wide ? 16 : 8;
  auto idx = [&](ull i) -> ull { return wide ? ((const ull*)(b + off))[i] : (ull)((const uint32_t*)(b + off))[i]; };
  const int* adj = (const int*)(b + off + (size_t)(n + 1) * (wide ? 8 : 4));
  std::mt19937_64 r(7); int S = 20000, bad = 0, self = 0;
  for (int i = 0; i < S; i++) {
    ull e = r() % m; int lo = 0, hi = n;
    while (hi - lo > 1) { int mid = (lo + hi) / 2; if (idx(mid) <= e) lo = mid; else hi = mid; }
    int s = lo, d = adj[e]; if (s == d) self++;
    const int* p = adj + idx(d); const int* q = adj + idx(d + 1);
    bool found = std::is_sorted(p, q) ? std::binary_search(p, q, s) : std::find(p, q, s) != q;
    if (!found) bad++;
  }
  ull maxd = 0; for (int i = 0; i < n; i++) maxd = std::max(maxd, idx(i + 1) - idx(i));
  printf("n=%d m=%llu (%.2f GB edges) maxdeg=%llu  reverse-edge missing %d/%d (%.1f%%) self-loops %d  m>2^32-1: %s\n",
         n, m, m * 4.0 / 1e9, maxd, bad, S, 100.0 * bad / S, self, m > 4294967295ull ? "YES" : "no");
}
