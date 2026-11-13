// evict_file.c drop a file's pages from the OS page cache without root
// (posix_fadvise DONTNEED), so end-to-end runs can start from a cold cache
// for one graph file instead of `echo 3 > /proc/sys/vm/drop_caches`.
//
// Build: gcc -O2 evict_file.c -o ../bin/evict_file
// Usage: evict_file <file> [<file> ...]

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

// fraction of the file's pages currently resident in the page cache
static double resident_fraction(int fd, size_t sz) {
    if (sz == 0) return 0.0;
    void* p = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return -1.0;
    long pg = sysconf(_SC_PAGESIZE);
    size_t npages = (sz + pg - 1) / pg;
    unsigned char* vec = malloc(npages);
    double frac = -1.0;
    if (vec && mincore(p, sz, vec) == 0) {
        size_t r = 0;
        for (size_t i = 0; i < npages; i++) r += vec[i] & 1;
        frac = (double)r / npages;
    }
    free(vec);
    munmap(p, sz);
    return frac;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s [--report] <file> [...]\n", argv[0]); return 1; }
    int rc = 0;
    if (!strcmp(argv[1], "--report")) {   // print resident % only, evict nothing
        for (int i = 2; i < argc; i++) {
            int fd = open(argv[i], O_RDONLY);
            if (fd < 0) { perror(argv[i]); rc = 1; continue; }
            struct stat st;
            fstat(fd, &st);
            printf("%.1f\n", 100 * resident_fraction(fd, st.st_size));
            close(fd);
        }
        return rc;
    }
    for (int i = 1; i < argc; i++) {
        int fd = open(argv[i], O_RDONLY);
        if (fd < 0) { perror(argv[i]); rc = 1; continue; }
        struct stat st;
        fstat(fd, &st);
        double before = resident_fraction(fd, st.st_size);
        fdatasync(fd);
        int e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        double after = resident_fraction(fd, st.st_size);
        printf("%s: %.1f%% -> %.1f%% resident%s\n", argv[i], 100 * before, 100 * after,
               e ? " (fadvise failed)" : "");
        if (e || after > 0.01) rc = 1;
        close(fd);
    }
    return rc;
}
