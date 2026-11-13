// Build: nvcc -O2 gpu_ballast.cu -o ../bin/gpu_ballast
// Usage: CUDA_VISIBLE_DEVICES=<d> gpu_ballast <budget_GB>

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <vector>

static volatile sig_atomic_t stop = 0;
static void on_signal(int) { stop = 1; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <budget_GB>\n", argv[0]); return 1; }
    const size_t budget = (size_t)(atof(argv[1]) * (1ull << 30));
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    if (freeB <= budget) { fprintf(stderr, "only %.2f GB free, below budget\n", freeB / 1073741824.0); return 1; }

    // grab in 256 MB pieces so fragmentation can't stop us short of the target
    std::vector<void*> held;
    const size_t piece = 256ull << 20;
    while (true) {
        cudaMemGetInfo(&freeB, &totalB);
        if (freeB <= budget) break;
        size_t want = freeB - budget < piece ? freeB - budget : piece;
        void* p;
        if (cudaMalloc(&p, want) != cudaSuccess) break;
        held.push_back(p);
    }
    cudaMemGetInfo(&freeB, &totalB);
    printf("READY free=%.2f GB (budget %.2f GB) of %.2f GB\n", freeB / 1073741824.0,
           budget / 1073741824.0, totalB / 1073741824.0);
    fflush(stdout);
    while (!stop) pause();
    for (void* p : held) cudaFree(p);
    return 0;
}
