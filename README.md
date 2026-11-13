# ExCent: External Memory Iterative Centrality on Large Graphs over Memory-Constrained GPUs

ExCent is a runtime that helps computes iterative vertex centrality measures such as **PageRank, Katz and eigenvector centrality** on a static graph that stays on storage. The graph is a binary CSR file (EGR) read through `mmap`. It never has to fit in host memory or in GPU memory. Only the metadata such as vertex values and residuals live on the GPU, part of the graph is cached in the GPU memory that is left, and host memory holds two staging buffers and a bitmap. The same binary runs on discrete GPUs (explicit host-to-GPU copies) and on devices where the CPU and GPU share one memory, such as the Jetson AGX Orin (zero-copy access).

All these measures are asynchronous delta-push iterations of the same form: each vertex holds a value and a residual, and a vertex whose residual is above the threshold is *pending*; pushing it folds the residual into its value and adds a weighted share to each neighbour's residual. A run ends when no vertex is pending.

| Measure | `--measure` | Push weight | Given constants |
|---|---|---|---|
| PageRank | `pr` | d / deg(v), d = 0.85 | — |
| Katz | `katz` | alpha | alpha, beta |
| Eigenvector | `ev` | 1 / lambda1 | lambda1 |
| All three over one stream of the graph | `all` | | alpha, beta, lambda1 |

Every pass has three steps:

1. **Demand snapshot.** The GPU counts, for each chunk of the graph, its pending vertices and their total adjacency size (the chunk's *demand*).
2. **Fetch plan.** The CPU decides, chunk by chunk, whether to skip it, use its GPU-cached copy, *stream* the whole chunk from storage (when demand / edges ≥ τ), or *gather* only the pending vertices' adjacency lists. τ is measured once per graph and machine at start-up. Gathers request each storage page once, as asynchronous readahead.
3. **Overlapped execution.** Reads go into two pinned staging buffers, so the GPU pushes one chunk while the CPU reads the next.

## Requirements

- A CUDA-capable GPU and the CUDA toolkit (`nvcc`). Tested on an NVIDIA L40S GPU with CUDA 12.4 and on an NVIDIA Jetson AGX Orin. `nvcc` needs to support `-arch=native` (CUDA ≥ 11.6), or you pass your GPU's architecture explicitly (see [Building](#building)).
- A C++17 host compiler with OpenMP support (e.g. a recent `gcc`/`g++`).
- Linux (the runtime uses `mmap`, `pread`, `madvise` and `posix_fadvise`).
- Python 3 with `numpy` for `tools/efficient_converter.py` only.

## Cloning

```bash
git clone <this-repository-url> ExCent
cd ExCent
```

## Building

`run.sh` builds the binary the first time you run it, so normal use needs no separate build step. To build it directly:

```bash
nvcc -O3 -std=c++17 -Xcompiler -fopenmp -arch=native -Iinclude src/excent.cu -o bin/excent
```

On a toolkit without `-arch=native`, use your GPU's architecture instead, e.g. `-arch=sm_89` for an L40S or `-arch=sm_87` for a Jetson AGX Orin. `run.sh` reads it from the `EXCENT_ARCH_FLAG` environment variable.

## Running

On the small bundled sample graph:

```bash
./run.sh --measure=pr --sample
./run.sh --measure=katz --sample
./run.sh --measure=ev --sample
./run.sh --measure=all --sample
```

On your own graph:

```bash
./run.sh --measure=ev --graph=path/to/graph.egr --gpu=0
```

- `--measure` —> `pr`, `katz`, `ev`, or `all`.
- `--graph` —> the graph, as an `.egr` binary CSR file (see [Using your own graph](#using-your-own-graph)).
- `--params` —> the graph's given constants (lambda1, alpha, thresholds). Katz and eigenvector need them. If omitted, `run.sh` computes them once with `tools/excent_params.cu` and saves them as `<graph>.env` next to the graph.
- `--gpu` —> CUDA device ID. Defaults to `0`.
- Anything after `--` goes to the binary unchanged, e.g. `./run.sh --measure=pr --sample -- --cache none --dump out`.

Or call the binary directly:

```bash
bin/excent graph.egr --measure pr
bin/excent graph.egr --measure katz --alpha A --beta 1 --eps-katz E
bin/excent graph.egr --measure ev   --lambda L --eps-ev E
bin/excent graph.egr --measure all  --alpha A --lambda L --eps-katz E --eps-ev E
```

Each run prints one line per pass (pending vertices, demand ratio, and how many chunks were cached, streamed or gathered) and a summary with the bytes read from storage and the time breakdown. `--dump PREFIX` writes the result to `PREFIX.<measure>.bin` (float32 per vertex).

### Options

| Flag | Values | Meaning |
|---|---|---|
| `--stream-ratio` | `auto` (default), or a number | τ: stream a chunk whole when demand / edges ≥ τ, otherwise gather its pending lists. `auto` measures τ on this graph's storage at the first run and caches it in `~/.cache/excent/` (`--recalibrate` measures again). `0` streams only; a value above 1 gathers only. |
| `--cache` | `auto` (default), `chunks`, `hot`, `none` | GPU cache of the graph. `chunks`: whole chunks in file order. `hot`: adjacency lists of the highest-degree vertices. `auto`: `chunks` if the whole graph fits, otherwise `hot`. |
| `--zero-copy` | `auto` (default), `on`, `off` | Staging buffers pinned and GPU-mapped, read in place by the kernels with no host-to-GPU copy. `auto`: on for integrated GPUs (Jetson), off for discrete GPUs. |
| `--dedup` | `on` (default), `off` | Gather: request each storage page of the pending lists once, as asynchronous readahead, before copying the lists. |
| `--hot-when` | `end` (default), `start`, `both` | When the cached high-degree vertices are pushed within a pass (`--cache hot`). |
| `--chunk-medges` | default \|E\|/16, clamped to [4, 64] | Chunk size in millions of edges. |
| `--group` | default automatic | Threads per vertex (1–32); automatic = from the average degree of the vertices pushed. |
| `--list-frac` | default `0.5` | Push only the listed pending vertices of a chunk when fewer than this fraction are pending. |
| `--reserve-mb` | default `256` | GPU memory left free when sizing the cache. |
| `--eps-pr`, `--eps-katz`, `--eps-ev` | PageRank default `0.01` | Convergence thresholds (pending iff \|residual\| > eps). |
| `--max-passes` | default `100000` | Stop after this many passes. |

Diagnostics: `--trace-pending FILE` (uint16 per vertex: passes pending), `--count-pushes` (adjacency entries pushed; slows the kernels).

## Using your own graph

1. **Convert to `.egr`.** The input is a MatrixMarket-style `.mtx` edge list, 1-indexed, with every undirected edge listed in **both** directions (`u v` and `v u`). Eigenvector centrality needs a symmetric graph. `datasets/sample/sample.mtx` is a working example.

   ```bash
   python3 tools/efficient_converter.py your_graph.mtx your_graph.egr
   ```

2. **Compute the given constants** (lambda1 by power iteration, Katz alpha = 0.85 / lambda1, beta = 1, and thresholds of 0.01 × each measure's fixed-point mean). `run.sh` does this automatically; to do it yourself:

   ```bash
   nvcc -O3 -std=c++17 -arch=native tools/excent_params.cu -o bin/excent_params
   bin/excent_params your_graph.egr your_graph.env
   ```

   This tool loads the whole graph onto the GPU, so run it once on a GPU that can hold the graph. It is not part of a timed run.

3. **Check a result** against an exact double-precision reference (also loads the whole graph onto the GPU):

   ```bash
   nvcc -O3 -std=c++17 -arch=native tools/centrality_reference.cu -o bin/centrality_reference
   bin/centrality_reference your_graph.egr pr ref.pr.bin
   ./run.sh --measure=pr --graph=your_graph.egr -- --dump out
   bin/centrality_reference --compare pr ref.pr.bin out.pr.bin      # L1/L2 error, top-100 overlap
   bin/centrality_reference --rank pr ref.pr.bin out.pr.bin         # precision@k, Kendall tau@1000
   ```

## Measuring under memory limits

The tools below set up the conditions used in the paper's experiments:

- `tools/evict_file.c` drops a graph file from the OS page cache without root (`posix_fadvise`), for cold-cache runs:
  `gcc -O2 tools/evict_file.c -o bin/evict_file && bin/evict_file your_graph.egr`
- `tools/gpu_ballast.cu` holds GPU memory so that only a given budget stays free, giving every system the same GPU memory budget without modifying it:
  `nvcc -O2 tools/gpu_ballast.cu -o bin/gpu_ballast && CUDA_VISIBLE_DEVICES=0 bin/gpu_ballast 8 &` (wait for `READY`, run, then kill it).
- Host memory is capped with a cgroup that also charges the page cache, with swap off: `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 ./run.sh ...`.
- `tools/egr_info.cpp` prints an `.egr` header, maximum degree and a sampled symmetry check without loading the graph.

## File formats

**`.egr` binary CSR** (little-endian):
```
int32   nodes
int32   edge_count_or_sentinel   // -1 if the real count needs 64 bits
[int64  edge_count]              // present only if the above was -1
int32/int64  nindex[nodes + 1]   // int64 entries iff the sentinel was used
int32   nlist[edge_count]
```
`nindex[v] .. nindex[v+1]` indexes into `nlist` for vertex `v`'s neighbours.

**Params file** (`.env`, shell syntax): `LAMBDA`, `ALPHA`, `BETA`, `EPS_PR`, `EPS_KATZ`, `EPS_EV`. See `datasets/sample/sample.env`.

**Result dumps**: `float32[nodes]`, one value per vertex in vertex order.

## Project layout

```
ExCent/
├── run.sh                      # build (if needed) + run one measure
├── src/excent.cu               # the runtime: demand snapshot, fetch plan, page engine, GPU cache, kernels
├── include/StaticGraph.h       # storage-resident .egr reader: mmap'd offsets, parallel pread
├── tools/
│   ├── efficient_converter.py  # .mtx edge list -> .egr
│   ├── excent_params.cu        # lambda1, Katz alpha, thresholds (given constants)
│   ├── centrality_reference.cu # exact references and result comparison
│   ├── evict_file.c            # drop a file from the page cache (cold runs)
│   ├── gpu_ballast.cu          # fix the free GPU memory to a budget
│   └── egr_info.cpp            # .egr header and symmetry check
├── datasets/sample/            # small bundled graph (.mtx, .egr) and its params
└── bin/                        # build output
```
