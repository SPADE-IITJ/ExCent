#!/bin/bash
# run.sh -- build (if needed) and run ExCent on one graph.
#
# Usage:
#   ./run.sh --measure=pr|katz|ev|all --graph=<graph.egr> [--params=<graph.env>] [--gpu=ID] [-- excent flags...]
#   ./run.sh --measure=pr|katz|ev|all --sample [-- excent flags...]
#
# Katz and eigenvector centrality need the graph's given constants (lambda1, alpha, thresholds).
# Without --params they are computed once with tools/excent_params.cu and saved next to the
# graph as <graph>.env, which later runs reuse.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $0 --measure=pr|katz|ev|all --graph=<graph.egr> [--params=<graph.env>] [--gpu=ID] [-- excent flags...]
       $0 --measure=pr|katz|ev|all --sample [-- excent flags...]

  --measure   pr (PageRank), katz, ev (eigenvector), or all (the three together, one stream of the graph)
  --graph     path to the graph, as an .egr binary CSR file (see README, "Using your own graph")
  --params    env file with LAMBDA, ALPHA, BETA, EPS_PR, EPS_KATZ, EPS_EV
              (default: <graph>.env, computed with bin/excent_params if it does not exist)
  --gpu       CUDA device id to run on (default: 0)
  --sample    use the small bundled graph under datasets/sample/
  --          everything after it is passed to bin/excent unchanged (e.g. -- --cache none --dump out)

Environment overrides:
  NVCC               nvcc binary to build with (default: nvcc on PATH)
  EXCENT_ARCH_FLAG   nvcc architecture flag (default: -arch=native, needs CUDA >= 11.6;
                     on an older toolkit set it explicitly, e.g. EXCENT_ARCH_FLAG=-arch=sm_87)
EOF
}

MEASURE=""
GRAPH=""
PARAMS=""
GPU="0"
USE_SAMPLE=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --measure=*) MEASURE="${1#--measure=}" ;;
        --graph=*)   GRAPH="${1#--graph=}" ;;
        --params=*)  PARAMS="${1#--params=}" ;;
        --gpu=*)     GPU="${1#--gpu=}" ;;
        --sample)    USE_SAMPLE=1 ;;
        --)          shift; EXTRA=("$@"); break ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "ERROR: unrecognized argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

case "$MEASURE" in
    pr|katz|ev|all) ;;
    "") echo "ERROR: --measure=pr|katz|ev|all is required" >&2; usage; exit 1 ;;
    *)  echo "ERROR: --measure must be pr, katz, ev, or all, got '$MEASURE'" >&2; exit 1 ;;
esac

if [ "$USE_SAMPLE" = "1" ]; then
    GRAPH="${GRAPH:-$SCRIPT_DIR/datasets/sample/sample.egr}"
    PARAMS="${PARAMS:-$SCRIPT_DIR/datasets/sample/sample.env}"
fi
[ -n "$GRAPH" ] || { echo "ERROR: --graph is required (or use --sample)" >&2; usage; exit 1; }
[ -f "$GRAPH" ] || { echo "ERROR: graph file not found: $GRAPH" >&2; exit 1; }

NVCC="${NVCC:-nvcc}"
ARCH="${EXCENT_ARCH_FLAG:--arch=native}"
build() {   # build <source> <binary>
    if [ ! -x "$2" ]; then
        echo "Building $2 ..."
        mkdir -p "$SCRIPT_DIR/bin"
        "$NVCC" -O3 -std=c++17 -Xcompiler -fopenmp $ARCH -I"$SCRIPT_DIR/include" "$1" -o "$2"
    fi
}

export CUDA_VISIBLE_DEVICES="$GPU"

ARGS=(--measure "$MEASURE")
if [ "$MEASURE" != "pr" ]; then
    PARAMS="${PARAMS:-${GRAPH%.egr}.env}"
    if [ ! -f "$PARAMS" ]; then
        build "$SCRIPT_DIR/tools/excent_params.cu" "$SCRIPT_DIR/bin/excent_params"
        echo "Computing the given constants of $GRAPH -> $PARAMS ..."
        "$SCRIPT_DIR/bin/excent_params" "$GRAPH" "$PARAMS"
    fi
    # shellcheck disable=SC1090
    source "$PARAMS"
    case "$MEASURE" in
        katz) ARGS+=(--alpha "$ALPHA" --beta "$BETA" --eps-katz "$EPS_KATZ") ;;
        ev)   ARGS+=(--lambda "$LAMBDA" --eps-ev "$EPS_EV") ;;
        all)  ARGS+=(--alpha "$ALPHA" --beta "$BETA" --lambda "$LAMBDA"
                     --eps-pr "$EPS_PR" --eps-katz "$EPS_KATZ" --eps-ev "$EPS_EV") ;;
    esac
fi

build "$SCRIPT_DIR/src/excent.cu" "$SCRIPT_DIR/bin/excent"
exec "$SCRIPT_DIR/bin/excent" "$GRAPH" "${ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}"
