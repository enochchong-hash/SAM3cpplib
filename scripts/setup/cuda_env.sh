#!/usr/bin/env bash
# Discover CUDA in common versioned layouts. Source this file to update the
# caller's environment, or execute it to inspect the selected toolkit.

_sam3_cuda_sourced=false
[[ "${BASH_SOURCE[0]}" != "$0" ]] && _sam3_cuda_sourced=true

_sam3_cuda_usage() {
    cat <<'EOF'
Usage: scripts/setup/cuda_env.sh [--cuda-root PATH] [--quiet] [--print-env]

Selection order: --cuda-root; SAM3_CUDA_ROOT/CUDA_HOME/CUDA_PATH;
/usr/local/cuda-12; /usr/local/cuda; newest /usr/local/cuda-* or /opt/cuda-*.
Both lib, lib64, and targets/<architecture>/lib layouts are supported.
EOF
}

_sam3_cuda_main() {
    local requested="" quiet=false print_env=false candidate root="" lib existing seen version
    local -a candidates=() libdirs=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cuda-root) [[ $# -ge 2 ]] || return 2; requested=$2; shift 2 ;;
            --quiet) quiet=true; shift ;;
            --print-env) print_env=true; shift ;;
            --help) _sam3_cuda_usage; return 0 ;;
            *) echo "Error: unknown option: $1" >&2; _sam3_cuda_usage >&2; return 2 ;;
        esac
    done
    if [[ -n "$requested" ]]; then
        candidates+=("$requested")
    else
        [[ -n "${SAM3_CUDA_ROOT:-}" ]] && candidates+=("$SAM3_CUDA_ROOT")
        [[ -n "${CUDA_HOME:-}" ]] && candidates+=("$CUDA_HOME")
        [[ -n "${CUDA_PATH:-}" ]] && candidates+=("$CUDA_PATH")
        candidates+=(/usr/local/cuda-12 /usr/local/cuda)
        while IFS= read -r candidate; do [[ -n "$candidate" ]] && candidates+=("$candidate"); done \
            < <(find /usr/local /opt -maxdepth 1 -type d -name 'cuda-*' -print 2>/dev/null | sort -Vr)
    fi
    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate/bin/nvcc" ]] || continue
        root="$(cd "$candidate" && pwd -P)"
        break
    done
    if [[ -z "$root" ]]; then
        [[ "$quiet" == true ]] || echo "Error: no CUDA toolkit containing bin/nvcc was found" >&2
        return 1
    fi
    for lib in "$root/lib" "$root/lib64" \
        "$root/targets/$(uname -m)-linux/lib" "$root/targets/x86_64-linux/lib" "$root/targets/aarch64-linux/lib"; do
        [[ -d "$lib" ]] || continue
        seen=false
        for existing in "${libdirs[@]}"; do [[ "$existing" == "$lib" ]] && seen=true; done
        [[ "$seen" == true ]] || libdirs+=("$lib")
    done
    [[ ${#libdirs[@]} -gt 0 ]] || { [[ "$quiet" == true ]] || echo "Error: CUDA libraries not found below $root" >&2; return 1; }

    export SAM3_CUDA_ROOT="$root" CUDA_HOME="$root" CUDA_PATH="$root" CUDACXX="$root/bin/nvcc"
    case ":$PATH:" in *":$root/bin:"*) ;; *) export PATH="$root/bin:$PATH" ;; esac
    for lib in "${libdirs[@]}"; do
        case ":${LD_LIBRARY_PATH:-}:" in *":$lib:"*) ;; *) export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;; esac
    done
    # nvcc may report a non-zero status after a downstream `head` closes its
    # pipe. Do not let a caller's inherited `set -o pipefail` turn successful
    # toolkit discovery into a failure.
    version=$("$CUDACXX" --version 2>/dev/null | sed -n 's/.*release \([^,]*\).*/\1/p' | sed -n '1p') || version=""
    if [[ "$quiet" != true ]]; then
        echo "CUDA toolkit: $CUDA_HOME${version:+ (version $version)}"
        echo "CUDA compiler: $CUDACXX"
        printf 'CUDA libraries:'; printf ' %s' "${libdirs[@]}"; printf '\n'
    fi
    if [[ "$print_env" == true ]]; then
        printf 'export SAM3_CUDA_ROOT=%q\nexport CUDA_HOME=%q\nexport CUDA_PATH=%q\nexport CUDACXX=%q\nexport PATH=%q\nexport LD_LIBRARY_PATH=%q\n' \
            "$SAM3_CUDA_ROOT" "$CUDA_HOME" "$CUDA_PATH" "$CUDACXX" "$PATH" "${LD_LIBRARY_PATH:-}"
    fi
    return 0
}

if _sam3_cuda_main "$@"; then _sam3_cuda_status=0; else _sam3_cuda_status=$?; fi
unset -f _sam3_cuda_main _sam3_cuda_usage
if [[ "$_sam3_cuda_sourced" == true ]]; then unset _sam3_cuda_sourced; return "$_sam3_cuda_status"; fi
unset _sam3_cuda_sourced
exit "$_sam3_cuda_status"
