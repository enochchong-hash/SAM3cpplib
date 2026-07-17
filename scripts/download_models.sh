#!/usr/bin/env bash
# Download the same pre-converted GGML checkpoints used by release/sam3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${SAM3_MODEL_DIR:-$ROOT_DIR/resources/models}"
BASE_URL="${SAM3_MODEL_BASE_URL:-https://huggingface.co/PABannier/sam3.cpp/resolve/main}"
SOURCE_DIR=""
FORCE=false
DRY_RUN=false
VARIANTS=()
SUPPORTED=(q8_0 f16 q4_0 q4_1 f32)

usage() {
    cat <<'EOF'
Usage: scripts/download_models.sh [options] [VARIANT ...]

Download ready-to-use sam3-VARIANT.ggml checkpoints. With no variant, q8_0
is selected (about 1.1 GB). Supported variants: q8_0, f16, q4_0, q4_1, f32.

Options:
  --model-dir PATH   Destination (default: resources/models)
  --base-url URL     Alternate mirror/base URL
  --source-dir PATH  Import from an existing/offline checkpoint directory
  --all              Acquire every supported variant
  --force            Replace an existing checkpoint
  --dry-run          Print planned sources and destinations only
  --list             List supported variants
  --help             Show this help

Environment equivalents: SAM3_MODEL_DIR, SAM3_MODEL_BASE_URL.
EOF
}

is_supported() {
    local wanted=$1 item
    for item in "${SUPPORTED[@]}"; do [[ "$wanted" == "$item" ]] && return 0; done
    return 1
}

validate_model() {
    local path=$1 size magic
    [[ -f "$path" ]] || return 1
    size=$(wc -c < "$path")
    (( size >= 1000000 )) || return 1
    magic=$(od -An -tx1 -N4 "$path" | tr -d ' \n')
    [[ "$magic" == "336d6173" ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; MODEL_DIR=$2; shift 2 ;;
        --base-url) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; BASE_URL=${2%/}; shift 2 ;;
        --source-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; SOURCE_DIR=$2; shift 2 ;;
        --all) VARIANTS=("${SUPPORTED[@]}"); shift ;;
        --force) FORCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --list) printf '%s\n' "${SUPPORTED[@]}"; exit 0 ;;
        --help) usage; exit 0 ;;
        --) shift; VARIANTS+=("$@"); break ;;
        -*) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) VARIANTS+=("$1"); shift ;;
    esac
done
[[ ${#VARIANTS[@]} -gt 0 ]] || VARIANTS=(q8_0)

if [[ -n "$SOURCE_DIR" ]]; then
    [[ -d "$SOURCE_DIR" ]] || { echo "Error: source directory not found: $SOURCE_DIR" >&2; exit 1; }
elif ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "Error: install curl or wget, or use --source-dir for an offline import" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"
for variant in "${VARIANTS[@]}"; do
    is_supported "$variant" || { echo "Error: unsupported model variant: $variant" >&2; exit 2; }
    filename="sam3-$variant.ggml"
    target="$MODEL_DIR/$filename"
    if [[ "$FORCE" == false ]] && validate_model "$target"; then
        echo "Already present: $target"
        continue
    fi
    if [[ -n "$SOURCE_DIR" ]]; then source="$SOURCE_DIR/$filename"; else source="$BASE_URL/$filename"; fi
    if [[ "$DRY_RUN" == true ]]; then
        echo "$source -> $target"
        continue
    fi
    partial="$target.part"
    echo "Acquiring $filename"
    if [[ -n "$SOURCE_DIR" ]]; then
        validate_model "$source" || { echo "Error: invalid or missing checkpoint: $source" >&2; exit 1; }
        cp --reflink=auto "$source" "$partial" 2>/dev/null || cp "$source" "$partial"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --retry-delay 2 --continue-at - --output "$partial" "$source"
    else
        wget --continue --output-document="$partial" "$source"
    fi
    validate_model "$partial" || { echo "Error: downloaded file is not a valid SAM3 GGML checkpoint" >&2; exit 1; }
    mv "$partial" "$target"
    echo "Ready: $target ($(du -h "$target" | cut -f1))"
done
