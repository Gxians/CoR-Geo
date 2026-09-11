#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINO_ROOT="$PROJECT_ROOT/third_party/dinov2"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/dinov2"
CHECKPOINT_PATH="$CHECKPOINT_ROOT/dinov2_vitb14_pretrain.pth"
DINO_COMMIT="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

SOURCE_ONLY=false
case "${1:-}" in
    "") ;;
    --source-only) SOURCE_ONLY=true ;;
    *) echo "Usage: bash scripts/setup_dinov2.sh [--source-only]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/setup_dinov2.sh [--source-only]" >&2
    exit 2
fi

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
mkdir -p "$PROJECT_ROOT/third_party"

if [[ -e "$DINO_ROOT" && ! -d "$DINO_ROOT/.git" ]]; then
    echo "Existing path is not a Git repository: $DINO_ROOT" >&2
    exit 1
fi

if [[ ! -d "$DINO_ROOT/.git" ]]; then
    git clone https://github.com/facebookresearch/dinov2.git "$DINO_ROOT"
fi

if [[ -n "$(git -C "$DINO_ROOT" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to change a modified DINOv2 checkout: $DINO_ROOT" >&2
    exit 1
fi

if ! git -C "$DINO_ROOT" cat-file -e "${DINO_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$DINO_ROOT" fetch origin
fi
git -C "$DINO_ROOT" checkout --detach "$DINO_COMMIT"

if [[ "$SOURCE_ONLY" == true ]]; then
    echo "DINOv2 source:  $DINO_ROOT"
    exit 0
fi

mkdir -p "$CHECKPOINT_ROOT"
if [[ ! -s "$CHECKPOINT_PATH" ]]; then
    PARTIAL_PATH="${CHECKPOINT_PATH}.part"
    if command -v wget >/dev/null 2>&1; then
        wget --continue --output-document="$PARTIAL_PATH" "$CHECKPOINT_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --continue-at - --output "$PARTIAL_PATH" "$CHECKPOINT_URL"
    else
        echo "wget or curl is required to download DINOv2 weights" >&2
        exit 1
    fi
    mv "$PARTIAL_PATH" "$CHECKPOINT_PATH"
fi

echo "DINOv2 source:  $DINO_ROOT"
echo "DINOv2 weights: $CHECKPOINT_PATH"
