#!/usr/bin/env bash
# Download the pretrained SEA-RAFT optical flow model used by MAVFusion.
# The checkpoint is required at: checkpoint/Tartan-C-T-TSKH-spring540x960-S.pth
# (this path is referenced by config/module/spring-S.json)
set -e

DST_DIR="checkpoint"
CKPT_NAME="Tartan-C-T-TSKH-spring540x960-S.pth"

if [ ! -d "$DST_DIR" ]; then
    mkdir -p "$DST_DIR"
fi

if [ -f "$DST_DIR/$CKPT_NAME" ]; then
    echo "SEA-RAFT checkpoint already present at $DST_DIR/$CKPT_NAME — skipping."
    exit 0
fi

cd "$DST_DIR"

# Try the official Princeton-VL SEA-RAFT release.
URL_PRIMARY="https://drive.google.com/uc?id=1R4f2SGJF3JJc2s6cPeogm4gfo7jhcFtH&export=download"
URL_FALLBACK="https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/Tartan-C-T-TSKH-spring540x960-S.pth"

if command -v gdown >/dev/null 2>&1; then
    echo "Downloading SEA-RAFT checkpoint via gdown..."
    gdown "$URL_PRIMARY" -O "$CKPT_NAME" || {
        echo "Primary download failed. Trying fallback URL..."
        wget -nv --show-progress "$URL_FALLBACK" -O "$CKPT_NAME"
    }
else
    echo "Downloading SEA-RAFT checkpoint via wget (fallback URL)..."
    wget -nv --show-progress "$URL_FALLBACK" -O "$CKPT_NAME"
fi

echo "Done. SEA-RAFT checkpoint saved to $DST_DIR/$CKPT_NAME"
