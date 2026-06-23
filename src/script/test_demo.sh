#!/usr/bin/env bash
# MAVFusion demo: run inference on a short IR-Visible video sequence.
# This script assumes a demo dataset has been prepared; see README for the
# expected directory layout.
set -e
set -x

python test_demo.py \
    --task_name IVF \
    --dataset_name VTMOT \
    --exp_path MAVFusion \
    --fps 24 \
    --bitrate 50000
