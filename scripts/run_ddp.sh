#!/bin/bash
# DDP training launch script
# Usage: bash scripts/run_ddp.sh [num_gpus]

NUM_GPUS=${1:-2}

echo "Starting DDP training with $NUM_GPUS GPUs..."

torchrun --nproc_per_node=$NUM_GPUS \
    -m bulba1.cli \
    --config configs/default.yaml \
    --distributed

echo "DDP training completed!"