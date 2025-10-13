#!/bin/bash

echo "Start multi-GPU training..."

# Set environment variables to avoid conflicts
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CUDA_LAUNCH_BLOCKING=1

# Check GPU count
GPU_COUNT=2
echo "Detected GPU count: $GPU_COUNT"

# Choose config based on GPU count
if [ $GPU_COUNT -ge 4 ]; then
    echo "Using 4-GPU config..."
    NUM_PROCESSES=4
elif [ $GPU_COUNT -ge 3 ]; then
    echo "Using 3-GPU config..."
    NUM_PROCESSES=3
else
    echo "Using 2-GPU config..."
    NUM_PROCESSES=2
fi

# Unified NCCL configuration
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

echo "Training with $NUM_PROCESSES GPUs"

# Method 1: use accelerate launch (recommended)
echo "Launching with accelerate..."
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_PROCESSES \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29500 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m akgr.abduction_model.main \
    --modelname='mydream' \
    --data_root='./sampled_data/' \
    -d='WN18RR' \
    --checkpoint_root='./checkpoints/' \
    --save_frequency 5 \
    --mode='training' \
    --training_mode='sft' \
    --merge_prob=0.0 \
    --attention_all


