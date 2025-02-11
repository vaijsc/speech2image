#!/bin/bash

# Ensure the user provides the GPU_ID as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

GPU_ID=$1  # GPU_ID is taken from the first command-line argument

for JOB_ID in {1..8}; do
    echo "Running inference for GPU_ID=${GPU_ID}, JOB_ID=${JOB_ID}"
    python infer.py --gpu_id ${GPU_ID} --job_id ${JOB_ID} > logs/gpu${GPU_ID}-job${JOB_ID}.log 2>&1 &
done
