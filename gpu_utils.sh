#!/bin/bash

################################################################################
# GPU Utility Functions
# Helper functions for GPU selection and management
################################################################################

# Find the least busy GPU based on memory usage
find_least_busy_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    sort -t',' -k2 -n | \
    head -1 | \
    cut -d',' -f1 | \
    tr -d ' '
}

# Get list of available GPUs (less than 80% memory used)
get_available_gpus() {
    local threshold=${1:-80}  # Default 80% threshold
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F',' -v thresh="$threshold" '{
        used = $2;
        total = $3;
        percent = (used / total) * 100;
        if (percent < thresh) print $1
    }' | tr -d ' '
}

# Display GPU status
show_gpu_status() {
    echo "GPU Status:"
    echo "==========="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=table
}

# Wait for a GPU to become available
wait_for_gpu() {
    local threshold=${1:-80}
    local timeout=${2:-3600}  # Default 1 hour timeout
    local elapsed=0
    local check_interval=30

    while [ $elapsed -lt $timeout ]; do
        local available=$(get_available_gpus $threshold | head -1)
        if [ -n "$available" ]; then
            echo "$available"
            return 0
        fi
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    echo "Timeout waiting for available GPU" >&2
    return 1
}

# Export functions
export -f find_least_busy_gpu
export -f get_available_gpus
export -f show_gpu_status
export -f wait_for_gpu
