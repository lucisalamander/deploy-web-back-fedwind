#!/bin/bash

################################################################################
# Parallel Experiment Runner
#
# This script runs multiple Flower experiments in parallel across available GPUs
#
# Usage:
#   ./run_parallel_experiments.sh --config experiments.conf
#   ./run_parallel_experiments.sh --grid-search
#
# Examples:
#   # Run predefined experiments from config file
#   ./run_parallel_experiments.sh --config my_experiments.conf
#
#   # Quick grid search over learning rates
#   ./run_parallel_experiments.sh --grid-search --param lr 0.0001,0.0005,0.001
#
#   # Full grid search
#   ./run_parallel_experiments.sh --grid-search \
#       --param lr 0.0001,0.0005,0.001 \
#       --param pred-len 24,96,168
################################################################################

# Don't use set -e - we handle errors explicitly to allow experiments to fail gracefully

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_SCRIPT="$SCRIPT_DIR/run_flower_experiment.sh"

# Default settings
MAX_PARALLEL=3  # Max parallel experiments
CONFIG_FILE=""
GRID_SEARCH=false
declare -A GRID_PARAMS

print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --config FILE             Run experiments from config file
    --grid-search             Enable grid search mode
    --param NAME VALUES       Add parameter for grid search (comma-separated)
                             Example: --param lr 0.0001,0.0005,0.001
    --max-parallel NUM        Maximum parallel experiments (default: 16)
    -h, --help               Show this help message

Config File Format:
  Each line defines one experiment with space-separated parameters:
  --rounds 10 --lr 0.001 --seq-len 336 --pred-len 96
  --rounds 20 --lr 0.0005 --seq-len 512 --pred-len 120
  --mode quick

Examples:
    # Run experiments from config
    $0 --config experiments.conf

    # Grid search over learning rates
    $0 --grid-search --param lr 0.0001,0.0005,0.001

    # Grid search over multiple parameters
    $0 --grid-search \\
        --param lr 0.0001,0.0005,0.001 \\
        --param pred-len 24,96,168 \\
        --param llm-layers 2,4,6

    # Limit parallel experiments
    $0 --config experiments.conf --max-parallel 8

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --grid-search)
                GRID_SEARCH=true
                shift
                ;;
            --param)
                PARAM_NAME="$2"
                PARAM_VALUES="$3"
                GRID_PARAMS["$PARAM_NAME"]="$PARAM_VALUES"
                shift 3
                ;;
            --max-parallel)
                MAX_PARALLEL="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Get available GPUs (less than 50% memory used)
get_available_gpus() {
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F',' '{
        used = $2;
        total = $3;
        percent = (used / total) * 100;
        if (percent < 50) print $1
    }' | tr -d ' '
}

# Get available GPU from pool (excluding currently assigned ones)
get_next_available_gpu() {
    local assigned_gpus=("$@")
    local timeout=3600  # 1 hour
    local elapsed=0
    local check_interval=30

    while [ $elapsed -lt $timeout ]; do
        # Get all available GPUs from nvidia-smi
        local available_gpus=($(get_available_gpus))

        # Find first GPU not currently assigned
        for gpu in "${available_gpus[@]}"; do
            local is_assigned=false
            for assigned in "${assigned_gpus[@]}"; do
                if [ "$gpu" = "$assigned" ]; then
                    is_assigned=true
                    break
                fi
            done

            if [ "$is_assigned" = false ]; then
                echo "$gpu"
                return 0
            fi
        done

        # No unassigned GPU found
        if [ ${#assigned_gpus[@]} -gt 0 ]; then
            print_info "All available GPUs currently assigned (${assigned_gpus[*]}), waiting ${check_interval}s..."
        else
            print_info "No available GPUs, waiting ${check_interval}s..."
        fi
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    print_error "Timeout waiting for available GPU"
    return 1
}

# Generate experiments from grid search
generate_grid_experiments() {
    local param_names=()
    local param_values=()

    # Parse parameters
    for param_name in "${!GRID_PARAMS[@]}"; do
        param_names+=("$param_name")
        IFS=',' read -ra values <<< "${GRID_PARAMS[$param_name]}"
        param_values+=("$(printf '%s\n' "${values[@]}")")
    done

    # Generate all combinations
    local experiments=()

    if [ ${#param_names[@]} -eq 0 ]; then
        print_error "No parameters specified for grid search"
        return 1
    fi

    # Simple case: 1 parameter
    if [ ${#param_names[@]} -eq 1 ]; then
        IFS=',' read -ra values <<< "${GRID_PARAMS[${param_names[0]}]}"
        for value in "${values[@]}"; do
            echo "--${param_names[0]} $value"
        done
        return 0
    fi

    # Multiple parameters: generate Cartesian product
    local temp_file=$(mktemp)

    # Start with first parameter
    IFS=',' read -ra values <<< "${GRID_PARAMS[${param_names[0]}]}"
    for value in "${values[@]}"; do
        echo "--${param_names[0]} $value" >> "$temp_file"
    done

    # Add remaining parameters
    for ((i=1; i<${#param_names[@]}; i++)); do
        local param="${param_names[$i]}"
        IFS=',' read -ra values <<< "${GRID_PARAMS[$param]}"

        local temp_file2=$(mktemp)
        while IFS= read -r line; do
            for value in "${values[@]}"; do
                echo "$line --$param $value" >> "$temp_file2"
            done
        done < "$temp_file"

        mv "$temp_file2" "$temp_file"
    done

    cat "$temp_file"
    rm "$temp_file"
}

# Run single experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local exp_args=$2
    local exp_num=$3

    print_info "[Experiment $exp_num] Starting on GPU $gpu_id with args: $exp_args"

    # Run experiment with GPU selection (--yes for non-interactive mode)
    CUDA_VISIBLE_DEVICES=$gpu_id $RUNNER_SCRIPT --yes $exp_args 2>&1 | \
        sed "s/^/[Exp$exp_num GPU$gpu_id] /"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        print_success "[Experiment $exp_num] Completed successfully on GPU $gpu_id"
    else
        print_error "[Experiment $exp_num] Failed on GPU $gpu_id (exit code: $exit_code)"
    fi

    return $exit_code
}

# Main execution
main() {
    parse_args "$@"

    print_header "Parallel Experiment Runner"

    # Check for GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. GPU required for parallel experiments."
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_info "Found $GPU_COUNT GPUs"

    # Adjust max parallel if needed
    if [ $MAX_PARALLEL -gt $GPU_COUNT ]; then
        print_warning "Reducing max-parallel from $MAX_PARALLEL to $GPU_COUNT (available GPUs)"
        MAX_PARALLEL=$GPU_COUNT
    fi

    # Generate experiment list
    declare -a experiments

    if [ "$GRID_SEARCH" = true ]; then
        print_info "Generating grid search experiments..."

        if [ ${#GRID_PARAMS[@]} -eq 0 ]; then
            print_error "No parameters specified for grid search. Use --param NAME VALUES"
            exit 1
        fi

        mapfile -t experiments < <(generate_grid_experiments)

    elif [ -n "$CONFIG_FILE" ]; then
        if [ ! -f "$CONFIG_FILE" ]; then
            print_error "Config file not found: $CONFIG_FILE"
            exit 1
        fi

        print_info "Loading experiments from: $CONFIG_FILE"
        mapfile -t experiments < <(grep -v '^#' "$CONFIG_FILE" | grep -v '^$')
    else
        print_error "Either --config or --grid-search must be specified"
        usage
        exit 1
    fi

    TOTAL_EXPERIMENTS=${#experiments[@]}
    print_success "Generated $TOTAL_EXPERIMENTS experiments"

    echo ""
    print_header "Experiment List"
    for i in "${!experiments[@]}"; do
        echo "  $((i+1)). ${experiments[$i]}"
    done
    echo ""

    read -p "Start experiments? (y/n): " confirm
    if [[ $confirm != "y" ]]; then
        print_warning "Aborted by user"
        exit 0
    fi

    # Track running experiments
    declare -A running_pids      # Maps PID -> experiment number
    declare -A running_gpus      # Maps PID -> GPU ID
    declare -A gpu_available     # Maps GPU ID -> 1 if available, 0 if in use
    completed=0
    failed=0

    # Initialize all GPUs as available
    available_gpus=($(get_available_gpus))
    if [ ${#available_gpus[@]} -eq 0 ]; then
        print_error "No GPUs available (all are >50% memory used)"
        exit 1
    fi

    for gpu in "${available_gpus[@]}"; do
        gpu_available[$gpu]=1
    done

    print_info "Available GPUs: ${available_gpus[*]}"

    print_header "Running Experiments"

    # Process all experiments sequentially, running MAX_PARALLEL at a time
    exp_index=0
    while [ $exp_index -lt $TOTAL_EXPERIMENTS ]; do
        # Clean up completed experiments and free their GPUs
        for pid in "${!running_pids[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                wait $pid
                exit_code=$?

                exp_num=${running_pids[$pid]}
                gpu_id=${running_gpus[$pid]}

                if [ $exit_code -eq 0 ]; then
                    ((completed++))
                    print_success "[Experiment $exp_num] Completed on GPU $gpu_id"
                else
                    ((failed++))
                    print_error "[Experiment $exp_num] Failed on GPU $gpu_id (exit code: $exit_code)"
                fi

                # Free the GPU for reuse
                gpu_available[$gpu_id]=1

                unset running_pids[$pid]
                unset running_gpus[$pid]
            fi
        done

        # If at max parallel, wait before starting new experiment
        if [ ${#running_pids[@]} -ge $MAX_PARALLEL ]; then
            sleep 2
            continue
        fi

        # Find an available GPU
        gpu_id=""
        for gpu in "${!gpu_available[@]}"; do
            if [ "${gpu_available[$gpu]}" -eq 1 ]; then
                gpu_id=$gpu
                break
            fi
        done

        # If no GPU available, wait and retry
        if [ -z "$gpu_id" ]; then
            sleep 2
            # Refresh available GPUs in case new ones became available
            new_available_gpus=($(get_available_gpus))
            for gpu in "${new_available_gpus[@]}"; do
                if [ -z "${gpu_available[$gpu]}" ]; then
                    gpu_available[$gpu]=1
                fi
            done
            continue
        fi

        # Start next experiment
        exp_num=$((exp_index + 1))
        exp_args="${experiments[$exp_index]}"

        # Mark GPU as in use
        gpu_available[$gpu_id]=0

        # Start experiment in background
        run_experiment $gpu_id "$exp_args" $exp_num &
        pid=$!

        running_pids[$pid]=$exp_num
        running_gpus[$pid]=$gpu_id

        print_info "Started experiment $exp_num/$TOTAL_EXPERIMENTS on GPU $gpu_id (Running: ${#running_pids[@]}, Completed: $completed, Failed: $failed)"

        # Move to next experiment
        ((exp_index++))

        # Brief delay to allow process to start properly
        sleep 1
    done

    # Wait for all remaining experiments to complete
    print_info "All experiments started. Waiting for remaining ${#running_pids[@]} to complete..."

    while [ ${#running_pids[@]} -gt 0 ]; do
        for pid in "${!running_pids[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                wait $pid
                exit_code=$?

                exp_num=${running_pids[$pid]}
                gpu_id=${running_gpus[$pid]}

                if [ $exit_code -eq 0 ]; then
                    ((completed++))
                    print_success "[Experiment $exp_num] Completed on GPU $gpu_id"
                else
                    ((failed++))
                    print_error "[Experiment $exp_num] Failed on GPU $gpu_id (exit code: $exit_code)"
                fi

                unset running_pids[$pid]
                unset running_gpus[$pid]
            fi
        done

        if [ ${#running_pids[@]} -gt 0 ]; then
            sleep 2
        fi
    done

    # Final summary
    echo ""
    print_header "Experiment Summary"
    echo -e "  ${CYAN}Total:${NC}     $TOTAL_EXPERIMENTS"
    echo -e "  ${GREEN}Completed:${NC} $completed"
    echo -e "  ${RED}Failed:${NC}    $failed"
    echo ""

    if [ $failed -eq 0 ]; then
        print_success "All experiments completed successfully!"
        exit 0
    else
        print_warning "$failed experiment(s) failed"
        exit 1
    fi
}

main "$@"
