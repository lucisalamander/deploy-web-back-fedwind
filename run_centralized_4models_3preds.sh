#!/bin/bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_PATH="Long-term_Forecasting/flower_app/run_centralized.py"

# Default prediction lengths (can be overridden)
PRED_LENS=(1 72 432)

# Dataset parameters
DATASET_NAME=""
TARGET_COLUMN=""

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_cuda() {
    print_info "Checking CUDA availability..."

    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "Found CUDA $CUDA_VERSION with $GPU_COUNT GPU(s)"

        # Show GPU info with memory usage
        echo ""
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        echo ""
    else
        print_warning "CUDA not found - will use CPU (training will be slow)"
        SELECTED_GPU=""
    fi
}

select_gpu() {
    print_info "Selecting GPU..."

    if command -v nvidia-smi &> /dev/null; then
        # Find least busy GPU (lowest memory usage)
        SELECTED_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
                       sort -t',' -k2 -n | \
                       head -1 | \
                       cut -d',' -f1 | \
                       tr -d ' ')

        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $SELECTED_GPU)
        GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $SELECTED_GPU)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $SELECTED_GPU)

        print_success "Selected GPU $SELECTED_GPU: $GPU_NAME ($GPU_MEM_USED MB / $GPU_MEM_TOTAL MB used)"
        export CUDA_VISIBLE_DEVICES=$SELECTED_GPU
    else
        print_warning "No GPU available, will use CPU"
        SELECTED_GPU=""
    fi
}

usage() {
  cat <<'USAGE'
Usage:
  ./run_centralized_4models_3preds.sh <config_path> [OPTIONS]

Options:
  --dataset-name NAME       Dataset name (e.g., VNMET, custom, nasa_almaty)
  --target-column COL       Target column name for forecasting
  --pred-lens LENS...       Prediction lengths (space-separated, default: 1 72 432)
                           Example: --pred-lens 1 72 432 144

Example:
  ./run_centralized_4models_3preds.sh centralized_configs/centralized_gpt.conf
  ./run_centralized_4models_3preds.sh centralized_configs/centralized_gpt.conf --dataset-name VNMET --target-column "Vavg80 [m/s]"
  ./run_centralized_4models_3preds.sh centralized_configs/centralized_gpt.conf --pred-lens 1 72 432 144 720

This runs centralized experiments sequentially with specified pred_len values
for the given config.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG_PATH="$1"
shift || true

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --target-column)
      TARGET_COLUMN="$2"
      shift 2
      ;;
    --pred-lens)
      shift
      PRED_LENS=()
      while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
        PRED_LENS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  print_info "Using pre-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  check_cuda
  select_gpu
fi

echo "----------------------------------------------------------------"
echo "Centralized sequential run"
echo "Config: $CONFIG_PATH"
echo "Pred lens: ${PRED_LENS[*]}"
echo "----------------------------------------------------------------"

for pred_len in "${PRED_LENS[@]}"; do
  echo "  -> pred_len=$pred_len"
  python "$SCRIPT_PATH" --config "$CONFIG_PATH" --pred-len "$pred_len"
done

echo ""
echo "----------------------------------------------------------------"
echo "Run Completed Successfully"
echo "----------------------------------------------------------------"
