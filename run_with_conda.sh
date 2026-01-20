#!/bin/bash
# Wrapper script to ensure conda environment is available

# Initialize conda for bash
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

# Activate flwr39 environment
conda activate flwr39 2>/dev/null || {
    echo "Error: Could not activate conda environment 'flwr39'"
    echo "Please ensure you're in the conda environment before running this script:"
    echo "  conda activate flwr39"
    exit 1
}

# Run the actual script with all arguments
exec "$(dirname "$0")/run_flower_experiment.sh" "$@"
