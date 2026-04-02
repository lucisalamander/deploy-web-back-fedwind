"""
Federated Training Client - Bridge to the external federated training repository.

==========================================================================
INTEGRATION GUIDE
==========================================================================

This module mirrors the architecture of training_client.py but targets the
federated training pipeline (Flower-based) instead of centralized training.

The external federated repo is expected to contain a `run_federated.py`
script that orchestrates a Flower simulation with the specified number
of clients, aggregation algorithm, and model configuration.

OPTION A  -  subprocess (recommended, matches training_client.py pattern)
------------------------------------------------------------------
  Call run_federated.py as a subprocess, passing all parameters via CLI
  arguments.  The script writes results to an experiment directory.

OPTION B  -  direct import
------------------------------------------------------------------
  import sys
  sys.path.insert(0, "/path/to/federated-training-repo")
  from run_federated import train_federated

==========================================================================
EXPECTED CLI INTERFACE (run_federated.py)
==========================================================================

  python run_federated.py \\
      --exp-dir /tmp/federated_web_xxx \\
      --rounds 5 \\
      --num-clients 5 \\
      --pred-len 6 \\
      --lr 0.0001 \\
      --batch-size 32 \\
      --seq-len 336 \\
      --dropout 0.2 \\
      --model gpt4ts_nonlinear \\
      --algorithm FedAvg

  Outputs in exp-dir:
      training_summary.csv   (round, train_loss, val_loss, test_mae, test_rmse)
      predictions/           (client0_round{N}_test.csv  for best round)

==========================================================================
"""

import os
import sys
import time
import logging
import tempfile
import subprocess
import shutil
import math
import random

import numpy as np
import pandas as pd

from app.schemas import FederatedTrainingInput, FederatedTrainingOutput

logger = logging.getLogger(__name__)


# The Flower app is located in: federated-cme/Long-term_Forecasting/flower_app/my_flower_app
TRAINING_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Long-term_Forecasting", "flower_app")
)
FLOWER_APP_DIR = os.path.join(TRAINING_REPO_ROOT, "my_flower_app")
RUN_SCRIPT = os.path.join(TRAINING_REPO_ROOT, "run_federated.py")
TRAINING_PYTHON = os.environ.get(
    "TRAINING_PYTHON",
    "/home/tin_trungchau/miniconda3/envs/flwr39/bin/python"
)

# Model name mapping (same as centralized)
MODEL_NAME_MAP = {
    "GPT4TS": "gpt4ts_nonlinear",
    "LLAMA": "llama",
    "BERT": "bert",
    "BART": "bart",
}

# Algorithm mapping (CLI argument values)
ALGORITHM_MAP = {
    "FedAvg": "fedavg",
    "FedProx": "fedprox",
    "FedBN": "fedbn",
    "FedPer": "fedper",
    "SCAFFOLD": "scaffold",
}


def run_federated_training(inp: FederatedTrainingInput) -> FederatedTrainingOutput:
    """
    Execute federated training by calling run_federated.py as a subprocess.

    This follows the exact same pattern as training_client.run_centralized_training()
    but passes federated-specific parameters (algorithm, num_clients, rounds).

    ---------------------------------------------------------------
    TODO: Replace the mock body below with your real training call
    once the external federated repo is available on the machine.
    ---------------------------------------------------------------
    """

    start = time.time()

    logger.info("=" * 70)
    logger.info("FEDERATED TRAINING REQUEST RECEIVED")
    logger.info("=" * 70)
    logger.info(f"  model_name:            {inp.model_name}")
    logger.info(f"  prediction_length:     {inp.prediction_length}")
    logger.info(f"  dropout_rate:          {inp.dropout_rate}")
    logger.info(f"  federated_algorithm:   {inp.federated_algorithm}")
    logger.info(f"  num_clients:           {inp.num_clients}")
    logger.info(f"  rounds:                {inp.rounds}")
    logger.info(f"  learning_rate:         {inp.learning_rate}")
    logger.info(f"  batch_size:            {inp.batch_size}")
    logger.info(f"  seq_len:               {inp.seq_len}")
    logger.info(f"  csv_path:              {inp.csv_path}")
    logger.info("-" * 70)

    # ------------------------------------------------------------------
    # Check if the external repo + script exist.
    # If they do, run the real subprocess; otherwise fall back to mock.
    # ------------------------------------------------------------------
    # Check for either run_federated.py or the Flower app directory
    use_real = (
        os.path.isdir(TRAINING_REPO_ROOT)
        and (os.path.isfile(RUN_SCRIPT) or os.path.isdir(FLOWER_APP_DIR))
        and (os.path.isfile(TRAINING_PYTHON) or shutil.which(TRAINING_PYTHON))
    )

    if use_real:
        return _run_real_federated(inp, start)
    else:
        logger.warning("External federated repo not found — using mock implementation")
        return _run_mock_federated(inp, start)


# ---------------------------------------------------------------------------
# Real implementation (mirrors training_client.py structure)
# ---------------------------------------------------------------------------

def _run_real_federated(inp: FederatedTrainingInput, start: float) -> FederatedTrainingOutput:
    """Run federated training via the external repo subprocess."""

    # 1. Verify Python interpreter has torch
    logger.info("[STEP 1] Checking TRAINING_PYTHON can import torch + flwr...")
    try:
        check = subprocess.run(
            [TRAINING_PYTHON, "-c", "import torch; import flwr; print(torch.__version__, flwr.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        if check.returncode == 0:
            logger.info(f"  torch + flwr available: {check.stdout.strip()}")
        else:
            logger.warning(f"  Import check failed: {check.stderr.strip()}")
    except subprocess.TimeoutExpired:
        logger.warning("  Import check timed out — proceeding anyway")

    # 2. Create experiment directory
    exp_dir = tempfile.mkdtemp(prefix="federated_web_")
    logger.info(f"[STEP 2] Experiment directory: {exp_dir}")

    # 3. Map model + algorithm names
    internal_model = MODEL_NAME_MAP.get(inp.model_name, inp.model_name.lower())
    internal_algo = ALGORITHM_MAP.get(inp.federated_algorithm, inp.federated_algorithm.lower())
    logger.info(f"[STEP 3] Model: {inp.model_name} -> {internal_model}")
    logger.info(f"         Algorithm: {inp.federated_algorithm} -> {internal_algo}")

    # 4. Build command
    cmd = [
        TRAINING_PYTHON,
        RUN_SCRIPT,
        "--exp-dir", exp_dir,
        "--rounds", str(inp.rounds),
        "--num-clients", str(inp.num_clients),
        "--pred-len", str(inp.prediction_length),
        "--lr", str(inp.learning_rate),
        "--batch-size", str(inp.batch_size),
        "--seq-len", str(inp.seq_len),
        "--dropout", str(inp.dropout_rate),
        "--model", internal_model,
        "--algorithm", internal_algo,
    ]

    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = TRAINING_REPO_ROOT + (
        os.pathsep + existing_pypath if existing_pypath else ""
    )

    logger.info(f"[STEP 4] Command: {' '.join(cmd)}")
    logger.info(f"  cwd: {TRAINING_REPO_ROOT}")

    # 5. Launch subprocess
    logger.info("=" * 70)
    logger.info("[STEP 5] LAUNCHING run_federated.py ...")
    logger.info("=" * 70)

    try:
        result = subprocess.run(
            cmd,
            cwd=TRAINING_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600 * 6,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("[STEP 5] TIMEOUT — federated training exceeded 6 hours")
        raise RuntimeError("Federated training timed out after 6 hours")
    except FileNotFoundError as e:
        logger.error(f"[STEP 5] FileNotFoundError: {e}")
        raise RuntimeError(f"Could not launch federated training: {e}")

    # 6. Process output
    logger.info(f"[STEP 6] Return code: {result.returncode}")

    if result.stdout:
        stdout_lines = result.stdout.strip().split("\n")
        logger.info(f"  STDOUT ({len(stdout_lines)} lines, showing last 30):")
        for line in stdout_lines[-30:]:
            logger.info(f"    | {line}")

    if result.stderr:
        stderr_lines = result.stderr.strip().split("\n")
        logger.warning(f"  STDERR ({len(stderr_lines)} lines, showing last 20):")
        for line in stderr_lines[-20:]:
            logger.warning(f"    | {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"run_federated.py exited with code {result.returncode}.\n"
            f"STDERR: {result.stderr[-1000:] if result.stderr else 'empty'}"
        )

    logger.info("[STEP 6] Federated training completed successfully")

    # 7. Parse training_summary.csv (same format as centralized)
    summary_path = os.path.join(exp_dir, "training_summary.csv")
    logger.info(f"[STEP 7] Looking for: {summary_path}")

    if not os.path.exists(summary_path):
        log_path = os.path.join(exp_dir, "training.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_tail = f.readlines()[-20:]
            logger.error(f"  training.log tail:\n{''.join(log_tail)}")
        raise RuntimeError(f"No training_summary.csv in {exp_dir}")

    df_summary = pd.read_csv(summary_path)
    logger.info(f"  Loaded training_summary.csv: {len(df_summary)} rows")

    if df_summary.empty:
        raise RuntimeError("training_summary.csv is empty")

    # Best round by lowest val_loss
    best_idx = df_summary["val_loss"].idxmin()
    best_row = df_summary.loc[best_idx]
    best_round = int(best_row["round"])

    mae = round(float(best_row["test_mae"]), 4)
    rmse = round(float(best_row["test_rmse"]), 4)

    logger.info(f"  Best round: {best_round}")
    logger.info(f"  MAE:  {mae}")
    logger.info(f"  RMSE: {rmse}")

    # 8. Parse predictions
    pred_len = inp.prediction_length
    predictions: list[float] = []
    actuals: list[float] = []

    pred_dir = os.path.join(exp_dir, "predictions")
    test_csv = os.path.join(pred_dir, f"client0_round{best_round}_test.csv")
    logger.info(f"[STEP 8] Looking for predictions: {test_csv}")

    if os.path.exists(test_csv):
        df_pred = pd.read_csv(test_csv)
        if len(df_pred) > 0:
            last_sample = df_pred.iloc[-1]
            pred_cols = [f"pred_t{t}" for t in range(pred_len)]
            true_cols = [f"true_t{t}" for t in range(pred_len)]
            predictions = [round(float(last_sample[c]), 4) for c in pred_cols if c in df_pred.columns]
            actuals = [round(float(last_sample[c]), 4) for c in true_cols if c in df_pred.columns]
    else:
        logger.warning(f"  Predictions file not found: {test_csv}")

    # 9. Compute MAPE
    mape = None
    if actuals and predictions and len(actuals) == len(predictions):
        mape_vals = [abs((a - p) / a) for a, p in zip(actuals, predictions) if a != 0]
        if mape_vals:
            mape = round(100.0 * sum(mape_vals) / len(mape_vals), 2)

    elapsed = round(time.time() - start, 2)

    logger.info("=" * 70)
    logger.info("FEDERATED TRAINING COMPLETE")
    logger.info(f"  MAE:  {mae}  |  RMSE: {rmse}  |  MAPE: {mape}")
    logger.info(f"  Best round: {best_round}  |  Time: {elapsed}s")
    logger.info("=" * 70)

    return FederatedTrainingOutput(
        mae=mae,
        rmse=rmse,
        mape=mape,
        training_time_seconds=elapsed,
        predictions=predictions,
        actuals=actuals,
        best_round=best_round,
        num_clients=inp.num_clients,
        federated_algorithm=inp.federated_algorithm,
    )


# ---------------------------------------------------------------------------
# Mock implementation (used when external repo is not available)
# ---------------------------------------------------------------------------

def _run_mock_federated(inp: FederatedTrainingInput, start: float) -> FederatedTrainingOutput:
    """
    Generate plausible mock results for federated training.
    Remove this once the external federated repo is wired in.
    """
    import time

    # Simulate training delay: rounds * clients * small factor
    delay = min(inp.rounds * inp.num_clients * 0.15, 5.0)
    time.sleep(delay)

    # Generate plausible predictions
    pred_len = inp.prediction_length
    base_speed = 8.0 + random.uniform(-2, 2)
    predictions = []
    actuals = []

    for i in range(pred_len):
        t = i / max(pred_len - 1, 1) * 2 * math.pi
        actual = base_speed + 3 * math.sin(t) + random.gauss(0, 0.3)
        # Federated models have slightly more noise than centralized
        noise = random.gauss(0, 0.35)
        predicted = actual + noise
        actuals.append(round(actual, 2))
        predictions.append(round(predicted, 2))

    # Compute mock metrics (federated typically slightly worse than centralized)
    errors = [abs(a - p) for a, p in zip(actuals, predictions)]
    mae = round(sum(errors) / len(errors), 4)
    rmse = round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 4)
    mape = round(
        100 * sum(abs((a - p) / a) for a, p in zip(actuals, predictions) if a != 0) / len(errors),
        2,
    )

    best_round = random.randint(2, min(3, inp.rounds))
    elapsed = round(time.time() - start, 2)

    return FederatedTrainingOutput(
        mae=mae,
        rmse=rmse,
        mape=mape,
        training_time_seconds=elapsed,
        predictions=predictions,
        actuals=actuals,
        best_round=best_round,
        num_clients=inp.num_clients,
        federated_algorithm=inp.federated_algorithm,
    )
