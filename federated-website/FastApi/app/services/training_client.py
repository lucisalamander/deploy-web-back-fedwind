"""
Training Client - Bridge to the external centralized training repository.

==========================================================================
INTEGRATION GUIDE
==========================================================================

This module is a STUB.  It defines the interface that the service layer
calls.  You must replace the body of `run_centralized_training()` with
actual calls to your existing training repo.

Your training repo (the one with Dataset_Custom, GPT4TS, etc.) should be
importable as a Python package.  There are two ways to wire it in:

OPTION A  -  pip install (recommended)
------------------------------------------------------------------
  1. In your training repo, add a minimal setup.py / pyproject.toml
  2. pip install -e /path/to/centralized-training-repo
  3. Then import directly:
       from centralized_training.run import train_centralized

OPTION B  -  sys.path hack (quick & dirty for development)
------------------------------------------------------------------
  import sys
  sys.path.insert(0, "/absolute/path/to/centralized-training-repo")
  from run_centralized import train_centralized

==========================================================================
EXPECTED FUNCTION SIGNATURE IN YOUR TRAINING REPO
==========================================================================

  def train_centralized(
      csv_path: str,
      model_name: str,        # "GPT4TS" | "LLAMA" | "BERT" | "BART"
      pred_len: int,          # prediction horizon  (e.g. 6, 36, 144)
      dropout: float,         # 0.0 - 0.5
      seq_len: int = 336,     # input window length
      batch_size: int = 32,
      lr: float = 0.0001,
      epochs: int = 10,
  ) -> dict:
      '''
      Returns a dict like:
      {
          "mae": 0.6757,
          "rmse": 0.8863,
          "mape": 5.8,              # optional
          "training_time_seconds": 42.5,
          "predictions": [8.1, 8.3, ...],   # length = pred_len
          "actuals":     [8.2, 8.5, ...],   # optional, same length
      }
      '''

==========================================================================
"""

import os
import sys
import time
import logging
import tempfile

import numpy as np
import pandas as pd

from app.schemas import TrainingInput, TrainingOutput

logger = logging.getLogger(__name__)

TRAINING_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Long-term_Forecasting")
)
RUN_SCRIPT = os.path.join(TRAINING_REPO_ROOT, "run_centralized.py")

# Frontend enum values -> run_centralized.py --model values
MODEL_NAME_MAP = {
    "GPT4TS": "gpt4ts_nonlinear",
    "LLAMA": "llama",
    "BERT": "bert",
    "BART": "bart",
}


def run_centralized_training(inp: TrainingInput) -> TrainingOutput:
    """
    Execute centralized training by calling run_centralized.py as a subprocess.

    Steps:
      1. Build CLI arguments from TrainingInput
      2. Launch run_centralized.py in a subprocess
      3. Parse training_summary.csv for metrics
      4. Find the BEST round (lowest val_loss) and parse its prediction CSV
      5. Return TrainingOutput for the service layer
    """
    import subprocess

    start = time.time()

    
    # 0. Verify training repo exists
    
    if not os.path.isfile(RUN_SCRIPT):
        raise RuntimeError(
            f"Training script not found at {RUN_SCRIPT}. "
            f"Expected repo at {TRAINING_REPO_ROOT}"
        )

    
    # 1. Create a temp experiment directory for this run
    
    exp_dir = tempfile.mkdtemp(prefix="centralized_web_")
    logger.info(f"Experiment directory: {exp_dir}")

    
    # 2. Map frontend model name to internal model name
    
    internal_model = MODEL_NAME_MAP.get(inp.model_name, inp.model_name.lower())

    
    # 3. Build subprocess command
    #
    #    run_centralized.py CLI args (from build_parser()):
    #      --exp-dir, --rounds, --pred-len, --lr, --batch-size,
    #      --seq-len, --dropout, --model, --dataset-name, etc.
    
    cmd = [
        sys.executable,
        RUN_SCRIPT,
        "--exp-dir", exp_dir,
        "--rounds", str(inp.epochs),
        "--pred-len", str(inp.prediction_length),
        "--lr", str(inp.learning_rate),
        "--batch-size", str(inp.batch_size),
        "--seq-len", str(inp.seq_len),
        "--dropout", str(inp.dropout_rate),
        "--model", internal_model,
    ]

    logger.info(f"Launching training: {' '.join(cmd)}")

    
    # 4. Run the training process
    
    try:
        result = subprocess.run(
            cmd,
            cwd=TRAINING_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600 * 6,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Training timed out after 6 hours")
    except FileNotFoundError:
        raise RuntimeError(
            f"Python interpreter or training script not found. "
            f"Command: {' '.join(cmd)}"
        )

    if result.stdout:
        logger.info(f"Training STDOUT (last 2000 chars):\n{result.stdout[-2000:]}")
    if result.stderr:
        logger.warning(f"Training STDERR (last 2000 chars):\n{result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(
            f"run_centralized.py exited with code {result.returncode}.\n"
            f"STDERR: {result.stderr[-1000:] if result.stderr else 'empty'}"
        )

    
    # 5. Parse training_summary.csv
    #
    #    Columns written by run_centralized.py (line ~340):
    #      round, train_loss, val_loss, val_mae, val_rmse,
    #      test_loss, test_mae, test_rmse, best_loss,
    #      round_duration_sec, train_duration_sec,
    #      val_duration_sec, test_duration_sec
    
    summary_path = os.path.join(exp_dir, "training_summary.csv")
    if not os.path.exists(summary_path):
        raise RuntimeError(
            f"Training completed but no training_summary.csv found in {exp_dir}. "
            f"Check {os.path.join(exp_dir, 'training.log')}"
        )

    df_summary = pd.read_csv(summary_path)
    if df_summary.empty:
        raise RuntimeError("training_summary.csv is empty")

    
    # 5a. Find the BEST round (lowest val_loss)
    #
    #     run_centralized.py restores the best checkpoint before saving
    #     final_model.pt, so the best round's predictions are the ones
    #     that match the final model.
    
    best_idx = df_summary["val_loss"].idxmin()
    best_row = df_summary.loc[best_idx]
    best_round = int(best_row["round"])

    # Use the best round's test metrics (these match the restored model)
    mae = round(float(best_row["test_mae"]), 4)
    rmse = round(float(best_row["test_rmse"]), 4)

    logger.info(
        f"Training finished: {len(df_summary)} rounds completed. "
        f"Best round: {best_round} (val_loss={best_row['val_loss']:.6f}, "
        f"test_mae={mae}, test_rmse={rmse})"
    )

    
    # 6. Parse prediction CSVs for forecast chart data
    #
    #    run_centralized.py saves via _save_predictions_to_csv():
    #      {exp_dir}/predictions/client{city_id}_round{round_num}_{split}.csv
    #
    #    Columns:
    #      sample_idx, client_id, round, split,
    #      pred_t0, pred_t1, ..., pred_t{pred_len-1},
    #      true_t0, true_t1, ..., true_t{pred_len-1}
    #
    #    We use city 0's test predictions from the BEST round.
    
    pred_len = inp.prediction_length
    predictions: list[float] = []
    actuals: list[float] = []

    pred_dir = os.path.join(exp_dir, "predictions")
    test_csv = os.path.join(
        pred_dir, f"client0_round{best_round}_test.csv"
    )

    if os.path.exists(test_csv):
        df_pred = pd.read_csv(test_csv)
        if len(df_pred) > 0:
            # Use the last sample as the representative forecast
            last_sample = df_pred.iloc[-1]

            # Column names: pred_t0, pred_t1, ..., true_t0, true_t1, ...
            pred_cols = [f"pred_t{t}" for t in range(pred_len)]
            true_cols = [f"true_t{t}" for t in range(pred_len)]

            # Verify columns exist before reading
            missing_pred = [c for c in pred_cols if c not in df_pred.columns]
            missing_true = [c for c in true_cols if c not in df_pred.columns]

            if missing_pred:
                logger.warning(
                    f"Missing prediction columns: {missing_pred}. "
                    f"Available: {list(df_pred.columns)}"
                )
            if missing_true:
                logger.warning(
                    f"Missing true columns: {missing_true}. "
                    f"Available: {list(df_pred.columns)}"
                )

            predictions = [
                round(float(last_sample[c]), 4)
                for c in pred_cols
                if c in df_pred.columns
            ]
            actuals = [
                round(float(last_sample[c]), 4)
                for c in true_cols
                if c in df_pred.columns
            ]

            logger.info(
                f"Parsed {len(df_pred)} samples from {test_csv}. "
                f"Returning last sample: {len(predictions)} predictions, "
                f"{len(actuals)} actuals"
            )
    else:
        # List what IS in predictions dir for debugging
        if os.path.isdir(pred_dir):
            available = os.listdir(pred_dir)
            logger.warning(
                f"Expected {test_csv} not found. "
                f"Available files: {available[:20]}"
            )
        else:
            logger.warning(f"Predictions directory not found: {pred_dir}")

    
    # 7. Compute MAPE from predictions

    mape = None
    if actuals and predictions and len(actuals) == len(predictions):
        mape_vals = [
            abs((a - p) / a)
            for a, p in zip(actuals, predictions)
            if a != 0
        ]
        if mape_vals:
            mape = round(100.0 * sum(mape_vals) / len(mape_vals), 2)

    elapsed = round(time.time() - start, 2)

    return TrainingOutput(
        mae=mae,
        rmse=rmse,
        mape=mape,
        training_time_seconds=elapsed,
        predictions=predictions,
        actuals=actuals,
    )