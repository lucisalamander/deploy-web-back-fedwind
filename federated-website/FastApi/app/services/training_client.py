"""
Training Client - Bridge to the external centralized training repository.
"""

import os
import sys
import time
import logging
import tempfile
import subprocess
import shutil

import numpy as np
import pandas as pd

from app.schemas import TrainingInput, TrainingOutput

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
TRAINING_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Long-term_Forecasting", "flower_app")
)
RUN_SCRIPT = os.path.join(TRAINING_REPO_ROOT, "run_centralized.py")
TRAINING_PYTHON = os.environ.get(
    "TRAINING_PYTHON",
    "/raid/tin_trungchau/conda_env/envs/flwr39/bin/python"
)

MODEL_NAME_MAP = {
    "GPT4TS": "gpt4ts_nonlinear",
    "LLAMA": "llama",
    "BERT": "bert",
    "BART": "bart",
}


def run_centralized_training(inp: TrainingInput) -> TrainingOutput:
    """
    Execute centralized training by calling run_centralized.py as a subprocess.
    """
    import subprocess

    start = time.time()

    logger.info("=" * 70)
    logger.info("CENTRALIZED TRAINING REQUEST RECEIVED")
    logger.info("=" * 70)
    logger.info(f"  model_name:        {inp.model_name}")
    logger.info(f"  prediction_length: {inp.prediction_length}")
    logger.info(f"  dropout_rate:      {inp.dropout_rate}")
    logger.info(f"  learning_rate:     {inp.learning_rate}")
    logger.info(f"  batch_size:        {inp.batch_size}")
    logger.info(f"  seq_len:           {inp.seq_len}")
    logger.info(f"  epochs:            {inp.epochs}")
    logger.info(f"  csv_path:          {getattr(inp, 'csv_path', 'N/A')}")
    logger.info("-" * 70)

    logger.info("[STEP 1] Verifying paths...")
    logger.info(f"  TRAINING_REPO_ROOT: {TRAINING_REPO_ROOT}")
    logger.info(f"  RUN_SCRIPT:         {RUN_SCRIPT}")
    logger.info(f"  TRAINING_PYTHON:    {TRAINING_PYTHON}")

    if not os.path.isdir(TRAINING_REPO_ROOT):
        msg = f"Training repo directory NOT FOUND: {TRAINING_REPO_ROOT}"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info(f"  ✓ Training repo directory exists")

    if not os.path.isfile(RUN_SCRIPT):
        msg = f"run_centralized.py NOT FOUND: {RUN_SCRIPT}"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info(f"  ✓ run_centralized.py exists")

    if not os.path.isfile(TRAINING_PYTHON) and not shutil.which(TRAINING_PYTHON):
        msg = f"Python interpreter NOT FOUND: {TRAINING_PYTHON}"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info(f"  ✓ Python interpreter exists")


    logger.info("[STEP 2] Checking if TRAINING_PYTHON can import torch...")
    try:
        check = subprocess.run(
            [TRAINING_PYTHON, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        if check.returncode == 0:
            logger.info(f"  ✓ torch {check.stdout.strip()} available")
        else:
            msg = (
                f"TRAINING_PYTHON cannot import torch!\n"
                f"  Python: {TRAINING_PYTHON}\n"
                f"  STDERR: {check.stderr.strip()}"
            )
            logger.error(msg)
            raise RuntimeError(msg)
    except subprocess.TimeoutExpired:
        logger.warning("  ⚠ torch check timed out — proceeding anyway")

    exp_base = os.environ.get("CENTRALIZED_WEB_DIR", "/raid/tin_trungchau/tmp")
    os.makedirs(exp_base, exist_ok=True)
    exp_dir = tempfile.mkdtemp(prefix="centralized_web_", dir=exp_base)
    logger.info(f"[STEP 3] Experiment directory created: {exp_dir}")

    internal_model = MODEL_NAME_MAP.get(inp.model_name, inp.model_name.lower())
    logger.info(f"[STEP 4] Model mapping: {inp.model_name} -> {internal_model}")

    cmd = [
        TRAINING_PYTHON,
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

    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = TRAINING_REPO_ROOT + (
        os.pathsep + existing_pypath if existing_pypath else ""
    )

    logger.info(f"[STEP 4] Full command:")
    logger.info(f"  {' '.join(cmd)}")
    logger.info(f"  cwd: {TRAINING_REPO_ROOT}")
    logger.info(f"  PYTHONPATH: {env['PYTHONPATH']}")


    logger.info("=" * 70)
    logger.info("[STEP 5] LAUNCHING run_centralized.py ...")
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
        logger.error("[STEP 5] ✗ TIMEOUT — training exceeded 6 hours")
        raise RuntimeError("Training timed out after 6 hours")
    except FileNotFoundError as e:
        logger.error(f"[STEP 5] ✗ FileNotFoundError: {e}")
        raise RuntimeError(f"Could not launch: {e}")

    logger.info(f"[STEP 6] Subprocess finished. Return code: {result.returncode}")

    if result.stdout:
        # Log last N lines of stdout
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
        logger.error(f"[STEP 6] ✗ TRAINING FAILED (exit code {result.returncode})")
        raise RuntimeError(
            f"run_centralized.py exited with code {result.returncode}.\n"
            f"STDERR: {result.stderr[-1000:] if result.stderr else 'empty'}"
        )

    logger.info("[STEP 6] ✓ Training completed successfully")

    logger.info(f"[STEP 7] Experiment directory contents:")
    for root, dirs, files in os.walk(exp_dir):
        level = root.replace(exp_dir, "").count(os.sep)
        indent = "  " * (level + 1)
        logger.info(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 2)
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            logger.info(f"{sub_indent}{f} ({size:,} bytes)")

    summary_path = os.path.join(exp_dir, "training_summary.csv")
    logger.info(f"[STEP 8] Looking for: {summary_path}")

    if not os.path.exists(summary_path):
        log_path = os.path.join(exp_dir, "training.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_tail = f.readlines()[-20:]
            logger.error(f"  training.log tail:\n{''.join(log_tail)}")
        raise RuntimeError(f"No training_summary.csv in {exp_dir}")

    df_summary = pd.read_csv(summary_path)
    logger.info(f"  ✓ Loaded training_summary.csv: {len(df_summary)} rows")
    logger.info(f"  Columns: {list(df_summary.columns)}")
    logger.info(f"  Last row:\n{df_summary.iloc[-1].to_string()}")

    if df_summary.empty:
        raise RuntimeError("training_summary.csv is empty")

    # Best round by lowest val_loss
    best_idx = df_summary["val_loss"].idxmin()
    best_row = df_summary.loc[best_idx]
    best_round = int(best_row["round"])

    mae = round(float(best_row["test_mae"]), 4)
    rmse = round(float(best_row["test_rmse"]), 4)

    logger.info(f"  Best round: {best_round}")
    logger.info(f"  val_loss:   {best_row['val_loss']:.6f}")
    logger.info(f"  test_mae:   {mae}")
    logger.info(f"  test_rmse:  {rmse}")

    pred_len = inp.prediction_length
    predictions: list[float] = []
    actuals: list[float] = []

    pred_dir = os.path.join(exp_dir, "predictions")
    test_csv = os.path.join(pred_dir, f"client0_round{best_round}_test.csv")
    logger.info(f"[STEP 9] Looking for predictions: {test_csv}")

    if os.path.exists(test_csv):
        df_pred = pd.read_csv(test_csv)
        logger.info(f"  ✓ Loaded {test_csv}: {len(df_pred)} rows")
        logger.info(f"  Columns: {list(df_pred.columns)}")

        if len(df_pred) > 0:
            last_sample = df_pred.iloc[-1]
            pred_cols = [f"pred_t{t}" for t in range(pred_len)]
            true_cols = [f"true_t{t}" for t in range(pred_len)]

            missing_pred = [c for c in pred_cols if c not in df_pred.columns]
            missing_true = [c for c in true_cols if c not in df_pred.columns]
            if missing_pred:
                logger.warning(f"  ⚠ Missing pred columns: {missing_pred}")
            if missing_true:
                logger.warning(f"  ⚠ Missing true columns: {missing_true}")

            predictions = [round(float(last_sample[c]), 4) for c in pred_cols if c in df_pred.columns]
            actuals = [round(float(last_sample[c]), 4) for c in true_cols if c in df_pred.columns]

            logger.info(f"  predictions (first 5): {predictions[:5]}")
            logger.info(f"  actuals     (first 5): {actuals[:5]}")
    else:
        if os.path.isdir(pred_dir):
            available = os.listdir(pred_dir)
            logger.warning(f"  ⚠ File not found. Available in predictions/: {available[:20]}")
        else:
            logger.warning(f"  ⚠ predictions/ directory does not exist")

    mape = None
    if actuals and predictions and len(actuals) == len(predictions):
        mape_vals = [abs((a - p) / a) for a, p in zip(actuals, predictions) if a != 0]
        if mape_vals:
            mape = round(100.0 * sum(mape_vals) / len(mape_vals), 2)

    elapsed = round(time.time() - start, 2)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE — RETURNING RESULTS")
    logger.info(f"  MAE:  {mae}")
    logger.info(f"  RMSE: {rmse}")
    logger.info(f"  MAPE: {mape}")
    logger.info(f"  Time: {elapsed}s")
    logger.info(f"  Predictions: {len(predictions)} values")
    logger.info(f"  Actuals:     {len(actuals)} values")
    logger.info("=" * 70)

    return TrainingOutput(
        mae=mae,
        rmse=rmse,
        mape=mape,
        training_time_seconds=elapsed,
        predictions=predictions,
        actuals=actuals,
    )
