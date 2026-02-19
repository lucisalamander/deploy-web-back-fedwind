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

import time
import math
import random
from app.schemas import TrainingInput, TrainingOutput


def run_centralized_training(inp: TrainingInput) -> TrainingOutput:
    """
    Execute centralized training and return metrics + predictions.

    ---------------------------------------------------------------
    TODO: Replace the mock body below with your real training call.
    ---------------------------------------------------------------

    Example real implementation:

        # Option A: installed package
        from centralized_training.run import train_centralized

        result = train_centralized(
            csv_path=inp.csv_path,
            model_name=inp.model_name,
            pred_len=inp.prediction_length,
            dropout=inp.dropout_rate,
            seq_len=inp.seq_len,
            batch_size=inp.batch_size,
            lr=inp.learning_rate,
            epochs=inp.epochs,
        )
        return TrainingOutput(**result)

    ---------------------------------------------------------------
    """

    # ---- MOCK IMPLEMENTATION (remove when integrating) ----
    start = time.time()

    # Simulate training delay proportional to epochs
    time.sleep(min(inp.epochs * 0.3, 3.0))

    # Generate plausible mock predictions
    pred_len = inp.prediction_length
    base_speed = 8.0 + random.uniform(-2, 2)
    predictions = []
    actuals = []
    for i in range(pred_len):
        t = i / max(pred_len - 1, 1) * 2 * math.pi
        actual = base_speed + 3 * math.sin(t) + random.gauss(0, 0.3)
        noise = random.gauss(0, 0.25)
        predicted = actual + noise
        actuals.append(round(actual, 2))
        predictions.append(round(predicted, 2))

    # Compute mock metrics
    errors = [abs(a - p) for a, p in zip(actuals, predictions)]
    mae = round(sum(errors) / len(errors), 4)
    rmse = round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 4)
    mape = round(
        100 * sum(abs((a - p) / a) for a, p in zip(actuals, predictions) if a != 0) / len(errors),
        2,
    )

    elapsed = round(time.time() - start, 2)

    return TrainingOutput(
        mae=mae,
        rmse=rmse,
        mape=mape,
        training_time_seconds=elapsed,
        predictions=predictions,
        actuals=actuals,
    )
