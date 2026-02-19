"""
Training Service - Orchestrates the centralized training flow.

Responsibilities:
  1. Validate the uploaded CSV (check columns, data quality)
  2. Build a TrainingInput from the user config + saved file path
  3. Call the training client (bridge to external repo)
  4. Convert TrainingOutput into a TrainingResult for the frontend
"""

import os
import csv
from datetime import datetime
from typing import List

from app.schemas import (
    TrainingConfig,
    TrainingInput,
    TrainingResult,
    TrainingMetrics,
    ForecastPoint,
    VALID_PREDICTION_LENGTHS,
)
from app.services.training_client import run_centralized_training


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")

# Expected CSV columns from NASA POWER hourly data
REQUIRED_COLUMNS = {"YEAR", "MO", "DY", "HR", "WS10M"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_csv(file_path: str) -> dict:
    """
    Validate that the CSV has the expected NASA POWER format.
    Returns {"valid": True, "rows": N, "columns": [...]} or raises ValueError.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV file is empty")

        columns = [c.strip() for c in header]
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}. "
                f"Expected: {', '.join(sorted(REQUIRED_COLUMNS))}. "
                f"Found: {', '.join(columns)}"
            )

        row_count = sum(1 for _ in reader)

    if row_count < 10:
        raise ValueError(
            f"CSV has only {row_count} data rows. Need at least 10 for training."
        )

    return {"valid": True, "rows": row_count, "columns": columns}


def validate_config(config: TrainingConfig) -> None:
    """Raise ValueError if config values are out of range."""
    if config.prediction_length not in VALID_PREDICTION_LENGTHS:
        raise ValueError(
            f"Invalid prediction_length={config.prediction_length}. "
            f"Allowed: {VALID_PREDICTION_LENGTHS}"
        )
    if not (0.0 <= config.dropout_rate <= 0.5):
        raise ValueError("dropout_rate must be between 0.0 and 0.5")


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

def start_training(filename: str, config: TrainingConfig) -> TrainingResult:
    """
    Full centralized training pipeline:
      1. Resolve file path
      2. Validate CSV + config
      3. Build TrainingInput
      4. Call training client
      5. Return TrainingResult

    Raises ValueError for validation errors (caught by router as 422).
    Raises RuntimeError for training failures (caught by router as 500).
    """

    # 1. Resolve file path
    file_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))
    if not os.path.isfile(file_path):
        raise ValueError(f"Uploaded file not found: {filename}")

    # 2. Validate
    validate_csv(file_path)
    validate_config(config)

    # 3. Build training input
    training_input = TrainingInput(
        csv_path=file_path,
        model_name=config.training_model.value,
        prediction_length=config.prediction_length,
        dropout_rate=config.dropout_rate,
    )

    # 4. Call training client (external repo bridge)
    try:
        output = run_centralized_training(training_input)
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")

    # 5. Build forecast points for the frontend chart
    forecast: List[ForecastPoint] = []
    for i, pred in enumerate(output.predictions):
        actual = output.actuals[i] if output.actuals and i < len(output.actuals) else None
        forecast.append(ForecastPoint(
            step=i + 1,
            predicted=pred,
            actual=actual,
        ))

    return TrainingResult(
        success=True,
        message=(
            f"Training complete using {config.training_model.value} "
            f"with {config.prediction_length}-step horizon"
        ),
        model_name=config.training_model.value,
        prediction_length=config.prediction_length,
        dropout_rate=config.dropout_rate,
        training_time_seconds=output.training_time_seconds,
        metrics=TrainingMetrics(
            mae=output.mae,
            rmse=output.rmse,
            mape=output.mape,
        ),
        forecast=forecast,
    )
