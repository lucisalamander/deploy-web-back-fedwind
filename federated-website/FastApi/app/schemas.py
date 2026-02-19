"""
Pydantic schemas for request/response validation.

Defines the contract between:
  - Frontend -> FastAPI  (request schemas)
  - FastAPI -> Frontend  (response schemas)
  - FastAPI -> Training module (internal schemas)

CSV Format (NASA POWER hourly data):
  YEAR, MO, DY, HR, WS10M
  2026,  1,  1,  0,  8.65
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ---------------------------------------------------------------------------
# Enums - valid configuration values
# ---------------------------------------------------------------------------

class TrainingModelName(str, Enum):
    GPT4TS = "GPT4TS"
    LLAMA = "LLAMA"
    BERT = "BERT"
    BART = "BART"


class TrainingMode(str, Enum):
    CENTRALIZED = "centralized"
    FEDERATED = "federated"


VALID_PREDICTION_LENGTHS = [1, 3, 6, 36, 72, 144, 432]


# ---------------------------------------------------------------------------
# Request schemas  (Frontend -> FastAPI)
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """
    Training configuration sent alongside file upload.
    Maps 1-to-1 with dashboard form fields.
    """
    training_model: TrainingModelName = Field(
        default=TrainingModelName.GPT4TS,
        description="LLM backbone: GPT4TS | LLAMA | BERT | BART",
    )
    prediction_length: int = Field(
        default=6,
        description="Forecast horizon in hourly steps",
    )
    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Dropout regularization (0.0 - 0.5)",
    )
    mode: TrainingMode = Field(
        default=TrainingMode.CENTRALIZED,
        description="centralized or federated",
    )


# ---------------------------------------------------------------------------
# Response schemas  (FastAPI -> Frontend)
# ---------------------------------------------------------------------------

class FileInfo(BaseModel):
    """Metadata about a single uploaded file."""
    filename: str
    original_name: str
    size_bytes: int
    rows: int
    columns: int
    column_names: List[str]
    preview: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    """Returned after POST /api/upload."""
    success: bool
    message: str
    file: FileInfo


class TrainingMetrics(BaseModel):
    """Evaluation metrics returned from the training module."""
    mae: float = Field(description="Mean Absolute Error")
    rmse: float = Field(description="Root Mean Squared Error")
    mape: Optional[float] = Field(default=None, description="Mean Absolute Percentage Error (%)")


class ForecastPoint(BaseModel):
    """A single timestamp in the forecast output."""
    step: int
    timestamp: Optional[str] = None
    actual: Optional[float] = None
    predicted: float


class TrainingResult(BaseModel):
    """
    Returned after POST /api/train.
    This is what the frontend renders in the results panel.
    """
    success: bool
    message: str
    model_name: str
    prediction_length: int
    dropout_rate: float
    training_time_seconds: float
    metrics: TrainingMetrics
    forecast: List[ForecastPoint]


class TrainRequest(BaseModel):
    """
    Body for POST /api/train.
    References a previously uploaded file plus config.
    """
    filename: str = Field(description="Saved filename from upload response")
    config: TrainingConfig


class FileListItem(BaseModel):
    """Single file entry in the file list response."""
    filename: str
    size: int
    modified: str


class FileListResponse(BaseModel):
    """Returned by GET /api/files."""
    files: List[FileListItem]
    total: int


class HealthResponse(BaseModel):
    """Returned by GET /health."""
    status: str
    version: str
    environment: str
    timestamp: str
    services: Dict[str, str]


# ---------------------------------------------------------------------------
# Internal schemas  (FastAPI service -> Training module)
# ---------------------------------------------------------------------------

class TrainingInput(BaseModel):
    """
    Structured input passed from the service layer to the
    centralized training module.  This is the bridge object.
    """
    csv_path: str = Field(description="Absolute path to the saved CSV file")
    model_name: str = Field(description="GPT4TS | LLAMA | BERT | BART")
    prediction_length: int = Field(description="Forecast horizon (steps)")
    dropout_rate: float = Field(description="Dropout for regularization")
    # Defaults used by the training repo
    seq_len: int = Field(default=336, description="Input sequence length")
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    epochs: int = Field(default=10, description="Training epochs")


class TrainingOutput(BaseModel):
    """
    Structured output returned by the centralized training module.
    The service layer converts this into TrainingResult for the frontend.
    """
    mae: float
    rmse: float
    mape: Optional[float] = None
    training_time_seconds: float
    predictions: List[float]
    actuals: Optional[List[float]] = None
