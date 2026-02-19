"""
Training Router - Triggers centralized model training.

Endpoint:
  POST /api/train  -  Start training with uploaded file + config

Data flow:
  Frontend (config + filename)
    -> This router (validates request)
    -> training_service.start_training()
    -> training_client.run_centralized_training()
    -> TrainingResult back to frontend
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas import TrainRequest, TrainingResult


router = APIRouter(prefix="/api")


@router.post(
    "/train",
    response_model=TrainingResult,
    summary="Start centralized training",
    description=(
        "Triggers centralized model training using a previously uploaded CSV file. "
        "Pass the `filename` from the upload response along with training config. "
        "Returns metrics (MAE, RMSE, MAPE) and forecast predictions."
    ),
)
async def train(request: TrainRequest):
    """
    Start centralized training.

    Request body example:
    {
        "filename": "20260213_143022_POWER_Hourly_Data.csv",
        "config": {
            "training_model": "GPT4TS",
            "prediction_length": 6,
            "dropout_rate": 0.2,
            "mode": "centralized"
        }
    }

    Response example:
    {
        "success": true,
        "message": "Training complete using GPT4TS with 6-step horizon",
        "model_name": "GPT4TS",
        "prediction_length": 6,
        "dropout_rate": 0.2,
        "training_time_seconds": 3.21,
        "metrics": { "mae": 0.6757, "rmse": 0.8863, "mape": 5.8 },
        "forecast": [
            { "step": 1, "predicted": 8.12, "actual": 8.20 },
            { "step": 2, "predicted": 8.35, "actual": 8.50 },
            ...
        ]
    }
    """
    # Import here to keep router thin
    from app.services.training_service import start_training

    try:
        result = start_training(request.filename, request.config)
        return result

    except ValueError as e:
        # Validation errors (bad CSV, bad config)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except RuntimeError as e:
        # Training execution errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )
