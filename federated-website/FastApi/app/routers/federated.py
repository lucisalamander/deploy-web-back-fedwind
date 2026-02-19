
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field


router = APIRouter()

model_updates_queue: List[Dict[str, Any]] = []

# Directory for storing model files
MODELS_DIR = "models"


# Request/Response Models

class ModelUpdate(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client")
    round_number: int = Field(..., ge=0, description="Current federated learning round")
    training_model: str = Field(default="GPT4TS", description="Model used: GPT4TS, LLAMA, BERT, BART")
    federated_algorithm: str = Field(default="FedAvg", description="Algorithm: FedAvg, FedProx, FedBN, FedPer, SCAFFOLD")
    prediction_length: int = Field(default=6, description="Prediction horizon in hours")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate for regularization")
    num_clients: int = Field(default=5, ge=1, le=10, description="Total participating clients")
    model_weights: Dict[str, Any] = Field(
        ..., 
        description="Model weights/parameters as a dictionary (layer_name -> values)"
    )
    num_samples: int = Field(
        ..., 
        ge=1, 
        description="Number of training samples used (for weighted averaging)"
    )
    training_loss: Optional[float] = Field(
        None, 
        description="Final training loss (optional, for monitoring)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (training time, epochs, etc.)"
    )


class ModelUpdateResponse(BaseModel):
    """Response after receiving a model update."""
    success: bool
    message: str
    update_id: str
    client_id: str
    round_number: int
    training_model: str
    federated_algorithm: str
    received_at: str
    queue_position: int
    total_clients_in_round: int


class ModelSkeletonResponse(BaseModel):
    """Response containing the model skeleton/architecture for local training."""
    success: bool
    model_type: str
    model_version: str
    version: str
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    instructions: str


class QueueStatusResponse(BaseModel):
    """Status of the model updates queue."""
    total_updates: int
    updates_by_round: Dict[int, int]
    unique_clients: int
    oldest_update: Optional[str]
    newest_update: Optional[str]


# Endpoints

@router.post(
    "/model-update",
    response_model=ModelUpdateResponse,
    summary="Submit model update from federated client",
    description="""
    
    **Federated Learning Flow:**
    1. Client downloads model skeleton via `/model-update/skeleton`
    2. Client trains model locally on their private data
    3. Client sends ONLY model updates (weights) via this endpoint
    4. Backend queues updates for later aggregation
    """
)
async def receive_model_update(update: ModelUpdate):
    """
    Receive and queue a model update from a federated client with training parameters.
    
    The update contains only model weights - no raw training data.
    Updates are stored in memory for later aggregation by the LLM module.
    """
    # Generate unique ID for this update
    update_id = str(uuid4())[:8]
    received_at = datetime.now().isoformat()
    
    # Store the update in the queue
    update_record = {
        "update_id": update_id,
        "client_id": update.client_id,
        "round_number": update.round_number,
        "training_model": training_model,
        "federated_algorithm": algorithm,
        "prediction_length": update.prediction_length,
        "dropout_rate": update.dropout_rate,
        "num_clients": update.num_clients,
        "model_weights": update.model_weights,
        "num_samples": update.num_samples,
        "training_loss": update.training_loss,
        "metadata": update.metadata,
        "received_at": received_at
    }
    
    model_updates_queue.append(update_record)
    
    return ModelUpdateResponse(
        success=True,
        message=f"Model update received from client '{update.client_id}' (round {update.round_number}, {algorithm})",
        update_id=update_id,
        client_id=update.client_id,
        round_number=update.round_number,
        training_model=training_model,
        federated_algorithm=algorithm,
        received_at=received_at,
        queue_position=len(model_updates_queue),
        total_clients_in_round=len(set(u["client_id"] for u in model_updates_queue if u["round_number"] == update.round_number))
    )


@router.get(
    "/model-update/skeleton",
    response_model=ModelSkeletonResponse,
    summary="Get model skeleton for local training",
    description="""
    """
)
async def get_model_skeleton():
    """
    Return the model skeleton/architecture for clients to use in local training.
    
    This is a placeholder structure - the actual model architecture will be
    defined when the LLM/Transformer module is integrated.
    """
    return ModelSkeletonResponse(
        success=True,
        model_type="TransformerForecaster",
        model_version="1.0.0",
        version="1.0.0",
        architecture={
            "type": "TransformerForecaster",
            "description": "Transformer-based wind forecasting model with LLM support",
            "supported_models": ["GPT4TS", "LLAMA", "BERT", "BART"],
            "supported_algorithms": ["FedAvg", "FedProx", "FedBN", "FedPer", "SCAFFOLD"],
            "layers": {
                "embedding": {"input_dim": 64, "output_dim": 128},
                "transformer_encoder": {
                    "num_layers": 4,
                    "num_heads": 8,
                    "d_model": 128,
                    "d_ff": 512,
                    "dropout": 0.1
                },
                "output": {"input_dim": 128, "output_dim": 1}
            },
            "input_features": [
                "wind_speed", "wind_direction", "temperature",
                "pressure", "humidity", "hour", "day_of_year"
            ],
            "output_features": ["wind_speed_forecast"]
        },
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "local_epochs": 5,
            "optimizer": "Adam",
            "loss_function": "MSE",
            "supported_prediction_lengths": [1, 3, 6, 36, 72, 144, 432],
            "dropout_range": [0.0, 0.5]
        },
        instructions="""
        1. Choose your training model (GPT4TS, LLAMA, BERT, BART)
        2. Select federated algorithm (FedAvg, FedProx, FedBN, FedPer, SCAFFOLD)
        3. Initialize your local model with this architecture
        4. Load your private wind data (CSV from NASA POWER)
        5. Set dropout rate and prediction length as configured
        6. Train for the specified number of local_epochs
        7. Extract model weights as a dictionary
        8. Send weights via POST /model-update with algorithm info
        
        Note: Your raw data stays on your device. Only model weights are shared.
        Aggregation will use your selected federated algorithm.
        """
    )


@router.get(
    "/model-update/queue",
    response_model=QueueStatusResponse,
    summary="Get model updates queue status",
    description="Check the current status of queued model updates waiting for aggregation."
)
async def get_queue_status():
    """Return the status of the model updates queue."""
    if not model_updates_queue:
        return QueueStatusResponse(
            total_updates=0,
            updates_by_round={},
            unique_clients=0,
            oldest_update=None,
            newest_update=None
        )
    
    # Count updates by round
    updates_by_round: Dict[int, int] = {}
    unique_clients = set()
    
    for update in model_updates_queue:
        round_num = update["round_number"]
        updates_by_round[round_num] = updates_by_round.get(round_num, 0) + 1
        unique_clients.add(update["client_id"])
    
    return QueueStatusResponse(
        total_updates=len(model_updates_queue),
        updates_by_round=updates_by_round,
        unique_clients=len(unique_clients),
        oldest_update=model_updates_queue[0]["received_at"],
        newest_update=model_updates_queue[-1]["received_at"]
    )


@router.delete(
    "/model-update/queue",
    summary="Clear model updates queue",
    description=""
)
async def clear_queue():
    """Clear all model updates from the queue."""
    count = len(model_updates_queue)
    model_updates_queue.clear()
    
    return {
        "success": True,
        "message": f"Cleared {count} model update(s) from queue",
        "cleared_at": datetime.now().isoformat()
    }
