"""
Feedback Router
===============
Simple feedback/comments endpoint for users to leave messages for creators.

Each feedback entry is stored as a JSON line in a file.
This is minimalistic storage - for production, use a database.
"""

import json
import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])

FEEDBACK_FILE = "feedback/user_feedback.jsonl"


class FeedbackCreate(BaseModel):
    """Schema for creating a feedback entry."""
    message: str = Field(..., min_length=1, max_length=10000, description="Feedback message")
    name: Optional[str] = Field(None, max_length=100, description="Optional name")
    context: Optional[str] = Field(None, max_length=200, description="Context (e.g., page, feature)")


class FeedbackEntry(BaseModel):
    """Schema for a stored feedback entry."""
    id: str
    message: str
    name: Optional[str]
    context: Optional[str]
    created_at: str


class FeedbackResponse(BaseModel):
    """Response after creating feedback."""
    success: bool
    message: str
    entry: FeedbackEntry


class FeedbackListResponse(BaseModel):
    """Response for listing all feedback (creator-only view)."""
    entries: list[FeedbackEntry]
    total: int


def _ensure_feedback_dir():
    """Ensure the feedback directory exists."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


def _generate_id():
    """Generate a simple unique ID."""
    import uuid
    return str(uuid.uuid4())[:8]


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackCreate):
    """
    Submit user feedback.
    
    Users can leave short comments/feedback that will be visible to creators only.
    """
    _ensure_feedback_dir()
    
    entry = FeedbackEntry(
        id=_generate_id(),
        message=feedback.message.strip(),
        name=feedback.name.strip() if feedback.name else None,
        context=feedback.context,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    
    # Append to JSONL file
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.model_dump()) + "\n")
    
    return FeedbackResponse(
        success=True,
        message="Thank you for your feedback!",
        entry=entry,
    )


@router.get("", response_model=FeedbackListResponse)
async def list_feedback():
    """
    List all feedback entries (for creators).
    
    Returns all submitted feedback in reverse chronological order.
    """
    _ensure_feedback_dir()
    
    entries: list[FeedbackEntry] = []
    
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(FeedbackEntry(**data))
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    # Reverse to show newest first
    entries.reverse()
    
    return FeedbackListResponse(entries=entries, total=len(entries))


@router.delete("/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """
    Delete a feedback entry by ID (for creators).
    """
    _ensure_feedback_dir()
    
    if not os.path.exists(FEEDBACK_FILE):
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Read all entries, filter out the one to delete
    remaining_entries = []
    found = False
    
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if data.get("id") != feedback_id:
                        remaining_entries.append(line)
                    else:
                        found = True
                except (json.JSONDecodeError, ValueError):
                    remaining_entries.append(line)
    
    if not found:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Write back the remaining entries
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        for line in remaining_entries:
            f.write(line + "\n")
    
    return {"success": True, "message": "Feedback deleted"}
