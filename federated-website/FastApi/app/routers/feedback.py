"""
Feedback Router
===============
Stores website feedback/questions, supports answering them,
exposes public answered questions for the website,
and processes Telegram webhook replies.

Storage is still JSONL for simplicity.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.telegram_service import (
    SUPPORT_CHAT_ID,
    delete_telegram_webhook,
    get_telegram_webhook_info,
    send_telegram_message,
    set_telegram_webhook,
)

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])

FEEDBACK_FILE = "feedback/user_feedback.jsonl"


class FeedbackCreate(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="Feedback/question text")
    name: Optional[str] = Field(None, max_length=100, description="Optional name")
    context: Optional[str] = Field(None, max_length=200, description="Context (e.g. dashboard)")


class FeedbackEntry(BaseModel):
    id: str
    message: str
    name: Optional[str] = None
    context: Optional[str] = None
    created_at: str

    status: str = "pending"  # pending | answered
    answer_text: Optional[str] = None
    answered_at: Optional[str] = None
    answered_by: Optional[str] = None
    is_public: bool = False

    telegram_message_id: Optional[int] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    entry: FeedbackEntry


class FeedbackListResponse(BaseModel):
    entries: list[FeedbackEntry]
    total: int


class AnswerFeedbackRequest(BaseModel):
    answer_text: str = Field(..., min_length=1, max_length=10000)
    answered_by: Optional[str] = Field(default="Developer", max_length=100)
    is_public: bool = True


class PublicAnswerItem(BaseModel):
    id: str
    question: str
    answer_text: str
    created_at: str
    answered_at: str
    asked_by: str


class PublicAnswersResponse(BaseModel):
    entries: list[PublicAnswerItem]
    total: int


class TelegramWebhookSetupRequest(BaseModel):
    public_base_url: str = Field(..., min_length=1, description="Public HTTPS base URL, without trailing slash")


def _ensure_feedback_dir():
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


def _generate_id() -> str:
    return str(uuid.uuid4())[:8]


def _normalize_entry(data: dict) -> FeedbackEntry:
    telegram_message_id = data.get("telegram_message_id")
    if telegram_message_id is not None:
        try:
            telegram_message_id = int(telegram_message_id)
        except (TypeError, ValueError):
            telegram_message_id = None

    return FeedbackEntry(
        id=str(data.get("id") or _generate_id()),
        message=str(data.get("message") or "").strip(),
        name=str(data["name"]).strip() if data.get("name") else None,
        context=str(data["context"]).strip() if data.get("context") else None,
        created_at=str(data.get("created_at") or (datetime.utcnow().isoformat() + "Z")),
        status=str(data.get("status") or "pending"),
        answer_text=str(data["answer_text"]).strip() if data.get("answer_text") else None,
        answered_at=str(data["answered_at"]) if data.get("answered_at") else None,
        answered_by=str(data["answered_by"]).strip() if data.get("answered_by") else None,
        is_public=bool(data.get("is_public", False)),
        telegram_message_id=telegram_message_id,
    )


def _read_entries() -> list[FeedbackEntry]:
    _ensure_feedback_dir()
    entries: list[FeedbackEntry] = []

    if not os.path.exists(FEEDBACK_FILE):
        return entries

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entries.append(_normalize_entry(raw))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    return entries


def _write_entries(entries: list[FeedbackEntry]) -> None:
    _ensure_feedback_dir()
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")


def _build_telegram_question_text(entry: FeedbackEntry) -> str:
    return (
        f"📩 New Website Question\n\n"
        f"🆔 ID: {entry.id}\n"
        f"👤 Name: {entry.name or 'Anonymous'}\n"
        f"📍 Context: {entry.context or 'N/A'}\n"
        f"🕒 Created At: {entry.created_at}\n"
        f"📌 Status: {entry.status}\n\n"
        f"❓ Question:\n{entry.message}\n\n"
        f"Reply to this message with:\n"
        f"ANSWER: your answer here"
    )


def _find_entry_by_telegram_message_id(entries: list[FeedbackEntry], telegram_message_id: int) -> Optional[int]:
    for index, entry in enumerate(entries):
        if entry.telegram_message_id == telegram_message_id:
            return index
    return None


def _extract_sender_name(message_from: dict) -> str:
    username = (message_from.get("username") or "").strip()
    first_name = (message_from.get("first_name") or "").strip()
    last_name = (message_from.get("last_name") or "").strip()

    if username:
        return f"@{username}"
    full_name = f"{first_name} {last_name}".strip()
    return full_name or "Developer"


def _apply_answer(entry: FeedbackEntry, answer_text: str, answered_by: str, is_public: bool = True) -> FeedbackEntry:
    entry.answer_text = answer_text.strip()
    entry.answered_at = datetime.utcnow().isoformat() + "Z"
    entry.answered_by = answered_by.strip() if answered_by else "Developer"
    entry.status = "answered"
    entry.is_public = is_public
    return entry


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackCreate):
    entries = _read_entries()

    entry = FeedbackEntry(
        id=_generate_id(),
        message=feedback.message.strip(),
        name=feedback.name.strip() if feedback.name else None,
        context=feedback.context.strip() if feedback.context else None,
        created_at=datetime.utcnow().isoformat() + "Z",
        status="pending",
        answer_text=None,
        answered_at=None,
        answered_by=None,
        is_public=False,
        telegram_message_id=None,
    )

    entries.append(entry)

    try:
        telegram_response = send_telegram_message(_build_telegram_question_text(entry))
        message_id = telegram_response.get("result", {}).get("message_id")
        if isinstance(message_id, int):
            entry.telegram_message_id = message_id
    except Exception as e:
        print(f"Failed to send feedback to Telegram: {e}")

    _write_entries(entries)

    return FeedbackResponse(
        success=True,
        message="Thank you! Your message has been sent.",
        entry=entry,
    )


@router.get("", response_model=FeedbackListResponse)
async def list_feedback():
    entries = _read_entries()
    entries.reverse()
    return FeedbackListResponse(entries=entries, total=len(entries))


@router.patch("/{feedback_id}/answer", response_model=FeedbackResponse)
async def answer_feedback(feedback_id: str, payload: AnswerFeedbackRequest):
    entries = _read_entries()

    for i, entry in enumerate(entries):
        if entry.id == feedback_id:
            entry = _apply_answer(
                entry=entry,
                answer_text=payload.answer_text,
                answered_by=payload.answered_by or "Developer",
                is_public=payload.is_public,
            )
            entries[i] = entry
            _write_entries(entries)

            return FeedbackResponse(
                success=True,
                message="Feedback answered successfully.",
                entry=entry,
            )

    raise HTTPException(status_code=404, detail="Feedback not found")


@router.get("/public-answers", response_model=PublicAnswersResponse)
async def list_public_answers():
    entries = _read_entries()

    public_entries = [
        PublicAnswerItem(
            id=entry.id,
            question=entry.message,
            answer_text=entry.answer_text or "",
            created_at=entry.created_at,
            answered_at=entry.answered_at or "",
            asked_by="Anonymous",
        )
        for entry in entries
        if entry.status == "answered" and entry.is_public and entry.answer_text
    ]

    public_entries.sort(key=lambda x: x.answered_at, reverse=True)

    return PublicAnswersResponse(entries=public_entries, total=len(public_entries))


@router.post("/telegram/set-webhook")
async def setup_telegram_webhook(payload: TelegramWebhookSetupRequest):
    base_url = payload.public_base_url.strip().rstrip("/")
    webhook_url = f"{base_url}/api/feedback/telegram/webhook"

    result = set_telegram_webhook(webhook_url)

    return {
        "success": True,
        "message": "Telegram webhook set successfully.",
        "webhook_url": webhook_url,
        "telegram_response": result,
    }


@router.get("/telegram/webhook-info")
async def telegram_webhook_info():
    result = get_telegram_webhook_info()
    return {
        "success": True,
        "telegram_response": result,
    }


@router.post("/telegram/delete-webhook")
async def remove_telegram_webhook():
    result = delete_telegram_webhook()
    return {
        "success": True,
        "message": "Telegram webhook deleted successfully.",
        "telegram_response": result,
    }


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update: Any = await request.json()

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True, "ignored": "No message in update"}

    chat = message.get("chat") or {}
    incoming_chat_id = str(chat.get("id"))

    if SUPPORT_CHAT_ID and incoming_chat_id != str(SUPPORT_CHAT_ID):
        return {"ok": True, "ignored": "Message came from another chat"}

    text = (message.get("text") or "").strip()
    if not text:
        return {"ok": True, "ignored": "No text message"}

    reply_to_message = message.get("reply_to_message")
    if not reply_to_message:
        return {"ok": True, "ignored": "Not a reply"}

    if ":" not in text:
        return {"ok": True, "ignored": "No ANSWER: prefix"}

    prefix, body = text.split(":", 1)
    if prefix.strip().lower() != "answer":
        return {"ok": True, "ignored": "Message is not public answer format"}

    answer_text = body.strip()
    if not answer_text:
        return {"ok": True, "ignored": "Answer text is empty"}

    replied_message_id = reply_to_message.get("message_id")
    if not isinstance(replied_message_id, int):
        return {"ok": True, "ignored": "Replied message_id missing"}

    entries = _read_entries()
    target_index = _find_entry_by_telegram_message_id(entries, replied_message_id)

    if target_index is None:
        return {"ok": True, "ignored": "No matching feedback entry"}

    answered_by = _extract_sender_name(message.get("from") or {})

    entry = entries[target_index]
    entry = _apply_answer(
        entry=entry,
        answer_text=answer_text,
        answered_by=answered_by,
        is_public=True,
    )
    entries[target_index] = entry
    _write_entries(entries)

    return {
        "ok": True,
        "message": "Answer saved successfully.",
        "feedback_id": entry.id,
    }


@router.delete("/{feedback_id}")
async def delete_feedback(feedback_id: str):
    entries = _read_entries()

    remaining_entries = [entry for entry in entries if entry.id != feedback_id]

    if len(remaining_entries) == len(entries):
        raise HTTPException(status_code=404, detail="Feedback not found")

    _write_entries(remaining_entries)
    return {"success": True, "message": "Feedback deleted"}