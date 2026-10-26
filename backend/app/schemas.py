"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


HEALTHCARE_DISCLAIMER = (
    "This assistant provides educational information only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "If you think you may be having a medical emergency, call your local emergency number immediately."
)

INSUFFICIENT_INFO_MESSAGE = (
    "I could not find sufficient information in the available medical sources."
)

EMERGENCY_MESSAGE = (
    "Your message may describe a medical emergency. "
    "Please seek emergency medical care immediately or call your local emergency services. "
    "This assistant cannot provide emergency medical care."
)


class HealthResponse(BaseModel):
    status: str
    version: str
    database_ok: bool
    inference_mode: str
    base_model: str
    adapter_loaded: bool
    rag_enabled: bool
    embedding_model: str
    disclaimer: str = HEALTHCARE_DISCLAIMER
    detail: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    conversation_id: Optional[str] = None
    use_rag: bool = False


class SourceCitation(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    model_mode: str
    latency_ms: float
    sources: list[SourceCitation] = []
    safety_flags: dict[str, Any] = {}
    disclaimer: str = HEALTHCARE_DISCLAIMER
    sufficient: bool = True


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    model_mode: Optional[str] = None
    latency_ms: Optional[float] = None
    sources: Optional[list[Any]] = None
    safety_flags: Optional[dict[str, Any]] = None
    created_at: datetime


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class DocumentOut(BaseModel):
    id: str
    filename: str
    original_name: str
    page_count: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class DatasetStatsOut(BaseModel):
    source_name: str
    total_raw: int
    valid: int
    duplicates_removed: int
    train_count: int
    val_count: int
    test_count: int
    stats_json: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None
    processed_available: bool = False


class TrainingRunOut(BaseModel):
    id: str
    base_model: str
    adapter_path: Optional[str]
    hardware_profile: str
    status: str
    config: Optional[dict[str, Any]]
    metrics: Optional[dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]


class EvalRunOut(BaseModel):
    id: str
    name: str
    model_mode: str
    sample_size: int
    metrics: dict[str, Any]
    comparison: Optional[dict[str, Any]]
    notes: Optional[str]
    created_at: datetime


class TrainRequest(BaseModel):
    max_samples: int = 2000
    epochs: float = 1.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 2
    hardware_profile: Optional[str] = None
