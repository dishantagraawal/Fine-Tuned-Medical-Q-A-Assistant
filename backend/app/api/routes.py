"""HTTP API routes."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.config import get_settings
from app.db.base import get_db
from app.db.models import DatasetStats, Document, EvalRun, TrainingRun
from app.schemas import (
    HEALTHCARE_DISCLAIMER,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    DatasetStatsOut,
    DocumentOut,
    EvalRunOut,
    HealthResponse,
    MessageOut,
    TrainRequest,
    TrainingRunOut,
)
from app.services.chat import chat, get_conversation, list_conversations
from app.services.inference import get_inference_engine
from app.services.rag import ingest_pdf, list_documents

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    settings = get_settings()
    database_ok = False
    detail = None
    try:
        await db.execute(text("SELECT 1"))
        database_ok = True
    except Exception as exc:  # noqa: BLE001
        detail = f"database: {exc}"

    engine = get_inference_engine()
    # Do not force-load heavy weights on health; report configured/mock state
    mode = engine.mode if engine.mode != "unloaded" else (
        "mock" if settings.mock_inference else "lazy"
    )
    adapter = engine._resolve_adapter()
    status = "ok" if database_ok else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        database_ok=database_ok,
        inference_mode=mode,
        base_model=settings.base_model_id,
        adapter_loaded=adapter is not None,
        rag_enabled=settings.rag_enabled,
        embedding_model=settings.embedding_model,
        disclaimer=HEALTHCARE_DISCLAIMER,
        detail=detail,
    )


@router.post("/chat", response_model=ChatResponse)
async def post_chat(body: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty")
    try:
        return await chat(db, body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(db: AsyncSession = Depends(get_db)) -> list[ConversationSummary]:
    rows = await list_conversations(db)
    return [ConversationSummary(**r) for r in rows]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: str, db: AsyncSession = Depends(get_db)
) -> ConversationDetail:
    conv = await get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = sorted(conv.messages, key=lambda m: m.created_at or m.id)
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                model_mode=m.model_mode,
                latency_ms=m.latency_ms,
                sources=m.sources,
                safety_flags=m.safety_flags,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.get("/documents", response_model=list[DocumentOut])
async def get_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentOut]:
    docs = await list_documents(db)
    return [
        DocumentOut(
            id=d.id,
            filename=d.filename,
            original_name=d.original_name,
            page_count=d.page_count,
            chunk_count=d.chunk_count,
            status=d.status,
            error_message=d.error_message,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.post("/documents/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = await ingest_pdf(db, dest, file.filename)
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        original_name=doc.original_name,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error_message=doc.error_message,
        created_at=doc.created_at,
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await db.delete(doc)
    await db.commit()
    return {"deleted": document_id}


@router.get("/dataset/stats", response_model=DatasetStatsOut)
async def dataset_stats(db: AsyncSession = Depends(get_db)) -> DatasetStatsOut:
    settings = get_settings()
    stats_path = settings.processed_dir / "stats.json"
    result = await db.execute(
        select(DatasetStats).where(DatasetStats.is_current.is_(True)).order_by(DatasetStats.id.desc())
    )
    row = result.scalars().first()
    if row:
        return DatasetStatsOut(
            source_name=row.source_name,
            total_raw=row.total_raw,
            valid=row.valid,
            duplicates_removed=row.duplicates_removed,
            train_count=row.train_count,
            val_count=row.val_count,
            test_count=row.test_count,
            stats_json=row.stats_json,
            created_at=row.created_at,
            processed_available=stats_path.exists(),
        )
    if stats_path.exists():
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        return DatasetStatsOut(
            source_name=data.get("source_name", "HealthCareMagic-100k"),
            total_raw=data.get("total_raw", 0),
            valid=data.get("valid", 0),
            duplicates_removed=data.get("duplicates_removed", 0),
            train_count=data.get("train_count", 0),
            val_count=data.get("val_count", 0),
            test_count=data.get("test_count", 0),
            stats_json=data,
            processed_available=True,
        )
    return DatasetStatsOut(
        source_name="HealthCareMagic-100k",
        total_raw=0,
        valid=0,
        duplicates_removed=0,
        train_count=0,
        val_count=0,
        test_count=0,
        processed_available=False,
    )


@router.get("/training/runs", response_model=list[TrainingRunOut])
async def training_runs(db: AsyncSession = Depends(get_db)) -> list[TrainingRunOut]:
    result = await db.execute(select(TrainingRun).order_by(TrainingRun.created_at.desc()))
    rows = result.scalars().all()
    return [
        TrainingRunOut(
            id=r.id,
            base_model=r.base_model,
            adapter_path=r.adapter_path,
            hardware_profile=r.hardware_profile,
            status=r.status,
            config=r.config,
            metrics=r.metrics,
            error_message=r.error_message,
            created_at=r.created_at,
            finished_at=r.finished_at,
        )
        for r in rows
    ]


@router.post("/training/runs", response_model=TrainingRunOut)
async def create_training_run(
    body: TrainRequest, db: AsyncSession = Depends(get_db)
) -> TrainingRunOut:
    """Register a training run config. Actual training is started via CLI."""
    settings = get_settings()
    run = TrainingRun(
        id=str(uuid.uuid4()),
        base_model=settings.base_model_id,
        hardware_profile=body.hardware_profile or settings.hardware_profile,
        status="registered",
        config=body.model_dump(),
        metrics=None,
    )
    session_metrics_path = settings.adapters_dir / "latest_training_metrics.json"
    if session_metrics_path.exists():
        run.metrics = json.loads(session_metrics_path.read_text(encoding="utf-8"))
        run.status = "completed_offline"
        run.adapter_path = str(settings.adapters_dir / "latest")
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return TrainingRunOut(
        id=run.id,
        base_model=run.base_model,
        adapter_path=run.adapter_path,
        hardware_profile=run.hardware_profile,
        status=run.status,
        config=run.config,
        metrics=run.metrics,
        error_message=run.error_message,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )


@router.get("/evaluation/runs", response_model=list[EvalRunOut])
async def evaluation_runs(db: AsyncSession = Depends(get_db)) -> list[EvalRunOut]:
    settings = get_settings()
    # Sync file-based eval results into DB if present and DB empty of matching id
    eval_path = settings.processed_dir / "eval_results.json"
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        existing = await db.execute(
            select(EvalRun).where(EvalRun.name == payload.get("name", "offline_eval"))
        )
        if existing.scalars().first() is None:
            run = EvalRun(
                id=str(uuid.uuid4()),
                name=payload.get("name", "offline_eval"),
                model_mode=payload.get("model_mode", "unknown"),
                sample_size=payload.get("sample_size", 0),
                metrics=payload.get("metrics", {}),
                comparison=payload.get("comparison"),
                notes=payload.get("notes"),
            )
            db.add(run)
            await db.commit()

    result = await db.execute(select(EvalRun).order_by(EvalRun.created_at.desc()))
    rows = result.scalars().all()
    return [
        EvalRunOut(
            id=r.id,
            name=r.name,
            model_mode=r.model_mode,
            sample_size=r.sample_size,
            metrics=r.metrics or {},
            comparison=r.comparison,
            notes=r.notes,
            created_at=r.created_at,
        )
        for r in rows
    ]
