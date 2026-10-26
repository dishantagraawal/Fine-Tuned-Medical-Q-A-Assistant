"""Chat orchestration: safety → optional RAG → inference → persistence."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Message
from app.schemas import ChatRequest, ChatResponse, SourceCitation
from app.services.inference import get_inference_engine
from app.services.rag import retrieve
from app.services.safety import (
    analyze_user_message,
    disclaimer,
    emergency_response,
    harmful_response,
)


def _title_from_message(message: str) -> str:
    text = message.strip().replace("\n", " ")
    return (text[:60] + "…") if len(text) > 60 else text or "New conversation"


async def chat(session: AsyncSession, req: ChatRequest) -> ChatResponse:
    flags = analyze_user_message(req.message)

    if req.conversation_id:
        conv = await session.get(Conversation, req.conversation_id)
        if conv is None:
            conv = Conversation(id=str(uuid.uuid4()), title=_title_from_message(req.message))
            session.add(conv)
    else:
        conv = Conversation(id=str(uuid.uuid4()), title=_title_from_message(req.message))
        session.add(conv)
    await session.flush()

    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="user",
        content=req.message.strip(),
        safety_flags=flags,
    )
    session.add(user_msg)

    sources: list[SourceCitation] = []
    if flags["block_generation"]:
        answer = emergency_response() if flags["is_emergency"] else harmful_response()
        model_mode = "safety"
        latency_ms = 0.0
        sufficient = True
    else:
        context = None
        sufficient = True
        if req.use_rag:
            sufficient, sources, context_or_msg = await retrieve(session, req.message)
            if not sufficient:
                answer = context_or_msg
                model_mode = "rag_insufficient"
                latency_ms = 0.0
            else:
                context = context_or_msg
                engine = get_inference_engine()
                result = engine.generate(req.message, context=context)
                answer = result.answer
                model_mode = result.model_mode
                latency_ms = result.latency_ms
                flags = {**flags, "uncertain": result.uncertain}
        else:
            engine = get_inference_engine()
            result = engine.generate(req.message)
            answer = result.answer
            model_mode = result.model_mode
            latency_ms = result.latency_ms
            flags = {**flags, "uncertain": result.uncertain}

    assistant_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        model_mode=model_mode,
        latency_ms=latency_ms,
        sources=[s.model_dump() for s in sources] if sources else None,
        safety_flags=flags,
    )
    session.add(assistant_msg)
    await session.commit()

    return ChatResponse(
        conversation_id=conv.id,
        message_id=assistant_msg.id,
        answer=answer,
        model_mode=model_mode,
        latency_ms=latency_ms,
        sources=sources,
        safety_flags=flags,
        disclaimer=disclaimer(),
        sufficient=sufficient,
    )


async def list_conversations(session: AsyncSession) -> list[dict[str, Any]]:
    msg_count = (
        select(Message.conversation_id, func.count(Message.id).label("cnt"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    q = (
        select(Conversation, func.coalesce(msg_count.c.cnt, 0))
        .outerjoin(msg_count, Conversation.id == msg_count.c.conversation_id)
        .order_by(Conversation.updated_at.desc())
    )
    rows = (await session.execute(q)).all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at,
            "updated_at": c.updated_at,
            "message_count": int(cnt),
        }
        for c, cnt in rows
    ]


async def get_conversation(session: AsyncSession, conversation_id: str) -> Conversation | None:
    result = await session.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    return result.scalar_one_or_none()
