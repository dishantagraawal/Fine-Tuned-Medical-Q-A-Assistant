"""PDF ingest, chunking, indexing, and pgvector retrieval."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import Chunk, Document
from app.schemas import INSUFFICIENT_INFO_MESSAGE, SourceCitation
from app.services.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        pages.append((i, page.get_text("text") or ""))
    doc.close()
    return pages


async def ingest_pdf(
    session: AsyncSession,
    file_path: Path,
    original_name: str,
    settings: Settings | None = None,
) -> Document:
    settings = settings or get_settings()
    document = Document(
        id=str(uuid.uuid4()),
        filename=file_path.name,
        original_name=original_name,
        status="processing",
    )
    session.add(document)
    await session.flush()

    try:
        pages = extract_pdf_pages(file_path)
        all_texts: list[str] = []
        meta: list[tuple[int, int, str]] = []  # chunk_index, page, text
        idx = 0
        for page_number, page_text in pages:
            for piece in chunk_text(page_text, settings.chunk_size, settings.chunk_overlap):
                meta.append((idx, page_number, piece))
                all_texts.append(piece)
                idx += 1

        vectors = embed_texts(all_texts) if all_texts else []
        for i, (chunk_index, page_number, piece) in enumerate(meta):
            emb = vectors[i].tolist() if len(vectors) else None
            session.add(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    text=piece,
                    embedding=emb,
                )
            )

        document.page_count = len(pages)
        document.chunk_count = len(meta)
        document.status = "ready"
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF ingest failed")
        document.status = "error"
        document.error_message = str(exc)

    await session.commit()
    await session.refresh(document)
    return document


async def retrieve(
    session: AsyncSession,
    query: str,
    settings: Settings | None = None,
) -> tuple[bool, list[SourceCitation], str]:
    """Return (sufficient, citations, context_block)."""
    settings = settings or get_settings()
    vector = embed_query(query)
    # Cosine distance with pgvector: 1 - cosine_similarity for <=> on normalized vectors
    # Using cosine distance operator
    sql = text(
        """
        SELECT c.id, c.text, c.page_number, d.original_name,
               1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready' AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
    )
    emb_literal = "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"
    result = await session.execute(
        sql,
        {
            "embedding": emb_literal,
            "top_k": settings.top_k,
        },
    )
    rows = result.fetchall()
    citations: list[SourceCitation] = []
    for row in rows:
        score = float(row.score or 0.0)
        if score < settings.similarity_threshold:
            continue
        citations.append(
            SourceCitation(
                document_name=row.original_name,
                page_number=int(row.page_number),
                chunk_id=str(row.id),
                score=score,
                excerpt=(row.text[:400] + "…") if len(row.text) > 400 else row.text,
            )
        )

    if not citations:
        return False, [], INSUFFICIENT_INFO_MESSAGE

    parts = []
    for i, c in enumerate(citations, start=1):
        parts.append(
            f"[{i}] document={c.document_name} | page={c.page_number} | "
            f"score={c.score:.4f}\n{c.excerpt}"
        )
    return True, citations, "\n\n".join(parts)


async def list_documents(session: AsyncSession) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())
