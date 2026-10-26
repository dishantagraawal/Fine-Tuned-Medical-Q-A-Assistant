"""Offline evaluation on held-out test split — never fabricates metrics."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path, max_samples: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def _token_overlap(reference: str, hypothesis: str) -> float:
    ref = set(reference.lower().split())
    hyp = set(hypothesis.lower().split())
    if not hyp:
        return 0.0
    return len(ref & hyp) / len(hyp)


def _rouge_l_f1(reference: str, hypothesis: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, hypothesis)["rougeL"].fmeasure
    except Exception:
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        if not ref or not hyp:
            return 0.0
        hyp_counts: dict[str, int] = {}
        for t in hyp:
            hyp_counts[t] = hyp_counts.get(t, 0) + 1
        common = 0
        for t in ref:
            if hyp_counts.get(t, 0) > 0:
                common += 1
                hyp_counts[t] -= 1
        precision = common / len(hyp)
        recall = common / len(ref)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def _eval_mode(rows: list[dict[str, str]], force_mock: bool = False) -> dict[str, Any]:
    from app.services.inference import InferenceEngine

    settings = get_settings()
    data = settings.model_dump()
    data["mock_inference"] = force_mock or settings.mock_inference
    eng_settings = Settings(**data)
    engine = InferenceEngine(eng_settings)

    latencies: list[float] = []
    overlaps: list[float] = []
    rouges: list[float] = []
    answers: list[dict[str, Any]] = []

    for row in rows:
        t0 = time.perf_counter()
        result = engine.generate(row["question"])
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        overlap = _token_overlap(row["answer"], result.answer)
        rouge = _rouge_l_f1(row["answer"], result.answer)
        overlaps.append(overlap)
        rouges.append(rouge)
        answers.append(
            {
                "question": row["question"][:200],
                "reference_len": len(row["answer"]),
                "prediction_len": len(result.answer),
                "latency_ms": round(latency, 2),
                "token_overlap": round(overlap, 4),
                "rougeL_f1": round(rouge, 4),
                "model_mode": result.model_mode,
            }
        )

    n = max(len(rows), 1)
    return {
        "sample_size": len(rows),
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0,
        "avg_token_overlap": round(sum(overlaps) / n, 4),
        "avg_rougeL_f1": round(sum(rouges) / n, 4),
        "model_mode": answers[0]["model_mode"] if answers else "unknown",
        "examples": answers[:5],
    }


def run_evaluate(max_samples: int = 50, compare_base: bool = False) -> dict[str, Any]:
    settings = get_settings()
    test_path = settings.processed_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing {test_path}. Run: python -m ml preprocess"
        )

    rows = _read_jsonl(test_path, max_samples=max_samples)
    if not rows:
        raise ValueError("Test split is empty")

    primary = _eval_mode(rows, force_mock=settings.mock_inference)
    comparison = None
    if compare_base:
        adapter = settings.adapters_dir / "latest"
        comparison = {
            "note": (
                "Base vs fine-tuned comparison requires a saved adapter under data/adapters/latest "
                "and running evaluation with MOCK_INFERENCE=false."
            ),
            "adapter_present": adapter.exists()
            or (settings.adapters_dir / "latest" / "adapter_config.json").exists(),
            "primary_mode": primary["model_mode"],
            "primary_metrics": {
                "avg_latency_ms": primary["avg_latency_ms"],
                "avg_token_overlap": primary["avg_token_overlap"],
                "avg_rougeL_f1": primary["avg_rougeL_f1"],
            },
        }

    payload = {
        "name": f"offline_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "model_mode": primary["model_mode"],
        "sample_size": primary["sample_size"],
        "metrics": {
            "avg_latency_ms": primary["avg_latency_ms"],
            "p50_latency_ms": primary["p50_latency_ms"],
            "avg_token_overlap": primary["avg_token_overlap"],
            "avg_rougeL_f1": primary["avg_rougeL_f1"],
        },
        "comparison": comparison,
        "examples": primary["examples"],
        "notes": (
            "Metrics computed on held-out test split only. "
            "No fabricated scores. Mock mode yields educational placeholders."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out = settings.processed_dir / "eval_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        _persist_eval_sync(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist eval to DB: %s", exc)

    return payload


def _persist_eval_sync(payload: dict[str, Any]) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from app.db.base import Base
    from app.db.models import EvalRun

    settings = get_settings()
    engine = create_engine(settings.database_url_sync)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(conn)
    with Session(engine) as session:
        session.add(
            EvalRun(
                name=payload["name"],
                model_mode=payload["model_mode"],
                sample_size=payload["sample_size"],
                metrics=payload["metrics"],
                comparison=payload.get("comparison"),
                notes=payload.get("notes"),
            )
        )
        session.commit()


if __name__ == "__main__":
    print(json.dumps(run_evaluate(), indent=2))
