"""Zip-aware HealthCareMagic preprocessing with leak-free splits."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app.config import get_settings

QUESTION_KEYS = ("input", "question", "query", "patient_question")
ANSWER_KEYS = ("output", "answer", "response", "doctor_answer")
INSTRUCTION_KEYS = ("instruction", "Instruction")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _first_field(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _is_zip(path: Path) -> bool:
    if zipfile.is_zipfile(path):
        return True
    with path.open("rb") as f:
        return f.read(4) == b"PK\x03\x04"


def extract_source_json(source: Path, extract_dir: Path) -> Path:
    """Return path to HealthCareMagic JSON, extracting from zip if needed."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    if _is_zip(source):
        with zipfile.ZipFile(source, "r") as zf:
            names = zf.namelist()
            target = next(
                (n for n in names if n.endswith("HealthCareMagic-100k.json") and not n.endswith("/")),
                None,
            )
            if target is None:
                target = next((n for n in names if n.endswith(".json") and "iCliniq" not in n), None)
            if target is None:
                raise FileNotFoundError("No HealthCareMagic JSON found inside archive")
            out = extract_dir / Path(target).name
            if not out.exists() or out.stat().st_size == 0:
                with zf.open(target) as src, out.open("wb") as dst:
                    dst.write(src.read())
            return out
    return source


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise ValueError("Unsupported JSON object structure")
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list of records")
    return data


def validate_and_map(records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    mapped: list[dict[str, str]] = []
    invalid = 0
    for rec in records:
        if not isinstance(rec, dict):
            invalid += 1
            continue
        question = _first_field(rec, QUESTION_KEYS)
        answer = _first_field(rec, ANSWER_KEYS)
        instruction = _first_field(rec, INSTRUCTION_KEYS) or (
            "If you are a doctor, please answer the medical questions based on the patient's description."
        )
        if not question or not answer:
            invalid += 1
            continue
        if len(question) < 10 or len(answer) < 10:
            invalid += 1
            continue
        mapped.append(
            {
                "instruction": instruction,
                "question": question,
                "answer": answer,
            }
        )
    return mapped, {"invalid": invalid}


def dedupe(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    dupes = 0
    for rec in records:
        key = hashlib.sha256(
            (_normalize(rec["question"]) + "||" + _normalize(rec["answer"])).encode("utf-8")
        ).hexdigest()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        unique.append(rec)
    return unique, dupes


def split_records(
    records: list[dict[str, str]], seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Hash-based 80/10/10 split on normalized question to avoid leakage."""
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for rec in records:
        h = hashlib.sha256(f"{seed}|{_normalize(rec['question'])}".encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 100
        if bucket < 80:
            train.append(rec)
        elif bucket < 90:
            val.append(rec)
        else:
            test.append(rec)
    return train, val, test


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_preprocess(
    source: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    settings = get_settings()
    source_path = Path(source) if source else settings.raw_dataset_path
    output = Path(out_dir) if out_dir else settings.processed_dir
    output.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    extracted = extract_source_json(source_path, settings.data_dir / "raw")
    raw = load_records(extracted)
    mapped, counts = validate_and_map(raw)
    unique, dupes = dedupe(mapped)
    train, val, test = split_records(unique, seed=seed)

    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "val.jsonl", val)
    write_jsonl(output / "test.jsonl", test)

    # Leakage check: no normalized question shared across splits
    def qset(rs: list[dict[str, str]]) -> set[str]:
        return {_normalize(r["question"]) for r in rs}

    leak_tv = len(qset(train) & qset(val))
    leak_tt = len(qset(train) & qset(test))
    leak_vt = len(qset(val) & qset(test))

    stats: dict[str, Any] = {
        "source_name": "HealthCareMagic-100k",
        "source_path": str(source_path),
        "extracted_path": str(extracted),
        "total_raw": len(raw),
        "valid": len(mapped),
        "invalid": counts["invalid"],
        "duplicates_removed": dupes,
        "unique": len(unique),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "split_ratios": "80/10/10",
        "split_method": "sha256(seed|normalized_question) % 100",
        "seed": seed,
        "leakage_question_overlap": {
            "train_val": leak_tv,
            "train_test": leak_tt,
            "val_test": leak_vt,
        },
        "fields": {"question": "input", "answer": "output", "instruction": "instruction"},
        "avg_question_len": round(sum(len(r["question"]) for r in unique) / max(len(unique), 1), 2),
        "avg_answer_len": round(sum(len(r["answer"]) for r in unique) / max(len(unique), 1), 2),
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Best-effort DB persistence
    try:
        _persist_stats_sync(stats)
    except Exception:
        pass

    return stats


def _persist_stats_sync(stats: dict[str, Any]) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from app.config import get_settings
    from app.db.models import DatasetStats
    from app.db.base import Base

    settings = get_settings()
    engine = create_engine(settings.database_url_sync)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(conn)
        conn.execute(text("UPDATE dataset_stats SET is_current = false WHERE is_current = true"))
    with Session(engine) as session:
        session.add(
            DatasetStats(
                source_name=stats["source_name"],
                total_raw=stats["total_raw"],
                valid=stats["valid"],
                duplicates_removed=stats["duplicates_removed"],
                train_count=stats["train_count"],
                val_count=stats["val_count"],
                test_count=stats["test_count"],
                stats_json=stats,
                is_current=True,
            )
        )
        session.commit()


if __name__ == "__main__":
    print(json.dumps(run_preprocess(), indent=2))
