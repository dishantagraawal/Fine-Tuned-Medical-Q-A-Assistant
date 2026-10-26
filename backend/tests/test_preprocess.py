"""Dataset preprocessing unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from ml.preprocess import dedupe, split_records, validate_and_map, write_jsonl


def test_validate_and_map_filters_invalid():
    records = [
        {
            "instruction": "Answer medical questions.",
            "input": "I have had a headache for three days, what could it be?",
            "output": "Headaches can have many causes; rest and hydration may help, but see a doctor if severe.",
        },
        {"instruction": "x", "input": "short", "output": "no"},
        {"foo": "bar"},
    ]
    mapped, counts = validate_and_map(records)
    assert len(mapped) == 1
    assert mapped[0]["question"].startswith("I have had")
    assert counts["invalid"] == 2


def test_dedupe_removes_duplicates():
    rec = {
        "instruction": "i",
        "question": "What is fever?",
        "answer": "Fever is an elevated body temperature.",
    }
    unique, dupes = dedupe([rec, dict(rec), dict(rec)])
    assert len(unique) == 1
    assert dupes == 2


def test_split_no_leakage(tmp_path: Path):
    records = [
        {
            "instruction": "i",
            "question": f"Question number {i} about symptoms and care?",
            "answer": f"Answer number {i} with enough detail here.",
        }
        for i in range(200)
    ]
    train, val, test = split_records(records, seed=42)
    assert len(train) + len(val) + len(test) == 200
    tq = {r["question"].lower() for r in train}
    vq = {r["question"].lower() for r in val}
    eq = {r["question"].lower() for r in test}
    assert not (tq & vq)
    assert not (tq & eq)
    assert not (vq & eq)

    out = tmp_path / "train.jsonl"
    write_jsonl(out, train[:3])
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "question" in json.loads(lines[0])
