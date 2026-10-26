"""Pytest fixtures — force mock inference for unit tests."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MOCK_INFERENCE", "true")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://medical:medical@localhost:5433/medical_qa")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql://medical:medical@localhost:5433/medical_qa")


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MOCK_INFERENCE", "true")
    settings = get_settings()
    settings.mock_inference = True
    yield
    get_settings.cache_clear()
