"""API tests with mocked DB init where possible."""

from __future__ import annotations

import os

os.environ["MOCK_INFERENCE"] = "true"

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.safety import analyze_user_message, emergency_response


def test_root():
    app = create_app()
    # Bypass lifespan DB requirement by using TestClient carefully
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/")
        assert res.status_code == 200
        body = res.json()
        assert "disclaimer" in body
        assert body["version"]


def test_safety_logic_used_by_chat():
    flags = analyze_user_message("I think I am having a heart attack")
    assert flags["is_emergency"]
    assert "emergency" in emergency_response().lower()
