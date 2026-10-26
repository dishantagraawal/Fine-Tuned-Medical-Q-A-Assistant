"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Fine-Tuned Medical Q&A Assistant"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    database_url: str = (
        "postgresql+asyncpg://medical:medical@localhost:5433/medical_qa"
    )
    database_url_sync: str = (
        "postgresql://medical:medical@localhost:5433/medical_qa"
    )

    data_dir: Path = ROOT / "data"
    raw_dataset_path: Path = ROOT / "HealthCareMagic-100k.json"
    processed_dir: Path = ROOT / "data" / "processed"
    uploads_dir: Path = ROOT / "data" / "uploads"
    adapters_dir: Path = ROOT / "data" / "adapters"

    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_adapter_path: Optional[Path] = None
    hardware_profile: str = "cpu"
    max_new_tokens: int = 512
    inference_enabled: bool = True
    mock_inference: bool = False

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    similarity_threshold: float = 0.25
    rag_enabled: bool = True

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"
    use_ollama_fallback: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)
    return settings
