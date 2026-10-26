"""HF model inference with optional LoRA adapter; mock mode for tests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from app.config import Settings, get_settings
from app.services.safety import wrap_answer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful medical education assistant. "
    "Explain medical concepts in simple, understandable language. "
    "Do not diagnose patients or prescribe treatments. "
    "If uncertain, say so clearly and recommend consulting a clinician. "
    "Educational information only — not a substitute for professional medical advice."
)


@dataclass
class InferenceResult:
    answer: str
    model_mode: str
    latency_ms: float
    uncertain: bool = False


class InferenceEngine:
    """Lazy-loaded transformers generator."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._mode = "unloaded"
        self._lock = Lock()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def adapter_loaded(self) -> bool:
        return self._mode == "fine_tuned"

    def _resolve_adapter(self) -> Optional[Path]:
        if self.settings.model_adapter_path:
            path = Path(self.settings.model_adapter_path)
            if path.exists():
                return path
        # Prefer newest adapter dir under adapters/
        adapters = self.settings.adapters_dir
        if adapters.exists():
            candidates = sorted(
                [p for p in adapters.iterdir() if p.is_dir() and (p / "adapter_config.json").exists()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
        return None

    def load(self) -> None:
        if self.settings.mock_inference or not self.settings.inference_enabled:
            self._mode = "mock"
            return

        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_id = self.settings.base_model_id
                adapter = self._resolve_adapter()
                device = self.settings.hardware_profile
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                if device == "mps" and not getattr(torch.backends, "mps", None):
                    device = "cpu"

                logger.info("Loading tokenizer/model %s on %s", model_id, device)
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                )
                if device != "cuda":
                    model = model.to(device if device != "cpu" else "cpu")

                if adapter is not None:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(adapter))
                    self._mode = "fine_tuned"
                    logger.info("Loaded LoRA adapter from %s", adapter)
                else:
                    self._mode = "base"

                model.eval()
                self._model = model
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load HF model: %s", exc)
                self._mode = "mock"
                self._model = None
                self._tokenizer = None

    def generate(
        self,
        question: str,
        *,
        context: str | None = None,
    ) -> InferenceResult:
        started = time.perf_counter()
        self.load()

        if self._mode == "mock" or self._model is None or self._tokenizer is None:
            answer = self._mock_answer(question, context)
            return InferenceResult(
                answer=wrap_answer(answer, uncertain=True),
                model_mode="mock",
                latency_ms=(time.perf_counter() - started) * 1000,
                uncertain=True,
            )

        user_content = question.strip()
        if context:
            user_content = (
                f"Use the following medical excerpts when relevant.\n\n"
                f"EVIDENCE:\n{context}\n\nQUESTION:\n{question.strip()}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        uncertain = len(text) < 20 or "not sure" in text.lower()
        return InferenceResult(
            answer=wrap_answer(text or "I am not sure how to answer that safely.", uncertain=uncertain),
            model_mode=self._mode,
            latency_ms=(time.perf_counter() - started) * 1000,
            uncertain=uncertain,
        )

    def _mock_answer(self, question: str, context: str | None) -> str:
        if context:
            return (
                f"(Mock inference — model weights not loaded.) "
                f"Based on retrieved sources, here is an educational overview related to: "
                f"{question[:200]}. Please consult a clinician for personal medical advice."
            )
        return (
            f"(Mock inference — model weights not loaded.) "
            f"Educational note about: {question[:200]}. "
            f"Load the base model or a fine-tuned adapter to get real model answers. "
            f"This is not medical advice."
        )


_engine: InferenceEngine | None = None


def get_inference_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
