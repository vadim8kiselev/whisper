from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel


MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "/models")
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("local-whisper")

app = FastAPI(title="Local Whisper", version="1.0.0")
model_lock = threading.Lock()
model: WhisperModel | None = None


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None


def get_model() -> WhisperModel:
    global model
    if model is None:
        logger.info(
            "Loading model=%s device=%s compute_type=%s",
            MODEL_NAME,
            DEVICE,
            COMPUTE_TYPE,
        )
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_DIR,
        )
        logger.info("Model loaded")
    return model


@app.on_event("startup")
def startup() -> None:
    get_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: Annotated[UploadFile, File(description="Audio file: wav, mp3, m4a, ogg, flac, webm, etc.")],
    language: Annotated[str, Form(description="Language code or auto")] = DEFAULT_LANGUAGE,
) -> TranscriptionResponse:
    suffix = Path(file.filename or "audio").suffix or ".audio"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    temp_path = write_temp_audio(audio_bytes, suffix)
    try:
        return run_transcription(temp_path, language)
    finally:
        temp_path.unlink(missing_ok=True)


def write_temp_audio(audio_bytes: bytes, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix="local_whisper_", suffix=suffix)
    path = Path(name)
    with os.fdopen(fd, "wb") as handle:
        handle.write(audio_bytes)
    return path


def run_transcription(audio_path: Path, language: str) -> TranscriptionResponse:
    selected_language = None if language.lower() == "auto" else language
    whisper_model = get_model()

    with model_lock:
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language=selected_language,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = normalize_text(" ".join(segment.text.strip() for segment in segments))

    return TranscriptionResponse(
        text=text,
        language=getattr(info, "language", None),
        duration=getattr(info, "duration", None),
    )


def normalize_text(text: str) -> str:
    text = " ".join(text.split())
    return text.rstrip(".")
