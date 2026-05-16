from __future__ import annotations

import argparse
import ctypes
import logging
import os
import queue
import site
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "dictate_hold.log"


@dataclass(frozen=True)
class Settings:
    hotkey: str = "f13"
    language: str | None = None
    model_size: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    sample_rate: int = 16_000
    channels: int = 1
    min_seconds: float = 0.35
    paste_delay_seconds: float = 0.08
    beam_size: int = 5


class HoldToDictate:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.recording = threading.Event()
        self.transcribing = threading.Lock()
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.keyboard_controller = keyboard.Controller()
        self.model: WhisperModel | None = None

    def load_model(self) -> None:
        add_cuda_library_paths()
        logging.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.settings.model_size,
            self.settings.device,
            self.settings.compute_type,
        )
        self.model = WhisperModel(
            self.settings.model_size,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            download_root=str(ROOT / "models"),
        )
        logging.info("Model loaded")

    def audio_callback(self, indata: np.ndarray, _frames: int, _time_info, status) -> None:
        if status:
            logging.warning("Audio callback status: %s", status)
        if self.recording.is_set():
            self.audio_queue.put(indata.copy())

    def start_recording(self) -> None:
        if self.recording.is_set() or self.transcribing.locked():
            return
        logging.info("Recording started")
        self.frames = []
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        self.recording.set()

    def stop_recording(self) -> None:
        if not self.recording.is_set():
            return
        self.recording.clear()
        while not self.audio_queue.empty():
            self.frames.append(self.audio_queue.get_nowait())
        logging.info("Recording stopped; chunks=%s", len(self.frames))
        threading.Thread(target=self.transcribe_and_paste, daemon=True).start()

    def transcribe_and_paste(self) -> None:
        if not self.transcribing.acquire(blocking=False):
            return
        try:
            audio = self.collect_audio()
            if audio is None:
                logging.info("Audio skipped: too short")
                return

            wav_path = self.write_temp_wav(audio)
            try:
                text = self.transcribe(wav_path)
            finally:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    logging.exception("Failed to remove temp wav")

            if text:
                self.paste_text(text)
                logging.info("Pasted text: %s", text)
            else:
                logging.info("No speech recognized")
        except Exception:
            logging.exception("Transcription failed")
        finally:
            self.transcribing.release()

    def collect_audio(self) -> np.ndarray | None:
        while not self.audio_queue.empty():
            self.frames.append(self.audio_queue.get_nowait())
        if not self.frames:
            return None

        audio = np.concatenate(self.frames, axis=0)
        seconds = len(audio) / self.settings.sample_rate
        if seconds < self.settings.min_seconds:
            return None
        return np.squeeze(audio)

    def write_temp_wav(self, audio: np.ndarray) -> Path:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)

        fd, name = tempfile.mkstemp(prefix="dictate_", suffix=".wav")
        os.close(fd)
        path = Path(name)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.settings.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.settings.sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return path

    def transcribe(self, wav_path: Path) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        segments, _info = self.model.transcribe(
            str(wav_path),
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=True,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return normalize_text(text)

    def paste_text(self, text: str) -> None:
        pyperclip.copy(text)
        time.sleep(self.settings.paste_delay_seconds)
        with self.keyboard_controller.pressed(keyboard.Key.ctrl):
            self.keyboard_controller.press("v")
            self.keyboard_controller.release("v")

    def on_press(self, key) -> None:
        if key_name(key) == self.settings.hotkey:
            self.start_recording()

    def on_release(self, key) -> None:
        if key_name(key) == self.settings.hotkey:
            self.stop_recording()

    def run(self) -> None:
        self.load_model()
        self.stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            dtype="float32",
            callback=self.audio_callback,
        )
        with self.stream:
            logging.info("Ready. Hold %s to dictate.", self.settings.hotkey.upper())
            with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()


def key_name(key) -> str:
    if isinstance(key, keyboard.KeyCode):
        return (key.char or "").lower()
    name = getattr(key, "name", "")
    return str(name).lower()


def normalize_text(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    return text.rstrip(".")


def add_cuda_library_paths() -> None:
    candidate_dirs: list[Path] = []
    for base in site.getsitepackages():
        nvidia_dir = Path(base) / "nvidia"
        if nvidia_dir.exists():
            candidate_dirs.extend(path for path in nvidia_dir.glob("*/lib") if path.is_dir())

    if candidate_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(str(path) for path in candidate_dirs) + ":" + existing

    for lib_name in ("libcublas.so.12", "libcudnn.so.9"):
        for path in candidate_dirs:
            for lib_path in path.glob(f"{lib_name}*"):
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    logging.debug("Could not load CUDA library: %s", lib_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hold F13 to dictate with faster-whisper on Linux.")
    parser.add_argument("--hotkey", default="f13", help="Key to hold. Recommended: f13.")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--compute-type", default="float16", help="CTranslate2 compute type.")
    parser.add_argument("--language", default="auto", help="Language code, or auto for autodetect.")
    parser.add_argument("--download-only", action="store_true", help="Download/load the model, then exit.")
    return parser.parse_args()


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    setup_logging()
    args = parse_args()
    settings = Settings(
        hotkey=args.hotkey.lower(),
        language=None if args.language.lower() == "auto" else args.language,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    app = HoldToDictate(settings)
    if args.download_only:
        app.load_model()
        logging.info("Download-only check completed")
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
