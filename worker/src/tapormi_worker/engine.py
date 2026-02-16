from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
import wave

DEFAULT_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
DEFAULT_PARTIAL_EVERY_CHUNKS = 4
DEFAULT_MIN_PARTIAL_SECONDS = 0.4
SUPPORTED_SAMPLE_RATE = 16000

_LANGUAGE_HINTS = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}


class EngineError(RuntimeError):
    """Base class for realtime engine failures."""


class EngineUnavailableError(EngineError):
    """Raised when realtime backend dependencies are not available."""


class EngineDecodeError(EngineError):
    """Raised when the backend fails to decode buffered audio."""


@dataclass
class SessionState:
    session_id: str
    sample_rate: int = SUPPORTED_SAMPLE_RATE
    language: str = "ko"
    chunk_count: int = 0
    bytes_received: int = 0
    buffer: bytearray = field(default_factory=bytearray)
    last_partial_text: str = ""


class RealtimeAsrEngine:
    """Realtime ASR engine with pluggable backends.

    Backends:
    - auto (default): try mlx-qwen3, fallback to mock if unavailable
    - mlx_qwen3: require mlx-audio + Qwen3 model
    - mock: protocol-only fake backend
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        model_id: str | None = None,
    ) -> None:
        selected = (backend or os.getenv("TAPORMI_ASR_BACKEND", "auto")).strip().lower()
        resolved_model = model_id or os.getenv("TAPORMI_ASR_MODEL", DEFAULT_MODEL_ID)
        self._warning: str | None = None

        if selected in {"mlx_qwen3", "qwen3", "mlx", "auto"}:
            try:
                self._backend: _EngineBackend = _MlxQwen3Backend(model_id=resolved_model)
            except EngineUnavailableError as exc:
                if selected == "auto":
                    self._backend = _MockBackend()
                    self._warning = str(exc)
                else:
                    raise
        elif selected == "mock":
            self._backend = _MockBackend()
        else:
            raise ValueError(
                f"unsupported TAPORMI_ASR_BACKEND={selected!r}; "
                "expected one of: auto, mlx_qwen3, mock"
            )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def warm(self) -> bool:
        return self._backend.warm

    @property
    def warning(self) -> str | None:
        return self._warning

    def start(
        self,
        session_id: str,
        *,
        sample_rate: int = SUPPORTED_SAMPLE_RATE,
        language: str = "ko",
    ) -> SessionState:
        return self._backend.start(session_id, sample_rate=sample_rate, language=language)

    def push_chunk(self, state: SessionState, pcm_bytes: bytes) -> str | None:
        return self._backend.push_chunk(state, pcm_bytes)

    def stop(self, state: SessionState) -> str:
        return self._backend.stop(state)

    def warmup(self) -> bool:
        return self._backend.warmup()


class _EngineBackend:
    name: str
    model_name: str
    warm: bool

    def start(
        self,
        session_id: str,
        *,
        sample_rate: int = SUPPORTED_SAMPLE_RATE,
        language: str = "ko",
    ) -> SessionState:
        raise NotImplementedError

    def push_chunk(self, state: SessionState, pcm_bytes: bytes) -> str | None:
        raise NotImplementedError

    def stop(self, state: SessionState) -> str:
        raise NotImplementedError

    def warmup(self) -> bool:
        raise NotImplementedError


class _MockBackend(_EngineBackend):
    name = "mock"
    model_name = "mock"
    warm = True

    def start(
        self,
        session_id: str,
        *,
        sample_rate: int = SUPPORTED_SAMPLE_RATE,
        language: str = "ko",
    ) -> SessionState:
        return SessionState(
            session_id=session_id,
            sample_rate=sample_rate,
            language=language,
        )

    def push_chunk(self, state: SessionState, pcm_bytes: bytes) -> str | None:
        state.chunk_count += 1
        state.bytes_received += len(pcm_bytes)
        state.buffer.extend(pcm_bytes)

        if state.chunk_count % DEFAULT_PARTIAL_EVERY_CHUNKS == 0:
            seconds = state.bytes_received / (state.sample_rate * 2)
            return f"listening... {seconds:.1f}s"
        return None

    def stop(self, state: SessionState) -> str:
        seconds = state.bytes_received / (state.sample_rate * 2)
        return f"(mock) captured {seconds:.2f}s audio"

    def warmup(self) -> bool:
        self.warm = True
        return True


class _MlxQwen3Backend(_EngineBackend):
    name = "mlx_qwen3"

    def __init__(self, *, model_id: str) -> None:
        if importlib.util.find_spec("mlx_audio") is None:
            raise EngineUnavailableError(
                "mlx-audio is not installed. Run `uv sync --python 3.10` (Python>=3.10) to enable Qwen3."
            )

        self.model_name = model_id
        self.warm = False
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._partial_every_chunks = _int_env(
            "TAPORMI_PARTIAL_EVERY_CHUNKS",
            default=DEFAULT_PARTIAL_EVERY_CHUNKS,
            min_value=1,
        )
        self._min_partial_seconds = _float_env(
            "TAPORMI_MIN_PARTIAL_SECONDS",
            default=DEFAULT_MIN_PARTIAL_SECONDS,
            min_value=0.0,
        )

    def start(
        self,
        session_id: str,
        *,
        sample_rate: int = SUPPORTED_SAMPLE_RATE,
        language: str = "ko",
    ) -> SessionState:
        if sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate={sample_rate} is unsupported; only {SUPPORTED_SAMPLE_RATE}Hz mono PCM16 is supported"
            )
        return SessionState(
            session_id=session_id,
            sample_rate=sample_rate,
            language=language,
        )

    def push_chunk(self, state: SessionState, pcm_bytes: bytes) -> str | None:
        state.chunk_count += 1
        state.bytes_received += len(pcm_bytes)
        state.buffer.extend(pcm_bytes)

        if state.chunk_count % self._partial_every_chunks != 0:
            return None

        if state.bytes_received < int(state.sample_rate * 2 * self._min_partial_seconds):
            return None

        text = self._transcribe(state, stream=True)
        if text and text != state.last_partial_text:
            state.last_partial_text = text
            return text
        return None

    def stop(self, state: SessionState) -> str:
        if not state.buffer:
            return ""
        text = self._transcribe(state, stream=False)
        if text:
            state.last_partial_text = text
            return text
        return state.last_partial_text

    def warmup(self) -> bool:
        self._ensure_model()
        return self.warm

    def _transcribe(self, state: SessionState, *, stream: bool) -> str:
        model = self._ensure_model()
        language = _normalize_language(state.language)

        with tempfile.TemporaryDirectory(prefix="tapormi-asr-") as tmp_dir:
            wav_path = Path(tmp_dir) / "session.wav"
            _write_pcm16_mono_wav(
                path=wav_path,
                pcm16_bytes=bytes(state.buffer),
                sample_rate=state.sample_rate,
            )
            with self._infer_lock:
                text = self._transcribe_with_model(
                    model=model,
                    wav_path=wav_path,
                    language=language,
                    stream=stream,
                )
        return text.strip()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            errors: list[str] = []
            loaded_model: Any | None = None

            try:
                stt_module = importlib.import_module("mlx_audio.stt")
                load_fn = getattr(stt_module, "load", None)
                if callable(load_fn):
                    loaded_model = load_fn(self.model_name)
            except Exception as exc:
                errors.append(f"mlx_audio.stt.load failed: {exc}")

            if loaded_model is None:
                try:
                    stt_utils_module = importlib.import_module("mlx_audio.stt.utils")
                    load_model_fn = getattr(stt_utils_module, "load_model", None)
                    if callable(load_model_fn):
                        loaded_model = load_model_fn(self.model_name)
                except Exception as exc:
                    errors.append(f"mlx_audio.stt.utils.load_model failed: {exc}")

            if loaded_model is None:
                detail = "; ".join(errors) if errors else "no load entrypoints found"
                raise EngineUnavailableError(
                    f"failed to load model {self.model_name!r} via mlx-audio ({detail})"
                )

            self._model = loaded_model
            self.warm = True
            return loaded_model

    def _transcribe_with_model(
        self,
        *,
        model: Any,
        wav_path: Path,
        language: str,
        stream: bool,
    ) -> str:
        generate = getattr(model, "generate", None)
        if callable(generate):
            text = self._try_model_generate(
                generate=generate,
                wav_path=wav_path,
                language=language,
                stream=stream,
            )
            if text:
                return text

        # Fallback for older mlx-audio APIs.
        text = self._try_generate_transcription_function(
            model=model,
            wav_path=wav_path,
            language=language,
            stream=stream,
        )
        if text:
            return text

        raise EngineDecodeError("mlx-audio returned empty transcription")

    def _try_model_generate(
        self,
        *,
        generate: Any,
        wav_path: Path,
        language: str,
        stream: bool,
    ) -> str:
        signature = inspect.signature(generate)
        params = signature.parameters
        base_kwargs: dict[str, Any] = {}
        if "language" in params:
            base_kwargs["language"] = language
        if "verbose" in params:
            base_kwargs["verbose"] = False

        kwargs_options: list[dict[str, Any]] = []
        stream_kwargs = dict(base_kwargs)
        if stream and "stream" in params:
            stream_kwargs["stream"] = True
        kwargs_options.append(stream_kwargs)

        if "language" in stream_kwargs:
            no_language = dict(stream_kwargs)
            no_language.pop("language", None)
            kwargs_options.append(no_language)
        kwargs_options.append({})

        seen: set[tuple[tuple[str, str], ...]] = set()
        last_type_error: TypeError | None = None
        for kwargs in kwargs_options:
            key = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            try:
                result = generate(str(wav_path), **kwargs)
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:
                raise EngineDecodeError(f"mlx-audio generate failed: {exc}") from exc

            if kwargs.get("stream"):
                return _consume_stream_text(result)
            return _extract_text(result)

        if last_type_error is not None:
            raise EngineDecodeError(f"mlx-audio generate rejected arguments: {last_type_error}") from last_type_error
        return ""

    def _try_generate_transcription_function(
        self,
        *,
        model: Any,
        wav_path: Path,
        language: str,
        stream: bool,
    ) -> str:
        try:
            generate_module = importlib.import_module("mlx_audio.stt.generate")
            generate_transcription = getattr(generate_module, "generate_transcription", None)
            if not callable(generate_transcription):
                return ""
        except Exception:
            return ""

        output_path = str(wav_path.with_suffix(""))
        base_kwargs: dict[str, Any] = {
            "model": model,
            "output_path": output_path,
            "format": "txt",
        }
        if stream:
            base_kwargs["stream"] = True

        kwargs_options: list[dict[str, Any]] = []
        kwargs_options.append({**base_kwargs, "audio": str(wav_path)})
        kwargs_options.append({**base_kwargs, "audio_path": str(wav_path)})

        if language:
            kwargs_options = [{**kwargs, "language": language} for kwargs in kwargs_options] + kwargs_options

        last_type_error: TypeError | None = None
        for kwargs in kwargs_options:
            try:
                result = generate_transcription(**kwargs, verbose=False)
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:
                raise EngineDecodeError(f"mlx-audio generate_transcription failed: {exc}") from exc
            return _extract_text(result)

        if last_type_error is not None:
            raise EngineDecodeError(
                f"mlx-audio generate_transcription rejected arguments: {last_type_error}"
            ) from last_type_error
        return ""


def _write_pcm16_mono_wav(*, path: Path, pcm16_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)


def _normalize_language(language: str) -> str:
    value = (language or "ko").strip()
    if not value:
        return "Korean"
    lowered = value.lower()
    return _LANGUAGE_HINTS.get(lowered, value)


def _extract_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()

    for attr in ("text", "transcript", "content"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            return value.strip()

    if isinstance(result, dict):
        for key in ("text", "transcript", "content"):
            value = result.get(key)
            if isinstance(value, str):
                return value.strip()
        segments = result.get("segments")
        if isinstance(segments, list):
            pieces = [_extract_text(item) for item in segments]
            return " ".join(piece for piece in pieces if piece).strip()

    if isinstance(result, list):
        pieces = [_extract_text(item) for item in result]
        return " ".join(piece for piece in pieces if piece).strip()

    return str(result).strip()


def _consume_stream_text(results: Any) -> str:
    if isinstance(results, str):
        return results.strip()
    if not isinstance(results, Iterable):
        return _extract_text(results)

    text = ""
    for item in results:
        piece = _extract_text(item).strip()
        if not piece:
            continue
        if piece.startswith(text):
            text = piece
            continue
        if text.startswith(piece):
            continue
        text = f"{text} {piece}".strip()
    return text


def _int_env(name: str, *, default: int, min_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _float_env(name: str, *, default: float, min_value: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, value)
