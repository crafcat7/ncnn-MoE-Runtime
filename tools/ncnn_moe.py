#!/usr/bin/env python3
"""Unified CLI/TUI for ncnn MoE examples."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO


# --- merged model adapters (tools/ncnn_moe_adapters.py) ---
NATIVE_TEXT_VERSION = "ncnn-moe-text-v2"


class AdapterError(RuntimeError):
    """The model or native worker cannot provide the required text interface."""


def model_asset_root(model: Path) -> Path:
    return model.parent if model.is_file() else model


def model_config_path(model: Path) -> Path:
    return model_asset_root(model) / "config.json"


def model_weight_files(model: Path) -> list[Path]:
    if model.is_file():
        if model.suffix.lower() != ".gguf":
            raise AdapterError(f"model file must be GGUF: {model}")
        split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name, re.IGNORECASE)
        if split is None:
            return [model]
        shards = [model.with_name(model.name[:split.start(2)] + f"{index:05d}" + model.name[split.end(2):])
                  for index in range(1, int(split[3]) + 1)]
        if not shards or any(not path.is_file() for path in shards):
            raise AdapterError(f"GGUF split package is incomplete: {model}")
        return shards
    files = [path for path in model.iterdir() if path.is_file()]
    safetensors = sorted(path for path in files if path.suffix == ".safetensors")
    if safetensors:
        return safetensors
    return sorted(path for path in files if path.suffix.lower() == ".gguf")


def _gguf_prefix_size(path: Path) -> int:
    """Find metadata and tensor descriptors without visiting weight payloads."""
    with path.open("rb") as stream:
        size = path.stat().st_size

        def take(count: int) -> bytes:
            if count < 0 or count > size - stream.tell():
                raise AdapterError(f"truncated GGUF header: {path}")
            value = stream.read(count)
            if len(value) != count:
                raise AdapterError(f"truncated GGUF header: {path}")
            return value

        def skip(count: int) -> None:
            if count < 0 or count > size - stream.tell():
                raise AdapterError(f"truncated GGUF header: {path}")
            stream.seek(count, 1)

        def string() -> None:
            skip(struct.unpack("<Q", take(8))[0])

        scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def value(kind: int, depth: int = 0) -> None:
            if kind == 8:
                string()
            elif kind in scalar_sizes:
                skip(scalar_sizes[kind])
            elif kind == 9 and depth < 8:
                element, count = struct.unpack("<IQ", take(12))
                if element in scalar_sizes:
                    skip(scalar_sizes[element] * count)
                else:
                    if count > size - stream.tell():
                        raise AdapterError(f"invalid GGUF array: {path}")
                    for _ in range(count):
                        value(element, depth + 1)
            else:
                raise AdapterError(f"unsupported GGUF metadata type: {kind}")

        magic, version, tensors, metadata = struct.unpack("<4sIQQ", take(24))
        if magic != b"GGUF" or version not in (2, 3) or metadata > size // 12 or tensors > size // 24:
            raise AdapterError(f"invalid GGUF header: {path}")
        for _ in range(metadata):
            string()
            value(struct.unpack("<I", take(4))[0])
        for _ in range(tensors):
            string()
            dimensions = struct.unpack("<I", take(4))[0]
            if dimensions > 8:
                raise AdapterError(f"invalid GGUF tensor dimensions: {path}")
            skip(dimensions * 8 + 12)
        return stream.tell()


def weight_fingerprint_samples(path: Path) -> tuple[bytes, bytes]:
    size = path.stat().st_size
    prefix = _gguf_prefix_size(path) if path.suffix.lower() == ".gguf" else 4096
    if prefix > 256 * 1024 * 1024:
        raise AdapterError(f"GGUF metadata exceeds the fingerprint limit: {path}")
    with path.open("rb") as stream:
        head = stream.read(prefix)
        tail = b""
        if size > prefix:
            stream.seek(-min(size, 4096), 2)
            tail = stream.read(4096)
    return head, tail


@dataclass(frozen=True)
class Completion:
    reasoning: str
    answer: str

    @property
    def text(self) -> str:
        return self.answer or self.reasoning


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AdapterError(f"config.json is missing from: {path.parent}") from error
    except json.JSONDecodeError as error:
        raise AdapterError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdapterError(f"model config must be a JSON object: {path}")
    return value


def _split_thinking(text: str, thinking: bool) -> Completion:
    """Split the first thinking boundary, accepting an optional opening tag."""
    if not thinking:
        return Completion("", text.strip())
    text = text.lstrip().removeprefix("<think>")
    marker = "</think>"
    if marker in text:
        reasoning, answer = text.split(marker, 1)
        return Completion(reasoning.strip(), answer.lstrip("\r\n"))
    return Completion(text.strip(), "")


def _decode_deepseek(text: str, thinking: bool) -> Completion:
    if not thinking:
        return Completion("", text.strip())
    split = _split_thinking(text, thinking)
    return Completion(split.reasoning.strip(), split.answer.strip())


class ModelAdapter:
    """A small model profile; tokenization and decoding stay in the worker."""

    def __init__(
        self,
        model: Path,
        *,
        thinking: bool = True,
        native_stop_tokens: list[int] | None = None,
        native_model_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = Path(model).resolve()
        self._asset_root = model_asset_root(self.model)
        self._config_path = model_config_path(self.model)
        self._weight_files = model_weight_files(self.model)
        self._gguf = bool(self._weight_files and self._weight_files[0].suffix.lower() == ".gguf")
        self.config = _load_json(self._config_path) if self._config_path.is_file() else {}
        self._native_model_info = native_model_info or {}
        if not self.config and not self._gguf:
            raise AdapterError(f"config.json is missing from: {self._asset_root}")
        config_type = self.config.get("model_type")
        native_type = self._native_model_info.get("model_type")
        # GGUF metadata is authoritative even when an unrelated HF config is
        # present beside a self-contained package.
        resolved_type = (native_type or config_type) if self._gguf else (config_type or native_type)
        self.model_type = str(resolved_type or "unknown").lower()
        self.name = self._model_name(self.model_type)
        # PhiMoE is a plain instruct model without a reasoning channel.
        self.thinking = bool(thinking) and self.name != "phimoe"
        self._stop_tokens = self._validate_stop_tokens(native_stop_tokens)
        self.model_fingerprint, self.legacy_model_fingerprints = self._fingerprints()

    @staticmethod
    def _model_name(model_type: str) -> str:
        if model_type in {"gpt_oss", "gpt-oss", "gpt_oss_moe"}:
            return "gpt-oss"
        if model_type in {"deepseek_v4", "deepseek-v4", "deepseek_v4_flash", "deepseek4"}:
            return "deepseek-v4"
        if model_type in {"qwen4_exp", "qwen4_exp_text", "qwen4exp"}:
            return "qwen3.8"
        if model_type == "qwen3":
            return "qwen3"
        if model_type in {"qwen3_5_moe", "qwen3_6", "qwen3.6", "qwen3_5", "qwen35", "qwen35moe"}:
            return "qwen3.6"
        if model_type == "phimoe":
            return "phimoe"
        if model_type == "bailing_hybrid":
            return "ling"
        raise AdapterError(f"unsupported model_type: {model_type or '<missing>'}")

    @staticmethod
    def _validate_stop_tokens(value: list[int] | None) -> list[int]:
        if value is None:
            return []
        if not isinstance(value, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in value
        ):
            raise AdapterError("worker native stop-token metadata is invalid")
        return list(dict.fromkeys(value))

    @property
    def stop_tokens(self) -> list[int]:
        return list(self._stop_tokens)

    @property
    def context_limit(self) -> int | None:
        native_limit = self._native_model_info.get("max_context_tokens")
        valid_native_limit = isinstance(native_limit, int) and not isinstance(native_limit, bool) and native_limit > 0
        if self._gguf and valid_native_limit:
            return native_limit
        configs = [self.config]
        text_config = self.config.get("text_config")
        if isinstance(text_config, dict):
            configs.append(text_config)
        for config in configs:
            for key in (
                "max_position_embeddings",
                "max_sequence_length",
                "max_seq_len",
                "max_context_length",
            ):
                value = config.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return native_limit if valid_native_limit else None

    def _fingerprints(self) -> tuple[str, tuple[str, ...]]:
        """Hash current and pre-v2 identities while scanning model weights once."""
        legacy = hashlib.sha256()
        legacy.update((self.model_type + "\0").encode("utf-8"))
        config_path = self._config_path
        if config_path.is_file():
            legacy.update(config_path.read_bytes())

        current = hashlib.sha256()
        current.update(
            (NATIVE_TEXT_VERSION + "\0" + self.name + "\0" + self.model_type + "\0").encode("utf-8")
        )
        model_files = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        )
        for name in model_files:
            path = self._config_path if name == "config.json" else self._asset_root / name
            current.update(name.encode("utf-8") + b"\0")
            if not path.is_file():
                current.update(b"\0")
                continue
            current.update(b"\1")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    current.update(block)

        for name in ("tokenizer.json", "chat_template.jinja", "encoding/encoding_dsv4.py"):
            path = self._asset_root / name
            if path.is_file():
                legacy.update(name.encode("utf-8"))
                legacy.update(str(path.stat().st_size).encode("ascii"))

        for path in self._weight_files:
            relative = path.name if self._gguf else path.relative_to(self._asset_root).as_posix()
            size = path.stat().st_size
            legacy.update(relative.encode("utf-8"))
            legacy.update(str(size).encode("ascii"))
            current.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0")
            head, tail = weight_fingerprint_samples(path)
            legacy.update(head)
            if tail:
                legacy.update(tail)
            current.update(head)
            if tail:
                current.update(tail)

        legacy_base = legacy.hexdigest()[:20]
        prior_fingerprints = [legacy_base]
        if self.name.startswith("qwen") and not self._gguf:
            old_qwen = hashlib.sha256()
            old_qwen.update(legacy_base.encode("ascii"))
            old_qwen.update(b"ncnn-moe-tokenizer-json-v1\0")
            if self.model_type == "qwen3_5_moe":
                old_qwen.update(b"ncnn-moe-qwen3.6-nfc9-regex16-v1\0")
            for name in model_files:
                path = self._asset_root / name
                old_qwen.update(name.encode("utf-8") + b"\0")
                if not path.is_file():
                    old_qwen.update(b"\0")
                    continue
                old_qwen.update(b"\1")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        old_qwen.update(block)
            prior_fingerprints.insert(0, old_qwen.hexdigest()[:20])
        return current.hexdigest()[:20], () if self._gguf else tuple(prior_fingerprints)

    def validate(self) -> None:
        if not self._config_path.is_file() and not self._gguf:
            raise AdapterError(f"config.json is missing from: {self._asset_root}")

    def decode_completion_text(self, text: str) -> Completion:
        if self.name == "gpt-oss":
            raise AdapterError(
                "GPT-OSS completion decoding requires native token events to preserve Harmony boundaries"
            )
        if self.name == "phimoe":
            return Completion("", text.strip())
        if self.name == "deepseek-v4":
            return _decode_deepseek(text, self.thinking)
        return _split_thinking(text, self.thinking)

    @staticmethod
    def visible_completion(completion: Completion, *, final_only: bool = False) -> str:
        if final_only:
            return completion.answer
        if completion.answer:
            if completion.reasoning:
                return f"[reasoning]\n{completion.reasoning}\n[answer]\n{completion.answer}"
            return f"[answer]\n{completion.answer}"
        return f"[reasoning]\n{completion.reasoning}" if completion.reasoning else ""


def create_adapter(model: Path, **options: Any) -> ModelAdapter:
    adapter = ModelAdapter(model, **options)
    adapter.validate()
    return adapter


# --- merged worker protocol (tools/ncnn_moe_protocol.py) ---
class WorkerError(RuntimeError):
    """A protocol, process, or native-runtime error."""

    def __init__(self, message: str, event: dict[str, Any] | None = None):
        super().__init__(message)
        self.event = event or {}


EventCallback = Callable[[dict[str, Any]], None]


class WorkerClient:
    """Strict UTF-8 JSONL client; generation runs asynchronously in C++."""

    def __init__(
        self,
        worker: Path,
        model: Path,
        runtime_args: Iterable[str] = (),
        *,
        verbose: bool = False,
    ) -> None:
        self.worker = Path(worker).resolve()
        self.model = Path(model).resolve()
        if not self.worker.is_file():
            raise WorkerError(f"worker does not exist: {self.worker}")
        if not self.model.is_dir() and not (self.model.is_file() and self.model.suffix.lower() == ".gguf"):
            raise WorkerError(f"model directory or GGUF file does not exist: {self.model}")
        self._process = subprocess.Popen(
            [str(self.worker), str(self.model), *map(str, runtime_args)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        self._closed = False
        try:
            ready = self._read_event()
            if ready.get("event") == "error":
                raise WorkerError(ready.get("message", "worker failed to load the model"), ready)
            if ready.get("event") != "ready":
                raise WorkerError(f"worker did not send ready event: {ready}", ready)
            self.ready = ready
        except BaseException:
            self._terminate()
            raise

    @property
    def process(self) -> subprocess.Popen[str]:
        return self._process

    def _send(self, payload: dict[str, Any]) -> None:
        if self._closed or self._process.stdin is None:
            raise WorkerError("worker is closed")
        try:
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except UnicodeEncodeError as error:
            raise WorkerError("worker request must contain valid UTF-8 text") from error
        self._process.stdin.flush()

    def _read_event(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise WorkerError("worker stdout is unavailable")
        try:
            line = self._process.stdout.readline()
        except UnicodeDecodeError as error:
            raise WorkerError("worker emitted invalid UTF-8") from error
        if not line:
            return_code = self._process.poll()
            raise WorkerError(f"worker exited before sending an event (return code {return_code})")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkerError(f"worker emitted invalid JSONL: {line.rstrip()!r}") from error
        if not isinstance(event, dict):
            raise WorkerError(f"worker event must be a JSON object: {event!r}")
        return event

    @staticmethod
    def _raise_for_error(event: dict[str, Any]) -> None:
        if event.get("event") == "error":
            code = event.get("code", "worker_error")
            message = event.get("message", "native worker error")
            raise WorkerError(f"{code}: {message}", event)

    def request(
        self,
        payload: dict[str, Any],
        *,
        expected: str | tuple[str, ...],
    ) -> dict[str, Any]:
        self._send(payload)
        event = self._read_event()
        self._raise_for_error(event)
        expected_events = (expected,) if isinstance(expected, str) else expected
        if event.get("event") not in expected_events:
            raise WorkerError(
                f"expected {expected_events}, received {event.get('event')!r}",
                event,
            )
        return event

    def create_session(
        self,
        session_id: str,
        *,
        seed: int = 0,
        prefill_chunk_size: int = 512,
        enable_speculative_context: bool = True,
    ) -> dict[str, Any]:
        return self.request(
            {
                "op": "create_session",
                "session_id": session_id,
                "seed": seed,
                "prefill_chunk_size": prefill_chunk_size,
                "enable_speculative_context": enable_speculative_context,
            },
            expected="session_created",
        )

    def generate(
        self,
        session_id: str,
        prompt_tokens: list[int] | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool | None = None,
        context_tokens: int | None = None,
        request_id: str = "generate",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        stop_tokens: list[int] | None = None,
        enable_speculative: bool = True,
        speculative_confidence: float = 0.5,
        speculative_max_draft: int = 0,
        metrics_enabled: bool = True,
        metrics_interval_ms: int = 1000,
        on_event: EventCallback | None = None,
    ) -> tuple[dict[str, Any], list[int]]:
        if (prompt_tokens is None) == (messages is None):
            raise WorkerError("generate requires exactly one of prompt_tokens or messages")
        payload: dict[str, Any] = {
            "op": "generate",
            "request_id": request_id,
            "session_id": session_id,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "enable_speculative": enable_speculative,
            "speculative_confidence": speculative_confidence,
            "speculative_max_draft": speculative_max_draft,
            "metrics_enabled": metrics_enabled,
            "metrics_interval_ms": metrics_interval_ms,
        }
        if messages is None:
            payload["prompt_tokens"] = prompt_tokens
            if stop_tokens is not None:
                payload["stop_tokens"] = stop_tokens
        else:
            payload["messages"] = messages
            if enable_thinking is not None:
                payload["enable_thinking"] = enable_thinking
            if context_tokens is not None:
                payload["context_tokens"] = context_tokens
        self._send(payload)
        tokens: list[int] = []
        callback_error: Exception | None = None
        try:
            while True:
                event = self._read_event()
                self._raise_for_error(event)
                if on_event is not None and callback_error is None:
                    try:
                        on_event(event)
                    except Exception as error:
                        callback_error = error
                if event.get("event") == "token":
                    tokens.append(int(event["token_id"]))
                elif event.get("event") == "done":
                    if callback_error is not None:
                        raise callback_error
                    return event, tokens
        except KeyboardInterrupt as error:
            # A synchronous client cannot call cancel() while it is reading a
            # generation. Send the same protocol operation directly, drain the
            # native completion, then let the REPL report the interruption.
            try:
                self._send({"op": "cancel", "request_id": request_id})
                while True:
                    event = self._read_event()
                    self._raise_for_error(event)
                    if on_event is not None and callback_error is None:
                        try:
                            on_event(event)
                        except Exception as callback_exception:
                            callback_error = callback_exception
                    if event.get("event") == "token":
                        tokens.append(int(event["token_id"]))
                    elif event.get("event") == "done":
                        break
            except (OSError, WorkerError):
                pass
            raise WorkerError("generation cancelled by user") from error

    def reset(self, session_id: str) -> dict[str, Any]:
        return self.request(
            {"op": "reset", "session_id": session_id}, expected="reset"
        )

    def compact(
        self,
        session_id: str,
        replay_tokens: list[int] | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool | None = None,
        context_tokens: int | None = None,
    ) -> dict[str, Any]:
        if (replay_tokens is None) == (messages is None):
            raise WorkerError("compact requires exactly one of replay_tokens or messages")
        payload: dict[str, Any] = {"op": "compact", "session_id": session_id}
        if messages is None:
            payload["replay_tokens"] = replay_tokens
        else:
            payload["messages"] = messages
            if enable_thinking is not None:
                payload["enable_thinking"] = enable_thinking
            if context_tokens is not None:
                payload["context_tokens"] = context_tokens
        return self.request(
            payload,
            expected="compacted",
        )

    def stats(
        self,
        session_id: str,
        *,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": "stats", "session_id": session_id}
        if messages is not None:
            payload["messages"] = messages
            if enable_thinking is not None:
                payload["enable_thinking"] = enable_thinking
        return self.request(
            payload, expected="stats"
        )

    def cancel(self, request_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"op": "cancel"}
        if request_id:
            payload["request_id"] = request_id
        return self.request(payload, expected="cancel_requested")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None and self._process.stdin is not None:
                self._process.stdin.write('{"op":"shutdown"}\n')
                self._process.stdin.flush()
                while True:
                    event = self._read_event()
                    if event.get("event") in {"shutdown", "error"}:
                        break
        except (BrokenPipeError, OSError, WorkerError):
            pass
        finally:
            self._terminate()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# --- merged persistent state (tools/ncnn_moe_state.py) ---
def state_root() -> Path:
    """Return the project-local state root unless explicitly overridden."""
    override = os.environ.get("NCNN_MOE_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "CMakeLists.txt").is_file():
        return source_root / ".ncnn-moe"
    return Path.cwd().resolve() / ".ncnn-moe"


def now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return value[:80] or "session"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def hardware_fingerprint(ready: dict[str, Any]) -> str:
    capabilities = ready.get("capabilities", {})
    stable = {
        "physical_memory_bytes": capabilities.get("physical_memory_bytes"),
        "logical_cpu_count": capabilities.get("logical_cpu_count"),
        "physical_cpu_core_count": capabilities.get("physical_cpu_core_count"),
        "cpu_isa": capabilities.get("cpu_isa"),
        "vulkan_devices": [
            {
                "vendor_id": device.get("vendor_id"),
                "device_id": device.get("device_id"),
                "name": device.get("name"),
                "heap_budget_bytes": device.get("heap_budget_bytes"),
            }
            for device in capabilities.get("vulkan_devices", [])
        ],
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or state_root()).resolve()
        self.sessions_dir = self.root / "sessions"
        self.config_path = self.root / "config.json"
        self.profiles_path = self.root / "tuning_profiles.json"

    def user_config(self) -> dict[str, Any]:
        value = read_json(self.config_path, {})
        return value if isinstance(value, dict) else {}

    def save_user_config(self, value: dict[str, Any]) -> None:
        write_json(self.config_path, value)

    def profiles(self) -> dict[str, Any]:
        value = read_json(self.profiles_path, {})
        return value if isinstance(value, dict) else {}

    def save_profile(self, key: str, value: dict[str, Any]) -> None:
        profiles = self.profiles()
        profiles[key] = value
        write_json(self.profiles_path, profiles)

    def profile(self, key: str) -> dict[str, Any] | None:
        value = self.profiles().get(key)
        return value if isinstance(value, dict) else None

    def new_id(self) -> str:
        return _datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{safe_name(session_id)}.json"

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        value = read_json(path, None)
        if not isinstance(value, dict):
            return None
        return value

    def save(self, record: dict[str, Any]) -> None:
        session_id = str(record.get("id", ""))
        if not session_id:
            raise ValueError("session record requires id")
        record = dict(record)
        record["updated_at"] = now_iso()
        write_json(self.path_for(session_id), record)

    def delete(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def rename(self, session_id: str, title: str) -> dict[str, Any]:
        record = self.load(session_id)
        if record is None:
            raise ValueError(f"session does not exist: {session_id}")
        record["title"] = title.strip() or session_id
        self.save(record)
        return record

    def list(self) -> list[dict[str, Any]]:
        if not self.sessions_dir.is_dir():
            return []
        records = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            value = read_json(path, None)
            if isinstance(value, dict) and value.get("id"):
                records.append(value)
        return records


def profile_key(
    *,
    hardware: str,
    model: str,
    context_tokens: int,
    session_count: int,
) -> str:
    return f"{hardware}:{model}:{context_tokens}:{session_count}"


_EXPERT_IO_KEYS = ("mmap_experts", "buffered_expert_io")


def _validate_expert_io_settings(settings: dict[str, Any]) -> None:
    if sum(bool(settings.get(key)) for key in _EXPERT_IO_KEYS) > 1:
        raise ValueError("Expert mmap and buffered I/O modes are mutually exclusive")


def runtime_args_from_settings(settings: dict[str, Any]) -> list[str]:
    _validate_expert_io_settings(settings)
    args: list[str] = []
    backend = settings.get("backend")
    if backend and backend != "auto":
        backend_flags = {
            "cpu": "--cpu",
            "hybrid": "--hybrid",
        }
        backend_flag = backend_flags.get(backend) if isinstance(backend, str) else None
        if backend_flag is None:
            raise ValueError(f"unknown backend in runtime settings: {backend}")
        args.append(backend_flag)
    cpu_packed_weights = settings.get("cpu_packed_weights")
    if cpu_packed_weights is not None:
        if cpu_packed_weights not in ("off", "on"):
            raise ValueError(
                "CPU packed weights in runtime settings must be off or on"
            )
        args.extend(["--cpu-packed-weights", cpu_packed_weights])
    for key, option in (
        ("host_memory_mb", "--host-memory-mb"),
        ("expert_cache_mb", "--expert-cache-mb"),
        ("expert_gpu_cache_mb", "--expert-gpu-cache-mb"),
        ("expert_gpu_victim_cache_mb", "--expert-gpu-victim-cache-mb"),
        ("expert_io_workers", "--expert-io-workers"),
        ("expert_memory", "--expert-memory"),
        ("expected_concurrency", "--expected-concurrency"),
    ):
        value = settings.get(key)
        if value is not None and value != 0:
            args.extend([option, str(value)])
    if settings.get("vulkan_device") is not None:
        args.extend(["--vulkan-device", str(settings["vulkan_device"])])
    if settings.get("vulkan_devices"):
        devices = settings["vulkan_devices"]
        if isinstance(devices, (list, tuple)):
            devices = ",".join(str(device) for device in devices)
        args.extend(["--vulkan-devices", str(devices)])
    for key, option in (
        ("mmap_experts", "--mmap-experts"),
        ("buffered_expert_io", "--buffered-expert-io"),
        ("disable_gpu_victim_execution", "--disable-gpu-victim-execution"),
        ("disable_vulkan_joint_cache_eviction", "--disable-vulkan-joint-cache-eviction"),
        ("disable_vulkan_mxfp4_row_tile", "--disable-vulkan-mxfp4-row-tile"),
        ("disable_vulkan_expert_direct_staging", "--disable-vulkan-expert-direct-staging"),
        ("disable_vulkan_shared_expert_overlap", "--disable-vulkan-shared-expert-overlap"),
    ):
        if settings.get(key):
            args.append(option)
    return args


def merge_runtime_settings(
    *,
    cli: dict[str, Any],
    session: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    user: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in (user or {}, profile or {}, session or {}, cli):
        io_settings = {key: source[key] for key in _EXPERT_IO_KEYS if source.get(key) is not None}
        _validate_expert_io_settings(io_settings)
        if io_settings:
            # A higher-priority layer selects the whole I/O mode. Explicit
            # false values select Auto; omitted/None preferences inherit.
            for key in _EXPERT_IO_KEYS:
                result.pop(key, None)
        for key, value in source.items():
            if value is not None:
                result[key] = value
    # Ignore removed experimental options in saved sessions/configs/profiles.
    for key in (
        "release_vulkan_dense_host",
        "no_speculative",
        "direct_expert_io",
        "gpu_decode_experts",
        "profiling",
        "vulkan_expert_concurrent_commands",
        "disable_vulkan_expert_concurrent_commands",
    ):
        result.pop(key, None)
    result.setdefault("backend", "auto")
    return result


def configure_standard_streams() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # Keep Python's console/explicit codec; default redirected text to UTF-8.
            encoding = stream.encoding if stream.isatty() or os.environ.get("PYTHONIOENCODING") else "utf-8"
            errors = "strict" if stream is sys.stdin else "backslashreplace"
            reconfigure(encoding=encoding, errors=errors)


def _optional_rich() -> tuple[Any, Any]:
    try:
        from rich.console import Console
        from rich.panel import Panel

        return Console, Panel
    except ImportError:
        return None, None


def _optional_prompt_session() -> Any:
    try:
        from prompt_toolkit import PromptSession

        return PromptSession
    except ImportError:
        return None


def _format_gpu_utilization(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A (unavailable)"
    utilization = value.get("utilization_percent")
    if utilization is not None:
        return f"{float(utilization):.1f}%"
    reason = value.get("reason") or "unavailable"
    return f"N/A ({reason})"


def _format_duration_microseconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    microseconds = float(value)
    if microseconds >= 1_000_000:
        return f"{microseconds / 1_000_000:.2f} s"
    return f"{microseconds / 1_000:.2f} ms"


def _format_bytes_gb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value) / 1_000_000_000:.2f} GB"


def _format_metric_count(value: Any, *, available: bool = True) -> str:
    if not available or not isinstance(value, (int, float)):
        return "N/A"
    return f"{int(value):,}"


def _format_rate(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value):.2f}"


def _numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _prompt_rate(metrics: dict[str, Any]) -> float | None:
    for name in ("prompt_tok_per_second", "prompt_tokens_per_second"):
        value = _numeric_metric(metrics.get(name))
        if value is not None:
            return value
    prompt_elapsed_microseconds = _numeric_metric(metrics.get("prompt_elapsed_microseconds"))
    input_tokens = _numeric_metric(metrics.get("input_tokens"))
    if prompt_elapsed_microseconds is not None and prompt_elapsed_microseconds > 0.0 and input_tokens:
        return input_tokens * 1_000_000.0 / prompt_elapsed_microseconds
    ttft_microseconds = _numeric_metric(metrics.get("ttft_microseconds"))
    if ttft_microseconds is not None and ttft_microseconds > 0.0 and input_tokens:
        return input_tokens * 1_000_000.0 / ttft_microseconds
    return None


def _generation_rate(metrics: dict[str, Any]) -> float | None:
    for name in (
        "generation_tok_per_second",
        "generation_tokens_per_second",
        "decode_tok_per_second",
        "tokens_per_second",
        "token_per_second",
    ):
        value = _numeric_metric(metrics.get(name))
        if value is not None:
            return value
    tpot_microseconds = _numeric_metric(metrics.get("tpot_microseconds"))
    if tpot_microseconds is not None and tpot_microseconds > 0.0:
        return 1_000_000.0 / tpot_microseconds
    return None


def _format_runtime_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return "metrics\n  unavailable"
    expert = metrics.get("expert", {})
    if not isinstance(expert, dict):
        expert = {}
    cache_hit_rate = expert.get("cache_hit_rate")
    if not isinstance(cache_hit_rate, (int, float)):
        cache_hit = expert.get("cache_hit")
        cache_miss = expert.get("cache_miss")
        if isinstance(cache_hit, (int, float)) and isinstance(cache_miss, (int, float)) and cache_hit + cache_miss:
            cache_hit_rate = cache_hit / (cache_hit + cache_miss)
    gpu = metrics.get("gpu", {})
    if not isinstance(gpu, dict):
        gpu = {}
    gpu_device = metrics.get("gpu_device", {})
    if not isinstance(gpu_device, dict) or not gpu_device:
        gpu_device = metrics.get("gpu", {}) if isinstance(metrics.get("gpu"), dict) else {}
    process = metrics.get("process", {})
    gpu_available = bool(gpu.get("available", gpu_device.get("active", False)))
    process_cpu = process.get("cpu_percent") if isinstance(process, dict) else None
    process_cpu_text = (
        f"{float(process_cpu):.1f}%"
        if isinstance(process_cpu, (int, float))
        else "N/A"
    )
    cache_hit_text = _format_metric_count(expert.get("cache_hit"))
    if isinstance(cache_hit_rate, (int, float)):
        cache_hit_text += f" ({float(cache_hit_rate) * 100:.1f}%)"
    lines = list(
        (
            "metrics",
            "  Prompt: "
            f"{_format_rate(_prompt_rate(metrics))} t/s | Generation: "
            f"{_format_rate(_generation_rate(metrics))} t/s"
            " · TTFT "
            f"{_format_duration_microseconds(metrics.get('ttft_microseconds'))}"
            " · TPOT "
            f"{_format_duration_microseconds(metrics.get('tpot_microseconds'))}",
            "  Expert: cache hit "
            f"{cache_hit_text}"
            " · cache miss "
            f"{_format_metric_count(expert.get('cache_miss'))}"
            " · IO "
            f"{_format_bytes_gb(expert.get('io_bytes'))}",
            "  GPU Expert: cache hit "
            f"{_format_metric_count(expert.get('gpu_cache_hit'))}"
            " / miss "
            f"{_format_metric_count(expert.get('gpu_cache_miss'))}"
            " / exec batches "
            f"{_format_metric_count(expert.get('gpu_executions'))}"
            " / resident "
            f"{_format_bytes_gb(expert.get('gpu_cache_resident_bytes'))}"
            " / dropped "
            f"{_format_metric_count(expert.get('gpu_cache_dropped_admissions'))}",
            "  Expert routes: total "
            f"{_format_metric_count(expert.get('assignments'))}"
            " / GPU "
            f"{_format_metric_count(expert.get('gpu_routes'))}"
            " / CPU "
            f"{_format_metric_count(expert.get('cpu_routes'))}"
            " · batches total/CPU "
            f"{_format_metric_count(expert.get('batches'))}"
            "/"
            f"{_format_metric_count(expert.get('cpu_batches'))}"
            " · CPU fallbacks "
            f"{_format_metric_count(expert.get('cpu_fallbacks'))}",
            f"  Process CPU: {process_cpu_text}",
            "  GPU: compute submit "
            f"{_format_metric_count(gpu.get('submit_count'), available=gpu_available)}"
            " · utilization "
            f"{_format_gpu_utilization(gpu_device)}"
            " / attention "
            f"{_format_metric_count(gpu.get('attention_blocks'), available=gpu_available)}"
            " / linear "
            f"{_format_metric_count(gpu.get('linear_dispatches'), available=gpu_available)}"
            " / gated-delta "
            f"{_format_metric_count(gpu.get('gated_delta_fusions'), available=gpu_available)}"
            " / upload/download "
            f"{_format_metric_count(gpu.get('batch_uploads'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('batch_downloads'), available=gpu_available)}"
            " / qkv-rope/device-rope/ring "
            f"{_format_metric_count(gpu.get('attention_qkv_rope_fusions'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('attention_device_rope_fusions'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('attention_qkv_ring_fusions'), available=gpu_available)}"
            " / kv-materialize "
            f"{_format_metric_count(gpu.get('attention_cache_materializations'), available=gpu_available)}"
            " / attention-cpu-fallback "
            f"{_format_metric_count(gpu.get('attention_cpu_fallbacks'), available=gpu_available)}",
            "  Device blocks: Qwen "
            f"{_format_metric_count(gpu.get('qwen_layers'), available=gpu_available)}"
            " / gated-residual "
            f"{_format_metric_count(gpu.get('gated_residual_blocks'), available=gpu_available)}"
            " / PLE "
            f"{_format_metric_count(gpu.get('ple_blocks'), available=gpu_available)}",
        )
    )

    if "weight_transfer_submissions" in gpu:
        lines.append(
            "  GPU weight transfer: copy/visibility submit "
            f"{_format_metric_count(gpu.get('weight_transfer_submissions'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('weight_visibility_submissions'), available=gpu_available)}"
            " · bytes "
            f"{_format_bytes_gb(gpu.get('weight_upload_bytes'))}"
            " · fence waits "
            f"{_format_metric_count(gpu.get('weight_transfer_fence_waits'), available=gpu_available)}"
        )
    if gpu.get("resident_graph_submissions", 0):
        lines.append(
            "  GPU resident graph: submit "
            f"{_format_metric_count(gpu.get('resident_graph_submissions'), available=gpu_available)}"
            " · activation H2D/D2H "
            f"{_format_metric_count(gpu.get('resident_graph_activation_upload_bytes'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('resident_graph_activation_download_bytes'), available=gpu_available)} B"
            " · route D2H "
            f"{_format_metric_count(gpu.get('resident_graph_route_download_bytes'), available=gpu_available)} B"
        )
    return "\n".join(lines)


def _add_model_position(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_pos", nargs="?", help="Model directory")
    parser.add_argument("--model", dest="model_opt", help="Model directory or GGUF file")


def _add_worker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worker",
        help="Override the default worker under build-ncnn",
    )
    parser.add_argument("--config-dir", help="Override the persistent configuration directory")
    parser.add_argument("--backend", choices=("auto", "cpu", "hybrid"))
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--cpu", dest="backend", action="store_const", const="cpu")
    backend_group.add_argument("--hybrid", dest="backend", action="store_const", const="hybrid")
    parser.add_argument("--expert-memory", choices=("auto", "eager", "on-demand"))
    parser.add_argument(
        "--cpu-packed-weights",
        choices=("off", "on"),
        help="Enable the optional CPU Expert weight repack (default: off)",
    )
    parser.add_argument(
        "--cpu-qnk-q8",
        choices=("off", "on"),
        help=(
            "Quantize CPU activations to int8 for block-quantized projections, "
            "cutting activation traffic fourfold at some accuracy cost (default: off)"
        ),
    )
    parser.add_argument("--host-memory-mb", type=int)
    parser.add_argument("--expert-cache-mb", type=int)
    parser.add_argument("--expert-gpu-cache-mb", type=int)
    parser.add_argument("--expert-gpu-victim-cache-mb", type=int)
    parser.add_argument("--expert-io-workers", type=int)
    parser.add_argument("--vulkan-device", type=int)
    parser.add_argument("--vulkan-devices", help="Comma-separated Vulkan device indices")
    parser.add_argument("--expected-concurrency", type=int)
    expert_io_group = parser.add_mutually_exclusive_group()
    expert_io_group.add_argument("--mmap-experts", action="store_true", default=None)
    expert_io_group.add_argument("--buffered-expert-io", action="store_true", default=None)
    parser.add_argument("--disable-gpu-victim-execution", action="store_true", default=None)
    parser.add_argument("--disable-vulkan-joint-cache-eviction", action="store_true", default=None)
    parser.add_argument("--disable-vulkan-mxfp4-row-tile", action="store_true", default=None)
    parser.add_argument("--disable-vulkan-expert-direct-staging", action="store_true", default=None)
    parser.add_argument("--disable-vulkan-shared-expert-overlap", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true")


def _add_metrics_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics-interval-ms",
        type=int,
        default=1000,
        help="Periodic metrics event interval; 0 disables the metrics trace",
    )
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--metrics",
        dest="metrics_enabled",
        action="store_true",
        help="Enable periodic metrics events",
    )
    metrics_group.add_argument(
        "--no-metrics",
        dest="metrics_enabled",
        action="store_false",
        help="Disable periodic metrics events while retaining final statistics",
    )
    parser.set_defaults(metrics_enabled=False)


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help=(
            "Maximum generated tokens, including reasoning tokens (default: 0); "
            "0 generates until a stop token or the remaining context limit"
        ),
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speculative-confidence", type=float, default=0.5)
    parser.add_argument("--speculative-max-draft", type=int, default=0)
    _add_metrics_options(parser)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Prefill tokens per chunk; 0 processes all remaining input as one batch (unchunked)",
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", help="Stream generated text (default)")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false", help="Print the completed response only")
    parser.set_defaults(stream=True)
    parser.add_argument("--stream-final-only", action="store_true")
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--show-reasoning", dest="show_reasoning", action="store_true", help="Show reasoning (default)")
    reasoning_group.add_argument("--hide-reasoning", dest="show_reasoning", action="store_false", help="Hide reasoning from the display")
    parser.set_defaults(show_reasoning=True)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ncnn-moe",
        description="Auto-configured one-shot generation, chat sessions, inspection, and tuning for ncnn MoE models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one prompt")
    _add_model_position(run)
    run.add_argument("prompt_pos", nargs="?", help="User prompt")
    run.add_argument("--prompt", dest="prompt_opt", help="User prompt")
    run.add_argument("--system", default="")
    run.add_argument("--no-thinking", action="store_true")
    run.add_argument("--ephemeral", action="store_true", help="Do not read or write a session")
    _add_worker_options(run)
    _add_generation_options(run)

    chat = subparsers.add_parser("chat", help="Start an interactive continuous conversation")
    _add_model_position(chat)
    chat.add_argument("--session", help="Session ID to resume")
    chat.add_argument("--title", default="", help="Title for a new persistent session")
    chat.add_argument("--system", default="")
    chat.add_argument("--no-thinking", action="store_true")
    chat.add_argument("--ephemeral", action="store_true", help="Keep the conversation in memory only")
    _add_worker_options(chat)
    _add_generation_options(chat)

    inspect = subparsers.add_parser("inspect", help="Inspect hardware, model, and effective resources")
    _add_model_position(inspect)
    inspect.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    _add_worker_options(inspect)

    tune = subparsers.add_parser("tune", help="Run an on-demand calibration and save a profile")
    _add_model_position(tune)
    tune.add_argument("prompt_pos", nargs="?", help="Calibration prompt")
    tune.add_argument("--prompt", dest="prompt_opt")
    tune.add_argument("--runs", type=int, default=2)
    tune.add_argument("--warmup", type=int, default=1)
    tune.add_argument("--no-thinking", action="store_true")
    tune.add_argument("--max-new-tokens", type=int, default=64)
    tune.add_argument("--context-tokens", type=int, default=0)
    tune.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Prefill tokens per chunk; 0 processes all remaining input as one batch (unchunked)",
    )
    tune.add_argument("--seed", type=int, default=0)
    _add_metrics_options(tune)
    _add_worker_options(tune)

    sessions = subparsers.add_parser("sessions", help="List, rename, or delete persistent sessions")
    sessions.add_argument("action", nargs="?", choices=("list", "delete", "rename"), default="list")
    sessions.add_argument("session_id", nargs="?")
    sessions.add_argument("title", nargs="?")
    sessions.add_argument("--config-dir")

    return parser.parse_args(argv)


def resolve_model(arguments: argparse.Namespace, *, required: bool = True) -> Path | None:
    value = getattr(arguments, "model_opt", None) or getattr(arguments, "model_pos", None)
    if not value:
        if required:
            raise ValueError("a model path is required; pass --model PATH or a positional model")
        return None
    model = Path(value).expanduser().resolve()
    if not model.is_dir() and not (model.is_file() and model.suffix.lower() == ".gguf"):
        raise ValueError(f"model directory or GGUF file does not exist: {model}")
    if model.is_dir() and not model_config_path(model).is_file() and not model_weight_files(model):
        raise ValueError(
            f"no model files found directly in directory: {model}; "
            "expected config.json or .gguf files. Subdirectories are not searched; "
            "pass the directory containing the model files or a GGUF file"
        )
    return model


def resolve_prompt(arguments: argparse.Namespace, *, required: bool = True) -> str | None:
    value = getattr(arguments, "prompt_opt", None)
    if value is None:
        value = getattr(arguments, "prompt_pos", None)
    if value is None:
        if required:
            raise ValueError("a prompt is required; pass --prompt TEXT or a positional prompt")
        return None
    return value


def default_worker_path(root: Path) -> Path:
    """Return the one conventional worker location used by the CLI."""
    worker_name = "ncnn_moe_worker.exe" if os.name == "nt" else "ncnn_moe_worker"
    build_dir = root / "build-ncnn"
    if os.name == "nt":
        return build_dir / "Release" / worker_name
    return build_dir / worker_name


def find_worker(explicit: str | None, root: Path) -> Path:
    """Resolve an explicitly selected worker or the fixed build-ncnn default."""
    source = "default"
    if explicit:
        candidate = Path(explicit).expanduser()
        source = "--worker"
    else:
        environment = os.environ.get("NCNN_MOE_WORKER")
        if environment:
            candidate = Path(environment).expanduser()
            source = "NCNN_MOE_WORKER"
        else:
            candidate = default_worker_path(root)

    if candidate.is_file():
        return candidate.resolve()

    resolved = candidate.resolve()
    if source == "default":
        raise ValueError(
            f"default worker was not found at {resolved}; build ncnn_moe_worker "
            "in build-ncnn or pass --worker PATH"
        )
    raise ValueError(f"worker from {source} was not found at {resolved}")


def lightweight_model_fingerprint(model: Path) -> str:
    digest = hashlib.sha256()
    config = model_config_path(model)
    if config.is_file():
        digest.update(config.read_bytes())
    root = model_asset_root(model)
    for path in model_weight_files(model):
        relative = path.name if path.suffix.lower() == ".gguf" else path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        head, tail = weight_fingerprint_samples(path)
        digest.update(head)
        if tail:
            digest.update(tail)
    return digest.hexdigest()[:20]


def cli_runtime_settings(arguments: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "backend",
        "expert_memory",
        "cpu_packed_weights",
        "host_memory_mb",
        "expert_cache_mb",
        "expert_gpu_cache_mb",
        "expert_gpu_victim_cache_mb",
        "expert_io_workers",
        "vulkan_device",
        "vulkan_devices",
        "expected_concurrency",
        "mmap_experts",
        "buffered_expert_io",
        "disable_gpu_victim_execution",
        "disable_vulkan_joint_cache_eviction",
        "disable_vulkan_mxfp4_row_tile",
        "disable_vulkan_expert_direct_staging",
        "disable_vulkan_shared_expert_overlap",
    )
    return {key: getattr(arguments, key) for key in keys if getattr(arguments, key, None) is not None}


def user_runtime_settings(store: SessionStore) -> dict[str, Any]:
    value = store.user_config()
    runtime = value.get("runtime", value)
    return runtime if isinstance(runtime, dict) else {}


def context_budget(adapter: ModelAdapter, arguments: argparse.Namespace, ready: dict[str, Any] | None = None) -> int | None:
    override = getattr(arguments, "context_tokens", 0) or 0
    if override > 0:
        return override
    model_limit = adapter.context_limit
    if model_limit is None and ready:
        model_limit = int(ready.get("model", {}).get("max_context_tokens", 0) or 0) or None
    if model_limit is None:
        return None
    reserve = max(128, int(getattr(arguments, "max_new_tokens", 1024) or 1024))
    return max(1, model_limit - reserve)


def runtime_cpu_qnk_q8_args(args: list[str], arguments: argparse.Namespace) -> list[str]:
    """Forward the opt-in int8 activation switch the shared runtime mapper does not know."""
    value = getattr(arguments, "cpu_qnk_q8", None)
    if value is None:
        return args
    return [*args, "--cpu-qnk-q8", str(value)]


def open_worker(
    arguments: argparse.Namespace,
    model: Path,
    store: SessionStore,
    *,
    adapter: ModelAdapter | None = None,
    session: dict[str, Any] | None = None,
    select_adapter: bool = False,
) -> tuple[WorkerClient, dict[str, Any], dict[str, Any] | None, ModelAdapter | None]:
    source_root = Path(__file__).resolve().parents[1]
    root = source_root if (source_root / "CMakeLists.txt").is_file() else Path.cwd().resolve()
    worker = find_worker(getattr(arguments, "worker", None), root)
    user = user_runtime_settings(store)
    cli = cli_runtime_settings(arguments)
    session_settings = session.get("settings", {}) if session else None
    if not isinstance(session_settings, dict):
        session_settings = {}
    initial = merge_runtime_settings(cli=cli, session=session_settings, profile=None, user=user)
    client = WorkerClient(
        worker,
        model,
        runtime_cpu_qnk_q8_args(runtime_args_from_settings(initial), arguments),
        verbose=getattr(arguments, "verbose", False),
    )
    if select_adapter and adapter is None:
        try:
            adapter = load_adapter(arguments, model, client.ready)
        except BaseException:
            client.close()
            raise
    model_fingerprint = adapter.model_fingerprint if adapter else lightweight_model_fingerprint(model)
    budget = context_budget(adapter, arguments, client.ready) if adapter else int(getattr(arguments, "context_tokens", 0) or 0)
    budget = budget or int(client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
    key = profile_key(
        hardware=hardware_fingerprint(client.ready),
        model=model_fingerprint,
        context_tokens=budget,
        session_count=1,
    )
    profile_record = store.profile(key)
    profile_settings = profile_record.get("settings", {}) if profile_record else {}
    if not isinstance(profile_settings, dict):
        profile_settings = {}
    merged = merge_runtime_settings(
        cli=cli,
        session=session_settings,
        profile=profile_settings,
        user=user,
    )
    if merged != initial:
        client.close()
        client = WorkerClient(
            worker,
            model,
            runtime_cpu_qnk_q8_args(runtime_args_from_settings(merged), arguments),
            verbose=getattr(arguments, "verbose", False),
        )
    return client, merged, profile_record, adapter


def default_generation_values(adapter: ModelAdapter, arguments: argparse.Namespace) -> tuple[float, int, float]:
    if arguments.temperature is None:
        temperature = 1.0
    else:
        temperature = arguments.temperature
    if arguments.top_k is None:
        top_k = 0 if adapter.name in {"gpt-oss", "deepseek-v4"} else 20
    else:
        top_k = arguments.top_k
    if arguments.top_p is None:
        top_p = 1.0 if adapter.name in {"gpt-oss", "deepseek-v4"} else 0.95
    else:
        top_p = arguments.top_p
    return temperature, top_k, top_p


def validate_generation_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if arguments.top_k is not None and arguments.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if arguments.speculative_max_draft < 0:
        raise ValueError("--speculative-max-draft must be non-negative")
    if not 0.0 <= arguments.speculative_confidence <= 1.0:
        raise ValueError("--speculative-confidence must be between 0 and 1")
    if arguments.metrics_interval_ms < 0:
        raise ValueError("--metrics-interval-ms must be non-negative")
    if arguments.context_tokens < 0 or arguments.prefill_chunk_size < 0:
        raise ValueError("context and prefill sizes must be non-negative (context 0=auto; prefill 0=unchunked)")
    for name in (
        "host_memory_mb",
        "expert_cache_mb",
        "expert_gpu_cache_mb",
        "expert_gpu_victim_cache_mb",
        "expert_io_workers",
        "vulkan_device",
        "expected_concurrency",
    ):
        value = getattr(arguments, name, None)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def metrics_trace_enabled(arguments: argparse.Namespace) -> bool:
    return bool(getattr(arguments, "metrics_enabled", True)) and getattr(arguments, "metrics_interval_ms", 1000) > 0


class ConversationApp:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        client: WorkerClient,
        store: SessionStore,
        arguments: argparse.Namespace,
        settings: dict[str, Any],
        record: dict[str, Any] | None,
        ephemeral: bool,
    ) -> None:
        self.adapter = adapter
        self.client = client
        self.store = store
        self.arguments = arguments
        self.settings = settings
        self.ephemeral = ephemeral
        self.console_class, self.panel_class = _optional_rich()
        self.console = self.console_class(stderr=True) if self.console_class else None
        self.prompt_session_class = _optional_prompt_session()
        self.native_session_id = "main"
        self.summary_session_id = "summary"
        self.summary_session_created = False
        self.messages: list[dict[str, str]] = []
        self.summary = ""
        self.native_context_count = 0
        self.last_reasoning = ""
        self.last_metrics: dict[str, Any] = {}
        self.record = record or {
            "id": store.new_id(),
            "title": getattr(arguments, "title", "") or "New conversation",
            "model": str(adapter.model),
            "model_type": adapter.model_type,
            "model_fingerprint": adapter.model_fingerprint,
            "messages": [],
            "summary": "",
            "settings": settings,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if record:
            self.messages = [
                {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
                for message in record.get("messages", [])
                if isinstance(message, dict) and message.get("content") is not None
            ]
            self.summary = str(record.get("summary", "") or "")
        if getattr(arguments, "system", "") and not any(message.get("role") == "system" for message in self.messages):
            self.messages.insert(0, {"role": "system", "content": arguments.system})
        if self.adapter.name == "qwen3.6":
            leading_system_count = 0
            while (
                leading_system_count < len(self.messages)
                and self.messages[leading_system_count].get("role") == "system"
            ):
                leading_system_count += 1
            if leading_system_count > 1:
                summary_marker = f"Conversation summary:\n{self.summary}" if self.summary else ""
                summary_suffix = f"\n\n{summary_marker}" if summary_marker else ""
                summary_seen = False
                system_parts: list[str] = []
                for message in self.messages[:leading_system_count]:
                    content = message["content"]
                    if summary_marker and content == summary_marker:
                        if summary_seen:
                            continue
                        summary_seen = True
                    elif summary_suffix and content.endswith(summary_suffix):
                        original = content[: -len(summary_suffix)]
                        if summary_seen:
                            content = original
                        else:
                            if original:
                                system_parts.append(original)
                            system_parts.append(summary_marker)
                            summary_seen = True
                            continue
                    if content:
                        system_parts.append(content)
                merged_system = "\n\n".join(system_parts)
                self.messages = [
                    {"role": "system", "content": merged_system},
                    *self.messages[leading_system_count:],
                ]
        self.client.create_session(
            self.native_session_id,
            seed=arguments.seed,
            prefill_chunk_size=arguments.prefill_chunk_size,
        )
        replay_succeeded = False
        if record and self.messages:
            try:
                replay = self._compact_messages(self.native_session_id, self.messages)
                self.native_context_count = int(replay.get("sequence_length", 0) or 0)
                replay_succeeded = True
                self._status(f"resumed {record.get('id')} ({len(self.messages)} messages, {self.native_context_count} context tokens)")
            except (AdapterError, WorkerError, ValueError) as error:
                self.native_context_count = 0
                self._status(f"session replay deferred: {error}")
        if (
            record
            and record.get("model_fingerprint") != self.adapter.model_fingerprint
            and (replay_succeeded or not self.messages)
        ):
            self._save()

    @property
    def context_limit(self) -> int | None:
        return context_budget(self.adapter, self.arguments, self.client.ready)

    @property
    def worker_context_tokens(self) -> int | None:
        override = int(getattr(self.arguments, "context_tokens", 0) or 0)
        if override > 0:
            return override
        model_limit = int(self.client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
        return model_limit or None

    def _generate_messages(self, session_id: str, messages: list[dict[str, str]], **options: Any) -> tuple[dict[str, Any], list[int]]:
        options.pop("stop_tokens", None)
        return self.client.generate(
            session_id,
            messages=messages,
            enable_thinking=self.adapter.thinking,
            context_tokens=self.worker_context_tokens,
            **options,
        )

    def _compact_messages(self, session_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        replay = self.client.compact(
            session_id,
            messages=messages,
            enable_thinking=self.adapter.thinking,
            context_tokens=self.worker_context_tokens,
        )
        sequence_length = replay.get("sequence_length")
        self.native_context_count = (
            sequence_length
            if isinstance(sequence_length, int) and not isinstance(sequence_length, bool)
            else 0
        )
        return replay

    def _prompt_count(self, session_id: str, messages: list[dict[str, str]]) -> int:
        event = self.client.stats(
            session_id,
            messages=messages,
            enable_thinking=self.adapter.thinking,
        )
        count = event.get("prompt_count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise WorkerError("worker did not return the native prompt count", event)
        return count

    def _status(self, message: str) -> None:
        if self.console:
            self.console.print(message)
        else:
            print(message, file=sys.stderr)

    def show_ready(self) -> None:
        model = self.client.ready.get("model", {})
        resources = self.client.ready.get("resources", {})
        self._status(
            f"model {model.get('model_type', self.adapter.model_type)} · "
            f"backend {resources.get('backend', 'N/A')} · "
            f"context {self.context_limit or 'auto'} tokens · session {self.record['id']}"
        )

    def _save(self) -> None:
        self.record["messages"] = self.messages
        self.record["summary"] = self.summary
        self.record["settings"] = self.settings
        self.record["model"] = str(self.adapter.model)
        self.record["model_type"] = self.adapter.model_type
        stored_fingerprint = self.record.get("model_fingerprint")
        legacy_fingerprints = getattr(self.adapter, "legacy_model_fingerprints", ())
        migration_ready = not self.messages or self.native_context_count > 0
        fingerprint_migrated = False
        compatible_fingerprint = (
            stored_fingerprint is None
            or stored_fingerprint == self.adapter.model_fingerprint
            or stored_fingerprint in legacy_fingerprints
        )
        if compatible_fingerprint and (
            stored_fingerprint == self.adapter.model_fingerprint or migration_ready
        ):
            fingerprint_migrated = stored_fingerprint != self.adapter.model_fingerprint
            self.record["model_fingerprint"] = self.adapter.model_fingerprint
        self.record["context_token_count"] = self.native_context_count
        if not self.ephemeral:
            self.store.save(self.record)
            if fingerprint_migrated:
                self._status("session tokenizer profile updated")

    def _transcript(self) -> str:
        parts = []
        for message in self.messages:
            parts.append(f"{message['role']}: {message['content']}")
        return "\n".join(parts)[-12000:]

    def _fallback_summary(self) -> str:
        text = self._transcript()
        if len(text) > 3500:
            text = text[-3500:]
        return "Earlier conversation (sliding-window fallback):\n" + text

    def _summarize(self) -> str:
        try:
            summary_system = (
                "Summarize the earlier conversation for a future assistant. Preserve decisions, "
                "constraints, names, open tasks, and facts. Be concise and do not answer the user."
            )
            summary_prompt = [
                {"role": "system", "content": summary_system},
                {"role": "user", "content": self._transcript()},
            ]
            if not self.summary_session_created:
                self.client.create_session(
                    self.summary_session_id,
                    seed=self.arguments.seed,
                    prefill_chunk_size=self.arguments.prefill_chunk_size,
                    enable_speculative_context=False,
                )
                self.summary_session_created = True
            else:
                self.client.reset(self.summary_session_id)
            summary_stream: dict[str, Any] = {"chunks": [], "decoded_chunks": []}
            self._generate_messages(
                self.summary_session_id,
                summary_prompt,
                request_id="compact-summary",
                max_new_tokens=min(512, self.arguments.max_new_tokens) if self.arguments.max_new_tokens else 512,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                min_p=0.0,
                enable_speculative=False,
                metrics_enabled=metrics_trace_enabled(self.arguments),
                metrics_interval_ms=self.arguments.metrics_interval_ms,
                on_event=lambda event: self._on_generation_event(
                    event, summary_stream, display=False
                ),
            )
            completion = self._decode_completion(summary_stream)
            summary = completion.answer or completion.reasoning or completion.text
            if summary.strip():
                return summary.strip()
        except (AdapterError, WorkerError, ValueError):
            pass
        return self._fallback_summary()

    def compact(self, *, announce: bool = True) -> None:
        if len(self.messages) <= 2:
            self.client.reset(self.native_session_id)
            self.native_context_count = 0
            if announce:
                self._status("context reset; not enough history to summarize")
            return
        old_messages = self.messages
        old_summary = self.summary
        summary = self._summarize()
        system_messages = [message for message in old_messages if message.get("role") == "system"]
        retained = [message for message in old_messages if message.get("role") != "system"][-4:]
        previous_summary = f"Conversation summary:\n{old_summary}" if old_summary else ""
        previous_suffix = f"\n\n{previous_summary}" if previous_summary else ""
        system_content_parts = []
        for message in system_messages:
            content = message["content"]
            if previous_summary and content == previous_summary:
                continue
            if previous_suffix and content.endswith(previous_suffix):
                content = content[: -len(previous_suffix)]
            if content:
                system_content_parts.append(content)
        system_content = "\n\n".join(system_content_parts)
        summary_content = f"Conversation summary:\n{summary}"
        combined_system = "\n\n".join(content for content in (system_content, summary_content) if content)
        self.messages = [{"role": "system", "content": combined_system}] + retained
        self.summary = summary
        try:
            replay = self._compact_messages(self.native_session_id, self.messages)
            self.native_context_count = int(replay.get("sequence_length", 0) or 0)
        except WorkerError:
            self.messages = old_messages
            self.summary = old_summary
            self._trim_sliding_window()
            self.client.reset(self.native_session_id)
            self.native_context_count = 0
            if announce:
                self._status("native compact failed; using sliding-window fallback")
            return
        if announce:
            self._status(f"context compacted · retained {len(self.messages)} messages")

    def _trim_sliding_window(self, prompt_count: int | None = None) -> None:
        budget = self.worker_context_tokens
        if budget is None:
            return
        if type(prompt_count) is not int or prompt_count <= 0:
            prompt_count = None
        for _ in range(100):
            if prompt_count is None:
                prompt_count = self._prompt_count(self.native_session_id, self.messages)
            generation_budget = self.arguments.max_new_tokens or 1
            if prompt_count + generation_budget <= budget:
                return
            prompt_count = None
            non_system = [index for index, message in enumerate(self.messages) if message.get("role") != "system"]
            if len(non_system) <= 1:
                if non_system:
                    message = self.messages[non_system[0]]
                    message["content"] = message["content"][-max(32, len(message["content"]) // 2) :]
                return
            del self.messages[non_system[0]]

    def _write_stream_delta(self, streamed: dict[str, Any], value: str) -> None:
        if not value:
            return
        streamed.setdefault("chunks", []).append(value)
        streamed["printed_length"] = streamed.get("printed_length", 0) + len(value)
        sys.stdout.write(value)
        sys.stdout.flush()

    def _stream_segment(
        self,
        segment: str,
        kind: str,
        state: dict[str, Any],
        streamed: dict[str, Any],
        *,
        terminal: bool = False,
    ) -> None:
        final_only = self.arguments.stream_final_only or not self.arguments.show_reasoning
        output: list[str] = []
        position = 0
        leading_key = "reasoning_leading" if kind == "reasoning" else "answer_leading"
        if kind == "reasoning":
            while position < len(segment) and state[leading_key] and segment[position].isspace():
                position += 1
        elif kind == "body" or state.get("answer_strip_all_leading", False):
            while position < len(segment) and state[leading_key] and segment[position].isspace():
                position += 1
        else:
            while position < len(segment) and state[leading_key] and segment[position] in "\r\n":
                position += 1
        if position < len(segment):
            state[leading_key] = False

        tail: list[str] = state["tail"]
        while position < len(segment):
            start = position
            candidate = segment[position].isspace() or segment[position] == "\ufffd"
            position += 1
            if candidate:
                while position < len(segment) and (segment[position].isspace() or segment[position] == "\ufffd"):
                    position += 1
                tail.append(segment[start:position])
                continue
            while position < len(segment) and not (segment[position].isspace() or segment[position] == "\ufffd"):
                position += 1
            safe_text = segment[start:position]
            pending = "".join(tail)
            tail.clear()
            if kind == "reasoning":
                if not final_only:
                    if not state["reasoning_nonempty"]:
                        output.append("[reasoning]\n")
                    if pending:
                        output.append(pending)
                    output.append(safe_text)
                state["reasoning_nonempty"] = True
            else:
                if not final_only and not state["answer_nonempty"]:
                    label = "\n[answer]\n" if state["reasoning_nonempty"] else "[answer]\n"
                    output.append(label)
                if pending:
                    output.append(pending)
                output.append(safe_text)
                state["answer_nonempty"] = True

        if terminal:
            pending = "".join(tail).rstrip()
            tail.clear()
            if pending:
                if kind == "reasoning":
                    if not final_only:
                        if not state["reasoning_nonempty"]:
                            output.append("[reasoning]\n")
                        output.append(pending)
                    state["reasoning_nonempty"] = True
                else:
                    if not final_only and not state["answer_nonempty"]:
                        label = "\n[answer]\n" if state["reasoning_nonempty"] else "[answer]\n"
                        output.append(label)
                    output.append(pending)
                    state["answer_nonempty"] = True
        self._write_stream_delta(streamed, "".join(output))

    def _stream_thinking_text(self, text: str, streamed: dict[str, Any]) -> None:
        state = streamed.setdefault(
            "thinking_state",
            {
                "phase": "reasoning" if self.adapter.thinking else "body",
                "marker_carry": "",
                "tail": [],
                "reasoning_leading": True,
                "answer_leading": True,
                "reasoning_nonempty": False,
                "answer_nonempty": False,
                "answer_strip_all_leading": self.adapter.name == "deepseek-v4",
            },
        )
        marker = "</think>"
        if state["phase"] == "reasoning":
            combined = state["marker_carry"] + text
            if not state.get("opening_marker_done"):
                combined = combined.lstrip()
                opening = "<think>"
                if len(combined) < len(opening) and opening.startswith(combined):
                    state["marker_carry"] = combined
                    return
                combined = combined.removeprefix(opening)
                state["opening_marker_done"] = True
            marker_index = combined.find(marker)
            if marker_index >= 0:
                self._stream_segment(combined[:marker_index], "reasoning", state, streamed, terminal=True)
                state["phase"] = "answer"
                state["marker_carry"] = ""
                self._stream_segment(combined[marker_index + len(marker) :], "answer", state, streamed)
                return
            carry_length = 0
            for length in range(min(len(marker) - 1, len(combined)), 0, -1):
                if combined.endswith(marker[:length]):
                    carry_length = length
                    break
            state["marker_carry"] = combined[-carry_length:] if carry_length else ""
            safe_text = combined[:-carry_length] if carry_length else combined
            self._stream_segment(safe_text, "reasoning", state, streamed)
            return
        self._stream_segment(text, state["phase"], state, streamed)

    def _stream_harmony_event(
        self,
        token_id: int,
        text: str,
        streamed: dict[str, Any],
        *,
        display: bool,
    ) -> None:
        state = streamed.setdefault(
            "harmony_state",
            {
                "body": False,
                "channel": "answer",
                "expect_channel": False,
                "channel_text": [],
                "reasoning_parts": [],
                "answer_parts": [],
                "reasoning_leading": True,
                "answer_leading": True,
                "reasoning_nonempty": False,
                "answer_nonempty": False,
                "tail": [],
            },
        )
        special = {
            200006: ("start", "<|start|>"),
            200007: ("end", "<|end|>"),
            200008: ("message", "<|message|>"),
            200005: ("channel", "<|channel|>"),
            200002: ("return", "<|return|>"),
            200012: ("call", "<|call|>"),
        }.get(token_id)
        kind, framing_text = special if special is not None else (None, "")
        body_text = text[:-len(framing_text)] if framing_text and text.endswith(framing_text) else text

        if state["body"] and kind is None:
            state["reasoning_parts" if state["channel"] == "reasoning" else "answer_parts"].append(text)
            if display:
                self._stream_segment(text, state["channel"], state, streamed)
            return

        if state["body"] and kind is not None:
            if body_text:
                state["reasoning_parts" if state["channel"] == "reasoning" else "answer_parts"].append(body_text)
                if display:
                    self._stream_segment(body_text, state["channel"], state, streamed)
            if display:
                self._stream_segment("", state["channel"], state, streamed, terminal=True)
            else:
                state["tail"].clear()
            state["body"] = False

        if kind == "start":
            state["channel"] = "answer"
            state["expect_channel"] = False
            state["channel_text"].clear()
        elif kind == "channel":
            state["expect_channel"] = True
            state["channel_text"].clear()
        elif kind == "message":
            channel = "".join(state["channel_text"]).strip()
            state["channel"] = "reasoning" if channel == "analysis" else "answer"
            state["channel_text"].clear()
            state["expect_channel"] = False
            state["body"] = True
        elif kind in {"end", "return", "call"}:
            state["body"] = False
            state["channel"] = "answer"
            state["expect_channel"] = False
            state["channel_text"].clear()
        elif state["expect_channel"]:
            state["channel_text"].append(text)

    def _on_generation_event(
        self,
        event: dict[str, Any],
        streamed: dict[str, Any],
        *,
        display: bool = True,
    ) -> None:
        if event.get("event") == "metrics":
            self.last_metrics = event.get("metrics", {})
            if display:
                self._status(_format_runtime_metrics(self.last_metrics))
            return
        if event.get("event") != "token":
            return

        text = event.get("text")
        if not isinstance(text, str):
            raise WorkerError("native worker token event did not include decoded text", event)
        if self.adapter.name == "deepseek-v4":
            token_id = event.get("token_id")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkerError("DeepSeek V4 worker token event did not include a token ID", event)
            if token_id == 1:
                eos_marker = "<｜end▁of▁sentence｜>"
                if text.endswith(eos_marker):
                    text = text[: -len(eos_marker)]
                if not text:
                    return
        if self.adapter.name == "gpt-oss":
            token_id = event.get("token_id")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkerError("GPT-OSS worker token event did not include a token ID", event)
            self._stream_harmony_event(token_id, text, streamed, display=display and self.arguments.stream)
            return
        if self.adapter.name.startswith("qwen"):
            token_id = event.get("token_id")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkerError("Qwen worker token event did not include a token ID", event)
            streamed["ended_with_stop_token"] = token_id in self.adapter.stop_tokens

        streamed.setdefault("decoded_chunks", []).append(text)
        if display and self.arguments.stream:
            self._stream_thinking_text(text, streamed)

    def _decode_completion(self, streamed: dict[str, Any]) -> Completion:
        if self.adapter.name == "gpt-oss":
            state = streamed.get("harmony_state", {})
            return Completion(
                "".join(state.get("reasoning_parts", [])).strip(),
                "".join(state.get("answer_parts", [])).strip(),
            )
        text = "".join(streamed.get("decoded_chunks", []))
        if self.adapter.name.startswith("qwen") and streamed.get("ended_with_stop_token"):
            text = text.rstrip().rstrip("\ufffd").rstrip()
        return self.adapter.decode_completion_text(text)

    def send(self, user_text: str) -> tuple[Completion, dict[str, Any], bool]:
        self.messages.append({"role": "user", "content": user_text})
        streamed: dict[str, Any] = {"chunks": [], "decoded_chunks": []}
        temperature, top_k, top_p = default_generation_values(self.adapter, self.arguments)
        request_id = f"chat-{int(time.time() * 1000)}"
        options = {
            "request_id": request_id,
            "max_new_tokens": self.arguments.max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": self.arguments.min_p,
            "speculative_confidence": self.arguments.speculative_confidence,
            "speculative_max_draft": self.arguments.speculative_max_draft,
            "metrics_enabled": metrics_trace_enabled(self.arguments),
            "metrics_interval_ms": self.arguments.metrics_interval_ms,
            "on_event": lambda event: self._on_generation_event(event, streamed),
        }
        def generate() -> tuple[dict[str, Any], list[int]]:
            try:
                return self._generate_messages(self.native_session_id, self.messages, **options)
            except WorkerError as error:
                if error.event.get("code") not in {"context_overflow", "invalid_request"}:
                    self.native_context_count = 0
                raise

        try:
            done, _ = generate()
        except WorkerError as error:
            if error.event.get("code") != "context_overflow":
                raise
            pending_user = self.messages.pop()
            self.compact()
            self.messages.append(pending_user)
            try:
                done, _ = generate()
            except WorkerError as retry_error:
                if retry_error.event.get("code") != "context_overflow":
                    raise
                self._trim_sliding_window(retry_error.event.get("prompt_count"))
                done, _ = generate()

        reused = bool(done.get("prefix_reused", False))
        completion = self._decode_completion(streamed)
        if self.arguments.stream and (streamed.get("decoded_chunks") or streamed.get("harmony_state")):
            visible = self.adapter.visible_completion(
                completion,
                final_only=self.arguments.stream_final_only or not self.arguments.show_reasoning,
            )
            chunks = streamed["chunks"]
            printed = "".join(chunks)
            if visible.startswith(printed):
                delta = visible[len(printed) :]
                if delta:
                    chunks.append(delta)
                    streamed["printed_length"] = streamed.get("printed_length", 0) + len(delta)
                    sys.stdout.write(delta)
                    sys.stdout.flush()
        sequence_length = done.get("sequence_length")
        self.native_context_count = (
            sequence_length
            if isinstance(sequence_length, int) and not isinstance(sequence_length, bool)
            else 0
        )
        self.last_reasoning = completion.reasoning
        self.messages.append({"role": "assistant", "content": completion.answer or completion.reasoning})
        self._save()
        if self.arguments.stream and streamed.get("printed_length", 0):
            sys.stdout.write("\n")
            sys.stdout.flush()
        return completion, done, reused

    def display_completion(self, completion: Completion) -> None:
        if completion.reasoning and self.arguments.show_reasoning:
            print("[reasoning]")
            print(completion.reasoning)
        elif completion.reasoning and not self.arguments.show_reasoning:
            self._status("reasoning hidden · use --show-reasoning or /reasoning")
        if completion.answer:
            print("[answer]")
            print(completion.answer)

    def status_after(self, done: dict[str, Any], reused: bool) -> None:
        metrics = done.get("metrics", {})
        self.last_metrics = metrics if isinstance(metrics, dict) else {}
        context_count = self.native_context_count
        status = (
            _format_runtime_metrics(self.last_metrics)
            + "\n  Context "
            + f"{context_count}/{self.context_limit or 'auto'}"
            + " tokens"
            + f" · prefix {'reused' if reused else 'replayed'}"
        )
        generated_tokens = done.get("generated_tokens")
        max_new_tokens = getattr(self.arguments, "max_new_tokens", None)
        stop_flags_clear = all(
            done.get(name) is False
            for name in ("stopped_by_stop_token", "stopped_by_callback", "cancelled")
        )
        if type(generated_tokens) is int and generated_tokens > 0 and stop_flags_clear:
            if type(max_new_tokens) is int and max_new_tokens > 0 and generated_tokens == max_new_tokens:
                status += (
                    f"\n  Response may be incomplete: generated {generated_tokens}/{max_new_tokens} tokens. "
                    "Set --max-new-tokens 0 to generate until a stop token or the remaining context limit, "
                    "or increase --max-new-tokens."
                )
            elif type(max_new_tokens) is int and max_new_tokens == 0:
                sequence_length = done.get("sequence_length")
                worker_context_tokens = self.worker_context_tokens
                if (
                    type(sequence_length) is int
                    and type(worker_context_tokens) is int
                    and worker_context_tokens > 0
                    # The last sampled token may not have been committed to KV.
                    and sequence_length in (worker_context_tokens - 1, worker_context_tokens)
                ):
                    status += (
                        f"\n  Response may be incomplete: reached the context limit after "
                        f"{generated_tokens} generated tokens. Increase the context limit or shorten the prompt."
                    )
        self._status(status)

    def show_context(self) -> None:
        tokens = self._prompt_count(self.native_session_id, self.messages) if self.messages else 0
        budget = self.context_limit
        self._status(
            f"messages={len(self.messages)} prompt_tokens={tokens} "
            f"budget={budget or 'unknown'} native_sequence={self.native_context_count}"
        )

    def show_settings(self) -> None:
        print(json.dumps(self.settings, ensure_ascii=False, indent=2, sort_keys=True))

    def tune_current(self) -> None:
        user_messages = [message for message in self.messages if message.get("role") == "user"]
        prompt = user_messages[-1]["content"] if user_messages else "Calibrate a short response."
        tune_session = "tune-live"
        try:
            if not getattr(self, "tune_session_created", False):
                self.client.create_session(tune_session, seed=self.arguments.seed, prefill_chunk_size=self.arguments.prefill_chunk_size, enable_speculative_context=False)
                self.tune_session_created = True
            else:
                self.client.reset(tune_session)
            temperature, top_k, top_p = default_generation_values(self.adapter, self.arguments)
            done, _ = self._generate_messages(
                tune_session,
                [{"role": "user", "content": prompt}],
                request_id="tune-live",
                max_new_tokens=min(32, self.arguments.max_new_tokens) if self.arguments.max_new_tokens else 32,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=self.arguments.min_p,
                enable_speculative=False,
                metrics_enabled=metrics_trace_enabled(self.arguments),
                metrics_interval_ms=self.arguments.metrics_interval_ms,
            )
            budget = self.context_limit or int(self.client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
            key = profile_key(
                hardware=hardware_fingerprint(self.client.ready),
                model=self.adapter.model_fingerprint,
                context_tokens=budget,
                session_count=1,
            )
            self.store.save_profile(
                key,
                {
                    "settings": self.settings,
                    "average_tokens_per_second": done.get("tokens_per_second"),
                    "samples": [done.get("tokens_per_second")],
                    "model": self.adapter.model_fingerprint,
                    "hardware": hardware_fingerprint(self.client.ready),
                    "context_tokens": budget,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            self._status(f"tuning profile saved · {done.get('tokens_per_second', 'N/A')} token/s · {key}")
        except (AdapterError, WorkerError, ValueError) as error:
            self._status(f"tune failed: {error}")

    def show_stats(self) -> None:
        event = self.client.stats(self.native_session_id)
        print(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True))

    def new_session(self) -> None:
        self._save()
        self.native_session_id = f"main-{int(time.time() * 1000)}"
        self.client.create_session(self.native_session_id, seed=self.arguments.seed, prefill_chunk_size=self.arguments.prefill_chunk_size)
        self.messages = []
        if self.arguments.system:
            self.messages.append({"role": "system", "content": self.arguments.system})
        self.native_context_count = 0
        self.record = {
            "id": self.store.new_id(),
            "title": "New conversation",
            "model": str(self.adapter.model),
            "model_type": self.adapter.model_type,
            "model_fingerprint": self.adapter.model_fingerprint,
            "messages": self.messages,
            "summary": "",
            "settings": self.settings,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._status(f"started session {self.record['id']}")

    def command(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            self._status(str(error))
            return True
        if not parts:
            return True
        command = parts[0].lower()
        if command in {"/quit", "/exit", "/q"}:
            return False
        if command == "/help":
            self._status("/context /compact /reset /new /sessions /settings /stats /reasoning /tune /quit")
        elif command == "/context":
            self.show_context()
        elif command == "/compact":
            self.compact()
        elif command == "/reset":
            self.native_context_count = 0
            self.client.reset(self.native_session_id)
            self.messages = [message for message in self.messages if message.get("role") == "system"]
            self._save()
            self._status("conversation reset")
        elif command == "/new":
            self.new_session()
        elif command == "/sessions":
            for record in self.store.list():
                print(f"{record.get('id')}  {record.get('title', '')}  {record.get('updated_at', '')}")
        elif command == "/settings":
            self.show_settings()
            if len(parts) > 1:
                self._status("resource changes require restarting the worker; apply them with CLI options or a new chat task")
        elif command == "/stats":
            self.show_stats()
        elif command == "/reasoning":
            print(self.last_reasoning or "(no reasoning in the last response)")
        elif command == "/tune":
            self.tune_current()
        else:
            self._status(f"unknown command: {command}; use /help")
        return True

    def chat(self, source: TextIO | None = None) -> int:
        """Read Unicode lines from a frontend-owned text stream or console."""
        self.show_ready()
        if source is None and not sys.stdin.isatty():
            source = sys.stdin
        prompt_session = self.prompt_session_class() if source is None and self.prompt_session_class else None
        while True:
            try:
                if source is None:
                    line = prompt_session.prompt("you> ") if prompt_session else input("you> ")
                else:
                    line = source.readline()
                    if not isinstance(line, str):
                        raise ValueError("chat input must be a decoded text stream")
                    if not line:
                        break
                    line = line.rstrip("\r\n")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line.strip():
                continue
            if line.startswith("/"):
                try:
                    if not self.command(line):
                        break
                except (AdapterError, WorkerError, ValueError) as error:
                    self._status(f"error: {error}")
                continue
            try:
                completion, done, reused = self.send(line)
                if not self.arguments.stream:
                    self.display_completion(completion)
                self.status_after(done, reused)
            except (AdapterError, WorkerError, ValueError) as error:
                self._status(f"error: {error}")
        self._save()
        return 0


def load_adapter(arguments: argparse.Namespace, model: Path, ready: dict[str, Any] | None = None) -> ModelAdapter:
    thinking = not getattr(arguments, "no_thinking", False)
    model_info = ready.get("model", {}) if isinstance(ready, dict) else {}
    if not isinstance(model_info, dict) or model_info.get("native_text_supported") is not True:
        raise AdapterError(
            "native text is unavailable: this runtime or its tokenizer assets "
            "do not support the model"
        )
    native_version = model_info.get("native_text_version")
    if native_version != NATIVE_TEXT_VERSION:
        raise AdapterError(
            f"worker native text version {native_version!r} is unsupported; "
            f"expected {NATIVE_TEXT_VERSION}"
        )
    stop_tokens = model_info.get("native_stop_tokens")
    if not isinstance(stop_tokens, list) or not stop_tokens:
        raise AdapterError("worker did not provide native stop-token metadata")
    return create_adapter(
        model,
        thinking=thinking,
        native_stop_tokens=stop_tokens,
        native_model_info=model_info,
    )


def run_command(arguments: argparse.Namespace) -> int:
    validate_generation_arguments(arguments)
    model = resolve_model(arguments)
    prompt = resolve_prompt(arguments)
    assert model is not None and prompt is not None
    store = SessionStore()
    client, settings, _, adapter = open_worker(arguments, model, store, select_adapter=True)
    assert adapter is not None
    record = None
    try:
        app = ConversationApp(
            adapter=adapter,
            client=client,
            store=store,
            arguments=arguments,
            settings=settings,
            record=record,
            ephemeral=True,
        )
        completion, done, reused = app.send(prompt)
        if not arguments.stream:
            app.display_completion(completion)
        app.status_after(done, reused)
        return 0
    finally:
        client.close()


def chat_command(arguments: argparse.Namespace) -> int:
    validate_generation_arguments(arguments)
    model = resolve_model(arguments)
    assert model is not None
    store = SessionStore()
    record = store.load(arguments.session) if arguments.session else None
    if arguments.session and record is None:
        raise ValueError(f"session does not exist: {arguments.session}")
    client, settings, _, adapter = open_worker(arguments, model, store, session=record, select_adapter=True)
    assert adapter is not None
    try:
        legacy_fingerprints = getattr(adapter, "legacy_model_fingerprints", ())
        stored_fingerprint = record.get("model_fingerprint") if record else None
        if (
            record
            and stored_fingerprint is not None
            and stored_fingerprint != adapter.model_fingerprint
            and stored_fingerprint not in legacy_fingerprints
        ):
            raise ValueError("the requested session belongs to a different model fingerprint")
        app = ConversationApp(
            adapter=adapter,
            client=client,
            store=store,
            arguments=arguments,
            settings=settings,
            record=record,
            ephemeral=arguments.ephemeral,
        )
        return app.chat()
    finally:
        client.close()


def inspect_command(arguments: argparse.Namespace) -> int:
    model = resolve_model(arguments)
    assert model is not None
    store = SessionStore()
    client, _, _, _ = open_worker(arguments, model, store)
    try:
        if arguments.json:
            print(json.dumps(client.ready, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            ready = client.ready
            model_info = ready.get("model", {})
            resources = ready.get("resources", {})
            capabilities = ready.get("capabilities", {})
            telemetry = ready.get("telemetry", {})
            print(f"model: {model_info.get('model_type', 'N/A')}")
            print(f"backend: {resources.get('backend', 'N/A')}")
            print(
                "CPU packed weights: "
                f"{resources.get('selected_cpu_packed_weights', 'off')}"
            )
            print(f"host memory: {_format_bytes_gb(resources.get('host_memory_budget_bytes'))}")
            print(f"Expert cache: {_format_bytes_gb(resources.get('expert_cache_bytes'))}")
            print(
                "packed Expert estimate: "
                f"{_format_bytes_gb(resources.get('estimated_cpu_packed_expert_bytes'))}"
            )
            print(f"Expert IO workers: {resources.get('expert_io_workers', 'N/A')}")
            print(f"CPU: {capabilities.get('physical_cpu_core_count', 'N/A')} physical / {capabilities.get('logical_cpu_count', 'N/A')} logical")
            print(f"Vulkan devices: {capabilities.get('vulkan_device_count', 0)}")
            print(f"GPU telemetry: {_format_gpu_utilization(telemetry)}")
            for device in capabilities.get("vulkan_devices", []):
                print(
                    f"  [{device.get('index')}] {device.get('name')} "
                    f"({device.get('type')}) heap={_format_bytes_gb(device.get('heap_budget_bytes'))}"
                )
        return 0
    finally:
        client.close()


def tune_command(arguments: argparse.Namespace) -> int:
    if arguments.runs <= 0 or arguments.warmup < 0:
        raise ValueError("--runs must be positive and --warmup must be non-negative")
    model = resolve_model(arguments)
    prompt = resolve_prompt(arguments)
    assert model is not None and prompt is not None
    store = SessionStore()
    client, settings, _, adapter = open_worker(arguments, model, store, select_adapter=True)
    assert adapter is not None
    try:
        client.create_session("tune", seed=arguments.seed, prefill_chunk_size=arguments.prefill_chunk_size, enable_speculative_context=False)
        prompt_messages = [{"role": "user", "content": prompt}]
        temperature = 0.0 if adapter.name == "gpt-oss" else 1.0
        rates: list[float] = []
        for index in range(arguments.warmup + arguments.runs):
            if index:
                client.reset("tune")
            options: dict[str, Any] = {
                "request_id": f"tune-{index}",
                "max_new_tokens": arguments.max_new_tokens,
                "temperature": temperature,
                "top_k": 0,
                "top_p": 1.0,
                "enable_speculative": False,
                "metrics_enabled": metrics_trace_enabled(arguments),
                "metrics_interval_ms": arguments.metrics_interval_ms if hasattr(arguments, "metrics_interval_ms") else 1000,
            }
            model_limit = int(client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
            context_tokens = int(getattr(arguments, "context_tokens", 0) or 0) or model_limit or None
            done, _ = client.generate(
                "tune",
                messages=prompt_messages,
                enable_thinking=adapter.thinking,
                context_tokens=context_tokens,
                **options,
            )
            if index >= arguments.warmup and isinstance(done.get("tokens_per_second"), (int, float)):
                rates.append(float(done["tokens_per_second"]))
        average = sum(rates) / len(rates) if rates else 0.0
        budget = context_budget(adapter, arguments, client.ready) or int(client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
        key = profile_key(
            hardware=hardware_fingerprint(client.ready),
            model=adapter.model_fingerprint,
            context_tokens=budget,
            session_count=1,
        )
        store.save_profile(
            key,
            {
                "settings": settings,
                "average_tokens_per_second": average,
                "samples": rates,
                "model": adapter.model_fingerprint,
                "hardware": hardware_fingerprint(client.ready),
                "context_tokens": budget,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        print(json.dumps({"profile_key": key, "average_tokens_per_second": average, "samples": rates}, indent=2))
        return 0
    finally:
        client.close()


def sessions_command(arguments: argparse.Namespace) -> int:
    store = SessionStore()
    if arguments.action == "list":
        records = store.list()
        if not records:
            print("No persistent sessions.")
            return 0
        for record in records:
            print(f"{record.get('id')}\t{record.get('title', '')}\t{record.get('model_type', '')}\t{record.get('updated_at', '')}")
        return 0
    if not arguments.session_id:
        raise ValueError(f"sessions {arguments.action} requires a session ID")
    if arguments.action == "delete":
        if not store.delete(arguments.session_id):
            raise ValueError(f"session does not exist: {arguments.session_id}")
        print(f"deleted {arguments.session_id}")
        return 0
    if not arguments.title:
        raise ValueError("sessions rename requires a title")
    store.rename(arguments.session_id, arguments.title)
    print(f"renamed {arguments.session_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_standard_streams()
    arguments = parse_arguments(argv)
    if getattr(arguments, "config_dir", None):
        os.environ["NCNN_MOE_CONFIG_DIR"] = str(Path(arguments.config_dir).expanduser().resolve())
    try:
        if arguments.command == "run":
            return run_command(arguments)
        if arguments.command == "chat":
            return chat_command(arguments)
        if arguments.command == "inspect":
            return inspect_command(arguments)
        if arguments.command == "tune":
            return tune_command(arguments)
        if arguments.command == "sessions":
            return sessions_command(arguments)
        raise ValueError(f"unknown command: {arguments.command}")
    except (AdapterError, WorkerError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
