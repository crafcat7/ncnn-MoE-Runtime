"""Model identity and text presentation for the native ncnn-MoE worker."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    """Split the first generated Qwen thinking boundary, if present."""
    if not thinking:
        return Completion("", text.strip())
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
        self.thinking = bool(thinking)
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
        if model_type in {"qwen3_5_moe", "qwen3_6", "qwen3.6", "qwen3_5", "qwen35", "qwen35moe"}:
            return "qwen3.6"
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
