"""Model identity and text presentation for the native ncnn-MoE worker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NATIVE_TEXT_VERSION = "ncnn-moe-text-v2"


class AdapterError(RuntimeError):
    """The model or native worker cannot provide the required text interface."""


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
    ) -> None:
        self.model = Path(model).resolve()
        self.config = _load_json(self.model / "config.json")
        self.model_type = str(self.config.get("model_type", "unknown")).lower()
        self.name = self._model_name(self.model_type)
        self.thinking = bool(thinking)
        self._stop_tokens = self._validate_stop_tokens(native_stop_tokens)
        self.model_fingerprint, self.legacy_model_fingerprints = self._fingerprints()

    @staticmethod
    def _model_name(model_type: str) -> str:
        if model_type in {"gpt_oss", "gpt-oss", "gpt_oss_moe"}:
            return "gpt-oss"
        if model_type in {"deepseek_v4", "deepseek-v4", "deepseek_v4_flash"}:
            return "deepseek-v4"
        if model_type in {"qwen4_exp", "qwen4_exp_text"}:
            return "qwen3.8"
        if model_type in {"qwen3_5_moe", "qwen3_6", "qwen3.6", "qwen3_5"}:
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
        return None

    def _fingerprints(self) -> tuple[str, tuple[str, ...]]:
        """Hash current and pre-v2 identities while scanning model weights once."""
        legacy = hashlib.sha256()
        legacy.update((self.model_type + "\0").encode("utf-8"))
        config_path = self.model / "config.json"
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
            path = self.model / name
            current.update(name.encode("utf-8") + b"\0")
            if not path.is_file():
                current.update(b"\0")
                continue
            current.update(b"\1")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    current.update(block)

        for name in ("tokenizer.json", "chat_template.jinja", "encoding/encoding_dsv4.py"):
            path = self.model / name
            if path.is_file():
                legacy.update(name.encode("utf-8"))
                legacy.update(str(path.stat().st_size).encode("ascii"))

        for path in sorted(self.model.rglob("*.safetensors")):
            relative = path.relative_to(self.model).as_posix()
            size = path.stat().st_size
            legacy.update(relative.encode("utf-8"))
            legacy.update(str(size).encode("ascii"))
            current.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0")
            with path.open("rb") as stream:
                head = stream.read(4096)
                tail = b""
                if size > 4096:
                    stream.seek(-4096, 2)
                    tail = stream.read(4096)
            legacy.update(head)
            if tail:
                legacy.update(tail)
            current.update(head)
            if tail:
                current.update(tail)

        legacy_base = legacy.hexdigest()[:20]
        prior_fingerprints = [legacy_base]
        if self.name.startswith("qwen"):
            old_qwen = hashlib.sha256()
            old_qwen.update(legacy_base.encode("ascii"))
            old_qwen.update(b"ncnn-moe-tokenizer-json-v1\0")
            if self.model_type == "qwen3_5_moe":
                old_qwen.update(b"ncnn-moe-qwen3.6-nfc9-regex16-v1\0")
            for name in model_files:
                path = self.model / name
                old_qwen.update(name.encode("utf-8") + b"\0")
                if not path.is_file():
                    old_qwen.update(b"\0")
                    continue
                old_qwen.update(b"\1")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        old_qwen.update(block)
            prior_fingerprints.insert(0, old_qwen.hexdigest()[:20])
        return current.hexdigest()[:20], tuple(prior_fingerprints)

    def validate(self) -> None:
        if not (self.model / "config.json").is_file():
            raise AdapterError(f"config.json is missing from: {self.model}")

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
