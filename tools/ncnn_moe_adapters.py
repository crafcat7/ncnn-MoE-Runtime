"""Tokenizer and chat-template adapters for the unified ncnn_moe CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import ModuleType
from typing import Any


class AdapterError(RuntimeError):
    """The model is unsupported or its official text components are missing."""


@dataclass(frozen=True)
class Completion:
    reasoning: str
    answer: str

    @property
    def text(self) -> str:
        return self.answer or self.reasoning


def _split_thinking(text: str, thinking: bool) -> Completion:
    marker = "</think>"
    if not thinking:
        return Completion("", text.strip())
    if marker in text:
        reasoning, answer = text.split(marker, 1)
        return Completion(reasoning.strip(), answer.lstrip("\r\n"))
    return Completion(text.strip(), "")


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


def _required(model: Path, *names: str) -> None:
    for name in names:
        if not (model / name).is_file():
            raise AdapterError(f"{name} is missing from: {model}")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - depends on model package
        raise AdapterError(f"cannot import {path}: {error}") from error
    return module


def _load_transformers_tokenizer(model: Path) -> Any:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - dependency availability varies
        raise AdapterError(
            "transformers is required for this model; install it with "
            "'python -m pip install -U transformers'"
        ) from error
    try:
        return AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    except Exception as error:  # pragma: no cover - tokenizer implementation varies
        raise AdapterError(f"cannot load the official tokenizer: {error}") from error


def _normalize_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer output into the native worker's flat token list."""
    if isinstance(value, Mapping):
        value = value.get("input_ids")

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    # Tokenizers may return one batch dimension when tensors or a batch
    # encoding is requested. The native protocol accepts one prompt only.
    while isinstance(value, (list, tuple)) and len(value) == 1:
        first = value[0]
        first_tolist = getattr(first, "tolist", None)
        if callable(first_tolist):
            first = first_tolist()
        if not isinstance(first, (list, tuple)):
            break
        value = first

    if not isinstance(value, (list, tuple)) or not value:
        raise AdapterError("the Qwen chat template did not return token IDs")

    tokens: list[int] = []
    for token in value:
        item = getattr(token, "item", None)
        if callable(item):
            token = item()
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise AdapterError("the Qwen chat template did not return token IDs")
        tokens.append(int(token))
    return tokens


class ModelAdapter:
    name = "generic"

    def __init__(self, model: Path, *, thinking: bool = True, thinking_mode: str = "thinking") -> None:
        self.model = model.resolve()
        self.config = _load_json(self.model / "config.json")
        self.thinking = thinking
        self.thinking_mode = thinking_mode
        self.model_type = str(self.config.get("model_type", "unknown"))
        self.model_fingerprint = self._fingerprint()

    @classmethod
    def detect(cls, model: Path, **options: Any) -> "ModelAdapter":
        config = _load_json(model.resolve() / "config.json")
        model_type = str(config.get("model_type", "")).lower()
        if model_type in {"gpt_oss", "gpt-oss", "gpt_oss_moe"}:
            return GptOssAdapter(model, **options)
        if model_type in {"deepseek_v4", "deepseek-v4", "deepseek_v4_flash"}:
            return DeepSeekAdapter(model, **options)
        if model_type in {
            "qwen3_5_moe",
            "qwen3_6",
            "qwen3.6",
            "qwen3_5",
            "qwen4_exp",
            "qwen4_exp_text",
        }:
            return QwenAdapter(model, **options)
        raise AdapterError(f"unsupported model_type: {model_type or '<missing>'}")

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update((self.model_type + "\0").encode("utf-8"))
        digest.update((self.model / "config.json").read_bytes())
        for name in ("tokenizer.json", "chat_template.jinja", "encoding/encoding_dsv4.py"):
            path = self.model / name
            if path.is_file():
                digest.update(name.encode("utf-8"))
                digest.update(str(path.stat().st_size).encode("ascii"))
        for path in sorted(self.model.rglob("*.safetensors")):
            relative = path.relative_to(self.model).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            with path.open("rb") as stream:
                digest.update(stream.read(4096))
                if path.stat().st_size > 4096:
                    stream.seek(-4096, 2)
                    digest.update(stream.read(4096))
        return digest.hexdigest()[:20]

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

    @property
    def stop_tokens(self) -> list[int]:
        return []

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        raise NotImplementedError

    def decode_text(self, tokens: list[int]) -> str:
        raise NotImplementedError

    def decode_completion(self, tokens: list[int]) -> Completion:
        return _split_thinking(self.decode_text(tokens), self.thinking)

    def validate(self) -> None:
        _required(self.model, "config.json")

    def stream_visible(self, tokens: list[int], *, final_only: bool = False) -> str:
        completion = self.decode_completion(tokens)
        if final_only:
            return completion.answer
        if completion.answer:
            if completion.reasoning:
                return f"[reasoning]\n{completion.reasoning}\n[answer]\n{completion.answer}"
            return f"[answer]\n{completion.answer}"
        return f"[reasoning]\n{completion.reasoning}" if completion.reasoning else ""


class GptOssAdapter(ModelAdapter):
    name = "gpt-oss"

    def __init__(self, model: Path, **options: Any) -> None:
        super().__init__(model, **options)
        _required(self.model, "config.json")
        try:
            from openai_harmony import (
                Conversation,
                HarmonyEncodingName,
                Message,
                Role,
                SystemContent,
                load_harmony_encoding,
            )
        except ImportError as error:  # pragma: no cover - dependency availability varies
            raise AdapterError(
                "openai_harmony is required for GPT-OSS; install the official package"
            ) from error
        self._Conversation = Conversation
        self._Message = Message
        self._Role = Role
        self._SystemContent = SystemContent
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    @property
    def stop_tokens(self) -> list[int]:
        return list(self.encoding.stop_tokens_for_assistant_actions())

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        converted = []
        for message in messages:
            role_name = message.get("role", "user").lower()
            try:
                role = getattr(self._Role, role_name.upper())
            except AttributeError as error:
                raise AdapterError(f"GPT-OSS does not support message role: {role_name}") from error
            content = message.get("content", "")
            if role_name == "system" and not content:
                content = self._SystemContent.new()
            converted.append(self._Message.from_role_and_content(role, content))
        conversation = self._Conversation.from_messages(converted)
        return list(self.encoding.render_conversation_for_completion(conversation, self._Role.ASSISTANT))

    def decode_text(self, tokens: list[int]) -> str:
        text = self.encoding.decode(tokens)
        return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)

    def decode_completion(self, tokens: list[int]) -> Completion:
        reasoning: list[str] = []
        answer: list[str] = []
        try:
            messages = self.encoding.parse_messages_from_completion_tokens(
                tokens, self._Role.ASSISTANT, strict=False
            )
            for message in messages:
                text = "".join(
                    str(getattr(content, "text", ""))
                    for content in getattr(message, "content", [])
                    if getattr(content, "text", None)
                )
                if getattr(message, "channel", None) == "analysis":
                    reasoning.append(text)
                elif getattr(message, "channel", None) == "final":
                    answer.append(text)
        except (AssertionError, ValueError, TypeError):
            pass
        if reasoning or answer:
            return Completion("".join(reasoning).strip(), "".join(answer).strip())
        return Completion("", self.decode_text(tokens).strip())


class DeepSeekAdapter(ModelAdapter):
    name = "deepseek-v4"

    def __init__(self, model: Path, **options: Any) -> None:
        super().__init__(model, **options)
        _required(self.model, "config.json", "tokenizer.json", "encoding/encoding_dsv4.py")
        self.tokenizer = _load_transformers_tokenizer(self.model)
        module = _load_module(
            self.model / "encoding" / "encoding_dsv4.py",
            f"ncnn_moe_encoding_dsv4_{self.model_fingerprint}",
        )
        if not hasattr(module, "encode_messages") or not hasattr(module, "parse_message_from_completion_text"):
            raise AdapterError("encoding_dsv4.py must export encode_messages and parse_message_from_completion_text")
        self._encode_messages = module.encode_messages
        self._parse_message = module.parse_message_from_completion_text
        if self.tokenizer.eos_token_id != 1:
            raise AdapterError(f"expected DeepSeek V4 EOS token 1, got {self.tokenizer.eos_token_id}")

    @property
    def stop_tokens(self) -> list[int]:
        return [int(self.tokenizer.eos_token_id)]

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        text = self._encode_messages(messages, thinking_mode=self.thinking_mode)
        return list(self.tokenizer.encode(text))

    def decode_text(self, tokens: list[int]) -> str:
        return str(self.tokenizer.decode(tokens, skip_special_tokens=False))

    def decode_completion(self, tokens: list[int]) -> Completion:
        text = self.decode_text(tokens)
        try:
            message = self._parse_message(text, thinking_mode=self.thinking_mode)
            return Completion(
                str(message.get("reasoning_content", "") or "").strip(),
                str(message.get("content", "") or "").strip(),
            )
        except (AssertionError, ValueError, TypeError, KeyError):
            return _split_thinking(text, self.thinking)


class QwenAdapter(ModelAdapter):
    name = "qwen3.6"

    def __init__(self, model: Path, **options: Any) -> None:
        super().__init__(model, **options)
        _required(self.model, "config.json", "tokenizer.json")
        if self.model_type in {"qwen4_exp", "qwen4_exp_text"}:
            self.name = "qwen3.8"
        self.tokenizer = _load_transformers_tokenizer(self.model)
        if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
            raise AdapterError("the official Qwen tokenizer does not provide a chat template")
        stop_tokens: list[int] = []
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, Integral):
            stop_tokens.append(int(tokenizer_eos))
        generation_path = self.model / "generation_config.json"
        if generation_path.is_file():
            generation = _load_json(generation_path)
            configured_eos = generation.get("eos_token_id")
            if isinstance(configured_eos, Integral):
                configured_eos = [configured_eos]
            if isinstance(configured_eos, list):
                for token in configured_eos:
                    if isinstance(token, Integral) and int(token) not in stop_tokens:
                        stop_tokens.append(int(token))
        self._stop_tokens = stop_tokens

    @property
    def stop_tokens(self) -> list[int]:
        configured = getattr(self, "_stop_tokens", None)
        if configured is not None:
            return list(configured)
        value = getattr(self.tokenizer, "eos_token_id", None)
        return [] if value is None else [int(value)]

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        return _normalize_token_ids(encoded)

    def decode_text(self, tokens: list[int]) -> str:
        text = str(self.tokenizer.decode(tokens, skip_special_tokens=True))
        if tokens and tokens[-1] in self.stop_tokens:
            # Qwen may expose an incomplete trailing byte as U+FFFD directly
            # before its EOS token. It is tokenizer framing, not user text.
            text = text.rstrip().rstrip("\ufffd").rstrip()
        return text


def create_adapter(model: Path, **options: Any) -> ModelAdapter:
    adapter = ModelAdapter.detect(model, **options)
    adapter.validate()
    return adapter
