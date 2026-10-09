#!/usr/bin/env python3
"""Dependency-free tests for the native-text worker client and chat flow."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncnn_moe_adapters import (  # noqa: E402
    AdapterError,
    Completion,
    NATIVE_TEXT_VERSION,
    create_adapter,
)
from ncnn_moe import (  # noqa: E402
    ConversationApp,
    _format_bytes_gb,
    _format_runtime_metrics,
    configure_standard_streams,
    default_worker_path,
    find_worker,
    load_adapter,
    parse_arguments,
)
from ncnn_moe_protocol import WorkerClient, WorkerError  # noqa: E402
from ncnn_moe_state import runtime_args_from_settings  # noqa: E402
import benchmark_prompt  # noqa: E402


FAMILY_CASES = (
    ("gpt_oss", "gpt-oss", [200002, 200012]),
    ("deepseek_v4", "deepseek-v4", [1]),
    ("qwen3_5_moe", "qwen3.6", [248046, 248044]),
    ("qwen4_exp", "qwen3.8", [248046, 248044]),
)


def _model(root: Path, model_type: str, *, context: int = 2048) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "text_config": {"max_position_embeddings": context},
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def _ready(stop_tokens: list[int] | None = None, *, supported: bool = True) -> dict[str, object]:
    model: dict[str, object] = {
        "native_text_supported": supported,
        "native_text_version": NATIVE_TEXT_VERSION,
        "native_stop_tokens": [1] if stop_tokens is None else stop_tokens,
        "max_context_tokens": 4096,
    }
    return {"event": "ready", "model": model}


def _arguments(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "system": "",
        "title": "",
        "seed": 7,
        "prefill_chunk_size": 8,
        "no_speculative": False,
        "max_new_tokens": 8,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "min_p": 0.0,
        "speculative_confidence": 0.5,
        "speculative_max_draft": 0,
        "metrics_enabled": False,
        "metrics_interval_ms": 0,
        "stream": False,
        "stream_final_only": False,
        "show_reasoning": True,
        "context_tokens": 0,
        "no_thinking": False,
        "verbose": False,
        "ephemeral": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []
        self._next_id = 0

    def new_id(self) -> str:
        self._next_id += 1
        return f"session-{self._next_id}"

    def save(self, record: dict[str, object]) -> None:
        self.saved.append(copy.deepcopy(record))

    def save_profile(self, *_: object) -> None:
        pass

    def list(self) -> list[dict[str, object]]:
        return []


def _message_count(messages: list[dict[str, str]]) -> int:
    return sum(3 + max(1, (len(message.get("content", "")) + 3) // 4) for message in messages)


class FakeWorkerClient:
    def __init__(self, *, response: list[str | tuple[int, str]] | None = None) -> None:
        self.ready = {
            "model": {"max_context_tokens": 4096},
            "resources": {"backend": "cpu"},
        }
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.compact_payloads: list[tuple[str, list[dict[str, str]], bool, int | None]] = []
        self.generate_payloads: list[list[dict[str, str]]] = []
        self.stats_payloads: list[list[dict[str, str]] | None] = []
        self.resets: list[str] = []
        self.responses: list[list[str | tuple[int, str]]] = [response] if response is not None else []

    def create_session(self, session_id: str, **_: object) -> dict[str, object]:
        self.sessions[session_id] = []
        return {"event": "session_created", "session_id": session_id}

    def reset(self, session_id: str) -> dict[str, object]:
        self.resets.append(session_id)
        self.sessions[session_id] = []
        return {"event": "reset", "session_id": session_id}

    def compact(
        self,
        session_id: str,
        replay_tokens: list[int] | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool | None = None,
        context_tokens: int | None = None,
    ) -> dict[str, object]:
        del replay_tokens
        copied = copy.deepcopy(messages or [])
        self.compact_payloads.append((session_id, copied, bool(enable_thinking), context_tokens))
        self.sessions[session_id] = copied
        count = _message_count(copied)
        return {"event": "compacted", "sequence_length": count, "replayed_tokens": count}

    def stats(
        self,
        session_id: str,
        *,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, object]:
        del enable_thinking
        self.stats_payloads.append(copy.deepcopy(messages))
        current = messages if messages is not None else self.sessions.get(session_id, [])
        return {
            "event": "stats",
            "prompt_count": _message_count(current),
            "sequence_length": _message_count(self.sessions.get(session_id, [])),
        }

    def generate(
        self,
        session_id: str,
        prompt_tokens: list[int] | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        enable_thinking: bool | None = None,
        context_tokens: int | None = None,
        on_event: object = None,
        **_: object,
    ) -> tuple[dict[str, object], list[int]]:
        del prompt_tokens, enable_thinking, context_tokens
        assert messages is not None
        previous = self.sessions.get(session_id, [])
        reused = bool(previous) and messages[: len(previous)] == previous
        copied = copy.deepcopy(messages)
        self.generate_payloads.append(copied)
        pieces = self.responses.pop(0) if self.responses else ["answer"]
        if callable(on_event):
            for index, piece in enumerate(pieces, start=1):
                token_id, text = piece if isinstance(piece, tuple) else (300000 + index, piece)
                on_event({"event": "token", "token_id": token_id, "text": text})
        text_pieces = [piece[1] if isinstance(piece, tuple) else piece for piece in pieces]
        count = _message_count(copied) + len(pieces)
        self.sessions[session_id] = copied + [{"role": "assistant", "content": "".join(text_pieces)}]
        return (
            {
                "event": "done",
                "sequence_length": count,
                "prefix_reused": reused,
                "generated_tokens": len(pieces),
                "metrics": {"input_tokens": _message_count(copied)},
            },
            [piece[0] if isinstance(piece, tuple) else 300000 + index for index, piece in enumerate(pieces, start=1)],
        )


class AdapterTests(unittest.TestCase):
    def test_four_model_families_need_no_python_tokenizer_packages(self) -> None:
        original_import = __import__

        def import_without_tokenizer(name: str, *args: object, **kwargs: object) -> ModuleType:
            if name == "transformers" or name == "openai_harmony" or name.startswith("transformers."):
                raise AssertionError(f"Python tokenizer dependency imported: {name}")
            return original_import(name, *args, **kwargs)

        arguments = _arguments()
        for command in ("run", "chat", "tune"):
            with self.subTest(command=command):
                parsed = parse_arguments([command, "--no-thinking"])
                self.assertTrue(parsed.no_thinking)
                self.assertFalse(parse_arguments([command]).no_thinking)
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-native-adapters-") as directory:
            root = Path(directory)
            for model_type, expected_name, expected_stops in FAMILY_CASES:
                with self.subTest(model_type=model_type), patch("builtins.__import__", import_without_tokenizer):
                    model = _model(root / model_type, model_type)
                    adapter = load_adapter(arguments, model, _ready(expected_stops))
                    self.assertEqual(adapter.name, expected_name)
                    self.assertEqual(adapter.stop_tokens, expected_stops)
                    self.assertEqual(adapter.context_limit, 2048)
                    self.assertTrue(adapter.thinking)
                    self.assertFalse(load_adapter(_arguments(no_thinking=True), model, _ready(expected_stops)).thinking)

    def test_native_unavailable_or_wrong_protocol_fails_explicitly(self) -> None:
        arguments = _arguments()
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-native-unavailable-") as directory:
            model = _model(Path(directory), "gpt_oss")
            cases = (
                (_ready(supported=False), "tokenizer assets"),
                ({"event": "ready", "model": {"native_text_supported": True}}, "version"),
                (
                    {"event": "ready", "model": {
                        "native_text_supported": True,
                        "native_text_version": "ncnn-moe-text-v1",
                        "native_stop_tokens": [1],
                    }},
                    NATIVE_TEXT_VERSION,
                ),
                (_ready([1, True]), "stop-token"),
                (_ready([]), "stop-token"),
            )
            for ready, expected in cases:
                with self.subTest(ready=ready), self.assertRaisesRegex(AdapterError, expected):
                    load_adapter(arguments, model, ready)

    def test_gpt_completion_requires_token_id_aware_parser(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-harmony-") as directory:
            model = _model(Path(directory), "gpt_oss")
            adapter = create_adapter(model, thinking=False)
            with self.assertRaisesRegex(AdapterError, "native token events"):
                adapter.decode_completion_text("<|return|> is ordinary user-visible text")

    def test_deepseek_and_qwen_text_modes_and_unicode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-text-modes-") as directory:
            root = Path(directory)
            deepseek = create_adapter(_model(root / "ds", "deepseek_v4"), thinking=True)
            self.assertEqual(
                deepseek.decode_completion_text("I should reason.</think>\nAnswer"),
                Completion("I should reason.", "Answer"),
            )
            self.assertEqual(
                deepseek.decode_completion_text("<｜Assistant｜> is body</think>Answer"),
                Completion("<｜Assistant｜> is body", "Answer"),
            )
            chat = create_adapter(_model(root / "chat", "deepseek_v4"), thinking=False)
            self.assertEqual(
                chat.decode_completion_text("</think>\nPlain answer"),
                Completion("", "</think>\nPlain answer"),
            )
            qwen = create_adapter(_model(root / "qwen", "qwen3_5_moe"), thinking=True)
            self.assertEqual(qwen.decode_completion_text("still reasoning"), Completion("still reasoning", ""))
            self.assertEqual(
                qwen.decode_completion_text("body <think> is literal"),
                Completion("body <think> is literal", ""),
            )
            self.assertEqual(
                qwen.decode_completion_text("reason</think>  answer  "),
                Completion("reason", "  answer  "),
            )
            plain = create_adapter(_model(root / "plain", "qwen4_exp"), thinking=False)
            self.assertEqual(plain.decode_completion_text("你好🙂"), Completion("", "你好🙂"))

    def test_fingerprints_expose_previous_versions_and_protect_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-fingerprint-") as directory:
            root = Path(directory)
            for model_type, _, _ in FAMILY_CASES:
                with self.subTest(model_type=model_type):
                    family = create_adapter(_model(root / model_type, model_type))
                    self.assertIn(family.legacy_model_fingerprints[-1], family.legacy_model_fingerprints)
                    self.assertNotIn(family.model_fingerprint, family.legacy_model_fingerprints)
            model = _model(root / "qwen", "qwen3_5_moe")
            (model / "tokenizer_config.json").write_text('{"eos_token":"x"}', encoding="utf-8")
            original = create_adapter(model)
            self.assertEqual(len(original.legacy_model_fingerprints), 2)
            self.assertEqual(original.legacy_model_fingerprints[-1], original.legacy_model_fingerprints[1])
            self.assertNotEqual(original.model_fingerprint, original.legacy_model_fingerprints[-1])

            changed_model = _model(root / "changed", "qwen3_5_moe", context=1024)
            changed = create_adapter(changed_model)
            self.assertNotEqual(original.model_fingerprint, changed.model_fingerprint)
            self.assertTrue(set(original.legacy_model_fingerprints).isdisjoint(changed.legacy_model_fingerprints))


class ProtocolTests(unittest.TestCase):
    class QueuedClient(WorkerClient):
        def __init__(self, events: list[dict[str, object]], *, interrupt: bool = False) -> None:
            self.events = list(events)
            self.sent: list[dict[str, object]] = []
            self.interrupt = interrupt

        def _send(self, payload: dict[str, object]) -> None:
            self.sent.append(payload)

        def _read_event(self) -> dict[str, object]:
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt
            return self.events.pop(0)

    @staticmethod
    def _failing_callback(_: dict[str, object]) -> None:
        raise RuntimeError("callback failed")

    def test_console_codec_and_redirected_input_are_frontend_owned(self) -> None:
        text = "中文 café 😀"
        for tty, explicit_codec in ((True, False), (False, False), (False, True)):
            with self.subTest(tty=tty, explicit_codec=explicit_codec):
                encoding = "gbk" if tty or explicit_codec else "utf-8"
                stdin = io.TextIOWrapper(io.BytesIO("中文\n".encode(encoding)), encoding="gbk")
                stdout = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
                stderr = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
                try:
                    with (
                        patch.object(sys, "stdin", stdin),
                        patch.object(sys, "stdout", stdout),
                        patch.object(sys, "stderr", stderr),
                        patch.object(stdin, "isatty", return_value=tty),
                        patch.object(stdout, "isatty", return_value=tty),
                        patch.object(stderr, "isatty", return_value=tty),
                        patch.dict(os.environ, {"PYTHONIOENCODING": "gbk"} if explicit_codec else {}, clear=True),
                    ):
                        configure_standard_streams()
                        self.assertEqual(stdin.readline(), "中文\n")
                        for stream in (stdout, stderr):
                            stream.write(text)
                            stream.flush()
                            self.assertEqual(stream.buffer.getvalue(), text.encode(encoding, errors="backslashreplace"))
                finally:
                    for stream in (stdin, stdout, stderr):
                        stream.close()

        with io.TextIOWrapper(io.BytesIO(b"\xc4\xe3\xba\xc3"), encoding="gbk") as stdin:
            with (
                patch.object(sys, "stdin", stdin),
                patch.object(sys, "stdout", io.StringIO()),
                patch.object(sys, "stderr", io.StringIO()),
                patch.dict(os.environ, {}, clear=True),
            ):
                configure_standard_streams()
                with self.assertRaises(UnicodeDecodeError):
                    stdin.read()

        with (
            patch.object(sys, "stdin", io.StringIO()),
            patch.object(sys, "stdout", io.StringIO()),
            patch.object(sys, "stderr", io.StringIO()),
        ):
            configure_standard_streams()

    def test_worker_protocol_preserves_utf8_and_rejects_invalid_text(self) -> None:
        text = "中文 café 😀"
        ready = json.dumps({"event": "ready", "text": text}, ensure_ascii=False) + "\n"
        sent = io.BytesIO()
        process = SimpleNamespace(
            stdin=io.TextIOWrapper(sent, encoding="utf-8", errors="strict"),
            stdout=io.TextIOWrapper(io.BytesIO(ready.encode("utf-8")), encoding="utf-8", errors="strict"),
            poll=lambda: 0,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("ncnn_moe_protocol.subprocess.Popen", return_value=process) as popen:
                client = WorkerClient(Path(sys.executable), Path(directory))
            try:
                self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
                self.assertEqual(popen.call_args.kwargs["errors"], "strict")
                self.assertEqual(client.ready["text"], text)
                payload = {"op": "generate", "messages": [{"role": "user", "content": text}]}
                client._send(payload)
                wire = sent.getvalue()
                self.assertIn(text.encode("utf-8"), wire)
                self.assertEqual(json.loads(wire.decode("utf-8")), payload)
                with self.assertRaisesRegex(WorkerError, "valid UTF-8"):
                    client._send({"op": "generate", "messages": [{"role": "user", "content": "\ud800"}]})
                self.assertEqual(sent.getvalue(), wire)
                process.stdout.close()
                process.stdout = io.TextIOWrapper(io.BytesIO(b'{"event":"token","text":"\xff"}\n'), encoding="utf-8")
                with self.assertRaisesRegex(WorkerError, "invalid UTF-8"):
                    client._read_event()
            finally:
                client.close()

    def test_callback_failure_drains_done_and_keeps_next_request_synchronized(self) -> None:
        normal = self.QueuedClient(
            [
                {"event": "token", "token_id": 1, "text": "a"},
                {"event": "done"},
                {"event": "stats", "sequence_length": 1},
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            normal.generate("normal", [0], on_event=self._failing_callback)
        self.assertEqual(normal.stats("normal")["sequence_length"], 1)
        self.assertFalse(normal.events)

        drain = self.QueuedClient(
            [
                {"event": "cancel_requested", "request_id": "interrupt"},
                {"event": "token", "token_id": 2, "text": "b"},
                {"event": "done"},
                {"event": "stats", "sequence_length": 2},
            ],
            interrupt=True,
        )
        with self.assertRaisesRegex(WorkerError, "generation cancelled by user"):
            drain.generate(
                "drain",
                [0],
                request_id="interrupt",
                on_event=self._failing_callback,
            )
        self.assertEqual(drain.sent[1], {"op": "cancel", "request_id": "interrupt"})
        self.assertEqual(drain.stats("drain")["sequence_length"], 2)
        self.assertFalse(drain.events)

    def test_messages_protocol_has_enable_thinking_and_uses_runtime_stops(self) -> None:
        client = self.QueuedClient(
            [
                {"event": "token", "token_id": 5, "text": "hi"},
                {"event": "done"},
            ]
        )
        client.generate(
            "text",
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=False,
            stop_tokens=[99],
        )
        payload = client.sent[0]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertIs(payload["enable_thinking"], False)
        self.assertNotIn("stop_tokens", payload)


class BenchmarkTests(unittest.TestCase):
    def test_direct_token_benchmark_does_not_require_native_text(self) -> None:
        class Client:
            def __init__(self, stops: list[int] | None) -> None:
                model: dict[str, object] = {"native_text_supported": False}
                if stops is not None:
                    model["native_stop_tokens"] = stops
                self.ready = {"model": model}
                self.call: tuple[str, list[int], dict[str, object]] | None = None

            def __enter__(self) -> Client:
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def create_session(self, *_: object, **__: object) -> None:
                pass

            def generate(
                self,
                session_id: str,
                prompt_tokens: list[int],
                **options: object,
            ) -> tuple[dict[str, object], list[int]]:
                self.call = (session_id, list(prompt_tokens), dict(options))
                return (
                    {
                        "prompt_tok_per_second": 1.0,
                        "generation_tok_per_second": 2.0,
                        "elapsed_seconds": 1.0,
                        "ttft_microseconds": 1000,
                        "tpot_microseconds": 2000,
                        "generated_tokens": 1,
                        "metrics": {"expert": {}, "gpu": {}, "cpu": {}},
                    },
                    [4],
                )

        with tempfile.TemporaryDirectory(prefix="ncnn-moe-token-benchmark-") as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                model=root / "model-without-text-assets",
                worker=root / "worker",
                backend="cpu",
                vulkan_device=0,
                host_memory_mb=0,
                expert_cache_mb=0,
                expert_io_workers=0,
                expert_gpu_cache_mb=0,
                expert_gpu_victim_cache_mb=0,
                prompt="unused",
                prompt_token_ids=[0, 1, 2],
                max_new_tokens=1,
                warmup=0,
                runs=1,
                prefill_chunk_size=8,
                expert_memory="auto",
                enable_speculative=False,
                json_output=True,
            )
            for ready_stops in (None, [88, 88]):
                client = Client(ready_stops)
                with (
                    patch.object(benchmark_prompt, "_parse_args", return_value=arguments),
                    patch.object(benchmark_prompt, "WorkerClient", return_value=client),
                    patch.object(
                        benchmark_prompt,
                        "load_adapter",
                        side_effect=AssertionError("direct token IDs must bypass text adapters"),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(benchmark_prompt.main(), 0)
                assert client.call is not None
                self.assertEqual(client.call[1], [0, 1, 2])
                self.assertEqual(client.call[2].get("stop_tokens"), [])


class ConversationTests(unittest.TestCase):
    def _app(
        self,
        model_type: str,
        *,
        response: list[str | tuple[int, str]] | None = None,
        record: dict[str, object] | None = None,
        arguments: SimpleNamespace | None = None,
    ) -> tuple[ConversationApp, FakeWorkerClient, FakeStore]:
        temporary = tempfile.TemporaryDirectory(prefix="ncnn-moe-conversation-")
        self.addCleanup(temporary.cleanup)
        model = _model(Path(temporary.name), model_type, context=4096)
        stop_tokens = next(stops for family, _, stops in FAMILY_CASES if family == model_type)
        adapter = create_adapter(
            model,
            thinking=not (arguments or _arguments()).no_thinking,
            native_stop_tokens=stop_tokens,
        )
        client = FakeWorkerClient(response=response)
        store = FakeStore()
        with contextlib.redirect_stderr(io.StringIO()):
            app = ConversationApp(
                adapter=adapter,
                client=client,  # type: ignore[arg-type]
                store=store,  # type: ignore[arg-type]
                arguments=arguments or _arguments(),
                settings={},
                record=record,
                ephemeral=True,
            )
        app._status = lambda _: None  # type: ignore[method-assign]
        return app, client, store

    def test_chat_accepts_text_streams_with_independent_source_encodings(self) -> None:
        text = "中文 hello"
        for encoding in ("utf-8", "gbk", "utf-16", "shift_jis"):
            with self.subTest(encoding=encoding):
                app, client, _ = self._app("qwen3_5_moe", response=["answer"], arguments=_arguments(stream=False))
                with io.TextIOWrapper(io.BytesIO((text + "\r\n/exit\r\n").encode(encoding)), encoding=encoding) as source:
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        patch.object(app, "prompt_session_class", side_effect=AssertionError("input source invoked terminal UI")),
                        patch("builtins.input", side_effect=AssertionError("input source invoked console input")),
                    ):
                        self.assertEqual(app.chat(source), 0)
                    self.assertFalse(source.closed)
                self.assertEqual(client.generate_payloads, [[{"role": "user", "content": text}]])
                self.assertEqual(app.messages[-1], {"role": "assistant", "content": "answer"})

    def test_chat_handles_redirected_eof_and_rejects_undecoded_input(self) -> None:
        app, client, _ = self._app("qwen3_5_moe", response=["answer"], arguments=_arguments(stream=False))
        with (
            patch.object(sys, "stdin", io.StringIO("中文 hello")),
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(app, "prompt_session_class", side_effect=AssertionError("pipe invoked terminal UI")),
            patch("builtins.input", side_effect=AssertionError("pipe invoked console input")),
        ):
            self.assertEqual(app.chat(), 0)
        self.assertEqual(client.generate_payloads, [[{"role": "user", "content": "中文 hello"}]])

        for source in (io.BytesIO(b"hello"), io.TextIOWrapper(io.BytesIO(b"\xc4\xe3\xba\xc3"), encoding="utf-8")):
            app, client, _ = self._app("qwen3_5_moe")
            with source, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises((ValueError, UnicodeDecodeError)):
                    app.chat(source)  # type: ignore[arg-type]
            self.assertFalse(client.generate_payloads)

    def test_resume_compact_reset_context_and_qwen_system_merge(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-resume-model-") as directory:
            model = _model(Path(directory), "qwen3_5_moe")
            adapter = create_adapter(model)
            record: dict[str, object] = {
                "id": "old-session",
                "title": "Saved",
                "model": str(model),
                "model_type": "qwen3_5_moe",
                "model_fingerprint": adapter.legacy_model_fingerprints[0],
                "messages": [
                    {"role": "system", "content": "base instructions"},
                    {"role": "system", "content": "extra instructions"},
                    {"role": "system", "content": "Conversation summary:\nprior summary"},
                    {"role": "user", "content": "old question one"},
                    {"role": "assistant", "content": "old answer one"},
                    {"role": "user", "content": "old question two"},
                    {"role": "assistant", "content": "old answer two"},
                    {"role": "user", "content": "old question three"},
                    {"role": "assistant", "content": "old answer three"},
                ],
                "summary": "prior summary",
                "settings": {},
            }
            client = FakeWorkerClient()
            store = FakeStore()
            with contextlib.redirect_stderr(io.StringIO()):
                app = ConversationApp(
                    adapter=adapter,
                    client=client,  # type: ignore[arg-type]
                    store=store,  # type: ignore[arg-type]
                    arguments=_arguments(system="configured system"),
                    settings={},
                    record=record,
                    ephemeral=False,
                )
            app._status = lambda _: None  # type: ignore[method-assign]
            self.assertEqual(client.compact_payloads[0][0], "main")
            self.assertIn(
                "base instructions\n\nextra instructions\n\nConversation summary:\nprior summary",
                app.messages[0]["content"],
            )
            self.assertEqual(record["model_fingerprint"], adapter.model_fingerprint)
            self.assertGreater(record["context_token_count"], 0)

            self.assertEqual(app._prompt_count("main", app.messages), _message_count(app.messages))
            app._summarize = lambda: "fresh summary"  # type: ignore[method-assign]
            app.compact(announce=False)
            self.assertEqual(app.summary, "fresh summary")
            self.assertEqual(len([message for message in app.messages if message["role"] != "system"]), 4)
            self.assertNotIn("old question one", app.messages[1]["content"])

            app.command("/reset")
            self.assertEqual(client.resets[-1], "main")
            self.assertEqual([message["role"] for message in app.messages], ["system"])
            self.assertEqual(app.native_context_count, 0)
            app.new_session()
            self.assertEqual(app.messages, [{"role": "system", "content": "configured system"}])
            self.assertEqual(app.native_context_count, 0)
            compact_calls = len(client.compact_payloads)
            app.compact(announce=False)
            self.assertEqual(client.resets[-1], app.native_session_id)
            self.assertEqual(len(client.compact_payloads), compact_calls)
            self.assertEqual(app.native_context_count, 0)

    def test_send_does_not_preflight_prompt_stats(self) -> None:
        app, client, _ = self._app("qwen3_5_moe")
        self.assertEqual(client.stats_payloads, [])
        app.send("question")
        self.assertEqual(client.stats_payloads, [])
        self.assertEqual(len(client.generate_payloads), 1)

    def test_streaming_handles_harmony_thinking_without_open_marker_and_unicode(self) -> None:
        harmony_events = [
            (200006, "<|start|>"),
            (300001, "assistant"),
            (200005, "<|channel|>"),
            (300002, "analysis"),
            (200008, "<|message|>"),
            (300003, "assistant is a role. literal <|return|>\n你好。"),
            (200006, "\ufffd<|start|>"),
            (300004, "assistant"),
            (200005, "<|channel|>"),
            (300005, "final"),
            (200008, "<|message|>"),
            (300006, "Final text, including assistant as prose."),
            (200002, "\ufffd<|return|>"),
        ]
        app, _, _ = self._app(
            "gpt_oss",
            response=harmony_events,
            arguments=_arguments(stream=True, no_thinking=True, show_reasoning=True),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(
            completion,
            Completion(
                "assistant is a role. literal <|return|>\n你好。\ufffd",
                "Final text, including assistant as prose.\ufffd",
            ),
        )
        self.assertEqual(
            output.getvalue().rstrip("\n"),
            "[reasoning]\nassistant is a role. literal <|return|>\n你好。\ufffd"
            "\n[answer]\nFinal text, including assistant as prose.\ufffd",
        )

        # Non-streamed decoding must use event IDs too: marker-looking BPE text
        # inside the body remains ordinary model output.
        app, _, _ = self._app("gpt_oss", response=harmony_events)
        completion, _, _ = app.send("question")
        self.assertEqual(completion.answer, "Final text, including assistant as prose.\ufffd")
        self.assertIn("literal <|return|>", completion.reasoning)

        deepseek_events = [
            (300001, "  consider"),
            (300002, " the question\ufffd</thi"),
            (300003, "nk>\r\n  Answer"),
            (1, "\ufffd<｜end▁of▁sentence｜>"),
        ]
        app, _, _ = self._app(
            "deepseek_v4",
            response=deepseek_events,
            arguments=_arguments(stream=True),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(
            completion,
            Completion("consider the question\ufffd", "Answer\ufffd"),
        )
        self.assertEqual(
            output.getvalue().rstrip("\n"),
            "[reasoning]\nconsider the question\ufffd\n[answer]\nAnswer\ufffd",
        )
        app, _, _ = self._app("deepseek_v4", response=deepseek_events)
        replayed, _, _ = app.send("question")
        self.assertEqual(replayed, completion)

        app, _, _ = self._app(
            "deepseek_v4",
            response=["reasoning without an opening or closing marker"],
            arguments=_arguments(stream=True),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(completion, Completion("reasoning without an opening or closing marker", ""))
        self.assertEqual(
            output.getvalue().rstrip("\n"),
            "[reasoning]\nreasoning without an opening or closing marker",
        )

        app, _, _ = self._app(
            "deepseek_v4",
            response=["  ", "plain", " answer  "],
            arguments=_arguments(stream=True, no_thinking=True),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(completion, Completion("", "plain answer"))
        self.assertEqual(output.getvalue().rstrip("\n"), "[answer]\nplain answer")

        app, _, _ = self._app(
            "qwen4_exp",
            response=["你", "好", "🙂"],
            arguments=_arguments(stream=True, no_thinking=True),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(completion, Completion("", "你好🙂"))
        self.assertEqual(output.getvalue().rstrip("\n"), "[answer]\n你好🙂")

        qwen_events = [
            (300001, "  reason\ufffd</thi"),
            (300002, "nk>\r\n  answer\ufffd \t"),
            (248046, ""),
        ]
        app, _, _ = self._app("qwen3_5_moe", response=qwen_events, arguments=_arguments(stream=True))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(completion, Completion("reason\ufffd", "  answer"))
        self.assertEqual(
            output.getvalue().rstrip("\n"),
            "[reasoning]\nreason\ufffd\n[answer]\n  answer",
        )
        app, _, _ = self._app("qwen3_5_moe", response=qwen_events)
        replayed, _, _ = app.send("question")
        self.assertEqual(replayed, completion)

        qwen_max_token_events = [
            (300001, "  reason\ufffd</thi"),
            (300002, "nk>\r\n  answer\ufffd "),
            (300003, "\t"),
        ]
        app, _, _ = self._app(
            "qwen3_5_moe",
            response=qwen_max_token_events,
            arguments=_arguments(stream=True),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            completion, _, _ = app.send("question")
        self.assertEqual(completion, Completion("reason\ufffd", "  answer\ufffd \t"))
        self.assertEqual(
            output.getvalue(),
            "[reasoning]\nreason\ufffd\n[answer]\n  answer\ufffd \t\n",
        )

    def test_only_matching_legacy_fingerprints_migrate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-legacy-check-") as directory:
            root = Path(directory)
            original_model = _model(root / "original", "deepseek_v4")
            original_adapter = create_adapter(original_model)
            changed_model = _model(root / "changed", "deepseek_v4", context=1024)
            changed_adapter = create_adapter(changed_model)
            old_fingerprint = original_adapter.legacy_model_fingerprints[-1]
            self.assertNotIn(old_fingerprint, changed_adapter.legacy_model_fingerprints)

            record: dict[str, object] = {
                "id": "foreign",
                "title": "Foreign",
                "model_fingerprint": old_fingerprint,
                "messages": [{"role": "user", "content": "saved"}],
                "settings": {},
            }
            client = FakeWorkerClient()
            store = FakeStore()
            with contextlib.redirect_stderr(io.StringIO()):
                app = ConversationApp(
                    adapter=changed_adapter,
                    client=client,  # type: ignore[arg-type]
                    store=store,  # type: ignore[arg-type]
                    arguments=_arguments(),
                    settings={},
                    record=record,
                    ephemeral=True,
                )
            self.assertEqual(record["model_fingerprint"], old_fingerprint)
            app.native_context_count = 100
            app._save()
            self.assertEqual(record["model_fingerprint"], old_fingerprint)

    def test_each_family_migrates_its_legacy_fingerprint_after_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-family-fingerprint-migration-") as directory:
            root = Path(directory)
            for model_type, _, _ in FAMILY_CASES:
                with self.subTest(model_type=model_type):
                    model = _model(root / model_type, model_type)
                    adapter = create_adapter(model)
                    record: dict[str, object] = {
                        "id": model_type,
                        "title": "Saved",
                        "model_fingerprint": adapter.legacy_model_fingerprints[0],
                        "messages": [{"role": "user", "content": "saved turn"}],
                        "settings": {},
                    }
                    with contextlib.redirect_stderr(io.StringIO()):
                        ConversationApp(
                            adapter=adapter,
                            client=FakeWorkerClient(),  # type: ignore[arg-type]
                            store=FakeStore(),  # type: ignore[arg-type]
                            arguments=_arguments(),
                            settings={},
                            record=record,
                            ephemeral=True,
                        )
                    self.assertEqual(record["model_fingerprint"], adapter.model_fingerprint)


def _runtime_smoke(worker: Path, model: Path, *, auto: bool) -> None:
    runtime_args = [] if auto else ["--cpu"]
    with WorkerClient(worker, model, runtime_args) as client:
        assert client.ready["event"] == "ready"
        assert client.ready["resources"]["backend"] in {"cpu", "hybrid"}
        assert "provider" in client.ready.get("telemetry", {})
        assert "reason" in client.ready.get("telemetry", {})
        try:
            client.request({"op": "unknown"}, expected="error")
        except WorkerError as error:
            assert error.event.get("code") == "invalid_request"
        else:
            raise AssertionError("unknown worker operation did not return an error")

        client.create_session("first", enable_speculative_context=False)
        for invalid_text in (b"\xc4\xe3\xba\xc3", b"\xed\xa0\x80", b"\xf4\x90\x80\x80"):
            request = b'{"op":"stats","session_id":"first","messages":[{"role":"user","content":"' + invalid_text + b'"}]}\n'
            client.process.stdin.buffer.write(request)
            client.process.stdin.buffer.flush()
            error = client._read_event()
            assert error["event"] == "error" and error["code"] == "invalid_request", error
            assert "UTF-8" in error["message"], error
            assert client.stats("first")["sequence_length"] == 0
        compacted = client.compact("first", [0, 1])
        assert compacted["replayed_tokens"] == 2
        vocabulary_size = int(client.ready["model"]["vocabulary_size"])
        for invalid_token in (-1, vocabulary_size):
            try:
                client.compact("first", [invalid_token])
            except WorkerError as error:
                assert error.event.get("code") == "invalid_request"
            else:
                raise AssertionError("compact accepted a token outside the vocabulary")
            assert client.stats("first")["sequence_length"] == 2
        empty = client.compact("first", [])
        assert empty["replayed_tokens"] == 0
        assert empty["sequence_length"] == 0

        events: list[dict[str, object]] = []
        done, tokens = client.generate(
            "first",
            [0],
            request_id="python-smoke",
            max_new_tokens=3,
            temperature=0.0,
            enable_speculative=False,
            metrics_enabled=False,
            metrics_interval_ms=1,
            on_event=events.append,
        )
        assert done["event"] == "done"
        assert len(tokens) == 3
        assert not any(event.get("event") == "metrics" for event in events)
        assert "prompt_tok_per_second" in done["metrics"]
        assert "generation_tok_per_second" in done["metrics"]
        assert "expert" in done["metrics"]
        assert "ttft_microseconds" in done["metrics"]
        assert "tpot_microseconds" in done["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--auto", action="store_true")
    arguments = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if arguments.worker is None and arguments.model is None:
        return 0
    if arguments.worker is None or arguments.model is None:
        parser.error("--worker and --model must be provided together")
    _runtime_smoke(arguments.worker, arguments.model, auto=arguments.auto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
