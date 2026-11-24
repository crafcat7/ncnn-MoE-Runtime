"""Focused unit tests for the Python CLI helpers that are pure functions.

The runtime itself is covered by the C++ suite; these tests pin the Gemma-4
marker/stream processing and the generation-default resolution that used to be
hardcoded per model.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import ncnn_moe  # noqa: E402  (path setup must run first)


def make_adapter(**attributes: object) -> ncnn_moe.ModelAdapter:
    adapter = ncnn_moe.ModelAdapter.__new__(ncnn_moe.ModelAdapter)
    for key, value in attributes.items():
        setattr(adapter, key, value)
    return adapter


def generation_arguments(**overrides: object) -> argparse.Namespace:
    values = {"temperature": None, "top_k": None, "top_p": None}
    values.update(overrides)
    return argparse.Namespace(**values)


class Gemma4NormalizeTest(unittest.TestCase):
    def test_keeps_thought_channel_when_thinking(self) -> None:
        adapter = make_adapter(thinking=True)
        text = "<|channel>thought\nreason<channel|>answer"
        self.assertEqual(adapter._gemma4_normalize(text), "<thinkreason</thinkanswer")

    def test_strips_thought_channel_without_thinking(self) -> None:
        adapter = make_adapter(thinking=False)
        text = "before<|channel>thought\nreason<channel|>after"
        self.assertEqual(adapter._gemma4_normalize(text), "beforeafter")


class Gemma4StreamTest(unittest.TestCase):
    CHUNKS = ["<|chan", "nel>thought\nsec", "ret<chan", "nel|>vis", "ible"]

    def _stream(self, thinking: bool) -> str:
        adapter = make_adapter(thinking=thinking)
        streamed: dict[str, object] = {}
        return "".join(adapter._gemma4_stream(chunk, streamed) for chunk in self.CHUNKS)

    def test_hides_thought_channel_when_thinking(self) -> None:
        self.assertEqual(self._stream(False), "visible")

    def test_streams_thought_channel_when_thinking(self) -> None:
        self.assertEqual(self._stream(True), "<thinksecret</thinkvisible")

    def test_non_thought_channel_passes_through(self) -> None:
        adapter = make_adapter(thinking=False)
        streamed: dict[str, object] = {}
        self.assertEqual(adapter._gemma4_stream("<|channel>final\nhello", streamed),
                         "<|channel>final\nhello")


class GenerationDefaultsTest(unittest.TestCase):
    def test_reads_generation_config(self) -> None:
        adapter = make_adapter(
            name="gemma4",
            generation_config={"temperature": 0.7, "top_k": 64, "top_p": 0.9},
        )
        self.assertEqual(ncnn_moe.default_generation_values(adapter, generation_arguments()),
                         (0.7, 64, 0.9))

    def test_cli_overrides_win(self) -> None:
        adapter = make_adapter(
            name="gemma4",
            generation_config={"temperature": 0.7, "top_k": 64, "top_p": 0.9},
        )
        self.assertEqual(
            ncnn_moe.default_generation_values(adapter, generation_arguments(temperature=0.0, top_k=1, top_p=0.5)),
            (0.0, 1, 0.5),
        )

    def test_falls_back_without_generation_config(self) -> None:
        gemma = make_adapter(name="gemma4", generation_config={})
        self.assertEqual(ncnn_moe.default_generation_values(gemma, generation_arguments()), (1.0, 20, 0.95))
        gpt_oss = make_adapter(name="gpt-oss", generation_config={})
        self.assertEqual(ncnn_moe.default_generation_values(gpt_oss, generation_arguments()), (1.0, 0, 1.0))

    def test_rejects_invalid_generation_values(self) -> None:
        adapter = make_adapter(
            name="gemma4",
            generation_config={"temperature": 0, "top_k": -1, "top_p": 2.0},
        )
        self.assertEqual(ncnn_moe.default_generation_values(adapter, generation_arguments()), (1.0, 20, 0.95))

    def test_read_generation_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = make_adapter(_asset_root=root)
            self.assertEqual(adapter._read_generation_config(), {})
            (root / "generation_config.json").write_text(
                json.dumps({"top_k": 64, "top_p": 0.95}), encoding="utf-8")
            self.assertEqual(adapter._read_generation_config(), {"top_k": 64, "top_p": 0.95})
            (root / "generation_config.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(adapter._read_generation_config(), {})


def make_chat_app(adapter: ncnn_moe.ModelAdapter) -> ncnn_moe.ConversationApp:
    app = ncnn_moe.ConversationApp.__new__(ncnn_moe.ConversationApp)
    app.adapter = adapter
    app.arguments = argparse.Namespace(stream=False)
    return app


class Gemma4StopTokenTest(unittest.TestCase):
    @staticmethod
    def _app(thinking: bool) -> ncnn_moe.ConversationApp:
        return make_chat_app(make_adapter(name="gemma4", thinking=thinking, _stop_tokens=[1, 106]))

    def test_stop_token_text_is_not_rendered(self) -> None:
        app = self._app(thinking=True)
        streamed: dict[str, object] = {}
        app._on_generation_event(
            {"event": "token", "token_id": 106, "text": "<turn|>", "is_stop_token": True},
            streamed,
            display=False,
        )
        self.assertFalse(streamed.get("decoded_chunks"))
        self.assertTrue(streamed.get("ended_with_stop_token"))

        app._on_generation_event(
            {"event": "token", "token_id": 50, "text": "answer", "is_stop_token": False},
            streamed,
            display=False,
        )
        self.assertEqual(streamed["decoded_chunks"], ["answer"])

    def test_stop_token_id_without_flag(self) -> None:
        app = self._app(thinking=False)
        streamed: dict[str, object] = {}
        app._on_generation_event(
            {"event": "token", "token_id": 106, "text": "<turn|>"},
            streamed,
            display=False,
        )
        self.assertFalse(streamed.get("decoded_chunks"))
        self.assertTrue(streamed.get("ended_with_stop_token"))

    def test_decode_strips_legacy_turn_marker(self) -> None:
        app = self._app(thinking=False)
        completion = app._decode_completion(
            {"decoded_chunks": ["answer<turn|>"], "ended_with_stop_token": True}
        )
        self.assertEqual(completion.answer, "answer")

    def test_plain_reply_is_answer_when_thinking(self) -> None:
        adapter = make_adapter(name="gemma4", thinking=True, _stop_tokens=[1, 106])
        self.assertEqual(adapter.decode_completion_text("plain answer").answer, "plain answer")


class RuntimeSettingsTest(unittest.TestCase):
    def test_cpu_toggle_settings(self) -> None:
        arguments = argparse.Namespace(cpu_f16_attention="on")
        captured = ncnn_moe.cli_runtime_settings(arguments)
        self.assertEqual(captured["cpu_f16_attention"], "on")
        merged = ncnn_moe.merge_runtime_settings(
            cli={
                "backend": "hybrid",
                "cpu_f16_attention": "off",
            },
            session=None,
            profile=None,
            user=None,
        )
        args = ncnn_moe.runtime_args_from_settings(merged)
        self.assertIn("--hybrid", args)
        self.assertEqual(args[args.index("--cpu-f16-attention") + 1], "off")

    def test_rejects_unknown_cpu_toggle_values(self) -> None:
        with self.assertRaises(ValueError):
            ncnn_moe.runtime_args_from_settings({"cpu_f16_attention": "maybe"})


if __name__ == "__main__":
    unittest.main()
