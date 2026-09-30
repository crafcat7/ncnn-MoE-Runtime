#!/usr/bin/env python3
"""Small dependency-light smoke test for the unified worker and GPT-OSS adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import UserDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncnn_moe_adapters import (  # noqa: E402
    Completion,
    QwenAdapter,
    _normalize_token_ids,
    create_adapter,
)
from ncnn_moe import (  # noqa: E402
    ConversationApp,
    _format_bytes_gb,
    _format_runtime_metrics,
    default_worker_path,
    find_worker,
    parse_arguments,
)
from ncnn_moe_protocol import WorkerClient, WorkerError  # noqa: E402
from ncnn_moe_state import runtime_args_from_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--auto", action="store_true")
    arguments = parser.parse_args()

    assert _format_bytes_gb(1_000_000_000) == "1.00 GB"
    assert _format_bytes_gb(None) == "N/A"
    formatted_metrics = _format_runtime_metrics(
        {
            "prompt_tok_per_second": 3.7,
            "generation_tok_per_second": 13.33,
            "tpot_microseconds": 75_030.0,
            "gpu": {
                "available": True,
                "kernel_time_available": False,
                "reason": "gpu_expert_execution_not_observed",
            },
            "gpu_device": {},
        }
    )
    assert "Prompt: 3.70 t/s | Generation: 13.33 t/s" in formatted_metrics
    assert "kernel N/A (no GPU Expert execution)" in formatted_metrics

    defaults = parse_arguments(["run", "--model", "model", "--prompt", "hello"])
    assert defaults.stream is True
    assert defaults.show_reasoning is True
    assert defaults.metrics_enabled is False
    overrides = parse_arguments(
        [
            "run",
            "--model",
            "model",
            "--prompt",
            "hello",
            "--no-stream",
            "--hide-reasoning",
            "--metrics",
        ]
    )
    assert overrides.stream is False
    assert overrides.show_reasoning is False
    assert overrides.metrics_enabled is True

    assert runtime_args_from_settings({"backend": "auto"}) == []
    assert runtime_args_from_settings({"backend": "cpu"}) == ["--cpu"]
    assert runtime_args_from_settings({"backend": "hybrid"}) == ["--hybrid"]
    assert runtime_args_from_settings({"cpu_packed_weights": "off"}) == [
        "--cpu-packed-weights",
        "off",
    ]
    assert runtime_args_from_settings({"cpu_packed_weights": "on"}) == [
        "--cpu-packed-weights",
        "on",
    ]
    try:
        runtime_args_from_settings({"cpu_packed_weights": "auto"})
    except ValueError as error:
        assert "must be off or on" in str(error)
    else:
        raise AssertionError("automatic CPU packed weights must not be accepted")
    try:
        runtime_args_from_settings({"backend": "vulkan"})
    except ValueError as error:
        assert "unknown backend" in str(error)
    else:
        raise AssertionError("Vulkan-only backend must not be accepted from saved settings")

    class FakeQwenTokenizer:
        eos_token_id = 99

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "answer\ufffd" if tokens and tokens[-1] == self.eos_token_id else "partial\ufffd"

    qwen = object.__new__(QwenAdapter)
    qwen.tokenizer = FakeQwenTokenizer()
    assert qwen.decode_text([1, 99]) == "answer"
    assert qwen.decode_text([1]) == "partial\ufffd"

    class FakeQwen4Tokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, messages: object, **options: object) -> list[int]:
            del messages, options
            return [1, 2, 3]

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model: str, *, local_files_only: bool) -> FakeQwen4Tokenizer:
            assert Path(model).is_dir()
            assert local_files_only
            return FakeQwen4Tokenizer()

    transformers = ModuleType("transformers")
    transformers.PreTrainedTokenizerFast = FakeTokenizerLoader  # type: ignore[attr-defined]
    previous_transformers = sys.modules.get("transformers")
    sys.modules["transformers"] = transformers
    try:
        with tempfile.TemporaryDirectory(prefix="ncnn-moe-qwen4-adapter-") as directory:
            qwen4_root = Path(directory)
            (qwen4_root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen4_exp",
                        "text_config": {"max_position_embeddings": 262144},
                    }
                ),
                encoding="utf-8",
            )
            (qwen4_root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (qwen4_root / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [248046, 248044]}),
                encoding="utf-8",
            )
            qwen4 = create_adapter(qwen4_root)
            assert isinstance(qwen4, QwenAdapter)
            assert qwen4.name == "qwen3.8"
            assert qwen4.model_type == "qwen4_exp"
            assert qwen4.context_limit == 262144
            assert qwen4.stop_tokens == [99, 248046, 248044]
    finally:
        if previous_transformers is None:
            del sys.modules["transformers"]
        else:
            sys.modules["transformers"] = previous_transformers

    class PrefixClient:
        def __init__(self) -> None:
            self.responses: list[tuple[dict[str, object], list[int]] | Exception] = []
            self.prompt_batches: list[list[int]] = []
            self.resets: list[str] = []

        def reset(self, session_id: str) -> None:
            self.resets.append(session_id)

        def generate(
            self, session_id: str, prompt_tokens: list[int], **options: object
        ) -> tuple[dict[str, object], list[int]]:
            del session_id, options
            self.prompt_batches.append(list(prompt_tokens))
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    prefix_client = PrefixClient()
    prefix_app = object.__new__(ConversationApp)
    prefix_app.adapter = SimpleNamespace(
        name="qwen3.6",
        model=Path("prefix-test-model"),
        model_type="qwen3_5",
        model_fingerprint="prefix-test",
        stop_tokens=[],
        decode_completion=lambda tokens: Completion("", "answer"),
    )
    prefix_app.client = prefix_client
    prefix_app.arguments = SimpleNamespace(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        no_speculative=True,
        speculative_confidence=0.5,
        speculative_max_draft=0,
        max_new_tokens=4,
        min_p=0.0,
        metrics_enabled=False,
        metrics_interval_ms=0,
        stream=False,
    )
    prefix_app.native_session_id = "prefix-test"
    prefix_app.native_context_tokens = []
    prefix_app.messages = []
    prefix_app.summary = ""
    prefix_app.settings = {}
    prefix_app.record = {}
    prefix_app.ephemeral = True

    # The worker says only the first generated token reached KV. The second
    # token must be sent with the next prompt instead of being treated as cache.
    first_prompt = [10, 11]
    second_prompt = [10, 11, 20, 21, 12]
    prompts = iter((first_prompt, second_prompt))
    prefix_app._prompt_tokens_with_budget = lambda: next(prompts)
    prefix_client.responses.extend(
        [
            ({"sequence_length": 3}, [20, 21]),
            ({"sequence_length": 6}, [30, 31]),
        ]
    )
    prefix_app.send("first turn")
    assert prefix_app.native_context_tokens == [10, 11, 20]
    prefix_app.send("second turn")
    assert prefix_client.prompt_batches[-2:] == [[10, 11], [21, 12]]
    assert prefix_app.native_context_tokens == [10, 11, 20, 21, 12, 30]
    assert first_prompt == [10, 11]
    assert second_prompt == [10, 11, 20, 21, 12]

    # Use the reported committed count rather than assuming stop, max-token,
    # cancellation, or speculative generation commits a fixed output suffix.
    for reason, prompt, tokens, committed in (
        ("stop", [1, 2], [50], 0),
        ("max tokens", [3, 4], [60, 61, 62], 1),
        ("cancel", [5, 6], [70, 71, 72], 2),
        ("partial speculative output", [7, 8], [80, 81, 82], 3),
    ):
        prefix_app.native_context_tokens = []
        prefix_app.messages = []
        prefix_app._prompt_tokens_with_budget = lambda prompt=prompt: prompt
        original_prompt = list(prompt)
        prefix_client.responses.append(
            ({"sequence_length": len(prompt) + committed}, tokens)
        )
        prefix_app.send("counted turn")
        assert prefix_app.native_context_tokens == prompt + tokens[:committed], reason
        assert prompt == original_prompt

    # A missing or malformed length makes the cache unusable; the next request
    # must reset before sending a full prompt.
    invalid_done_events = (
        {},
        {"sequence_length": None},
        {"sequence_length": True},
        {"sequence_length": "2"},
        {"sequence_length": 2.0},
        {"sequence_length": 1},
        {"sequence_length": 5},
    )
    for invalid_done in invalid_done_events:
        prefix_app.native_context_tokens = [3]
        prefix_app.messages = []
        prefix_app._prompt_tokens_with_budget = lambda: [3, 4]
        reset_count = len(prefix_client.resets)
        prefix_client.responses.append((invalid_done, [90, 91]))
        prefix_app.send("invalid sequence length")
        assert prefix_app.native_context_tokens == []
        assert len(prefix_client.resets) == reset_count

        prefix_app._prompt_tokens_with_budget = lambda: [3, 4, 5]
        prefix_client.responses.append(({"sequence_length": 3}, [92]))
        prefix_app.send("reset after invalid length")
        assert len(prefix_client.resets) == reset_count + 1

    # An interrupted or disconnected generate call may have changed native KV
    # without returning a committed length. The next turn must reset and replay
    # its complete prompt instead of reusing the stale Python prefix.
    for failure in (
        WorkerError("generation cancelled by user"),
        WorkerError("worker exited before sending an event (return code 1)"),
    ):
        prefix_app.native_context_tokens = [41, 42]
        prefix_app.messages = []
        failed_prompt = [41, 42, 43]
        prefix_app._prompt_tokens_with_budget = lambda: failed_prompt
        reset_count = len(prefix_client.resets)
        prefix_client.responses.append(failure)
        try:
            prefix_app.send("failed generation")
        except WorkerError as error:
            assert str(error) == str(failure)
        else:
            raise AssertionError("failed generation unexpectedly returned")
        assert prefix_client.prompt_batches[-1] == [43]
        assert prefix_app.native_context_tokens == []
        assert len(prefix_client.resets) == reset_count

        recovery_prompt = [41, 42, 43, 44]
        prefix_app._prompt_tokens_with_budget = lambda: recovery_prompt
        prefix_client.responses.append(({"sequence_length": 4}, [101]))
        prefix_app.send("recovery generation")
        assert prefix_client.prompt_batches[-1] == recovery_prompt
        assert prefix_client.resets[-1] == "prefix-test"
        assert len(prefix_client.resets) == reset_count + 1
        assert prefix_app.native_context_tokens == recovery_prompt

    with tempfile.TemporaryDirectory(prefix="ncnn-moe-worker-path-") as directory:
        root = Path(directory)
        worker_name = "ncnn_moe_worker.exe" if os.name == "nt" else "ncnn_moe_worker"
        expected = root / "build-ncnn"
        if os.name == "nt":
            expected /= "Release"
        expected /= worker_name
        assert default_worker_path(root) == expected

        other = root / "build-other" / "Release" / worker_name
        other.parent.mkdir(parents=True)
        other.touch()
        try:
            find_worker(None, root)
        except ValueError as error:
            assert "build-ncnn" in str(error)
        else:
            raise AssertionError("worker resolution must not scan other build directories")

        assert find_worker(str(other), root) == other.resolve()

    # Transformers returns BatchEncoding (a Mapping, not necessarily dict),
    # and tensor-backed tokenizers may add a single batch dimension.
    assert _normalize_token_ids(UserDict({"input_ids": [4, 5, 6]})) == [4, 5, 6]
    assert _normalize_token_ids(UserDict({"input_ids": [[7, 8, 9]]})) == [7, 8, 9]

    if importlib.util.find_spec("openai_harmony") is None:
        print(
            f"SKIP: openai_harmony is not installed for {sys.executable}; install with "
            f'"{sys.executable}" -m pip install -e ".[gpt-oss]"',
            file=sys.stderr,
        )
        return 77

    adapter = create_adapter(arguments.model)
    if adapter.model_type != "gpt_oss":
        raise AssertionError(f"unexpected adapter: {adapter.model_type}")

    runtime_args = [] if arguments.auto else ["--cpu"]
    with WorkerClient(arguments.worker, arguments.model, runtime_args) as client:
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
        client.create_session("second", enable_speculative_context=False)
        compacted = client.compact("first", [0, 1])
        assert compacted["replayed_tokens"] == 2
        events: list[dict[str, object]] = []
        done, tokens = client.generate(
            "first",
            [0],
            request_id="fixture-generation",
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
        assert "metrics" in done
        assert "prompt_tok_per_second" in done["metrics"]
        assert "generation_tok_per_second" in done["metrics"]
        assert "expert" in done["metrics"]
        assert "ttft_microseconds" in done["metrics"]
        assert "tpot_microseconds" in done["metrics"]
        if arguments.auto and client.ready["resources"]["backend"] == "hybrid":
            assert done["stats"]["generation_gpu"]["submit_count"] > 0
        else:
            assert done["metrics"]["gpu"]["submit_count"] is None
            assert done["metrics"]["gpu"]["reason"] == "runtime_backend_cpu_only"
        stats = client.stats("first")
        assert stats["event"] == "stats"
        assert stats["sequence_length"] == done["sequence_length"]

        worker_tokens: list[int] = []

        def decode_worker_tokens(tokens: list[int]) -> Completion:
            worker_tokens[:] = tokens
            return Completion("", "answer")

        worker_app = prefix_app
        worker_app.adapter.name = "gpt-oss"
        worker_app.adapter.model = arguments.model
        worker_app.adapter.model_type = "gpt_oss"
        worker_app.adapter.model_fingerprint = "prefix-worker-test"
        worker_app.adapter.stop_tokens = []
        worker_app.adapter.decode_completion = decode_worker_tokens
        worker_app.client = client
        worker_app.native_session_id = "prefix-reuse"
        worker_app.native_context_tokens = []
        worker_app.messages = []
        client.create_session(
            "prefix-reuse", seed=17, enable_speculative_context=False
        )
        client.create_session(
            "prefix-fresh", seed=17, enable_speculative_context=False
        )
        first_worker_prompt = [0, 1]
        worker_app._prompt_tokens_with_budget = lambda: first_worker_prompt
        _, first_worker_done, first_reused = worker_app.send("first worker turn")
        first_worker_output = list(worker_tokens)
        assert not first_reused
        committed_count = first_worker_done["sequence_length"] - len(first_worker_prompt)
        assert 0 <= committed_count <= len(first_worker_output)
        assert worker_app.native_context_tokens == (
            first_worker_prompt + first_worker_output[:committed_count]
        )
        assert client.stats("prefix-reuse")["sequence_length"] == len(
            worker_app.native_context_tokens
        )

        replay_prompt = first_worker_prompt + first_worker_output + [0]
        worker_app._prompt_tokens_with_budget = lambda: replay_prompt
        _, resumed_done, reused = worker_app.send("second worker turn")
        resumed_output = list(worker_tokens)
        assert reused
        committed_count = resumed_done["sequence_length"] - len(replay_prompt)
        assert 0 <= committed_count <= len(resumed_output)
        assert worker_app.native_context_tokens == (
            replay_prompt + resumed_output[:committed_count]
        )
        _, fresh_output = client.generate(
            "prefix-fresh",
            replay_prompt,
            request_id="fresh-prefix-replay",
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            stop_tokens=[],
            enable_speculative=False,
            metrics_enabled=False,
            metrics_interval_ms=0,
        )
        assert resumed_output == fresh_output
        client.reset("first")
        assert client.stats("first")["sequence_length"] == 0

    process = subprocess.Popen(
        [str(arguments.worker), str(arguments.model), *runtime_args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    ready = json.loads(process.stdout.readline())
    assert ready["event"] == "ready"
    for request, session_id in (
        (r'{"op":"create_session","session_id":"json-\u4e2d\u6587\ud83d\ude00"}', "json-中文😀"),
        (r'{"op":"create_session","session_id":"\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008\u0009\u000a\u000b\u000c\u000d\u000e\u000f\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f"}', "".join(chr(codepoint) for codepoint in range(32))),
        ('{"metadata":{"session_id":"shadow"},"op":"create_session","session_id":"json-top"}', "json-top"),
        (r'{"op":"create_session","session_\u0069d":"json-key"}', "json-key"),
    ):
        process.stdin.write(request + "\n")
        process.stdin.flush()
        event = json.loads(process.stdout.readline())
        assert event["event"] == "session_created", event
        assert event["session_id"] == session_id, event
    process.stdin.write(r'{"op":"create_session","session_id":"bad-\ud800"}' + "\n")
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["event"] == "error"
    process.stdin.write(
        '{"metadata":{"prompt_tokens":[0,1,0,1]},"op":"generate",'
        '"session_id":"json-top","prompt_tokens":[0,1],"max_new_tokens":1,'
        '"temperature":0,"enable_speculative":false,"metrics_enabled":false}\n'
    )
    process.stdin.flush()
    for line in process.stdout:
        event = json.loads(line)
        assert event["event"] != "error", event
        if event["event"] == "done":
            assert event["sequence_length"] == 2, event
            break
    else:
        raise AssertionError("JSON field generation did not finish")
    process.stdin.write('{"op":"create_session","session_id":"cancel"}\n')
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["event"] == "session_created"
    # Reject the old oversized budget before mutation, then exercise actual
    # cancellation with a budget that fits the fixture's context.
    process.stdin.write(
        '{"op":"generate","request_id":"cancel-me","session_id":"cancel",'
        '"prompt_tokens":[0,1,0,1],"max_new_tokens":1000,"temperature":0,'
        '"enable_speculative":false}\n'
    )
    process.stdin.flush()
    event = json.loads(process.stdout.readline())
    assert event["event"] == "error" and event["code"] == "context_overflow", event
    process.stdin.write('{"op":"stats","session_id":"cancel"}\n')
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["sequence_length"] == 0
    cancel_limit = min(1000, int(ready["model"]["max_context_tokens"]) - 4)
    assert cancel_limit > 1
    process.stdin.write(
        json.dumps(
            {
                "op": "generate",
                "request_id": "cancel-me",
                "session_id": "cancel",
                "prompt_tokens": [0, 1, 0, 1],
                "max_new_tokens": cancel_limit,
                "temperature": 0,
                "enable_speculative": False,
            }
        )
        + "\n"
    )
    process.stdin.flush()
    cancelled = False
    completed = False
    for line in process.stdout:
        event = json.loads(line)
        if event.get("event") == "token" and not cancelled:
            process.stdin.write('{"op":"cancel","request_id":"cancel-me"}\n')
            process.stdin.flush()
            cancelled = True
        elif event.get("event") == "error":
            process.terminate()
            process.wait(timeout=5)
            raise AssertionError(f"cancellation generation failed: {event}")
        elif event.get("event") == "done":
            completed = True
            assert event.get("cancelled") is True
            break
    assert cancelled and completed
    process.stdin.write('{"op":"shutdown"}\n')
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["event"] == "shutdown"
    process.wait(timeout=5)
    print("ncnn_moe Python worker/adapter smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
