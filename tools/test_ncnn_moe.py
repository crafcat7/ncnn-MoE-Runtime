#!/usr/bin/env python3
"""Small dependency-light smoke test for the unified worker and GPT-OSS adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncnn_moe_adapters import QwenAdapter, _normalize_token_ids, create_adapter  # noqa: E402
from ncnn_moe import _format_bytes_gb, _format_runtime_metrics, default_worker_path, find_worker, parse_arguments  # noqa: E402
from ncnn_moe_protocol import WorkerClient, WorkerError  # noqa: E402


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
            "decode_tok_per_second": None,
            "tokens_per_second": None,
            "tpot_microseconds": 75_030.0,
            "gpu": {
                "available": True,
                "kernel_time_available": False,
                "reason": "gpu_expert_execution_not_observed",
            },
            "gpu_device": {},
        }
    )
    assert "Decode tok/s 13.33" in formatted_metrics
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

    class FakeQwenTokenizer:
        eos_token_id = 99

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "answer\ufffd" if tokens and tokens[-1] == self.eos_token_id else "partial\ufffd"

    qwen = object.__new__(QwenAdapter)
    qwen.tokenizer = FakeQwenTokenizer()
    assert qwen.decode_text([1, 99]) == "answer"
    assert qwen.decode_text([1]) == "partial\ufffd"

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

    adapter = create_adapter(arguments.model)
    if adapter.model_type != "gpt_oss":
        raise AssertionError(f"unexpected adapter: {adapter.model_type}")

    runtime_args = [] if arguments.auto else ["--cpu"]
    startup_events: list[dict[str, object]] = []
    with WorkerClient(
        arguments.worker,
        arguments.model,
        runtime_args,
        startup_callback=startup_events.append,
    ) as client:
        assert client.ready["event"] == "ready"
        assert startup_events
        assert startup_events[-1]["completed_steps"] == startup_events[-1]["total_steps"]
        assert startup_events[-1]["phase"] == "worker"
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
    startup_lines: list[dict[str, object]] = []
    while True:
        ready = json.loads(process.stdout.readline())
        if ready.get("event") != "init":
            break
        startup_lines.append(ready)
    assert ready["event"] == "ready"
    assert startup_lines and startup_lines[-1]["phase"] == "worker"
    process.stdin.write('{"op":"create_session","session_id":"cancel"}\n')
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["event"] == "session_created"
    process.stdin.write(
        '{"op":"generate","request_id":"cancel-me","session_id":"cancel",'
        '"prompt_tokens":[0,1,0,1],"max_new_tokens":1000,"temperature":0,'
        '"enable_speculative":false}\n'
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
