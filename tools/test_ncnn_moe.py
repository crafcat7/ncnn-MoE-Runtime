#!/usr/bin/env python3
"""Small dependency-light smoke test for the unified worker and GPT-OSS adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncnn_moe_adapters import _normalize_token_ids, create_adapter  # noqa: E402
from ncnn_moe_protocol import WorkerClient, WorkerError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--auto", action="store_true")
    arguments = parser.parse_args()

    # Transformers returns BatchEncoding (a Mapping, not necessarily dict),
    # and tensor-backed tokenizers may add a single batch dimension.
    assert _normalize_token_ids(UserDict({"input_ids": [4, 5, 6]})) == [4, 5, 6]
    assert _normalize_token_ids(UserDict({"input_ids": [[7, 8, 9]]})) == [7, 8, 9]

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
        if arguments.auto and client.ready["resources"]["backend"] == "hybrid":
            assert done["stats"]["vulkan_linear_dispatches"] > 0
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
    ready = json.loads(process.stdout.readline())
    assert ready["event"] == "ready"
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
