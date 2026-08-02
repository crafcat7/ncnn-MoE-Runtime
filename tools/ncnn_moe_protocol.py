"""Client for the ncnn_moe_worker stdio JSONL protocol.

Startup ``init`` events are consumed before the first ``ready`` event.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable


class WorkerError(RuntimeError):
    """A protocol, process, or native-runtime error."""

    def __init__(self, message: str, event: dict[str, Any] | None = None):
        super().__init__(message)
        self.event = event or {}


EventCallback = Callable[[dict[str, Any]], None]


class WorkerClient:
    """Synchronous JSONL client; generation itself runs asynchronously in C++."""

    def __init__(
        self,
        worker: Path,
        model: Path,
        runtime_args: Iterable[str] = (),
        *,
        startup_callback: EventCallback | None = None,
        verbose: bool = False,
    ) -> None:
        self.worker = Path(worker).resolve()
        self.model = Path(model).resolve()
        if not self.worker.is_file():
            raise WorkerError(f"worker does not exist: {self.worker}")
        if not self.model.is_dir():
            raise WorkerError(f"model directory does not exist: {self.model}")
        self._process = subprocess.Popen(
            [str(self.worker), str(self.model), *map(str, runtime_args)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._closed = False
        try:
            while True:
                ready = self._read_event()
                if ready.get("event") == "init":
                    if startup_callback is not None:
                        startup_callback(ready)
                    continue
                if ready.get("event") == "error":
                    raise WorkerError(ready.get("message", "worker failed to load the model"), ready)
                if ready.get("event") != "ready":
                    raise WorkerError(f"worker did not send ready event: {ready}", ready)
                self.ready = ready
                break
        except BaseException:
            self._terminate()
            raise

    @property
    def process(self) -> subprocess.Popen[str]:
        return self._process

    def _send(self, payload: dict[str, Any]) -> None:
        if self._closed or self._process.stdin is None:
            raise WorkerError("worker is closed")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _read_event(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise WorkerError("worker stdout is unavailable")
        while True:
            line = self._process.stdout.readline()
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
        prefill_chunk_size: int = 256,
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
        prompt_tokens: list[int],
        *,
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
        self._send(
            {
                "op": "generate",
                "request_id": request_id,
                "session_id": session_id,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "stop_tokens": stop_tokens or [],
                "enable_speculative": enable_speculative,
                "speculative_confidence": speculative_confidence,
                "speculative_max_draft": speculative_max_draft,
                "metrics_enabled": metrics_enabled,
                "metrics_interval_ms": metrics_interval_ms,
            }
        )
        tokens: list[int] = []
        try:
            while True:
                event = self._read_event()
                self._raise_for_error(event)
                if on_event is not None:
                    on_event(event)
                if event.get("event") == "token":
                    tokens.append(int(event["token_id"]))
                elif event.get("event") == "done":
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
                    if on_event is not None:
                        on_event(event)
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

    def compact(self, session_id: str, replay_tokens: list[int]) -> dict[str, Any]:
        return self.request(
            {
                "op": "compact",
                "session_id": session_id,
                "replay_tokens": replay_tokens,
            },
            expected="compacted",
        )

    def stats(self, session_id: str) -> dict[str, Any]:
        return self.request(
            {"op": "stats", "session_id": session_id}, expected="stats"
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
