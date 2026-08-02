#!/usr/bin/env python3
"""Unified CLI/TUI for ncnn MoE examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ncnn_moe_adapters import AdapterError, Completion, ModelAdapter, create_adapter
    from ncnn_moe_protocol import WorkerClient, WorkerError
    from ncnn_moe_state import (
        SessionStore,
        hardware_fingerprint,
        merge_runtime_settings,
        profile_key,
        runtime_args_from_settings,
    )
except ModuleNotFoundError:  # Installed entry point: tools is a package.
    from .ncnn_moe_adapters import AdapterError, Completion, ModelAdapter, create_adapter
    from .ncnn_moe_protocol import WorkerClient, WorkerError
    from .ncnn_moe_state import (
        SessionStore,
        hardware_fingerprint,
        merge_runtime_settings,
        profile_key,
        runtime_args_from_settings,
    )


def configure_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


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
        return str(utilization)
    reason = value.get("reason") or "unavailable"
    return f"N/A ({reason})"


def _add_model_position(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_pos", nargs="?", help="Model directory")
    parser.add_argument("--model", dest="model_opt", help="Model directory")


def _add_worker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--worker", help="Path to ncnn_moe_worker")
    parser.add_argument("--config-dir", help="Override the persistent configuration directory")
    parser.add_argument("--backend", choices=("auto", "cpu", "vulkan", "hybrid", "hybrid-prefetch"))
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--cpu", dest="backend", action="store_const", const="cpu")
    backend_group.add_argument("--hybrid", dest="backend", action="store_const", const="hybrid")
    backend_group.add_argument("--hybrid-prefetch", dest="backend", action="store_const", const="hybrid-prefetch")
    parser.add_argument("--expert-memory", choices=("auto", "eager", "on-demand"))
    parser.add_argument("--host-memory-mb", type=int)
    parser.add_argument("--expert-cache-mb", type=int)
    parser.add_argument("--expert-gpu-cache-mb", type=int)
    parser.add_argument("--expert-gpu-victim-cache-mb", type=int)
    parser.add_argument("--expert-io-workers", type=int)
    parser.add_argument("--vulkan-device", type=int)
    parser.add_argument("--vulkan-devices", help="Comma-separated Vulkan device indices")
    parser.add_argument("--expected-concurrency", type=int)
    parser.add_argument("--mmap-experts", action="store_true", default=None)
    parser.add_argument("--direct-expert-io", action="store_true", default=None)
    parser.add_argument("--buffered-expert-io", action="store_true", default=None)
    parser.add_argument("--disable-gpu-victim-execution", action="store_true", default=None)
    parser.add_argument("--router-prediction", action="store_true", default=None)
    parser.add_argument("--async-router-prediction", action="store_true", default=None)
    parser.add_argument("--forward-aware-cache", action="store_true", default=None)
    parser.add_argument("--rank-adaptive-prefetch", action="store_true", default=None)
    parser.add_argument("--cross-expert-read-coalescing", action="store_true", default=None)
    parser.add_argument("--release-vulkan-dense-host", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true")


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--speculative-confidence", type=float, default=0.5)
    parser.add_argument("--speculative-max-draft", type=int, default=0)
    parser.add_argument(
        "--metrics-interval-ms",
        type=int,
        default=1000,
        help="Periodic metrics event interval; 0 disables the metrics trace",
    )
    parser.add_argument(
        "--no-metrics",
        dest="metrics_enabled",
        action="store_false",
        default=True,
        help="Disable periodic metrics events while retaining final statistics",
    )
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-final-only", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")


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
    run.add_argument("--thinking-mode", choices=("chat", "thinking"), default="thinking")
    run.add_argument("--no-thinking", action="store_true")
    run.add_argument("--ephemeral", action="store_true", help="Do not read or write a session")
    _add_worker_options(run)
    _add_generation_options(run)

    chat = subparsers.add_parser("chat", help="Start an interactive continuous conversation")
    _add_model_position(chat)
    chat.add_argument("--session", help="Session ID to resume")
    chat.add_argument("--title", default="", help="Title for a new persistent session")
    chat.add_argument("--system", default="")
    chat.add_argument("--thinking-mode", choices=("chat", "thinking"), default="thinking")
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
    tune.add_argument("--thinking-mode", choices=("chat", "thinking"), default="thinking")
    tune.add_argument("--no-thinking", action="store_true")
    tune.add_argument("--max-new-tokens", type=int, default=64)
    tune.add_argument("--context-tokens", type=int, default=0)
    tune.add_argument("--prefill-chunk-size", type=int, default=256)
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument(
        "--metrics-interval-ms",
        type=int,
        default=1000,
        help="Periodic metrics event interval; 0 disables the metrics trace",
    )
    tune.add_argument(
        "--no-metrics",
        dest="metrics_enabled",
        action="store_false",
        default=True,
        help="Disable periodic metrics events while retaining final statistics",
    )
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
            raise ValueError("a model directory is required; pass --model PATH or a positional model")
        return None
    model = Path(value).expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
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


def _build_vulkan_state(build_dir: Path) -> bool | None:
    cache = build_dir / "CMakeCache.txt"
    try:
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("NCNN_MOE_USE_VULKAN:BOOL="):
                return line.rsplit("=", 1)[-1].strip().upper() == "ON"
    except OSError:
        pass
    return None


def find_worker(explicit: str | None, root: Path) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    environment = os.environ.get("NCNN_MOE_WORKER")
    if environment:
        candidates.append(Path(environment))
    names = ("ncnn_moe_worker.exe", "ncnn_moe_worker")
    build_candidates: list[tuple[int, str, Path]] = []
    for build_dir in sorted(root.glob("build*")):
        vulkan_state = _build_vulkan_state(build_dir)
        # Auto should prefer a build that can actually expose Vulkan. An
        # explicit --worker or NCNN_MOE_WORKER remains authoritative.
        priority = 0 if vulkan_state is True else 2 if vulkan_state is False else 1
        for name in names:
            build_candidates.append((priority, build_dir.name, build_dir / "Release" / name))
            build_candidates.append((priority, build_dir.name, build_dir / name))
    candidates.extend(candidate for _, _, candidate in sorted(build_candidates, key=lambda item: (item[0], item[1], str(item[2]))))
    for name in names:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        "ncnn_moe_worker was not found; build the examples or pass --worker PATH"
    )


def lightweight_model_fingerprint(model: Path) -> str:
    digest = hashlib.sha256()
    digest.update((model / "config.json").read_bytes())
    for path in sorted(model.rglob("*.safetensors")):
        digest.update(path.relative_to(model).as_posix().encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        with path.open("rb") as stream:
            digest.update(stream.read(4096))
            if path.stat().st_size > 4096:
                stream.seek(-4096, 2)
                digest.update(stream.read(4096))
    return digest.hexdigest()[:20]


def cli_runtime_settings(arguments: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "backend",
        "expert_memory",
        "host_memory_mb",
        "expert_cache_mb",
        "expert_gpu_cache_mb",
        "expert_gpu_victim_cache_mb",
        "expert_io_workers",
        "vulkan_device",
        "vulkan_devices",
        "expected_concurrency",
        "mmap_experts",
        "direct_expert_io",
        "buffered_expert_io",
        "disable_gpu_victim_execution",
        "router_prediction",
        "async_router_prediction",
        "forward_aware_cache",
        "rank_adaptive_prefetch",
        "cross_expert_read_coalescing",
        "release_vulkan_dense_host",
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


def open_worker(
    arguments: argparse.Namespace,
    model: Path,
    store: SessionStore,
    *,
    adapter: ModelAdapter | None = None,
    session: dict[str, Any] | None = None,
) -> tuple[WorkerClient, dict[str, Any], dict[str, Any] | None]:
    root = Path(__file__).resolve().parents[1]
    worker = find_worker(getattr(arguments, "worker", None), root)
    user = user_runtime_settings(store)
    cli = cli_runtime_settings(arguments)
    initial = merge_runtime_settings(cli=cli, session=session.get("settings", {}) if session else None, profile=None, user=user)
    client = WorkerClient(worker, model, runtime_args_from_settings(initial), verbose=getattr(arguments, "verbose", False))
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
        session=session.get("settings", {}) if session else None,
        profile=profile_settings,
        user=user,
    )
    if merged != initial:
        client.close()
        client = WorkerClient(worker, model, runtime_args_from_settings(merged), verbose=getattr(arguments, "verbose", False))
    return client, merged, profile_record


def default_generation_values(adapter: ModelAdapter, arguments: argparse.Namespace) -> tuple[float, int, float]:
    if arguments.temperature is None:
        temperature = 0.0 if adapter.name == "gpt-oss" else 1.0
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
    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if arguments.top_k is not None and arguments.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if arguments.speculative_max_draft < 0:
        raise ValueError("--speculative-max-draft must be non-negative")
    if not 0.0 <= arguments.speculative_confidence <= 1.0:
        raise ValueError("--speculative-confidence must be between 0 and 1")
    if arguments.metrics_interval_ms < 0:
        raise ValueError("--metrics-interval-ms must be non-negative")
    if arguments.context_tokens < 0 or arguments.prefill_chunk_size <= 0:
        raise ValueError("context and prefill sizes must be positive or zero for auto")
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
        self.native_context_tokens: list[int] = []
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
        self.client.create_session(
            self.native_session_id,
            seed=arguments.seed,
            prefill_chunk_size=arguments.prefill_chunk_size,
            enable_speculative_context=not arguments.no_speculative,
        )
        if record and self.messages:
            try:
                replay_tokens = self.adapter.encode_messages(self.messages)
                self._status(f"resuming {record.get('id')} · replaying {len(replay_tokens)} context tokens")
                self.client.compact(self.native_session_id, replay_tokens)
                self.native_context_tokens = replay_tokens
                self._status(f"resumed {record.get('id')} ({len(self.messages)} messages)")
            except (AdapterError, WorkerError, ValueError) as error:
                self.native_context_tokens = []
                self._status(f"session replay deferred: {error}")

    @property
    def context_limit(self) -> int | None:
        return context_budget(self.adapter, self.arguments, self.client.ready)

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
        self.record["model_fingerprint"] = self.adapter.model_fingerprint
        self.record["context_token_count"] = len(self.native_context_tokens)
        if not self.ephemeral:
            self.store.save(self.record)

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
            prompt_tokens = self.adapter.encode_messages(summary_prompt)
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
            _, tokens = self.client.generate(
                self.summary_session_id,
                prompt_tokens,
                request_id="compact-summary",
                max_new_tokens=min(512, self.arguments.max_new_tokens),
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                min_p=0.0,
                stop_tokens=self.adapter.stop_tokens,
                enable_speculative=False,
                metrics_enabled=metrics_trace_enabled(self.arguments),
                metrics_interval_ms=self.arguments.metrics_interval_ms,
            )
            completion = self.adapter.decode_completion(tokens)
            summary = completion.answer or completion.reasoning or completion.text
            if summary.strip():
                return summary.strip()
        except (AdapterError, WorkerError, ValueError):
            pass
        return self._fallback_summary()

    def compact(self, *, announce: bool = True) -> None:
        if len(self.messages) <= 2:
            try:
                self.client.compact(self.native_session_id, [])
            except WorkerError:
                self.client.reset(self.native_session_id)
            self.native_context_tokens = []
            if announce:
                self._status("context reset; not enough history to summarize")
            return
        old_messages = self.messages
        old_summary = self.summary
        summary = self._summarize()
        system_messages = [message for message in old_messages if message.get("role") == "system"]
        retained = [message for message in old_messages if message.get("role") != "system"][-4:]
        summary_message = {"role": "system", "content": f"Conversation summary:\n{summary}"}
        self.messages = system_messages[:1] + [summary_message] + retained
        self.summary = summary
        try:
            self.client.compact(self.native_session_id, [])
        except WorkerError:
            self.messages = old_messages
            self.summary = old_summary
            self._trim_sliding_window()
            self.client.reset(self.native_session_id)
            self.native_context_tokens = []
            if announce:
                self._status("native compact failed; using sliding-window fallback")
            return
        self.native_context_tokens = []
        if announce:
            self._status(f"context compacted · retained {len(self.messages)} messages")

    def _trim_sliding_window(self) -> None:
        budget = self.context_limit
        if budget is None:
            return
        for _ in range(100):
            tokens = self.adapter.encode_messages(self.messages)
            if len(tokens) + self.arguments.max_new_tokens <= budget:
                return
            non_system = [index for index, message in enumerate(self.messages) if message.get("role") != "system"]
            if len(non_system) <= 1:
                if non_system:
                    message = self.messages[non_system[0]]
                    message["content"] = message["content"][-max(32, len(message["content"]) // 2) :]
                return
            del self.messages[non_system[0]]

    def _prompt_tokens_with_budget(self) -> list[int]:
        tokens = self.adapter.encode_messages(self.messages)
        budget = self.context_limit
        if budget is not None and len(tokens) + self.arguments.max_new_tokens > budget:
            self.compact()
            tokens = self.adapter.encode_messages(self.messages)
            if len(tokens) + self.arguments.max_new_tokens > budget:
                self._status("summary did not fit; using sliding-window fallback")
                self._trim_sliding_window()
                tokens = self.adapter.encode_messages(self.messages)
        return tokens

    def _on_generation_event(self, event: dict[str, Any], generated_tokens: list[int], streamed: dict[str, str]) -> None:
        if event.get("event") == "metrics":
            streamed["metrics"] = json.dumps(event.get("metrics", {}), ensure_ascii=False)
            self.last_metrics = event.get("metrics", {})
            metrics = self.last_metrics
            process = metrics.get("process", {}) if isinstance(metrics, dict) else {}
            self._status(
                f"metrics · {metrics.get('tokens_per_second', 'N/A')} token/s · "
                f"CPU {process.get('cpu_percent', 'N/A')} · "
                f"GPU {_format_gpu_utilization(metrics.get('gpu'))} · "
                f"IO {process.get('read_bytes', 'N/A')}/{process.get('write_bytes', 'N/A')} bytes"
            )
            return
        if event.get("event") != "token" or not self.arguments.stream:
            return
        generated_tokens.append(int(event["token_id"]))
        visible = self.adapter.stream_visible(
            generated_tokens,
            final_only=self.arguments.stream_final_only or not self.arguments.show_reasoning,
        )
        previous = streamed.get("text", "")
        if visible.startswith(previous):
            delta = visible[len(previous) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        streamed["text"] = visible

    def send(self, user_text: str) -> tuple[Completion, dict[str, Any], bool]:
        self.messages.append({"role": "user", "content": user_text})
        prompt_tokens = self._prompt_tokens_with_budget()
        reused = bool(self.native_context_tokens and prompt_tokens[: len(self.native_context_tokens)] == self.native_context_tokens)
        if reused:
            input_tokens = prompt_tokens[len(self.native_context_tokens) :]
            native_base = list(self.native_context_tokens)
        else:
            self.client.reset(self.native_session_id)
            input_tokens = prompt_tokens
            native_base = []
        if not input_tokens:
            self.client.reset(self.native_session_id)
            input_tokens = prompt_tokens
            native_base = []
        generated_tokens: list[int] = []
        streamed: dict[str, str] = {}
        temperature, top_k, top_p = default_generation_values(self.adapter, self.arguments)
        done, generated_tokens = self.client.generate(
            self.native_session_id,
            input_tokens,
            request_id=f"chat-{int(time.time() * 1000)}",
            max_new_tokens=self.arguments.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=self.arguments.min_p,
            stop_tokens=self.adapter.stop_tokens,
            enable_speculative=not self.arguments.no_speculative,
            speculative_confidence=self.arguments.speculative_confidence,
            speculative_max_draft=self.arguments.speculative_max_draft,
            metrics_enabled=metrics_trace_enabled(self.arguments),
            metrics_interval_ms=self.arguments.metrics_interval_ms,
            on_event=lambda event: self._on_generation_event(event, generated_tokens, streamed),
        )
        completion = self.adapter.decode_completion(generated_tokens)
        self.last_reasoning = completion.reasoning
        self.messages.append({"role": "assistant", "content": completion.answer or completion.reasoning})
        self.native_context_tokens = native_base + input_tokens + generated_tokens
        self._save()
        if self.arguments.stream and streamed.get("text"):
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
        elif not self.arguments.stream:
            print(completion.reasoning)

    def status_after(self, done: dict[str, Any], reused: bool) -> None:
        stats = done.get("stats", {})
        metrics = done.get("telemetry", {})
        tokens = done.get("generated_tokens", 0)
        rate = done.get("tokens_per_second")
        cache_hits = stats.get("expert_cache_hits", 0)
        cache_misses = stats.get("expert_cache_misses", 0)
        hit_rate = None if cache_hits + cache_misses == 0 else cache_hits / (cache_hits + cache_misses)
        parts = [
            f"{tokens} tokens · {rate:.2f} token/s" if isinstance(rate, (int, float)) else f"{tokens} tokens",
            f"context {len(self.native_context_tokens)}/{self.context_limit or 'auto'}",
            f"cache {hit_rate:.0%}" if hit_rate is not None else "cache N/A",
            f"CPU {metrics.get('cpu_percent', 'N/A')}",
            f"prefix {'reused' if reused else 'replayed'}",
        ]
        self._status(" · ".join(parts))

    def show_context(self) -> None:
        tokens = self.adapter.encode_messages(self.messages) if self.messages else []
        budget = self.context_limit
        self._status(
            f"messages={len(self.messages)} prompt_tokens={len(tokens)} "
            f"budget={budget or 'unknown'} native_sequence={len(self.native_context_tokens)}"
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
            prompt_tokens = self.adapter.encode_messages([{"role": "user", "content": prompt}])
            temperature, top_k, top_p = default_generation_values(self.adapter, self.arguments)
            done, _ = self.client.generate(
                tune_session,
                prompt_tokens,
                request_id="tune-live",
                max_new_tokens=min(32, self.arguments.max_new_tokens),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=self.arguments.min_p,
                stop_tokens=self.adapter.stop_tokens,
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
        self.native_context_tokens = []
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
            self.client.reset(self.native_session_id)
            self.native_context_tokens = []
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

    def chat(self) -> int:
        self.show_ready()
        prompt_session = self.prompt_session_class() if self.prompt_session_class else None
        while True:
            try:
                line = prompt_session.prompt("you> ") if prompt_session else input("you> ")
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
                if not self.arguments.stream or not completion.answer:
                    self.display_completion(completion)
                self.status_after(done, reused)
            except (AdapterError, WorkerError, ValueError) as error:
                self._status(f"error: {error}")
        self._save()
        return 0


def load_adapter(arguments: argparse.Namespace, model: Path) -> ModelAdapter:
    thinking_mode = getattr(arguments, "thinking_mode", "thinking")
    thinking = thinking_mode != "chat" and not getattr(arguments, "no_thinking", False)
    return create_adapter(model, thinking=thinking, thinking_mode=thinking_mode)


def run_command(arguments: argparse.Namespace) -> int:
    validate_generation_arguments(arguments)
    model = resolve_model(arguments)
    prompt = resolve_prompt(arguments)
    assert model is not None and prompt is not None
    adapter = load_adapter(arguments, model)
    store = SessionStore()
    client, settings, _ = open_worker(arguments, model, store, adapter=adapter)
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
        if not arguments.stream or not completion.answer:
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
    adapter = load_adapter(arguments, model)
    if record and record.get("model_fingerprint") not in {None, adapter.model_fingerprint}:
        raise ValueError("the requested session belongs to a different model fingerprint")
    client, settings, _ = open_worker(arguments, model, store, adapter=adapter, session=record)
    try:
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
    client, _, _ = open_worker(arguments, model, store)
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
            print(f"host memory: {resources.get('host_memory_budget_bytes', 'N/A')} bytes")
            print(f"Expert cache: {resources.get('expert_cache_bytes', 'N/A')} bytes")
            print(f"Expert IO workers: {resources.get('expert_io_workers', 'N/A')}")
            print(f"CPU: {capabilities.get('physical_cpu_core_count', 'N/A')} physical / {capabilities.get('logical_cpu_count', 'N/A')} logical")
            print(f"Vulkan devices: {capabilities.get('vulkan_device_count', 0)}")
            print(f"GPU telemetry: {_format_gpu_utilization(telemetry)}")
            for device in capabilities.get("vulkan_devices", []):
                print(f"  [{device.get('index')}] {device.get('name')} ({device.get('type')}) heap={device.get('heap_budget_bytes')} bytes")
        return 0
    finally:
        client.close()


def tune_command(arguments: argparse.Namespace) -> int:
    if arguments.runs <= 0 or arguments.warmup < 0:
        raise ValueError("--runs must be positive and --warmup must be non-negative")
    model = resolve_model(arguments)
    prompt = resolve_prompt(arguments)
    assert model is not None and prompt is not None
    adapter = load_adapter(arguments, model)
    store = SessionStore()
    client, settings, _ = open_worker(arguments, model, store, adapter=adapter)
    try:
        client.create_session("tune", seed=arguments.seed, prefill_chunk_size=arguments.prefill_chunk_size, enable_speculative_context=False)
        prompt_tokens = adapter.encode_messages([{"role": "user", "content": prompt}])
        temperature = 0.0 if adapter.name == "gpt-oss" else 1.0
        rates: list[float] = []
        for index in range(arguments.warmup + arguments.runs):
            if index:
                client.reset("tune")
            done, _ = client.generate(
                "tune",
                prompt_tokens,
                request_id=f"tune-{index}",
                max_new_tokens=arguments.max_new_tokens,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                stop_tokens=adapter.stop_tokens,
                enable_speculative=False,
                metrics_enabled=metrics_trace_enabled(arguments),
                metrics_interval_ms=arguments.metrics_interval_ms if hasattr(arguments, "metrics_interval_ms") else 1000,
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
