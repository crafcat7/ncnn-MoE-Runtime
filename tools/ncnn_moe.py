#!/usr/bin/env python3
"""Unified CLI/TUI for ncnn MoE examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ncnn_moe_adapters import AdapterError, Completion, ModelAdapter, NativeQwenAdapter, create_adapter
    from ncnn_moe_protocol import WorkerClient, WorkerError
    from ncnn_moe_state import (
        SessionStore,
        hardware_fingerprint,
        merge_runtime_settings,
        profile_key,
        runtime_args_from_settings,
    )
except ModuleNotFoundError:  # Installed entry point: tools is a package.
    from .ncnn_moe_adapters import AdapterError, Completion, ModelAdapter, NativeQwenAdapter, create_adapter
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
        return f"{float(utilization):.1f}%"
    reason = value.get("reason") or "unavailable"
    return f"N/A ({reason})"


def _format_duration_microseconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    microseconds = float(value)
    if microseconds >= 1_000_000:
        return f"{microseconds / 1_000_000:.2f} s"
    return f"{microseconds / 1_000:.2f} ms"


def _format_bytes_gb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value) / 1_000_000_000:.2f} GB"


def _format_metric_count(value: Any, *, available: bool = True) -> str:
    if not available or not isinstance(value, (int, float)):
        return "N/A"
    return f"{int(value):,}"


def _format_rate(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value):.2f}"


def _numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _prompt_rate(metrics: dict[str, Any]) -> float | None:
    for name in ("prompt_tok_per_second", "prompt_tokens_per_second"):
        value = _numeric_metric(metrics.get(name))
        if value is not None:
            return value
    prompt_elapsed_microseconds = _numeric_metric(metrics.get("prompt_elapsed_microseconds"))
    input_tokens = _numeric_metric(metrics.get("input_tokens"))
    if prompt_elapsed_microseconds is not None and prompt_elapsed_microseconds > 0.0 and input_tokens:
        return input_tokens * 1_000_000.0 / prompt_elapsed_microseconds
    ttft_microseconds = _numeric_metric(metrics.get("ttft_microseconds"))
    if ttft_microseconds is not None and ttft_microseconds > 0.0 and input_tokens:
        return input_tokens * 1_000_000.0 / ttft_microseconds
    return None


def _generation_rate(metrics: dict[str, Any]) -> float | None:
    for name in (
        "generation_tok_per_second",
        "generation_tokens_per_second",
        "decode_tok_per_second",
        "tokens_per_second",
        "token_per_second",
    ):
        value = _numeric_metric(metrics.get(name))
        if value is not None:
            return value
    tpot_microseconds = _numeric_metric(metrics.get("tpot_microseconds"))
    if tpot_microseconds is not None and tpot_microseconds > 0.0:
        return 1_000_000.0 / tpot_microseconds
    return None


def _format_gpu_kernel_time(gpu: dict[str, Any], available: bool) -> str:
    if not available:
        return "N/A (CPU-only)"
    if gpu.get("kernel_time_available", False):
        return _format_duration_microseconds(gpu.get("kernel_time_microseconds"))
    if gpu.get("reason") == "gpu_expert_execution_not_observed":
        return "N/A (no GPU Expert execution)"
    return "N/A"


def _format_runtime_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return "metrics\n  unavailable"
    expert = metrics.get("expert", {})
    if not isinstance(expert, dict):
        expert = {}
    cache_hit_rate = expert.get("cache_hit_rate")
    if not isinstance(cache_hit_rate, (int, float)):
        cache_hit = expert.get("cache_hit")
        cache_miss = expert.get("cache_miss")
        if isinstance(cache_hit, (int, float)) and isinstance(cache_miss, (int, float)) and cache_hit + cache_miss:
            cache_hit_rate = cache_hit / (cache_hit + cache_miss)
    cpu = metrics.get("cpu", {})
    if not isinstance(cpu, dict):
        cpu = {"expert_compute_time_microseconds": metrics.get("expert_compute_time_microseconds")}
    gpu = metrics.get("gpu", {})
    if not isinstance(gpu, dict):
        gpu = {}
    gpu_device = metrics.get("gpu_device", {})
    if not isinstance(gpu_device, dict) or not gpu_device:
        gpu_device = metrics.get("gpu", {}) if isinstance(metrics.get("gpu"), dict) else {}
    process = metrics.get("process", {})
    gpu_available = bool(gpu.get("available", gpu_device.get("active", False)))
    process_cpu = process.get("cpu_percent") if isinstance(process, dict) else None
    process_cpu_text = (
        f"{float(process_cpu):.1f}%"
        if isinstance(process_cpu, (int, float))
        else "N/A"
    )
    cache_hit_text = _format_metric_count(expert.get("cache_hit"))
    if isinstance(cache_hit_rate, (int, float)):
        cache_hit_text += f" ({float(cache_hit_rate) * 100:.1f}%)"
    return "\n".join(
        (
            "metrics",
            "  Prompt: "
            f"{_format_rate(_prompt_rate(metrics))} t/s | Generation: "
            f"{_format_rate(_generation_rate(metrics))} t/s"
            " · TTFT "
            f"{_format_duration_microseconds(metrics.get('ttft_microseconds'))}"
            " · TPOT "
            f"{_format_duration_microseconds(metrics.get('tpot_microseconds'))}",
            "  Expert: cache hit "
            f"{cache_hit_text}"
            " · cache miss "
            f"{_format_metric_count(expert.get('cache_miss'))}"
            " · IO "
            f"{_format_bytes_gb(expert.get('io_bytes'))}",
            "  GPU Expert: cache hit "
            f"{_format_metric_count(expert.get('gpu_cache_hit'))}"
            " / miss "
            f"{_format_metric_count(expert.get('gpu_cache_miss'))}"
            " / exec "
            f"{_format_metric_count(expert.get('gpu_executions'))}"
            " / resident "
            f"{_format_bytes_gb(expert.get('gpu_cache_resident_bytes'))}"
            " / dropped "
            f"{_format_metric_count(expert.get('gpu_cache_dropped_admissions'))}",
            "  CPU: expert compute "
            f"{_format_duration_microseconds(cpu.get('expert_compute_time_microseconds'))}"
            f" · process {process_cpu_text}",
            "  GPU: submit "
            f"{_format_metric_count(gpu.get('submit_count'), available=gpu_available)}"
            " · wait "
            f"{_format_duration_microseconds(gpu.get('wait_time_microseconds') if gpu_available else None)}"
            " · kernel "
            f"{_format_gpu_kernel_time(gpu, gpu_available)}"
            " · utilization "
            f"{_format_gpu_utilization(gpu_device)}"
            " / attention "
            f"{_format_metric_count(gpu.get('attention_blocks'), available=gpu_available)}"
            " / linear "
            f"{_format_metric_count(gpu.get('linear_dispatches'), available=gpu_available)}"
            " / gated-delta "
            f"{_format_metric_count(gpu.get('gated_delta_fusions'), available=gpu_available)}"
            " / upload/download "
            f"{_format_metric_count(gpu.get('batch_uploads'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('batch_downloads'), available=gpu_available)}"
            " / qkv-rope/device-rope/ring "
            f"{_format_metric_count(gpu.get('attention_qkv_rope_fusions'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('attention_device_rope_fusions'), available=gpu_available)}"
            "/"
            f"{_format_metric_count(gpu.get('attention_qkv_ring_fusions'), available=gpu_available)}"
            " / kv-materialize "
            f"{_format_metric_count(gpu.get('attention_cache_materializations'), available=gpu_available)}"
            " / attention-cpu-fallback "
            f"{_format_metric_count(gpu.get('attention_cpu_fallbacks'), available=gpu_available)}",
        )
    )


def _add_model_position(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_pos", nargs="?", help="Model directory")
    parser.add_argument("--model", dest="model_opt", help="Model directory")


def _add_worker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worker",
        help="Override the default worker under build-ncnn",
    )
    parser.add_argument("--config-dir", help="Override the persistent configuration directory")
    parser.add_argument("--backend", choices=("auto", "cpu", "hybrid"))
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--cpu", dest="backend", action="store_const", const="cpu")
    backend_group.add_argument("--hybrid", dest="backend", action="store_const", const="hybrid")
    parser.add_argument("--expert-memory", choices=("auto", "eager", "on-demand"))
    parser.add_argument(
        "--cpu-packed-weights",
        choices=("off", "on"),
        help="Enable the optional CPU Expert weight repack (default: off)",
    )
    parser.add_argument("--host-memory-mb", type=int)
    parser.add_argument("--expert-cache-mb", type=int)
    parser.add_argument("--expert-gpu-cache-mb", type=int)
    parser.add_argument("--expert-gpu-victim-cache-mb", type=int)
    parser.add_argument("--expert-io-workers", type=int)
    parser.add_argument("--vulkan-device", type=int)
    parser.add_argument("--vulkan-devices", help="Comma-separated Vulkan device indices")
    parser.add_argument("--expected-concurrency", type=int)
    expert_io_group = parser.add_mutually_exclusive_group()
    expert_io_group.add_argument("--mmap-experts", action="store_true", default=None)
    expert_io_group.add_argument("--direct-expert-io", action="store_true", default=None)
    expert_io_group.add_argument("--buffered-expert-io", action="store_true", default=None)
    parser.add_argument("--disable-gpu-victim-execution", action="store_true", default=None)
    parser.add_argument("--release-vulkan-dense-host", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true")


def _add_metrics_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics-interval-ms",
        type=int,
        default=1000,
        help="Periodic metrics event interval; 0 disables the metrics trace",
    )
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--metrics",
        dest="metrics_enabled",
        action="store_true",
        help="Enable periodic metrics events",
    )
    metrics_group.add_argument(
        "--no-metrics",
        dest="metrics_enabled",
        action="store_false",
        help="Disable periodic metrics events while retaining final statistics",
    )
    parser.set_defaults(metrics_enabled=False)


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
    _add_metrics_options(parser)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", help="Stream generated text (default)")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false", help="Print the completed response only")
    parser.set_defaults(stream=True)
    parser.add_argument("--stream-final-only", action="store_true")
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--show-reasoning", dest="show_reasoning", action="store_true", help="Show reasoning (default)")
    reasoning_group.add_argument("--hide-reasoning", dest="show_reasoning", action="store_false", help="Hide reasoning from the display")
    parser.set_defaults(show_reasoning=True)


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
    tune.add_argument("--prefill-chunk-size", type=int, default=512)
    tune.add_argument("--seed", type=int, default=0)
    _add_metrics_options(tune)
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


def default_worker_path(root: Path) -> Path:
    """Return the one conventional worker location used by the CLI."""
    worker_name = "ncnn_moe_worker.exe" if os.name == "nt" else "ncnn_moe_worker"
    build_dir = root / "build-ncnn"
    if os.name == "nt":
        return build_dir / "Release" / worker_name
    return build_dir / worker_name


def find_worker(explicit: str | None, root: Path) -> Path:
    """Resolve an explicitly selected worker or the fixed build-ncnn default."""
    source = "default"
    if explicit:
        candidate = Path(explicit).expanduser()
        source = "--worker"
    else:
        environment = os.environ.get("NCNN_MOE_WORKER")
        if environment:
            candidate = Path(environment).expanduser()
            source = "NCNN_MOE_WORKER"
        else:
            candidate = default_worker_path(root)

    if candidate.is_file():
        return candidate.resolve()

    resolved = candidate.resolve()
    if source == "default":
        raise ValueError(
            f"default worker was not found at {resolved}; build ncnn_moe_worker "
            "in build-ncnn or pass --worker PATH"
        )
    raise ValueError(f"worker from {source} was not found at {resolved}")


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
        "cpu_packed_weights",
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
    select_adapter: bool = False,
) -> tuple[WorkerClient, dict[str, Any], dict[str, Any] | None, ModelAdapter | None]:
    source_root = Path(__file__).resolve().parents[1]
    root = source_root if (source_root / "CMakeLists.txt").is_file() else Path.cwd().resolve()
    worker = find_worker(getattr(arguments, "worker", None), root)
    user = user_runtime_settings(store)
    cli = cli_runtime_settings(arguments)
    session_settings = session.get("settings", {}) if session else None
    if not isinstance(session_settings, dict):
        session_settings = {}
    initial = merge_runtime_settings(cli=cli, session=session_settings, profile=None, user=user)
    client = WorkerClient(
        worker,
        model,
        runtime_args_from_settings(initial),
        verbose=getattr(arguments, "verbose", False),
    )
    if select_adapter and adapter is None:
        try:
            adapter = load_adapter(arguments, model, client.ready)
        except BaseException:
            client.close()
            raise
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
        session=session_settings,
        profile=profile_settings,
        user=user,
    )
    if merged != initial:
        client.close()
        client = WorkerClient(
            worker,
            model,
            runtime_args_from_settings(merged),
            verbose=getattr(arguments, "verbose", False),
        )
    return client, merged, profile_record, adapter


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
    if arguments.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
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
        if not self.native_text:
            self.native_context_tokens: list[int] = []
        self.native_context_count = 0
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
        if self.adapter.model_type == "qwen3_5_moe":
            leading_system_count = 0
            while (
                leading_system_count < len(self.messages)
                and self.messages[leading_system_count].get("role") == "system"
            ):
                leading_system_count += 1
            if leading_system_count > 1:
                summary_marker = f"Conversation summary:\n{self.summary}" if self.summary else ""
                summary_suffix = f"\n\n{summary_marker}" if summary_marker else ""
                summary_seen = False
                system_parts: list[str] = []
                for message in self.messages[:leading_system_count]:
                    content = message["content"]
                    if summary_marker and content == summary_marker:
                        if summary_seen:
                            continue
                        summary_seen = True
                    elif summary_suffix and content.endswith(summary_suffix):
                        original = content[: -len(summary_suffix)]
                        if summary_seen:
                            content = original
                        else:
                            if original:
                                system_parts.append(original)
                            system_parts.append(summary_marker)
                            summary_seen = True
                            continue
                    if content:
                        system_parts.append(content)
                merged_system = "\n\n".join(system_parts)
                self.messages = [
                    {"role": "system", "content": merged_system},
                    *self.messages[leading_system_count:],
                ]
        self.client.create_session(
            self.native_session_id,
            seed=arguments.seed,
            prefill_chunk_size=arguments.prefill_chunk_size,
            enable_speculative_context=not arguments.no_speculative,
        )
        replay_succeeded = False
        if record and self.messages:
            try:
                replay = self._compact_messages(self.native_session_id, self.messages)
                self.native_context_count = int(replay.get("sequence_length", 0) or 0)
                replay_succeeded = True
                self._status(f"resumed {record.get('id')} ({len(self.messages)} messages, {self.native_context_count} context tokens)")
            except (AdapterError, WorkerError, ValueError) as error:
                if not self.native_text:
                    self.native_context_tokens = []
                self.native_context_count = 0
                self._status(f"session replay deferred: {error}")
        if (
            record
            and hasattr(self.adapter, "_legacy_model_fingerprint")
            and record.get("model_fingerprint") != self.adapter.model_fingerprint
            and (replay_succeeded or not self.messages)
        ):
            self._save()

    @property
    def context_limit(self) -> int | None:
        return context_budget(self.adapter, self.arguments, self.client.ready)

    @property
    def native_text(self) -> bool:
        return bool(getattr(self.adapter, "native_text", False))

    @property
    def worker_context_tokens(self) -> int | None:
        override = int(getattr(self.arguments, "context_tokens", 0) or 0)
        if override > 0:
            return override
        model_limit = int(self.client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
        return model_limit or None

    def _generate_messages(self, session_id: str, messages: list[dict[str, str]], **options: Any) -> tuple[dict[str, Any], list[int]]:
        if self.native_text:
            options.pop("stop_tokens", None)
            return self.client.generate(
                session_id,
                messages=messages,
                thinking=self.adapter.thinking,
                context_tokens=self.worker_context_tokens,
                **options,
            )
        return self.client.generate(session_id, self.adapter.encode_messages(messages), **options)

    def _compact_messages(self, session_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.native_text:
            return self.client.compact(
                session_id,
                messages=messages,
                thinking=self.adapter.thinking,
                context_tokens=self.worker_context_tokens,
            )
        prompt_tokens = self.adapter.encode_messages(messages)
        replay = self.client.compact(session_id, prompt_tokens)
        sequence_length = replay.get("sequence_length")
        if (
            isinstance(sequence_length, int)
            and not isinstance(sequence_length, bool)
            and sequence_length == len(prompt_tokens)
        ):
            self.native_context_tokens = prompt_tokens
        else:
            self.native_context_tokens = []
        return replay

    def _prompt_count(self, session_id: str, messages: list[dict[str, str]]) -> int:
        if self.native_text:
            event = self.client.stats(session_id, messages=messages, thinking=self.adapter.thinking)
            count = event.get("prompt_count")
            if isinstance(count, bool) or not isinstance(count, int):
                raise WorkerError("worker did not return the native prompt count", event)
            return count
        return len(self.adapter.encode_messages(messages))

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
        stored_fingerprint = self.record.get("model_fingerprint")
        legacy_fingerprint = getattr(self.adapter, "_legacy_model_fingerprint", None)
        migration_ready = not self.messages or (
            self.native_context_count > 0 if self.native_text else bool(self.native_context_tokens)
        )
        fingerprint_migrated = False
        if hasattr(self.adapter, "_legacy_model_fingerprint"):
            compatible_fingerprint = (
                stored_fingerprint is None
                or stored_fingerprint == self.adapter.model_fingerprint
                or stored_fingerprint == legacy_fingerprint
            )
            if compatible_fingerprint and (
                stored_fingerprint == self.adapter.model_fingerprint or migration_ready
            ):
                fingerprint_migrated = stored_fingerprint != self.adapter.model_fingerprint
                self.record["model_fingerprint"] = self.adapter.model_fingerprint
        else:
            self.record["model_fingerprint"] = self.adapter.model_fingerprint
        self.record["context_token_count"] = self.native_context_count if self.native_text else len(self.native_context_tokens)
        if not self.ephemeral:
            self.store.save(self.record)
            if fingerprint_migrated:
                self._status("session tokenizer profile updated")

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
            summary_tokens: list[int] = []
            summary_stream: dict[str, Any] = {}
            _, tokens = self._generate_messages(
                self.summary_session_id,
                summary_prompt,
                request_id="compact-summary",
                max_new_tokens=min(512, self.arguments.max_new_tokens) if self.arguments.max_new_tokens else 512,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                min_p=0.0,
                stop_tokens=self.adapter.stop_tokens,
                enable_speculative=False,
                metrics_enabled=metrics_trace_enabled(self.arguments),
                metrics_interval_ms=self.arguments.metrics_interval_ms,
                on_event=lambda event: self._on_generation_event(
                    event, summary_tokens, summary_stream, display=False
                ),
            )
            completion = self._decode_completion(tokens, summary_stream)
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
            if not self.native_text:
                self.native_context_tokens = []
            self.native_context_count = 0
            if announce:
                self._status("context reset; not enough history to summarize")
            return
        old_messages = self.messages
        old_summary = self.summary
        summary = self._summarize()
        system_messages = [message for message in old_messages if message.get("role") == "system"]
        retained = [message for message in old_messages if message.get("role") != "system"][-4:]
        previous_summary = f"Conversation summary:\n{old_summary}" if old_summary else ""
        previous_suffix = f"\n\n{previous_summary}" if previous_summary else ""
        system_content_parts = []
        for message in system_messages:
            content = message["content"]
            if previous_summary and content == previous_summary:
                continue
            if previous_suffix and content.endswith(previous_suffix):
                content = content[: -len(previous_suffix)]
            if content:
                system_content_parts.append(content)
        system_content = "\n\n".join(system_content_parts)
        summary_content = f"Conversation summary:\n{summary}"
        combined_system = "\n\n".join(content for content in (system_content, summary_content) if content)
        self.messages = [{"role": "system", "content": combined_system}] + retained
        self.summary = summary
        try:
            if self.native_text:
                replay = self._compact_messages(self.native_session_id, self.messages)
                self.native_context_count = int(replay.get("sequence_length", 0) or 0)
            else:
                self.client.compact(self.native_session_id, [])
                self.native_context_tokens = []
        except WorkerError:
            self.messages = old_messages
            self.summary = old_summary
            self._trim_sliding_window()
            self.client.reset(self.native_session_id)
            if not self.native_text:
                self.native_context_tokens = []
            self.native_context_count = 0
            if announce:
                self._status("native compact failed; using sliding-window fallback")
            return
        if not self.native_text:
            self.native_context_tokens = []
            self.native_context_count = 0
        if announce:
            self._status(f"context compacted · retained {len(self.messages)} messages")

    def _trim_sliding_window(self, prompt_count: int | None = None) -> None:
        budget = self.worker_context_tokens if self.native_text else self.context_limit
        if budget is None:
            return
        if type(prompt_count) is not int or prompt_count <= 0:
            prompt_count = None
        for _ in range(100):
            if prompt_count is None:
                prompt_count = self._prompt_count(self.native_session_id, self.messages)
            generation_budget = self.arguments.max_new_tokens or 1
            if prompt_count + generation_budget <= budget:
                return
            prompt_count = None
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

    def _on_generation_event(
        self,
        event: dict[str, Any],
        generated_tokens: list[int],
        streamed: dict[str, Any],
        *,
        display: bool = True,
    ) -> None:
        if event.get("event") == "metrics":
            streamed["metrics"] = json.dumps(event.get("metrics", {}), ensure_ascii=False)
            self.last_metrics = event.get("metrics", {})
            if display:
                self._status(_format_runtime_metrics(self.last_metrics))
            return
        if event.get("event") != "token":
            return
        token_id = int(event["token_id"])
        text = event.get("text")
        if self.native_text and not isinstance(text, str):
            raise WorkerError("native Worker token event did not include decoded text", event)
        if self.native_text and isinstance(text, str):
            streamed.setdefault("decoded_chunks", []).append(text)
            streamed["native_started"] = True
        if not display or not self.arguments.stream:
            return
        chunks = streamed.setdefault("chunks", [])
        if not self.native_text:
            generated_tokens.append(token_id)
            visible = self.adapter.stream_visible(
                generated_tokens,
                final_only=self.arguments.stream_final_only or not self.arguments.show_reasoning,
            )
            printed = "".join(chunks)
            if visible.startswith(printed):
                delta = visible[len(printed) :]
                if delta:
                    chunks.append(delta)
                    streamed["printed_length"] = streamed.get("printed_length", 0) + len(delta)
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            return

        if not isinstance(text, str):
            return
        if "phase" not in streamed:
            streamed["phase"] = "reasoning" if self.adapter.thinking else "body"
            streamed["marker_carry"] = ""
            streamed["tail"] = []
            streamed["reasoning_leading"] = True
            streamed["body_leading"] = True
            streamed["answer_leading"] = True
            streamed["reasoning_nonempty"] = False
            streamed["answer_nonempty"] = False

        output: list[str] = []
        marker = "</think>"
        phase = streamed["phase"]
        segments: list[tuple[str, str, bool]]
        if phase == "reasoning":
            combined = streamed["marker_carry"] + text
            marker_index = combined.find(marker)
            if marker_index >= 0:
                segments = [
                    ("reasoning", combined[:marker_index], True),
                    ("answer", combined[marker_index + len(marker) :], False),
                ]
                streamed["phase"] = "answer"
                streamed["marker_carry"] = ""
            else:
                carry_length = 0
                for length in range(min(len(marker) - 1, len(combined)), 0, -1):
                    if combined.endswith(marker[:length]):
                        carry_length = length
                        break
                streamed["marker_carry"] = combined[-carry_length:] if carry_length else ""
                safe_text = combined[:-carry_length] if carry_length else combined
                segments = [("reasoning", safe_text, False)]
        else:
            segments = [(phase, text, False)]

        final_only = self.arguments.stream_final_only or not self.arguments.show_reasoning
        tail: list[str] = streamed["tail"]
        for kind, segment, terminal in segments:
            position = 0
            if kind == "reasoning":
                while position < len(segment) and streamed["reasoning_leading"] and segment[position].isspace():
                    position += 1
                if position < len(segment):
                    streamed["reasoning_leading"] = False
            elif kind == "body":
                while position < len(segment) and streamed["body_leading"] and segment[position].isspace():
                    position += 1
                if position < len(segment):
                    streamed["body_leading"] = False
            else:
                while position < len(segment) and streamed["answer_leading"] and segment[position] in "\r\n":
                    position += 1
                if position < len(segment):
                    streamed["answer_leading"] = False

            while position < len(segment):
                start = position
                candidate = segment[position].isspace() or segment[position] == "\ufffd"
                position += 1
                if candidate:
                    while position < len(segment) and (segment[position].isspace() or segment[position] == "\ufffd"):
                        position += 1
                    tail.append(segment[start:position])
                    continue
                while position < len(segment) and not (segment[position].isspace() or segment[position] == "\ufffd"):
                    position += 1
                safe_text = segment[start:position]
                pending = "".join(tail)
                tail.clear()
                if kind == "reasoning":
                    if not final_only:
                        if not streamed["reasoning_nonempty"]:
                            output.append("[reasoning]\n")
                        if pending:
                            output.append(pending)
                        output.append(safe_text)
                    streamed["reasoning_nonempty"] = True
                elif kind == "answer":
                    if not final_only and not streamed["answer_nonempty"]:
                        label = "\n[answer]\n" if streamed["reasoning_nonempty"] else "[answer]\n"
                        output.append(label)
                    if pending:
                        output.append(pending)
                    output.append(safe_text)
                    streamed["answer_nonempty"] = True
                else:
                    if not final_only and not streamed["answer_nonempty"]:
                        output.append("[answer]\n")
                    if pending:
                        output.append(pending)
                    output.append(safe_text)
                    streamed["answer_nonempty"] = True

            if terminal:
                pending = "".join(tail).rstrip()
                tail.clear()
                if pending:
                    if not final_only:
                        if not streamed["reasoning_nonempty"]:
                            output.append("[reasoning]\n")
                        output.append(pending)
                    streamed["reasoning_nonempty"] = True

        delta = "".join(output)
        if delta:
            chunks.append(delta)
            streamed["printed_length"] = streamed.get("printed_length", 0) + len(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()

    def _decode_completion(self, tokens: list[int], streamed: dict[str, Any]) -> Completion:
        if self.native_text:
            text = "".join(streamed.get("decoded_chunks", []))
            return self.adapter.decode_native_completion(text, tokens)
        return self.adapter.decode_completion(tokens)

    def send(self, user_text: str) -> tuple[Completion, dict[str, Any], bool]:
        self.messages.append({"role": "user", "content": user_text})
        generated_tokens: list[int] = []
        streamed: dict[str, Any] = {"chunks": [], "decoded_chunks": []}
        temperature, top_k, top_p = default_generation_values(self.adapter, self.arguments)
        request_id = f"chat-{int(time.time() * 1000)}"
        options = {
            "request_id": request_id,
            "max_new_tokens": self.arguments.max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": self.arguments.min_p,
            "stop_tokens": self.adapter.stop_tokens,
            "enable_speculative": not self.arguments.no_speculative,
            "speculative_confidence": self.arguments.speculative_confidence,
            "speculative_max_draft": self.arguments.speculative_max_draft,
            "metrics_enabled": metrics_trace_enabled(self.arguments),
            "metrics_interval_ms": self.arguments.metrics_interval_ms,
            "on_event": lambda event: self._on_generation_event(event, generated_tokens, streamed),
        }
        if self.native_text:
            def generate_native() -> tuple[dict[str, Any], list[int]]:
                try:
                    return self._generate_messages(self.native_session_id, self.messages, **options)
                except WorkerError as error:
                    if error.event.get("code") not in {"context_overflow", "invalid_request"}:
                        self.native_context_count = 0
                    raise

            try:
                done, generated_tokens = generate_native()
            except WorkerError as error:
                if error.event.get("code") != "context_overflow":
                    raise
                pending_user = self.messages.pop()
                self.compact()
                self.messages.append(pending_user)
                try:
                    done, generated_tokens = generate_native()
                except WorkerError as retry_error:
                    if retry_error.event.get("code") != "context_overflow":
                        raise
                    self._trim_sliding_window(retry_error.event.get("prompt_count"))
                    done, generated_tokens = generate_native()
            reused = bool(done.get("prefix_reused", False))
            completion = self._decode_completion(generated_tokens, streamed)
            if self.arguments.stream and streamed.get("native_started"):
                visible = self.adapter.visible_completion(
                    completion,
                    final_only=self.arguments.stream_final_only or not self.arguments.show_reasoning,
                )
                chunks = streamed["chunks"]
                printed = "".join(chunks)
                if visible.startswith(printed):
                    delta = visible[len(printed) :]
                    if delta:
                        chunks.append(delta)
                        streamed["printed_length"] = streamed.get("printed_length", 0) + len(delta)
                        sys.stdout.write(delta)
                        sys.stdout.flush()
            sequence_length = done.get("sequence_length")
            self.native_context_count = (
                sequence_length
                if isinstance(sequence_length, int) and not isinstance(sequence_length, bool)
                else 0
            )
        else:
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
            self.native_context_tokens = []
            done, generated_tokens = self.client.generate(self.native_session_id, input_tokens, **options)
            completion = self._decode_completion(generated_tokens, streamed)
            base_count = len(native_base) + len(input_tokens)
            sequence_length = done.get("sequence_length")
            if (
                isinstance(sequence_length, int)
                and not isinstance(sequence_length, bool)
                and base_count <= sequence_length <= base_count + len(generated_tokens)
            ):
                committed_count = sequence_length - base_count
                native_base.extend(input_tokens)
                native_base.extend(generated_tokens[:committed_count])
                self.native_context_tokens = native_base
            else:
                self.native_context_tokens = []
        self.last_reasoning = completion.reasoning
        self.messages.append({"role": "assistant", "content": completion.answer or completion.reasoning})
        self._save()
        if self.arguments.stream and streamed.get("printed_length", 0):
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
        metrics = done.get("metrics", {})
        self.last_metrics = metrics if isinstance(metrics, dict) else {}
        context_count = self.native_context_count if self.native_text else len(self.native_context_tokens)
        self._status(
            _format_runtime_metrics(self.last_metrics)
            + "\n  Context "
            + f"{context_count}/{self.context_limit or 'auto'}"
            + " tokens"
            + f" · prefix {'reused' if reused else 'replayed'}"
        )

    def show_context(self) -> None:
        tokens = self._prompt_count(self.native_session_id, self.messages) if self.messages else 0
        budget = self.context_limit
        self._status(
            f"messages={len(self.messages)} prompt_tokens={tokens} "
            f"budget={budget or 'unknown'} native_sequence={self.native_context_count if self.native_text else len(self.native_context_tokens)}"
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
            temperature, top_k, top_p = default_generation_values(self.adapter, self.arguments)
            done, _ = self._generate_messages(
                tune_session,
                [{"role": "user", "content": prompt}],
                request_id="tune-live",
                max_new_tokens=min(32, self.arguments.max_new_tokens) if self.arguments.max_new_tokens else 32,
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
        if not self.native_text:
            self.native_context_tokens = []
        self.native_context_count = 0
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
            if not self.native_text:
                self.native_context_tokens = []
            self.native_context_count = 0
            self.client.reset(self.native_session_id)
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


def load_adapter(arguments: argparse.Namespace, model: Path, ready: dict[str, Any] | None = None) -> ModelAdapter:
    thinking_mode = getattr(arguments, "thinking_mode", "thinking")
    thinking = thinking_mode != "chat" and not getattr(arguments, "no_thinking", False)
    model_info = ready.get("model", {}) if isinstance(ready, dict) else {}
    native_version = model_info.get("native_text_version") if isinstance(model_info, dict) else None
    if (
        isinstance(model_info, dict)
        and model_info.get("native_text_supported") is True
        and native_version == "ncnn-moe-qwen3.6-nfc9-regex16-v1"
    ):
        stop_tokens = model_info.get("native_stop_tokens")
        if not isinstance(stop_tokens, list) or not stop_tokens:
            stop_tokens = None
        if stop_tokens is None:
            return create_adapter(model, thinking=thinking, thinking_mode=thinking_mode)
        try:
            adapter = NativeQwenAdapter(
                model,
                native_text_version=native_version,
                native_stop_tokens=stop_tokens,
                thinking=thinking,
                thinking_mode=thinking_mode,
            )
            adapter.validate()
            return adapter
        except (AdapterError, OSError):
            pass
    return create_adapter(model, thinking=thinking, thinking_mode=thinking_mode)


def run_command(arguments: argparse.Namespace) -> int:
    validate_generation_arguments(arguments)
    model = resolve_model(arguments)
    prompt = resolve_prompt(arguments)
    assert model is not None and prompt is not None
    store = SessionStore()
    client, settings, _, adapter = open_worker(arguments, model, store, select_adapter=True)
    assert adapter is not None
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
    client, settings, _, adapter = open_worker(arguments, model, store, session=record, select_adapter=True)
    assert adapter is not None
    try:
        legacy_fingerprint = getattr(adapter, "_legacy_model_fingerprint", None)
        stored_fingerprint = record.get("model_fingerprint") if record else None
        if (
            record
            and stored_fingerprint is not None
            and stored_fingerprint != adapter.model_fingerprint
            and stored_fingerprint != legacy_fingerprint
        ):
            raise ValueError("the requested session belongs to a different model fingerprint")
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
    client, _, _, _ = open_worker(arguments, model, store)
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
            print(
                "CPU packed weights: "
                f"{resources.get('selected_cpu_packed_weights', 'off')}"
            )
            print(f"host memory: {_format_bytes_gb(resources.get('host_memory_budget_bytes'))}")
            print(f"Expert cache: {_format_bytes_gb(resources.get('expert_cache_bytes'))}")
            print(
                "packed Expert estimate: "
                f"{_format_bytes_gb(resources.get('estimated_cpu_packed_expert_bytes'))}"
            )
            print(f"Expert IO workers: {resources.get('expert_io_workers', 'N/A')}")
            print(f"CPU: {capabilities.get('physical_cpu_core_count', 'N/A')} physical / {capabilities.get('logical_cpu_count', 'N/A')} logical")
            print(f"Vulkan devices: {capabilities.get('vulkan_device_count', 0)}")
            print(f"GPU telemetry: {_format_gpu_utilization(telemetry)}")
            for device in capabilities.get("vulkan_devices", []):
                print(
                    f"  [{device.get('index')}] {device.get('name')} "
                    f"({device.get('type')}) heap={_format_bytes_gb(device.get('heap_budget_bytes'))}"
                )
        return 0
    finally:
        client.close()


def tune_command(arguments: argparse.Namespace) -> int:
    if arguments.runs <= 0 or arguments.warmup < 0:
        raise ValueError("--runs must be positive and --warmup must be non-negative")
    model = resolve_model(arguments)
    prompt = resolve_prompt(arguments)
    assert model is not None and prompt is not None
    store = SessionStore()
    client, settings, _, adapter = open_worker(arguments, model, store, select_adapter=True)
    assert adapter is not None
    try:
        client.create_session("tune", seed=arguments.seed, prefill_chunk_size=arguments.prefill_chunk_size, enable_speculative_context=False)
        prompt_messages = [{"role": "user", "content": prompt}]
        native_text = bool(getattr(adapter, "native_text", False))
        prompt_tokens = None if native_text else adapter.encode_messages(prompt_messages)
        temperature = 0.0 if adapter.name == "gpt-oss" else 1.0
        rates: list[float] = []
        for index in range(arguments.warmup + arguments.runs):
            if index:
                client.reset("tune")
            options: dict[str, Any] = {
                "request_id": f"tune-{index}",
                "max_new_tokens": arguments.max_new_tokens,
                "temperature": temperature,
                "top_k": 0,
                "top_p": 1.0,
                "enable_speculative": False,
                "metrics_enabled": metrics_trace_enabled(arguments),
                "metrics_interval_ms": arguments.metrics_interval_ms if hasattr(arguments, "metrics_interval_ms") else 1000,
            }
            if not native_text:
                options["stop_tokens"] = adapter.stop_tokens
            if native_text:
                model_limit = int(client.ready.get("model", {}).get("max_context_tokens", 0) or 0)
                context_tokens = int(getattr(arguments, "context_tokens", 0) or 0) or model_limit or None
                done, _ = client.generate(
                    "tune",
                    messages=prompt_messages,
                    thinking=adapter.thinking,
                    context_tokens=context_tokens,
                    **options,
                )
            else:
                done, _ = client.generate("tune", prompt_tokens, **options)
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
