"""Benchmark one tokenizer-backed prompt through the native worker.

This intentionally reports native ``done`` metrics instead of inferring GPU
execution from device memory occupancy.  It is useful for before/after
comparisons of CPU and Hybrid runtime changes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from ncnn_moe_protocol import WorkerClient  # noqa: E402


DEFAULT_PROMPT = "你好，你能做些什么"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "hybrid"), default="cpu")
    parser.add_argument("--vulkan-device", type=int, default=0)
    parser.add_argument(
        "--expert-io-workers",
        type=int,
        default=0,
        help="override the file-backed Expert I/O worker count",
    )
    parser.add_argument(
        "--host-memory-mb",
        type=int,
        default=0,
        help="explicit host model/Expert cache budget in MiB",
    )
    parser.add_argument(
        "--expert-cache-mb",
        type=int,
        default=0,
        help="explicit CPU Expert cache capacity in MiB",
    )
    parser.add_argument(
        "--expert-gpu-cache-mb",
        type=int,
        default=0,
        help="explicit executable GPU Expert cache capacity in MiB",
    )
    parser.add_argument(
        "--expert-gpu-victim-cache-mb",
        type=int,
        default=0,
        help="explicit GPU Expert victim cache capacity in MiB",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-token-ids",
        type=int,
        nargs="+",
        default=None,
        help="use these native token IDs directly; useful for model-specific prompt tests",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="maximum generated tokens; 0 runs until EOS/stop or model context limit",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--expert-memory",
        choices=("auto", "eager", "on-demand"),
        default="auto",
        help="Expert residency mode passed to the ncnn-MoE worker",
    )
    parser.add_argument("--enable-speculative", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def _runtime_args(arguments: argparse.Namespace) -> list[str]:
    if arguments.backend == "cpu":
        runtime_args = ["--cpu"]
    else:
        mode = "--hybrid"
        runtime_args = [mode, "--vulkan-device", str(arguments.vulkan_device)]
    if arguments.host_memory_mb > 0:
        runtime_args.extend(["--host-memory-mb", str(arguments.host_memory_mb)])
    if arguments.expert_cache_mb > 0:
        runtime_args.extend(["--expert-cache-mb", str(arguments.expert_cache_mb)])
    if arguments.expert_io_workers > 0:
        runtime_args.extend(["--expert-io-workers", str(arguments.expert_io_workers)])
    if arguments.expert_gpu_cache_mb > 0:
        runtime_args.extend(["--expert-gpu-cache-mb", str(arguments.expert_gpu_cache_mb)])
    if arguments.expert_gpu_victim_cache_mb > 0:
        runtime_args.extend(["--expert-gpu-victim-cache-mb", str(arguments.expert_gpu_victim_cache_mb)])
    if arguments.expert_memory:
        runtime_args.extend(["--expert-memory", arguments.expert_memory])
    return runtime_args


def _summary(done_events: list[dict[str, Any]], generated: list[list[int]]) -> dict[str, Any]:
    prompt_tok_s = [
        float(event["prompt_tok_per_second"])
        for event in done_events
        if event.get("prompt_tok_per_second") is not None
    ]
    generation_tok_s = [
        float(event.get("generation_tok_per_second", event.get("decode_tok_per_second")))
        for event in done_events
        if event.get("generation_tok_per_second", event.get("decode_tok_per_second")) is not None
    ]
    elapsed = [float(event["elapsed_seconds"]) for event in done_events]
    ttft = [
        float(event["ttft_microseconds"]) / 1_000_000.0
        if event.get("ttft_microseconds") is not None
        else None
        for event in done_events
    ]
    tpot = [
        float(event["tpot_microseconds"]) / 1_000_000.0
        if event.get("tpot_microseconds") is not None
        else None
        for event in done_events
    ]
    last = done_events[-1]
    metrics = last.get("metrics", {})
    expert = metrics.get("expert", {})
    gpu = metrics.get("gpu", {})
    sequence_match = bool(generated) and all(token_ids == generated[0] for token_ids in generated[1:])
    return {
        "runs": len(done_events),
        "prompt_tokens_per_second": prompt_tok_s,
        "median_prompt_tokens_per_second": statistics.median(prompt_tok_s) if prompt_tok_s else None,
        "generation_tokens_per_second": generation_tok_s,
        "median_generation_tokens_per_second": statistics.median(generation_tok_s) if generation_tok_s else None,
        # Backward-compatible alias.
        "decode_tokens_per_second": generation_tok_s,
        "median_decode_tokens_per_second": statistics.median(generation_tok_s) if generation_tok_s else None,
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": statistics.median(elapsed),
        "ttft_seconds": ttft,
        "median_ttft_seconds": statistics.median(value for value in ttft if value is not None),
        "tpot_seconds": tpot,
        "median_tpot_seconds": statistics.median(value for value in tpot if value is not None) if any(value is not None for value in tpot) else None,
        "generated_tokens": int(last.get("generated_tokens", len(generated[-1]))),
        "generated_token_ids": generated[-1],
        "generated_sequences_match": sequence_match,
        "generated_token_ids_by_run": generated,
        "expert_gpu_executions": int(expert.get("gpu_executions", 0)),
        "expert_gpu_execution_failures": int(expert.get("gpu_execution_failures", 0)),
        "expert_gpu_route_aggregation_batches": int(expert.get("gpu_route_aggregation_batches", 0)),
        "expert_gpu_route_aggregation_routes": int(expert.get("gpu_route_aggregation_routes", 0)),
        "expert_gpu_route_aggregation_bytes_saved": int(expert.get("gpu_route_aggregation_bytes_saved", 0)),
        "expert_gpu_cache_hits": int(expert.get("gpu_cache_hit", 0)),
        "expert_gpu_cache_misses": int(expert.get("gpu_cache_miss", 0)),
        "expert_gpu_cache_pending_bytes": int(expert.get("gpu_cache_pending_bytes", 0)),
        "expert_gpu_cache_resident_bytes": int(expert.get("gpu_cache_resident_bytes", 0)),
        "expert_cpu_seconds": float(metrics.get("cpu", {}).get("expert_compute_time_microseconds") or 0) / 1_000_000.0,
        "gpu_wait_seconds": float(gpu.get("wait_time_microseconds") or 0) / 1_000_000.0,
        "gpu_available": bool(gpu.get("available", False)),
        "gpu_linear_dispatches": int(gpu.get("linear_dispatches", 0)),
        "gpu_attention_blocks": int(gpu.get("attention_blocks", 0)),
        "gpu_gated_delta_fusions": int(gpu.get("gated_delta_fusions", 0)),
        "gpu_gated_delta_submissions": int(gpu.get("gated_delta_submissions", 0)),
        "gpu_batch_uploads": int(gpu.get("batch_uploads", 0)),
        "gpu_batch_downloads": int(gpu.get("batch_downloads", 0)),
        "gpu_execution_evidence": "expert_execution_counter"
        if int(expert.get("gpu_executions", 0)) > 0
        else "dense_attention_only_or_cpu_fallback",
    }


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def main() -> int:
    arguments = _parse_args()
    if arguments.max_new_tokens < 0 or arguments.warmup < 0 or arguments.runs <= 0:
        raise SystemExit("--max-new-tokens must be non-negative; --runs must be positive; --warmup cannot be negative")
    if (
        arguments.host_memory_mb < 0
        or arguments.expert_cache_mb < 0
        or arguments.expert_io_workers < 0
        or arguments.expert_gpu_cache_mb < 0
        or arguments.expert_gpu_victim_cache_mb < 0
    ):
        raise SystemExit("Expert worker and cache sizes cannot be negative")
    if arguments.backend != "hybrid" and (arguments.expert_gpu_cache_mb or arguments.expert_gpu_victim_cache_mb):
        raise SystemExit("GPU Expert cache options require --backend hybrid")
    model = arguments.model.resolve()
    worker = arguments.worker.resolve()
    adapter = None
    if arguments.prompt_token_ids is not None:
        prompt_tokens = list(arguments.prompt_token_ids)
        stop_tokens: list[int] = []
    else:
        from ncnn_moe_adapters import create_adapter

        adapter = create_adapter(model)
        prompt_tokens = adapter.encode_messages([{"role": "user", "content": arguments.prompt}])
        stop_tokens = adapter.stop_tokens
    done_events: list[dict[str, Any]] = []
    generated: list[list[int]] = []

    with WorkerClient(worker, model, _runtime_args(arguments)) as client:
        client.create_session(
            "prompt-benchmark",
            prefill_chunk_size=arguments.prefill_chunk_size,
            enable_speculative_context=arguments.enable_speculative,
        )
        for run_index in range(arguments.warmup + arguments.runs):
            if run_index:
                client.reset("prompt-benchmark")
            done, token_ids = client.generate(
                "prompt-benchmark",
                prompt_tokens,
                request_id=f"prompt-benchmark-{run_index}",
                max_new_tokens=arguments.max_new_tokens,
                temperature=0.0,
                stop_tokens=stop_tokens,
                enable_speculative=arguments.enable_speculative,
                metrics_enabled=False,
                metrics_interval_ms=0,
            )
            if run_index >= arguments.warmup:
                done_events.append(done)
                generated.append(token_ids)

    if not generated:
        raise SystemExit("no measured generation completed")
    result = {
        "prompt": arguments.prompt if arguments.prompt_token_ids is None else None,
        "adapter": getattr(adapter, "name", None),
        "prompt_tokens": prompt_tokens,
        "model": str(model),
        "worker": str(worker),
        "backend": arguments.backend,
        "vulkan_device": arguments.vulkan_device if arguments.backend != "cpu" else None,
        **_summary(done_events, generated),
    }
    if arguments.json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        if result["prompt"] is None:
            print("prompt token ids:", *result["prompt_tokens"])
        else:
            print(f"prompt: {result['prompt']}")
        print(f"backend: {result['backend']}, prompt tokens: {len(prompt_tokens)}, generated: {result['generated_tokens']}")
        print(f"prompt Token/s: {result['prompt_tokens_per_second']}")
        print(f"median Prompt Token/s: {_format_metric(result['median_prompt_tokens_per_second'])}")
        print(f"generation Token/s: {result['generation_tokens_per_second']}")
        print(f"median Generation Token/s: {_format_metric(result['median_generation_tokens_per_second'])}")
        print(
            "median TTFT: "
            f"{_format_metric(result['median_ttft_seconds'])} s, median TPOT: "
            f"{_format_metric(result['median_tpot_seconds'])} s"
        )
        print(
            "GPU evidence: executions={expert_gpu_executions}, failures={expert_gpu_execution_failures}, "
            "dense dispatches={gpu_linear_dispatches}, "
            "attention blocks={gpu_attention_blocks}, GDN fusions/submissions={gpu_gated_delta_fusions}/{gpu_gated_delta_submissions}, "
            "uploads/downloads={gpu_batch_uploads}/{gpu_batch_downloads}; {gpu_execution_evidence}".format(**result)
        )
        print(
            "GPU Expert route aggregation: batches={expert_gpu_route_aggregation_batches}, "
            "routes={expert_gpu_route_aggregation_routes}, bytes-saved={expert_gpu_route_aggregation_bytes_saved}".format(**result)
        )
        print(
            "Expert: CPU={expert_cpu_seconds:.3f} s, GPU cache hit/miss={expert_gpu_cache_hits}/{expert_gpu_cache_misses}, "
            "pending={expert_gpu_cache_pending_bytes} bytes, GPU wait={gpu_wait_seconds:.3f} s".format(**result)
        )
        print("generated token ids:", *result["generated_token_ids"])
        if not result["generated_sequences_match"]:
            print("warning: measured runs produced different token sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
