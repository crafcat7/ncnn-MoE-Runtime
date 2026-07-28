#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


GPT_OSS_PROMPT_TOKEN_IDS = [
    50117,
    261,
    82463,
    30547,
    328,
    70531,
    13236,
    10978,
    13108,
    12,
    149295,
    91643,
    395,
    2698,
    10220,
    13,
]
DEEPSEEK_PROMPT_TOKEN_IDS = [0] * 16


def add_common_arguments(parser):
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--short-tokens", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=256)
    parser.add_argument("--vulkan-device-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a unified ncnn_moe reference performance matrix."
    )
    parser.add_argument(
        "--benchmark",
        default="tools/benchmark_runtime.py",
        help="Path to the single-workload benchmark script.",
    )
    families = parser.add_subparsers(dest="family", required=True)

    gpt_oss = families.add_parser("gpt-oss")
    add_common_arguments(gpt_oss)
    gpt_oss.add_argument("--model-20b", required=True)
    gpt_oss.add_argument("--model-120b", required=True)
    gpt_oss.add_argument("--skip-offload", action="store_true")
    gpt_oss.add_argument("--skip-20b", action="store_true")
    gpt_oss.add_argument("--skip-service", action="store_true")
    gpt_oss.add_argument("--skip-single-120b", action="store_true")

    deepseek = families.add_parser("deepseek-v4")
    add_common_arguments(deepseek)
    deepseek.add_argument("--model", required=True)
    deepseek.add_argument("--host-memory-mb", type=int, default=28672)
    deepseek.add_argument("--expert-cache-mb", type=int, default=16384)
    deepseek.add_argument(
        "--storage-cache-mb",
        type=int,
        nargs="+",
        default=(1024, 10240, 16384),
    )
    deepseek.add_argument("--expert-io-workers", type=int, default=4)
    deepseek.add_argument("--parallel-sessions", type=int, default=4)
    deepseek.add_argument("--skip-storage", action="store_true")
    deepseek.add_argument("--skip-single", action="store_true")
    deepseek.add_argument("--skip-service", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if arguments.short_tokens <= 0 or arguments.long_tokens <= 0:
        raise ValueError("token windows must be positive")
    if arguments.short_tokens >= arguments.long_tokens:
        raise ValueError("--short-tokens must be smaller than --long-tokens")
    if arguments.vulkan_device_index < 0:
        raise ValueError("--vulkan-device-index must be non-negative")
    for path in (arguments.benchmark, arguments.runner):
        if not Path(path).is_file():
            raise ValueError(f"file does not exist: {path}")

    if arguments.family == "gpt-oss":
        model_paths = (arguments.model_20b, arguments.model_120b)
    else:
        model_paths = (arguments.model,)
        if arguments.host_memory_mb <= 0 or arguments.expert_cache_mb <= 0:
            raise ValueError("memory limits must be positive")
        if arguments.expert_cache_mb > arguments.host_memory_mb:
            raise ValueError("--expert-cache-mb cannot exceed host memory")
        if not arguments.storage_cache_mb:
            raise ValueError("--storage-cache-mb must not be empty")
        if any(value <= 0 for value in arguments.storage_cache_mb):
            raise ValueError("--storage-cache-mb values must be positive")
        if len(set(arguments.storage_cache_mb)) != len(
            arguments.storage_cache_mb
        ):
            raise ValueError("--storage-cache-mb values must be unique")
        if max(arguments.storage_cache_mb) > arguments.host_memory_mb:
            raise ValueError("--storage-cache-mb cannot exceed host memory")
        if not 0 <= arguments.expert_io_workers <= 64:
            raise ValueError("--expert-io-workers must be between 0 and 64")
        if not 2 <= arguments.parallel_sessions <= 64:
            raise ValueError("--parallel-sessions must be between 2 and 64")

    for path in model_paths:
        if not Path(path).is_dir():
            raise ValueError(f"directory does not exist: {path}")


def warmup_options(warmup):
    if warmup:
        return ["--warmup", "1", "--cache-warmup-runs", "1"]
    return ["--warmup", "0", "--cache-warmup-runs", "0"]


def common_arguments(arguments, model, tokens, warmup, extra):
    prompt = (
        GPT_OSS_PROMPT_TOKEN_IDS
        if arguments.family == "gpt-oss"
        else DEEPSEEK_PROMPT_TOKEN_IDS
    )
    command = [
        sys.executable,
        str(Path(arguments.benchmark).resolve()),
        str(Path(arguments.runner).resolve()),
        str(Path(model).resolve()),
        "--prompt-token-ids",
        *[str(token) for token in prompt],
        "--max-new-tokens",
        str(tokens),
    ]
    if arguments.family == "deepseek-v4":
        command.extend(("--temperature", "0", "--no-speculative"))
    command.extend(
        (
            *warmup_options(warmup),
            "--repeats",
            str(arguments.repeats),
            *extra,
        )
    )
    return command


def gpt_oss_cases(arguments):
    cases = []
    windows = (("short", arguments.short_tokens), ("long", arguments.long_tokens))
    warmups = (("cold", False), ("warm", True))

    if not arguments.skip_20b:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": f"gpt-oss-20b-single-{window_name}-{warmup_name}",
                        "family": "gpt-oss-20b",
                        "workload": "single-session",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_20b,
                        "extra": [
                            "--backend",
                            "hybrid",
                            "--expert-memory",
                            "eager",
                            "--vulkan-device-index",
                            str(arguments.vulkan_device_index),
                        ],
                    }
                )

    if not arguments.skip_single_120b:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": f"gpt-oss-120b-single-{window_name}-{warmup_name}",
                        "family": "gpt-oss-120b",
                        "workload": "single-session",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_120b,
                        "extra": [
                            "--backend",
                            "hybrid",
                            "--expert-memory",
                            "on-demand",
                            "--host-memory-mb",
                            "28672",
                            "--expert-cache-mb",
                            "20480",
                            "--expert-io-workers",
                            "4",
                            "--direct-expert-io",
                            "--vulkan-device-index",
                            str(arguments.vulkan_device_index),
                        ],
                    }
                )

    if not arguments.skip_service:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": (
                            f"gpt-oss-120b-service-2x{window_name}-"
                            f"{warmup_name}"
                        ),
                        "family": "gpt-oss-120b",
                        "workload": "two-session-service",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_120b,
                        "extra": [
                            "--backend",
                            "hybrid",
                            "--expert-memory",
                            "on-demand",
                            "--host-memory-mb",
                            "28672",
                            "--expert-cache-mb",
                            "22528",
                            "--expert-gpu-cache-mb",
                            "512",
                            "--expert-gpu-victim-cache-mb",
                            "2560",
                            "--expert-gpu-victim-reuse-probe",
                            "1",
                            "--expert-io-workers",
                            "4",
                            "--parallel-sessions",
                            "2",
                            "--scheduler-expert-threads",
                            "8",
                            "--scheduler-cross-call",
                            "--direct-expert-io",
                            "--vulkan-device-index",
                            str(arguments.vulkan_device_index),
                        ],
                    }
                )

    if not arguments.skip_offload:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": f"gpt-oss-120b-offload-{window_name}-{warmup_name}",
                        "family": "gpt-oss-120b",
                        "workload": "cpu-expert-cache-sweep",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_120b,
                        "extra": [
                            "--backend",
                            "cpu",
                            "--expert-memory",
                            "on-demand",
                            "--host-memory-mb",
                            "28672",
                            "--expert-io-workers",
                            "4",
                            "--direct-expert-io",
                            "--expert-cache-sweep-mb",
                            "1024",
                            "10240",
                            "16384",
                        ],
                    }
                )
    return cases


def deepseek_cases(arguments):
    cases = []
    windows = (("short", arguments.short_tokens), ("long", arguments.long_tokens))
    warmups = (("cold", False), ("warm", True))
    storage = [
        "--expert-memory",
        "on-demand",
        "--host-memory-mb",
        str(arguments.host_memory_mb),
        "--expert-io-workers",
        str(arguments.expert_io_workers),
        "--buffered-expert-io",
    ]

    if not arguments.skip_single:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": f"deepseek-v4-single-{window_name}-{warmup_name}",
                        "family": "deepseek-v4-flash-dspark",
                        "workload": "single-session",
                        "backend": "hybrid",
                        "scheduler_staging": "auto",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            "--backend",
                            "hybrid",
                            *storage,
                            "--expert-cache-mb",
                            str(arguments.expert_cache_mb),
                            "--vulkan-device-index",
                            str(arguments.vulkan_device_index),
                        ],
                    }
                )

    if not arguments.skip_service:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": (
                            f"deepseek-v4-service-{arguments.parallel_sessions}x"
                            f"{window_name}-{warmup_name}"
                        ),
                        "family": "deepseek-v4-flash-dspark",
                        "workload": (
                            f"{arguments.parallel_sessions}-session-service"
                        ),
                        "backend": "hybrid",
                        "scheduler_staging": "force",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            "--backend",
                            "hybrid",
                            *storage,
                            "--expert-cache-mb",
                            str(arguments.expert_cache_mb),
                            "--parallel-sessions",
                            str(arguments.parallel_sessions),
                            "--scheduler-staging",
                            "force",
                            "--vulkan-device-index",
                            str(arguments.vulkan_device_index),
                        ],
                    }
                )

    if not arguments.skip_storage:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": f"deepseek-v4-storage-{window_name}-{warmup_name}",
                        "family": "deepseek-v4-flash-dspark",
                        "workload": "cpu-storage-control",
                        "backend": "cpu",
                        "scheduler_staging": "auto",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            "--backend",
                            "cpu",
                            *storage,
                            "--expert-cache-sweep-mb",
                            *[
                                str(cache_mb)
                                for cache_mb in arguments.storage_cache_mb
                            ],
                        ],
                    }
                )
    return cases


def load_report(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_case(arguments, case, output_dir):
    output_path = output_dir / f"{case['name']}.json"
    command = common_arguments(
        arguments,
        case["model"],
        case["generated_tokens"],
        case["warmup"] == "warm",
        case["extra"] + ["--json-output", str(output_path.resolve())],
    )
    if arguments.resume and output_path.is_file():
        report = load_report(output_path)
        if report is not None:
            print(f"\n=== {case['name']} (reused) ===", flush=True)
            return {
                "case": case,
                "command": command,
                "report_path": str(output_path.resolve()),
                "report": report,
            }

    print(f"\n=== {case['name']} ===", flush=True)
    print("command:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed for {case['name']} with exit code "
            f"{completed.returncode}"
        )
    report = load_report(output_path)
    if report is None:
        raise RuntimeError(f"benchmark did not write valid JSON: {output_path}")
    return {
        "case": case,
        "command": command,
        "report_path": str(output_path.resolve()),
        "report": report,
    }


def write_aggregate(arguments, results, started, output_dir):
    prompt = (
        GPT_OSS_PROMPT_TOKEN_IDS
        if arguments.family == "gpt-oss"
        else DEEPSEEK_PROMPT_TOKEN_IDS
    )
    benchmark = (
        "gpt_oss_performance_matrix"
        if arguments.family == "gpt-oss"
        else "deepseek_v4_performance_matrix"
    )
    aggregate = {
        "schema_version": 1,
        "benchmark": benchmark,
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "prompt_token_ids": prompt,
        "short_tokens": arguments.short_tokens,
        "long_tokens": arguments.long_tokens,
        "repeats": arguments.repeats,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    aggregate_path = output_dir / "report.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate_path


def main():
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    output_dir = Path(arguments.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        gpt_oss_cases(arguments)
        if arguments.family == "gpt-oss"
        else deepseek_cases(arguments)
    )
    if not cases:
        print("all matrix workloads were skipped", file=sys.stderr)
        return 2

    results = []
    started = time.time()
    for case in cases:
        results.append(run_case(arguments, case, output_dir))
        write_aggregate(arguments, results, started, output_dir)

    aggregate_path = write_aggregate(arguments, results, started, output_dir)
    print(f"\nMatrix report: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
