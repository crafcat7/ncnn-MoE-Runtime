#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROMPT_TOKEN_IDS = [
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the unified GPT-OSS performance matrix."
    )
    parser.add_argument(
        "--benchmark",
        default="tools/benchmark_gpt_oss.py",
        help="Path to the single-workload benchmark script.",
    )
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model-20b", required=True)
    parser.add_argument("--model-120b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--short-tokens", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=256)
    parser.add_argument("--vulkan-device-index", type=int, default=0)
    parser.add_argument("--skip-offload", action="store_true")
    parser.add_argument("--skip-20b", action="store_true")
    parser.add_argument("--skip-service", action="store_true")
    parser.add_argument("--skip-single-120b", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if arguments.short_tokens <= 0 or arguments.long_tokens <= 0:
        raise ValueError("token windows must be positive")
    if arguments.vulkan_device_index < 0:
        raise ValueError("--vulkan-device-index must be non-negative")
    for path in (arguments.benchmark, arguments.runner):
        if not Path(path).is_file():
            raise ValueError(f"file does not exist: {path}")
    for path in (arguments.model_20b, arguments.model_120b):
        if not Path(path).is_dir():
            raise ValueError(f"directory does not exist: {path}")


def warmup_options(warmup):
    if warmup:
        return ["--warmup", "1", "--cache-warmup-runs", "1"]
    return ["--warmup", "0", "--cache-warmup-runs", "0"]


def common_arguments(arguments, model, tokens, warmup, extra):
    command = [
        sys.executable,
        str(Path(arguments.benchmark).resolve()),
        str(Path(arguments.runner).resolve()),
        str(Path(model).resolve()),
        "--prompt-token-ids",
        *[str(token) for token in PROMPT_TOKEN_IDS],
        "--max-new-tokens",
        str(tokens),
        *warmup_options(warmup),
        "--repeats",
        str(arguments.repeats),
        *extra,
    ]
    return command


def build_cases(arguments):
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
                        "name": f"gpt-oss-120b-service-2x{window_name}-{warmup_name}",
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


def run_case(arguments, case, output_dir):
    output_path = output_dir / f"{case['name']}.json"
    command = common_arguments(
        arguments,
        case["model"],
        case["generated_tokens"],
        case["warmup"] == "warm",
        case["extra"] + ["--json-output", str(output_path.resolve())],
    )
    print(f"\n=== {case['name']} ===", flush=True)
    print("command:", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed for {case['name']} with exit code "
            f"{completed.returncode}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        "case": case,
        "command": command,
        "report_path": str(output_path.resolve()),
        "report": report,
    }


def main():
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    output_dir = Path(arguments.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(arguments)
    results = []
    started = time.time()
    for case in cases:
        results.append(run_case(arguments, case, output_dir))

    aggregate = {
        "schema_version": 1,
        "benchmark": "gpt_oss_performance_matrix",
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "short_tokens": arguments.short_tokens,
        "long_tokens": arguments.long_tokens,
        "repeats": arguments.repeats,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    aggregate_path = output_dir / "performance-matrix.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nMatrix report: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
