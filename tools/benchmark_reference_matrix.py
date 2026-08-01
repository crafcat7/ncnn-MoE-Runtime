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
DEEPSEEK_FAMILIES = (
    "deepseek-v4-flash",
    "deepseek-v4-flash-dspark",
)
QWEN_FAMILY = "qwen3.6-35b-a3b"
QWEN_PROMPT_TEXT = (
    "Explain the tradeoffs of mixture-of-experts inference in at least "
    "1000 words."
)
QWEN_PROMPT_TOKEN_IDS = [
    248045,
    846,
    198,
    814,
    20139,
    279,
    6355,
    31410,
    314,
    20340,
    8404,
    17830,
    15089,
    42903,
    303,
    506,
    3140,
    220,
    16,
    15,
    15,
    15,
    4105,
    13,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    198,
]


def add_common_arguments(parser):
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--short-tokens", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=256)
    parser.add_argument(
        "--matrix-backend",
        choices=("hybrid", "cpu"),
        default="hybrid",
        help=(
            "Backend used by the matrix workloads. CPU mode omits Vulkan "
            "device and GPU-cache options."
        ),
    )
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

    for family in DEEPSEEK_FAMILIES:
        deepseek = families.add_parser(family)
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

    qwen = families.add_parser(QWEN_FAMILY)
    add_common_arguments(qwen)
    qwen.add_argument("--model", required=True)
    qwen.add_argument(
        "--model-revision",
        default="",
        help="Checkpoint revision or commit recorded in each case report.",
    )
    qwen.add_argument("--parallel-sessions", type=int, default=4)
    qwen.add_argument("--skip-cpu", action="store_true")
    qwen.add_argument("--skip-single", action="store_true")
    qwen.add_argument("--skip-service", action="store_true")
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
    elif arguments.family in DEEPSEEK_FAMILIES:
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
    else:
        model_paths = (arguments.model,)
        if not 2 <= arguments.parallel_sessions <= 64:
            raise ValueError("--parallel-sessions must be between 2 and 64")

    for path in model_paths:
        if not Path(path).is_dir():
            raise ValueError(f"directory does not exist: {path}")


def warmup_options(warmup):
    if warmup:
        return ["--warmup", "1", "--cache-warmup-runs", "1"]
    return ["--warmup", "0", "--cache-warmup-runs", "0"]


def backend_options(arguments, backend=None):
    selected_backend = backend or arguments.matrix_backend
    options = ["--backend", selected_backend]
    if selected_backend == "hybrid":
        options.extend(
            ("--vulkan-device-index", str(arguments.vulkan_device_index))
        )
    return options


def prompt_token_ids(family):
    if family == "gpt-oss":
        return GPT_OSS_PROMPT_TOKEN_IDS
    if family == QWEN_FAMILY:
        return QWEN_PROMPT_TOKEN_IDS
    return DEEPSEEK_PROMPT_TOKEN_IDS


def common_arguments(arguments, model, tokens, warmup, extra):
    prompt = prompt_token_ids(arguments.family)
    command = [
        sys.executable,
        str(Path(arguments.benchmark).resolve()),
        str(Path(arguments.runner).resolve()),
        str(Path(model).resolve()),
    ]
    if (
        arguments.family == QWEN_FAMILY
        and arguments.model_revision
    ):
        command.extend(("--model-revision", arguments.model_revision))
    command.extend(
        (
            "--prompt-token-ids",
            *[str(token) for token in prompt],
            "--max-new-tokens",
            str(tokens),
        )
    )
    if arguments.family in ("deepseek-v4-flash", QWEN_FAMILY):
        command.extend(("--temperature", "0", "--no-speculative"))
    elif arguments.family == "deepseek-v4-flash-dspark":
        command.extend(
            (
                "--temperature",
                "0",
                "--speculative-confidence",
                "0.5",
                "--require-speculative",
            )
        )
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
    backend = arguments.matrix_backend
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
                        "backend": backend,
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_20b,
                        "extra": [
                            *backend_options(arguments),
                            "--expert-memory",
                            "eager",
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
                        "backend": backend,
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_120b,
                        "extra": [
                            *backend_options(arguments),
                            "--expert-memory",
                            "on-demand",
                            "--host-memory-mb",
                            "28672",
                            "--expert-cache-mb",
                            "20480",
                            "--expert-io-workers",
                            "4",
                            "--direct-expert-io",
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
                        "backend": backend,
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_120b,
                        "extra": [
                            *backend_options(arguments),
                            "--expert-memory",
                            "on-demand",
                            "--host-memory-mb",
                            "28672",
                            "--expert-cache-mb",
                            "22528",
                            "--expert-io-workers",
                            "4",
                            "--parallel-sessions",
                            "2",
                            "--scheduler-expert-threads",
                            "8",
                            "--scheduler-cross-call",
                            "--direct-expert-io",
                            *(
                                [
                                    "--expert-gpu-cache-mb",
                                    "512",
                                    "--expert-gpu-victim-cache-mb",
                                    "2560",
                                    "--expert-gpu-victim-reuse-probe",
                                    "1",
                                ]
                                if backend == "hybrid"
                                else []
                            ),
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
                        "backend": "cpu",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model_120b,
                        "extra": [
                            *backend_options(arguments, "cpu"),
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
    backend = arguments.matrix_backend
    prefix = arguments.family
    speculative = arguments.family == "deepseek-v4-flash-dspark"
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
                        "name": f"{prefix}-single-{window_name}-{warmup_name}",
                        "family": prefix,
                        "workload": "single-session",
                        "backend": backend,
                        "scheduler_staging": "auto",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            *backend_options(arguments),
                            *storage,
                            "--expert-cache-mb",
                            str(arguments.expert_cache_mb),
                        ],
                    }
                )

    if not arguments.skip_service:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": (
                            f"{prefix}-service-{arguments.parallel_sessions}x"
                            f"{window_name}-{warmup_name}"
                        ),
                        "family": prefix,
                        "workload": (
                            (
                                f"{arguments.parallel_sessions}-session-"
                                "independent-speculative"
                            )
                            if speculative
                            else f"{arguments.parallel_sessions}-session-service"
                        ),
                        "backend": backend,
                        "scheduler_staging": (
                            "independent-speculative"
                            if speculative
                            else "force"
                        ),
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            *backend_options(arguments),
                            *storage,
                            "--expert-cache-mb",
                            str(arguments.expert_cache_mb),
                            "--parallel-sessions",
                            str(arguments.parallel_sessions),
                            *(
                                ["--parallel-speculative"]
                                if speculative
                                else ["--scheduler-staging", "force"]
                            ),
                        ],
                    }
                )

    if not arguments.skip_storage:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": f"{prefix}-storage-{window_name}-{warmup_name}",
                        "family": prefix,
                        "workload": "cpu-storage-control",
                        "backend": "cpu",
                        "scheduler_staging": "auto",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            *backend_options(arguments, "cpu"),
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


def qwen_cases(arguments):
    cases = []
    backend = arguments.matrix_backend
    prefix = arguments.family
    windows = (
        ("short", arguments.short_tokens),
        ("long", arguments.long_tokens),
    )
    warmups = (("cold", False), ("warm", True))
    eager_experts = ["--expert-memory", "eager"]

    if backend == "hybrid" and not arguments.skip_cpu:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": (
                            f"{prefix}-single-cpu-{window_name}-"
                            f"{warmup_name}"
                        ),
                        "family": prefix,
                        "workload": "single-session-cpu-control",
                        "backend": "cpu",
                        "scheduler_staging": "auto",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            *backend_options(arguments, "cpu"),
                            *eager_experts,
                        ],
                    }
                )

    if not arguments.skip_single and not (backend == "cpu" and arguments.skip_cpu):
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": (
                            f"{prefix}-single-{backend}-{window_name}-"
                            f"{warmup_name}"
                        ),
                        "family": prefix,
                        "workload": (
                            "single-session"
                            if backend == "hybrid"
                            else "single-session-cpu-control"
                        ),
                        "backend": backend,
                        "scheduler_staging": "auto",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            *backend_options(arguments),
                            *eager_experts,
                        ],
                    }
                )

    if not arguments.skip_service:
        for window_name, tokens in windows:
            for warmup_name, warmup in warmups:
                cases.append(
                    {
                        "name": (
                            f"{prefix}-service-"
                            f"{arguments.parallel_sessions}x"
                            f"{window_name}-{warmup_name}"
                        ),
                        "family": prefix,
                        "workload": (
                            f"{arguments.parallel_sessions}-session-service"
                        ),
                        "backend": backend,
                        "scheduler_staging": "force",
                        "token_window": window_name,
                        "generated_tokens": tokens,
                        "warmup": warmup_name,
                        "model": arguments.model,
                        "extra": [
                            *backend_options(arguments),
                            *eager_experts,
                            "--parallel-sessions",
                            str(arguments.parallel_sessions),
                            "--scheduler-staging",
                            "force",
                        ],
                    }
                )
    return cases


def load_report(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_qwen_results(results):
    references = {}
    for result in results:
        case = result["case"]
        report = result["report"]
        sessions = report.get("parallel_sessions", 1)
        requested_tokens = case["generated_tokens"]
        tokens = report.get("generated_token_ids", [])
        expected_tokens = requested_tokens * sessions
        if len(tokens) != expected_tokens:
            raise RuntimeError(
                f"{case['name']} generated {len(tokens)} token(s); "
                f"expected {expected_tokens}"
            )

        for session in range(sessions):
            begin = session * requested_tokens
            sequence = tokens[begin : begin + requested_tokens]
            reference = references.setdefault(requested_tokens, sequence)
            if sequence != reference:
                first_difference = next(
                    index
                    for index, (expected, actual) in enumerate(
                        zip(reference, sequence)
                    )
                    if expected != actual
                )
                raise RuntimeError(
                    f"{case['name']} differs from the Qwen "
                    f"{requested_tokens}-token reference at token "
                    f"{first_difference}"
                )


def validate_backend_result(case, report):
    expected_backend = case.get("backend")
    if expected_backend is None:
        return
    if report.get("backend") != expected_backend:
        raise RuntimeError(
            f"{case['name']} reported backend {report.get('backend')!r}; "
            f"expected {expected_backend!r}"
        )
    if expected_backend == "cpu":
        evidence = report.get("execution_evidence")
        if evidence is None:
            raise RuntimeError(
                f"{case['name']} has no CPU-only execution evidence; "
                "rebuild the runner with the current diagnostics"
            )
        if not evidence.get("cpu_only_execution_verified", False):
            raise RuntimeError(
                f"{case['name']} was requested as CPU-only but reported "
                f"GPU execution (runtime backend: "
                f"{evidence.get('runtime_backend')!r}, Vulkan context: "
                f"{evidence.get('vulkan_context_initialized')!r})"
            )


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
            validate_backend_result(case, report)
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
    validate_backend_result(case, report)
    return {
        "case": case,
        "command": command,
        "report_path": str(output_path.resolve()),
        "report": report,
    }


def write_aggregate(arguments, results, started, output_dir):
    prompt = prompt_token_ids(arguments.family)
    benchmark = (
        "gpt_oss_performance_matrix"
        if arguments.family == "gpt-oss"
        else f"{arguments.family.replace('-', '_')}_performance_matrix"
    )
    aggregate = {
        "schema_version": 3,
        "benchmark": benchmark,
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "prompt_token_ids": prompt,
        "prompt_text": (
            QWEN_PROMPT_TEXT if arguments.family == QWEN_FAMILY else None
        ),
        "model_revision": (
            arguments.model_revision
            if arguments.family == QWEN_FAMILY
            and arguments.model_revision
            else None
        ),
        "short_tokens": arguments.short_tokens,
        "long_tokens": arguments.long_tokens,
        "repeats": arguments.repeats,
        "matrix_backend": arguments.matrix_backend,
        "case_backends": sorted(
            {result["case"].get("backend") for result in results} - {None}
        ),
        "speculative_enabled": (
            arguments.family == "deepseek-v4-flash-dspark"
        ),
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
    if arguments.family == "gpt-oss":
        cases = gpt_oss_cases(arguments)
    elif arguments.family in DEEPSEEK_FAMILIES:
        cases = deepseek_cases(arguments)
    else:
        cases = qwen_cases(arguments)
    if not cases:
        print("all matrix workloads were skipped", file=sys.stderr)
        return 2

    results = []
    started = time.time()
    for case in cases:
        results.append(run_case(arguments, case, output_dir))
        if arguments.family == QWEN_FAMILY:
            validate_qwen_results(results)
        write_aggregate(arguments, results, started, output_dir)

    aggregate_path = write_aggregate(arguments, results, started, output_dir)
    print(f"\nMatrix report: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
