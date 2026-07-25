#!/usr/bin/env python3

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


MIB = 1024 * 1024


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ncnn_moe_gpt_oss with repeatable runtime, memory, "
            "and output-token reporting."
        )
    )
    parser.add_argument("runner", help="Path to ncnn_moe_gpt_oss")
    parser.add_argument("model", help="Path to an official GPT-OSS model directory")
    parser.add_argument(
        "--model-revision",
        default="",
        help="Checkpoint revision or commit recorded in the report.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=int,
        nargs="+",
        default=[0],
        help="Input token IDs passed to the runner (default: 0).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "hybrid", "hybrid-prefetch"),
        default="auto",
    )
    parser.add_argument(
        "--expert-memory",
        choices=("auto", "eager", "on-demand"),
        default="auto",
    )
    parser.add_argument("--host-memory-mb", type=int, default=0)
    parser.add_argument("--expert-cache-mb", type=int, default=0)
    parser.add_argument("--expert-gpu-cache-mb", type=int, default=0)
    parser.add_argument("--expert-io-workers", type=int, default=0)
    parser.add_argument(
        "--mmap-experts",
        action="store_true",
        help="Use memory-mapped on-demand MXFP4 ranges.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=int,
        default=100,
        help="Host/GPU memory sampling interval (default: 100).",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="NVIDIA GPU index monitored with nvidia-smi (default: 0).",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path for the complete machine-readable report.",
    )
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if arguments.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if (
        arguments.host_memory_mb < 0
        or arguments.expert_cache_mb < 0
        or arguments.expert_gpu_cache_mb < 0
    ):
        raise ValueError("memory limits must be non-negative")
    if arguments.expert_io_workers < 0 or arguments.expert_io_workers > 64:
        raise ValueError("--expert-io-workers must be between 0 and 64")
    if arguments.sample_interval_ms < 20:
        raise ValueError("--sample-interval-ms must be at least 20")
    if arguments.gpu_index < 0:
        raise ValueError("--gpu-index must be non-negative")


def runner_command(arguments):
    command = [
        str(Path(arguments.runner)),
        str(Path(arguments.model)),
        *[str(token) for token in arguments.prompt_token_ids],
        "--max-new-tokens",
        str(arguments.max_new_tokens),
    ]
    if arguments.backend != "auto":
        command.append(f"--{arguments.backend}")
    if arguments.expert_memory != "auto":
        command.extend(["--expert-memory", arguments.expert_memory])
    if arguments.host_memory_mb:
        command.extend(["--host-memory-mb", str(arguments.host_memory_mb)])
    if arguments.expert_cache_mb:
        command.extend(["--expert-cache-mb", str(arguments.expert_cache_mb)])
    if arguments.expert_gpu_cache_mb:
        command.extend(
            ["--expert-gpu-cache-mb", str(arguments.expert_gpu_cache_mb)]
        )
    if arguments.expert_io_workers:
        command.extend(["--expert-io-workers", str(arguments.expert_io_workers)])
    if arguments.mmap_experts:
        command.append("--mmap-experts")
    return command


class WindowsProcessMemory:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
    ]

    def __init__(self, process_id):
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel32.OpenProcess.argtypes = [
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self.kernel32.OpenProcess.restype = ctypes.c_void_p
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_int
        self.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self.Counters),
            ctypes.c_ulong,
        ]
        self.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        access = 0x1000 | 0x0010
        self.handle = self.kernel32.OpenProcess(access, False, process_id)

    def sample(self):
        if not self.handle:
            return None
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)
        succeeded = self.psapi.GetProcessMemoryInfo(
            self.handle,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if succeeded else None

    def close(self):
        if self.handle:
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


class PortableProcessMemory:
    def __init__(self, process_id):
        self.process_id = process_id
        self.peak_bytes = 0
        self.windows = (
            WindowsProcessMemory(process_id)
            if os.name == "nt"
            else None
        )

    def sample(self):
        if self.windows:
            value = self.windows.sample()
        elif sys.platform.startswith("linux"):
            value = self._linux_rss()
        else:
            value = self._ps_rss()
        if value is not None:
            self.peak_bytes = max(self.peak_bytes, value)

    def close(self):
        self.sample()
        if self.windows:
            self.windows.close()

    def _linux_rss(self):
        try:
            status = Path(f"/proc/{self.process_id}/status").read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeError):
            return None
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
        return int(match.group(1)) * 1024 if match else None

    def _ps_rss(self):
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.process_id)],
            text=True,
            capture_output=True,
        )
        try:
            return int(completed.stdout.strip()) * 1024
        except ValueError:
            return None


class NvidiaMemoryMonitor:
    def __init__(self, gpu_index, interval_ms):
        self.executable = shutil.which("nvidia-smi")
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.device = self._query_device()
        self.baseline_mib = self._query_once()
        self.process = None

    def _base_command(self):
        return [
            self.executable,
            f"--id={self.gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]

    def _query_once(self):
        if not self.executable:
            return None
        completed = subprocess.run(
            self._base_command(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            return None
        try:
            return int(completed.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            return None

    def _query_device(self):
        if not self.executable:
            return None
        completed = subprocess.run(
            [
                self.executable,
                f"--id={self.gpu_index}",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            return None
        lines = completed.stdout.splitlines()
        if not lines:
            return None
        fields = [field.strip() for field in lines[0].split(",")]
        if len(fields) != 3:
            return None
        try:
            memory_mib = int(fields[2])
        except ValueError:
            memory_mib = None
        return {
            "name": fields[0],
            "driver_version": fields[1],
            "memory_total_mib": memory_mib,
        }

    def start(self):
        if not self.executable:
            return
        self.process = subprocess.Popen(
            [*self._base_command(), f"--loop-ms={self.interval_ms}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        if not self.process:
            return None, None
        self.process.terminate()
        try:
            output, _ = self.process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            output, _ = self.process.communicate()
        samples = []
        for line in output.splitlines():
            try:
                samples.append(int(line.strip()))
            except ValueError:
                continue
        peak_mib = max(samples) if samples else None
        delta_mib = (
            max(0, peak_mib - self.baseline_mib)
            if peak_mib is not None and self.baseline_mib is not None
            else None
        )
        return peak_mib, delta_mib


def extract_number(output, pattern, cast=float):
    match = re.search(pattern, output, re.MULTILINE)
    return cast(match.group(1)) if match else None


def parse_runner_output(output):
    token_match = re.search(
        r"^generated token ids:(.*)$",
        output,
        re.MULTILINE,
    )
    tokens = (
        [int(token) for token in token_match.group(1).split()]
        if token_match
        else []
    )
    result = {
        "load_seconds": extract_number(
            output, r"^loaded gpt_oss in ([0-9.]+) s,"
        ),
        "generated_tokens": extract_number(
            output, r"^generated (\d+) token\(s\) in", int
        ),
        "generation_seconds": extract_number(
            output, r"^generated \d+ token\(s\) in ([0-9.]+) s"
        ),
        "attention_ms": extract_number(
            output, r"^Attention time: ([0-9.]+) ms"
        ),
        "router_ms": extract_number(output, r"^Router time: ([0-9.]+) ms"),
        "expert_ms": extract_number(output, r"^Expert time: ([0-9.]+) ms"),
        "expert_cache_hits": extract_number(
            output, r"^Expert cache: (\d+) hit", int
        ),
        "expert_cache_misses": extract_number(
            output, r"^Expert cache: \d+ hit\(s\), (\d+) miss", int
        ),
        "expert_cache_evictions": extract_number(
            output,
            r"^Expert cache: \d+ hit\(s\), \d+ miss\(es\), (\d+) eviction",
            int,
        ),
        "expert_cache_bytes_read": extract_number(
            output, r"^Expert cache:.*?(\d+) bytes read", int
        ),
        "expert_cache_resident_bytes": extract_number(
            output, r"^Expert cache:.*?(\d+) bytes resident", int
        ),
        "expert_cache_mapped_ranges": extract_number(
            output, r"^Expert mmap: (\d+) range", int
        ),
        "expert_cache_mapped_bytes": extract_number(
            output, r"^Expert mmap:.*?(\d+) bytes", int
        ),
        "expert_gpu_cache_hits": extract_number(
            output, r"^Expert GPU cache: (\d+) hit", int
        ),
        "expert_gpu_cache_misses": extract_number(
            output, r"^Expert GPU cache: \d+ hit\(s\), (\d+) miss", int
        ),
        "expert_gpu_cache_admissions": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) admission", int
        ),
        "expert_gpu_cache_stores": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) store", int
        ),
        "expert_gpu_cache_evictions": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) eviction", int
        ),
        "expert_gpu_cache_dropped_admissions": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) dropped admission", int
        ),
        "expert_gpu_cache_restore_failures": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) restore failure", int
        ),
        "expert_gpu_cache_bytes_uploaded": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) bytes uploaded", int
        ),
        "expert_gpu_cache_bytes_downloaded": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) bytes downloaded", int
        ),
        "expert_gpu_cache_restore_ms": extract_number(
            output, r"^Expert GPU cache:.*?([0-9.]+) ms restoring"
        ),
        "expert_gpu_cache_mapped_stores": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) mapped store", int
        ),
        "expert_gpu_cache_mapped_restores": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) mapped restore", int
        ),
        "expert_gpu_cache_resident_bytes": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) bytes resident", int
        ),
        "expert_gpu_cache_pending_bytes": extract_number(
            output, r"^Expert GPU cache:.*?(\d+) bytes pending", int
        ),
        "generated_token_ids": tokens,
    }
    if result["generation_seconds"] and result["generated_tokens"]:
        result["decode_tokens_per_second"] = (
            result["generated_tokens"] / result["generation_seconds"]
        )
    else:
        result["decode_tokens_per_second"] = None
    cache_lookups = (
        (result["expert_cache_hits"] or 0)
        + (result["expert_cache_misses"] or 0)
    )
    result["expert_cache_hit_rate"] = (
        result["expert_cache_hits"] / cache_lookups
        if cache_lookups
        else None
    )
    return result


def execute_sample(command, gpu_index, sample_interval_ms):
    gpu_monitor = NvidiaMemoryMonitor(gpu_index, sample_interval_ms)
    gpu_monitor.start()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    memory = PortableProcessMemory(process.pid)
    while process.poll() is None:
        memory.sample()
        time.sleep(sample_interval_ms / 1000.0)
    stdout, stderr = process.communicate()
    memory.close()
    peak_gpu_mib, gpu_delta_mib = gpu_monitor.stop()
    if process.returncode != 0:
        raise RuntimeError(
            f"runner exited with {process.returncode}\n{stdout}\n{stderr}"
        )
    parsed = parse_runner_output(stdout)
    if (
        parsed["generation_seconds"] is None
        or len(parsed["generated_token_ids"]) != parsed["generated_tokens"]
    ):
        raise RuntimeError(f"runner output is incomplete\n{stdout}\n{stderr}")
    parsed["peak_rss_mib"] = memory.peak_bytes / MIB
    parsed["gpu_baseline_mib"] = gpu_monitor.baseline_mib
    parsed["peak_gpu_mib"] = peak_gpu_mib
    parsed["peak_gpu_delta_mib"] = gpu_delta_mib
    parsed["gpu_device"] = gpu_monitor.device
    return parsed


def median_field(samples, field):
    values = [sample[field] for sample in samples if sample[field] is not None]
    return statistics.median(values) if values else None


def main():
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    command = runner_command(arguments)
    samples = []
    total_runs = arguments.warmup + arguments.repeats
    for run_index in range(total_runs):
        kind = "warmup" if run_index < arguments.warmup else "measured"
        ordinal = (
            run_index + 1
            if kind == "warmup"
            else run_index - arguments.warmup + 1
        )
        print(f"{kind} run {ordinal}...", file=sys.stderr, flush=True)
        try:
            sample = execute_sample(
                command,
                arguments.gpu_index,
                arguments.sample_interval_ms,
            )
        except (OSError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return 1
        if kind == "measured":
            samples.append(sample)

    reference_tokens = samples[0]["generated_token_ids"]
    if any(sample["generated_token_ids"] != reference_tokens for sample in samples[1:]):
        print("measured runs produced different token sequences", file=sys.stderr)
        return 1

    report = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(Path(arguments.model).resolve()),
        "model_revision": arguments.model_revision or None,
        "runner": str(Path(arguments.runner).resolve()),
        "command": command,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "gpu_index": arguments.gpu_index,
            "gpu": samples[0]["gpu_device"],
        },
        "warmup_runs": arguments.warmup,
        "measured_runs": arguments.repeats,
        "median": {
            "load_seconds": median_field(samples, "load_seconds"),
            "generation_seconds": median_field(samples, "generation_seconds"),
            "decode_tokens_per_second": median_field(
                samples, "decode_tokens_per_second"
            ),
            "attention_ms": median_field(samples, "attention_ms"),
            "router_ms": median_field(samples, "router_ms"),
            "expert_ms": median_field(samples, "expert_ms"),
            "expert_cache_hit_rate": median_field(
                samples, "expert_cache_hit_rate"
            ),
            "expert_cache_bytes_read": median_field(
                samples, "expert_cache_bytes_read"
            ),
            "expert_cache_mapped_ranges": median_field(
                samples, "expert_cache_mapped_ranges"
            ),
            "expert_cache_mapped_bytes": median_field(
                samples, "expert_cache_mapped_bytes"
            ),
            "expert_gpu_cache_hits": median_field(
                samples, "expert_gpu_cache_hits"
            ),
            "expert_gpu_cache_bytes_uploaded": median_field(
                samples, "expert_gpu_cache_bytes_uploaded"
            ),
            "expert_gpu_cache_bytes_downloaded": median_field(
                samples, "expert_gpu_cache_bytes_downloaded"
            ),
            "expert_gpu_cache_restore_ms": median_field(
                samples, "expert_gpu_cache_restore_ms"
            ),
            "expert_gpu_cache_mapped_stores": median_field(
                samples, "expert_gpu_cache_mapped_stores"
            ),
            "expert_gpu_cache_mapped_restores": median_field(
                samples, "expert_gpu_cache_mapped_restores"
            ),
        },
        "maximum": {
            "peak_rss_mib": max(sample["peak_rss_mib"] for sample in samples),
            "peak_gpu_mib": max(
                (
                    sample["peak_gpu_mib"]
                    for sample in samples
                    if sample["peak_gpu_mib"] is not None
                ),
                default=None,
            ),
            "peak_gpu_delta_mib": max(
                (
                    sample["peak_gpu_delta_mib"]
                    for sample in samples
                    if sample["peak_gpu_delta_mib"] is not None
                ),
                default=None,
            ),
        },
        "generated_token_ids": reference_tokens,
        "samples": samples,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.json_output:
        output_path = Path(arguments.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    median = report["median"]
    maximum = report["maximum"]
    print(
        f"median generation: {median['generation_seconds']:.4f} s "
        f"({median['decode_tokens_per_second']:.3f} token/s)"
    )
    print(f"median Expert time: {median['expert_ms']:.3f} ms")
    if median["expert_gpu_cache_hits"] is not None:
        print(
            "median GPU Expert cache: "
            f"{median['expert_gpu_cache_hits']:.0f} hit(s), "
            f"{median['expert_gpu_cache_bytes_downloaded']:.0f} bytes restored, "
            f"{median['expert_gpu_cache_mapped_restores']:.0f} direct-mapped"
        )
    print(f"peak RSS: {maximum['peak_rss_mib']:.1f} MiB")
    if maximum["peak_gpu_mib"] is not None:
        print(
            f"peak NVIDIA memory: {maximum['peak_gpu_mib']} MiB "
            f"(+{maximum['peak_gpu_delta_mib']} MiB)"
        )
    print("generated token ids:", *reference_tokens)
    if arguments.json_output:
        print(f"JSON report: {Path(arguments.json_output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
