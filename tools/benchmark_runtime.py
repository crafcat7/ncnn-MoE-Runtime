#!/usr/bin/env python3

import argparse
import ctypes
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


MIB = 1024 * 1024
DEFAULT_PROMPT_TOKEN_IDS = [
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
QWEN_MXFP4_ARTIFACT = "ncnn-moe-qwen3.6-mxfp4.safetensors"


def inspect_compiled_artifact(model):
    path = Path(model).resolve() / QWEN_MXFP4_ARTIFACT
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise ValueError(f"truncated compiled artifact header: {path}")
        (header_length,) = struct.unpack("<Q", encoded_length)
        if not 0 < header_length <= 128 * 1024 * 1024:
            raise ValueError(f"invalid compiled artifact header: {path}")
        encoded = stream.read(header_length)
        if len(encoded) != header_length:
            raise ValueError(f"truncated compiled artifact header: {path}")
    header = json.loads(encoded)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError(f"compiled artifact metadata is missing: {path}")
    fields = (
        "format",
        "model_type",
        "scope",
        "quantization",
        "source_config_sha256",
        "source_index_sha256",
        "source_config_fnv1a64",
        "source_index_fnv1a64",
    )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "metadata": {field: metadata.get(field) for field in fields},
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an ncnn_moe model runner with repeatable runtime, "
            "memory, and output-token reporting."
        )
    )
    parser.add_argument("runner", help="Path to an ncnn_moe model runner")
    parser.add_argument("model", help="Path to a supported model directory")
    parser.add_argument(
        "--model-revision",
        default="",
        help="Checkpoint revision or commit recorded in the report.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_TOKEN_IDS,
        help="Input token IDs passed to the runner (default: fixed 16-token prompt).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed explicitly to the runner (default: greedy).",
    )
    parser.add_argument(
        "--no-speculative",
        action="store_true",
        help="Disable model-specific speculative decoding in the runner.",
    )
    parser.add_argument(
        "--speculative-confidence",
        type=float,
        default=None,
        help="Override the runner's speculative confidence threshold.",
    )
    parser.add_argument(
        "--speculative-max-draft",
        type=int,
        default=0,
        help="Limit speculative draft tokens; zero keeps the model block size.",
    )
    parser.add_argument(
        "--parallel-speculative",
        action="store_true",
        help="Run parallel Sessions independently so each can use speculation.",
    )
    parser.add_argument(
        "--parallel-independent",
        action="store_true",
        help=(
            "Run parallel Sessions independently, with or without "
            "speculation, for like-for-like A/B measurements."
        ),
    )
    parser.add_argument(
        "--require-speculative",
        action="store_true",
        help="Fail when a measured run has no proposal or draft activity.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--cache-warmup-runs",
        type=int,
        default=0,
        help="Warm the shared model Expert cache before each measured run.",
    )
    parser.add_argument(
        "--parallel-sessions",
        type=int,
        default=1,
        help=(
            "Generate independent sequences through the cross-session "
            "scheduler and report aggregate throughput."
        ),
    )
    parser.add_argument(
        "--scheduler-expert-threads",
        type=int,
        default=0,
        help=(
            "Override Expert OpenMP threads per scheduler worker "
            "(zero keeps the runner's topology default)."
        ),
    )
    parser.add_argument(
        "--scheduler-staging",
        choices=("auto", "force", "off"),
        default="auto",
        help="Control cross-session staged decode batching.",
    )
    parser.add_argument(
        "--scheduler-cross-call",
        action="store_true",
        help=(
            "Submit each Session as an independent scheduler call so the "
            "Runtime cross-call collector is exercised."
        ),
    )
    parser.add_argument(
        "--scheduler-collection-us",
        type=int,
        default=200,
        help="Maximum cross-call collection window in microseconds.",
    )
    parser.add_argument(
        "--scheduler-max-micro-batch",
        type=int,
        default=0,
        help="Maximum collected requests; zero follows scheduler workers.",
    )
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
    parser.add_argument(
        "--expert-cache-sweep-mb",
        type=int,
        nargs="+",
        default=None,
        help="Run independent process repeats for each Expert cache size.",
    )
    parser.add_argument("--expert-gpu-cache-mb", type=int, default=0)
    parser.add_argument(
        "--expert-gpu-victim-cache-mb",
        type=int,
        default=0,
        help=(
            "Use a compressed-weight Vulkan L2 behind the host Expert ARC; "
            "entries can execute in place or restore to CPU."
        ),
    )
    parser.add_argument(
        "--disable-gpu-victim-execution",
        action="store_true",
        help=(
            "Disable in-place execution from the Vulkan victim tier for an "
            "A/B control or compatibility fallback."
        ),
    )
    parser.add_argument("--disable-router-prediction", action="store_true")
    parser.add_argument(
        "--release-vulkan-dense-host",
        action="store_true",
        help=(
            "Drop host copies of dense tensors after Vulkan operators take ownership."
        ),
    )
    parser.add_argument(
        "--disable-async-router-prediction", action="store_true"
    )
    parser.add_argument("--disable-forward-aware-cache", action="store_true")
    parser.add_argument("--disable-rank-adaptive-prefetch", action="store_true")
    parser.add_argument(
        "--disable-cross-expert-read-coalescing", action="store_true"
    )
    parser.add_argument("--router-prediction", action="store_true")
    parser.add_argument("--async-router-prediction", action="store_true")
    parser.add_argument("--forward-aware-cache", action="store_true")
    parser.add_argument("--rank-adaptive-prefetch", action="store_true")
    parser.add_argument("--cross-expert-read-coalescing", action="store_true")
    parser.add_argument(
        "--expert-gpu-victim-reuse-probe",
        type=int,
        default=1,
        help=(
            "Admit Experts observed across host-ARC residencies, plus one "
            "of each N first-observation evictions as a victim-tier probe."
        ),
    )
    parser.add_argument("--expert-io-workers", type=int, default=0)
    parser.add_argument(
        "--vulkan-device-index",
        type=int,
        default=None,
        help="Vulkan device index selected by the runtime.",
    )
    parser.add_argument(
        "--vulkan-device-indices",
        default=None,
        help=(
            "Comma-separated Vulkan devices used for weighted layer and "
            "Expert placement."
        ),
    )
    parser.add_argument(
        "--mmap-experts",
        action="store_true",
        help="Use memory-mapped on-demand MXFP4 ranges.",
    )
    parser.add_argument(
        "--direct-expert-io",
        action="store_true",
        help="Use aligned direct reads for on-demand MXFP4 ranges.",
    )
    parser.add_argument(
        "--buffered-expert-io",
        action="store_true",
        help="Force buffered reads instead of adaptive Expert I/O.",
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
    feature_pairs = (
        (
            arguments.router_prediction,
            arguments.disable_router_prediction,
            "router prediction",
        ),
        (
            arguments.async_router_prediction,
            arguments.disable_async_router_prediction,
            "async Router prediction",
        ),
        (
            arguments.forward_aware_cache,
            arguments.disable_forward_aware_cache,
            "Forward-aware cache",
        ),
        (
            arguments.rank_adaptive_prefetch,
            arguments.disable_rank_adaptive_prefetch,
            "Rank-adaptive prefetch",
        ),
        (
            arguments.cross_expert_read_coalescing,
            arguments.disable_cross_expert_read_coalescing,
            "cross-Expert read coalescing",
        ),
    )
    for enabled, disabled, name in feature_pairs:
        if enabled and disabled:
            raise ValueError(f"{name} cannot be both enabled and disabled")
    if (
        arguments.async_router_prediction
        and arguments.disable_router_prediction
    ):
        raise ValueError(
            "async Router prediction cannot disable Router prediction"
        )
    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not math.isfinite(arguments.temperature) or arguments.temperature < 0:
        raise ValueError("--temperature must be finite and non-negative")
    if arguments.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if arguments.cache_warmup_runs < 0:
        raise ValueError("--cache-warmup-runs must be non-negative")
    if (
        arguments.parallel_sessions <= 0
        or arguments.parallel_sessions > 64
    ):
        raise ValueError("--parallel-sessions must be between 1 and 64")
    if (
        arguments.scheduler_expert_threads < 0
        or arguments.scheduler_expert_threads > 1024
    ):
        raise ValueError(
            "--scheduler-expert-threads must be between 0 and 1024"
        )
    if (
        arguments.scheduler_collection_us < 0
        or arguments.scheduler_collection_us > 1000000
    ):
        raise ValueError(
            "--scheduler-collection-us must be between 0 and 1000000"
        )
    if (
        arguments.scheduler_max_micro_batch < 0
        or arguments.scheduler_max_micro_batch > 1024
    ):
        raise ValueError(
            "--scheduler-max-micro-batch must be between 0 and 1024"
        )
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if (
        arguments.speculative_confidence is not None
        and not 0.0 <= arguments.speculative_confidence <= 1.0
    ):
        raise ValueError("--speculative-confidence must be between 0 and 1")
    if arguments.speculative_max_draft < 0:
        raise ValueError("--speculative-max-draft must be non-negative")
    if arguments.no_speculative and (
        arguments.require_speculative or arguments.parallel_speculative
    ):
        raise ValueError(
            "--no-speculative cannot be combined with speculative requirements"
        )
    if (
        arguments.parallel_speculative
        and arguments.parallel_independent
    ):
        raise ValueError(
            "--parallel-speculative and --parallel-independent are mutually exclusive"
        )
    if (
        arguments.parallel_speculative or arguments.parallel_independent
    ) and arguments.parallel_sessions < 2:
        raise ValueError(
            "parallel independent generation requires at least two Sessions"
        )
    if (
        arguments.host_memory_mb < 0
        or arguments.expert_cache_mb < 0
        or arguments.expert_gpu_cache_mb < 0
        or arguments.expert_gpu_victim_cache_mb < 0
    ):
        raise ValueError("memory limits must be non-negative")
    if arguments.expert_cache_sweep_mb is not None:
        if any(cache_mb <= 0 for cache_mb in arguments.expert_cache_sweep_mb):
            raise ValueError("--expert-cache-sweep-mb values must be positive")
        if len(set(arguments.expert_cache_sweep_mb)) != len(
            arguments.expert_cache_sweep_mb
        ):
            raise ValueError("--expert-cache-sweep-mb values must be unique")
        if arguments.expert_cache_mb:
            raise ValueError(
                "--expert-cache-mb and --expert-cache-sweep-mb are mutually exclusive"
            )
    if arguments.expert_io_workers < 0 or arguments.expert_io_workers > 64:
        raise ValueError("--expert-io-workers must be between 0 and 64")
    if (
        arguments.expert_gpu_victim_reuse_probe < 1
        or arguments.expert_gpu_victim_reuse_probe > 1024
    ):
        raise ValueError(
            "--expert-gpu-victim-reuse-probe must be between 1 and 1024"
        )
    if (
        arguments.vulkan_device_index is not None
        and arguments.vulkan_device_index < 0
    ):
        raise ValueError("--vulkan-device-index must be non-negative")
    if arguments.vulkan_device_indices is not None:
        try:
            device_indices = [
                int(value)
                for value in arguments.vulkan_device_indices.split(",")
            ]
        except ValueError as error:
            raise ValueError(
                "--vulkan-device-indices must contain comma-separated integers"
            ) from error
        if not device_indices or any(index < 0 for index in device_indices):
            raise ValueError(
                "--vulkan-device-indices must contain non-negative indices"
            )
        if len(set(device_indices)) != len(device_indices):
            raise ValueError("--vulkan-device-indices must be unique")
        if (
            arguments.vulkan_device_index is not None
            and arguments.vulkan_device_index != device_indices[0]
        ):
            raise ValueError(
                "--vulkan-device-index must match the first multi-device index"
            )
    if arguments.sample_interval_ms < 20:
        raise ValueError("--sample-interval-ms must be at least 20")
    if arguments.gpu_index < 0:
        raise ValueError("--gpu-index must be non-negative")
    selected_io_modes = sum(
        (
            arguments.mmap_experts,
            arguments.direct_expert_io,
            arguments.buffered_expert_io,
        )
    )
    if selected_io_modes > 1:
        raise ValueError(
            "Expert mmap, direct I/O, and buffered I/O are mutually exclusive"
        )


def runner_command(arguments):
    command = [
        str(Path(arguments.runner)),
        str(Path(arguments.model)),
        *[str(token) for token in arguments.prompt_token_ids],
        "--max-new-tokens",
        str(arguments.max_new_tokens),
        "--temperature",
        str(arguments.temperature),
        "--report-throughput",
    ]
    if arguments.no_speculative:
        command.append("--no-speculative")
    else:
        command.append("--speculative")
    if arguments.speculative_confidence is not None:
        command.extend(
            [
                "--speculative-confidence",
                str(arguments.speculative_confidence),
            ]
        )
    if arguments.speculative_max_draft:
        command.extend(
            [
                "--speculative-max-draft",
                str(arguments.speculative_max_draft),
            ]
        )
    if arguments.parallel_speculative:
        command.append("--parallel-speculative")
    if arguments.parallel_independent:
        command.append("--parallel-independent")
    if arguments.cache_warmup_runs:
        command.extend(
            ["--cache-warmup-runs", str(arguments.cache_warmup_runs)]
        )
    if arguments.parallel_sessions != 1:
        command.extend(
            ["--parallel-sessions", str(arguments.parallel_sessions)]
        )
    if arguments.scheduler_expert_threads:
        command.extend(
            [
                "--scheduler-expert-threads",
                str(arguments.scheduler_expert_threads),
            ]
        )
    if arguments.scheduler_staging != "auto":
        command.extend(["--scheduler-staging", arguments.scheduler_staging])
    if arguments.scheduler_cross_call:
        command.append("--scheduler-cross-call")
    if arguments.scheduler_collection_us != 200:
        command.extend(
            [
                "--scheduler-collection-us",
                str(arguments.scheduler_collection_us),
            ]
        )
    if arguments.scheduler_max_micro_batch:
        command.extend(
            [
                "--scheduler-max-micro-batch",
                str(arguments.scheduler_max_micro_batch),
            ]
        )
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
    if arguments.expert_gpu_victim_cache_mb:
        command.extend(
            [
                "--expert-gpu-victim-cache-mb",
                str(arguments.expert_gpu_victim_cache_mb),
            ]
        )
        if arguments.disable_gpu_victim_execution:
            command.append("--disable-gpu-victim-execution")
        command.extend(
            [
                "--expert-gpu-victim-reuse-probe",
                str(arguments.expert_gpu_victim_reuse_probe),
            ]
        )
    if arguments.disable_router_prediction:
        command.append("--disable-router-prediction")
    if arguments.release_vulkan_dense_host:
        command.append("--release-vulkan-dense-host")
    if arguments.disable_async_router_prediction:
        command.append("--disable-async-router-prediction")
    if arguments.disable_forward_aware_cache:
        command.append("--disable-forward-aware-cache")
    if arguments.disable_rank_adaptive_prefetch:
        command.append("--disable-rank-adaptive-prefetch")
    if arguments.disable_cross_expert_read_coalescing:
        command.append("--disable-cross-expert-read-coalescing")
    if arguments.router_prediction:
        command.append("--router-prediction")
    if arguments.async_router_prediction:
        command.append("--async-router-prediction")
    if arguments.forward_aware_cache:
        command.append("--forward-aware-cache")
    if arguments.rank_adaptive_prefetch:
        command.append("--rank-adaptive-prefetch")
    if arguments.cross_expert_read_coalescing:
        command.append("--cross-expert-read-coalescing")
    if arguments.expert_io_workers:
        command.extend(["--expert-io-workers", str(arguments.expert_io_workers)])
    if arguments.vulkan_device_index is not None:
        command.extend(
            ["--vulkan-device", str(arguments.vulkan_device_index)]
        )
    if arguments.vulkan_device_indices is not None:
        command.extend(
            ["--vulkan-devices", arguments.vulkan_device_indices]
        )
    if arguments.mmap_experts:
        command.append("--mmap-experts")
    if arguments.direct_expert_io:
        command.append("--direct-expert-io")
    if arguments.buffered_expert_io:
        command.append("--buffered-expert-io")
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

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
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
        self.kernel32.GetProcessIoCounters.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self.IoCounters),
        ]
        self.kernel32.GetProcessIoCounters.restype = ctypes.c_int
        access = 0x0400 | 0x1000 | 0x0010
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

    def io_counters(self):
        if not self.handle:
            return None
        counters = self.IoCounters()
        if not self.kernel32.GetProcessIoCounters(
            self.handle, ctypes.byref(counters)
        ):
            return None
        return {
            "logical_read_bytes": int(counters.ReadTransferCount),
            "logical_read_operations": int(counters.ReadOperationCount),
            "physical_read_bytes": None,
            "source": "Windows GetProcessIoCounters ReadTransferCount",
        }

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

    def io_counters(self):
        if self.windows:
            return self.windows.io_counters()
        if sys.platform.startswith("linux"):
            try:
                values = {}
                for line in Path(
                    f"/proc/{self.process_id}/io"
                ).read_text(encoding="utf-8").splitlines():
                    key, value = line.split(":", 1)
                    values[key.strip()] = int(value.strip())
            except (OSError, ValueError):
                return None
            return {
                "logical_read_bytes": values.get("rchar"),
                "logical_read_operations": values.get("syscr"),
                "physical_read_bytes": values.get("read_bytes"),
                "source": "Linux /proc/<pid>/io rchar/read_bytes",
            }
        return None

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


class WindowsPhysicalDiskMonitor:
    COUNTER = r"\\PhysicalDisk(_Total)\\Disk Read Bytes/sec"

    def __init__(self):
        self.executable = shutil.which("typeperf")
        self.powershell = shutil.which("powershell")
        self.output = None
        self.process = None
        self.sample_source = None

    def start(self):
        if not self.executable and not self.powershell:
            return
        if self.executable:
            self.output = tempfile.TemporaryFile(mode="w+b")
            try:
                self.process = subprocess.Popen(
                    [self.executable, self.COUNTER, "-si", "1"],
                    stdout=self.output,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.05)
                if self.process.poll() is not None:
                    self.process = None
                    self.output.close()
                    self.output = None
            except OSError:
                if self.output:
                    self.output.close()
                self.output = None
        if self.output is not None or not self.powershell:
            self.sample_source = "typeperf"
            return
        self.output = tempfile.TemporaryFile(mode="w+b")
        command = (
            "$ErrorActionPreference='SilentlyContinue'; "
            "while ($true) { "
            "$counter = Get-CimInstance "
            "Win32_PerfFormattedData_PerfDisk_PhysicalDisk | "
            "Where-Object { $_.Name -eq '_Total' }; "
            "if ($counter) { [Console]::WriteLine($counter.DiskReadBytesPersec) }; "
            "Start-Sleep -Seconds 1 }"
        )
        try:
            self.process = subprocess.Popen(
                [self.powershell, "-NoProfile", "-Command", command],
                stdout=self.output,
                stderr=subprocess.DEVNULL,
            )
            self.sample_source = "CIM"
        except OSError:
            self.output.close()
            self.output = None

    def stop(self):
        if not self.process or not self.output:
            return None
        self.process.terminate()
        try:
            self.process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
        self.output.flush()
        self.output.seek(0)
        text = self.output.read().decode("utf-8", errors="replace")
        self.output.close()
        self.output = None
        values = []
        for line in text.splitlines():
            value = line.rsplit(",", 1)[-1].strip().strip('"')
            try:
                values.append(float(value.replace(",", "")))
            except ValueError:
                continue
        if not values:
            return None
        return {
            "physical_read_bytes": int(sum(values)),
            "source": (
                f"Windows {self.sample_source} physical disk read bytes/sec "
                "(system total, one-second samples)"
            ),
        }


class LinuxPhysicalDiskMonitor:
    def __init__(self):
        self.before = None

    @staticmethod
    def snapshot():
        try:
            block_devices = {
                path.name for path in Path("/sys/block").iterdir()
            }
            values = {}
            for line in Path("/proc/diskstats").read_text(
                encoding="utf-8"
            ).splitlines():
                fields = line.split()
                if len(fields) < 14 or fields[2] not in block_devices:
                    continue
                values[fields[2]] = int(fields[5]) * 512
            return values
        except (OSError, ValueError):
            return None

    def start(self):
        self.before = self.snapshot()

    def stop(self):
        if self.before is None:
            return None
        after = self.snapshot()
        if after is None:
            return None
        total = sum(
            max(0, after.get(name, 0) - value)
            for name, value in self.before.items()
        )
        return {
            "physical_read_bytes": total,
            "source": "Linux /proc/diskstats base block devices (system total)",
        }


class PhysicalDiskMonitor:
    def __init__(self):
        if os.name == "nt":
            self.monitor = WindowsPhysicalDiskMonitor()
        elif sys.platform.startswith("linux"):
            self.monitor = LinuxPhysicalDiskMonitor()
        else:
            self.monitor = None

    def start(self):
        if self.monitor:
            self.monitor.start()

    def stop(self):
        return self.monitor.stop() if self.monitor else None


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


def extract_text(output, pattern):
    match = re.search(pattern, output, re.MULTILINE)
    return match.group(1) if match else None


def io_delta(before, after, field):
    if not before or not after:
        return None
    before_value = before.get(field)
    after_value = after.get(field)
    if before_value is None or after_value is None:
        return None
    return max(0, after_value - before_value)


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
    hotsets = [
        {
            "capacity_mib": int(match.group(1)),
            "resident_experts": int(match.group(2)),
            "active_experts": int(match.group(3)),
            "resident_bytes": int(match.group(4)),
            "batch_weight_coverage": float(match.group(5)) / 100.0,
            "route_weight_coverage": float(match.group(6)) / 100.0,
        }
        for match in re.finditer(
            r"^Expert static hotset at (\d+) MiB: "
            r"(\d+)/(\d+) Expert\(s\), (\d+) bytes resident, "
            r"([0-9.]+)% batch-byte coverage, "
            r"([0-9.]+)% route-byte coverage$",
            output,
            re.MULTILINE,
        )
    ]
    route_ranks = [
        {
            "rank": int(match.group(1)),
            "predictions": int(match.group(2)),
            "matches": int(match.group(3)),
            "demands": int(match.group(4)),
            "demand_queue_time_us": int(match.group(5)),
        }
        for match in re.finditer(
            r"\br(\d+) (\d+) predicted/(\d+) matched/"
            r"(\d+) demanded/(\d+)us queued;",
            output,
        )
    ]
    result = {
        "load_seconds": extract_number(
            output, r"^loaded [a-zA-Z0-9_-]+ in ([0-9.]+) s,"
        ),
        "runtime_backend": extract_text(
            output,
            r"^loaded [a-zA-Z0-9_-]+ in [0-9.]+ s, backend (.+)$",
        ),
        "routed_expert_format": extract_text(
            output, r"^Routed Expert format: (.+)$"
        ),
        "generated_tokens": extract_number(
            output, r"^generated (\d+) token\(s\) in", int
        ),
        "parallel_sessions": extract_number(
            output, r"^Parallel sessions: (\d+)", int
        ),
        "reported_aggregate_tokens_per_second": extract_number(
            output,
            r"^(?:Parallel sessions: \d+, aggregate throughput: "
            r"|Aggregate throughput: )([0-9.]+) token/s",
        ),
        "generation_seconds": extract_number(
            output, r"^generated \d+ token\(s\) in ([0-9.]+) s"
        ),
        "attention_ms": extract_number(
            output, r"^Attention time: ([0-9.]+) ms"
        ),
        "router_ms": extract_number(output, r"^Router time: ([0-9.]+) ms"),
        "expert_ms": extract_number(output, r"^Expert time: ([0-9.]+) ms"),
        "expert_batch_weight_bytes": extract_number(
            output, r"^Expert weight demand: (\d+) batched bytes", int
        ),
        "expert_route_weight_bytes": extract_number(
            output,
            r"^Expert weight demand: \d+ batched bytes, (\d+) route bytes",
            int,
        ),
        "expert_cache_wait_ms": extract_number(
            output, r"^Expert cache wait time: ([0-9.]+) ms"
        ),
        "expert_cache_management_ms": extract_number(
            output, r"^Expert cache management time: ([0-9.]+) ms"
        ),
        "expert_engine_ms": extract_number(
            output, r"^Expert engine wall time: ([0-9.]+) ms"
        ),
        "expert_compute_ms": extract_number(
            output, r"^Expert compute(?: wall)? time: ([0-9.]+) ms"
        ),
        "expert_orchestration_ms": extract_number(
            output, r"^Expert orchestration wall time: ([0-9.]+) ms"
        ),
        "expert_regroup_ms": extract_number(
            output, r"^Expert regroup time: ([0-9.]+) ms"
        ),
        "expert_combine_ms": extract_number(
            output, r"^Expert combine time: ([0-9.]+) ms"
        ),
        "embedding_ms": extract_number(
            output, r"^Embedding time: ([0-9.]+) ms"
        ),
        "final_norm_ms": extract_number(
            output, r"^Final norm time: ([0-9.]+) ms"
        ),
        "lm_head_ms": extract_number(
            output, r"^LM head time: ([0-9.]+) ms"
        ),
        "speculative_proposals": extract_number(
            output, r"^Speculative decoding: (\d+) proposal", int
        ),
        "speculative_draft_tokens": extract_number(
            output,
            r"^Speculative decoding: \d+ proposal\(s\), (\d+) draft token",
            int,
        ),
        "speculative_accepted_tokens": extract_number(
            output,
            r"^Speculative decoding: \d+ proposal\(s\), \d+ draft token\(s\), "
            r"(\d+) accepted token",
            int,
        ),
        "speculative_context_ms": extract_number(
            output, r"^Speculative time: ([0-9.]+) ms context"
        ),
        "speculative_draft_ms": extract_number(
            output,
            r"^Speculative time: [0-9.]+ ms context, ([0-9.]+) ms draft",
        ),
        "speculative_verify_ms": extract_number(
            output,
            r"^Speculative time: [0-9.]+ ms context, [0-9.]+ ms draft, "
            r"([0-9.]+) ms verify",
        ),
        "physical_cpu_core_count": extract_number(
            output, r"^CPU topology: (\d+) physical core", int
        ),
        "runtime_logical_cpu_count": extract_number(
            output,
            r"^CPU topology: \d+ physical core\(s\), (\d+) logical processor",
            int,
        ),
        "runtime_openmp_thread_count": extract_number(
            output, r"^CPU topology:.*?(\d+) OpenMP thread", int
        ),
        "cpu_isa": extract_text(
            output, r"^CPU ISA capabilities: (.+)$"
        ),
        "mxfp4_kernel": extract_text(
            output, r"^MXFP4 CPU kernel: (.+)$"
        ),
        "mxfp4_decode_row_pair_group_size": extract_number(
            output, r"^MXFP4 decode row-pair group: (\d+)", int
        ),
        "activation_kernel": extract_text(
            output, r"^Activation CPU kernel: (.+)$"
        ),
        "vulkan_runtime_device_count": extract_number(
            output, r"^Vulkan runtime devices: (\d+)", int
        ),
        "vulkan_heap_budget_mib": extract_number(
            output, r"^Vulkan heap budget: (\d+) MiB", int
        ),
        "vulkan_linear_dispatches": extract_number(
            output, r"^Vulkan linear dispatches: (\d+)", int
        ),
        "vulkan_compute_submissions": extract_number(
            output, r"^Vulkan compute submissions: (\d+)", int
        ),
        "vulkan_batch_uploads": extract_number(
            output, r"^Vulkan batch transfers: (\d+) upload", int
        ),
        "vulkan_batch_downloads": extract_number(
            output,
            r"^Vulkan batch transfers: \d+ upload\(s\), (\d+) download",
            int,
        ),
        "vulkan_submit_wait_ms": extract_number(
            output, r"^Vulkan submit/wait time: ([0-9.eE+-]+) ms", float
        ),
        "vulkan_auxiliary_uploads": extract_number(
            output,
            r"^Vulkan auxiliary uploads: (\d+) upload",
            int,
        ),
        "vulkan_auxiliary_upload_bytes": extract_number(
            output,
            r"^Vulkan auxiliary uploads: \d+ upload\(s\), (\d+) bytes",
            int,
        ),
        "vulkan_model_device_index": extract_number(
            output, r"^Vulkan model device: (\d+)", int
        ),
        "vulkan_model_devices": extract_text(
            output, r"^Vulkan model devices: ([^;]+);"
        ),
        "vulkan_layer_placement": extract_text(
            output, r"^Vulkan model devices: [^;]+; layer placement: (.+)$"
        ),
        "vulkan_attention_blocks": extract_number(
            output, r"^Vulkan attention blocks: (\d+)", int
        ),
        "vulkan_kernel_features": extract_text(
            output, r"^Vulkan kernel features: (.+)$"
        ),
        "vulkan_command_buffer_reuses": extract_number(
            output,
            r"^Vulkan command buffer reuses: (\d+)",
            int,
        ),
        "vulkan_attention_qkv_rope_fusions": extract_number(
            output,
            r"^Vulkan attention fusion: (\d+) QKV\+RoPE block",
            int,
        ),
        "vulkan_attention_qkv_ring_fusions": extract_number(
            output,
            r"^Vulkan attention fusion: \d+ QKV\+RoPE block\(s\), (\d+) QKV->ring block",
            int,
        ),
        "vulkan_attention_decode_sdpa_fusions": extract_number(
            output,
            r"^Vulkan attention fusion: \d+ QKV\+RoPE block\(s\), \d+ QKV->ring block\(s\), (\d+) Decode-SDPA block",
            int,
        ),
        "vulkan_kv_ring_appends": extract_number(
            output, r"^Vulkan KV ring: (\d+) append", int
        ),
        "vulkan_kv_ring_resizes": extract_number(
            output, r"^Vulkan KV ring: \d+ append\(s\), (\d+) resize", int
        ),
        "vulkan_kv_ring_wrapped_views": extract_number(
            output,
            r"^Vulkan KV ring: \d+ append\(s\), \d+ resize\(s\), (\d+) wrapped",
            int,
        ),
        "scheduler_staged_batches": extract_number(
            output, r"^Scheduler staging: (\d+) batch", int
        ),
        "scheduler_staged_requests": extract_number(
            output,
            r"^Scheduler staging: \d+ batch\(es\), (\d+) request",
            int,
        ),
        "scheduler_staging_bypassed_batches": extract_number(
            output,
            r"^Scheduler staging: \d+ batch\(es\), \d+ request\(s\), (\d+) bypassed",
            int,
        ),
        "scheduler_logical_expert_batches": extract_number(
            output,
            r"^Scheduler staging:.*?(\d+) logical Expert batch",
            int,
        ),
        "scheduler_physical_expert_batches": extract_number(
            output,
            r"^Scheduler staging:.*?-> (\d+) physical",
            int,
        ),
        "scheduler_coalesced_expert_routes": extract_number(
            output,
            r"^Scheduler staging:.*?(\d+) coalesced route",
            int,
        ),
        "scheduler_max_coalesced_expert_batch_size": extract_number(
            output,
            r"^Scheduler staging:.*?max (\d+) row",
            int,
        ),
        "scheduler_adaptive_staged_decisions": extract_number(
            output,
            r"^Scheduler adaptive policy: (\d+) staged decision",
            int,
        ),
        "scheduler_adaptive_independent_decisions": extract_number(
            output,
            r"^Scheduler adaptive policy: \d+ staged decision\(s\), (\d+) independent",
            int,
        ),
        "scheduler_adaptive_probe_decisions": extract_number(
            output,
            r"^Scheduler adaptive policy:.*?(\d+) probe",
            int,
        ),
        "scheduler_adaptive_policy_switches": extract_number(
            output,
            r"^Scheduler adaptive policy:.*?(\d+) switch",
            int,
        ),
        "scheduler_adaptive_staged_ms_per_request": extract_number(
            output,
            r"^Scheduler adaptive policy:.*?([0-9.]+)/[0-9.]+ mean ms/request",
        ),
        "scheduler_adaptive_independent_ms_per_request": extract_number(
            output,
            r"^Scheduler adaptive policy:.*?[0-9.]+/([0-9.]+) mean ms/request",
        ),
        "scheduler_adaptive_resident_decisions": extract_number(
            output,
            r"^Scheduler adaptive phases: (\d+)/",
            int,
        ),
        "scheduler_adaptive_mixed_decisions": extract_number(
            output,
            r"^Scheduler adaptive phases: \d+/(\d+)/",
            int,
        ),
        "scheduler_adaptive_storage_decisions": extract_number(
            output,
            r"^Scheduler adaptive phases: \d+/\d+/(\d+) resident",
            int,
        ),
        "scheduler_adaptive_resident_observations": extract_number(
            output,
            r"^Scheduler adaptive phases:.*?, (\d+)/\d+/\d+ observation",
            int,
        ),
        "scheduler_adaptive_mixed_observations": extract_number(
            output,
            r"^Scheduler adaptive phases:.*?, \d+/(\d+)/\d+ observation",
            int,
        ),
        "scheduler_adaptive_storage_observations": extract_number(
            output,
            r"^Scheduler adaptive phases:.*?, \d+/\d+/(\d+) observation",
            int,
        ),
        "scheduler_adaptive_phase_changes": extract_number(
            output,
            r"^Scheduler adaptive phases:.*?(\d+) phase change",
            int,
        ),
        "scheduler_adaptive_noisy_switch_rejections": extract_number(
            output,
            r"^Scheduler adaptive phases:.*?(\d+) noisy switch rejection",
            int,
        ),
        "scheduler_cross_call_collected_batches": extract_number(
            output,
            r"^Scheduler cross-call: (\d+) collected batch",
            int,
        ),
        "scheduler_cross_call_collected_requests": extract_number(
            output,
            r"^Scheduler cross-call: \d+ collected batch\(es\), (\d+) collected request",
            int,
        ),
        "scheduler_cross_call_collection_probes": extract_number(
            output,
            r"^Scheduler cross-call:.*?(\d+) probe",
            int,
        ),
        "scheduler_cross_call_collection_timeouts": extract_number(
            output,
            r"^Scheduler cross-call:.*?(\d+) timeout",
            int,
        ),
        "scheduler_cross_call_collection_bypasses": extract_number(
            output,
            r"^Scheduler cross-call:.*?(\d+) bypass",
            int,
        ),
        "scheduler_cross_call_collection_wait_us": extract_number(
            output,
            r"^Scheduler cross-call:.*?(\d+) us waiting",
            int,
        ),
        "scheduler_cross_call_max_batch_size": extract_number(
            output,
            r"^Scheduler cross-call:.*?max batch (\d+)",
            int,
        ),
        "scheduler_cross_call_max_pending": extract_number(
            output,
            r"^Scheduler cross-call:.*?max pending (\d+)",
            int,
        ),
        "expert_route_predictions": extract_number(
            output,
            r"^Expert route prediction: (\d+) prediction",
            int,
        ),
        "expert_route_prediction_matches": extract_number(
            output,
            r"^Expert route prediction: \d+ prediction\(s\), (\d+) match",
            int,
        ),
        "expert_route_prediction_cache_hits": extract_number(
            output,
            r"^Expert route prediction:.*?(\d+) cache-ready",
            int,
        ),
        "expert_route_prediction_cache_misses": extract_number(
            output,
            r"^Expert route prediction:.*?(\d+) not-ready",
            int,
        ),
        "expert_route_prediction_ms": extract_number(
            output,
            r"^Expert route prediction:.*?([0-9.]+) ms predictor",
        ),
        "expert_route_prediction_wait_ms": extract_number(
            output,
            r"^Expert route prediction:.*?([0-9.]+) ms waiting",
        ),
        "expert_route_prediction_async_submissions": extract_number(
            output,
            r"^Expert route prediction:.*?(\d+) async submission",
            int,
        ),
        "expert_route_prediction_async_completions": extract_number(
            output,
            r"^Expert route prediction:.*?(\d+) completion",
            int,
        ),
        "expert_route_prediction_async_fallbacks": extract_number(
            output,
            r"^Expert route prediction:.*?(\d+) fallback",
            int,
        ),
        "expert_route_ranks": route_ranks,
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
        "expert_cache_queued_reads": extract_number(
            output, r"^Expert cache:.*?(\d+) queued read", int
        ),
        "expert_cache_speculative_reads": extract_number(
            output, r"^Expert cache:.*?(\d+) speculative read", int
        ),
        "expert_cache_cancelled_speculative_reads": extract_number(
            output,
            r"^Expert cache:.*?(\d+) cancelled speculative read",
            int,
        ),
        "expert_cache_dropped_speculative_admissions": extract_number(
            output,
            r"^Expert cache:.*?(\d+) dropped speculative admission",
            int,
        ),
        "expert_cache_unused_speculative_reads": extract_number(
            output,
            r"^Expert cache:.*?(\d+) unused speculative read",
            int,
        ),
        "expert_cache_short_term_reloads": extract_number(
            output, r"^Expert cache:.*?(\d+) short-term reload", int
        ),
        "expert_cache_arc_recent_bytes": extract_number(
            output, r"^Expert ARC: T1 (\d+) bytes", int
        ),
        "expert_cache_arc_frequent_bytes": extract_number(
            output, r"^Expert ARC:.*?T2 (\d+) bytes", int
        ),
        "expert_cache_arc_recent_target_bytes": extract_number(
            output, r"^Expert ARC:.*?target T1 (\d+) bytes", int
        ),
        "expert_cache_arc_recent_ghost_bytes": extract_number(
            output, r"^Expert ARC:.*?B1 (\d+) bytes", int
        ),
        "expert_cache_arc_frequent_ghost_bytes": extract_number(
            output, r"^Expert ARC:.*?B2 (\d+) bytes", int
        ),
        "expert_cache_arc_recent_ghost_hits": extract_number(
            output,
            r"^Expert ARC:.*?B1 \d+ bytes, B2 \d+ bytes, B1 (\d+) hit",
            int,
        ),
        "expert_cache_arc_frequent_ghost_hits": extract_number(
            output, r"^Expert ARC:.*?B2 (\d+) hit", int
        ),
        "expert_cache_mapped_ranges": extract_number(
            output, r"^Expert mmap: (\d+) range", int
        ),
        "expert_cache_mapped_bytes": extract_number(
            output, r"^Expert mmap:.*?(\d+) bytes", int
        ),
        "expert_cache_read_policy": extract_text(
            output, r"^Expert I/O policy: ([^,]+)"
        ),
        "expert_cache_direct_read_ranges": extract_number(
            output, r"^Expert I/O policy:.*?(\d+) direct range", int
        ),
        "expert_cache_direct_read_bytes": extract_number(
            output,
            r"^Expert I/O policy:.*?direct range\(s\), (\d+) direct byte",
            int,
        ),
        "expert_cache_direct_read_fallbacks": extract_number(
            output, r"^Expert I/O policy:.*?(\d+) fallback", int
        ),
        "expert_cache_buffered_read_ranges": extract_number(
            output, r"^Expert I/O policy:.*?(\d+) buffered range", int
        ),
        "expert_cache_buffered_read_bytes": extract_number(
            output,
            r"^Expert I/O policy:.*?buffered range\(s\), (\d+) buffered byte",
            int,
        ),
        "expert_cache_coalesced_read_batches": extract_number(
            output, r"^Expert I/O policy:.*?(\d+) coalesced batch", int
        ),
        "expert_cache_coalesced_experts": extract_number(
            output, r"^Expert I/O policy:.*?(\d+) coalesced Expert", int
        ),
        "expert_cache_coalesced_read_ranges_saved": extract_number(
            output, r"^Expert I/O policy:.*?(\d+) physical range\(s\) saved", int
        ),
        "expert_gpu_cache_hits": extract_number(
            output, r"^Expert GPU execution cache: (\d+) hit", int
        ),
        "expert_gpu_cache_misses": extract_number(
            output,
            r"^Expert GPU execution cache: \d+ hit\(s\), (\d+) miss",
            int,
        ),
        "expert_gpu_cache_admissions": extract_number(
            output, r"^Expert GPU execution cache:.*?(\d+) admission", int
        ),
        "expert_gpu_cache_stores": extract_number(
            output, r"^Expert GPU execution cache:.*?(\d+) store", int
        ),
        "expert_gpu_cache_evictions": extract_number(
            output, r"^Expert GPU execution cache:.*?(\d+) eviction", int
        ),
        "expert_gpu_cache_dropped_admissions": extract_number(
            output,
            r"^Expert GPU execution cache:.*?(\d+) dropped admission",
            int,
        ),
        "expert_gpu_cache_bytes_uploaded": extract_number(
            output,
            r"^Expert GPU execution cache:.*?(\d+) bytes uploaded",
            int,
        ),
        "expert_gpu_cache_resident_bytes": extract_number(
            output,
            r"^Expert GPU execution cache:.*?(\d+) bytes resident",
            int,
        ),
        "expert_gpu_cache_pending_bytes": extract_number(
            output,
            r"^Expert GPU execution cache:.*?(\d+) bytes pending",
            int,
        ),
        "expert_gpu_victim_cache_hits": extract_number(
            output, r"^Expert GPU victim cache: (\d+) hit", int
        ),
        "expert_gpu_victim_cache_misses": extract_number(
            output,
            r"^Expert GPU victim cache: \d+ hit\(s\), (\d+) miss",
            int,
        ),
        "expert_gpu_victim_cache_admissions": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) admission", int
        ),
        "expert_gpu_victim_cache_filtered_admissions": extract_number(
            output,
            r"^Expert GPU victim cache:.*?(\d+) filtered admission",
            int,
        ),
        "expert_gpu_victim_cache_reused_admissions": extract_number(
            output,
            r"^Expert GPU victim cache:.*?(\d+) reused admission",
            int,
        ),
        "expert_gpu_victim_cache_probe_admissions": extract_number(
            output,
            r"^Expert GPU victim cache:.*?(\d+) probe admission",
            int,
        ),
        "expert_gpu_victim_cache_stores": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) store", int
        ),
        "expert_gpu_victim_cache_evictions": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) eviction", int
        ),
        "expert_gpu_victim_cache_dropped_admissions": extract_number(
            output,
            r"^Expert GPU victim cache:.*?(\d+) dropped admission",
            int,
        ),
        "expert_gpu_victim_cache_restore_failures": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) restore failure", int
        ),
        "expert_gpu_victim_cache_bytes_uploaded": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) bytes uploaded", int
        ),
        "expert_gpu_victim_cache_bytes_downloaded": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) bytes downloaded", int
        ),
        "expert_gpu_victim_cache_restore_ms": extract_number(
            output, r"^Expert GPU victim cache:.*?([0-9.]+) ms restoring"
        ),
        "expert_gpu_victim_cache_mapped_stores": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) mapped store", int
        ),
        "expert_gpu_victim_cache_mapped_restores": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) mapped restore", int
        ),
        "expert_gpu_victim_cache_resident_bytes": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) bytes resident", int
        ),
        "expert_gpu_victim_cache_pending_bytes": extract_number(
            output, r"^Expert GPU victim cache:.*?(\d+) bytes pending", int
        ),
        "expert_gpu_executions": extract_number(
            output, r"^Expert GPU execution: (\d+) execution", int
        ),
        "expert_gpu_execution_failures": extract_number(
            output, r"^Expert GPU execution:.*?(\d+) failure", int
        ),
        "expert_gpu_cpu_preferred": extract_number(
            output, r"^Expert GPU execution:.*?(\d+) CPU-preferred", int
        ),
        "expert_gpu_execution_ms": extract_number(
            output, r"^Expert GPU execution:.*?([0-9.]+) ms executing"
        ),
        "expert_gpu_device_source_hits": extract_number(
            output, r"^Expert GPU device source: (\d+) hit", int
        ),
        "expert_gpu_device_source_misses": extract_number(
            output,
            r"^Expert GPU device source: \d+ hit\(s\), (\d+) miss",
            int,
        ),
        "expert_gpu_device_source_executions": extract_number(
            output, r"^Expert GPU device source:.*?(\d+) execution", int
        ),
        "expert_gpu_device_source_execution_failures": extract_number(
            output, r"^Expert GPU device source:.*?(\d+) failure", int
        ),
        "expert_gpu_arc_recent_bytes": extract_number(
            output, r"^Expert GPU ARC: (\d+) recent byte", int
        ),
        "expert_gpu_arc_frequent_bytes": extract_number(
            output, r"^Expert GPU ARC:.*?(\d+) frequent byte", int
        ),
        "expert_gpu_arc_recent_target_bytes": extract_number(
            output, r"^Expert GPU ARC:.*?(\d+) recent target byte", int
        ),
        "expert_gpu_arc_recent_ghost_bytes": extract_number(
            output, r"^Expert GPU ARC:.*?(\d+) recent ghost byte", int
        ),
        "expert_gpu_arc_frequent_ghost_bytes": extract_number(
            output, r"^Expert GPU ARC:.*?(\d+) frequent ghost byte", int
        ),
        "expert_static_hotsets": hotsets,
        "generated_token_ids": tokens,
    }
    def has_positive(field):
        return (result[field] or 0) > 0

    result["gpu_compute_detected"] = any(
        has_positive(field)
        for field in (
            "vulkan_linear_dispatches",
            "vulkan_compute_submissions",
            "vulkan_attention_blocks",
            "expert_gpu_executions",
            "expert_gpu_device_source_executions",
        )
    )
    result["gpu_transfer_detected"] = any(
        has_positive(field)
        for field in (
            "vulkan_batch_uploads",
            "vulkan_batch_downloads",
            "vulkan_auxiliary_uploads",
            "expert_gpu_cache_bytes_uploaded",
            "expert_gpu_victim_cache_bytes_uploaded",
        )
    )
    if result["generation_seconds"] and result["generated_tokens"]:
        result["decode_tokens_per_second"] = (
            result["generated_tokens"] / result["generation_seconds"]
        )
    else:
        result["decode_tokens_per_second"] = None
    if result["generated_tokens"] and result["expert_batch_weight_bytes"] is not None:
        result["expert_batch_weight_bytes_per_generated_token"] = (
            result["expert_batch_weight_bytes"] / result["generated_tokens"]
        )
        result["required_pcie_gib_per_second_at_20_tps"] = (
            result["expert_batch_weight_bytes_per_generated_token"]
            * 20
            / (1024**3)
        )
    else:
        result["expert_batch_weight_bytes_per_generated_token"] = None
        result["required_pcie_gib_per_second_at_20_tps"] = None
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


def summarize_execution_evidence(arguments, observations, maximum):
    runtime_backends = sorted(
        {
            observation["runtime_backend"]
            for observation in observations
            if observation["runtime_backend"] is not None
        }
    )
    runtime_device_counts = [
        observation["vulkan_runtime_device_count"]
        for observation in observations
    ]
    vulkan_context_initialized = (
        None
        if any(count is None for count in runtime_device_counts)
        else any(count > 0 for count in runtime_device_counts)
    )
    model_vulkan_devices = any(
        observation["vulkan_model_devices"]
        for observation in observations
    )
    gpu_compute_detected = any(
        observation["gpu_compute_detected"]
        for observation in observations
    )
    gpu_transfer_detected = any(
        observation["gpu_transfer_detected"]
        for observation in observations
    )
    strict_no_vulkan_context_verified = (
        None
        if any(count is None for count in runtime_device_counts)
        else arguments.backend == "cpu"
        and all(count == 0 for count in runtime_device_counts)
    )
    cpu_only_execution_verified = (
        arguments.backend == "cpu"
        and bool(runtime_backends)
        and all(
            backend in ("cpu", "cpu-only")
            for backend in runtime_backends
        )
        and not model_vulkan_devices
        and not gpu_compute_detected
        and not gpu_transfer_detected
    )
    return {
        "requested_backend": arguments.backend,
        "runtime_backends": runtime_backends,
        "runtime_backend": observations[0]["runtime_backend"],
        "vulkan_runtime_device_count": observations[0][
            "vulkan_runtime_device_count"
        ],
        "vulkan_context_initialized": vulkan_context_initialized,
        "model_vulkan_devices_present": model_vulkan_devices,
        "gpu_compute_detected": gpu_compute_detected,
        "gpu_transfer_detected": gpu_transfer_detected,
        "cpu_only_execution_verified": cpu_only_execution_verified,
        "strict_no_vulkan_context_verified": strict_no_vulkan_context_verified,
        "gpu_memory_observation": {
            "source": "nvidia-smi query-gpu=memory.used",
            "scope": "whole GPU, not child-process attribution",
            "peak_mib": maximum["peak_gpu_mib"],
            "peak_delta_mib": maximum["peak_gpu_delta_mib"],
            "observed": maximum["peak_gpu_mib"] is not None,
            "attributed_to_process": False,
        },
    }


def build_execution_evidence(arguments, samples, maximum):
    return summarize_execution_evidence(
        arguments,
        [
            {
                "runtime_backend": sample["runtime_backend"],
                "vulkan_runtime_device_count": sample[
                    "vulkan_runtime_device_count"
                ],
                "vulkan_model_devices": sample["vulkan_model_devices"],
                "gpu_compute_detected": sample["gpu_compute_detected"],
                "gpu_transfer_detected": sample["gpu_transfer_detected"],
            }
            for sample in samples
        ],
        maximum,
    )


def build_cache_sweep_execution_evidence(arguments, reports, maximum):
    observations = []
    for report in reports:
        evidence = report.get("execution_evidence")
        if evidence is None:
            raise ValueError(
                "cache sweep child report is missing execution_evidence"
            )
        observations.append(
            {
                "runtime_backend": evidence["runtime_backend"],
                "vulkan_runtime_device_count": evidence[
                    "vulkan_runtime_device_count"
                ],
                "vulkan_model_devices": evidence[
                    "model_vulkan_devices_present"
                ],
                "gpu_compute_detected": evidence["gpu_compute_detected"],
                "gpu_transfer_detected": evidence["gpu_transfer_detected"],
            }
        )
    return summarize_execution_evidence(arguments, observations, maximum)


def execute_sample(command, gpu_index, sample_interval_ms):
    gpu_monitor = NvidiaMemoryMonitor(gpu_index, sample_interval_ms)
    gpu_monitor.start()
    disk_monitor = PhysicalDiskMonitor()
    disk_monitor.start()
    # The runner emits enough diagnostics for a Windows anonymous pipe to
    # fill. Keeping both streams in temporary files lets the parent continue
    # sampling memory without deadlocking while the child writes its report.
    with tempfile.TemporaryFile(
        mode="w+", encoding="utf-8", errors="replace"
    ) as stdout_file, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8", errors="replace"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        memory = PortableProcessMemory(process.pid)
        io_before = memory.io_counters()
        while process.poll() is None:
            memory.sample()
            time.sleep(sample_interval_ms / 1000.0)
        io_after = memory.io_counters()
        process.wait()
        stdout_file.flush()
        stderr_file.flush()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    memory.close()
    peak_gpu_mib, gpu_delta_mib = gpu_monitor.stop()
    physical_io = disk_monitor.stop()
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
    parsed["runtime_logical_read_bytes"] = parsed["expert_cache_bytes_read"]
    parsed["process_logical_read_bytes"] = io_delta(
        io_before, io_after, "logical_read_bytes"
    )
    parsed["process_logical_read_operations"] = io_delta(
        io_before, io_after, "logical_read_operations"
    )
    parsed["process_physical_read_bytes"] = io_delta(
        io_before, io_after, "physical_read_bytes"
    )
    parsed["system_physical_read_bytes"] = (
        physical_io["physical_read_bytes"] if physical_io else None
    )
    parsed["logical_read_source"] = (
        io_after["source"] if io_after else None
    )
    parsed["physical_read_source"] = (
        physical_io["source"] if physical_io else None
    )
    return parsed


def median_field(samples, field):
    values = [sample[field] for sample in samples if sample[field] is not None]
    return statistics.median(values) if values else None


def median_route_ranks(samples):
    ranks = {}
    for sample in samples:
        for item in sample.get("expert_route_ranks", []):
            rank = item["rank"]
            aggregate = ranks.setdefault(
                rank,
                {
                    "predictions": [],
                    "matches": [],
                    "demands": [],
                    "demand_queue_time_us": [],
                },
            )
            for field in aggregate:
                aggregate[field].append(item[field])
    return [
        {
            "rank": rank,
            **{
                field: statistics.median(values)
                for field, values in aggregate.items()
            },
        }
        for rank, aggregate in sorted(ranks.items())
    ]


def sweep_child_arguments(cache_mb, report_path):
    arguments = sys.argv[1:]
    child_arguments = []
    index = 0
    has_cache_limit = False
    has_json_output = False
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--expert-cache-sweep-mb":
            index += 1
            while index < len(arguments) and not arguments[index].startswith(
                "--"
            ):
                index += 1
            continue
        if argument == "--expert-cache-mb":
            child_arguments.extend([argument, str(cache_mb)])
            has_cache_limit = True
            index += 2
            continue
        if argument == "--json-output":
            child_arguments.extend([argument, str(report_path)])
            has_json_output = True
            index += 2
            continue
        child_arguments.append(argument)
        index += 1
    if not has_cache_limit:
        child_arguments.extend(["--expert-cache-mb", str(cache_mb)])
    if not has_json_output:
        child_arguments.extend(["--json-output", str(report_path)])
    return [sys.executable, str(Path(__file__).resolve()), *child_arguments]


def run_cache_sweep(arguments):
    reports = []
    with tempfile.TemporaryDirectory(prefix="ncnn-moe-cache-sweep-") as directory:
        for cache_mb in arguments.expert_cache_sweep_mb:
            report_path = Path(directory) / f"cache-{cache_mb}.json"
            command = sweep_child_arguments(cache_mb, report_path)
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
            )
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            if completed.returncode != 0:
                return completed.returncode
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    rows = []
    for report in reports:
        median = report["median"]
        maximum = report["maximum"]
        rows.append(
            {
                "expert_cache_mb": report["expert_cache_mb"],
                "decode_tokens_per_second": median["decode_tokens_per_second"],
                "peak_rss_mib": maximum["peak_rss_mib"],
                "expert_cache_hit_rate": median["expert_cache_hit_rate"],
                "runtime_logical_read_bytes": median[
                    "runtime_logical_read_bytes"
                ],
                "process_logical_read_bytes": median[
                    "process_logical_read_bytes"
                ],
                "process_physical_read_bytes": median[
                    "process_physical_read_bytes"
                ],
                "system_physical_read_bytes": median[
                    "system_physical_read_bytes"
                ],
                "speculative_proposals": median["speculative_proposals"],
                "speculative_draft_tokens": median[
                    "speculative_draft_tokens"
                ],
                "speculative_accepted_tokens": median[
                    "speculative_accepted_tokens"
                ],
                "speculative_draft_ms": median["speculative_draft_ms"],
                "speculative_verify_ms": median["speculative_verify_ms"],
            }
        )

    maximum = {
        "peak_rss_mib": max(
            report["maximum"]["peak_rss_mib"] for report in reports
        ),
        "peak_gpu_mib": max(
            (
                report["maximum"]["peak_gpu_mib"]
                for report in reports
                if report["maximum"]["peak_gpu_mib"] is not None
            ),
            default=None,
        ),
        "peak_gpu_delta_mib": max(
            (
                report["maximum"]["peak_gpu_delta_mib"]
                for report in reports
                if report["maximum"]["peak_gpu_delta_mib"] is not None
            ),
            default=None,
        ),
    }
    try:
        execution_evidence = build_cache_sweep_execution_evidence(
            arguments, reports, maximum
        )
    except (KeyError, ValueError) as error:
        print(f"invalid cache sweep child report: {error}", file=sys.stderr)
        return 1
    report = {
        "schema_version": 3,
        "benchmark": "runtime_expert_cache_sweep",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(Path(arguments.model).resolve()),
        "runner": str(Path(arguments.runner).resolve()),
        "prompt_token_ids": arguments.prompt_token_ids,
        "max_new_tokens": arguments.max_new_tokens,
        "temperature": arguments.temperature,
        "speculative_enabled": not arguments.no_speculative,
        "speculative_confidence": arguments.speculative_confidence,
        "speculative_max_draft": arguments.speculative_max_draft,
        "parallel_speculative": arguments.parallel_speculative,
        "parallel_independent": arguments.parallel_independent,
        "backend": arguments.backend,
        "execution_evidence": execution_evidence,
        "warmup_runs": arguments.warmup,
        "cache_warmup_runs": arguments.cache_warmup_runs,
        "repeats": arguments.repeats,
        "cache_sizes_mb": arguments.expert_cache_sweep_mb,
        "rows": rows,
        "maximum": maximum,
        "reports": reports,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.json_output:
        output_path = Path(arguments.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    print("cache sweep:")
    for row in rows:
        print(
            f"  {row['expert_cache_mb']} MiB: "
            f"{row['decode_tokens_per_second']:.3f} token/s, "
            f"RSS {row['peak_rss_mib']:.1f} MiB, "
            f"hit rate {row['expert_cache_hit_rate']:.1%}, "
            f"Runtime reads {row['runtime_logical_read_bytes']} bytes, "
            f"system physical reads {row['system_physical_read_bytes']} bytes"
        )
    if arguments.json_output:
        print(f"JSON report: {Path(arguments.json_output).resolve()}")
    return 0


def main():
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if arguments.expert_cache_sweep_mb is not None:
        return run_cache_sweep(arguments)

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
    for sample_index, sample in enumerate(samples[1:], start=2):
        candidate_tokens = sample["generated_token_ids"]
        if candidate_tokens == reference_tokens:
            continue
        first_difference = next(
            (
                index
                for index, (expected, actual) in enumerate(
                    zip(reference_tokens, candidate_tokens)
                )
                if expected != actual
            ),
            min(len(reference_tokens), len(candidate_tokens)),
        )
        print(
            "measured runs produced different token sequences: "
            f"run 1 vs run {sample_index}, first difference at "
            f"flattened token {first_difference}",
            file=sys.stderr,
        )
        print("run 1 tokens:", *reference_tokens, file=sys.stderr)
        print(
            f"run {sample_index} tokens:",
            *candidate_tokens,
            file=sys.stderr,
        )
        return 1

    if arguments.require_speculative:
        inactive_runs = [
            index
            for index, sample in enumerate(samples, start=1)
            if not sample["speculative_proposals"]
            or not sample["speculative_draft_tokens"]
        ]
        if inactive_runs:
            print(
                "speculative execution was required but complete proposal "
                f"and draft activity was not reported in measured run(s): "
                f"{inactive_runs}",
                file=sys.stderr,
            )
            return 1

    maximum = {
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
    }
    execution_evidence = build_execution_evidence(
        arguments, samples, maximum
    )

    report = {
        "schema_version": 3,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(Path(arguments.model).resolve()),
        "model_revision": arguments.model_revision or None,
        "routed_expert_format": samples[0]["routed_expert_format"],
        "compiled_artifact": inspect_compiled_artifact(arguments.model),
        "runner": str(Path(arguments.runner).resolve()),
        "command": command,
        "prompt_token_ids": arguments.prompt_token_ids,
        "max_new_tokens": arguments.max_new_tokens,
        "temperature": arguments.temperature,
        "speculative_enabled": not arguments.no_speculative,
        "speculative_confidence": arguments.speculative_confidence,
        "speculative_max_draft": arguments.speculative_max_draft,
        "parallel_speculative": arguments.parallel_speculative,
        "parallel_independent": arguments.parallel_independent,
        "backend": arguments.backend,
        "execution_evidence": execution_evidence,
        "expert_memory": arguments.expert_memory,
        "host_memory_mb": arguments.host_memory_mb,
        "expert_cache_mb": arguments.expert_cache_mb,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_core_count": samples[0][
                "physical_cpu_core_count"
            ],
            "runtime_logical_cpu_count": samples[0][
                "runtime_logical_cpu_count"
            ],
            "runtime_openmp_thread_count": samples[0][
                "runtime_openmp_thread_count"
            ],
            "mxfp4_kernel": samples[0]["mxfp4_kernel"],
            "cpu_isa": samples[0]["cpu_isa"],
            "mxfp4_decode_row_pair_group_size": samples[0][
                "mxfp4_decode_row_pair_group_size"
            ],
            "activation_kernel": samples[0]["activation_kernel"],
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "runtime_backend": samples[0]["runtime_backend"],
            "vulkan_runtime_device_count": samples[0][
                "vulkan_runtime_device_count"
            ],
            "vulkan_heap_budget_mib": samples[0]["vulkan_heap_budget_mib"],
            "gpu_index": arguments.gpu_index,
            "vulkan_device_index": arguments.vulkan_device_index,
            "vulkan_device_indices": arguments.vulkan_device_indices,
            "runtime_vulkan_model_device_index": samples[0][
                "vulkan_model_device_index"
            ],
            "runtime_vulkan_model_devices": samples[0][
                "vulkan_model_devices"
            ],
            "vulkan_layer_placement": samples[0][
                "vulkan_layer_placement"
            ],
            "vulkan_kernel_features": samples[0][
                "vulkan_kernel_features"
            ],
            "gpu": samples[0]["gpu_device"],
        },
        "warmup_runs": arguments.warmup,
        "cache_warmup_runs": arguments.cache_warmup_runs,
        "parallel_sessions": arguments.parallel_sessions,
        "scheduler_expert_threads": arguments.scheduler_expert_threads,
        "scheduler_staging": arguments.scheduler_staging,
        "scheduler_cross_call": arguments.scheduler_cross_call,
        "scheduler_collection_us": (
            arguments.scheduler_collection_us
        ),
        "scheduler_max_micro_batch": (
            arguments.scheduler_max_micro_batch
        ),
        "expert_gpu_cache_mb": arguments.expert_gpu_cache_mb,
        "expert_gpu_victim_cache_mb": (
            arguments.expert_gpu_victim_cache_mb
        ),
        "expert_gpu_victim_reuse_probe": (
            arguments.expert_gpu_victim_reuse_probe
        ),
        "measured_runs": arguments.repeats,
        "median": {
            "load_seconds": median_field(samples, "load_seconds"),
            "generation_seconds": median_field(samples, "generation_seconds"),
            "decode_tokens_per_second": median_field(
                samples, "decode_tokens_per_second"
            ),
            "reported_aggregate_tokens_per_second": median_field(
                samples, "reported_aggregate_tokens_per_second"
            ),
            "attention_ms": median_field(samples, "attention_ms"),
            "router_ms": median_field(samples, "router_ms"),
            "expert_ms": median_field(samples, "expert_ms"),
            "expert_batch_weight_bytes": median_field(
                samples, "expert_batch_weight_bytes"
            ),
            "expert_route_weight_bytes": median_field(
                samples, "expert_route_weight_bytes"
            ),
            "expert_batch_weight_bytes_per_generated_token": median_field(
                samples,
                "expert_batch_weight_bytes_per_generated_token",
            ),
            "required_pcie_gib_per_second_at_20_tps": median_field(
                samples,
                "required_pcie_gib_per_second_at_20_tps",
            ),
            "expert_static_hotsets": samples[-1][
                "expert_static_hotsets"
            ],
            "expert_cache_wait_ms": median_field(
                samples, "expert_cache_wait_ms"
            ),
            "expert_cache_management_ms": median_field(
                samples, "expert_cache_management_ms"
            ),
            "expert_engine_ms": median_field(
                samples, "expert_engine_ms"
            ),
            "expert_compute_ms": median_field(
                samples, "expert_compute_ms"
            ),
            "expert_orchestration_ms": median_field(
                samples, "expert_orchestration_ms"
            ),
            "expert_regroup_ms": median_field(
                samples, "expert_regroup_ms"
            ),
            "expert_combine_ms": median_field(
                samples, "expert_combine_ms"
            ),
            "embedding_ms": median_field(samples, "embedding_ms"),
            "final_norm_ms": median_field(samples, "final_norm_ms"),
            "lm_head_ms": median_field(samples, "lm_head_ms"),
            "speculative_proposals": median_field(
                samples, "speculative_proposals"
            ),
            "speculative_draft_tokens": median_field(
                samples, "speculative_draft_tokens"
            ),
            "speculative_accepted_tokens": median_field(
                samples, "speculative_accepted_tokens"
            ),
            "speculative_context_ms": median_field(
                samples, "speculative_context_ms"
            ),
            "speculative_draft_ms": median_field(
                samples, "speculative_draft_ms"
            ),
            "speculative_verify_ms": median_field(
                samples, "speculative_verify_ms"
            ),
            "vulkan_linear_dispatches": median_field(
                samples, "vulkan_linear_dispatches"
            ),
            "vulkan_compute_submissions": median_field(
                samples, "vulkan_compute_submissions"
            ),
            "vulkan_batch_uploads": median_field(
                samples, "vulkan_batch_uploads"
            ),
            "vulkan_batch_downloads": median_field(
                samples, "vulkan_batch_downloads"
            ),
            "vulkan_submit_wait_ms": median_field(
                samples, "vulkan_submit_wait_ms"
            ),
            "vulkan_auxiliary_uploads": median_field(
                samples, "vulkan_auxiliary_uploads"
            ),
            "vulkan_auxiliary_upload_bytes": median_field(
                samples, "vulkan_auxiliary_upload_bytes"
            ),
            "vulkan_command_buffer_reuses": median_field(
                samples, "vulkan_command_buffer_reuses"
            ),
            "vulkan_attention_qkv_rope_fusions": median_field(
                samples, "vulkan_attention_qkv_rope_fusions"
            ),
            "vulkan_attention_qkv_ring_fusions": median_field(
                samples, "vulkan_attention_qkv_ring_fusions"
            ),
            "vulkan_attention_decode_sdpa_fusions": median_field(
                samples, "vulkan_attention_decode_sdpa_fusions"
            ),
            "vulkan_attention_blocks": median_field(
                samples, "vulkan_attention_blocks"
            ),
            "vulkan_kv_ring_appends": median_field(
                samples, "vulkan_kv_ring_appends"
            ),
            "vulkan_kv_ring_resizes": median_field(
                samples, "vulkan_kv_ring_resizes"
            ),
            "vulkan_kv_ring_wrapped_views": median_field(
                samples, "vulkan_kv_ring_wrapped_views"
            ),
            "scheduler_staged_batches": median_field(
                samples, "scheduler_staged_batches"
            ),
            "scheduler_staged_requests": median_field(
                samples, "scheduler_staged_requests"
            ),
            "scheduler_staging_bypassed_batches": median_field(
                samples, "scheduler_staging_bypassed_batches"
            ),
            "scheduler_logical_expert_batches": median_field(
                samples, "scheduler_logical_expert_batches"
            ),
            "scheduler_physical_expert_batches": median_field(
                samples, "scheduler_physical_expert_batches"
            ),
            "scheduler_coalesced_expert_routes": median_field(
                samples, "scheduler_coalesced_expert_routes"
            ),
            "scheduler_max_coalesced_expert_batch_size": median_field(
                samples, "scheduler_max_coalesced_expert_batch_size"
            ),
            "scheduler_adaptive_staged_decisions": median_field(
                samples, "scheduler_adaptive_staged_decisions"
            ),
            "scheduler_adaptive_independent_decisions": median_field(
                samples, "scheduler_adaptive_independent_decisions"
            ),
            "scheduler_adaptive_probe_decisions": median_field(
                samples, "scheduler_adaptive_probe_decisions"
            ),
            "scheduler_adaptive_policy_switches": median_field(
                samples, "scheduler_adaptive_policy_switches"
            ),
            "scheduler_adaptive_staged_ms_per_request": median_field(
                samples, "scheduler_adaptive_staged_ms_per_request"
            ),
            "scheduler_adaptive_independent_ms_per_request": median_field(
                samples, "scheduler_adaptive_independent_ms_per_request"
            ),
            "scheduler_adaptive_resident_decisions": median_field(
                samples, "scheduler_adaptive_resident_decisions"
            ),
            "scheduler_adaptive_mixed_decisions": median_field(
                samples, "scheduler_adaptive_mixed_decisions"
            ),
            "scheduler_adaptive_storage_decisions": median_field(
                samples, "scheduler_adaptive_storage_decisions"
            ),
            "scheduler_adaptive_resident_observations": median_field(
                samples, "scheduler_adaptive_resident_observations"
            ),
            "scheduler_adaptive_mixed_observations": median_field(
                samples, "scheduler_adaptive_mixed_observations"
            ),
            "scheduler_adaptive_storage_observations": median_field(
                samples, "scheduler_adaptive_storage_observations"
            ),
            "scheduler_adaptive_phase_changes": median_field(
                samples, "scheduler_adaptive_phase_changes"
            ),
            "scheduler_adaptive_noisy_switch_rejections": median_field(
                samples, "scheduler_adaptive_noisy_switch_rejections"
            ),
            "scheduler_cross_call_collected_batches": median_field(
                samples, "scheduler_cross_call_collected_batches"
            ),
            "scheduler_cross_call_collected_requests": median_field(
                samples, "scheduler_cross_call_collected_requests"
            ),
            "scheduler_cross_call_collection_probes": median_field(
                samples, "scheduler_cross_call_collection_probes"
            ),
            "scheduler_cross_call_collection_timeouts": median_field(
                samples, "scheduler_cross_call_collection_timeouts"
            ),
            "scheduler_cross_call_collection_bypasses": median_field(
                samples, "scheduler_cross_call_collection_bypasses"
            ),
            "scheduler_cross_call_collection_wait_us": median_field(
                samples, "scheduler_cross_call_collection_wait_us"
            ),
            "scheduler_cross_call_max_batch_size": median_field(
                samples, "scheduler_cross_call_max_batch_size"
            ),
            "scheduler_cross_call_max_pending": median_field(
                samples, "scheduler_cross_call_max_pending"
            ),
            "expert_route_predictions": median_field(
                samples, "expert_route_predictions"
            ),
            "expert_route_prediction_matches": median_field(
                samples, "expert_route_prediction_matches"
            ),
            "expert_route_prediction_cache_hits": median_field(
                samples, "expert_route_prediction_cache_hits"
            ),
            "expert_route_prediction_cache_misses": median_field(
                samples, "expert_route_prediction_cache_misses"
            ),
            "expert_route_prediction_ms": median_field(
                samples, "expert_route_prediction_ms"
            ),
            "expert_route_prediction_wait_ms": median_field(
                samples, "expert_route_prediction_wait_ms"
            ),
            "expert_route_prediction_async_submissions": median_field(
                samples, "expert_route_prediction_async_submissions"
            ),
            "expert_route_prediction_async_completions": median_field(
                samples, "expert_route_prediction_async_completions"
            ),
            "expert_route_prediction_async_fallbacks": median_field(
                samples, "expert_route_prediction_async_fallbacks"
            ),
            "expert_route_ranks": median_route_ranks(samples),
            "expert_cache_hit_rate": median_field(
                samples, "expert_cache_hit_rate"
            ),
            "expert_cache_bytes_read": median_field(
                samples, "expert_cache_bytes_read"
            ),
            "runtime_logical_read_bytes": median_field(
                samples, "runtime_logical_read_bytes"
            ),
            "process_logical_read_bytes": median_field(
                samples, "process_logical_read_bytes"
            ),
            "process_logical_read_operations": median_field(
                samples, "process_logical_read_operations"
            ),
            "process_physical_read_bytes": median_field(
                samples, "process_physical_read_bytes"
            ),
            "system_physical_read_bytes": median_field(
                samples, "system_physical_read_bytes"
            ),
            "expert_cache_queued_reads": median_field(
                samples, "expert_cache_queued_reads"
            ),
            "expert_cache_speculative_reads": median_field(
                samples, "expert_cache_speculative_reads"
            ),
            "expert_cache_cancelled_speculative_reads": median_field(
                samples, "expert_cache_cancelled_speculative_reads"
            ),
            "expert_cache_dropped_speculative_admissions": median_field(
                samples, "expert_cache_dropped_speculative_admissions"
            ),
            "expert_cache_unused_speculative_reads": median_field(
                samples, "expert_cache_unused_speculative_reads"
            ),
            "expert_cache_short_term_reloads": median_field(
                samples, "expert_cache_short_term_reloads"
            ),
            "expert_cache_arc_recent_bytes": median_field(
                samples, "expert_cache_arc_recent_bytes"
            ),
            "expert_cache_arc_frequent_bytes": median_field(
                samples, "expert_cache_arc_frequent_bytes"
            ),
            "expert_cache_arc_recent_target_bytes": median_field(
                samples, "expert_cache_arc_recent_target_bytes"
            ),
            "expert_cache_arc_recent_ghost_bytes": median_field(
                samples, "expert_cache_arc_recent_ghost_bytes"
            ),
            "expert_cache_arc_frequent_ghost_bytes": median_field(
                samples, "expert_cache_arc_frequent_ghost_bytes"
            ),
            "expert_cache_arc_recent_ghost_hits": median_field(
                samples, "expert_cache_arc_recent_ghost_hits"
            ),
            "expert_cache_arc_frequent_ghost_hits": median_field(
                samples, "expert_cache_arc_frequent_ghost_hits"
            ),
            "expert_cache_mapped_ranges": median_field(
                samples, "expert_cache_mapped_ranges"
            ),
            "expert_cache_mapped_bytes": median_field(
                samples, "expert_cache_mapped_bytes"
            ),
            "expert_cache_read_policy": samples[-1][
                "expert_cache_read_policy"
            ],
            "expert_cache_direct_read_ranges": median_field(
                samples, "expert_cache_direct_read_ranges"
            ),
            "expert_cache_direct_read_bytes": median_field(
                samples, "expert_cache_direct_read_bytes"
            ),
            "expert_cache_direct_read_fallbacks": median_field(
                samples, "expert_cache_direct_read_fallbacks"
            ),
            "expert_cache_buffered_read_ranges": median_field(
                samples, "expert_cache_buffered_read_ranges"
            ),
            "expert_cache_buffered_read_bytes": median_field(
                samples, "expert_cache_buffered_read_bytes"
            ),
            "expert_cache_coalesced_read_batches": median_field(
                samples, "expert_cache_coalesced_read_batches"
            ),
            "expert_cache_coalesced_experts": median_field(
                samples, "expert_cache_coalesced_experts"
            ),
            "expert_cache_coalesced_read_ranges_saved": median_field(
                samples, "expert_cache_coalesced_read_ranges_saved"
            ),
            "expert_gpu_cache_hits": median_field(
                samples, "expert_gpu_cache_hits"
            ),
            "expert_gpu_cache_misses": median_field(
                samples, "expert_gpu_cache_misses"
            ),
            "expert_gpu_cache_admissions": median_field(
                samples, "expert_gpu_cache_admissions"
            ),
            "expert_gpu_cache_stores": median_field(
                samples, "expert_gpu_cache_stores"
            ),
            "expert_gpu_cache_evictions": median_field(
                samples, "expert_gpu_cache_evictions"
            ),
            "expert_gpu_cache_dropped_admissions": median_field(
                samples, "expert_gpu_cache_dropped_admissions"
            ),
            "expert_gpu_cache_bytes_uploaded": median_field(
                samples, "expert_gpu_cache_bytes_uploaded"
            ),
            "expert_gpu_cache_resident_bytes": median_field(
                samples, "expert_gpu_cache_resident_bytes"
            ),
            "expert_gpu_cache_pending_bytes": median_field(
                samples, "expert_gpu_cache_pending_bytes"
            ),
            "expert_gpu_victim_cache_hits": median_field(
                samples, "expert_gpu_victim_cache_hits"
            ),
            "expert_gpu_victim_cache_misses": median_field(
                samples, "expert_gpu_victim_cache_misses"
            ),
            "expert_gpu_victim_cache_admissions": median_field(
                samples, "expert_gpu_victim_cache_admissions"
            ),
            "expert_gpu_victim_cache_filtered_admissions": median_field(
                samples, "expert_gpu_victim_cache_filtered_admissions"
            ),
            "expert_gpu_victim_cache_reused_admissions": median_field(
                samples, "expert_gpu_victim_cache_reused_admissions"
            ),
            "expert_gpu_victim_cache_probe_admissions": median_field(
                samples, "expert_gpu_victim_cache_probe_admissions"
            ),
            "expert_gpu_victim_cache_stores": median_field(
                samples, "expert_gpu_victim_cache_stores"
            ),
            "expert_gpu_victim_cache_evictions": median_field(
                samples, "expert_gpu_victim_cache_evictions"
            ),
            "expert_gpu_victim_cache_dropped_admissions": median_field(
                samples, "expert_gpu_victim_cache_dropped_admissions"
            ),
            "expert_gpu_victim_cache_restore_failures": median_field(
                samples, "expert_gpu_victim_cache_restore_failures"
            ),
            "expert_gpu_victim_cache_bytes_uploaded": median_field(
                samples, "expert_gpu_victim_cache_bytes_uploaded"
            ),
            "expert_gpu_victim_cache_bytes_downloaded": median_field(
                samples, "expert_gpu_victim_cache_bytes_downloaded"
            ),
            "expert_gpu_victim_cache_restore_ms": median_field(
                samples, "expert_gpu_victim_cache_restore_ms"
            ),
            "expert_gpu_victim_cache_mapped_stores": median_field(
                samples, "expert_gpu_victim_cache_mapped_stores"
            ),
            "expert_gpu_victim_cache_mapped_restores": median_field(
                samples, "expert_gpu_victim_cache_mapped_restores"
            ),
            "expert_gpu_victim_cache_resident_bytes": median_field(
                samples, "expert_gpu_victim_cache_resident_bytes"
            ),
            "expert_gpu_victim_cache_pending_bytes": median_field(
                samples, "expert_gpu_victim_cache_pending_bytes"
            ),
            "expert_gpu_executions": median_field(
                samples, "expert_gpu_executions"
            ),
            "expert_gpu_execution_failures": median_field(
                samples, "expert_gpu_execution_failures"
            ),
            "expert_gpu_cpu_preferred": median_field(
                samples, "expert_gpu_cpu_preferred"
            ),
            "expert_gpu_execution_ms": median_field(
                samples, "expert_gpu_execution_ms"
            ),
            "expert_gpu_device_source_hits": median_field(
                samples, "expert_gpu_device_source_hits"
            ),
            "expert_gpu_device_source_misses": median_field(
                samples, "expert_gpu_device_source_misses"
            ),
            "expert_gpu_device_source_executions": median_field(
                samples, "expert_gpu_device_source_executions"
            ),
            "expert_gpu_device_source_execution_failures": median_field(
                samples, "expert_gpu_device_source_execution_failures"
            ),
            "expert_gpu_arc_recent_bytes": median_field(
                samples, "expert_gpu_arc_recent_bytes"
            ),
            "expert_gpu_arc_frequent_bytes": median_field(
                samples, "expert_gpu_arc_frequent_bytes"
            ),
            "expert_gpu_arc_recent_target_bytes": median_field(
                samples, "expert_gpu_arc_recent_target_bytes"
            ),
            "expert_gpu_arc_recent_ghost_bytes": median_field(
                samples, "expert_gpu_arc_recent_ghost_bytes"
            ),
            "expert_gpu_arc_frequent_ghost_bytes": median_field(
                samples, "expert_gpu_arc_frequent_ghost_bytes"
            ),
        },
        "maximum": maximum,
        "read_traffic": {
            "runtime_logical": "Expert cache bytes read",
            "process_logical": samples[0]["logical_read_source"],
            "process_physical": (
                "Linux /proc/<pid>/io read_bytes"
                if samples[0]["process_physical_read_bytes"] is not None
                else None
            ),
            "system_physical": samples[0]["physical_read_source"],
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
            "median GPU Expert execution cache: "
            f"{median['expert_gpu_cache_hits']:.0f} hit(s), "
            f"{median['expert_gpu_executions']:.0f} execution(s), "
            f"{median['expert_gpu_cpu_preferred']:.0f} CPU-preferred"
        )
    if median["expert_gpu_victim_cache_hits"] is not None:
        print(
            "median GPU Expert victim cache: "
            f"{median['expert_gpu_victim_cache_hits']:.0f} hit(s), "
            f"{median['expert_gpu_victim_cache_misses']:.0f} miss(es), "
            f"{median['expert_gpu_victim_cache_restore_ms']:.3f} ms restoring"
        )
    if median["expert_gpu_device_source_hits"] is not None:
        print(
            "median GPU Expert device source: "
            f"{median['expert_gpu_device_source_hits']:.0f} hit(s), "
            f"{median['expert_gpu_device_source_executions']:.0f} execution(s), "
            f"{median['expert_gpu_device_source_execution_failures']:.0f} failure(s)"
        )
    print(f"peak RSS: {maximum['peak_rss_mib']:.1f} MiB")
    if maximum["peak_gpu_mib"] is not None:
        print(
            f"peak NVIDIA memory: {maximum['peak_gpu_mib']} MiB "
            f"(+{maximum['peak_gpu_delta_mib']} MiB)"
        )
    if arguments.backend == "cpu":
        print(
            "CPU-only execution: "
            f"{'verified' if execution_evidence['cpu_only_execution_verified'] else 'not verified'}; "
            f"Vulkan context initialized: "
            f"{execution_evidence['vulkan_context_initialized']}; "
            f"GPU compute: {execution_evidence['gpu_compute_detected']}; "
            f"GPU transfers: {execution_evidence['gpu_transfer_detected']}"
        )
        print(
            "GPU memory observation: system-wide nvidia-smi total; "
            f"attributed to process: "
            f"{execution_evidence['gpu_memory_observation']['attributed_to_process']}"
        )
    if median["runtime_logical_read_bytes"] is not None:
        print(
            "median Runtime logical reads: "
            f"{median['runtime_logical_read_bytes']} bytes"
        )
    if median["process_logical_read_bytes"] is not None:
        print(
            "median process logical reads: "
            f"{median['process_logical_read_bytes']} bytes "
            f"({median['process_logical_read_operations']} operations)"
        )
    if median["process_physical_read_bytes"] is not None:
        print(
            "median process physical reads: "
            f"{median['process_physical_read_bytes']} bytes"
        )
    if median["system_physical_read_bytes"] is not None:
        print(
            "median system physical disk reads: "
            f"{median['system_physical_read_bytes']} bytes"
        )
    print("generated token ids:", *reference_tokens)
    if median["speculative_proposals"] is not None:
        print(
            "median speculative activity: "
            f"{median['speculative_proposals']} proposal(s), "
            f"{median['speculative_draft_tokens']} draft token(s), "
            f"{median['speculative_accepted_tokens']} accepted token(s)"
        )
    if arguments.json_output:
        print(f"JSON report: {Path(arguments.json_output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
