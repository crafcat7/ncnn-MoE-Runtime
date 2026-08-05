"""Persistent configuration, tuning profiles, and conversation records."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


def state_root() -> Path:
    """Return the project-local state root unless explicitly overridden."""
    override = os.environ.get("NCNN_MOE_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "CMakeLists.txt").is_file():
        return source_root / ".ncnn-moe"
    return Path.cwd().resolve() / ".ncnn-moe"


def now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return value[:80] or "session"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def hardware_fingerprint(ready: dict[str, Any]) -> str:
    capabilities = ready.get("capabilities", {})
    stable = {
        "physical_memory_bytes": capabilities.get("physical_memory_bytes"),
        "logical_cpu_count": capabilities.get("logical_cpu_count"),
        "physical_cpu_core_count": capabilities.get("physical_cpu_core_count"),
        "cpu_isa": capabilities.get("cpu_isa"),
        "vulkan_devices": [
            {
                "vendor_id": device.get("vendor_id"),
                "device_id": device.get("device_id"),
                "name": device.get("name"),
                "heap_budget_bytes": device.get("heap_budget_bytes"),
            }
            for device in capabilities.get("vulkan_devices", [])
        ],
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or state_root()).resolve()
        self.sessions_dir = self.root / "sessions"
        self.config_path = self.root / "config.json"
        self.profiles_path = self.root / "tuning_profiles.json"

    def user_config(self) -> dict[str, Any]:
        value = read_json(self.config_path, {})
        return value if isinstance(value, dict) else {}

    def save_user_config(self, value: dict[str, Any]) -> None:
        write_json(self.config_path, value)

    def profiles(self) -> dict[str, Any]:
        value = read_json(self.profiles_path, {})
        return value if isinstance(value, dict) else {}

    def save_profile(self, key: str, value: dict[str, Any]) -> None:
        profiles = self.profiles()
        profiles[key] = value
        write_json(self.profiles_path, profiles)

    def profile(self, key: str) -> dict[str, Any] | None:
        value = self.profiles().get(key)
        return value if isinstance(value, dict) else None

    def new_id(self) -> str:
        return _datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{safe_name(session_id)}.json"

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        value = read_json(path, None)
        if not isinstance(value, dict):
            return None
        return value

    def save(self, record: dict[str, Any]) -> None:
        session_id = str(record.get("id", ""))
        if not session_id:
            raise ValueError("session record requires id")
        record = dict(record)
        record["updated_at"] = now_iso()
        write_json(self.path_for(session_id), record)

    def delete(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def rename(self, session_id: str, title: str) -> dict[str, Any]:
        record = self.load(session_id)
        if record is None:
            raise ValueError(f"session does not exist: {session_id}")
        record["title"] = title.strip() or session_id
        self.save(record)
        return record

    def list(self) -> list[dict[str, Any]]:
        if not self.sessions_dir.is_dir():
            return []
        records = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            value = read_json(path, None)
            if isinstance(value, dict) and value.get("id"):
                records.append(value)
        return records


def profile_key(
    *,
    hardware: str,
    model: str,
    context_tokens: int,
    session_count: int,
) -> str:
    return f"{hardware}:{model}:{context_tokens}:{session_count}"


def runtime_args_from_settings(settings: dict[str, Any]) -> list[str]:
    args: list[str] = []
    backend = settings.get("backend")
    if backend and backend != "auto":
        args.append(f"--{backend}")
    for key, option in (
        ("host_memory_mb", "--host-memory-mb"),
        ("expert_cache_mb", "--expert-cache-mb"),
        ("expert_gpu_cache_mb", "--expert-gpu-cache-mb"),
        ("expert_gpu_victim_cache_mb", "--expert-gpu-victim-cache-mb"),
        ("expert_io_workers", "--expert-io-workers"),
        ("expert_memory", "--expert-memory"),
        ("expected_concurrency", "--expected-concurrency"),
        ("optimization_flags", "--optimization-flags"),
    ):
        value = settings.get(key)
        if value is not None and value != 0:
            args.extend([option, str(value)])
    if settings.get("vulkan_device") is not None:
        args.extend(["--vulkan-device", str(settings["vulkan_device"])])
    if settings.get("vulkan_devices"):
        devices = settings["vulkan_devices"]
        if isinstance(devices, (list, tuple)):
            devices = ",".join(str(device) for device in devices)
        args.extend(["--vulkan-devices", str(devices)])
    for key, option in (
        ("mmap_experts", "--mmap-experts"),
        ("direct_expert_io", "--direct-expert-io"),
        ("buffered_expert_io", "--buffered-expert-io"),
        ("disable_gpu_victim_execution", "--disable-gpu-victim-execution"),
        ("router_prediction", "--router-prediction"),
        ("async_router_prediction", "--async-router-prediction"),
        ("forward_aware_cache", "--forward-aware-cache"),
        ("rank_adaptive_prefetch", "--rank-adaptive-prefetch"),
        ("cross_expert_read_coalescing", "--cross-expert-read-coalescing"),
        ("release_vulkan_dense_host", "--release-vulkan-dense-host"),
    ):
        if settings.get(key):
            args.append(option)
    return args


def merge_runtime_settings(
    *,
    cli: dict[str, Any],
    session: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    user: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in (user or {}, profile or {}, session or {}, cli):
        for key, value in source.items():
            if value is not None:
                result[key] = value
    result.setdefault("backend", "auto")
    return result
