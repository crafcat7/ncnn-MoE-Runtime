#!/usr/bin/env python3
"""Pack llama.cpp-compatible Q2_K..Q8_K Expert tensors into a sidecar.

The source tensor must be U8 and have one contiguous slice per Expert.  The
logical source layout is normally [expert_count, rows, block_count,
block_bytes], although any shape with the same byte count is accepted.  The
sidecar stores one tensor per selected Expert using the names consumed by
SafetensorsArchive.load_qnk_expert().  It preserves the original Qn_K bytes;
the runtime's CPU packer performs the lossless 8-row tile reorder lazily.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


PACKED_PREFIX = "__ncnn_moe_packed__."
DEFAULT_OUTPUT = "ncnn-moe-packed-qnk.safetensors"
COPY_CHUNK_BYTES = 16 * 1024 * 1024
BLOCK_BYTES = {
    "q2_k": 84,
    "q3_k": 110,
    "q4_k": 144,
    "q5_k": 176,
    "q6_k": 210,
    "q8_k": 292,
}


@dataclass(frozen=True)
class TensorRange:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    byte_count: int


@dataclass(frozen=True)
class CopyRange:
    name: str
    source: TensorRange
    source_offset: int
    byte_count: int
    shape: tuple[int, ...]
    dtype: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack Qn_K Expert U8 tensors into a Safetensors sidecar")
    parser.add_argument("model", type=Path, help="Model directory")
    parser.add_argument("--tensor", action="append", required=True, help="Source Expert tensor name; repeat for gate/up/down")
    parser.add_argument("--dtype", choices=sorted(BLOCK_BYTES), required=True, help="Qn_K encoding, for example q5_k")
    parser.add_argument("--rows", type=int, required=True, help="Logical rows in each Expert matrix")
    parser.add_argument("--columns", type=int, required=True, help="Logical columns in each Expert matrix; must be divisible by 256")
    parser.add_argument("--expert-count", type=int, required=True, help="Expert count in each source tensor")
    parser.add_argument("--experts", help="Optional Expert IDs or inclusive ranges, for example 0,3,8-15")
    parser.add_argument("--output", type=Path, help=f"Output path (default: MODEL/{DEFAULT_OUTPUT})")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing sidecar")
    return parser.parse_args()


def load_safetensor_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated Safetensors header length: {path}")
        (length,) = struct.unpack("<Q", length_bytes)
        if length == 0 or length > 128 * 1024 * 1024:
            raise ValueError(f"invalid Safetensors header length: {path}")
        header_bytes = stream.read(length)
        if len(header_bytes) != length:
            raise ValueError(f"truncated Safetensors header: {path}")
    parsed = json.loads(header_bytes)
    if not isinstance(parsed, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    parsed["__ncnn_data_start__"] = 8 + length
    return parsed


def scan_tensors(model: Path, output: Path) -> dict[str, TensorRange]:
    tensors: dict[str, TensorRange] = {}
    for path in sorted(model.glob("*.safetensors")):
        if path.resolve() == output.resolve():
            continue
        header = load_safetensor_header(path)
        data_start = int(header.pop("__ncnn_data_start__"))
        file_bytes = path.stat().st_size
        for name, raw_info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(raw_info, dict):
                raise ValueError(f"invalid tensor metadata for {name}: {path}")
            dtype = raw_info.get("dtype")
            shape = raw_info.get("shape")
            offsets = raw_info.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"invalid tensor metadata for {name}: {path}")
            begin = data_start + offsets[0]
            end = data_start + offsets[1]
            if end > file_bytes:
                raise ValueError(f"tensor range exceeds shard for {name}: {path}")
            if name in tensors:
                raise ValueError(f"duplicate tensor name: {name}")
            tensors[name] = TensorRange(path, dtype, tuple(shape), begin, end - begin)
    if not tensors:
        raise ValueError(f"no Safetensors tensors found in {model}")
    return tensors


def parse_selection(value: str | None, maximum: int) -> list[int]:
    if value is None:
        return list(range(maximum))
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--experts contains an empty item")
        if "-" in item:
            begin_text, end_text = item.split("-", 1)
            begin, end = int(begin_text), int(end_text)
            if end < begin:
                raise ValueError(f"invalid Expert range: {item}")
            values = range(begin, end + 1)
        else:
            values = (int(item),)
        for expert in values:
            if expert < 0 or expert >= maximum:
                raise ValueError(f"Expert index {expert} is outside [0, {maximum})")
            selected.add(expert)
    if not selected:
        raise ValueError("--experts must select at least one Expert")
    return sorted(selected)


def build_copy_plan(
    tensors: dict[str, TensorRange],
    names: list[str],
    dtype: str,
    rows: int,
    columns: int,
    expert_count: int,
    expert_ids: list[int],
) -> list[CopyRange]:
    block_bytes = BLOCK_BYTES[dtype]
    if rows <= 0 or columns <= 0 or columns % 256 != 0:
        raise ValueError("--rows/--columns must be positive and columns must be divisible by 256")
    blocks = columns // 256
    expert_bytes = rows * blocks * block_bytes
    if expert_bytes <= 0:
        raise ValueError("Qn_K Expert byte count is invalid")
    plan: list[CopyRange] = []
    for name in names:
        source = tensors.get(name)
        if source is None:
            raise ValueError(f"missing Qn_K tensor: {name}")
        if source.dtype != "U8" or source.byte_count != expert_count * expert_bytes:
            raise ValueError(
                f"{name} must be U8 with {expert_count * expert_bytes} bytes; "
                f"found dtype={source.dtype!r}, bytes={source.byte_count}"
            )
        for expert in expert_ids:
            plan.append(
                CopyRange(
                    name=f"{PACKED_PREFIX}{expert}.{name}",
                    source=source,
                    source_offset=source.offset + expert * expert_bytes,
                    byte_count=expert_bytes,
                    shape=(rows, blocks, block_bytes),
                    dtype="U8",
                )
            )
    return plan


def encode_header(plan: list[CopyRange], dtype: str, rows: int, columns: int) -> bytes:
    header: dict[str, object] = {
        "__metadata__": {
            "format": "ncnn-moe-packed-qnk-v1",
            "dtype": dtype,
            "qk_k": 256,
            "rows": str(rows),
            "columns": str(columns),
        }
    }
    offset = 0
    for item in plan:
        header[item.name] = {
            "dtype": item.dtype,
            "shape": list(item.shape),
            "data_offsets": [offset, offset + item.byte_count],
        }
        offset += item.byte_count
    encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return encoded + b" " * ((-len(encoded)) % 8)


def copy_exact(source: BinaryIO, destination: BinaryIO, offset: int, byte_count: int) -> None:
    source.seek(offset)
    remaining = byte_count
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise ValueError("source shard ended during Qn_K packing")
        destination.write(chunk)
        remaining -= len(chunk)


def write_sidecar(output: Path, plan: list[CopyRange], header: bytes, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    handles: dict[Path, BinaryIO] = {}
    expected_data_bytes = sum(item.byte_count for item in plan)
    try:
        with temporary.open("wb") as destination:
            destination.write(struct.pack("<Q", len(header)))
            destination.write(header)
            for item in plan:
                source = handles.get(item.source.path)
                if source is None:
                    source = item.source.path.open("rb")
                    handles[item.source.path] = source
                copy_exact(source, destination, item.source_offset, item.byte_count)
            destination.flush()
            os.fsync(destination.fileno())
        expected_file_bytes = 8 + len(header) + expected_data_bytes
        if temporary.stat().st_size != expected_file_bytes:
            raise ValueError("packed sidecar size mismatch")
        os.replace(temporary, output)
    finally:
        for handle in handles.values():
            handle.close()
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    arguments = parse_arguments()
    model = arguments.model.resolve()
    output = arguments.output.resolve() if arguments.output else model / DEFAULT_OUTPUT
    try:
        if not model.is_dir():
            raise ValueError(f"model directory does not exist: {model}")
        expert_ids = parse_selection(arguments.experts, arguments.expert_count)
        tensors = scan_tensors(model, output)
        plan = build_copy_plan(
            tensors,
            arguments.tensor,
            arguments.dtype,
            arguments.rows,
            arguments.columns,
            arguments.expert_count,
            expert_ids,
        )
        header = encode_header(plan, arguments.dtype, arguments.rows, arguments.columns)
        packed_bytes = sum(item.byte_count for item in plan)
        print(f"packing {len(arguments.tensor) * len(expert_ids)} Qn_K Expert slices ({packed_bytes / (1024 ** 2):.2f} MiB)", flush=True)
        write_sidecar(output, plan, header, arguments.overwrite)
        print(f"packed Qn_K Expert sidecar: {output}", flush=True)
        return 0
    except (FileExistsError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
