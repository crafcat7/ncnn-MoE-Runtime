#!/usr/bin/env python3
"""Create a read-optimized MXFP4 Expert sidecar.

The sidecar is a normal Safetensors file. It keeps each Expert's gate/up and
down blocks/scales adjacent so the runtime can satisfy a cache miss with one
large range read instead of four random range reads. Original model shards are
never modified.
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
DEFAULT_OUTPUT = "ncnn-moe-packed-experts.safetensors"
COPY_CHUNK_BYTES = 16 * 1024 * 1024


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack GPT-OSS MXFP4 Expert slices into a read-optimized "
            "Safetensors sidecar."
        )
    )
    parser.add_argument("model", type=Path, help="Model directory")
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output path (default: MODEL/{DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sidecar",
    )
    return parser.parse_args()


def load_safetensor_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        header_length_bytes = stream.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"truncated Safetensors header length: {path}")
        (header_length,) = struct.unpack("<Q", header_length_bytes)
        if header_length == 0 or header_length > 128 * 1024 * 1024:
            raise ValueError(f"invalid Safetensors header length: {path}")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated Safetensors header: {path}")
    try:
        parsed = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Safetensors JSON header: {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    parsed["__ncnn_data_start__"] = 8 + header_length
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
            tensors[name] = TensorRange(
                path=path,
                dtype=dtype,
                shape=tuple(shape),
                offset=begin,
                byte_count=end - begin,
            )
    if not tensors:
        raise ValueError(f"no Safetensors tensors found in {model}")
    return tensors


def require_u8_expert_tensor(
    tensors: dict[str, TensorRange],
    name: str,
    expert_count: int,
) -> TensorRange:
    tensor = tensors.get(name)
    if tensor is None:
        raise ValueError(f"missing Expert tensor: {name}")
    if (
        tensor.dtype != "U8"
        or not tensor.shape
        or tensor.shape[0] != expert_count
        or tensor.byte_count % expert_count != 0
    ):
        raise ValueError(f"invalid MXFP4 Expert tensor: {name}")
    return tensor


def build_gpt_oss_copy_plan(
    tensors: dict[str, TensorRange],
    layer_count: int,
    expert_count: int,
) -> list[CopyRange]:
    plan: list[CopyRange] = []
    for layer_id in range(layer_count):
        prefix = f"model.layers.{layer_id}.mlp.experts."
        names = (
            prefix + "gate_up_proj_blocks",
            prefix + "gate_up_proj_scales",
            prefix + "down_proj_blocks",
            prefix + "down_proj_scales",
        )
        sources = tuple(
            require_u8_expert_tensor(tensors, name, expert_count)
            for name in names
        )
        for expert_id in range(expert_count):
            for name, source in zip(names, sources, strict=True):
                slice_bytes = source.byte_count // expert_count
                plan.append(
                    CopyRange(
                        name=f"{PACKED_PREFIX}{expert_id}.{name}",
                        source=source,
                        source_offset=source.offset + expert_id * slice_bytes,
                        byte_count=slice_bytes,
                        shape=source.shape[1:],
                    )
                )
    return plan


def encode_header(plan: list[CopyRange], model_type: str) -> bytes:
    header: dict[str, object] = {
        "__metadata__": {
            "format": "ncnn-moe-packed-experts-v1",
            "model_type": model_type,
        }
    }
    offset = 0
    for item in plan:
        header[item.name] = {
            "dtype": "U8",
            "shape": list(item.shape),
            "data_offsets": [offset, offset + item.byte_count],
        }
        offset += item.byte_count
    encoded = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + b" " * padding


def copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    offset: int,
    byte_count: int,
) -> None:
    source.seek(offset)
    remaining = byte_count
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise ValueError("source shard ended during Expert packing")
        destination.write(chunk)
        remaining -= len(chunk)


def write_sidecar(
    output: Path,
    plan: list[CopyRange],
    header: bytes,
    overwrite: bool,
    layer_count: int,
    expert_count: int,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists (pass --overwrite to replace it): {output}"
        )
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
            ranges_per_layer = expert_count * 4
            for index, item in enumerate(plan):
                source = handles.get(item.source.path)
                if source is None:
                    source = item.source.path.open("rb")
                    handles[item.source.path] = source
                copy_exact(
                    source,
                    destination,
                    item.source_offset,
                    item.byte_count,
                )
                if (index + 1) % ranges_per_layer == 0:
                    layer = (index + 1) // ranges_per_layer
                    print(
                        f"packed layer {layer}/{layer_count}",
                        flush=True,
                    )
            destination.flush()
            os.fsync(destination.fileno())
        expected_file_bytes = 8 + len(header) + expected_data_bytes
        actual_file_bytes = temporary.stat().st_size
        if actual_file_bytes != expected_file_bytes:
            raise ValueError(
                "packed sidecar size mismatch: "
                f"expected {expected_file_bytes}, got {actual_file_bytes}"
            )
        os.replace(temporary, output)
    finally:
        for handle in handles.values():
            handle.close()
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    arguments = parse_arguments()
    model = arguments.model.resolve()
    output = (
        arguments.output.resolve()
        if arguments.output
        else model / DEFAULT_OUTPUT
    )
    try:
        if not model.is_dir():
            raise ValueError(f"model directory does not exist: {model}")
        config_path = model / "config.json"
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        model_type = config.get("model_type")
        if model_type != "gpt_oss":
            raise ValueError(
                "the current packer supports GPT-OSS source layouts; "
                f"found model_type={model_type!r}"
            )
        layer_count = int(config["num_hidden_layers"])
        expert_count = int(config["num_local_experts"])
        if layer_count <= 0 or expert_count <= 0:
            raise ValueError("invalid GPT-OSS layer or Expert count")

        tensors = scan_tensors(model, output)
        plan = build_gpt_oss_copy_plan(
            tensors,
            layer_count,
            expert_count,
        )
        header = encode_header(plan, model_type)
        packed_bytes = sum(item.byte_count for item in plan)
        print(
            f"packing {layer_count * expert_count} Experts "
            f"({packed_bytes / (1024 ** 3):.2f} GiB) into {output}",
            flush=True,
        )
        write_sidecar(
            output,
            plan,
            header,
            arguments.overwrite,
            layer_count,
            expert_count,
        )
        print(f"packed Expert sidecar: {output}", flush=True)
        return 0
    except (
        FileExistsError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
