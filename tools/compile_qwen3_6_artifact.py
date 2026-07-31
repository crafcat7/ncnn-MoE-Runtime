#!/usr/bin/env python3
"""Compile Qwen3.6 routed BF16 Experts into an optional MXFP4 sidecar."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


ARTIFACT_NAME = "ncnn-moe-qwen3.6-mxfp4.safetensors"
ARTIFACT_PREFIX = "__ncnn_moe_qwen3_6_mxfp4__"
ARTIFACT_FORMAT = "ncnn-moe-qwen3.6-mxfp4-v3"
MAX_HEADER_BYTES = 128 * 1024 * 1024
FNV1A64_OFFSET = 14695981039346656037
FNV1A64_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1
E2M1_THRESHOLDS = np.asarray(
    (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
    dtype=np.float32,
)


@dataclass(frozen=True)
class TensorRange:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    byte_count: int


@dataclass(frozen=True)
class ArtifactTensor:
    name: str
    shape: tuple[int, ...]
    byte_count: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the routed Experts from an official Qwen3.6-35B-A3B "
            "BF16 checkpoint into an optional OCP MXFP4 Safetensors sidecar."
        )
    )
    parser.add_argument("model", type=Path, help="Official model directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing compiled artifact",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) // 2)),
        help="Parallel Expert quantizers (default: half the logical CPUs, capped at 8)",
    )
    return parser.parse_args()


def load_safetensor_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise ValueError(f"truncated Safetensors header length: {path}")
        (header_length,) = struct.unpack("<Q", encoded_length)
        if header_length == 0 or header_length > MAX_HEADER_BYTES:
            raise ValueError(f"invalid Safetensors header length: {path}")
        encoded = stream.read(header_length)
        if len(encoded) != header_length:
            raise ValueError(f"truncated Safetensors header: {path}")
    try:
        header = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Safetensors JSON header: {path}: {error}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    return header, 8 + header_length


def scan_tensors(model: Path, artifact: Path) -> dict[str, TensorRange]:
    tensors: dict[str, TensorRange] = {}
    for path in sorted(model.glob("*.safetensors")):
        if path.resolve() == artifact.resolve():
            continue
        header, data_start = load_safetensor_header(path)
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
    return tensors


def required_source(
    tensors: dict[str, TensorRange],
    name: str,
    shape: tuple[int, ...],
) -> TensorRange:
    source = tensors.get(name)
    if source is None:
        raise ValueError(f"missing source tensor: {name}")
    if (
        source.dtype != "BF16"
        or source.shape != shape
        or source.byte_count != math.prod(shape) * 2
    ):
        raise ValueError(
            f"invalid BF16 source tensor {name}: "
            f"dtype={source.dtype}, shape={source.shape}"
        )
    return source


def artifact_tensor_plan(
    layer_count: int,
    mtp_layer_count: int,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    config_fnv1a64: int,
    index_fnv1a64: int,
) -> list[ArtifactTensor]:
    identity = (
        f"{ARTIFACT_PREFIX}.identity.v3."
        f"{layer_count}.{mtp_layer_count}.{expert_count}."
        f"{hidden_size}.{intermediate_size}."
        f"{config_fnv1a64:016x}.{index_fnv1a64:016x}"
    )
    plan = [
        ArtifactTensor(
            name=identity,
            shape=(0,),
            byte_count=0,
        )
    ]
    def append_bank(prefix: str) -> None:
        for role, rows, columns in (
            ("gate_up", intermediate_size * 2, hidden_size),
            ("down", hidden_size, intermediate_size),
        ):
            groups = columns // 32
            block_shape = (expert_count, rows, groups, 16)
            scale_shape = (expert_count, rows, groups)
            plan.append(
                ArtifactTensor(
                    name=prefix + role + ".blocks",
                    shape=block_shape,
                    byte_count=math.prod(block_shape),
                )
            )
            plan.append(
                ArtifactTensor(
                    name=prefix + role + ".scales",
                    shape=scale_shape,
                    byte_count=math.prod(scale_shape),
                )
            )
    for layer_id in range(layer_count):
        append_bank(f"{ARTIFACT_PREFIX}.layers.{layer_id}.experts.")
    for layer_id in range(mtp_layer_count):
        append_bank(
            f"{ARTIFACT_PREFIX}.mtp.layers.{layer_id}.experts."
        )
    return plan


def encode_header(
    plan: list[ArtifactTensor],
    config_sha256: str,
    index_sha256: str,
    config_fnv1a64: int,
    index_fnv1a64: int,
) -> bytes:
    header: dict[str, object] = {
        "__metadata__": {
            "format": ARTIFACT_FORMAT,
            "model_type": "qwen3_5_moe",
            "scope": "main-and-mtp-routed-experts",
            "quantization": "ocp-e2m1-e8m0-block32-maxabs",
            "source_config_sha256": config_sha256,
            "source_index_sha256": index_sha256,
            "source_config_fnv1a64": f"{config_fnv1a64:016x}",
            "source_index_fnv1a64": f"{index_fnv1a64:016x}",
        }
    }
    offset = 0
    for tensor in plan:
        header[tensor.name] = {
            "dtype": "U8",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.byte_count],
        }
        offset += tensor.byte_count
    encoded = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encoded + b" " * ((-len(encoded)) % 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64_file(path: Path) -> int:
    value = FNV1A64_OFFSET
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            for byte in chunk:
                value ^= byte
                value = (value * FNV1A64_PRIME) & UINT64_MASK
    return value


def bfloat16_rows_to_float32(rows: np.ndarray) -> np.ndarray:
    bits = np.asarray(rows, dtype=np.uint16).astype(np.uint32)
    bits <<= np.uint32(16)
    return bits.view(np.float32)


def quantize_mxfp4_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = bfloat16_rows_to_float32(rows)
    if not np.isfinite(values).all():
        raise ValueError("routed Expert source contains NaN or infinity")
    row_count, columns = values.shape
    groups = columns // 32
    grouped = values.reshape(row_count, groups, 32)
    maximum = np.max(np.abs(grouped), axis=2)
    exponent = np.full(maximum.shape, -127, dtype=np.int16)
    nonzero = maximum > 0.0
    exponent[nonzero] = np.ceil(
        np.log2(maximum[nonzero] / np.float32(6.0))
    ).astype(np.int16)
    np.clip(exponent, -127, 127, out=exponent)
    scale = np.ldexp(
        np.ones(exponent.shape, dtype=np.float32),
        exponent,
    )
    normalized = np.abs(grouped) / scale[:, :, None]
    magnitude = np.searchsorted(
        E2M1_THRESHOLDS,
        normalized,
        side="left",
    ).astype(np.uint8)
    codes = magnitude | (
        np.signbit(grouped).astype(np.uint8) << np.uint8(3)
    )
    packed = codes[:, :, 0::2] | (codes[:, :, 1::2] << np.uint8(4))
    scales = (exponent + 127).astype(np.uint8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


def compile_bank(
    destination: BinaryIO,
    source: TensorRange,
    expert_count: int,
    rows: int,
    columns: int,
    workers: int,
    interleave_gate_up: bool,
) -> None:
    mapped = np.memmap(
        source.path,
        mode="r",
        dtype="<u2",
        offset=source.offset,
        shape=source.shape,
        order="C",
    )
    groups = columns // 32
    scales = np.empty((expert_count, rows, groups), dtype=np.uint8)
    if interleave_gate_up:
        intermediate = rows // 2
        row_order = np.empty(rows, dtype=np.int64)
        row_order[0::2] = np.arange(intermediate, dtype=np.int64)
        row_order[1::2] = intermediate + np.arange(
            intermediate,
            dtype=np.int64,
        )
    else:
        row_order = None

    def quantize_expert(expert_id: int) -> tuple[np.ndarray, np.ndarray]:
        blocks, expert_scales = quantize_mxfp4_rows(mapped[expert_id])
        if row_order is not None:
            blocks = np.ascontiguousarray(blocks[row_order])
            expert_scales = np.ascontiguousarray(expert_scales[row_order])
        return blocks, expert_scales

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for expert_id, (blocks, expert_scales) in enumerate(
            executor.map(quantize_expert, range(expert_count))
        ):
            destination.write(blocks.tobytes(order="C"))
            scales[expert_id] = expert_scales
    destination.write(scales.tobytes(order="C"))
    del mapped


def compile_artifact(
    model: Path,
    output: Path,
    tensors: dict[str, TensorRange],
    header: bytes,
    layer_count: int,
    mtp_layer_count: int,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    workers: int,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"artifact already exists (pass --overwrite to replace it): {output}"
        )
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    expected_data_bytes = (layer_count + mtp_layer_count) * expert_count * (
        intermediate_size * 2 * hidden_size * 17 // 32
        + hidden_size * intermediate_size * 17 // 32
    )
    required_bytes = 8 + len(header) + expected_data_bytes
    if shutil.disk_usage(output.parent).free < required_bytes:
        raise OSError(
            f"insufficient free space for {required_bytes / (1024 ** 3):.2f} GiB artifact"
        )

    try:
        with temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
            destination.write(struct.pack("<Q", len(header)))
            destination.write(header)
            for layer_id in range(layer_count):
                source_prefix = (
                    f"model.language_model.layers.{layer_id}.mlp.experts."
                )
                gate_up = required_source(
                    tensors,
                    source_prefix + "gate_up_proj",
                    (
                        expert_count,
                        intermediate_size * 2,
                        hidden_size,
                    ),
                )
                compile_bank(
                    destination,
                    gate_up,
                    expert_count,
                    intermediate_size * 2,
                    hidden_size,
                    workers,
                    True,
                )
                print(
                    f"compiled layer {layer_id + 1}/{layer_count} gate/up",
                    flush=True,
                )
                down = required_source(
                    tensors,
                    source_prefix + "down_proj",
                    (
                        expert_count,
                        hidden_size,
                        intermediate_size,
                    ),
                )
                compile_bank(
                    destination,
                    down,
                    expert_count,
                    hidden_size,
                    intermediate_size,
                    workers,
                    False,
                )
                print(
                    f"compiled layer {layer_id + 1}/{layer_count} down",
                    flush=True,
                )
            for layer_id in range(mtp_layer_count):
                source_prefix = f"mtp.layers.{layer_id}.mlp.experts."
                gate_up = required_source(
                    tensors,
                    source_prefix + "gate_up_proj",
                    (
                        expert_count,
                        intermediate_size * 2,
                        hidden_size,
                    ),
                )
                compile_bank(
                    destination,
                    gate_up,
                    expert_count,
                    intermediate_size * 2,
                    hidden_size,
                    workers,
                    True,
                )
                print(
                    f"compiled MTP layer {layer_id + 1}/"
                    f"{mtp_layer_count} gate/up",
                    flush=True,
                )
                down = required_source(
                    tensors,
                    source_prefix + "down_proj",
                    (
                        expert_count,
                        hidden_size,
                        intermediate_size,
                    ),
                )
                compile_bank(
                    destination,
                    down,
                    expert_count,
                    hidden_size,
                    intermediate_size,
                    workers,
                    False,
                )
                print(
                    f"compiled MTP layer {layer_id + 1}/"
                    f"{mtp_layer_count} down",
                    flush=True,
                )
            destination.flush()
            os.fsync(destination.fileno())
        actual_bytes = temporary.stat().st_size
        if actual_bytes != required_bytes:
            raise ValueError(
                f"compiled artifact size mismatch: expected {required_bytes}, got {actual_bytes}"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.workers <= 0:
            raise ValueError("--workers must be positive")
        model = arguments.model.resolve()
        if not model.is_dir():
            raise ValueError(f"model directory does not exist: {model}")
        config_path = model / "config.json"
        index_path = model / "model.safetensors.index.json"
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("model_type") != "qwen3_5_moe":
            raise ValueError(
                "the compiler requires the official qwen3_5_moe package"
            )
        text_config = config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("config.json is missing text_config")
        layer_count = int(text_config["num_hidden_layers"])
        mtp_layer_count = int(text_config["mtp_num_hidden_layers"])
        expert_count = int(text_config["num_experts"])
        hidden_size = int(text_config["hidden_size"])
        intermediate_size = int(text_config["moe_intermediate_size"])
        if (
            layer_count <= 0
            or mtp_layer_count != 1
            or expert_count <= 0
            or hidden_size <= 0
            or intermediate_size <= 0
            or hidden_size % 32
            or intermediate_size % 32
            or bool(text_config.get("mtp_use_dedicated_embeddings", True))
        ):
            raise ValueError("unsupported Qwen routed Expert dimensions")

        output = model / ARTIFACT_NAME
        tensors = scan_tensors(model, output)
        config_fnv1a64 = fnv1a64_file(config_path)
        index_fnv1a64 = fnv1a64_file(index_path)
        plan = artifact_tensor_plan(
            layer_count,
            mtp_layer_count,
            expert_count,
            hidden_size,
            intermediate_size,
            config_fnv1a64,
            index_fnv1a64,
        )
        header = encode_header(
            plan,
            sha256_file(config_path),
            sha256_file(index_path),
            config_fnv1a64,
            index_fnv1a64,
        )
        artifact_bytes = 8 + len(header) + sum(
            tensor.byte_count for tensor in plan
        )
        print(
            f"compiling {(layer_count + mtp_layer_count) * expert_count} "
            f"main/MTP routed Experts "
            f"into {artifact_bytes / (1024 ** 3):.2f} GiB at {output}",
            flush=True,
        )
        compile_artifact(
            model,
            output,
            tensors,
            header,
            layer_count,
            mtp_layer_count,
            expert_count,
            hidden_size,
            intermediate_size,
            arguments.workers,
            arguments.overwrite,
        )
        print(f"compiled Qwen3.6 MXFP4 artifact: {output}", flush=True)
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
