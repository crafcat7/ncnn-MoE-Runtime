#!/usr/bin/env python3
"""Create a dependency-free miniature checkpoint in the official GPT-OSS layout."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Tensor:
    dtype: str
    shape: list[int]
    data: bytes


def bfloat16(values: list[float]) -> bytes:
    encoded = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits += 0x7FFF + ((bits >> 16) & 1)
        encoded.extend(struct.pack("<H", bits >> 16))
    return bytes(encoded)


def zeros(count: int) -> list[float]:
    return [0.0] * count


def identity(rows: int, columns: int) -> list[float]:
    values = zeros(rows * columns)
    for index in range(min(rows, columns)):
        values[index * columns + index] = 1.0
    return values


def add_bfloat16(
    tensors: dict[str, Tensor], name: str, shape: list[int], values: list[float]
) -> None:
    count = 1
    for dimension in shape:
        count *= dimension
    if len(values) != count:
        raise ValueError(f"{name}: expected {count} values, got {len(values)}")
    tensors[name] = Tensor("BF16", shape, bfloat16(values))


def add_mxfp4(
    tensors: dict[str, Tensor], prefix: str, experts: int, rows: int, columns: int
) -> None:
    if columns % 32:
        raise ValueError("MXFP4 columns must be divisible by 32")
    groups = columns // 32
    tensors[prefix + "_blocks"] = Tensor(
        "U8", [experts, rows, groups, 16], bytes(experts * rows * groups * 16)
    )
    tensors[prefix + "_scales"] = Tensor(
        "U8", [experts, rows, groups], bytes([127]) * (experts * rows * groups)
    )


def write_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name, tensor in tensors.items():
        header[name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + len(tensor.data)],
        }
        offset += len(tensor.data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for tensor in tensors.values():
            stream.write(tensor.data)


def build_fixture(output: Path) -> None:
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    config = {
        "model_type": "gpt_oss",
        "vocab_size": 2,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_local_experts": 1,
        "experts_per_token": 1,
        "num_experts_per_tok": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "sliding_window": 4,
        "initial_context_length": 16,
        "max_position_embeddings": 32,
        "rope_theta": 150000.0,
        "rope_scaling": {
            "factor": 1.0,
            "beta_slow": 1.0,
            "beta_fast": 32.0,
        },
        "attention_bias": True,
        "rms_norm_eps": 1e-5,
        "swiglu_limit": 7.0,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    tensors: dict[str, Tensor] = {}
    add_bfloat16(tensors, "model.embed_tokens.weight", [2, 32], identity(2, 32))
    prefix = "model.layers.0."
    add_bfloat16(tensors, prefix + "input_layernorm.weight", [32], [1.0] * 32)
    add_bfloat16(tensors, prefix + "self_attn.q_proj.weight", [32, 32], identity(32, 32))
    add_bfloat16(tensors, prefix + "self_attn.q_proj.bias", [32], zeros(32))
    add_bfloat16(tensors, prefix + "self_attn.k_proj.weight", [8, 32], identity(8, 32))
    add_bfloat16(tensors, prefix + "self_attn.k_proj.bias", [8], zeros(8))
    add_bfloat16(tensors, prefix + "self_attn.v_proj.weight", [8, 32], identity(8, 32))
    add_bfloat16(tensors, prefix + "self_attn.v_proj.bias", [8], zeros(8))
    add_bfloat16(tensors, prefix + "self_attn.o_proj.weight", [32, 32], identity(32, 32))
    add_bfloat16(tensors, prefix + "self_attn.o_proj.bias", [32], zeros(32))
    add_bfloat16(
        tensors,
        prefix + "self_attn.sinks",
        [8],
        [0.125 * index for index in range(8)],
    )
    add_bfloat16(
        tensors, prefix + "post_attention_layernorm.weight", [32], [1.0] * 32
    )
    add_bfloat16(tensors, prefix + "mlp.router.weight", [1, 32], zeros(32))
    add_bfloat16(tensors, prefix + "mlp.router.bias", [1], zeros(1))
    add_bfloat16(
        tensors, prefix + "mlp.experts.gate_up_proj_bias", [1, 64], zeros(64)
    )
    add_bfloat16(
        tensors, prefix + "mlp.experts.down_proj_bias", [1, 32], zeros(32)
    )
    add_mxfp4(tensors, prefix + "mlp.experts.gate_up_proj", 1, 64, 32)
    add_mxfp4(tensors, prefix + "mlp.experts.down_proj", 1, 32, 32)
    add_bfloat16(tensors, "model.norm.weight", [32], [1.0] * 32)
    add_bfloat16(tensors, "lm_head.weight", [2, 32], identity(2, 32))
    write_safetensors(output / "model-00000-of-00001.safetensors", tensors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-runner", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_fixture(args.output.resolve())
    if args.verify_runner:
        prompt_file = args.output.resolve() / "prompt.tokens"
        prompt_file.write_text("0 1 0 1\n", encoding="ascii")
        common_arguments = [
            str(args.verify_runner.resolve()),
            str(args.output.resolve()),
            "--prompt-token-file",
            str(prompt_file),
            "--max-new-tokens",
            "3",
            "--temperature",
            "0",
        ]
        automatic = subprocess.run(
            common_arguments,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        cpu = subprocess.run(
            common_arguments + ["--cpu", "--cpu-packed-weights", "off"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        packed = subprocess.run(
            common_arguments + ["--cpu", "--cpu-packed-weights", "on"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        paged = subprocess.run(
            common_arguments + ["--cpu", "--expert-cache-mb", "1"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(automatic.stdout, end="")
        if automatic.returncode:
            raise SystemExit(automatic.returncode)
        if cpu.returncode:
            print(cpu.stdout, end="")
            raise SystemExit(cpu.returncode)
        if paged.returncode:
            print(paged.stdout, end="")
            raise SystemExit(paged.returncode)
        if "CPU packed weights off" not in automatic.stdout:
            raise SystemExit("default GPT-OSS run did not keep CPU repack disabled")
        if "CPU packed weights off" not in cpu.stdout:
            print(cpu.stdout, end="")
            raise SystemExit("explicit CPU packed-weight off mode was not selected")
        packed_supported = packed.returncode == 0
        if packed_supported and "CPU packed weights on" not in packed.stdout:
            print(packed.stdout, end="")
            raise SystemExit("explicit CPU packed-weight on mode was not selected")
        if not packed_supported and "CPU packed weights are unavailable" not in packed.stdout:
            print(packed.stdout, end="")
            raise SystemExit(packed.returncode)
        if "loaded gpt_oss" not in automatic.stdout or "generated token ids:" not in automatic.stdout:
            raise SystemExit("GPT-OSS fixture runner output is incomplete")
        def generated_tokens(output: str) -> str:
            return next(
                line.partition(":")[2].strip()
                for line in output.splitlines()
                if line.startswith("generated token ids:")
            )

        automatic_tokens = generated_tokens(automatic.stdout)
        cpu_tokens = generated_tokens(cpu.stdout)
        paged_tokens = generated_tokens(paged.stdout)
        if automatic_tokens != cpu_tokens or automatic_tokens != paged_tokens:
            print(cpu.stdout, end="")
            print(paged.stdout, end="")
            raise SystemExit("automatic and CPU GPT-OSS outputs differ")
        if packed_supported and automatic_tokens != generated_tokens(packed.stdout):
            print(packed.stdout, end="")
            raise SystemExit("packed and default GPT-OSS outputs differ")
        if "Expert cache: 2 hit(s), 1 miss(es)" not in paged.stdout:
            print(paged.stdout, end="")
            raise SystemExit("paged GPT-OSS fixture did not exercise the expert cache")
        if "backend vulkan-dense/cpu-experts" in automatic.stdout \
                and "Vulkan attention blocks: 0" in automatic.stdout:
            raise SystemExit("Vulkan GPT-OSS run did not execute GPU attention")


if __name__ == "__main__":
    main()
