#!/usr/bin/env python3
"""Create a deterministic Tiny MoE package for the CPU reference runtime."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path
from typing import BinaryIO, Iterable


def write_floats(stream: BinaryIO, values: Iterable[float]) -> None:
    chunk: list[float] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == 4096:
            stream.write(struct.pack(f"<{len(chunk)}f", *chunk))
            chunk.clear()
    if chunk:
        stream.write(struct.pack(f"<{len(chunk)}f", *chunk))


def write_int8_matrix(
    stream: BinaryIO, values: list[float], rows: int, columns: int
) -> None:
    quantized: list[int] = []
    scales: list[float] = []
    for row in range(rows):
        row_values = values[row * columns : (row + 1) * columns]
        maximum = max(abs(value) for value in row_values)
        scale = maximum / 127.0 if maximum > 0.0 else 1.0
        scales.append(scale)
        quantized.extend(
            max(-127, min(127, round(value / scale))) for value in row_values
        )
    stream.write(struct.pack(f"<{len(quantized)}b", *quantized))
    write_floats(stream, scales)


def write_expert_matrix(
    stream: BinaryIO,
    values: list[float],
    rows: int,
    columns: int,
    dtype: str,
) -> None:
    if dtype == "float32":
        write_floats(stream, values)
    else:
        write_int8_matrix(stream, values, rows, columns)


def random_tensor(rng: random.Random, count: int, fan_in: int) -> list[float]:
    limit = 1.0 / math.sqrt(float(fan_in))
    return [rng.uniform(-limit, limit) for _ in range(count)]


def build_package(args: argparse.Namespace) -> None:
    if args.top_k < 1 or args.top_k > args.experts:
        raise ValueError("--top-k must be between 1 and --experts")
    for name in ("vocabulary", "hidden", "intermediate", "layers", "experts"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")

    output: Path = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    manifest = {
        "format_version": 1,
        "model_type": "tiny_moe",
        "vocabulary_size": args.vocabulary,
        "hidden_size": args.hidden,
        "intermediate_size": args.intermediate,
        "layer_count": args.layers,
        "expert_count": args.experts,
        "experts_per_token": args.top_k,
        "expert_activation": "silu",
        "expert_layout": "gate_up_down",
        "expert_weight_dtype": args.expert_weight_dtype,
        "normalize_topk_weights": True,
        "use_expert_bias": False,
        "norm_epsilon": 1e-5,
        "weights_file": "model.ncnnmoe.bin",
    }
    (output / "model.ncnnmoe.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with (output / "model.ncnnmoe.bin").open("wb") as weights:
        # token_embedding.weight [vocabulary, hidden]
        write_floats(
            weights,
            random_tensor(rng, args.vocabulary * args.hidden, args.hidden),
        )

        for _layer in range(args.layers):
            # pre_ffn_norm.weight [hidden]
            write_floats(weights, (1.0 for _ in range(args.hidden)))
            # router.weight [experts, hidden]
            write_floats(
                weights,
                random_tensor(rng, args.experts * args.hidden, args.hidden),
            )
            for _expert in range(args.experts):
                # gate.weight and up.weight [intermediate, hidden]
                for _projection in range(2):
                    write_expert_matrix(
                        weights,
                        random_tensor(
                            rng, args.intermediate * args.hidden, args.hidden
                        ),
                        args.intermediate,
                        args.hidden,
                        args.expert_weight_dtype,
                    )
                # down.weight [hidden, intermediate]
                write_expert_matrix(
                    weights,
                    random_tensor(
                        rng, args.hidden * args.intermediate, args.intermediate
                    ),
                    args.hidden,
                    args.intermediate,
                    args.expert_weight_dtype,
                )

        # final_norm.weight [hidden]
        write_floats(weights, (1.0 for _ in range(args.hidden)))
        # lm_head.weight [vocabulary, hidden]
        write_floats(
            weights,
            random_tensor(rng, args.vocabulary * args.hidden, args.hidden),
        )

    print(f"wrote Tiny MoE package to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output model directory")
    parser.add_argument("--vocabulary", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--expert-weight-dtype",
        choices=("float32", "int8"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


if __name__ == "__main__":
    build_package(parse_args())
