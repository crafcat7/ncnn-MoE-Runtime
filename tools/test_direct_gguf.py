#!/usr/bin/env python3
"""Compare native GGUF graphs with independent decoded HF fixture graphs.

Uses the existing nonzero family fixtures and the official GGUF writer/decoder;
the runtime parser, layout transforms and kernels are exercised independently.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from tools import test_native_quantized as fixtures
except ModuleNotFoundError:
    import test_native_quantized as fixtures


def add_metadata(writer, architecture: str, model: dict) -> None:
    text = model.get("text_config", model)
    prefix = architecture + "."
    writer.add_array("tokenizer.ggml.tokens", [f"token{i}" for i in range(text["vocab_size"])])
    common = {
        "context_length": "max_position_embeddings", "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads", "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
    }
    for key, source in common.items():
        writer.add_uint32(prefix + key, text[source])
    writer.add_float32(prefix + "attention.layer_norm_rms_epsilon", text["rms_norm_eps"])
    rope = text.get("rope_parameters", {})
    writer.add_float32(prefix + "rope.freq_base", rope.get("rope_theta", text.get("rope_theta", 10000.0)))
    if architecture.startswith("qwen"):
        for key, source in {
            "ssm.group_count": "linear_num_key_heads", "ssm.state_size": "linear_key_head_dim",
            "ssm.time_step_rank": "linear_num_value_heads", "ssm.conv_kernel": "linear_conv_kernel_dim",
        }.items():
            writer.add_uint32(prefix + key, text[source])
        writer.add_uint32(prefix + "ssm.inner_size", text["linear_num_value_heads"] * text["linear_value_head_dim"])
        writer.add_uint32(prefix + "rope.dimension_count", round(text["head_dim"] * rope["partial_rotary_factor"]))
        if architecture == "qwen35":
            writer.add_uint32(prefix + "feed_forward_length", text["intermediate_size"])
        else:
            for key, source in {
                "expert_count": "num_experts", "expert_used_count": "num_experts_per_tok",
                "expert_feed_forward_length": "moe_intermediate_size",
                "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
            }.items():
                writer.add_uint32(prefix + key, text[source])
        if architecture == "qwen4exp":
            for key, source in {
                "hyper_connection.count": "hc_count", "hyper_connection.low_rank": "hc_lowrank",
                "attention.indexer.head_count": "indexer_n_heads", "attention.indexer.key_length": "indexer_head_dim",
                "attention.indexer.top_k": "indexer_budget", "ple.ngram_size": "ngram_size",
                "ple.heads_per_ngram": "heads_per_ngram", "ple.conv_kernel": "ple_conv_kernel_size",
                "ple.eos_token_id": "eos_token_id",
            }.items():
                writer.add_uint32(prefix + key, text[source])
            writer.add_array(prefix + "attention.compress_ratios", [0, text["indexer_compress_ratio"]])
            writer.add_array(prefix + "ple.layers", [value - 1 for value in text["ple_layer_ids"]])
    elif architecture == "gpt-oss":
        for key, source in {
            "expert_count": "num_local_experts", "expert_used_count": "experts_per_token",
            "expert_feed_forward_length": "intermediate_size", "attention.sliding_window": "sliding_window",
        }.items():
            writer.add_uint32(prefix + key, text[source])
        writer.add_float32(prefix + "swiglu_clamp_exp", text["swiglu_limit"])
        writer.add_float32(prefix + "rope.scaling.factor", text["rope_scaling"]["factor"])
        writer.add_uint32(prefix + "rope.scaling.original_context_length", text["initial_context_length"])
    else:
        for key, source in {
            "expert_count": "n_routed_experts", "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts", "expert_feed_forward_length": "moe_intermediate_size",
            "attention.q_lora_rank": "q_lora_rank", "attention.output_group_count": "o_groups",
            "attention.output_lora_rank": "o_lora_rank", "attention.sliding_window": "sliding_window",
            "attention.indexer.head_count": "index_n_heads", "attention.indexer.key_length": "index_head_dim",
            "attention.indexer.top_k": "index_topk", "rope.dimension_count": "qk_rope_head_dim",
            "hyper_connection.count": "hc_mult", "hyper_connection.sinkhorn_iterations": "hc_sinkhorn_iters",
            "hash_layer_count": "num_hash_layers",
        }.items():
            writer.add_uint32(prefix + key, text[source])
        writer.add_array(prefix + "attention.compress_ratios", text["compress_ratios"])
        writer.add_float32(prefix + "expert_weights_scale", text["routed_scaling_factor"])
        writer.add_float32(prefix + "hyper_connection.epsilon", text["hc_eps"])
        writer.add_float32(prefix + "swiglu_clamp_exp", text["swiglu_limit"])
        writer.add_uint32(prefix + "expert_gating_func", 4)
        writer.add_bool(prefix + "expert_weights_norm", True)
        writer.add_float32(prefix + "attention.compress_rope_freq_base", 160000.0)
        writer.add_string(prefix + "rope.scaling.type", "yarn")
        writer.add_float32(prefix + "rope.scaling.factor", 16.0)
        writer.add_float32(prefix + "rope.scaling.yarn_beta_fast", 32.0)
        writer.add_float32(prefix + "rope.scaling.yarn_beta_slow", 1.0)
        writer.add_uint32(prefix + "rope.scaling.original_context_length", text["original_max_position_embeddings"])


def floating_copy(source: Path, destination: Path) -> None:
    """Keep metadata/transformed layout, decode only fixture Qn_K blocks."""
    gguf = fixtures.gguf
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(destination, reader.fields["general.architecture"].contents())
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    for tensor in reader.tensors:
        values = gguf.dequantize(tensor.data, tensor.tensor_type)
        writer.add_tensor(tensor.name, np.asarray(values, np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def mxfp4_copy(fixture, directory: Path) -> tuple[Path, Path]:
    """Independent E2M1 codes, exponent scales, and GGUF half-block packing."""
    gguf = fixtures.gguf
    reader = gguf.GGUFReader(fixture.source)
    destination = directory / "mxfp4.gguf"
    architecture = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(destination, architecture)
    for name, field in reader.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    reference = {name: (info.dtype, info.read().copy())
                 for name, info in fixtures.importer.scan_safetensors(fixture.baseline).items()}
    table = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    for index, tensor in enumerate(reader.tensors):
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if not match:
            writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)
            continue
        shape = tuple(int(value) for value in tensor.shape[::-1])
        codes = np.random.default_rng(index).integers(0, 16, shape, dtype=np.uint8)
        grouped = codes.reshape(*shape[:-1], shape[-1] // 32, 32)
        exponents = np.random.default_rng(index + 100).integers(118, 122, grouped.shape[:-1], dtype=np.uint8)
        raw = np.empty((*grouped.shape[:-1], 17), np.uint8)
        raw[..., 0] = exponents
        raw[..., 1:] = grouped[..., :16] | (grouped[..., 16:] << 4)
        writer.add_tensor(tensor.name, raw.reshape(*shape[:-1], -1), raw_dtype=gguf.GGMLQuantizationType.MXFP4)
        decoded = (table[grouped] * np.exp2(exponents.astype(np.float32)[..., None] - 127)).reshape(shape)
        for expert in range(shape[0]):
            if architecture == "deepseek4":
                role = {"gate": "w1", "up": "w3", "down": "w2"}[match[2]]
                name = f"layers.{match[1]}.ffn.experts.{expert}.{role}.weight"
            else:
                prefix = "model." if architecture == "gpt-oss" else fixtures.PREFIX
                name = f"{prefix}layers.{match[1]}.mlp.experts.{expert}.{match[2]}_proj.weight"
            reference[name] = ("BF16", fixtures.bfloat16(decoded[expert]))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    baseline = directory / "mxfp4-baseline"
    fixtures.assets(baseline, json.loads((fixture.baseline / "config.json").read_text()))
    fixtures.write_safetensors(baseline / "model.safetensors", reference)
    return baseline, destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    if fixtures.gguf is None:
        print("SKIP: optional official gguf package is unavailable")
        return 77
    real_writer = fixtures.gguf.GGUFWriter
    real_family = fixtures.family_tensors

    def family_tensors(family):
        model, tensors, native = real_family(family)
        if family == "deepseek_v4":
            model.update(rope_theta=10000.0, partial_rotary_factor=0.5)
        return model, tensors, native

    with tempfile.TemporaryDirectory(prefix="ncnn-direct-gguf-", ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        cases = [("qwen35", False, False), ("qwen35moe", True, False), ("qwen35-reordered", False, True),
                 ("gpt_oss", True, False), ("deepseek_v4", True, False), ("qwen4_exp", True, False)]
        for label, moe, reordered in cases:
            family = label if label in ("gpt_oss", "deepseek_v4", "qwen4_exp") else None
            model = family_tensors(family)[0] if family else fixtures.config(moe, reordered)

            def writer(path, architecture, *extra, **kwargs):
                result = real_writer(path, architecture, *extra, **kwargs)
                add_metadata(result, architecture, model)
                return result

            directory = root / label
            directory.mkdir()
            with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                fixture = (fixtures.prepare_family_gguf(directory, family, fp32=False) if family
                           else fixtures.prepare_gguf(directory, moe, reordered, fp32=False))
            for source in (fixture.source, directory / "floating.gguf"):
                if source != fixture.source:
                    floating_copy(fixture.source, source)
                command = [str(args.runner.resolve()), str(fixture.baseline), str(source)]
                if args.hybrid:
                    command.append("--hybrid")
                print(f"Checking {label}: {source.name}", flush=True)
                completed = subprocess.run(command)
                if completed.returncode == 77:
                    return 77
                completed.check_returncode()
            if moe:
                baseline, source = mxfp4_copy(fixture, directory)
                command = [str(args.runner.resolve()), str(baseline), str(source), "--on-demand"]
                if args.hybrid:
                    command.append("--hybrid")
                print(f"Checking {label}: MXFP4 with on-demand Expert cache", flush=True)
                completed = subprocess.run(command)
                if completed.returncode == 77:
                    return 77
                completed.check_returncode()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
