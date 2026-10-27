#!/usr/bin/env python3
"""Offline checkpoint -> native worker regression against independent
decoded-weight oracles.

Requires NumPy; real GGUF cases additionally require the official gguf package.
The fixtures exercise prefill/decode, recurrent/full attention, dense/shared/routed
FFNs, the output head, mixed BF16/INT4 weights, GPTQ act-order, and memory plans.
No checkpoint downloads are used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import import_quantized as importer  # noqa: E402
from tools.ncnn_moe_protocol import WorkerClient, WorkerError  # noqa: E402
from tools.test_import_quantized import iq4_nl_blocks, iq4_xs_blocks, pack_columns, pack_gptq, write_safetensors  # noqa: E402

try:
    import gguf
except ImportError:
    gguf = None

HIDDEN = 256
INTERMEDIATE = 256
VOCABULARY = 8
GROUP_SIZE = 128
PREFIX = "model.language_model."


def bfloat16(values: np.ndarray) -> np.ndarray:
    """Independent round-to-nearest-even reference; no importer dequantizer."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def float32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def plain(shape: tuple[int, ...], seed: int, scale: float = 1 / 512) -> np.ndarray:
    values = np.random.default_rng(seed).integers(-8, 9, size=shape).astype(np.float32)
    return bfloat16(values * scale)


def config(moe: bool = False, reordered: bool = False) -> dict:
    model_type = "qwen3_5_moe" if moe else "qwen3_5"
    text = {
        "model_type": model_type + "_text", "vocab_size": VOCABULARY,
        "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 64,
        "linear_num_key_heads": 8, "linear_num_value_heads": 8,
        "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 2, "max_position_embeddings": 32,
        "rms_norm_eps": 0.000001,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "layer_types": ["linear_attention", "full_attention"], "hidden_act": "silu",
        "dtype": "bfloat16", "mamba_ssm_dtype": "float32",
        "attention_bias": False, "attn_output_gate": True, "tie_word_embeddings": False,
    }
    if moe:
        text.update(moe_intermediate_size=INTERMEDIATE,
                    shared_expert_intermediate_size=INTERMEDIATE,
                    num_experts=2, num_experts_per_tok=1,
                    mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    if reordered:
        text.update(linear_num_key_heads=2, linear_num_value_heads=6, linear_value_head_dim=128)
    return {"model_type": model_type, "text_config": text}


def assets(root: Path, model_config: dict) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(model_config))
    (root / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
    (root / "chat_template.jinja").write_text("{{ messages }}")


def base_tensors(moe: bool, reordered: bool = False) -> tuple[dict, list[str]]:
    """Return BF16 weights and the Linear matrices eligible for INT4/Qn_K."""
    result = {
        PREFIX + "embed_tokens.weight": ("BF16", plain((VOCABULARY, HIDDEN), 1, 1 / 32)),
        PREFIX + "norm.weight": ("BF16", bfloat16(np.zeros(HIDDEN))),
        "lm_head.weight": ("BF16", plain((VOCABULARY, HIDDEN), 2)),
    }
    quantized = ["lm_head.weight"]
    text = config(moe, reordered)["text_config"]
    key_heads, value_heads = text["linear_num_key_heads"], text["linear_num_value_heads"]
    key_dimension, value_dimension = text["linear_key_head_dim"], text["linear_value_head_dim"]
    value_width = value_heads * value_dimension
    qkv_width = 2 * key_heads * key_dimension + value_width
    seed = 10
    for layer in range(2):
        prefix = PREFIX + f"layers.{layer}."
        for role in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            result[prefix + role] = ("BF16", bfloat16(np.zeros(HIDDEN)))
        matrices = {}
        if layer == 0:
            matrices.update({
                "linear_attn.in_proj_qkv.weight": (qkv_width, HIDDEN),
                "linear_attn.in_proj_z.weight": (value_width, HIDDEN),
                "linear_attn.in_proj_b.weight": (value_heads, HIDDEN),
                "linear_attn.in_proj_a.weight": (value_heads, HIDDEN),
                "linear_attn.out_proj.weight": (HIDDEN, value_width),
            })
            result[prefix + "linear_attn.conv1d.weight"] = ("BF16", plain((qkv_width, 1, 2), seed, 1 / 8))
            result[prefix + "linear_attn.dt_bias"] = ("BF16", bfloat16(np.full(value_heads, -1.0)))
            result[prefix + "linear_attn.A_log"] = ("BF16", bfloat16(np.zeros(value_heads)))
            result[prefix + "linear_attn.norm.weight"] = ("BF16", bfloat16(np.ones(value_dimension)))
        else:
            matrices.update({
                "self_attn.q_proj.weight": (2 * HIDDEN, HIDDEN),
                "self_attn.k_proj.weight": (HIDDEN // 2, HIDDEN),
                "self_attn.v_proj.weight": (HIDDEN // 2, HIDDEN),
                "self_attn.o_proj.weight": (HIDDEN, HIDDEN),
            })
            for role in ("q_norm.weight", "k_norm.weight"):
                result[prefix + "self_attn." + role] = ("BF16", bfloat16(np.zeros(64)))
        if moe:
            # The first two prompt embeddings favor different routed Experts.
            embedding = float32(result[PREFIX + "embed_tokens.weight"][1])
            result[prefix + "mlp.gate.weight"] = ("BF16", bfloat16(embedding[[1, 2]]))
            result[prefix + "mlp.shared_expert_gate.weight"] = ("BF16", plain((1, HIDDEN), seed + 2))
            for expert in ("shared_expert", "experts.0", "experts.1"):
                for role in ("gate", "up", "down"):
                    matrices[f"mlp.{expert}.{role}_proj.weight"] = (INTERMEDIATE, HIDDEN)
        else:
            matrices.update({f"mlp.{role}_proj.weight": (INTERMEDIATE, HIDDEN) for role in ("gate", "up", "down")})
        for role, shape in matrices.items():
            name = prefix + role
            result[name] = ("BF16", plain(shape, seed))
            # Leave one routed projection in BF16 to exercise mixed Expert storage.
            if not (moe and ".experts.1.up_proj." in name):
                quantized.append(name)
            seed += 1
    return result, quantized


@dataclass
class Fixture:
    baseline: Path
    imported: Path
    source: Path
    expected_dense: int
    expected_expert: int
    bf16_dense: int
    bf16_expert: int
    native_matrices: int
    moe: bool
    int4: bool = False
    family: str = "qwen"
    hybrid_dense_release: int = 0


def gpt_oss_tensors() -> tuple[dict, dict, list[str]]:
    model_config = {
        "model_type": "gpt_oss", "vocab_size": VOCABULARY, "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE, "num_hidden_layers": 2,
        "num_local_experts": 2, "experts_per_token": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "sliding_window": 8, "initial_context_length": 32, "max_position_embeddings": 32,
        "rope_theta": 150000.0, "rope_scaling": {"factor": 1.0, "beta_slow": 1.0, "beta_fast": 32.0},
        "attention_bias": True, "rms_norm_eps": 1e-5, "swiglu_limit": 0.25,
    }
    tensors = {"model.embed_tokens.weight": ("BF16", plain((VOCABULARY, HIDDEN), 1, 1 / 32)),
               "model.norm.weight": ("BF16", bfloat16(np.ones(HIDDEN))),
               "lm_head.weight": ("BF16", plain((VOCABULARY, HIDDEN), 2))}
    native = ["lm_head.weight"]
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for role in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            tensors[prefix + role] = ("BF16", bfloat16(np.ones(HIDDEN)))
        for role, rows in (("q", HIDDEN), ("k", HIDDEN // 2), ("v", HIDDEN // 2), ("o", HIDDEN)):
            name = prefix + f"self_attn.{role}_proj.weight"
            tensors[name] = ("BF16", plain((rows, HIDDEN), 10 + layer))
            native.append(name)
            tensors[name[:-6] + "bias"] = ("BF16", plain((rows,), 20 + layer, 1 / 1024))
        tensors[prefix + "self_attn.sinks"] = ("BF16", bfloat16(np.arange(4) / 16))
        embedding = float32(tensors["model.embed_tokens.weight"][1])
        tensors[prefix + "mlp.router.weight"] = ("BF16", bfloat16(embedding[[1, 2]]))
        tensors[prefix + "mlp.router.bias"] = ("BF16", bfloat16(np.zeros(2)))
        for expert in range(2):
            for role in ("gate", "up", "down"):
                name = prefix + f"mlp.experts.{expert}.{role}_proj.weight"
                tensors[name] = ("BF16", plain((INTERMEDIATE, HIDDEN), 30 + expert))
                if not (expert == 1 and role == "up"):
                    native.append(name)
                tensors[name[:-6] + "bias"] = ("BF16", plain((HIDDEN,), 40 + expert, 1 / 1024))
    return model_config, tensors, native


def deepseek_tensors() -> tuple[dict, dict, list[str]]:
    model_config = {
        "model_type": "deepseek_v4", "vocab_size": VOCABULARY, "hidden_size": HIDDEN,
        "moe_intermediate_size": INTERMEDIATE, "num_hidden_layers": 2,
        "n_routed_experts": 2, "num_experts_per_tok": 1, "n_shared_experts": 1,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128,
        "q_lora_rank": 256, "qk_rope_head_dim": 64, "o_groups": 2, "o_lora_rank": 256,
        "sliding_window": 8, "max_position_embeddings": 32, "original_max_position_embeddings": 32,
        "num_hash_layers": 0, "index_n_heads": 2, "index_head_dim": 128, "index_topk": 8,
        "hc_mult": 4, "hc_sinkhorn_iters": 4, "compress_ratios": [0, 0],
        "expert_dtype": "bfloat16", "scoring_func": "sqrtsoftplus", "rms_norm_eps": 1e-6,
        "hc_eps": 1e-6, "swiglu_limit": 0.25, "routed_scaling_factor": 1.5,
    }
    tensors = {"embed.weight": ("BF16", plain((VOCABULARY, HIDDEN), 1, 1 / 32)),
               "norm.weight": ("BF16", bfloat16(np.ones(HIDDEN))),
               "head.weight": ("BF16", plain((VOCABULARY, HIDDEN), 2)),
               "hc_head_fn": ("F32", np.zeros((4, 4 * HIDDEN), np.float32)),
               "hc_head_base": ("F32", np.zeros(4, np.float32)),
               "hc_head_scale": ("F32", np.ones(1, np.float32))}
    native = ["head.weight"]
    for layer in range(2):
        prefix = f"layers.{layer}."
        for block in ("attn", "ffn"):
            tensors[prefix + f"hc_{block}_fn"] = ("F32", np.zeros((24, 4 * HIDDEN), np.float32))
            tensors[prefix + f"hc_{block}_base"] = ("F32", np.arange(24, dtype=np.float32) * np.float32(0.0001234567))
            tensors[prefix + f"hc_{block}_scale"] = ("F32", np.asarray([0.999997, 1.000003, 0.33333334], np.float32))
            tensors[prefix + f"{block}_norm.weight"] = ("BF16", bfloat16(np.ones(HIDDEN)))
        tensors[prefix + "attn.q_norm.weight"] = ("BF16", bfloat16(np.ones(256)))
        tensors[prefix + "attn.kv_norm.weight"] = ("BF16", bfloat16(np.ones(128)))
        tensors[prefix + "attn.attn_sink"] = ("F32", np.arange(4, dtype=np.float32) * np.float32(0.12345679))
        embedding = float32(tensors["embed.weight"][1])
        tensors[prefix + "ffn.gate.weight"] = ("BF16", bfloat16(embedding[[1, 2]]))
        tensors[prefix + "ffn.gate.bias"] = ("F32", np.asarray([-0.0001234567, 0.0001234567], np.float32))
        matrices = {"attn.wq_a": (256, HIDDEN), "attn.wq_b": (512, 256),
                    "attn.wkv": (128, HIDDEN), "attn.wo_a": (512, 256), "attn.wo_b": (HIDDEN, 512)}
        for expert in ("shared_experts", "experts.0", "experts.1"):
            matrices.update({f"ffn.{expert}.{role}": (INTERMEDIATE, HIDDEN) for role in ("w1", "w3", "w2")})
        for index, (role, shape) in enumerate(matrices.items()):
            name = prefix + role + ".weight"
            tensors[name] = ("BF16", plain(shape, 50 + index + layer))
            if not role.endswith("experts.1.w3"):
                native.append(name)
    return model_config, tensors, native


def dspark_tensors() -> tuple[dict, dict, list[str]]:
    model_config, tensors, native = deepseek_tensors()
    model_config.update(dspark_target_layer_ids=[0], dspark_block_size=2, dspark_noise_token_id=0,
                        dspark_markov_rank=256)
    for name, tensor in list(tensors.items()):
        if name.startswith("layers.0."):
            target = "mtp.0." + name[len("layers.0."):]
            tensors[target] = (tensor[0], tensor[1].copy())
            if name in native:
                native.append(target)
    for role in ("norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
        tensors["mtp.0." + role] = tensors[role]
    tensors["mtp.0.main_norm.weight"] = ("BF16", bfloat16(np.ones(HIDDEN)))
    for role, shape in (("main_proj", (HIDDEN, HIDDEN)),
                        ("markov_head.markov_w1", (VOCABULARY, 256)),
                        ("markov_head.markov_w2", (VOCABULARY, 256))):
        name = "mtp.0." + role + ".weight"
        tensors[name] = ("BF16", plain(shape, 80))
        native.append(name)
    codes = np.arange(HIDDEN + 256, dtype=np.int32).reshape(1, -1) % 16
    tensors["mtp.0.confidence_head.proj.weight"] = ("BF16", bfloat16((codes - 8).astype(np.float32) / 512))
    return model_config, tensors, native


def qwen38_tensors() -> tuple[dict, dict, list[str]]:
    model_config = config(True)
    model_config["model_type"] = "qwen4_exp"
    text = model_config["text_config"]
    text.update(model_type="qwen4_exp_text", hc_count=4, hc_lowrank=256, output_gate_type="sigmoid",
                indexer_n_heads=2, indexer_head_dim=64, indexer_kv_heads=1,
                indexer_compress_ratio=4, indexer_budget=16, ple_embed_dim=HIDDEN, ple_conv_kernel_size=2,
                ngram_size=2, heads_per_ngram=1, ngram_vocab_size_base=2,
                make_ngram_vocab_size_divisible_by=4, split_ngram_parts=2, ple_layer_ids=[1],
                eos_token_id=7, mtp_num_hidden_layers=0)
    tensors, native = base_tensors(True)
    del tensors[PREFIX + "norm.weight"]
    for layer in range(2):
        for role in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            del tensors[PREFIX + f"layers.{layer}." + role]
    for prefix in ([PREFIX + "hyper_connection_mixer."]
                   + [PREFIX + f"layers.{layer}.{block}_hyper_connection."
                      for layer in range(2) for block in ("attn", "mlp")]):
        tensors[prefix + "hc_norm.weight"] = ("BF16", bfloat16(np.zeros(4 * HIDDEN)))
        for role, shape in (("input_mix_weight_down", (256, 4 * HIDDEN)),
                            ("input_mix_weight_up", (4 * HIDDEN, 256))):
            name = prefix + role + ".weight"
            tensors[name] = ("BF16", plain(shape, 70, 1 / 4096))
            native.append(name)
        if not prefix.endswith("hyper_connection_mixer."):
            name = prefix + "block_inject_weight.weight"
            tensors[name] = ("BF16", plain((4, 4 * HIDDEN), 71, 1 / 4096))
    prefix = PREFIX + "layers.1.self_attn.indexer."
    tensors[prefix + "index_qk_proj.weight"] = ("BF16", plain((192, HIDDEN), 72))
    native.append(prefix + "index_qk_proj.weight")
    for role in ("q_layernorm", "k_layernorm"):
        tensors[prefix + role + ".weight"] = ("BF16", bfloat16(np.zeros(64)))
    prefix = PREFIX + "layers.0.ple."
    for role, shape in (("key_proj", (4 * HIDDEN, HIDDEN)), ("value_proj", (HIDDEN, HIDDEN))):
        tensors[prefix + role + ".weight"] = ("BF16", plain(shape, 73))
        native.append(prefix + role + ".weight")
    for role in ("norm_key", "norm_query", "norm_conv"):
        tensors[prefix + role + ".weight"] = ("BF16", bfloat16(np.zeros(4 * HIDDEN)))
    tensors[prefix + "conv1d.weight"] = ("BF16", plain((4 * HIDDEN, 1, 2), 74, 1 / 8))
    for role, values in (("layer_multipliers", [1, 3]), ("ngram_heads_vocab_sizes", [2]), ("ngram_heads_offsets", [0])):
        tensors[prefix + "ple_embedding." + role] = ("I64", np.asarray(values, np.int64))
    for shard in range(2):
        tensors[prefix + f"ple_embedding.ngram_embedding.shard_{shard}.weight"] = ("BF16", plain((2, 256), 75 + shard))
    return model_config, tensors, native


def family_tensors(family: str) -> tuple[dict, dict, list[str]]:
    return {"gpt_oss": gpt_oss_tensors, "deepseek_v4": deepseek_tensors,
            "deepseek_dspark": dspark_tensors, "qwen4_exp": qwen38_tensors}[family]()


def add_test_gguf_metadata(writer, architecture: str, model: dict) -> None:
    """Write the small model metadata set required by direct GGUF loading."""
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


def gguf_writer_with_test_metadata(model: dict):
    real_writer = gguf.GGUFWriter

    def writer(path, architecture, *extra, **kwargs):
        result = real_writer(path, architecture, *extra, **kwargs)
        add_test_gguf_metadata(result, architecture, model)
        return result

    return writer


def check_preserved_auxiliary(loaded: dict, reference: dict,
                              normalized_f32_names: set[str] | None = None) -> None:
    normalized_f32_names = normalized_f32_names or set()
    for name, (dtype, values) in reference.items():
        if name in normalized_f32_names:
            assert dtype == "BF16" and loaded[name].dtype == "BF16", name
            np.testing.assert_array_equal(loaded[name].read(), values, err_msg=name)
            continue
        if dtype in ("F32", "I64"):
            if (dtype == "F32" and loaded[name].dtype == "BF16"
                    and re.search(r"\.self_attn\.[qkvo]_proj\.weight$", name)):
                # The general importer stores ordinary GGUF F32 projections
                # as BF16; preserve and verify that explicit normalization.
                np.testing.assert_array_equal(loaded[name].read(), bfloat16(values))
                continue
            assert loaded[name].dtype == dtype, name
            np.testing.assert_array_equal(loaded[name].read().view(np.uint8), values.view(np.uint8))


def weight_sizes(tensors: dict, packed_sizes: dict[str, int]) -> tuple[int, int]:
    dense = expert = 0
    for name, (dtype, values) in tensors.items():
        size = packed_sizes.get(name, values.size * {"BF16": 2, "F32": 4, "I64": 8}[dtype])
        if ".experts." in name:
            expert += size
        else:
            dense += size
    return dense, expert


def mapped_ple_dense_bytes(path: Path) -> int:
    """Count PLE payload bytes excluded from resident totals as file-mapped.

    BF16 shards and native GGUF Qn_K block shards can remain mapped. F16/F32
    PLE sources are converted to owned BF16 and remain fully charged. MXFP4
    codes and scales are decoded into owned arrays, so they remain charged.
    Projection weights, scales and metadata keep their independent byte expectations.
    """
    if not path.is_dir():
        return 0
    model_config = json.loads((path / "config.json").read_text())
    if model_config.get("model_type") != "qwen4_exp":
        return 0
    text = model_config.get("text_config", model_config)
    layers = set(text["ple_layer_ids"])
    shards = text["split_ngram_parts"]
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    columns = text["ple_embed_dim"] // heads
    pattern = re.escape(PREFIX) + r"layers\.(0|[1-9][0-9]*)\.ple\.ple_embedding\.ngram_embedding\.shard_(0|[1-9][0-9]*)\.weight"
    total = 0
    tensors = importer.scan_safetensors(path)
    native_blocks = {tag: (elements, block_bytes)
                     for tag, elements, block_bytes in importer.GGUF_NATIVE_FORMATS.values()}
    for name, info in tensors.items():
        match = re.fullmatch(pattern, name)
        if not (match and int(match[1]) + 1 in layers and int(match[2]) < shards):
            continue
        if (info.dtype == "BF16" and len(info.shape) == 2
                and info.shape[0] > 0 and info.shape[1] == columns):
            total += info.size
        elif info.dtype == "U8":
            shape_info = tensors.get(name + ".__qnk_shape")
            if (shape_info is None or shape_info.dtype != "I64"
                    or shape_info.shape != (3,)):
                continue
            tag, rows, width = (int(value) for value in shape_info.read())
            encoding = native_blocks.get(tag)
            if encoding is None:
                continue
            block_elements, block_bytes = encoding
            expected_size = rows * (width // block_elements) * block_bytes
            if rows > 0 and width == columns and width % block_elements == 0 and info.size == expected_size:
                total += info.size
    return total


def hybrid_dense_host_release(tensors: dict, packed_sizes: dict[str, int], *,
                              family: str = "qwen", fp8: bool = False,
                              query_selection_bytes: int = 0,
                              q8_0_dense_bytes: dict[str, int] | None = None) -> int:
    """Independently count the fixture weights owned by resident Vulkan ops.

    INT4/Qn_K storage remains resident on host. Ordinary unquantized Shared
    Expert router gates are separate projections and can be reclaimed. The
    Qwen4 continuous chain owns its router gates inside the Expert operation,
    so their host copies remain, while QSA owns and releases query/key norms.
    Native Standard GQA Attention owns its pre-attention norm, sinks, and
    (for Qwen) query/key norms for each complete Vulkan layer, as well as its
    unquantized projection copies. Q8_0 dense Linear operators release their
    raw GGML payloads; embeddings, Expert banks, and Qwen dense-FFN Expert
    chains retain theirs.
    Its HC/PLE/indexer weights also retain their host storage despite GPU
    execution. FP8
    reclaims codes and scales of the output head, standard/latent Attention,
    and dense FFN projections. GatedDeltaNet's five projections retain their
    CPU prefill fallback, and FP8 Shared Experts retain their CPU chain. The
    Other norms, biases, routing weights and auxiliaries remain in the exact
    retained-byte expectation. Qwen4 continuous execution rejects FP8.
    """
    released = query_selection_bytes if fp8 else 0
    qwen_family = family in ("qwen", "qwen_dense", "qwen_moe")
    if qwen_family or family == "gpt_oss":
        tensor_prefix = PREFIX if qwen_family else "model."
        layer_pattern = re.compile(re.escape(tensor_prefix) + r"layers\.(\d+)\.self_attn\.[qkvo]_proj\.weight$")
        layer_ids = {int(match[1]) for name in tensors if (match := layer_pattern.fullmatch(name))}
        for layer_id in sorted(layer_ids):
            prefix = tensor_prefix + f"layers.{layer_id}.self_attn."
            projection_names = [prefix + f"{role}_proj.weight" for role in ("q", "k", "v", "o")]
            if not all(name in tensors for name in projection_names):
                continue
            query_shape = tensors[projection_names[0]][1].shape
            key_shape = tensors[projection_names[1]][1].shape
            value_shape = tensors[projection_names[2]][1].shape
            if (len(query_shape) != 2 or len(key_shape) != 2 or len(value_shape) != 2
                    or query_shape[0] <= key_shape[0] or key_shape[0] != value_shape[0]):
                continue
            # These fixture layers exercise native Attention when at least
            # one projection uses the packed GGML/INT4 representation.
            if not any(name in packed_sizes for name in projection_names):
                continue
            for name in projection_names:
                if name not in packed_sizes:
                    released += tensors[name][1].nbytes
            owned_constants = [tensor_prefix + f"layers.{layer_id}.input_layernorm.weight"]
            if qwen_family:
                owned_constants.extend((prefix + "q_norm.weight", prefix + "k_norm.weight"))
            else:
                owned_constants.append(prefix + "sinks")
            for name in owned_constants:
                if name in tensors and name not in packed_sizes:
                    released += tensors[name][1].nbytes

    for name, raw_bytes in (q8_0_dense_bytes or {}).items():
        if name.endswith("embed_tokens.weight") or name in ("model.embed_tokens.weight", "embed.weight"):
            continue
        if ".experts." in name:
            continue
        if qwen_family and re.search(r"\.mlp\.(?:gate|up|down)_proj\.weight$", name):
            continue
        released += raw_bytes

    for name, (dtype, values) in tensors.items():
        if not fp8:
            qsa_owned_norm = family == "qwen4_exp" and bool(re.search(r"\.self_attn\.[qk]_norm\.weight$", name))
            separate_router_gate = family != "qwen4_exp" and name.endswith("mlp.shared_expert_gate.weight")
            if (qsa_owned_norm or separate_router_gate) and name not in packed_sizes:
                released += values.size * {"BF16": 2, "F32": 4}[dtype]
            continue
        if name not in packed_sizes:
            continue
        output_head = name in ("lm_head.weight", "head.weight")
        standard_attention = bool(re.search(r"\.self_attn\.[qkvo]_proj\.weight$", name))
        latent_attention = bool(re.search(r"\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", name))
        dense_ffn = bool(re.search(r"\.mlp\.(?:gate|up|down)_proj\.weight$", name))
        if output_head or standard_attention or latent_attention or dense_ffn:
            released += packed_sizes[name]
    return released


def prepare_int4(root: Path, method: str, moe: bool, family: str = "qwen") -> Fixture:
    if family == "qwen":
        model_config = config(moe)
        reference, native_names = base_tensors(moe)
    else:
        model_config, reference, native_names = family_tensors(family)
    # Common HF AWQ/GPTQ assets have float16 dtype and unquantized F16 tensors.
    source_tensors = {name: ("F16", float32(values).astype(np.float16)) if dtype == "BF16" else (dtype, values)
                      for name, (dtype, values) in reference.items()}
    specs, packed_sizes = {}, {}
    for number, name in enumerate(native_names):
        outputs, inputs = reference[name][1].shape
        rng = np.random.default_rng(100 + number)
        codes = rng.integers(0, 16, size=(inputs, outputs), dtype=np.int32)
        groups = (inputs + GROUP_SIZE - 1) // GROUP_SIZE
        zeros = rng.integers(1, 17 if method == "gptq" else 16, size=(groups, outputs), dtype=np.int32)
        scales = rng.integers(1, 4, size=(groups, outputs)).astype(np.float32) / 512
        # Interleaved act-order groups are deliberately different from k // group_size.
        indices = (np.arange(inputs) % groups).astype(np.int32) if method == "gptq" else None
        logical_groups = indices if indices is not None else np.arange(inputs) // GROUP_SIZE
        expected = np.empty((outputs, inputs), dtype=np.float32)
        for output in range(outputs):
            expected[output] = (codes[:, output] - zeros[logical_groups, output]) * scales[logical_groups, output]
        reference[name] = ("BF16", bfloat16(expected))
        # Dyadic scales keep all reference values exactly representable in BF16.
        np.testing.assert_array_equal(float32(reference[name][1]), expected)
        prefix = name[:-len("weight")]
        del source_tensors[name]
        source_tensors[prefix + "qweight"] = ("I32", pack_columns(codes, True) if method == "awq" else pack_gptq(codes))
        source_tensors[prefix + "qzeros"] = ("I32", pack_columns(zeros, True) if method == "awq" else pack_columns(zeros - 1))
        source_tensors[prefix + "scales"] = ("F16", scales.astype(np.float16))
        if indices is not None:
            source_tensors[prefix + "g_idx"] = ("I32", indices)
        resident = outputs * ((inputs + 1) // 2) + groups * outputs * 5
        if indices is not None:
            resident += inputs * 4
            if family in ("qwen", "qwen4_exp") and name.endswith("self_attn.q_proj.weight"):
                resident += inputs * 4  # Query/gate split each owns its group-index vector.
        packed_sizes[name] = resident
        specs[name] = (codes, scales, zeros, indices)
    dense_bytes, expert_bytes = weight_sizes(reference, packed_sizes)
    bf16_dense, bf16_expert = weight_sizes(reference, {})
    baseline = root / "baseline"
    assets(baseline, model_config)
    baseline_tensors = dict(reference)
    for name in native_names:
        baseline_tensors[name] = ("F32", float32(reference[name][1]))
    write_safetensors(baseline / "model.safetensors", baseline_tensors)
    source = root / "source"
    source_config = copy.deepcopy(model_config)
    source_config.get("text_config", source_config)["dtype"] = "float16"
    quant = {"quant_method": method, "bits": 4, "group_size": GROUP_SIZE}
    if method == "awq":
        quant.update(version="GEMM", zero_point=True)
    else:
        quant.update(desc_act=True, sym=False, checkpoint_format="gptq")
    source_config["quantization_config"] = quant
    assets(source, source_config)
    write_safetensors(source / "model.safetensors", source_tensors)
    imported = root / "imported"
    importer.import_package(source, imported)
    loaded = importer.scan_safetensors(imported)
    check_preserved_auxiliary(loaded, reference)
    for name, (codes, scales, zeros, indices) in specs.items():
        assert loaded[name].dtype == "U8", name
        np.testing.assert_array_equal(loaded[name + ".__int4_shape"].read(), [codes.shape[1], codes.shape[0], GROUP_SIZE])
        packed = loaded[name].read()
        actual_codes = np.empty((codes.shape[1], packed.shape[1] * 2), dtype=np.uint8)
        actual_codes[:, ::2], actual_codes[:, 1::2] = packed & 15, packed >> 4
        np.testing.assert_array_equal(actual_codes[:, :codes.shape[0]], codes.T)
        np.testing.assert_array_equal(loaded[name + ".__int4_scales"].read(), scales.T)
        np.testing.assert_array_equal(loaded[name + ".__int4_zeros"].read(), zeros.T)
        if indices is not None:
            np.testing.assert_array_equal(loaded[name + ".__int4_g_idx"].read(), indices)
    return Fixture(baseline, imported, source, dense_bytes, expert_bytes, bf16_dense, bf16_expert, len(specs), moe, True, family,
                   hybrid_dense_host_release(reference, packed_sizes, family=family))


def quantize_tiny_confidence(fixture: Fixture) -> Fixture:
    # HF AWQ/GPTQ pack output zero points eight at a time. This one-row
    # auxiliary head uses the runtime's canonical INT4 schema directly.
    name = "mtp.0.confidence_head.proj.weight"
    loaded = importer.scan_safetensors(fixture.imported)
    info = loaded[name]
    values = float32(info.read())
    codes = np.rint(values * 512 + 8).astype(np.uint8)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    columns = values.shape[1]
    metadata = {name + ".__int4_shape": ("I64", np.asarray([1, columns, GROUP_SIZE], np.int64)),
                name + ".__int4_scales": ("F32", np.full((1, columns // GROUP_SIZE), 1 / 512, np.float32)),
                name + ".__int4_zeros": ("U8", np.full((1, columns // GROUP_SIZE), 8, np.uint8))}
    write_safetensors(info.path, {name: ("U8", packed)})
    filename = "native-confidence-metadata.safetensors"
    write_safetensors(fixture.imported / filename, metadata)
    index_path = fixture.imported / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({key: filename for key in metadata})
    size = packed.nbytes + sum(array.nbytes for _, array in metadata.values())
    index["metadata"]["total_size"] += size - info.size
    index_path.write_text(json.dumps(index))
    resident = packed.nbytes + (columns // GROUP_SIZE) * 5
    return replace(fixture, expected_dense=fixture.expected_dense - info.size + resident,
                   native_matrices=fixture.native_matrices + 1)


def gguf_name(name: str) -> str:
    names = {PREFIX + "embed_tokens.weight": "token_embd.weight",
             PREFIX + "norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"}
    if name in names:
        return names[name]
    layer, role = name[len(PREFIX + "layers."):].split(".", 1)
    roles = {
        "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
        "linear_attn.in_proj_qkv.weight": "attn_qkv.weight", "linear_attn.in_proj_z.weight": "attn_gate.weight",
        "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
        "linear_attn.out_proj.weight": "ssm_out.weight", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
        "linear_attn.A_log": "ssm_a", "linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.norm.weight": "ssm_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight", "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
        "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
        "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
        "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    }
    return f"blk.{layer}." + roles[role]


def qnk_blocks(rows: int, bits: int, seed: int, columns: int = 256) -> np.ndarray:
    """Construct finite, nonzero GGML blocks without an importer quantizer."""
    rng = np.random.default_rng(seed)
    assert columns % 256 == 0
    block_bytes = {2: 84, 3: 110, 4: 144, 5: 176, 6: 210, 8: 292}[bits]
    raw = np.empty((rows, columns // 256 * block_bytes), dtype=np.uint8)
    for block in range(columns // 256):
        if bits == 2:
            values = np.zeros((rows, 84), dtype=np.uint8)
            values[:, :16] = 1  # subgroup scale 1, min 0 for each of 16 groups.
            values[:, 80:82] = np.asarray([1 / 512], dtype="<f2").view(np.uint8)
            for row in range(rows):
                codes = (np.arange(256, dtype=np.uint16) + seed + row + block) % 4
                packed = np.zeros(64, dtype=np.uint8)
                for index, code in enumerate(codes):
                    group, within = divmod(index % 128, 32)
                    half, lane = divmod(within, 16)
                    byte_index = index // 128 * 32 + half * 16 + lane
                    packed[byte_index] |= np.uint8(code << (group * 2))
                values[row, 16:80] = packed
        elif bits == 3:
            values = np.zeros((rows, 110), dtype=np.uint8)
            values[:, :32] = 0xff  # enable the low two-bit q values.
            values[:, 96:104] = 0x11  # low four bits encode scale 1.
            values[:, 104:108] = 0xaa  # remaining two scale bits encode 1.
            values[:, 108:110] = np.asarray([1 / 512], dtype="<f2").view(np.uint8)
            for row in range(rows):
                codes = (np.arange(256, dtype=np.uint16) + seed + row + block) % 4
                packed = np.zeros(64, dtype=np.uint8)
                for index, code in enumerate(codes):
                    group, within = divmod(index % 128, 32)
                    half, lane = divmod(within, 16)
                    byte_index = index // 128 * 32 + half * 16 + lane
                    packed[byte_index] |= np.uint8(code << (group * 2))
                values[row, 32:96] = packed
        elif bits == 4:
            values = np.zeros((rows, 144), dtype=np.uint8)
            for row in range(rows):
                values[row, :2] = np.asarray([(row % 3 + 1) / 1024], dtype="<f2").view(np.uint8)
                values[row, 2:4] = np.asarray([1 / 2048], dtype="<f2").view(np.uint8)
            values[:, 4:16] = 1
            values[:, 16:] = rng.integers(0, 256, size=(rows, 128), dtype=np.uint8)
        elif bits == 5:
            values = np.zeros((rows, 176), dtype=np.uint8)
            values[:, :2] = np.asarray([1 / 512], dtype="<f2").view(np.uint8)
            values[:, 4:16] = 1  # one-bit group scales; dmin is zero.
            for row in range(rows):
                codes = (np.arange(256, dtype=np.uint16) + seed + row + block) % 32
                low = np.zeros(128, dtype=np.uint8)
                high = np.zeros(32, dtype=np.uint8)
                for group in range(4):
                    for half in range(2):
                        for lane in range(32):
                            code = int(codes[group * 64 + half * 32 + lane])
                            low[group * 32 + lane] |= (code & 15) << (half * 4)
                            high[lane] |= ((code >> 4) & 1) << (group * 2 + half)
                values[row, 16:48] = high
                values[row, 48:] = low
        else:
            if bits == 6:
                values = rng.integers(0, 256, size=(rows, 210), dtype=np.uint8)
                values[:, 192:208] = rng.integers(1, 4, size=(rows, 16), dtype=np.uint8)
                for row in range(rows):
                    values[row, 208:] = np.asarray([(row % 2 + 1) / 2048], dtype="<f2").view(np.uint8)
            else:
                values = np.zeros((rows, 292), dtype=np.uint8)
                values[:, :4] = np.asarray([1 / 512], dtype="<f4").view(np.uint8)
                for row in range(rows):
                    q = ((np.arange(256, dtype=np.int16) + seed + row + block) % 17 - 8).astype(np.int8)
                    values[row, 4:260] = q.view(np.uint8)
                    values[row, 260:] = q.reshape(16, 16).sum(axis=1, dtype=np.int16).astype("<i2").view(np.uint8)
        raw[:, block * block_bytes:(block + 1) * block_bytes] = values
    return raw


NATIVE_GGUF_FORMATS = ("Q4_0", "Q4_1", "Q5_0", "IQ4_XS")
NATIVE_GQA_GGUF_FORMATS = tuple(name for name in (
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "Q8_0",
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "IQ4_XS",
) if name in importer.GGUF_NATIVE_FORMATS)
assert len(NATIVE_GQA_GGUF_FORMATS) == 13


def q8_0_blocks(rows: int, columns: int, seed: int) -> np.ndarray:
    if columns % 32:
        raise ValueError(f"Q8_0 fixture columns are not block aligned: {columns}")
    blocks = np.zeros((rows, columns // 32, 34), dtype=np.uint8)
    blocks[..., :2] = np.asarray([1 / 512], dtype="<f2").view(np.uint8).reshape(1, 1, 2)
    codes = ((np.arange(rows * columns, dtype=np.int16).reshape(rows, columns) + seed) % 17 - 8).astype(np.int8)
    blocks[..., 2:] = codes.reshape(rows, columns // 32, 32).view(np.uint8)
    return blocks.reshape(rows, -1)


def q5_1_blocks(rows: int, columns: int, seed: int) -> np.ndarray:
    if columns % 32:
        raise ValueError(f"Q5_1 fixture columns are not block aligned: {columns}")
    block_shape = (rows, columns // 32)
    codes = ((np.arange(rows * columns, dtype=np.uint8).reshape(rows, columns) + seed) % 31 + 1)
    codes = codes.reshape(*block_shape, 32)
    blocks = np.zeros((*block_shape, 24), dtype=np.uint8)
    blocks[..., :2] = np.asarray([1 / 512], dtype="<f2").view(np.uint8)
    blocks[..., 4:8] = np.packbits(codes >> 4, axis=-1, bitorder="little")
    blocks[..., 8:] = (codes[..., :16] & 15) | (codes[..., 16:] << 4)
    return blocks.reshape(rows, -1)


def scale_gguf_block_scales(raw: np.ndarray, format_name: str, rows: int, columns: int,
                            amplitude_scale: float) -> None:
    """Scale native GGUF values without changing their dtype or packed codes."""
    if not math.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError(f"invalid native GGUF fixture amplitude scale: {amplitude_scale}")
    if amplitude_scale == 1:
        return
    _, block_elements, block_bytes = importer.GGUF_NATIVE_FORMATS[format_name]
    blocks = raw.reshape(rows, columns // block_elements, block_bytes)
    if format_name == "Q2_K":
        fields, field_type = (80, 82), "f16"
    elif format_name == "Q3_K":
        fields, field_type = (108,), "f16"
    elif format_name in ("Q4_K", "Q5_K"):
        fields, field_type = (0, 2), "f16"
    elif format_name == "Q6_K":
        fields, field_type = (208,), "f16"
    elif format_name == "Q8_K":
        fields, field_type = (0,), "f32"
    elif format_name in ("Q4_1", "Q5_1"):
        fields, field_type = (0, 2), "f16"
    else:
        fields, field_type = (0,), "f16"
    for offset in fields:
        width = 4 if field_type == "f32" else 2
        values = blocks[..., offset:offset + width].copy()
        if field_type == "f32":
            scaled = (values.view("<f4").reshape(blocks.shape[:-1]) * amplitude_scale).astype("<f4")
        else:
            scaled = (values.view("<f2").reshape(blocks.shape[:-1]).astype(np.float32)
                      * amplitude_scale).astype("<f2")
        blocks[..., offset:offset + width] = scaled.view(np.uint8).reshape(blocks.shape[:-1] + (width,))


FP4_DOUBLED_E2M1 = np.asarray([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def fp4_gguf_matrix(rows: int, columns: int, format_name: str,
                    amplitude_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build GGUF MXFP4/NVFP4 blocks and decode them from the documented ABI."""
    if not math.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError(f"invalid native GGUF fixture amplitude scale: {amplitude_scale}")
    power = round(math.log2(amplitude_scale))
    if not math.isclose(math.ldexp(1.0, power), amplitude_scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("FP4 fixture scaling must be an exactly representable power of two")
    codes = np.random.default_rng(seed).integers(0, 16, (rows, columns), dtype=np.uint8)
    if format_name == "MXFP4":
        scale_byte = 128 + power
        if scale_byte < 2 or scale_byte > 255:
            raise ValueError("MXFP4 fixture scale is outside the E8M0 range")
        blocks = codes.reshape(rows, columns // 32, 32)
        raw = np.empty((rows, columns // 32, 17), dtype=np.uint8)
        raw[..., 0] = scale_byte
        raw[..., 1:] = blocks[..., :16] | (blocks[..., 16:] << 4)
        half_scale = math.ldexp(1.0, scale_byte - 128)
        decoded = (FP4_DOUBLED_E2M1[codes] * half_scale).reshape(rows, columns)
    elif format_name == "NVFP4":
        exponent = 8 + power
        if exponent <= 0:
            mantissa = int(math.ldexp(amplitude_scale, 10))
            if mantissa < 1 or mantissa > 7 or math.ldexp(float(mantissa), -10) != amplitude_scale:
                raise ValueError("NVFP4 fixture scale is outside the UE4M3 subnormal range")
            scale_byte = mantissa
        else:
            if exponent >= 15:
                raise ValueError("NVFP4 fixture scale is outside the UE4M3 normal range")
            scale_byte = exponent << 3
        blocks = codes.reshape(rows, columns // 64, 4, 16)
        raw = np.empty((rows, columns // 64, 36), dtype=np.uint8)
        raw[..., :4] = scale_byte
        raw[..., 4:] = (blocks[..., :8] | (blocks[..., 8:] << 4)).reshape(rows, columns // 64, 32)
        half_scale = amplitude_scale
        decoded = (FP4_DOUBLED_E2M1[codes] * half_scale).reshape(rows, columns)
    else:
        raise ValueError(f"unsupported independent FP4 fixture format: {format_name}")
    return raw.reshape(rows, -1), decoded.astype(np.float32)


def quantized_gguf_matrix(values: np.ndarray, format_name: str, *,
                          amplitude_scale: float = 1.0) -> tuple[np.ndarray, Any, np.ndarray, int]:
    """Quantize a full matrix and return raw bytes with a BF16-exact GGUF decode."""
    encoding = ((39, 32, 17) if format_name == "MXFP4"
                else importer.GGUF_NATIVE_FORMATS[format_name])
    tag, block_elements, block_bytes = encoding
    source = np.asarray(values, dtype=np.float32)
    rows, columns = source.shape
    if columns % block_elements:
        raise ValueError(f"fixture matrix is not aligned for {format_name}: {source.shape}")
    dtype = getattr(gguf.GGMLQuantizationType, format_name)
    if format_name in ("MXFP4", "NVFP4"):
        seed = int(abs(float(source.sum())) * 1e5) % (1 << 32)
        raw, decoded = fp4_gguf_matrix(rows, columns, format_name, amplitude_scale, seed)
    elif format_name == "IQ4_XS":
        raw = iq4_xs_blocks(rows, columns, seed=int(abs(float(source.sum())) * 1e5) % 251)
    elif format_name in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"):
        bits = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_K": 8}[format_name]
        raw = qnk_blocks(rows, bits, seed=int(abs(float(source.sum())) * 1e5) % 251, columns=columns)
    elif format_name == "Q8_0":
        raw = q8_0_blocks(rows, columns, seed=int(abs(float(source.sum())) * 1e5) % 251)
    elif format_name == "Q5_1":
        raw = q5_1_blocks(rows, columns, seed=int(abs(float(source.sum())) * 1e5) % 251)
    elif format_name == "IQ4_NL":
        raw = iq4_nl_blocks(rows, columns)
        blocks = raw.reshape(rows, columns // 32, 18)
        lanes = np.arange(16, dtype=np.uint16)
        for row in range(rows):
            for block in range(columns // 32):
                low = ((lanes * 3 + row + block) & 15).astype(np.uint8)
                high = ((lanes * 5 + row * 2 + block * 3) & 15).astype(np.uint8)
                blocks[row, block, 2:] = low | (high << 4)
    else:
        raw = np.asarray(gguf.quantize(source, dtype)).view(np.uint8).reshape(rows, -1).copy()
    expected_bytes = rows * (columns // block_elements) * block_bytes
    if raw.nbytes != expected_bytes:
        raise ValueError(f"invalid {format_name} fixture byte count: {raw.nbytes} != {expected_bytes}")
    if format_name not in ("MXFP4", "NVFP4"):
        scale_gguf_block_scales(raw, format_name, rows, columns, amplitude_scale)
    if format_name in ("Q4_0", "Q4_1", "Q5_0"):
        blocks = raw.reshape(rows, columns // block_elements, block_bytes)
        original_d = blocks[..., :2].copy().view("<f2").reshape(rows, -1)
        normalized_d = np.copysign(np.full_like(original_d, np.float16(1 / 512)), original_d)
        blocks[..., :2] = normalized_d.astype("<f2").view(np.uint8).reshape(rows, -1, 2)
        if format_name == "Q4_1":
            # Keep the quantized codes and use an exactly representable offset.
            blocks[..., 2:4] = 0
    if format_name in ("MXFP4", "NVFP4"):
        pass
    elif format_name == "Q8_0":
        # Decode the GGML ABI independently: each 34-byte block is an F16
        # scale followed by 32 signed 8-bit values.
        blocks = raw.reshape(rows, columns // 32, 34)
        scales = blocks[..., :2].copy().view("<f2").reshape(rows, columns // 32).astype(np.float32)
        codes = blocks[..., 2:].copy().view(np.int8).astype(np.float32)
        decoded = (scales[..., None] * codes).reshape(source.shape)
        try:
            upstream = np.asarray(gguf.dequantize(raw, dtype), dtype=np.float32).reshape(source.shape)
        except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            upstream = None
        if upstream is not None:
            np.testing.assert_array_equal(upstream, decoded)
    elif format_name == "Q8_K":
        # gguf 0.19 may not expose Q8_K dequantize; this independent GGML
        # specification decode is d(float32) * qs(int8), with bsums checked
        # below. When the official decoder is available, compare it as well.
        blocks = raw.reshape(rows, columns // 256, 292)
        scales = blocks[..., :4].copy().view("<f4").reshape(rows, columns // 256)
        codes = blocks[..., 4:260].copy().view(np.int8).reshape(rows, columns // 256, 256)
        decoded = (scales[..., None] * codes.astype(np.float32)).reshape(source.shape)
        sums = blocks[..., 260:].copy().view("<i2").reshape(rows, columns // 256, 16)
        np.testing.assert_array_equal(sums, codes.reshape(rows, columns // 256, 16, 16).sum(axis=-1))
        try:
            upstream = np.asarray(gguf.dequantize(raw, dtype), dtype=np.float32).reshape(source.shape)
        except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            upstream = None
        if upstream is not None:
            np.testing.assert_array_equal(upstream, decoded)
    else:
        decoded = np.asarray(gguf.dequantize(raw, dtype), dtype=np.float32).reshape(source.shape)
    self_check = np.isfinite(decoded).all() and np.any(decoded != 0)
    if not self_check:
        raise ValueError(f"invalid or all-zero {format_name} fixture decode")
    np.testing.assert_array_equal(float32(bfloat16(decoded)), decoded)
    if format_name in ("MXFP4", "NVFP4"):
        tag = 39 if format_name == "MXFP4" else importer.GGUF_NATIVE_FORMATS[format_name][0]
    return raw, dtype, decoded, tag


def assert_imported_gguf_native(loaded: dict, name: str, raw: np.ndarray, tag: int,
                                rows: int, columns: int,
                                input_columns: np.ndarray | None = None,
                                family: str = "qwen") -> None:
    """Check the importer's lossless native representation for one GGUF matrix."""
    if tag == 39:
        source = np.asarray(raw, dtype=np.uint8).reshape(rows, columns // 32, 17)
        codes = np.empty((rows, columns // 32, 32), dtype=np.uint8)
        codes[..., :16] = source[..., 1:] & 15
        codes[..., 16:] = source[..., 1:] >> 4
        adjacent = codes.reshape(rows, columns // 2, 2)
        expected_blocks = (adjacent[..., 0] | (adjacent[..., 1] << 4)).astype(np.uint8).view(np.int8)
        scale_values = source[..., 0]
        if family == "gpt_oss" and ".mlp.experts." in name:
            role = re.search(r"\.experts\.\d+\.(gate|up|down)_proj\.weight$", name)[1]
            if role == "down":
                prefix = name[:-len(".weight")]
                blocks_name, scales_name = prefix + "_blocks", prefix + "_scales"
                expected_role_blocks, expected_role_scales = expected_blocks, scale_values
            else:
                prefix = re.sub(r"\.(?:gate|up)_proj\.weight$", ".gate_up_proj", name)
                blocks_name, scales_name = prefix + "_blocks", prefix + "_scales"
                selector = 0 if role == "gate" else 1
                expected_role_blocks = expected_blocks
                expected_role_scales = scale_values
                stored_blocks, stored_scales = loaded[blocks_name], loaded[scales_name]
                assert stored_blocks.dtype == "I8" and stored_blocks.shape[0] == rows * 2, blocks_name
                assert stored_scales.dtype == "F8_E8M0" and stored_scales.shape[0] == rows * 2, scales_name
                np.testing.assert_array_equal(stored_blocks.read()[selector::2], expected_role_blocks)
                np.testing.assert_array_equal(stored_scales.read()[selector::2], expected_role_scales)
                return
            weight, scales = loaded[blocks_name], loaded[scales_name]
        else:
            weight = loaded[name]
            scales_name = name[:-len("weight")] + "scale"
            scales = loaded[scales_name]
        assert weight.dtype == "I8" and weight.shape == expected_blocks.shape, name
        assert scales.dtype == "F8_E8M0" and scales.shape == (rows, columns // 32), scales_name
        np.testing.assert_array_equal(weight.read(), expected_blocks)
        np.testing.assert_array_equal(scales.read(), scale_values)
        if input_columns is not None:
            np.testing.assert_array_equal(loaded[name + ".__mxfp4_columns"].read(), input_columns)
        return
    weight = loaded[name]
    assert weight.dtype == "U8", name
    np.testing.assert_array_equal(weight.read().reshape(raw.shape), raw)
    np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [tag, rows, columns])
    if input_columns is not None:
        np.testing.assert_array_equal(loaded[name + ".__qnk_columns"].read(), input_columns)


def native_mxfp4_pair(rows: int, columns: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an adjacent-nibble MXFP4 pair and decode it independently."""
    if columns <= 0 or columns % 32:
        raise ValueError(f"MXFP4 fixture matrix is not block aligned: {(rows, columns)}")
    codes = ((np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns) * 7 + seed * 3 + 1)
             & 15).astype(np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(np.int8)
    scales = np.full((rows, columns // 32), 127, dtype=np.uint8)
    decoded = (FP4_DOUBLED_E2M1[codes] * np.float32(0.5)).astype(np.float32)
    return packed, scales, decoded


def native_nvfp4_blocks(rows: int, columns: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create GGUF-style UE4M3-scaled NVFP4 blocks and decode their raw bytes."""
    if columns <= 0 or columns % 64:
        raise ValueError(f"NVFP4 fixture matrix is not block aligned: {(rows, columns)}")
    codes = ((np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns) * 5 + seed * 7 + 3)
             & 15).astype(np.uint8)
    grouped = codes.reshape(rows, columns // 64, 4, 16)
    # UE4M3 exponent 8, mantissa 0 gives scale 1 under the NVFP4 ABI.
    scales = np.full((rows, columns // 64, 4), 8 << 3, dtype=np.uint8)
    raw = np.empty((rows, columns // 64, 36), dtype=np.uint8)
    raw[..., :4] = scales
    raw[..., 4:] = (grouped[..., :8] | (grouped[..., 8:] << 4)).reshape(rows, columns // 64, 32)
    decoded = FP4_DOUBLED_E2M1[codes]
    return raw.reshape(rows, -1), decoded.astype(np.float32)


def fixture_native_format(name: str, native_formats: tuple[str, ...], dense_index: int) -> str:
    expert = re.search(r"(?:mlp|ffn)\.experts\.(\d+)\.(.+)\.weight$", name)
    if expert:
        role = expert[2].split(".")[-1].removesuffix("_proj")
        role_index = 0 if role in {"gate", "w1", "up", "w3"} else 1
        layer = int(re.search(r"layers\.(\d+)\.", name)[1])
        return native_formats[(layer * 2 + role_index) % len(native_formats)]
    return native_formats[dense_index % len(native_formats)]


def logical_to_gguf_columns(keys: int, heads: int, dimension: int) -> np.ndarray:
    """Independent explicit key-major HF -> repetition-major GGUF indexing."""
    result = np.empty(heads * dimension, dtype=np.int64)
    repetitions = heads // keys
    for key in range(keys):
        for repetition in range(repetitions):
            for feature in range(dimension):
                logical = (key * repetitions + repetition) * dimension + feature
                result[logical] = (repetition * keys + key) * dimension + feature
    return result


def prepare_gguf(root: Path, moe: bool = False, reordered: bool = False, *, fp32: bool = True,
                 native_formats: tuple[str, ...] | None = None,
                 native_overrides: dict[str, str] | None = None,
                 native_scale: float = 1.0,
                 fp32_native_baseline: bool = False) -> Fixture:
    model_config = config(moe, reordered)
    reference, native_names = base_tensors(moe, reordered)
    native_names.append(PREFIX + "embed_tokens.weight")
    native_names.extend(name for name in reference if ".experts." in name and name not in native_names)
    if moe:
        native_names.extend(name for name in reference if name.endswith(("mlp.gate.weight", "mlp.shared_expert_gate.weight")))
    native_overrides = validate_native_overrides(native_names, native_overrides)
    assets(root / "assets", model_config)
    path = root / "source.gguf"
    writer = gguf.GGUFWriter(path, "qwen35moe" if moe else "qwen35")
    writer.add_block_count(2)
    specs, packed_sizes, expert_banks = {}, {}, {}
    q8_0_dense_bytes = {}
    text = model_config["text_config"]
    keys, heads, dimension = text["linear_num_key_heads"], text["linear_num_value_heads"], text["linear_value_head_dim"]
    value_order = logical_to_gguf_columns(keys, heads, dimension)
    head_order = logical_to_gguf_columns(keys, heads, 1)
    qk_width = 2 * keys * text["linear_key_head_dim"]
    dense_format_index = 0
    for index, (name, (_, values)) in enumerate(reference.items()):
        if name in native_names:
            rows, columns = values.shape
            assert columns % 256 == 0
            override = native_overrides.get(name)
            if override in ("BF16", "F32"):
                dense_values = (float32(values) if values.dtype == np.uint16 else np.asarray(values, np.float32))
                dense_values = dense_values * native_scale
                reference[name] = write_dense_gguf_override(writer, gguf_name(name), dense_values, override)
                continue
            if native_formats:
                format_name = override or fixture_native_format(name, native_formats, dense_format_index)
                if ".experts." not in name:
                    dense_format_index += 1
                raw, dtype, decoded, tag = quantized_gguf_matrix(
                    values, format_name, amplitude_scale=native_scale)
            elif override:
                raw, dtype, decoded, tag = quantized_gguf_matrix(
                    values, override, amplitude_scale=native_scale)
            else:
                bits = 6 if any(role in name for role in ("down_proj", "out_proj", "o_proj", "lm_head")) else 4
                dtype = gguf.GGMLQuantizationType.Q4_K if bits == 4 else gguf.GGMLQuantizationType.Q6_K
                raw = qnk_blocks(rows, bits, index, columns)
                decoded = gguf.dequantize(raw, dtype).reshape(rows, columns)
                tag = bits
            if getattr(dtype, "name", None) == "Q8_0":
                q8_0_dense_bytes[name] = raw.nbytes
            expected_raw = raw
            column_indices = None
            if reordered and ".linear_attn." in name:
                if name.endswith("in_proj_qkv.weight"):
                    selection = np.concatenate((np.arange(qk_width), qk_width + value_order))
                    decoded, expected_raw = decoded[selection], raw[selection]
                elif name.endswith("in_proj_z.weight"):
                    decoded, expected_raw = decoded[value_order], raw[value_order]
                elif name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
                    decoded, expected_raw = decoded[head_order], raw[head_order]
                elif name.endswith("out_proj.weight"):
                    decoded = decoded[:, value_order]
                    column_indices = np.argsort(value_order)
                    assert not np.array_equal(column_indices[column_indices], np.arange(columns))
            reference[name] = ("BF16", bfloat16(decoded))
            np.testing.assert_array_equal(float32(reference[name][1]), decoded)
            if ".experts." in name:
                layer = name[len(PREFIX + "layers."):].split(".", 1)[0]
                role = name.rsplit(".", 2)[1].removesuffix("_proj")
                expert_banks.setdefault((layer, role, dtype), []).append(raw)
            else:
                writer.add_tensor(gguf_name(name), raw, raw_dtype=dtype)
            specs[name] = (expected_raw, tag, columns, column_indices)
            packed_sizes[name] = raw.nbytes + (columns * 4 if column_indices is not None else 0)
        else:
            decoded = float32(values)
            if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
                decoded = decoded + 1
            if name.endswith("linear_attn.A_log"):
                decoded = -np.exp(decoded)
            if reordered and name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
                decoded = decoded[np.argsort(head_order)]
            if name.endswith("linear_attn.conv1d.weight"):
                decoded = decoded[:, 0, :]
                if reordered:
                    decoded = np.concatenate((decoded[:qk_width], decoded[qk_width:][np.argsort(value_order)]))
            writer.add_tensor(gguf_name(name), decoded)
    for (layer, role, dtype), values in expert_banks.items():
        writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", np.stack(values), raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    baseline = root / "baseline"
    assets(baseline, model_config)
    baseline_tensors = dict(reference)
    if fp32_native_baseline or (fp32 and not native_formats):
        for name in native_names:
            if name != PREFIX + "embed_tokens.weight":
                source_dtype, values = reference[name]
                baseline_tensors[name] = ("F32", np.asarray(values, dtype=np.float32).copy()
                                          if source_dtype == "F32" else float32(values))
    write_safetensors(baseline / "model.safetensors", baseline_tensors)
    imported = root / "imported"
    importer.import_package(path, imported, root / "assets")
    loaded = importer.scan_safetensors(imported)
    check_preserved_auxiliary(loaded, reference,
                              {name for name, dtype in native_overrides.items() if dtype == "F32"})
    for name, (raw, tag, columns, column_indices) in specs.items():
        assert_imported_gguf_native(loaded, name, raw, tag, raw.shape[0], columns, column_indices, "qwen")
    dense_bytes, expert_bytes = weight_sizes(reference, packed_sizes)
    bf16_dense, bf16_expert = weight_sizes(reference, {})
    return Fixture(baseline, imported, path, dense_bytes, expert_bytes, bf16_dense, bf16_expert, len(specs), moe,
                   hybrid_dense_release=hybrid_dense_host_release(
                       reference, packed_sizes, q8_0_dense_bytes=q8_0_dense_bytes))


def family_gguf_name(name: str, family: str) -> str:
    if family == "gpt_oss":
        roots = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
                 "lm_head.weight": "output.weight"}
        if name in roots:
            return roots[name]
        layer, role = name[len("model.layers."):].split(".", 1)
        roles = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
                 "self_attn.sinks": "attn_sinks.weight", "mlp.router.weight": "ffn_gate_inp.weight",
                 "mlp.router.bias": "ffn_gate_inp.bias"}
        for hf, gg in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "output")):
            for suffix in ("weight", "bias"):
                roles[f"self_attn.{hf}_proj.{suffix}"] = f"attn_{gg}.{suffix}"
        return f"blk.{layer}." + roles[role]
    if family == "deepseek_v4":
        roots = {"embed.weight": "token_embd.weight", "norm.weight": "output_norm.weight", "head.weight": "output.weight",
                 "hc_head_fn": "output_hc_fn.weight", "hc_head_base": "output_hc_base.weight", "hc_head_scale": "output_hc_scale.weight"}
        if name in roots:
            return roots[name]
        layer, role = name[len("layers."):].split(".", 1)
        roles = {"attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
                 "attn.q_norm.weight": "attn_q_a_norm.weight", "attn.kv_norm.weight": "attn_kv_norm.weight",
                 "attn.attn_sink": "attn_sinks.weight", "ffn.gate.weight": "ffn_gate_inp.weight",
                 "ffn.gate.bias": "exp_probs_b.bias"}
        for block in ("attn", "ffn"):
            for kind in ("fn", "base", "scale"):
                roles[f"hc_{block}_{kind}"] = f"hc_{block}_{kind}.weight"
        for hf, gg in (("wq_a", "q_a"), ("wq_b", "q_b"), ("wkv", "kv"), ("wo_a", "output_a"), ("wo_b", "output_b")):
            roles[f"attn.{hf}.weight"] = f"attn_{gg}.weight"
        for hf, gg in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            roles[f"ffn.shared_experts.{hf}.weight"] = f"ffn_{gg}_shexp.weight"
        return f"blk.{layer}." + roles[role]
    head = PREFIX + "hyper_connection_mixer."
    hc = {"hc_norm.weight": "norm.weight", "input_mix_weight_down.weight": "down.weight",
          "input_mix_weight_up.weight": "up.weight", "block_inject_weight.weight": "inject.weight"}
    if name.startswith(head):
        return "output_hc_" + hc[name[len(head):]]
    if "_hyper_connection." in name:
        layer, tail = name[len(PREFIX + "layers."):].split(".", 1)
        block, role = tail.split("_hyper_connection.")
        return f"blk.{layer}.hc_{'attn' if block == 'attn' else 'ffn'}_" + hc[role]
    if ".ple." in name or ".self_attn.indexer." in name:
        layer, role = name[len(PREFIX + "layers."):].split(".", 1)
        roles = {"ple.key_proj.weight": "ple_key.weight", "ple.value_proj.weight": "ple_value.weight",
                 "ple.norm_key.weight": "ple_norm_key.weight", "ple.norm_query.weight": "ple_norm_query.weight",
                 "ple.norm_conv.weight": "ple_norm_conv.weight", "ple.conv1d.weight": "ple_conv1d.weight",
                 "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
                 "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight"}
        return f"blk.{layer}." + roles[role]
    return gguf_name(name)


def standard_gqa_projection_names(family: str) -> dict[str, str]:
    if family == "gpt_oss":
        prefix = "model.layers.0.self_attn."
    elif family in ("qwen", "qwen35", "qwen35moe", "qwen35_reordered"):
        prefix = PREFIX + "layers.1.self_attn."
    else:
        raise ValueError(f"no Standard GQA fixture mapping for {family}")
    return {role: prefix + f"{role}_proj.weight" for role in ("q", "k", "v", "o")}


def standard_gqa_projection_types(source: Path, family: str) -> dict[str, str]:
    if gguf is None:
        raise RuntimeError("standard GQA GGUF inspection requires the optional gguf package")
    names = standard_gqa_projection_names(family)
    to_gguf_name = family_gguf_name if family == "gpt_oss" else lambda name, _family: gguf_name(name)
    expected = {role: to_gguf_name(name, family) for role, name in names.items()}
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(source).tensors}
    missing = set(expected.values()) - tensors.keys()
    if missing:
        raise AssertionError(f"GGUF Standard GQA fixture is missing projections: {sorted(missing)}")
    return {role: tensors[name].tensor_type.name for role, name in expected.items()}


def standard_gqa_mixed_overrides(family: str) -> dict[str, str]:
    names = standard_gqa_projection_names(family)
    return {names["q"]: "Q2_K", names["k"]: "BF16",
            names["v"]: "Q8_0", names["o"]: "F32"}


def validate_native_overrides(native_names: Any, overrides: dict[str, str] | None) -> dict[str, str]:
    result = dict(overrides or {})
    unknown_names = set(result) - set(native_names)
    if unknown_names:
        raise ValueError(f"native overrides refer to non-native fixture tensors: {sorted(unknown_names)}")
    supported = set(NATIVE_GQA_GGUF_FORMATS) | {"BF16", "F32"}
    invalid = {name: dtype for name, dtype in result.items() if dtype not in supported}
    if invalid:
        raise ValueError(f"unsupported native GGUF fixture overrides: {invalid}")
    return result


def write_dense_gguf_override(writer, name: str, values: np.ndarray,
                              dtype: str) -> tuple[str, np.ndarray]:
    source = np.asarray(values)
    decoded = float32(source) if source.dtype == np.uint16 else np.asarray(source, dtype=np.float32)
    if dtype == "BF16":
        stored = bfloat16(decoded)
        writer.add_tensor(name, stored, raw_dtype=gguf.GGMLQuantizationType.BF16)
        return "BF16", stored
    if dtype == "F32":
        stored = np.asarray(decoded, dtype=np.float32).copy()
        writer.add_tensor(name, stored)
        # Keep the GGUF tensor tag as F32, but account for the loader's
        # established F32 -> BF16 normalization in the fixture oracle.
        return "BF16", bfloat16(stored)
    raise ValueError(f"unsupported dense GGUF override: {dtype}")


def prepare_family_gguf(root: Path, family: str, *, canonical_dspark: bool = False, fp32: bool = True,
                         native_formats: tuple[str, ...] | None = None,
                         ple_format: str | None = None,
                         native_overrides: dict[str, str] | None = None,
                         native_scale: float = 1.0,
                         fp32_native_baseline: bool = False) -> Fixture:
    if ple_format is not None and family != "qwen4_exp":
        raise ValueError("only Qwen4Exp has a GGUF PLE table")
    model_config, reference, native_names = family_tensors(family)
    embedding = "embed.weight" if family == "deepseek_v4" else ("model." if family == "gpt_oss" else PREFIX) + "embed_tokens.weight"
    native = set(native_names) | {embedding}
    native.update(name for name, (_, values) in reference.items()
                  if ".experts." in name and values.ndim == 2)
    native_overrides = validate_native_overrides(native, native_overrides)
    source_config = copy.deepcopy(model_config)
    if canonical_dspark:
        source_config.update(dspark_target_layer_ids=[0], dspark_block_size=2, dspark_noise_token_id=0,
                             dspark_markov_rank=256)
    assets(root / "assets", source_config)
    path = root / "source.gguf"
    writer = gguf.GGUFWriter(path, {"gpt_oss": "gpt-oss", "deepseek_v4": "deepseek4", "qwen4_exp": "qwen4exp"}[family])
    writer.add_block_count(2)
    specs, packed_sizes, banks = {}, {}, {}
    q8_0_dense_bytes = {}
    if family == "qwen4_exp":
        ple = PREFIX + "layers.0.ple.ple_embedding."
        for key, source in (("layer_multipliers", "layer_multipliers"), ("head_offsets", "ngram_heads_offsets"),
                            ("head_vocab_sizes", "ngram_heads_vocab_sizes")):
            writer.add_array("qwen4exp.ple." + key, reference[ple + source][1].tolist())
        ple_values = np.concatenate([
            float32(reference[ple + f"ngram_embedding.shard_{shard}.weight"][1]) for shard in range(2)])
        if ple_format:
            ple_raw, ple_dtype, decoded, tag = quantized_gguf_matrix(
                ple_values, ple_format, amplitude_scale=native_scale)
            writer.add_tensor("per_layer_token_embd.weight", ple_raw, raw_dtype=ple_dtype)
            for shard in range(2):
                name = ple + f"ngram_embedding.shard_{shard}.weight"
                begin, end = ple_values.shape[0] * shard // 2, ple_values.shape[0] * (shard + 1) // 2
                shard_raw = ple_raw[begin:end]
                shard_values = decoded[begin:end]
                reference[name] = ("BF16", bfloat16(shard_values))
                np.testing.assert_array_equal(float32(reference[name][1]), shard_values)
                specs[name] = (shard_raw, tag, shard_values.shape[0], shard_values.shape[1])
                packed_sizes[name] = shard_raw.nbytes
                native.add(name)
        else:
            writer.add_tensor("per_layer_token_embd.weight", ple_values)
    dense_format_index = 0
    for index, (name, (source_dtype, values)) in enumerate(list(reference.items())):
        if family == "qwen4_exp" and ".ple.ple_embedding." in name:
            continue
        expert = re.search(r"(?:mlp|ffn)\.experts\.(\d+)\.(.+)\.(weight|bias)$", name)
        if expert:
            layer = re.search(r"layers\.(\d+)\.", name)[1]
            role = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down", "w1": "gate", "w3": "up", "w2": "down"}[expert[2]]
            bank_name = f"blk.{layer}.ffn_{role}_exps.{expert[3]}"
        if name in native:
            rows, columns = values.shape
            override = native_overrides.get(name)
            if override in ("BF16", "F32"):
                dense_values = (float32(values) if values.dtype == np.uint16 else np.asarray(values, np.float32))
                dense_values = dense_values * native_scale
                reference[name] = write_dense_gguf_override(writer, family_gguf_name(name, family),
                                                             dense_values, override)
                continue
            if native_formats:
                format_name = override or fixture_native_format(name, native_formats, dense_format_index)
                if ".experts." not in name:
                    dense_format_index += 1
                raw, dtype, decoded, tag = quantized_gguf_matrix(
                    values, format_name, amplitude_scale=native_scale)
            elif override:
                raw, dtype, decoded, tag = quantized_gguf_matrix(
                    values, override, amplitude_scale=native_scale)
            else:
                bits = 6 if any(role in name for role in ("down_proj", ".w2.", "out_proj", "o_proj", "wo_b", "head.weight")) else 4
                dtype = gguf.GGMLQuantizationType.Q4_K if bits == 4 else gguf.GGMLQuantizationType.Q6_K
                raw = qnk_blocks(rows, bits, index, columns)
                decoded = gguf.dequantize(raw, dtype).reshape(rows, columns)
                tag = bits
            if getattr(dtype, "name", None) == "Q8_0":
                q8_0_dense_bytes[name] = raw.nbytes
            reference[name] = ("BF16", bfloat16(decoded))
            np.testing.assert_array_equal(float32(reference[name][1]), decoded)
            specs[name] = (raw, tag, rows, columns)
            packed_sizes[name] = raw.nbytes
            if expert:
                banks.setdefault((bank_name, dtype), []).append(raw)
            elif name.endswith("self_attn.indexer.index_qk_proj.weight"):
                layer = re.search(r"layers\.(\d+)\.", name)[1]
                writer.add_tensor(f"blk.{layer}.indexer.q_proj.weight", raw[:128], raw_dtype=dtype)
                writer.add_tensor(f"blk.{layer}.indexer.k_proj.weight", raw[128:], raw_dtype=dtype)
            else:
                writer.add_tensor(family_gguf_name(name, family), raw, raw_dtype=dtype)
        else:
            decoded = float32(values) if source_dtype == "BF16" else values
            if family == "qwen4_exp":
                if (name.endswith("norm.weight") or name.endswith(("ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight"))) and not name.endswith("linear_attn.norm.weight"):
                    decoded = decoded + 1
                if name.endswith("linear_attn.A_log"):
                    decoded = -np.exp(decoded)
                if name.endswith("conv1d.weight"):
                    decoded = decoded[:, 0, :]
            if expert:
                banks.setdefault((bank_name, None), []).append(decoded)
            else:
                writer.add_tensor(family_gguf_name(name, family), decoded)
    for (name, dtype), values in banks.items():
        writer.add_tensor(name, np.stack(values), **({"raw_dtype": dtype} if dtype is not None else {}))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    baseline = root / "baseline"
    assets(baseline, model_config)
    baseline_tensors = dict(reference)
    if fp32_native_baseline or (fp32 and not native_formats and ple_format is None):
        for name in native:
            if name != embedding:
                source_dtype, values = reference[name]
                baseline_tensors[name] = ("F32", np.asarray(values, dtype=np.float32).copy()
                                          if source_dtype == "F32" else float32(values))
    write_safetensors(baseline / "model.safetensors", baseline_tensors)
    imported = root / "imported"
    importer.import_package(path, imported, root / "assets")
    loaded = importer.scan_safetensors(imported)
    check_preserved_auxiliary(loaded, reference,
                              {name for name, dtype in native_overrides.items() if dtype == "F32"})
    if canonical_dspark:
        imported_config = json.loads((imported / "config.json").read_text())
        assert not any(name.startswith("dspark_") for name in imported_config)
        assert not any(name.startswith("mtp.") for name in loaded)
    for name, (raw, tag, rows, columns) in specs.items():
        assert_imported_gguf_native(loaded, name, raw, tag, rows, columns, family=family)
    dense, expert = weight_sizes(reference, packed_sizes)
    bf16_dense, bf16_expert = weight_sizes(reference, {})
    return Fixture(baseline, imported, path, dense, expert, bf16_dense, bf16_expert, len(specs), True, False, family,
                   hybrid_dense_host_release(reference, packed_sizes, family=family,
                                             q8_0_dense_bytes=q8_0_dense_bytes))


def prepare_dspark_fp4(root: Path, format_name: str) -> Fixture:
    """Build a native FP4 DSpark fixture and its independently decoded baseline."""
    if format_name not in ("MXFP4", "NVFP4"):
        raise ValueError(f"unsupported DSpark FP4 format: {format_name}")
    model_config, tensors, native_names = dspark_tensors()
    native = set(native_names)
    if format_name == "MXFP4":
        # DeepSeek selects the interleaved path from w1, so its w1/w3 pairs
        # must use one layout across both target and draft Experts.
        native.update(name for name, (_, values) in tensors.items()
                      if ".ffn.experts." in name and name.endswith(".weight") and values.ndim == 2)
    source = root / "source"
    assets(source, model_config)
    source_tensors = {}
    reference = {name: (dtype, values.copy()) for name, (dtype, values) in tensors.items()}
    packed_sizes = {}
    native_data = {}
    for index, (name, (dtype, values)) in enumerate(tensors.items()):
        if name not in native:
            source_tensors[name] = (dtype, values)
            continue
        if not name.endswith(".weight") or values.ndim != 2:
            raise ValueError(f"unsupported DSpark native {format_name} fixture tensor: {name}")
        rows, columns = values.shape
        if format_name == "MXFP4":
            raw, scales, decoded = native_mxfp4_pair(rows, columns, index + 1)
            scale_name = name[:-len("weight")] + "scale"
            source_tensors[name] = ("I8", raw)
            source_tensors[scale_name] = ("F8_E8M0", scales)
            native_data[name] = (raw, scales)
            packed_sizes[name] = raw.nbytes + scales.nbytes
        else:
            raw, decoded = native_nvfp4_blocks(rows, columns, index + 1)
            source_tensors[name] = ("U8", raw)
            source_tensors[name + ".__qnk_shape"] = ("I64", np.asarray([296, rows, columns], dtype="<i8"))
            native_data[name] = raw
            packed_sizes[name] = raw.nbytes
        reference[name] = ("BF16", bfloat16(decoded))
    write_safetensors(source / "model.safetensors", source_tensors)
    imported = source
    if format_name == "MXFP4":
        imported = root / "imported"
        importer.import_package(source, imported)
    loaded = importer.scan_safetensors(imported)
    for name, data in native_data.items():
        if format_name == "MXFP4":
            raw, scales = data
            scale_name = name[:-len("weight")] + "scale"
            assert loaded[name].dtype == "I8", name
            assert loaded[scale_name].dtype == "F8_E8M0", scale_name
            np.testing.assert_array_equal(loaded[name].read(), raw)
            np.testing.assert_array_equal(loaded[scale_name].read(), scales)
        else:
            np.testing.assert_array_equal(loaded[name].read(), data)
            np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(),
                                          [296, *tensors[name][1].shape])
    baseline = root / "baseline"
    assets(baseline, model_config)
    write_safetensors(baseline / "model.safetensors", reference)
    dense, expert = weight_sizes(reference, packed_sizes)
    bf16_dense, bf16_expert = weight_sizes(reference, {})
    return Fixture(baseline, imported, source, dense, expert, bf16_dense, bf16_expert, len(native), True,
                   False, "deepseek_dspark")

def prepare_fp8(root: Path, family: str) -> Fixture:
    """Use independent E4M3 arithmetic and nonuniform block scales as an oracle."""
    if family in {"qwen_dense", "qwen_moe"}:
        model_config = config(family == "qwen_moe")
        reference, names = base_tensors(family == "qwen_moe")
    else:
        model_config, reference, names = family_tensors(family)
    codes = np.arange(256, dtype=np.uint16)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    positive = np.where(exponent == 0, mantissa / 512,
                        np.ldexp(1 + mantissa / 8, exponent.astype(np.int16) - 7))
    values = np.where(codes & 128, -positive, positive).astype(np.float32)
    values[(codes & 127) == 127] = np.nan
    encoding = {float(value): int(code) for code, value in enumerate(values)
                if np.isfinite(value)}
    encoding[0.0] = 0
    source_tensors = dict(reference)
    packed_sizes = {}
    # Qwen's interleaved query/gate row selection preserves FP8 codes and
    # represents source block scales per selected row, with input blocks 128.
    selection_metadata = 0
    for name in names:
        rows, columns = reference[name][1].shape
        original = float32(reference[name][1])
        packed = np.empty((rows, columns), dtype=np.uint8)
        scales = np.empty(((rows + 127) // 128, (columns + 127) // 128), dtype=np.float32)
        for row in range(0, rows, 128):
            for column in range(0, columns, 128):
                scale = np.float32(1 / (16 if (row // 128 + column // 128) % 2 else 32))
                block = original[row:row + 128, column:column + 128]
                encoded = np.asarray([encoding[float(value / scale)] for value in block.reshape(-1)], dtype=np.uint8).reshape(block.shape)
                packed[row:row + 128, column:column + 128] = encoded
                scales[row // 128, column // 128] = scale
                np.testing.assert_array_equal(values[encoded] * scale, block)
        source_tensors[name] = ("F8_E4M3", packed)
        suffix = "scale" if family == "deepseek_v4" else "weight_scale_inv"
        source_tensors[name[:-len("weight")] + suffix] = ("F32", scales)
        packed_sizes[name] = packed.nbytes + scales.nbytes
        if family.startswith("qwen") and name.endswith(".self_attn.q_proj.weight"):
            selection_metadata += rows * scales.shape[1] * 4 - scales.nbytes
    model_config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0",
                                           "weight_block_size": [128, 128], "activation_scheme": "dynamic"}
    source = root / "source"
    assets(source, model_config)
    write_safetensors(source / "model.safetensors", source_tensors)
    imported = root / "imported"
    importer.import_package(source, imported)
    loaded = importer.scan_safetensors(imported)
    for name in names:
        target = importer.canonical_tensor_name(name, model_config)
        assert loaded[target].dtype == "F8_E4M3", target
        np.testing.assert_array_equal(loaded[target].read(), source_tensors[name][1])
        scale_name = name[:-len("weight")] + ("scale" if family == "deepseek_v4" else "weight_scale_inv")
        imported_scale = importer.canonical_tensor_name(name[:-len("weight")] + "scale", model_config)
        assert loaded[imported_scale].dtype == "F32", imported_scale
        np.testing.assert_array_equal(loaded[imported_scale].read(), source_tensors[scale_name][1])
    baseline = root / "baseline"
    assets(baseline, importer.normalized_config(model_config))
    write_safetensors(baseline / "model.safetensors", reference)
    dense, expert = weight_sizes(reference, packed_sizes)
    bf16_dense, bf16_expert = weight_sizes(reference, {})
    moe = family != "qwen_dense"
    return Fixture(baseline, imported, source, dense + selection_metadata, expert, bf16_dense, bf16_expert,
                   len(names), moe, False, family,
                   hybrid_dense_host_release(reference, packed_sizes, family=family, fp8=True, query_selection_bytes=selection_metadata))


def add_deepseek_mxfp4_experts(fixture: Fixture) -> Path:
    """Keep FP8 dense projections and replace routed banks with official MXFP4 pairs."""
    source_tensors = {name: (info.dtype, info.read().copy())
                      for name, info in importer.scan_safetensors(fixture.source).items()}
    oracle_tensors = dict(source_tensors)
    blocks = np.full((INTERMEDIATE, HIDDEN // 2), 0x11, dtype=np.uint8).view(np.int8)
    scales = np.full((INTERMEDIATE, HIDDEN // 32), 127, dtype=np.uint8)
    oracle_weight = bfloat16(np.full((INTERMEDIATE, HIDDEN), 0.5, dtype=np.float32))
    for layer in range(2):
        for expert in range(2):
            for role in ("w1", "w3", "w2"):
                name = f"layers.{layer}.ffn.experts.{expert}.{role}.weight"
                scale_name = name[:-len("weight")] + "scale"
                source_tensors.pop(name, None)
                source_tensors.pop(scale_name, None)
                source_tensors[name] = ("I8", blocks)
                source_tensors[scale_name] = ("F8_E8M0", scales)
                oracle_tensors.pop(scale_name, None)
                oracle_tensors[name] = ("BF16", oracle_weight)
    write_safetensors(fixture.source / "model.safetensors", source_tensors)
    oracle = fixture.source.parent / "bf16_oracle"
    assets(oracle, json.loads((fixture.source / "config.json").read_text()))
    write_safetensors(oracle / "model.safetensors", oracle_tensors)
    return oracle

class NativeQuantizedTests(unittest.TestCase):
    worker: Path
    backend = "cpu"
    vulkan_device_index: int | None = None
    verbose_worker = False

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ncnn-native-quantized-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_hybrid_dense_release_expected_accounting(self):
        # Vary shapes and element widths so the oracle counts tensor storage,
        # rather than encoding the release difference of one model fixture.
        for dimension in (5, 13):
            with self.subTest(dimension=dimension):
                query_norm = np.zeros(dimension, np.uint16)
                key_norm = np.zeros(dimension + 2, np.float32)
                first_gate = np.zeros((1, dimension + 4), np.uint16)
                second_gate = np.zeros((1, dimension + 6), np.float32)
                tensors = {
                    PREFIX + "layers.1.self_attn.q_norm.weight": ("BF16", query_norm),
                    PREFIX + "layers.1.self_attn.k_norm.weight": ("F32", key_norm),
                    PREFIX + "layers.0.mlp.shared_expert_gate.weight": ("BF16", first_gate),
                    PREFIX + "layers.1.mlp.shared_expert_gate.weight": ("F32", second_gate),
                    PREFIX + "hyper_connection_mixer.hc_norm.weight": ("BF16", np.zeros(dimension * 4, np.uint16)),
                    PREFIX + "layers.0.ple.norm_key.weight": ("BF16", np.zeros(dimension * 4, np.uint16)),
                    PREFIX + "layers.1.self_attn.indexer.q_layernorm.weight": ("BF16", np.zeros(dimension, np.uint16)),
                    PREFIX + "layers.1.self_attn.indexer.k_layernorm.weight": ("BF16", np.zeros(dimension, np.uint16)),
                    PREFIX + "layers.1.mlp.gate.weight": ("BF16", np.zeros((2, dimension), np.uint16)),
                    "lm_head.weight": ("BF16", np.zeros((3, dimension), np.uint16)),
                }
                packed = {"lm_head.weight": dimension * 3}
                self.assertEqual(hybrid_dense_host_release(tensors, packed, family="qwen4_exp"),
                                 query_norm.nbytes + key_norm.nbytes)
                self.assertEqual(hybrid_dense_host_release(tensors, packed, family="qwen"),
                                 first_gate.nbytes + second_gate.nbytes)

    def test_mapped_ple_expected_accounting(self):
        model_config = config(True)
        model_config["model_type"] = "qwen4_exp"
        model_config["text_config"].update(ple_layer_ids=[1], split_ngram_parts=2,
                                         ple_embed_dim=256, ngram_size=3, heads_per_ngram=1)
        prefix = PREFIX + "layers.0.ple.ple_embedding.ngram_embedding."
        for dtype, storage in (("BF16", np.uint16), ("F16", np.float16), ("F32", np.float32)):
            with self.subTest(dtype=dtype):
                path = self.root / dtype
                assets(path, model_config)
                tensors = {prefix + f"shard_{shard}.weight": (dtype, np.zeros((4, 128), storage))
                           for shard in range(2)}
                # Other layers/shards and side metadata never receive PLE credit.
                tensors[prefix + "shard_2.weight"] = ("BF16", np.zeros((4, 128), np.uint16))
                tensors[PREFIX + "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"] = tensors[prefix + "shard_2.weight"]
                tensors[prefix + "shard_0.weight.__int4_scales"] = ("F32", np.ones(128, np.float32))
                write_safetensors(path / "model.safetensors", tensors)
                self.assertEqual(mapped_ple_dense_bytes(path), 2048 if dtype == "BF16" else 0)
        native_path = self.root / "MXFP4"
        assets(native_path, model_config)
        tensors = {}
        for shard in range(2):
            name = prefix + f"shard_{shard}.weight"
            tensors[name] = ("I8", np.zeros((4, 64), np.int8))
            tensors[name[:-len("weight")] + "scale"] = ("F8_E8M0", np.zeros((4, 4), np.uint8))
        write_safetensors(native_path / "model.safetensors", tensors)
        # Native MXFP4 code/scale data is loaded into owned arrays, not mapped PLE payload.
        self.assertEqual(mapped_ple_dense_bytes(native_path), 0)

    def run_worker(self, path: Path, moe: bool, expected: Fixture | None = None,
                   *, backend: str | None = None, extra_args: tuple[str, ...] = (),
                   speculative: bool = False, pure_auto: bool = False) -> dict:
        if pure_auto and backend is not None:
            raise ValueError("pure Auto worker execution cannot select a backend argument")
        selected_backend = "auto" if pure_auto else (backend or self.backend)
        worker_args = [] if pure_auto else ["--backend", selected_backend, "--cpu-packed-weights", "off"]
        if selected_backend == "hybrid":
            self.assertIsNotNone(self.vulkan_device_index)
        if not pure_auto and selected_backend in ("hybrid", "auto") and self.vulkan_device_index is not None:
            worker_args.extend(("--vulkan-device", str(self.vulkan_device_index)))
        with WorkerClient(self.worker, path, [*worker_args, *extra_args],
                          verbose=self.verbose_worker) as client:
            ready = client.ready
            if selected_backend != "auto":
                self.assertEqual(ready["resources"]["backend"], selected_backend)
            hybrid = ready["resources"]["backend"] == "hybrid"
            if hybrid:
                capabilities = ready["capabilities"]
                devices = {device["index"]: device for device in capabilities["vulkan_devices"]}
                selected_device_index = ready["resources"]["vulkan_device_index"]
                self.assertIn(selected_device_index, devices)
                self.assertTrue(devices[selected_device_index]["name"])
                if not pure_auto:
                    self.assertIsNotNone(self.vulkan_device_index)
                    self.assertEqual(selected_device_index, self.vulkan_device_index)
                device = devices[selected_device_index]
                self.assertNotEqual(device["type"], "cpu")
                self.assertGreater(device["compute_queue_count"], 0)
            self.assertEqual(ready["model"]["expert_count"], 2 if moe else 0)
            if expected is not None:
                resources = ready["resources"]
                mapped_ple = mapped_ple_dense_bytes(path)
                dense_bytes = expected.expected_dense - mapped_ple - (expected.hybrid_dense_release if hybrid else 0)
                self.assertEqual(resources["estimated_dense_bytes"], dense_bytes)
                self.assertEqual(resources["estimated_expert_bytes"], expected.expected_expert)
                self.assertLess(resources["estimated_dense_bytes"], expected.bf16_dense - mapped_ple)
                if moe:
                    self.assertLess(resources["estimated_expert_bytes"], expected.bf16_expert)
                else:
                    self.assertEqual(resources["expert_cache_bytes"], 0)
            result = {"resources": ready["resources"]}
            has_gpu_expert_cache = False
            if hybrid and moe and expected is not None:
                cache_bytes = resources["expert_gpu_cache_bytes"]
                has_gpu_expert_cache = cache_bytes > 0
                if not has_gpu_expert_cache:
                    print("SKIP optional routed GPU Expert execution assertions: no eligible Expert GPU cache "
                          "was constructed", flush=True)
            sessions = [("greedy", 0.0, 0), ("sampled", 1.0, 23)]
            if hybrid and moe and expected is not None:
                sessions.append(("reused", 1.0, 23))
            for label, temperature, seed in sessions:
                warm_cache_is_populated = False
                if label == "reused":
                    sampled_cumulative = result["sampled_stats"]["cumulative"]
                    routed_expert_entries = (ready["model"]["expert_count"]
                                             * ready["model"]["layer_count"])
                    stores = sampled_cumulative["gpu_cache_stores"]
                    evictions = sampled_cumulative["gpu_cache_evictions"]
                    warm_cache_is_populated = evictions == 0 and stores >= routed_expert_entries
                    if not warm_cache_is_populated:
                        print(f"SKIP optional warm-cache residency assertions: sampled generation stored "
                              f"{stores} of {routed_expert_entries} routed Expert entries and evicted "
                              f"{evictions}", flush=True)
                client.create_session(label, seed=seed, prefill_chunk_size=8, enable_speculative_context=speculative)
                events = []
                new_tokens = 6 if speculative else 4
                done, tokens = client.generate(label, [1, 1, 2, 2, 3, 3, 4, 4], max_new_tokens=new_tokens, temperature=temperature,
                                              stop_tokens=[], enable_speculative=speculative,
                                              speculative_confidence=0, speculative_max_draft=2,
                                              metrics_enabled=True, on_event=events.append)
                self.assertTrue(done["ok"], done)
                self.assertEqual(len(tokens), new_tokens)
                probabilities = [event["probability"] for event in events if event.get("event") == "token"]
                self.assertEqual(len(probabilities), new_tokens)
                self.assertTrue(all(math.isfinite(value) and 0 < value <= 1 for value in probabilities))
                if speculative:
                    self.assertGreater(done["stats"]["generation"]["speculative_proposals"], 0,
                                       "DSpark model loaded but executed no draft proposal")
                    self.assertGreater(done["stats"]["generation"]["speculative_draft_tokens"], 0)
                if expected is not None:
                    gpu = done["stats"]["generation_gpu"]
                    native_gqa = expected.native_matrices > 0 and expected.family in ("qwen", "gpt_oss")
                    if native_gqa:
                        self.assertEqual(gpu["attention_cpu_fallbacks"], 0,
                                         "native GQA Attention fell back to CPU")
                        if hybrid:
                            self.assertGreater(gpu["attention_blocks"], 0,
                                               "native GQA model executed no Vulkan Attention blocks")
                            self.assertGreater(gpu["attention_device_rope_fusions"], 0,
                                               "native GQA model executed no Vulkan device-RoPE")
                            self.assertGreater(gpu["attention_qkv_ring_fusions"], 0,
                                               "native GQA decode executed no KV-ring append")
                            self.assertEqual(gpu["model_compute_cpu_fallbacks"], 0,
                                             "native GQA model compute fell back to CPU")
                        else:
                            self.assertEqual(gpu["attention_blocks"], 0,
                                             "explicit CPU GQA profile unexpectedly executed Vulkan Attention")
                    if hybrid:
                        self.assertTrue(gpu["available"])
                        self.assertGreater(gpu["submit_count"], 0)
                        self.assertGreater(gpu["linear_dispatches"], 0)
                        if expected.family in ("qwen", "qwen4_exp"):
                            self.assertGreater(gpu["gated_delta_submissions"], 0,
                                               "hybrid quantized DeltaNet executed no Vulkan recurrence")
                        if expected.family == "qwen4_exp":
                            self.assertGreater(gpu["ple_blocks"], 0,
                                               "hybrid Qwen4 model executed no PLE selected rows")
                        if expected.int4:
                            self.assertGreater(gpu["int4_dispatches"], 0,
                                               "hybrid INT4 model executed no native INT4 Vulkan projection")
                        if moe and label == "sampled" and has_gpu_expert_cache:
                            self.assertGreater(done["stats"]["cumulative"]["gpu_executions"], 0,
                                               "hybrid MoE model executed no routed Expert on Vulkan")
                            self.assertGreater(done["stats"]["generation"]["gpu_routes"], 0,
                                               "hybrid MoE model routed no Expert to Vulkan")
                            self.assertEqual(done["stats"]["generation"]["cpu_fallbacks"], 0,
                                             "eligible hybrid Expert execution fell back to CPU")
                            self.assertEqual(done["stats"]["generation"]["gpu_execution_failures"], 0)
                            if expected.family == "qwen4_exp":
                                self.assertEqual(done["stats"]["generation"]["cpu_routes"], 0)
                        if moe and label == "reused" and has_gpu_expert_cache:
                            generation = done["stats"]["generation"]
                            self.assertGreater(generation["gpu_executions"], 0,
                                               "warm MoE session executed no routed Expert on Vulkan")
                            self.assertEqual(generation["cpu_fallbacks"], 0,
                                             "warm eligible Expert execution fell back to CPU")
                            if warm_cache_is_populated:
                                self.assertEqual(generation["gpu_cache_stores"], 0,
                                                 "warm MoE session uploaded an Expert again")
                                self.assertEqual(generation["gpu_cache_evictions"], 0,
                                                 "warm MoE session evicted an already resident Expert")
                    else:
                        self.assertEqual(gpu["int4_dispatches"], 0)
                result[label] = (tokens, probabilities)
                result[label + "_stats"] = done["stats"]
            if "reused" in result:
                self.assertEqual(result["reused"][0], result["sampled"][0])
                np.testing.assert_allclose(result["reused"][1], result["sampled"][1], rtol=0, atol=3e-5)
            return result

    def compare(self, fixture: Fixture, *, direct: bool, auto: bool = False,
                include_imported: bool = True) -> None:
        # Native GGUF format fixtures use a BF16 oracle after checking that the
        # official GGUF dequantization is exactly representable in BF16.
        speculative = fixture.family == "deepseek_dspark"
        baseline = self.run_worker(fixture.baseline, fixture.moe, backend="cpu", speculative=speculative)
        paths = [("imported", fixture.imported)] if include_imported else []
        if direct:
            paths.append(("direct", fixture.source))
        for label, path in paths:
            with self.subTest(path=label, native_matrices=fixture.native_matrices):
                result = self.run_worker(path, fixture.moe, fixture, speculative=speculative)
                for mode in ("greedy", "sampled"):
                    self.assertEqual(result[mode][0], baseline[mode][0])
                    np.testing.assert_allclose(result[mode][1], baseline[mode][1], rtol=0, atol=3e-5)
                print(json.dumps({"family": fixture.family, "path": label,
                                  "backend": result["resources"]["backend"], "native_matrices": fixture.native_matrices,
                                  "greedy": result["greedy"], "sampled": result["sampled"],
                                  "reference_greedy": baseline["greedy"], "reference_sampled": baseline["sampled"],
                                  "gpu": result["sampled_stats"]["generation_gpu"],
                                  "speculative": result["sampled_stats"]["generation"].get("speculative_proposals"),
                                  "warm_experts": result.get("reused_stats", {}).get("generation")}), flush=True)
        if auto:
            auto_result = self.run_worker(fixture.source, fixture.moe, fixture, pure_auto=True,
                                          speculative=speculative)
            for mode in ("greedy", "sampled"):
                self.assertEqual(auto_result[mode][0], baseline[mode][0])
                np.testing.assert_allclose(auto_result[mode][1], baseline[mode][1], rtol=0, atol=3e-5)
            print(json.dumps({"family": fixture.family, "path": "direct", "backend": auto_result["resources"]["backend"],
                              "native_matrices": fixture.native_matrices, "profile": "default-auto",
                              "gpu": auto_result["sampled_stats"]["generation_gpu"]}), flush=True)

    def compare_expert_memory_modes(self, fixture: Fixture) -> None:
        cache_args = ("--expert-gpu-cache-mb", "16") if self.backend == "hybrid" else ()
        eager = self.run_worker(fixture.imported, True, fixture,
                                extra_args=("--expert-memory", "eager"))
        # Biases and input-column maps stay resident outside the Expert cache.
        fixed_bytes = sum(info.size // 2 if name.endswith(".__qnk_columns") else info.size
                          for name, info in importer.scan_safetensors(fixture.imported).items()
                          if ".experts." in name
                          and (name.endswith(".bias") or name.endswith(".__qnk_columns")))
        deferred_fixture = replace(fixture, expected_dense=fixture.expected_dense + fixed_bytes)
        deferred = self.run_worker(fixture.imported, True, deferred_fixture,
                                   extra_args=(*cache_args, "--expert-memory", "on-demand",
                                               "--expert-cache-mb", "1"))
        resources = deferred["resources"]
        self.assertEqual(resources["selected_expert_memory"], "on-demand")
        self.assertTrue(resources["file_backed_experts"])
        self.assertEqual(resources["expert_cache_bytes"], 1024 * 1024)
        for mode in ("greedy", "sampled"):
            self.assertEqual(deferred[mode][0], eager[mode][0])
            np.testing.assert_allclose(deferred[mode][1], eager[mode][1], rtol=0, atol=3e-5)
            generation = deferred[mode + "_stats"]["generation"]
            self.assertLessEqual(generation["cache_resident_bytes"], resources["expert_cache_bytes"])
        self.assertGreater(deferred["greedy_stats"]["generation"]["io_bytes"], 0)

    def test_awq_dense(self):
        self.compare(prepare_int4(self.root, "awq", False), direct=True)

    def test_gptq_dense_act_order(self):
        self.compare(prepare_int4(self.root, "gptq", False), direct=True)

    def test_awq_moe_mixed_expert_storage(self):
        self.compare(prepare_int4(self.root, "awq", True), direct=True)

    def test_gptq_moe_mixed_expert_storage(self):
        self.compare(prepare_int4(self.root, "gptq", True), direct=True)

    def test_gpt_oss_native_int4(self):
        self.compare_native_family("gpt_oss")

    def test_deepseek_v4_native_int4(self):
        self.compare_native_family("deepseek_v4")

    def test_qwen38_flash_native_int4(self):
        self.compare_native_family("qwen4_exp")

    def test_deepseek_dspark_native_proposals(self):
        for method in ("awq", "gptq"):
            with self.subTest(method=method):
                root = self.root / method
                root.mkdir()
                fixture = prepare_int4(root, method, True, "deepseek_dspark")
                self.compare(fixture, direct=True)
                with self.subTest(confidence="canonical_int4_one_row"):
                    self.compare(quantize_tiny_confidence(fixture), direct=False)

    def test_deepseek_dspark_native_fp4_proposals(self):
        for format_name in ("MXFP4", "NVFP4"):
            with self.subTest(format=format_name):
                root = self.root / format_name.lower()
                root.mkdir()
                fixture = prepare_dspark_fp4(root, format_name)
                self.compare(fixture, direct=True, include_imported=format_name == "MXFP4")

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_deepseek_gguf_target_only(self):
        fixture = prepare_family_gguf(self.root, "deepseek_v4", canonical_dspark=True)
        self.compare(fixture, direct=False)
        baseline = self.run_worker(fixture.baseline, True, backend="cpu")
        worker_args = ["--backend", self.backend, "--cpu-packed-weights", "off"]
        if self.backend == "hybrid":
            self.assertIsNotNone(self.vulkan_device_index)
        if self.backend in ("hybrid", "auto") and self.vulkan_device_index is not None:
            worker_args.extend(("--vulkan-device", str(self.vulkan_device_index)))
        with WorkerClient(self.worker, fixture.imported, worker_args) as client:
            client.create_session("target-only", prefill_chunk_size=8, enable_speculative_context=True)
            events = []
            done, tokens = client.generate("target-only", [1, 1, 2, 2, 3, 3, 4, 4], max_new_tokens=4,
                                          stop_tokens=[], enable_speculative=True, on_event=events.append)
            self.assertTrue(done["ok"], done)
            self.assertEqual(done["stats"]["generation"]["speculative_proposals"], 0)
            self.assertEqual(tokens, baseline["greedy"][0])
            probabilities = [event["probability"] for event in events if event.get("event") == "token"]
            np.testing.assert_allclose(probabilities, baseline["greedy"][1], rtol=0, atol=3e-5)

    def compare_native_family(self, family: str):
        for method in ("awq", "gptq"):
            with self.subTest(family=family, method=method):
                root = self.root / method
                root.mkdir()
                self.compare(prepare_int4(root, method, True, family), direct=True)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_real_gguf_q4k_q6k_dense(self):
        self.compare(prepare_gguf(self.root), direct=False)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_real_gguf_q4k_q6k_moe(self):
        self.compare(prepare_gguf(self.root, True), direct=False)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_real_gguf_recurrent_columns_and_embedding(self):
        self.compare(prepare_gguf(self.root, reordered=True), direct=False)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_real_gguf_native_families(self):
        for family in ("gpt_oss", "deepseek_v4", "qwen4_exp"):
            with self.subTest(family=family):
                root = self.root / family
                root.mkdir()
                self.compare(prepare_family_gguf(root, family), direct=False)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_native_gguf_block_fixture_oracles(self):
        source = plain((2, 256), 913, 1 / 64)
        for format_name in NATIVE_GQA_GGUF_FORMATS:
            with self.subTest(format=format_name):
                raw, dtype, decoded, tag = quantized_gguf_matrix(source, format_name)
                block_elements, block_bytes = importer.GGUF_NATIVE_FORMATS[format_name][1:]
                self.assertEqual(dtype, getattr(gguf.GGMLQuantizationType, format_name))
                self.assertEqual(tag, importer.GGUF_NATIVE_FORMATS[format_name][0])
                self.assertEqual(raw.nbytes, source.shape[0] * (source.shape[1] // block_elements) * block_bytes)
                self.assertTrue(np.isfinite(decoded).all())
                self.assertTrue(np.any(decoded != 0))
                np.testing.assert_array_equal(float32(bfloat16(decoded)), decoded)
                if format_name == "Q8_K":
                    # The official 0.19 package omits Q8_K dequantize on some
                    # builds; validate the standard d(float32) * qs(int8)
                    # layout and valid signed subgroup sums independently.
                    blocks = raw.reshape(2, 1, 292)
                    scales = blocks[..., :4].copy().view("<f4").reshape(2, 1)
                    codes = blocks[..., 4:260].copy().view(np.int8).reshape(2, 1, 256)
                    expected = (scales[..., None] * codes.astype(np.float32)).reshape(source.shape)
                    sums = blocks[..., 260:].copy().view("<i2").reshape(2, 1, 16)
                    np.testing.assert_array_equal(decoded, expected)
                    np.testing.assert_array_equal(
                        sums, codes.reshape(2, 1, 16, 16).sum(axis=-1))

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_native_gqa_mixed_projection_dtypes(self):
        expected_types = {"q": "Q2_K", "k": "BF16", "v": "Q8_0", "o": "F32"}
        for family in ("qwen35moe", "gpt_oss"):
            projection_family = "qwen" if family == "qwen35moe" else family
            with self.subTest(family=family, fixture="13-format-matrix"):
                matrix_root = self.root / f"{family}-13-format-matrix"
                matrix_root.mkdir()
                matrix_fixture = (prepare_gguf(matrix_root, True, native_formats=NATIVE_GQA_GGUF_FORMATS)
                                  if family == "qwen35moe"
                                  else prepare_family_gguf(matrix_root, family, native_formats=NATIVE_GQA_GGUF_FORMATS))
                matrix_types = standard_gqa_projection_types(matrix_fixture.source, projection_family)
                self.assertEqual(len(set(matrix_types.values())), 4,
                                 f"13-format matrix did not give Q/K/V/O distinct dtypes: {matrix_types}")

            with self.subTest(family=family, fixture="Q2_K-BF16-Q8_0-F32"):
                mixed_root = self.root / f"{family}-mixed-gqa"
                mixed_root.mkdir()
                overrides = standard_gqa_mixed_overrides(projection_family)
                model = config(True) if family == "qwen35moe" else family_tensors(family)[0]
                with patch.object(gguf, "GGUFWriter", gguf_writer_with_test_metadata(model)):
                    fixture = (prepare_gguf(mixed_root, True, native_formats=NATIVE_GQA_GGUF_FORMATS,
                                            native_overrides=overrides, native_scale=1 / 64,
                                            fp32_native_baseline=True)
                               if family == "qwen35moe"
                               else prepare_family_gguf(mixed_root, family, native_formats=NATIVE_GQA_GGUF_FORMATS,
                                                        native_overrides=overrides, native_scale=1 / 64,
                                                        fp32_native_baseline=True))
                projection_types = standard_gqa_projection_types(fixture.source, projection_family)
                self.assertEqual(projection_types, expected_types)
                loaded = importer.scan_safetensors(fixture.imported)
                for role, name in standard_gqa_projection_names(projection_family).items():
                    imported_dtype = "BF16" if expected_types[role] in ("BF16", "F32") else "U8"
                    self.assertEqual(loaded[name].dtype, imported_dtype, name)
                # Reuse the CPU oracle, the selected CPU/Hybrid backend, Auto,
                # and real Vulkan Attention/fallback counters. The generic
                # importer normalizes ordinary GGUF F32 matrices to BF16, so
                # execute the original mixed-dtype GGUF source for this case.
                self.compare(fixture, direct=True, auto=True, include_imported=False)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_native_gqa_ggml_blocks_all_formats(self):
        for family in ("qwen35", "gpt_oss"):
            for format_name in NATIVE_GQA_GGUF_FORMATS:
                with self.subTest(family=family, format=format_name):
                    root = self.root / f"{family}-{format_name.lower()}"
                    root.mkdir()
                    model = config(False) if family == "qwen35" else family_tensors(family)[0]
                    with patch.object(gguf, "GGUFWriter", gguf_writer_with_test_metadata(model)):
                        fixture = (prepare_gguf(root, native_formats=(format_name,), native_scale=1 / 64,
                                               fp32_native_baseline=True)
                                   if family == "qwen35"
                                   else prepare_family_gguf(root, family, native_formats=(format_name,),
                                                            native_scale=1 / 64, fp32_native_baseline=True))
                    self.compare(fixture, direct=True, auto=self.backend == "cpu")

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_native_ggml_block_formats_all_architectures(self):
        formats = NATIVE_GGUF_FORMATS
        for family in ("qwen35", "qwen35_reordered", "qwen35moe", "gpt_oss", "deepseek_v4", "qwen4_exp"):
            with self.subTest(family=family):
                root = self.root / family
                root.mkdir()
                if family == "qwen35":
                    fixture = prepare_gguf(root, native_formats=formats)
                elif family == "qwen35_reordered":
                    fixture = prepare_gguf(root, reordered=True, native_formats=formats)
                elif family == "qwen35moe":
                    fixture = prepare_gguf(root, True, native_formats=formats)
                else:
                    fixture = prepare_family_gguf(
                        root, family, native_formats=formats,
                        ple_format="IQ4_XS" if family == "qwen4_exp" else None)
                self.compare(fixture, direct=False)
                if fixture.moe:
                    self.compare_expert_memory_modes(fixture)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_native_fp4_all_matrices_all_architectures(self):
        for format_name in ("MXFP4", "NVFP4"):
            for family in ("qwen35", "qwen35_reordered", "qwen35moe", "gpt_oss", "deepseek_v4", "qwen4_exp"):
                with self.subTest(family=family, format=format_name):
                    root = self.root / (family + "-" + format_name.lower())
                    root.mkdir()
                    # Keep decoded native matrices near the existing fixture activation scale.
                    # Their exact FP32 oracle avoids adding a second BF16 rounding source.
                    native_fixture_options = {"native_scale": 1 / 512, "fp32_native_baseline": True}
                    model = (family_tensors(family)[0] if family in ("gpt_oss", "deepseek_v4", "qwen4_exp")
                             else config(family == "qwen35moe", family == "qwen35_reordered"))
                    with patch.object(gguf, "GGUFWriter", gguf_writer_with_test_metadata(model)):
                        if family == "qwen35":
                            fixture = prepare_gguf(root, native_formats=(format_name,), **native_fixture_options)
                        elif family == "qwen35_reordered":
                            fixture = prepare_gguf(root, reordered=True, native_formats=(format_name,),
                                                    **native_fixture_options)
                        elif family == "qwen35moe":
                            fixture = prepare_gguf(root, True, native_formats=(format_name,), **native_fixture_options)
                        else:
                            fixture = prepare_family_gguf(
                                root, family, native_formats=(format_name,),
                                ple_format=format_name if family == "qwen4_exp" else None,
                                **native_fixture_options)
                    if format_name == "MXFP4" and self.backend == "cpu":
                        # MXFP4's optional CPU Q8 activation path is approximate.
                        # Use direct GGUF CPU execution as the path-loading oracle;
                        # strict FP32 parity is covered with CpuMxfp4Q8 disabled.
                        cpu_native = self.run_worker(fixture.source, fixture.moe,
                                                     fixture, backend="cpu")
                        imported = self.run_worker(fixture.imported, fixture.moe,
                                                   fixture, backend="cpu")
                        for mode in ("greedy", "sampled"):
                            self.assertEqual(imported[mode][0], cpu_native[mode][0])
                            np.testing.assert_allclose(imported[mode][1], cpu_native[mode][1],
                                                       rtol=0, atol=3e-5)
                    else:
                        self.compare(fixture, direct=False)
                    if fixture.moe:
                        self.compare_expert_memory_modes(fixture)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_qnk_expert_memory_modes(self):
        for family in ("qwen", "gpt_oss", "deepseek_v4", "qwen4_exp"):
            with self.subTest(family=family):
                root = self.root / family
                root.mkdir()
                fixture = (prepare_gguf(root, True) if family == "qwen"
                           else prepare_family_gguf(root, family))
                self.compare_expert_memory_modes(fixture)


    def test_native_fp8_all_families(self):
        for family in ("qwen_dense", "qwen_moe", "qwen4_exp", "gpt_oss", "deepseek_v4"):
            with self.subTest(family=family):
                root = self.root / family
                root.mkdir()
                fixture = prepare_fp8(root, family)
                # FP8 rounds activations too, so native CPU execution is the oracle.
                cpu_fp8 = self.run_worker(fixture.source, fixture.moe, backend="cpu")
                for label, path in (("direct", fixture.source), ("imported", fixture.imported)):
                    with self.subTest(path=label):
                        if family == "qwen4_exp" and self.backend == "hybrid":
                            with self.assertRaisesRegex(WorkerError, "continuous Qwen Vulkan execution does not support FP8"):
                                self.run_worker(path, fixture.moe)
                            continue
                        is_cpu_reference = self.backend == "cpu" and label == "direct"
                        result = cpu_fp8 if is_cpu_reference else self.run_worker(path, fixture.moe)
                        resources = result["resources"]
                        mapped_ple = mapped_ple_dense_bytes(path)
                        dense_bytes = fixture.expected_dense - mapped_ple - (fixture.hybrid_dense_release
                                                                              if resources["backend"] == "hybrid" else 0)
                        self.assertEqual(resources["estimated_dense_bytes"], dense_bytes)
                        self.assertEqual(resources["estimated_expert_bytes"], fixture.expected_expert)
                        self.assertLess(resources["estimated_dense_bytes"], fixture.bf16_dense - mapped_ple)
                        if fixture.moe:
                            self.assertLess(resources["estimated_expert_bytes"], fixture.bf16_expert)
                        if not is_cpu_reference:
                            for mode in ("greedy", "sampled"):
                                self.assertEqual(result[mode][0], cpu_fp8[mode][0])
                                np.testing.assert_allclose(result[mode][1], cpu_fp8[mode][1], rtol=0, atol=3e-5)
                        gpu = result["sampled_stats"]["generation_gpu"]
                        if resources["backend"] == "hybrid":
                            self.assertTrue(gpu["available"])
                            self.assertGreater(gpu["submit_count"], 0)
                            self.assertGreater(gpu["linear_dispatches"] + gpu["attention_blocks"], 0,
                                               "native FP8 model executed no Vulkan projection or attention")
                        self.assertEqual(gpu["int4_dispatches"], 0)
                        print(json.dumps({"family": family, "format": "native_fp8", "path": label,
                                          "backend": resources["backend"], "gpu": gpu,
                                          "dense_bytes": resources["estimated_dense_bytes"],
                                          "expert_bytes": resources["estimated_expert_bytes"]}), flush=True)

    def test_deepseek_fp8_dense_mxfp4_expert_memory_modes(self):
        if self.backend != "cpu":
            self.skipTest("mixed FP8/MXFP4 memory mode regression runs with --backend cpu")
        root = self.root / "deepseek-fp8-mxfp4"
        root.mkdir()
        fixture = prepare_fp8(root, "deepseek_v4")
        oracle = add_deepseek_mxfp4_experts(fixture)

        def execute(model_path: Path, memory_mode: str) -> tuple[list[int], list[float], dict]:
            with WorkerClient(self.worker, model_path,
                              ["--backend", "cpu", "--cpu-packed-weights", "off",
                               "--expert-memory", memory_mode]) as client:
                self.assertEqual(client.ready["resources"]["selected_expert_memory"], memory_mode)
                client.create_session("mixed", seed=0, prefill_chunk_size=8)
                events = []
                done, tokens = client.generate("mixed", [1, 1, 2, 2, 3, 3, 4, 4],
                                               max_new_tokens=4, temperature=0.0,
                                               stop_tokens=[], enable_speculative=False,
                                               metrics_enabled=True, on_event=events.append)
                self.assertTrue(done["ok"], done)
                self.assertEqual(len(tokens), 4)
                probabilities = [event["probability"] for event in events if event.get("event") == "token"]
                self.assertEqual(len(probabilities), 4)
                self.assertTrue(all(math.isfinite(value) and 0 < value <= 1 for value in probabilities))
                return tokens, probabilities, done["stats"]["generation"]

        on_demand = execute(fixture.source, "on-demand")
        eager = execute(fixture.source, "eager")
        expected = execute(oracle, "eager")
        self.assertGreater(on_demand[2]["cache_miss"], 0,
                           "OnDemand FP8/MXFP4 generation did not load a routed Expert from file storage")
        self.assertGreater(on_demand[2]["cache_hit"], 0,
                           "OnDemand FP8/MXFP4 generation did not reuse its file-backed Expert cache entry")
        self.assertGreater(on_demand[2]["io_bytes"], 0,
                           "OnDemand FP8/MXFP4 generation read no routed Expert bytes")
        self.assertEqual(on_demand[0], eager[0])
        np.testing.assert_allclose(on_demand[1], eager[1], rtol=0, atol=3e-5)
        self.assertEqual(eager[0], expected[0])
        np.testing.assert_allclose(eager[1], expected[1], rtol=0, atol=3e-5)

    def test_hybrid_release_dense_host(self):
        if self.backend != "hybrid":
            self.skipTest("dense host release regression runs with --backend hybrid")
        methods = ["awq", "gptq"]
        if gguf is not None:
            methods.append("gguf")
        for method in methods:
            with self.subTest(method=method):
                root = self.root / method
                root.mkdir()
                fixture = (prepare_gguf(root, reordered=True) if method == "gguf"
                           else prepare_int4(root, method, False))
                baseline = self.run_worker(fixture.baseline, False, backend="cpu")
                # Direct HF F16 auxiliaries and imported Qn_K columns both
                # need host projections for fresh multi-token prefill.
                path = fixture.source if method == "awq" else fixture.imported
                result = self.run_worker(path, False, fixture, backend="hybrid")
                for mode in ("greedy", "sampled"):
                    self.assertEqual(result[mode][0], baseline[mode][0])
                    np.testing.assert_allclose(result[mode][1], baseline[mode][1], rtol=0, atol=3e-5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "auto", "hybrid"), default="cpu")
    parser.add_argument("--verbose-worker", action="store_true")
    args, test_args = parser.parse_known_args()
    NativeQuantizedTests.worker = args.worker.resolve()
    NativeQuantizedTests.backend = args.backend
    NativeQuantizedTests.vulkan_device_index = None
    NativeQuantizedTests.verbose_worker = args.verbose_worker
    if args.backend in ("hybrid", "auto"):
        # A CPU worker can report the actual device inventory without trying
        # any optional GPU operator. Hybrid skips only without an eligible
        # device; checkpoint and operator failures still fail the tests.
        with tempfile.TemporaryDirectory(prefix="ncnn-native-gpu-probe-") as temporary:
            probe = Path(temporary) / "model"
            assets(probe, config())
            weights, _ = base_tensors(False)
            write_safetensors(probe / "model.safetensors", weights)
            with WorkerClient(args.worker, probe, ["--backend", "cpu", "--cpu-packed-weights", "off"]) as client:
                devices = client.ready["capabilities"]["vulkan_devices"]
                eligible_device = next((device for device in devices
                                         if device["type"] != "cpu" and device["compute_queue_count"] > 0), None)
                if eligible_device is None and args.backend == "hybrid":
                    print("SKIP: no non-CPU Vulkan device with a compute queue")
                    return 77
                if eligible_device is not None:
                    NativeQuantizedTests.vulkan_device_index = eligible_device["index"]
    suite = (unittest.defaultTestLoader.loadTestsFromNames(test_args, module=sys.modules[__name__])
             if test_args else unittest.defaultTestLoader.loadTestsFromTestCase(NativeQuantizedTests))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
