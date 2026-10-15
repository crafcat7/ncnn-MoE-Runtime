#!/usr/bin/env python3
"""Import quantized checkpoints into native runtime Safetensors packages.

AWQ/GPTQ retain their integer codes, scales, zero points and group indices.
AWQ GEMM and legacy GPTQ use their upstream packing conventions:
https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py
https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py
GGUF decoding uses llama.cpp's optional gguf package. The Qwen inverse
transforms and family mappings follow its official converters, pinned for
review at e467e5b5e5ba43071d6151b3d881deb9300d000e:
https://github.com/ggml-org/llama.cpp/tree/e467e5b5e5ba43071d6151b3d881deb9300d000e/conversion
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


MAX_HEADER_BYTES = 128 * 1024 * 1024
AWQ_REVERSE_ORDER = np.asarray([0, 4, 1, 5, 2, 6, 3, 7])
DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "I32": "<i4",
          "I64": "<i8", "U8": "u1", "I8": "i1", "F8_E4M3": "u1", "F8_E8M0": "u1"}
ASSETS = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
          "merges.txt", "special_tokens_map.json", "added_tokens.json",
          "generation_config.json", "chat_template.jinja", "preprocessor_config.json")
QWEN_TYPES = {"qwen3_5", "qwen3_5_moe", "qwen4_exp"}
MODEL_TYPES = QWEN_TYPES | {"gpt_oss", "deepseek_v4"}


@dataclass(frozen=True)
class TensorRange:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int

    def read(self) -> np.ndarray:
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported Safetensors dtype: {self.dtype}")
        dtype = np.dtype(DTYPES[self.dtype])
        if self.size != math.prod(self.shape) * dtype.itemsize:
            raise ValueError(f"invalid tensor byte count: {self.path}")
        if self.size == 0:
            return np.empty(self.shape, dtype=dtype)
        return np.memmap(self.path, dtype=dtype, mode="r", offset=self.offset, shape=self.shape)


def scan_safetensors(source: Path) -> dict[str, TensorRange]:
    paths = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]
    tensors: dict[str, TensorRange] = {}
    for path in paths:
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated Safetensors header: {path}")
            length = struct.unpack("<Q", raw)[0]
            if length == 0 or length > MAX_HEADER_BYTES:
                raise ValueError(f"invalid Safetensors header length: {path}")
            encoded = stream.read(length)
            if len(encoded) != length:
                raise ValueError(f"truncated Safetensors header: {path}")
            header = json.loads(encoded)
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header: {path}")
        intervals = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict):
                raise ValueError(f"invalid tensor metadata: {name}")
            shape, offsets, dtype = info.get("shape"), info.get("data_offsets"), info.get("dtype")
            if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(x) is not int or x < 0 for x in offsets)
                    or offsets[1] < offsets[0] or 8 + length + offsets[1] > path.stat().st_size
                    or not isinstance(dtype, str) or name in tensors):
                raise ValueError(f"invalid or duplicate tensor metadata: {name}")
            intervals.append(tuple(offsets))
            tensors[name] = TensorRange(path, dtype, tuple(shape), 8 + length + offsets[0], offsets[1] - offsets[0])
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(f"overlapping tensor ranges: {path}")
    if not tensors:
        raise ValueError("source contains no Safetensors tensors")
    return tensors


def as_float(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (values.astype(np.uint32) << 16).view(np.float32)
    if dtype not in {"F16", "F32"}:
        raise ValueError(f"expected floating-point scales, got {dtype}")
    return np.asarray(values, dtype=np.float32)


def to_bfloat16(values: np.ndarray) -> np.ndarray:
    floating = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(floating)):
        raise ValueError("conversion produced non-finite weights")
    bits = floating.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    result = (rounded >> 16).astype("<u2")
    if np.any((result & 0x7f80) == 0x7f80):
        raise ValueError("weight overflows BF16")
    return result


def preserve_float32(name: str, config: dict[str, Any]) -> bool:
    return config.get("model_type") == "deepseek_v4" and bool(
        re.search(r"(?:^|\.)(?:hc_(?:head|attn|ffn)_(?:fn|base|scale)|attn_sink|ape)$", name)
        or name.endswith((".ffn.gate.bias", ".ffn.gate.bias_vl")))


def unpack_columns(packed: np.ndarray, awq: bool = False) -> np.ndarray:
    if packed.dtype != np.dtype("int32") or packed.ndim != 2:
        raise ValueError("packed 4-bit values must be an I32 matrix")
    values = ((packed.astype(np.uint32)[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15)
    if awq:
        values = values[..., AWQ_REVERSE_ORDER]
    return values.reshape(packed.shape[0], packed.shape[1] * 8).astype(np.int16)


def validate_quantization(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be an object")
    outer, inner = config.get("quantization_config"), text.get("quantization_config")
    if outer is not None and inner is not None and outer != inner:
        raise ValueError("conflicting nested quantization_config")
    quant = outer if outer is not None else inner
    if quant is None:
        return "plain", {}
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must be an object")
    method = str(quant.get("quant_method", "")).lower()
    if method in {"awq", "gptq"}:
        bits = quant.get("bits", quant.get("w_bit"))
        group = quant.get("group_size", quant.get("q_group_size"))
        if ("bits" in quant and "w_bit" in quant and quant["bits"] != quant["w_bit"]
                or "group_size" in quant and "q_group_size" in quant and quant["group_size"] != quant["q_group_size"]):
            raise ValueError("conflicting quantization bit/group metadata")
        if type(bits) is not int or bits != 4 or type(group) is not int or group == 0 or group < -1:
            raise ValueError("only 4-bit AWQ/GPTQ with a positive group_size or -1 is supported")
        if quant.get("backend") not in {None, "auto", "auto_trainable"}:
            raise ValueError("repacked AWQ/GPTQ backends are not supported")
        if quant.get("modules_in_block_to_quantize") or quant.get("dynamic"):
            raise ValueError("per-module quantization overrides are not supported")
        if method == "awq":
            if str(quant.get("version", "gemm")).lower() != "gemm" or quant.get("zero_point", True) is not True:
                raise ValueError("only asymmetric AWQ GEMM with explicit zero points is supported")
        else:
            if any(key in quant and quant[key] is not None and quant[key] is not False
                   for key in ("is_marlin_format", "use_marlin")):
                raise ValueError("Marlin GPTQ checkpoints are unsupported")
            if str(quant.get("checkpoint_format", "gptq")).lower() != "gptq":
                raise ValueError("only legacy GPTQ checkpoint_format=gptq is supported (not GPTQ v2/Marlin)")
            if type(quant.get("desc_act", False)) is not bool or type(quant.get("sym", True)) is not bool:
                raise ValueError("invalid GPTQ desc_act/sym metadata")
        return method, {**quant, "group_size": group}
    if method == "fp8":
        if (quant.get("fmt", "e4m3") != "e4m3" or quant.get("weight_block_size") != [128, 128]
                or quant.get("activation_scheme", "dynamic") not in {"dynamic", "static"}):
            raise ValueError("only FP8 E4M3 with 128x128 weight_scale_inv blocks is supported")
        return method, quant
    if method == "mxfp4" and config.get("model_type") == "gpt_oss":
        return method, quant
    raise ValueError(f"unsupported quant_method: {method!r}; native MXFP4/DeepSeek packages should load directly")


def repack_linear(weight: np.ndarray, zeros: np.ndarray, scales: np.ndarray,
                  method: str, group_size: int, g_idx: np.ndarray | None = None,
                  desc_act: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Repack losslessly to output-major INT4 without a floating-point weight copy."""
    if weight.dtype != np.dtype("int32") or weight.ndim != 2 or 0 in weight.shape:
        raise ValueError("packed 4-bit values must be a nonempty I32 matrix")
    if method == "awq":
        inputs, outputs = weight.shape[0], weight.shape[1] * 8
        zero_values = unpack_columns(zeros, awq=True)
    elif method == "gptq":
        inputs, outputs = weight.shape[0] * 8, weight.shape[1]
        # Legacy GPTQ's stored zero - 1 can reconstruct zero point 16.
        zero_values = unpack_columns(zeros) + 1
    else:
        raise ValueError(f"unsupported quantization method: {method}")
    group_size = inputs if group_size == -1 else group_size
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("invalid INT4 group_size")
    groups = (inputs + group_size - 1) // group_size
    if (scales.shape != (groups, outputs) or zero_values.shape != scales.shape
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
        raise ValueError("qweight/qzeros/scales shapes or values disagree with group_size")
    canonical = np.arange(inputs) // group_size
    if g_idx is None:
        if desc_act:
            raise ValueError("GPTQ desc_act requires g_idx")
    else:
        if (method != "gptq" or g_idx.dtype != np.dtype("int32") or g_idx.shape != (inputs,)
                or np.any(g_idx < 0) or np.any(g_idx >= groups)):
            raise ValueError("invalid GPTQ g_idx")
        if not np.array_equal(np.bincount(g_idx, minlength=groups), np.bincount(canonical, minlength=groups)):
            raise ValueError("GPTQ g_idx group sizes disagree with group_size")
        if not desc_act and not np.array_equal(g_idx, canonical):
            raise ValueError("noncanonical g_idx requires desc_act=true")
    packed = np.zeros((outputs, (inputs + 1) // 2), dtype=np.uint8)
    if method == "gptq":
        # Each source I32 contains four consecutive output bytes. Transposing
        # these bytes changes layout without decoding or requantizing a value.
        for pair in range(4):
            packed[:, pair::4] = ((weight.view(np.uint32) >> (pair * 8)) & 255).T
    else:
        # Bound temporary unpacked storage independently of the input dimension.
        for begin in range(0, inputs, 256):
            codes = unpack_columns(weight[begin:begin + 256], awq=True).T.astype(np.uint8)
            block = codes[:, ::2].copy()
            block[:, :codes.shape[1] // 2] |= codes[:, 1::2] << 4
            packed[:, begin // 2:begin // 2 + block.shape[1]] = block
    return (packed, np.asarray([outputs, inputs, group_size], dtype="<i8"),
            np.asarray(scales.T, dtype="<f4"), zero_values.T.astype(np.uint8),
            None if g_idx is None else g_idx.astype("<i8"))


def dequantize_fp8(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if values.dtype != np.dtype("uint8") or values.ndim < 2:
        raise ValueError("FP8 weights must have at least two dimensions")
    rows, columns = values.shape[-2:]
    if (scales.shape != (*values.shape[:-2], (rows + 127) // 128, (columns + 127) // 128)
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
        raise ValueError("invalid FP8 weight_scale_inv shape or values")
    bits = np.arange(256, dtype=np.uint8)
    exponent, mantissa = ((bits >> 3) & 15).astype(np.int16), (bits & 7).astype(np.float32)
    table = np.where(exponent == 0, mantissa * np.float32(2 ** -9),
                     np.ldexp(1 + mantissa / 8, exponent - 7))
    table = np.where(bits & 128, -table, table).astype(np.float32)
    table[(bits & 127) == 127] = np.nan
    output = table[values]
    # weight_scale_inv is the multiplier used to reconstruct the weight.
    for row in range(0, rows, 128):
        for column in range(0, columns, 128):
            output[..., row:row + 128, column:column + 128] *= scales[..., row // 128, column // 128, None, None]
    return output


def safetensor_weights(source: Path, config: dict[str, Any]) -> Iterator[tuple[str, str, np.ndarray]]:
    method, quant = validate_quantization(config)
    tensors = scan_safetensors(source)
    consumed: set[str] = set()
    quantized_count = 0
    # Source MXFP4 tensors already use the runtime's adjacent-nibble layout.
    # Validate pairs without ever materializing floating-point expert matrices.
    for name, info in sorted(tensors.items()):
        if name.endswith("_blocks"):
            scale_name = name[:-len("blocks")] + "scales"
            scale = tensors.get(scale_name)
            if (config.get("model_type") != "gpt_oss" or scale is None
                    or info.dtype != "U8" or scale.dtype != "U8"
                    or len(info.shape) != 4 or info.shape[-1] != 16
                    or scale.shape != info.shape[:-1]):
                raise ValueError(f"invalid source MXFP4 pair: {name}")
            consumed.update((name, scale_name))
            quantized_count += 1
            yield name, "U8", info.read()
            yield scale_name, "U8", scale.read()
        elif info.dtype == "I8" and name.endswith(".weight"):
            scale_name = name[:-len("weight")] + "scale"
            scale = tensors.get(scale_name)
            if (config.get("model_type") != "deepseek_v4" or scale is None
                    or scale.dtype != "F8_E8M0" or len(info.shape) != 2
                    or info.shape[1] % 16 or scale.shape != (info.shape[0], info.shape[1] // 16)):
                raise ValueError(f"invalid source MXFP4 pair: {name}")
            consumed.update((name, scale_name))
            quantized_count += 1
            yield name, "I8", info.read()
            yield scale_name, "F8_E8M0", scale.read()
    for name in sorted(tensors):
        if not name.endswith(".qweight"):
            continue
        if method not in {"awq", "gptq"}:
            raise ValueError("packed qweight requires AWQ/GPTQ quantization_config")
        prefix = name[:-len("qweight")]
        required = [name, prefix + "qzeros", prefix + "scales"]
        if any(item not in tensors for item in required) or prefix + "weight" in tensors:
            raise ValueError(f"incomplete or duplicate quantized linear: {prefix}")
        if tensors[name].dtype != "I32" or tensors[required[1]].dtype != "I32":
            raise ValueError("AWQ/GPTQ qweight and qzeros must use I32")
        scale_info = tensors[required[2]]
        scales = as_float(scale_info.read(), scale_info.dtype)
        index_name = prefix + "g_idx"
        g_idx = None
        if index_name in tensors:
            if tensors[index_name].dtype != "I32":
                raise ValueError("GPTQ g_idx must use I32")
            g_idx = tensors[index_name].read()
            consumed.add(index_name)
        packed, shape, scales, zeros, indices = repack_linear(
            tensors[name].read(), tensors[required[1]].read(), scales,
            method, quant["group_size"], g_idx, quant.get("desc_act", False))
        consumed.update(required)
        quantized_count += 1
        target = prefix + "weight"
        yield target, "U8", packed
        yield target + ".__int4_shape", "I64", shape
        yield target + ".__int4_scales", "F32", scales
        yield target + ".__int4_zeros", "U8", zeros
        if indices is not None:
            yield target + ".__int4_g_idx", "I64", indices
    for name, info in sorted(tensors.items()):
        if (name in consumed or (name.endswith(".scale") and name[:-len("scale")] + "weight" in tensors
                                and tensors[name[:-len("scale")] + "weight"].dtype == "F8_E4M3")):
            continue
        if info.dtype == "F8_E4M3":
            if method != "fp8" or not name.endswith(".weight"):
                raise ValueError(f"FP8 tensor requires a supported FP8 config: {name}")
            scale_name = next((candidate for candidate in
                               (name[:-len("weight")] + "scale", name[:-len("weight")] + "weight_scale_inv")
                               if candidate in tensors), None)
            if scale_name is None:
                raise ValueError(f"missing FP8 weight_scale_inv: {name}")
            scale_info = tensors[scale_name]
            scales = (np.exp2(scale_info.read().astype(np.float32) - 127)
                      if scale_info.dtype == "F8_E8M0" else as_float(scale_info.read(), scale_info.dtype))
            expected = (*info.shape[:-2], (info.shape[-2] + 127) // 128, (info.shape[-1] + 127) // 128)
            if (len(info.shape) != 2 or scales.shape != expected
                    or not np.all(np.isfinite(scales)) or np.any(scales <= 0)
                    or np.any((info.read() & 127) == 127)):
                raise ValueError(f"invalid FP8 weight/scale shape or values: {name}")
            consumed.add(scale_name)
            quantized_count += 1
            yield name, "F8_E4M3", info.read()
            yield name[:-len("weight")] + "scale", "F32", np.asarray(scales, dtype="<f4")
        elif name.endswith((".qzeros", ".scales", ".g_idx", ".weight_scale_inv", ".input_scale")):
            # A scale must have been consumed by the matching weight. No silent loss.
            raise ValueError(f"orphan or unsupported quantization tensor: {name}")
        elif info.dtype in {"F16", "F32"}:
            if preserve_float32(name, config) and info.dtype == "F32":
                yield name, "F32", info.read()
            else:
                yield name, "BF16", to_bfloat16(info.read())
        elif info.dtype in {"BF16", "I64"}:
            yield name, info.dtype, info.read()
        else:
            raise ValueError(f"unsupported tensor {name} ({info.dtype}); native MXFP4 packages should load directly")
    if method != "plain" and quantized_count == 0:
        raise ValueError("quantization_config is present but no supported quantized weights were found")


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be an object")
    return text


def canonical_tensor_name(name: str, config: dict[str, Any] | None = None) -> str:
    if ((config is None or config.get("model_type") in QWEN_TYPES)
            and (name.startswith("model.layers.") or name.startswith("model.hyper_connection_mixer.")
                 or name in {"model.embed_tokens.weight", "model.norm.weight"})):
        return "model.language_model." + name[len("model."):]
    return name


def gguf_tensor_name(name: str, wrapped: bool, family: str = "qwen3_5") -> str:
    prefix = "model.language_model." if wrapped else "model."
    if family == "deepseek_v4":
        roots = {"token_embd.weight": "embed.weight", "output_norm.weight": "norm.weight",
                 "output.weight": "head.weight", "output_hc_fn.weight": "hc_head_fn",
                 "output_hc_base.weight": "hc_head_base", "output_hc_scale.weight": "hc_head_scale"}
        if name in roots:
            return roots[name]
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        mapping = {
            "hc_attn_fn.weight": "hc_attn_fn", "hc_attn_base.weight": "hc_attn_base", "hc_attn_scale.weight": "hc_attn_scale",
            "hc_ffn_fn.weight": "hc_ffn_fn", "hc_ffn_base.weight": "hc_ffn_base", "hc_ffn_scale.weight": "hc_ffn_scale",
            "attn_sinks.weight": "attn.attn_sink", "attn_q_a.weight": "attn.wq_a.weight",
            "attn_q_b.weight": "attn.wq_b.weight", "attn_q_a_norm.weight": "attn.q_norm.weight",
            "attn_kv.weight": "attn.wkv.weight", "attn_kv_norm.weight": "attn.kv_norm.weight",
            "attn_output_a.weight": "attn.wo_a.weight", "attn_output_b.weight": "attn.wo_b.weight",
            "attn_compressor_ape.weight": "attn.compressor.ape", "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
            "attn_compressor_gate.weight": "attn.compressor.wgate.weight", "attn_compressor_norm.weight": "attn.compressor.norm.weight",
            "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight", "indexer.proj.weight": "attn.indexer.weights_proj.weight",
            "indexer_compressor_ape.weight": "attn.indexer.compressor.ape", "indexer_compressor_kv.weight": "attn.indexer.compressor.wkv.weight",
            "indexer_compressor_gate.weight": "attn.indexer.compressor.wgate.weight", "indexer_compressor_norm.weight": "attn.indexer.compressor.norm.weight",
            "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
            "ffn_gate_inp.weight": "ffn.gate.weight", "exp_probs_b.bias": "ffn.gate.bias",
            "exp_probs_b_vl.bias": "ffn.gate.bias_vl", "ffn_gate_tid2eid.weight": "ffn.gate.tid2eid",
            "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight", "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
            "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
        }
        if match and match[2] in mapping:
            return f"layers.{match[1]}." + mapping[match[2]]
        raise ValueError(f"unsupported DeepSeek GGUF tensor: {name}")
    global_names = {"token_embd.weight": prefix + "embed_tokens.weight",
                    "output_norm.weight": prefix + "norm.weight", "output.weight": "lm_head.weight"}
    if family == "qwen4_exp":
        for role, source in (("norm", "hc_norm"), ("down", "input_mix_weight_down"), ("up", "input_mix_weight_up")):
            global_names[f"output_hc_{role}.weight"] = prefix + "hyper_connection_mixer." + source + ".weight"
    if name in global_names:
        return global_names[name]
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"unsupported GGUF tensor: {name}")
    mapping = {
        "attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight", "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_a": "linear_attn.A_log", "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight", "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
        "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
        "ffn_gate_up_exps.weight": "mlp.experts.gate_up_proj", "ffn_down_exps.weight": "mlp.experts.down_proj",
    }
    if family == "gpt_oss":
        mapping.update({"attn_sinks.weight": "self_attn.sinks", "ffn_gate_inp.bias": "mlp.router.bias",
                        "ffn_gate_inp.weight": "mlp.router.weight"})
        for role in ("q", "k", "v", "output"):
            mapping[f"attn_{role}.bias"] = "self_attn." + ("o" if role == "output" else role) + "_proj.bias"
    if family == "qwen4_exp":
        for gg_role, hf_role in (("attn", "attn"), ("ffn", "mlp")):
            for role, source in (("norm", "hc_norm"), ("down", "input_mix_weight_down"),
                                 ("up", "input_mix_weight_up"), ("inject", "block_inject_weight")):
                mapping[f"hc_{gg_role}_{role}.weight"] = hf_role + "_hyper_connection." + source + ".weight"
        mapping.update({"indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
                        "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight"})
        for role, source in (("key", "key_proj"), ("value", "value_proj"), ("norm_key", "norm_key"),
                             ("norm_query", "norm_query"), ("norm_conv", "norm_conv"), ("conv1d", "conv1d")):
            mapping[f"ple_{role}.weight"] = "ple." + source + ".weight"
    if match[2] not in mapping:
        raise ValueError(f"unsupported GGUF tensor: {name}")
    return prefix + "layers." + match[1] + "." + mapping[match[2]]


def undo_v_reorder(values: np.ndarray, axis: int, keys: int, heads: int, dimension: int) -> np.ndarray:
    if keys <= 0 or heads % keys or values.shape[axis] != heads * dimension:
        raise ValueError("invalid Qwen GGUF V-head shape/config")
    shape = list(values.shape)
    expanded = shape[:axis] + [heads // keys, keys, dimension] + shape[axis + 1:]
    return np.swapaxes(values.reshape(expanded), axis, axis + 1).reshape(shape)


def undo_gguf_transform(name: str, values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    text = text_config(config)
    if config.get("model_type") is not None and config.get("model_type") not in QWEN_TYPES:
        return values
    if (name.endswith("norm.weight") or name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight"))) and not name.endswith("linear_attn.norm.weight"):
        values = values - 1
    if name.endswith(".ple.conv1d.weight"):
        if values.ndim != 2:
            raise ValueError("GGUF PLE convolution must be a matrix")
        return values[:, None, :]
    if ".linear_attn." not in name:
        return values
    keys, heads = text.get("linear_num_key_heads", 0), text.get("linear_num_value_heads", 0)
    kd, vd = text.get("linear_key_head_dim", 0), text.get("linear_value_head_dim", 0)
    if name.endswith(".A_log"):
        if np.any(values >= 0):
            raise ValueError("GGUF SSM A must be negative to reconstruct A_log")
        values = np.log(-values)
    if keys != heads:
        if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
            qk = 2 * keys * kd
            values = np.concatenate((values[:qk], undo_v_reorder(values[qk:], 0, keys, heads, vd)))
        elif name.endswith("in_proj_z.weight"):
            values = undo_v_reorder(values, 0, keys, heads, vd)
        elif name.endswith(("in_proj_a.weight", "in_proj_b.weight", ".A_log", ".dt_bias")):
            values = undo_v_reorder(values, 0, keys, heads, 1)
        elif name.endswith("out_proj.weight"):
            values = undo_v_reorder(values, 1, keys, heads, vd)
    if name.endswith("conv1d.weight"):
        if values.ndim != 2:
            raise ValueError("GGUF SSM convolution must be a matrix")
        values = values[:, None, :]
    return values


def validate_gguf_metadata(reader: Any, architecture: str, config: dict[str, Any]) -> tuple[int, int]:
    """Check available layout metadata before applying any inverse permutation."""
    text = text_config(config)
    target_layers = text.get("num_hidden_layers")
    canonical_mtp = text.get("mtp_num_hidden_layers", text.get("num_nextn_predict_layers", 0))
    if type(target_layers) is not int or target_layers <= 0 or type(canonical_mtp) is not int or canonical_mtp < 0:
        raise ValueError("invalid canonical layer/MTP count")
    total_field = reader.get_field(architecture + ".block_count")
    mtp_field = reader.get_field(architecture + ".nextn_predict_layers")
    total = total_field.contents() if total_field is not None else None
    source_mtp = mtp_field.contents() if mtp_field is not None else 0
    if (type(total) is not int or total <= 0 or type(source_mtp) is not int or source_mtp < 0
            or total - source_mtp != target_layers or source_mtp > canonical_mtp):
        raise ValueError("GGUF block_count/nextn_predict_layers disagree with canonical target/MTP counts")
    rope = text.get("rope_parameters", text)
    if not isinstance(rope, dict):
        raise ValueError("canonical rope_parameters must be an object")
    heads, hidden = text.get("num_attention_heads"), text.get("hidden_size")
    head_dimension = text.get("head_dim")
    if head_dimension is None and type(heads) is int and heads > 0 and type(hidden) is int and hidden % heads == 0:
        head_dimension = hidden // heads
    partial = rope.get("partial_rotary_factor", text.get("partial_rotary_factor"))
    rope_dimension = None
    if type(head_dimension) is int and type(partial) in {int, float}:
        product = head_dimension * partial
        if math.isfinite(product) and math.isclose(product, round(product), abs_tol=1e-4):
            rope_dimension = round(product)
    value_heads, value_dimension = text.get("linear_num_value_heads"), text.get("linear_value_head_dim")
    inner = value_heads * value_dimension if type(value_heads) is int and type(value_dimension) is int else None
    expected = {
        "embedding_length": hidden,
        "vocab_size": text.get("vocab_size", config.get("vocab_size")),
        "attention.head_count": heads,
        "attention.head_count_kv": text.get("num_key_value_heads"),
        "ssm.group_count": text.get("linear_num_key_heads"),
        "ssm.state_size": text.get("linear_key_head_dim"),
        "ssm.time_step_rank": value_heads,
        "ssm.inner_size": inner,
        "ssm.conv_kernel": text.get("linear_conv_kernel_dim"),
        "rope.freq_base": rope.get("rope_theta", text.get("rope_theta")),
        "rope.dimension_count": rope_dimension,
        "attention.layer_norm_rms_epsilon": text.get("rms_norm_eps"),
    }
    if architecture == "qwen35":
        expected["feed_forward_length"] = text.get("intermediate_size")
    elif architecture in {"qwen35moe", "qwen4exp"}:
        expected["expert_feed_forward_length"] = text.get("moe_intermediate_size")
        expected["expert_shared_feed_forward_length"] = text.get("shared_expert_intermediate_size")
        expected["expert_count"] = text.get("num_experts")
        expected["expert_used_count"] = text.get("num_experts_per_tok")
    elif architecture == "gpt-oss":
        expected.update({"expert_feed_forward_length": text.get("intermediate_size"),
                         "expert_count": text.get("num_local_experts"), "expert_used_count": text.get("experts_per_token"),
                         "attention.sliding_window": text.get("sliding_window")})
    elif architecture == "deepseek4":
        expected.update({"expert_feed_forward_length": text.get("moe_intermediate_size"),
                         "expert_count": text.get("n_routed_experts"), "expert_used_count": text.get("num_experts_per_tok"),
                         "attention.q_lora_rank": text.get("q_lora_rank"), "attention.output_group_count": text.get("o_groups"),
                         "attention.output_lora_rank": text.get("o_lora_rank"), "hyper_connection.count": text.get("hc_mult")})
    if architecture == "qwen4exp":
        expected.update({"hyper_connection.count": text.get("hc_count"), "hyper_connection.low_rank": text.get("hc_lowrank"),
                         "attention.indexer.head_count": text.get("indexer_n_heads"),
                         "attention.indexer.key_length": text.get("indexer_head_dim"),
                         "attention.indexer.top_k": text.get("indexer_budget"),
                         "ple.ngram_size": text.get("ngram_size"), "ple.heads_per_ngram": text.get("heads_per_ngram"),
                         "ple.conv_kernel": text.get("ple_conv_kernel_size")})
    float_keys = {"rope.freq_base", "attention.layer_norm_rms_epsilon"}
    for key, canonical in expected.items():
        field = reader.get_field(architecture + "." + key)
        if field is None:
            continue
        actual = field.contents()
        if key in float_keys:
            matches = (type(actual) in {int, float} and type(canonical) in {int, float}
                       and math.isfinite(actual) and math.isfinite(canonical)
                       and math.isclose(actual, canonical, rel_tol=1e-6, abs_tol=1e-12))
        else:
            matches = type(actual) is int and type(canonical) is int and actual == canonical
        if not matches:
            raise ValueError(f"GGUF {key} ({actual!r}) disagrees with canonical config ({canonical!r})")
    return total, source_mtp


QNK_BITS = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_K": 8}
QNK_BLOCK_BYTES = {2: 84, 3: 110, 4: 144, 5: 176, 6: 210, 8: 292}


def native_gguf_matrix(name: str, tensor: Any, shape: tuple[int, ...],
                       config: dict[str, Any], data: np.ndarray | None = None
                       ) -> Iterator[tuple[str, str, np.ndarray]]:
    """Keep GGML blocks intact and describe column transforms on input vectors."""
    bits = QNK_BITS.get(tensor.tensor_type.name)
    if bits is None or len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[1] % 256:
        return
    if (not name.endswith(".weight") and not preserve_float32(name, config)
            or name.endswith(("norm.weight", "conv1d.weight"))):
        return
    text = text_config(config)
    keys, heads = text.get("linear_num_key_heads", 0), text.get("linear_num_value_heads", 0)
    rows, columns = shape
    raw = np.asarray(tensor.data if data is None else data).view(np.uint8)
    row_bytes = columns // 256 * QNK_BLOCK_BYTES[bits]
    if raw.size != rows * row_bytes:
        raise ValueError(f"invalid GGUF Qn_K matrix byte count: {name}")
    raw = raw.reshape(rows, row_bytes)
    input_columns = None
    if ".linear_attn." in name and keys != heads:
        kd, vd = text.get("linear_key_head_dim", 0), text.get("linear_value_head_dim", 0)
        if name.endswith("in_proj_qkv.weight"):
            qk = 2 * keys * kd
            raw = np.concatenate((raw[:qk], undo_v_reorder(raw[qk:], 0, keys, heads, vd)))
        elif name.endswith("in_proj_z.weight"):
            raw = undo_v_reorder(raw, 0, keys, heads, vd)
        elif name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
            raw = undo_v_reorder(raw, 0, keys, heads, 1)
        elif name.endswith("out_proj.weight"):
            # Undoing a column permutation would cross Qn_K block boundaries.
            # Apply its inverse to input activations instead: packed input[j]
            # is logical input[input_columns[j]], so the raw blocks stay exact.
            logical_to_packed = undo_v_reorder(np.arange(columns, dtype="<i8"), 0, keys, heads, vd)
            input_columns = np.argsort(logical_to_packed).astype("<i8")
    yield name, "U8", raw.reshape(-1)
    yield name + ".__qnk_shape", "I64", np.asarray([bits, rows, columns], dtype="<i8")
    if input_columns is not None:
        yield name + ".__qnk_columns", "I64", input_columns


def gguf_mxfp4_rows(data: np.ndarray, rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """GGML separates the first/last 16 nibbles; runtime packs adjacent columns."""
    raw = np.asarray(data).view(np.uint8)
    if columns <= 0 or columns % 32 or raw.size != rows * (columns // 32) * 17:
        raise ValueError("invalid GGUF MXFP4 block shape")
    blocks = raw.reshape(rows, columns // 32, 17)
    source = blocks[..., 1:]
    adjacent = np.empty((*source.shape[:-1], 16), dtype=np.uint8)
    adjacent[..., :8] = (source[..., ::2] & 15) | ((source[..., 1::2] & 15) << 4)
    adjacent[..., 8:] = (source[..., ::2] >> 4) | (source[..., 1::2] & 240)
    return adjacent.reshape(rows, columns // 2).view(np.int8), blocks[..., 0].copy()


def gguf_float_values(tensor: Any, gguf: Any, auxiliary: bool = False) -> np.ndarray:
    shape = tuple(reversed(tensor.shape))
    if tensor.tensor_type.name not in {"F32", "F16", "BF16"} and len(shape) >= 2 and not auxiliary:
        raise ValueError(f"GGUF matrix requires native Qn_K/MXFP4 quantization: {tensor.name} ({tensor.tensor_type.name})")
    try:
        return np.asarray(gguf.dequantize(tensor.data, tensor.tensor_type), dtype=np.float32).reshape(shape)
    except (NotImplementedError, ValueError, KeyError) as error:
        raise ValueError(f"unsupported GGUF quantization: {tensor.name} ({tensor.tensor_type.name})") from error


def expert_geometry(config: dict[str, Any]) -> tuple[int, int, int]:
    text = text_config(config)
    family = config.get("model_type")
    count = text.get("num_local_experts" if family == "gpt_oss" else "n_routed_experts" if family == "deepseek_v4" else "num_experts")
    hidden = text.get("hidden_size")
    intermediate = text.get("intermediate_size" if family == "gpt_oss" else "moe_intermediate_size")
    if any(type(value) is not int or value <= 0 for value in (count, hidden, intermediate)):
        raise ValueError("GGUF experts require canonical expert count/hidden/intermediate dimensions")
    return count, hidden, intermediate


@dataclass
class GgufArchive:
    readers: list[Any]
    tensors: list[Any]
    fields: dict[str, Any]
    endianess: Any

    def get_field(self, name: str) -> Any:
        return self.fields.get(name)


def open_gguf(source: Path, gguf: Any) -> GgufArchive:
    """Open standard split packages while retaining each shard's mapped views."""
    initial = gguf.GGUFReader(str(source))
    field = initial.get_field("split.count")
    count = field.contents() if field is not None else 1
    if type(count) is not int or count <= 0:
        raise ValueError("invalid GGUF split.count")
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", source.name, re.IGNORECASE)
    if count == 1:
        if match and (int(match[2]) != 1 or int(match[3]) != 1):
            raise ValueError("GGUF split filename disagrees with split.count")
        return GgufArchive([initial], list(initial.tensors), dict(initial.fields), initial.endianess)
    if match is None or int(match[3]) != count or not 1 <= int(match[2]) <= count:
        raise ValueError("split GGUF requires standard -00001-of-000NN.gguf filenames matching split.count")
    readers = []
    architecture = None
    total = None
    fields: dict[str, Any] = {}
    tensors = []
    for index in range(count):
        path = source.with_name(f"{match[1]}-{index + 1:05d}-of-{count:05d}.gguf")
        if not path.is_file():
            raise ValueError(f"missing GGUF shard: {path.name}")
        reader = initial if path == source else gguf.GGUFReader(str(path))
        def value(key: str) -> Any:
            item = reader.get_field(key)
            return item.contents() if item is not None else None
        shard_arch = value("general.architecture")
        shard_total = value("split.tensors.count")
        if (type(value("split.no")) is not int or value("split.no") != index or value("split.count") != count
                or type(shard_total) is not int or shard_total <= 0
                or reader.endianess != gguf.GGUFEndian.LITTLE):
            raise ValueError(f"invalid GGUF split metadata: {path.name}")
        if index == 0:
            architecture, total = shard_arch, shard_total
        elif shard_arch != architecture or shard_total != total:
            raise ValueError("GGUF shards disagree on architecture or split.tensors.count")
        for key, item in reader.fields.items():
            if key.startswith(str(architecture) + ".") and key in fields and item.contents() != fields[key].contents():
                raise ValueError(f"GGUF shards disagree on layout metadata: {key}")
            fields.setdefault(key, item)
        readers.append(reader)
        tensors.extend(reader.tensors)
    if len(tensors) != total:
        raise ValueError("GGUF split.tensors.count disagrees with actual tensor count")
    return GgufArchive(readers, tensors, fields, readers[0].endianess)


def gguf_weights(source: Path, config: dict[str, Any]) -> Iterator[tuple[str, str, np.ndarray]]:
    try:
        import gguf
    except ImportError as error:
        raise ValueError("GGUF import requires the official llama.cpp Python package: pip install gguf") from error
    text = text_config(config)
    family = config.get("model_type")
    expected_architecture = {"qwen3_5": "qwen35", "qwen3_5_moe": "qwen35moe", "qwen4_exp": "qwen4exp",
                             "gpt_oss": "gpt-oss", "deepseek_v4": "deepseek4"}.get(family)
    if expected_architecture is None:
        raise ValueError(f"unsupported canonical model_type: {family!r}")
    reader = open_gguf(source, gguf)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError("big-endian GGUF is not supported")
    field = reader.get_field("general.architecture")
    architecture = field.contents() if field is not None else None
    if architecture == "dflash":
        raise ValueError("DSpark dflash GGUF contains only draft weights; provide the complete HF target+mtp package")
    if architecture != expected_architecture:
        raise ValueError("GGUF architecture disagrees with canonical model config")
    source_layers, source_mtp_layers = validate_gguf_metadata(reader, architecture, config)
    if any("hadamard" in key.lower() for key in reader.fields):
        raise ValueError("GGUF Hadamard-transformed weights are not supported")
    wrapped = isinstance(config.get("text_config"), dict)
    seen: set[str] = set()
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    if len(by_name) != len(reader.tensors):
        raise ValueError("duplicate GGUF tensor names")
    hidden, vocabulary = text.get("hidden_size"), text.get("vocab_size", config.get("vocab_size"))
    if type(hidden) is not int or hidden <= 0 or type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError("canonical GGUF import requires positive hidden_size/vocab_size")
    if "token_embd.weight" not in by_name:
        raise ValueError("GGUF has no token_embd.weight")
    for name in ("token_embd.weight", "output.weight"):
        if name in by_name and tuple(by_name[name].shape) != (hidden, vocabulary):
            raise ValueError(f"GGUF {name} shape disagrees with canonical hidden_size/vocab_size")
    tokens = reader.get_field("tokenizer.ggml.tokens")
    if tokens is not None:
        token_list = tokens.contents()
        if not isinstance(token_list, list) or len(token_list) != vocabulary:
            raise ValueError("GGUF tokenizer.ggml.tokens length disagrees with canonical vocab_size")
    target_layers = source_layers - source_mtp_layers
    skipped_mtp = 0
    consumed: set[str] = set()
    if family == "qwen4_exp":
        # llama.cpp moves these I64 tensors into metadata to avoid FP32
        # rounding. Restore their exact integer values before writing shards.
        ple_layers = text.get("ple_layer_ids", [])
        table = by_name.get("per_layer_token_embd.weight")
        if table is not None:
            parts = text.get("split_ngram_parts")
            if (not isinstance(ple_layers, list) or len(ple_layers) != 1
                    or type(ple_layers[0]) is not int or not 1 <= ple_layers[0] <= target_layers
                    or type(parts) is not int or parts <= 0):
                raise ValueError("Qwen4Exp PLE requires one canonical layer and positive split_ngram_parts")
            prefix = ("model.language_model." if wrapped else "model.") + f"layers.{ple_layers[0] - 1}.ple.ple_embedding."
            for key, source_name in (("layer_multipliers", "layer_multipliers"), ("head_offsets", "ngram_heads_offsets"),
                                     ("head_vocab_sizes", "ngram_heads_vocab_sizes")):
                field = reader.get_field("qwen4exp.ple." + key)
                values = field.contents() if field is not None else None
                if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
                    raise ValueError(f"Qwen4Exp PLE requires exact integer metadata: {key}")
                yield prefix + source_name, "I64", np.asarray(values, dtype="<i8")
            shape = tuple(reversed(table.shape))
            if len(shape) != 2 or shape[0] < parts:
                raise ValueError("invalid Qwen4Exp PLE embedding shape/shard count")
            rows, columns = shape
            bits = QNK_BITS.get(table.tensor_type.name)
            if bits is not None:
                if columns % 256:
                    raise ValueError("PLE Qn_K embedding columns must be divisible by 256")
                values = np.asarray(table.data).view(np.uint8).reshape(rows, columns // 256 * QNK_BLOCK_BYTES[bits])
            else:
                values = gguf_float_values(table, gguf)
            for shard in range(parts):
                begin, end = rows * shard // parts, rows * (shard + 1) // parts
                name = prefix + f"ngram_embedding.shard_{shard}.weight"
                native = list(native_gguf_matrix(name, table, (end - begin, columns), config, values[begin:end]))
                if native:
                    yield from native
                else:
                    yield name, "BF16", to_bfloat16(values[begin:end])
            consumed.add(table.name)
        elif ple_layers:
            raise ValueError("Qwen4Exp GGUF is missing its PLE embedding table")
    for tensor in reader.tensors:
        if tensor.name in consumed:
            continue
        block = re.match(r"blk\.(\d+)\.", tensor.name)
        if block and int(block[1]) >= target_layers:
            if int(block[1]) >= source_layers:
                raise ValueError(f"GGUF layer exceeds source block_count: {tensor.name}")
            skipped_mtp += 1
            continue
        indexer = re.fullmatch(r"blk\.(\d+)\.indexer\.(q|k)_proj\.weight", tensor.name)
        if indexer and family == "qwen4_exp":
            qname, kname = f"blk.{indexer[1]}.indexer.q_proj.weight", f"blk.{indexer[1]}.indexer.k_proj.weight"
            query, key = by_name.get(qname), by_name.get(kname)
            if query is None or key is None:
                raise ValueError("Qwen4Exp GGUF requires both indexer q/k projections")
            qs, ks = tuple(reversed(query.shape)), tuple(reversed(key.shape))
            if (len(qs) != 2 or len(ks) != 2 or qs[1] != hidden or ks[1] != hidden
                    or qs[0] != text.get("indexer_n_heads", 0) * text.get("indexer_head_dim", 0)
                    or ks[0] != text.get("indexer_head_dim")):
                raise ValueError("Qwen4Exp indexer q/k shape disagrees with canonical config")
            name = ("model.language_model." if wrapped else "model.") + f"layers.{indexer[1]}.self_attn.indexer.index_qk_proj.weight"
            qbits, kbits = QNK_BITS.get(query.tensor_type.name), QNK_BITS.get(key.tensor_type.name)
            if qbits is not None or kbits is not None:
                if qbits != kbits:
                    raise ValueError("Qwen4Exp packed indexer q/k must share a Qn_K type")
                raw = np.concatenate((np.asarray(query.data).reshape(qs[0], -1), np.asarray(key.data).reshape(ks[0], -1)))
                yield from native_gguf_matrix(name, query, (qs[0] + ks[0], hidden), config, raw)
            else:
                values = np.concatenate((gguf_float_values(query, gguf), gguf_float_values(key, gguf)))
                yield name, "BF16", to_bfloat16(values)
            consumed.update((qname, kname))
            continue
        # GPT-OSS keeps projection biases; GGUF stores gate/up separately.
        bias = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.bias", tensor.name)
        if bias:
            if family != "gpt_oss":
                raise ValueError(f"unexpected expert bias: {tensor.name}")
            count, hidden, intermediate = expert_geometry(config)
            width = hidden if bias[2] == "down" else intermediate
            values = gguf_float_values(tensor, gguf)
            if values.shape != (count, width):
                raise ValueError(f"invalid GGUF expert bias shape: {tensor.name}")
            matrix = by_name.get(f"blk.{bias[1]}.ffn_{bias[2]}_exps.weight")
            if bias[2] in {"gate", "up"} and matrix is not None and matrix.tensor_type.name == "MXFP4":
                gate_name, up_name = f"blk.{bias[1]}.ffn_gate_exps.bias", f"blk.{bias[1]}.ffn_up_exps.bias"
                gate, up = by_name.get(gate_name), by_name.get(up_name)
                if gate is None or up is None:
                    raise ValueError("GPT-OSS requires both gate/up expert biases")
                gate_values, up_values = gguf_float_values(gate, gguf), gguf_float_values(up, gguf)
                if gate_values.shape != (count, width) or up_values.shape != (count, width):
                    raise ValueError("invalid GPT-OSS gate/up expert bias shapes")
                for index in range(count):
                    combined = np.empty(width * 2, dtype=np.float32)
                    combined[::2], combined[1::2] = gate_values[index], up_values[index]
                    yield f"model.layers.{bias[1]}.mlp.experts.{index}.gate_up_proj.bias", "BF16", to_bfloat16(combined)
                consumed.update((gate_name, up_name))
                continue
            for index in range(count):
                yield f"model.layers.{bias[1]}.mlp.experts.{index}.{bias[2]}_proj.bias", "BF16", to_bfloat16(values[index])
            continue
        expert = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight", tensor.name)
        if expert:
            count, hidden, intermediate = expert_geometry(config)
            role = expert[2]
            rows, columns = ((hidden, intermediate) if role == "down" else
                             (intermediate * (2 if role == "gate_up" else 1), hidden))
            if tuple(reversed(tensor.shape)) != (count, rows, columns):
                raise ValueError(f"GGUF expert shape disagrees with canonical config: {tensor.name}")
            bits = QNK_BITS.get(tensor.tensor_type.name)
            raw = np.asarray(tensor.data)
            if tensor.tensor_type.name == "MXFP4":
                if family not in {"gpt_oss", "deepseek_v4"} or role == "gate_up":
                    raise ValueError(f"unsupported native MXFP4 expert layout: {tensor.name}")
                if family == "gpt_oss" and role in {"gate", "up"}:
                    gate_name = f"blk.{expert[1]}.ffn_gate_exps.weight"
                    up_name = f"blk.{expert[1]}.ffn_up_exps.weight"
                    gate, up = by_name.get(gate_name), by_name.get(up_name)
                    if (gate is None or up is None or gate.tensor_type.name != "MXFP4" or up.tensor_type.name != "MXFP4"
                            or tuple(gate.shape) != tuple(up.shape)):
                        raise ValueError("GPT-OSS MXFP4 requires matching gate/up expert banks")
                    consumed.update((gate_name, up_name))
                    for index in range(count):
                        gate_blocks, gate_scales = gguf_mxfp4_rows(np.asarray(gate.data)[index], rows, columns)
                        up_blocks, up_scales = gguf_mxfp4_rows(np.asarray(up.data)[index], rows, columns)
                        blocks = np.empty((rows * 2, columns // 2), dtype=np.int8)
                        scales = np.empty((rows * 2, columns // 32), dtype=np.uint8)
                        blocks[::2], blocks[1::2] = gate_blocks, up_blocks
                        scales[::2], scales[1::2] = gate_scales, up_scales
                        prefix = f"model.layers.{expert[1]}.mlp.experts.{index}.gate_up_proj"
                        yield prefix + "_blocks", "I8", blocks
                        yield prefix + "_scales", "F8_E8M0", scales
                else:
                    for index in range(count):
                        blocks, scales = gguf_mxfp4_rows(raw[index], rows, columns)
                        if family == "deepseek_v4":
                            projection = {"gate": "w1", "up": "w3", "down": "w2"}[role]
                            prefix = f"layers.{expert[1]}.ffn.experts.{index}.{projection}."
                            yield prefix + "weight", "I8", blocks
                            yield prefix + "scale", "F8_E8M0", scales
                        else:
                            prefix = f"model.layers.{expert[1]}.mlp.experts.{index}.down_proj"
                            yield prefix + "_blocks", "I8", blocks
                            yield prefix + "_scales", "F8_E8M0", scales
                continue
            if bits is not None:
                if columns % 256:
                    raise ValueError(f"GGUF Qn_K expert columns must be divisible by 256: {tensor.name}")
                row_bytes = columns // 256 * QNK_BLOCK_BYTES[bits]
                if raw.nbytes != count * rows * row_bytes:
                    raise ValueError(f"invalid GGUF Qn_K expert byte count: {tensor.name}")
                values = raw.view(np.uint8).reshape(count, rows, row_bytes)
            else:
                values = gguf_float_values(tensor, gguf).reshape(count, rows, columns)
            for index in range(count):
                for projection, begin, end in (("gate", 0, intermediate), ("up", intermediate, rows)) if role == "gate_up" else ((role, 0, rows),):
                    if family == "deepseek_v4":
                        source_projection = {"gate": "w1", "up": "w3", "down": "w2"}[projection]
                        name = f"layers.{expert[1]}.ffn.experts.{index}.{source_projection}.weight"
                    else:
                        name = (("model.language_model." if wrapped else "model.") +
                                f"layers.{expert[1]}.mlp.experts.{index}.{projection}_proj.weight")
                    if name in seen:
                        raise ValueError(f"duplicate mapped GGUF tensor: {name}")
                    seen.add(name)
                    block_values = values[index, begin:end]
                    native = list(native_gguf_matrix(name, tensor, (end - begin, columns), config, block_values))
                    if native:
                        yield from native
                    else:
                        yield name, "BF16", to_bfloat16(block_values)
            continue
        name = gguf_tensor_name(tensor.name, wrapped, family)
        if name in seen:
            raise ValueError(f"duplicate mapped GGUF tensor: {name}")
        seen.add(name)
        native = list(native_gguf_matrix(name, tensor, tuple(reversed(tensor.shape)), config))
        if native:
            yield from native
            continue
        if family == "deepseek_v4" and name.endswith(".ffn.gate.tid2eid"):
            if tensor.tensor_type.name != "I32":
                raise ValueError("DeepSeek GGUF routing table must be I32")
            yield name, "I64", np.asarray(tensor.data).astype("<i8").reshape(tuple(reversed(tensor.shape)))
            continue
        values = gguf_float_values(tensor, gguf, name.endswith(("norm.weight", "conv1d.weight")))
        values = undo_gguf_transform(name, np.asarray(values, dtype=np.float32), config)
        yield name, "F32" if preserve_float32(name, config) else "BF16", values if preserve_float32(name, config) else to_bfloat16(values)
    if skipped_mtp:
        print(f"Skipped {skipped_mtp} GGUF MTP tensors; imported package supports target-only inference.", file=sys.stderr)
    embedding = "embed.weight" if family == "deepseek_v4" else ("model.language_model." if wrapped else "model.") + "embed_tokens.weight"
    head = "head.weight" if family == "deepseek_v4" else "lm_head.weight"
    if head not in seen:
        if text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)) and embedding in seen:
            tensor = next(item for item in reader.tensors if item.name == "token_embd.weight")
            native = list(native_gguf_matrix(head, tensor, tuple(reversed(tensor.shape)), config))
            if native:
                yield from native
            else:
                values = gguf_float_values(tensor, gguf)
                yield head, "BF16", to_bfloat16(values)
        else:
            raise ValueError("GGUF has no output.weight and canonical config does not tie embeddings")


class ShardWriter:
    """One tensor per shard bounds working memory to the largest tensor."""

    def __init__(self, root: Path):
        self.root = root
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self.shard_count = 0

    def add(self, name: str, dtype: str, values: np.ndarray) -> None:
        if name in self.weight_map or dtype not in {"BF16", "F32", "I64", "U8", "I8", "F8_E4M3", "F8_E8M0"}:
            raise ValueError(f"duplicate output tensor or unsupported output dtype: {name}")
        values = np.ascontiguousarray(values, dtype=DTYPES[dtype])
        self.shard_count += 1
        filename = f"model-{self.shard_count:06d}.safetensors"
        header = json.dumps({name: {"dtype": dtype, "shape": list(values.shape),
                                   "data_offsets": [0, values.nbytes]}}, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        with (self.root / filename).open("wb") as stream:
            stream.write(struct.pack("<Q", len(header)))
            stream.write(header)
            # tofile avoids constructing another whole-tensor bytes copy.
            values.tofile(stream)
        self.weight_map[name] = filename
        self.total_size += values.nbytes

    def merge_expert_banks(self, config: dict[str, Any]) -> None:
        """Reassemble unfused HF experts using a disk-backed destination bank."""
        if config.get("model_type") not in QWEN_TYPES:
            return
        pattern = re.compile(r"(.+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")
        banks: dict[str, dict[tuple[int, str], str]] = {}
        for name in self.weight_map:
            match = pattern.fullmatch(name)
            if match:
                banks.setdefault(match[1], {})[(int(match[2]), match[3])] = name
        for prefix, sources in banks.items():
            text = text_config(config)
            count, hidden, intermediate = (text.get("num_experts"), text.get("hidden_size"), text.get("moe_intermediate_size"))
            if any(type(value) is not int or value <= 0 for value in (count, hidden, intermediate)):
                raise ValueError("unfused Qwen experts require num_experts/hidden_size/moe_intermediate_size")
            if set(sources) != {(expert, role) for expert in range(count) for role in ("gate", "up", "down")}:
                raise ValueError(f"incomplete unfused Expert bank: {prefix}")
            if any(name + suffix in self.weight_map for name in sources.values()
                   for suffix in (".__int4_shape", ".__qnk_shape")) or any(
                       self.weight_map.get(name[:-len("weight")] + "scale") for name in sources.values()):
                # Keep the entire bank separate, including any floating-point
                # projections; the runtime binds each expert projection once.
                for (_, role), name in sources.items():
                    expected = (hidden, intermediate) if role == "down" else (intermediate, hidden)
                    suffix = next((suffix for suffix in (".__int4_shape", ".__qnk_shape")
                                   if name + suffix in self.weight_map), None)
                    if suffix is None:
                        info = scan_safetensors(self.root / self.weight_map[name])[name]
                        shape = info.shape
                    else:
                        companion = name + suffix
                        metadata = scan_safetensors(self.root / self.weight_map[companion])[companion].read()
                        shape = tuple(metadata[:2] if suffix == ".__int4_shape" else metadata[1:])
                    if shape != expected:
                        raise ValueError(f"invalid unfused Expert shape: {name}")
                continue
            for destination, roles, shape in (("gate_up_proj", ("gate", "up"), (count, intermediate * 2, hidden)),
                                               ("down_proj", ("down",), (count, hidden, intermediate))):
                path = self.root / "expert-bank.tmp"
                bank = np.memmap(path, dtype="<u2", mode="w+", shape=shape)
                for expert in range(count):
                    for index, role in enumerate(roles):
                        name = sources[(expert, role)]
                        filename = self.weight_map[name]
                        info = scan_safetensors(self.root / filename)[name]
                        values = info.read()
                        expected = (hidden, intermediate) if role == "down" else (intermediate, hidden)
                        if info.shape != expected or info.dtype not in {"BF16", "F32"}:
                            raise ValueError(f"invalid unfused Expert shape/dtype: {name}")
                        converted = values if info.dtype == "BF16" else to_bfloat16(values)
                        bank[expert, index * expected[0]:(index + 1) * expected[0], :] = converted
                        self.total_size -= info.size
                        del self.weight_map[name]
                        del values, converted
                        (self.root / filename).unlink()
                bank.flush()
                self.add(prefix + "." + destination, "BF16", bank)
                del bank
                path.unlink()

    def finish(self) -> None:
        if not self.weight_map:
            raise ValueError("no output tensors")
        index = {"metadata": {"total_size": self.total_size}, "weight_map": self.weight_map}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def normalized_config(config: dict[str, Any], gguf_target_only: bool = False) -> dict[str, Any]:
    result = copy.deepcopy(config)
    if gguf_target_only and result.get("model_type") == "deepseek_v4":
        # The official deepseek4 GGUF contains the target model. DSpark is
        # exported separately as dflash and cannot be inferred from it.
        for key in ("dspark_target_layer_ids", "dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank"):
            result.pop(key, None)
    for node in (result, result.get("text_config")):
        if isinstance(node, dict):
            node.pop("quantization_config", None)
            node["dtype"] = "bfloat16"
            if "torch_dtype" in node:
                node["torch_dtype"] = "bfloat16"
    return result


def import_package(source: Path, output: Path, config_directory: Path | None = None) -> int:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}; choose a new directory")
    if source.suffix.lower() == ".gguf" and config_directory is None:
        raise ValueError("GGUF requires --config with matching canonical HF config/tokenizer assets")
    assets = (config_directory or (source if source.is_dir() else source.parent)).resolve()
    config = json.loads((assets / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("config.json must be an object")
    model_type = config.get("model_type")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unsupported model_type: {model_type!r}")
    if model_type in QWEN_TYPES and not isinstance(config.get("text_config"), dict):
        raise ValueError("import requires the canonical wrapped Qwen config with text_config")
    validate_quantization(config)
    if not (assets / "tokenizer.json").is_file():
        raise ValueError("canonical tokenizer.json is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".import-", dir=output.parent))
    try:
        writer = ShardWriter(temporary)
        iterator = gguf_weights(source, config) if source.suffix.lower() == ".gguf" else safetensor_weights(source, config)
        for name, dtype, values in iterator:
            writer.add(canonical_tensor_name(name, config), dtype, values)
        writer.merge_expert_banks(config)
        writer.finish()
        for name in ASSETS:
            if (assets / name).is_file():
                shutil.copyfile(assets / name, temporary / name)
        (temporary / "config.json").write_text(json.dumps(normalized_config(config, source.suffix.lower() == ".gguf"), indent=2) + "\n")
        os.rename(temporary, output)
        return writer.total_size
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Requires NumPy; GGUF additionally requires gguf. Outputs one tensor per shard.\n"
               "AWQ/GPTQ retain packed INT4 codes, scales, zero points and GPTQ group indices.\n"
               "GGUF Qn_K embeddings and linear matrices retain blocks, with input column maps when needed.\n"
               "GGUF MXFP4 experts and HF FP8/MXFP4 retain codes and scales. Unsupported quantized matrices fail.\n"
               "Supports AWQ GEMM 4-bit; legacy GPTQ 4-bit (including g_idx/desc_act);\n"
               "FP8 E4M3 weight_scale_inv 128x128; plain BF16/F16/F32 Safetensors.\n"
               "Uses canonical config/tokenizer assets; Qwen requires wrapped text_config.\n"
               "GGUF: qwen35/qwen35moe/qwen4exp/gpt-oss/deepseek4, target-only (MTP tensors skipped).\n"
               "Standalone DSpark dflash contains only draft weights and requires a complete HF target+mtp source. No Hadamard,\n"
               "Standard numbered split GGUF packages are supported. No GPTQ v2/Marlin, AWQ GEMV, or compressed-tensors packages.\n"
               "Unknown schemas/tensors fail; runtime architecture restrictions still apply.")
    parser.add_argument("source", type=Path, help="HF Safetensors directory/file or GGUF file (any shard of a standard split)")
    parser.add_argument("output", type=Path, help="New destination directory; must not exist")
    parser.add_argument("--config", type=Path, help="Canonical HF config/tokenizer directory (required for GGUF)")
    args = parser.parse_args()
    print("Offline import preserves native AWQ/GPTQ INT4 and eligible GGUF Qn_K matrices.", file=sys.stderr)
    try:
        size = import_package(args.source, args.output, args.config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"import failed: {error}", file=sys.stderr)
        return 1
    tensors = scan_safetensors(args.output)
    int4 = sum(name.endswith(".__int4_shape") for name in tensors)
    qnk = sum(name.endswith(".__qnk_shape") for name in tensors)
    bf16 = sum(info.dtype == "BF16" for info in tensors.values())
    print(f"Imported {len(tensors)} tensors ({size:,} weight bytes) into {args.output}: "
          f"{int4} native INT4 matrices, {qnk} native Qn_K matrices, {bf16} BF16 tensors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
