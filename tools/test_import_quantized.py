#!/usr/bin/env python3
"""Deterministic offline importer checks; no model download or GPU required."""

from __future__ import annotations

import json
import gc
import contextlib
import io
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from tools import import_quantized as importer
except ModuleNotFoundError:
    import import_quantized as importer

try:
    import gguf
except ImportError:
    gguf = None


def pack_columns(values: np.ndarray, awq: bool = False) -> np.ndarray:
    rows, columns = values.shape
    packed = np.zeros((rows, columns // 8), dtype=np.uint32)
    # AWQ GEMM packs logical columns 0,2,4,6,1,3,5,7 into successive nibbles.
    order = (0, 2, 4, 6, 1, 3, 5, 7) if awq else tuple(range(8))
    for row in range(rows):
        for column in range(columns // 8):
            for lane, logical in enumerate(order):
                packed[row, column] |= np.uint32(int(values[row, column * 8 + logical]) << (lane * 4))
    return packed.view(np.int32)


def pack_gptq(values: np.ndarray) -> np.ndarray:
    inputs, outputs = values.shape
    packed = np.zeros((inputs // 8, outputs), dtype=np.uint32)
    for row in range(inputs):
        for column in range(outputs):
            packed[row // 8, column] |= np.uint32(int(values[row, column]) << (row % 8 * 4))
    return packed.view(np.int32)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, arrays, offset = {}, [], 0
    for name, (dtype, values) in tensors.items():
        values = np.ascontiguousarray(values, dtype=importer.DTYPES[dtype])
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + values.nbytes]}
        offset += values.nbytes
        arrays.append(values)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for values in arrays:
            values.tofile(stream)


def iq4_xs_blocks(rows: int, columns: int, seed: int = 0) -> np.ndarray:
    """Build valid IQ4_XS blocks from its GGML ABI, without a quantizer."""
    if rows <= 0 or columns <= 0 or columns % 256:
        raise ValueError("IQ4_XS fixtures require positive rows and columns divisible by 256")
    result = np.zeros((rows, columns // 256 * 136), dtype=np.uint8)
    blocks = result.reshape(rows, columns // 256, 136)
    # Signed local scales are 0, +/-1, +/-2, +/-4 and +/-8. With the
    # power-of-two super-block delta, every decoded codebook value is BF16-exact.
    base_scales = np.asarray([32, 33, 31, 34, 30, 36, 28, 40], dtype=np.uint8)
    for row in range(rows):
        for block in range(columns // 256):
            encoded = base_scales
            block_data = blocks[row, block]
            block_data[:2] = np.asarray([1 / 512], dtype="<f2").view(np.uint8)
            scale_high = 0
            for subgroup, scale in enumerate(encoded):
                block_data[4 + subgroup // 2] |= int(scale & 15) << (4 * (subgroup % 2))
                scale_high |= int(scale >> 4) << (2 * subgroup)
            block_data[2:4] = np.asarray([scale_high], dtype="<u2").view(np.uint8)
            for subgroup in range(8):
                lanes = np.arange(16, dtype=np.int32)
                low = ((lanes * 3 + seed + row + block + subgroup) & 15).astype(np.uint8)
                high = ((lanes * 5 + seed + row * 2 + block + subgroup * 3) & 15).astype(np.uint8)
                block_data[8 + subgroup * 16:8 + (subgroup + 1) * 16] = low | (high << 4)
    return result


def iq4_nl_blocks(rows: int, columns: int) -> np.ndarray:
    """Build valid IQ4_NL blocks without relying on an upstream quantizer."""
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("IQ4_NL fixtures require positive rows and columns divisible by 32")
    result = np.zeros((rows, columns // 32 * 18), dtype=np.uint8)
    blocks = result.reshape(rows, columns // 32, 18)
    blocks[..., :2] = np.asarray([1 / 512], dtype="<f2").view(np.uint8)
    codes = np.arange(16, dtype=np.uint8)
    blocks[..., 2:] = codes.reshape(1, 1, 16)
    return result


def repacked_float(*args, **kwargs):
    packed, shape, scales, zeros, indices = importer.repack_linear(*args, **kwargs)
    _, columns, group_size = shape
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int16)
    codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
    if indices is None:
        indices = np.arange(columns) // group_size
    return (codes[:, :columns] - zeros[:, indices]) * scales[:, indices]


def native_codes(loaded, name):
    shape = loaded[name + ".__int4_shape"].read()
    rows, columns, _ = shape
    packed = loaded[name].read()
    codes = np.empty((rows, packed.shape[1] * 2), dtype=np.uint8)
    codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
    return codes[:, :columns]


def native_float(loaded, name):
    _, columns, group_size = loaded[name + ".__int4_shape"].read()
    indices = (loaded[name + ".__int4_g_idx"].read() if name + ".__int4_g_idx" in loaded
               else np.arange(columns) // group_size)
    zeros, scales = loaded[name + ".__int4_zeros"].read(), loaded[name + ".__int4_scales"].read()
    return (native_codes(loaded, name).astype(np.int16) - zeros[:, indices]) * scales[:, indices]


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        # Mock call histories can retain GGUF mmap views in reference cycles.
        gc.collect()
        self.temporary.cleanup()

    def test_deepseek_gguf_kv_a_norm_alias_maps_to_kv_norm(self):
        self.assertEqual(
            importer.gguf_tensor_name("blk.0.attn_kv_a_norm.weight", False, "deepseek_v4"),
            "layers.0.attn.kv_norm.weight",
        )

    def assets(self, quant=None, model_type="qwen3_5") -> Path:
        root = self.root / "source"
        root.mkdir(exist_ok=True)
        config = {"model_type": model_type, "text_config": {
            "model_type": model_type + "_text", "hidden_size": 8, "vocab_size": 4, "num_hidden_layers": 1,
            "mtp_num_hidden_layers": 1, "tie_word_embeddings": False, "dtype": "float16"}}
        if quant is not None:
            config["quantization_config"] = quant
        (root / "config.json").write_text(json.dumps(config))
        (root / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
        (root / "chat_template.jinja").write_text("{{ messages }}")
        return root

    def test_awq_gemm_known_values(self):
        # Each column's code and scale is distinct; swapping nibbles or transposing
        # either matrix therefore changes the result.
        codes = (np.arange(16 * 16).reshape(16, 16) * 7 + np.arange(16)[:, None]) % 16
        zeros = np.tile(np.arange(16), (2, 1))
        scales = np.arange(1, 33, dtype=np.float32).reshape(2, 16) / 8
        decoded = repacked_float(pack_columns(codes, True), pack_columns(zeros, True), scales, "awq", 8)
        expected = np.empty((16, 16), dtype=np.float32)
        for output in range(16):
            for input_ in range(16):
                expected[output, input_] = (int(codes[input_, output]) - int(zeros[input_ // 8, output])) * scales[input_ // 8, output]
        np.testing.assert_array_equal(decoded, expected)

    def test_gptq_gidx_activation_order(self):
        codes = np.arange(128).reshape(16, 8) % 16
        zeros = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
        scales = np.asarray([[0.5] * 8, [2.0] * 8], dtype=np.float32)
        g_idx = np.asarray([0, 1] * 8, dtype=np.int32)
        decoded = repacked_float(pack_gptq(codes), pack_columns(zeros - 1), scales, "gptq", 8, g_idx, True)
        expected = np.asarray([[(codes[input_, output] - zeros[g_idx[input_], output]) * scales[g_idx[input_], output]
                                for input_ in range(16)] for output in range(8)], dtype=np.float32)
        np.testing.assert_array_equal(decoded, expected)
        with self.assertRaisesRegex(ValueError, "desc_act"):
            repacked_float(pack_gptq(codes), pack_columns(zeros - 1), scales, "gptq", 8, g_idx, False)
        with self.assertRaisesRegex(ValueError, "requires g_idx"):
            repacked_float(pack_gptq(codes), pack_columns(zeros - 1), scales, "gptq", 8, desc_act=True)
        with self.assertRaisesRegex(ValueError, "group sizes"):
            repacked_float(pack_gptq(codes), pack_columns(zeros - 1), scales, "gptq", 8, np.zeros(16, dtype=np.int32), True)

    def test_quantization_schema_rejections(self):
        cases = [
            {"quant_method": "awq", "bits": 4, "group_size": 8, "version": "gemv"},
            {"quant_method": "awq", "bits": 4, "group_size": 8, "zero_point": False},
            {"quant_method": "gptq", "bits": 4, "group_size": 8, "checkpoint_format": "gptq_v2"},
            {"quant_method": "gptq", "bits": 3, "group_size": 8},
            {"quant_method": "gptq", "bits": 4, "group_size": 0},
            {"quant_method": "gptq", "bits": 4, "w_bit": 8, "group_size": 8},
            {"quant_method": "gptq", "bits": 4, "group_size": 8, "backend": "marlin"},
            {"quant_method": "gptq", "bits": 4, "group_size": 8, "is_marlin_format": True},
            {"quant_method": "gptq", "bits": 4, "group_size": 8, "use_marlin": True},
            {"quant_method": "compressed-tensors"},
            {"quant_method": "fp8", "weight_block_size": [64, 64]},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                importer.validate_quantization({"quantization_config": case})

        for key in ("is_marlin_format", "use_marlin"):
            self.assertEqual(importer.validate_quantization({"quantization_config": {
                "quant_method": "gptq", "bits": 4, "group_size": 8, key: False}})[0], "gptq")

    def test_fp8_blocks_and_subnormals(self):
        values = np.full((129, 129), 0x38, dtype=np.uint8)  # E4M3 1.0
        values[0, :4] = [0, 1, 0xb8, 0x7e]  # zero, subnormal, -1, max 448
        scales = np.asarray([[2, 3], [4, 5]], dtype=np.float32)
        decoded = importer.dequantize_fp8(values, scales)
        np.testing.assert_array_equal(decoded[0, :4], [0, 2 ** -8, -2, 896])
        self.assertEqual(decoded[0, 128], 3)
        self.assertEqual(decoded[128, 0], 4)
        self.assertEqual(decoded[128, 128], 5)
        with self.assertRaises(ValueError):
            importer.dequantize_fp8(values, np.zeros((2, 2), dtype=np.float32))
        values[0, 0] = 0x7f
        with self.assertRaisesRegex(ValueError, "non-finite"):
            importer.to_bfloat16(importer.dequantize_fp8(values, scales))

    def test_bf16_round_to_even(self):
        values = np.asarray([0x3f808000, 0x3f818000, 0xbf808000], dtype=np.uint32).view(np.float32)
        np.testing.assert_array_equal(importer.to_bfloat16(values), [0x3f80, 0x3f82, 0xbf80])

    def test_awq_package_roundtrip_and_assets(self):
        source = self.assets({"quant_method": "awq", "bits": 4, "group_size": 8, "version": "GEMM", "zero_point": True})
        codes = np.arange(64).reshape(8, 8) % 16
        zeros = np.full((1, 8), 3)
        scales = np.full((1, 8), 0.5, dtype=np.float16)
        prefix = "model.layers.0.mlp.gate_proj."
        write_safetensors(source / "model.safetensors", {
            prefix + "qweight": ("I32", pack_columns(codes, True)),
            prefix + "qzeros": ("I32", pack_columns(zeros, True)),
            prefix + "scales": ("F16", scales),
            "model.norm.weight": ("F16", np.ones(8, dtype=np.float16)),
        })
        original = (source / "model.safetensors").read_bytes()
        output = self.root / "output"
        self.assertEqual(importer.import_package(source, output), 112)
        loaded = importer.scan_safetensors(output)
        target = importer.canonical_tensor_name(prefix + "weight")
        self.assertEqual(loaded[target].dtype, "U8")
        np.testing.assert_array_equal(native_codes(loaded, target), codes.T)
        np.testing.assert_array_equal(loaded[target + ".__int4_scales"].read(), scales.T.astype(np.float32))
        np.testing.assert_array_equal(loaded[target + ".__int4_zeros"].read(), zeros.T)
        values = native_float(loaded, target)
        np.testing.assert_array_equal(values, ((codes - 3) * 0.5).T)
        config = json.loads((output / "config.json").read_text())
        self.assertNotIn("quantization_config", config)
        self.assertEqual(config["dtype"], "bfloat16")
        self.assertEqual(config["model_type"], "qwen3_5")
        self.assertEqual((output / "chat_template.jinja").read_text(), "{{ messages }}")
        self.assertEqual(original, (source / "model.safetensors").read_bytes())
        with self.assertRaisesRegex(ValueError, "already exists"):
            importer.import_package(source, output)

    def test_fp8_package_and_failure_cleanup(self):
        source = self.assets({"quant_method": "fp8", "weight_block_size": [128, 128], "activation_scheme": "dynamic"})
        name = "model.layers.0.mlp.up_proj."
        write_safetensors(source / "model.safetensors", {
            name + "weight": ("F8_E4M3", np.full((2, 4), 0x38, dtype=np.uint8)),
            name + "weight_scale_inv": ("F32", np.full((1, 1), 2.0, dtype=np.float32)),
        })
        output = self.root / "output"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        target = importer.canonical_tensor_name(name + "weight")
        self.assertEqual(set(loaded), {target, importer.canonical_tensor_name(name + "scale")})
        self.assertEqual(loaded[target].dtype, "F8_E4M3")
        np.testing.assert_array_equal(loaded[target].read(), 0x38)
        np.testing.assert_array_equal(loaded[importer.canonical_tensor_name(name + "scale")].read(), 2)
        write_safetensors(source / "model.safetensors", {name + "weight": ("F8_E4M3", np.zeros((2, 4), dtype=np.uint8))})
        failed = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "missing FP8"):
            importer.import_package(source, failed)
        self.assertFalse(failed.exists())
        self.assertEqual(list(self.root.glob("failed.import-*")), [])

    def test_plain_mixed_precision_package_normalizes_every_float(self):
        source = self.assets()
        tensors = {
            "model.embed_tokens.weight": ("F32", np.asarray([[1.25, -2.5]], dtype=np.float32)),
            "model.norm.weight": ("F32", np.asarray([0.25, 1.5], dtype=np.float32)),
            "model.layers.0.linear_attn.A_log": ("F32", np.asarray([-3.25], dtype=np.float32)),
            "model.layers.0.linear_attn.dt_bias": ("F16", np.asarray([0.5], dtype=np.float16)),
            "model.position_ids": ("I64", np.asarray([0, 1], dtype=np.int64)),
        }
        write_safetensors(source / "model.safetensors", tensors)
        output = self.root / "output"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        for name, (dtype, expected) in tensors.items():
            actual = loaded[importer.canonical_tensor_name(name)]
            if dtype == "I64":
                self.assertEqual(actual.dtype, "I64")
                np.testing.assert_array_equal(actual.read(), expected)
            else:
                self.assertEqual(actual.dtype, "BF16")
                np.testing.assert_array_equal(importer.as_float(actual.read(), "BF16"), expected)
        normalized = json.loads((output / "config.json").read_text())
        self.assertEqual(normalized["text_config"]["dtype"], "bfloat16")

    def test_bad_packing_and_orphan_rejected(self):
        source = self.assets({"quant_method": "awq", "bits": 4, "group_size": 8})
        write_safetensors(source / "model.safetensors", {"model.layers.0.qzeros": ("I32", np.ones((1, 1), dtype=np.int32))})
        with self.assertRaisesRegex(ValueError, "orphan"):
            list(importer.safetensor_weights(source, json.loads((source / "config.json").read_text())))
        with self.assertRaisesRegex(ValueError, "I32"):
            repacked_float(np.zeros((8, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int32),
                                       np.ones((1, 8), dtype=np.float32), "awq", 8)
        with self.assertRaisesRegex(ValueError, "shapes"):
            repacked_float(np.zeros((8, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.int32),
                                       np.ones((2, 8), dtype=np.float32), "awq", 8)

    def test_gptq_complete_package(self):
        source = self.assets({"quant_method": "gptq", "bits": 4, "group_size": 8,
                              "desc_act": True, "sym": False, "checkpoint_format": "gptq"})
        codes = np.arange(128).reshape(16, 8) % 16
        groups = np.asarray([1, 0] * 8, dtype=np.int32)
        zeros = np.asarray([[3] * 8, [7] * 8])
        scales = np.asarray([[0.5] * 8, [2.0] * 8], dtype=np.float32)
        prefix = "model.layers.0.mlp.down_proj."
        write_safetensors(source / "model.safetensors", {
            prefix + "qweight": ("I32", pack_gptq(codes)),
            prefix + "qzeros": ("I32", pack_columns(zeros - 1)),
            prefix + "scales": ("F32", scales), prefix + "g_idx": ("I32", groups),
        })
        output = self.root / "output"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        self.assertEqual(len(loaded), 5)
        target = importer.canonical_tensor_name(prefix + "weight")
        self.assertEqual(loaded[target].dtype, "U8")
        np.testing.assert_array_equal(native_codes(loaded, target), codes.T)
        np.testing.assert_array_equal(loaded[target + ".__int4_g_idx"].read(), groups)
        expected = ((codes - zeros[groups]) * scales[groups]).T
        actual = native_float(loaded, target)
        np.testing.assert_array_equal(actual, expected)

    def test_native_int4_random_codes_scales_and_storage(self):
        rng = np.random.default_rng(38)
        for method in ("awq", "gptq"):
            with self.subTest(method=method):
                source = self.assets({"quant_method": method, "bits": 4, "group_size": 64})
                codes = rng.integers(0, 16, (256, 256), dtype=np.int32)
                zeros = rng.integers(0 if method == "awq" else 1, 16 if method == "awq" else 17,
                                     (4, 256), dtype=np.int32)
                scales = rng.uniform(0.001, 0.09, (4, 256)).astype(np.float32)
                prefix = "model.layers.0.self_attn.q_proj."
                write_safetensors(source / "model.safetensors", {
                    prefix + "qweight": ("I32", pack_columns(codes, True) if method == "awq" else pack_gptq(codes)),
                    prefix + "qzeros": ("I32", pack_columns(zeros if method == "awq" else zeros - 1, method == "awq")),
                    prefix + "scales": ("F32", scales),
                })
                output = self.root / method
                with patch.object(importer, "to_bfloat16", side_effect=AssertionError("INT4 was expanded to BF16")):
                    size = importer.import_package(source, output)
                loaded = importer.scan_safetensors(output)
                target = importer.canonical_tensor_name(prefix + "weight")
                self.assertFalse(any(info.dtype == "BF16" for info in loaded.values()))
                self.assertEqual(loaded[target].size, codes.size // 2)
                self.assertEqual(size, codes.size // 2 + scales.size * 5 + 24)
                self.assertLess(size, codes.size)
                np.testing.assert_array_equal(native_codes(loaded, target), codes.T)
                np.testing.assert_array_equal(loaded[target + ".__int4_scales"].read(), scales.T)
                np.testing.assert_array_equal(loaded[target + ".__int4_zeros"].read(), zeros.T)
                expected = (codes - zeros[np.arange(256) // 64]).astype(np.float32) * scales[np.arange(256) // 64]
                np.testing.assert_array_equal(native_float(loaded, target), expected.T)

    def test_awq_odd_columns_and_single_group(self):
        codes = np.arange(17 * 8).reshape(17, 8) % 16
        zeros = np.zeros((1, 8), dtype=np.int32)
        scales = np.arange(1, 9, dtype=np.float32)[None, :]
        packed, shape, output_scales, output_zeros, indices = importer.repack_linear(
            pack_columns(codes, True), pack_columns(zeros, True), scales, "awq", -1)
        np.testing.assert_array_equal(shape, [8, 17, 17])
        np.testing.assert_array_equal(packed[:, -1], codes[-1])
        np.testing.assert_array_equal(output_scales, scales.T)
        np.testing.assert_array_equal(output_zeros, zeros.T)
        self.assertIsNone(indices)

    def test_native_int4_validation(self):
        weight = np.zeros((2, 8), dtype=np.int32)
        zeros = np.zeros((2, 1), dtype=np.int32)
        scales = np.ones((2, 8), dtype=np.float32)
        cases = ((weight.astype(np.int64), zeros, scales, None, False),
                 (weight, zeros.astype(np.int64), scales, None, False),
                 (weight, zeros, scales[:1], None, False),
                 (weight, zeros, np.zeros_like(scales), None, False),
                 (weight, zeros, scales, None, True),
                 (weight, zeros, scales, np.zeros(16, dtype=np.int32), True))
        for values, zero, scale, index, desc_act in cases:
            with self.subTest(desc_act=desc_act), self.assertRaises(ValueError):
                importer.repack_linear(values, zero, scale, "gptq", 8, index, desc_act)

    def test_quantized_experts_stay_separate_with_mixed_projections(self):
        source = self.assets({"quant_method": "awq", "bits": 4, "group_size": 8}, "qwen3_5_moe")
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(num_experts=2, moe_intermediate_size=8)
        (source / "config.json").write_text(json.dumps(config))
        tensors = {}
        for expert in range(2):
            for role in ("gate", "up", "down"):
                prefix = f"model.layers.0.mlp.experts.{expert}.{role}_proj."
                if role == "up":
                    tensors[prefix + "weight"] = ("F16", np.ones((8, 8), dtype=np.float16))
                else:
                    tensors[prefix + "qweight"] = ("I32", pack_columns(np.full((8, 8), expert + 2), True))
                    tensors[prefix + "qzeros"] = ("I32", pack_columns(np.ones((1, 8)), True))
                    tensors[prefix + "scales"] = ("F32", np.ones((1, 8), dtype=np.float32))
        write_safetensors(source / "model.safetensors", tensors)
        output = self.root / "output"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        self.assertEqual(len(loaded), 4 * 4 + 2)
        self.assertNotIn("model.language_model.layers.0.mlp.experts.gate_up_proj", loaded)
        for expert in range(2):
            prefix = f"model.language_model.layers.0.mlp.experts.{expert}."
            np.testing.assert_array_equal(native_codes(loaded, prefix + "gate_proj.weight"), expert + 2)
            self.assertEqual(loaded[prefix + "up_proj.weight"].dtype, "BF16")

    def test_unfused_experts_disk_bank_merge(self):
        source = self.assets(model_type="qwen3_5_moe")
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(num_experts=2, moe_intermediate_size=3)
        (source / "config.json").write_text(json.dumps(config))
        tensors = {}
        for expert in range(2):
            for index, role in enumerate(("gate", "up", "down")):
                shape = (8, 3) if role == "down" else (3, 8)
                values = np.full(shape, expert * 10 + index, dtype=np.float16)
                tensors[f"model.layers.0.mlp.experts.{expert}.{role}_proj.weight"] = ("F16", values)
        write_safetensors(source / "model.safetensors", tensors)
        output = self.root / "output"
        size = importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        self.assertEqual(size, 288)
        self.assertEqual(len(loaded), 2)
        gate_up = importer.as_float(loaded["model.language_model.layers.0.mlp.experts.gate_up_proj"].read(), "BF16")
        self.assertEqual(gate_up.shape, (2, 6, 8))
        np.testing.assert_array_equal(gate_up[0, :3], 0)
        np.testing.assert_array_equal(gate_up[0, 3:], 1)
        np.testing.assert_array_equal(gate_up[1, :3], 10)
        np.testing.assert_array_equal(gate_up[1, 3:], 11)
        self.assertFalse((output / "expert-bank.tmp").exists())

    def test_inverse_qwen_gguf_transforms(self):
        config = {"linear_num_key_heads": 2, "linear_num_value_heads": 4,
                  "linear_key_head_dim": 1, "linear_value_head_dim": 2}
        original = np.arange(16, dtype=np.float32).reshape(8, 2)
        tiled = original.reshape(2, 2, 2, 2).swapaxes(0, 1).reshape(8, 2)
        name = "model.layers.0.linear_attn.in_proj_z.weight"
        np.testing.assert_array_equal(importer.undo_gguf_transform(name, tiled, config), original)
        values = np.asarray([-1, -4, -2, -8], dtype=np.float32)  # tiled K0V0,K1V0,K0V1,K1V1
        expected = np.log(np.asarray([1, 2, 4, 8], dtype=np.float32))
        np.testing.assert_allclose(importer.undo_gguf_transform("model.layers.0.linear_attn.A_log", values, config), expected)
        np.testing.assert_array_equal(importer.undo_gguf_transform("model.norm.weight", np.asarray([1.5]), config), [0.5])
        with self.assertRaisesRegex(ValueError, "unsupported GGUF tensor"):
            importer.gguf_tensor_name("blk.0.surprise.weight", False)

    def test_gguf_layout_metadata_checked_before_inverse_transform(self):
        config = {"text_config": {
            "num_hidden_layers": 1, "mtp_num_hidden_layers": 1, "intermediate_size": 128,
            "hidden_size": 64, "vocab_size": 16, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
        }}
        metadata = {"block_count": 2, "nextn_predict_layers": 1, "feed_forward_length": 128,
                    "embedding_length": 64, "vocab_size": 16, "attention.head_count": 4, "attention.head_count_kv": 2,
                    "ssm.group_count": 2, "ssm.state_size": 8, "ssm.time_step_rank": 4,
                    "ssm.inner_size": 64, "ssm.conv_kernel": 4, "rope.freq_base": 1e6,
                    "rope.dimension_count": 8, "attention.layer_norm_rms_epsilon": float(np.float32(1e-6))}
        def reader_for(values):
            fields = {"qwen35." + key: SimpleNamespace(contents=lambda value=value: value) for key, value in values.items()}
            return SimpleNamespace(get_field=fields.get)
        importer.validate_gguf_metadata(reader_for(metadata), "qwen35", config)
        for key in metadata:
            invalid = dict(metadata)
            invalid[key] = metadata[key] * 2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "disagree.*with canonical"):
                importer.validate_gguf_metadata(reader_for(invalid), "qwen35", config)
        absent = {"text_config": {"hidden_size": 64, "num_hidden_layers": 1}}
        with self.assertRaisesRegex(ValueError, "ssm.group_count"):
            importer.validate_gguf_metadata(reader_for({"block_count": 1, "ssm.group_count": 2}), "qwen35", absent)
        # --no-mtp exports retain the canonical MTP count but have source total=target.
        self.assertEqual(importer.validate_gguf_metadata(reader_for({"block_count": 1}), "qwen35", config), (1, 0))
        for invalid in ({}, {"block_count": 2}, {"block_count": 3, "nextn_predict_layers": 2},
                        {"block_count": 1, "nextn_predict_layers": 1}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "block_count"):
                importer.validate_gguf_metadata(reader_for(invalid), "qwen35", config)
        moe_config = {"text_config": {"num_hidden_layers": 1, "moe_intermediate_size": 32,
                                      "shared_expert_intermediate_size": 64}}
        for key, size in (("expert_feed_forward_length", 32), ("expert_shared_feed_forward_length", 64)):
            fields = {"qwen35moe.block_count": SimpleNamespace(contents=lambda: 1),
                      "qwen35moe." + key: SimpleNamespace(contents=lambda: size)}
            reader = SimpleNamespace(get_field=fields.get)
            importer.validate_gguf_metadata(reader, "qwen35moe", moe_config)
            fields["qwen35moe." + key] = SimpleNamespace(contents=lambda: size + 1)
            with self.assertRaisesRegex(ValueError, key):
                importer.validate_gguf_metadata(reader, "qwen35moe", moe_config)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_actual_gguf_reader_quantization_and_native_qnk(self):
        source = self.assets()
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(hidden_size=256, vocab_size=8, intermediate_size=2)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "tiny.gguf"
        writer = gguf.GGUFWriter(path, "qwen35")
        writer.add_block_count(2)
        writer.add_nextn_predict_layers(1)
        embedding = np.arange(8 * 256, dtype=np.float32).reshape(8, 256) / 8
        writer.add_tensor("token_embd.weight", embedding)
        raw_output = np.arange(8 * 210, dtype=np.uint8).reshape(8, 210)
        writer.add_tensor("output.weight", raw_output, raw_dtype=gguf.GGMLQuantizationType.Q6_K)
        writer.add_tensor("output_norm.weight", np.full(256, 1.5, dtype=np.float32))
        raw_q4k = np.arange(2 * 144, dtype=np.uint8).reshape(2, 144)
        writer.add_tensor("blk.0.ffn_gate.weight", raw_q4k, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        raw_attention = np.arange(16 * 110, dtype=np.uint8).reshape(16, 110)
        writer.add_tensor("blk.0.attn_q.weight", raw_attention, raw_dtype=gguf.GGMLQuantizationType.Q3_K)
        writer.add_tensor("blk.1.nextn.eh_proj.weight", np.ones((8, 8), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "requires --config"):
            importer.import_package(path, output)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            importer.import_package(path, output, source)
        self.assertIn("Skipped 1 GGUF MTP tensors", stderr.getvalue())
        self.assertEqual(json.loads((output / "config.json").read_text())["text_config"]["mtp_num_hidden_layers"], 1)
        loaded = importer.scan_safetensors(output)
        expected = importer.to_bfloat16(embedding)
        np.testing.assert_array_equal(loaded["model.language_model.embed_tokens.weight"].read(), expected)
        np.testing.assert_array_equal(importer.as_float(loaded["model.language_model.norm.weight"].read(), "BF16"), 0.5)
        self.assertEqual(loaded["model.language_model.layers.0.mlp.gate_proj.weight"].dtype, "U8")
        np.testing.assert_array_equal(loaded["model.language_model.layers.0.mlp.gate_proj.weight"].read(), raw_q4k.reshape(-1))
        np.testing.assert_array_equal(loaded["model.language_model.layers.0.mlp.gate_proj.weight.__qnk_shape"].read(), [4, 2, 256])
        np.testing.assert_array_equal(loaded["model.language_model.layers.0.self_attn.q_proj.weight"].read(), raw_attention.reshape(-1))
        np.testing.assert_array_equal(loaded["model.language_model.layers.0.self_attn.q_proj.weight.__qnk_shape"].read(), [3, 16, 256])
        np.testing.assert_array_equal(loaded["lm_head.weight"].read(), raw_output.reshape(-1))
        np.testing.assert_array_equal(loaded["lm_head.weight.__qnk_shape"].read(), [6, 8, 256])

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_short_block_and_iq4_xs_import_preserves_native_bytes(self):
        source = self.assets()
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(hidden_size=256, vocab_size=8, intermediate_size=2)
        (source / "config.json").write_text(json.dumps(config))
        formats = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "IQ4_XS")
        expected_shapes = {
            "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
            "Q5_1": (32, 24), "Q8_0": (32, 34), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
        }
        for format_name in formats:
            with self.subTest(format=format_name):
                ggml_type = getattr(gguf.GGMLQuantizationType, format_name)
                block_elements, block_bytes = expected_shapes[format_name]
                path = self.root / f"{format_name}.gguf"
                writer = gguf.GGUFWriter(path, "qwen35")
                writer.add_block_count(2)
                writer.add_nextn_predict_layers(1)
                writer.add_tensor("token_embd.weight", np.ones((8, 256), dtype=np.float32))
                raw_tensors = {}
                for name, rows, seed in (("output.weight", 8, 1),
                                         ("blk.0.ffn_gate.weight", 2, 2),
                                         ("blk.0.attn_q.weight", 16, 3)):
                    values = np.random.default_rng(seed).normal(0, 0.125, (rows, 256)).astype(np.float32)
                    raw = (iq4_xs_blocks(rows, 256, seed) if format_name == "IQ4_XS" else
                           iq4_nl_blocks(rows, 256) if format_name == "IQ4_NL" else
                           np.asarray(gguf.quantize(values, ggml_type)).view(np.uint8))
                    expected_bytes = rows * (256 // block_elements) * block_bytes
                    self.assertEqual(raw.nbytes, expected_bytes)
                    raw_tensors[name] = (raw.reshape(rows, -1), values)
                    writer.add_tensor(name, raw_tensors[name][0], raw_dtype=ggml_type)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()

                output = self.root / f"{format_name}-output"
                importer.import_package(path, output, source)
                loaded = importer.scan_safetensors(output)
                tag, _, _ = importer.GGUF_NATIVE_FORMATS[format_name]
                for name, (raw, values) in raw_tensors.items():
                    target = ("lm_head.weight" if name == "output.weight" else
                              "model.language_model.layers.0.mlp.gate_proj.weight" if name.endswith("ffn_gate.weight") else
                              "model.language_model.layers.0.self_attn.q_proj.weight")
                    expected_size = values.shape[0] * (256 // block_elements) * block_bytes
                    self.assertEqual(loaded[target].dtype, "U8")
                    self.assertEqual(loaded[target].size, expected_size)
                    np.testing.assert_array_equal(loaded[target].read(), raw.reshape(-1))
                    np.testing.assert_array_equal(loaded[target + ".__qnk_shape"].read(),
                                                  [tag, values.shape[0], 256])
                    decoded = gguf.dequantize(raw, ggml_type).reshape(values.shape)
                    self.assertTrue(np.isfinite(decoded).all())
                    self.assertGreater(float(np.max(np.abs(decoded))), 0.0)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_expert_gate_up_reassembly(self):
        source = self.assets(model_type="qwen3_5_moe")
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(num_experts=2, moe_intermediate_size=3)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "tiny.gguf"
        writer = gguf.GGUFWriter(path, "qwen35moe")
        writer.add_block_count(1)  # --no-mtp source, canonical config still declares MTP=1.
        gate = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
        up = gate + 100
        writer.add_tensor("token_embd.weight", np.ones((4, 8), dtype=np.float32))
        writer.add_tensor("output.weight", np.ones((4, 8), dtype=np.float32))
        writer.add_tensor("blk.0.ffn_gate_exps.weight", gate)
        writer.add_tensor("blk.0.ffn_up_exps.weight", up)
        writer.add_tensor("blk.0.ffn_down_exps.weight", np.ones((2, 8, 3), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        output = self.root / "output"
        importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        actual = importer.as_float(loaded["model.language_model.layers.0.mlp.experts.gate_up_proj"].read(), "BF16")
        np.testing.assert_array_equal(actual, np.concatenate((gate, up), axis=1))

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_quantized_experts_and_shared_expert_preserve_blocks(self):
        source = self.assets(model_type="qwen3_5_moe")
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(hidden_size=256, num_experts=2, moe_intermediate_size=256,
                                     shared_expert_intermediate_size=256)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "experts.gguf"
        writer = gguf.GGUFWriter(path, "qwen35moe")
        writer.add_block_count(1)
        writer.add_expert_count(2)
        writer.add_tensor("token_embd.weight", np.ones((4, 256), dtype=np.float32))
        writer.add_tensor("output.weight", np.ones((4, 256), dtype=np.float32))
        rng = np.random.default_rng(97)
        arrays = {}
        for role, type_, bytes_ in (("gate", gguf.GGMLQuantizationType.Q2_K, 84),
                                     ("up", gguf.GGMLQuantizationType.Q5_K, 176),
                                     ("down", gguf.GGMLQuantizationType.Q8_K, 292)):
            arrays[role] = rng.integers(0, 256, (2, 256, bytes_), dtype=np.uint8)
            writer.add_tensor(f"blk.0.ffn_{role}_exps.weight", arrays[role], raw_dtype=type_)
        shared = rng.integers(0, 256, (256, 144), dtype=np.uint8)
        writer.add_tensor("blk.0.ffn_gate_shexp.weight", shared, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        output = self.root / "output"
        importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        for role, bits in (("gate", 2), ("up", 5), ("down", 8)):
            for expert in range(2):
                name = f"model.language_model.layers.0.mlp.experts.{expert}.{role}_proj.weight"
                self.assertEqual(loaded[name].dtype, "U8")
                np.testing.assert_array_equal(loaded[name].read(), arrays[role][expert].reshape(-1))
                np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [bits, 256, 256])
        np.testing.assert_array_equal(loaded["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"].read(), shared.reshape(-1))
        self.assertNotIn("model.language_model.layers.0.mlp.experts.gate_up_proj", loaded)

    def test_native_qnk_inverse_row_permutation_and_column_map(self):
        config = {"linear_num_key_heads": 2, "linear_num_value_heads": 4,
                  "linear_key_head_dim": 1, "linear_value_head_dim": 2}
        original = np.arange(8 * 144, dtype=np.uint8).reshape(8, 144)
        tiled = original.reshape(2, 2, 2, 144).swapaxes(0, 1).reshape(8, 144)
        tensor = SimpleNamespace(tensor_type=SimpleNamespace(name="Q4_K"), data=tiled)
        name = "model.layers.0.linear_attn.in_proj_z.weight"
        results = list(importer.native_gguf_matrix(name, tensor, (8, 256), config))
        np.testing.assert_array_equal(results[0][2], original.reshape(-1))
        np.testing.assert_array_equal(results[1][2], [4, 8, 256])
        output = np.arange(8 * 288, dtype=np.uint8).reshape(8, 288)
        tensor.data = output
        config["linear_value_head_dim"] = 128
        name = "model.layers.0.linear_attn.out_proj.weight"
        results = list(importer.native_gguf_matrix(name, tensor, (8, 512), config))
        np.testing.assert_array_equal(results[0][2], output.reshape(-1))
        np.testing.assert_array_equal(results[1][2], [4, 8, 512])
        self.assertEqual(results[2][0], name + ".__qnk_columns")
        np.testing.assert_array_equal(results[2][2], np.concatenate((np.arange(128), np.arange(256, 384),
                                                                   np.arange(128, 256), np.arange(384, 512))))

    def test_native_nvfp4_uses_reserved_ggml_tag_without_repacking(self):
        raw = np.arange(2 * 2 * 36, dtype=np.uint8).reshape(2, 72)
        tensor = SimpleNamespace(tensor_type=SimpleNamespace(name="NVFP4"), data=raw)
        results = list(importer.native_gguf_matrix("model.embed_tokens.weight", tensor, (2, 128), {}))
        self.assertEqual(importer.GGUF_NATIVE_FORMATS["NVFP4"], (296, 64, 36))
        self.assertEqual([item[0] for item in results], ["model.embed_tokens.weight", "model.embed_tokens.weight.__qnk_shape"])
        self.assertEqual(results[0][1], "U8")
        np.testing.assert_array_equal(results[0][2], raw.reshape(-1))
        np.testing.assert_array_equal(results[1][2], [296, 2, 128])

    def test_native_mxfp4_general_safetensor_roles_stay_as_code_scale_pairs(self):
        source, _ = self.family_assets("qwen4_exp", {"quant_method": "mxfp4"})
        roles = (
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
        )
        tensors = {}
        expected = {}
        for index, name in enumerate(roles):
            packed = (np.arange(2 * 32, dtype=np.uint8).reshape(2, 32) + index * 17).view(np.int8)
            scales = np.full((2, 2), 119 + index, dtype=np.uint8)
            tensors[name] = ("I8", packed)
            tensors[name[:-len("weight")] + "scale"] = ("F8_E8M0", scales)
            expected[name], expected[name[:-len("weight")] + "scale"] = packed.copy(), scales.copy()
        columns_name = roles[-1] + ".__mxfp4_columns"
        columns = np.arange(64, dtype="<i8")[::-1].copy()
        tensors[columns_name] = ("I64", columns)
        expected[columns_name] = columns
        write_safetensors(source / "model.safetensors", tensors)
        output = self.root / "native-general-mxfp4"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        for name, values in expected.items():
            np.testing.assert_array_equal(loaded[name].read(), values)
        for name in roles:
            self.assertEqual(loaded[name].dtype, "I8")
            self.assertEqual(loaded[name[:-len("weight")] + "scale"].dtype, "F8_E8M0")

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_native_embedding_and_recurrent_output_column_direction(self):
        source = self.assets()
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(hidden_size=256, linear_num_key_heads=2, linear_num_value_heads=4,
                                     linear_key_head_dim=32, linear_value_head_dim=128)
        for heads in (4, 6):
            with self.subTest(heads=heads):
                columns = heads * 128
                config["text_config"]["linear_num_value_heads"] = heads
                (source / "config.json").write_text(json.dumps(config))
                path = self.root / f"recurrent-{heads}.gguf"
                writer = gguf.GGUFWriter(path, "qwen35")
                writer.add_block_count(1)
                embedding = np.random.default_rng(39).integers(0, 256, (4, 144), dtype=np.uint8)
                writer.add_tensor("token_embd.weight", embedding, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
                writer.add_tensor("output.weight", np.ones((4, 256), dtype=np.float32))
                # Construct valid Q4_K blocks independently: d/dmin(F16),
                # scales(U8[12]), qs(U8[128]). Positive exact scales avoid a floating-point
                # conversion hiding any packed-column permutation error.
                raw = np.random.default_rng(97).integers(0, 256, (3, columns // 256, 144), dtype=np.uint8)
                for row in range(3):
                    for block in range(columns // 256):
                        raw[row, block, :4] = np.asarray([2 ** (-4 - row - block), 2 ** (-7 - row - block)], dtype="<f2").view(np.uint8)
                writer.add_tensor("blk.0.ssm_out.weight", raw.reshape(3, -1), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                output = self.root / f"output-{heads}"
                with patch.object(importer, "to_bfloat16", wraps=importer.to_bfloat16) as convert:
                    importer.import_package(path, output, source)
                    # Only the unquantized output.weight needs BF16 conversion.
                    self.assertEqual(convert.call_count, 1)
                loaded = importer.scan_safetensors(output)
                embedding_name = "model.language_model.embed_tokens.weight"
                np.testing.assert_array_equal(loaded[embedding_name].read(), embedding.reshape(-1))
                np.testing.assert_array_equal(loaded[embedding_name + ".__qnk_shape"].read(), [4, 4, 256])
                name = "model.language_model.layers.0.linear_attn.out_proj.weight"
                np.testing.assert_array_equal(loaded[name].read(), raw.reshape(-1))
                indices = loaded[name + ".__qnk_columns"].read()
                self.assertEqual(loaded[name + ".__qnk_columns"].dtype, "I64")
                expected_indices = np.asarray([((head % 2) * (heads // 2) + head // 2) * 128 + lane
                                               for head in range(heads) for lane in range(128)])
                np.testing.assert_array_equal(indices, expected_indices)
                np.testing.assert_array_equal(np.sort(indices), np.arange(columns))
                decoded = gguf.dequantize(raw.reshape(3, -1), gguf.GGMLQuantizationType.Q4_K).reshape(3, columns).astype(np.float64)
                reference = importer.undo_gguf_transform(name, decoded, config)
                logical_input = np.random.default_rng(413).uniform(-1, 1, (5, columns))
                expected = logical_input @ reference.T
                actual = logical_input[:, indices] @ decoded.T
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
                if heads == 6:
                    wrong_direction = logical_input[:, np.argsort(indices)] @ decoded.T
                    self.assertGreater(np.max(np.abs(actual - wrong_direction)), 0.1)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_tied_quantized_embedding_stays_native_for_lm_head(self):
        source = self.assets()
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(hidden_size=256, tie_word_embeddings=True)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "tied.gguf"
        writer = gguf.GGUFWriter(path, "qwen35")
        writer.add_block_count(1)
        embedding = np.random.default_rng(55).integers(0, 256, (4, 210), dtype=np.uint8)
        writer.add_tensor("token_embd.weight", embedding, raw_dtype=gguf.GGMLQuantizationType.Q6_K)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        output = self.root / "output"
        with patch.object(importer, "to_bfloat16", side_effect=AssertionError("Qn_K embedding was expanded to BF16")):
            importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            self.assertEqual(loaded[name].dtype, "U8")
            np.testing.assert_array_equal(loaded[name].read(), embedding.reshape(-1))
            np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [6, 4, 256])

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_fused_quantized_experts_split_without_requantization(self):
        source = self.assets(model_type="qwen3_5_moe")
        config = json.loads((source / "config.json").read_text())
        config["text_config"].update(hidden_size=256, num_experts=2, moe_intermediate_size=256)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "fused.gguf"
        writer = gguf.GGUFWriter(path, "qwen35moe")
        writer.add_block_count(1)
        writer.add_tensor("token_embd.weight", np.ones((4, 256), dtype=np.float32))
        writer.add_tensor("output.weight", np.ones((4, 256), dtype=np.float32))
        raw = np.random.default_rng(19).integers(0, 256, (2, 512, 144), dtype=np.uint8)
        writer.add_tensor("blk.0.ffn_gate_up_exps.weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        writer.add_tensor("blk.0.ffn_down_exps.weight", np.ones((2, 256, 256), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        output = self.root / "output"
        importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        for expert in range(2):
            for role, begin in (("gate", 0), ("up", 256)):
                name = f"model.language_model.layers.0.mlp.experts.{expert}.{role}_proj.weight"
                np.testing.assert_array_equal(loaded[name].read(), raw[expert, begin:begin + 256].reshape(-1))
                np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [4, 256, 256])
            self.assertEqual(loaded[f"model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight"].dtype, "BF16")

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_rejects_missing_block_metadata_and_wrong_geometry(self):
        source = self.assets()
        cases = ("missing_count", "embedding_shape", "output_shape", "token_count", "undeclared_layer")
        for case in cases:
            with self.subTest(case=case):
                path = self.root / (case + ".gguf")
                writer = gguf.GGUFWriter(path, "qwen35")
                if case != "missing_count":
                    writer.add_block_count(1)
                embedding_shape = (4, 16) if case == "embedding_shape" else (4, 8)
                output_shape = (5, 8) if case == "output_shape" else (4, 8)
                writer.add_tensor("token_embd.weight", np.ones(embedding_shape, dtype=np.float32))
                writer.add_tensor("output.weight", np.ones(output_shape, dtype=np.float32))
                if case == "token_count":
                    writer.add_token_list(["a", "b", "c"])
                if case == "undeclared_layer":
                    writer.add_tensor("blk.1.attn_norm.weight", np.ones(8, dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                output = self.root / (case + "-output")
                with self.assertRaises(ValueError):
                    importer.import_package(path, output, source)
                self.assertFalse(output.exists())


    def family_assets(self, family, quant=None):
        source = self.assets(quant, family) if family in importer.QWEN_TYPES else self.assets()
        if family in importer.QWEN_TYPES:
            config = json.loads((source / "config.json").read_text())
            text = config["text_config"]
        else:
            config = {"model_type": family, "dtype": "bfloat16"}
            text = config
            if quant is not None:
                config["quantization_config"] = quant
        text.update(hidden_size=256, vocab_size=4, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=64, num_experts=2, num_local_experts=2,
                    n_routed_experts=2, moe_intermediate_size=256, intermediate_size=256,
                    tie_word_embeddings=False, mtp_num_hidden_layers=0)
        (source / "config.json").write_text(json.dumps(config))
        return source, config

    def test_all_family_hf_awq_gptq_preserve_native_names(self):
        for family in ("gpt_oss", "deepseek_v4", "qwen4_exp"):
            for method in ("awq", "gptq"):
                with self.subTest(family=family, method=method):
                    source, config = self.family_assets(family, {"quant_method": method, "bits": 4, "group_size": 8})
                    prefix = ("layers.0.ffn.experts.0.w1." if family == "deepseek_v4" else
                              "model.layers.0.self_attn.q_proj.")
                    codes = (np.arange(16 * 8).reshape(16, 8) * 3) % 16
                    zeros = np.tile(np.arange(8), (2, 1))
                    scales = (np.arange(16).reshape(2, 8) + 1).astype(np.float32) / 8
                    write_safetensors(source / "model.safetensors", {
                        prefix + "qweight": ("I32", pack_columns(codes, True) if method == "awq" else pack_gptq(codes)),
                        prefix + "qzeros": ("I32", pack_columns(zeros, True) if method == "awq" else pack_columns((zeros - 1) % 16)),
                        prefix + "scales": ("F16", scales.astype(np.float16)),
                    })
                    output = self.root / (family + method)
                    importer.import_package(source, output)
                    loaded = importer.scan_safetensors(output)
                    target = importer.canonical_tensor_name(prefix + "weight", config)
                    np.testing.assert_array_equal(native_codes(loaded, target), codes.T)
                    np.testing.assert_array_equal(loaded[target + ".__int4_scales"].read(), scales.T)
                    if family != "qwen4_exp":
                        self.assertFalse(any(name.startswith("model.language_model.") for name in loaded))
                    self.assertNotIn("quantization_config", json.loads((output / "config.json").read_text()))

    def test_deepseek_source_fp8_mxfp4_and_fp32_parameters_stay_native(self):
        source, _ = self.family_assets("deepseek_v4", {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0",
                                                       "weight_block_size": [128, 128]})
        bits = np.arange(256 * 128, dtype=np.uint8).reshape(256, 128) % 126
        blocks = np.arange(256 * 64, dtype=np.uint8).view(np.int8).reshape(256, 64)
        exact = np.asarray([1.000031, 2.000061], dtype=np.float32)
        write_safetensors(source / "model.safetensors", {
            "layers.0.attn.wq_a.weight": ("F8_E4M3", bits),
            "layers.0.attn.wq_a.scale": ("F32", exact.reshape(2, 1)),
            "layers.0.ffn.experts.0.w1.weight": ("I8", blocks),
            "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", np.full((256, 4), 125, dtype=np.uint8)),
            "mtp.0.hc_head_scale": ("F32", exact),
            "layers.0.ffn.gate.bias": ("F32", exact),
            "layers.0.attn_norm.weight": ("F32", np.ones(256, dtype=np.float32)),
        })
        output = self.root / "native-ds"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        np.testing.assert_array_equal(loaded["layers.0.attn.wq_a.weight"].read(), bits)
        np.testing.assert_array_equal(loaded["layers.0.attn.wq_a.scale"].read(), exact.reshape(2, 1))
        self.assertEqual(loaded["layers.0.attn.wq_a.weight"].dtype, "F8_E4M3")
        np.testing.assert_array_equal(loaded["layers.0.ffn.experts.0.w1.weight"].read(), blocks)
        np.testing.assert_array_equal(loaded["mtp.0.hc_head_scale"].read(), exact)
        self.assertEqual(loaded["layers.0.attn_norm.weight"].dtype, "BF16")

    def test_gptoss_source_mxfp4_banks_preserved(self):
        source, _ = self.family_assets("gpt_oss", {"quant_method": "mxfp4"})
        blocks = np.arange(2 * 4 * 2 * 16, dtype=np.uint8).reshape(2, 4, 2, 16)
        scales = np.arange(2 * 4 * 2, dtype=np.uint8).reshape(2, 4, 2)
        prefix = "model.layers.0.mlp.experts.gate_up_proj_"
        write_safetensors(source / "model.safetensors", {prefix + "blocks": ("U8", blocks), prefix + "scales": ("U8", scales)})
        output = self.root / "native-gptoss"
        importer.import_package(source, output)
        loaded = importer.scan_safetensors(output)
        np.testing.assert_array_equal(loaded[prefix + "blocks"].read(), blocks)
        np.testing.assert_array_equal(loaded[prefix + "scales"].read(), scales)

    @staticmethod
    def q4k_rows(rows, columns=256, seed=0):
        blocks = np.zeros((rows, columns // 256, 144), dtype=np.uint8)
        blocks[..., :2] = np.asarray([0.125], dtype=np.float16).view(np.uint8)
        blocks[..., 2:4] = np.asarray([0.03125], dtype=np.float16).view(np.uint8)
        blocks[..., 4:16] = (np.arange(12, dtype=np.uint8) + seed) % 32
        blocks[..., 16:] = (np.arange(128, dtype=np.uint8) + seed) % 251
        return blocks.reshape(rows, -1)

    @staticmethod
    def q8_0_fixture(rows, columns, seed):
        if columns % 32:
            raise ValueError("Q8_0 fixture columns must be divisible by 32")
        blocks = np.zeros((rows, columns // 32, 34), dtype=np.uint8)
        blocks[..., :2] = np.asarray([1 / 512], dtype="<f2").view(np.uint8).reshape(1, 1, 2)
        codes = ((np.arange(rows * columns, dtype=np.int16).reshape(rows, columns) + seed) % 17 - 8).astype(np.int8)
        blocks[..., 2:] = codes.reshape(rows, columns // 32, 32).view(np.uint8)
        raw = blocks.reshape(rows, -1)
        # This is the independent GGML Q8_0 decode oracle: F16 block scale
        # multiplied by the 32 signed codes in each block.
        scales = blocks[..., :2].copy().view("<f2").reshape(rows, columns // 32).astype(np.float32)
        decoded_codes = blocks[..., 2:].copy().view(np.int8).astype(np.float32)
        decoded = (scales[..., None] * decoded_codes).reshape(rows, columns)
        return raw, decoded

    @staticmethod
    def write_gguf(path, arch, tensors, metadata=None):
        writer = gguf.GGUFWriter(path, arch)
        writer.add_block_count(1)
        if metadata:
            metadata(writer)
        for name, values, dtype in tensors:
            writer.add_tensor(name, values, raw_dtype=dtype) if dtype is not None else writer.add_tensor(name, values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gptoss_gguf_packed_experts_bias_and_attention(self):
        source, _ = self.family_assets("gpt_oss")
        path = self.root / "gpt-oss.gguf"
        raw = self.q4k_rows(2 * 256, seed=3).reshape(2, 256, 144)
        gate_bias = np.arange(512, dtype=np.float32).reshape(2, 256) / 8
        tensors = [("token_embd.weight", np.ones((4, 256), dtype=np.float32), None),
                   ("output.weight", self.q4k_rows(4), gguf.GGMLQuantizationType.Q4_K),
                   ("blk.0.attn_q.bias", np.arange(256, dtype=np.float32), None),
                   ("blk.0.attn_sinks.weight", np.arange(4, dtype=np.float32), None)]
        for role in ("gate", "up", "down"):
            tensors.append((f"blk.0.ffn_{role}_exps.weight", raw, gguf.GGMLQuantizationType.Q4_K))
            tensors.append((f"blk.0.ffn_{role}_exps.bias", gate_bias + (1 if role == "up" else 0), None))
        self.write_gguf(path, "gpt-oss", tensors)
        output = self.root / "gpt-oss-output"
        importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        for expert in range(2):
            for role in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{expert}.{role}_proj.weight"
                np.testing.assert_array_equal(loaded[name].read(), raw[expert].reshape(-1))
                np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [4, 256, 256])
            np.testing.assert_array_equal(loaded[f"model.layers.0.mlp.experts.{expert}.up_proj.bias"].read(), importer.to_bfloat16(gate_bias[expert] + 1))
        np.testing.assert_array_equal(importer.as_float(loaded["model.layers.0.self_attn.q_proj.bias"].read(), "BF16"), np.arange(256))
        self.assertIn("model.layers.0.self_attn.sinks", loaded)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_deepseek_gguf_hyper_connections_hash_and_experts(self):
        source, config = self.family_assets("deepseek_v4")
        config.update(dspark_target_layer_ids=[0], dspark_block_size=2, dspark_noise_token_id=3,
                      dspark_markov_rank=32, num_nextn_predict_layers=1,
                      compress_ratios=[4], head_dim=512, index_n_heads=2,
                      index_head_dim=128, index_topk=8, q_lora_rank=256,
                      o_groups=2, o_lora_rank=256, hc_mult=4)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "deepseek.gguf"
        raw = self.q4k_rows(512, seed=5).reshape(2, 256, 144)
        hc = self.q4k_rows(6, seed=7)
        routing = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32)
        exact = np.asarray([1.000031, 2.000061], dtype=np.float32)
        ape_raw, ape_oracle = self.q8_0_fixture(4, 1024, seed=19)
        index_ape_raw, index_ape_oracle = self.q8_0_fixture(4, 256, seed=29)
        compressor_kv, _ = self.q8_0_fixture(1024, 256, seed=31)
        compressor_gate, _ = self.q8_0_fixture(1024, 256, seed=37)
        index_compressor_kv, _ = self.q8_0_fixture(256, 256, seed=41)
        index_compressor_gate, _ = self.q8_0_fixture(256, 256, seed=43)
        index_query, _ = self.q8_0_fixture(256, 256, seed=47)
        index_weights, _ = self.q8_0_fixture(2, 256, seed=53)
        tensors = [("token_embd.weight", np.ones((4, 256), dtype=np.float32), None),
                   ("output.weight", self.q4k_rows(4), gguf.GGMLQuantizationType.Q4_K),
                   ("blk.0.hc_attn_fn.weight", hc, gguf.GGMLQuantizationType.Q4_K),
                   ("blk.0.hc_attn_base.weight", exact, None),
                   ("blk.0.exp_probs_b.bias", exact, None),
                   ("blk.0.attn_compressor_ape.weight", ape_raw, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.indexer_compressor_ape.weight", index_ape_raw, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.attn_compressor_kv.weight", compressor_kv, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.attn_compressor_gate.weight", compressor_gate, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.indexer_compressor_kv.weight", index_compressor_kv, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.indexer_compressor_gate.weight", index_compressor_gate, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.indexer.attn_q_b.weight", index_query, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.indexer.proj.weight", index_weights, gguf.GGMLQuantizationType.Q8_0),
                   ("blk.0.ffn_gate_tid2eid.weight", routing, None)]
        for role in ("gate", "up", "down"):
            tensors.append((f"blk.0.ffn_{role}_exps.weight", raw, gguf.GGMLQuantizationType.Q4_K))
        self.write_gguf(path, "deepseek4", tensors)
        output = self.root / "deepseek-output"
        importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        for projection in ("w1", "w2", "w3"):
            np.testing.assert_array_equal(loaded[f"layers.0.ffn.experts.1.{projection}.weight"].read(), raw[1].reshape(-1))
        np.testing.assert_array_equal(loaded["layers.0.hc_attn_fn"].read(), hc.reshape(-1))
        np.testing.assert_array_equal(loaded["layers.0.hc_attn_fn.__qnk_shape"].read(), [4, 6, 256])
        np.testing.assert_array_equal(loaded["layers.0.ffn.gate.tid2eid"].read(), routing.astype(np.int64))
        for name, expected in (("layers.0.hc_attn_base", exact),
                               ("layers.0.ffn.gate.bias", exact),
                               ("layers.0.attn.compressor.ape", ape_oracle),
                               ("layers.0.attn.indexer.compressor.ape", index_ape_oracle)):
            self.assertEqual(loaded[name].dtype, "F32")
            np.testing.assert_array_equal(loaded[name].read(), expected)
            self.assertNotIn(name + ".__qnk_shape", loaded)
        native_q8 = {
            "layers.0.attn.compressor.wkv.weight": (compressor_kv, 1024, 256),
            "layers.0.attn.compressor.wgate.weight": (compressor_gate, 1024, 256),
            "layers.0.attn.indexer.compressor.wkv.weight": (index_compressor_kv, 256, 256),
            "layers.0.attn.indexer.compressor.wgate.weight": (index_compressor_gate, 256, 256),
            "layers.0.attn.indexer.wq_b.weight": (index_query, 256, 256),
            "layers.0.attn.indexer.weights_proj.weight": (index_weights, 2, 256),
        }
        for name, (packed, rows, columns) in native_q8.items():
            self.assertEqual(loaded[name].dtype, "U8", name)
            np.testing.assert_array_equal(loaded[name].read().reshape(rows, -1), packed)
            np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [264, rows, columns])
        saved = json.loads((output / "config.json").read_text())
        for key in ("dspark_target_layer_ids", "dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank"):
            self.assertNotIn(key, saved)
        self.assertEqual(json.loads((source / "config.json").read_text())["dspark_target_layer_ids"], [0])
        self.assertEqual(saved["num_nextn_predict_layers"], 1)
        self.assertFalse(any(name.startswith("mtp.") for name in loaded))

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_native_mxfp4_gguf_lossless_codes_scales_and_expert_order(self):
        for family, architecture in (("gpt_oss", "gpt-oss"), ("deepseek_v4", "deepseek4")):
            with self.subTest(family=family):
                source, _ = self.family_assets(family)
                path = self.root / (family + "-mx.gguf")
                blocks = np.arange(2 * 256 * 8 * 17, dtype=np.uint8).reshape(2, 256, 8, 17)
                blocks[..., 0] = 126
                raw = blocks.reshape(2, 256, 8 * 17)
                tensors = [("token_embd.weight", np.ones((4, 256), dtype=np.float32), None),
                           ("output.weight", np.ones((4, 256), dtype=np.float32), None)]
                for index, role in enumerate(("gate", "up", "down")):
                    values = raw.copy()
                    values[..., 1::17] ^= index
                    tensors.append((f"blk.0.ffn_{role}_exps.weight", values, gguf.GGMLQuantizationType.MXFP4))
                    if family == "gpt_oss":
                        tensors.append((f"blk.0.ffn_{role}_exps.bias", np.full((2, 256), index + 1, dtype=np.float32), None))
                self.write_gguf(path, architecture, tensors)
                output = self.root / (family + "-mx-output")
                importer.import_package(path, output, source)
                loaded = importer.scan_safetensors(output)
                table = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
                for expert in range(2):
                    for role_index, role in enumerate(("gate", "up", "down")):
                        if family == "deepseek_v4":
                            projection = {"gate": "w1", "up": "w3", "down": "w2"}[role]
                            prefix = f"layers.0.ffn.experts.{expert}.{projection}."
                            packed, scales = loaded[prefix + "weight"].read().view(np.uint8), loaded[prefix + "scale"].read()
                        else:
                            prefix = f"model.layers.0.mlp.experts.{expert}." + ("down_proj" if role == "down" else "gate_up_proj")
                            packed, scales = loaded[prefix + "_blocks"].read().view(np.uint8), loaded[prefix + "_scales"].read()
                            if role != "down":
                                packed, scales = packed[role_index::2], scales[role_index::2]
                        codes = np.empty((256, 256), dtype=np.uint8)
                        codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
                        decoded = table[codes] * np.exp2(scales.astype(np.float32) - 127).repeat(32, axis=1)
                        original = tensors[2 + role_index * (2 if family == "gpt_oss" else 1)][1][expert]
                        expected = gguf.dequantize(original, gguf.GGMLQuantizationType.MXFP4)
                        np.testing.assert_array_equal(decoded, expected)
                        # The GGUF stream can be reconstructed byte for byte.
                        reconstructed = np.empty((256, 8, 17), dtype=np.uint8)
                        reconstructed[..., 0] = scales
                        logical = codes.reshape(256, 8, 32)
                        reconstructed[..., 1:] = logical[..., :16] | (logical[..., 16:] << 4)
                        np.testing.assert_array_equal(reconstructed.reshape(original.shape), original)
                    if family == "gpt_oss":
                        bias = importer.as_float(loaded[f"model.layers.0.mlp.experts.{expert}.gate_up_proj.bias"].read(), "BF16")
                        np.testing.assert_array_equal(bias[::2], 1)
                        np.testing.assert_array_equal(bias[1::2], 2)

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_qwen4exp_gguf_qsa_ple_hash_and_native_embedding_rows(self):
        source, config = self.family_assets("qwen4_exp")
        config["text_config"].update(indexer_n_heads=2, indexer_head_dim=4, ple_layer_ids=[1], split_ngram_parts=2)
        (source / "config.json").write_text(json.dumps(config))
        path = self.root / "qwen4exp.gguf"
        query, key, ple = self.q4k_rows(8, seed=1), self.q4k_rows(4, seed=2), self.q4k_rows(10, seed=3)
        multipliers = [2 ** 45 + 7, 2 ** 46 + 11, 2 ** 47 + 13]
        def metadata(writer):
            writer.add_key_value("qwen4exp.ple.layer_multipliers", multipliers, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)
            writer.add_array("qwen4exp.ple.head_offsets", [0, 5])
            writer.add_array("qwen4exp.ple.head_vocab_sizes", [5, 5])
        self.write_gguf(path, "qwen4exp", [
            ("token_embd.weight", np.ones((4, 256), dtype=np.float32), None),
            ("output.weight", np.ones((4, 256), dtype=np.float32), None),
            ("blk.0.indexer.q_proj.weight", query, gguf.GGMLQuantizationType.Q4_K),
            ("blk.0.indexer.k_proj.weight", key, gguf.GGMLQuantizationType.Q4_K),
            ("blk.0.hc_attn_down.weight", self.q4k_rows(2), gguf.GGMLQuantizationType.Q4_K),
            ("blk.0.hc_attn_norm.weight", np.full(256, 1.25, dtype=np.float32), None),
            ("blk.0.indexer.q_norm.weight", np.full(4, 1.5, dtype=np.float32), None),
            ("blk.0.ple_norm_key.weight", np.full(256, 1.75, dtype=np.float32), None),
            ("per_layer_token_embd.weight", ple, gguf.GGMLQuantizationType.Q4_K),
        ], metadata)
        output = self.root / "qwen4exp-output"
        importer.import_package(path, output, source)
        loaded = importer.scan_safetensors(output)
        prefix = "model.language_model.layers.0."
        np.testing.assert_array_equal(loaded[prefix + "self_attn.indexer.index_qk_proj.weight"].read(), np.concatenate((query, key)).reshape(-1))
        np.testing.assert_array_equal(loaded[prefix + "self_attn.indexer.index_qk_proj.weight.__qnk_shape"].read(), [4, 12, 256])
        np.testing.assert_array_equal(importer.as_float(loaded[prefix + "attn_hyper_connection.hc_norm.weight"].read(), "BF16"), .25)
        np.testing.assert_array_equal(importer.as_float(loaded[prefix + "self_attn.indexer.q_layernorm.weight"].read(), "BF16"), .5)
        np.testing.assert_array_equal(importer.as_float(loaded[prefix + "ple.norm_key.weight"].read(), "BF16"), .75)
        np.testing.assert_array_equal(loaded[prefix + "ple.ple_embedding.layer_multipliers"].read(), multipliers)
        for shard in range(2):
            name = prefix + f"ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            np.testing.assert_array_equal(loaded[name].read(), ple[shard * 5:(shard + 1) * 5].reshape(-1))
            np.testing.assert_array_equal(loaded[name + ".__qnk_shape"].read(), [4, 5, 256])

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_gguf_rejects_unsupported_quantized_matrix_and_standalone_dspark(self):
        for family, arch in (("qwen3_5", "qwen35"), ("deepseek_v4", "dflash")):
            source, _ = self.family_assets(family)
            path = self.root / (arch + "-unsupported.gguf")
            raw = np.zeros((4, 320), dtype=np.uint8)  # Q8_1 has 40 bytes per 32-value block.
            self.write_gguf(path, arch, [("token_embd.weight", raw, gguf.GGMLQuantizationType.Q8_1),
                                         ("output.weight", np.ones((4, 256), dtype=np.float32), None)])
            output = self.root / (arch + "-unsupported-output")
            with self.assertRaisesRegex(ValueError, "native Qn_K/MXFP4|only draft weights"):
                importer.import_package(path, output, source)
            self.assertFalse(output.exists())


    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_split_gguf_maps_native_shards_from_any_member(self):
        source, _ = self.family_assets("gpt_oss")
        embedding, output = self.q4k_rows(4, seed=7), self.q4k_rows(4, seed=9)
        paths = [self.root / f"gpt-oss-0000{index + 1}-of-00002.gguf" for index in range(2)]
        for index, path in enumerate(paths):
            def metadata(writer, index=index):
                writer.add_uint16("split.no", index)
                writer.add_uint16("split.count", 2)
                writer.add_int32("split.tensors.count", 3)
            tensors = [("token_embd.weight", embedding, gguf.GGMLQuantizationType.Q4_K)] if index == 0 else [
                ("output.weight", output, gguf.GGMLQuantizationType.Q4_K),
                ("output_norm.weight", np.ones(256, dtype=np.float32), None)]
            self.write_gguf(path, "gpt-oss", tensors, metadata)
        destination = self.root / "split-output"
        importer.import_package(paths[1], destination, source)
        loaded = importer.scan_safetensors(destination)
        np.testing.assert_array_equal(loaded["model.embed_tokens.weight"].read(), embedding.reshape(-1))
        np.testing.assert_array_equal(loaded["lm_head.weight"].read(), output.reshape(-1))
        np.testing.assert_array_equal(loaded["model.norm.weight"].read(), importer.to_bfloat16(np.ones(256)))

    def test_open_gguf_preserves_split_filename_casing_from_any_member(self):
        source = self.root / "Model-00002-OF-00002.GGUF"
        expected = [self.root / "Model-00001-OF-00002.GGUF", source]
        paths_seen = []
        little_endian = object()

        class Field:
            def __init__(self, value):
                self.value = value

            def contents(self):
                return self.value

        class Reader:
            def __init__(self, path):
                self.path = Path(path)
                paths_seen.append(self.path)
                index = 0 if self.path.name == expected[0].name else 1
                self.fields = {
                    "split.count": Field(2),
                    "split.no": Field(index),
                    "split.tensors.count": Field(2),
                    "general.architecture": Field("gpt-oss"),
                }
                self.tensors = [self.path.name]
                self.endianess = little_endian

            def get_field(self, name):
                return self.fields.get(name)

        fake_gguf = SimpleNamespace(GGUFReader=Reader, GGUFEndian=SimpleNamespace(LITTLE=little_endian))
        expected_paths = {str(path) for path in expected}
        with patch.object(Path, "is_file", lambda path: str(path) in expected_paths):
            archive = importer.open_gguf(source, fake_gguf)

        self.assertEqual([str(path) for path in paths_seen], [str(source), str(expected[0])])
        self.assertEqual([str(reader.path) for reader in archive.readers], [str(expected[0]), str(source)])
        self.assertEqual(archive.tensors, [expected[0].name, source.name])

    @unittest.skipIf(gguf is None, "optional official gguf package is not installed")
    def test_split_gguf_rejects_missing_duplicate_and_inconsistent_shards(self):
        source, _ = self.family_assets("gpt_oss")
        for case in ("missing", "index", "count", "architecture", "duplicate", "layout", "layout_later"):
            with self.subTest(case=case):
                paths = [self.root / f"{case}-0000{index + 1}-of-00002.gguf" for index in range(2)]
                for index, path in enumerate(paths):
                    if case == "missing" and index == 1:
                        continue
                    def metadata(writer, index=index):
                        writer.add_uint16("split.no", 0 if case == "index" else index)
                        writer.add_uint16("split.count", 2)
                        writer.add_int32("split.tensors.count", 3 if case == "count" else 2)
                        if case == "layout":
                            writer.add_embedding_length(256 if index == 0 else 512)
                        elif case == "layout_later" and index == 1:
                            writer.add_embedding_length(512)
                    tensor = "token_embd.weight" if index == 0 or case == "duplicate" else "output.weight"
                    arch = "deepseek4" if case == "architecture" and index == 1 else "gpt-oss"
                    self.write_gguf(path, arch, [(tensor, np.ones((4, 256), dtype=np.float32), None)], metadata)
                destination = self.root / (case + "-split-output")
                expected_error = "embedding_length" if case == "layout_later" else None
                error_context = (self.assertRaisesRegex(ValueError, expected_error)
                                 if expected_error else self.assertRaises(ValueError))
                with error_context:
                    importer.import_package(paths[0], destination, source)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
