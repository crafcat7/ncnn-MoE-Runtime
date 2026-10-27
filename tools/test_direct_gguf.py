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


add_metadata = fixtures.add_test_gguf_metadata


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


def fp4_copy(fixture, directory: Path, format_name: str) -> tuple[Path, Path]:
    """Build independently packed MXFP4 or NVFP4 Expert banks and their baseline."""
    if format_name not in ("MXFP4", "NVFP4"):
        raise ValueError(f"unsupported FP4 format: {format_name}")
    gguf = fixtures.gguf
    reader = gguf.GGUFReader(fixture.source)
    destination = directory / f"{format_name.lower()}.gguf"
    architecture = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(destination, architecture)
    for name, field in reader.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    reference = {name: (info.dtype, info.read().copy())
                 for name, info in fixtures.importer.scan_safetensors(fixture.baseline).items()}
    table = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
                       if format_name == "MXFP4" else [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                       np.float32)
    for index, tensor in enumerate(reader.tensors):
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if not match:
            writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)
            continue
        shape = tuple(int(value) for value in tensor.shape[::-1])
        code_seed = index if format_name == "MXFP4" else index + 300
        codes = np.random.default_rng(code_seed).integers(0, 16, shape, dtype=np.uint8)
        if format_name == "MXFP4":
            grouped = codes.reshape(*shape[:-1], shape[-1] // 32, 32)
            exponents = np.random.default_rng(index + 100).integers(118, 122, grouped.shape[:-1], dtype=np.uint8)
            raw = np.empty((*grouped.shape[:-1], 17), np.uint8)
            raw[..., 0] = exponents
            raw[..., 1:] = grouped[..., :16] | (grouped[..., 16:] << 4)
            scale_values = np.exp2(exponents.astype(np.float32) - 127)
            dtype = gguf.GGMLQuantizationType.MXFP4
        else:
            columns = shape[-1]
            grouped = codes.reshape(*shape[:-1], columns // 64, 4, 16)
            rng = np.random.default_rng(index + 600)
            exponents = rng.integers(0, 2, grouped.shape[:-1], dtype=np.uint8)
            mantissas = rng.integers(1, 8, grouped.shape[:-1], dtype=np.uint8)
            scale_bytes = (exponents << 3) | mantissas
            raw = np.empty((*shape[:-1], columns // 64, 36), np.uint8)
            raw[..., :4] = scale_bytes
            raw[..., 4:] = (grouped[..., :8] | (grouped[..., 8:] << 4)).reshape(
                *shape[:-1], columns // 64, 32)
            scale_values = np.where(exponents == 0, mantissas.astype(np.float32) * 2 ** -10,
                                    (1 + mantissas.astype(np.float32) / 8) * np.exp2(exponents.astype(np.float32) - 8))
            dtype = gguf.GGMLQuantizationType.NVFP4
        writer.add_tensor(tensor.name, raw.reshape(*shape[:-1], -1), raw_dtype=dtype)
        decoded = (table[grouped] * scale_values[..., None]).reshape(shape)
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
    baseline = directory / f"{format_name.lower()}-baseline"
    fixtures.assets(baseline, json.loads((fixture.baseline / "config.json").read_text()))
    fixtures.write_safetensors(baseline / "model.safetensors", reference)
    return baseline, destination


def q8_0_copy(baseline: Path, source: Path, directory: Path, label: str) -> tuple[Path, Path]:
    """Native 34-byte blocks and an independently decoded DeepSeek HF oracle."""
    gguf = fixtures.gguf
    reader = gguf.GGUFReader(source)
    architecture = reader.fields["general.architecture"].contents()
    assert architecture == "deepseek4"
    destination = directory / f"{label}.gguf"
    writer = gguf.GGUFWriter(destination, architecture)
    for name, field in reader.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    reference = {name: (info.dtype, info.read().copy())
                 for name, info in fixtures.importer.scan_safetensors(baseline).items()}
    dense_names = {fixtures.family_gguf_name(name, "deepseek_v4"): name
                   for name in reference if ".ffn.experts." not in name}
    converted = set()
    for index, tensor in enumerate(reader.tensors):
        # Preserve the independent MXFP4 oracle, F32 auxiliary weights and
        # layout metadata. Existing native Qn_K matrices cover embedding,
        # LM Head, every MLA projection, Shared FFN and routed Expert banks.
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
            writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)
            continue
        shape = tuple(int(value) for value in tensor.shape[::-1])
        assert len(shape) in (2, 3) and shape[-1] % 32 == 0
        values = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(shape)
        blocks = values.reshape(*shape[:-1], shape[-1] // 32, 32)
        maximum = np.max(np.abs(blocks), axis=-1)
        # Powers of two keep the independently decoded oracle exactly BF16
        # representable. Negative scales exercise the actual signed Q8_0 ABI.
        exponents = np.ceil(np.log2(np.where(maximum > 0, maximum / 127, 1 / 1024)))
        scales = np.exp2(exponents).astype(np.float32)
        signs = np.arange(scales.size).reshape(scales.shape) + index
        scales = np.where(signs % 3 == 1, -scales, scales).astype("<f2")
        codes = np.clip(np.rint(blocks / scales[..., None]), -128, 127).astype(np.int8)
        raw = np.empty((*blocks.shape[:-1], 34), np.uint8)
        raw[..., :2] = scales.view(np.uint8).reshape(*scales.shape, 2)
        raw[..., 2:] = codes.view(np.uint8)
        packed = raw.reshape(*shape[:-1], -1)
        writer.add_tensor(tensor.name, packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        # Decode bytes directly rather than sharing the runtime/importer or
        # the official decoder that generated the original Qn_K fixture.
        decoded_scales = raw[..., :2].copy().view("<f2").reshape(scales.shape).astype(np.float32)
        decoded_codes = raw[..., 2:].copy().view(np.int8).astype(np.float32)
        decoded = (decoded_scales[..., None] * decoded_codes).reshape(shape)
        np.testing.assert_array_equal(decoded, gguf.dequantize(packed, gguf.GGMLQuantizationType.Q8_0).reshape(shape))
        rounded = fixtures.bfloat16(decoded)
        np.testing.assert_array_equal(fixtures.float32(rounded), decoded)
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if match:
            role = {"gate": "w1", "up": "w3", "down": "w2"}[match[2]]
            for expert in range(shape[0]):
                reference[f"layers.{match[1]}.ffn.experts.{expert}.{role}.weight"] = ("BF16", rounded[expert])
        else:
            reference[dense_names[tensor.name]] = ("BF16", rounded)
        converted.add(tensor.name)
    assert {"token_embd.weight", "output.weight", "blk.0.attn_q_a.weight", "blk.0.attn_output_a.weight",
            "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight"} <= converted
    if "fp4" not in label:
        assert "blk.0.ffn_gate_exps.weight" in converted
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    decoded_baseline = directory / f"{label}-baseline"
    fixtures.assets(decoded_baseline, json.loads((baseline / "config.json").read_text()))
    fixtures.write_safetensors(decoded_baseline / "model.safetensors", reference)
    return decoded_baseline, destination


def flash_next_blocks_copy(fixture, directory: Path) -> tuple[Path, Path]:
    """Actual Flash-Next roles: IQ4_NL PLE rows and mixed Q5_1 Expert down banks."""
    gguf = fixtures.gguf
    reader = gguf.GGUFReader(fixture.source)
    architecture = reader.fields["general.architecture"].contents()
    assert architecture == "qwen4exp"
    destination = directory / "flash-next-blocks.gguf"
    writer = gguf.GGUFWriter(destination, architecture)
    for name, field in reader.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    reference = {name: (info.dtype, info.read().copy())
                 for name, info in fixtures.importer.scan_safetensors(fixture.baseline).items()}
    ple_shards = sorted(name for name in reference if re.fullmatch(
        re.escape(fixtures.PREFIX) + r"layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight", name))
    ple_table = next(tensor for tensor in reader.tensors if tensor.name == "per_layer_token_embd.weight")
    assert len(ple_shards) == 2
    assert sum(reference[name][1].shape[0] for name in ple_shards) == int(ple_table.shape[1])
    assert all(reference[name][1].shape[0] > 0 and reference[name][1].shape[1] == int(ple_table.shape[0])
               for name in ple_shards)
    converted = set()
    for index, tensor in enumerate(reader.tensors):
        down = re.fullmatch(r"blk\.(\d+)\.ffn_down_exps\.weight", tensor.name)
        ple = tensor.name == "per_layer_token_embd.weight"
        if not ple and not down:
            writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)
            continue
        shape = tuple(int(value) for value in tensor.shape[::-1])
        assert shape[-1] % 32 == 0 and len(shape) == (2 if ple else 3)
        block_shape = (*shape[:-1], shape[-1] // 32)
        block_index = np.arange(np.prod(block_shape)).reshape(block_shape)
        # Small power-of-two scales make every decoded value exactly BF16.
        scales = np.exp2(-9 - block_index % 3).astype("<f2")
        values = np.arange(np.prod(shape)).reshape(*block_shape, 32)
        codes = ((values + block_index[..., None] * 5 + index) % (16 if ple else 32)).astype(np.uint8)
        if ple:
            dtype = gguf.GGMLQuantizationType.IQ4_NL
            scales = np.where(block_index % 3 == 1, -scales, scales).astype("<f2")
            raw = np.empty((*block_shape, 18), np.uint8)
            raw[..., :2] = scales.view(np.uint8).reshape(*block_shape, 2)
            raw[..., 2:] = codes[..., :16] | (codes[..., 16:] << 4)
        else:
            dtype = gguf.GGMLQuantizationType.Q5_1
            minimum = (-16 * scales.astype(np.float32)).astype("<f2")
            raw = np.empty((*block_shape, 24), np.uint8)
            raw[..., :2] = scales.view(np.uint8).reshape(*block_shape, 2)
            raw[..., 2:4] = minimum.view(np.uint8).reshape(*block_shape, 2)
            raw[..., 4:8] = np.packbits(codes >> 4, axis=-1, bitorder="little")
            raw[..., 8:] = (codes[..., :16] & 15) | (codes[..., 16:] << 4)
        packed = raw.reshape(*shape[:-1], -1)
        writer.add_tensor(tensor.name, packed, raw_dtype=dtype)
        # The upstream GGUF decoder supplies the reference, independently of
        # the runtime's block decoder, row fetch, transforms and cache paths.
        decoded = gguf.dequantize(packed, dtype).reshape(shape)
        assert np.any(decoded != 0)
        if down:
            np.testing.assert_array_equal(decoded.reshape(*block_shape, 32),
                                          scales.astype(np.float32)[..., None] * (codes.astype(np.float32) - 16))
        rounded = fixtures.bfloat16(decoded)
        np.testing.assert_array_equal(fixtures.float32(rounded), decoded)
        if ple:
            offset = 0
            for name in ple_shards:
                rows, columns = reference[name][1].shape
                assert columns == shape[1]
                reference[name] = ("BF16", rounded[offset:offset + rows].copy())
                offset += rows
            assert offset == shape[0] and not np.array_equal(rounded[0], rounded[reference[ple_shards[0]][1].shape[0]])
        else:
            for expert in range(shape[0]):
                name = f"{fixtures.PREFIX}layers.{down[1]}.mlp.experts.{expert}.down_proj.weight"
                reference[name] = ("BF16", rounded[expert])
        converted.add(tensor.name)
    assert converted == {"per_layer_token_embd.weight", "blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight"}
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    mixed = gguf.GGUFReader(destination)
    observed = {tensor.name: tensor.tensor_type for tensor in mixed.tensors}
    assert observed["per_layer_token_embd.weight"] == gguf.GGMLQuantizationType.IQ4_NL
    for layer in range(2):
        assert observed[f"blk.{layer}.ffn_down_exps.weight"] == gguf.GGMLQuantizationType.Q5_1
        assert observed[f"blk.{layer}.ffn_gate_exps.weight"] == gguf.GGMLQuantizationType.Q4_K
        assert observed[f"blk.{layer}.ffn_up_exps.weight"] == gguf.GGMLQuantizationType.Q4_K
    assert gguf.GGMLQuantizationType.Q6_K in observed.values()
    baseline = directory / "flash-next-blocks-baseline"
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
    native_formats = fixtures.NATIVE_GGUF_FORMATS
    gqa_native_formats = fixtures.NATIVE_GQA_GGUF_FORMATS

    def run_graph(label: str, baseline: Path, source: Path, on_demand: bool = False) -> None:
        command = [str(args.runner.resolve()), str(baseline), str(source)]
        if on_demand:
            command.append("--on-demand")
        if args.hybrid:
            command.append("--hybrid")
        mode = "on-demand" if on_demand else "eager"
        print(f"Checking {label}: {mode} native GGML blocks", flush=True)
        completed = subprocess.run(command)
        if completed.returncode == 77:
            raise SystemExit(77)
        completed.check_returncode()

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
                add_tensor = result.add_tensor

                def tensor(name, values, *tensor_args, **tensor_kwargs):
                    # GGML Qwen shared router gates omit their singleton output
                    # row. Keep the independent HF oracle as [1, hidden].
                    if (architecture == "qwen4exp" and name.endswith(".ffn_gate_inp_shexp.weight")
                            and values.dtype == np.float32):
                        assert values.shape == (1, model["text_config"]["hidden_size"])
                        values = values.reshape(-1)
                    return add_tensor(name, values, *tensor_args, **tensor_kwargs)

                result.add_tensor = tensor
                return result

            directory = root / label
            directory.mkdir()
            with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                fixture = (fixtures.prepare_family_gguf(directory, family, fp32=False) if family
                           else fixtures.prepare_gguf(directory, moe, reordered, fp32=False))
            if label == "qwen4_exp":
                gates = [tensor for tensor in fixtures.gguf.GGUFReader(fixture.source).tensors
                         if tensor.name.endswith(".ffn_gate_inp_shexp.weight")]
                assert len(gates) == 2 and all(tuple(tensor.shape) == (model["text_config"]["hidden_size"],)
                                               for tensor in gates)
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
                mxfp4 = fp4_copy(fixture, directory, "MXFP4")
                nvfp4 = fp4_copy(fixture, directory, "NVFP4")
                deferred = [(fixture.baseline, fixture.source, "QnK"),
                            (*mxfp4, "MXFP4"), (*nvfp4, "NVFP4")]
                for baseline, source, encoding in deferred:
                    command = [str(args.runner.resolve()), str(baseline), str(source), "--on-demand"]
                    if args.hybrid:
                        command.append("--hybrid")
                    print(f"Checking {label}: {encoding} with on-demand Expert cache", flush=True)
                    completed = subprocess.run(command)
                    if completed.returncode == 77:
                        return 77
                    completed.check_returncode()
                if family == "deepseek_v4":
                    mixed_sources = (q8_0_copy(fixture.baseline, fixture.source, directory, "q8_0"),
                                     q8_0_copy(*mxfp4, directory, "q8_0-mxfp4"),
                                     q8_0_copy(*nvfp4, directory, "q8_0-nvfp4"))
                    for baseline, source in mixed_sources:
                        for on_demand in (False, True):
                            command = [str(args.runner.resolve()), str(baseline), str(source)]
                            if on_demand:
                                command.append("--on-demand")
                            if args.hybrid:
                                command.append("--hybrid")
                            mode = "on-demand" if on_demand else "eager"
                            print(f"Checking {label}: {source.name} with {mode} Expert memory", flush=True)
                            completed = subprocess.run(command)
                            if completed.returncode == 77:
                                return 77
                            completed.check_returncode()
                if family == "qwen4_exp":
                    baseline, source = flash_next_blocks_copy(fixture, directory)
                    for on_demand in (False, True):
                        command = [str(args.runner.resolve()), str(baseline), str(source)]
                        if on_demand:
                            command.append("--on-demand")
                        if args.hybrid:
                            command.append("--hybrid")
                        mode = "on-demand" if on_demand else "eager"
                        print(f"Checking {label}: IQ4_NL PLE and mixed Q5_1 Experts with {mode} memory", flush=True)
                        completed = subprocess.run(command)
                        if completed.returncode == 77:
                            return 77
                        completed.check_returncode()
            block_root = root / (label + "-native-block-mixed")
            block_root.mkdir()
            with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                if family:
                    mixed = fixtures.prepare_family_gguf(
                        block_root, family, fp32=False, native_formats=native_formats,
                        ple_format=native_formats[0] if family == "qwen4_exp" else None)
                else:
                    mixed = fixtures.prepare_gguf(block_root, moe, reordered, fp32=False, native_formats=native_formats)
            run_graph(label, mixed.baseline, mixed.source)
            if moe:
                run_graph(label, mixed.baseline, mixed.source, on_demand=True)
            for fp4_format in ("MXFP4", "NVFP4"):
                fp4_root = root / (label + "-" + fp4_format.lower() + "-all-matrices")
                fp4_root.mkdir()
                # Keep decoded native matrices near the fixture activation scale and use an exact FP32 oracle.
                with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                    if family:
                        fp4_fixture = fixtures.prepare_family_gguf(
                            fp4_root, family, fp32=False, native_formats=(fp4_format,),
                            ple_format=fp4_format if family == "qwen4_exp" else None,
                            native_scale=1 / 512, fp32_native_baseline=True)
                    else:
                        fp4_fixture = fixtures.prepare_gguf(
                            fp4_root, moe, reordered, fp32=False, native_formats=(fp4_format,),
                            native_scale=1 / 512, fp32_native_baseline=True)
                run_graph(label + " all matrices " + fp4_format, fp4_fixture.baseline, fp4_fixture.source)
                if moe:
                    run_graph(label + " all matrices " + fp4_format,
                              fp4_fixture.baseline, fp4_fixture.source, on_demand=True)
            if label in ("qwen35", "qwen35moe") or family == "gpt_oss":
                gqa_root = root / (label + "-native-gqa-13-formats")
                gqa_root.mkdir()
                with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                    if family:
                        gqa_fixture = fixtures.prepare_family_gguf(
                            gqa_root, family, fp32=False, native_formats=gqa_native_formats,
                            native_scale=1 / 64, fp32_native_baseline=True)
                    else:
                        gqa_fixture = fixtures.prepare_gguf(
                            gqa_root, moe, reordered, fp32=False, native_formats=gqa_native_formats,
                            native_scale=1 / 64, fp32_native_baseline=True)
                projection_family = family or "qwen"
                projection_types = fixtures.standard_gqa_projection_types(gqa_fixture.source, projection_family)
                assert len(set(projection_types.values())) == 4, (
                    f"{label} 13-format GQA matrix did not give Q/K/V/O distinct dtypes: {projection_types}")
                run_graph(label + " standard GQA with all 13 GGML formats",
                          gqa_fixture.baseline, gqa_fixture.source)
                if moe:
                    run_graph(label + " standard GQA with all 13 GGML formats",
                              gqa_fixture.baseline, gqa_fixture.source, on_demand=True)

                mixed_root = root / (label + "-native-gqa-mixed-dtypes")
                mixed_root.mkdir()
                mixed_overrides = fixtures.standard_gqa_mixed_overrides(projection_family)
                with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                    if family:
                        mixed_fixture = fixtures.prepare_family_gguf(
                            mixed_root, family, fp32=False, native_formats=gqa_native_formats,
                            native_overrides=mixed_overrides, native_scale=1 / 64,
                            fp32_native_baseline=True)
                    else:
                        mixed_fixture = fixtures.prepare_gguf(
                            mixed_root, moe, reordered, fp32=False, native_formats=gqa_native_formats,
                            native_overrides=mixed_overrides, native_scale=1 / 64,
                            fp32_native_baseline=True)
                mixed_types = fixtures.standard_gqa_projection_types(mixed_fixture.source, projection_family)
                expected_mixed_types = {"q": "Q2_K", "k": "BF16", "v": "Q8_0", "o": "F32"}
                assert mixed_types == expected_mixed_types, (
                    f"{label} mixed GQA projection dtypes mismatch: {mixed_types}")
                run_graph(label + " mixed GQA Q2_K/BF16/Q8_0/F32",
                          mixed_fixture.baseline, mixed_fixture.source)
                if moe:
                    run_graph(label + " mixed GQA Q2_K/BF16/Q8_0/F32",
                              mixed_fixture.baseline, mixed_fixture.source, on_demand=True)
            if family == "qwen4_exp":
                # The continuous Qwen path must decode every new format
                # from PLE selected rows; all other native matrices stay
                # mixed across the full-model fixture.
                for ple_format in native_formats[1:]:
                    ple_root = root / (label + "-ple-" + ple_format.lower())
                    ple_root.mkdir()
                    with patch.object(fixtures.gguf, "GGUFWriter", writer), patch.object(fixtures, "family_tensors", family_tensors):
                        ple_fixture = fixtures.prepare_family_gguf(
                            ple_root, family, fp32=False, native_formats=native_formats,
                            ple_format=ple_format)
                    run_graph(label + " PLE " + ple_format, ple_fixture.baseline, ple_fixture.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
