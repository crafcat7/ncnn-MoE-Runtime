#include "models/gguf.h"
#include "models/ggufmodel.h"
#include "models/modeladapter_deepseekv4.h"
#include "models/safetensors.h"
#include "kernels/qnk.h"
#include "storage/mappedfile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void append_unsigned(std::vector<uint8_t>& bytes, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i)
        bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
}

static void append_string(std::vector<uint8_t>& bytes, const std::string& value)
{
    append_unsigned(bytes, value.size(), 8);
    bytes.insert(bytes.end(), value.begin(), value.end());
}

struct FixtureMetadata
{
    std::string key;
    uint32_t type;
    std::vector<uint8_t> payload;
};

struct FixtureTensor
{
    std::string name;
    uint32_t type;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> payload;
};

static FixtureMetadata integer_metadata(const std::string& key, uint64_t value)
{
    FixtureMetadata result{key, 10, {}};
    append_unsigned(result.payload, value, 8);
    return result;
}

static FixtureMetadata string_metadata(const std::string& key, const std::string& value)
{
    FixtureMetadata result{key, 8, {}};
    append_string(result.payload, value);
    return result;
}

static FixtureMetadata float_metadata(const std::string& key, float value)
{
    FixtureMetadata result{key, 6, {}};
    append_unsigned(result.payload, std::bit_cast<uint32_t>(value), sizeof(value));
    return result;
}

static FixtureMetadata float_array_metadata(const std::string& key, const std::vector<float>& values)
{
    FixtureMetadata result{key, 9, {}};
    append_unsigned(result.payload, 6, 4);
    append_unsigned(result.payload, values.size(), 8);
    for (float value : values)
        append_unsigned(result.payload, std::bit_cast<uint32_t>(value), sizeof(value));
    return result;
}

static FixtureMetadata single_token_metadata()
{
    FixtureMetadata tokens{"tokenizer.ggml.tokens", 9, {}};
    append_unsigned(tokens.payload, 8, 4);
    append_unsigned(tokens.payload, 1, 8);
    append_string(tokens.payload, "x");
    return tokens;
}

static std::vector<FixtureMetadata> deepseek_test_metadata()
{
    return {string_metadata("general.architecture", "deepseek4"),
            single_token_metadata(),
            integer_metadata("deepseek4.block_count", 1),
            integer_metadata("deepseek4.nextn_predict_layers", 0),
            integer_metadata("deepseek4.context_length", 128),
            integer_metadata("deepseek4.embedding_length", 32),
            integer_metadata("deepseek4.expert_count", 1),
            integer_metadata("deepseek4.expert_used_count", 1),
            integer_metadata("deepseek4.expert_shared_count", 1),
            integer_metadata("deepseek4.expert_feed_forward_length", 32),
            float_metadata("deepseek4.expert_weights_scale", 1.0f),
            integer_metadata("deepseek4.attention.head_count", 1),
            integer_metadata("deepseek4.attention.head_count_kv", 1),
            integer_metadata("deepseek4.attention.key_length", 32),
            integer_metadata("deepseek4.attention.q_lora_rank", 1),
            integer_metadata("deepseek4.attention.output_group_count", 1),
            integer_metadata("deepseek4.attention.output_lora_rank", 1),
            integer_metadata("deepseek4.attention.sliding_window", 128),
            integer_metadata("deepseek4.attention.indexer.head_count", 1),
            integer_metadata("deepseek4.attention.indexer.key_length", 32),
            integer_metadata("deepseek4.attention.indexer.top_k", 1),
            integer_metadata("deepseek4.rope.dimension_count", 32),
            float_metadata("deepseek4.rope.freq_base", 10000.0f),
            float_metadata("deepseek4.attention.compress_rope_freq_base", 10000.0f),
            float_array_metadata("deepseek4.attention.compress_ratios", {1.0f}),
            float_metadata("deepseek4.attention.layer_norm_rms_epsilon", 1.0e-5f),
            integer_metadata("deepseek4.hyper_connection.count", 1),
            integer_metadata("deepseek4.hyper_connection.sinkhorn_iterations", 1),
            float_metadata("deepseek4.hyper_connection.epsilon", 1.0e-6f),
            integer_metadata("deepseek4.hash_layer_count", 0),
            integer_metadata("deepseek4.expert_gating_func", 4),
            FixtureMetadata{"deepseek4.expert_weights_norm", 7, {1}},
            float_metadata("deepseek4.swiglu_clamp_exp", 1.0f),
            string_metadata("deepseek4.rope.scaling.type", "yarn"),
            float_metadata("deepseek4.rope.scaling.factor", 1.0f),
            integer_metadata("deepseek4.rope.scaling.original_context_length", 128),
            float_metadata("deepseek4.rope.scaling.yarn_beta_fast", 32.0f),
            float_metadata("deepseek4.rope.scaling.yarn_beta_slow", 1.0f)};
}

static void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(stream), "cannot write GGUF fixture");
}

static std::vector<uint8_t> fixture_bytes(const std::vector<FixtureMetadata>& metadata,
                                          const std::vector<FixtureTensor>& tensors)
{
    std::vector<uint8_t> bytes{'G', 'G', 'U', 'F'};
    append_unsigned(bytes, 3, 4);
    append_unsigned(bytes, tensors.size(), 8);
    append_unsigned(bytes, metadata.size(), 8);
    for (const auto& item : metadata)
    {
        append_string(bytes, item.key);
        append_unsigned(bytes, item.type, 4);
        bytes.insert(bytes.end(), item.payload.begin(), item.payload.end());
    }
    uint64_t offset = 0;
    for (const auto& tensor : tensors)
    {
        append_string(bytes, tensor.name);
        append_unsigned(bytes, tensor.shape.size(), 4);
        for (uint64_t extent : tensor.shape)
            append_unsigned(bytes, extent, 8);
        append_unsigned(bytes, tensor.type, 4);
        append_unsigned(bytes, offset, 8);
        offset += tensor.payload.size();
        offset += (32 - offset % 32) % 32;
    }
    bytes.resize(bytes.size() + (32 - bytes.size() % 32) % 32);
    for (const auto& tensor : tensors)
    {
        bytes.insert(bytes.end(), tensor.payload.begin(), tensor.payload.end());
        bytes.resize(bytes.size() + (32 - bytes.size() % 32) % 32);
    }
    return bytes;
}

static void check_single_file(const std::filesystem::path& root)
{
    FixtureMetadata tokens{"tokenizer.ggml.tokens", 9, {}};
    append_unsigned(tokens.payload, 8, 4);
    append_unsigned(tokens.payload, 2, 8);
    append_string(tokens.payload, "hello");
    append_string(tokens.payload, "world");
    const auto path = root / "single" / "model.gguf";
    write_bytes(path, fixture_bytes({string_metadata("general.architecture", "deepseek4"), tokens,
                                     integer_metadata("test.integer", UINT64_C(0xffffffffffffffff)),
                                     FixtureMetadata{"test.boolean", 7, {1}}},
                                    {{"weight", 30, {4, 2}, std::vector<uint8_t>(16, 0x3f)}}));
    auto archive = GgufArchive::open(path.parent_path());
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    const auto* info = archive.value().find("weight");
    check(info && info->dtype == DType::BFloat16 && info->shape == std::vector<uint32_t>({2, 4}) && info->size == 16,
          "GGUF tensor shape/type/range mismatch");
    const auto* vocabulary = archive.value().find_metadata("tokenizer.ggml.tokens");
    check(vocabulary && vocabulary->array() && vocabulary->array()->size() == 2
              && (*vocabulary->array())[1].string() && *(*vocabulary->array())[1].string() == "world",
          "GGUF string-array metadata mismatch");
    const auto* integer = archive.value().find_metadata("test.integer");
    check(integer && integer->unsigned_integer() == UINT64_MAX && !integer->integer() && !integer->boolean(),
          "GGUF integer metadata was narrowed or mistyped");
    check(archive.value().find_metadata("test.boolean")->boolean() == true, "GGUF boolean metadata mismatch");
    auto mapping = MappedFileRange::open(info->path, info->offset, info->size);
    check(static_cast<bool>(mapping), mapping ? "" : mapping.error().message);
    check(mapping.value()->size() == 16 && mapping.value()->data()[15] == 0x3f,
          "GGUF mapping did not retain the original payload range");
    archive.value() = {};
    check(mapping.value()->data()[0] == 0x3f, "GGUF tensor range lost its mapping owner");
}

static void check_split(const std::filesystem::path& root)
{
    const auto directory = root / "split" / "UD-Q8_K_XL";
    const auto first = directory / "model-00001-of-00002.gguf";
    const auto second = directory / "model-00002-of-00002.gguf";
    auto metadata = std::vector<FixtureMetadata>{string_metadata("general.architecture", "deepseek4"),
                                                 integer_metadata("split.count", 2), integer_metadata("split.no", 0),
                                                 integer_metadata("split.tensors.count", 1)};
    write_bytes(first, fixture_bytes(metadata, {}));
    check(!GgufArchive::open(first), "GGUF accepted a missing split shard");
    metadata[2] = integer_metadata("split.no", 1);
    write_bytes(second, fixture_bytes(metadata, {{"weight", 39, {32}, std::vector<uint8_t>(17, 0x52)}}));
    check(!has_gguf_source(root / "split") && !GgufArchive::open(root / "split"),
          "GGUF split selection searched a child directory");
    for (const auto& source : {directory, first, second})
    {
        auto archive = GgufArchive::open(source);
        check(static_cast<bool>(archive), archive ? "" : archive.error().message);
        check(archive.value().tensor_infos().size() == 1 && archive.value().find("weight")->dtype == DType::MxFp4,
              "GGUF split discovery or zero-tensor first shard failed");
    }
    metadata[2] = integer_metadata("split.no", 0);
    write_bytes(second, fixture_bytes(metadata, {{"weight", 39, {32}, std::vector<uint8_t>(17)}}));
    check(!GgufArchive::open(first), "GGUF accepted an inconsistent shard index");
    metadata[3] = integer_metadata("split.tensors.count", UINT64_C(1024) * 1024 + 1);
    write_bytes(first, fixture_bytes(metadata, {}));
    auto oversized = GgufArchive::open(first);
    check(!oversized && oversized.error().message.find("tensor count limit") != std::string::npos,
          "GGUF did not reject an oversized package catalog before expanding shards");

    metadata[2] = integer_metadata("split.no", 0);
    metadata[3] = integer_metadata("split.tensors.count", 1);
    const auto uppercase_directory = root / "split-uppercase";
    write_bytes(uppercase_directory / "model-00001-OF-00002.GGUF", fixture_bytes(metadata, {}));
    metadata[2] = integer_metadata("split.no", 1);
    write_bytes(uppercase_directory / "model-00002-OF-00002.GGUF",
                fixture_bytes(metadata, {{"weight", 39, {32}, std::vector<uint8_t>(17, 0x52)}}));
    auto uppercase = GgufArchive::open(uppercase_directory);
    check(uppercase && uppercase.value().tensor_infos().size() == 1
              && uppercase.value().find("weight") && uppercase.value().find("weight")->dtype == DType::MxFp4,
          "GGUF did not discover uppercase -OF-/.GGUF shards");

    const std::array<std::pair<const char*, uint64_t>, 6> invalid_names = {{
        {"-00001-of-00002.gguf", 0},
        {"model-0001-of-00002.gguf", 0},
        {"model-00001-of-00002-extra.gguf", 0},
        {"model-00000-of-00002.gguf", 0},
        {"model-00003-of-00002.gguf", 2},
        {"model-00001-of-00003.gguf", 0},
    }};
    for (size_t index = 0; index < invalid_names.size(); ++index)
    {
        const auto& [name, shard_number] = invalid_names[index];
        auto invalid_metadata = metadata;
        invalid_metadata[2] = integer_metadata("split.no", shard_number);
        invalid_metadata[3] = integer_metadata("split.tensors.count", 0);
        const auto invalid_path = root / "split-invalid" / name;
        write_bytes(invalid_path, fixture_bytes(invalid_metadata, {}));
        auto invalid = GgufArchive::open(invalid_path);
        const std::string expected_error = index < 3 ? "requires standard" : "disagrees";
        check(!invalid && invalid.error().code == ErrorCode::InvalidModel
                  && invalid.error().message.find(expected_error) != std::string::npos,
              "GGUF accepted malformed split filename with split.count=2: " + std::string(name));
    }
}

static void check_encodings(const std::filesystem::path& root)
{
    const std::array<uint32_t, 19> types = {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 20, 23, 26, 30, 39, 40};
    const std::array<size_t, 19> sizes = {1024, 512, 144, 160, 176, 192, 272, 84, 110, 144, 176, 210, 292, 144, 136, 1024, 512, 136, 144};
    const std::array<DType, 19> dtypes = {DType::Float32, DType::Float16, DType::Q4_0, DType::Q4_1, DType::Q5_0,
                                          DType::Q5_1, DType::Q8_0, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K,
                                          DType::Q6K, DType::Q8K, DType::IQ4NL, DType::IQ4XS, DType::Int32, DType::BFloat16, DType::MxFp4, DType::NvFp4};
    for (size_t i = 0; i < types.size(); ++i)
    {
        const auto path = root / "encodings" / (std::to_string(types[i]) + ".gguf");
        write_bytes(path, fixture_bytes({}, {{"weight", types[i], {256}, std::vector<uint8_t>(sizes[i])}}));
        auto archive = GgufArchive::open(path);
        check(static_cast<bool>(archive), archive ? "" : archive.error().message);
        check(archive.value().find("weight")->dtype == dtypes[i] && archive.value().find("weight")->size == sizes[i],
              "GGUF encoding size mismatch");
    }
    check(!GgufArchive::open(root / "encodings"), "GGUF directory discovery silently chose an ambiguous package");
}

static void check_native_mxfp4_safetensor_pair(const std::filesystem::path& root)
{
    constexpr uint32_t rows = 2;
    constexpr uint32_t columns = 64;
    std::vector<uint8_t> header;
    const std::string json = R"({"layer.weight":{"dtype":"I8","shape":[2,32],"data_offsets":[0,64]},"layer.scale":{"dtype":"F8_E8M0","shape":[2,2],"data_offsets":[64,68]},"layer.weight.__mxfp4_columns":{"dtype":"I64","shape":[64],"data_offsets":[68,580]}})";
    append_unsigned(header, json.size() + (8 - json.size() % 8) % 8, 8);
    std::string padded = json;
    padded.append((8 - padded.size() % 8) % 8, ' ');
    header.insert(header.end(), padded.begin(), padded.end());
    std::vector<uint8_t> payload(580, 0);
    for (size_t index = 0; index < 64; ++index)
        payload[index] = static_cast<uint8_t>(index * 3 + 1);
    payload[64] = 120;
    payload[65] = 121;
    payload[66] = 122;
    payload[67] = 123;
    for (uint32_t index = 0; index < columns; ++index)
    {
        const int64_t value = columns - 1 - index;
        std::memcpy(payload.data() + 68 + static_cast<size_t>(index) * sizeof(value), &value, sizeof(value));
    }
    header.insert(header.end(), payload.begin(), payload.end());
    const auto path = root / "native-mxfp4" / "model.safetensors";
    write_bytes(path, header);
    auto archive = SafetensorsArchive::open_file(path);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    auto dtype = archive.value().tensor_dtype("layer.weight");
    check(dtype && dtype.value() == DType::MxFp4, "native MXFP4 I8/scale pair was not inferred");
    auto loaded = archive.value().load_tensor("layer.weight");
    check(loaded && loaded.value().dtype == DType::MxFp4
              && loaded.value().shape == std::vector<uint32_t>{rows, columns}
              && loaded.value().mxfp4_blocks.size() == 64
              && loaded.value().mxfp4_scales.size() == 4
              && loaded.value().input_column_indices.size() == columns,
          "native MXFP4 pair did not load in its canonical compact form");
    for (size_t index = 0; index < 64; ++index)
        check(loaded.value().mxfp4_blocks[index] == static_cast<uint8_t>(index * 3 + 1),
              "native MXFP4 packed codes changed during loading");
    check(loaded.value().input_column_indices.front() == columns - 1
              && loaded.value().input_column_indices.back() == 0,
          "native MXFP4 input-column map was not preserved");
}

static void check_invalid_files(const std::filesystem::path& root)
{
    const auto path = root / "invalid" / "model.gguf";
    const auto reject = [&](const std::vector<uint8_t>& bytes, const std::string& context) {
        write_bytes(path, bytes);
        check(!GgufArchive::open(path), context);
    };
    const FixtureTensor valid{"weight", 30, {8}, std::vector<uint8_t>(16)};
    auto bytes = fixture_bytes({}, {valid});
    bytes[4] = 2;
    reject(bytes, "GGUF accepted a non-v3 header");
    bytes = fixture_bytes({}, {});
    for (size_t i = 8; i < 16; ++i)
        bytes[i] = 0xff;
    reject(bytes, "GGUF accepted an oversized tensor descriptor count");
    bytes = fixture_bytes({}, {});
    for (size_t i = 16; i < 24; ++i)
        bytes[i] = 0xff;
    reject(bytes, "GGUF accepted an oversized metadata count");
    bytes = fixture_bytes({}, {valid});
    bytes.resize(bytes.size() - 20);
    reject(bytes, "GGUF accepted a truncated tensor payload");
    reject(fixture_bytes({}, {{"weight", 9, {32}, std::vector<uint8_t>(36)}}), "GGUF accepted unsupported Q8_1");
    reject(fixture_bytes({}, {{"weight", 8, {48}, std::vector<uint8_t>(51)}}), "GGUF accepted a Q8_0 row crossing quantization blocks");
    reject(fixture_bytes({}, {{"weight", 12, {128, 2}, std::vector<uint8_t>(144)}}), "GGUF accepted a row crossing quantization blocks");
    reject(fixture_bytes({}, {{"weight", 0, {UINT64_C(1) + UINT32_MAX}, {}}}), "GGUF narrowed an oversized tensor dimension");
    reject(fixture_bytes({}, {valid, valid}), "GGUF accepted duplicate tensor names");
    reject(fixture_bytes({integer_metadata("same", 1), integer_metadata("same", 2)}, {}), "GGUF accepted duplicate metadata keys");
    reject(fixture_bytes({FixtureMetadata{"bad", 13, {}}}, {}), "GGUF accepted an unknown metadata type");
    reject(fixture_bytes({FixtureMetadata{"bad", 7, {2}}}, {}), "GGUF accepted a non-boolean metadata value");
    FixtureMetadata oversized{"bad", 8, {}};
    append_unsigned(oversized.payload, UINT64_MAX, 8);
    reject(fixture_bytes({oversized}, {}), "GGUF accepted an oversized string length");
    oversized = {"bad", 9, {}};
    append_unsigned(oversized.payload, 8, 4);
    append_unsigned(oversized.payload, UINT64_MAX, 8);
    reject(fixture_bytes({oversized}, {}), "GGUF accepted an oversized array length");
}

static std::vector<uint8_t> q8_0_oracle_payload(std::vector<float>& expected)
{
    std::vector<uint8_t> payload;
    constexpr std::array<uint16_t, 2> scales = {0x3800, 0x3400}; // 0.5 and 0.25.
    expected.clear();
    expected.reserve(64);
    for (size_t block = 0; block < scales.size(); ++block)
    {
        append_unsigned(payload, scales[block], sizeof(uint16_t));
        const float scale = block == 0 ? 0.5f : 0.25f;
        for (int index = 0; index < 32; ++index)
        {
            const int magnitude = index % 15 + 1;
            const int8_t quantized = static_cast<int8_t>((index & 1) == 0 ? magnitude : -magnitude);
            payload.push_back(static_cast<uint8_t>(quantized));
            expected.push_back(scale * quantized);
        }
    }
    return payload;
}

static void check_gguf_explicit_float32_auxiliary_loading(const std::filesystem::path& root)
{
    const auto path = root / "explicit-f32-aux" / "model.gguf";
    std::vector<float> q8_expected;
    const std::vector<uint8_t> q8_payload = q8_0_oracle_payload(q8_expected);
    std::vector<uint8_t> f16_payload;
    for (uint16_t value : {uint16_t{0x3c01}, uint16_t{0xbe01}, uint16_t{0x3555}})
        append_unsigned(f16_payload, value, sizeof(value));
    std::vector<uint8_t> bf16_payload;
    for (uint16_t value : {uint16_t{0x3f81}, uint16_t{0xc020}})
        append_unsigned(bf16_payload, value, sizeof(value));
    std::vector<uint8_t> int32_payload;
    append_unsigned(int32_payload, 1, sizeof(int32_t));
    append_unsigned(int32_payload, 2, sizeof(int32_t));
    const std::vector<FixtureTensor> tensors = {
        {"blk.0.attn_compressor_ape.weight", 8, {32, 2}, q8_payload},
        {"blk.0.attn_compressor_kv.weight", 1, {3}, f16_payload},
        {"blk.0.attn_compressor_gate.weight", 30, {2}, bf16_payload},
        {"blk.0.attn_norm.weight", 26, {2}, int32_payload},
        {"blk.0.attn_output_a.weight", 8, {32, 2}, q8_payload}};
    write_bytes(path, fixture_bytes(deepseek_test_metadata(), tensors));

    const std::array<std::string, 3> malformed_layer_names = {
        "blk.x.attn_norm.weight", "blk.0x.attn_norm.weight", "blk.18446744073709551616.attn_norm.weight"};
    for (size_t index = 0; index < malformed_layer_names.size(); ++index)
    {
        auto malformed_tensors = tensors;
        malformed_tensors[3].name = malformed_layer_names[index];
        const auto malformed_path = root / ("deepseek-invalid-layer-" + std::to_string(index)) / "model.gguf";
        write_bytes(malformed_path, fixture_bytes(deepseek_test_metadata(), malformed_tensors));
        auto malformed_archive = GgufArchive::open(malformed_path);
        check(static_cast<bool>(malformed_archive), malformed_archive ? "" : malformed_archive.error().message);
        auto malformed_mapping = gguf_model_tensors(malformed_archive.value());
        check(!malformed_mapping,
              "DeepSeek GGUF model mapping accepted malformed layer id: " + malformed_layer_names[index]);
    }

    auto archive = SafetensorsArchive::open(path);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    const std::string ape_name = "layers.0.attn.compressor.ape";
    const std::string main_weight_name = "layers.0.attn.wo_a.weight";
    const SafetensorInfo* ape_info = archive.value().find(ape_name);
    const SafetensorInfo* main_weight_info = archive.value().find(main_weight_name);
    check(ape_info && ape_info->dtype == "U8" && ape_info->ggml_type == 8 && !ape_info->transform
              && ape_info->shape == std::vector<uint32_t>{2, 32},
          "DeepSeek GGUF APE alias changed its native Q8_0 range or shape");
    check(main_weight_info && main_weight_info->dtype == "U8" && main_weight_info->ggml_type == 8
              && !main_weight_info->transform,
          "DeepSeek GGUF matmul alias changed its native Q8_0 encoding");
    {
        auto native_ape = archive.value().load_tensor(ape_name);
        check(native_ape && native_ape.value().dtype == DType::Q8_0
                  && native_ape.value().shape == std::vector<uint32_t>{2, 32}
                  && native_ape.value().float32_data.empty()
                  && native_ape.value().qnk_values().size() == q8_payload.size()
                  && std::equal(q8_payload.begin(), q8_payload.end(), native_ape.value().qnk_values().begin()),
              "default APE loading dequantized or changed native GGUF Q8_0 blocks");
        auto explicit_ape = archive.value().load_float32_tensor(ape_name);
        const auto ape_values = explicit_ape ? explicit_ape.value().float32_values() : std::span<const float>{};
        check(explicit_ape && explicit_ape.value().dtype == DType::Float32
                  && explicit_ape.value().shape == std::vector<uint32_t>{2, 32}
                  && ape_values.size() == q8_expected.size()
                  && std::equal(ape_values.begin(), ape_values.end(), q8_expected.begin()),
              "explicit Q8_0 APE loading disagreed with the independent block oracle");
        auto native_main = archive.value().load_tensor(main_weight_name);
        check(native_main && native_main.value().dtype == DType::Q8_0
                  && native_main.value().float32_data.empty()
                  && native_main.value().qnk_values().size() == q8_payload.size()
                  && std::equal(q8_payload.begin(), q8_payload.end(), native_main.value().qnk_values().begin()),
              "ordinary DeepSeek matmul loading expanded its native Q8_0 weights");
    }

    auto explicit_f16 = archive.value().load_float32_tensor("layers.0.attn.compressor.wkv.weight");
    constexpr std::array<float, 3> f16_expected = {1.0009765625f, -1.5009765625f, 0.333251953125f};
    const auto f16_values = explicit_f16 ? explicit_f16.value().float32_values() : std::span<const float>{};
    check(explicit_f16 && explicit_f16.value().dtype == DType::Float32
              && explicit_f16.value().shape == std::vector<uint32_t>{3}
              && f16_values.size() == f16_expected.size()
              && std::equal(f16_values.begin(), f16_values.end(), f16_expected.begin()),
          "explicit F16 auxiliary loading did not widen its stored values directly");
    auto explicit_bf16 = archive.value().load_float32_tensor("layers.0.attn.compressor.wgate.weight");
    constexpr std::array<float, 2> bf16_expected = {1.0078125f, -2.5f};
    const auto bf16_values = explicit_bf16 ? explicit_bf16.value().float32_values() : std::span<const float>{};
    check(explicit_bf16 && explicit_bf16.value().dtype == DType::Float32
              && explicit_bf16.value().shape == std::vector<uint32_t>{2}
              && bf16_values.size() == bf16_expected.size()
              && std::equal(bf16_values.begin(), bf16_values.end(), bf16_expected.begin()),
          "explicit BF16 auxiliary loading did not widen its stored values");
    auto integer = archive.value().load_float32_tensor("layers.0.attn_norm.weight");
    check(!integer && integer.error().code == ErrorCode::InvalidModel,
          "explicit F32 loading accepted an integer GGUF tensor");

    const auto truncated_path = root / "explicit-f32-aux-truncated" / "model.gguf";
    write_bytes(truncated_path, fixture_bytes(deepseek_test_metadata(), tensors));
    auto truncated_archive = SafetensorsArchive::open(truncated_path);
    check(static_cast<bool>(truncated_archive), truncated_archive ? "" : truncated_archive.error().message);
    const SafetensorInfo* truncated_info = truncated_archive.value().find(ape_name);
    check(truncated_info && truncated_info->size > 0, "truncation fixture lost its APE range");
    std::filesystem::resize_file(truncated_path, truncated_info->offset + truncated_info->size - 1);
    check(!truncated_archive.value().load_float32_tensor(ape_name),
          "explicit F32 loading accepted a truncated GGUF Q8_0 payload");
}

static void check_raw_qnk_dtype_ambiguity(const std::filesystem::path& root)
{
    std::string header = R"({"q4_ambiguous":{"dtype":"U8","shape":[1,18],"data_offsets":[0,18]},"legacy_q4k":{"dtype":"U8","shape":[2,1,144],"data_offsets":[18,306]}})";
    while (header.size() % 8 != 0)
        header.push_back(' ');
    std::vector<uint8_t> bytes;
    append_unsigned(bytes, header.size(), 8);
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.resize(bytes.size() + 306, 0);
    const auto path = root / "raw-qnk" / "weights.safetensors";
    write_bytes(path, bytes);
    auto archive = SafetensorsArchive::open_file(path);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    check(!archive.value().find_qnk_expert_dtype("q4_ambiguous", 1, 1, 32),
          "untagged 18-byte U8 data was guessed to be Q4_0 or IQ4_NL");
    check(archive.value().find_qnk_expert_dtype("legacy_q4k", 2, 1, 256) == DType::Q4K,
          "legacy untagged Q4_K expert bank inference changed");
}

static void check_qnk_defer_fallback(const std::filesystem::path& root, DType dtype, uint32_t ggml_type, uint32_t columns = 0,
                                     bool vector_shared_gate = false, bool verify_float32_transform = false)
{
    constexpr uint32_t rows = 8;
    if (columns == 0)
        columns = qnk_block_size(dtype);
    const std::string name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    const auto path = root / ("qwen-qnk-" + std::to_string(ggml_type) + "-" + std::to_string(columns) + (vector_shared_gate ? "-vector-gate" : "-matrix-gate")) / "model.gguf";
    FixtureMetadata tokens{"tokenizer.ggml.tokens", 9, {}};
    append_unsigned(tokens.payload, 8, 4);
    append_unsigned(tokens.payload, 1, 8);
    append_string(tokens.payload, "x");

    std::vector<uint8_t> qkv(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)));
    const size_t row_bytes = static_cast<size_t>(qnk_storage_bytes(dtype, 1, columns));
    for (uint32_t row = 0; row < rows; ++row)
        std::fill_n(qkv.begin() + static_cast<size_t>(row) * row_bytes, row_bytes, static_cast<uint8_t>(row + 1));

    std::vector<FixtureMetadata> metadata = {
        string_metadata("general.architecture", "qwen35"),
        integer_metadata("qwen35.block_count", 1),
        integer_metadata("qwen35.context_length", 128),
        integer_metadata("qwen35.embedding_length", columns),
        integer_metadata("qwen35.attention.head_count", 1),
        integer_metadata("qwen35.attention.head_count_kv", 1),
        integer_metadata("qwen35.rope.dimension_count", 1),
        integer_metadata("qwen35.ssm.group_count", 2),
        integer_metadata("qwen35.ssm.state_size", 1),
        integer_metadata("qwen35.ssm.time_step_rank", 4),
        integer_metadata("qwen35.ssm.inner_size", 4),
        integer_metadata("qwen35.ssm.conv_kernel", 1),
        integer_metadata("qwen35.attention.key_length", 1),
        integer_metadata("qwen35.feed_forward_length", 4),
        std::move(tokens)};
    const std::vector<uint8_t> hidden(4 * columns * sizeof(uint16_t));
    const bool native_qnk_block = is_qnk_dtype(dtype);
    const std::vector<uint8_t> embedding(native_qnk_block ? row_bytes : columns * sizeof(float), 0x35);
    std::vector<uint8_t> shared_gate;
    for (uint32_t column = 0; column < columns; ++column)
        append_unsigned(shared_gate, 0x3f000000u, sizeof(float)); // Nonzero 0.5f.
    std::vector<FixtureTensor> tensors = {{"token_embd.weight", native_qnk_block ? ggml_type : 0u, {columns, 1}, embedding},
                                          {"blk.0.attn_qkv.weight", ggml_type, {columns, rows}, std::move(qkv)},
                                          {"blk.0.attn_gate.weight", 30, {columns, 4}, hidden},
                                          {"blk.0.ssm_alpha.weight", 30, {columns, 4}, hidden},
                                          {"blk.0.ssm_beta.weight", 30, {columns, 4}, hidden},
                                          {"blk.0.ssm_out.weight", 30, {4, columns}, hidden},
                                          {"blk.0.ssm_conv1d.weight", 30, {1, rows}, std::vector<uint8_t>(rows * sizeof(uint16_t))},
                                          {"blk.0.ffn_gate_inp_shexp.weight", 0,
                                           vector_shared_gate ? std::vector<uint64_t>{columns} : std::vector<uint64_t>{columns, 1}, shared_gate}};
    if (verify_float32_transform)
    {
        std::vector<uint8_t> norm;
        norm.reserve(static_cast<size_t>(columns) * sizeof(uint16_t));
        for (uint32_t index = 0; index < columns; ++index)
        {
            const uint16_t value = index == 0 ? 0x3c01 : index == 1 ? 0x3e01
                                                                    : 0x3c00;
            append_unsigned(norm, value, sizeof(value));
        }
        tensors.push_back({"blk.0.attn_norm.weight", 1, {columns}, std::move(norm)});
    }
    write_bytes(path, fixture_bytes(metadata, tensors));

    auto archive = SafetensorsArchive::open(path);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    if (verify_float32_transform)
    {
        const std::string norm_name = "model.language_model.layers.0.input_layernorm.weight";
        const SafetensorInfo* norm_info = archive.value().find(norm_name);
        auto norm = archive.value().load_float32_tensor(norm_name);
        std::vector<float> expected(columns, 0.0f);
        expected[0] = 0.0009765625f;
        expected[1] = 0.5009765625f;
        const auto values = norm ? norm.value().float32_values() : std::span<const float>{};
        check(norm_info && norm_info->transform && norm_info->ggml_type == 1
                  && norm && norm.value().dtype == DType::Float32 && norm.value().shape == std::vector<uint32_t>{columns}
                  && values.size() == expected.size() && std::equal(values.begin(), values.end(), expected.begin()),
              "explicit F32 transformed Qwen GGUF F16 tensor passed through BF16 precision");
    }
    const std::string shared_name = "model.language_model.layers.0.mlp.shared_expert_gate.weight";
    auto original = GgufArchive::open(path);
    check(static_cast<bool>(original), "cannot reopen shared gate fixture metadata");
    const GgufTensorInfo* source_gate = original.value().find("blk.0.ffn_gate_inp_shexp.weight");
    const SafetensorInfo* mapped_gate = archive.value().find(shared_name);
    check(source_gate && mapped_gate && mapped_gate->shape == std::vector<uint32_t>{1, columns}
              && !mapped_gate->transform && mapped_gate->path == source_gate->path
              && mapped_gate->offset == source_gate->offset && mapped_gate->size == source_gate->size
              && mapped_gate->ggml_type == source_gate->ggml_type,
          "Qwen singleton shared gate did not preserve its file range and logical Linear shape");
    auto gate = archive.value().load_tensor(shared_name);
    check(gate && gate.value().shape == std::vector<uint32_t>{1, columns}
              && gate.value().bfloat16_values().size() == columns
              && std::all_of(gate.value().bfloat16_values().begin(), gate.value().bfloat16_values().end(),
                             [](uint16_t value) { return value == 0x3f00; }),
          "Qwen vector shared gate lost its nonzero payload");
    auto loaded = archive.value().load_tensor(name, SafetensorLoadDeferMxfp4Data);
    check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
    check(loaded.value().dtype == dtype && loaded.value().shape == std::vector<uint32_t>{rows, columns}
              && !loaded.value().quantized_data.empty() && !loaded.value().mxfp4_file_storage,
          "unsupported deferred Qn_K row transform did not fall back to eager loading");
    constexpr std::array<uint32_t, rows> source_rows = {0, 1, 2, 3, 4, 6, 5, 7};
    const auto values = loaded.value().qnk_values();
    for (uint32_t row = 0; row < rows; ++row)
        check(values[static_cast<size_t>(row) * row_bytes] == source_rows[row] + 1,
              "eager Qn_K fallback did not preserve transformed row order");

    if (native_qnk_block)
    {
        auto raw = archive.value().load_tensor("model.language_model.embed_tokens.weight");
        check(raw && raw.value().dtype == dtype
                  && raw.value().shape == std::vector<uint32_t>{1, columns}
                  && raw.value().mapped_data && raw.value().quantized_data.empty()
                  && raw.value().float32_data.empty() && raw.value().bfloat16_data.empty()
                  && raw.value().qnk_values().size() == embedding.size()
                  && std::equal(embedding.begin(), embedding.end(), raw.value().qnk_values().begin()),
              "native GGML loading expanded or changed compact mapped storage");
        check(archive.value().find_qnk_expert_dtype("model.language_model.embed_tokens.weight", 1, 1, columns) == dtype,
              "native Expert bank detection confused equal byte sizes or required K blocks");
    }

    auto invalid = archive.value().load_qnk_tensor(name, dtype, rows - 1, columns, SafetensorLoadDeferMxfp4Data);
    check(!invalid && invalid.error().code == ErrorCode::InvalidModel
              && invalid.error().message == "GGUF Qn_K transform shape mismatch",
          "invalid deferred Qn_K request was hidden by eager fallback");
    if (vector_shared_gate)
    {
        tensors.back().shape = {static_cast<uint64_t>(columns) + 1};
        append_unsigned(tensors.back().payload, 0x3f000000u, sizeof(float));
        const auto invalid_path = path.parent_path() / "wrong-gate-width.gguf";
        write_bytes(invalid_path, fixture_bytes(metadata, tensors));
        auto bad_gate = SafetensorsArchive::open(invalid_path);
        check(!bad_gate && bad_gate.error().code == ErrorCode::InvalidModel
                  && bad_gate.error().message.find("shared Expert gate shape") != std::string::npos,
              "Qwen GGUF accepted a singleton shared gate with the wrong hidden width");
    }
}

static void check_directory_selection(const std::filesystem::path& root)
{
    const auto directory = root / "discovery";
    const auto nested = directory / "package" / "nested" / "model.gguf";
    const auto bytes = fixture_bytes({}, {{"weight", 8, {32}, std::vector<uint8_t>(34)}});
    write_bytes(nested, bytes);
    check(!has_gguf_source(directory) && !GgufArchive::open(directory),
          "GGUF selection searched a nested directory");
    auto direct = GgufArchive::open(nested);
    check(static_cast<bool>(direct), "GGUF direct nested file could not be selected");
    const auto child = directory / "package" / "model.gguf";
    write_bytes(child, bytes);
    auto missing = GgufArchive::open(directory);
    check(!has_gguf_source(directory) && !missing
              && missing.error().message.find("model directory contains no GGUF files") != std::string::npos,
          "GGUF selection searched its only child package");
    check(!SafetensorsArchive::open(directory), "checkpoint selection searched a child GGUF package");
    auto explicit_child = GgufArchive::open(child.parent_path());
    check(explicit_child && has_gguf_source(child.parent_path())
              && explicit_child.value().tensor_infos().at("weight").path == std::filesystem::absolute(child),
          "GGUF could not open an explicitly selected package directory");
    write_bytes(directory / "other" / "model.gguf", bytes);
    check(!has_gguf_source(directory) && !GgufArchive::open(directory),
          "GGUF selection searched multiple child packages");
    const auto top = directory / "model.gguf";
    write_bytes(top, bytes);
    auto selected = GgufArchive::open(directory);
    check(selected && has_gguf_source(directory)
              && selected.value().tensor_infos().at("weight").path == std::filesystem::absolute(top),
          "GGUF did not match files in the selected directory");
    write_bytes(directory / "other.gguf", bytes);
    auto ambiguous = GgufArchive::open(directory);
    check(!ambiguous && ambiguous.error().message.find("select a .gguf file or package directory explicitly") != std::string::npos,
          "GGUF silently chose one of multiple packages in the selected directory");
}

static void check_real_model(const std::filesystem::path& source)
{
    auto archive = GgufArchive::open(source);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    auto manifest = gguf_model_manifest(archive.value());
    check(static_cast<bool>(manifest), manifest ? "" : manifest.error().message);
    auto aliases = gguf_model_tensors(archive.value());
    check(static_cast<bool>(aliases), aliases ? "" : aliases.error().message);
    const auto absolute_source = std::filesystem::absolute(source);
    const auto package_root = std::filesystem::is_directory(absolute_source) ? absolute_source : absolute_source.parent_path();
    ModelPackage package{package_root,
                         ModelManifest{"deepseek_v4", manifest.value()},
                         0,
                         absolute_source};
    auto descriptor = DeepSeekV4ModelAdapter().parse_model(package);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    std::array<size_t, 41> types{};
    for (const auto& [name, tensor] : archive.value().tensor_infos())
    {
        (void)name;
        if (tensor.ggml_type < types.size())
            ++types[tensor.ggml_type];
    }
    std::cout << "GGUF source=" << source.string()
              << " tensors=" << archive.value().tensor_infos().size()
              << " aliases=" << aliases.value().size()
              << " layers=" << descriptor.value().layers.size()
              << " F32=" << types[0] << " BF16=" << types[30]
              << " Q8_0=" << types[8] << " I32=" << types[26] << " MXFP4=" << types[39]
              << " NVFP4=" << types[40] << '\n';
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string(argv[1]) == "--model")
        {
            ncnn::moe::check_real_model(argv[2]);
            return 0;
        }
        if (argc != 1)
            throw std::runtime_error("usage: test_gguf [--model PATH]");
        const auto root = std::filesystem::temp_directory_path()
                          / ("ncnn_moe_gguf_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ncnn::moe::check_single_file(root);
        ncnn::moe::check_split(root);
        ncnn::moe::check_encodings(root);
        ncnn::moe::check_native_mxfp4_safetensor_pair(root);
        ncnn::moe::check_invalid_files(root);
        ncnn::moe::check_gguf_explicit_float32_auxiliary_loading(root);
        ncnn::moe::check_raw_qnk_dtype_ambiguity(root);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q4K, 12);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q4_0, 2);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q4_1, 3);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q5_0, 6);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4XS, 23);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q8_0, 8, 0, false, true);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q5_1, 7);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4NL, 20);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4NL, 20, 160);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4NL, 20, 256);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::NvFp4, 40, 64);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q8_0, 8, 2560, true);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q8_0, 8, 2560, false);
        ncnn::moe::check_directory_selection(root);
        std::filesystem::remove_all(root);
        std::cout << "GGUF parser tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
