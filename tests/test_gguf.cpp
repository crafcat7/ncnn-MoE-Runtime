#include "models/gguf.h"
#include "models/ggufmodel.h"
#include "models/modeladapter_deepseekv4.h"
#include "models/safetensors.h"
#include "kernels/qnk.h"
#include "storage/mappedfile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
}

static void check_encodings(const std::filesystem::path& root)
{
    const std::array<uint32_t, 17> types = {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 20, 23, 26, 30};
    const std::array<size_t, 17> sizes = {1024, 512, 144, 160, 176, 192, 272, 84, 110, 144, 176, 210, 292, 144, 136, 1024, 512};
    const std::array<DType, 17> dtypes = {DType::Float32, DType::Float16, DType::Q4_0, DType::Q4_1, DType::Q5_0,
                                          DType::Q5_1, DType::Q8_0, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K,
                                          DType::Q6K, DType::Q8K, DType::IQ4NL, DType::IQ4XS, DType::Int32, DType::BFloat16};
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
                                     bool vector_shared_gate = false)
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
    write_bytes(path, fixture_bytes(metadata, tensors));

    auto archive = SafetensorsArchive::open(path);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
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
              && explicit_child.value().source_path() == std::filesystem::absolute(child),
          "GGUF could not open an explicitly selected package directory");
    write_bytes(directory / "other" / "model.gguf", bytes);
    check(!has_gguf_source(directory) && !GgufArchive::open(directory),
          "GGUF selection searched multiple child packages");
    const auto top = directory / "model.gguf";
    write_bytes(top, bytes);
    auto selected = GgufArchive::open(directory);
    check(selected && has_gguf_source(directory)
              && selected.value().source_path() == std::filesystem::absolute(top),
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
    ModelPackage package{archive.value().root_path(), ModelManifest{"deepseek_v4", manifest.value()}, 0, archive.value().source_path()};
    auto descriptor = DeepSeekV4ModelAdapter().parse_model(package);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    std::array<size_t, 40> types{};
    for (const auto& [name, tensor] : archive.value().tensor_infos())
    {
        (void)name;
        if (tensor.ggml_type < types.size())
            ++types[tensor.ggml_type];
    }
    std::cout << "GGUF source=" << archive.value().source_path().string()
              << " tensors=" << archive.value().tensor_infos().size()
              << " aliases=" << aliases.value().size()
              << " layers=" << descriptor.value().layers.size()
              << " F32=" << types[0] << " BF16=" << types[30]
              << " Q8_0=" << types[8] << " I32=" << types[26] << " MXFP4=" << types[39] << '\n';
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
        ncnn::moe::check_invalid_files(root);
        ncnn::moe::check_raw_qnk_dtype_ambiguity(root);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q4K, 12);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q4_0, 2);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q4_1, 3);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q5_0, 6);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4XS, 23);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q8_0, 8);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::Q5_1, 7);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4NL, 20);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4NL, 20, 160);
        ncnn::moe::check_qnk_defer_fallback(root, ncnn::moe::DType::IQ4NL, 20, 256);
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
