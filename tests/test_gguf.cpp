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
    for (const auto& source : {root / "split", first, second})
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
    const std::array<uint32_t, 10> types = {0, 1, 10, 11, 12, 13, 14, 15, 26, 30};
    const std::array<size_t, 10> sizes = {1024, 512, 84, 110, 144, 176, 210, 292, 1024, 512};
    const std::array<DType, 10> dtypes = {DType::Float32, DType::Float16, DType::Q2K, DType::Q3K, DType::Q4K,
                                          DType::Q5K, DType::Q6K, DType::Q8K, DType::Int32, DType::BFloat16};
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
    reject(fixture_bytes({}, {{"weight", 8, {32}, std::vector<uint8_t>(34)}}), "GGUF silently accepted unsupported Q8_0");
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

static void check_qnk_defer_fallback(const std::filesystem::path& root)
{
    constexpr uint32_t rows = 8;
    constexpr uint32_t columns = 256;
    const std::string name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    const auto path = root / "qwen-qnk" / "model.gguf";
    FixtureMetadata tokens{"tokenizer.ggml.tokens", 9, {}};
    append_unsigned(tokens.payload, 8, 4);
    append_unsigned(tokens.payload, 1, 8);
    append_string(tokens.payload, "x");

    std::vector<uint8_t> qkv(static_cast<size_t>(qnk_storage_bytes(DType::Q4K, rows, columns)));
    constexpr size_t row_bytes = 144;
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
    write_bytes(path, fixture_bytes(metadata,
                                    {{"token_embd.weight", 0, {columns, 1}, std::vector<uint8_t>(columns * sizeof(float))},
                                     {"blk.0.attn_qkv.weight", 12, {columns, rows}, std::move(qkv)},
                                     {"blk.0.attn_gate.weight", 30, {columns, 4}, hidden},
                                     {"blk.0.ssm_alpha.weight", 30, {columns, 4}, hidden},
                                     {"blk.0.ssm_beta.weight", 30, {columns, 4}, hidden},
                                     {"blk.0.ssm_out.weight", 30, {4, columns}, hidden},
                                     {"blk.0.ssm_conv1d.weight", 30, {1, rows}, std::vector<uint8_t>(rows * sizeof(uint16_t))}}));

    auto archive = SafetensorsArchive::open(path);
    check(static_cast<bool>(archive), archive ? "" : archive.error().message);
    auto loaded = archive.value().load_tensor(name, SafetensorLoadDeferMxfp4Data);
    check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
    check(loaded.value().dtype == DType::Q4K && loaded.value().shape == std::vector<uint32_t>{rows, columns}
              && !loaded.value().quantized_data.empty() && !loaded.value().mxfp4_file_storage,
          "unsupported deferred Qn_K row transform did not fall back to eager loading");
    constexpr std::array<uint32_t, rows> source_rows = {0, 1, 2, 3, 4, 6, 5, 7};
    const auto values = loaded.value().qnk_values();
    for (uint32_t row = 0; row < rows; ++row)
        check(values[static_cast<size_t>(row) * row_bytes] == source_rows[row] + 1,
              "eager Qn_K fallback did not preserve transformed row order");

    auto invalid = archive.value().load_qnk_tensor(name, DType::Q4K, rows - 1, columns, SafetensorLoadDeferMxfp4Data);
    check(!invalid && invalid.error().code == ErrorCode::InvalidModel
              && invalid.error().message == "GGUF Qn_K transform shape mismatch",
          "invalid deferred Qn_K request was hidden by eager fallback");
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
              << " I32=" << types[26] << " MXFP4=" << types[39] << '\n';
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
        ncnn::moe::check_qnk_defer_fallback(root);
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
