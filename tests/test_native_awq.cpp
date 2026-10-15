#include "models/safetensors.h"
#include "kernels/int4.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void native_awq_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("native AWQ/GPTQ test failed: ") + message);
}

class NativeAwqDirectory
{
public:
    NativeAwqDirectory()
    {
        static uint32_t sequence = 0;
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("ncnn_moe_native_awq_" + std::to_string(timestamp) + "_" + std::to_string(sequence++));
        native_awq_check(std::filesystem::create_directory(path), "create fixture directory");
    }

    ~NativeAwqDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    void config(const std::string& quant, bool nested = false) const
    {
        std::ofstream stream(path / "config.json");
        stream << (nested ? "{\"text_config\":{\"quantization_config\":" : "{\"quantization_config\":") << quant << (nested ? "}}" : "}");
        native_awq_check(static_cast<bool>(stream), "write quantization config");
    }

    std::filesystem::path path;
};

static uint8_t native_awq_code(uint32_t row, uint32_t column)
{
    return static_cast<uint8_t>((row * 11 + column * 7 + 3) % 16);
}

static uint8_t native_awq_zero(uint32_t row, uint32_t group, bool awq)
{
    return static_cast<uint8_t>((row * 7 + group * 11 + 15) % 16 + (awq ? 0 : 1));
}

static float native_awq_scale(uint32_t index, const std::string& dtype)
{
    if (dtype == "BF16")
        return 0.125f + static_cast<float>(index % 32 + 1) / 1024.0f;
    return 0.125f + static_cast<float>(index % 32 + 1) / 8192.0f + (dtype == "F32" ? 1.0f / 65536.0f : 0.0f);
}

class NativeAwqArchive
{
public:
    void add(const std::string& name, const std::string& dtype, const std::vector<uint32_t>& shape, const void* values, size_t size)
    {
        if (count++ != 0)
            header << ',';
        header << '"' << name << "\":{\"dtype\":\"" << dtype << "\",\"shape\":[";
        for (size_t index = 0; index < shape.size(); ++index)
        {
            if (index != 0)
                header << ',';
            header << shape[index];
        }
        header << "],\"data_offsets\":[" << data.size() << ',' << data.size() + size << "]}";
        const size_t offset = data.size();
        data.resize(offset + size);
        if (size != 0)
            std::memcpy(data.data() + offset, values, size);
    }

    void projection(const std::string& prefix, bool awq, uint32_t columns, int64_t group_size,
                    const std::string& scale_dtype, bool indices, bool permuted, int malformed = -1)
    {
        constexpr uint32_t rows = 16;
        constexpr std::array<uint32_t, 8> order = {0, 4, 1, 5, 2, 6, 3, 7};
        const uint32_t effective_group = group_size == -1 ? columns : static_cast<uint32_t>(group_size);
        const uint32_t groups = columns / effective_group + (columns % effective_group != 0);
        std::vector<uint32_t> weights(static_cast<size_t>(rows) * columns / 8);
        std::vector<uint32_t> zeros(static_cast<size_t>(groups) * rows / 8);
        std::vector<uint32_t> g_idx(columns);
        std::vector<uint16_t> half_scales(static_cast<size_t>(groups) * rows);
        std::vector<float> scales(half_scales.size());
        for (uint32_t row = 0; row < rows; ++row)
        {
            for (uint32_t column = 0; column < columns; ++column)
            {
                const size_t index = awq ? static_cast<size_t>(column) * (rows / 8) + row / 8 : static_cast<size_t>(column / 8) * rows + row;
                const uint32_t shift = (awq ? order[row % 8] : column % 8) * 4;
                weights[index] |= static_cast<uint32_t>(native_awq_code(row, column)) << shift;
            }
            for (uint32_t group = 0; group < groups; ++group)
            {
                const size_t index = static_cast<size_t>(group) * rows + row;
                const uint8_t code = static_cast<uint8_t>(native_awq_zero(row, group, awq) - (awq ? 0 : 1));
                zeros[index / 8] |= static_cast<uint32_t>(code) << ((awq ? order[row % 8] : row % 8) * 4);
                half_scales[index] = static_cast<uint16_t>((scale_dtype == "BF16" ? 0x3e01 : 0x3001) + index % 32);
                scales[index] = native_awq_scale(static_cast<uint32_t>(index), scale_dtype);
            }
        }
        for (uint32_t column = 0; column < columns; ++column)
            g_idx[column] = (permuted ? (column + 1) % columns : column) / effective_group;
        if (malformed == 9)
        {
            half_scales[0] = 0;
            scales[0] = 0.0f;
        }
        if (malformed == 10 || malformed == 11 || malformed == 12)
        {
            half_scales[0] = malformed == 10 ? 0x7e00 : malformed == 11 ? 0x7c00
                                                                        : 0xb000;
            scales[0] = malformed == 10 ? std::bit_cast<float>(UINT32_C(0x7fc00000)) : malformed == 11 ? std::bit_cast<float>(UINT32_C(0x7f800000))
                                                                                                       : -1.0f;
        }
        if (malformed == 19)
            g_idx[0] = UINT32_MAX;
        if (malformed == 20)
            g_idx[0] = groups;
        if (malformed == 21)
            std::fill(g_idx.begin(), g_idx.end(), 0);
        std::vector<uint32_t> weight_shape = awq ? std::vector<uint32_t>{columns, rows / 8} : std::vector<uint32_t>{columns / 8, rows};
        if (malformed == 1)
            ++weight_shape[0];
        add(prefix + "qweight", malformed == 0 ? "U8" : "I32", weight_shape, weights.data(), weights.size() * sizeof(uint32_t) - (malformed == 2 ? sizeof(uint32_t) : 0));
        if (malformed != 13)
            add(prefix + "qzeros", malformed == 3 ? "U8" : "I32", {groups - (malformed == 4 ? 1u : 0u), rows / 8}, zeros.data(), zeros.size() * sizeof(uint32_t) - (malformed == 5 ? sizeof(uint32_t) : 0));
        if (malformed != 14)
        {
            const size_t bytes = half_scales.size() * (scale_dtype == "F32" ? sizeof(float) : sizeof(uint16_t));
            const void* values = scale_dtype == "F32" ? static_cast<const void*>(scales.data()) : static_cast<const void*>(half_scales.data());
            add(prefix + "scales", malformed == 6 ? "I32" : scale_dtype, {groups, rows - (malformed == 7 ? 1u : 0u)}, values, bytes - (malformed == 8 ? 2 : 0));
        }
        if (indices)
            add(prefix + "g_idx", malformed == 16 ? "I64" : "I32", {columns - (malformed == 17 ? 1u : 0u)}, g_idx.data(), g_idx.size() * sizeof(uint32_t) - (malformed == 18 ? sizeof(uint32_t) : 0));
        if (malformed == 15)
        {
            const float value = 1.0f;
            add(prefix + "weight", "F32", {1}, &value, sizeof(value));
        }
    }

    void write(const std::filesystem::path& root) const
    {
        std::string encoded = '{' + header.str() + '}';
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        const uint64_t length = encoded.size();
        std::ofstream stream(root / "model.safetensors", std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        stream.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        native_awq_check(static_cast<bool>(stream), "write source archive");
    }

private:
    size_t count = 0;
    std::ostringstream header;
    std::vector<uint8_t> data;
};

static std::string native_awq_config(bool awq, int64_t group_size, bool desc_act = false)
{
    return std::string("{\"quant_method\":\"") + (awq ? "awq" : "gptq") + "\",\"bits\":4,\"group_size\":" + std::to_string(group_size)
           + (awq ? ",\"version\":\"gemm\",\"zero_point\":true}" : desc_act ? ",\"desc_act\":true}"
                                                                            : ",\"desc_act\":false}");
}

static void check_native_awq_projection(bool awq, const std::string& dtype, int64_t group_size,
                                        bool indices, bool permuted, bool alias, bool desc_act = false)
{
    NativeAwqDirectory directory;
    directory.config(native_awq_config(awq, group_size, desc_act), alias);
    NativeAwqArchive fixture;
    const std::string source = alias ? "model.layers.0.self_attn.q_proj." : "model.language_model.layers.0.self_attn.q_proj.";
    const std::string logical = "model.language_model.layers.0.self_attn.q_proj.weight";
    const uint32_t columns = awq ? 19 : 24;
    fixture.projection(source, awq, columns, group_size, dtype, indices, permuted);
    fixture.write(directory.path);
    auto opened = SafetensorsArchive::open(directory.path);
    native_awq_check(static_cast<bool>(opened), "valid source archive rejected");
    const SafetensorsArchive& archive = opened.value();
    native_awq_check(archive.has_native_quantized_tensors() && archive.contains_tensor(logical), "logical quantized tensor was not detected");
    auto inferred = archive.tensor_dtype(logical);
    native_awq_check(inferred && inferred.value() == DType::Int4, "source dtype was not INT4");
    auto loaded = archive.load_tensor(logical);
    native_awq_check(static_cast<bool>(loaded), "valid source INT4 loading failed");
    const TensorData& matrix = loaded.value();
    const uint32_t group = group_size == -1 ? columns : static_cast<uint32_t>(group_size);
    const uint32_t groups = columns / group + (columns % group != 0);
    native_awq_check(matrix.dtype == DType::Int4 && matrix.shape == std::vector<uint32_t>{16, columns}
                         && matrix.quantization_group_size == group && matrix.float32_data.empty() && matrix.bfloat16_data.empty(),
                     "source did not retain native INT4 storage");
    native_awq_check(matrix.int4_values().size() == static_cast<size_t>(16) * ((columns + 1) / 2),
                     "source packed byte count changed");
    for (uint32_t row = 0; row < 16; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint8_t byte = matrix.int4_values()[row * ((columns + 1) / 2) + column / 2];
            native_awq_check(((byte >> ((column & 1u) * 4)) & 15u) == native_awq_code(row, column), "packed integer code changed");
        }
        for (uint32_t index = 0; index < groups; ++index)
        {
            const size_t metadata = static_cast<size_t>(row) * groups + index;
            native_awq_check(matrix.quantization_scales[metadata] == native_awq_scale(index * 16 + row, dtype), "source scale precision changed");
            native_awq_check(matrix.quantization_zero_points[metadata] == native_awq_zero(row, index, awq), "source zero point changed");
        }
    }
    native_awq_check(matrix.quantization_group_indices.size() == (indices && permuted ? columns : 0),
                     "canonical g_idx was not released or noncanonical g_idx was lost");
    ActivationBuffer input(5, columns);
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t column = 0; column < columns; ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((token * 3 + column * 5) % 13) - 6) / 8.0f;
    ActivationBuffer output;
    native_awq_check(forward_linear_int4(matrix, input, output), "source native execution failed");
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t row = 0; row < 16; ++row)
        {
            float expected = 0.0f;
            for (uint32_t column = 0; column < columns; ++column)
            {
                const uint32_t selected = (permuted ? (column + 1) % columns : column) / group;
                const float weight = (static_cast<int>(native_awq_code(row, column)) - native_awq_zero(row, selected, awq)) * native_awq_scale(selected * 16 + row, dtype);
                expected += input.row(token)[column] * weight;
                if (indices && permuted)
                    native_awq_check(matrix.quantization_group_indices[column] == selected, "GPTQ g_idx changed");
            }
            native_awq_check(std::abs(output.row(token)[row] - expected) < 1e-5f, "direct source scalar reference mismatch");
        }
    }
}

static void test_native_awq_invalid_inputs()
{
    for (int malformed = 0; malformed <= 21; ++malformed)
    {
        NativeAwqDirectory directory;
        directory.config(native_awq_config(false, 5, true));
        NativeAwqArchive fixture;
        fixture.projection("linear.", false, 24, 5, "F16", true, true, malformed);
        fixture.write(directory.path);
        auto opened = SafetensorsArchive::open(directory.path);
        if (opened)
            native_awq_check(!opened.value().load_tensor("linear.weight"), "malformed source quantization was accepted");
    }
    const std::vector<std::string> configs = {
        R"({"quant_method":"awq","bits":8,"group_size":7})",
        R"({"quant_method":"awq","bits":4,"w_bit":8,"group_size":7})",
        R"({"quant_method":"awq","bits":4,"group_size":7,"q_group_size":8})",
        R"({"quant_method":"awq","bits":4,"group_size":0})",
        R"({"quant_method":"awq","bits":4,"group_size":-2})",
        R"({"quant_method":"awq","bits":true,"group_size":7})",
        R"({"quant_method":"awq","bits":4,"group_size":7,"version":"gemv"})",
        R"({"quant_method":"awq","bits":4,"group_size":7,"zero_point":false})",
        R"({"quant_method":"awq","bits":4,"group_size":7,"backend":"marlin"})",
        R"({"quant_method":"awq","bits":4,"group_size":7,"dynamic":{"linear":{}}})",
        R"({"quant_method":"awq","bits":4,"group_size":7,"modules_in_block_to_quantize":["linear"]})",
        R"({"quant_method":"gptq","bits":4,"group_size":5,"checkpoint_format":"gptq_v2"})",
        R"({"quant_method":"gptq","bits":4,"group_size":5,"checkpoint_format":"marlin"})",
        R"({"quant_method":"gptq","bits":4,"group_size":5,"desc_act":"true"})",
        R"({"quant_method":"gptq","bits":4,"group_size":5,"sym":0})",
        R"({"quant_method":"gptq","bits":4,"group_size":5,"use_marlin":true})",
    };
    for (const std::string& config : configs)
    {
        NativeAwqDirectory directory;
        directory.config(config);
        NativeAwqArchive fixture;
        fixture.projection("linear.", true, 19, 7, "F16", false, false);
        fixture.write(directory.path);
        native_awq_check(!SafetensorsArchive::open(directory.path), "unsupported source quantization config accepted");
    }
    for (int variant = 0; variant < 4; ++variant)
    {
        NativeAwqDirectory directory;
        const bool awq = variant == 3;
        if (variant != 0)
            directory.config(native_awq_config(awq, 5, variant == 1));
        NativeAwqArchive fixture;
        fixture.projection("linear.", awq, 24, 5, "F16", variant >= 2, true);
        fixture.write(directory.path);
        auto opened = SafetensorsArchive::open(directory.path);
        if (opened)
            native_awq_check(!opened.value().load_tensor("linear.weight"), "missing/noncanonical GPTQ g_idx or AWQ g_idx accepted");
    }
}

static void test_native_awq_unquantized_weights()
{
    for (bool source_quantized : {false, true})
    {
        NativeAwqDirectory directory;
        NativeAwqArchive fixture;
        if (source_quantized)
        {
            directory.config(native_awq_config(true, 7));
            fixture.projection("linear.", true, 19, 7, "F16", false, false);
        }
        const std::array<uint16_t, 7> values = {0x3c01, 0xbc03, 0x0001, 0x8000, 0x0400, 0x3c04, 0x3c0c};
        const std::array<uint16_t, 7> expected = {0x3f80, 0xbf80, 0x3380, 0x8000, 0x3880, 0x3f80, 0x3f82};
        fixture.add("model.embed_tokens.weight", "F16", {1, 7}, values.data(), sizeof(values));
        const std::array<float, 2> norms = {1.00390625f, 1.01171875f};
        fixture.add("model.norm.weight", "F32", {2}, norms.data(), sizeof(norms));
        const uint16_t canonical = 0x4000;
        fixture.add("model.language_model.norm.weight", "BF16", {1}, &canonical, sizeof(canonical));
        fixture.write(directory.path);
        auto opened = SafetensorsArchive::open(directory.path);
        native_awq_check(static_cast<bool>(opened), "unquantized source fixture rejected");
        auto embedding = opened.value().load_tensor("model.language_model.embed_tokens.weight");
        native_awq_check(embedding && embedding.value().dtype == DType::BFloat16, "F16 alias did not convert to BF16");
        native_awq_check(std::equal(expected.begin(), expected.end(), embedding.value().bfloat16_values().begin()), "F16 BF16 rounding or subnormals changed");
        auto norm = opened.value().load_tensor("model.norm.weight");
        native_awq_check(static_cast<bool>(norm), "F32 norm loading failed");
        if (source_quantized)
            native_awq_check(norm.value().dtype == DType::BFloat16 && norm.value().bfloat16_values()[0] == 0x3f80 && norm.value().bfloat16_values()[1] == 0x3f82, "source F32 norm did not normalize to BF16");
        else
            native_awq_check(norm.value().dtype == DType::Float32 && norm.value().float32_values()[0] == norms[0], "ordinary F32 source changed dtype");
        auto exact = opened.value().load_tensor("model.language_model.norm.weight");
        native_awq_check(exact && exact.value().shape == std::vector<uint32_t>{1} && exact.value().bfloat16_values()[0] == canonical, "exact canonical name did not take priority over alias");
    }
}

static void test_native_awq_orphan_metadata()
{
    const uint32_t index = 0;
    const std::array<float, 2> scale = {1.0f, 2.0f};
    for (bool awq : {false, true})
    {
        for (const char* suffix : {"qzeros", "g_idx"})
        {
            NativeAwqDirectory directory;
            directory.config(native_awq_config(awq, 7));
            NativeAwqArchive fixture;
            fixture.projection("linear.", awq, 24, 7, "F16", false, false);
            fixture.add(std::string("orphan.") + suffix, "I32", {1}, &index, sizeof(index));
            fixture.add("orphan.scales", "F32", {1, 2}, scale.data(), sizeof(scale));
            fixture.write(directory.path);
            auto archive = SafetensorsArchive::open(directory.path);
            native_awq_check(!archive && archive.error().code == ErrorCode::InvalidModel
                                 && archive.error().message.find("orphan") != std::string::npos,
                             "orphan source quantization metadata was accepted");
        }
        NativeAwqDirectory directory;
        directory.config(native_awq_config(awq, 7));
        NativeAwqArchive fixture;
        fixture.projection("linear.", awq, 24, 7, "F16", false, false);
        fixture.add("other_layer.scales", "F32", {2}, scale.data(), sizeof(scale));
        fixture.add("other.q_proj.scales", "F32", {1, 2}, scale.data(), sizeof(scale));
        fixture.write(directory.path);
        native_awq_check(static_cast<bool>(SafetensorsArchive::open(directory.path)), "ordinary layer scales were rejected as orphan quantization metadata");
    }
    NativeAwqDirectory directory;
    NativeAwqArchive fixture;
    fixture.add("ordinary.g_idx", "I32", {1}, &index, sizeof(index));
    fixture.add("ordinary.scales", "F32", {2}, scale.data(), sizeof(scale));
    fixture.write(directory.path);
    native_awq_check(static_cast<bool>(SafetensorsArchive::open(directory.path)), "plain archive metadata was treated as AWQ/GPTQ");
}

static void test_native_awq_manifest_snapshot()
{
    NativeAwqDirectory directory;
    directory.config(native_awq_config(true, 5));
    NativeAwqArchive fixture;
    fixture.projection("linear.", true, 19, 7, "F16", false, false);
    fixture.write(directory.path);
    native_awq_check(!SafetensorsArchive::open(directory.path), "mismatched disk quantization config was accepted");
    const std::string manifest = "{\"quantization_config\":" + native_awq_config(true, 7) + '}';
    auto opened = SafetensorsArchive::open(directory.path, manifest);
    native_awq_check(static_cast<bool>(opened), "manifest snapshot did not take priority over disk config");
    auto tensor = opened.value().load_tensor("linear.weight");
    native_awq_check(tensor && tensor.value().quantization_group_size == 7, "source tensor ignored manifest quantization group size");
}

void test_native_awq()
{
    for (bool awq : {false, true})
    {
        for (const std::string dtype : {"F16", "BF16", "F32"})
        {
            check_native_awq_projection(awq, dtype, awq ? 7 : 5, false, false, false);
            check_native_awq_projection(awq, dtype, -1, false, false, true);
            if (!awq)
            {
                check_native_awq_projection(false, dtype, 5, true, true, false, true);
                check_native_awq_projection(false, dtype, 5, true, false, true);
                check_native_awq_projection(false, dtype, 5, true, false, false, true);
            }
        }
    }
    test_native_awq_invalid_inputs();
    test_native_awq_unquantized_weights();
    test_native_awq_orphan_metadata();
    test_native_awq_manifest_snapshot();
}

} // namespace moe
} // namespace ncnn
