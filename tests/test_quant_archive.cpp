#include "models/modeladapter.h"
#include "models/safetensors.h"
#include "kernels/qnk.h"
#include "kernels/ops.h"
#include "kernels/int4.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void quant_archive_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

class QuantArchiveTestDirectory
{
public:
    QuantArchiveTestDirectory()
    {
        static uint32_t sequence = 0;
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path()
               / ("ncnn_moe_quant_archive_" + std::to_string(timestamp) + "_" + std::to_string(sequence++));
        quant_archive_check(std::filesystem::create_directory(path), "create quant archive test directory");
    }

    ~QuantArchiveTestDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

class QuantArchiveTestFile
{
public:
    void add(const std::string& name, const char* dtype, const std::vector<uint32_t>& shape,
             const void* values, size_t size)
    {
        if (count++ != 0)
            header << ',';
        header << '"' << name << "\":{\"dtype\":\"" << dtype << "\",\"shape\":[";
        for (size_t dimension = 0; dimension < shape.size(); ++dimension)
        {
            if (dimension != 0)
                header << ',';
            header << shape[dimension];
        }
        header << "],\"data_offsets\":[" << data.size() << ',' << data.size() + size << "]}";
        const size_t offset = data.size();
        data.resize(offset + size);
        if (size != 0)
            std::memcpy(data.data() + offset, values, size);
    }

    void add_int4(const std::string& name, int malformed = -1,
                  bool include_indices = true, bool canonical_indices = false)
    {
        int64_t shape[3] = {4, 5, 2};
        uint8_t raw[12] = {0x21, 0x43, 0x05, 0x76, 0x98, 0x0a, 0xba, 0xdc, 0x0e, 0x10, 0x32, 0x04};
        float scales[12] = {0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f, 3.f};
        uint8_t zeros[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16};
        int64_t indices[5] = {2, 0, 1, 0, 2};
        if (canonical_indices)
        {
            const int64_t canonical[5] = {0, 0, 1, 1, 2};
            for (size_t index = 0; index < 5; ++index)
                indices[index] = canonical[index];
        }
        if (malformed == 2)
            scales[0] = std::numeric_limits<float>::quiet_NaN();
        if (malformed == 3)
            zeros[0] = 17;
        if (malformed == 4)
            indices[0] = -1;
        if (malformed == 5)
            indices[0] = 3;
        if (malformed == 10)
            shape[0] = -1;
        if (malformed == 11)
            shape[2] = 0;
        if (malformed == 14)
            scales[0] = 0.f;
        if (malformed == 15)
            scales[0] = std::numeric_limits<float>::infinity();
        if (malformed == 16)
            scales[0] = -1.f;
        add(name, malformed == 9 ? "BF16" : "U8", malformed == 17 ? std::vector<uint32_t>{2, 6} : std::vector<uint32_t>{4, 3},
            raw, malformed == 8 ? sizeof(raw) - 1 : sizeof(raw));
        if (malformed != 0)
            add(name + ".__int4_shape", "I64", {3}, shape, sizeof(shape));
        if (malformed != 18)
            add(name + ".__int4_scales", malformed == 12 ? "BF16" : "F32", {4, malformed == 1 ? 2u : 3u},
                scales, malformed == 20 ? sizeof(scales) - sizeof(float) : sizeof(scales));
        if (malformed != 6)
            add(name + ".__int4_zeros", "U8", malformed == 19 ? std::vector<uint32_t>{12} : std::vector<uint32_t>{4, 3}, zeros, sizeof(zeros));
        if (include_indices)
        {
            add(name + ".__int4_g_idx", malformed == 13 ? "I32" : "I64", {5}, indices,
                malformed == 21 ? sizeof(indices) - sizeof(int64_t) : sizeof(indices));
        }
        if (malformed == 7)
        {
            const int64_t qnk_shape[3] = {4, 4, 256};
            add(name + ".__qnk_shape", "I64", {3}, qnk_shape, sizeof(qnk_shape));
        }
    }

    void add_qnk(const std::string& name, DType dtype, uint32_t bits, bool reordered = false)
    {
        const size_t stride = static_cast<size_t>(qnk_storage_bytes(dtype, 1, 256));
        std::vector<uint8_t> raw(stride * 4);
        for (size_t row = 0; row < 4; ++row)
            std::fill_n(raw.data() + row * stride, stride, static_cast<uint8_t>(row + 1));
        add(name, "U8", {4, 1, static_cast<uint32_t>(qnk_block_bytes(dtype))}, raw.data(), raw.size());
        const int64_t shape[3] = {bits, 4, 256};
        add(name + ".__qnk_shape", "I64", {3}, shape, sizeof(shape));
        if (reordered)
        {
            std::array<int64_t, 256> columns;
            for (uint32_t column = 0; column < columns.size(); ++column)
                columns[column] = column ^ 17u;
            add(name + ".__qnk_columns", "I64", {256}, columns.data(), sizeof(columns));
        }
    }

    void add_attention_tail(const std::string& prefix)
    {
        const uint16_t value = 0x3f80;
        for (const char* suffix : {"k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight"})
            add(prefix + suffix, "BF16", {1}, &value, sizeof(value));
    }

    void write(const std::filesystem::path& path) const
    {
        std::string encoded = '{' + header.str() + '}';
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        const uint64_t header_size = encoded.size();
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        quant_archive_check(static_cast<bool>(stream), "write native quant archive");
    }

private:
    uint32_t count = 0;
    std::ostringstream header;
    std::vector<uint8_t> data;
};

static void test_int4_archive()
{
    QuantArchiveTestDirectory directory;
    QuantArchiveTestFile file;
    file.add_int4("model.self_attn.q_proj.weight");
    file.add_attention_tail("model.self_attn.");
    file.write(directory.path / "model.safetensors");
    TensorData retained;
    {
        auto archive = SafetensorsArchive::open(directory.path);
        quant_archive_check(static_cast<bool>(archive), "open native INT4 archive");
        quant_archive_check(archive.value().has_native_quantized_tensors(), "detect native quant archive");
        auto dtype = archive.value().tensor_dtype("model.self_attn.q_proj.weight");
        quant_archive_check(dtype && dtype.value() == DType::Int4, "inspect native INT4 dtype");
        auto tensor = archive.value().load_tensor("model.self_attn.q_proj.weight");
        quant_archive_check(static_cast<bool>(tensor), "load native INT4 tensor");
        const TensorData& source = tensor.value();
        quant_archive_check(source.shape == std::vector<uint32_t>{4, 5}, "native INT4 logical shape");
        quant_archive_check(source.mapped_data && source.quantized_data.empty() && source.bfloat16_data.empty(), "INT4 weights stay packed and mapped");
        quant_archive_check(source.int4_values().size() == 12 && source.int4_values()[7] == 0xdc, "preserve packed INT4 bytes");
        quant_archive_check(source.quantization_group_indices == std::vector<uint32_t>{2, 0, 1, 0, 2}, "preserve nonmonotonic GPTQ group indices");
        quant_archive_check(source.quantization_scales[11] == 3.f && source.quantization_zero_points[11] == 16, "preserve scale and zero point values");
        auto selected = select_tensor_rows(source, {2, 0});
        quant_archive_check(static_cast<bool>(selected), "select reordered INT4 rows");
        quant_archive_check(selected.value().shape == std::vector<uint32_t>{2, 5}, "INT4 selected logical shape");
        quant_archive_check(selected.value().int4_values()[0] == 0xba && selected.value().int4_values()[3] == 0x21, "select packed INT4 row bytes");
        quant_archive_check(selected.value().quantization_scales == std::vector<float>{1.75f, 2.f, 2.25f, 0.25f, 0.5f, 0.75f}, "select INT4 row scales");
        quant_archive_check(selected.value().quantization_zero_points == std::vector<uint8_t>{6, 7, 8, 0, 1, 2}, "select INT4 row zero points");
        auto contiguous = select_tensor_rows(source, {1, 2});
        quant_archive_check(contiguous && contiguous.value().mapped_data.get() == source.mapped_data.get() + 3, "share contiguous packed INT4 row mapping");
        retained = std::move(contiguous).value();
        quant_archive_check(!select_tensor_rows(source, {4}) && !select_tensor_rows(source, {}), "reject invalid INT4 row selection");
        WeightMapping mapping;
        auto status = add_qwen_attention(mapping, archive.value(), "model.", "layers.0.", 2, 1, 5, "test");
        quant_archive_check(static_cast<bool>(status), "split native INT4 query gate");
        const TensorData& query = mapping.at("layers.0.attention.query.weight");
        const TensorData& gate = mapping.at("layers.0.attention.output_gate.weight");
        quant_archive_check(query.dtype == DType::Int4 && gate.dtype == DType::Int4, "query and gate retain INT4 dtype");
        quant_archive_check(query.int4_values()[0] == 0x21 && query.int4_values()[3] == 0xba, "query head row order");
        quant_archive_check(gate.int4_values()[0] == 0x76 && gate.int4_values()[3] == 0x10, "gate head row order");
        quant_archive_check(gate.quantization_zero_points == std::vector<uint8_t>{3, 4, 5, 9, 15, 16}, "query gate preserves native metadata");
    }
    quant_archive_check(retained.int4_values()[0] == 0x76 && retained.int4_values()[5] == 0x0e, "packed row mapping outlives archive");
    QuantArchiveTestFile canonical_file;
    canonical_file.add_int4("weight", -1, true, true);
    canonical_file.write(directory.path / "canonical.safetensors");
    QuantArchiveTestFile no_indices_file;
    no_indices_file.add_int4("weight", -1, false);
    no_indices_file.write(directory.path / "no-indices.safetensors");
    auto canonical_archive = SafetensorsArchive::open_file(directory.path / "canonical.safetensors");
    auto no_indices_archive = SafetensorsArchive::open_file(directory.path / "no-indices.safetensors");
    quant_archive_check(canonical_archive && no_indices_archive, "open canonical/no-index INT4 fixtures");
    auto canonical = canonical_archive.value().load_tensor("weight");
    auto no_indices = no_indices_archive.value().load_tensor("weight");
    quant_archive_check(canonical && no_indices, "load canonical/no-index INT4 fixtures");
    quant_archive_check(canonical.value().quantization_group_indices.empty()
                            && no_indices.value().quantization_group_indices.empty(),
                        "canonical native g_idx was not normalized away");
    ActivationBuffer input(3, 5);
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t column = 0; column < 5; ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>(token * 3 + column) - 2) / 4.0f;
    ActivationBuffer canonical_output;
    ActivationBuffer no_indices_output;
    quant_archive_check(forward_linear_int4(canonical.value(), input, canonical_output)
                            && forward_linear_int4(no_indices.value(), input, no_indices_output),
                        "execute canonical/no-index native INT4 fixtures");
    for (size_t row = 0; row < canonical_output.rows(); ++row)
        for (uint32_t column = 0; column < canonical_output.columns(); ++column)
            quant_archive_check(std::abs(canonical_output.row(row)[column] - no_indices_output.row(row)[column]) < 1e-6f,
                                "canonical native g_idx changed INT4 output");
    for (int malformed = 0; malformed <= 21; ++malformed)
    {
        QuantArchiveTestFile invalid;
        invalid.add_int4("weight", malformed);
        invalid.write(directory.path / "invalid.safetensors");
        auto archive = SafetensorsArchive::open_file(directory.path / "invalid.safetensors");
        quant_archive_check(static_cast<bool>(archive), "open malformed INT4 fixture");
        auto tensor = archive.value().load_tensor("weight");
        quant_archive_check(!tensor && tensor.error().code == ErrorCode::InvalidModel, "reject malformed native INT4 metadata");
    }
}

static void test_qnk_archive()
{
    const std::array<DType, 6> dtypes = {DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K};
    const std::array<uint32_t, 6> bits = {2, 3, 4, 5, 6, 8};
    QuantArchiveTestDirectory directory;
    TensorData retained;
    for (size_t index = 0; index < dtypes.size(); ++index)
    {
        QuantArchiveTestFile file;
        file.add_qnk("model.self_attn.q_proj.weight", dtypes[index], bits[index], true);
        file.add_attention_tail("model.self_attn.");
        // Retained rows keep the preceding fixture mapped until the next row
        // selection. Windows does not permit truncating that mapped file.
        const auto path = directory.path / ("qnk-" + std::to_string(bits[index]) + ".safetensors");
        file.write(path);
        auto archive = SafetensorsArchive::open_file(path);
        quant_archive_check(static_cast<bool>(archive), "open tagged Qn_K archive");
        auto dtype = archive.value().tensor_dtype("model.self_attn.q_proj.weight");
        quant_archive_check(dtype && dtype.value() == dtypes[index], "inspect tagged Qn_K dtype");
        auto loaded = archive.value().load_tensor("model.self_attn.q_proj.weight");
        quant_archive_check(loaded && loaded.value().mapped_data, "map tagged Qn_K bytes");
        quant_archive_check(!archive.value().load_qnk_tensor("model.self_attn.q_proj.weight", dtypes[index], 2, 512),
                            "reject reinterpretation of tagged Qn_K logical dimensions");
        quant_archive_check(loaded.value().input_column_indices.size() == 256
                                && loaded.value().input_column_indices[0] == 17
                                && loaded.value().input_column_indices[17] == 0,
                            "load Qn_K input-column permutation");
        const size_t stride = static_cast<size_t>(qnk_storage_bytes(dtypes[index], 1, 256));
        auto selected = select_tensor_rows(loaded.value(), {3, 0});
        quant_archive_check(selected && selected.value().qnk_values().size() == 2 * stride, "select Qn_K packed rows");
        quant_archive_check(selected.value().qnk_values()[0] == 4 && selected.value().qnk_values()[stride] == 1, "preserve Qn_K selected row blocks");
        quant_archive_check(selected.value().input_column_indices == loaded.value().input_column_indices, "row gather preserves Qn_K input-column permutation");
        auto contiguous = select_tensor_rows(loaded.value(), {1, 2});
        quant_archive_check(contiguous && contiguous.value().mapped_data.get() == loaded.value().mapped_data.get() + stride,
                            "share reordered Qn_K contiguous row mapping");
        retained = std::move(contiguous).value();
        WeightMapping mapping;
        auto status = add_qwen_attention(mapping, archive.value(), "model.", "layers.0.", 2, 1, 256, "test");
        quant_archive_check(static_cast<bool>(status), "split tagged Qn_K query gate");
        const TensorData& query = mapping.at("layers.0.attention.query.weight");
        const TensorData& gate = mapping.at("layers.0.attention.output_gate.weight");
        quant_archive_check(query.dtype == dtypes[index] && gate.dtype == dtypes[index], "query gate retain Qn_K dtype");
        quant_archive_check(query.qnk_values()[0] == 1 && query.qnk_values()[stride] == 3, "Qn_K query head row order");
        quant_archive_check(gate.qnk_values()[0] == 2 && gate.qnk_values()[stride] == 4, "Qn_K gate head row order");
        quant_archive_check(query.input_column_indices == loaded.value().input_column_indices
                                && gate.input_column_indices == loaded.value().input_column_indices,
                            "Qn_K query and gate retain input-column permutation");
    }
    quant_archive_check(retained.qnk_values()[0] == 2 && retained.input_column_indices[0] == 17,
                        "Qn_K mapped rows and column permutation outlive archive");
    for (uint32_t malformed = 0; malformed < 12; ++malformed)
    {
        QuantArchiveTestFile invalid;
        const std::vector<uint8_t> raw(144, 3);
        invalid.add("weight", "U8", {malformed == 4 ? 145u : 144u}, raw.data(), raw.size());
        int64_t shape[3] = {4, 1, 256};
        if (malformed == 0)
            shape[0] = 7;
        if (malformed == 1)
            shape[2] = 255;
        if (malformed == 2)
            shape[1] = 2;
        if (malformed != 3 && malformed != 11)
            invalid.add("weight.__qnk_shape", "I64", {3}, shape, sizeof(shape));
        if (malformed >= 5)
        {
            std::array<int64_t, 256> columns;
            for (uint32_t column = 0; column < columns.size(); ++column)
                columns[column] = column ^ 17u;
            if (malformed == 7)
                columns[1] = columns[0];
            if (malformed == 8)
                columns[0] = -1;
            if (malformed == 9)
                columns[0] = 256;
            invalid.add("weight.__qnk_columns", malformed == 5 ? "I32" : "I64", {malformed == 6 ? 255u : 256u},
                        columns.data(), malformed == 10 ? sizeof(columns) - sizeof(int64_t) : sizeof(columns));
        }
        invalid.write(directory.path / "invalid.safetensors");
        auto archive = SafetensorsArchive::open_file(directory.path / "invalid.safetensors");
        quant_archive_check(static_cast<bool>(archive), "open malformed Qn_K fixture");
        quant_archive_check(!archive.value().load_tensor("weight"), "reject malformed Qn_K columns or untagged Qn_K bytes");
        if (malformed <= 4)
            quant_archive_check(!archive.value().tensor_dtype("weight"), "reject malformed Qn_K dtype metadata");
    }
}

static void write_shape_test_archive(const std::filesystem::path& path,
                                     const std::string& shape,
                                     size_t data_size,
                                     const std::string& offsets = {})
{
    std::string header = "{\"weight\":{\"dtype\":\"BF16\",\"shape\":" + shape
                         + ",\"data_offsets\":" + (offsets.empty() ? "[0," + std::to_string(data_size) + "]" : offsets) + "}}";
    header.append((8 - header.size() % 8) % 8, ' ');
    const uint64_t header_size = header.size();
    const std::vector<uint8_t> payload(data_size, 0);
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!payload.empty())
        stream.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    quant_archive_check(static_cast<bool>(stream), "write shape parsing test archive");
}

static void test_safetensor_shape_metadata()
{
    QuantArchiveTestDirectory directory;
    const std::filesystem::path path = directory.path / "model.safetensors";
    for (const char* shape : {"[-2]", "[2.0]", "[2e0]", "[2junk]", "[2]junk", "[2,]", "[true]", "[\"2\"]", "[4294967296]", "[18446744073709551616]"})
    {
        write_shape_test_archive(path, shape, 4);
        auto archive = SafetensorsArchive::open_file(path);
        quant_archive_check(!archive && archive.error().code == ErrorCode::InvalidModel,
                            "reject negative, noninteger or malformed shape dimensions");
    }
    write_shape_test_archive(path, "[]", 2);
    auto scalar = SafetensorsArchive::open_file(path);
    quant_archive_check(scalar && scalar.value().find("weight")->shape.empty(), "preserve empty scalar shape metadata");
    write_shape_test_archive(path, "[0, 2]", 0);
    auto empty = SafetensorsArchive::open_file(path);
    quant_archive_check(empty && empty.value().find("weight")->shape == std::vector<uint32_t>{0, 2}, "preserve zero dimension shape metadata");
    write_shape_test_archive(path, "[ 2 ]", 4);
    auto normal = SafetensorsArchive::open_file(path);
    quant_archive_check(normal && normal.value().find("weight")->shape == std::vector<uint32_t>{2}, "preserve valid integer shape metadata");
    write_shape_test_archive(path, "[2]", 4, "[ 0 , 4 ]");
    auto spaced = SafetensorsArchive::open_file(path);
    quant_archive_check(spaced && spaced.value().find("weight")->size == 4,
                        "preserve whitespace in data offsets");
    for (const char* offsets : {"[4,0]", "[-1,4]", "[0,4.0]", "[0,4e0]", "[0,18446744073709551615]", "[0,18446744073709551616]", "[0,4],", "[0,4]}junk"})
    {
        write_shape_test_archive(path, "[2]", 4, offsets);
        auto invalid = SafetensorsArchive::open_file(path);
        quant_archive_check(!invalid && invalid.error().code == ErrorCode::InvalidModel,
                            "reject reversed, noninteger, overflowing or out-of-file data offsets");
    }
    QuantArchiveTestFile escaped;
    const std::array<uint16_t, 2> values = {0x3f80, 0x4000};
    escaped.add("wei\\u0067ht", "BF16", {2}, values.data(), sizeof(values));
    escaped.write(path);
    auto unicode = SafetensorsArchive::open_file(path);
    quant_archive_check(unicode && unicode.value().find("weight"), "decode escaped tensor names with the shared JSON parser");
    auto tensor = unicode.value().load_tensor("weight");
    quant_archive_check(tensor && tensor.value().bfloat16_values()[1] == values[1], "preserve escaped-name tensor payload");
}

static void test_native_float_scales()
{
    QuantArchiveTestDirectory directory;
    const std::vector<uint8_t> values(129 * 131, 0x38);
    for (const char* scale_dtype : {"F32", "F16", "BF16", "F8_E8M0"})
    {
        QuantArchiveTestFile file;
        file.add("projection.weight", "F8_E4M3", {129, 131}, values.data(), values.size());
        std::array<float, 4> scales = {0.1234567f, 0.25f, 0.375f, 0.5f};
        const std::array<uint16_t, 4> half_scales = {0x3000, 0x3400, 0x3600, 0x3800};
        const std::array<uint16_t, 4> bf16_scales = {0x3e00, 0x3e80, 0x3ec0, 0x3f00};
        const std::array<uint8_t, 4> exponent_scales = {124, 125, 126, 127};
        const std::string dtype(scale_dtype);
        const void* data = dtype == "F32"    ? static_cast<const void*>(scales.data())
                           : dtype == "F16"  ? static_cast<const void*>(half_scales.data())
                           : dtype == "BF16" ? static_cast<const void*>(bf16_scales.data())
                                             : static_cast<const void*>(exponent_scales.data());
        file.add("projection.weight_scale_inv", scale_dtype, {2, 2}, data,
                 dtype == "F32" ? 16 : dtype == "F8_E8M0" ? 4
                                                          : 8);
        file.add("retained", "F32", {4}, scales.data(), sizeof(scales));
        file.write(directory.path / "model.safetensors");
        std::ofstream config(directory.path / "config.json");
        config << R"({"quantization_config":{"quant_method":"gptq","bits":4,"group_size":128}})";
        config.close();
        auto archive = SafetensorsArchive::open(directory.path);
        quant_archive_check(static_cast<bool>(archive), "open native float scale archive");
        auto tensor = archive.value().load_tensor("projection.weight");
        quant_archive_check(tensor && tensor.value().dtype == DType::Float8E4M3
                                && tensor.value().float8_values().size() == values.size()
                                && tensor.value().float8_scale_row_size == 128
                                && tensor.value().bfloat16_data.empty(),
                            "retain native FP8 matrix storage");
        quant_archive_check(archive.value().tensor_dtype("projection.weight").value() == DType::Float8E4M3,
                            "FP8 dtype admission");
        const std::array<int32_t, 2> ids = {0, 128};
        ActivationBuffer embedding;
        forward_embedding(tensor.value(), ids, embedding);
        for (size_t row = 0; row < ids.size(); ++row)
            for (uint32_t column = 0; column < embedding.columns(); ++column)
                quant_archive_check(embedding.row(row)[column]
                                        == tensor.value().quantization_scales[(ids[row] / 128) * 2 + column / 128],
                                    "selected FP8 rows retain both scale dimensions");
        for (size_t index = 0; index < scales.size(); ++index)
        {
            const float expected = dtype == "F32"       ? scales[index]
                                   : dtype == "F8_E8M0" ? std::ldexp(1.0f, exponent_scales[index] - 127)
                                                        : (index + 1) * 0.125f;
            quant_archive_check(tensor.value().quantization_scales[index] == expected, "retain exact native FP8 scales");
        }
        if (dtype == "F32")
        {
            auto retained = archive.value().load_float32_tensor("retained");
            auto normalized = archive.value().load_tensor("retained");
            quant_archive_check(retained && retained.value().dtype == DType::Float32
                                    && retained.value().float32_values()[0] == scales[0],
                                "preserve explicit F32 auxiliary values");
            quant_archive_check(normalized && normalized.value().dtype == DType::BFloat16,
                                "default source floating-point normalization");
            quant_archive_check(!archive.value().load_float32_tensor("projection.weight"), "reject non-F32 exact loader");
        }
    }
}

static void test_float8_row_selection()
{
    constexpr uint32_t source_rows = 257;
    constexpr uint32_t columns = 131;
    const std::array<uint8_t, 5> codes = {0x30, 0x38, 0x40, 0xb8, 0xb0};
    const std::array<float, 5> decoded = {0.5f, 1.f, 2.f, -1.f, -0.5f};
    auto storage = std::make_shared<std::vector<uint8_t>>(source_rows * columns);
    for (uint32_t row = 0; row < source_rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
            (*storage)[static_cast<size_t>(row) * columns + column] = codes[(row + column) % codes.size()];
    const std::vector<uint8_t> original_bytes = *storage;
    const std::vector<float> original_scales = {0.1234567f, 0.2345678f, 0.3456789f, 0.4567891f, 0.5678912f, 0.6789123f};
    TensorData source;
    source.dtype = DType::Float8E4M3;
    source.shape = {source_rows, columns};
    source.quantization_scales = original_scales;
    source.mapped_data = std::shared_ptr<const uint8_t>(storage, storage->data());
    source.mapped_size = storage->size();
    const std::vector<uint32_t> selection = {128, 127, 256, 0, 129};
    auto selected = select_tensor_rows(source, selection);
    quant_archive_check(static_cast<bool>(selected), "select FP8 rows across both 128-row scale boundaries");
    quant_archive_check(selected.value().quantized_data.size() == selection.size() * columns
                            && !selected.value().mapped_data,
                        "noncontiguous FP8 rows expose owned packed bytes");
    const auto verify = [&](const TensorData& tensor, const std::vector<uint32_t>& rows) {
        quant_archive_check(tensor.dtype == DType::Float8E4M3
                                && tensor.shape == std::vector<uint32_t>{static_cast<uint32_t>(rows.size()), columns}
                                && tensor.float8_scale_row_size == 1
                                && tensor.quantization_scales.size() == rows.size() * 2
                                && tensor.bfloat16_data.empty() && tensor.float32_data.empty(),
                            "selected FP8 rows retain byte storage and per-row FP32 scales");
        for (size_t row = 0; row < rows.size(); ++row)
        {
            quant_archive_check(std::equal(tensor.float8_values().begin() + row * columns,
                                           tensor.float8_values().begin() + (row + 1) * columns,
                                           original_bytes.begin() + static_cast<size_t>(rows[row]) * columns),
                                "FP8 row selection preserves exact codes");
            for (size_t block = 0; block < 2; ++block)
            {
                const float actual = tensor.quantization_scales[row * 2 + block];
                const float expected = original_scales[(rows[row] / 128) * 2 + block];
                quant_archive_check(std::bit_cast<uint32_t>(actual) == std::bit_cast<uint32_t>(expected),
                                    "FP8 row selection preserves exact scale bits");
            }
        }
        std::vector<int32_t> ids;
        for (size_t row = 0; row < rows.size(); ++row)
            ids.push_back(static_cast<int32_t>(row));
        ActivationBuffer embedding;
        forward_embedding(tensor, ids, embedding);
        for (size_t row = 0; row < rows.size(); ++row)
            for (uint32_t column = 0; column < columns; ++column)
                quant_archive_check(embedding.row(row)[column]
                                        == decoded[(rows[row] + column) % codes.size()]
                                               * original_scales[(rows[row] / 128) * 2 + column / 128],
                                    "FP8 embedding uses selected row scales and original input blocks");
        ActivationBuffer input(2, columns);
        input.row(0)[0] = 1.f;
        input.row(1)[130] = 1.f;
        ActivationBuffer projection;
        forward_linear(tensor, input, projection, 0);
        for (size_t row = 0; row < rows.size(); ++row)
        {
            quant_archive_check(std::abs(projection.row(0)[row] - embedding.row(row)[0]) < 1e-6f
                                    && std::abs(projection.row(1)[row] - embedding.row(row)[130]) < 1e-6f,
                                "FP8 projection uses selected row scales across input block boundaries");
        }
    };
    verify(selected.value(), selection);
    auto nested = select_tensor_rows(selected.value(), {4, 1, 0});
    quant_archive_check(static_cast<bool>(nested), "select FP8 rows from already selected scale rows");
    verify(nested.value(), {129, 127, 128});
    auto contiguous = select_tensor_rows(source, {127, 128});
    quant_archive_check(contiguous && contiguous.value().mapped_data
                            && contiguous.value().float8_values().data() == source.float8_values().data() + 127 * columns,
                        "contiguous FP8 rows alias original mmap storage");
    TensorData malformed = source;
    malformed.float8_scale_row_size = 0;
    quant_archive_check(!select_tensor_rows(malformed, {0}), "reject zero FP8 scale row size");
    malformed.float8_scale_row_size = 64;
    quant_archive_check(!select_tensor_rows(malformed, {0}), "reject unsupported FP8 scale row size");
    malformed = source;
    malformed.quantization_scales.pop_back();
    quant_archive_check(!select_tensor_rows(malformed, {0}), "reject malformed FP8 row scale count");
    malformed = source;
    malformed.quantization_scales[0] = std::numeric_limits<float>::quiet_NaN();
    quant_archive_check(!select_tensor_rows(malformed, {0}), "reject nonfinite FP8 row scales");
    malformed = source;
    malformed.mapped_size -= 1;
    quant_archive_check(!select_tensor_rows(malformed, {0}), "reject malformed FP8 row payload");
    malformed = source;
    malformed.quantized_data = original_bytes;
    quant_archive_check(!select_tensor_rows(malformed, {0}), "reject duplicate FP8 row storage");
    malformed = {};
    source = {};
    storage.reset();
    verify(contiguous.value(), {127, 128});
}

void test_quant_archive()
{
    test_native_float_scales();
    test_float8_row_selection();
    test_int4_archive();
    test_qnk_archive();
    test_safetensor_shape_metadata();
}

} // namespace moe
} // namespace ncnn
