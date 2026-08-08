#include "safetensors.h"
#include "kernels/cpu_qnk.h"
#include "storage/mapped_file.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <cmath>
#include <limits>
#include <regex>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

static Result<std::vector<uint8_t>> read_range(const std::filesystem::path& path, uint64_t offset, uint64_t byte_count)
{
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) || byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return Error{ErrorCode::InvalidModel, "safetensors range is too large: " + path.string()};

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open safetensors shard: " + path.string()};
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot seek safetensors shard: " + path.string()};

    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    if (byte_count > 0 && !stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byte_count)))
        return Error{ErrorCode::InvalidModel, "safetensors shard is truncated: " + path.string()};
    return bytes;
}

static std::string packed_tensor_name(const std::string& name)
{
    return "__ncnn_moe_packed__." + name;
}

static bool parse_json_string(const std::string& json, size_t& position, std::string& value)
{
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
        ++position;
    if (position >= json.size() || json[position] != '"')
        return false;
    ++position;
    value.clear();
    while (position < json.size())
    {
        const char character = json[position++];
        if (character == '"')
            return true;
        if (character == '\\' && position < json.size())
        {
            value.push_back(json[position++]);
        }
        else
        {
            value.push_back(character);
        }
    }
    return false;
}

static bool find_object_end(const std::string& json, size_t begin, size_t& end)
{
    if (begin >= json.size() || json[begin] != '{')
        return false;
    uint32_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t position = begin; position < json.size(); ++position)
    {
        const char character = json[position];
        if (in_string)
        {
            if (escaped)
            {
                escaped = false;
            }
            else if (character == '\\')
            {
                escaped = true;
            }
            else if (character == '"')
            {
                in_string = false;
            }
            continue;
        }
        if (character == '"')
        {
            in_string = true;
        }
        else if (character == '{')
        {
            ++depth;
        }
        else if (character == '}')
        {
            if (--depth == 0)
            {
                end = position + 1;
                return true;
            }
        }
    }
    return false;
}

static Result<std::vector<uint32_t>> parse_shape(const std::string& text)
{
    const std::regex number_expression("[0-9]+");
    std::vector<uint32_t> shape;
    for (std::sregex_iterator iterator(text.begin(), text.end(), number_expression), end; iterator != end; ++iterator)
    {
        try
        {
            const unsigned long long value = std::stoull(iterator->str());
            if (value > std::numeric_limits<uint32_t>::max())
                return Error{ErrorCode::InvalidModel, "safetensors dimension is out of range"};
            shape.push_back(static_cast<uint32_t>(value));
        }
        catch (const std::exception&)
        {
            return Error{ErrorCode::InvalidModel, "invalid safetensors dimension"};
        }
    }
    return shape;
}

static Result<void> parse_header(
    const std::filesystem::path& path,
    const std::string& json,
    uint64_t data_start,
    uint64_t file_bytes,
    std::unordered_map<std::string, SafetensorInfo>& tensors)
{
    const std::regex dtype_expression("\"dtype\"\\s*:\\s*\"([^\"]+)\"");
    const std::regex shape_expression("\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
    const std::regex offsets_expression("\"data_offsets\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    size_t position = 1;
    while (position < json.size())
    {
        while (position < json.size() && (std::isspace(static_cast<unsigned char>(json[position])) || json[position] == ','))
            ++position;
        if (position >= json.size() || json[position] == '}')
            break;

        std::string name;
        if (!parse_json_string(json, position, name))
            return Error{ErrorCode::InvalidModel, "invalid safetensors header key: " + path.string()};
        while (position < json.size() && (std::isspace(static_cast<unsigned char>(json[position])) || json[position] == ':'))
            ++position;
        size_t object_end = 0;
        if (!find_object_end(json, position, object_end))
            return Error{ErrorCode::InvalidModel, "invalid safetensors tensor object: " + path.string()};
        const std::string object = json.substr(position, object_end - position);
        position = object_end;
        if (name == "__metadata__")
            continue;

        std::smatch dtype_match;
        std::smatch shape_match;
        std::smatch offsets_match;
        if (!std::regex_search(object, dtype_match, dtype_expression)
            || !std::regex_search(object, shape_match, shape_expression)
            || !std::regex_search(object, offsets_match, offsets_expression))
            return Error{ErrorCode::InvalidModel, "incomplete safetensors tensor metadata: " + name};

        auto shape = parse_shape(shape_match[1].str());
        if (!shape)
            return shape.error();
        uint64_t begin = 0;
        uint64_t end = 0;
        try
        {
            begin = std::stoull(offsets_match[1].str());
            end = std::stoull(offsets_match[2].str());
        }
        catch (const std::exception&)
        {
            return Error{ErrorCode::InvalidModel, "invalid safetensors data offsets: " + name};
        }
        if (end < begin
            || data_start > std::numeric_limits<uint64_t>::max() - end
            || data_start + end > file_bytes)
            return Error{ErrorCode::InvalidModel, "invalid safetensors data range: " + name};
        if (tensors.contains(name))
            return Error{ErrorCode::InvalidModel, "duplicate safetensors tensor: " + name};
        tensors.emplace(std::move(name), SafetensorInfo{path, dtype_match[1].str(), std::move(shape).value(), data_start + begin, end - begin});
    }
    return {};
}

static Result<void> open_safetensors_file(
    const std::filesystem::path& path,
    std::unordered_map<std::string, SafetensorInfo>& tensors)
{
    std::error_code size_error;
    const uint64_t file_bytes = std::filesystem::file_size(path, size_error);
    if (size_error)
        return Error{ErrorCode::IoError, "cannot inspect safetensors shard: " + path.string()};

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open safetensors shard: " + path.string()};
    uint64_t header_length = 0;
    stream.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    if (!stream || header_length == 0 || header_length > 128 * 1024 * 1024)
        return Error{ErrorCode::InvalidModel, "invalid safetensors header length: " + path.string()};
    if (header_length > file_bytes - std::min<uint64_t>(file_bytes, sizeof(header_length)))
        return Error{ErrorCode::InvalidModel, "truncated safetensors header: " + path.string()};
    std::string header(static_cast<size_t>(header_length), '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header_length));
    if (!stream)
        return Error{ErrorCode::InvalidModel, "truncated safetensors header: " + path.string()};
    return parse_header(
        path,
        header,
        sizeof(header_length) + header_length,
        file_bytes,
        tensors);
}

Result<SafetensorsArchive> SafetensorsArchive::open(const std::filesystem::path& root)
{
    SafetensorsArchive archive;
    std::error_code filesystem_error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(root, filesystem_error))
    {
        if (filesystem_error)
            return Error{ErrorCode::IoError, "cannot enumerate model directory: " + root.string()};
        if (!entry.is_regular_file() || entry.path().extension() != ".safetensors")
            continue;

        auto status = open_safetensors_file(entry.path(), archive.tensors_);
        if (!status)
            return status.error();
    }
    if (archive.tensors_.empty())
        return Error{ErrorCode::InvalidModel, "model directory contains no safetensors tensors"};
    return archive;
}

Result<SafetensorsArchive> SafetensorsArchive::open_file(const std::filesystem::path& path)
{
    SafetensorsArchive archive;
    auto status = open_safetensors_file(path, archive.tensors_);
    if (!status)
        return status.error();
    if (archive.tensors_.empty())
        return Error{ErrorCode::InvalidModel, "safetensors file contains no tensors: " + path.string()};
    return archive;
}

const SafetensorInfo* SafetensorsArchive::find(const std::string& name) const noexcept
{
    const auto iterator = tensors_.find(name);
    return iterator == tensors_.end() ? nullptr : &iterator->second;
}

Result<TensorData> SafetensorsArchive::load_tensor(const std::string& name) const
{
    const SafetensorInfo* info = find(name);
    if (!info)
        return Error{ErrorCode::InvalidModel, "missing safetensors tensor: " + name};

    TensorData tensor;
    tensor.shape = info->shape;
    if (info->dtype == "BF16")
    {
        if (info->byte_count % sizeof(uint16_t) != 0)
            return Error{ErrorCode::InvalidModel, "invalid BF16 byte count: " + name};
        tensor.dtype = DType::BFloat16;
        auto mapped = MappedFileRange::open(info->path, info->offset, info->byte_count);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(uint16_t) == 0)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_byte_count = info->byte_count;
            return tensor;
        }
        auto bytes = read_range(info->path, info->offset, info->byte_count);
        if (!bytes)
            return bytes.error();
        tensor.bfloat16_data.resize(static_cast<size_t>(info->byte_count / sizeof(uint16_t)));
        std::memcpy(tensor.bfloat16_data.data(), bytes.value().data(), static_cast<size_t>(info->byte_count));
    }
    else if (info->dtype == "F32")
    {
        if (info->byte_count % sizeof(float) != 0)
            return Error{ErrorCode::InvalidModel, "invalid F32 byte count: " + name};
        tensor.dtype = DType::Float32;
        auto mapped = MappedFileRange::open(info->path, info->offset, info->byte_count);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(float) == 0)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_byte_count = info->byte_count;
            return tensor;
        }
        auto bytes = read_range(info->path, info->offset, info->byte_count);
        if (!bytes)
            return bytes.error();
        tensor.float32_data.resize(static_cast<size_t>(info->byte_count / sizeof(float)));
        std::memcpy(tensor.float32_data.data(), bytes.value().data(), static_cast<size_t>(info->byte_count));
    }
    else if (info->dtype == "I64")
    {
        if (info->byte_count % sizeof(int64_t) != 0)
            return Error{ErrorCode::InvalidModel, "invalid I64 byte count: " + name};
        tensor.dtype = DType::Int64;
        auto mapped = MappedFileRange::open(info->path, info->offset, info->byte_count);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(int64_t) == 0)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_byte_count = info->byte_count;
            return tensor;
        }
        auto bytes = read_range(info->path, info->offset, info->byte_count);
        if (!bytes)
            return bytes.error();
        tensor.int64_data.resize(static_cast<size_t>(info->byte_count / sizeof(int64_t)));
        std::memcpy(tensor.int64_data.data(), bytes.value().data(), static_cast<size_t>(info->byte_count));
    }
    else
    {
        return Error{ErrorCode::UnsupportedModel, "unsupported safetensors dtype for tensor: " + name};
    }
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_qnk_tensor(
    const std::string& name,
    DType dtype,
    uint32_t rows,
    uint32_t columns) const
{
    if (!qnk_shape_supported(dtype, rows, columns))
        return Error{ErrorCode::InvalidArgument, "invalid Qn_K tensor shape: " + name};
    const uint64_t expected_bytes = qnk_storage_bytes(dtype, rows, columns);
    const SafetensorInfo* source = find(packed_tensor_name(name));
    if (!source)
        source = find(name);
    if (!source || source->dtype != "U8" || source->byte_count != expected_bytes)
        return Error{ErrorCode::InvalidModel, "invalid Qn_K safetensors tensor: " + name};

    TensorData tensor;
    tensor.dtype = dtype;
    tensor.shape = {rows, columns};
    auto mapped = MappedFileRange::open(source->path, source->offset, source->byte_count);
    if (mapped)
    {
        mapped.value()->prefault();
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_byte_count = source->byte_count;
        return tensor;
    }
    auto bytes = read_range(source->path, source->offset, source->byte_count);
    if (!bytes)
        return bytes.error();
    tensor.quantized_data = std::move(bytes).value();
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_qnk_expert(
    const std::string& name,
    DType dtype,
    uint32_t expert_id,
    uint32_t expert_count,
    uint32_t rows,
    uint32_t columns) const
{
    if (expert_count == 0 || expert_id >= expert_count || !qnk_shape_supported(dtype, rows, columns))
        return Error{ErrorCode::InvalidArgument, "invalid Qn_K Expert shape: " + name};
    const uint64_t expert_bytes = qnk_storage_bytes(dtype, rows, columns);
    const std::string packed_name = "__ncnn_moe_packed__." + std::to_string(expert_id) + "." + name;
    const SafetensorInfo* source = find(packed_name);
    uint64_t offset = 0;
    if (source)
    {
        if (source->dtype != "U8" || source->byte_count != expert_bytes)
            return Error{ErrorCode::InvalidModel, "invalid packed Qn_K Expert tensor: " + name};
        offset = source->offset;
    }
    else
    {
        source = find(name);
        if (!source || source->dtype != "U8"
            || expert_bytes > std::numeric_limits<uint64_t>::max() / expert_count
            || source->byte_count != expert_bytes * expert_count)
        {
            return Error{ErrorCode::InvalidModel, "invalid Qn_K Expert tensor: " + name};
        }
        offset = source->offset + expert_id * expert_bytes;
    }

    TensorData tensor;
    tensor.dtype = dtype;
    tensor.shape = {rows, columns};
    auto mapped = MappedFileRange::open(source->path, offset, expert_bytes);
    if (mapped)
    {
        mapped.value()->prefault();
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_byte_count = expert_bytes;
        return tensor;
    }
    auto bytes = read_range(source->path, offset, expert_bytes);
    if (!bytes)
        return bytes.error();
    tensor.quantized_data = std::move(bytes).value();
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_float8_tensor(const std::string& weight_name, const std::string& scale_name) const
{
    const SafetensorInfo* weight = find(weight_name);
    const SafetensorInfo* scale = find(scale_name);
    if (!weight || !scale || weight->dtype != "F8_E4M3" || scale->dtype != "F8_E8M0" || weight->shape.size() != 2 || scale->shape.size() != 2)
        return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 tensor: " + weight_name};
    const uint32_t output_blocks = (weight->shape[0] + 127) / 128;
    const uint32_t input_blocks = (weight->shape[1] + 127) / 128;
    if (scale->shape != std::vector<uint32_t>{output_blocks, input_blocks}
        || weight->byte_count != static_cast<uint64_t>(weight->shape[0]) * weight->shape[1]
        || scale->byte_count != static_cast<uint64_t>(output_blocks) * input_blocks)
    {
        return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 tensor shape: " + weight_name};
    }

    TensorData tensor;
    tensor.dtype = DType::Float8E4M3;
    tensor.shape = weight->shape;
    auto mapped = MappedFileRange::open(weight->path, weight->offset, weight->byte_count);
    if (mapped)
    {
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_byte_count = weight->byte_count;
    }
    else
    {
        return Error{ErrorCode::IoError, "cannot map blockwise FP8 tensor: " + weight_name};
    }

    auto scale_bytes = read_range(scale->path, scale->offset, scale->byte_count);
    if (!scale_bytes)
        return scale_bytes.error();
    tensor.quantization_scales.resize(scale_bytes.value().size());
    for (size_t index = 0; index < scale_bytes.value().size(); ++index)
        tensor.quantization_scales[index] = std::ldexp(1.0f, static_cast<int>(scale_bytes.value()[index]) - 127);
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_mxfp4_tensor(const std::string& blocks_name, const std::string& scales_name, uint32_t rows, uint32_t columns, uint32_t flags) const
{
    const SafetensorInfo* blocks = find(blocks_name);
    const SafetensorInfo* scales = find(scales_name);
    const SafetensorInfo* packed_blocks = find(packed_tensor_name(blocks_name));
    const SafetensorInfo* packed_scales = find(packed_tensor_name(scales_name));
    if ((packed_blocks == nullptr) != (packed_scales == nullptr))
    {
        return Error{ErrorCode::InvalidModel, "incomplete packed MXFP4 tensor pair: " + blocks_name};
    }
    if (packed_blocks)
    {
        blocks = packed_blocks;
        scales = packed_scales;
    }
    if (!blocks || !scales || blocks->dtype != "I8" || scales->dtype != "F8_E8M0"
        || columns % 32 != 0
        || blocks->shape != std::vector<uint32_t>{rows, columns / 2}
        || scales->shape != std::vector<uint32_t>{rows, columns / 32})
    {
        return Error{ErrorCode::InvalidModel, "invalid packed MXFP4 tensor: " + blocks_name};
    }

    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    if (has_flag(flags, SafetensorLoadDeferMxfp4Data))
    {
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = blocks->path.string();
        storage->blocks_offset = blocks->offset;
        storage->blocks_bytes = blocks->byte_count;
        storage->scales_path = scales->path.string();
        storage->scales_offset = scales->offset;
        storage->scales_bytes = scales->byte_count;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    }

    auto block_mapping = MappedFileRange::open(blocks->path, blocks->offset, blocks->byte_count);
    auto scale_mapping = MappedFileRange::open(scales->path, scales->offset, scales->byte_count);
    if (block_mapping && scale_mapping)
    {
        block_mapping.value()->prefault();
        scale_mapping.value()->prefault();
        tensor.mxfp4_blocks = block_mapping.value()->share_bytes();
        tensor.mxfp4_scales = scale_mapping.value()->share_bytes();
        return tensor;
    }
    auto block_data = read_range(blocks->path, blocks->offset, blocks->byte_count);
    auto scale_data = read_range(scales->path, scales->offset, scales->byte_count);
    if (!block_data)
        return block_data.error();
    if (!scale_data)
        return scale_data.error();
    tensor.mxfp4_blocks.assign(block_data.value().data(), block_data.value().size());
    tensor.mxfp4_scales.assign(scale_data.value().data(), scale_data.value().size());
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_interleaved_mxfp4_tensor(const std::string& gate_blocks_name, const std::string& gate_scales_name,
                                                                     const std::string& up_blocks_name, const std::string& up_scales_name,
                                                                     uint32_t rows, uint32_t columns, uint32_t flags) const
{
    if (has_flag(flags, SafetensorLoadDeferMxfp4Data))
    {
        const SafetensorInfo* gate_blocks = find(gate_blocks_name);
        const SafetensorInfo* gate_scales = find(gate_scales_name);
        const SafetensorInfo* up_blocks = find(up_blocks_name);
        const SafetensorInfo* up_scales = find(up_scales_name);
        const SafetensorInfo* packed_gate_blocks = find(packed_tensor_name(gate_blocks_name));
        const SafetensorInfo* packed_gate_scales = find(packed_tensor_name(gate_scales_name));
        const SafetensorInfo* packed_up_blocks = find(packed_tensor_name(up_blocks_name));
        const SafetensorInfo* packed_up_scales = find(packed_tensor_name(up_scales_name));
        const bool any_packed = packed_gate_blocks || packed_gate_scales
                                || packed_up_blocks || packed_up_scales;
        const bool all_packed = packed_gate_blocks && packed_gate_scales
                                && packed_up_blocks && packed_up_scales;
        if (any_packed && !all_packed)
        {
            return Error{ErrorCode::InvalidModel, "incomplete packed interleaved MXFP4 tensor: " + gate_blocks_name};
        }
        if (all_packed)
        {
            gate_blocks = packed_gate_blocks;
            gate_scales = packed_gate_scales;
            up_blocks = packed_up_blocks;
            up_scales = packed_up_scales;
        }
        if (!gate_blocks || !gate_scales || !up_blocks || !up_scales
            || gate_blocks->dtype != "I8" || up_blocks->dtype != "I8"
            || gate_scales->dtype != "F8_E8M0" || up_scales->dtype != "F8_E8M0"
            || gate_blocks->shape != std::vector<uint32_t>{rows, columns / 2}
            || up_blocks->shape != gate_blocks->shape
            || gate_scales->shape != std::vector<uint32_t>{rows, columns / 32}
            || up_scales->shape != gate_scales->shape)
        {
            return Error{ErrorCode::InvalidModel, "invalid interleaved MXFP4 tensor: " + gate_blocks_name};
        }
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {rows * 2, columns};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = gate_blocks->path.string();
        storage->blocks_offset = gate_blocks->offset;
        storage->blocks_bytes = gate_blocks->byte_count;
        storage->scales_path = gate_scales->path.string();
        storage->scales_offset = gate_scales->offset;
        storage->scales_bytes = gate_scales->byte_count;
        storage->secondary_blocks_path = up_blocks->path.string();
        storage->secondary_blocks_offset = up_blocks->offset;
        storage->secondary_blocks_bytes = up_blocks->byte_count;
        storage->secondary_scales_path = up_scales->path.string();
        storage->secondary_scales_offset = up_scales->offset;
        storage->secondary_scales_bytes = up_scales->byte_count;
        storage->interleave_rows = true;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    }

    auto gate = load_mxfp4_tensor(gate_blocks_name, gate_scales_name, rows, columns);
    auto up = load_mxfp4_tensor(up_blocks_name, up_scales_name, rows, columns);
    if (!gate)
        return gate.error();
    if (!up)
        return up.error();
    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows * 2, columns};
    const size_t block_row_bytes = columns / 2;
    const size_t scale_row_bytes = columns / 32;
    tensor.mxfp4_blocks.resize(static_cast<size_t>(rows) * block_row_bytes * 2);
    tensor.mxfp4_scales.resize(static_cast<size_t>(rows) * scale_row_bytes * 2);
    for (uint32_t row = 0; row < rows; ++row)
    {
        std::memcpy(tensor.mxfp4_blocks.data() + static_cast<size_t>(row * 2) * block_row_bytes, gate.value().mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_bytes, block_row_bytes);
        std::memcpy(tensor.mxfp4_blocks.data() + static_cast<size_t>(row * 2 + 1) * block_row_bytes, up.value().mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_bytes, block_row_bytes);
        std::memcpy(tensor.mxfp4_scales.data() + static_cast<size_t>(row * 2) * scale_row_bytes, gate.value().mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_bytes, scale_row_bytes);
        std::memcpy(tensor.mxfp4_scales.data() + static_cast<size_t>(row * 2 + 1) * scale_row_bytes, up.value().mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_bytes, scale_row_bytes);
    }
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_bfloat16_slice(const std::string& name, uint32_t index, std::vector<uint32_t> shape) const
{
    const SafetensorInfo* info = find(name);
    if (!info || info->dtype != "BF16" || info->shape.empty() || index >= info->shape[0])
        return Error{ErrorCode::InvalidModel, "invalid BF16 tensor slice: " + name};
    uint64_t element_count = 1;
    for (uint32_t dimension : shape)
        element_count *= dimension;
    const uint64_t byte_count = element_count * sizeof(uint16_t);
    if (byte_count * info->shape[0] != info->byte_count)
        return Error{ErrorCode::InvalidModel, "BF16 tensor slice shape mismatch: " + name};

    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    auto mapped = MappedFileRange::open(info->path, info->offset + index * byte_count, byte_count);
    if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(uint16_t) == 0)
    {
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_byte_count = byte_count;
        return tensor;
    }
    auto bytes = read_range(info->path, info->offset + index * byte_count, byte_count);
    if (!bytes)
        return bytes.error();
    tensor.bfloat16_data.resize(static_cast<size_t>(element_count));
    std::memcpy(tensor.bfloat16_data.data(), bytes.value().data(), static_cast<size_t>(byte_count));
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_mxfp4_expert(
    const std::string& blocks_name,
    const std::string& scales_name,
    uint32_t expert_id,
    uint32_t rows,
    uint32_t columns,
    uint32_t flags) const
{
    const SafetensorInfo* blocks = find(blocks_name);
    const SafetensorInfo* scales = find(scales_name);
    if (!blocks || !scales || blocks->dtype != "U8" || scales->dtype != "U8"
        || columns % 32 != 0 || blocks->shape.size() != 4 || scales->shape.size() != 3)
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 expert tensors: " + blocks_name};
    const std::vector<uint32_t> expected_blocks = {blocks->shape[0], rows, columns / 32, 16};
    const std::vector<uint32_t> expected_scales = {scales->shape[0], rows, columns / 32};
    if (blocks->shape != expected_blocks || scales->shape != expected_scales
        || blocks->shape[0] != scales->shape[0] || expert_id >= blocks->shape[0])
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 expert tensors: " + blocks_name};

    const uint64_t block_bytes = static_cast<uint64_t>(rows) * columns / 2;
    const uint64_t scale_bytes = static_cast<uint64_t>(rows) * columns / 32;
    const std::string packed_prefix = "__ncnn_moe_packed__." + std::to_string(expert_id) + ".";
    const SafetensorInfo* packed_blocks = find(packed_prefix + blocks_name);
    const SafetensorInfo* packed_scales = find(packed_prefix + scales_name);
    if ((packed_blocks == nullptr) != (packed_scales == nullptr))
    {
        return Error{ErrorCode::InvalidModel, "incomplete packed MXFP4 Expert pair: " + blocks_name};
    }
    if (packed_blocks
        && (packed_blocks->dtype != "U8"
            || packed_scales->dtype != "U8"
            || packed_blocks->byte_count != block_bytes
            || packed_scales->byte_count != scale_bytes
            || packed_blocks->shape != std::vector<uint32_t>{rows, columns / 32, 16}
            || packed_scales->shape != std::vector<uint32_t>{rows, columns / 32}))
    {
        return Error{ErrorCode::InvalidModel, "invalid packed MXFP4 Expert tensors: " + blocks_name};
    }
    const SafetensorInfo* selected_blocks = packed_blocks ? packed_blocks : blocks;
    const SafetensorInfo* selected_scales = packed_scales ? packed_scales : scales;
    const uint64_t block_offset = selected_blocks->offset + (packed_blocks ? 0 : expert_id * block_bytes);
    const uint64_t scale_offset = selected_scales->offset + (packed_scales ? 0 : expert_id * scale_bytes);
    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    if (has_flag(flags, SafetensorLoadDeferMxfp4Data))
    {
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = selected_blocks->path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_bytes = block_bytes;
        storage->scales_path = selected_scales->path.string();
        storage->scales_offset = scale_offset;
        storage->scales_bytes = scale_bytes;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    }

    auto block_mapping = MappedFileRange::open(selected_blocks->path, block_offset, block_bytes);
    auto scale_mapping = MappedFileRange::open(selected_scales->path, scale_offset, scale_bytes);
    if (block_mapping && scale_mapping)
    {
        block_mapping.value()->prefault();
        scale_mapping.value()->prefault();
        tensor.mxfp4_blocks = block_mapping.value()->share_bytes();
        tensor.mxfp4_scales = scale_mapping.value()->share_bytes();
        return tensor;
    }

    auto block_data = read_range(selected_blocks->path, block_offset, block_bytes);
    if (!block_data)
        return block_data.error();
    auto scale_data = read_range(selected_scales->path, scale_offset, scale_bytes);
    if (!scale_data)
        return scale_data.error();
    tensor.mxfp4_blocks.assign(block_data.value().data(), block_data.value().size());
    tensor.mxfp4_scales.assign(scale_data.value().data(), scale_data.value().size());
    return tensor;
}

} // namespace moe
} // namespace ncnn
