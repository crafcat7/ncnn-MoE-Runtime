#ifndef NCNN_MOE_SAFETENSORS_H
#define NCNN_MOE_SAFETENSORS_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

struct GgufTensorTransform;

struct SafetensorInfo
{
    std::filesystem::path path;
    std::string dtype;
    std::vector<uint32_t> shape;
    uint64_t offset = 0;
    uint64_t size = 0;
    // Original GGML encoding for directly indexed GGUF ranges; UINT32_MAX for Safetensors.
    uint32_t ggml_type = UINT32_MAX;
    std::shared_ptr<const GgufTensorTransform> transform;
};

enum class GgufValueTransform
{
    None,
    SubtractOne,
    LogNegative
};

struct GgufTensorTransform
{
    // Raw ranges, with transform unset. Multiple matrices concatenate rows.
    std::vector<SafetensorInfo> sources;
    // Destination row/column -> stored row/column for floating-point tensors.
    std::vector<uint32_t> row_indices;
    std::vector<uint32_t> column_indices;
    // Stored quantized input column -> logical input column; blocks stay raw.
    std::vector<uint32_t> input_column_indices;
    GgufValueTransform value_transform = GgufValueTransform::None;
    std::vector<int64_t> generated_i64;
};

#define NCNN_MOE_SAFETENSOR_DEFER_MXFP4_BIT 0

enum SafetensorLoadFlag : uint32_t
{
    // Defers native MXFP4 and Qn_K expert ranges until cache admission.
    SafetensorLoadDeferMxfp4Data = UINT32_C(1) << NCNN_MOE_SAFETENSOR_DEFER_MXFP4_BIT
};

class SafetensorsArchive
{
public:
    [[nodiscard]] static Result<SafetensorsArchive> open(const std::filesystem::path& root, std::string_view manifest_json = {});
    [[nodiscard]] static Result<SafetensorsArchive> open_file(const std::filesystem::path& path);
    [[nodiscard]] const SafetensorInfo* find(const std::string& name) const noexcept;
    [[nodiscard]] bool contains_tensor(const std::string& name) const noexcept;
    [[nodiscard]] bool has_native_quantized_tensors() const noexcept;
    [[nodiscard]] bool has_float8_tensors() const noexcept;
    [[nodiscard]] Result<DType> tensor_dtype(const std::string& name) const;
    [[nodiscard]] Result<TensorData> load_tensor(const std::string& name, uint32_t flags = 0) const;
    [[nodiscard]] Result<TensorData> load_float32_tensor(const std::string& name) const;
    [[nodiscard]] Result<TensorData> load_bfloat16_slice(const std::string& name, uint32_t index, std::vector<uint32_t> shape) const;
    [[nodiscard]] Result<TensorData> load_float8_tensor(const std::string& weight_name, const std::string& scale_name) const;
    // Detect a native Q2_K..Q8_K or Q8_0 Expert bank by byte count and, for GGUF,
    // its original encoding. Q8_0 retains 32-value blocks.
    [[nodiscard]] std::optional<DType> find_qnk_expert_dtype(const std::string& name,
                                                             uint32_t expert_count,
                                                             uint32_t rows,
                                                             uint32_t columns) const noexcept;
    // Load a raw native GGML block-quantized matrix from the source checkpoint.
    [[nodiscard]] Result<TensorData> load_qnk_tensor(const std::string& name, DType dtype, uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
    // Load one Expert slice from a bank tensor.
    [[nodiscard]] Result<TensorData> load_qnk_expert(const std::string& name, DType dtype, uint32_t expert_id, uint32_t expert_count,
                                                     uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
    [[nodiscard]] Result<TensorData> load_mxfp4_tensor(const std::string& blocks_name, const std::string& scales_name, uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
    [[nodiscard]] Result<TensorData> load_interleaved_mxfp4_tensor(const std::string& gate_blocks_name, const std::string& gate_scales_name,
                                                                   const std::string& up_blocks_name, const std::string& up_scales_name,
                                                                   uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
    [[nodiscard]] Result<TensorData> load_mxfp4_expert(const std::string& blocks_name, const std::string& scales_name, uint32_t expert_id, uint32_t rows, uint32_t columns, uint32_t flags = 0) const;

private:
    [[nodiscard]] static Result<TensorData> load_gguf_source(SafetensorInfo info,
                                                             std::optional<uint32_t> slice,
                                                             bool preserve_float32);
    [[nodiscard]] Result<TensorData> load_gguf_tensor(const SafetensorInfo& info,
                                                      std::optional<uint32_t> slice = std::nullopt,
                                                      std::vector<uint32_t> shape = {},
                                                      bool preserve_float32 = false) const;
    [[nodiscard]] Result<TensorData> load_source_int4_tensor(const std::string& name) const;

    std::unordered_map<std::string, SafetensorInfo> tensors;
    std::string source_quantization_method;
    int64_t source_quantization_group_size = 0;
    bool source_quantization_desc_act = false;
    bool gguf_source = false;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SAFETENSORS_H
