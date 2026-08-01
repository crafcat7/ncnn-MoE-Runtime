#ifndef NCNN_MOE_SAFETENSORS_H
#define NCNN_MOE_SAFETENSORS_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

struct SafetensorInfo
{
    std::filesystem::path path;
    std::string dtype;
    std::vector<uint32_t> shape;
    uint64_t offset = 0;
    uint64_t byte_count = 0;
};

#define NCNN_MOE_SAFETENSOR_DEFER_MXFP4_BIT 0

enum SafetensorLoadFlag : uint32_t
{
    SafetensorLoadDeferMxfp4Data = UINT32_C(1) << NCNN_MOE_SAFETENSOR_DEFER_MXFP4_BIT
};

class SafetensorsArchive
{
private:
    std::unordered_map<std::string, SafetensorInfo> tensors_;

public:
    [[nodiscard]] static Result<SafetensorsArchive> open(const std::filesystem::path& root);
    [[nodiscard]] static Result<SafetensorsArchive> open_file(const std::filesystem::path& path);
    [[nodiscard]] const SafetensorInfo* find(const std::string& name) const noexcept;
    [[nodiscard]] Result<TensorData> load_tensor(const std::string& name) const;
    [[nodiscard]] Result<TensorData> load_bfloat16_slice(const std::string& name, uint32_t index, std::vector<uint32_t> shape) const;
    [[nodiscard]] Result<TensorData> load_float8_tensor(const std::string& weight_name, const std::string& scale_name) const;
    [[nodiscard]] Result<TensorData> load_mxfp4_tensor(const std::string& blocks_name, const std::string& scales_name, uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
    [[nodiscard]] Result<TensorData> load_interleaved_mxfp4_tensor(const std::string& gate_blocks_name, const std::string& gate_scales_name,
                                                                   const std::string& up_blocks_name, const std::string& up_scales_name,
                                                                   uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
    [[nodiscard]] Result<TensorData> load_mxfp4_expert(const std::string& blocks_name, const std::string& scales_name, uint32_t expert_id, uint32_t rows, uint32_t columns, uint32_t flags = 0) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SAFETENSORS_H
