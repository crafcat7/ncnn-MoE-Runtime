#ifndef NCNN_MOE_MODELS_MODELADAPTER_H
#define NCNN_MOE_MODELS_MODELADAPTER_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

class SafetensorsArchive;
struct ModelPackage;

// Manifest readers inspect only direct members of the supplied JSON object.
[[nodiscard]] std::optional<std::string> find_manifest_member(const std::string& json, const std::string& key);
[[nodiscard]] Result<std::string> read_manifest_object(const std::string& json, const std::string& key, const char* prefix = "");
[[nodiscard]] Result<uint32_t> read_manifest_uint32(const std::string& json, const std::string& key, const char* prefix = "");
[[nodiscard]] Result<std::string> read_manifest_string(const std::string& json, const std::string& key, const char* prefix = "");
[[nodiscard]] Result<float> read_manifest_float(const std::string& json, const std::string& key, const char* prefix = "");
[[nodiscard]] Result<bool> read_manifest_bool(const std::string& json, const std::string& key, const char* prefix = "");
[[nodiscard]] Result<std::vector<uint32_t>> read_manifest_uint32_array(const std::string& json, const std::string& key, const char* prefix = "");
[[nodiscard]] Result<std::vector<std::string>> read_manifest_string_array(const std::string& json, const std::string& key, const char* prefix = "");
float optional_manifest_float(const std::string& json, const std::string& key, float fallback);
[[nodiscard]] Result<uint32_t> get_rotary_dimension(uint32_t head_dimension,
                                                    float partial_rotary_factor,
                                                    const char* description);

[[nodiscard]] Result<bool> optional_artifact_exists(const std::filesystem::path& path,
                                                    const char* description);

[[nodiscard]] Result<void> validate_mxfp4_artifact_identity(const SafetensorsArchive& archive,
                                                            const ModelPackage& package,
                                                            const char* identity_prefix,
                                                            uint32_t layer_count,
                                                            uint32_t mtp_layer_count,
                                                            uint32_t expert_count,
                                                            uint32_t hidden_size,
                                                            uint32_t intermediate_size,
                                                            const char* identity_description,
                                                            const char* artifact_description);

[[nodiscard]] Result<void> validate_mxfp4_artifact_expert_bank(const SafetensorsArchive& archive,
                                                               const std::string& prefix,
                                                               uint32_t expert_count,
                                                               uint32_t hidden_size,
                                                               uint32_t intermediate_size,
                                                               const char* description);

[[nodiscard]] Result<void> add_tensor(WeightMapping& mapping,
                                      const SafetensorsArchive& archive,
                                      const std::string& target,
                                      const std::string& source);

[[nodiscard]] Result<void> add_bfloat16_slice(WeightMapping& mapping,
                                              const SafetensorsArchive& archive,
                                              const std::string& target_name,
                                              const std::string& source_name,
                                              uint32_t index,
                                              std::vector<uint32_t> shape);

[[nodiscard]] Result<void> add_bfloat16_expert_bank(WeightMapping& mapping,
                                                    const SafetensorsArchive& archive,
                                                    const std::string& target_prefix,
                                                    const std::string& target_suffix,
                                                    const std::string& source_name,
                                                    uint32_t expert_count,
                                                    const std::vector<uint32_t>& shape);

[[nodiscard]] Result<void> add_qwen_attention(WeightMapping& mapping, const SafetensorsArchive& archive,
                                              const std::string& source_prefix, const std::string& target_prefix,
                                              uint32_t head_count, uint32_t head_dimension, uint32_t hidden_size, const char* description);

[[nodiscard]] Result<void> add_qwen_gated_delta_net(WeightMapping& mapping, const SafetensorsArchive& archive,
                                                    const std::string& source_prefix, const std::string& target_prefix);

[[nodiscard]] Result<void> add_qwen_shared_expert(WeightMapping& mapping, const SafetensorsArchive& archive,
                                                  const std::string& source_prefix, const std::string& target_prefix);

[[nodiscard]] Result<void> add_mxfp4_expert(WeightMapping& mapping,
                                            const SafetensorsArchive& archive,
                                            const std::string& target_name,
                                            const std::string& blocks_name,
                                            const std::string& scales_name,
                                            uint32_t expert_id,
                                            uint32_t rows,
                                            uint32_t columns,
                                            uint32_t flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELS_MODELADAPTER_H
