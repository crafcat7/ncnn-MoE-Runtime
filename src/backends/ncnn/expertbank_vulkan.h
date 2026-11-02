#ifndef NCNN_MOE_EXPERTBANK_VULKAN_H
#define NCNN_MOE_EXPERTBANK_VULKAN_H

#include "vulkan.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

namespace ncnn {
class VkCompute;
namespace moe {
class VulkanContext;

struct ExpertBankWeights_vulkan
{
    // gate_up is the separate gate matrix when layout == GateUpDown.
    const TensorData* gate_up = nullptr;
    const TensorData* up = nullptr;
    const TensorData* down = nullptr;
    const TensorData* gate_up_bias = nullptr;
    const TensorData* up_bias = nullptr;
    const TensorData* down_bias = nullptr;
    ExpertLayout layout = ExpertLayout::InterleavedGateUpDown;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    // A loader may retain a cache lease here. create copies this pair into
    // mapped upload storage before releasing it or loading the next expert.
    std::shared_ptr<const void> owner;
};

struct ExpertBankPlan_vulkan
{
    DType dtype = DType::Float32;
    ExpertLayout layout = ExpertLayout::InterleavedGateUpDown;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    uint32_t expert_count = 0;
    uint32_t input_columns = 0;
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    uint32_t device_index = 0;
    uint64_t storage_size = 0;
    uint64_t physical_size = 0;
    uint64_t alignment = 4;
    // Byte offsets and expert strides. Gate offsets are relative to each gate
    // slice; down_scales/down_bias are relative to each down slice, whose bank
    // begins at down_weights in the single immutable weight buffer.
    uint64_t gate_stride = 0;
    uint64_t down_stride = 0;
    uint64_t gate_scales = 0;
    uint64_t gate_bias = 0;
    uint64_t down_weights = 0;
    uint64_t down_scales = 0;
    uint64_t down_bias = 0;
};

using ExpertBankLoader_vulkan = std::function<Result<ExpertBankWeights_vulkan>(uint32_t expert)>;

// Immutable full-layer expert weights. Unsupported capability is decided by
// plan before selection; failure to create or record a selected bank is an error.
class ExpertBank_vulkan
{
public:
    ~ExpertBank_vulkan();
    // Metadata only: TensorData payloads may be deferred. UnsupportedModel
    // means the existing GPU streaming plan must be selected before execution.
    [[nodiscard]] static Result<ExpertBankPlan_vulkan> plan(std::span<const ExpertBankWeights_vulkan> experts,
                                                            const std::shared_ptr<VulkanContext>& context);
    [[nodiscard]] static std::vector<uint64_t> scratch_elements(const ExpertBankPlan_vulkan& plan,
                                                                 uint64_t rows, uint32_t top_k);
    [[nodiscard]] static Result<std::shared_ptr<ExpertBank_vulkan>> create(const ExpertBankPlan_vulkan& plan,
                                                                           const std::shared_ptr<VulkanContext>& context,
                                                                           const ExpertBankLoader_vulkan& loader);
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    [[nodiscard]] const ExpertBankPlan_vulkan& description() const noexcept;
    [[nodiscard]] uint64_t physical_size() const noexcept;
#if NCNN_MOE_WITH_VULKAN
    // Caller owns the command lock and retains bank, operands, output, and
    // workspace until its fence or command reset. Never submits or locks.
    // selected is FP32 [rows, 2*K+1]: IDs, weights, router status. validation
    // is a caller-zeroed shared GPU uint32 flag, ORed on invalid routes. The
    // caller must read it after the final fence and reject a nonzero value.
    [[nodiscard]] bool record(const ncnn::VkMat& normalized, const ncnn::VkMat& selected,
                              uint32_t top_k, ncnn::VkMat& combined, ncnn::VkMat& validation,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                              const ncnn::VkMat* shared = nullptr, const ncnn::VkMat* residual = nullptr) const;
#endif

private:
    class Implementation;
    ExpertBank_vulkan();
    std::unique_ptr<Implementation> d;
};
} // namespace moe
} // namespace ncnn

#endif
