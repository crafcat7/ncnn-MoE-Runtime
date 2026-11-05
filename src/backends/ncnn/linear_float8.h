#ifndef NCNN_MOE_LINEAR_FLOAT8_H
#define NCNN_MOE_LINEAR_FLOAT8_H

#include "linear_core.h"

namespace ncnn {
namespace moe {

class Float8Linear_vulkan
{
public:
    ~Float8Linear_vulkan();

    [[nodiscard]] static std::shared_ptr<Float8Linear_vulkan> create(const TensorData& matrix,
                                                                     const TensorData* bias,
                                                                     uint32_t input_group_count,
                                                                     uint32_t vulkan_device_index,
                                                                     const VulkanRuntimePtr& vulkan_runtime,
                                                                     uint64_t optimization_flags);
    [[nodiscard]] static std::vector<uint64_t> scratch_elements(uint32_t input_columns,
                                                                uint64_t rows);
    // Device input includes its FP32 clone; host input quantizes and uploads before recording.
    [[nodiscard]] static std::vector<uint64_t> swiglu_scratch_elements(uint32_t input_columns,
                                                                       uint32_t intermediate_columns,
                                                                       uint64_t rows,
                                                                       bool device_input = true);
    [[nodiscard]] bool prepare_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool prepare_input_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_chain(const ActivationBuffer& input, const Float8Linear_vulkan& next, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_rms_norm_chain(const ActivationBuffer& input, const Float8Linear_vulkan& next, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                       const Float8Linear_vulkan& next,
                                                       const Float8Linear_vulkan& parallel,
                                                       ActivationBuffer& output,
                                                       ActivationBuffer& parallel_output,
                                                       DeviceTensor_vulkan* retained_output = nullptr) const;
    [[nodiscard]] bool forward_input_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                             const Float8Linear_vulkan& next,
                                                             const Float8Linear_vulkan& parallel,
                                                             ActivationBuffer& output,
                                                             ActivationBuffer& parallel_output,
                                                             DeviceTensor_vulkan* retained_output = nullptr) const;
    // Extends the FP8 Q/KV chain with one or more independent BF16
    // projections from the original (pre-FP8-quantized) input.  All
    // projections share one command submission; extra outputs are returned in
    // the same order as extra_operators.
    [[nodiscard]] bool forward_rms_norm_chain_parallel_bfloat16(const ActivationBuffer& input,
                                                                const Float8Linear_vulkan& next,
                                                                const Float8Linear_vulkan& parallel,
                                                                std::span<const Bfloat16Linear_vulkan*> extra_operators,
                                                                std::span<ActivationBuffer*> extra_outputs,
                                                                ActivationBuffer& output,
                                                                ActivationBuffer& parallel_output,
                                                                DeviceTensor_vulkan* retained_output = nullptr) const;
    [[nodiscard]] bool forward_swiglu_chain(const ActivationBuffer& input,
                                            const Float8Linear_vulkan& up,
                                            const Float8Linear_vulkan& down,
                                            ExpertActivation activation,
                                            float activation_limit,
                                            ActivationBuffer& output) const;
    // Keeps normalized input and the completed FP8 Shared Expert output on the device.
    [[nodiscard]] bool forward_swiglu_chain_device(const DeviceTensor_vulkan& input,
                                                   const Float8Linear_vulkan& up,
                                                   const Float8Linear_vulkan& down,
                                                   ExpertActivation activation,
                                                   float activation_limit,
                                                   DeviceTensor_vulkan& output) const;

#if NCNN_MOE_WITH_VULKAN
    // Internal recording bridge; caller owns the command and holds the context lock.
    [[nodiscard]] bool record_input_normalization(const ncnn::VkMat& input,
                                                  ncnn::VkMat& output,
                                                  ncnn::VkCompute& cmd) const;
    [[nodiscard]] bool record_rms_norm_chain_parallel(const ncnn::VkMat& input,
                                                      const Float8Linear_vulkan& next,
                                                      const Float8Linear_vulkan& parallel,
                                                      ncnn::VkMat& query,
                                                      ncnn::VkMat& kv,
                                                      ncnn::VkMat& query_rank,
                                                      ncnn::VkCompute& cmd,
                                                      std::vector<ncnn::VkMat>& workspace) const;
#endif

    [[nodiscard]] bool materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const;

private:
    friend class Attention_vulkan;
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;
    friend class LatentAttention_vulkan;
    friend class LayerHead_vulkan;
    friend class SharedFfn_vulkan;

#if NCNN_MOE_WITH_VULKAN
    // Internal record-only path: caller owns submission, operand/workspace
    // lifetime, and publication of the collected statistics after completion.
    [[nodiscard]] bool record_swiglu_chain(const ncnn::VkMat& input,
                                           const Float8Linear_vulkan& up, const Float8Linear_vulkan& down,
                                           ExpertActivation activation, float activation_limit,
                                           ncnn::VkMat& output, ncnn::VkCompute& command,
                                           std::vector<ncnn::VkMat>& workspace,
                                           FfnRecordStatistics_vulkan& statistics) const;
#endif

    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;
    [[nodiscard]] bool record_forward(const ncnn::VkMat& input,
                                      ncnn::VkMat& output,
                                      ncnn::VkCompute& command,
                                      std::vector<ncnn::VkMat>& workspace,
                                      uint32_t output_column_offset = 0,
                                      uint32_t output_stride = 0) const;

    class Implementation;

    [[nodiscard]] bool prepare_rms_norm_weight(const TensorData& weight,
                                               uint32_t expected_columns,
                                               float epsilon,
                                               ncnn::VkMat& destination);
    [[nodiscard]] bool forward_rms_norm_chain_parallel_impl(const ActivationBuffer& input,
                                                            const Float8Linear_vulkan& next,
                                                            const Float8Linear_vulkan& parallel,
                                                            ActivationBuffer& output,
                                                            ActivationBuffer& parallel_output,
                                                            bool normalize_input,
                                                            DeviceTensor_vulkan* retained_output) const;

    Float8Linear_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif
