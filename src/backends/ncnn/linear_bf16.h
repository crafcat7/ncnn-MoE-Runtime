#ifndef NCNN_MOE_LINEAR_BF16_H
#define NCNN_MOE_LINEAR_BF16_H

#include "linear_core.h"

namespace ncnn {
namespace moe {

class Bfloat16Linear_vulkan
{
public:
    ~Bfloat16Linear_vulkan();

    // Owned BF16 weights, FP32 bias, optional input RMSNorm and allocator slack.
    [[nodiscard]] static uint64_t storage_bytes(const TensorData& matrix, uint64_t alignment = 256) noexcept;
    [[nodiscard]] static std::shared_ptr<Bfloat16Linear_vulkan> create(const TensorData& matrix,
                                                                       const TensorData* bias,
                                                                       uint32_t vulkan_device_index,
                                                                       const VulkanRuntimePtr& vulkan_runtime,
                                                                       uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<Bfloat16Linear_vulkan> create_fused(const std::vector<const TensorData*>& matrices,
                                                                             const std::vector<const TensorData*>& biases,
                                                                             uint32_t vulkan_device_index,
                                                                             const VulkanRuntimePtr& vulkan_runtime,
                                                                             uint64_t optimization_flags);
    [[nodiscard]] static std::vector<uint64_t> swiglu_scratch_elements(uint32_t intermediate_columns,
                                                                       bool router_gate, uint64_t rows);
    [[nodiscard]] bool prepare_rms_norm(const TensorData& weight,
                                        float epsilon,
                                        float weight_offset = 0.0f);
    [[nodiscard]] bool has_rms_norm_chain() const noexcept;
    [[nodiscard]] bool prepare_input_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool prepare_output_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    // Records a device projection without submitting or publishing counters.
    // The caller holds the context lock and retains inputs/outputs until completion.
    [[nodiscard]] int forward(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
    // Executes two independent BF16 projections from one input batch in a
    // single command buffer.  The outputs are downloaded separately, while
    // the input upload and submit/wait are shared.
    [[nodiscard]] bool forward_parallel(const ActivationBuffer& input,
                                        const Bfloat16Linear_vulkan& parallel_operator,
                                        ActivationBuffer& output,
                                        ActivationBuffer& parallel_output) const;
    [[nodiscard]] bool forward_rms_norm_chain(const ActivationBuffer& input,
                                              ActivationBuffer& output) const;
    // Fuses a BF16 gate/up projection with SiLU gating and the BF16 Down
    // projection.  The fused input operator may contain one extra scalar
    // column for a shared-Expert router gate; that scalar is applied on the
    // device before the result is downloaded.
    [[nodiscard]] bool forward_swiglu_chain(const ActivationBuffer& input,
                                            const Bfloat16Linear_vulkan& down_operator,
                                            uint32_t intermediate_columns,
                                            ExpertActivation activation,
                                            float activation_limit,
                                            bool apply_router_gate,
                                            ActivationBuffer& output) const;
    [[nodiscard]] bool forward_swiglu_chain_device(const DeviceTensor_vulkan& input,
                                                   const Bfloat16Linear_vulkan& down_operator,
                                                   uint32_t intermediate_columns,
                                                   ExpertActivation activation,
                                                   float activation_limit,
                                                   bool apply_router_gate,
                                                   DeviceTensor_vulkan& output) const;
    [[nodiscard]] bool materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const;

#if NCNN_MOE_WITH_VULKAN
    // Records the prepared RMSNorm and BF16 projection as one dispatch.
    // The caller holds the context lock, retains input/output until completion,
    // and owns submission and success counters. False does not change output.
    [[nodiscard]] bool record_rms_norm_chain(const ncnn::VkMat& input,
                                             ncnn::VkMat& output,
                                             ncnn::VkCompute& command) const;
    // Caller owns the command, holds the context lock and retains workspace until submission.
    [[nodiscard]] bool record_input_normalization(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd) const;
    [[nodiscard]] bool record_rms_norm_chain_parallel(const ncnn::VkMat& input,
                                                      const Bfloat16Linear_vulkan& next,
                                                      const Bfloat16Linear_vulkan& parallel,
                                                      ncnn::VkMat& query,
                                                      ncnn::VkMat& kv,
                                                      ncnn::VkMat& query_rank,
                                                      ncnn::VkCompute& cmd,
                                                      std::vector<ncnn::VkMat>& workspace) const;
#endif

private:
    friend class Float8Linear_vulkan;
    friend class LatentAttention_vulkan;
    friend class LayerHead_vulkan;
    friend class GatedDeltaNet_vulkan;
    friend class Attention_vulkan;
    friend class Bfloat16Expert_vulkan;
    friend class SharedFfn_vulkan;
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;

#if NCNN_MOE_WITH_VULKAN
    // Internal record-only path: caller owns submission, operand/workspace
    // lifetime, and publication of the collected statistics after completion.
    [[nodiscard]] bool record_swiglu_chain(const ncnn::VkMat& input,
                                           const Bfloat16Linear_vulkan& down,
                                           uint32_t intermediate_columns, ExpertActivation activation,
                                           float activation_limit, bool apply_router_gate,
                                           ncnn::VkMat& output, ncnn::VkCompute& command,
                                           std::vector<ncnn::VkMat>& workspace,
                                           FfnRecordStatistics_vulkan& statistics) const;
#endif

    // Caller provides allocated outputs and holds the context lock; scalar uses readonly bindings.
    void record_scalar_projection(const ncnn::VkMat& input,
                                  ncnn::VkMat& output,
                                  ncnn::VkCompute& cmd) const;
    void record_rms_norm_projection(const ncnn::VkMat& input,
                                    ncnn::VkMat& output,
                                    ncnn::VkCompute& cmd) const;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

    [[nodiscard]] static std::shared_ptr<Bfloat16Linear_vulkan> create_with_allocator(const TensorData& matrix,
                                                                                      const TensorData* bias,
                                                                                      uint32_t vulkan_device_index,
                                                                                      ncnn::VkAllocator* weight_allocator,
                                                                                      const VulkanRuntimePtr& vulkan_runtime,
                                                                                      uint64_t optimization_flags,
                                                                                      VulkanWeightUploadBatch* upload_batch = nullptr);

    class Implementation;

    Bfloat16Linear_vulkan();
    std::unique_ptr<Implementation> d;
};

// Fused Vulkan BF16 Expert chain.  The backend owns one pair of BF16
// projections per resident Expert and can record a whole routed batch in a
// single command submission.
class Bfloat16Expert_vulkan
{
public:
    ~Bfloat16Expert_vulkan();

    [[nodiscard]] static std::shared_ptr<Bfloat16Expert_vulkan> create(const TensorData& gate_up,
                                                                       const TensorData* gate_up_bias,
                                                                       const TensorData& down,
                                                                       const TensorData* down_bias,
                                                                       float activation_limit,
                                                                       uint32_t vulkan_device_index,
                                                                       ExpertActivation activation,
                                                                       const VulkanRuntimePtr& vulkan_runtime,
                                                                       uint64_t optimization_flags);
    [[nodiscard]] static std::vector<uint64_t> scratch_elements(uint32_t gate_output_columns,
                                                                uint64_t rows);
    [[nodiscard]] bool forward(const ActivationBuffer& input,
                               ActivationBuffer& output) const;

private:
    friend class VulkanExpertBackend;

    [[nodiscard]] static std::shared_ptr<Bfloat16Expert_vulkan> create_with_allocator(const TensorData& gate_up,
                                                                                      const TensorData* gate_up_bias,
                                                                                      const TensorData& down,
                                                                                      const TensorData* down_bias,
                                                                                      float activation_limit,
                                                                                      uint32_t vulkan_device_index,
                                                                                      ncnn::VkAllocator* weight_allocator,
                                                                                      ExpertActivation activation,
                                                                                      const VulkanRuntimePtr& vulkan_runtime,
                                                                                      uint64_t optimization_flags,
                                                                                      VulkanWeightUploadBatch* upload_batch = nullptr);
    [[nodiscard]] static bool forward_batch(std::span<const Bfloat16Expert_vulkan*> experts,
                                            std::span<const ActivationBuffer*> inputs,
                                            std::span<ActivationBuffer*> outputs);
    [[nodiscard]] bool record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                              std::vector<ncnn::VkMat>& bindings,
                              std::vector<ncnn::vk_constant_type>& constants) const;
    [[nodiscard]] uint32_t output_columns() const noexcept;

    class Implementation;

    Bfloat16Expert_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif
