#ifndef NCNN_MOE_LINEAR_H
#define NCNN_MOE_LINEAR_H

#include "kernels/activationbuffer.h"

#include "ncnn/moe/types.h"
#include "ncnn/moe/option.h"
#include "vulkan.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
class VkAllocator;
class VkCompute;
class VkMat;
class Option;

namespace moe {

class VulkanContext;
class VulkanWeightUploadBatch;
class GatedDeltaNet_vulkan;

enum class LinearDevice
{
    Cpu,
    Vulkan
};

class Linear
{
public:
    ~Linear();

    [[nodiscard]] static std::shared_ptr<Linear> create(const TensorData& matrix, const TensorData* bias,
                                                        LinearDevice device,
                                                        uint32_t vulkan_device_index,
                                                        const VulkanRuntimePtr& vulkan_runtime,
                                                        uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<Linear> create_fused(const std::vector<const TensorData*>& matrices,
                                                              const std::vector<const TensorData*>& biases, LinearDevice device,
                                                              uint32_t vulkan_device_index,
                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                              uint64_t optimization_flags);
    [[nodiscard]] static const char* cpu_small_bfloat16_linear_policy(uint64_t optimization_flags) noexcept;
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] bool uses_vulkan() const noexcept;

private:
    friend class CommandGraph_vulkan;
    friend class Attention_vulkan;

    // Records a device projection; the caller holds the Vulkan context lock.
    [[nodiscard]] int forward(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

    class Implementation;

    Linear();
    std::unique_ptr<Implementation> d;
};

// A small graph-level bridge for generic ncnn Linear operators.  Tensors stay
// in Vulkan storage until the caller explicitly requests a download.  This is
// intentionally narrower than the model executor: it provides a reusable
// lifetime/synchronization contract while keeping the existing CPU fallback.
class DeviceTensor_vulkan
{
public:
    DeviceTensor_vulkan();
    ~DeviceTensor_vulkan();

    DeviceTensor_vulkan(DeviceTensor_vulkan&&) noexcept;
    DeviceTensor_vulkan& operator=(DeviceTensor_vulkan&&) noexcept;
    DeviceTensor_vulkan(const DeviceTensor_vulkan&) = delete;
    DeviceTensor_vulkan& operator=(const DeviceTensor_vulkan&) = delete;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] uint32_t columns() const noexcept;

private:
    class Implementation;
    std::unique_ptr<Implementation> d;

    friend class CommandGraph_vulkan;
};

class CommandGraph_vulkan
{
public:
    ~CommandGraph_vulkan();

    CommandGraph_vulkan(const CommandGraph_vulkan&) = delete;
    CommandGraph_vulkan& operator=(const CommandGraph_vulkan&) = delete;
    CommandGraph_vulkan(CommandGraph_vulkan&&) noexcept;
    CommandGraph_vulkan& operator=(CommandGraph_vulkan&&) noexcept;

    // Creates a graph using the Vulkan context and activation storage policy
    // of seed.  Returns null for CPU-only operators/devices.
    [[nodiscard]] static std::unique_ptr<CommandGraph_vulkan> create(const Linear& seed);

    [[nodiscard]] bool upload(const ActivationBuffer& input,
                              DeviceTensor_vulkan& output);
    [[nodiscard]] bool linear(const Linear& op,
                              const DeviceTensor_vulkan& input,
                              DeviceTensor_vulkan& output);
    [[nodiscard]] bool download(const DeviceTensor_vulkan& input,
                                ActivationBuffer& output);
    [[nodiscard]] bool submit();
    [[nodiscard]] bool wait();

private:
    class Implementation;
    std::unique_ptr<Implementation> d;

    explicit CommandGraph_vulkan(std::unique_ptr<Implementation> implementation);
};

class Bfloat16Linear_vulkan
{
public:
    ~Bfloat16Linear_vulkan();

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
    [[nodiscard]] bool prepare_rms_norm(const TensorData& weight,
                                        float epsilon,
                                        float weight_offset = 0.0f);
    [[nodiscard]] bool has_rms_norm_chain() const noexcept;
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
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

private:
    friend class Float8Linear_vulkan;
    friend class GatedDeltaNet_vulkan;
    friend class Attention_vulkan;
    friend class Bfloat16Expert_vulkan;

    // Records a scalar device projection; the caller holds the Vulkan context lock.
    [[nodiscard]] int forward(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
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
                                                                                      uint64_t optimization_flags);

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
                                                                                      uint64_t optimization_flags);
    [[nodiscard]] static bool forward_batch(std::span<const Bfloat16Expert_vulkan*> experts,
                                            std::span<const ActivationBuffer*> inputs,
                                            std::span<ActivationBuffer*> outputs);
    [[nodiscard]] uint32_t output_columns() const noexcept;

    class Implementation;

    Bfloat16Expert_vulkan();
    std::unique_ptr<Implementation> d;
};

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
    [[nodiscard]] bool prepare_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool prepare_input_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_chain(const ActivationBuffer& input, const Float8Linear_vulkan& next, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_rms_norm_chain(const ActivationBuffer& input, const Float8Linear_vulkan& next, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                       const Float8Linear_vulkan& next,
                                                       const Float8Linear_vulkan& parallel,
                                                       ActivationBuffer& output,
                                                       ActivationBuffer& parallel_output) const;
    [[nodiscard]] bool forward_input_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                             const Float8Linear_vulkan& next,
                                                             const Float8Linear_vulkan& parallel,
                                                             ActivationBuffer& output,
                                                             ActivationBuffer& parallel_output) const;
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
                                                                ActivationBuffer& parallel_output) const;
    [[nodiscard]] bool forward_swiglu_chain(const ActivationBuffer& input,
                                            const Float8Linear_vulkan& up,
                                            const Float8Linear_vulkan& down,
                                            ExpertActivation activation,
                                            float activation_limit,
                                            ActivationBuffer& output) const;

private:
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
                                                            bool normalize_input) const;

    Float8Linear_vulkan();
    std::unique_ptr<Implementation> d;
};

class Mxfp4Linear_vulkan
{
public:
    ~Mxfp4Linear_vulkan();

    [[nodiscard]] static std::shared_ptr<Mxfp4Linear_vulkan> create(const TensorData& matrix, const TensorData* bias,
                                                                    uint32_t vulkan_device_index,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

private:
    friend class Mxfp4Expert_vulkan;
    friend class VulkanExpertBackend;

    [[nodiscard]] static std::shared_ptr<Mxfp4Linear_vulkan> create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                                   uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags,
                                                                                   VulkanWeightUploadBatch* upload_batch = nullptr);
    class Implementation;

    Mxfp4Linear_vulkan();
    std::unique_ptr<Implementation> d;
};

// Vulkan Qn_K projection with CPU fallback.
class QnkLinear_vulkan
{
public:
    ~QnkLinear_vulkan();

    [[nodiscard]] static std::shared_ptr<QnkLinear_vulkan> create(const TensorData& matrix,
                                                                  const TensorData* bias,
                                                                  uint32_t vulkan_device_index,
                                                                  const VulkanRuntimePtr& vulkan_runtime,
                                                                  uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

private:
    friend class QnkExpert_vulkan;

    [[nodiscard]] static std::shared_ptr<QnkLinear_vulkan> create_with_allocator(const TensorData& matrix,
                                                                                 const TensorData* bias,
                                                                                 uint32_t vulkan_device_index,
                                                                                 ncnn::VkAllocator* weight_allocator,
                                                                                 const VulkanRuntimePtr& vulkan_runtime,
                                                                                 uint64_t optimization_flags);
    class Implementation;

    QnkLinear_vulkan();
    std::unique_ptr<Implementation> d;
};

// Fused Vulkan Qn_K Expert chain.
class QnkExpert_vulkan
{
public:
    ~QnkExpert_vulkan();

    [[nodiscard]] static std::shared_ptr<QnkExpert_vulkan> create(const TensorData& gate_up,
                                                                  const TensorData* gate_up_bias,
                                                                  const TensorData& down,
                                                                  const TensorData* down_bias,
                                                                  float activation_limit,
                                                                  uint32_t vulkan_device_index,
                                                                  ExpertActivation activation,
                                                                  const VulkanRuntimePtr& vulkan_runtime,
                                                                  uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;

private:
    friend class VulkanExpertBackend;

    [[nodiscard]] static std::shared_ptr<QnkExpert_vulkan> create_with_allocator(const TensorData& gate_up,
                                                                                 const TensorData* gate_up_bias,
                                                                                 const TensorData& down,
                                                                                 const TensorData* down_bias,
                                                                                 float activation_limit,
                                                                                 uint32_t vulkan_device_index,
                                                                                 ncnn::VkAllocator* weight_allocator,
                                                                                 ExpertActivation activation,
                                                                                 const VulkanRuntimePtr& vulkan_runtime,
                                                                                 uint64_t optimization_flags);
    [[nodiscard]] static bool forward_batch(std::span<const QnkExpert_vulkan*> experts,
                                            std::span<const ActivationBuffer*> inputs,
                                            std::span<ActivationBuffer*> outputs);
    [[nodiscard]] uint32_t output_columns() const noexcept;
    class Implementation;

    QnkExpert_vulkan();
    std::unique_ptr<Implementation> d;
};

struct Mxfp4DeviceMatrixView_vulkan
{
    uint32_t output_columns = 0;
    uint32_t input_columns = 0;
    uint64_t packed_size = 0;
    uint64_t scales_size = 0;
    size_t packed_offset = 0;
    size_t scales_offset = 0;
};

class Mxfp4Expert_vulkan
{
public:
    ~Mxfp4Expert_vulkan();

    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                    const TensorData& down, const TensorData* down_bias, float activation_limit,
                                                                    uint32_t vulkan_device_index,
                                                                    ExpertActivation activation,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create_from_device_storage(const Mxfp4DeviceMatrixView_vulkan& gate_up, const TensorData* gate_up_bias, const Mxfp4DeviceMatrixView_vulkan& down,
                                                                                        const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage,
                                                                                        ExpertActivation activation, const VulkanRuntimePtr& vulkan_runtime,
                                                                                        uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;

private:
    friend class VulkanExpertBackend;

    // Caller preallocates FP32 buffers, holds the context lock, and keeps them alive until submit.
    void record(const ncnn::VkMat& input,
                ncnn::VkMat& intermediate,
                ncnn::VkMat& output,
                ncnn::VkCompute& cmd) const;

    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                   const TensorData& down, const TensorData* down_bias,
                                                                                   float activation_limit, uint32_t vulkan_device_index,
                                                                                   ncnn::VkAllocator* weight_allocator,
                                                                                   ExpertActivation activation,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags,
                                                                                   VulkanWeightUploadBatch* upload_batch = nullptr);

    class Implementation;

    Mxfp4Expert_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_H
