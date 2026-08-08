#ifndef NCNN_MOE_NCNN_LINEAR_H
#define NCNN_MOE_NCNN_LINEAR_H

#include "kernels/cpu_batch.h"

#include "ncnn/moe/compiled_operator.h"
#include "ncnn/moe/types.h"
#include "ncnn/moe/runtime_config.h"
#include "ncnn/moe/vulkan_context.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
class VkAllocator;
class VkMat;
class Option;

namespace moe {

struct CpuLayerCache;
class NcnnVulkanContext;

struct NcnnVulkanGatedDeltaBatchEntry
{
    const ActivationBuffer* normalized = nullptr;
    CpuLayerCache* cache = nullptr;
    ActivationBuffer* projected = nullptr;
};

enum class NcnnVulkanGatedDeltaBatchResult
{
    NotExecuted,
    Executed,
    Failed
};

enum class NcnnLinearDevice
{
    Cpu,
    Vulkan
};

struct NcnnVulkanRuntimeCounters
{
    uint64_t compute_submissions = 0;
    uint64_t submit_wait_time_microseconds = 0;
    uint64_t batch_uploads = 0;
    uint64_t batch_downloads = 0;
    uint64_t auxiliary_uploads = 0;
    uint64_t auxiliary_upload_bytes = 0;
    uint64_t staging_slot_resizes = 0;
    uint64_t staging_slot_reuses = 0;
    uint64_t staging_slot_acquisitions = 0;
    uint64_t staging_slot_contentions = 0;
    uint64_t command_buffer_reuses = 0;
    uint64_t command_graph_submissions = 0;
    uint64_t command_graph_operations = 0;
    uint64_t direct_host_input_bindings = 0;
    uint64_t direct_host_output_bindings = 0;
    uint64_t attention_qkv_rope_fusions = 0;
    uint64_t attention_device_rope_fusions = 0;
    uint64_t attention_qkv_ring_fusions = 0;
    uint64_t attention_qkv_rope_pipeline_failures = 0;
    uint64_t attention_qkv_rope_shape_failures = 0;
    uint64_t attention_qkv_rope_source_failures = 0;
    uint64_t attention_qkv_rope_norm_failures = 0;
    uint64_t attention_qkv_rope_ring_failures = 0;
    uint64_t attention_qkv_rope_allocation_failures = 0;
    uint64_t attention_precondition_failures = 0;
    uint64_t attention_staging_failures = 0;
    uint64_t attention_norm_failures = 0;
    uint64_t attention_qkv_failures = 0;
    uint64_t attention_cache_failures = 0;
    uint64_t attention_sdpa_failures = 0;
    uint64_t attention_projection_failures = 0;
    uint64_t attention_output_failures = 0;
    uint64_t attention_submit_failures = 0;
    uint64_t attention_decode_sdpa_fusions = 0;
    uint64_t attention_cache_materializations = 0;
    uint64_t attention_cpu_fallbacks = 0;
    uint64_t shared_expert_swiglu_fusions = 0;
    uint64_t gated_delta_fusions = 0;
    uint64_t gated_delta_submissions = 0;
    uint64_t rms_norm_linear_fusions = 0;
    uint64_t kv_ring_appends = 0;
    uint64_t kv_ring_resizes = 0;
    uint64_t kv_ring_wrapped_views = 0;
    uint64_t kv_cache_promotions = 0;
    uint64_t kv_cache_promotion_bytes = 0;
    uint64_t bfloat16_cooperative_matrix_dispatches = 0;
    uint64_t command_dispatches = 0;
    uint64_t command_pipeline_binds = 0;
    uint64_t command_redundant_pipeline_binds = 0;
    uint64_t command_descriptor_bindings = 0;
    uint64_t command_push_constant_updates = 0;
    uint64_t command_resource_barrier_calls = 0;
    uint64_t command_buffer_resource_barriers = 0;
    uint64_t command_image_resource_barriers = 0;
};

struct NcnnVulkanExecutionSnapshot
{
    uint64_t dispatches = 0;
    uint64_t attention_blocks = 0;
    NcnnVulkanRuntimeCounters counters;
};

class NcnnLinearOperator
{
private:
    friend class NcnnVulkanCommandGraph;
    friend class NcnnVulkanAttentionOperator;

    class Implementation;

    NcnnLinearOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnLinearOperator();

    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create(const TensorData& matrix, const TensorData* bias,
                                                                    NcnnLinearDevice device,
                                                                    uint32_t vulkan_device_index,
                                                                    const NcnnVulkanContextInstancePtr& context_instance,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create_fused(const std::vector<const TensorData*>& matrices,
                                                                          const std::vector<const TensorData*>& biases, NcnnLinearDevice device,
                                                                          uint32_t vulkan_device_index,
                                                                          const NcnnVulkanContextInstancePtr& context_instance,
                                                                          uint64_t optimization_flags);
    [[nodiscard]] static const char* cpu_small_bfloat16_linear_policy(
        uint64_t optimization_flags) noexcept;
    [[nodiscard]] static uint32_t vulkan_device_count() noexcept;
    [[nodiscard]] static uint64_t vulkan_heap_budget_bytes() noexcept;
    [[nodiscard]] static std::vector<VulkanDeviceCapabilities> vulkan_device_capabilities();
    [[nodiscard]] static NcnnVulkanExecutionSnapshot vulkan_execution_snapshot(
        const NcnnVulkanContextInstancePtr& context_instance) noexcept;
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] bool uses_vulkan() const noexcept;
};

// A small graph-level bridge for generic ncnn Linear operators.  Tensors stay
// in Vulkan storage until the caller explicitly requests a download.  This is
// intentionally narrower than the model executor: it provides a reusable
// lifetime/synchronization contract while keeping the existing CPU fallback.
class NcnnVulkanDeviceTensor
{
private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;

    friend class NcnnVulkanCommandGraph;

public:
    NcnnVulkanDeviceTensor();
    ~NcnnVulkanDeviceTensor();

    NcnnVulkanDeviceTensor(NcnnVulkanDeviceTensor&&) noexcept;
    NcnnVulkanDeviceTensor& operator=(NcnnVulkanDeviceTensor&&) noexcept;
    NcnnVulkanDeviceTensor(const NcnnVulkanDeviceTensor&) = delete;
    NcnnVulkanDeviceTensor& operator=(const NcnnVulkanDeviceTensor&) = delete;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] uint32_t columns() const noexcept;
};

class NcnnVulkanCommandGraph
{
private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;

    explicit NcnnVulkanCommandGraph(std::unique_ptr<Implementation> implementation);

public:
    ~NcnnVulkanCommandGraph();

    NcnnVulkanCommandGraph(const NcnnVulkanCommandGraph&) = delete;
    NcnnVulkanCommandGraph& operator=(const NcnnVulkanCommandGraph&) = delete;
    NcnnVulkanCommandGraph(NcnnVulkanCommandGraph&&) noexcept;
    NcnnVulkanCommandGraph& operator=(NcnnVulkanCommandGraph&&) noexcept;

    // Creates a graph using the Vulkan context and activation storage policy
    // of seed_operator.  Returns null for CPU-only operators/devices.
    [[nodiscard]] static std::unique_ptr<NcnnVulkanCommandGraph> create(
        const NcnnLinearOperator& seed_operator);

    [[nodiscard]] bool upload(
        const ActivationBuffer& input,
        NcnnVulkanDeviceTensor& output);
    [[nodiscard]] bool linear(
        const NcnnLinearOperator& operator_instance,
        const NcnnVulkanDeviceTensor& input,
        NcnnVulkanDeviceTensor& output);
    [[nodiscard]] bool download(
        const NcnnVulkanDeviceTensor& input,
        ActivationBuffer& output);
    [[nodiscard]] bool submit();
    [[nodiscard]] bool wait();
};

class NcnnVulkanBfloat16Operator
{
private:
    friend class NcnnVulkanFloat8Operator;
    friend class NcnnVulkanGatedDeltaNetOperator;
    friend class NcnnVulkanAttentionOperator;

    class Implementation;

    NcnnVulkanBfloat16Operator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanBfloat16Operator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanBfloat16Operator> create(
        const TensorData& matrix,
        const TensorData* bias,
        uint32_t vulkan_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<NcnnVulkanBfloat16Operator> create_fused(
        const std::vector<const TensorData*>& matrices,
        const std::vector<const TensorData*>& biases,
        uint32_t vulkan_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] bool prepare_rms_norm(
        const TensorData& weight,
        float epsilon,
        float weight_offset = 0.0f);
    [[nodiscard]] bool has_rms_norm_chain() const noexcept;
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    // Executes two independent BF16 projections from one input batch in a
    // single command buffer.  The outputs are downloaded separately, while
    // the input upload and submit/wait are shared.
    [[nodiscard]] bool forward_parallel(
        const ActivationBuffer& input,
        const NcnnVulkanBfloat16Operator& parallel_operator,
        ActivationBuffer& output,
        ActivationBuffer& parallel_output) const;
    [[nodiscard]] bool forward_rms_norm_chain(
        const ActivationBuffer& input,
        ActivationBuffer& output) const;
    // Executes the prepared RMSNorm + projection and keeps the vocabulary
    // reduction on the device. One greedy token id is returned per input row.
    // Fuses a BF16 gate/up projection with SiLU gating and the BF16 Down
    // projection.  The fused input operator may contain one extra scalar
    // column for a shared-Expert router gate; that scalar is applied on the
    // device before the result is downloaded.
    [[nodiscard]] bool forward_swiglu_chain(
        const ActivationBuffer& input,
        const NcnnVulkanBfloat16Operator& down_operator,
        uint32_t intermediate_columns,
        ExpertActivation activation,
        float activation_limit,
        bool apply_router_gate,
        ActivationBuffer& output) const;
};

// Backend-owned fused path.  The executor only asks whether the compiled
// operator can consume the typed activation buffer; it does not inspect or
// branch on a backend enum for this optimization.
[[nodiscard]] bool try_fused_rms_norm_linear(
    const CompiledOperator& operator_entry,
    const ActivationBuffer& input,
    ActivationBuffer& output);

class NcnnVulkanGatedDeltaState
{
private:
    friend class NcnnVulkanGatedDeltaNetOperator;

public:
    class Implementation;
    NcnnVulkanGatedDeltaState();
    std::unique_ptr<Implementation> implementation_;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanGatedDeltaState> create(
        const std::shared_ptr<NcnnVulkanContext>& context,
        uint32_t convolution_size,
        uint32_t kernel_size,
        uint32_t head_count,
        uint32_t head_dimension,
        uint32_t value_head_dimension,
        const ncnn::Option& option);

public:
    ~NcnnVulkanGatedDeltaState();

    [[nodiscard]] bool begin_transaction(size_t expected_rows) noexcept;
    [[nodiscard]] bool prepare_transaction_finish(
        size_t committed_rows,
        size_t recorded_rows) noexcept;
    void complete_transaction() noexcept;
    [[nodiscard]] bool prepare_cpu_state(
        const std::vector<float>& convolution,
        const std::vector<float>& recurrent);
    [[nodiscard]] bool download(std::vector<float>& convolution, std::vector<float>& recurrent) const;
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
};

class NcnnVulkanGatedDeltaNetOperator
{
private:
    class Implementation;

    [[nodiscard]] bool forward_impl(
        const ActivationBuffer& input,
        CpuLayerCache& cache,
        ActivationBuffer& projected,
        bool apply_input_rms_norm) const;

    NcnnVulkanGatedDeltaNetOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanGatedDeltaNetOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanGatedDeltaNetOperator> create(
        const std::shared_ptr<NcnnVulkanBfloat16Operator>& fused_input,
        const TensorData& convolution_weight,
        const TensorData& time_bias,
        const TensorData& decay_log,
        const TensorData& norm_weight,
        const std::shared_ptr<NcnnVulkanBfloat16Operator>& output_projection,
        uint32_t head_count,
        uint32_t kv_head_count,
        uint32_t head_dimension,
        uint32_t value_head_dimension,
        uint32_t convolution_kernel_size,
        float norm_epsilon,
        uint32_t vulkan_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);

    [[nodiscard]] bool forward(
        const ActivationBuffer& normalized,
        CpuLayerCache& cache,
        ActivationBuffer& projected) const;
    [[nodiscard]] bool forward_input_rms_norm(
        const ActivationBuffer& input,
        CpuLayerCache& cache,
        ActivationBuffer& projected) const;
    [[nodiscard]] bool has_input_rms_norm() const noexcept;
    [[nodiscard]] NcnnVulkanGatedDeltaBatchResult forward_batch(
        std::span<const NcnnVulkanGatedDeltaBatchEntry> entries) const;
};

class NcnnVulkanFloat8Operator
{
private:
    class Implementation;

    [[nodiscard]] bool prepare_rms_norm_weight(
        const TensorData& weight,
        uint32_t expected_columns,
        float epsilon,
        ncnn::VkMat& destination);
    [[nodiscard]] bool forward_rms_norm_chain_parallel_impl(
        const ActivationBuffer& input,
        const NcnnVulkanFloat8Operator& next,
        const NcnnVulkanFloat8Operator& parallel,
        ActivationBuffer& output,
        ActivationBuffer& parallel_output,
        bool normalize_input) const;

    NcnnVulkanFloat8Operator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanFloat8Operator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanFloat8Operator> create(
        const TensorData& matrix,
        const TensorData* bias,
        uint32_t input_group_count,
        uint32_t vulkan_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] bool prepare_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool prepare_input_rms_norm(const TensorData& weight, float epsilon);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_chain(const ActivationBuffer& input, const NcnnVulkanFloat8Operator& next, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_rms_norm_chain(const ActivationBuffer& input, const NcnnVulkanFloat8Operator& next, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_rms_norm_chain_parallel(
        const ActivationBuffer& input,
        const NcnnVulkanFloat8Operator& next,
        const NcnnVulkanFloat8Operator& parallel,
        ActivationBuffer& output,
        ActivationBuffer& parallel_output) const;
    [[nodiscard]] bool forward_input_rms_norm_chain_parallel(
        const ActivationBuffer& input,
        const NcnnVulkanFloat8Operator& next,
        const NcnnVulkanFloat8Operator& parallel,
        ActivationBuffer& output,
        ActivationBuffer& parallel_output) const;
    // Extends the FP8 Q/KV chain with one or more independent BF16
    // projections from the original (pre-FP8-quantized) input.  All
    // projections share one command submission; extra outputs are returned in
    // the same order as extra_operators.
    [[nodiscard]] bool forward_rms_norm_chain_parallel_bfloat16(
        const ActivationBuffer& input,
        const NcnnVulkanFloat8Operator& next,
        const NcnnVulkanFloat8Operator& parallel,
        std::span<const NcnnVulkanBfloat16Operator*> extra_operators,
        std::span<ActivationBuffer*> extra_outputs,
        ActivationBuffer& output,
        ActivationBuffer& parallel_output) const;
    [[nodiscard]] bool forward_swiglu_chain(
        const ActivationBuffer& input,
        const NcnnVulkanFloat8Operator& up,
        const NcnnVulkanFloat8Operator& down,
        ExpertActivation activation,
        float activation_limit,
        ActivationBuffer& output) const;
};

class NcnnVulkanMxfp4Operator
{
private:
    friend class NcnnVulkanMxfp4ExpertOperator;
    friend class VulkanMxfp4ExpertBackend;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4Operator> create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                                        uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator,
                                                                                        const NcnnVulkanContextInstancePtr& context_instance,
                                                                                        uint64_t optimization_flags);
    class Implementation;

    NcnnVulkanMxfp4Operator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanMxfp4Operator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4Operator> create(const TensorData& matrix, const TensorData* bias,
                                                                         uint32_t vulkan_device_index,
                                                                         const NcnnVulkanContextInstancePtr& context_instance,
                                                                         uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;
};

// Experimental Vulkan projection for raw Q2_K-Q8_K
// matrices.  The operator keeps the canonical block bytes on device and
// performs dequantization inside the projection shader; callers can fall
// back to the CPU Qn_K path when creation or execution fails.
class NcnnVulkanQnkOperator
{
private:
    friend class NcnnVulkanQnkExpertOperator;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanQnkOperator> create_with_allocator(
        const TensorData& matrix,
        const TensorData* bias,
        uint32_t vulkan_device_index,
        ncnn::VkAllocator* weight_allocator,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    class Implementation;

    NcnnVulkanQnkOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanQnkOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanQnkOperator> create(
        const TensorData& matrix,
        const TensorData* bias,
        uint32_t vulkan_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;
};

// Fused Vulkan Qn_K Expert chain.  Gate/up and down projections remain
// quantized on device; only the final Expert output crosses the transfer
// boundary.  CPU execution remains the caller's fallback on failure.
class NcnnVulkanQnkExpertOperator
{
private:
    friend class VulkanMxfp4ExpertBackend;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanQnkExpertOperator> create_with_allocator(
        const TensorData& gate_up,
        const TensorData* gate_up_bias,
        const TensorData& down,
        const TensorData* down_bias,
        float activation_limit,
        uint32_t vulkan_device_index,
        ncnn::VkAllocator* weight_allocator,
        ExpertActivation activation,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] static bool forward_batch(
        std::span<const NcnnVulkanQnkExpertOperator*> experts,
        std::span<const ActivationBuffer*> inputs,
        std::span<ActivationBuffer*> outputs);
    class Implementation;

    NcnnVulkanQnkExpertOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanQnkExpertOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanQnkExpertOperator> create(
        const TensorData& gate_up,
        const TensorData* gate_up_bias,
        const TensorData& down,
        const TensorData* down_bias,
        float activation_limit,
        uint32_t vulkan_device_index,
        ExpertActivation activation,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
};

struct NcnnVulkanMxfp4DeviceMatrixView
{
    uint32_t output_columns = 0;
    uint32_t input_columns = 0;
    uint64_t packed_bytes = 0;
    uint64_t scales_bytes = 0;
    size_t packed_offset = 0;
    size_t scales_offset = 0;
};

class NcnnVulkanMxfp4ExpertOperator
{
private:
    friend class VulkanMxfp4ExpertBackend;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                              const TensorData& down, const TensorData* down_bias,
                                                                                              float activation_limit, uint32_t vulkan_device_index,
                                                                                              ncnn::VkAllocator* weight_allocator,
                                                                                              ExpertActivation activation,
                                                                                              const NcnnVulkanContextInstancePtr& context_instance,
                                                                                              uint64_t optimization_flags);

    class Implementation;

    NcnnVulkanMxfp4ExpertOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanMxfp4ExpertOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> create(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                               const TensorData& down, const TensorData* down_bias, float activation_limit,
                                                                               uint32_t vulkan_device_index,
                                                                               ExpertActivation activation,
                                                                               const NcnnVulkanContextInstancePtr& context_instance,
                                                                               uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> create_from_device_storage(
        const NcnnVulkanMxfp4DeviceMatrixView& gate_up, const TensorData* gate_up_bias, const NcnnVulkanMxfp4DeviceMatrixView& down,
        const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage,
        ExpertActivation activation, const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_LINEAR_H
