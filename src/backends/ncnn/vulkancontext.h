#ifndef NCNN_MOE_VULKANCONTEXT_H
#define NCNN_MOE_VULKANCONTEXT_H

#include "vulkan.h"
#include "ncnn/moe/option.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
class ActivationBuffer;
struct TensorData;
class VulkanContext;

struct VulkanContextCacheKey
{
    uint32_t device_index = 0;
    uint64_t optimization_flags = 0;

    [[nodiscard]] bool operator==(const VulkanContextCacheKey& other) const noexcept
    {
        return device_index == other.device_index
               && optimization_flags == other.optimization_flags;
    }
};

struct VulkanContextCacheKeyHash
{
    [[nodiscard]] size_t operator()(const VulkanContextCacheKey& key) const noexcept
    {
        const size_t device_hash = std::hash<uint32_t>{}(key.device_index);
        const size_t flags_hash = std::hash<uint64_t>{}(key.optimization_flags);
        return device_hash
               ^ (flags_hash + static_cast<size_t>(0x9e3779b9u)
                  + (device_hash << 6)
                  + (device_hash >> 2));
    }
};

class AtomicRuntimeCounter
{
public:
    AtomicRuntimeCounter() noexcept = default;
    AtomicRuntimeCounter(const AtomicRuntimeCounter&) = delete;
    AtomicRuntimeCounter& operator=(const AtomicRuntimeCounter&) = delete;

    AtomicRuntimeCounter& operator++() noexcept
    {
        value.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    AtomicRuntimeCounter& operator+=(uint64_t amount) noexcept
    {
        value.fetch_add(amount, std::memory_order_relaxed);
        return *this;
    }

    [[nodiscard]] uint64_t load() const noexcept
    {
        return value.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> value{0};
};

struct VulkanRuntimeState
{
    AtomicRuntimeCounter compute_submissions;
    AtomicRuntimeCounter submit_wait_time_microseconds;
    AtomicRuntimeCounter batch_uploads;
    AtomicRuntimeCounter batch_downloads;
    AtomicRuntimeCounter auxiliary_uploads;
    AtomicRuntimeCounter auxiliary_upload_bytes;
    AtomicRuntimeCounter staging_slot_resizes;
    AtomicRuntimeCounter staging_slot_reuses;
    AtomicRuntimeCounter staging_slot_acquisitions;
    AtomicRuntimeCounter staging_slot_contentions;
    AtomicRuntimeCounter command_buffer_reuses;
    AtomicRuntimeCounter command_graph_submissions;
    AtomicRuntimeCounter command_graph_operations;
    AtomicRuntimeCounter direct_host_input_bindings;
    AtomicRuntimeCounter direct_host_output_bindings;
    AtomicRuntimeCounter attention_qkv_rope_fusions;
    AtomicRuntimeCounter attention_device_rope_fusions;
    AtomicRuntimeCounter attention_qkv_ring_fusions;
    AtomicRuntimeCounter attention_decode_sdpa_fusions;
    AtomicRuntimeCounter attention_cache_materializations;
    AtomicRuntimeCounter attention_cpu_fallbacks;
    AtomicRuntimeCounter shared_expert_swiglu_fusions;
    AtomicRuntimeCounter gated_delta_fusions;
    AtomicRuntimeCounter gated_delta_submissions;
    AtomicRuntimeCounter rms_norm_linear_fusions;
    AtomicRuntimeCounter kv_ring_appends;
    AtomicRuntimeCounter kv_ring_resizes;
    AtomicRuntimeCounter kv_ring_wrapped_views;
    AtomicRuntimeCounter kv_cache_promotions;
    AtomicRuntimeCounter kv_cache_promotion_bytes;
    AtomicRuntimeCounter bfloat16_cooperative_matrix_dispatches;
    AtomicRuntimeCounter command_dispatches;
    AtomicRuntimeCounter command_pipeline_binds;
    AtomicRuntimeCounter command_redundant_pipeline_binds;
    AtomicRuntimeCounter command_descriptor_bindings;
    AtomicRuntimeCounter command_push_constant_updates;
    AtomicRuntimeCounter command_resource_barrier_calls;
    AtomicRuntimeCounter command_buffer_resource_barriers;
    AtomicRuntimeCounter command_image_resource_barriers;
    AtomicRuntimeCounter dispatches;
    AtomicRuntimeCounter attention_blocks;

    [[nodiscard]] VulkanStatistics snapshot() const noexcept;
};

int submit_compute_and_wait(ncnn::VkCompute& command,
                            VulkanRuntimeState& runtime_state);

bool prepare_staging_batch(ncnn::VkMat& buffer,
                           size_t rows,
                           uint32_t columns,
                           ncnn::VkAllocator* allocator,
                           VulkanRuntimeState& runtime_state,
                           size_t element_size = sizeof(float));

bool prepare_staging_tensor(ncnn::VkMat& buffer,
                            int width,
                            int height,
                            int channels,
                            size_t element_size,
                            ncnn::VkAllocator* allocator,
                            VulkanRuntimeState& runtime_state);

bool record_mapped_upload(ncnn::VkMat& staging,
                          ncnn::VkMat& destination,
                          ncnn::VkCompute& command,
                          const ncnn::Option& option);

bool record_mapped_activation_upload(ncnn::VkMat& staging,
                                     ncnn::VkMat& destination,
                                     ncnn::VkCompute& command,
                                     ncnn::VulkanDevice* device,
                                     const ncnn::Option& option,
                                     DType source_dtype = DType::Float32);

[[nodiscard]] inline size_t vulkan_activation_storage_variant(const ncnn::Option& option) noexcept
{
    return option.use_fp16_storage
               ? 1
           : option.use_bf16_storage ? 2
                                     : 0;
}

[[nodiscard]] inline size_t vulkan_activation_element_size(const ncnn::Option& option) noexcept
{
    return vulkan_activation_storage_variant(option) == 0
               ? sizeof(float)
               : sizeof(uint16_t);
}

[[nodiscard]] inline ncnn::VkMat row_view(const ncnn::VkMat& source,
                                          size_t first_row,
                                          size_t rows)
{
    if (source.empty() || source.dims != 2
        || first_row + rows > static_cast<size_t>(source.h))
    {
        return {};
    }
    ncnn::VkMat view = source;
    view.h = static_cast<int>(rows);
    view.offset += first_row * static_cast<size_t>(source.w) * source.elemsize;
    return view;
}

ncnn::VkMat bind_direct_host_input(ncnn::VkMat& staging,
                                   VulkanRuntimeState& runtime_state);

ncnn::VkMat prepare_direct_host_output(ncnn::VkMat& staging,
                                       VulkanRuntimeState& runtime_state);

bool fill_staging_upload(const ActivationBuffer& input,
                         ncnn::VkMat& staging,
                         ncnn::VkAllocator* allocator,
                         VulkanRuntimeState& runtime_state);

bool fill_staging_values(const void* source,
                         size_t count,
                         size_t element_size,
                         ncnn::VkMat& staging,
                         ncnn::VkAllocator* allocator,
                         VulkanRuntimeState& runtime_state);

bool record_prepared_staging_upload(const ncnn::VkMat& staging,
                                    size_t rows,
                                    ncnn::VkMat& destination,
                                    ncnn::VkCompute& command,
                                    ncnn::VulkanDevice* device,
                                    const ncnn::Option& option,
                                    DType source_dtype = DType::Float32);

bool record_prepared_staging_download(const ncnn::VkMat& source,
                                      size_t rows,
                                      uint32_t columns,
                                      ncnn::VkMat& staging,
                                      ncnn::VkCompute& command,
                                      const ncnn::Option& option);

bool record_prepared_activation_staging_download(const ncnn::VkMat& source,
                                                 size_t rows,
                                                 uint32_t columns,
                                                 ncnn::VkMat& staging,
                                                 ncnn::VkCompute& command,
                                                 ncnn::VulkanDevice* device,
                                                 const ncnn::Option& option,
                                                 DType output_dtype = DType::Float32);

bool copy_staging_to_cpu_batch(ncnn::VkMat& staging,
                               ActivationBuffer& output);

// Batch scatter requires distinct outputs; only same-index input/output alias is allowed.
bool copy_staging_to_cpu_batches(ncnn::VkMat& staging,
                                 std::span<const ActivationBuffer*> inputs,
                                 std::span<ActivationBuffer*> outputs,
                                 uint32_t columns);

bool prepare_float_tensor_upload(const TensorData& source,
                                 ncnn::Mat& destination);

class VulkanRuntime
{
    friend class VulkanContext;
    friend VulkanStatistics get_vulkan_statistics(const VulkanRuntimePtr& vulkan_runtime) noexcept;

    mutable std::mutex initialization_mutex;
    bool initialization_attempted = false;
    bool instance_ready = false;
    VulkanRuntimeState state;
    mutable std::mutex context_mutex;
    std::unordered_map<
        VulkanContextCacheKey,
        std::weak_ptr<VulkanContext>,
        VulkanContextCacheKeyHash>
        contexts;
};
#else
class VulkanRuntime
{
};
#endif

#if NCNN_MOE_WITH_VULKAN
struct VulkanTransferSlot
{
    std::mutex mutex;
    ncnn::VkAllocator* staging_allocator = nullptr;
    ncnn::VkCompute* command = nullptr;
    bool command_used = false;
    ncnn::VkMat upload;
    ncnn::VkMat download;
    ncnn::VkMat expert_slots;
    ncnn::VkMat expert_row_offsets;
    ncnn::VkMat route_offsets;
    ncnn::VkMat route_rows;
    ncnn::VkMat route_weights;
    ncnn::VkMat rope_cosine;
    ncnn::VkMat rope_sine;
    ncnn::VkMat attention_mask;
    ncnn::VkMat attention_cache_key;
    ncnn::VkMat attention_cache_value;
};

class VulkanTransferLease
{
public:
    VulkanTransferLease(VulkanTransferSlot& slot, std::unique_lock<std::mutex> _lock);

    VulkanTransferLease(const VulkanTransferLease&) = delete;
    VulkanTransferLease& operator=(const VulkanTransferLease&) = delete;
    VulkanTransferLease(VulkanTransferLease&&) noexcept = default;
    VulkanTransferLease& operator=(VulkanTransferLease&&) noexcept = default;

    [[nodiscard]] VulkanTransferSlot& slot() const noexcept
    {
        return *transfer_slot;
    }

private:
    VulkanTransferSlot* transfer_slot = nullptr;
    std::unique_lock<std::mutex> lock;
};

class VulkanContext
{
public:
    struct ShaderCacheKey
    {
        const char* source = nullptr;
        uint64_t variant = 0;

        [[nodiscard]] bool operator==(const ShaderCacheKey& other) const noexcept
        {
            return source == other.source && variant == other.variant;
        }
    };

    struct ShaderCacheKeyHash
    {
        [[nodiscard]] size_t operator()(const ShaderCacheKey& key) const noexcept
        {
            const size_t source_hash = std::hash<const void*>{}(static_cast<const void*>(key.source));
            const size_t variant_hash = std::hash<uint64_t>{}(key.variant);
            return source_hash
                   ^ (variant_hash + static_cast<size_t>(0x9e3779b9u)
                      + (source_hash << 6)
                      + (source_hash >> 2));
        }
    };

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    ~VulkanContext();

    [[nodiscard]] static std::shared_ptr<VulkanContext> acquire(uint32_t requested_device_index,
                                                                const VulkanRuntimePtr& vulkan_runtime,
                                                                uint64_t optimization_flags);

    [[nodiscard]] ncnn::VulkanDevice* device() const noexcept
    {
        return vkdev;
    }

    [[nodiscard]] bool support_direct_host_buffer(size_t size, DType dtype) const noexcept
    {
        // Direct bindings require FP32 host storage; staging handles other casts.
        // Keep large transfers and discrete-GPU buffers in device-local storage.
        // The command recorder supplies the shader-write -> host-read barrier.
        constexpr size_t maximum_direct_host_size = 64 * 1024;
        if (dtype != DType::Float32 || size > maximum_direct_host_size)
            return false;
        return vkdev && vkdev->info.type() > 0;
    }

    [[nodiscard]] ncnn::VkAllocator* blob_allocator() const noexcept
    {
        return blob_alloc;
    }

    [[nodiscard]] ncnn::VkAllocator* staging_allocator() const noexcept
    {
        return staging_alloc;
    }

    [[nodiscard]] uint64_t optimization_flags() const noexcept
    {
        return flags;
    }

    [[nodiscard]] const VulkanRuntimePtr& runtime() const noexcept
    {
        return vulkan_runtime;
    }

    [[nodiscard]] uint32_t command_optimization_flags() const noexcept
    {
        return command_flags;
    }

    [[nodiscard]] VulkanRuntimeState& runtime_state() noexcept
    {
        return vulkan_runtime->state;
    }

    [[nodiscard]] std::shared_ptr<const std::vector<uint32_t>> shader_binary(const char* source,
                                                                             int source_length,
                                                                             const ncnn::Option& option,
                                                                             uint64_t variant);

    [[nodiscard]] std::shared_ptr<ncnn::Pipeline> find_pipeline(const char* source,
                                                                uint64_t variant) const;

    void cache_pipeline(const char* source,
                        uint64_t variant,
                        const std::shared_ptr<ncnn::Pipeline>& pipeline);

    [[nodiscard]] std::mutex& command_mutex() noexcept
    {
        return command_lock;
    }

    [[nodiscard]] VulkanTransferLease acquire_transfer_slot();

private:
    explicit VulkanContext(ncnn::VulkanDevice* device,
                           VulkanRuntimePtr _vulkan_runtime,
                           uint64_t optimization_flags,
                           uint32_t command_optimization_flags);

    // ncnn owns Vulkan teardown through atexit; transfer commands share that lifetime.

    ncnn::VulkanDevice* vkdev = nullptr;
    VulkanRuntimePtr vulkan_runtime;
    uint64_t flags = OptimizationDefaultFlags;
    uint32_t command_flags = 0;
    ncnn::VkAllocator* blob_alloc = nullptr;
    ncnn::VkAllocator* staging_alloc = nullptr;
    std::mutex command_lock;
    // Staging slots require independent allocators while commands are in flight.
    std::array<VulkanTransferSlot, 2> transfer_slots;
    std::atomic<size_t> next_transfer_slot{0};
    mutable std::mutex pipeline_cache_mutex;
    std::unordered_map<
        ShaderCacheKey,
        std::shared_ptr<const std::vector<uint32_t>>,
        ShaderCacheKeyHash>
        shader_binaries;
    std::unordered_map<
        ShaderCacheKey,
        std::weak_ptr<ncnn::Pipeline>,
        ShaderCacheKeyHash>
        pipelines;
};

// Weight admissions are produced by one background worker, but each Expert
// used to create and wait for a separate transfer command. Keep one transfer
// command and its staging allocator alive for a bounded group of MXFP4 Experts
// so the device sees one submission for the whole group. The caller owns the
// context command lock while the batch is recording; this keeps ncnn's
// allocator and command domain serialized with foreground execution.
class VulkanWeightUploadBatch
{
public:
    explicit VulkanWeightUploadBatch(const std::shared_ptr<VulkanContext>& _context);

    VulkanWeightUploadBatch(const VulkanWeightUploadBatch&) = delete;
    VulkanWeightUploadBatch& operator=(const VulkanWeightUploadBatch&) = delete;

    [[nodiscard]] bool record(const ncnn::Mat& source,
                              ncnn::VkMat& destination,
                              const ncnn::Option& option,
                              ncnn::VkAllocator* weight_allocator);

    [[nodiscard]] bool submit();

private:
    std::shared_ptr<VulkanContext> context;
    // Keep this before cmd so the allocator outlives VkTransfer's retained
    // staging VkMats during destruction.
    std::unique_ptr<ncnn::VkWeightStagingAllocator> staging_allocator;
    ncnn::VkTransfer cmd;
    std::unique_lock<std::mutex> command_lock;
};
#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKANCONTEXT_H
