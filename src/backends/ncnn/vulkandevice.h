#ifndef NCNN_MOE_NCNN_VULKANDEVICE_H
#define NCNN_MOE_NCNN_VULKANDEVICE_H

#include "vulkancontext.h"
#include "ncnn/moe/option.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
class NcnnVulkanContext;

struct NcnnVulkanContextCacheKey
{
    uint32_t device_index = 0;
    uint64_t optimization_flags = 0;

    [[nodiscard]] bool operator==(
        const NcnnVulkanContextCacheKey& other) const noexcept
    {
        return device_index == other.device_index
               && optimization_flags == other.optimization_flags;
    }
};

struct NcnnVulkanContextCacheKeyHash
{
    [[nodiscard]] size_t operator()(
        const NcnnVulkanContextCacheKey& key) const noexcept
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

struct NcnnVulkanRuntimeState
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
    AtomicRuntimeCounter attention_qkv_rope_pipeline_failures;
    AtomicRuntimeCounter attention_qkv_rope_shape_failures;
    AtomicRuntimeCounter attention_qkv_rope_source_failures;
    AtomicRuntimeCounter attention_qkv_rope_norm_failures;
    AtomicRuntimeCounter attention_qkv_rope_ring_failures;
    AtomicRuntimeCounter attention_qkv_rope_allocation_failures;
    AtomicRuntimeCounter attention_precondition_failures;
    AtomicRuntimeCounter attention_staging_failures;
    AtomicRuntimeCounter attention_norm_failures;
    AtomicRuntimeCounter attention_qkv_failures;
    AtomicRuntimeCounter attention_cache_failures;
    AtomicRuntimeCounter attention_sdpa_failures;
    AtomicRuntimeCounter attention_projection_failures;
    AtomicRuntimeCounter attention_output_failures;
    AtomicRuntimeCounter attention_submit_failures;
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

    [[nodiscard]] NcnnVulkanExecutionSnapshot snapshot() const noexcept;
};

int submit_compute_and_wait(
    ncnn::VkCompute& command,
    NcnnVulkanRuntimeState& runtime_state);

class NcnnVulkanContextInstance
{
    friend class NcnnVulkanContext;
    friend NcnnVulkanExecutionSnapshot get_vulkan_execution_snapshot(
        const NcnnVulkanContextInstancePtr& context_instance) noexcept;

    mutable std::mutex initialization_mutex;
    bool initialization_attempted = false;
    bool instance_ready = false;
    NcnnVulkanRuntimeState state;
    mutable std::mutex context_mutex;
    std::unordered_map<
        NcnnVulkanContextCacheKey,
        std::weak_ptr<NcnnVulkanContext>,
        NcnnVulkanContextCacheKeyHash>
        contexts;
};
#else
class NcnnVulkanContextInstance
{
};
#endif

#if NCNN_MOE_WITH_VULKAN
struct NcnnVulkanTransferSlot
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

class NcnnVulkanTransferLease
{
public:
    NcnnVulkanTransferLease(NcnnVulkanTransferSlot& slot, std::unique_lock<std::mutex> _lock);

    NcnnVulkanTransferLease(const NcnnVulkanTransferLease&) = delete;
    NcnnVulkanTransferLease& operator=(const NcnnVulkanTransferLease&) = delete;
    NcnnVulkanTransferLease(NcnnVulkanTransferLease&&) noexcept = default;
    NcnnVulkanTransferLease& operator=(NcnnVulkanTransferLease&&) noexcept = default;

    [[nodiscard]] NcnnVulkanTransferSlot& slot() const noexcept
    {
        return *transfer_slot;
    }

private:
    NcnnVulkanTransferSlot* transfer_slot = nullptr;
    std::unique_lock<std::mutex> lock;
};

class NcnnVulkanContext
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

    NcnnVulkanContext(const NcnnVulkanContext&) = delete;
    NcnnVulkanContext& operator=(const NcnnVulkanContext&) = delete;

    ~NcnnVulkanContext();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanContext> acquire(
        uint32_t requested_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags);

    [[nodiscard]] ncnn::VulkanDevice* device() const noexcept
    {
        return vkdev;
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

    [[nodiscard]] const NcnnVulkanContextInstancePtr& instance() const noexcept
    {
        return context_instance;
    }

    [[nodiscard]] uint32_t command_optimization_flags() const noexcept
    {
        return command_flags;
    }

    [[nodiscard]] NcnnVulkanRuntimeState& runtime_state() noexcept
    {
        return context_instance->state;
    }

    [[nodiscard]] std::shared_ptr<const std::vector<uint32_t>> shader_binary(
        const char* source,
        int source_length,
        const ncnn::Option& option,
        uint64_t variant);

    [[nodiscard]] std::shared_ptr<ncnn::Pipeline> find_pipeline(
        const char* source,
        uint64_t variant) const;

    void cache_pipeline(
        const char* source,
        uint64_t variant,
        const std::shared_ptr<ncnn::Pipeline>& pipeline);

    [[nodiscard]] std::mutex& command_mutex() noexcept
    {
        return command_lock;
    }

    [[nodiscard]] NcnnVulkanTransferLease acquire_transfer_slot();

private:
    explicit NcnnVulkanContext(
        ncnn::VulkanDevice* device,
        NcnnVulkanContextInstancePtr _context_instance,
        uint64_t optimization_flags,
        uint32_t command_optimization_flags);

    // ncnn owns Vulkan teardown through atexit; transfer commands share that lifetime.

    ncnn::VulkanDevice* vkdev = nullptr;
    NcnnVulkanContextInstancePtr context_instance;
    uint64_t flags = OptimizationDefaultFlags;
    uint32_t command_flags = 0;
    ncnn::VkAllocator* blob_alloc = nullptr;
    ncnn::VkAllocator* staging_alloc = nullptr;
    std::mutex command_lock;
    // Staging slots require independent allocators while commands are in flight.
    std::array<NcnnVulkanTransferSlot, 2> transfer_slots;
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
// used to create and wait for a separate transfer command.  Keep one transfer
// command and its staging allocators alive for a bounded group of MXFP4
// Experts so the device sees one submission for the whole group.  The caller
// owns the context command lock while the batch is recording; this keeps
// ncnn's allocators and command domain serialized with foreground execution.
class NcnnVulkanWeightUploadBatch
{
public:
    explicit NcnnVulkanWeightUploadBatch(const std::shared_ptr<NcnnVulkanContext>& _context);

    NcnnVulkanWeightUploadBatch(const NcnnVulkanWeightUploadBatch&) = delete;
    NcnnVulkanWeightUploadBatch& operator=(const NcnnVulkanWeightUploadBatch&) = delete;

    [[nodiscard]] bool record(
        const ncnn::Mat& source,
        ncnn::VkMat& destination,
        const ncnn::Option& option,
        ncnn::VkAllocator* weight_allocator);

    [[nodiscard]] bool submit();

private:
    std::shared_ptr<NcnnVulkanContext> context;
    std::vector<std::unique_ptr<ncnn::VkWeightStagingAllocator>> staging_allocators;
    ncnn::VkTransfer cmd;
    std::unique_lock<std::mutex> command_lock;
};
#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_VULKANDEVICE_H
