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

inline constexpr uint64_t vulkan_weight_block_size = 64 * 1024;

[[nodiscard]] inline uint64_t vulkan_kv_size(DType dtype, uint32_t device_index) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (dtype == DType::BFloat16)
    {
        const int count = ncnn::get_gpu_count();
        if (count <= 0 || device_index >= static_cast<uint32_t>(count)) return UINT64_MAX;
        if (ncnn::get_gpu_info(static_cast<int>(device_index)).support_bf16_storage()) return sizeof(uint16_t);
    }
#else
    (void)dtype;
    (void)device_index;
#endif
    return sizeof(float);
}

#if NCNN_MOE_WITH_VULKAN
class ActivationBuffer;
struct TensorData;
class VulkanContext;
class VulkanIndependentWeightTransfer;
class VulkanExpertTransferPool;
class VulkanGraphAllocator;

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

class AtomicCounter
{
public:
    AtomicCounter() noexcept = default;
    AtomicCounter(const AtomicCounter&) = delete;
    AtomicCounter& operator=(const AtomicCounter&) = delete;

    AtomicCounter& operator++() noexcept
    {
        value.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    AtomicCounter& operator+=(uint64_t amount) noexcept
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
    AtomicCounter compute_submissions;
    AtomicCounter batch_uploads;
    AtomicCounter batch_downloads;
    AtomicCounter weight_upload_bytes;
    AtomicCounter weight_transfer_submissions;
    AtomicCounter weight_visibility_submissions;
    AtomicCounter weight_transfer_fence_waits;
    AtomicCounter weight_transfer_wait_microseconds;
    AtomicCounter resident_graph_submissions;
    AtomicCounter resident_graph_activation_upload_bytes;
    AtomicCounter resident_graph_activation_download_bytes;
    AtomicCounter resident_graph_route_download_bytes;
    AtomicCounter attention_qkv_rope_fusions;
    AtomicCounter attention_device_rope_fusions;
    AtomicCounter attention_qkv_ring_fusions;
    AtomicCounter attention_decode_sdpa_fusions;
    AtomicCounter attention_cache_materializations;
    AtomicCounter attention_cpu_fallbacks;
    AtomicCounter shared_expert_swiglu_fusions;
    AtomicCounter gated_delta_fusions;
    AtomicCounter gated_delta_submissions;
    AtomicCounter rms_norm_linear_fusions;
    AtomicCounter kv_ring_appends;
    AtomicCounter kv_ring_resizes;
    AtomicCounter kv_ring_wrapped_views;
    AtomicCounter bfloat16_cooperative_matrix_dispatches;
    AtomicCounter dispatches;
    AtomicCounter int4_dispatches;
    AtomicCounter attention_blocks;
};

// Wait an already submitted command without submitting it again. Return only
// after its fence completes or the owning device is confirmed lost; callers
// keep every command resource/owner alive through this call.
int wait_submitted_compute(ncnn::VkCompute& command, const ncnn::VulkanDevice* probe_device,
                           VkQueue borrowed_queue = VK_NULL_HANDLE);

// The device must belong to this command and stay alive through the call. It
// is used only to detect device loss after a failed fence wait. Without one,
// the original wait keeps retrying until completion rather than release owners.
int submit_compute_and_wait(ncnn::VkCompute& command,
                            const ncnn::VulkanDevice* probe_device);

bool prepare_staging_batch(ncnn::VkMat& buffer,
                           size_t rows,
                           uint32_t columns,
                           ncnn::VkAllocator* allocator,
                           size_t element_size = sizeof(float));

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
    if (source.empty() || source.dims != 2 || rows == 0
#if NCNN_BATCH
        || source.n != 1
#endif
        || first_row > static_cast<size_t>(source.h)
        || rows > static_cast<size_t>(source.h) - first_row)
    {
        return {};
    }
    ncnn::VkMat view = source;
    view.h = static_cast<int>(rows);
    view.offset += first_row * static_cast<size_t>(source.w) * source.elemsize;
    view.cstep = static_cast<size_t>(source.w) * rows;
#if NCNN_BATCH
    view.nstep = view.cstep;
#endif
    return view;
}

ncnn::VkMat bind_direct_host_input(ncnn::VkMat& staging);

bool fill_staging_upload(const ActivationBuffer& input,
                         ncnn::VkMat& staging,
                         ncnn::VkAllocator* allocator);

bool fill_staging_values(const void* source,
                         size_t count,
                         size_t element_size,
                         ncnn::VkMat& staging,
                         ncnn::VkAllocator* allocator);

bool record_prepared_staging_upload(const ncnn::VkMat& staging,
                                    size_t rows,
                                    ncnn::VkMat& destination,
                                    ncnn::VkCompute& command,
                                    ncnn::VulkanDevice* device,
                                    const ncnn::Option& option,
                                    DType source_dtype = DType::Float32);

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
public:
    VulkanRuntime() noexcept = default;

private:
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
public:
    VulkanRuntime() noexcept = default;
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
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    ~VulkanContext();

    [[nodiscard]] static std::shared_ptr<VulkanContext> acquire(uint32_t requested_device_index,
                                                                const VulkanRuntimePtr& vulkan_runtime,
                                                                uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<VulkanContext> find(uint32_t requested_device_index,
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

    [[nodiscard]] bool reserve_graph_compute(uint64_t bytes);
    [[nodiscard]] VulkanGraphArenaSize graph_compute_physical_size() const;

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
    [[nodiscard]] VulkanExpertTransferPool& expert_transfer_pool() noexcept
    { return *weight_transfer_pool; }

private:
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

    friend class VulkanComputeBatch;
    // Reused under command_mutex; each batch drains before releasing the lock.
    std::vector<std::unique_ptr<ncnn::VkCompute>> ordered_commands;
    explicit VulkanContext(ncnn::VulkanDevice* device,
                           VulkanRuntimePtr _vulkan_runtime,
                           uint64_t optimization_flags);

    // Release dependent commands and tensors before freeing their allocator.
    void release_resources() noexcept;

    ncnn::VulkanDevice* vkdev = nullptr;
    VulkanRuntimePtr vulkan_runtime;
    uint64_t flags = OptimizationDefaultFlags;
    ncnn::VkAllocator* blob_alloc = nullptr;
    std::unique_ptr<VulkanGraphAllocator> graph_allocator;
    ncnn::VkAllocator* staging_alloc = nullptr;
    std::mutex command_lock;
    // Staging slots require independent allocators while commands are in flight.
    std::array<VulkanTransferSlot, 2> transfer_slots;
    std::atomic<size_t> next_transfer_slot{0};
    std::unique_ptr<VulkanExpertTransferPool> weight_transfer_pool;
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

// Ordered asynchronous submissions within one planned resource lifetime.
// Caller holds context.command_mutex() and all operand/pipeline owners until
// reset completes. Each boundary preserves the VkMat producer access state.
class VulkanComputeBatch
{
public:
    explicit VulkanComputeBatch(VulkanContext& context);
    ~VulkanComputeBatch();
    VulkanComputeBatch(const VulkanComputeBatch&) = delete;
    VulkanComputeBatch& operator=(const VulkanComputeBatch&) = delete;
    [[nodiscard]] ncnn::VkCompute* next();
    [[nodiscard]] int submit();
    [[nodiscard]] int reset() noexcept;
    [[nodiscard]] size_t submission_count() const noexcept
    { return submitted; }

private:
    VulkanContext& context;
    VkQueue queue = VK_NULL_HANDLE;
    size_t recorded = 0, submitted = 0;
    int completion_error = 0;
};

// Two transactional host-visible output slots per execution workspace. A
// published ActivationBuffer leases its slot; only unleased slots are reused.
// Callers plan the payload bound before recording and serialize preparation
// under the context command mutex. Each slot owns its allocator so a published
// output may safely outlive the workspace or be released on another thread.
class VulkanHostOutput
{
public:
    VulkanHostOutput();
    ~VulkanHostOutput();
    VulkanHostOutput(const VulkanHostOutput&) = delete;
    VulkanHostOutput& operator=(const VulkanHostOutput&) = delete;
    [[nodiscard]] bool prepare(const std::shared_ptr<VulkanContext>& context,
                               size_t rows, uint32_t columns, uint64_t slot_capacity_limit,
                               ncnn::VkMat& destination, ActivationBuffer& pending);
    [[nodiscard]] static bool complete(ncnn::VkMat& destination, const ActivationBuffer& pending);
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
    [[nodiscard]] uint64_t planned_capacity() const noexcept
    { return slot_capacity; }

private:
    struct Storage;
    std::array<std::shared_ptr<Storage>, 2> slots;
    uint64_t slot_capacity = 0;
};

// Caller serializes allocations and frees in the command domain. Only free
// buffers are cached; retained upload commands keep their active buffers alive.
class VulkanUploadStagingAllocator final : public ncnn::VkStagingAllocator
{
public:
    VulkanUploadStagingAllocator(const ncnn::VulkanDevice* device, uint64_t cache_limit);

    using ncnn::VkStagingAllocator::fastFree;
    using ncnn::VkStagingAllocator::fastMalloc;
    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* buffer) override;
    void clear() override;

private:
    struct CachedBuffer
    {
        ncnn::VkBufferMemory* buffer = nullptr;
        uint64_t allocation_size = 0;
    };
    const uint64_t cache_limit;
    std::array<CachedBuffer, 32> cached_buffers{};
    size_t cached_count = 0;
    uint64_t cached_size = 0;
};

// Bound one group of immutable Expert uploads. Runtime-owned weight buffers
// use an isolated transfer command pool and concurrent queue-family sharing;
// staging ownership remains local to the serialized admission lane. Legacy
// allocators retain ncnn's transfer recorder and foreground command mutex.
class VulkanWeightUploadBatch
{
public:
    // A borrowed staging allocator belongs exclusively to the admission
    // transfer domain; do not reuse buffers previously owned by compute-family
    // commands. The caller serializes its allocation/free operations.
    explicit VulkanWeightUploadBatch(const std::shared_ptr<VulkanContext>& _context,
                                     ncnn::VkAllocator* borrowed_staging_allocator = nullptr,
                                     ncnn::Mat* borrowed_upload_scratch = nullptr);

    ~VulkanWeightUploadBatch();

    VulkanWeightUploadBatch(const VulkanWeightUploadBatch&) = delete;
    VulkanWeightUploadBatch& operator=(const VulkanWeightUploadBatch&) = delete;

    [[nodiscard]] bool record(const ncnn::Mat& source,
                              ncnn::VkMat& destination,
                              const ncnn::Option& option,
                              ncnn::VkAllocator* weight_allocator);

    // Pack the final GPU layout directly into mapped staging. Only runtime
    // concurrent-family weight allocators support this path. Reservations
    // cannot escape this batch; submission rejects an unrecorded reservation.
    [[nodiscard]] std::span<uint8_t> prepare_storage(size_t bytes, ncnn::VkAllocator* weight_allocator);
    [[nodiscard]] bool record_prepared_storage(ncnn::VkMat& destination,
                                               ncnn::VkAllocator* weight_allocator);
    [[nodiscard]] bool submit();

    // record_upload copies this source synchronously. Reuse at most 16 MiB
    // of ordinary host scratch between matrices; larger sources are transient.
    [[nodiscard]] ncnn::Mat host_storage(size_t bytes);

    // Failed factories retain their last pipeline owners until this batch
    // settles/destroys commands and releases any legacy context lock.
    void retain_owner(std::shared_ptr<const void> owner);

private:
    void settle_legacy_upload() noexcept;

    std::shared_ptr<VulkanContext> context;
    // Preallocated before native recording/submission, so an OOM recovery
    // never allocates while retaining pending command/staging resources.
    std::vector<VkQueue> legacy_queue_scratch;
    // Keep this before cmd so the allocator outlives VkTransfer's retained
    // staging VkMats during destruction.
    std::unique_ptr<ncnn::VkWeightStagingAllocator> staging_allocator;
    ncnn::VkAllocator* borrowed_staging_allocator = nullptr;
    ncnn::Mat* borrowed_upload_scratch = nullptr;
    ncnn::Mat upload_scratch;
    std::vector<std::shared_ptr<const void>> retained_owners;
    // Destroy cmd (and return its staging buffers) before unlocking context.
    std::unique_lock<std::mutex> command_lock;
    std::unique_ptr<ncnn::VkTransfer> cmd;
    std::unique_ptr<VulkanIndependentWeightTransfer> independent_command;
    bool failed = false;
    bool submitted = false;
};
#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKANCONTEXT_H
