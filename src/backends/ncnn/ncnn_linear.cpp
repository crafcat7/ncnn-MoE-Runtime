#include "ncnn_linear.h"

#include "ncnn/moe/runtime.h"
#include "engine/expert_backend.h"
#include "engine/cpu_features.h"
#include "engine/cpu_session_state.h"
#include "kernels/cpu_float8.h"
#include "kernels/cpu_ops.h"
#include "kernels/cpu_state_cache.h"
#include "storage/expert_victim_cache.h"
#include "ncnn_attention.h"

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <list>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#endif

namespace ncnn {
namespace moe {

struct TransparentStringHash
{
    using is_transparent = void;

    [[nodiscard]] size_t operator()(std::string_view value) const noexcept
    {
        return std::hash<std::string_view>{}(value);
    }
};

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
        const size_t device_hash =
            std::hash<uint32_t>{}(key.device_index);
        const size_t flags_hash =
            std::hash<uint64_t>{}(key.optimization_flags);
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
        value_.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    AtomicRuntimeCounter& operator+=(uint64_t amount) noexcept
    {
        value_.fetch_add(amount, std::memory_order_relaxed);
        return *this;
    }

    [[nodiscard]] uint64_t load() const noexcept
    {
        return value_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> value_{0};
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

    [[nodiscard]] NcnnVulkanExecutionSnapshot snapshot() const noexcept
    {
        NcnnVulkanExecutionSnapshot result;
        result.dispatches = dispatches.load();
        result.attention_blocks = attention_blocks.load();
        result.counters.compute_submissions = compute_submissions.load();
        result.counters.submit_wait_time_microseconds = submit_wait_time_microseconds.load();
        result.counters.batch_uploads = batch_uploads.load();
        result.counters.batch_downloads = batch_downloads.load();
        result.counters.auxiliary_uploads = auxiliary_uploads.load();
        result.counters.auxiliary_upload_bytes = auxiliary_upload_bytes.load();
        result.counters.staging_slot_resizes = staging_slot_resizes.load();
        result.counters.staging_slot_reuses = staging_slot_reuses.load();
        result.counters.staging_slot_acquisitions = staging_slot_acquisitions.load();
        result.counters.staging_slot_contentions = staging_slot_contentions.load();
        result.counters.command_buffer_reuses = command_buffer_reuses.load();
        result.counters.command_graph_submissions = command_graph_submissions.load();
        result.counters.command_graph_operations = command_graph_operations.load();
        result.counters.direct_host_input_bindings = direct_host_input_bindings.load();
        result.counters.direct_host_output_bindings = direct_host_output_bindings.load();
        result.counters.attention_qkv_rope_fusions = attention_qkv_rope_fusions.load();
        result.counters.attention_device_rope_fusions = attention_device_rope_fusions.load();
        result.counters.attention_qkv_ring_fusions = attention_qkv_ring_fusions.load();
        result.counters.attention_qkv_rope_pipeline_failures = attention_qkv_rope_pipeline_failures.load();
        result.counters.attention_qkv_rope_shape_failures = attention_qkv_rope_shape_failures.load();
        result.counters.attention_qkv_rope_source_failures = attention_qkv_rope_source_failures.load();
        result.counters.attention_qkv_rope_norm_failures = attention_qkv_rope_norm_failures.load();
        result.counters.attention_qkv_rope_ring_failures = attention_qkv_rope_ring_failures.load();
        result.counters.attention_qkv_rope_allocation_failures = attention_qkv_rope_allocation_failures.load();
        result.counters.attention_precondition_failures = attention_precondition_failures.load();
        result.counters.attention_staging_failures = attention_staging_failures.load();
        result.counters.attention_norm_failures = attention_norm_failures.load();
        result.counters.attention_qkv_failures = attention_qkv_failures.load();
        result.counters.attention_cache_failures = attention_cache_failures.load();
        result.counters.attention_sdpa_failures = attention_sdpa_failures.load();
        result.counters.attention_projection_failures = attention_projection_failures.load();
        result.counters.attention_output_failures = attention_output_failures.load();
        result.counters.attention_submit_failures = attention_submit_failures.load();
        result.counters.attention_decode_sdpa_fusions = attention_decode_sdpa_fusions.load();
        result.counters.attention_cache_materializations = attention_cache_materializations.load();
        result.counters.attention_cpu_fallbacks = attention_cpu_fallbacks.load();
        result.counters.shared_expert_swiglu_fusions = shared_expert_swiglu_fusions.load();
        result.counters.gated_delta_fusions = gated_delta_fusions.load();
        result.counters.gated_delta_submissions = gated_delta_submissions.load();
        result.counters.rms_norm_linear_fusions = rms_norm_linear_fusions.load();
        result.counters.kv_ring_appends = kv_ring_appends.load();
        result.counters.kv_ring_resizes = kv_ring_resizes.load();
        result.counters.kv_ring_wrapped_views = kv_ring_wrapped_views.load();
        result.counters.kv_cache_promotions = kv_cache_promotions.load();
        result.counters.kv_cache_promotion_bytes = kv_cache_promotion_bytes.load();
        result.counters.bfloat16_cooperative_matrix_dispatches = bfloat16_cooperative_matrix_dispatches.load();
        result.counters.command_dispatches = command_dispatches.load();
        result.counters.command_pipeline_binds = command_pipeline_binds.load();
        result.counters.command_redundant_pipeline_binds = command_redundant_pipeline_binds.load();
        result.counters.command_descriptor_bindings = command_descriptor_bindings.load();
        result.counters.command_push_constant_updates = command_push_constant_updates.load();
        result.counters.command_resource_barrier_calls = command_resource_barrier_calls.load();
        result.counters.command_buffer_resource_barriers = command_buffer_resource_barriers.load();
        result.counters.command_image_resource_barriers = command_image_resource_barriers.load();
        return result;
    }
};

class NcnnVulkanContextInstance
{
    friend class NcnnVulkanContext;
    friend class NcnnLinearOperator;

    mutable std::mutex initialization_mutex_;
    bool initialization_attempted_ = false;
    bool instance_ready_ = false;
    NcnnVulkanRuntimeState runtime_state_;
    mutable std::mutex context_mutex_;
    std::unordered_map<
        NcnnVulkanContextCacheKey,
        std::weak_ptr<NcnnVulkanContext>,
        NcnnVulkanContextCacheKeyHash>
        contexts_;
};
#else
class NcnnVulkanContextInstance
{
};
#endif

NcnnVulkanContextInstancePtr create_ncnn_vulkan_context_instance()
{
    return std::make_shared<NcnnVulkanContextInstance>();
}

#if NCNN_MOE_USE_NCNN
static constexpr uint64_t max_ncnn_linear_weight_bytes = 64ull * 1024ull * 1024ull;

static bool ncnn_cpu_bfloat16_linear_enabled(uint64_t optimization_flags) noexcept
{
    const uint64_t isa = detect_cpu_isa_capabilities().flags;
    // Keep ncnn's FP32-expanded path as the conservative default on
    // unbenchmarked ISAs. The direct BF16 path is default only where its
    // AVX512 implementation has been validated end to end.
    const bool default_enabled = (isa & CpuIsaX86Avx512) == 0;
    return runtime_optimization_enabled(
               optimization_flags,
               RuntimeOptimizationNcnnCpuBfloat16Linear)
           && (default_enabled
               || runtime_optimization_enabled(
                   optimization_flags,
                   RuntimeOptimizationNcnnCpuBfloat16LinearForce));
}
#endif

#if NCNN_MOE_WITH_VULKAN
enum class VulkanBfloat16CooperativeMatrixPolicy
{
    Automatic,
    Disabled,
    Forced
};

static VulkanBfloat16CooperativeMatrixPolicy
vulkan_bfloat16_cooperative_matrix_policy(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
               optimization_flags,
               RuntimeOptimizationVulkanBfloat16CoopMatrix)
               ? VulkanBfloat16CooperativeMatrixPolicy::Automatic
               : VulkanBfloat16CooperativeMatrixPolicy::Disabled;
}

static uint32_t vulkan_command_optimization_flags(uint64_t optimization_flags) noexcept
{
    uint32_t flags = 0;
    if (runtime_optimization_enabled(
            optimization_flags,
            RuntimeOptimizationVulkanPipelineBindElision))
    {
        flags |= ncnn::VkComputeOptimizationPipelineBindElision;
    }
    if (runtime_optimization_enabled(
            optimization_flags,
            RuntimeOptimizationVulkanReadonlyBindings))
    {
        flags |= ncnn::VkComputeOptimizationReadonlyBindings;
    }
    if (runtime_optimization_enabled(
            optimization_flags,
            RuntimeOptimizationVulkanBatchBufferBarriers))
    {
        flags |= ncnn::VkComputeOptimizationBatchBufferBarriers;
    }
    if (runtime_optimization_enabled(
            optimization_flags,
            RuntimeOptimizationVulkanStackDescriptorPayload))
    {
        flags |= ncnn::VkComputeOptimizationStackDescriptorPayload;
    }
    return flags;
}
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
    NcnnVulkanTransferLease(NcnnVulkanTransferSlot& slot, std::unique_lock<std::mutex> lock)
        : slot_(&slot), lock_(std::move(lock))
    {
    }

    NcnnVulkanTransferLease(const NcnnVulkanTransferLease&) = delete;
    NcnnVulkanTransferLease& operator=(const NcnnVulkanTransferLease&) = delete;
    NcnnVulkanTransferLease(NcnnVulkanTransferLease&&) noexcept = default;
    NcnnVulkanTransferLease& operator=(NcnnVulkanTransferLease&&) noexcept = default;

    [[nodiscard]] NcnnVulkanTransferSlot& slot() const noexcept
    {
        return *slot_;
    }

private:
    NcnnVulkanTransferSlot* slot_ = nullptr;
    std::unique_lock<std::mutex> lock_;
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
            const size_t source_hash =
                std::hash<const void*>{}(static_cast<const void*>(key.source));
            const size_t variant_hash = std::hash<uint64_t>{}(key.variant);
            return source_hash
                   ^ (variant_hash + static_cast<size_t>(0x9e3779b9u)
                      + (source_hash << 6)
                      + (source_hash >> 2));
        }
    };

    NcnnVulkanContext(const NcnnVulkanContext&) = delete;
    NcnnVulkanContext& operator=(const NcnnVulkanContext&) = delete;

    ~NcnnVulkanContext()
    {
        const std::lock_guard<std::mutex> lock(command_mutex_);
        for (NcnnVulkanTransferSlot& slot : transfer_slots_)
        {
            delete slot.command;
            slot.command = nullptr;
        }
    }

    [[nodiscard]] static std::shared_ptr<NcnnVulkanContext> acquire(
        uint32_t requested_device_index,
        const NcnnVulkanContextInstancePtr& context_instance,
        uint64_t optimization_flags)
    {
        if (!context_instance)
            return {};
        {
            const std::lock_guard<std::mutex> lock(
                context_instance->initialization_mutex_);
            if (!context_instance->initialization_attempted_)
            {
                context_instance->initialization_attempted_ = true;
#if defined(__APPLE__) && defined(NCNN_MOE_MOLTENVK_LIBRARY_PATH)
                context_instance->instance_ready_ = ncnn::create_gpu_instance(NCNN_MOE_MOLTENVK_LIBRARY_PATH) == 0 && ncnn::get_gpu_count() > 0;
#else
                context_instance->instance_ready_ = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
#endif
            }
        }
        if (!context_instance->instance_ready_)
            return {};

        const uint32_t device_index = requested_device_index == automatic_vulkan_device_index ? static_cast<uint32_t>(ncnn::get_default_gpu_index()) : requested_device_index;
        if (device_index >= static_cast<uint32_t>(ncnn::get_gpu_count()))
        {
            return {};
        }
        const uint32_t command_flags = vulkan_command_optimization_flags(optimization_flags);
        ncnn::VulkanDevice* device = ncnn::get_gpu_device(static_cast<int>(device_index));
        if (!device)
            return {};
        const NcnnVulkanContextCacheKey context_key{
            device_index,
            optimization_flags};
        const std::lock_guard<std::mutex> lock(
            context_instance->context_mutex_);
        const auto existing =
            context_instance->contexts_.find(context_key);
        if (existing != context_instance->contexts_.end())
        {
            if (const std::shared_ptr<NcnnVulkanContext> context =
                    existing->second.lock())
            {
                return context;
            }
            context_instance->contexts_.erase(existing);
        }
        auto context = std::shared_ptr<NcnnVulkanContext>(new NcnnVulkanContext(
            device,
            context_instance,
            optimization_flags,
            command_flags));
        context_instance->contexts_.emplace(context_key, context);
        return context;
    }

    [[nodiscard]] ncnn::VulkanDevice* device() const noexcept
    {
        return device_;
    }

    [[nodiscard]] ncnn::VkAllocator* blob_allocator() const noexcept
    {
        return blob_allocator_;
    }

    [[nodiscard]] ncnn::VkAllocator* staging_allocator() const noexcept
    {
        return staging_allocator_;
    }

    [[nodiscard]] uint64_t optimization_flags() const noexcept
    {
        return optimization_flags_;
    }

    [[nodiscard]] const NcnnVulkanContextInstancePtr& instance() const noexcept
    {
        return context_instance_;
    }

    [[nodiscard]] uint32_t command_optimization_flags() const noexcept
    {
        return command_optimization_flags_;
    }

    [[nodiscard]] NcnnVulkanRuntimeState& runtime_state() noexcept
    {
        return context_instance_->runtime_state_;
    }

    [[nodiscard]] std::shared_ptr<const std::vector<uint32_t>> shader_binary(
        const char* source,
        int source_length,
        const ncnn::Option& option,
        uint64_t variant)
    {
        if (!source || source_length <= 0)
            return {};
        const ShaderCacheKey key{source, variant};
        const std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
        const auto cached = shader_binaries_.find(key);
        if (cached != shader_binaries_.end())
            return cached->second;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(
                source,
                source_length,
                option,
                spirv)
                != 0
            || spirv.empty())
        {
            const std::shared_ptr<const std::vector<uint32_t>> failed =
                std::make_shared<const std::vector<uint32_t>>();
            shader_binaries_.emplace(key, failed);
            return failed;
        }
        const std::shared_ptr<const std::vector<uint32_t>> binary =
            std::make_shared<const std::vector<uint32_t>>(std::move(spirv));
        shader_binaries_.emplace(key, binary);
        return binary;
    }

    [[nodiscard]] std::shared_ptr<ncnn::Pipeline> find_pipeline(
        const char* source,
        uint64_t variant) const
    {
        const std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
        const auto cached = pipelines_.find({source, variant});
        if (cached == pipelines_.end())
            return {};
        return cached->second.lock();
    }

    void cache_pipeline(
        const char* source,
        uint64_t variant,
        const std::shared_ptr<ncnn::Pipeline>& pipeline)
    {
        const std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
        pipelines_[{source, variant}] = pipeline;
    }

    [[nodiscard]] std::mutex& command_mutex() noexcept
    {
        return command_mutex_;
    }

    [[nodiscard]] NcnnVulkanTransferLease acquire_transfer_slot()
    {
        const size_t slot_index = next_transfer_slot_.fetch_add(1, std::memory_order_relaxed) % transfer_slots_.size();
        NcnnVulkanTransferSlot& slot = transfer_slots_[slot_index];
        std::unique_lock<std::mutex> lock(slot.mutex, std::try_to_lock);
        if (!lock.owns_lock())
        {
            ++runtime_state().staging_slot_contentions;
            lock.lock();
        }
        ++runtime_state().staging_slot_acquisitions;
        return NcnnVulkanTransferLease(slot, std::move(lock));
    }

private:
    explicit NcnnVulkanContext(
        ncnn::VulkanDevice* device,
        NcnnVulkanContextInstancePtr context_instance,
        uint64_t optimization_flags,
        uint32_t command_optimization_flags)
        : device_(device),
          context_instance_(std::move(context_instance)),
          optimization_flags_(optimization_flags),
          command_optimization_flags_(command_optimization_flags),
          blob_allocator_(device->acquire_blob_allocator()),
          staging_allocator_(device->acquire_staging_allocator())
    {
        for (NcnnVulkanTransferSlot& slot : transfer_slots_)
            slot.staging_allocator = device->acquire_staging_allocator();
        for (NcnnVulkanTransferSlot& slot : transfer_slots_)
        {
            slot.command = new ncnn::VkCompute(device, command_optimization_flags_);
        }
    }

    // ncnn owns Vulkan teardown through atexit; transfer commands share that lifetime.

    ncnn::VulkanDevice* device_ = nullptr;
    NcnnVulkanContextInstancePtr context_instance_;
    uint64_t optimization_flags_ = RuntimeOptimizationDefaultFlags;
    uint32_t command_optimization_flags_ = 0;
    ncnn::VkAllocator* blob_allocator_ = nullptr;
    ncnn::VkAllocator* staging_allocator_ = nullptr;
    std::mutex command_mutex_;
    // Staging slots require independent allocators while commands are in flight.
    std::array<NcnnVulkanTransferSlot, 2> transfer_slots_;
    std::atomic<size_t> next_transfer_slot_{0};
    mutable std::mutex pipeline_cache_mutex_;
    std::unordered_map<
        ShaderCacheKey,
        std::shared_ptr<const std::vector<uint32_t>>,
        ShaderCacheKeyHash> shader_binaries_;
    std::unordered_map<
        ShaderCacheKey,
        std::weak_ptr<ncnn::Pipeline>,
        ShaderCacheKeyHash> pipelines_;
};

class VulkanExpertVictimCache final : public IExpertVictimCache
{
public:
    VulkanExpertVictimCache(std::shared_ptr<NcnnVulkanContext> context, uint64_t capacity_bytes)
        : context_(std::move(context)),
          capacity_bytes_(capacity_bytes),
          // The pending queue is the admission pipeline for the executable
          // cache. Capping it at 256 MiB made a large device look full while
          // most of its weight capacity was still unused: every burst beyond
          // that cap was dropped and had to execute on the CPU. The queue
          // owns shared host-weight references, so its bound is the same
          // per-device byte budget as the cache itself.
          maximum_pending_bytes_(capacity_bytes),
          upload_staging_allocator_(context_->device()->acquire_staging_allocator()),
          worker_(&VulkanExpertVictimCache::worker_loop, this)
    {
        for (DownloadSlot& slot : download_slots_)
        {
            slot.staging_allocator = context_->device()->acquire_staging_allocator();
        }
    }

    ~VulkanExpertVictimCache() override
    {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            while (!pending_.empty())
            {
                pending_bytes_ -= pending_.front().bytes;
                pending_keys_.erase(pending_.front().key);
                pending_.pop_front();
                ++dropped_admissions_;
            }
        }
        work_available_.notify_all();
        if (worker_.joinable())
            worker_.join();
        upload_staging_ = ncnn::VkMat();
        context_->device()->reclaim_staging_allocator(upload_staging_allocator_);
        for (DownloadSlot& slot : download_slots_)
        {
            slot.staging = ncnn::VkMat();
            context_->device()->reclaim_staging_allocator(slot.staging_allocator);
        }
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down, uint32_t residency_group,
               ExpertVictimExecutionMetadata execution) override
    {
        (void)residency_group;
        if (!gate_up || !down)
            return;
        const uint64_t gate_blocks = gate_up->mxfp4_blocks.size();
        const uint64_t gate_scales = gate_up->mxfp4_scales.size();
        const uint64_t down_blocks = down->mxfp4_blocks.size();
        const uint64_t down_scales = down->mxfp4_scales.size();
        const uint64_t alignment = std::max<uint64_t>(4, context_->device()->info.buffer_offset_alignment());
        uint64_t cursor = 0;
        uint64_t gate_blocks_offset = 0;
        uint64_t gate_scales_offset = 0;
        uint64_t down_blocks_offset = 0;
        uint64_t down_scales_offset = 0;
        if (!append_segment(gate_blocks, alignment, cursor, gate_blocks_offset) || !append_segment(gate_scales, alignment, cursor, gate_scales_offset)
            || !append_segment(down_blocks, alignment, cursor, down_blocks_offset) || !append_segment(down_scales, alignment, cursor, down_scales_offset))
        {
            return;
        }
        const uint64_t bytes = cursor;
        if (bytes == 0 || bytes > capacity_bytes_ || bytes > maximum_pending_bytes_ || bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            return;
        }

        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || entries_.find(key) != entries_.end() || pending_keys_.find(key) != pending_keys_.end())
        {
            return;
        }
        while (!pending_.empty() && pending_bytes_ > maximum_pending_bytes_ - bytes)
        {
            pending_bytes_ -= pending_.front().bytes;
            pending_keys_.erase(pending_.front().key);
            pending_.pop_front();
            ++dropped_admissions_;
        }
        if (pending_bytes_ > maximum_pending_bytes_ - bytes)
        {
            ++dropped_admissions_;
            return;
        }

        PendingAdmission admission;
        admission.key = std::move(key);
        admission.gate_up = std::move(gate_up);
        admission.down = std::move(down);
        admission.bytes = bytes;
        admission.gate_blocks_offset = gate_blocks_offset;
        admission.gate_scales_offset = gate_scales_offset;
        admission.down_blocks_offset = down_blocks_offset;
        admission.down_scales_offset = down_scales_offset;
        admission.execution = execution;
        pending_bytes_ += bytes;
        pending_keys_.insert(admission.key);
        pending_.push_back(std::move(admission));
        ++admissions_;
        work_available_.notify_one();
    }

    struct DeviceOperationLease
    {
        std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
        std::shared_ptr<const void> pin;
    };

    std::optional<DeviceOperationLease> find_device_operation(std::string_view key)
    {
        std::shared_ptr<DeviceEntry> entry;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            const auto existing = entries_.find(key);
            if (existing == entries_.end() || !existing->second->execution.enabled)
            {
                return std::nullopt;
            }
            entry = existing->second;
            if (entry->operation)
            {
                return DeviceOperationLease{
                    entry->operation,
                    entry,
                };
            }
            if (entry->operation_attempted)
                return std::nullopt;
            entry->operation_attempted = true;
        }
        const NcnnVulkanMxfp4DeviceMatrixView gate_up{
            entry->gate_output_columns,
            entry->gate_input_columns,
            entry->gate_blocks,
            entry->gate_scales,
            static_cast<size_t>(entry->gate_blocks_offset),
            static_cast<size_t>(entry->gate_scales_offset),
        };
        const NcnnVulkanMxfp4DeviceMatrixView down{
            entry->down_output_columns,
            entry->down_input_columns,
            entry->down_blocks,
            entry->down_scales,
            static_cast<size_t>(entry->down_blocks_offset),
            static_cast<size_t>(entry->down_scales_offset),
        };
        auto operation = NcnnVulkanMxfp4ExpertOperator ::create_from_device_storage(
            gate_up, entry->execution.gate_up_bias, down, entry->execution.down_bias, entry->execution.activation_limit,
            static_cast<uint32_t>(context_->device()->info.device_index()), entry->data, entry->execution.activation,
            context_->instance(),
            context_->optimization_flags());
        if (!operation)
            return std::nullopt;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            entry->operation = operation;
        }
        return DeviceOperationLease{
            std::move(operation),
            std::move(entry),
        };
    }

    void touch_device_operations(std::span<const std::string_view> keys)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (std::string_view key : keys)
        {
            const auto existing = entries_.find(key);
            if (existing != entries_.end())
                existing->second->used_at = ++clock_;
        }
    }

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source) override
    {
        std::shared_ptr<DeviceEntry> entry;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            const auto existing = entries_.find(key);
            if (existing == entries_.end())
            {
                ++misses_;
                return std::nullopt;
            }
            entry = existing->second;
            entry->used_at = ++clock_;
        }

        ExpertVictimPair restored;
        const bool mapped_restore = entry->data.mapped_ptr() != nullptr;
        const auto restore_started = std::chrono::steady_clock::now();
        if (!download(*entry, gate_up_source, down_source, restored))
        {
            const uint64_t restore_microseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - restore_started).count());
            const std::lock_guard<std::mutex> lock(mutex_);
            restore_time_microseconds_ += restore_microseconds;
            ++restore_failures_;
            const auto existing = entries_.find(key);
            if (existing != entries_.end() && existing->second == entry)
            {
                resident_bytes_ -= entry->bytes;
                entries_.erase(existing);
            }
            return std::nullopt;
        }
        const uint64_t restore_microseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - restore_started).count());

        {
            const std::lock_guard<std::mutex> lock(mutex_);
            ++hits_;
            bytes_downloaded_ += entry->bytes;
            restore_time_microseconds_ += restore_microseconds;
            if (mapped_restore)
                ++mapped_restores_;
        }
        return restored;
    }

    void wait_for_background_work() override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [this] { return pending_.empty() && active_admissions_ == 0; });
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return {hits_,
                misses_,
                admissions_,
                0,
                0,
                0,
                stores_,
                evictions_,
                dropped_admissions_,
                restore_failures_,
                bytes_uploaded_,
                bytes_downloaded_,
                restore_time_microseconds_,
                mapped_stores_,
                mapped_restores_,
                resident_bytes_,
                pending_bytes_};
    }

    uint64_t capacity_bytes() const noexcept override
    {
        return capacity_bytes_;
    }

private:
    struct DeviceEntry
    {
        ncnn::VkMat data;
        std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
        uint64_t bytes = 0;
        uint64_t gate_blocks = 0;
        uint64_t gate_scales = 0;
        uint64_t down_blocks = 0;
        uint64_t down_scales = 0;
        uint64_t gate_blocks_offset = 0;
        uint64_t gate_scales_offset = 0;
        uint64_t down_blocks_offset = 0;
        uint64_t down_scales_offset = 0;
        uint32_t gate_output_columns = 0;
        uint32_t gate_input_columns = 0;
        uint32_t down_output_columns = 0;
        uint32_t down_input_columns = 0;
        uint64_t used_at = 0;
        ExpertVictimExecutionMetadata execution;
        bool operation_attempted = false;
        bool host_mapped = false;
    };

    struct PendingAdmission
    {
        std::string key;
        std::shared_ptr<const TensorData> gate_up;
        std::shared_ptr<const TensorData> down;
        uint64_t bytes = 0;
        uint64_t gate_blocks_offset = 0;
        uint64_t gate_scales_offset = 0;
        uint64_t down_blocks_offset = 0;
        uint64_t down_scales_offset = 0;
        ExpertVictimExecutionMetadata execution;
    };

    struct DownloadSlot
    {
        std::mutex mutex;
        ncnn::VkAllocator* staging_allocator = nullptr;
        ncnn::VkMat staging;
    };

    static bool append_segment(uint64_t bytes, uint64_t alignment, uint64_t& cursor, uint64_t& offset)
    {
        if (bytes == 0 || alignment == 0)
            return false;
        const uint64_t remainder = cursor % alignment;
        const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
        if (cursor > std::numeric_limits<uint64_t>::max() - padding)
        {
            return false;
        }
        cursor += padding;
        offset = cursor;
        if (cursor > std::numeric_limits<uint64_t>::max() - bytes)
        {
            return false;
        }
        cursor += bytes;
        return true;
    }

    static void copy_payload(const PendingAdmission& admission, uint8_t* destination)
    {
        std::memcpy(destination + admission.gate_blocks_offset, admission.gate_up->mxfp4_blocks.data(), admission.gate_up->mxfp4_blocks.size());
        std::memcpy(destination + admission.gate_scales_offset, admission.gate_up->mxfp4_scales.data(), admission.gate_up->mxfp4_scales.size());
        std::memcpy(destination + admission.down_blocks_offset, admission.down->mxfp4_blocks.data(), admission.down->mxfp4_blocks.size());
        std::memcpy(destination + admission.down_scales_offset, admission.down->mxfp4_scales.data(), admission.down->mxfp4_scales.size());
    }

    static MxFp4ByteBuffer copy_bytes(const uint8_t* source, uint64_t offset, uint64_t byte_count)
    {
        MxFp4ByteBuffer result;
        result.assign(source + offset, static_cast<size_t>(byte_count));
        return result;
    }

    static void materialize(const DeviceEntry& entry, const TensorData& gate_up_source, const TensorData& down_source, const uint8_t* source,
                            ExpertVictimPair& restored)
    {
        restored.gate_up = std::make_shared<TensorData>();
        restored.gate_up->dtype = DType::MxFp4;
        restored.gate_up->shape = gate_up_source.shape;
        restored.gate_up->mxfp4_blocks = copy_bytes(source, entry.gate_blocks_offset, entry.gate_blocks);
        restored.gate_up->mxfp4_scales = copy_bytes(source, entry.gate_scales_offset, entry.gate_scales);
        restored.down = std::make_shared<TensorData>();
        restored.down->dtype = DType::MxFp4;
        restored.down->shape = down_source.shape;
        restored.down->mxfp4_blocks = copy_bytes(source, entry.down_blocks_offset, entry.down_blocks);
        restored.down->mxfp4_scales = copy_bytes(source, entry.down_scales_offset, entry.down_scales);
    }

    std::shared_ptr<DeviceEntry> upload(const PendingAdmission& admission)
    {
        auto entry = std::make_shared<DeviceEntry>();
        entry->bytes = admission.bytes;
        entry->gate_blocks = admission.gate_up->mxfp4_blocks.size();
        entry->gate_scales = admission.gate_up->mxfp4_scales.size();
        entry->down_blocks = admission.down->mxfp4_blocks.size();
        entry->down_scales = admission.down->mxfp4_scales.size();
        entry->gate_blocks_offset = admission.gate_blocks_offset;
        entry->gate_scales_offset = admission.gate_scales_offset;
        entry->down_blocks_offset = admission.down_blocks_offset;
        entry->down_scales_offset = admission.down_scales_offset;
        if (admission.gate_up->shape.size() == 2)
        {
            entry->gate_output_columns = admission.gate_up->shape[0];
            entry->gate_input_columns = admission.gate_up->shape[1];
        }
        if (admission.down->shape.size() == 2)
        {
            entry->down_output_columns = admission.down->shape[0];
            entry->down_input_columns = admission.down->shape[1];
        }
        entry->execution = admission.execution;
        {
            const std::lock_guard<std::mutex> command_lock(context_->command_mutex());
            entry->data.create(static_cast<int>(admission.bytes), sizeof(uint8_t), context_->blob_allocator());
        }
        if (entry->data.empty())
            return {};
        if (entry->data.mapped_ptr())
        {
            entry->host_mapped = true;
            std::memset(entry->data.mapped_ptr(), 0, static_cast<size_t>(admission.bytes));
            copy_payload(admission, static_cast<uint8_t*>(entry->data.mapped_ptr()));
            entry->data.allocator->flush(entry->data.data);
            entry->data.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
            entry->data.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
            return entry;
        }

        upload_staging_.create(static_cast<int>(admission.bytes), sizeof(uint8_t), upload_staging_allocator_);
        if (upload_staging_.empty() || !upload_staging_.mapped_ptr())
            return {};

        uint8_t* destination = static_cast<uint8_t*>(upload_staging_.mapped_ptr());
        std::memset(destination, 0, static_cast<size_t>(admission.bytes));
        copy_payload(admission, destination);
        upload_staging_.allocator->flush(upload_staging_.data);
        upload_staging_.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        upload_staging_.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

        const std::lock_guard<std::mutex> command_lock(context_->command_mutex());
        ncnn::Option option;
        option.blob_vkallocator = context_->blob_allocator();
        option.workspace_vkallocator = context_->blob_allocator();
        option.staging_vkallocator = upload_staging_allocator_;
        ncnn::VkCompute command(context_->device(), context_->command_optimization_flags());
        command.record_clone(upload_staging_, entry->data, option);
        if (entry->data.empty() || command.submit_and_wait() != 0)
            return {};
        return entry;
    }

    bool download(const DeviceEntry& entry, const TensorData& gate_up_source, const TensorData& down_source, ExpertVictimPair& restored)
    {
        if (!gate_up_source.mxfp4_file_storage || !down_source.mxfp4_file_storage)
        {
            return false;
        }
        const MxFp4FileStorage& gate_file = *gate_up_source.mxfp4_file_storage;
        const MxFp4FileStorage& down_file = *down_source.mxfp4_file_storage;
        if (gate_file.blocks_bytes != entry.gate_blocks || gate_file.scales_bytes != entry.gate_scales || down_file.blocks_bytes != entry.down_blocks
            || down_file.scales_bytes != entry.down_scales)
        {
            return false;
        }

        if (entry.data.mapped_ptr())
        {
            entry.data.allocator->invalidate(entry.data.data);
            materialize(entry, gate_up_source, down_source, static_cast<const uint8_t*>(entry.data.mapped_ptr()), restored);
            entry.data.data->access_flags = VK_ACCESS_HOST_READ_BIT;
            entry.data.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
            return true;
        }

        DownloadSlot& slot = download_slots_[next_download_slot_.fetch_add(1, std::memory_order_relaxed) % download_slots_.size()];
        const std::lock_guard<std::mutex> slot_lock(slot.mutex);
        slot.staging.create(static_cast<int>(entry.bytes), sizeof(uint8_t), slot.staging_allocator);
        if (slot.staging.empty() || !slot.staging.mapped_ptr())
            return false;

        {
            const std::lock_guard<std::mutex> command_lock(context_->command_mutex());
            ncnn::Option option;
            option.blob_vkallocator = slot.staging_allocator;
            option.workspace_vkallocator = slot.staging_allocator;
            option.staging_vkallocator = slot.staging_allocator;
        ncnn::VkCompute command(context_->device(), context_->command_optimization_flags());
            command.record_clone(entry.data, slot.staging, option);
            if (slot.staging.empty() || command.submit_and_wait() != 0)
            {
                return false;
            }
        }
        slot.staging.allocator->invalidate(slot.staging.data);
        const uint8_t* source = static_cast<const uint8_t*>(slot.staging.mapped_ptr());
        materialize(entry, gate_up_source, down_source, source, restored);
        slot.staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
        slot.staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        return true;
    }

    void worker_loop()
    {
        for (;;)
        {
            PendingAdmission admission;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_available_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
                if (stopping_ && pending_.empty())
                    return;
                admission = std::move(pending_.front());
                pending_.pop_front();
                ++active_admissions_;
            }

            std::shared_ptr<DeviceEntry> entry = upload(admission);
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                pending_bytes_ -= admission.bytes;
                pending_keys_.erase(admission.key);
                if (entry)
                {
                    while (resident_bytes_ > capacity_bytes_ - entry->bytes)
                    {
                        auto victim = entries_.end();
                        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator)
                        {
                            if (iterator->second.use_count() != 1)
                            {
                                continue;
                            }
                            if (victim == entries_.end() || iterator->second->used_at < victim->second->used_at)
                            {
                                victim = iterator;
                            }
                        }
                        if (victim == entries_.end())
                            break;
                        resident_bytes_ -= victim->second->bytes;
                        entries_.erase(victim);
                        ++evictions_;
                    }
                    if (resident_bytes_ <= capacity_bytes_ - entry->bytes)
                    {
                        entry->used_at = ++clock_;
                        resident_bytes_ += entry->bytes;
                        bytes_uploaded_ += entry->bytes;
                        if (entry->host_mapped)
                            ++mapped_stores_;
                        ++stores_;
                        entries_[admission.key] = std::move(entry);
                    }
                    else
                    {
                        ++dropped_admissions_;
                    }
                }
                --active_admissions_;
                if (pending_.empty() && active_admissions_ == 0)
                {
                    idle_.notify_all();
                }
            }
        }
    }

    std::shared_ptr<NcnnVulkanContext> context_;
    uint64_t capacity_bytes_ = 0;
    uint64_t maximum_pending_bytes_ = 0;
    ncnn::VkAllocator* upload_staging_allocator_ = nullptr;
    mutable std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable idle_;
    std::deque<PendingAdmission> pending_;
    std::unordered_set<std::string, TransparentStringHash, std::equal_to<>> pending_keys_;
    std::unordered_map<std::string, std::shared_ptr<DeviceEntry>, TransparentStringHash, std::equal_to<>> entries_;
    uint64_t pending_bytes_ = 0;
    uint32_t active_admissions_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t admissions_ = 0;
    uint64_t stores_ = 0;
    uint64_t evictions_ = 0;
    uint64_t dropped_admissions_ = 0;
    uint64_t restore_failures_ = 0;
    uint64_t bytes_uploaded_ = 0;
    uint64_t bytes_downloaded_ = 0;
    uint64_t restore_time_microseconds_ = 0;
    uint64_t mapped_stores_ = 0;
    uint64_t mapped_restores_ = 0;
    bool stopping_ = false;
    ncnn::VkMat upload_staging_;
    std::array<DownloadSlot, 2> download_slots_;
    std::atomic<size_t> next_download_slot_{0};
    std::thread worker_;
};

static bool has_batch_shape(const ncnn::VkMat& buffer, size_t rows, uint32_t columns, size_t element_size = sizeof(float))
{
    return buffer.dims == 2 && buffer.w == static_cast<int>(columns) && buffer.h == static_cast<int>(rows) && buffer.elemsize == element_size
           && buffer.elempack == 1;
}

static bool prepare_staging_batch(ncnn::VkMat& buffer, size_t rows, uint32_t columns, ncnn::VkAllocator* allocator, NcnnVulkanRuntimeState& runtime_state,
                                  size_t element_size = sizeof(float))
{
    const bool reused = has_batch_shape(buffer, rows, columns, element_size);
    buffer.create(static_cast<int>(columns), static_cast<int>(rows), element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++runtime_state.staging_slot_reuses;
    else
        ++runtime_state.staging_slot_resizes;
    return true;
}

static bool prepare_staging_matrix(ncnn::VkMat& buffer, int width, int height, size_t element_size, ncnn::VkAllocator* allocator, NcnnVulkanRuntimeState& runtime_state)
{
    const bool reused = buffer.dims == 2 && buffer.w == width && buffer.h == height && buffer.elemsize == element_size && buffer.elempack == 1;
    buffer.create(width, height, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++runtime_state.staging_slot_reuses;
    else
        ++runtime_state.staging_slot_resizes;
    return true;
}

static bool prepare_staging_tensor(ncnn::VkMat& buffer, int width, int height, int channels, size_t element_size, ncnn::VkAllocator* allocator, NcnnVulkanRuntimeState& runtime_state)
{
    const bool reused = buffer.dims == 3 && buffer.w == width && buffer.h == height && buffer.c == channels && buffer.elemsize == element_size && buffer.elempack == 1;
    buffer.create(width, height, channels, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++runtime_state.staging_slot_reuses;
    else
        ++runtime_state.staging_slot_resizes;
    return true;
}

static bool record_mapped_upload(ncnn::VkMat& staging, ncnn::VkMat& destination, ncnn::VkCompute& command, const ncnn::Option& option)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    command.record_clone(staging, destination, option);
    return !destination.empty();
}

static ncnn::Option activation_source_option(const ncnn::Option& option, DType source_dtype)
{
    ncnn::Option source_option = option;
    if (source_dtype == DType::BFloat16)
    {
        source_option.use_fp16_storage = false;
        source_option.use_fp16_packed = false;
        source_option.use_bf16_storage = true;
        source_option.use_bf16_packed = true;
    }
    else if (source_dtype == DType::Float16)
    {
        source_option.use_bf16_storage = false;
        source_option.use_bf16_packed = false;
        source_option.use_fp16_storage = true;
        source_option.use_fp16_packed = true;
    }
    return source_option;
}

static size_t vulkan_activation_storage_variant(
    const ncnn::Option& option) noexcept;
static size_t vulkan_activation_element_size(
    const ncnn::Option& option) noexcept;

static bool record_mapped_activation_upload(
    ncnn::VkMat& staging,
    ncnn::VkMat& destination,
    ncnn::VkCompute& command,
    ncnn::VulkanDevice* device,
    const ncnn::Option& option,
    DType source_dtype = DType::Float32)
{
    if (vulkan_activation_storage_variant(option) == 0)
    {
        if (source_dtype == DType::Float32)
            return record_mapped_upload(staging, destination, command, option);
        if (!device || staging.empty() || !staging.mapped_ptr())
            return false;
        staging.allocator->flush(staging.data);
        staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        const ncnn::Option source_option =
            activation_source_option(option, source_dtype);
        const int destination_elempack = staging.h % 4 == 0 ? 4 : 1;
        device->convert_packing(
            staging,
            destination,
            destination_elempack,
            1,
            command,
            source_option);
        return !destination.empty();
    }
    if (!device || staging.empty() || !staging.mapped_ptr())
        return false;
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    const int cast_type = option.use_bf16_storage ? 5 : 2;
    const ncnn::Option source_option =
        activation_source_option(option, source_dtype);
    device->convert_packing(
        staging,
        destination,
        1,
        cast_type,
        command,
        source_option);
    return !destination.empty()
           && destination.elemsize
                  == vulkan_activation_element_size(option);
}

static size_t vulkan_activation_storage_variant(
    const ncnn::Option& option) noexcept
{
    return option.use_fp16_storage
               ? 1
               : option.use_bf16_storage ? 2 : 0;
}

static bool vulkan_fp16_activations_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanFp16Activations);
}

static size_t vulkan_activation_element_size(
    const ncnn::Option& option) noexcept
{
    return vulkan_activation_storage_variant(option) == 0
               ? sizeof(float)
               : sizeof(uint16_t);
}

static int submit_compute_and_wait(
    ncnn::VkCompute& command,
    NcnnVulkanRuntimeState& runtime_state)
{
    const ncnn::VkComputeCommandStatistics command_recording =
        command.command_statistics();
    const auto started = std::chrono::steady_clock::now();
    const int result = command.submit_and_wait();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started);
    runtime_state.submit_wait_time_microseconds +=
        static_cast<uint64_t>(elapsed.count());
    runtime_state.command_dispatches +=
        command_recording.dispatches;
    runtime_state.command_pipeline_binds +=
        command_recording.pipeline_binds;
    runtime_state.command_redundant_pipeline_binds +=
        command_recording.redundant_pipeline_binds;
    runtime_state.command_descriptor_bindings +=
        command_recording.descriptor_bindings;
    runtime_state.command_push_constant_updates +=
        command_recording.push_constant_updates;
    runtime_state.command_resource_barrier_calls +=
        command_recording.resource_barrier_calls;
    runtime_state.command_buffer_resource_barriers +=
        command_recording.buffer_resource_barriers;
    runtime_state.command_image_resource_barriers +=
        command_recording.image_resource_barriers;
    return result;
}

static bool vulkan_attention_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationVulkanAttention);
}

static bool vulkan_attention_batch_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanAttentionBatch);
}

static bool vulkan_attention_kv_promotion_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanKvPromotion);
}

static bool vulkan_attention_device_rope_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanDeviceRope);
}

static bool direct_host_buffer_enabled(
    const NcnnVulkanContext& context,
    size_t bytes,
    bool enabled,
    size_t maximum_discrete_default_bytes) noexcept
{
    // Host-visible buffers are useful for small activation tensors, but are
    // not a replacement for device-local storage on discrete GPUs.  Keep
    // this policy generic: dtype/layout and device class decide it, not a
    // model adapter or checkpoint family.
    constexpr size_t maximum_direct_host_bytes = 64 * 1024;
    if (bytes > maximum_direct_host_bytes)
        return false;

    if (!enabled)
        return false;

    const ncnn::VulkanDevice* device = context.device();
    return device
           && (device->info.type() > 0
               || (maximum_discrete_default_bytes != 0
                   && bytes <= maximum_discrete_default_bytes));
}

static bool direct_host_input_enabled(
    const NcnnVulkanContext& context,
    size_t input_bytes,
    DType input_dtype) noexcept
{
    // A direct host binding bypasses the cast recorded by the staging path.
    // It is therefore valid only when the host payload already matches the
    // FP32 storage contract used by this path.
    if (input_dtype != DType::Float32)
        return false;
    return direct_host_buffer_enabled(
        context,
        input_bytes,
        runtime_optimization_enabled(
            context.optimization_flags(),
            RuntimeOptimizationVulkanDirectHostInput),
        16 * 1024);
}

static bool direct_host_output_enabled(
    const NcnnVulkanContext& context,
    size_t output_bytes,
    DType output_dtype) noexcept
{
    // The command layer now records the shader-write -> host-read barrier
    // explicitly.  Keep this conservative on discrete GPUs by default: a
    // host-visible output can save a device-to-staging copy for small tensors,
    // but PCIe write latency can outweigh that saving for larger outputs.
    if (output_dtype != DType::Float32)
        return false;
    return direct_host_buffer_enabled(
        context,
        output_bytes,
        runtime_optimization_enabled(
            context.optimization_flags(),
            RuntimeOptimizationVulkanDirectHostOutput),
        0);
}

static ncnn::VkMat bind_direct_host_input(ncnn::VkMat& staging, NcnnVulkanRuntimeState& runtime_state)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ++runtime_state.direct_host_input_bindings;
    return staging;
}

static ncnn::VkMat prepare_direct_host_output(ncnn::VkMat& staging, NcnnVulkanRuntimeState& runtime_state)
{
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ++runtime_state.direct_host_output_bindings;
    return staging;
}

static bool fill_staging_upload(const ActivationBuffer& input, ncnn::VkMat& staging, ncnn::VkAllocator* allocator, NcnnVulkanRuntimeState& runtime_state)
{
    const size_t element_size = input.element_size();
    if (element_size == 0 || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || input.bytes().size() != input.rows() * static_cast<size_t>(input.columns()) * element_size)
        return false;
    if (!prepare_staging_batch(staging, input.rows(), input.columns(), allocator, runtime_state, element_size))
        return false;

    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.total() * mapped.elemsize < input.bytes().size())
        return false;
    std::memcpy(mapped.data, input.bytes().data(), input.bytes().size());
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static bool fill_staging_values(const void* source, size_t count, size_t element_size, ncnn::VkMat& staging, ncnn::VkAllocator* allocator, NcnnVulkanRuntimeState& runtime_state)
{
    if (!source || count == 0 || element_size == 0 || count > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    if (!prepare_staging_matrix(staging, static_cast<int>(count), 1, element_size, allocator, runtime_state))
        return false;
    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.total() < count)
        return false;
    std::memcpy(mapped.data, source, count * element_size);
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static bool record_prepared_staging_upload(const ncnn::VkMat& staging, size_t rows, ncnn::VkMat& destination, ncnn::VkCompute& command,
                                           ncnn::VulkanDevice* device, const ncnn::Option& option, DType source_dtype = DType::Float32)
{
    if (!device || staging.empty() || !staging.mapped_ptr())
        return false;
    const int packed_rows = static_cast<int>(rows);
    const int destination_elempack =
        packed_rows % 4 == 0 && (option.use_packing_layout || packed_rows <= 4) ? 4 : 1;
    int cast_type = source_dtype == DType::Float32 ? 0 : 1;
    if (option.use_bf16_storage || option.use_bf16_packed)
        cast_type = 5;
    else if (option.use_fp16_storage || option.use_fp16_packed)
        cast_type = 2;
    else if (device->info.type() != 0)
    {
        cast_type = 1;
    }
    const ncnn::Option source_option =
        activation_source_option(option, source_dtype);
    device->convert_packing(staging, destination, destination_elempack, cast_type, command, source_option);
    return !destination.empty();
}

static bool record_prepared_staging_download(const ncnn::VkMat& source, size_t rows, uint32_t columns, ncnn::VkMat& staging, ncnn::VkCompute& command,
                                             const ncnn::Option& option)
{
    if (!has_batch_shape(staging, rows, columns))
        return false;
    ncnn::Option staging_option = option;
    staging_option.blob_vkallocator = staging.allocator;
    staging_option.workspace_vkallocator = staging.allocator;
    staging_option.staging_vkallocator = staging.allocator;
    command.record_clone(source, staging, staging_option);
    return !staging.empty();
}

static bool record_prepared_activation_staging_download(
    const ncnn::VkMat& source,
    size_t rows,
    uint32_t columns,
    ncnn::VkMat& staging,
    ncnn::VkCompute& command,
    ncnn::VulkanDevice* device,
    const ncnn::Option& option,
    DType output_dtype = DType::Float32)
{
    const size_t output_element_size =
        output_dtype == DType::Float32
            ? sizeof(float)
            : (output_dtype == DType::Float16 || output_dtype == DType::BFloat16
                   ? sizeof(uint16_t)
                   : 0);
    if (output_element_size == 0)
        return false;

    const int output_cast_type =
        output_dtype == DType::Float32
            ? 1
            : output_dtype == DType::Float16 ? 2 : 5;
    const bool source_matches_cpu_batch =
        source.dims == 2
        && source.w == static_cast<int>(columns)
        && source.h == static_cast<int>(rows)
        && source.elempack == 1
        && source.elemsize == output_element_size;
    if (vulkan_activation_storage_variant(option) == 0
        && output_dtype == DType::Float32
        && source_matches_cpu_batch)
    {
        return record_prepared_staging_download(
            source,
            rows,
            columns,
            staging,
            command,
            option);
    }
    if (!device)
        return false;
    if (!staging.allocator)
        return false;
    staging.create(
        static_cast<int>(columns),
        static_cast<int>(rows),
        output_element_size,
        staging.allocator);
    if (staging.empty() || !staging.mapped_ptr())
        return false;
    if (!has_batch_shape(staging, rows, columns, output_element_size))
        return false;
    ncnn::Option staging_option = option;
    staging_option.blob_vkallocator = staging.allocator;
    staging_option.workspace_vkallocator = staging.allocator;
    staging_option.staging_vkallocator = staging.allocator;
    device->convert_packing(
        source,
        staging,
        1,
        output_cast_type,
        command,
        staging_option);
    const bool matches = has_batch_shape(
        staging,
        rows,
        columns,
        output_element_size);
    return matches;
}

static bool copy_staging_to_cpu_batch(ncnn::VkMat& staging, ActivationBuffer& output)
{
    const size_t element_size = output.element_size();
    if (element_size == 0 || staging.elemsize != element_size)
        return false;
    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.dims != 2 || mapped.w != static_cast<int>(output.columns()) || mapped.h != static_cast<int>(output.rows())
        || mapped.elempack != 1 || mapped.elemsize != element_size)
        return false;
    const size_t row_bytes = static_cast<size_t>(output.columns()) * element_size;
    const auto* source = static_cast<const std::byte*>(mapped.data);
    for (size_t row_index = 0; row_index < output.rows(); ++row_index)
        std::memcpy(output.mutable_row_bytes(row_index).data(), source + row_index * row_bytes, row_bytes);
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static ncnn::VkMat row_view(const ncnn::VkMat& source, size_t first_row, size_t rows)
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

#endif

class NcnnLinearOperator::Implementation
{
public:
#if NCNN_MOE_USE_NCNN
    ~Implementation()
    {
        if (layer)
        {
#if NCNN_MOE_WITH_VULKAN
            if (vulkan_context)
            {
                const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
                if (pipeline_created)
                    layer->destroy_pipeline(option);
            }
            else
#endif
                if (pipeline_created)
            {
                layer->destroy_pipeline(option);
            }
            delete layer;
        }
#if NCNN_MOE_WITH_VULKAN
        weight_staging_allocator.reset();
        weight_allocator.reset();
#endif
    }

    ncnn::Layer* layer = nullptr;
    ncnn::Option option;
    mutable std::mutex forward_mutex;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool pipeline_created = false;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
#endif
#endif
};

#if NCNN_MOE_WITH_VULKAN
struct NcnnVulkanCommandGraphState
{
};
#endif

class NcnnVulkanDeviceTensor::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::weak_ptr<NcnnVulkanCommandGraphState> graph;
    ncnn::VkMat value;
    size_t rows = 0;
    uint32_t columns = 0;
#endif
};

class NcnnVulkanCommandGraph::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    struct PendingDownload
    {
        ncnn::VkMat staging;
        ActivationBuffer* output = nullptr;
    };

    std::shared_ptr<NcnnVulkanContext> context;
    ncnn::Option option;
    std::shared_ptr<NcnnVulkanCommandGraphState> state;
    std::unique_ptr<NcnnVulkanTransferLease> transfer_lease;
    std::vector<ncnn::VkMat> upload_staging;
    std::vector<PendingDownload> pending_downloads;
    uint64_t recorded_operations = 0;
    uint64_t upload_count = 0;
    bool submitted = false;
    bool completed = false;
    ncnn::VkComputeCommandStatistics command_recording;
#endif
};

class NcnnVulkanAttentionCache
{
public:
#if NCNN_MOE_WITH_VULKAN
    // Double-written rows keep wrapped cache views contiguous.
    ncnn::VkMat key;
    ncnn::VkMat value;
#endif
};

class NcnnVulkanAttentionOperator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    struct DecodeSdpaPolicy
    {
        double fused_microseconds = 0.0;
        double ncnn_microseconds = 0.0;
        uint64_t fused_samples = 0;
        uint64_t ncnn_samples = 0;
        uint64_t decisions = 0;
        bool prefer_fused = false;
        bool preference_initialized = false;
        bool probe_pending = false;
        bool probe_fused = false;
    };

    [[nodiscard]] static uint64_t decode_sdpa_key(
        uint32_t head_dimension,
        uint32_t head_count,
        uint32_t key_value_head_count,
        uint64_t destination_count) noexcept
    {
        uint64_t context_bucket = 0;
        uint64_t upper_bound = 16;
        while (upper_bound < destination_count && context_bucket < 15)
        {
            upper_bound <<= 1;
            ++context_bucket;
        }
        return static_cast<uint64_t>(head_dimension) << 40
               | static_cast<uint64_t>(head_count) << 24
               | static_cast<uint64_t>(key_value_head_count) << 8
               | context_bucket;
    }

    [[nodiscard]] bool choose_decode_sdpa(
        uint32_t head_dimension,
        uint32_t head_count,
        uint32_t key_value_head_count,
        uint64_t destination_count) const
    {
        const uint64_t key = decode_sdpa_key(
            head_dimension,
            head_count,
            key_value_head_count,
            destination_count);
        const std::lock_guard<std::mutex> lock(decode_sdpa_mutex);
        DecodeSdpaPolicy& policy = decode_sdpa_policies[key];
        bool use_fused = false;
        if (policy.ncnn_samples < 2)
        {
            use_fused = false;
        }
        else if (policy.fused_samples < 2)
        {
            use_fused = true;
        }
        else
        {
            use_fused = policy.prefer_fused;
            if (policy.decisions != 0 && policy.decisions % 256 == 0)
            {
                use_fused = !use_fused;
                policy.probe_pending = true;
                policy.probe_fused = use_fused;
            }
        }
        ++policy.decisions;
        return use_fused;
    }

    void observe_decode_sdpa(
        uint32_t head_dimension,
        uint32_t head_count,
        uint32_t key_value_head_count,
        uint64_t destination_count,
        bool fused,
        uint64_t elapsed_microseconds) const
    {
        const uint64_t key = decode_sdpa_key(
            head_dimension,
            head_count,
            key_value_head_count,
            destination_count);
        const std::lock_guard<std::mutex> lock(decode_sdpa_mutex);
        DecodeSdpaPolicy& policy = decode_sdpa_policies[key];
        double& estimate = fused
                               ? policy.fused_microseconds
                               : policy.ncnn_microseconds;
        uint64_t& samples = fused ? policy.fused_samples : policy.ncnn_samples;
        estimate = samples == 0
                       ? static_cast<double>(elapsed_microseconds)
                       : estimate * 0.75
                             + static_cast<double>(elapsed_microseconds) * 0.25;
        ++samples;
        const bool initial_comparison =
            !policy.preference_initialized
            && policy.fused_samples >= 2
            && policy.ncnn_samples >= 2;
        const bool probe_comparison =
            policy.probe_pending && policy.probe_fused == fused;
        if (initial_comparison || probe_comparison)
        {
            if (policy.prefer_fused)
            {
                if (policy.fused_microseconds
                    > policy.ncnn_microseconds * 1.02)
                    policy.prefer_fused = false;
            }
            else if (policy.fused_microseconds * 1.02
                     < policy.ncnn_microseconds)
            {
                policy.prefer_fused = true;
            }
            policy.preference_initialized = true;
            policy.probe_pending = false;
        }
    }

    ~Implementation()
    {
        if (vulkan_context)
        {
            qkv_rope_pipeline.reset();
            qkv_norm_rope_pipeline.reset();
            output_gate_pipeline.reset();
            decode_sdpa_pipeline.reset();
            ring_append_pipeline.reset();
            ring_zero_pipeline.reset();
            const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
            for (ncnn::Layer* layer : layers)
            {
                layer->destroy_pipeline(option);
                delete layer;
            }
        }
        layers.clear();
        attention_sinks = ncnn::VkMat();
        rope_inverse_frequencies_gpu = ncnn::VkMat();
        query_norm_weight = ncnn::VkMat();
        key_norm_weight = ncnn::VkMat();
        weight_staging_allocator.reset();
        weight_allocator.reset();
    }

    ncnn::Layer* norm = nullptr;
    ncnn::Layer* slice_qkv = nullptr;
    ncnn::Layer* reshape_query = nullptr;
    ncnn::Layer* reshape_key_value = nullptr;
    ncnn::Layer* permute_heads_tokens = nullptr;
    ncnn::Layer* rotary = nullptr;
    ncnn::Layer* sdpa = nullptr;
    ncnn::Layer* reshape_attention = nullptr;
    ncnn::Layer* add = nullptr;
    std::shared_ptr<ncnn::Pipeline> qkv_rope_pipeline;
    std::shared_ptr<ncnn::Pipeline> qkv_norm_rope_pipeline;
    std::shared_ptr<ncnn::Pipeline> output_gate_pipeline;
    std::shared_ptr<ncnn::Pipeline> decode_sdpa_pipeline;
    std::shared_ptr<ncnn::Pipeline> ring_append_pipeline;
    std::shared_ptr<ncnn::Pipeline> ring_zero_pipeline;
    std::vector<ncnn::Layer*> layers;
    ncnn::Option option;
    ncnn::Option kv_option;
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    ncnn::VkMat attention_sinks;
    ncnn::VkMat rope_inverse_frequencies_gpu;
#endif
    std::shared_ptr<NcnnLinearOperator> fused_qkv;
    std::shared_ptr<NcnnVulkanBfloat16Operator> fused_qkv_gate;
    std::shared_ptr<NcnnLinearOperator> output_projection;
    std::shared_ptr<NcnnVulkanBfloat16Operator> output_projection_bfloat16;
    NcnnVulkanAttentionConfig config;
    std::vector<float> sinks;
    std::vector<float> rope_inverse_frequencies;
#if NCNN_MOE_WITH_VULKAN
    ncnn::VkMat query_norm_weight;
    ncnn::VkMat key_norm_weight;
#endif
#if NCNN_MOE_WITH_VULKAN
    mutable std::mutex decode_sdpa_mutex;
    mutable std::unordered_map<uint64_t, DecodeSdpaPolicy> decode_sdpa_policies;
#endif
    float rope_concentration = 1.0f;
};

NcnnLinearOperator::NcnnLinearOperator()
    : implementation_(new Implementation)
{
}

NcnnLinearOperator::~NcnnLinearOperator() = default;

NcnnVulkanDeviceTensor::NcnnVulkanDeviceTensor()
    : implementation_(new Implementation)
{
}

NcnnVulkanDeviceTensor::~NcnnVulkanDeviceTensor() = default;

NcnnVulkanDeviceTensor::NcnnVulkanDeviceTensor(
    NcnnVulkanDeviceTensor&&) noexcept = default;

NcnnVulkanDeviceTensor& NcnnVulkanDeviceTensor::operator=(
    NcnnVulkanDeviceTensor&&) noexcept = default;

bool NcnnVulkanDeviceTensor::empty() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return !implementation_ || implementation_->value.empty();
#else
    return true;
#endif
}

size_t NcnnVulkanDeviceTensor::rows() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return implementation_ ? implementation_->rows : 0;
#else
    return 0;
#endif
}

uint32_t NcnnVulkanDeviceTensor::columns() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return implementation_ ? implementation_->columns : 0;
#else
    return 0;
#endif
}

NcnnVulkanCommandGraph::NcnnVulkanCommandGraph(
    std::unique_ptr<Implementation> implementation)
    : implementation_(std::move(implementation))
{
}

NcnnVulkanCommandGraph::~NcnnVulkanCommandGraph()
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (implementation_ && implementation_->submitted
        && !implementation_->completed
        && implementation_->context
        && implementation_->transfer_lease)
    {
        (void)wait();
    }
    if (implementation_ && !implementation_->submitted
        && implementation_->context && implementation_->transfer_lease)
    {
        const std::lock_guard<std::mutex> lock(
            implementation_->context->command_mutex());
        (void)implementation_->transfer_lease->slot().command->reset();
    }
#endif
}

NcnnVulkanCommandGraph::NcnnVulkanCommandGraph(
    NcnnVulkanCommandGraph&&) noexcept = default;

NcnnVulkanCommandGraph& NcnnVulkanCommandGraph::operator=(
    NcnnVulkanCommandGraph&&) noexcept = default;

std::unique_ptr<NcnnVulkanCommandGraph> NcnnVulkanCommandGraph::create(
    const NcnnLinearOperator& seed_operator)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    const NcnnLinearOperator::Implementation& seed =
        *seed_operator.implementation_;
    if (!seed.layer || !seed.vulkan_context)
        return {};

    auto implementation = std::make_unique<Implementation>();
    implementation->context = seed.vulkan_context;
    implementation->option = seed.option;
    implementation->state =
        std::make_shared<NcnnVulkanCommandGraphState>();
    implementation->transfer_lease =
        std::make_unique<NcnnVulkanTransferLease>(
            implementation->context->acquire_transfer_slot());

    NcnnVulkanTransferSlot& transfer_slot =
        implementation->transfer_lease->slot();
    NcnnVulkanRuntimeState& runtime_state =
        implementation->context->runtime_state();
    const std::lock_guard<std::mutex> lock(
        implementation->context->command_mutex());
    if (transfer_slot.command_used)
    {
        if (transfer_slot.command->reset() != 0)
            return {};
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    return std::unique_ptr<NcnnVulkanCommandGraph>(
        new NcnnVulkanCommandGraph(std::move(implementation)));
#else
    (void)seed_operator;
    return {};
#endif
}

bool NcnnVulkanCommandGraph::upload(
    const ActivationBuffer& input,
    NcnnVulkanDeviceTensor& output)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!implementation_ || implementation_->submitted
        || !implementation_->context || input.rows() == 0
        || input.columns() == 0
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    ncnn::VkMat staging;
    if (!fill_staging_upload(
            input,
            staging,
            implementation_->transfer_lease->slot().staging_allocator,
            implementation_->context->runtime_state()))
    {
        return false;
    }

    ncnn::VkMat device_input;
    {
        const std::lock_guard<std::mutex> lock(
            implementation_->context->command_mutex());
        if (!record_prepared_staging_upload(
                staging,
                input.rows(),
                device_input,
                *implementation_->transfer_lease->slot().command,
                implementation_->context->device(),
                implementation_->option,
                input.dtype()))
        {
            return false;
        }
    }

    auto tensor = std::make_unique<NcnnVulkanDeviceTensor::Implementation>();
    tensor->graph = implementation_->state;
    tensor->value = std::move(device_input);
    tensor->rows = input.rows();
    tensor->columns = input.columns();
    output.implementation_ = std::move(tensor);
    implementation_->upload_staging.push_back(std::move(staging));
    ++implementation_->upload_count;
    ++implementation_->recorded_operations;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanCommandGraph::linear(
    const NcnnLinearOperator& operator_instance,
    const NcnnVulkanDeviceTensor& input,
    NcnnVulkanDeviceTensor& output)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!implementation_ || implementation_->submitted
        || !input.implementation_ || !output.implementation_
        || &input == &output || input.empty())
    {
        return false;
    }
    const std::shared_ptr<NcnnVulkanCommandGraphState> input_graph =
        input.implementation_->graph.lock();
    if (!input_graph
        || input_graph.get() != implementation_->state.get())
    {
        return false;
    }

    const NcnnLinearOperator::Implementation& implementation =
        *operator_instance.implementation_;
    if (!implementation.layer || !implementation.vulkan_context
        || implementation.vulkan_context.get()
               != implementation_->context.get()
        || implementation.input_columns != input.columns()
        || vulkan_activation_storage_variant(implementation.option)
               != vulkan_activation_storage_variant(
                   implementation_->option))
    {
        return false;
    }

    ncnn::VkMat device_output;
    {
        const std::lock_guard<std::mutex> lock(
            implementation_->context->command_mutex());
        if (implementation.layer->forward(
                input.implementation_->value,
                device_output,
                *implementation_->transfer_lease->slot().command,
                implementation.option)
            != 0)
        {
            return false;
        }
        if (device_output.elempack != 1)
        {
            ncnn::VkMat unpacked;
            implementation_->context->device()->convert_packing(
                device_output,
                unpacked,
                1,
                *implementation_->transfer_lease->slot().command,
                implementation.option);
            device_output = unpacked;
        }
    }
    if (device_output.empty())
        return false;

    auto tensor = std::make_unique<NcnnVulkanDeviceTensor::Implementation>();
    tensor->graph = implementation_->state;
    tensor->value = std::move(device_output);
    tensor->rows = input.rows();
    tensor->columns = implementation.output_columns;
    output.implementation_ = std::move(tensor);
    ++implementation_->recorded_operations;
    return true;
#else
    (void)operator_instance;
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanCommandGraph::download(
    const NcnnVulkanDeviceTensor& input,
    ActivationBuffer& output)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!implementation_ || implementation_->submitted
        || !input.implementation_ || input.empty())
    {
        return false;
    }
    const std::shared_ptr<NcnnVulkanCommandGraphState> input_graph =
        input.implementation_->graph.lock();
    if (!input_graph
        || input_graph.get() != implementation_->state.get())
    {
        return false;
    }

    NcnnVulkanCommandGraph::Implementation::PendingDownload pending;
    const DType output_dtype = output.dtype();
    if (!prepare_staging_batch(
            pending.staging,
            input.rows(),
            input.columns(),
            implementation_->transfer_lease->slot().staging_allocator,
            implementation_->context->runtime_state(),
            output.element_size()))
    {
        return false;
    }
    output.reset(input.rows(), input.columns(), false);
    {
        const std::lock_guard<std::mutex> lock(
            implementation_->context->command_mutex());
        if (!record_prepared_activation_staging_download(
                input.implementation_->value,
                input.rows(),
                input.columns(),
                pending.staging,
                *implementation_->transfer_lease->slot().command,
                implementation_->context->device(),
                implementation_->option,
                output_dtype))
        {
            return false;
        }
    }
    pending.output = &output;
    implementation_->pending_downloads.push_back(std::move(pending));
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanCommandGraph::submit()
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!implementation_ || implementation_->submitted
        || implementation_->recorded_operations == 0
        || !implementation_->context || !implementation_->transfer_lease)
    {
        return false;
    }

    ncnn::VkCompute& command =
        *implementation_->transfer_lease->slot().command;
    const std::lock_guard<std::mutex> lock(
        implementation_->context->command_mutex());
    implementation_->command_recording = command.command_statistics();
    NcnnVulkanRuntimeState& runtime_state =
        implementation_->context->runtime_state();
    if (command.submit() != 0)
        return false;
    implementation_->submitted = true;
    implementation_->completed = false;
    runtime_state.dispatches += implementation_->command_recording.dispatches;
    ++runtime_state.compute_submissions;
    runtime_state.batch_uploads +=
        implementation_->upload_count;
    runtime_state.batch_downloads +=
        implementation_->pending_downloads.size();
    ++runtime_state.command_graph_submissions;
    runtime_state.command_graph_operations +=
        implementation_->recorded_operations;
    return true;
#else
    return false;
#endif
}

bool NcnnVulkanCommandGraph::wait()
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!implementation_ || !implementation_->submitted
        || implementation_->completed
        || !implementation_->context || !implementation_->transfer_lease)
    {
        return false;
    }

    ncnn::VkCompute& command =
        *implementation_->transfer_lease->slot().command;
    NcnnVulkanRuntimeState& runtime_state =
        implementation_->context->runtime_state();
    const auto started = std::chrono::steady_clock::now();
    {
        const std::lock_guard<std::mutex> lock(
            implementation_->context->command_mutex());
        if (command.wait() != 0)
            return false;
    }
    runtime_state.submit_wait_time_microseconds += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count());

    for (NcnnVulkanCommandGraph::Implementation::PendingDownload& pending :
         implementation_->pending_downloads)
    {
        if (!pending.output
            || !copy_staging_to_cpu_batch(pending.staging, *pending.output))
        {
            return false;
        }
    }
    implementation_->completed = true;
    return true;
#else
    return false;
#endif
}

const char* NcnnLinearOperator::cpu_small_bfloat16_linear_policy(
    uint64_t optimization_flags) noexcept
{
#if NCNN_MOE_USE_NCNN
    return ncnn_cpu_bfloat16_linear_enabled(optimization_flags)
               ? "ncnn-fp32-expanded"
               : "moe-direct-bfloat16";
#else
    (void)optimization_flags;
    return "moe-direct-bfloat16";
#endif
}

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create(const TensorData& matrix, const TensorData* bias, NcnnLinearDevice device,
                                                               uint32_t vulkan_device_index,
                                                               const NcnnVulkanContextInstancePtr& context_instance,
                                                               uint64_t optimization_flags)
{
#if !NCNN_MOE_WITH_VULKAN
    (void)vulkan_device_index;
    (void)context_instance;
#endif
#if NCNN_MOE_USE_NCNN
    if (matrix.shape.size() != 2 || (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16))
        return {};
    if (device == NcnnLinearDevice::Cpu
        && matrix.dtype == DType::BFloat16
        && !ncnn_cpu_bfloat16_linear_enabled(optimization_flags))
    {
        return {};
    }

    const uint64_t element_size = matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
    if (device == NcnnLinearDevice::Cpu && matrix.element_count() > max_ncnn_linear_weight_bytes / element_size)
        return {};
    if (matrix.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};

    std::shared_ptr<NcnnLinearOperator> linear(new NcnnLinearOperator);
    Implementation& implementation = *linear->implementation_;
    implementation.optimization_flags = optimization_flags;
    implementation.input_columns = matrix.shape[1];
    implementation.output_columns = matrix.shape[0];
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;

#if NCNN_MOE_WITH_VULKAN
    if (device == NcnnLinearDevice::Vulkan)
    {
        implementation.vulkan_context = NcnnVulkanContext::acquire(
            vulkan_device_index,
            context_instance,
            optimization_flags);
        if (!implementation.vulkan_context)
            return {};

        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        implementation.option.use_vulkan_compute = true;
        implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
        implementation.option.use_cooperative_matrix = vkdev->info.support_cooperative_matrix();
        implementation.option.use_subgroup_ops = vkdev->info.support_subgroup_ops();
        implementation.layer = ncnn::create_layer_vulkan(ncnn::LayerType::InnerProduct);
        if (implementation.layer)
            implementation.layer->vkdev = vkdev;
    }
    else
#endif
    {
        implementation.layer = ncnn::create_layer_cpu(ncnn::LayerType::InnerProduct);
    }
    if (!implementation.layer)
        return {};

    ncnn::ParamDict parameters;
    parameters.set(0, static_cast<int>(implementation.output_columns));
    parameters.set(1, bias ? 1 : 0);
    parameters.set(2, static_cast<int>(matrix.element_count()));
    if (implementation.layer->load_param(parameters) != 0)
        return {};

    ncnn::Mat model_data[2];
    model_data[0].create(static_cast<int>(matrix.element_count()), sizeof(float));
    if (model_data[0].empty())
        return {};
    float* weight_data = static_cast<float*>(model_data[0].data);
    const std::span<const float> float32_weights = matrix.float32_values();
    const std::span<const uint16_t> bfloat16_weights = matrix.bfloat16_values();
    for (size_t index = 0; index < matrix.element_count(); ++index)
        weight_data[index] = matrix.dtype == DType::Float32 ? float32_weights[index] : bfloat16_to_float(bfloat16_weights[index]);

    if (bias)
    {
        model_data[1].create(static_cast<int>(implementation.output_columns), sizeof(float));
        if (model_data[1].empty())
            return {};
        float* bias_data = static_cast<float*>(model_data[1].data);
        const std::span<const float> float32_bias = bias->float32_values();
        const std::span<const uint16_t> bfloat16_bias = bias->bfloat16_values();
        for (uint32_t column = 0; column < implementation.output_columns; ++column)
        {
            bias_data[column] = bias->dtype == DType::Float32 ? float32_bias[column] : bfloat16_to_float(bfloat16_bias[column]);
        }
    }

    if (implementation.layer->load_model(ncnn::ModelBinFromMatArray(model_data)) != 0 || implementation.layer->create_pipeline(implementation.option) != 0)
        return {};
    implementation.pipeline_created = true;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context)
    {
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(vkdev);
        ncnn::Option upload_option = implementation.option;
        upload_option.blob_vkallocator = implementation.weight_allocator.get();
        upload_option.workspace_vkallocator = implementation.weight_allocator.get();
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        if (implementation.layer->upload_model(command, upload_option) != 0 || command.submit_and_wait() != 0)
            return {};
    }
#endif
    return linear;
#else
    (void)matrix;
    (void)bias;
    (void)device;
    (void)vulkan_device_index;
    (void)context_instance;
    return {};
#endif
}

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create_fused(const std::vector<const TensorData*>& matrices,
                                                                     const std::vector<const TensorData*>& biases, NcnnLinearDevice device,
                                                                     uint32_t vulkan_device_index,
                                                                     const NcnnVulkanContextInstancePtr& context_instance,
                                                                     uint64_t optimization_flags)
{
    if (matrices.empty() || matrices.size() != biases.size() || !matrices.front())
        return {};

    const DType dtype = matrices.front()->dtype;
    if (matrices.front()->shape.size() != 2 || (dtype != DType::Float32 && dtype != DType::BFloat16))
        return {};
    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix || matrix->dtype != dtype || matrix->shape.size() != 2 || matrix->shape[1] != input_columns || (bias != nullptr) != has_bias)
            return {};
        if (bias && (bias->dtype != dtype || bias->shape.size() != 1 || bias->shape[0] != matrix->shape[0]))
            return {};
        output_columns += matrix->shape[0];
        element_count += matrix->element_count();
    }
    if (output_columns > std::numeric_limits<uint32_t>::max())
        return {};

    TensorData fused_matrix;
    fused_matrix.dtype = dtype;
    fused_matrix.shape = {static_cast<uint32_t>(output_columns), input_columns};
    TensorData fused_bias;
    fused_bias.dtype = dtype;
    fused_bias.shape = {static_cast<uint32_t>(output_columns)};
    if (dtype == DType::Float32)
    {
        fused_matrix.float32_data.reserve(element_count);
        if (has_bias)
            fused_bias.float32_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index)
        {
            const std::span<const float> matrix_values = matrices[index]->float32_values();
            fused_matrix.float32_data.insert(fused_matrix.float32_data.end(), matrix_values.begin(), matrix_values.end());
            if (has_bias)
            {
                const std::span<const float> bias_values = biases[index]->float32_values();
                fused_bias.float32_data.insert(fused_bias.float32_data.end(), bias_values.begin(), bias_values.end());
            }
        }
    }
    else
    {
        fused_matrix.bfloat16_data.reserve(element_count);
        if (has_bias)
            fused_bias.bfloat16_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index)
        {
            const std::span<const uint16_t> matrix_values = matrices[index]->bfloat16_values();
            fused_matrix.bfloat16_data.insert(fused_matrix.bfloat16_data.end(), matrix_values.begin(), matrix_values.end());
            if (has_bias)
            {
                const std::span<const uint16_t> bias_values = biases[index]->bfloat16_values();
                fused_bias.bfloat16_data.insert(fused_bias.bfloat16_data.end(), bias_values.begin(), bias_values.end());
            }
        }
    }
    return create(
        fused_matrix,
        has_bias ? &fused_bias : nullptr,
        device,
        vulkan_device_index,
        context_instance,
        optimization_flags);
}

std::shared_ptr<IExpertVictimCache> create_vulkan_victim_cache(uint64_t capacity_bytes, uint32_t vulkan_device_index,
                                                               const NcnnVulkanContextInstancePtr& context_instance,
                                                               uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context =
        NcnnVulkanContext::acquire(
            vulkan_device_index,
            context_instance,
            optimization_flags);
    if (!context || capacity_bytes == 0)
        return {};
    return std::make_shared<VulkanExpertVictimCache>(context, capacity_bytes);
#else
    (void)capacity_bytes;
    (void)vulkan_device_index;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

uint32_t NcnnLinearOperator::vulkan_device_count() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::create_gpu_instance() != 0)
        return 0;
    return static_cast<uint32_t>(ncnn::get_gpu_count());
#else
    return 0;
#endif
}

uint64_t NcnnLinearOperator::vulkan_heap_budget_bytes() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() == 0)
        return 0;
    ncnn::VulkanDevice* device =
        ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    return device
               ? static_cast<uint64_t>(device->get_heap_budget()) * 1024 * 1024
               : 0;
#else
    return 0;
#endif
}

std::vector<VulkanDeviceCapabilities> NcnnLinearOperator::vulkan_device_capabilities()
{
    std::vector<VulkanDeviceCapabilities> result;
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::create_gpu_instance() != 0)
        return result;
    const int count = ncnn::get_gpu_count();
    result.reserve(static_cast<size_t>(count));
    for (int device_index = 0; device_index < count; ++device_index)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(device_index);
        VulkanDeviceCapabilities device;
        device.index = static_cast<uint32_t>(device_index);
        device.vendor_id = info.vendor_id();
        device.device_id = info.device_id();
        device.name = info.device_name();
        device.rough_score = info.rough_score();
        device.compute_queue_count = info.compute_queue_count();
        device.transfer_queue_count = info.transfer_queue_count();
        switch (info.type())
        {
        case 0: device.type = VulkanDeviceType::Discrete; break;
        case 1: device.type = VulkanDeviceType::Integrated; break;
        case 2: device.type = VulkanDeviceType::Virtual; break;
        case 3: device.type = VulkanDeviceType::Cpu; break;
        default: device.type = VulkanDeviceType::Unknown; break;
        }
        if (info.support_fp16_storage())
            device.flags |= VulkanDeviceFp16Storage;
        if (info.support_fp16_arithmetic())
            device.flags |= VulkanDeviceFp16Arithmetic;
        if (info.support_bf16_storage())
            device.flags |= VulkanDeviceBf16Storage;
        if (info.support_int8_storage())
            device.flags |= VulkanDeviceInt8Storage;
        if (info.support_int8_arithmetic())
            device.flags |= VulkanDeviceInt8Arithmetic;
        if (info.support_VK_KHR_shader_integer_dot_product())
            device.flags |= VulkanDeviceIntegerDotProduct;
        if (info.support_subgroup_ops() != 0)
            device.flags |= VulkanDeviceSubgroupOperations;
        if (info.support_cooperative_matrix())
            device.flags |= VulkanDeviceCooperativeMatrix;
        if (info.support_int8_cooperative_matrix())
            device.flags |= VulkanDeviceInt8CooperativeMatrix;
        if (info.unified_compute_transfer_queue())
            device.flags |= VulkanDeviceUnifiedComputeTransfer;
        if (info.resizable_bar_enabled())
            device.flags |= VulkanDeviceResizableBar;
        ncnn::VulkanDevice* profile_device = ncnn::get_gpu_device(device_index);
        if (profile_device)
        {
            device.heap_budget_bytes = static_cast<uint64_t>(profile_device->get_heap_budget()) * 1024 * 1024;
            device.heap_usage_bytes = static_cast<uint64_t>(profile_device->get_heap_usage()) * 1024 * 1024;
            device.heap_available_bytes = device.heap_usage_bytes < device.heap_budget_bytes
                                              ? device.heap_budget_bytes - device.heap_usage_bytes
                                              : 0;
        }
        if (device_index == ncnn::get_default_gpu_index())
        {
            device.flags |= VulkanDeviceSelected;
        }
        result.push_back(std::move(device));
    }
#endif
    return result;
}

NcnnVulkanExecutionSnapshot NcnnLinearOperator::vulkan_execution_snapshot(
    const NcnnVulkanContextInstancePtr& context_instance) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return context_instance ? context_instance->runtime_state_.snapshot()
                            : NcnnVulkanExecutionSnapshot{};
#else
    (void)context_instance;
    return {};
#endif
}

bool NcnnLinearOperator::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_USE_NCNN
    const Implementation& implementation = *implementation_;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context)
    {
        NcnnVulkanRuntimeState& runtime_state =
            implementation.vulkan_context->runtime_state();
        NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
        NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
        const DType output_dtype = output.dtype();
        if (!fill_staging_upload(
                input,
                transfer_slot.upload,
                transfer_slot.staging_allocator,
                runtime_state)
            || !prepare_staging_batch(
                transfer_slot.download,
                input.rows(),
                implementation.output_columns,
                transfer_slot.staging_allocator,
                runtime_state,
                output.element_size()))
            return false;

        output.reset(input.rows(), implementation.output_columns, false);
        std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        ncnn::VkCompute& command = *transfer_slot.command;
        if (transfer_slot.command_used)
        {
            if (command.reset() != 0)
                return false;
            ++runtime_state.command_buffer_reuses;
        }
        transfer_slot.command_used = true;
        ncnn::VkMat bottom_gpu;
        if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), bottom_gpu, command, vkdev, implementation.option, input.dtype()))
            return false;

        ncnn::VkMat top_gpu;
        if (implementation.layer->forward(bottom_gpu, top_gpu, command, implementation.option) != 0)
            return false;
        ncnn::VkMat download_gpu = top_gpu;
        if (top_gpu.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(top_gpu, unpacked, 1, command, implementation.option);
            download_gpu = unpacked;
        }
        if (!record_prepared_activation_staging_download(
                download_gpu,
                input.rows(),
                implementation.output_columns,
                transfer_slot.download,
                command,
                vkdev,
                implementation.option,
                output_dtype))
            return false;
        if (submit_compute_and_wait(command, runtime_state) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
            return false;
        ++runtime_state.dispatches;
        ++runtime_state.compute_submissions;
        ++runtime_state.batch_uploads;
        ++runtime_state.batch_downloads;
        return true;
    }
#endif

    const std::lock_guard<std::mutex> lock(
        implementation.forward_mutex);
    ncnn::Mat bottom(static_cast<int>(input.columns()), static_cast<int>(input.rows()), sizeof(float));
    if (bottom.empty())
        return false;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        std::copy_n(input.row(row_index), input.columns(), bottom.row<float>(static_cast<int>(row_index)));

    ncnn::Mat top;
    if (implementation.layer->forward(bottom, top, implementation.option) != 0 || top.empty()
        || top.total() * top.elempack != input.rows() * implementation.output_columns)
        return false;
    output.reset(input.rows(), implementation.output_columns, false);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = top.row<float>(static_cast<int>(row_index));
        std::copy_n(source, implementation.output_columns, output.row(row_index));
    }
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnLinearOperator::uses_vulkan() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return implementation_->vulkan_context != nullptr;
#else
    return false;
#endif
}

class NcnnVulkanBfloat16Operator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> cooperative_pipeline;
    mutable std::shared_ptr<ncnn::Pipeline> swiglu_down_pipeline;
    std::shared_ptr<ncnn::Pipeline> rms_norm_projection_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    ncnn::Option option;
    float rms_norm_epsilon = 0.0f;
    uint32_t rms_norm_output_subgroups = 0;
    uint32_t cooperative_tile_m = 0;
    uint32_t cooperative_tile_n = 0;
    uint32_t cooperative_tile_k = 0;
    uint32_t cooperative_subgroup_size = 0;
    bool cooperative_forced = false;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

#if NCNN_MOE_WITH_VULKAN
static constexpr char bfloat16_projection_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    sfp input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer bias_blob
{
    sfp bias_data[];
};
layout(binding = 3) writeonly buffer output_blob
{
    sfp output_data[];
};

layout(push_constant) uniform parameter
{
    uint input_columns;
    uint output_columns;
    uint block_count;
    uint token_count;
}
p;

float decode_bfloat16(uint value)
{
    return uintBitsToFloat(value << 16);
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum[32];
#endif

void main()
{
    const uint output_column = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const bool valid = output_column < p.output_columns && token < p.token_count;
    float sum = 0.0;
    if (valid)
    {
        const uint input_row = token * p.input_columns;
        const uint weight_row = output_column * p.input_columns;
        for (uint block = 0; block < p.block_count; ++block)
        {
            const uint column = block * 128 + lane * 4;
            if (column >= p.input_columns)
                continue;
            const uint first = packed_words[(weight_row + column) >> 1];
            const uint second = packed_words[(weight_row + column + 2) >> 1];
            sum += decode_bfloat16(first & 65535) * buffer_ld1(input_data, input_row + column);
            sum += decode_bfloat16(first >> 16) * buffer_ld1(input_data, input_row + column + 1);
            sum += decode_bfloat16(second & 65535) * buffer_ld1(input_data, input_row + column + 2);
            sum += decode_bfloat16(second >> 16) * buffer_ld1(input_data, input_row + column + 3);
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_sum = subgroupAdd(sum);
    if (valid && lane == 0)
        buffer_st1(output_data, token * p.output_columns + output_column, reduced_sum + buffer_ld1(bias_data, output_column));
#else
    partial_sum[lane] = sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[lane] += partial_sum[lane + stride];
        barrier();
    }
    if (valid && lane == 0)
        buffer_st1(output_data, token * p.output_columns + output_column, partial_sum[0] + buffer_ld1(bias_data, output_column));
#endif
}
)glsl";

// Mixed-storage BF16 GEMM for batched projections. Activations remain FP32 at
// the operator boundary and are rounded into BF16 shared-memory tiles inside
// this dispatch. The existing row-major BF16 weight buffer is reused directly,
// while accumulation and output remain FP32 for the following Vulkan operator.
static constexpr char bfloat16_cooperative_projection_shader[] = R"glsl(
#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require

layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;

layout(binding = 0) readonly buffer input_blob
{
    float input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer bias_blob
{
    float bias_data[];
};
layout(binding = 3) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint input_columns;
    uint output_columns;
    uint token_count;
    uint token_tile_count;
}
p;

shared uvec2 input_tile[(TILE_M * TILE_K + 3) / 4];
shared uvec2 weight_tile[(TILE_K * TILE_N + 3) / 4];
shared float result_tile[TILE_M * TILE_N];

vec4 load_input_values(uint token_base, uint column_base, uint packed_index)
{
    const uint scalar_index = packed_index * 4;
    const uint row = scalar_index / TILE_K;
    const uint column = scalar_index - row * TILE_K;
    vec4 values = vec4(0.0);
    if (token_base + row >= p.token_count)
        return values;
    const uint source = (token_base + row) * p.input_columns
                        + column_base + column;
    if (column_base + column < p.input_columns)
        values.x = input_data[source];
    if (column_base + column + 1 < p.input_columns)
        values.y = input_data[source + 1];
    if (column_base + column + 2 < p.input_columns)
        values.z = input_data[source + 2];
    if (column_base + column + 3 < p.input_columns)
        values.w = input_data[source + 3];
    return values;
}

uvec4 load_weight_bits(uint output_base, uint column_base, uint packed_index)
{
    const uint scalar_index = packed_index * 4;
    const uint row = scalar_index / TILE_N;
    const uint column = scalar_index - row * TILE_N;
    uvec4 values = uvec4(0);
    if (column_base + row >= p.input_columns)
        return values;
    const uint input_column = column_base + row;
    if (output_base + column < p.output_columns)
    {
        const uint word = packed_words[
            ((output_base + column) * p.input_columns + input_column) >> 1];
        values.x =
            (input_column & 1) == 0 ? word & 65535u : word >> 16;
    }
    if (output_base + column + 1 < p.output_columns)
    {
        const uint word = packed_words[
            ((output_base + column + 1) * p.input_columns + input_column) >> 1];
        values.y =
            (input_column & 1) == 0 ? word & 65535u : word >> 16;
    }
    if (output_base + column + 2 < p.output_columns)
    {
        const uint word = packed_words[
            ((output_base + column + 2) * p.input_columns + input_column) >> 1];
        values.z =
            (input_column & 1) == 0 ? word & 65535u : word >> 16;
    }
    if (output_base + column + 3 < p.output_columns)
    {
        const uint word = packed_words[
            ((output_base + column + 3) * p.input_columns + input_column) >> 1];
        values.w =
            (input_column & 1) == 0 ? word & 65535u : word >> 16;
    }
    return values;
}

void main()
{
    const uint lane = gl_SubgroupInvocationID;
    const uint tile_index = gl_WorkGroupID.x;
    const uint output_tile_index = tile_index / p.token_tile_count;
    const uint token_tile_index = tile_index
                                  - output_tile_index * p.token_tile_count;
    const uint token_base = token_tile_index * TILE_M;
    const uint output_base = output_tile_index * TILE_N;

    coopmat<float, gl_ScopeSubgroup, TILE_M, TILE_N,
            gl_MatrixUseAccumulator> sum =
        coopmat<float, gl_ScopeSubgroup, TILE_M, TILE_N,
                gl_MatrixUseAccumulator>(0.0);

    for (uint column_base = 0;
         column_base < p.input_columns;
         column_base += TILE_K)
    {
        for (uint index = lane;
             index < (TILE_M * TILE_K + 3) / 4;
             index += gl_SubgroupSize)
        {
            const vec4 values = load_input_values(
                token_base,
                column_base,
                index);
            input_tile[index] = uvec2(
                packBFloat2x16(values.xy),
                packBFloat2x16(values.zw));
        }
        for (uint index = lane;
             index < (TILE_K * TILE_N + 3) / 4;
             index += gl_SubgroupSize)
        {
            const uvec4 values = load_weight_bits(
                output_base,
                column_base,
                index);
            weight_tile[index] = uvec2(
                values.x | values.y << 16,
                values.z | values.w << 16);
        }
        barrier();

        coopmat<bfloat16_t, gl_ScopeSubgroup, TILE_M, TILE_K,
                gl_MatrixUseA> activation;
        coopmat<bfloat16_t, gl_ScopeSubgroup, TILE_K, TILE_N,
                gl_MatrixUseB> weight;
        coopMatLoad(
            activation,
            input_tile,
            0,
            (TILE_K + 3) / 4,
            gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(
            weight,
            weight_tile,
            0,
            (TILE_N + 3) / 4,
            gl_CooperativeMatrixLayoutRowMajor);
        sum = coopMatMulAdd(activation, weight, sum);
        barrier();
    }

    coopMatStore(
        sum,
        result_tile,
        0,
        TILE_N,
        gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (uint index = lane;
         index < TILE_M * TILE_N;
         index += gl_SubgroupSize)
    {
        const uint row = index / TILE_N;
        const uint column = index - row * TILE_N;
        if (token_base + row < p.token_count
            && output_base + column < p.output_columns)
        {
            output_data[(token_base + row) * p.output_columns
                        + output_base + column] =
                result_tile[index] + bias_data[output_base + column];
        }
    }
}
)glsl";

static constexpr char bfloat16_rms_norm_projection_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    sfp input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer bias_blob
{
    sfp bias_data[];
};
layout(binding = 3) readonly buffer norm_blob
{
    sfp norm_weight[];
};
layout(binding = 4) writeonly buffer output_blob
{
    sfp output_data[];
};

layout(push_constant) uniform parameter
{
    uint input_columns;
    uint output_columns;
    uint block_count;
    uint token_count;
    float norm_epsilon;
    uint output_subgroups;
}
p;

float decode_bfloat16(uint value)
{
    return uintBitsToFloat(value << 16);
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum[256];
#endif

void main()
{
    const uint local_id = gl_LocalInvocationID.x;
    const uint lane = local_id & 31;
    const uint subgroup = local_id >> 5;
    // ncnn injects the actual local size after GLSL compilation. Keep the
    // output-tile mapping in push constants instead of relying on the
    // gl_WorkGroupSize built-in surviving that injection path.
    const uint output_column =
        gl_WorkGroupID.x * p.output_subgroups + subgroup;
    const uint token = gl_WorkGroupID.y;
    const bool token_valid = token < p.token_count;
    const bool output_valid = token_valid && output_column < p.output_columns;
    float square_sum = 0.0;
    if (token_valid)
    {
        const uint input_row = token * p.input_columns;
        for (uint block = 0; block < p.block_count; ++block)
        {
            const uint column = block * 128 + lane * 4;
            if (column >= p.input_columns)
                continue;
            const float first = buffer_ld1(input_data, input_row + column);
            const float second = buffer_ld1(input_data, input_row + column + 1);
            const float third = buffer_ld1(input_data, input_row + column + 2);
            const float fourth = buffer_ld1(input_data, input_row + column + 3);
            square_sum += first * first + second * second
                          + third * third + fourth * fourth;
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_square_sum = subgroupAdd(square_sum);
    const float inverse_rms = 1.0 / sqrt(
        reduced_square_sum / float(p.input_columns) + p.norm_epsilon);
#else
    partial_sum[local_id] = square_sum;
    barrier();
    const uint partial_base = subgroup * 32;
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[partial_base + lane] += partial_sum[partial_base + lane + stride];
        barrier();
    }
    const float inverse_rms = 1.0 / sqrt(
        partial_sum[partial_base] / float(p.input_columns)
        + p.norm_epsilon);
#endif

    float sum = 0.0;
    if (output_valid)
    {
        const uint input_row = token * p.input_columns;
        const uint weight_row = output_column * p.input_columns;
        for (uint block = 0; block < p.block_count; ++block)
        {
            const uint column = block * 128 + lane * 4;
            if (column >= p.input_columns)
                continue;
            const uint first = packed_words[(weight_row + column) >> 1];
            const uint second = packed_words[(weight_row + column + 2) >> 1];
            sum += decode_bfloat16(first & 65535)
                       * buffer_ld1(input_data, input_row + column)
                       * buffer_ld1(norm_weight, column) * inverse_rms;
            sum += decode_bfloat16(first >> 16)
                       * buffer_ld1(input_data, input_row + column + 1)
                       * buffer_ld1(norm_weight, column + 1) * inverse_rms;
            sum += decode_bfloat16(second & 65535)
                       * buffer_ld1(input_data, input_row + column + 2)
                       * buffer_ld1(norm_weight, column + 2) * inverse_rms;
            sum += decode_bfloat16(second >> 16)
                       * buffer_ld1(input_data, input_row + column + 3)
                       * buffer_ld1(norm_weight, column + 3) * inverse_rms;
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_sum = subgroupAdd(sum);
    if (output_valid && lane == 0)
        buffer_st1(
            output_data,
            token * p.output_columns + output_column,
            reduced_sum + buffer_ld1(bias_data, output_column));
#else
    partial_sum[local_id] = sum;
    barrier();
    const uint output_partial_base = subgroup * 32;
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[output_partial_base + lane] += partial_sum[output_partial_base + lane + stride];
        barrier();
    }
    if (output_valid && lane == 0)
        buffer_st1(
            output_data,
            token * p.output_columns + output_column,
            partial_sum[output_partial_base]
                + buffer_ld1(bias_data, output_column));
#endif
}
)glsl";

static constexpr char bfloat16_swiglu_down_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer fused_input_blob
{
    sfp input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer bias_blob
{
    sfp bias_data[];
};
layout(binding = 3) writeonly buffer output_blob
{
    sfp output_data[];
};

layout(push_constant) uniform parameter
{
    uint intermediate_columns;
    uint fused_columns;
    uint output_columns;
    uint token_count;
    uint apply_router_gate;
}
p;

float decode_bfloat16(uint value)
{
    return uintBitsToFloat(value << 16);
}

float sigmoid(float value)
{
    return 1.0 / (1.0 + exp(-value));
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum[32];
#endif

void main()
{
    const uint output_column = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const bool valid = output_column < p.output_columns && token < p.token_count;
    float sum = 0.0;
    if (valid)
    {
        const uint input_row = token * p.fused_columns;
        const uint weight_row = output_column * p.intermediate_columns;
        for (uint block = 0; block < p.intermediate_columns / 128; ++block)
        {
            const uint column = block * 128 + lane * 4;
            const uint first = packed_words[(weight_row + column) >> 1];
            const uint second = packed_words[(weight_row + column + 2) >> 1];
            const float first_gate = buffer_ld1(input_data, input_row + column);
            const float first_up = buffer_ld1(input_data, input_row + p.intermediate_columns + column);
            const float second_gate = buffer_ld1(input_data, input_row + column + 2);
            const float second_up = buffer_ld1(input_data, input_row + p.intermediate_columns + column + 2);
            const float first_value = first_gate * sigmoid(first_gate) * first_up;
            const float second_value = second_gate * sigmoid(second_gate) * second_up;
            const float third_gate = buffer_ld1(input_data, input_row + column + 1);
            const float third_up = buffer_ld1(input_data, input_row + p.intermediate_columns + column + 1);
            const float fourth_gate = buffer_ld1(input_data, input_row + column + 3);
            const float fourth_up = buffer_ld1(input_data, input_row + p.intermediate_columns + column + 3);
            sum += decode_bfloat16(first & 65535) * first_value;
            sum += decode_bfloat16(first >> 16) * (third_gate * sigmoid(third_gate) * third_up);
            sum += decode_bfloat16(second & 65535) * second_value;
            sum += decode_bfloat16(second >> 16) * (fourth_gate * sigmoid(fourth_gate) * fourth_up);
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_sum = subgroupAdd(sum);
    if (valid && lane == 0)
    {
        float result = reduced_sum + buffer_ld1(bias_data, output_column);
        if (p.apply_router_gate != 0)
            result *= sigmoid(buffer_ld1(input_data, token * p.fused_columns + p.intermediate_columns * 2));
        buffer_st1(output_data, token * p.output_columns + output_column, result);
    }
#else
    partial_sum[lane] = sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[lane] += partial_sum[lane + stride];
        barrier();
    }
    if (valid && lane == 0)
    {
        float result = partial_sum[0] + buffer_ld1(bias_data, output_column);
        if (p.apply_router_gate != 0)
            result *= sigmoid(buffer_ld1(input_data, token * p.fused_columns + p.intermediate_columns * 2));
        buffer_st1(output_data, token * p.output_columns + output_column, result);
    }
#endif
}
)glsl";

// One workgroup owns the recurrent state.  The token loop is intentionally
// sequential, while convolution channels and value heads remain parallel.
// This keeps the state transition on the device without changing the
// cross-token ordering required by DeltaNet.
static constexpr char gated_delta_net_shader[] = R"glsl(
#version 450

layout(binding = 0) buffer fused_input_blob
{
    float fused_input[];
};
layout(binding = 1) buffer convolution_state_blob
{
    float convolution_state[];
};
layout(binding = 2) buffer recurrent_state_blob
{
    float recurrent_state[];
};
layout(binding = 3) readonly buffer convolution_weight_blob
{
    float convolution_weight[];
};
layout(binding = 4) readonly buffer time_bias_blob
{
    float time_bias[];
};
layout(binding = 5) readonly buffer decay_log_blob
{
    float decay_log[];
};
layout(binding = 6) readonly buffer norm_weight_blob
{
    float norm_weight[];
};
layout(binding = 7) buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint convolution_size;
    uint value_size;
    uint fused_columns;
    uint head_count;
    uint kv_head_count;
    uint head_dimension;
    uint value_head_dimension;
    uint kernel_size;
    uint token_count;
    float norm_epsilon;
}
p;

float sigmoid(float value)
{
    if (value >= 0.0)
        return 1.0 / (1.0 + exp(-value));
    const float exponential = exp(value);
    return exponential / (1.0 + exponential);
}

float softplus(float value)
{
    if (value > 20.0)
        return value;
    if (value < -20.0)
        return exp(value);
    return log(1.0 + exp(value));
}

shared float partial_sum[128];

void main()
{
    const uint lane = gl_LocalInvocationID.x;
    const uint key_size = p.kv_head_count * p.head_dimension;
    const uint head_ratio = p.head_count / p.kv_head_count;
    const float query_scale = 1.0 / sqrt(float(p.head_dimension));

    for (uint token = 0; token < p.token_count; ++token)
    {
        const uint input_row = token * p.fused_columns;

        for (uint channel = lane; channel < p.convolution_size; channel += 128)
        {
            const uint state_base = channel * p.kernel_size;
            for (uint tap = 0; tap + 1 < p.kernel_size; ++tap)
                convolution_state[state_base + tap] = convolution_state[state_base + tap + 1];
            convolution_state[state_base + p.kernel_size - 1] = fused_input[input_row + channel];
            float sum = 0.0;
            for (uint tap = 0; tap < p.kernel_size; ++tap)
                sum += convolution_state[state_base + tap]
                       * convolution_weight[state_base + tap];
            fused_input[input_row + channel] = sum * sigmoid(sum);
        }
        barrier();

        for (uint key_head = lane; key_head < p.kv_head_count; key_head += 128)
        {
            const uint query_base = input_row + key_head * p.head_dimension;
            const uint key_base = input_row + key_size + key_head * p.head_dimension;
            float query_square_sum = 0.0;
            float key_square_sum = 0.0;
            for (uint column = 0; column < p.head_dimension; ++column)
            {
                const float query = fused_input[query_base + column];
                const float key = fused_input[key_base + column];
                query_square_sum += query * query;
                key_square_sum += key * key;
            }
            const float query_inverse_norm = 1.0 / sqrt(query_square_sum + 1e-6);
            const float key_inverse_norm = 1.0 / sqrt(key_square_sum + 1e-6);
            for (uint column = 0; column < p.head_dimension; ++column)
            {
                fused_input[query_base + column] *= query_inverse_norm;
                fused_input[key_base + column] *= key_inverse_norm;
            }
        }
        barrier();

        for (uint value_head = 0; value_head < p.head_count; ++value_head)
        {
            const uint key_head = value_head / head_ratio;
            const uint query_base = input_row + key_head * p.head_dimension;
            const uint key_base = input_row + key_size + key_head * p.head_dimension;
            const uint value_base = input_row + key_size * 2
                                    + value_head * p.value_head_dimension;
            const uint recurrent_base = value_head * p.head_dimension
                                        * p.value_head_dimension;
            const uint output_base = token * p.value_size
                                     + value_head * p.value_head_dimension;
            const float beta = sigmoid(fused_input[input_row + p.convolution_size
                                                   + p.value_size + value_head]);
            const uint alpha_base = input_row + p.convolution_size + p.value_size
                                    + p.head_count;
            const float decay = exp(
                -exp(decay_log[value_head])
                * softplus(fused_input[alpha_base + value_head] + time_bias[value_head]));

            for (uint value_column = lane;
                 value_column < p.value_head_dimension;
                 value_column += 128)
            {
                float memory_value = 0.0;
                for (uint key_column = 0; key_column < p.head_dimension; ++key_column)
                {
                    memory_value += recurrent_state[recurrent_base
                                                       + key_column * p.value_head_dimension
                                                       + value_column]
                                    * decay * fused_input[key_base + key_column];
                }
                const float value = fused_input[value_base + value_column];
                const float delta = (value - memory_value) * beta;
                for (uint key_column = 0; key_column < p.head_dimension; ++key_column)
                {
                    recurrent_state[recurrent_base
                                    + key_column * p.value_head_dimension
                                    + value_column] = recurrent_state[recurrent_base
                                                                       + key_column * p.value_head_dimension
                                                                       + value_column] * decay
                                                       + fused_input[key_base + key_column] * delta;
                }
                float output_value = 0.0;
                for (uint key_column = 0; key_column < p.head_dimension; ++key_column)
                {
                    output_value += recurrent_state[recurrent_base
                                                    + key_column * p.value_head_dimension
                                                    + value_column]
                                   * fused_input[query_base + key_column];
                }
                output_data[output_base + value_column] = output_value * query_scale;
            }

            float square_sum = 0.0;
            for (uint value_column = lane;
                 value_column < p.value_head_dimension;
                 value_column += 128)
            {
                const float value = output_data[output_base + value_column];
                square_sum += value * value;
            }
            partial_sum[lane] = square_sum;
            barrier();
            for (uint stride = 64; stride > 0; stride >>= 1)
            {
                if (lane < stride)
                    partial_sum[lane] += partial_sum[lane + stride];
                barrier();
            }
            const float inverse_rms = 1.0 / sqrt(
                partial_sum[0] / float(p.value_head_dimension)
                + p.norm_epsilon);
            const uint gate_base = input_row + p.convolution_size;
            for (uint value_column = lane;
                 value_column < p.value_head_dimension;
                 value_column += 128)
            {
                const float gate = fused_input[gate_base
                                               + value_head * p.value_head_dimension
                                               + value_column];
                output_data[output_base + value_column] *= inverse_rms
                    * norm_weight[value_column] * gate * sigmoid(gate);
            }
            barrier();
        }
    }
}
)glsl";

static bool create_gated_delta_net_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            gated_delta_net_shader,
            static_cast<int>(sizeof(gated_delta_net_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(gated_delta_net_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(gated_delta_net_shader, 0, destination);
    return true;
}

static bool create_bfloat16_projection_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant =
        (option.use_subgroup_ops ? 1u : 0u) * 3
        + vulkan_activation_storage_variant(option);
    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            bfloat16_projection_shader,
            static_cast<int>(sizeof(bfloat16_projection_shader) - 1),
            compile_option,
            shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(
        bfloat16_projection_shader,
        shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination =
        std::shared_ptr<ncnn::Pipeline>(
            pipeline.release(),
            [context](ncnn::Pipeline* value) {
                const std::lock_guard<std::mutex> lock(
                    context->command_mutex());
                delete value;
            });
    context->cache_pipeline(
        bfloat16_projection_shader,
        shader_variant,
        destination);
    return true;
}

static bool create_bfloat16_cooperative_projection_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    uint32_t input_columns,
    uint32_t output_columns,
    std::shared_ptr<ncnn::Pipeline>& destination,
    uint32_t& tile_m,
    uint32_t& tile_n,
    uint32_t& tile_k,
    uint32_t& subgroup_size)
{
    ncnn::VulkanDevice* device = context->device();
    // The cooperative shader uses the BF16 storage macro family for its
    // matrix tiles. Activation storage must use the scalar/subgroup path until
    // activation and weight storage types are independently parameterized.
    if (vulkan_activation_storage_variant(option) != 0)
        return false;
    if (!device->info.support_VK_KHR_cooperative_matrix()
        || !device->info.support_bf16_cooperative_matrix()
        || !device->info.support_bf16_storage())
    {
        return false;
    }

    int selected_m = 0;
    int selected_n = 0;
    int selected_k = 0;
    int selected_subgroup_size = 0;
    device->info.get_optimal_cooperative_matrix_mnk(
        16,
        static_cast<int>(output_columns),
        static_cast<int>(input_columns),
        VK_COMPONENT_TYPE_BFLOAT16_KHR,
        VK_COMPONENT_TYPE_FLOAT32_KHR,
        VK_SCOPE_SUBGROUP_KHR,
        selected_m,
        selected_n,
        selected_k,
        selected_subgroup_size);
    if (selected_m <= 0
        || selected_n <= 0
        || selected_k <= 0
        || selected_subgroup_size <= 0
        || selected_m % 4 != 0
        || selected_n % 4 != 0
        || selected_k % 4 != 0)
    {
        return false;
    }
    const uint64_t shared_bytes =
        (static_cast<uint64_t>(selected_m) * selected_k
         + static_cast<uint64_t>(selected_k) * selected_n)
            * sizeof(uint16_t)
        + static_cast<uint64_t>(selected_m) * selected_n
              * sizeof(float);
    if (shared_bytes > device->info.max_shared_memory_size()
        || selected_m > std::numeric_limits<uint16_t>::max()
        || selected_n > std::numeric_limits<uint16_t>::max()
        || selected_k > std::numeric_limits<uint16_t>::max()
        || selected_subgroup_size > std::numeric_limits<uint16_t>::max())
    {
        return false;
    }

    ncnn::Option compile_option = option;
    compile_option.vulkan_device_index = device->info.device_index();
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_storage = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    compile_option.use_bf16_storage = true;
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            bfloat16_cooperative_projection_shader,
            static_cast<int>(
                sizeof(bfloat16_cooperative_projection_shader) - 1),
            compile_option,
            0);
    if (!spirv || spirv->empty())
        return false;

    const uint64_t pipeline_key =
        static_cast<uint64_t>(selected_m)
        | static_cast<uint64_t>(selected_n) << 16
        | static_cast<uint64_t>(selected_k) << 32
        | static_cast<uint64_t>(selected_subgroup_size) << 48;
    destination = context->find_pipeline(
        bfloat16_cooperative_projection_shader,
        pipeline_key);
    if (destination)
    {
        tile_m = static_cast<uint32_t>(selected_m);
        tile_n = static_cast<uint32_t>(selected_n);
        tile_k = static_cast<uint32_t>(selected_k);
        subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
        return true;
    }

    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = static_cast<uint32_t>(selected_m);
    specializations[1].u32 = static_cast<uint32_t>(selected_n);
    specializations[2].u32 = static_cast<uint32_t>(selected_k);
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(selected_subgroup_size);
    pipeline->set_local_size_xyz(selected_subgroup_size, 1, 1);
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(
                context->command_mutex());
            delete value;
        });
    context->cache_pipeline(
        bfloat16_cooperative_projection_shader,
        pipeline_key,
        destination);
    tile_m = static_cast<uint32_t>(selected_m);
    tile_n = static_cast<uint32_t>(selected_n);
    tile_k = static_cast<uint32_t>(selected_k);
    subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
    return true;
}

static bool create_bfloat16_rms_norm_projection_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination,
    uint32_t& output_subgroups)
{
    const size_t shader_variant =
        (option.use_subgroup_ops ? 1u : 0u) * 3
        + vulkan_activation_storage_variant(option);
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            bfloat16_rms_norm_projection_shader,
            static_cast<int>(
                sizeof(bfloat16_rms_norm_projection_shader) - 1),
            option,
            shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const uint32_t local_size =
        device->info.max_workgroup_invocations() >= 256
            && device->info.max_workgroup_size_x() >= 256
            ? 256
            : 128;
    output_subgroups = local_size / 32;
    destination = context->find_pipeline(
        bfloat16_rms_norm_projection_shader,
        shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(static_cast<int>(local_size), 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination =
        std::shared_ptr<ncnn::Pipeline>(
            pipeline.release(),
            [context](ncnn::Pipeline* value) {
                const std::lock_guard<std::mutex> lock(
                    context->command_mutex());
                delete value;
            });
    context->cache_pipeline(
        bfloat16_rms_norm_projection_shader,
        shader_variant,
        destination);
    return true;
}

static bool create_bfloat16_swiglu_down_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant =
        (option.use_subgroup_ops ? 1u : 0u) * 3
        + vulkan_activation_storage_variant(option);
    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            bfloat16_swiglu_down_shader,
            static_cast<int>(sizeof(bfloat16_swiglu_down_shader) - 1),
            compile_option,
            shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(
        bfloat16_swiglu_down_shader,
        shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(
        bfloat16_swiglu_down_shader,
        shader_variant,
        destination);
    return true;
}

static bool prepare_bfloat16_upload(
    std::span<const uint16_t> source,
    ncnn::Mat& destination)
{
    if (source.empty()
        || source.size() > static_cast<size_t>(std::numeric_limits<int>::max()) * 2)
    {
        return false;
    }
    const size_t word_count = (source.size() + 1) / 2;
    if (word_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    destination.create(static_cast<int>(word_count), sizeof(uint32_t));
    if (destination.empty())
        return false;
    uint32_t* words = static_cast<uint32_t*>(destination.data);
    for (size_t index = 0; index < word_count; ++index)
    {
        const size_t first = index * 2;
        const uint32_t low = source[first];
        const uint32_t high =
            first + 1 < source.size() ? source[first + 1] : 0;
        words[index] = low | high << 16;
    }
    return true;
}
#endif

NcnnVulkanBfloat16Operator::NcnnVulkanBfloat16Operator()
    : implementation_(new Implementation)
{
}

NcnnVulkanBfloat16Operator::~NcnnVulkanBfloat16Operator() = default;

bool NcnnVulkanBfloat16Operator::prepare_rms_norm(
    const TensorData& weight,
    float epsilon,
    float weight_offset)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context
        || implementation.rms_norm_projection_pipeline
        || epsilon <= 0.0f
        || weight.shape != std::vector<uint32_t>{implementation.input_columns}
        || (weight.dtype != DType::Float32
            && weight.dtype != DType::BFloat16))
    {
        return false;
    }

    ncnn::Mat values;
    values.create(static_cast<int>(implementation.input_columns), sizeof(float));
    if (values.empty())
        return false;
    float* destination = static_cast<float*>(values.data);
    if (weight.dtype == DType::Float32)
    {
        const std::span<const float> source = weight.float32_values();
        if (source.size() != implementation.input_columns)
            return false;
        for (uint32_t index = 0; index < implementation.input_columns; ++index)
            destination[index] = source[index] + weight_offset;
    }
    else
    {
        const std::span<const uint16_t> source = weight.bfloat16_values();
        if (source.size() != implementation.input_columns)
            return false;
        for (uint32_t index = 0; index < implementation.input_columns; ++index)
            destination[index] = bfloat16_to_float(source[index]) + weight_offset;
    }

    if (!create_bfloat16_rms_norm_projection_pipeline(
            implementation.vulkan_context,
            implementation.option,
            implementation.rms_norm_projection_pipeline,
            implementation.rms_norm_output_subgroups))
        return false;

    implementation.weight_staging_allocator.reset(
        new ncnn::VkWeightStagingAllocator(
            implementation.vulkan_context->device()));
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator =
        implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(implementation.vulkan_context->device());
        command.record_upload(
            values,
            implementation.rms_norm_weight,
            upload_option);
        uploaded = !implementation.rms_norm_weight.empty()
                   && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
    {
        implementation.rms_norm_projection_pipeline.reset();
        return false;
    }
    implementation.rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    (void)weight_offset;
    return false;
#endif
}

bool NcnnVulkanBfloat16Operator::has_rms_norm_chain() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return implementation_->rms_norm_projection_pipeline != nullptr
           && !implementation_->rms_norm_weight.empty();
#else
    return false;
#endif
}

bool NcnnVulkanBfloat16Operator::forward_rms_norm_chain(
    const ActivationBuffer& input,
    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || !implementation.rms_norm_projection_pipeline
        || implementation.rms_norm_weight.empty()
        || implementation.rms_norm_output_subgroups == 0
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_input_enabled(
            *implementation.vulkan_context,
            static_cast<size_t>(input.rows())
                * implementation.input_columns * sizeof(float),
            input.dtype());
    const bool direct_host_output =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_output_enabled(
            *implementation.vulkan_context,
            static_cast<size_t>(input.rows())
                * implementation.output_columns * sizeof(float),
            output.dtype());
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            implementation.output_columns,
            transfer_slot.staging_allocator,
            runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            ++runtime_state.attention_staging_failures;
            return false;
        }
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(
            transfer_slot.upload,
            input_gpu,
            command,
            implementation.vulkan_context->device(),
            implementation.option,
            input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(implementation.output_columns),
            static_cast<int>(input.rows()),
            vulkan_activation_element_size(implementation.option),
            implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;
    const std::vector<ncnn::VkMat> bindings = {
        input_gpu,
        implementation.packed,
        implementation.bias,
        implementation.rms_norm_weight,
        output_gpu};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    constants[4].f = implementation.rms_norm_epsilon;
    constants[5].u32 = implementation.rms_norm_output_subgroups;
    const uint64_t dispatch_width =
        ((static_cast<uint64_t>(implementation.output_columns)
          + implementation.rms_norm_output_subgroups - 1)
         / implementation.rms_norm_output_subgroups)
        * implementation.rms_norm_output_subgroups * 32;
    if (dispatch_width > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(dispatch_width);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(
        implementation.rms_norm_projection_pipeline.get(),
        bindings,
        constants,
        dispatcher);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
            output_gpu,
            input.rows(),
            implementation.output_columns,
            transfer_slot.download,
            command,
            implementation.vulkan_context->device(),
            implementation.option,
            output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.rms_norm_linear_fusions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool try_fused_rms_norm_linear(
    const CompiledOperator& operator_entry,
    const ActivationBuffer& input,
    ActivationBuffer& output)
{
    return operator_entry.bfloat16
           && operator_entry.bfloat16->forward_rms_norm_chain(
               input,
               output);
}

std::shared_ptr<NcnnVulkanBfloat16Operator>
NcnnVulkanBfloat16Operator::create(
    const TensorData& matrix,
    const TensorData* bias,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::BFloat16
        || matrix.shape.size() != 2
        || matrix.shape[0] == 0
        || matrix.shape[1] == 0
        || matrix.shape[1] % 4 != 0
        || matrix.shape[0]
               > static_cast<uint32_t>(
                   std::numeric_limits<int>::max() / 32)
        || matrix.shape[1]
               > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const std::span<const uint16_t> weights = matrix.bfloat16_values();
    if (weights.size() != matrix.element_count())
        return {};
    if (bias
        && (bias->shape.size() != 1
            || bias->shape[0] != output_columns
            || (bias->dtype != DType::Float32
                && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<NcnnVulkanBfloat16Operator> result(
        new NcnnVulkanBfloat16Operator);
    Implementation& implementation = *result->implementation_;
    implementation.input_columns = input_columns;
    implementation.output_columns = output_columns;
    implementation.block_count = (input_columns + 127) / 128;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context =
        NcnnVulkanContext::acquire(
            vulkan_device_index,
            context_instance,
            optimization_flags);
    if (!implementation.vulkan_context)
        return {};
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    if (device->info.subgroup_size() != 32)
        return {};
    implementation.option.vulkan_device_index = device->info.device_index();
    implementation.option.use_vulkan_compute = true;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage =
        vulkan_fp16_activations_enabled(implementation.optimization_flags)
        && device->info.support_fp16_storage();
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;
    implementation.option.blob_vkallocator =
        implementation.vulkan_context->blob_allocator();
    implementation.option.workspace_vkallocator =
        implementation.vulkan_context->blob_allocator();
    implementation.option.staging_vkallocator =
        implementation.vulkan_context->staging_allocator();
    implementation.option.use_cooperative_matrix =
        device->info.support_cooperative_matrix();
    implementation.option.use_subgroup_ops =
        device->info.support_subgroup_ops();

    const uint64_t weight_bytes =
        static_cast<uint64_t>(weights.size()) * sizeof(uint16_t);
    const uint64_t preferred_weight_bytes =
        weight_bytes
        + static_cast<uint64_t>(output_columns) * sizeof(float)
        + static_cast<uint64_t>(input_columns) * sizeof(float);
    if (preferred_weight_bytes
        > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return {};
    }
    implementation.weight_allocator.reset(
        new ncnn::VkWeightAllocator(
            device,
            static_cast<size_t>(preferred_weight_bytes)));
    implementation.weight_staging_allocator.reset(
        new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat packed;
    ncnn::Mat biases;
    if (!prepare_bfloat16_upload(weights, packed))
        return {};
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (biases.empty())
        return {};
    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        if (values.size() != output_columns)
            return {};
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values =
            bias->bfloat16_values();
        if (values.size() != output_columns)
            return {};
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(values[index]);
    }

    {
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        if (!create_bfloat16_projection_pipeline(
                implementation.vulkan_context,
                implementation.option,
                implementation.pipeline))
        {
            return {};
        }
        const VulkanBfloat16CooperativeMatrixPolicy cooperative_policy =
            vulkan_bfloat16_cooperative_matrix_policy(implementation.optimization_flags);
        implementation.cooperative_forced =
            cooperative_policy
            == VulkanBfloat16CooperativeMatrixPolicy::Forced;
        if (cooperative_policy
            != VulkanBfloat16CooperativeMatrixPolicy::Disabled)
        {
            create_bfloat16_cooperative_projection_pipeline(
                implementation.vulkan_context,
                implementation.option,
                input_columns,
                output_columns,
                implementation.cooperative_pipeline,
                implementation.cooperative_tile_m,
                implementation.cooperative_tile_n,
                implementation.cooperative_tile_k,
                implementation.cooperative_subgroup_size);
        }
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator =
        implementation.weight_allocator.get();
    upload_option.workspace_vkallocator =
        implementation.weight_allocator.get();
    upload_option.staging_vkallocator =
        implementation.weight_staging_allocator.get();
    ncnn::Option packed_upload_option = upload_option;
    packed_upload_option.use_fp16_storage = false;
    packed_upload_option.use_bf16_storage = false;
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(
            packed,
            implementation.packed,
            packed_upload_option);
        command.record_upload(
            biases,
            implementation.bias,
            upload_option);
        uploaded =
            !implementation.packed.empty()
            && !implementation.bias.empty()
            && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

std::shared_ptr<NcnnVulkanBfloat16Operator>
NcnnVulkanBfloat16Operator::create_fused(
    const std::vector<const TensorData*>& matrices,
    const std::vector<const TensorData*>& biases,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags)
{
    if (matrices.empty()
        || matrices.size() != biases.size()
        || !matrices.front()
        || matrices.front()->dtype != DType::BFloat16
        || matrices.front()->shape.size() != 2)
    {
        return {};
    }
    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix
            || matrix->dtype != DType::BFloat16
            || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns
            || (bias != nullptr) != has_bias
            || (bias
                && (bias->shape.size() != 1
                    || bias->shape[0] != matrix->shape[0]
                    || (bias->dtype != DType::Float32
                        && bias->dtype != DType::BFloat16))))
        {
            return {};
        }
        output_columns += matrix->shape[0];
        element_count += matrix->element_count();
    }
    if (output_columns > std::numeric_limits<uint32_t>::max()
        || element_count > std::numeric_limits<size_t>::max())
    {
        return {};
    }

    TensorData fused_matrix;
    fused_matrix.dtype = DType::BFloat16;
    fused_matrix.shape = {
        static_cast<uint32_t>(output_columns),
        input_columns};
    fused_matrix.bfloat16_data.reserve(
        static_cast<size_t>(element_count));
    TensorData fused_bias;
    fused_bias.dtype = DType::Float32;
    fused_bias.shape = {static_cast<uint32_t>(output_columns)};
    if (has_bias)
        fused_bias.float32_data.reserve(
            static_cast<size_t>(output_columns));
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const std::span<const uint16_t> matrix_values =
            matrices[index]->bfloat16_values();
        if (matrix_values.size() != matrices[index]->element_count())
            return {};
        fused_matrix.bfloat16_data.insert(
            fused_matrix.bfloat16_data.end(),
            matrix_values.begin(),
            matrix_values.end());
        if (!has_bias)
            continue;
        const TensorData& bias = *biases[index];
        if (bias.dtype == DType::Float32)
        {
            const std::span<const float> bias_values =
                bias.float32_values();
            fused_bias.float32_data.insert(
                fused_bias.float32_data.end(),
                bias_values.begin(),
                bias_values.end());
        }
        else
        {
            const std::span<const uint16_t> bias_values =
                bias.bfloat16_values();
            for (uint16_t value : bias_values)
                fused_bias.float32_data.push_back(
                    bfloat16_to_float(value));
        }
    }
    return create(
        fused_matrix,
        has_bias ? &fused_bias : nullptr,
        vulkan_device_index,
        context_instance,
        optimization_flags);
}

#if NCNN_MOE_WITH_VULKAN
static bool record_bfloat16_projection(
    const std::shared_ptr<ncnn::Pipeline>& scalar_pipeline,
    const std::shared_ptr<ncnn::Pipeline>& cooperative_pipeline,
    const ncnn::VkMat& input,
    const ncnn::VkMat& packed,
    const ncnn::VkMat& bias,
    ncnn::VkMat& output,
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t block_count,
    uint32_t token_count,
    uint32_t cooperative_tile_m,
    uint32_t cooperative_tile_n,
    uint32_t cooperative_tile_k,
    uint32_t cooperative_subgroup_size,
    bool cooperative_forced,
    ncnn::VkCompute& command)
{
    std::vector<ncnn::VkMat> bindings = {
        input,
        packed,
        bias,
        output};
    const uint64_t cooperative_output_work =
        static_cast<uint64_t>(token_count) * output_columns;
    if (cooperative_pipeline
        && token_count >= cooperative_tile_m
        && input_columns >= cooperative_tile_k
        && output_columns >= cooperative_tile_n
        && (cooperative_forced
            || cooperative_output_work >= UINT64_C(65536)))
    {
        const uint64_t token_tile_count =
            (token_count + cooperative_tile_m - 1)
            / cooperative_tile_m;
        const uint64_t output_tile_count =
            (output_columns + cooperative_tile_n - 1)
            / cooperative_tile_n;
        const uint64_t invocation_count =
            token_tile_count * output_tile_count
            * cooperative_subgroup_size;
        if (invocation_count
            <= static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].u32 = input_columns;
            constants[1].u32 = output_columns;
            constants[2].u32 = token_count;
            constants[3].u32 = static_cast<uint32_t>(token_tile_count);
            ncnn::VkMat dispatcher;
            dispatcher.w = static_cast<int>(invocation_count);
            dispatcher.h = 1;
            dispatcher.c = 1;
            command.record_pipeline_readonly(
                cooperative_pipeline.get(),
                bindings,
                {1, 1, 1, 0},
                constants,
                dispatcher);
            return true;
        }
    }

    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = input_columns;
    constants[1].u32 = output_columns;
    constants[2].u32 = block_count;
    constants[3].u32 = token_count;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(output_columns * 32);
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(
        scalar_pipeline.get(),
        bindings,
        {1, 1, 1, 0},
        constants,
        dispatcher);
    return false;
}
#endif

bool NcnnVulkanBfloat16Operator::forward(
    const ActivationBuffer& input,
    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();
    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_input_enabled(
            *implementation.vulkan_context,
            static_cast<size_t>(input.rows())
                * implementation.input_columns * sizeof(float),
            input.dtype());
    const bool direct_host_output =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_output_enabled(
            *implementation.vulkan_context,
            static_cast<size_t>(input.rows())
                * implementation.output_columns * sizeof(float),
            output.dtype());
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            implementation.output_columns,
            transfer_slot.staging_allocator,
            runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            ++runtime_state.attention_staging_failures;
            return false;
        }
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(
                 transfer_slot.upload,
                 input_gpu,
                 command,
                 implementation.vulkan_context->device(),
                 implementation.option,
                 input.dtype()))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(implementation.output_columns),
            static_cast<int>(input.rows()),
            vulkan_activation_element_size(implementation.option),
            implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    const bool used_cooperative_matrix = record_bfloat16_projection(
        implementation.pipeline,
        implementation.cooperative_pipeline,
        input_gpu,
        implementation.packed,
        implementation.bias,
        output_gpu,
        implementation.input_columns,
        implementation.output_columns,
        implementation.block_count,
        static_cast<uint32_t>(input.rows()),
        implementation.cooperative_tile_m,
        implementation.cooperative_tile_n,
        implementation.cooperative_tile_k,
        implementation.cooperative_subgroup_size,
        implementation.cooperative_forced,
        command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
                output_gpu,
                input.rows(),
                implementation.output_columns,
                transfer_slot.download,
                command,
                implementation.vulkan_context->device(),
                implementation.option,
                output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(
            transfer_slot.download,
            output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    if (used_cooperative_matrix)
    {
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanBfloat16Operator::forward_parallel(
    const ActivationBuffer& input,
    const NcnnVulkanBfloat16Operator& parallel_operator,
    ActivationBuffer& output,
    ActivationBuffer& parallel_output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *implementation_;
    const Implementation& parallel = *parallel_operator.implementation_;
    if (!first.vulkan_context
        || !parallel.vulkan_context
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline
        || !parallel.pipeline
        || vulkan_activation_storage_variant(first.option)
               != vulkan_activation_storage_variant(parallel.option)
        || input.rows() == 0
        || input.columns() != first.input_columns
        || input.columns() != parallel.input_columns
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    NcnnVulkanRuntimeState& runtime_state =
        first.vulkan_context->runtime_state();
    NcnnVulkanTransferLease transfer_lease =
        first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    ncnn::VkMat parallel_download;
    const bool direct_host_input =
        vulkan_activation_storage_variant(first.option) == 0
        && direct_host_input_enabled(
            *first.vulkan_context,
            static_cast<size_t>(input.rows()) * first.input_columns
                * sizeof(float),
            input.dtype());
    const bool direct_host_output =
        vulkan_activation_storage_variant(first.option) == 0
        && direct_host_output_enabled(
            *first.vulkan_context,
            static_cast<size_t>(input.rows()) * first.output_columns
                * sizeof(float),
            output.dtype());
    const bool parallel_direct_host_output =
        vulkan_activation_storage_variant(parallel.option) == 0
        && direct_host_output_enabled(
            *parallel.vulkan_context,
            static_cast<size_t>(input.rows()) * parallel.output_columns
                * sizeof(float),
            parallel_output.dtype());
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            first.output_columns,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_batch(
            parallel_download,
            input.rows(),
            parallel.output_columns,
            transfer_slot.staging_allocator,
            runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), first.output_columns, false);
    parallel_output.reset(input.rows(), parallel.output_columns, false);

    std::unique_lock<std::mutex> lock(
        first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(
                 transfer_slot.upload,
                 input_gpu,
                 command,
                 first.vulkan_context->device(),
                 first.option,
                 input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(first.output_columns),
            static_cast<int>(input.rows()),
            vulkan_activation_element_size(first.option),
            first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_output_gpu;
    if (parallel_direct_host_output)
        parallel_output_gpu = prepare_direct_host_output(parallel_download, runtime_state);
    else
        parallel_output_gpu.create(
            static_cast<int>(parallel.output_columns),
            static_cast<int>(input.rows()),
            vulkan_activation_element_size(parallel.option),
            parallel.vulkan_context->blob_allocator());
    if (output_gpu.empty() || parallel_output_gpu.empty())
        return false;

    const bool first_used_cooperative_matrix = record_bfloat16_projection(
        first.pipeline,
        first.cooperative_pipeline,
        input_gpu,
        first.packed,
        first.bias,
        output_gpu,
        first.input_columns,
        first.output_columns,
        first.block_count,
        static_cast<uint32_t>(input.rows()),
        first.cooperative_tile_m,
        first.cooperative_tile_n,
        first.cooperative_tile_k,
        first.cooperative_subgroup_size,
        first.cooperative_forced,
        command);
    const bool parallel_used_cooperative_matrix = record_bfloat16_projection(
        parallel.pipeline,
        parallel.cooperative_pipeline,
        input_gpu,
        parallel.packed,
        parallel.bias,
        parallel_output_gpu,
        parallel.input_columns,
        parallel.output_columns,
        parallel.block_count,
        static_cast<uint32_t>(input.rows()),
        parallel.cooperative_tile_m,
        parallel.cooperative_tile_n,
        parallel.cooperative_tile_k,
        parallel.cooperative_subgroup_size,
        parallel.cooperative_forced,
        command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
                output_gpu,
                input.rows(),
                first.output_columns,
                transfer_slot.download,
                command,
                first.vulkan_context->device(),
                first.option,
                output.dtype()))
        || (!parallel_direct_host_output
            && !record_prepared_activation_staging_download(
                   parallel_output_gpu,
                   input.rows(),
                   parallel.output_columns,
                   parallel_download,
                   command,
                   parallel.vulkan_context->device(),
                   parallel.option,
                   parallel_output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2;
    if (first_used_cooperative_matrix)
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    if (parallel_used_cooperative_matrix)
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    return true;
#else
    (void)input;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    return false;
#endif
}

bool NcnnVulkanBfloat16Operator::forward_swiglu_chain(
    const ActivationBuffer& input,
    const NcnnVulkanBfloat16Operator& down_operator,
    uint32_t intermediate_columns,
    ExpertActivation activation,
    float activation_limit,
    bool apply_router_gate,
    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *implementation_;
    const Implementation& down = *down_operator.implementation_;
    const uint64_t expected_fused_columns =
        static_cast<uint64_t>(intermediate_columns) * 2
        + (apply_router_gate ? 1u : 0u);
    (void)activation_limit;
    if (activation != ExpertActivation::Silu
        || !first.vulkan_context
        || !down.vulkan_context
        || first.vulkan_context.get() != down.vulkan_context.get()
        || vulkan_activation_storage_variant(first.option)
               != vulkan_activation_storage_variant(down.option)
        || !first.pipeline
        || first.output_columns != expected_fused_columns
        || down.input_columns != intermediate_columns
        || intermediate_columns == 0
        || intermediate_columns % 128 != 0
        || input.rows() == 0
        || input.columns() != first.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || down.packed.empty()
        || down.bias.empty())
    {
        return false;
    }

    NcnnVulkanRuntimeState& runtime_state =
        first.vulkan_context->runtime_state();
    NcnnVulkanTransferLease transfer_lease =
        first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input =
        vulkan_activation_storage_variant(first.option) == 0
        && direct_host_input_enabled(
            *first.vulkan_context,
            static_cast<size_t>(input.rows()) * first.input_columns
                * sizeof(float),
            input.dtype());
    const bool direct_host_output =
        vulkan_activation_storage_variant(first.option) == 0
        && direct_host_output_enabled(
            *first.vulkan_context,
            static_cast<size_t>(input.rows()) * down.output_columns
                * sizeof(float),
            output.dtype());
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            down.output_columns,
            transfer_slot.staging_allocator,
            runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), down.output_columns, false);

    std::unique_lock<std::mutex> lock(
        first.vulkan_context->command_mutex());
    if (!first.swiglu_down_pipeline
        && !create_bfloat16_swiglu_down_pipeline(
            first.vulkan_context,
            first.option,
            first.swiglu_down_pipeline))
    {
        return false;
    }
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(
                 transfer_slot.upload,
                 input_gpu,
                 command,
                 first.vulkan_context->device(),
                 first.option,
                 input.dtype()))
    {
        return false;
    }
    ncnn::VkMat fused_gpu;
    fused_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        vulkan_activation_element_size(first.option),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(down.output_columns),
            static_cast<int>(input.rows()),
            vulkan_activation_element_size(first.option),
            first.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || output_gpu.empty())
        return false;

    const bool used_cooperative_matrix = record_bfloat16_projection(
        first.pipeline,
        first.cooperative_pipeline,
        input_gpu,
        first.packed,
        first.bias,
        fused_gpu,
        first.input_columns,
        first.output_columns,
        first.block_count,
        static_cast<uint32_t>(input.rows()),
        first.cooperative_tile_m,
        first.cooperative_tile_n,
        first.cooperative_tile_k,
        first.cooperative_subgroup_size,
        first.cooperative_forced,
        command);

    std::vector<ncnn::VkMat> swiglu_bindings = {
        fused_gpu,
        down.packed,
        down.bias,
        output_gpu};
    std::vector<ncnn::vk_constant_type> swiglu_constants(5);
    swiglu_constants[0].u32 = intermediate_columns;
    swiglu_constants[1].u32 = first.output_columns;
    swiglu_constants[2].u32 = down.output_columns;
    swiglu_constants[3].u32 = static_cast<uint32_t>(input.rows());
    swiglu_constants[4].u32 = apply_router_gate ? 1u : 0u;
    ncnn::VkMat swiglu_dispatcher;
    swiglu_dispatcher.w = static_cast<int>(down.output_columns * 32);
    swiglu_dispatcher.h = static_cast<int>(input.rows());
    swiglu_dispatcher.c = 1;
    command.record_pipeline(
        first.swiglu_down_pipeline.get(),
        swiglu_bindings,
        swiglu_constants,
        swiglu_dispatcher);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
                output_gpu,
                input.rows(),
                down.output_columns,
                transfer_slot.download,
                command,
                first.vulkan_context->device(),
                first.option,
                output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.shared_expert_swiglu_fusions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    if (used_cooperative_matrix)
    {
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)down_operator;
    (void)intermediate_columns;
    (void)activation;
    (void)activation_limit;
    (void)apply_router_gate;
    (void)output;
    return false;
#endif
}



class NcnnVulkanGatedDeltaState::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    ncnn::Option option;
    ncnn::VkMat convolution;
    ncnn::VkMat recurrent;
    ncnn::VkMat zero_convolution;
    ncnn::VkMat zero_recurrent;
    ncnn::VkMat cpu_convolution;
    ncnn::VkMat cpu_recurrent;
    ncnn::VkMat initial_convolution;
    ncnn::VkMat initial_recurrent;
    std::vector<ncnn::VkMat> convolution_snapshots;
    std::vector<ncnn::VkMat> recurrent_snapshots;
    uint32_t convolution_size = 0;
    uint32_t kernel_size = 0;
    uint32_t head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t value_head_dimension = 0;
    bool initialized = false;
    bool cpu_state_pending = false;
    bool state_unknown = false;
    bool transaction_active = false;
    bool transaction_initial_recorded = false;
    bool transaction_initial_pending = false;
    bool transaction_row_pending = false;
    bool transaction_row_submitted = false;
    bool transaction_requires_restore = false;
    bool transaction_state_unknown = false;
    size_t expected_rows = 0;
    size_t recorded_rows = 0;
#endif
};

class NcnnVulkanGatedDeltaNetOperator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::shared_ptr<NcnnVulkanBfloat16Operator> fused_input;
    std::shared_ptr<NcnnVulkanBfloat16Operator> output_projection;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat convolution_weight;
    ncnn::VkMat time_bias;
    ncnn::VkMat decay_log;
    ncnn::VkMat norm_weight;
    ncnn::Option option;
    uint32_t convolution_size = 0;
    uint32_t value_size = 0;
    uint32_t fused_columns = 0;
    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t value_head_dimension = 0;
    uint32_t convolution_kernel_size = 0;
    float norm_epsilon = 0.0f;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
#endif
};

NcnnVulkanGatedDeltaState::NcnnVulkanGatedDeltaState()
    : implementation_(new Implementation)
{
}

NcnnVulkanGatedDeltaState::~NcnnVulkanGatedDeltaState() = default;

NcnnVulkanGatedDeltaNetOperator::NcnnVulkanGatedDeltaNetOperator()
    : implementation_(new Implementation)
{
}

NcnnVulkanGatedDeltaNetOperator::~NcnnVulkanGatedDeltaNetOperator() = default;

#if NCNN_MOE_WITH_VULKAN
static bool prepare_float_tensor_upload(
    const TensorData& source,
    ncnn::Mat& destination)
{
    if (source.element_count() == 0
        || source.element_count()
               > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    destination.create(static_cast<int>(source.element_count()), sizeof(float));
    if (destination.empty())
        return false;
    float* values = static_cast<float*>(destination.data);
    if (source.dtype == DType::Float32)
    {
        const std::span<const float> source_values = source.float32_values();
        if (source_values.size() != source.element_count())
            return false;
        std::copy(source_values.begin(), source_values.end(), values);
        return true;
    }
    if (source.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> source_values = source.bfloat16_values();
        if (source_values.size() != source.element_count())
            return false;
        for (size_t index = 0; index < source_values.size(); ++index)
            values[index] = bfloat16_to_float(source_values[index]);
        return true;
    }
    return false;
}

static bool create_state_storage(
    ncnn::VkMat& destination,
    uint32_t elements,
    ncnn::VkAllocator* allocator)
{
    if (elements == 0
        || elements > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    destination.create(static_cast<int>(elements), 1, sizeof(float), allocator);
    return !destination.empty();
}

static bool fill_zero_staging(ncnn::VkMat& destination, uint32_t elements)
{
    if (destination.empty() || !destination.mapped_ptr())
        return false;
    std::fill_n(static_cast<float*>(destination.mapped_ptr()), elements, 0.0f);
    destination.allocator->flush(destination.data);
    destination.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    destination.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static uint64_t vkmat_allocated_bytes(const ncnn::VkMat& value)
{
    return value.empty() ? 0 : static_cast<uint64_t>(value.buffer_capacity());
}

static std::shared_ptr<NcnnVulkanGatedDeltaState> create_gated_delta_state(
    const std::shared_ptr<NcnnVulkanContext>& context,
    uint32_t convolution_size,
    uint32_t kernel_size,
    uint32_t head_count,
    uint32_t head_dimension,
    uint32_t value_head_dimension,
    const ncnn::Option& option)
{
    if (!context || convolution_size == 0 || kernel_size == 0
        || head_count == 0 || head_dimension == 0
        || value_head_dimension == 0)
    {
        return {};
    }
    const uint64_t recurrent_elements =
        static_cast<uint64_t>(head_count) * head_dimension
        * value_head_dimension;
    const uint64_t convolution_elements =
        static_cast<uint64_t>(convolution_size) * kernel_size;
    if (recurrent_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || convolution_elements > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }

    std::shared_ptr<NcnnVulkanGatedDeltaState> result(
        new NcnnVulkanGatedDeltaState);
    NcnnVulkanGatedDeltaState::Implementation& implementation =
        *result->implementation_;
    implementation.vulkan_context = context;
    implementation.option = option;
    implementation.convolution_size = convolution_size;
    implementation.kernel_size = kernel_size;
    implementation.head_count = head_count;
    implementation.head_dimension = head_dimension;
    implementation.value_head_dimension = value_head_dimension;
    if (!create_state_storage(
            implementation.convolution,
            static_cast<uint32_t>(convolution_elements),
            context->blob_allocator())
        || !create_state_storage(
            implementation.recurrent,
            static_cast<uint32_t>(recurrent_elements),
            context->blob_allocator())
        || !create_state_storage(
            implementation.zero_convolution,
            static_cast<uint32_t>(convolution_elements),
            context->staging_allocator())
        || !create_state_storage(
            implementation.zero_recurrent,
            static_cast<uint32_t>(recurrent_elements),
            context->staging_allocator())
        || !fill_zero_staging(
            implementation.zero_convolution,
            static_cast<uint32_t>(convolution_elements))
        || !fill_zero_staging(
            implementation.zero_recurrent,
            static_cast<uint32_t>(recurrent_elements)))
    {
        return {};
    }
    return result;
}

static bool record_state_zero_initialization(
    NcnnVulkanGatedDeltaState::Implementation& implementation,
    ncnn::VkCompute& command)
{
    if (implementation.initialized)
        return true;
    const ncnn::VkMat& source_convolution =
        implementation.cpu_state_pending
            ? implementation.cpu_convolution
            : implementation.zero_convolution;
    const ncnn::VkMat& source_recurrent =
        implementation.cpu_state_pending
            ? implementation.cpu_recurrent
            : implementation.zero_recurrent;
    command.record_clone(source_convolution, implementation.convolution, implementation.option);
    command.record_clone(source_recurrent, implementation.recurrent, implementation.option);
    return !implementation.convolution.empty()
           && !implementation.recurrent.empty();
}

static bool record_state_transaction_initial(
    NcnnVulkanGatedDeltaState::Implementation& implementation,
    ncnn::VkCompute& command)
{
    if (!implementation.transaction_active
        || implementation.transaction_initial_recorded
        || implementation.transaction_initial_pending)
    {
        return true;
    }
    if (implementation.initial_convolution.empty()
        || implementation.initial_recurrent.empty())
    {
        return false;
    }
    command.record_clone(
        implementation.convolution,
        implementation.initial_convolution,
        implementation.option);
    command.record_clone(
        implementation.recurrent,
        implementation.initial_recurrent,
        implementation.option);
    implementation.transaction_initial_pending = true;
    return true;
}

static bool record_state_transaction_row(
    NcnnVulkanGatedDeltaState::Implementation& implementation,
    ncnn::VkCompute& command)
{
    if (!implementation.transaction_active)
        return true;
    if (implementation.transaction_row_pending
        || implementation.recorded_rows >= implementation.expected_rows)
        return false;
    if (implementation.recorded_rows < implementation.convolution_snapshots.size())
    {
        command.record_clone(
            implementation.convolution,
            implementation.convolution_snapshots[implementation.recorded_rows],
            implementation.option);
        command.record_clone(
            implementation.recurrent,
            implementation.recurrent_snapshots[implementation.recorded_rows],
            implementation.option);
    }
    implementation.transaction_row_pending = true;
    return true;
}

static void mark_state_transaction_submission(
    NcnnVulkanGatedDeltaState::Implementation& implementation) noexcept
{
    if (implementation.transaction_initial_pending)
    {
        implementation.transaction_initial_pending = false;
        implementation.transaction_initial_recorded = true;
    }
    if (implementation.transaction_row_pending)
    {
        implementation.transaction_row_submitted = true;
        implementation.transaction_requires_restore = true;
    }
}

static void commit_state_transaction_row(
    NcnnVulkanGatedDeltaState::Implementation& implementation) noexcept
{
    if (!implementation.transaction_row_pending
        || !implementation.transaction_row_submitted)
    {
        return;
    }
    ++implementation.recorded_rows;
    implementation.transaction_row_pending = false;
    implementation.transaction_row_submitted = false;
    implementation.transaction_requires_restore = false;
}

static void discard_state_transaction_recording(
    NcnnVulkanGatedDeltaState::Implementation& implementation) noexcept
{
    implementation.transaction_initial_pending = false;
    implementation.transaction_row_pending = false;
    implementation.transaction_row_submitted = false;
}

static void mark_state_transaction_submit_failure(
    NcnnVulkanGatedDeltaState::Implementation& implementation) noexcept
{
    if (implementation.transaction_active
        && implementation.transaction_row_pending)
    {
        if (implementation.transaction_initial_recorded)
            implementation.transaction_requires_restore = true;
        else
            implementation.transaction_state_unknown = true;
    }
}

class StateTransactionRecordingGuard
{
public:
    explicit StateTransactionRecordingGuard(
        std::span<NcnnVulkanGatedDeltaState::Implementation* const> states)
        : states_(states)
    {
    }

    ~StateTransactionRecordingGuard()
    {
        if (completed_)
            return;
        for (NcnnVulkanGatedDeltaState::Implementation* state : states_)
            discard_state_transaction_recording(*state);
    }

    void mark_submitted() noexcept
    {
        for (NcnnVulkanGatedDeltaState::Implementation* state : states_)
            mark_state_transaction_submission(*state);
    }

    void mark_submit_failed() noexcept
    {
        for (NcnnVulkanGatedDeltaState::Implementation* state : states_)
            mark_state_transaction_submit_failure(*state);
    }

    void commit_rows() noexcept
    {
        for (NcnnVulkanGatedDeltaState::Implementation* state : states_)
            commit_state_transaction_row(*state);
        completed_ = true;
    }

private:
    std::span<NcnnVulkanGatedDeltaState::Implementation* const> states_;
    bool completed_ = false;
};

static bool restore_state_snapshot(
    NcnnVulkanGatedDeltaState::Implementation& implementation,
    const ncnn::VkMat& convolution,
    const ncnn::VkMat& recurrent)
{
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();
    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute command(implementation.vulkan_context->device(), implementation.vulkan_context->command_optimization_flags());
    command.record_clone(
        convolution,
        implementation.convolution,
        implementation.option);
    command.record_clone(
        recurrent,
        implementation.recurrent,
        implementation.option);
    if (submit_compute_and_wait(command, runtime_state) != 0)
        return false;
    ++runtime_state.compute_submissions;
    return true;
}
#endif

std::shared_ptr<NcnnVulkanGatedDeltaState>
NcnnVulkanGatedDeltaState::create(
    const std::shared_ptr<NcnnVulkanContext>& context,
    uint32_t convolution_size,
    uint32_t kernel_size,
    uint32_t head_count,
    uint32_t head_dimension,
    uint32_t value_head_dimension,
    const ncnn::Option& option)
{
#if NCNN_MOE_WITH_VULKAN
    return create_gated_delta_state(
        context,
        convolution_size,
        kernel_size,
        head_count,
        head_dimension,
        value_head_dimension,
        option);
#else
    (void)context;
    (void)convolution_size;
    (void)kernel_size;
    (void)head_count;
    (void)head_dimension;
    (void)value_head_dimension;
    (void)option;
    return {};
#endif
}

bool NcnnVulkanGatedDeltaState::begin_transaction(
    size_t expected_rows) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (implementation.state_unknown
        || implementation.transaction_active)
        return false;
    implementation.transaction_active = false;
    implementation.transaction_initial_recorded = false;
    implementation.transaction_initial_pending = false;
    implementation.transaction_row_pending = false;
    implementation.transaction_row_submitted = false;
    implementation.transaction_requires_restore = false;
    implementation.transaction_state_unknown = false;
    implementation.expected_rows = 0;
    implementation.recorded_rows = 0;
    implementation.initial_convolution = ncnn::VkMat();
    implementation.initial_recurrent = ncnn::VkMat();
    implementation.convolution_snapshots.clear();
    implementation.recurrent_snapshots.clear();
    if (expected_rows == 0)
    {
        implementation.transaction_active = true;
        implementation.expected_rows = expected_rows;
        return true;
    }
    const uint64_t convolution_elements =
        static_cast<uint64_t>(implementation.convolution_size)
        * implementation.kernel_size;
    const uint64_t recurrent_elements =
        static_cast<uint64_t>(implementation.head_count)
        * implementation.head_dimension
        * implementation.value_head_dimension;
    try
    {
        if (!create_state_storage(
                implementation.initial_convolution,
                static_cast<uint32_t>(convolution_elements),
                implementation.vulkan_context->blob_allocator())
            || !create_state_storage(
                implementation.initial_recurrent,
                static_cast<uint32_t>(recurrent_elements),
                implementation.vulkan_context->blob_allocator()))
        {
            complete_transaction();
            return false;
        }
        const size_t snapshot_count = expected_rows > 0 ? expected_rows - 1 : 0;
        implementation.convolution_snapshots.resize(snapshot_count);
        implementation.recurrent_snapshots.resize(snapshot_count);
        for (size_t index = 0; index < snapshot_count; ++index)
        {
            if (!create_state_storage(
                    implementation.convolution_snapshots[index],
                    static_cast<uint32_t>(convolution_elements),
                    implementation.vulkan_context->blob_allocator())
                || !create_state_storage(
                    implementation.recurrent_snapshots[index],
                    static_cast<uint32_t>(recurrent_elements),
                    implementation.vulkan_context->blob_allocator()))
            {
                complete_transaction();
                return false;
            }
        }
    }
    catch (...)
    {
        complete_transaction();
        return false;
    }
    implementation.expected_rows = expected_rows;
    implementation.transaction_active = true;
    return true;
#else
    (void)expected_rows;
    return false;
#endif
}

bool NcnnVulkanGatedDeltaState::prepare_cpu_state(
    const std::vector<float>& convolution,
    const std::vector<float>& recurrent)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    const size_t expected_convolution = static_cast<size_t>(
        implementation.convolution_size) * implementation.kernel_size;
    const size_t expected_recurrent = static_cast<size_t>(
        implementation.head_count) * implementation.head_dimension
        * implementation.value_head_dimension;
    if (implementation.initialized
        || convolution.size() != expected_convolution
        || recurrent.size() != expected_recurrent)
    {
        return false;
    }
    implementation.cpu_convolution.create(
        static_cast<int>(expected_convolution),
        1,
        sizeof(float),
        implementation.vulkan_context->staging_allocator());
    implementation.cpu_recurrent.create(
        static_cast<int>(expected_recurrent),
        1,
        sizeof(float),
        implementation.vulkan_context->staging_allocator());
    if (implementation.cpu_convolution.empty()
        || implementation.cpu_recurrent.empty()
        || !implementation.cpu_convolution.mapped_ptr()
        || !implementation.cpu_recurrent.mapped_ptr())
    {
        return false;
    }
    std::copy(
        convolution.begin(),
        convolution.end(),
        static_cast<float*>(implementation.cpu_convolution.mapped_ptr()));
    std::copy(
        recurrent.begin(),
        recurrent.end(),
        static_cast<float*>(implementation.cpu_recurrent.mapped_ptr()));
    implementation.cpu_convolution.allocator->flush(
        implementation.cpu_convolution.data);
    implementation.cpu_recurrent.allocator->flush(
        implementation.cpu_recurrent.data);
    implementation.cpu_convolution.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    implementation.cpu_convolution.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    implementation.cpu_recurrent.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    implementation.cpu_recurrent.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    implementation.cpu_state_pending = true;
    return true;
#else
    (void)convolution;
    (void)recurrent;
    return false;
#endif
}

bool NcnnVulkanGatedDeltaState::prepare_transaction_finish(
    size_t committed_rows,
    size_t recorded_rows) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (!implementation.transaction_active
        || recorded_rows != implementation.recorded_rows
        || committed_rows > recorded_rows
        || implementation.transaction_state_unknown)
    {
        return false;
    }
    bool restored = true;
    if (committed_rows < recorded_rows
        || implementation.transaction_requires_restore)
    {
        if (committed_rows == 0)
        {
            restored = implementation.transaction_initial_recorded
                       && restore_state_snapshot(
                           implementation,
                           implementation.initial_convolution,
                           implementation.initial_recurrent);
        }
        else if (committed_rows - 1 < implementation.convolution_snapshots.size())
        {
            restored = restore_state_snapshot(
                implementation,
                implementation.convolution_snapshots[committed_rows - 1],
                implementation.recurrent_snapshots[committed_rows - 1]);
        }
        else
        {
            restored = false;
        }
    }
    return restored;
#else
    (void)committed_rows;
    (void)recorded_rows;
    return false;
#endif
}

void NcnnVulkanGatedDeltaState::complete_transaction() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    implementation.transaction_active = false;
    implementation.transaction_initial_recorded = false;
    implementation.transaction_initial_pending = false;
    implementation.transaction_row_pending = false;
    implementation.transaction_row_submitted = false;
    implementation.transaction_requires_restore = false;
    implementation.transaction_state_unknown = false;
    implementation.expected_rows = 0;
    implementation.recorded_rows = 0;
    implementation.initial_convolution = ncnn::VkMat();
    implementation.initial_recurrent = ncnn::VkMat();
    implementation.convolution_snapshots.clear();
    implementation.recurrent_snapshots.clear();
#endif
}

bool NcnnVulkanGatedDeltaState::download(
    std::vector<float>& convolution,
    std::vector<float>& recurrent) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (implementation.state_unknown
        || !implementation.vulkan_context
        || implementation.convolution.empty()
        || implementation.recurrent.empty())
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();
    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const size_t convolution_count = static_cast<size_t>(
        implementation.convolution_size) * implementation.kernel_size;
    const size_t recurrent_count = static_cast<size_t>(
        implementation.head_count) * implementation.head_dimension
        * implementation.value_head_dimension;
    if (!prepare_staging_matrix(
            transfer_slot.upload,
            static_cast<int>(convolution_count),
            1,
            sizeof(float),
            transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_matrix(
            transfer_slot.download,
            static_cast<int>(recurrent_count),
            1,
            sizeof(float),
                                        transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;
    // The two clones use separate staging buffers.  Reuse the slot's upload
    // and download allocations only after the first copy has completed.
    ncnn::VkMat convolution_staging = transfer_slot.upload;
    ncnn::VkMat recurrent_staging = transfer_slot.download;
    command.record_clone(
        implementation.convolution,
        convolution_staging,
        implementation.option);
    command.record_clone(
        implementation.recurrent,
        recurrent_staging,
        implementation.option);
    if (submit_compute_and_wait(command, runtime_state) != 0)
        return false;
    convolution_staging.allocator->invalidate(convolution_staging.data);
    recurrent_staging.allocator->invalidate(recurrent_staging.data);
    const ncnn::Mat convolution_mapped = convolution_staging.mapped();
    const ncnn::Mat recurrent_mapped = recurrent_staging.mapped();
    if (convolution_mapped.empty() || recurrent_mapped.empty())
        return false;
    convolution.assign(
        static_cast<const float*>(convolution_mapped.data),
        static_cast<const float*>(convolution_mapped.data) + convolution_count);
    recurrent.assign(
        static_cast<const float*>(recurrent_mapped.data),
        static_cast<const float*>(recurrent_mapped.data) + recurrent_count);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)convolution;
    (void)recurrent;
    return false;
#endif
}

uint64_t NcnnVulkanGatedDeltaState::allocated_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    uint64_t bytes = vkmat_allocated_bytes(implementation.convolution)
                     + vkmat_allocated_bytes(implementation.recurrent)
                     + vkmat_allocated_bytes(implementation.zero_convolution)
                     + vkmat_allocated_bytes(implementation.zero_recurrent)
                     + vkmat_allocated_bytes(implementation.cpu_convolution)
                     + vkmat_allocated_bytes(implementation.cpu_recurrent)
                     + vkmat_allocated_bytes(implementation.initial_convolution)
                     + vkmat_allocated_bytes(implementation.initial_recurrent);
    for (const ncnn::VkMat& snapshot : implementation.convolution_snapshots)
        bytes += vkmat_allocated_bytes(snapshot);
    for (const ncnn::VkMat& snapshot : implementation.recurrent_snapshots)
        bytes += vkmat_allocated_bytes(snapshot);
    return bytes;
#else
    return 0;
#endif
}

std::shared_ptr<NcnnVulkanGatedDeltaNetOperator>
NcnnVulkanGatedDeltaNetOperator::create(
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
    uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (!fused_input || !output_projection || head_count == 0
        || kv_head_count == 0 || head_count % kv_head_count != 0
        || head_dimension == 0 || value_head_dimension == 0
        || convolution_kernel_size == 0)
    {
        return {};
    }
    const uint32_t key_size = kv_head_count * head_dimension;
    const uint32_t value_size = head_count * value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const uint32_t fused_columns = convolution_size + value_size + head_count * 2;
    const NcnnVulkanBfloat16Operator::Implementation& first =
        *fused_input->implementation_;
    const NcnnVulkanBfloat16Operator::Implementation& output =
        *output_projection->implementation_;
    if (!first.vulkan_context || !output.vulkan_context
        || first.optimization_flags != optimization_flags
        || output.optimization_flags != optimization_flags
        || first.vulkan_context->instance().get() != context_instance.get()
        || first.vulkan_context.get() != output.vulkan_context.get()
        || !first.pipeline || !output.pipeline
        || first.output_columns != fused_columns
        || output.input_columns != value_size
        || convolution_weight.element_count()
               != static_cast<size_t>(convolution_size)
                      * convolution_kernel_size
        || time_bias.element_count() != head_count
        || decay_log.element_count() != head_count
        || norm_weight.element_count() != value_head_dimension
        || (convolution_weight.dtype != DType::Float32
            && convolution_weight.dtype != DType::BFloat16)
        || (time_bias.dtype != DType::Float32
            && time_bias.dtype != DType::BFloat16)
        || (decay_log.dtype != DType::Float32
            && decay_log.dtype != DType::BFloat16)
        || (norm_weight.dtype != DType::Float32
            && norm_weight.dtype != DType::BFloat16))
    {
        return {};
    }

    std::shared_ptr<NcnnVulkanGatedDeltaNetOperator> result(
        new NcnnVulkanGatedDeltaNetOperator);
    Implementation& implementation = *result->implementation_;
    implementation.vulkan_context = first.vulkan_context;
    implementation.optimization_flags = first.optimization_flags;
    implementation.fused_input = fused_input;
    implementation.output_projection = output_projection;
    implementation.option = first.option;
    implementation.convolution_size = convolution_size;
    implementation.value_size = value_size;
    implementation.fused_columns = fused_columns;
    implementation.head_count = head_count;
    implementation.kv_head_count = kv_head_count;
    implementation.head_dimension = head_dimension;
    implementation.value_head_dimension = value_head_dimension;
    implementation.convolution_kernel_size = convolution_kernel_size;
    implementation.norm_epsilon = norm_epsilon;
    {
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        if (!create_gated_delta_net_pipeline(
                implementation.vulkan_context,
                implementation.option,
                implementation.pipeline))
        {
            return {};
        }
    }

    implementation.weight_allocator.reset(
        new ncnn::VkWeightAllocator(
            implementation.vulkan_context->device()));
    implementation.weight_staging_allocator.reset(
        new ncnn::VkWeightStagingAllocator(
            implementation.vulkan_context->device()));
    ncnn::Mat convolution_model;
    ncnn::Mat time_model;
    ncnn::Mat decay_model;
    ncnn::Mat norm_model;
    if (!prepare_float_tensor_upload(convolution_weight, convolution_model)
        || !prepare_float_tensor_upload(time_bias, time_model)
        || !prepare_float_tensor_upload(decay_log, decay_model)
        || !prepare_float_tensor_upload(norm_weight, norm_model))
    {
        return {};
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    {
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(implementation.vulkan_context->device());
        command.record_upload(
            convolution_model,
            implementation.convolution_weight,
            upload_option);
        command.record_upload(time_model, implementation.time_bias, upload_option);
        command.record_upload(decay_model, implementation.decay_log, upload_option);
        command.record_upload(norm_model, implementation.norm_weight, upload_option);
        if (implementation.convolution_weight.empty()
            || implementation.time_bias.empty()
            || implementation.decay_log.empty()
            || implementation.norm_weight.empty()
            || command.submit_and_wait() != 0)
        {
            return {};
        }
    }
    implementation.weight_staging_allocator.reset();
    (void)vulkan_device_index;
    return result;
#else
    (void)fused_input;
    (void)convolution_weight;
    (void)time_bias;
    (void)decay_log;
    (void)norm_weight;
    (void)output_projection;
    (void)head_count;
    (void)kv_head_count;
    (void)head_dimension;
    (void)value_head_dimension;
    (void)convolution_kernel_size;
    (void)norm_epsilon;
    (void)vulkan_device_index;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

bool NcnnVulkanGatedDeltaNetOperator::forward_impl(
    const ActivationBuffer& input,
    CpuLayerCache& cache,
    ActivationBuffer& projected,
    bool apply_input_rms_norm) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.rows() == 0
        || input.columns() != implementation.fused_input->implementation_->input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || (cache.transaction.active && input.rows() != 1))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();

    bool created_device_state = false;
    if (!cache.gated_delta_device_state)
    {
        const bool has_cpu_state =
            !cache.gated_delta_convolution.empty()
            || !cache.gated_delta_recurrent.empty();
        cache.gated_delta_device_state =
            NcnnVulkanGatedDeltaState::create(
                implementation.vulkan_context,
                implementation.convolution_size,
                implementation.convolution_kernel_size,
                implementation.head_count,
                implementation.head_dimension,
                implementation.value_head_dimension,
                implementation.option);
        if (!cache.gated_delta_device_state)
            return false;
        created_device_state = true;
        if (has_cpu_state
            && !cache.gated_delta_device_state->prepare_cpu_state(
                cache.gated_delta_convolution,
                cache.gated_delta_recurrent))
        {
            cache.gated_delta_device_state.reset();
            return false;
        }
    }
    struct DeviceStateCreationAttempt
    {
        CpuLayerCache* cache = nullptr;

        ~DeviceStateCreationAttempt()
        {
            if (cache)
            {
                cache->gated_delta_device_state.reset();
                cache->device_allocated_bytes = 0;
            }
        }

        void complete() noexcept
        {
            cache = nullptr;
        }
    } creation_attempt{
        created_device_state ? &cache : nullptr};
    std::shared_ptr<NcnnVulkanGatedDeltaState> state =
        cache.gated_delta_device_state;
    NcnnVulkanGatedDeltaState::Implementation& state_implementation =
        *state->implementation_;
    if (state_implementation.state_unknown)
        return false;
    if (cache.transaction.active
        && !state_implementation.transaction_active)
    {
        if (!state->begin_transaction(
                cache.transaction.expected_rows))
            return false;
    }
    cache.device_allocated_bytes = state->allocated_bytes();

    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_input_enabled(
            *implementation.vulkan_context,
            input.rows() * input.columns() * sizeof(float),
            input.dtype());
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            implementation.output_projection->implementation_->output_columns,
            transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType projected_dtype = projected.dtype();
    projected.reset(
        input.rows(),
        implementation.output_projection->implementation_->output_columns,
        false);

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(
            transfer_slot.upload,
            input_gpu,
            command,
            implementation.option))
    {
        return false;
    }
    std::array<NcnnVulkanGatedDeltaState::Implementation*, 1>
        transaction_states = {&state_implementation};
    StateTransactionRecordingGuard transaction_recording(
        transaction_states);
    if (!record_state_zero_initialization(state_implementation, command)
        || !record_state_transaction_initial(state_implementation, command))
    {
        return false;
    }

    const NcnnVulkanBfloat16Operator::Implementation& first =
        *implementation.fused_input->implementation_;
    const NcnnVulkanBfloat16Operator::Implementation& output_operator =
        *implementation.output_projection->implementation_;
    const bool use_input_rms_norm =
        apply_input_rms_norm
        && first.rms_norm_projection_pipeline
        && !first.rms_norm_weight.empty()
        && first.rms_norm_output_subgroups != 0;
    ncnn::VkMat fused_gpu;
    fused_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat recurrent_output_gpu;
    recurrent_output_gpu.create(
        static_cast<int>(implementation.value_size),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    output_gpu.create(
        static_cast<int>(output_operator.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || recurrent_output_gpu.empty() || output_gpu.empty())
        return false;

    ncnn::VkMat projection_dispatcher;
    projection_dispatcher.h = static_cast<int>(input.rows());
    projection_dispatcher.c = 1;
    if (use_input_rms_norm)
    {
        std::vector<ncnn::VkMat> projection_bindings = {
            input_gpu,
            first.packed,
            first.bias,
            first.rms_norm_weight,
            fused_gpu};
        std::vector<ncnn::vk_constant_type> projection_constants(6);
        projection_constants[0].u32 = first.input_columns;
        projection_constants[1].u32 = first.output_columns;
        projection_constants[2].u32 = first.block_count;
        projection_constants[3].u32 = static_cast<uint32_t>(input.rows());
        projection_constants[4].f = first.rms_norm_epsilon;
        projection_constants[5].u32 = first.rms_norm_output_subgroups;
        projection_dispatcher.w = static_cast<int>(
            ((static_cast<uint64_t>(first.output_columns)
              + first.rms_norm_output_subgroups - 1)
             / first.rms_norm_output_subgroups)
            * first.rms_norm_output_subgroups * 32);
        command.record_pipeline_readonly(
            first.rms_norm_projection_pipeline.get(),
            projection_bindings,
            {1, 1, 1, 1, 0},
            projection_constants,
            projection_dispatcher);
    }
    else
    {
        std::vector<ncnn::VkMat> projection_bindings = {
            input_gpu,
            first.packed,
            first.bias,
            fused_gpu};
        std::vector<ncnn::vk_constant_type> projection_constants(4);
        projection_constants[0].u32 = first.input_columns;
        projection_constants[1].u32 = first.output_columns;
        projection_constants[2].u32 = first.block_count;
        projection_constants[3].u32 = static_cast<uint32_t>(input.rows());
        projection_dispatcher.w = static_cast<int>(first.output_columns * 32);
        command.record_pipeline_readonly(
            first.pipeline.get(),
            projection_bindings,
            {1, 1, 1, 0},
            projection_constants,
            projection_dispatcher);
    }

    std::vector<ncnn::VkMat> delta_bindings = {
        fused_gpu,
        state_implementation.convolution,
        state_implementation.recurrent,
        implementation.convolution_weight,
        implementation.time_bias,
        implementation.decay_log,
        implementation.norm_weight,
        recurrent_output_gpu};
    std::vector<ncnn::vk_constant_type> delta_constants(10);
    delta_constants[0].u32 = implementation.convolution_size;
    delta_constants[1].u32 = implementation.value_size;
    delta_constants[2].u32 = implementation.fused_columns;
    delta_constants[3].u32 = implementation.head_count;
    delta_constants[4].u32 = implementation.kv_head_count;
    delta_constants[5].u32 = implementation.head_dimension;
    delta_constants[6].u32 = implementation.value_head_dimension;
    delta_constants[7].u32 = implementation.convolution_kernel_size;
    delta_constants[8].u32 = static_cast<uint32_t>(input.rows());
    delta_constants[9].f = implementation.norm_epsilon;
    ncnn::VkMat delta_dispatcher;
    delta_dispatcher.w = 128;
    delta_dispatcher.h = 1;
    delta_dispatcher.c = 1;
    command.record_pipeline_readonly(
        implementation.pipeline.get(),
        delta_bindings,
        {0, 0, 0, 1, 1, 1, 1, 0},
        delta_constants,
        delta_dispatcher);
    if (!record_state_transaction_row(state_implementation, command))
        return false;

    std::vector<ncnn::VkMat> output_bindings = {
        recurrent_output_gpu,
        output_operator.packed,
        output_operator.bias,
        output_gpu};
    std::vector<ncnn::vk_constant_type> output_constants(4);
    output_constants[0].u32 = output_operator.input_columns;
    output_constants[1].u32 = output_operator.output_columns;
    output_constants[2].u32 = output_operator.block_count;
    output_constants[3].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat output_dispatcher;
    output_dispatcher.w = static_cast<int>(output_operator.output_columns * 32);
    output_dispatcher.h = static_cast<int>(input.rows());
    output_dispatcher.c = 1;
    command.record_pipeline_readonly(
        output_operator.pipeline.get(),
        output_bindings,
        {1, 1, 1, 0},
        output_constants,
        output_dispatcher);

    if (!record_prepared_activation_staging_download(
        output_gpu,
        input.rows(),
            output_operator.output_columns,
            transfer_slot.download,
            command,
            implementation.vulkan_context->device(),
            implementation.option,
            projected_dtype))
    {
        return false;
    }
    if (submit_compute_and_wait(command, runtime_state) != 0)
    {
        transaction_recording.mark_submit_failed();
        if (!state_implementation.transaction_active
            && !created_device_state)
        {
            state_implementation.state_unknown = true;
        }
        return false;
    }
    transaction_recording.mark_submitted();
    if (!copy_staging_to_cpu_batch(transfer_slot.download, projected))
    {
        if (!state_implementation.transaction_active
            && !created_device_state)
        {
            state_implementation.state_unknown = true;
        }
        return false;
    }
    transaction_recording.commit_rows();
    state_implementation.initialized = true;
    state_implementation.cpu_state_pending = false;
    state_implementation.cpu_convolution = ncnn::VkMat();
    state_implementation.cpu_recurrent = ncnn::VkMat();
    state_implementation.zero_convolution = ncnn::VkMat();
    state_implementation.zero_recurrent = ncnn::VkMat();
    runtime_state.dispatches += 3;
    ++runtime_state.gated_delta_fusions;
    ++runtime_state.gated_delta_submissions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    cache.device_allocated_bytes = state->allocated_bytes();
    creation_attempt.complete();
    return true;
#else
    (void)input;
    (void)cache;
    (void)projected;
    (void)apply_input_rms_norm;
    return false;
#endif
}

bool NcnnVulkanGatedDeltaNetOperator::forward(
    const ActivationBuffer& normalized,
    CpuLayerCache& cache,
    ActivationBuffer& projected) const
{
    return forward_impl(normalized, cache, projected, false);
}

bool NcnnVulkanGatedDeltaNetOperator::forward_input_rms_norm(
    const ActivationBuffer& input,
    CpuLayerCache& cache,
    ActivationBuffer& projected) const
{
    if (!has_input_rms_norm())
        return false;
    return forward_impl(input, cache, projected, true);
}

bool NcnnVulkanGatedDeltaNetOperator::has_input_rms_norm() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return implementation_
           && implementation_->fused_input
           && implementation_->fused_input->has_rms_norm_chain();
#else
    return false;
#endif
}

NcnnVulkanGatedDeltaBatchResult
NcnnVulkanGatedDeltaNetOperator::forward_batch(
    std::span<const NcnnVulkanGatedDeltaBatchEntry> entries) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (entries.empty())
        return NcnnVulkanGatedDeltaBatchResult::Executed;
    if (entries.size() == 1)
    {
        const NcnnVulkanGatedDeltaBatchEntry& entry = entries.front();
        if (!entry.normalized || !entry.cache || !entry.projected)
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        if (forward(*entry.normalized, *entry.cache, *entry.projected))
            return NcnnVulkanGatedDeltaBatchResult::Executed;
        return entry.cache->gated_delta_device_state
                   ? NcnnVulkanGatedDeltaBatchResult::Failed
                   : NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    }
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || entries.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    }
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();

    const NcnnVulkanBfloat16Operator::Implementation& first =
        *implementation.fused_input->implementation_;
    const NcnnVulkanBfloat16Operator::Implementation& output_operator =
        *implementation.output_projection->implementation_;
    const size_t total_rows = entries.size();
    ActivationBuffer combined_normalized;
    combined_normalized.reset(total_rows, first.input_columns, false);
    std::vector<std::shared_ptr<NcnnVulkanGatedDeltaState>> states;
    states.reserve(entries.size());
    std::vector<bool> created_states;
    created_states.reserve(entries.size());
    std::vector<CpuLayerCache*> created_state_caches;
    created_state_caches.reserve(entries.size());
    struct BatchDeviceStateCreationAttempt
    {
        explicit BatchDeviceStateCreationAttempt(
            std::vector<CpuLayerCache*>& created_caches)
            : created_caches_(&created_caches)
        {
        }

        ~BatchDeviceStateCreationAttempt()
        {
            if (!created_caches_)
                return;
            for (CpuLayerCache* cache : *created_caches_)
            {
                cache->gated_delta_device_state.reset();
                cache->device_allocated_bytes = 0;
            }
        }

        void complete() noexcept
        {
            created_caches_ = nullptr;
        }

    private:
        std::vector<CpuLayerCache*>* created_caches_;
    } creation_attempt(created_state_caches);

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        const NcnnVulkanGatedDeltaBatchEntry& entry = entries[entry_index];
        if (!entry.normalized || !entry.cache || !entry.projected
            || entry.normalized->rows() != 1
            || entry.normalized->columns() != first.input_columns)
        {
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        }
        std::copy_n(
            entry.normalized->row(0),
            first.input_columns,
            combined_normalized.row(entry_index));

        CpuLayerCache& cache = *entry.cache;
        if (cache.transaction.active
            && !cache.gated_delta_device_state)
        {
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        }
        bool created_device_state = false;
        if (!cache.gated_delta_device_state)
        {
            const bool has_cpu_state =
                !cache.gated_delta_convolution.empty()
                || !cache.gated_delta_recurrent.empty();
            cache.gated_delta_device_state =
                NcnnVulkanGatedDeltaState::create(
                    implementation.vulkan_context,
                    implementation.convolution_size,
                    implementation.convolution_kernel_size,
                    implementation.head_count,
                    implementation.head_dimension,
                    implementation.value_head_dimension,
                    implementation.option);
            if (!cache.gated_delta_device_state)
                return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
            created_device_state = true;
            if (has_cpu_state
                && !cache.gated_delta_device_state->prepare_cpu_state(
                    cache.gated_delta_convolution,
                    cache.gated_delta_recurrent))
            {
                cache.gated_delta_device_state.reset();
                return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
            }
            created_state_caches.push_back(&cache);
        }
        std::shared_ptr<NcnnVulkanGatedDeltaState> state =
            cache.gated_delta_device_state;
        NcnnVulkanGatedDeltaState::Implementation& state_implementation =
            *state->implementation_;
        if (state_implementation.state_unknown)
            return NcnnVulkanGatedDeltaBatchResult::Failed;
        if (state_implementation.vulkan_context != implementation.vulkan_context)
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        if (cache.transaction.active
            && !state_implementation.transaction_active)
        {
            if (!state->begin_transaction(
                    cache.transaction.expected_rows))
                return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        }
        cache.device_allocated_bytes = state->allocated_bytes();
        states.push_back(std::move(state));
        created_states.push_back(created_device_state);
    }

    const auto mark_nontransaction_existing_states_unknown = [&]() noexcept {
        for (size_t index = 0; index < states.size(); ++index)
        {
            if (!created_states[index]
                && !states[index]->implementation_->transaction_active)
            {
                states[index]->implementation_->state_unknown = true;
            }
        }
    };

    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_input_enabled(
            *implementation.vulkan_context,
            total_rows * first.input_columns * sizeof(float),
            combined_normalized.dtype());
    if (!fill_staging_upload(
            combined_normalized,
            transfer_slot.upload,
            transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            total_rows,
            output_operator.output_columns,
            transfer_slot.staging_allocator, runtime_state))
    {
        return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    }
    ActivationBuffer combined_projected;
    combined_projected.reset(total_rows, output_operator.output_columns, false);
    for (const NcnnVulkanGatedDeltaBatchEntry& entry : entries)
    {
        entry.projected->reset(
            1,
            output_operator.output_columns,
            false);
    }

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(
            transfer_slot.upload,
            input_gpu,
            command,
            implementation.option))
    {
        return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    }
    ncnn::VkMat fused_gpu;
    fused_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(total_rows),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat recurrent_output_gpu;
    recurrent_output_gpu.create(
        static_cast<int>(implementation.value_size),
        static_cast<int>(total_rows),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    output_gpu.create(
        static_cast<int>(output_operator.output_columns),
        static_cast<int>(total_rows),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || recurrent_output_gpu.empty() || output_gpu.empty())
        return NcnnVulkanGatedDeltaBatchResult::NotExecuted;

    std::vector<NcnnVulkanGatedDeltaState::Implementation*>
        transaction_states;
    transaction_states.reserve(states.size());
    for (const std::shared_ptr<NcnnVulkanGatedDeltaState>& state : states)
        transaction_states.push_back(state->implementation_.get());
    StateTransactionRecordingGuard transaction_recording(
        transaction_states);

    for (const std::shared_ptr<NcnnVulkanGatedDeltaState>& state : states)
    {
        NcnnVulkanGatedDeltaState::Implementation& state_implementation =
            *state->implementation_;
        if (!record_state_zero_initialization(state_implementation, command)
            || !record_state_transaction_initial(state_implementation, command))
        {
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        }
    }

    std::vector<ncnn::VkMat> projection_bindings = {
        input_gpu,
        first.packed,
        first.bias,
        fused_gpu};
    std::vector<ncnn::vk_constant_type> projection_constants(4);
    projection_constants[0].u32 = first.input_columns;
    projection_constants[1].u32 = first.output_columns;
    projection_constants[2].u32 = first.block_count;
    projection_constants[3].u32 = static_cast<uint32_t>(total_rows);
    ncnn::VkMat projection_dispatcher;
    projection_dispatcher.w = static_cast<int>(first.output_columns * 32);
    projection_dispatcher.h = static_cast<int>(total_rows);
    projection_dispatcher.c = 1;
    command.record_pipeline_readonly(
        first.pipeline.get(),
        projection_bindings,
        {1, 1, 1, 0},
        projection_constants,
        projection_dispatcher);

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        NcnnVulkanGatedDeltaState::Implementation& state_implementation =
            *states[entry_index]->implementation_;
        ncnn::VkMat fused_view = row_view(fused_gpu, entry_index, 1);
        ncnn::VkMat recurrent_view =
            row_view(recurrent_output_gpu, entry_index, 1);
        if (fused_view.empty() || recurrent_view.empty())
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
        std::vector<ncnn::VkMat> delta_bindings = {
            fused_view,
            state_implementation.convolution,
            state_implementation.recurrent,
            implementation.convolution_weight,
            implementation.time_bias,
            implementation.decay_log,
            implementation.norm_weight,
            recurrent_view};
        std::vector<ncnn::vk_constant_type> delta_constants(10);
        delta_constants[0].u32 = implementation.convolution_size;
        delta_constants[1].u32 = implementation.value_size;
        delta_constants[2].u32 = implementation.fused_columns;
        delta_constants[3].u32 = implementation.head_count;
        delta_constants[4].u32 = implementation.kv_head_count;
        delta_constants[5].u32 = implementation.head_dimension;
        delta_constants[6].u32 = implementation.value_head_dimension;
        delta_constants[7].u32 = implementation.convolution_kernel_size;
        delta_constants[8].u32 = 1;
        delta_constants[9].f = implementation.norm_epsilon;
        ncnn::VkMat delta_dispatcher;
        delta_dispatcher.w = 128;
        delta_dispatcher.h = 1;
        delta_dispatcher.c = 1;
        command.record_pipeline_readonly(
            implementation.pipeline.get(),
            delta_bindings,
            {0, 0, 0, 1, 1, 1, 1, 0},
            delta_constants,
            delta_dispatcher);
        if (!record_state_transaction_row(state_implementation, command))
            return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    }

    std::vector<ncnn::VkMat> output_bindings = {
        recurrent_output_gpu,
        output_operator.packed,
        output_operator.bias,
        output_gpu};
    std::vector<ncnn::vk_constant_type> output_constants(4);
    output_constants[0].u32 = output_operator.input_columns;
    output_constants[1].u32 = output_operator.output_columns;
    output_constants[2].u32 = output_operator.block_count;
    output_constants[3].u32 = static_cast<uint32_t>(total_rows);
    ncnn::VkMat output_dispatcher;
    output_dispatcher.w = static_cast<int>(output_operator.output_columns * 32);
    output_dispatcher.h = static_cast<int>(total_rows);
    output_dispatcher.c = 1;
    command.record_pipeline_readonly(
        output_operator.pipeline.get(),
        output_bindings,
        {1, 1, 1, 0},
        output_constants,
        output_dispatcher);

    if (!record_prepared_activation_staging_download(
            output_gpu,
            total_rows,
            output_operator.output_columns,
            transfer_slot.download,
            command,
            implementation.vulkan_context->device(),
            implementation.option,
            combined_projected.dtype()))
    {
        return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    }
    if (submit_compute_and_wait(command, runtime_state) != 0)
    {
        transaction_recording.mark_submit_failed();
        mark_nontransaction_existing_states_unknown();
        return NcnnVulkanGatedDeltaBatchResult::Failed;
    }
    transaction_recording.mark_submitted();
    if (!copy_staging_to_cpu_batch(
            transfer_slot.download,
            combined_projected))
    {
        mark_nontransaction_existing_states_unknown();
        return NcnnVulkanGatedDeltaBatchResult::Failed;
    }
    transaction_recording.commit_rows();
    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        NcnnVulkanGatedDeltaState::Implementation& state_implementation =
            *states[entry_index]->implementation_;
        state_implementation.initialized = true;
        state_implementation.cpu_state_pending = false;
        state_implementation.cpu_convolution = ncnn::VkMat();
        state_implementation.cpu_recurrent = ncnn::VkMat();
        state_implementation.zero_convolution = ncnn::VkMat();
        state_implementation.zero_recurrent = ncnn::VkMat();
    }
    lock.unlock();

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        CpuLayerCache& cache = *entries[entry_index].cache;
        std::copy_n(
            combined_projected.row(entry_index),
            output_operator.output_columns,
            entries[entry_index].projected->row(0));
        cache.gated_delta_convolution.clear();
        cache.gated_delta_recurrent.clear();
        cache.gated_delta_token_count += 1;
        cache.device_allocated_bytes = states[entry_index]->allocated_bytes();
    }
    runtime_state.dispatches += entries.size() + 2;
    runtime_state.gated_delta_fusions += entries.size();
    ++runtime_state.gated_delta_submissions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    creation_attempt.complete();
    return NcnnVulkanGatedDeltaBatchResult::Executed;
#else
    (void)entries;
    return NcnnVulkanGatedDeltaBatchResult::NotExecuted;
#endif
}

class NcnnVulkanFloat8Operator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> tiled4_pipeline;
    std::shared_ptr<ncnn::Pipeline> quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> swiglu_quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> rms_norm_quantize_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    ncnn::VkMat input_rms_norm_weight;
    ncnn::Option option;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
#endif
    uint32_t matrix_input_columns = 0;
    uint32_t logical_input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t output_columns_per_group = 0;
    uint32_t block_count = 0;
    uint32_t input_group_count = 1;
    float rms_norm_epsilon = 0.0f;
    float input_rms_norm_epsilon = 0.0f;
};

#if NCNN_MOE_WITH_VULKAN
static constexpr char float8_projection_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    float input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer scale_blob
{
    float scale_data[];
};
layout(binding = 3) readonly buffer bias_blob
{
    float bias_data[];
};
layout(binding = 4) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint matrix_input_columns;
    uint logical_input_columns;
    uint output_columns;
    uint output_columns_per_group;
    uint block_count;
    uint token_count;
}
p;

uint packed_byte(uint index)
{
    const uint word = packed_words[index >> 2];
    return (word >> ((index & 3) * 8)) & 255;
}

float decode_float8_e4m3(uint value)
{
    const uint magnitude = value & 127;
    const uint exponent = magnitude >> 3;
    const uint mantissa = magnitude & 7;
    float decoded = 0.0;
    if (exponent == 0)
        decoded = float(mantissa) * 0.001953125;
    else
        decoded = uintBitsToFloat(((exponent + 120) << 23) | (mantissa << 20));
    return (value & 128) == 0 ? decoded : -decoded;
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum[32];
#endif

void main()
{
    const uint output_column = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const bool valid = output_column < p.output_columns && token < p.token_count;
    const uint input_group = output_column / p.output_columns_per_group;
    const uint input_row = token * p.logical_input_columns + input_group * p.matrix_input_columns;
    const uint weight_row = output_column * p.matrix_input_columns;
    const uint scale_row = (output_column / 128) * p.block_count;
    float sum = 0.0;
    for (uint block = 0; block < p.block_count; ++block)
    {
        if (valid && lane < 32)
        {
            const uint block_begin = block * 128;
            const float scale = scale_data[scale_row + block];
            const uint column = block_begin + lane * 4;
            const uint word = packed_words[(weight_row + column) >> 2];
            for (uint offset = 0; offset < 4; ++offset)
            {
                const float weight = decode_float8_e4m3((word >> (offset * 8)) & 255);
                sum += weight * input_data[input_row + column + offset] * scale;
            }
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_sum = subgroupAdd(sum);
    if (valid && lane == 0)
        output_data[token * p.output_columns + output_column] = reduced_sum + bias_data[output_column];
#else
    partial_sum[lane] = sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[lane] += partial_sum[lane + stride];
        barrier();
    }
    if (valid && lane == 0)
        output_data[token * p.output_columns + output_column] = partial_sum[0] + bias_data[output_column];
#endif
}
)glsl";

// Four-output FP8 GEMV variant.  A workgroup shares each 128-value input
// block across four adjacent output rows, reducing repeated activation loads
// for decode-sized batches while retaining the exact FP32 accumulation and
// FP8 dequantization of the scalar path.  It is only selected for a single
// input group, where the four output rows address the same input row.
static constexpr char float8_projection_tiled4_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    float input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer scale_blob
{
    float scale_data[];
};
layout(binding = 3) readonly buffer bias_blob
{
    float bias_data[];
};
layout(binding = 4) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint matrix_input_columns;
    uint logical_input_columns;
    uint output_columns;
    uint output_columns_per_group;
    uint block_count;
    uint token_count;
}
p;

float decode_float8_e4m3(uint value)
{
    const uint magnitude = value & 127;
    const uint exponent = magnitude >> 3;
    const uint mantissa = magnitude & 7;
    float decoded = 0.0;
    if (exponent == 0)
        decoded = float(mantissa) * 0.001953125;
    else
        decoded = uintBitsToFloat(((exponent + 120) << 23) | (mantissa << 20));
    return (value & 128) == 0 ? decoded : -decoded;
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum0[32];
shared float partial_sum1[32];
shared float partial_sum2[32];
shared float partial_sum3[32];
#endif

void main()
{
    const uint output_base = gl_WorkGroupID.x * 4;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const bool token_valid = token < p.token_count;
    float sum0 = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;

    for (uint block = 0; block < p.block_count; ++block)
    {
        const uint block_begin = block * 128;
        const uint input_column = lane * 4;
        const float input0 = token_valid && block_begin + input_column < p.matrix_input_columns
            ? input_data[token * p.logical_input_columns + block_begin + input_column]
            : 0.0;
        const float input1 = token_valid && block_begin + input_column + 1 < p.matrix_input_columns
            ? input_data[token * p.logical_input_columns + block_begin + input_column + 1]
            : 0.0;
        const float input2 = token_valid && block_begin + input_column + 2 < p.matrix_input_columns
            ? input_data[token * p.logical_input_columns + block_begin + input_column + 2]
            : 0.0;
        const float input3 = token_valid && block_begin + input_column + 3 < p.matrix_input_columns
            ? input_data[token * p.logical_input_columns + block_begin + input_column + 3]
            : 0.0;

        if (token_valid)
        {
            const uint output0 = output_base;
            const uint output1 = output_base + 1;
            const uint output2 = output_base + 2;
            const uint output3 = output_base + 3;
            const uint word_base0 = (output0 * p.matrix_input_columns + block_begin + input_column) >> 2;
            const uint word_base1 = (output1 * p.matrix_input_columns + block_begin + input_column) >> 2;
            const uint word_base2 = (output2 * p.matrix_input_columns + block_begin + input_column) >> 2;
            const uint word_base3 = (output3 * p.matrix_input_columns + block_begin + input_column) >> 2;
            const float scale0 = output0 < p.output_columns
                ? scale_data[(output0 / 128) * p.block_count + block]
                : 0.0;
            const float scale1 = output1 < p.output_columns
                ? scale_data[(output1 / 128) * p.block_count + block]
                : 0.0;
            const float scale2 = output2 < p.output_columns
                ? scale_data[(output2 / 128) * p.block_count + block]
                : 0.0;
            const float scale3 = output3 < p.output_columns
                ? scale_data[(output3 / 128) * p.block_count + block]
                : 0.0;
            for (uint offset = 0; offset < 4; ++offset)
            {
                const float input_value = offset == 0
                    ? input0
                    : offset == 1
                        ? input1
                        : offset == 2
                            ? input2
                            : input3;
                if (output0 < p.output_columns)
                    sum0 += decode_float8_e4m3((packed_words[word_base0 + (offset >> 2)] >> ((offset & 3) * 8)) & 255) * input_value * scale0;
                if (output1 < p.output_columns)
                    sum1 += decode_float8_e4m3((packed_words[word_base1 + (offset >> 2)] >> ((offset & 3) * 8)) & 255) * input_value * scale1;
                if (output2 < p.output_columns)
                    sum2 += decode_float8_e4m3((packed_words[word_base2 + (offset >> 2)] >> ((offset & 3) * 8)) & 255) * input_value * scale2;
                if (output3 < p.output_columns)
                sum3 += decode_float8_e4m3((packed_words[word_base3 + (offset >> 2)] >> ((offset & 3) * 8)) & 255) * input_value * scale3;
            }
        }
    }

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_sum0 = subgroupAdd(sum0);
    const float reduced_sum1 = subgroupAdd(sum1);
    const float reduced_sum2 = subgroupAdd(sum2);
    const float reduced_sum3 = subgroupAdd(sum3);
    if (token_valid && lane == 0)
    {
        if (output_base < p.output_columns)
            output_data[token * p.output_columns + output_base] = reduced_sum0 + bias_data[output_base];
        if (output_base + 1 < p.output_columns)
            output_data[token * p.output_columns + output_base + 1] = reduced_sum1 + bias_data[output_base + 1];
        if (output_base + 2 < p.output_columns)
            output_data[token * p.output_columns + output_base + 2] = reduced_sum2 + bias_data[output_base + 2];
        if (output_base + 3 < p.output_columns)
            output_data[token * p.output_columns + output_base + 3] = reduced_sum3 + bias_data[output_base + 3];
    }
#else
    partial_sum0[lane] = sum0;
    partial_sum1[lane] = sum1;
    partial_sum2[lane] = sum2;
    partial_sum3[lane] = sum3;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            partial_sum0[lane] += partial_sum0[lane + stride];
            partial_sum1[lane] += partial_sum1[lane + stride];
            partial_sum2[lane] += partial_sum2[lane + stride];
            partial_sum3[lane] += partial_sum3[lane + stride];
        }
        barrier();
    }
    if (token_valid && lane == 0)
    {
        if (output_base < p.output_columns)
            output_data[token * p.output_columns + output_base] = partial_sum0[0] + bias_data[output_base];
        if (output_base + 1 < p.output_columns)
            output_data[token * p.output_columns + output_base + 1] = partial_sum1[0] + bias_data[output_base + 1];
        if (output_base + 2 < p.output_columns)
            output_data[token * p.output_columns + output_base + 2] = partial_sum2[0] + bias_data[output_base + 2];
        if (output_base + 3 < p.output_columns)
            output_data[token * p.output_columns + output_base + 3] = partial_sum3[0] + bias_data[output_base + 3];
    }
#endif
}
)glsl";

static constexpr char float8_quantize_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) buffer data_blob
{
    float data[];
};

layout(push_constant) uniform parameter
{
    uint columns;
    uint token_count;
}
p;

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float maxima[32];
#endif

float quantize_float8_e4m3(float value)
{
    const float sign_value = value < 0.0 ? -1.0 : 1.0;
    const float magnitude = min(abs(value), 448.0);
    if (magnitude < 0.015625)
    {
        const float mantissa = min(7.0, floor(magnitude * 512.0 + 0.5));
        return sign_value * mantissa * 0.001953125;
    }
    float exponent = floor(log2(magnitude));
    float encoded_exponent = exponent + 7.0;
    float mantissa = floor((magnitude * exp2(-exponent) - 1.0) * 8.0 + 0.5);
    if (mantissa >= 8.0)
    {
        mantissa = 0.0;
        encoded_exponent += 1.0;
    }
    if (encoded_exponent >= 15.0)
    {
        encoded_exponent = 15.0;
        mantissa = min(mantissa, 6.0);
    }
    const float decoded = encoded_exponent == 0.0
        ? mantissa * 0.001953125
        : (1.0 + mantissa * 0.125) * exp2(encoded_exponent - 7.0);
    return sign_value * decoded;
}

void main()
{
    const uint block = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint block_begin = block * 128;
    float maximum = 0.0001;
    for (uint offset = lane; offset < 128; offset += 32)
    {
        const uint column = block_begin + offset;
        if (token < p.token_count && column < p.columns)
            maximum = max(maximum, abs(data[token * p.columns + column]));
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float block_maximum = subgroupMax(maximum);
    const float scale = exp2(ceil(log2(block_maximum / 448.0)));
#else
    maxima[lane] = maximum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        barrier();
    }
    const float scale = exp2(ceil(log2(maxima[0] / 448.0)));
#endif
    for (uint offset = lane; offset < 128; offset += 32)
    {
        const uint column = block_begin + offset;
        if (token < p.token_count && column < p.columns)
        {
            const uint index = token * p.columns + column;
            data[index] = quantize_float8_e4m3(clamp(data[index] / scale, -448.0, 448.0)) * scale;
        }
    }
}
)glsl";

static constexpr char float8_rms_norm_quantize_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) buffer data_blob
{
    float data[];
};
layout(binding = 1) readonly buffer weight_blob
{
    float weights[];
};

layout(push_constant) uniform parameter
{
    uint columns;
    uint token_count;
    float epsilon;
}
p;

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial[32];
#endif

float quantize_float8_e4m3(float value)
{
    const float sign_value = value < 0.0 ? -1.0 : 1.0;
    const float magnitude = min(abs(value), 448.0);
    if (magnitude < 0.015625)
    {
        const float mantissa = min(7.0, floor(magnitude * 512.0 + 0.5));
        return sign_value * mantissa * 0.001953125;
    }
    float exponent = floor(log2(magnitude));
    float encoded_exponent = exponent + 7.0;
    float mantissa = floor((magnitude * exp2(-exponent) - 1.0) * 8.0 + 0.5);
    if (mantissa >= 8.0)
    {
        mantissa = 0.0;
        encoded_exponent += 1.0;
    }
    if (encoded_exponent >= 15.0)
    {
        encoded_exponent = 15.0;
        mantissa = min(mantissa, 6.0);
    }
    const float decoded = encoded_exponent == 0.0
        ? mantissa * 0.001953125
        : (1.0 + mantissa * 0.125) * exp2(encoded_exponent - 7.0);
    return sign_value * decoded;
}

void main()
{
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    float square_sum = 0.0;
    for (uint column = lane; column < p.columns; column += 32)
    {
        const float value = data[token * p.columns + column];
        square_sum += value * value;
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float total_square_sum = subgroupAdd(square_sum);
    const float inverse_rms = inversesqrt(total_square_sum / float(p.columns) + p.epsilon);
#else
    partial[lane] = square_sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial[lane] += partial[lane + stride];
        barrier();
    }
    const float inverse_rms = inversesqrt(partial[0] / float(p.columns) + p.epsilon);
#endif
    for (uint block_begin = 0; block_begin < p.columns; block_begin += 128)
    {
        float maximum = 0.0001;
        for (uint offset = lane; offset < 128; offset += 32)
        {
            const uint column = block_begin + offset;
            if (column < p.columns)
            {
                const uint index = token * p.columns + column;
                data[index] *= inverse_rms * weights[column];
                maximum = max(maximum, abs(data[index]));
            }
        }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
        const float block_maximum = subgroupMax(maximum);
        const float scale = exp2(ceil(log2(block_maximum / 448.0)));
#else
        partial[lane] = maximum;
        barrier();
        for (uint stride = 16; stride > 0; stride >>= 1)
        {
            if (lane < stride)
                partial[lane] = max(partial[lane], partial[lane + stride]);
            barrier();
        }
        const float scale = exp2(ceil(log2(partial[0] / 448.0)));
#endif
        for (uint offset = lane; offset < 128; offset += 32)
        {
            const uint column = block_begin + offset;
            if (column < p.columns)
            {
                const uint index = token * p.columns + column;
                data[index] = quantize_float8_e4m3(clamp(data[index] / scale, -448.0, 448.0)) * scale;
            }
        }
#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
        barrier();
#endif
    }
}
)glsl";

static constexpr char float8_swiglu_quantize_shader[] = R"glsl(
#version 450

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) buffer gate_blob
{
    float gate_data[];
};
layout(binding = 1) readonly buffer up_blob
{
    float up_data[];
};

layout(push_constant) uniform parameter
{
    uint columns;
    uint token_count;
    float activation_limit;
}
p;

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float maxima[32];
#endif

float quantize_float8_e4m3(float value)
{
    const float sign_value = value < 0.0 ? -1.0 : 1.0;
    const float magnitude = min(abs(value), 448.0);
    if (magnitude < 0.015625)
    {
        const float mantissa = min(7.0, floor(magnitude * 512.0 + 0.5));
        return sign_value * mantissa * 0.001953125;
    }
    float exponent = floor(log2(magnitude));
    float encoded_exponent = exponent + 7.0;
    float mantissa = floor((magnitude * exp2(-exponent) - 1.0) * 8.0 + 0.5);
    if (mantissa >= 8.0)
    {
        mantissa = 0.0;
        encoded_exponent += 1.0;
    }
    if (encoded_exponent >= 15.0)
    {
        encoded_exponent = 15.0;
        mantissa = min(mantissa, 6.0);
    }
    const float decoded = encoded_exponent == 0.0
        ? mantissa * 0.001953125
        : (1.0 + mantissa * 0.125) * exp2(encoded_exponent - 7.0);
    return sign_value * decoded;
}

void main()
{
    const uint block = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint block_begin = block * 128;
    float maximum = 0.0001;
    for (uint offset = lane; offset < 128; offset += 32)
    {
        const uint column = block_begin + offset;
        if (token < p.token_count && column < p.columns)
        {
            const uint index = token * p.columns + column;
            float gate = gate_data[index];
            float up = up_data[index];
            if (p.activation_limit > 0.0)
            {
                gate = min(gate, p.activation_limit);
                up = clamp(up, -p.activation_limit, p.activation_limit);
            }
            gate_data[index] = gate / (1.0 + exp(-gate)) * up;
            maximum = max(maximum, abs(gate_data[index]));
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float block_maximum = subgroupMax(maximum);
    const float scale = exp2(ceil(log2(block_maximum / 448.0)));
#else
    maxima[lane] = maximum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        barrier();
    }
    const float scale = exp2(ceil(log2(maxima[0] / 448.0)));
#endif
    for (uint offset = lane; offset < 128; offset += 32)
    {
        const uint column = block_begin + offset;
        if (token < p.token_count && column < p.columns)
        {
            const uint index = token * p.columns + column;
            gate_data[index] = quantize_float8_e4m3(clamp(gate_data[index] / scale, -448.0, 448.0)) * scale;
        }
    }
}
)glsl";

static bool create_float8_projection_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            float8_projection_shader,
            static_cast<int>(sizeof(float8_projection_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_projection_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_projection_shader, 0, destination);
    return true;
}

static bool create_float8_projection_tiled4_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            float8_projection_tiled4_shader,
            static_cast<int>(sizeof(float8_projection_tiled4_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_projection_tiled4_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(float8_projection_tiled4_shader, 0, destination);
    return true;
}

static bool create_float8_quantize_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            float8_quantize_shader,
            static_cast<int>(sizeof(float8_quantize_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_quantize_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_quantize_shader, 0, destination);
    return true;
}

static bool create_float8_rms_norm_quantize_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            float8_rms_norm_quantize_shader,
            static_cast<int>(sizeof(float8_rms_norm_quantize_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_rms_norm_quantize_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_rms_norm_quantize_shader, 0, destination);
    return true;
}

static bool create_float8_swiglu_quantize_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            float8_swiglu_quantize_shader,
            static_cast<int>(sizeof(float8_swiglu_quantize_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_swiglu_quantize_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_swiglu_quantize_shader, 0, destination);
    return true;
}

static size_t align_float8_upload(size_t bytes) noexcept
{
    return (bytes + 3) & ~static_cast<size_t>(3);
}

static bool prepare_float8_upload(std::span<const uint8_t> source, ncnn::Mat& destination)
{
    const size_t padded_size = align_float8_upload(source.size());
    if (padded_size == 0 || padded_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    destination.create(static_cast<int>(padded_size), sizeof(uint8_t));
    if (destination.empty())
        return false;
    std::memset(destination.data, 0, padded_size);
    std::memcpy(destination.data, source.data(), source.size());
    return true;
}

static bool vulkan_float8_tiled4_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationVulkanFp8Tile4);
}

static const ncnn::Pipeline* select_float8_projection_pipeline(
    const std::shared_ptr<ncnn::Pipeline>& scalar_pipeline,
    const std::shared_ptr<ncnn::Pipeline>& tiled4_pipeline,
    uint32_t input_group_count,
    uint32_t matrix_input_columns,
    uint32_t logical_input_columns,
    uint32_t output_columns,
    uint64_t optimization_flags)
{
    if (vulkan_float8_tiled4_enabled(optimization_flags)
        && tiled4_pipeline
        && input_group_count == 1
        && matrix_input_columns == logical_input_columns
        && output_columns >= 4)
    {
        return tiled4_pipeline.get();
    }
    return scalar_pipeline.get();
}

static bool vulkan_float8_input_quantize_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanInputQuantize);
}

static bool record_float8_input_quantize(
    const std::shared_ptr<ncnn::Pipeline>& quantize_pipeline,
    ncnn::VkMat& input,
    uint32_t columns,
    size_t token_count,
    ncnn::VkCompute& command)
{
    if (!quantize_pipeline || columns == 0 || columns % 128 != 0 || token_count == 0
        || token_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    std::vector<ncnn::VkMat> bindings = {input};
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = columns;
    constants[1].u32 = static_cast<uint32_t>(token_count);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>((columns / 128) * 32);
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline(
        quantize_pipeline.get(),
        bindings,
        constants,
        dispatcher);
    return true;
}
#endif

NcnnVulkanFloat8Operator::NcnnVulkanFloat8Operator()
    : implementation_(new Implementation)
{
}

NcnnVulkanFloat8Operator::~NcnnVulkanFloat8Operator() = default;

std::shared_ptr<NcnnVulkanFloat8Operator> NcnnVulkanFloat8Operator::create(
    const TensorData& matrix,
    const TensorData* bias,
    uint32_t input_group_count,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::Float8E4M3 || matrix.shape.size() != 2 || matrix.shape[0] == 0 || matrix.shape[1] == 0
        || matrix.shape[1] % 128 != 0 || input_group_count == 0 || matrix.shape[0] % input_group_count != 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 128;
    const uint32_t output_block_count = (output_columns + 127) / 128;
    const std::span<const uint8_t> weights = matrix.float8_values();
    if (weights.size() != matrix.element_count()
        || matrix.quantization_scales.size() != static_cast<size_t>(output_block_count) * block_count)
    {
        return {};
    }
    if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }
    const uint64_t logical_input_columns = static_cast<uint64_t>(input_columns) * input_group_count;
    if (logical_input_columns > std::numeric_limits<uint32_t>::max())
        return {};

    std::shared_ptr<NcnnVulkanFloat8Operator> result(new NcnnVulkanFloat8Operator);
    Implementation& implementation = *result->implementation_;
    implementation.matrix_input_columns = input_columns;
    implementation.logical_input_columns = static_cast<uint32_t>(logical_input_columns);
    implementation.output_columns = output_columns;
    implementation.output_columns_per_group = output_columns / input_group_count;
    implementation.block_count = block_count;
    implementation.input_group_count = input_group_count;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context = NcnnVulkanContext::acquire(
        vulkan_device_index,
        context_instance,
        optimization_flags);
    if (!implementation.vulkan_context)
        return {};
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.option.use_vulkan_compute = true;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;
    implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
    implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
    implementation.option.use_subgroup_ops = device->info.support_subgroup_ops();

    const uint64_t preferred_weight_bytes = align_float8_upload(weights.size())
                                            + static_cast<uint64_t>(matrix.quantization_scales.size()) * sizeof(float)
                                            + static_cast<uint64_t>(output_columns) * sizeof(float);
    if (preferred_weight_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return {};
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat packed;
    ncnn::Mat scales;
    ncnn::Mat biases;
    if (!prepare_float8_upload(weights, packed))
        return {};
    scales.create(static_cast<int>(matrix.quantization_scales.size()), sizeof(float));
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (scales.empty() || biases.empty())
        return {};
    std::copy(matrix.quantization_scales.begin(), matrix.quantization_scales.end(), static_cast<float*>(scales.data));
    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values = bias->bfloat16_values();
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(values[index]);
    }

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_float8_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline)
            || !create_float8_quantize_pipeline(implementation.vulkan_context, implementation.option, implementation.quantize_pipeline)
            || !create_float8_swiglu_quantize_pipeline(
                implementation.vulkan_context,
                implementation.option,
                implementation.swiglu_quantize_pipeline))
            return {};
        if (vulkan_float8_tiled4_enabled(optimization_flags))
        {
            (void)create_float8_projection_tiled4_pipeline(
                implementation.vulkan_context,
                implementation.option,
                implementation.tiled4_pipeline);
        }
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(packed, implementation.packed, upload_option);
        command.record_upload(scales, implementation.scales, upload_option);
        command.record_upload(biases, implementation.bias, upload_option);
        uploaded = !implementation.packed.empty() && !implementation.scales.empty() && !implementation.bias.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)input_group_count;
    (void)vulkan_device_index;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

bool NcnnVulkanFloat8Operator::prepare_rms_norm_weight(
    const TensorData& weight,
    uint32_t expected_columns,
    float epsilon,
    ncnn::VkMat& destination)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context || epsilon <= 0.0f
        || expected_columns == 0
        || weight.shape != std::vector<uint32_t>{expected_columns}
        || (weight.dtype != DType::Float32 && weight.dtype != DType::BFloat16))
    {
        return false;
    }
    ncnn::Mat values;
    values.create(static_cast<int>(expected_columns), sizeof(float));
    if (values.empty())
        return false;
    float* values_destination = static_cast<float*>(values.data);
    if (weight.dtype == DType::Float32)
    {
        const std::span<const float> source = weight.float32_values();
        if (source.size() != expected_columns)
            return false;
        std::copy(source.begin(), source.end(), values_destination);
    }
    else
    {
        const std::span<const uint16_t> source = weight.bfloat16_values();
        if (source.size() != expected_columns)
            return false;
        for (uint32_t index = 0; index < expected_columns; ++index)
            values_destination[index] = bfloat16_to_float(source[index]);
    }
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_float8_rms_norm_quantize_pipeline(
                implementation.vulkan_context,
                implementation.option,
                implementation.rms_norm_quantize_pipeline))
        {
            return false;
        }
    }
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(values, destination, upload_option);
        uploaded = !destination.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    return uploaded;
#else
    (void)weight;
    (void)expected_columns;
    (void)epsilon;
    (void)destination;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::prepare_rms_norm(
    const TensorData& weight,
    float epsilon)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (!prepare_rms_norm_weight(
            weight,
            implementation.output_columns,
            epsilon,
            implementation.rms_norm_weight))
        return false;
    implementation.rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::prepare_input_rms_norm(
    const TensorData& weight,
    float epsilon)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (!prepare_rms_norm_weight(
            weight,
            implementation.logical_input_columns,
            epsilon,
            implementation.input_rms_norm_weight))
        return false;
    implementation.input_rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context || !implementation.pipeline || input.rows() == 0
        || input.columns() != implementation.logical_input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = direct_host_input_enabled(
        *implementation.vulkan_context,
        static_cast<size_t>(input.rows())
            * implementation.logical_input_columns * sizeof(float),
        input.dtype());
    const bool direct_host_output = direct_host_output_enabled(
        *implementation.vulkan_context,
        static_cast<size_t>(input.rows()) * implementation.output_columns
            * sizeof(float),
        output.dtype());
    const bool gpu_input_quantize =
        vulkan_float8_input_quantize_enabled(implementation.optimization_flags) && implementation.quantize_pipeline;
    ActivationBuffer quantized_input;
    if (!gpu_input_quantize)
    {
        quantized_input = input;
        for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
            quantize_float8_e4m3_inplace(
                quantized_input.row(token_index),
                implementation.logical_input_columns,
                128,
                true,
                implementation.optimization_flags);
    }
    const ActivationBuffer& upload_input = gpu_input_quantize ? input : quantized_input;
    if (!fill_staging_upload(upload_input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output = ActivationBuffer(input.rows(), implementation.output_columns, output_dtype);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
        return false;
    if (gpu_input_quantize
        && !record_float8_input_quantize(
            implementation.quantize_pipeline,
            input_gpu,
            implementation.logical_input_columns,
            input.rows(),
            command))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(implementation.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(5);
    bindings[0] = input_gpu;
    bindings[1] = implementation.packed;
    bindings[2] = implementation.scales;
    bindings[3] = implementation.bias;
    bindings[4] = output_gpu;
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.matrix_input_columns;
    constants[1].u32 = implementation.logical_input_columns;
    constants[2].u32 = implementation.output_columns;
    constants[3].u32 = implementation.output_columns_per_group;
    constants[4].u32 = implementation.block_count;
    constants[5].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline_readonly(
        select_float8_projection_pipeline(
            implementation.pipeline,
            implementation.tiled4_pipeline,
            implementation.input_group_count,
            implementation.matrix_input_columns,
            implementation.logical_input_columns,
            implementation.output_columns,
            implementation.optimization_flags),
        bindings,
        {1, 1, 1, 1, 0},
        constants,
        dispatcher);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
                output_gpu,
                input.rows(),
                implementation.output_columns,
                transfer_slot.download,
                command,
                implementation.vulkan_context->device(),
                implementation.option,
                output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 1 + static_cast<uint64_t>(gpu_input_quantize);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_chain(const ActivationBuffer& input, const NcnnVulkanFloat8Operator& next, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *implementation_;
    const Implementation& second = *next.implementation_;
    if (!first.vulkan_context || !second.vulkan_context || first.vulkan_context.get() != second.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !first.quantize_pipeline || input.rows() == 0
        || input.columns() != first.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        first.vulkan_context->runtime_state();
    NcnnVulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = direct_host_output_enabled(
        *first.vulkan_context,
        static_cast<size_t>(input.rows()) * second.output_columns
            * sizeof(float),
        output.dtype());
    const bool gpu_input_quantize =
        vulkan_float8_input_quantize_enabled(first.optimization_flags) && first.quantize_pipeline;
    ActivationBuffer quantized_input;
    if (!gpu_input_quantize)
    {
        quantized_input = input;
        for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
            quantize_float8_e4m3_inplace(quantized_input.row(token_index), first.logical_input_columns, 128, true,
                                         first.optimization_flags);
    }
    const ActivationBuffer& upload_input = gpu_input_quantize ? input : quantized_input;
    if (!fill_staging_upload(upload_input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output = ActivationBuffer(input.rows(), second.output_columns, output_dtype);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    if (gpu_input_quantize
        && !record_float8_input_quantize(
            first.quantize_pipeline,
            input_gpu,
            first.logical_input_columns,
            input.rows(),
            command))
    {
        return false;
    }
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(second.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            first.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> first_bindings = {input_gpu, first.packed, first.scales, first.bias, intermediate_gpu};
    std::vector<ncnn::vk_constant_type> first_constants(6);
    first_constants[0].u32 = first.matrix_input_columns;
    first_constants[1].u32 = first.logical_input_columns;
    first_constants[2].u32 = first.output_columns;
    first_constants[3].u32 = first.output_columns_per_group;
    first_constants[4].u32 = first.block_count;
    first_constants[5].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat first_dispatcher;
    first_dispatcher.w = static_cast<int>(first.output_columns * 32);
    first_dispatcher.h = static_cast<int>(input.rows());
    first_dispatcher.c = 1;
    command.record_pipeline_readonly(
        select_float8_projection_pipeline(
            first.pipeline,
            first.tiled4_pipeline,
            first.input_group_count,
            first.matrix_input_columns,
            first.logical_input_columns,
            first.output_columns,
            first.optimization_flags),
        first_bindings,
        {1, 1, 1, 1, 0},
        first_constants,
        first_dispatcher);

    std::vector<ncnn::VkMat> quantize_bindings = {intermediate_gpu};
    std::vector<ncnn::vk_constant_type> quantize_constants(2);
    quantize_constants[0].u32 = first.output_columns;
    quantize_constants[1].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat quantize_dispatcher;
    quantize_dispatcher.w = static_cast<int>((first.output_columns / 128) * 32);
    quantize_dispatcher.h = static_cast<int>(input.rows());
    quantize_dispatcher.c = 1;
    command.record_pipeline(first.quantize_pipeline.get(), quantize_bindings, quantize_constants, quantize_dispatcher);

    std::vector<ncnn::VkMat> second_bindings = {intermediate_gpu, second.packed, second.scales, second.bias, output_gpu};
    std::vector<ncnn::vk_constant_type> second_constants(6);
    second_constants[0].u32 = second.matrix_input_columns;
    second_constants[1].u32 = second.logical_input_columns;
    second_constants[2].u32 = second.output_columns;
    second_constants[3].u32 = second.output_columns_per_group;
    second_constants[4].u32 = second.block_count;
    second_constants[5].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat second_dispatcher;
    second_dispatcher.w = static_cast<int>(second.output_columns * 32);
    second_dispatcher.h = static_cast<int>(input.rows());
    second_dispatcher.c = 1;
    command.record_pipeline_readonly(
        select_float8_projection_pipeline(
            second.pipeline,
            second.tiled4_pipeline,
            second.input_group_count,
            second.matrix_input_columns,
            second.logical_input_columns,
            second.output_columns,
            second.optimization_flags),
        second_bindings,
        {1, 1, 1, 1, 0},
        second_constants,
        second_dispatcher);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
             output_gpu,
             input.rows(),
             second.output_columns,
             transfer_slot.download,
             command,
             first.vulkan_context->device(),
             first.option,
             output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2 + static_cast<uint64_t>(gpu_input_quantize);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)next;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_rms_norm_chain(
    const ActivationBuffer& input,
    const NcnnVulkanFloat8Operator& next,
    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *implementation_;
    const Implementation& second = *next.implementation_;
    if (!first.vulkan_context || !second.vulkan_context || first.vulkan_context.get() != second.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || first.rms_norm_epsilon <= 0.0f || input.rows() == 0
        || input.columns() != first.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        first.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = direct_host_output_enabled(
        *first.vulkan_context,
        static_cast<size_t>(input.rows()) * second.output_columns
            * sizeof(float),
        output.dtype());
    const bool gpu_input_quantize =
        vulkan_float8_input_quantize_enabled(first.optimization_flags) && first.rms_norm_quantize_pipeline;
    ActivationBuffer quantized_input;
    if (!gpu_input_quantize)
    {
        quantized_input = input;
        for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
            quantize_float8_e4m3_inplace(quantized_input.row(token_index), first.logical_input_columns, 128, true,
                                         first.optimization_flags);
    }
    const ActivationBuffer& upload_input = gpu_input_quantize ? input : quantized_input;
    if (!fill_staging_upload(upload_input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output = ActivationBuffer(input.rows(), second.output_columns, output_dtype);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    if (gpu_input_quantize
        && !record_float8_input_quantize(
            first.quantize_pipeline,
            input_gpu,
            first.logical_input_columns,
            input.rows(),
            command))
    {
        return false;
    }
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(second.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            first.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> first_bindings = {input_gpu, first.packed, first.scales, first.bias, intermediate_gpu};
    std::vector<ncnn::vk_constant_type> first_constants(6);
    first_constants[0].u32 = first.matrix_input_columns;
    first_constants[1].u32 = first.logical_input_columns;
    first_constants[2].u32 = first.output_columns;
    first_constants[3].u32 = first.output_columns_per_group;
    first_constants[4].u32 = first.block_count;
    first_constants[5].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat first_dispatcher;
    first_dispatcher.w = static_cast<int>(first.output_columns * 32);
    first_dispatcher.h = static_cast<int>(input.rows());
    first_dispatcher.c = 1;
    command.record_pipeline_readonly(
        select_float8_projection_pipeline(
            first.pipeline,
            first.tiled4_pipeline,
            first.input_group_count,
            first.matrix_input_columns,
            first.logical_input_columns,
            first.output_columns,
            first.optimization_flags),
        first_bindings,
        {1, 1, 1, 1, 0},
        first_constants,
        first_dispatcher);

    std::vector<ncnn::VkMat> norm_bindings = {intermediate_gpu, first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline_readonly(
        first.rms_norm_quantize_pipeline.get(),
        norm_bindings,
        {0, 1},
        norm_constants,
        norm_dispatcher);

    std::vector<ncnn::VkMat> second_bindings = {intermediate_gpu, second.packed, second.scales, second.bias, output_gpu};
    std::vector<ncnn::vk_constant_type> second_constants(6);
    second_constants[0].u32 = second.matrix_input_columns;
    second_constants[1].u32 = second.logical_input_columns;
    second_constants[2].u32 = second.output_columns;
    second_constants[3].u32 = second.output_columns_per_group;
    second_constants[4].u32 = second.block_count;
    second_constants[5].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat second_dispatcher;
    second_dispatcher.w = static_cast<int>(second.output_columns * 32);
    second_dispatcher.h = static_cast<int>(input.rows());
    second_dispatcher.c = 1;
    command.record_pipeline_readonly(
        select_float8_projection_pipeline(
            second.pipeline,
            second.tiled4_pipeline,
            second.input_group_count,
            second.matrix_input_columns,
            second.logical_input_columns,
            second.output_columns,
            second.optimization_flags),
        second_bindings,
        {1, 1, 1, 1, 0},
        second_constants,
        second_dispatcher);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
             output_gpu,
             input.rows(),
             second.output_columns,
             transfer_slot.download,
             command,
             first.vulkan_context->device(),
             first.option,
             output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2 + static_cast<uint64_t>(gpu_input_quantize);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)next;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_rms_norm_chain_parallel_impl(
    const ActivationBuffer& input,
    const NcnnVulkanFloat8Operator& next,
    const NcnnVulkanFloat8Operator& parallel_operator,
    ActivationBuffer& output,
    ActivationBuffer& parallel_output,
    bool normalize_input) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *implementation_;
    const Implementation& second = *next.implementation_;
    const Implementation& parallel = *parallel_operator.implementation_;
    if (normalize_input
        && !runtime_optimization_enabled(
            first.optimization_flags,
            RuntimeOptimizationVulkanLatentInputRmsNorm))
    {
        return false;
    }
    if (!first.vulkan_context || !second.vulkan_context || !parallel.vulkan_context
        || first.vulkan_context.get() != second.vulkan_context.get()
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !parallel.pipeline
        || !first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || (normalize_input && first.input_rms_norm_weight.empty())
        || first.rms_norm_epsilon <= 0.0f || input.rows() == 0
        || (normalize_input && first.input_rms_norm_epsilon <= 0.0f)
        || input.columns() != first.logical_input_columns
        || input.columns() != parallel.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        first.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = direct_host_output_enabled(
        *first.vulkan_context,
        static_cast<size_t>(input.rows()) * second.output_columns
            * sizeof(float),
        output.dtype());
    const bool parallel_direct_host_output = direct_host_output_enabled(
        *parallel.vulkan_context,
        static_cast<size_t>(input.rows()) * parallel.output_columns
            * sizeof(float),
        parallel_output.dtype());
    const bool gpu_input_quantize =
        !normalize_input
        && vulkan_float8_input_quantize_enabled(first.optimization_flags)
        && first.quantize_pipeline;
    ActivationBuffer quantized_input;
    if (!normalize_input && !gpu_input_quantize)
    {
        quantized_input = input;
        for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
            quantize_float8_e4m3_inplace(quantized_input.row(token_index), first.logical_input_columns, 128, true,
                                         first.optimization_flags);
    }
    const ActivationBuffer& upload_input = normalize_input || gpu_input_quantize ? input : quantized_input;
    ncnn::VkMat parallel_download;
    if (!fill_staging_upload(upload_input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(parallel_download, input.rows(), parallel.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    const DType parallel_output_dtype = parallel_output.dtype();
    output = ActivationBuffer(input.rows(), second.output_columns, output_dtype);
    parallel_output = ActivationBuffer(input.rows(), parallel.output_columns, parallel_output_dtype);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    if (normalize_input)
    {
        std::vector<ncnn::VkMat> input_norm_bindings = {
            input_gpu,
            first.input_rms_norm_weight};
        std::vector<ncnn::vk_constant_type> input_norm_constants(3);
        input_norm_constants[0].u32 = first.logical_input_columns;
        input_norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
        input_norm_constants[2].f = first.input_rms_norm_epsilon;
        ncnn::VkMat input_norm_dispatcher;
        input_norm_dispatcher.w = 32;
        input_norm_dispatcher.h = static_cast<int>(input.rows());
        input_norm_dispatcher.c = 1;
        command.record_pipeline_readonly(
            first.rms_norm_quantize_pipeline.get(),
            input_norm_bindings,
            {0, 1},
            input_norm_constants,
            input_norm_dispatcher);
    }
    else if (gpu_input_quantize
        && !record_float8_input_quantize(
            first.quantize_pipeline,
            input_gpu,
            first.logical_input_columns,
            input.rows(),
            command))
    {
        return false;
    }
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(second.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
    if (parallel_direct_host_output)
        parallel_gpu = prepare_direct_host_output(parallel_download, runtime_state);
    else
        parallel_gpu.create(
            static_cast<int>(parallel.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            first.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty() || parallel_gpu.empty())
        return false;

    const auto record_projection = [&](const Implementation& projection, const ncnn::VkMat& projection_input, const ncnn::VkMat& projection_output) {
        std::vector<ncnn::VkMat> bindings = {
            projection_input,
            projection.packed,
            projection.scales,
            projection.bias,
            projection_output,
        };
        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].u32 = projection.matrix_input_columns;
        constants[1].u32 = projection.logical_input_columns;
        constants[2].u32 = projection.output_columns;
        constants[3].u32 = projection.output_columns_per_group;
        constants[4].u32 = projection.block_count;
        constants[5].u32 = static_cast<uint32_t>(input.rows());
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(projection.output_columns * 32);
        dispatcher.h = static_cast<int>(input.rows());
        dispatcher.c = 1;
        command.record_pipeline_readonly(
            select_float8_projection_pipeline(
                projection.pipeline,
                projection.tiled4_pipeline,
                projection.input_group_count,
                projection.matrix_input_columns,
                projection.logical_input_columns,
                projection.output_columns,
                projection.optimization_flags),
            bindings,
            {1, 1, 1, 1, 0},
            constants,
            dispatcher);
    };
    record_projection(first, input_gpu, intermediate_gpu);

    std::vector<ncnn::VkMat> norm_bindings = {intermediate_gpu, first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline_readonly(
        first.rms_norm_quantize_pipeline.get(),
        norm_bindings,
        {0, 1},
        norm_constants,
        norm_dispatcher);

    record_projection(second, intermediate_gpu, output_gpu);
    record_projection(parallel, input_gpu, parallel_gpu);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
             output_gpu,
             input.rows(),
             second.output_columns,
             transfer_slot.download,
             command,
             first.vulkan_context->device(),
             first.option,
             output_dtype))
        || (!parallel_direct_host_output
            && !record_prepared_activation_staging_download(
                parallel_gpu,
                input.rows(),
                parallel.output_columns,
                parallel_download,
                command,
                first.vulkan_context->device(),
                first.option,
                parallel_output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    runtime_state.dispatches += 3 + static_cast<uint64_t>(gpu_input_quantize);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2;
    if (normalize_input)
        ++runtime_state.rms_norm_linear_fusions;
    return true;
#else
    (void)input;
    (void)next;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    (void)normalize_input;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_rms_norm_chain_parallel(
    const ActivationBuffer& input,
    const NcnnVulkanFloat8Operator& next,
    const NcnnVulkanFloat8Operator& parallel_operator,
    ActivationBuffer& output,
    ActivationBuffer& parallel_output) const
{
    return forward_rms_norm_chain_parallel_impl(
        input,
        next,
        parallel_operator,
        output,
        parallel_output,
        false);
}

bool NcnnVulkanFloat8Operator::forward_input_rms_norm_chain_parallel(
    const ActivationBuffer& input,
    const NcnnVulkanFloat8Operator& next,
    const NcnnVulkanFloat8Operator& parallel_operator,
    ActivationBuffer& output,
    ActivationBuffer& parallel_output) const
{
    return forward_rms_norm_chain_parallel_impl(
        input,
        next,
        parallel_operator,
        output,
        parallel_output,
        true);
}

bool NcnnVulkanFloat8Operator::forward_rms_norm_chain_parallel_bfloat16(
    const ActivationBuffer& input,
    const NcnnVulkanFloat8Operator& next,
    const NcnnVulkanFloat8Operator& parallel_operator,
    std::span<const NcnnVulkanBfloat16Operator*> extra_operators,
    std::span<ActivationBuffer*> extra_outputs,
    ActivationBuffer& output,
    ActivationBuffer& parallel_output) const
{
#if NCNN_MOE_WITH_VULKAN
    if (extra_operators.empty())
        return forward_rms_norm_chain_parallel(
            input,
            next,
            parallel_operator,
            output,
            parallel_output);
    if (extra_operators.size() != extra_outputs.size())
        return false;

    const Implementation& first = *implementation_;
    const Implementation& second = *next.implementation_;
    const Implementation& parallel = *parallel_operator.implementation_;
    if (!first.vulkan_context || !second.vulkan_context || !parallel.vulkan_context
        || first.vulkan_context.get() != second.vulkan_context.get()
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !parallel.pipeline
        || !first.quantize_pipeline
        || !first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || first.rms_norm_epsilon <= 0.0f || input.rows() == 0
        || input.columns() != first.logical_input_columns
        || input.columns() != parallel.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    size_t extra_storage_variant = 0;
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const NcnnVulkanBfloat16Operator* extra_operator =
            extra_operators[index];
        ActivationBuffer* extra_output = extra_outputs[index];
        if (!extra_operator || !extra_output)
            return false;
        const NcnnVulkanBfloat16Operator::Implementation& extra =
            *extra_operator->implementation_;
        if (!extra.vulkan_context
            || extra.vulkan_context.get() != first.vulkan_context.get()
            || !extra.pipeline
            || extra.input_columns != input.columns())
        {
            return false;
        }
        const size_t variant = vulkan_activation_storage_variant(extra.option);
        if (index == 0)
            extra_storage_variant = variant;
        else if (variant != extra_storage_variant)
            return false;
    }
    if (extra_storage_variant != vulkan_activation_storage_variant(first.option)
        || extra_storage_variant != 0)
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        first.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease =
        first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = direct_host_output_enabled(
        *first.vulkan_context,
        static_cast<size_t>(input.rows()) * second.output_columns
            * sizeof(float),
        output.dtype());
    const bool parallel_direct_host_output = direct_host_output_enabled(
        *parallel.vulkan_context,
        static_cast<size_t>(input.rows()) * parallel.output_columns
            * sizeof(float),
        parallel_output.dtype());
    ncnn::VkMat parallel_download;
    std::vector<ncnn::VkMat> extra_downloads(extra_operators.size());
    std::vector<bool> extra_direct_host_outputs(
        extra_operators.size(), false);
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(
               transfer_slot.download,
               input.rows(),
               second.output_columns,
               transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(
               parallel_download,
               input.rows(),
               parallel.output_columns,
               transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->implementation_;
        if (!prepare_staging_batch(
                extra_downloads[index],
                input.rows(),
                extra.output_columns,
                transfer_slot.staging_allocator, runtime_state))
        {
            return false;
        }
        extra_direct_host_outputs[index] = direct_host_output_enabled(
            *extra.vulkan_context,
            static_cast<size_t>(input.rows()) * extra.output_columns
                * sizeof(float),
            extra_outputs[index]->dtype());
    }
    const DType output_dtype = output.dtype();
    const DType parallel_output_dtype = parallel_output.dtype();
    output = ActivationBuffer(input.rows(), second.output_columns, output_dtype);
    parallel_output = ActivationBuffer(input.rows(), parallel.output_columns, parallel_output_dtype);
    for (size_t index = 0; index < extra_outputs.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->implementation_;
        const DType extra_output_dtype = extra_outputs[index]->dtype();
        *extra_outputs[index] = ActivationBuffer(
            input.rows(),
            extra.output_columns,
            extra_output_dtype);
    }

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat original_input_gpu;
    if (!record_mapped_upload(
            transfer_slot.upload,
            original_input_gpu,
            command,
            first.option))
    {
        return false;
    }
    ncnn::VkMat input_gpu;
    command.record_clone(original_input_gpu, input_gpu, first.option);
    if (input_gpu.empty())
        return false;
    std::vector<ncnn::VkMat> input_quantize_bindings = {input_gpu};
    std::vector<ncnn::vk_constant_type> input_quantize_constants(2);
    input_quantize_constants[0].u32 = first.logical_input_columns;
    input_quantize_constants[1].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat input_quantize_dispatcher;
    input_quantize_dispatcher.w = static_cast<int>(
        (first.logical_input_columns / 128) * 32);
    input_quantize_dispatcher.h = static_cast<int>(input.rows());
    input_quantize_dispatcher.c = 1;
    command.record_pipeline(
        first.quantize_pipeline.get(),
        input_quantize_bindings,
        input_quantize_constants,
        input_quantize_dispatcher);

    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(second.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
    if (parallel_direct_host_output)
        parallel_gpu = prepare_direct_host_output(parallel_download, runtime_state);
    else
        parallel_gpu.create(
            static_cast<int>(parallel.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            first.vulkan_context->blob_allocator());
    std::vector<ncnn::VkMat> extra_gpu(extra_operators.size());
    if (intermediate_gpu.empty() || output_gpu.empty() || parallel_gpu.empty())
        return false;
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->implementation_;
        if (extra_direct_host_outputs[index])
            extra_gpu[index] = prepare_direct_host_output(
                extra_downloads[index], runtime_state);
        else
            extra_gpu[index].create(
                static_cast<int>(extra.output_columns),
                static_cast<int>(input.rows()),
                vulkan_activation_element_size(extra.option),
                first.vulkan_context->blob_allocator());
        if (extra_gpu[index].empty())
            return false;
    }

    const auto record_projection = [&](const Implementation& projection,
                                       const ncnn::VkMat& projection_input,
                                       const ncnn::VkMat& projection_output) {
        std::vector<ncnn::VkMat> bindings = {
            projection_input,
            projection.packed,
            projection.scales,
            projection.bias,
            projection_output};
        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].u32 = projection.matrix_input_columns;
        constants[1].u32 = projection.logical_input_columns;
        constants[2].u32 = projection.output_columns;
        constants[3].u32 = projection.output_columns_per_group;
        constants[4].u32 = projection.block_count;
        constants[5].u32 = static_cast<uint32_t>(input.rows());
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(projection.output_columns * 32);
        dispatcher.h = static_cast<int>(input.rows());
        dispatcher.c = 1;
        command.record_pipeline_readonly(
            select_float8_projection_pipeline(
                projection.pipeline,
                projection.tiled4_pipeline,
                projection.input_group_count,
                projection.matrix_input_columns,
                projection.logical_input_columns,
                projection.output_columns,
                projection.optimization_flags),
            bindings,
            {1, 1, 1, 1, 0},
            constants,
            dispatcher);
    };
    record_projection(first, input_gpu, intermediate_gpu);

    std::vector<ncnn::VkMat> norm_bindings = {
        intermediate_gpu,
        first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline_readonly(
        first.rms_norm_quantize_pipeline.get(),
        norm_bindings,
        {0, 1},
        norm_constants,
        norm_dispatcher);
    record_projection(second, intermediate_gpu, output_gpu);
    record_projection(parallel, input_gpu, parallel_gpu);

    std::vector<bool> cooperative_dispatches(extra_operators.size(), false);
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->implementation_;
        cooperative_dispatches[index] = record_bfloat16_projection(
            extra.pipeline,
            extra.cooperative_pipeline,
            original_input_gpu,
            extra.packed,
            extra.bias,
            extra_gpu[index],
            extra.input_columns,
            extra.output_columns,
            extra.block_count,
            static_cast<uint32_t>(input.rows()),
            extra.cooperative_tile_m,
            extra.cooperative_tile_n,
            extra.cooperative_tile_k,
            extra.cooperative_subgroup_size,
            extra.cooperative_forced,
            command);
    }

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
             output_gpu,
             input.rows(),
             second.output_columns,
             transfer_slot.download,
             command,
             first.vulkan_context->device(),
             first.option,
             output_dtype))
        || (!parallel_direct_host_output
            && !record_prepared_activation_staging_download(
                parallel_gpu,
                input.rows(),
                parallel.output_columns,
                parallel_download,
                command,
                first.vulkan_context->device(),
                first.option,
                parallel_output_dtype)))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->implementation_;
        if (!extra_direct_host_outputs[index]
            && !record_prepared_activation_staging_download(
                extra_gpu[index],
                input.rows(),
                extra.output_columns,
                extra_downloads[index],
                command,
                extra.vulkan_context->device(),
                extra.option,
                extra_outputs[index]->dtype()))
        {
            return false;
        }
    }
    if (submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        if (!copy_staging_to_cpu_batch(
                extra_downloads[index],
                *extra_outputs[index]))
        {
            return false;
        }
    }
    runtime_state.dispatches +=
        4 + static_cast<uint64_t>(extra_operators.size());
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads +=
        2 + extra_operators.size();
    for (bool used_cooperative_matrix : cooperative_dispatches)
    {
        if (used_cooperative_matrix)
            ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)next;
    (void)parallel_operator;
    (void)extra_operators;
    (void)extra_outputs;
    (void)output;
    (void)parallel_output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_swiglu_chain(
    const ActivationBuffer& input,
    const NcnnVulkanFloat8Operator& up_operator,
    const NcnnVulkanFloat8Operator& down_operator,
    ExpertActivation activation,
    float activation_limit,
    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& gate = *implementation_;
    const Implementation& up = *up_operator.implementation_;
    const Implementation& down = *down_operator.implementation_;
    if (activation != ExpertActivation::DeepSeekSwiGlu
        || !gate.vulkan_context || !up.vulkan_context || !down.vulkan_context
        || gate.vulkan_context.get() != up.vulkan_context.get()
        || gate.vulkan_context.get() != down.vulkan_context.get()
        || !gate.pipeline || !up.pipeline || !down.pipeline || !gate.swiglu_quantize_pipeline
        || input.rows() == 0 || input.columns() != gate.logical_input_columns
        || input.columns() != up.logical_input_columns
        || gate.output_columns != up.output_columns
        || gate.output_columns != down.logical_input_columns
        || gate.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        gate.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = direct_host_output_enabled(
        *gate.vulkan_context,
        static_cast<size_t>(input.rows()) * down.output_columns
            * sizeof(float),
        output.dtype());
    const bool gpu_input_quantize =
        vulkan_float8_input_quantize_enabled(gate.optimization_flags) && gate.quantize_pipeline;
    ActivationBuffer quantized_input;
    if (!gpu_input_quantize)
    {
        quantized_input = input;
        for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
            quantize_float8_e4m3_inplace(quantized_input.row(token_index), gate.logical_input_columns, 128, true,
                                         gate.optimization_flags);
    }
    const ActivationBuffer& upload_input = gpu_input_quantize ? input : quantized_input;
    if (!fill_staging_upload(upload_input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);

    std::unique_lock<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
        return false;
    if (gpu_input_quantize
        && !record_float8_input_quantize(
            gate.quantize_pipeline,
            input_gpu,
            gate.logical_input_columns,
            input.rows(),
            command))
    {
        return false;
    }
    ncnn::VkMat gate_gpu;
    gate_gpu.create(
        static_cast<int>(gate.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        gate.vulkan_context->blob_allocator());
    ncnn::VkMat up_gpu;
    up_gpu.create(
        static_cast<int>(up.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        gate.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(
            static_cast<int>(down.output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            gate.vulkan_context->blob_allocator());
    if (gate_gpu.empty() || up_gpu.empty() || output_gpu.empty())
        return false;

    const auto record_projection = [&](const Implementation& projection, const ncnn::VkMat& projection_input, const ncnn::VkMat& projection_output) {
        std::vector<ncnn::VkMat> bindings = {
            projection_input,
            projection.packed,
            projection.scales,
            projection.bias,
            projection_output,
        };
        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].u32 = projection.matrix_input_columns;
        constants[1].u32 = projection.logical_input_columns;
        constants[2].u32 = projection.output_columns;
        constants[3].u32 = projection.output_columns_per_group;
        constants[4].u32 = projection.block_count;
        constants[5].u32 = static_cast<uint32_t>(input.rows());
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(projection.output_columns * 32);
        dispatcher.h = static_cast<int>(input.rows());
        dispatcher.c = 1;
        command.record_pipeline_readonly(
            select_float8_projection_pipeline(
                projection.pipeline,
                projection.tiled4_pipeline,
                projection.input_group_count,
                projection.matrix_input_columns,
                projection.logical_input_columns,
                projection.output_columns,
                projection.optimization_flags),
            bindings,
            {1, 1, 1, 1, 0},
            constants,
            dispatcher);
    };
    record_projection(gate, input_gpu, gate_gpu);
    record_projection(up, input_gpu, up_gpu);

    std::vector<ncnn::VkMat> activation_bindings = {gate_gpu, up_gpu};
    std::vector<ncnn::vk_constant_type> activation_constants(3);
    activation_constants[0].u32 = gate.output_columns;
    activation_constants[1].u32 = static_cast<uint32_t>(input.rows());
    activation_constants[2].f = activation_limit;
    ncnn::VkMat activation_dispatcher;
    activation_dispatcher.w = static_cast<int>((gate.output_columns / 128) * 32);
    activation_dispatcher.h = static_cast<int>(input.rows());
    activation_dispatcher.c = 1;
    command.record_pipeline(
        gate.swiglu_quantize_pipeline.get(),
        activation_bindings,
        activation_constants,
        activation_dispatcher);

    record_projection(down, gate_gpu, output_gpu);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
             output_gpu,
             input.rows(),
             down.output_columns,
             transfer_slot.download,
             command,
             gate.vulkan_context->device(),
             gate.option,
             output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 3 + static_cast<uint64_t>(gpu_input_quantize);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)up_operator;
    (void)down_operator;
    (void)activation;
    (void)activation_limit;
    (void)output;
    return false;
#endif
}

class NcnnVulkanMxfp4Operator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    // Keep the three read-only MXFP4 projection inputs in one device buffer.
    // Besides reducing allocator fragmentation, this gives the indexed MoE
    // shaders one descriptor per Expert instead of one descriptor for every
    // packed/scales/bias tensor.
    ncnn::VkMat storage;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::Option option;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
    bool indexed_storage = false;
    uint32_t indexed_scales_word_offset = 0;
    uint32_t indexed_bias_word_offset = 0;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

#if NCNN_MOE_WITH_VULKAN
static constexpr char mxfp4_projection_shader[] = R"glsl(
#version 450

#ifdef NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    float input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer scale_blob
{
    uint scale_words[];
};
layout(binding = 3) readonly buffer bias_blob
{
    float bias_data[];
};
layout(binding = 4) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint input_columns;
    uint output_columns;
    uint block_count;
    uint token_count;
}
p;

uint packed_byte(uint index)
{
    const uint word = packed_words[index >> 2];
    return (word >> ((index & 3) * 8)) & 255;
}

uint scale_byte(uint index)
{
    const uint word = scale_words[index >> 2];
    return (word >> ((index & 3) * 8)) & 255;
}

float decode_mxfp4(uint value)
{
    const float magnitudes[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    const float magnitude = magnitudes[value & 7];
    return (value & 8) == 0 ? magnitude : -magnitude;
}

float decode_scale(uint value)
{
    // E8M0 maps directly to an IEEE-754 exponent.
    return uintBitsToFloat(value == 0 ? 0x00400000u : value << 23);
}

float mxfp4_product(float value, float input_value, float scale)
{
#ifdef NCNN_fp16_arithmetic
    return float(float16_t(value) * float16_t(input_value) * float16_t(scale));
#else
    return value * input_value * scale;
#endif
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum[32];
#endif

void main()
{
    const uint output_column = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const bool valid = output_column < p.output_columns && token < p.token_count;
    float sum = 0.0;
    if (valid && lane < 32)
    {
        const uint packed_row = output_column * p.block_count * 16;
        const uint scale_row = output_column * p.block_count;
        const uint input_row = token * p.input_columns;
        for (uint block = 0; block < p.block_count; ++block)
        {
            const float scale = decode_scale(scale_byte(scale_row + block));
            const uint value = packed_byte(packed_row + block * 16 + lane / 2);
            const uint nibble = (lane & 1) == 0 ? value & 15 : value >> 4;
            sum += mxfp4_product(
                decode_mxfp4(nibble),
                input_data[input_row + block * 32 + lane],
                scale);
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_sum = subgroupAdd(sum);
    if (valid && lane == 0)
    {
        output_data[token * p.output_columns + output_column] = reduced_sum + bias_data[output_column];
    }
#else
    partial_sum[lane] = sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[lane] += partial_sum[lane + stride];
        barrier();
    }
    if (valid && lane == 0)
        output_data[token * p.output_columns + output_column] = partial_sum[0] + bias_data[output_column];
#endif
}
)glsl";

static bool create_mxfp4_projection_pipeline(const std::shared_ptr<NcnnVulkanContext>& context, const ncnn::Option& option,
                                             std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            mxfp4_projection_shader,
            static_cast<int>(sizeof(mxfp4_projection_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_projection_shader, pipeline_key);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(mxfp4_projection_shader, pipeline_key, destination);
    return true;
}

static size_t align_to_uint32(size_t bytes) noexcept
{
    return (bytes + 3) & ~static_cast<size_t>(3);
}

static bool prepare_mxfp4_upload(const MxFp4ByteBuffer& source, ncnn::Mat& destination)
{
    const size_t padded_size = align_to_uint32(source.size());
    if (padded_size == 0 || padded_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    destination.create(static_cast<int>(padded_size), sizeof(uint8_t));
    if (destination.empty())
        return false;
    std::memset(destination.data, 0, padded_size);
    std::memcpy(destination.data, source.data(), source.size());
    return true;
}
#endif

NcnnVulkanMxfp4Operator::NcnnVulkanMxfp4Operator()
    : implementation_(new Implementation)
{
}

NcnnVulkanMxfp4Operator::~NcnnVulkanMxfp4Operator() = default;

std::shared_ptr<NcnnVulkanMxfp4Operator> NcnnVulkanMxfp4Operator::create(const TensorData& matrix, const TensorData* bias,
                                                                         uint32_t vulkan_device_index,
                                                                         const NcnnVulkanContextInstancePtr& context_instance,
                                                                         uint64_t optimization_flags)
{
    return create_with_allocator(
        matrix,
        bias,
        vulkan_device_index,
        nullptr,
        context_instance,
        optimization_flags);
}

std::shared_ptr<NcnnVulkanMxfp4Operator> NcnnVulkanMxfp4Operator::create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                                        uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator,
                                                                                        const NcnnVulkanContextInstancePtr& context_instance,
                                                                                        uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::MxFp4 || matrix.shape.size() != 2 || matrix.shape[0] == 0 || matrix.shape[1] == 0 || matrix.shape[1] % 32 != 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 64)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 32;
    const uint64_t expected_blocks = static_cast<uint64_t>(output_columns) * block_count * 16;
    const uint64_t expected_scales = static_cast<uint64_t>(output_columns) * block_count;
    if (matrix.mxfp4_blocks.size() != expected_blocks || matrix.mxfp4_scales.size() != expected_scales)
    {
        return {};
    }
    if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<NcnnVulkanMxfp4Operator> result(new NcnnVulkanMxfp4Operator);
    Implementation& implementation = *result->implementation_;
    implementation.input_columns = input_columns;
    implementation.output_columns = output_columns;
    implementation.block_count = block_count;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context = NcnnVulkanContext::acquire(
        vulkan_device_index,
        context_instance,
        optimization_flags);
    if (!implementation.vulkan_context)
        return {};
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.option.use_vulkan_compute = true;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;
    implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
    implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
    implementation.option.use_subgroup_ops = device->info.support_subgroup_ops();
    const uint64_t preferred_weight_bytes = expected_blocks + expected_scales + static_cast<uint64_t>(output_columns) * sizeof(float);
    if (preferred_weight_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return {};
    }
    if (!weight_allocator)
    {
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(device, static_cast<size_t>(preferred_weight_bytes)));
        weight_allocator = implementation.weight_allocator.get();
    }
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat packed;
    ncnn::Mat scales;
    ncnn::Mat biases;
    if (!prepare_mxfp4_upload(matrix.mxfp4_blocks, packed) || !prepare_mxfp4_upload(matrix.mxfp4_scales, scales))
    {
        return {};
    }
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (biases.empty())
        return {};
    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        if (values.size() != output_columns)
            return {};
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values = bias->bfloat16_values();
        if (values.size() != output_columns)
            return {};
        for (uint32_t index = 0; index < output_columns; ++index)
        {
            bias_values[index] = bfloat16_to_float(values[index]);
        }
    }

    const size_t packed_bytes = packed.total() * packed.elemsize;
    const size_t scales_bytes = scales.total() * scales.elemsize;
    const size_t bias_bytes = biases.total() * biases.elemsize;
    const size_t storage_alignment = std::max<size_t>(
        4,
        device->info.buffer_offset_alignment());
    const auto aligned_segment = [storage_alignment](size_t bytes) -> size_t {
        const size_t remainder = bytes % storage_alignment;
        if (remainder == 0)
            return bytes;
        const size_t padding = storage_alignment - remainder;
        if (bytes > std::numeric_limits<size_t>::max() - padding)
            return 0;
        return bytes + padding;
    };
    const size_t packed_segment_bytes = aligned_segment(packed_bytes);
    const size_t scales_segment_bytes = aligned_segment(scales_bytes);
    const size_t bias_segment_bytes = aligned_segment(bias_bytes);
    if (packed_segment_bytes == 0 || scales_segment_bytes == 0 || bias_segment_bytes == 0
        || packed_segment_bytes > std::numeric_limits<size_t>::max() - scales_segment_bytes
        || packed_segment_bytes + scales_segment_bytes > std::numeric_limits<size_t>::max() - bias_segment_bytes)
    {
        return {};
    }
    const size_t scales_offset = packed_segment_bytes;
    const size_t bias_offset = packed_segment_bytes + scales_segment_bytes;
    const size_t storage_bytes = bias_offset + bias_segment_bytes;
    if (storage_bytes == 0 || storage_bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};

    ncnn::Mat combined;
    combined.create(static_cast<int>(storage_bytes), sizeof(uint8_t));
    if (combined.empty())
        return {};
    std::memset(combined.data, 0, storage_bytes);
    std::memcpy(combined.data, packed.data, packed_bytes);
    std::memcpy(static_cast<std::byte*>(combined.data) + scales_offset, scales.data, scales_bytes);
    std::memcpy(static_cast<std::byte*>(combined.data) + bias_offset, biases.data, bias_bytes);

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_mxfp4_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline))
        {
            return {};
        }
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        // Executable-cache admission runs on a background thread. Serialize
        // weight uploads with foreground compute and other context users;
        // ncnn's transfer command and Vulkan allocators are not an independent
        // command domain for this shared device context.
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(device);
        command.record_upload(combined, implementation.storage, upload_option);
        uploaded = !implementation.storage.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
    {
        return {};
    }

    const auto storage_view = [&implementation](size_t offset, size_t bytes) {
        ncnn::VkMat view = implementation.storage;
        view.dims = 1;
        view.w = static_cast<int>(bytes);
        view.h = 1;
        view.d = 1;
        view.c = 1;
        view.elemsize = sizeof(uint8_t);
        view.elempack = 1;
        view.cstep = bytes;
#if NCNN_BATCH
        view.n = 1;
        view.nstep = bytes;
#endif
        view.offset += offset;
        return view;
    };
    implementation.packed = storage_view(0, packed_bytes);
    implementation.scales = storage_view(scales_offset, scales_bytes);
    implementation.bias = storage_view(bias_offset, bias_bytes);
    implementation.indexed_storage = true;
    implementation.indexed_scales_word_offset = static_cast<uint32_t>(scales_offset / sizeof(uint32_t));
    implementation.indexed_bias_word_offset = static_cast<uint32_t>(bias_offset / sizeof(uint32_t));
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

bool NcnnVulkanMxfp4Operator::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context || !implementation.pipeline || input.rows() == 0 || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = direct_host_input_enabled(
        *implementation.vulkan_context,
        static_cast<size_t>(input.rows()) * implementation.input_columns
            * sizeof(float),
        input.dtype());
    const bool direct_host_output = direct_host_output_enabled(
        *implementation.vulkan_context,
        static_cast<size_t>(input.rows()) * implementation.output_columns
            * sizeof(float),
        output.dtype());
      if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output = ActivationBuffer(input.rows(), implementation.output_columns, output_dtype);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns), static_cast<int>(input.rows()), sizeof(float),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(5);
    bindings[0] = input_gpu;
    bindings[1] = implementation.packed;
    bindings[2] = implementation.scales;
    bindings[3] = implementation.bias;
    bindings[4] = output_gpu;
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(implementation.pipeline.get(), bindings, constants, dispatcher);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(
             output_gpu,
             input.rows(),
             implementation.output_columns,
             transfer_slot.download,
             command,
             implementation.vulkan_context->device(),
             implementation.option,
             output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

uint32_t NcnnVulkanMxfp4Operator::input_columns() const noexcept
{
    return implementation_->input_columns;
}

uint32_t NcnnVulkanMxfp4Operator::output_columns() const noexcept
{
    return implementation_->output_columns;
}

class NcnnVulkanMxfp4ExpertOperator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<ncnn::Pipeline> gate_up_pipeline;
#endif
    std::shared_ptr<NcnnVulkanMxfp4Operator> gate_up;
    std::shared_ptr<NcnnVulkanMxfp4Operator> down;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
};

#if NCNN_MOE_WITH_VULKAN
static constexpr char mxfp4_gate_up_shader[] = R"glsl(
#version 450

#ifdef NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    float input_data[];
};
layout(binding = 1) readonly buffer packed_blob
{
    uint packed_words[];
};
layout(binding = 2) readonly buffer scale_blob
{
    uint scale_words[];
};
layout(binding = 3) readonly buffer bias_blob
{
    float bias_data[];
};
layout(binding = 4) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint input_columns;
    uint intermediate_columns;
    uint block_count;
    uint token_count;
    uint activation;
    float activation_limit;
}
p;

uint packed_byte(uint index)
{
    const uint word = packed_words[index >> 2];
    return (word >> ((index & 3) * 8)) & 255;
}

uint scale_byte(uint index)
{
    const uint word = scale_words[index >> 2];
    return (word >> ((index & 3) * 8)) & 255;
}

float decode_mxfp4(uint value)
{
    const float magnitudes[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    const float magnitude = magnitudes[value & 7];
    return (value & 8) == 0 ? magnitude : -magnitude;
}

float decode_scale(uint value)
{
    return uintBitsToFloat(value == 0 ? 0x00400000u : value << 23);
}

float mxfp4_product(float value, float input_value, float scale)
{
#ifdef NCNN_fp16_arithmetic
    return float(float16_t(value) * float16_t(input_value) * float16_t(scale));
#else
    return value * input_value * scale;
#endif
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float gate_partial[32];
shared float up_partial[32];
#endif

void main()
{
    const uint intermediate_column = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const bool valid = intermediate_column < p.intermediate_columns && token < p.token_count;
    float gate_sum = 0.0;
    float up_sum = 0.0;
    if (valid && lane < 32)
    {
        const uint gate_row = intermediate_column * 2;
        const uint up_row = gate_row + 1;
        const uint gate_packed_row = gate_row * p.block_count * 16;
        const uint up_packed_row = up_row * p.block_count * 16;
        const uint gate_scale_row = gate_row * p.block_count;
        const uint up_scale_row = up_row * p.block_count;
        const uint input_row = token * p.input_columns;
        for (uint block = 0; block < p.block_count; ++block)
        {
            const uint packed_index = block * 16 + lane / 2;
            const uint gate_value = packed_byte(gate_packed_row + packed_index);
            const uint up_value = packed_byte(up_packed_row + packed_index);
            const uint gate_nibble = (lane & 1) == 0 ? gate_value & 15 : gate_value >> 4;
            const uint up_nibble = (lane & 1) == 0 ? up_value & 15 : up_value >> 4;
            const float input_value = input_data[input_row + block * 32 + lane];
            gate_sum += mxfp4_product(
                decode_mxfp4(gate_nibble),
                input_value,
                decode_scale(scale_byte(gate_scale_row + block)));
            up_sum += mxfp4_product(
                decode_mxfp4(up_nibble),
                input_value,
                decode_scale(scale_byte(up_scale_row + block)));
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_gate = subgroupAdd(gate_sum);
    const float reduced_up = subgroupAdd(up_sum);
    if (valid && lane == 0)
    {
        float gate = reduced_gate + bias_data[intermediate_column * 2];
        float up = reduced_up + bias_data[intermediate_column * 2 + 1];
        if (p.activation_limit > 0.0)
        {
            gate = min(gate, p.activation_limit);
            up = clamp(up, -p.activation_limit, p.activation_limit);
        }
        if (p.activation == 2)
        {
            const float silu = gate / (1.0 + exp(-gate));
            output_data[token * p.intermediate_columns + intermediate_column] = silu * up;
        }
        else if (p.activation == 1)
        {
            const float silu = gate / (1.0 + exp(-gate));
            output_data[token * p.intermediate_columns + intermediate_column] = silu * up;
        }
        else
        {
            const float silu = gate / (1.0 + exp(-1.702 * gate));
            output_data[token * p.intermediate_columns + intermediate_column] = silu * (up + 1.0);
        }
    }
#else
    gate_partial[lane] = gate_sum;
    up_partial[lane] = up_sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            gate_partial[lane] += gate_partial[lane + stride];
            up_partial[lane] += up_partial[lane + stride];
        }
        barrier();
    }
    if (valid && lane == 0)
    {
        float gate = gate_partial[0] + bias_data[intermediate_column * 2];
        float up = up_partial[0] + bias_data[intermediate_column * 2 + 1];
        if (p.activation_limit > 0.0)
        {
            gate = min(gate, p.activation_limit);
            up = clamp(up, -p.activation_limit, p.activation_limit);
        }
        if (p.activation == 2)
        {
            const float silu = gate / (1.0 + exp(-gate));
            output_data[token * p.intermediate_columns + intermediate_column] = silu * up;
        }
        else if (p.activation == 1)
        {
            const float silu = gate / (1.0 + exp(-gate));
            output_data[token * p.intermediate_columns + intermediate_column] = silu * up;
        }
        else
        {
            const float silu = gate / (1.0 + exp(-1.702 * gate));
            output_data[token * p.intermediate_columns + intermediate_column] = silu * (up + 1.0);
        }
    }
#endif
}
)glsl";

static bool create_mxfp4_gate_up_pipeline(const std::shared_ptr<NcnnVulkanContext>& context, const ncnn::Option& option,
                                           std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            mxfp4_gate_up_shader,
            static_cast<int>(sizeof(mxfp4_gate_up_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;
    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_gate_up_shader, pipeline_key);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(mxfp4_gate_up_shader, pipeline_key, destination);
    return true;
}

static constexpr char mxfp4_route_aggregation_shader[] = R"glsl(
#version 450

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer expert_output_blob
{
    float expert_output[];
};
layout(binding = 1) readonly buffer route_offset_blob
{
    uint route_offsets[];
};
layout(binding = 2) readonly buffer route_row_blob
{
    uint route_rows[];
};
layout(binding = 3) readonly buffer route_weight_blob
{
    float route_weights[];
};
layout(binding = 4) writeonly buffer aggregate_output_blob
{
    float aggregate_output[];
};

layout(push_constant) uniform parameter
{
    uint output_columns;
    uint token_count;
}
p;

void main()
{
    const uint token = gl_WorkGroupID.y;
    const uint column = gl_WorkGroupID.x * 128u + gl_LocalInvocationID.x;
    if (token >= p.token_count || column >= p.output_columns)
        return;

    const uint begin = route_offsets[token];
    const uint end = route_offsets[token + 1u];
    float sum = 0.0;
    for (uint route = begin; route < end; ++route)
    {
        const uint row = route_rows[route];
        sum += route_weights[route] * expert_output[row * p.output_columns + column];
    }
    aggregate_output[token * p.output_columns + column] = sum;
}
)glsl";

static bool create_mxfp4_route_aggregation_pipeline(const std::shared_ptr<NcnnVulkanContext>& context, const ncnn::Option& option,
                                                    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            mxfp4_route_aggregation_shader,
            static_cast<int>(sizeof(mxfp4_route_aggregation_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(mxfp4_route_aggregation_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(mxfp4_route_aggregation_shader, 0, destination);
    return true;
}

static bool route_aggregation_enabled(
    std::span<const ExpertBackendRequest> requests,
    std::span<const size_t> selected_indices,
    uint32_t output_columns,
    ActivationBuffer*& output,
    uint32_t& token_count,
    uint64_t optimization_flags)
{
    output = nullptr;
    token_count = 0;
    size_t route_count = 0;
    if (selected_indices.empty())
        return false;

    const bool enabled = runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanRouteAggregation);
    const bool force_small = runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationVulkanRouteAggregationForce);
    if (!enabled)
        return false;

    bool require_all_requests = false;
    for (size_t request_index : selected_indices)
    {
        if (request_index >= requests.size())
            return false;
        const ExpertBackendRequest& request = requests[request_index];
        const ExpertBackendRequest::RouteAggregation& aggregation = request.route_aggregation;
        require_all_requests = require_all_requests || aggregation.require_all_requests;
        if (!request.input || !aggregation.output || !aggregation.completed || aggregation.token_count == 0
            || aggregation.routes.size() != request.input->rows()
            || aggregation.output->rows() != aggregation.token_count
            || aggregation.output->columns() != output_columns)
        {
            return false;
        }
        if (output && (output != aggregation.output || token_count != aggregation.token_count))
            return false;
        output = aggregation.output;
        token_count = aggregation.token_count;
        if (route_count > std::numeric_limits<uint32_t>::max() - aggregation.routes.size())
            return false;
        route_count += aggregation.routes.size();
        for (const ExpertRoute& route : aggregation.routes)
        {
            if (route.token_index >= token_count)
                return false;
        }
    }

    if (require_all_requests && selected_indices.size() != requests.size())
        return false;

    // Aggregating only pays when the selected GPU routes contain more rows
    // than the final token output.  For decode, the metadata upload and one
    // extra aggregation dispatch should be amortized by the output transfer
    // and CPU combine work that it removes.  Keep a byte threshold instead of
    // hard-coding a model or hidden size: this enables the optimization for
    // large generic MoE outputs while leaving tiny test and latency paths on
    // the simple per-route return.
    if (force_small)
        return true;
    if (route_count <= token_count)
        return false;
    if (token_count > 1)
        return true;
    if (output_columns == 0)
        return false;
    constexpr uint64_t minimum_saved_output_bytes = 32 * 1024;
    const uint64_t saved_rows = route_count - token_count;
    if (saved_rows > std::numeric_limits<uint64_t>::max()
                         / static_cast<uint64_t>(output_columns)
                         / sizeof(float))
    {
        return true;
    }
    const uint64_t saved_output_bytes =
        saved_rows * static_cast<uint64_t>(output_columns) * sizeof(float);
    return saved_output_bytes >= minimum_saved_output_bytes;
}

static bool build_route_aggregation_metadata(
    std::span<const ExpertBackendRequest> requests,
    std::span<const size_t> selected_indices,
    std::vector<uint32_t>& offsets,
    std::vector<uint32_t>& rows,
    std::vector<float>& weights)
{
    if (selected_indices.empty())
        return false;
    const ExpertBackendRequest::RouteAggregation& first = requests[selected_indices.front()].route_aggregation;
    const uint32_t token_count = first.token_count;
    offsets.assign(static_cast<size_t>(token_count) + 1, 0);
    size_t total_rows = 0;
    for (size_t request_index : selected_indices)
    {
        const ExpertBackendRequest& request = requests[request_index];
        const auto& aggregation = request.route_aggregation;
        if (aggregation.token_count != token_count || aggregation.routes.size() != request.input->rows())
            return false;
        total_rows += aggregation.routes.size();
        for (const ExpertRoute& route : aggregation.routes)
        {
            if (route.token_index >= token_count || offsets[route.token_index + 1] == std::numeric_limits<uint32_t>::max())
                return false;
            ++offsets[route.token_index + 1];
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
    {
        if (offsets[token] > std::numeric_limits<uint32_t>::max() - offsets[token + 1])
            return false;
        offsets[token + 1] += offsets[token];
    }
    if (offsets.back() != total_rows)
        return false;
    rows.resize(total_rows);
    weights.resize(total_rows);
    std::vector<uint32_t> cursors(offsets.begin(), offsets.end() - 1);
    size_t row_offset = 0;
    for (size_t request_index : selected_indices)
    {
        const ExpertBackendRequest& request = requests[request_index];
        for (size_t batch_index = 0; batch_index < request.route_aggregation.routes.size(); ++batch_index)
        {
            const ExpertRoute& route = request.route_aggregation.routes[batch_index];
            const uint32_t destination = cursors[route.token_index]++;
            if (destination >= rows.size() || row_offset + batch_index > std::numeric_limits<uint32_t>::max())
                return false;
            rows[destination] = static_cast<uint32_t>(row_offset + batch_index);
            weights[destination] = route.weight;
        }
        row_offset += request.input->rows();
    }
    return true;
}

// This is the ncnn-side equivalent of llama.cpp's MUL_MAT_ID path.  The
// selected Expert matrices are exposed as one storage-buffer binding per
// Expert, while expert_ids selects the matrix for each routed input row.  A
// single shader dispatch therefore covers all selected Experts in the batch.
// Eight bindings keep the descriptor set small enough for older Vulkan
// implementations; larger physical batches continue through forward_batch's
// descriptor-per-Expert fallback below.
static constexpr uint32_t mxfp4_indexed_max_experts = 8;

static constexpr char mxfp4_indexed_shader[] = R"glsl(
#version 450

#ifdef NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

layout(binding = 0) readonly buffer input_blob
{
    float input_data[];
};
layout(binding = 1) readonly buffer expert_blob_0
{
    uint expert_words_0[];
};
layout(binding = 2) readonly buffer expert_blob_1
{
    uint expert_words_1[];
};
layout(binding = 3) readonly buffer expert_blob_2
{
    uint expert_words_2[];
};
layout(binding = 4) readonly buffer expert_blob_3
{
    uint expert_words_3[];
};
layout(binding = 5) readonly buffer expert_blob_4
{
    uint expert_words_4[];
};
layout(binding = 6) readonly buffer expert_blob_5
{
    uint expert_words_5[];
};
layout(binding = 7) readonly buffer expert_blob_6
{
    uint expert_words_6[];
};
layout(binding = 8) readonly buffer expert_blob_7
{
    uint expert_words_7[];
};
layout(binding = 9) readonly buffer expert_id_blob
{
    uint expert_ids[];
};
layout(binding = 10) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint gate_input_columns;
    uint gate_output_columns;
    uint down_input_columns;
    uint down_output_columns;
    uint gate_block_count;
    uint down_block_count;
    uint token_count;
    uint expert_count;
    uint gate_packed_word_offset;
    uint gate_scale_word_offset;
    uint gate_bias_word_offset;
    uint down_packed_word_offset;
    uint down_scale_word_offset;
    uint down_bias_word_offset;
    uint mode;
    uint activation;
    float activation_limit;
    uint tile_mode;
}
p;

uint expert_word(uint slot, uint index)
{
    switch (slot)
    {
    case 0u:
        return expert_words_0[index];
    case 1u:
        return expert_words_1[index];
    case 2u:
        return expert_words_2[index];
    case 3u:
        return expert_words_3[index];
    case 4u:
        return expert_words_4[index];
    case 5u:
        return expert_words_5[index];
    case 6u:
        return expert_words_6[index];
    case 7u:
        return expert_words_7[index];
    default:
        return 0u;
    }
}

uint packed_byte(uint slot, uint index)
{
    const uint word_offset = p.mode == 0u
                                 ? p.gate_packed_word_offset
                                 : p.down_packed_word_offset;
    const uint word = expert_word(slot, word_offset + (index >> 2));
    return (word >> ((index & 3u) * 8u)) & 255u;
}

uint scale_byte(uint slot, uint index)
{
    const uint word_offset = p.mode == 0u
                                 ? p.gate_scale_word_offset
                                 : p.down_scale_word_offset;
    const uint word = expert_word(slot, word_offset + (index >> 2));
    return (word >> ((index & 3u) * 8u)) & 255u;
}

float bias_value(uint slot, uint index)
{
    const uint word_offset = p.mode == 0u
                                 ? p.gate_bias_word_offset
                                 : p.down_bias_word_offset;
    return uintBitsToFloat(expert_word(slot, word_offset + index));
}

float decode_mxfp4(uint value)
{
    const float magnitudes[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    const float magnitude = magnitudes[value & 7u];
    return (value & 8u) == 0u ? magnitude : -magnitude;
}

float decode_scale(uint value)
{
    return uintBitsToFloat(value == 0u ? 0x00400000u : value << 23);
}

float mxfp4_product(float value, float input_value, float scale)
{
#ifdef NCNN_fp16_arithmetic
    return float(float16_t(value) * float16_t(input_value) * float16_t(scale));
#else
    return value * input_value * scale;
#endif
}

#if !(ncnn_subgroup_basic && ncnn_subgroup_arithmetic)
shared float partial_sum_0[32];
shared float partial_sum_1[32];
shared float partial_sum_2[32];
shared float partial_sum_3[32];
#endif

void accumulate_indexed_row(
    uint slot,
    uint row,
    uint output_column,
    bool gate_mode,
    uint matrix_input_columns,
    uint block_count,
    inout float sum_0,
    inout float sum_1)
{
    const uint weight_row = gate_mode ? output_column * 2u : output_column;
    const uint packed_row_base = weight_row * block_count * 16u;
    const uint scale_row_base = weight_row * block_count;
    const uint input_row = row * matrix_input_columns;
    for (uint block = 0u; block < block_count; ++block)
    {
        const uint packed_index = block * 16u + gl_LocalInvocationID.x / 2u;
        const uint input_index = input_row + block * 32u + gl_LocalInvocationID.x;
        const float input_value = input_data[input_index];
        const float scale_0 = decode_scale(scale_byte(slot, scale_row_base + block));
        const uint value_0 = packed_byte(slot, packed_row_base + packed_index);
        const uint nibble_0 = (gl_LocalInvocationID.x & 1u) == 0u ? value_0 & 15u : value_0 >> 4u;
        sum_0 += mxfp4_product(decode_mxfp4(nibble_0), input_value, scale_0);
        if (gate_mode)
        {
            const float scale_1 = decode_scale(scale_byte(slot, scale_row_base + block_count + block));
            const uint value_1 = packed_byte(slot, packed_row_base + block_count * 16u + packed_index);
            const uint nibble_1 = (gl_LocalInvocationID.x & 1u) == 0u ? value_1 & 15u : value_1 >> 4u;
            sum_1 += mxfp4_product(decode_mxfp4(nibble_1), input_value, scale_1);
        }
    }
}

void store_indexed_result(uint slot, uint row, uint output_column, float sum_0, float sum_1)
{
    if (p.mode == 1u)
    {
        output_data[row * p.down_output_columns + output_column] =
            sum_0 + bias_value(slot, output_column);
        return;
    }

    float gate = sum_0 + bias_value(slot, output_column * 2u);
    float up = sum_1 + bias_value(slot, output_column * 2u + 1u);
    if (p.activation_limit > 0.0)
    {
        gate = min(gate, p.activation_limit);
        up = clamp(up, -p.activation_limit, p.activation_limit);
    }
    const float silu = p.activation == 0u
                           ? gate / (1.0 + exp(-1.702 * gate))
                           : gate / (1.0 + exp(-gate));
    output_data[row * (p.gate_output_columns / 2u) + output_column] =
        p.activation == 0u ? silu * (up + 1.0) : silu * up;
}

void main()
{
    const bool gate_mode = p.mode == 0u;
    const uint output_column = gl_WorkGroupID.x;
    const uint token = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;

    // Layout 1 is the tiled MUL_MAT_ID-style path.  The selected Expert
    // rows are contiguous in input/output order, so one workgroup processes
    // two rows for one Expert while each row keeps the scalar accumulation
    // order. This is deliberately kept in the same pipeline as the scalar
    // path so devices with unusual subgroup capabilities retain the proven
    // fallback.
    if (p.tile_mode == 1u)
    {
        const uint slot = gl_WorkGroupID.z;
        const uint row_tile = gl_WorkGroupID.y;
        const uint row_begin = expert_ids[slot] + row_tile * 2u;
        const uint row_end = expert_ids[slot + 1u];
        const bool output_valid = output_column < (gate_mode ? p.gate_output_columns / 2u : p.down_output_columns);
        const bool row_0_valid = slot < p.expert_count && row_begin < row_end && row_begin < p.token_count;
        const bool row_1_valid = slot < p.expert_count && row_begin + 1u < row_end && row_begin + 1u < p.token_count;
        const uint matrix_input_columns = gate_mode ? p.gate_input_columns : p.down_input_columns;
        const uint block_count = gate_mode ? p.gate_block_count : p.down_block_count;
        float tile_sum_0 = 0.0;
        float tile_sum_1 = 0.0;
        float tile_sum_2 = 0.0;
        float tile_sum_3 = 0.0;
        if (output_valid && (row_0_valid || row_1_valid))
        {
            if (row_0_valid)
                accumulate_indexed_row(slot, row_begin, output_column, gate_mode, matrix_input_columns, block_count, tile_sum_0, tile_sum_1);
            if (row_1_valid)
                accumulate_indexed_row(slot, row_begin + 1u, output_column, gate_mode, matrix_input_columns, block_count, tile_sum_2, tile_sum_3);
        }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
        const float reduced_tile_0 = subgroupAdd(tile_sum_0);
        const float reduced_tile_1 = subgroupAdd(tile_sum_1);
        const float reduced_tile_2 = subgroupAdd(tile_sum_2);
        const float reduced_tile_3 = subgroupAdd(tile_sum_3);
        if (lane == 0u && output_valid)
        {
            if (row_0_valid)
                store_indexed_result(slot, row_begin, output_column, reduced_tile_0, reduced_tile_1);
            if (row_1_valid)
                store_indexed_result(slot, row_begin + 1u, output_column, reduced_tile_2, reduced_tile_3);
        }
#else
        partial_sum_0[lane] = tile_sum_0;
        partial_sum_1[lane] = tile_sum_1;
        partial_sum_2[lane] = tile_sum_2;
        partial_sum_3[lane] = tile_sum_3;
        barrier();
        for (uint stride = 16u; stride > 0u; stride >>= 1u)
        {
            if (lane < stride)
            {
                partial_sum_0[lane] += partial_sum_0[lane + stride];
                partial_sum_1[lane] += partial_sum_1[lane + stride];
                partial_sum_2[lane] += partial_sum_2[lane + stride];
                partial_sum_3[lane] += partial_sum_3[lane + stride];
            }
            barrier();
        }
        if (lane == 0u && output_valid)
        {
            if (row_0_valid)
                store_indexed_result(slot, row_begin, output_column, partial_sum_0[0], partial_sum_1[0]);
            if (row_1_valid)
                store_indexed_result(slot, row_begin + 1u, output_column, partial_sum_2[0], partial_sum_3[0]);
        }
#endif
        return;
    }

    const uint matrix_input_columns = gate_mode ? p.gate_input_columns : p.down_input_columns;
    const uint block_count = gate_mode ? p.gate_block_count : p.down_block_count;
    const uint slot = token < p.token_count ? expert_ids[token] : 0u;
    const bool valid = output_column < (gate_mode ? p.gate_output_columns / 2u : p.down_output_columns)
                       && token < p.token_count
                       && slot < p.expert_count;
    float sum_0 = 0.0;
    float sum_1 = 0.0;
    if (valid && lane < 32u)
    {
        const uint input_row = token * matrix_input_columns;
        const uint row_0 = gate_mode ? output_column * 2u : output_column;
        const uint row_1 = row_0 + 1u;
        const uint row_0_packed = row_0 * block_count * 16u;
        const uint row_1_packed = row_1 * block_count * 16u;
        const uint row_0_scale = row_0 * block_count;
        const uint row_1_scale = row_1 * block_count;
        for (uint block = 0u; block < block_count; ++block)
        {
            const uint packed_index = block * 16u + lane / 2u;
            const uint input_index = input_row + block * 32u + lane;
            const float input_value = input_data[input_index];
            const float scale_0 = decode_scale(scale_byte(slot, row_0_scale + block));
            const uint value_0 = packed_byte(slot, row_0_packed + packed_index);
            const uint nibble_0 = (lane & 1u) == 0u ? value_0 & 15u : value_0 >> 4u;
            sum_0 += mxfp4_product(decode_mxfp4(nibble_0), input_value, scale_0);
            if (gate_mode)
            {
                const float scale_1 = decode_scale(scale_byte(slot, row_1_scale + block));
                const uint value_1 = packed_byte(slot, row_1_packed + packed_index);
                const uint nibble_1 = (lane & 1u) == 0u ? value_1 & 15u : value_1 >> 4u;
                sum_1 += mxfp4_product(decode_mxfp4(nibble_1), input_value, scale_1);
            }
        }
    }
#if ncnn_subgroup_basic && ncnn_subgroup_arithmetic
    const float reduced_0 = subgroupAdd(sum_0);
    const float reduced_1 = subgroupAdd(sum_1);
    if (valid && lane == 0u)
    {
        if (!gate_mode)
        {
            output_data[token * p.down_output_columns + output_column] =
                reduced_0 + bias_value(slot, output_column);
        }
        else
        {
            float gate = reduced_0 + bias_value(slot, output_column * 2u);
            float up = reduced_1 + bias_value(slot, output_column * 2u + 1u);
            if (p.activation_limit > 0.0)
            {
                gate = min(gate, p.activation_limit);
                up = clamp(up, -p.activation_limit, p.activation_limit);
            }
            const float silu = p.activation == 0u
                                   ? gate / (1.0 + exp(-1.702 * gate))
                                   : gate / (1.0 + exp(-gate));
            output_data[token * (p.gate_output_columns / 2u) + output_column] =
                p.activation == 0u ? silu * (up + 1.0) : silu * up;
        }
    }
#else
    partial_sum_0[lane] = sum_0;
    partial_sum_1[lane] = sum_1;
    barrier();
    for (uint stride = 16u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            partial_sum_0[lane] += partial_sum_0[lane + stride];
            partial_sum_1[lane] += partial_sum_1[lane + stride];
        }
        barrier();
    }
    if (valid && lane == 0u)
    {
        if (!gate_mode)
        {
            output_data[token * p.down_output_columns + output_column] =
                partial_sum_0[0] + bias_value(slot, output_column);
        }
        else
        {
            float gate = partial_sum_0[0] + bias_value(slot, output_column * 2u);
            float up = partial_sum_1[0] + bias_value(slot, output_column * 2u + 1u);
            if (p.activation_limit > 0.0)
            {
                gate = min(gate, p.activation_limit);
                up = clamp(up, -p.activation_limit, p.activation_limit);
            }
            const float silu = p.activation == 0u
                                   ? gate / (1.0 + exp(-1.702 * gate))
                                   : gate / (1.0 + exp(-gate));
            output_data[token * (p.gate_output_columns / 2u) + output_column] =
                p.activation == 0u ? silu * (up + 1.0) : silu * up;
        }
    }
#endif
}
)glsl";

static bool create_mxfp4_indexed_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(
            mxfp4_indexed_shader,
            static_cast<int>(sizeof(mxfp4_indexed_shader) - 1),
            option,
            0);
    if (!spirv || spirv->empty())
        return false;

    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_indexed_shader, pipeline_key);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(context->device()));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(mxfp4_indexed_shader, pipeline_key, destination);
    return true;
}
#endif

NcnnVulkanMxfp4ExpertOperator::NcnnVulkanMxfp4ExpertOperator()
    : implementation_(new Implementation)
{
}

NcnnVulkanMxfp4ExpertOperator::~NcnnVulkanMxfp4ExpertOperator() = default;

std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> NcnnVulkanMxfp4ExpertOperator::create(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                     const TensorData& down, const TensorData* down_bias,
                                                                                     float activation_limit, uint32_t vulkan_device_index,
                                                                                     ExpertActivation activation,
                                                                                     const NcnnVulkanContextInstancePtr& context_instance,
                                                                                     uint64_t optimization_flags)
{
    return create_with_allocator(
        gate_up,
        gate_up_bias,
        down,
        down_bias,
        activation_limit,
        vulkan_device_index,
        nullptr,
        activation,
        context_instance,
        optimization_flags);
}

std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> NcnnVulkanMxfp4ExpertOperator::create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                                    const TensorData& down, const TensorData* down_bias,
                                                                                                    float activation_limit, uint32_t vulkan_device_index,
                                                                                                    ncnn::VkAllocator* weight_allocator,
                                                                                                    ExpertActivation activation,
                                                                                                    const NcnnVulkanContextInstancePtr& context_instance,
                                                                                                    uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (gate_up.shape.size() != 2 || gate_up.shape[0] % 2 != 0 || down.shape.size() != 2 || down.shape[1] != gate_up.shape[0] / 2 || activation_limit < 0.0f)
    {
        return {};
    }

    std::shared_ptr<NcnnVulkanMxfp4Operator> gate_up_projection = NcnnVulkanMxfp4Operator::create_with_allocator(
        gate_up,
        gate_up_bias,
        vulkan_device_index,
        weight_allocator,
        context_instance,
        optimization_flags);
    std::shared_ptr<NcnnVulkanMxfp4Operator> down_projection = NcnnVulkanMxfp4Operator::create_with_allocator(
        down,
        down_bias,
        vulkan_device_index,
        weight_allocator,
        context_instance,
        optimization_flags);
    if (!gate_up_projection || !down_projection)
        return {};
    if (gate_up_projection->implementation_->vulkan_context != down_projection->implementation_->vulkan_context)
    {
        return {};
    }

    std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> result(new NcnnVulkanMxfp4ExpertOperator);
    Implementation& implementation = *result->implementation_;
    implementation.gate_up = std::move(gate_up_projection);
    implementation.down = std::move(down_projection);
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    implementation.optimization_flags = optimization_flags;
    NcnnVulkanMxfp4Operator::Implementation& gate_implementation = *implementation.gate_up->implementation_;
    const std::lock_guard<std::mutex> lock(gate_implementation.vulkan_context->command_mutex());
    if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline))
    {
        return {};
    }
    return result;
#else
    (void)gate_up;
    (void)gate_up_bias;
    (void)down;
    (void)down_bias;
    (void)activation_limit;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)activation;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> NcnnVulkanMxfp4ExpertOperator::create_from_device_storage(
    const NcnnVulkanMxfp4DeviceMatrixView& gate_up, const TensorData* gate_up_bias, const NcnnVulkanMxfp4DeviceMatrixView& down,
    const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage,
    ExpertActivation activation,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN && NCNN_BATCH
    if (storage.empty() || gate_up.output_columns == 0 || gate_up.output_columns % 2 != 0 || gate_up.input_columns == 0 || down.output_columns == 0
        || down.input_columns != gate_up.output_columns / 2 || activation_limit < 0.0f)
    {
        return {};
    }
    const auto create_projection = [&](const NcnnVulkanMxfp4DeviceMatrixView& matrix, const TensorData* bias) -> std::shared_ptr<NcnnVulkanMxfp4Operator> {
        if (matrix.output_columns == 0 || matrix.input_columns == 0 || matrix.input_columns % 32 != 0)
        {
            return {};
        }
        const uint32_t output_columns = matrix.output_columns;
        const uint32_t input_columns = matrix.input_columns;
        const uint32_t block_count = input_columns / 32;
        const uint64_t expected_blocks = static_cast<uint64_t>(output_columns) * block_count * 16;
        const uint64_t expected_scales = static_cast<uint64_t>(output_columns) * block_count;
        if (matrix.packed_bytes != expected_blocks || matrix.scales_bytes != expected_scales || expected_blocks > std::numeric_limits<int>::max()
            || expected_scales > std::numeric_limits<int>::max() || matrix.packed_offset > storage.buffer_capacity()
            || matrix.scales_offset > storage.buffer_capacity() || expected_blocks > storage.buffer_capacity() - matrix.packed_offset
            || expected_scales > storage.buffer_capacity() - matrix.scales_offset)
        {
            return {};
        }
        if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
        {
            return {};
        }

        std::shared_ptr<NcnnVulkanMxfp4Operator> projection(new NcnnVulkanMxfp4Operator);
        NcnnVulkanMxfp4Operator::Implementation& implementation = *projection->implementation_;
        implementation.input_columns = input_columns;
        implementation.output_columns = output_columns;
        implementation.block_count = block_count;
        implementation.vulkan_context = NcnnVulkanContext::acquire(
            vulkan_device_index,
            context_instance,
            optimization_flags);
        if (!implementation.vulkan_context)
            return {};
        ncnn::VulkanDevice* device = implementation.vulkan_context->device();
        const size_t alignment = std::max<size_t>(4, device->info.buffer_offset_alignment());
        if ((storage.buffer_offset() + matrix.packed_offset) % alignment != 0 || (storage.buffer_offset() + matrix.scales_offset) % alignment != 0)
        {
            return {};
        }
        implementation.option.use_vulkan_compute = true;
        implementation.option.use_fp16_packed = false;
        implementation.option.use_fp16_storage = false;
        implementation.option.use_fp16_arithmetic = false;
        implementation.option.use_bf16_packed = false;
        implementation.option.use_bf16_storage = false;
        implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
        implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
        implementation.option.use_subgroup_ops = device->info.support_subgroup_ops();

        const auto view = [&storage](size_t offset, uint64_t bytes) {
            ncnn::VkMat result = storage;
            result.dims = 1;
            result.w = static_cast<int>(align_to_uint32(static_cast<size_t>(bytes)));
            result.h = 1;
            result.d = 1;
            result.c = 1;
            result.elemsize = sizeof(uint8_t);
            result.elempack = 1;
            result.cstep = result.w;
            result.offset += offset;
            result.n = 1;
            result.nstep = result.w;
            return result;
        };
        implementation.packed = view(matrix.packed_offset, expected_blocks);
        implementation.scales = view(matrix.scales_offset, expected_scales);

        ncnn::Mat biases(static_cast<int>(output_columns), sizeof(float));
        if (biases.empty())
            return {};
        float* bias_values = static_cast<float*>(biases.data);
        if (!bias)
        {
            std::fill_n(bias_values, output_columns, 0.0f);
        }
        else if (bias->dtype == DType::Float32)
        {
            const auto values = bias->float32_values();
            if (values.size() != output_columns)
            {
                return {};
            }
            std::copy(values.begin(), values.end(), bias_values);
        }
        else
        {
            const auto values = bias->bfloat16_values();
            if (values.size() != output_columns)
            {
                return {};
            }
            for (uint32_t index = 0; index < output_columns; ++index)
            {
                bias_values[index] = bfloat16_to_float(values[index]);
            }
        }
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
        ncnn::Option upload_option = implementation.option;
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        {
            const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
            if (!create_mxfp4_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline))
            {
                return {};
            }
            ncnn::VkTransfer command(device);
            command.record_upload(biases, implementation.bias, upload_option);
            if (implementation.bias.empty() || command.submit_and_wait() != 0)
            {
                return {};
            }
        }
        implementation.weight_staging_allocator.reset();
        return projection;
    };

    auto gate_projection = create_projection(gate_up, gate_up_bias);
    auto down_projection = create_projection(down, down_bias);
    if (!gate_projection || !down_projection || gate_projection->implementation_->vulkan_context != down_projection->implementation_->vulkan_context)
    {
        return {};
    }
    std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> result(new NcnnVulkanMxfp4ExpertOperator);
    Implementation& implementation = *result->implementation_;
    implementation.gate_up = std::move(gate_projection);
    implementation.down = std::move(down_projection);
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    implementation.optimization_flags = optimization_flags;
    auto& gate_implementation = *implementation.gate_up->implementation_;
    {
        const std::lock_guard<std::mutex> lock(gate_implementation.vulkan_context->command_mutex());
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline))
        {
            return {};
        }
    }
    return result;
#else
    (void)gate_up;
    (void)gate_up_bias;
    (void)down;
    (void)down_bias;
    (void)activation_limit;
    (void)vulkan_device_index;
    (void)storage;
    (void)activation;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

bool NcnnVulkanMxfp4ExpertOperator::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.gate_up || !implementation.down || !implementation.gate_up_pipeline)
    {
        return false;
    }
    const NcnnVulkanMxfp4Operator::Implementation& gate = *implementation.gate_up->implementation_;
    const NcnnVulkanMxfp4Operator::Implementation& down = *implementation.down->implementation_;
    if (!gate.vulkan_context || gate.vulkan_context != down.vulkan_context || input.rows() == 0 || input.columns() != gate.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    const uint32_t intermediate_columns = gate.output_columns / 2;
    if (down.input_columns != intermediate_columns)
    {
        return false;
    }
    NcnnVulkanRuntimeState& runtime_state =
        gate.vulkan_context->runtime_state();

    NcnnVulkanTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);
    std::unique_lock<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
    {
        return false;
    }
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(intermediate_columns), static_cast<int>(input.rows()), sizeof(float), gate.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> gate_bindings(5);
    gate_bindings[0] = input_gpu;
    gate_bindings[1] = gate.packed;
    gate_bindings[2] = gate.scales;
    gate_bindings[3] = gate.bias;
    gate_bindings[4] = intermediate_gpu;
    std::vector<ncnn::vk_constant_type> gate_constants(6);
    gate_constants[0].u32 = gate.input_columns;
    gate_constants[1].u32 = intermediate_columns;
    gate_constants[2].u32 = gate.block_count;
    gate_constants[3].u32 = static_cast<uint32_t>(input.rows());
    gate_constants[4].u32 = implementation.activation == ExpertActivation::Silu
                                ? 2
                                : implementation.activation == ExpertActivation::DeepSeekSwiGlu ? 1 : 0;
    gate_constants[5].f = implementation.activation_limit;
    ncnn::VkMat gate_dispatcher;
    gate_dispatcher.w = static_cast<int>(intermediate_columns * 32);
    gate_dispatcher.h = static_cast<int>(input.rows());
    gate_dispatcher.c = 1;
    command.record_pipeline(implementation.gate_up_pipeline.get(), gate_bindings, gate_constants, gate_dispatcher);

    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(down.output_columns), static_cast<int>(input.rows()), sizeof(float), gate.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;
    std::vector<ncnn::VkMat> down_bindings(5);
    down_bindings[0] = intermediate_gpu;
    down_bindings[1] = down.packed;
    down_bindings[2] = down.scales;
    down_bindings[3] = down.bias;
    down_bindings[4] = output_gpu;
    std::vector<ncnn::vk_constant_type> down_constants(4);
    down_constants[0].u32 = down.input_columns;
    down_constants[1].u32 = down.output_columns;
    down_constants[2].u32 = down.block_count;
    down_constants[3].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat down_dispatcher;
    down_dispatcher.w = static_cast<int>(down.output_columns * 32);
    down_dispatcher.h = static_cast<int>(input.rows());
    down_dispatcher.c = 1;
    command.record_pipeline(down.pipeline.get(), down_bindings, down_constants, down_dispatcher);

    if (!record_prepared_activation_staging_download(
            output_gpu,
            input.rows(),
            down.output_columns,
            transfer_slot.download,
            command,
            gate.vulkan_context->device(),
            down.option,
            output_dtype)
        || submit_compute_and_wait(command, runtime_state) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
class VulkanMxfp4ExpertBackend final : public IExpertExecutionBackend
{
public:
    VulkanMxfp4ExpertBackend(
        uint64_t capacity_bytes,
        uint32_t vulkan_device_index,
        std::shared_ptr<VulkanExpertVictimCache> device_weight_source,
        NcnnVulkanContextInstancePtr context_instance,
        uint64_t optimization_flags)
        : capacity_bytes_(capacity_bytes),
          vulkan_device_index_(vulkan_device_index),
          context_instance_(std::move(context_instance)),
          optimization_flags_(optimization_flags),
          // The admission queue is part of the per-instance GPU cache
          // pipeline. A fixed 256 MiB queue made the queue fill long before
          // the configured device capacity, so cold layers silently fell
          // back to CPU while the GPU cache was still mostly empty.
          maximum_pending_bytes_(capacity_bytes),
          device_weight_source_(std::move(device_weight_source)),
          worker_(&VulkanMxfp4ExpertBackend::worker_loop, this),
          execution_worker_(&VulkanMxfp4ExpertBackend::execution_loop, this)
    {
        vulkan_context_ = NcnnVulkanContext::acquire(
            vulkan_device_index_,
            context_instance_,
            optimization_flags_);
        if (vulkan_context_ && capacity_bytes_ != 0)
        {
            const uint64_t allocator_block_bytes = std::min<uint64_t>(capacity_bytes_, UINT64_C(64) * 1024 * 1024);
            expert_weight_allocator_.reset(new ncnn::VkBlobAllocator(vulkan_context_->device(), static_cast<size_t>(allocator_block_bytes)));
        }
    }

    ~VulkanMxfp4ExpertBackend() override
    {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            dropped_admissions_ += pending_.size();
            pending_.clear();
            pending_keys_.clear();
            pending_bytes_ = 0;
        }
        work_available_.notify_all();
        execution_available_.notify_all();
        if (worker_.joinable())
            worker_.join();
        if (execution_worker_.joinable())
            execution_worker_.join();
        retired_entries_.clear();
        entries_.clear();
    }

    void set_foreground_active(bool active) noexcept override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (active)
        {
            if (foreground_depth_++ == 0)
            {
                admission_idle_.wait(lock, [this] {
                    return active_admissions_ == 0;
                });
            }
        }
        else if (foreground_depth_ != 0)
        {
            if (--foreground_depth_ == 0)
                work_available_.notify_all();
        }
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down,
               const TensorData* down_bias, uint32_t residency_group, uint32_t token_count, float activation_limit,
               ExpertActivation activation) override
    {
        if (key.empty() || !gate_up || !down || gate_up->dtype != DType::MxFp4 || down->dtype != DType::MxFp4 || gate_up->shape.size() != 2
            || down->shape.size() != 2 || gate_up->shape[0] % 2 != 0 || down->shape[1] != gate_up->shape[0] / 2 || activation_limit < 0.0f)
        {
            return;
        }
        const uint64_t bytes = mxfp4_bytes(*gate_up) + mxfp4_bytes(*down) + tensor_bytes(gate_up_bias) + tensor_bytes(down_bias);
        if (bytes == 0 || bytes > capacity_bytes_ || bytes > maximum_pending_bytes_)
        {
            return;
        }

        PendingAdmission admission;
        admission.key = std::move(key);
        admission.gate_up = std::move(gate_up);
        admission.down = std::move(down);
        if (gate_up_bias)
        {
            admission.gate_up_bias = std::make_shared<TensorData>(*gate_up_bias);
        }
        if (down_bias)
        {
            admission.down_bias = std::make_shared<TensorData>(*down_bias);
        }
        admission.activation_limit = activation_limit;
        admission.activation = activation;
        admission.residency_group = residency_group;
        admission.bytes = bytes;
        (void)token_count;

        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || entries_.find(admission.key) != entries_.end() || pending_keys_.find(admission.key) != pending_keys_.end())
        {
            return;
        }
        while (!pending_.empty() && pending_bytes_ > maximum_pending_bytes_ - bytes)
        {
            pending_bytes_ -= pending_.front().bytes;
            pending_keys_.erase(pending_.front().key);
            pending_.pop_front();
            ++dropped_admissions_;
        }
        if (pending_bytes_ > maximum_pending_bytes_ - bytes)
        {
            ++dropped_admissions_;
            return;
        }
        pending_bytes_ += bytes;
        pending_keys_.insert(admission.key);
        pending_.push_back(std::move(admission));
        ++admissions_;
        work_available_.notify_one();
    }

    ExpertBackendExecutionResult try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output) override
    {
        const ExpertBackendRequest request{key, &input, &output};
        std::vector<ExpertBackendExecutionResult> results = try_execute_batch(std::span<const ExpertBackendRequest>(&request, 1));
        return results.empty() ? ExpertBackendExecutionResult ::Failed : results.front();
    }

    std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests) override
    {
        auto submission = submit_batch(requests);
        if (!submission)
            return std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult ::Failed);
        const std::span<const ExpertBackendExecutionResult> planned = submission->reservations();
        std::vector<ExpertBackendExecutionResult> results = submission->wait();
        if (planned.size() != requests.size() || results.size() != requests.size())
        {
            submission->abort();
            return std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult ::Failed);
        }
        for (size_t index = 0; index < results.size(); ++index)
        {
            if (results[index] == ExpertBackendExecutionResult::Executed
                && planned[index] != ExpertBackendExecutionResult::Executed)
            {
                submission->abort();
                return std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult ::Failed);
            }
        }
        bool has_executed = false;
        for (ExpertBackendExecutionResult result : results)
            has_executed = has_executed || result == ExpertBackendExecutionResult::Executed;
        if (has_executed)
        {
            if (!submission->commit())
            {
                for (ExpertBackendExecutionResult& result : results)
                {
                    if (result == ExpertBackendExecutionResult::Executed)
                        result = ExpertBackendExecutionResult::Failed;
                }
                submission->abort();
            }
        }
        else
            submission->abort();
        return results;
    }

    std::unique_ptr<IExpertBackendBatchSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        auto work = std::make_shared<WorkItem>();
        work->client_requests.assign(requests.begin(), requests.end());
        work->requests = work->client_requests;
        work->private_outputs.resize(requests.size());
        work->private_route_completed.assign(requests.size(), 0);
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            ExpertBackendRequest& private_request = work->requests[request_index];
            private_request.output = &work->private_outputs[request_index];
            if (private_request.route_aggregation.output)
            {
                const ActivationBuffer* client_aggregation = work->client_requests[request_index].route_aggregation.output;
                if (!client_aggregation)
                    continue;
                if (work->private_aggregation.rows() == 0)
                {
                    work->private_aggregation.reset(
                        client_aggregation->rows(),
                        client_aggregation->columns(),
                        true);
                }
                private_request.route_aggregation.output = &work->private_aggregation;
                private_request.route_aggregation.completed = &work->private_route_completed[request_index];
            }
        }
        work->planned.assign(requests.size(), ExpertBackendExecutionResult ::NotResident);
        work->selected.reserve(requests.size());
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Admission is asynchronous and bounded; resident selection
            // below applies the fixed route-row policy for device execution.
            std::vector<Selection>& candidates = work->selected;
            for (size_t request_index = 0; request_index < requests.size(); ++request_index)
            {
                const ExpertBackendRequest& request = requests[request_index];
                if (!request.input || !request.output || request.input->rows() == 0)
                {
                    work->planned[request_index] = ExpertBackendExecutionResult ::Failed;
                    continue;
                }
                auto existing = entries_.find(request.key);
                if (existing == entries_.end())
                {
                    const bool source_allowed = request.weight_bytes == 0
                                                || request.input->rows() >= vulkan_expert_gpu_admission_min_rows;
                    std::optional<VulkanExpertVictimCache::DeviceOperationLease> device_lease;
                    const std::shared_ptr<VulkanExpertVictimCache> device_weight_source = device_weight_source_;
                    if (device_weight_source && source_allowed)
                    {
                        // Vulkan allocation must stay outside the scheduler lock.
                        lock.unlock();
                        device_lease = device_weight_source->find_device_operation(request.key);
                        lock.lock();
                        existing = entries_.find(request.key);
                    }
                    if (existing != entries_.end())
                    {
                        // Prefer an admission completed during victim lookup.
                        device_lease.reset();
                    }
                    else if (!device_lease)
                    {
                        ++misses_;
                        if (device_weight_source && source_allowed)
                        {
                            ++device_source_misses_;
                        }
                        continue;
                    }
                    else
                    {
                        std::shared_ptr<Entry> entry = std::make_shared<Entry>();
                        entry->key.assign(request.key);
                        entry->operation = std::move(device_lease->operation);
                        entry->device_source_pin = std::move(device_lease->pin);
                        entry->bytes = request.weight_bytes;
                        entry->device_source = true;
                        ++hits_;
                        ++device_source_hits_;
                        candidates.push_back({
                            request_index,
                            std::move(entry),
                        });
                        continue;
                    }
                }
                std::shared_ptr<Entry> entry = existing->second;
                touch_locked(*entry, true);
                ++hits_;
                candidates.push_back({
                    request_index,
                    std::move(entry),
                });
            }

            size_t selected_count = 0;
            bool direct_resident_execution = foreground_depth_ == 0 && !candidates.empty();
            if (direct_resident_execution)
            {
                for (const ExpertBackendRequest& request : requests)
                {
                    if (request.weight_bytes != 0)
                    {
                        direct_resident_execution = false;
                        break;
                    }
                }
            }
            if (direct_resident_execution)
            {
                // The weight-less backend API is an explicit resident
                // operation used by backend clients and validation. It must
                // execute every explicitly requested resident Expert.
                selected_count = candidates.size();
            }
            else if (runtime_optimization_enabled(
                         optimization_flags_,
                         RuntimeOptimizationVulkanExpertGpuPriority))
            {
                // Deterministic policy: only resident Experts with enough
                // routed rows amortize the host/device boundary.  The
                // threshold is fixed, so no CPU/GPU probe or timing sample
                // is needed to decide whether the request uses Vulkan.
                const auto gpu_end = std::stable_partition(
                    candidates.begin(),
                    candidates.end(),
                    [&requests](const Selection& selection) {
                        const ExpertBackendRequest& request =
                            requests[selection.request_index];
                        return request.input
                               && request.input->rows()
                                      >= vulkan_expert_gpu_min_rows;
                    });
                selected_count = static_cast<size_t>(
                    std::distance(candidates.begin(), gpu_end));
            }

            for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index)
            {
                const Selection& selection = candidates[candidate_index];
                if (candidate_index < selected_count)
                {
                    work->planned[selection.request_index] = ExpertBackendExecutionResult ::Executed;
                }
                else
                {
                    ++cpu_preferred_;
                    work->planned[selection.request_index] = ExpertBackendExecutionResult ::PreferCpu;
                }
            }
            work->selected.resize(selected_count);
            if (work->selected.empty())
            {
                work->final = work->planned;
                work->done = true;
            }
            else
            {
                execution_pending_.push_back(work);
                execution_available_.notify_one();
            }
        }
        return std::make_unique<Submission>(std::move(work));
    }

    void observe_cpu(uint32_t token_count, uint64_t weight_bytes, uint64_t elapsed_microseconds) override
    {
        // GPU/CPU placement is deterministic; runtime timing never changes it.
        (void)token_count;
        (void)weight_bytes;
        (void)elapsed_microseconds;
    }

    void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) override
    {
        // Keep the interface for backend composition, but do not spend work
        // collecting samples or changing the static placement policy.
        (void)token_count;
        (void)total_weight_bytes;
        (void)accelerated_weight_bytes;
        (void)elapsed_microseconds;
    }

    void wait_for_background_work() override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        admission_idle_.wait(lock, [this] { return pending_.empty() && active_admissions_ == 0; });
    }

    ExpertBackendStatistics statistics() const override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ExpertBackendStatistics result;
        result.hits = hits_;
        result.misses = misses_;
        result.admissions = admissions_;
        result.stores = stores_;
        result.evictions = evictions_;
        result.dropped_admissions = dropped_admissions_;
        result.executions = executions_;
        result.execution_failures = execution_failures_;
        result.cpu_preferred = cpu_preferred_;
        result.bytes_uploaded = bytes_uploaded_;
        result.resident_bytes = resident_bytes_;
        result.pending_bytes = pending_bytes_;
        result.execution_time_microseconds = execution_time_microseconds_;
        result.arc_recent_bytes = recent_bytes_;
        result.arc_frequent_bytes = frequent_bytes_;
        result.arc_recent_target_bytes = recent_target_bytes_;
        result.arc_recent_ghost_bytes = recent_ghost_bytes_;
        result.arc_frequent_ghost_bytes = frequent_ghost_bytes_;
        result.device_source_hits = device_source_hits_;
        result.device_source_misses = device_source_misses_;
        result.device_source_executions = device_source_executions_;
        result.device_source_execution_failures = device_source_execution_failures_;
        result.route_aggregation_batches = route_aggregation_batches_;
        result.route_aggregation_routes = route_aggregation_routes_;
        result.route_aggregation_bytes_saved = route_aggregation_bytes_saved_;
        return result;
    }

    std::vector<ExpertBackendDeviceStatistics> device_statistics() const override
    {
        return {{
            vulkan_device_index_,
            capacity_bytes_,
            statistics(),
        }};
    }

    uint64_t capacity_bytes() const noexcept override
    {
        return capacity_bytes_;
    }

private:
    enum class ArcList
    {
        Recent,
        Frequent
    };

    struct Entry
    {
        std::string key;
        std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
        std::shared_ptr<const void> device_source_pin;
        uint64_t bytes = 0;
        uint32_t residency_group = 0;
        bool device_source = false;
        ArcList list = ArcList::Recent;
        std::list<std::string>::iterator position;
    };

    struct Selection
    {
        size_t request_index = 0;
        std::shared_ptr<Entry> entry;
    };

    struct WorkItem
    {
        std::vector<ExpertBackendRequest> client_requests;
        std::vector<ExpertBackendRequest> requests;
        std::vector<ActivationBuffer> private_outputs;
        ActivationBuffer private_aggregation;
        std::vector<uint8_t> private_route_completed;
        std::vector<Selection> selected;
        std::vector<ExpertBackendExecutionResult> planned;
        std::vector<ExpertBackendExecutionResult> final;
        std::mutex mutex;
        std::condition_variable completed;
        bool done = false;
    };

    class Submission final : public IExpertBackendBatchSubmission
    {
    public:
        explicit Submission(std::shared_ptr<WorkItem> work)
            : work_(std::move(work))
        {
        }

        ~Submission() override
        {
            if (work_ && !waited_)
                (void)wait();
            if (work_ && !committed_ && !aborted_)
                abort();
        }

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override
        {
            return work_->planned;
        }

        std::vector<ExpertBackendExecutionResult> wait() override
        {
            std::unique_lock<std::mutex> lock(work_->mutex);
            work_->completed.wait(lock, [this] { return work_->done; });
            waited_ = true;
            return work_->final;
        }

        bool commit() override
        {
            if (!waited_)
                (void)wait();
            if (committed_ || aborted_)
                return committed_;
            // Validate the complete publication set before touching any
            // caller-owned buffer. A failed reservation must be a safe CPU
            // fallback, never a partially published batch.
            ActivationBuffer* route_output = nullptr;
            for (size_t index = 0; index < work_->final.size(); ++index)
            {
                if (work_->final[index] != ExpertBackendExecutionResult::Executed)
                    continue;
                ExpertBackendRequest& client = work_->client_requests[index];
                if (!client.output)
                {
                    aborted_ = true;
                    return false;
                }
                if (client.route_aggregation.output)
                {
                    if (work_->private_route_completed[index] == 0)
                    {
                        aborted_ = true;
                        return false;
                    }
                    if (route_output && route_output != client.route_aggregation.output)
                    {
                        aborted_ = true;
                        return false;
                    }
                    route_output = client.route_aggregation.output;
                }
            }
            bool route_published = false;
            for (size_t index = 0; index < work_->final.size(); ++index)
            {
                if (work_->final[index] != ExpertBackendExecutionResult::Executed)
                    continue;
                ExpertBackendRequest& client = work_->client_requests[index];
                client.output->swap(work_->private_outputs[index]);
                if (client.route_aggregation.output)
                {
                    if (!route_published)
                    {
                        client.route_aggregation.output->swap(work_->private_aggregation);
                        route_published = true;
                    }
                    if (client.route_aggregation.completed)
                        *client.route_aggregation.completed = 1;
                }
            }
            committed_ = true;
            return true;
        }

        void abort() noexcept override
        {
            if (committed_ || aborted_)
                return;
            aborted_ = true;
        }

    private:
        std::shared_ptr<WorkItem> work_;
        bool waited_ = false;
        bool committed_ = false;
        bool aborted_ = false;
    };

    struct PendingAdmission
    {
        std::string key;
        std::shared_ptr<const TensorData> gate_up;
        std::shared_ptr<const TensorData> down;
        std::shared_ptr<const TensorData> gate_up_bias;
        std::shared_ptr<const TensorData> down_bias;
        float activation_limit = 0.0f;
        ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
        uint32_t residency_group = 0;
        uint64_t bytes = 0;
    };

    struct Ghost
    {
        std::string key;
        uint64_t bytes = 0;
    };
    using GhostList = std::list<Ghost>;
    using GhostIndex = std::unordered_map<std::string, GhostList::iterator, TransparentStringHash, std::equal_to<>>;

    static uint64_t mxfp4_bytes(const TensorData& tensor)
    {
        return tensor.mxfp4_blocks.size() + tensor.mxfp4_scales.size();
    }

    static uint64_t tensor_bytes(const TensorData* tensor)
    {
        if (!tensor)
            return 0;
        if (tensor->dtype == DType::Float32)
            return tensor->float32_values().size() * sizeof(float);
        if (tensor->dtype == DType::BFloat16)
            return tensor->bfloat16_values().size() * sizeof(uint16_t);
        return 0;
    }

    void touch_locked(Entry& entry, bool repeated)
    {
        if (entry.list == ArcList::Recent && repeated)
        {
            recent_.erase(entry.position);
            recent_bytes_ -= entry.bytes;
            frequent_.push_back(entry.key);
            entry.position = std::prev(frequent_.end());
            entry.list = ArcList::Frequent;
            frequent_bytes_ += entry.bytes;
            return;
        }
        std::list<std::string>& list = entry.list == ArcList::Recent ? recent_ : frequent_;
        list.splice(list.end(), list, entry.position);
        entry.position = std::prev(list.end());
    }

    static void erase_ghost_entry(GhostIndex& index, GhostList& list, uint64_t& bytes, const std::string& key)
    {
        const auto existing = index.find(key);
        if (existing == index.end())
            return;
        bytes -= existing->second->bytes;
        list.erase(existing->second);
        index.erase(existing);
    }

    void erase_ghost_locked(const std::string& key)
    {
        erase_ghost_entry(recent_ghost_index_, recent_ghost_, recent_ghost_bytes_, key);
        erase_ghost_entry(frequent_ghost_index_, frequent_ghost_, frequent_ghost_bytes_, key);
    }

    void add_ghost_locked(const Entry& entry)
    {
        erase_ghost_locked(entry.key);
        Ghost ghost{entry.key, entry.bytes};
        if (entry.list == ArcList::Recent)
        {
            recent_ghost_.push_back(std::move(ghost));
            const auto position = std::prev(recent_ghost_.end());
            recent_ghost_index_[position->key] = position;
            recent_ghost_bytes_ += entry.bytes;
        }
        else
        {
            frequent_ghost_.push_back(std::move(ghost));
            const auto position = std::prev(frequent_ghost_.end());
            frequent_ghost_index_[position->key] = position;
            frequent_ghost_bytes_ += entry.bytes;
        }
        trim_ghosts_locked();
    }

    static void trim_ghost_front(GhostIndex& index, GhostList& list, uint64_t& bytes)
    {
        if (list.empty())
            return;
        bytes -= list.front().bytes;
        index.erase(list.front().key);
        list.pop_front();
    }

    void trim_ghosts_locked()
    {
        while (recent_ghost_bytes_ + frequent_ghost_bytes_ > capacity_bytes_)
        {
            if (recent_ghost_bytes_ >= frequent_ghost_bytes_ && !recent_ghost_.empty())
            {
                trim_ghost_front(recent_ghost_index_, recent_ghost_, recent_ghost_bytes_);
            }
            else
            {
                trim_ghost_front(frequent_ghost_index_, frequent_ghost_, frequent_ghost_bytes_);
            }
        }
    }

    uint64_t arc_delta(uint64_t required, uint64_t numerator, uint64_t denominator) const
    {
        if (denominator == 0 || numerator <= denominator)
            return required;
        const uint64_t ratio = numerator / denominator;
        if (required != 0 && ratio > capacity_bytes_ / required)
            return capacity_bytes_;
        return std::min(capacity_bytes_, std::max(required, required * ratio));
    }

    bool consume_ghost_locked(const std::string& key, uint64_t required, bool& frequent, bool& from_frequent)
    {
        frequent = false;
        from_frequent = false;
        const auto recent = recent_ghost_index_.find(key);
        if (recent != recent_ghost_index_.end())
        {
            const uint64_t adjustment = arc_delta(required, frequent_ghost_bytes_, recent_ghost_bytes_);
            recent_ghost_bytes_ -= recent->second->bytes;
            recent_ghost_.erase(recent->second);
            recent_ghost_index_.erase(recent);
            recent_target_bytes_ = std::min(capacity_bytes_, recent_target_bytes_ + std::min(adjustment, capacity_bytes_ - recent_target_bytes_));
            frequent = true;
            return true;
        }
        const auto frequent_entry = frequent_ghost_index_.find(key);
        if (frequent_entry == frequent_ghost_index_.end())
        {
            return false;
        }
        const uint64_t adjustment = arc_delta(required, recent_ghost_bytes_, frequent_ghost_bytes_);
        frequent_ghost_bytes_ -= frequent_entry->second->bytes;
        frequent_ghost_.erase(frequent_entry->second);
        frequent_ghost_index_.erase(frequent_entry);
        recent_target_bytes_ -= std::min(adjustment, recent_target_bytes_);
        frequent = true;
        from_frequent = true;
        return true;
    }

    std::shared_ptr<Entry> find_victim_locked(const std::list<std::string>& list, uint32_t residency_group)
    {
        for (const std::string& key : list)
        {
            const auto existing = entries_.find(key);
            if (existing == entries_.end() || existing->second.use_count() != 1
                || (residency_group != std::numeric_limits<uint32_t>::max() && existing->second->residency_group != residency_group))
            {
                continue;
            }
            return existing->second;
        }
        return {};
    }

    bool evict_one_locked(bool incoming_from_frequent, uint32_t incoming_group, uint64_t required)
    {
        const bool prefer_recent = recent_bytes_ > recent_target_bytes_ || (incoming_from_frequent && recent_bytes_ == recent_target_bytes_);
        uint32_t preferred_group = std::numeric_limits<uint32_t>::max();
        if (!residency_group_bytes_.empty())
        {
            const uint64_t fair_share = capacity_bytes_ / residency_group_bytes_.size();
            if (incoming_group < residency_group_bytes_.size() && residency_group_bytes_[incoming_group] + required > fair_share)
            {
                preferred_group = incoming_group;
            }
            else
            {
                uint64_t maximum_excess = 0;
                for (uint32_t group = 0; group < residency_group_bytes_.size(); ++group)
                {
                    const uint64_t bytes = residency_group_bytes_[group];
                    const uint64_t excess = bytes > fair_share ? bytes - fair_share : 0;
                    if (excess > maximum_excess)
                    {
                        maximum_excess = excess;
                        preferred_group = group;
                    }
                }
            }
        }
        std::shared_ptr<Entry> victim = prefer_recent ? find_victim_locked(recent_, preferred_group) : find_victim_locked(frequent_, preferred_group);
        if (!victim)
        {
            victim = prefer_recent ? find_victim_locked(frequent_, preferred_group) : find_victim_locked(recent_, preferred_group);
        }
        if (!victim && preferred_group != std::numeric_limits<uint32_t>::max())
        {
            victim = prefer_recent ? find_victim_locked(recent_, std::numeric_limits<uint32_t>::max())
                                   : find_victim_locked(frequent_, std::numeric_limits<uint32_t>::max());
            if (!victim)
            {
                victim = prefer_recent ? find_victim_locked(frequent_, std::numeric_limits<uint32_t>::max())
                                       : find_victim_locked(recent_, std::numeric_limits<uint32_t>::max());
            }
        }
        if (!victim)
            return false;
        add_ghost_locked(*victim);
        if (victim->list == ArcList::Recent)
        {
            recent_.erase(victim->position);
            recent_bytes_ -= victim->bytes;
        }
        else
        {
            frequent_.erase(victim->position);
            frequent_bytes_ -= victim->bytes;
        }
        resident_bytes_ -= victim->bytes;
        if (victim->residency_group < residency_group_bytes_.size())
        {
            residency_group_bytes_[victim->residency_group] -= victim->bytes;
        }
        entries_.erase(victim->key);
        retired_entries_.push_back(std::move(victim));
        ++evictions_;
        return true;
    }

    static ncnn::VkMat row_view(const ncnn::VkMat& source, size_t first_row, size_t rows)
    {
        if (source.empty() || source.dims != 2 || first_row + rows > static_cast<size_t>(source.h))
        {
            return {};
        }
        ncnn::VkMat view = source;
        view.h = static_cast<int>(rows);
        view.offset += first_row * static_cast<size_t>(source.w) * source.elemsize;
        return view;
    }

    enum class IndexedBatchResult
    {
        NotSupported,
        Executed,
        Failed
    };

    IndexedBatchResult forward_indexed_batch(
        std::span<const ExpertBackendRequest> requests,
        std::span<const Selection> selected)
    {
        // The indexed shader pays for a runtime Expert-buffer selector.  Keep
        // the tuned per-Expert kernel until a batch has enough distinct
        // Experts to amortize that cost.
        if (selected.size() < 4 || selected.size() > mxfp4_indexed_max_experts)
            return IndexedBatchResult::NotSupported;

        const NcnnVulkanMxfp4ExpertOperator::Implementation& first_expert =
            *selected.front().entry->operation->implementation_;
        if (!first_expert.gate_up || !first_expert.down || !first_expert.gate_up->implementation_->indexed_storage
            || !first_expert.down->implementation_->indexed_storage)
        {
            return IndexedBatchResult::NotSupported;
        }
        const NcnnVulkanMxfp4Operator::Implementation& first_gate =
            *first_expert.gate_up->implementation_;
        const NcnnVulkanMxfp4Operator::Implementation& first_down =
            *first_expert.down->implementation_;
        if (!first_gate.vulkan_context || first_gate.vulkan_context != first_down.vulkan_context
            || first_gate.storage.empty() || first_down.storage.empty())
        {
            return IndexedBatchResult::NotSupported;
        }

        const uint32_t input_columns = first_gate.input_columns;
        const uint32_t intermediate_columns = first_gate.output_columns / 2;
        const uint32_t output_columns = first_down.output_columns;
        if (input_columns == 0 || intermediate_columns == 0 || output_columns == 0
            || first_gate.output_columns % 2 != 0
            || first_down.input_columns != intermediate_columns)
        {
            return IndexedBatchResult::NotSupported;
        }

        size_t total_rows = 0;
        size_t maximum_request_rows = 0;
        std::vector<uint32_t> row_slots;
        std::vector<uint32_t> expert_row_offsets(selected.size() + 1, 0);
        std::vector<size_t> selected_request_indices;
        selected_request_indices.reserve(selected.size());
        for (size_t slot = 0; slot < selected.size(); ++slot)
        {
            const Selection& selection = selected[slot];
            if (selection.request_index >= requests.size())
                return IndexedBatchResult::NotSupported;
            const ExpertBackendRequest& request = requests[selection.request_index];
            if (!request.input || !request.output || request.input->rows() == 0
                || request.input->columns() != input_columns
                || request.input->dtype() != DType::Float32
                || total_rows > static_cast<size_t>(std::numeric_limits<int>::max()) - request.input->rows()
                || total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - request.input->rows())
            {
                return IndexedBatchResult::NotSupported;
            }
            const NcnnVulkanMxfp4ExpertOperator::Implementation& expert =
                *selection.entry->operation->implementation_;
            if (!expert.gate_up || !expert.down
                || expert.activation != first_expert.activation
                || expert.activation_limit != first_expert.activation_limit)
            {
                return IndexedBatchResult::NotSupported;
            }
            const NcnnVulkanMxfp4Operator::Implementation& gate =
                *expert.gate_up->implementation_;
            const NcnnVulkanMxfp4Operator::Implementation& down =
                *expert.down->implementation_;
            if (!gate.indexed_storage || !down.indexed_storage
                || gate.storage.empty() || down.storage.empty()
                || gate.vulkan_context != first_gate.vulkan_context
                || down.vulkan_context != first_gate.vulkan_context
                || gate.input_columns != input_columns
                || gate.output_columns != first_gate.output_columns
                || gate.block_count != first_gate.block_count
                || down.input_columns != intermediate_columns
                || down.output_columns != output_columns
                || down.block_count != first_down.block_count
                || gate.indexed_scales_word_offset != first_gate.indexed_scales_word_offset
                || gate.indexed_bias_word_offset != first_gate.indexed_bias_word_offset
                || down.indexed_scales_word_offset != first_down.indexed_scales_word_offset
                || down.indexed_bias_word_offset != first_down.indexed_bias_word_offset)
            {
                return IndexedBatchResult::NotSupported;
            }
            expert_row_offsets[slot] = static_cast<uint32_t>(total_rows);
            maximum_request_rows = std::max(maximum_request_rows, request.input->rows());
            total_rows += request.input->rows();
            selected_request_indices.push_back(selection.request_index);
        }
        if (total_rows == 0 || total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            return IndexedBatchResult::Failed;
        expert_row_offsets[selected.size()] = static_cast<uint32_t>(total_rows);

        // With at least two rows in the largest selected Expert, use the
        // MUL_MAT_ID-style row tile.  Single-row decode batches remain on the
        // scalar indexed kernel because there is no row reuse to amortize.
        const bool use_tiled_indexed = maximum_request_rows >= 2;

        if (!indexed_pipeline_
            && !create_mxfp4_indexed_pipeline(
                first_gate.vulkan_context,
                first_gate.option,
                indexed_pipeline_))
        {
            return IndexedBatchResult::NotSupported;
        }

        ActivationBuffer* aggregated_output = nullptr;
        uint32_t aggregated_token_count = 0;
        const bool use_route_aggregation = route_aggregation_enabled(
            requests,
            selected_request_indices,
            output_columns,
            aggregated_output,
            aggregated_token_count,
            first_gate.optimization_flags);
        std::vector<uint32_t> route_offsets;
        std::vector<uint32_t> route_rows;
        std::vector<float> route_weights;
        if (use_route_aggregation
            && !build_route_aggregation_metadata(
                requests,
                selected_request_indices,
                route_offsets,
                route_rows,
                route_weights))
        {
            return IndexedBatchResult::Failed;
        }

        NcnnVulkanRuntimeState& runtime_state =
            first_gate.vulkan_context->runtime_state();
        NcnnVulkanTransferLease transfer_lease =
            first_gate.vulkan_context->acquire_transfer_slot();
        NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
        if (!prepare_staging_batch(
                transfer_slot.upload,
                total_rows,
                input_columns,
                transfer_slot.staging_allocator,
                runtime_state,
                sizeof(float)))
        {
            return IndexedBatchResult::Failed;
        }
        ncnn::Mat mapped_input = transfer_slot.upload.mapped();
        if (mapped_input.empty() || mapped_input.dims != 2
            || mapped_input.w != static_cast<int>(input_columns)
            || mapped_input.h != static_cast<int>(total_rows)
            || mapped_input.elemsize != sizeof(float)
            || mapped_input.elempack != 1)
        {
            return IndexedBatchResult::Failed;
        }

        row_slots.assign(total_rows, 0);
        size_t row_offset = 0;
        auto* mapped_input_bytes = static_cast<std::byte*>(mapped_input.data);
        for (size_t slot = 0; slot < selected.size(); ++slot)
        {
            const ExpertBackendRequest& request =
                requests[selected[slot].request_index];
            const ActivationBuffer& input = *request.input;
            if (input.bytes().size() != input.rows() * static_cast<size_t>(input_columns) * sizeof(float))
                return IndexedBatchResult::Failed;
            std::memcpy(
                mapped_input_bytes + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                input.bytes().data(),
                input.bytes().size());
            std::fill(
                row_slots.begin() + static_cast<std::ptrdiff_t>(row_offset),
                row_slots.begin() + static_cast<std::ptrdiff_t>(row_offset + input.rows()),
                static_cast<uint32_t>(slot));
            row_offset += input.rows();
        }
        transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
        transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

        const bool direct_host_input = direct_host_input_enabled(
            *first_gate.vulkan_context,
            total_rows * input_columns * sizeof(float),
            DType::Float32);
        const bool direct_host_output = !use_route_aggregation
                                        && direct_host_output_enabled(
                                            *first_gate.vulkan_context,
                                            total_rows * output_columns * sizeof(float),
                                            DType::Float32);
        const size_t download_rows = use_route_aggregation ? aggregated_token_count : total_rows;
        if (!prepare_staging_batch(
                transfer_slot.download,
                download_rows,
                output_columns,
                transfer_slot.staging_allocator,
                runtime_state)
            || !fill_staging_values(
                row_slots.data(),
                row_slots.size(),
                sizeof(uint32_t),
                transfer_slot.expert_slots,
                transfer_slot.staging_allocator,
                runtime_state)
            || !fill_staging_values(
                expert_row_offsets.data(),
                expert_row_offsets.size(),
                sizeof(uint32_t),
                transfer_slot.expert_row_offsets,
                transfer_slot.staging_allocator,
                runtime_state))
        {
            return IndexedBatchResult::Failed;
        }
        if (use_route_aggregation
            && (!fill_staging_values(
                    route_offsets.data(),
                    route_offsets.size(),
                    sizeof(uint32_t),
                    transfer_slot.route_offsets,
                    transfer_slot.staging_allocator,
                    runtime_state)
                || !fill_staging_values(
                    route_rows.data(),
                    route_rows.size(),
                    sizeof(uint32_t),
                    transfer_slot.route_rows,
                    transfer_slot.staging_allocator,
                    runtime_state)
                || !fill_staging_values(
                    route_weights.data(),
                    route_weights.size(),
                    sizeof(float),
                    transfer_slot.route_weights,
                    transfer_slot.staging_allocator,
                    runtime_state)))
        {
            return IndexedBatchResult::Failed;
        }

        ActivationBuffer combined_output;
        if (!use_route_aggregation)
            combined_output.reset(total_rows, output_columns, false);

        std::unique_lock<std::mutex> lock(
            first_gate.vulkan_context->command_mutex());
        ncnn::VkCompute& command = *transfer_slot.command;
        if (transfer_slot.command_used)
        {
            if (command.reset() != 0)
                return IndexedBatchResult::Failed;
            ++runtime_state.command_buffer_reuses;
        }
        transfer_slot.command_used = true;

        ncnn::VkMat input_gpu;
        if (direct_host_input)
            input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
        else if (!record_mapped_upload(
                     transfer_slot.upload,
                     input_gpu,
                     command,
                     first_gate.option))
        {
            return IndexedBatchResult::Failed;
        }
        ncnn::VkMat expert_ids_gpu;
        if (!record_mapped_upload(
                transfer_slot.expert_slots,
                expert_ids_gpu,
                command,
                first_gate.option))
        {
            return IndexedBatchResult::Failed;
        }
        ncnn::VkMat expert_row_offsets_gpu;
        if (!record_mapped_upload(
                transfer_slot.expert_row_offsets,
                expert_row_offsets_gpu,
                command,
                first_gate.option))
        {
            return IndexedBatchResult::Failed;
        }

        ncnn::VkMat intermediate_gpu;
        intermediate_gpu.create(
            static_cast<int>(intermediate_columns),
            static_cast<int>(total_rows),
            sizeof(float),
            first_gate.vulkan_context->blob_allocator());
        if (intermediate_gpu.empty())
            return IndexedBatchResult::Failed;

        ncnn::VkMat output_gpu;
        if (direct_host_output)
        {
            output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
        }
        else
        {
            output_gpu.create(
                static_cast<int>(output_columns),
                static_cast<int>(total_rows),
                sizeof(float),
                first_gate.vulkan_context->blob_allocator());
        }
        if (output_gpu.empty())
            return IndexedBatchResult::Failed;

        const auto make_constants = [&](uint32_t mode) {
            std::vector<ncnn::vk_constant_type> constants(18);
            constants[0].u32 = first_gate.input_columns;
            constants[1].u32 = first_gate.output_columns;
            constants[2].u32 = first_down.input_columns;
            constants[3].u32 = first_down.output_columns;
            constants[4].u32 = first_gate.block_count;
            constants[5].u32 = first_down.block_count;
            constants[6].u32 = static_cast<uint32_t>(total_rows);
            constants[7].u32 = static_cast<uint32_t>(selected.size());
            constants[8].u32 = 0;
            constants[9].u32 = first_gate.indexed_scales_word_offset;
            constants[10].u32 = first_gate.indexed_bias_word_offset;
            constants[11].u32 = 0;
            constants[12].u32 = first_down.indexed_scales_word_offset;
            constants[13].u32 = first_down.indexed_bias_word_offset;
            constants[14].u32 = mode;
            constants[15].u32 = first_expert.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
            constants[16].f = first_expert.activation_limit;
            constants[17].u32 = use_tiled_indexed ? 1u : 0u;
            return constants;
        };

        std::vector<ncnn::VkMat> gate_bindings;
        gate_bindings.reserve(mxfp4_indexed_max_experts + 3);
        gate_bindings.push_back(input_gpu);
        for (uint32_t slot = 0; slot < mxfp4_indexed_max_experts; ++slot)
        {
            if (slot < selected.size())
            {
                gate_bindings.push_back(
                    selected[slot].entry->operation->implementation_->gate_up->implementation_->storage);
            }
            else
            {
                gate_bindings.emplace_back();
            }
        }
        gate_bindings.push_back(use_tiled_indexed ? expert_row_offsets_gpu : expert_ids_gpu);
        gate_bindings.push_back(intermediate_gpu);
        std::vector<unsigned char> readonly_bindings(gate_bindings.size(), 1);
        readonly_bindings.back() = 0;
        ncnn::VkMat gate_dispatcher;
        gate_dispatcher.w = static_cast<int>(intermediate_columns * 32);
        gate_dispatcher.h = static_cast<int>(use_tiled_indexed ? (maximum_request_rows + 1) / 2 : total_rows);
        gate_dispatcher.c = static_cast<int>(use_tiled_indexed ? selected.size() : 1);
        command.record_pipeline_readonly(
            indexed_pipeline_.get(),
            gate_bindings,
            readonly_bindings,
            make_constants(0),
            gate_dispatcher);

        std::vector<ncnn::VkMat> down_bindings;
        down_bindings.reserve(mxfp4_indexed_max_experts + 3);
        down_bindings.push_back(intermediate_gpu);
        for (uint32_t slot = 0; slot < mxfp4_indexed_max_experts; ++slot)
        {
            if (slot < selected.size())
            {
                down_bindings.push_back(
                    selected[slot].entry->operation->implementation_->down->implementation_->storage);
            }
            else
            {
                down_bindings.emplace_back();
            }
        }
        down_bindings.push_back(use_tiled_indexed ? expert_row_offsets_gpu : expert_ids_gpu);
        down_bindings.push_back(output_gpu);
        readonly_bindings.assign(down_bindings.size(), 1);
        readonly_bindings.back() = 0;
        ncnn::VkMat down_dispatcher;
        down_dispatcher.w = static_cast<int>(output_columns * 32);
        down_dispatcher.h = static_cast<int>(use_tiled_indexed ? (maximum_request_rows + 1) / 2 : total_rows);
        down_dispatcher.c = static_cast<int>(use_tiled_indexed ? selected.size() : 1);
        command.record_pipeline_readonly(
            indexed_pipeline_.get(),
            down_bindings,
            readonly_bindings,
            make_constants(1),
            down_dispatcher);

        if (use_route_aggregation)
        {
            ncnn::VkMat route_offsets_gpu;
            ncnn::VkMat route_rows_gpu;
            ncnn::VkMat route_weights_gpu;
            if (!record_mapped_upload(
                    transfer_slot.route_offsets,
                    route_offsets_gpu,
                    command,
                    first_gate.option)
                || !record_mapped_upload(
                    transfer_slot.route_rows,
                    route_rows_gpu,
                    command,
                    first_gate.option)
                || !record_mapped_upload(
                    transfer_slot.route_weights,
                    route_weights_gpu,
                    command,
                    first_gate.option))
            {
                return IndexedBatchResult::Failed;
            }
            if (!route_aggregation_pipeline_
                && !create_mxfp4_route_aggregation_pipeline(
                    first_gate.vulkan_context,
                    first_gate.option,
                    route_aggregation_pipeline_))
            {
                return IndexedBatchResult::Failed;
            }
            ncnn::VkMat aggregated_output_gpu;
            aggregated_output_gpu.create(
                static_cast<int>(output_columns),
                static_cast<int>(aggregated_token_count),
                sizeof(float),
                first_gate.vulkan_context->blob_allocator());
            if (aggregated_output_gpu.empty())
                return IndexedBatchResult::Failed;
            std::vector<ncnn::VkMat> aggregation_bindings = {
                output_gpu,
                route_offsets_gpu,
                route_rows_gpu,
                route_weights_gpu,
                aggregated_output_gpu,
            };
            std::vector<ncnn::vk_constant_type> aggregation_constants(2);
            aggregation_constants[0].u32 = output_columns;
            aggregation_constants[1].u32 = aggregated_token_count;
            ncnn::VkMat aggregation_dispatcher;
            aggregation_dispatcher.w = static_cast<int>(output_columns * 128);
            aggregation_dispatcher.h = static_cast<int>(aggregated_token_count);
            aggregation_dispatcher.c = 1;
            command.record_pipeline(
                route_aggregation_pipeline_.get(),
                aggregation_bindings,
                aggregation_constants,
                aggregation_dispatcher);
            if (!record_prepared_activation_staging_download(
                    aggregated_output_gpu,
                    aggregated_token_count,
                    output_columns,
                    transfer_slot.download,
                    command,
                    first_down.vulkan_context->device(),
                    first_down.option,
                    aggregated_output->dtype())
                || submit_compute_and_wait(command, runtime_state) != 0
                || !copy_staging_to_cpu_batch(transfer_slot.download, *aggregated_output))
            {
                return IndexedBatchResult::Failed;
            }
        }
        else if ((!direct_host_output
                  && !record_prepared_activation_staging_download(
                      output_gpu,
                      total_rows,
                      output_columns,
                      transfer_slot.download,
                      command,
                      first_down.vulkan_context->device(),
                      first_down.option,
                      combined_output.dtype()))
                 || submit_compute_and_wait(command, runtime_state) != 0
                 || !copy_staging_to_cpu_batch(transfer_slot.download, combined_output))
        {
            return IndexedBatchResult::Failed;
        }
        lock.unlock();

        if (use_route_aggregation)
        {
            for (const Selection& selection : selected)
            {
                const ExpertBackendRequest& request =
                    requests[selection.request_index];
                if (request.route_aggregation.completed)
                    *request.route_aggregation.completed = 1;
            }
        }
        else
        {
            row_offset = 0;
            for (const Selection& selection : selected)
            {
                const ExpertBackendRequest& request =
                    requests[selection.request_index];
                request.output->reset(request.input->rows(), output_columns, false);
                for (size_t row = 0; row < request.input->rows(); ++row)
                {
                    std::copy_n(
                        combined_output.row(row_offset + row),
                        output_columns,
                        request.output->row(row));
                }
                row_offset += request.input->rows();
            }
        }
        runtime_state.dispatches +=
            2 + static_cast<uint64_t>(use_route_aggregation);
        ++runtime_state.compute_submissions;
        ++runtime_state.batch_uploads;
        ++runtime_state.batch_downloads;
        return IndexedBatchResult::Executed;
    }

    bool forward_batch(std::span<const ExpertBackendRequest> requests, std::span<const Selection> selected)
    {
        if (selected.empty())
            return true;
        const NcnnVulkanMxfp4ExpertOperator::Implementation& first_expert = *selected.front().entry->operation->implementation_;
        if (!first_expert.gate_up || !first_expert.down || !first_expert.gate_up_pipeline)
        {
            return false;
        }
        const NcnnVulkanMxfp4Operator::Implementation& first_gate = *first_expert.gate_up->implementation_;
        const NcnnVulkanMxfp4Operator::Implementation& first_down = *first_expert.down->implementation_;
        if (!first_gate.vulkan_context || first_gate.vulkan_context != first_down.vulkan_context)
        {
            return false;
        }
        NcnnVulkanRuntimeState& runtime_state =
            first_gate.vulkan_context->runtime_state();
        const uint32_t input_columns = first_gate.input_columns;
        const uint32_t output_columns = first_down.output_columns;
        size_t total_rows = 0;
        std::vector<size_t> selected_request_indices;
        selected_request_indices.reserve(selected.size());
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            const auto& expert = *selection.entry->operation->implementation_;
            const auto& gate = *expert.gate_up->implementation_;
            const auto& down = *expert.down->implementation_;
            if (!request.input || !request.output || request.input->rows() == 0 || request.input->columns() != input_columns
                || gate.vulkan_context != first_gate.vulkan_context || down.vulkan_context != first_gate.vulkan_context || gate.input_columns != input_columns
                || gate.output_columns % 2 != 0 || down.input_columns != gate.output_columns / 2 || down.output_columns != output_columns
                || total_rows > static_cast<size_t>(std::numeric_limits<int>::max()) - request.input->rows())
            {
                return false;
            }
            total_rows += request.input->rows();
            selected_request_indices.push_back(selection.request_index);
        }
        if (total_rows == 0 || total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            return false;
        }

        ActivationBuffer* aggregated_output = nullptr;
        uint32_t aggregated_token_count = 0;
        const bool use_route_aggregation = route_aggregation_enabled(
            requests,
            selected_request_indices,
            output_columns,
            aggregated_output,
            aggregated_token_count,
            first_gate.optimization_flags);
        std::vector<uint32_t> route_offsets;
        std::vector<uint32_t> route_rows;
        std::vector<float> route_weights;
        if (use_route_aggregation
            && !build_route_aggregation_metadata(
                requests,
                selected_request_indices,
                route_offsets,
                route_rows,
                route_weights))
        {
            return false;
        }

        NcnnVulkanTransferLease transfer_lease = first_gate.vulkan_context->acquire_transfer_slot();
        NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
        const DType input_dtype = requests[selected.front().request_index].input->dtype();
        if (input_dtype != DType::Float32)
            return false;
        if (!prepare_staging_batch(
                transfer_slot.upload,
                total_rows,
                input_columns,
                transfer_slot.staging_allocator,
                runtime_state,
                sizeof(float)))
        {
            return false;
        }
        ncnn::Mat mapped_input = transfer_slot.upload.mapped();
        if (mapped_input.empty()
            || mapped_input.dims != 2
            || mapped_input.w != static_cast<int>(input_columns)
            || mapped_input.h != static_cast<int>(total_rows)
            || mapped_input.elemsize != sizeof(float)
            || mapped_input.elempack != 1)
        {
            return false;
        }
        size_t row_offset = 0;
        auto* mapped_input_bytes = static_cast<std::byte*>(mapped_input.data);
        for (const Selection& selection : selected)
        {
            const ActivationBuffer& input = *requests[selection.request_index].input;
            if (input.dtype() != input_dtype
                || input.bytes().size() != input.rows() * static_cast<size_t>(input_columns) * sizeof(float))
            {
                return false;
            }
            std::memcpy(
                mapped_input_bytes + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                input.bytes().data(),
                input.bytes().size());
            row_offset += input.rows();
        }
        transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
        transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        const bool direct_host_input = direct_host_input_enabled(
            *first_gate.vulkan_context,
            total_rows * input_columns * sizeof(float),
            input_dtype);
        const bool direct_host_output = !use_route_aggregation
                                        && direct_host_output_enabled(
                                            *first_gate.vulkan_context,
                                            total_rows * output_columns
                                                * sizeof(float),
                                            DType::Float32);
        const size_t download_rows = use_route_aggregation ? aggregated_token_count : total_rows;
        if (!prepare_staging_batch(transfer_slot.download, download_rows, output_columns, transfer_slot.staging_allocator, runtime_state)
            || (use_route_aggregation
                && (!fill_staging_values(
                        route_offsets.data(),
                        route_offsets.size(),
                        sizeof(uint32_t),
                        transfer_slot.route_offsets,
                        transfer_slot.staging_allocator, runtime_state)
                    || !fill_staging_values(
                        route_rows.data(),
                        route_rows.size(),
                        sizeof(uint32_t),
                        transfer_slot.route_rows,
                        transfer_slot.staging_allocator, runtime_state)
                    || !fill_staging_values(
                        route_weights.data(),
                        route_weights.size(),
                        sizeof(float),
                        transfer_slot.route_weights,
                        transfer_slot.staging_allocator, runtime_state))))
        {
            return false;
        }
        ActivationBuffer combined_output;
        if (!use_route_aggregation)
            combined_output.reset(total_rows, output_columns, false);
        std::unique_lock<std::mutex> lock(first_gate.vulkan_context->command_mutex());
        ncnn::VkCompute& command = *transfer_slot.command;
        if (transfer_slot.command_used)
        {
            if (command.reset() != 0)
                return false;
            ++runtime_state.command_buffer_reuses;
        }
        transfer_slot.command_used = true;
        ncnn::VkMat input_gpu;
        if (direct_host_input)
            input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
        else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first_gate.option))
        {
            return false;
        }
        ncnn::VkMat output_gpu;
        if (direct_host_output)
            output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
        else
            output_gpu.create(static_cast<int>(output_columns), static_cast<int>(total_rows), sizeof(float), first_gate.vulkan_context->blob_allocator());
        if (output_gpu.empty())
            return false;
        std::vector<ncnn::VkMat> intermediates;
        intermediates.reserve(selected.size());
        row_offset = 0;
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            const auto& expert = *selection.entry->operation->implementation_;
            const auto& gate = *expert.gate_up->implementation_;
            const auto& down = *expert.down->implementation_;
            const uint32_t intermediate_columns = gate.output_columns / 2;
            ncnn::VkMat input_view = row_view(input_gpu, row_offset, request.input->rows());
            ncnn::VkMat output_view = row_view(output_gpu, row_offset, request.input->rows());
            if (input_view.empty() || output_view.empty())
            {
                return false;
            }
            ncnn::VkMat intermediate;
            intermediate.create(static_cast<int>(intermediate_columns), static_cast<int>(request.input->rows()), sizeof(float),
                                first_gate.vulkan_context->blob_allocator());
            if (intermediate.empty())
                return false;
            intermediates.push_back(intermediate);

            std::vector<ncnn::VkMat> gate_bindings = {
                input_view,
                gate.packed,
                gate.scales,
                gate.bias,
                intermediate,
            };
            std::vector<ncnn::vk_constant_type> gate_constants(6);
            gate_constants[0].u32 = gate.input_columns;
            gate_constants[1].u32 = intermediate_columns;
            gate_constants[2].u32 = gate.block_count;
            gate_constants[3].u32 = static_cast<uint32_t>(request.input->rows());
            gate_constants[4].u32 = expert.activation == ExpertActivation::Silu
                                        ? 2
                                        : expert.activation == ExpertActivation::DeepSeekSwiGlu ? 1 : 0;
            gate_constants[5].f = expert.activation_limit;
            ncnn::VkMat gate_dispatcher;
            gate_dispatcher.w = static_cast<int>(intermediate_columns * 32);
            gate_dispatcher.h = static_cast<int>(request.input->rows());
            gate_dispatcher.c = 1;
            command.record_pipeline(expert.gate_up_pipeline.get(), gate_bindings, gate_constants, gate_dispatcher);

            std::vector<ncnn::VkMat> down_bindings = {
                intermediate,
                down.packed,
                down.scales,
                down.bias,
                output_view,
            };
            std::vector<ncnn::vk_constant_type> down_constants(4);
            down_constants[0].u32 = down.input_columns;
            down_constants[1].u32 = down.output_columns;
            down_constants[2].u32 = down.block_count;
            down_constants[3].u32 = static_cast<uint32_t>(request.input->rows());
            ncnn::VkMat down_dispatcher;
            down_dispatcher.w = static_cast<int>(down.output_columns * 32);
            down_dispatcher.h = static_cast<int>(request.input->rows());
            down_dispatcher.c = 1;
            command.record_pipeline(down.pipeline.get(), down_bindings, down_constants, down_dispatcher);
            row_offset += request.input->rows();
        }
        if (use_route_aggregation)
        {
            ncnn::VkMat route_offsets_gpu;
            ncnn::VkMat route_rows_gpu;
            ncnn::VkMat route_weights_gpu;
            if (!record_mapped_upload(transfer_slot.route_offsets, route_offsets_gpu, command, first_gate.option)
                || !record_mapped_upload(transfer_slot.route_rows, route_rows_gpu, command, first_gate.option)
                || !record_mapped_upload(transfer_slot.route_weights, route_weights_gpu, command, first_gate.option))
            {
                return false;
            }
            if (!route_aggregation_pipeline_
                && !create_mxfp4_route_aggregation_pipeline(
                    first_gate.vulkan_context,
                    first_gate.option,
                    route_aggregation_pipeline_))
            {
                return false;
            }
            ncnn::VkMat aggregated_output_gpu;
            aggregated_output_gpu.create(
                static_cast<int>(output_columns),
                static_cast<int>(aggregated_token_count),
                sizeof(float),
                first_gate.vulkan_context->blob_allocator());
            if (aggregated_output_gpu.empty())
                return false;
            std::vector<ncnn::VkMat> aggregation_bindings = {
                output_gpu,
                route_offsets_gpu,
                route_rows_gpu,
                route_weights_gpu,
                aggregated_output_gpu,
            };
            std::vector<ncnn::vk_constant_type> aggregation_constants(2);
            aggregation_constants[0].u32 = output_columns;
            aggregation_constants[1].u32 = aggregated_token_count;
            ncnn::VkMat aggregation_dispatcher;
            aggregation_dispatcher.w = static_cast<int>(output_columns * 128);
            aggregation_dispatcher.h = static_cast<int>(aggregated_token_count);
            aggregation_dispatcher.c = 1;
            command.record_pipeline(route_aggregation_pipeline_.get(), aggregation_bindings, aggregation_constants, aggregation_dispatcher);
            if (!record_prepared_activation_staging_download(
                    aggregated_output_gpu,
                    aggregated_token_count,
                    output_columns,
                    transfer_slot.download,
                    command,
                    first_down.vulkan_context->device(),
                    first_down.option,
                    aggregated_output->dtype())
                || submit_compute_and_wait(command, runtime_state) != 0
                || !copy_staging_to_cpu_batch(transfer_slot.download, *aggregated_output))
            {
                return false;
            }
        }
        else if ((!direct_host_output
                  && !record_prepared_activation_staging_download(
                      output_gpu,
                      total_rows,
                      output_columns,
                      transfer_slot.download,
                      command,
                      first_down.vulkan_context->device(),
                      first_down.option,
                      combined_output.dtype()))
                 || submit_compute_and_wait(command, runtime_state) != 0
                 || !copy_staging_to_cpu_batch(transfer_slot.download, combined_output))
        {
            return false;
        }
        lock.unlock();

        if (use_route_aggregation)
        {
            for (const Selection& selection : selected)
            {
                const ExpertBackendRequest& request = requests[selection.request_index];
                if (request.route_aggregation.completed)
                    *request.route_aggregation.completed = 1;
            }
        }
        else
        {
            row_offset = 0;
            for (const Selection& selection : selected)
            {
                ExpertBackendRequest const& request = requests[selection.request_index];
                request.output->reset(request.input->rows(), output_columns, false);
                for (size_t row = 0; row < request.input->rows(); ++row)
                {
                    std::copy_n(combined_output.row(row_offset + row), output_columns, request.output->row(row));
                }
                row_offset += request.input->rows();
            }
        }
        runtime_state.dispatches += selected.size() * 2 + (use_route_aggregation ? 1 : 0);
        ++runtime_state.compute_submissions;
        ++runtime_state.batch_uploads;
        ++runtime_state.batch_downloads;
        return true;
    }

    void execute_work_item(const std::shared_ptr<WorkItem>& work)
    {
        const auto started = std::chrono::steady_clock::now();
        const IndexedBatchResult indexed_result = runtime_optimization_enabled(
                                                     optimization_flags_,
                                                     RuntimeOptimizationVulkanIndexedExperts)
                                                     ? forward_indexed_batch(work->requests, work->selected)
                                                     : IndexedBatchResult::NotSupported;
        const bool executed = indexed_result == IndexedBatchResult::Executed
                              || forward_batch(work->requests, work->selected);
        const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
        uint64_t route_aggregation_routes = 0;
        uint32_t route_aggregation_token_count = 0;
        uint32_t route_aggregation_columns = 0;
        bool route_aggregated = false;
        if (executed)
        {
            for (const Selection& selection : work->selected)
            {
                if (selection.request_index >= work->requests.size())
                    continue;
                const ExpertBackendRequest& request = work->requests[selection.request_index];
                if (!request.route_aggregation.completed || *request.route_aggregation.completed == 0)
                    continue;
                if (!route_aggregated)
                {
                    route_aggregation_token_count = request.route_aggregation.token_count;
                    route_aggregation_columns = request.route_aggregation.output
                                                    ? request.route_aggregation.output->columns()
                                                    : 0;
                }
                route_aggregation_routes += request.route_aggregation.routes.size();
                route_aggregated = true;
            }
        }
        uint64_t route_aggregation_bytes_saved = 0;
        if (route_aggregated && route_aggregation_routes > route_aggregation_token_count && route_aggregation_columns != 0)
        {
            const uint64_t saved_rows = route_aggregation_routes - route_aggregation_token_count;
            if (saved_rows <= std::numeric_limits<uint64_t>::max() / route_aggregation_columns / sizeof(float))
            {
                route_aggregation_bytes_saved = saved_rows * route_aggregation_columns * sizeof(float);
            }
        }
        std::vector<ExpertBackendExecutionResult> final = work->planned;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            execution_time_microseconds_ += elapsed;
            if (!executed)
            {
                execution_failures_ += work->selected.size();
                for (const Selection& selection : work->selected)
                {
                    if (selection.entry->device_source)
                    {
                        ++device_source_execution_failures_;
                    }
                    final[selection.request_index] = ExpertBackendExecutionResult ::Failed;
                }
            }
            else
            {
                executions_ += work->selected.size();
                if (route_aggregated)
                {
                    ++route_aggregation_batches_;
                    route_aggregation_routes_ += route_aggregation_routes;
                    route_aggregation_bytes_saved_ += route_aggregation_bytes_saved;
                }
                for (const Selection& selection : work->selected)
                {
                    if (selection.entry->device_source)
                    {
                        ++device_source_executions_;
                    }
                }
            }
        }
        if (executed && device_weight_source_)
        {
            static constexpr size_t touch_batch_size = 256;
            std::array<std::string_view, touch_batch_size> keys;
            size_t key_count = 0;
            for (const Selection& selection : work->selected)
            {
                if (!selection.entry->device_source)
                    continue;
                keys[key_count++] = selection.entry->key;
                if (key_count == keys.size())
                {
                    device_weight_source_->touch_device_operations(std::span<const std::string_view>(keys.data(), key_count));
                    key_count = 0;
                }
            }
            if (key_count != 0)
                device_weight_source_->touch_device_operations(std::span<const std::string_view>(keys.data(), key_count));
        }
        {
            const std::lock_guard<std::mutex> lock(work->mutex);
            work->final = std::move(final);
            work->done = true;
        }
        work->completed.notify_all();
    }

    void execution_loop()
    {
        while (true)
        {
            std::shared_ptr<WorkItem> work;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                execution_available_.wait(lock, [this] { return stopping_ || !execution_pending_.empty(); });
                if (execution_pending_.empty())
                {
                    if (stopping_)
                        return;
                    continue;
                }
                work = std::move(execution_pending_.front());
                execution_pending_.pop_front();
            }
            execute_work_item(work);
        }
    }

    void finish_admission_locked()
    {
        --active_admissions_;
        if (active_admissions_ == 0)
            admission_idle_.notify_all();
    }

    void worker_loop()
    {
        while (true)
        {
            std::vector<std::shared_ptr<Entry>> retired;
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                retired.swap(retired_entries_);
            }
            retired.clear();
            PendingAdmission admission;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_available_.wait(lock, [this] {
                    // Admission is an asynchronous upload/compile stage.  It
                    // must keep making progress while a foreground Session
                    // is executing; otherwise every Expert admitted by that
                    // Session remains pending until the whole transaction
                    // completes, making Hybrid mode permanently CPU-only for
                    // cold Experts.  The worker remains bounded to one
                    // admission at a time.
                    return stopping_ || !pending_.empty();
                });
                if (stopping_)
                    return;
                admission = std::move(pending_.front());
                pending_.pop_front();
                ++active_admissions_;
            }

            auto operation = NcnnVulkanMxfp4ExpertOperator ::create_with_allocator(*admission.gate_up, admission.gate_up_bias.get(), *admission.down,
                                                                                   admission.down_bias.get(), admission.activation_limit, vulkan_device_index_,
                                                                                   expert_weight_allocator_.get(), admission.activation,
                                                                                   context_instance_,
                                                                                   optimization_flags_);
            std::shared_ptr<Entry> entry;
            if (operation)
            {
                entry = std::make_shared<Entry>();
                entry->key = admission.key;
                entry->operation = std::move(operation);
                entry->bytes = admission.bytes;
                entry->residency_group = admission.residency_group;
            }

            const std::lock_guard<std::mutex> lock(mutex_);
            pending_bytes_ -= admission.bytes;
            pending_keys_.erase(admission.key);
            if (stopping_)
            {
                finish_admission_locked();
                return;
            }
            if (!entry || entries_.find(entry->key) != entries_.end())
            {
                if (!entry)
                    ++dropped_admissions_;
                finish_admission_locked();
                continue;
            }
            bool frequent = false;
            bool from_frequent = false;
            (void)consume_ghost_locked(entry->key, entry->bytes, frequent, from_frequent);
            while (resident_bytes_ > capacity_bytes_ - entry->bytes)
            {
                if (!evict_one_locked(from_frequent, entry->residency_group, entry->bytes))
                {
                    ++dropped_admissions_;
                    entry.reset();
                    break;
                }
            }
            if (!entry)
            {
                finish_admission_locked();
                continue;
            }
            if (frequent)
            {
                frequent_.push_back(entry->key);
                entry->position = std::prev(frequent_.end());
                entry->list = ArcList::Frequent;
                frequent_bytes_ += entry->bytes;
            }
            else
            {
                recent_.push_back(entry->key);
                entry->position = std::prev(recent_.end());
                entry->list = ArcList::Recent;
                recent_bytes_ += entry->bytes;
            }
            resident_bytes_ += entry->bytes;
            if (entry->residency_group >= residency_group_bytes_.size())
            {
                residency_group_bytes_.resize(static_cast<size_t>(entry->residency_group) + 1, 0);
            }
            residency_group_bytes_[entry->residency_group] += entry->bytes;
            bytes_uploaded_ += entry->bytes;
            const std::string entry_key = entry->key;
            entries_.emplace(entry_key, std::move(entry));
            ++stores_;
            finish_admission_locked();
        }
    }

    const uint64_t capacity_bytes_;
    const uint32_t vulkan_device_index_;
    NcnnVulkanContextInstancePtr context_instance_;
    const uint64_t optimization_flags_;
    const uint64_t maximum_pending_bytes_;
    std::shared_ptr<NcnnVulkanContext> vulkan_context_;
    std::unique_ptr<ncnn::VkBlobAllocator> expert_weight_allocator_;
    std::shared_ptr<ncnn::Pipeline> indexed_pipeline_;
    std::shared_ptr<ncnn::Pipeline> route_aggregation_pipeline_;
    std::shared_ptr<VulkanExpertVictimCache> device_weight_source_;
    mutable std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable execution_available_;
    std::condition_variable admission_idle_;
    bool stopping_ = false;
    uint32_t foreground_depth_ = 0;
    std::deque<PendingAdmission> pending_;
    std::deque<std::shared_ptr<WorkItem>> execution_pending_;
    std::unordered_set<std::string, TransparentStringHash, std::equal_to<>> pending_keys_;
    uint64_t pending_bytes_ = 0;
    uint32_t active_admissions_ = 0;
    std::unordered_map<std::string, std::shared_ptr<Entry>, TransparentStringHash, std::equal_to<>> entries_;
    std::vector<std::shared_ptr<Entry>> retired_entries_;
    std::list<std::string> recent_;
    std::list<std::string> frequent_;
    uint64_t recent_bytes_ = 0;
    uint64_t frequent_bytes_ = 0;
    uint64_t recent_target_bytes_ = 0;
    GhostList recent_ghost_;
    GhostList frequent_ghost_;
    GhostIndex recent_ghost_index_;
    GhostIndex frequent_ghost_index_;
    uint64_t recent_ghost_bytes_ = 0;
    uint64_t frequent_ghost_bytes_ = 0;
    std::vector<uint64_t> residency_group_bytes_;
    uint64_t resident_bytes_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t admissions_ = 0;
    uint64_t stores_ = 0;
    uint64_t evictions_ = 0;
    uint64_t dropped_admissions_ = 0;
    uint64_t executions_ = 0;
    uint64_t execution_failures_ = 0;
    uint64_t cpu_preferred_ = 0;
    uint64_t bytes_uploaded_ = 0;
    uint64_t execution_time_microseconds_ = 0;
    uint64_t device_source_hits_ = 0;
    uint64_t device_source_misses_ = 0;
    uint64_t device_source_executions_ = 0;
    uint64_t device_source_execution_failures_ = 0;
    uint64_t route_aggregation_batches_ = 0;
    uint64_t route_aggregation_routes_ = 0;
    uint64_t route_aggregation_bytes_saved_ = 0;
    std::thread worker_;
    std::thread execution_worker_;
};

#endif

std::shared_ptr<IExpertExecutionBackend> create_vulkan_mxfp4_expert_backend(uint64_t capacity_bytes, uint32_t vulkan_device_index,
                                                                             std::shared_ptr<IExpertVictimCache> device_weight_source,
                                                                             const NcnnVulkanContextInstancePtr& context_instance,
                                                                             uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    auto source = std::dynamic_pointer_cast<VulkanExpertVictimCache>(std::move(device_weight_source));
    if ((capacity_bytes == 0 && !source) || NcnnLinearOperator::vulkan_device_count() == 0)
    {
        return {};
    }
    if (vulkan_device_index == automatic_vulkan_device_index)
    {
        vulkan_device_index = static_cast<uint32_t>(ncnn::get_default_gpu_index());
    }
    if (vulkan_device_index >= NcnnLinearOperator::vulkan_device_count())
    {
        return {};
    }
    return std::make_shared<VulkanMxfp4ExpertBackend>(
        capacity_bytes,
        vulkan_device_index,
        std::move(source),
        context_instance,
        optimization_flags);
#else
    (void)capacity_bytes;
    (void)vulkan_device_index;
    (void)device_weight_source;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

NcnnVulkanAttentionOperator::NcnnVulkanAttentionOperator()
    : implementation_(new Implementation)
{
}

NcnnVulkanAttentionOperator::~NcnnVulkanAttentionOperator() = default;

#if NCNN_MOE_WITH_VULKAN
static bool tensor_to_float_vector(const TensorData& tensor, std::vector<float>& values)
{
    if (tensor.dtype != DType::Float32 && tensor.dtype != DType::BFloat16)
        return false;
    values.resize(tensor.element_count());
    if (tensor.dtype == DType::Float32)
    {
        const std::span<const float> tensor_values = tensor.float32_values();
        if (tensor_values.size() != values.size())
            return false;
        std::copy(tensor_values.begin(), tensor_values.end(), values.begin());
    }
    else
    {
        const std::span<const uint16_t> tensor_values = tensor.bfloat16_values();
        if (tensor_values.size() != values.size())
            return false;
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = bfloat16_to_float(tensor_values[index]);
    }
    return true;
}

static bool fill_rope_staging_pair(ncnn::VkMat& cosine_staging, ncnn::VkMat& sine_staging, size_t token_count, uint64_t position_offset,
                                    const std::vector<float>& inverse_frequencies, float concentration, bool bfloat16_storage, ncnn::VkAllocator* allocator,
                                    NcnnVulkanRuntimeState& runtime_state)
{
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max()) || inverse_frequencies.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_matrix(cosine_staging, static_cast<int>(inverse_frequencies.size()), static_cast<int>(token_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator, runtime_state)
        || !prepare_staging_matrix(sine_staging, static_cast<int>(inverse_frequencies.size()), static_cast<int>(token_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator, runtime_state))
        return false;

    ncnn::Mat cosine_mapped = cosine_staging.mapped();
    ncnn::Mat sine_mapped = sine_staging.mapped();
    if (cosine_mapped.empty() || sine_mapped.empty())
        return false;
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        float* cosine_float_row = bfloat16_storage ? nullptr : cosine_mapped.row<float>(static_cast<int>(token_index));
        float* sine_float_row = bfloat16_storage ? nullptr : sine_mapped.row<float>(static_cast<int>(token_index));
        uint16_t* cosine_bfloat16_row = bfloat16_storage ? cosine_mapped.row<uint16_t>(static_cast<int>(token_index)) : nullptr;
        uint16_t* sine_bfloat16_row = bfloat16_storage ? sine_mapped.row<uint16_t>(static_cast<int>(token_index)) : nullptr;
        for (size_t index = 0; index < inverse_frequencies.size(); ++index)
        {
            const float angle = static_cast<float>(position_offset + token_index) * inverse_frequencies[index];
            const float cosine = std::cos(angle) * concentration;
            const float sine = std::sin(angle) * concentration;
            if (bfloat16_storage)
            {
                cosine_bfloat16_row[index] = float_to_bfloat16(cosine);
                sine_bfloat16_row[index] = float_to_bfloat16(sine);
            }
            else
            {
                cosine_float_row[index] = cosine;
                sine_float_row[index] = sine;
            }
        }
    }
    return true;
}

static bool fill_attention_mask_staging(ncnn::VkMat& staging, size_t token_count, uint64_t destination_count, uint64_t position_offset,
                                        const CpuLayerCache& cache, const NcnnVulkanAttentionConfig& config, const std::vector<float>& sinks,
                                        bool bfloat16_storage, ncnn::VkAllocator* allocator,
                                        NcnnVulkanRuntimeState& runtime_state)
{
    if (destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_tensor(staging, static_cast<int>(destination_count), static_cast<int>(token_count), static_cast<int>(config.head_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator, runtime_state))
        return false;

    // The finite sentinel avoids BF16 NaNs and still underflows after softmax.
    constexpr float masked_logit = -10000.0f;
    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty())
        return false;
    const uint64_t actual_end = cache.token_count + token_count;
    const bool use_attention_sink = has_flag(config.flags, NcnnAttentionSink);
    ncnn::Mat first_head = mapped.channel(0);
    for (size_t query_index = 0; query_index < token_count; ++query_index)
    {
        const uint64_t query_position = position_offset + query_index;
        if (bfloat16_storage)
        {
            uint16_t* row = first_head.row<uint16_t>(static_cast<int>(query_index));
            const uint16_t masked_value = float_to_bfloat16(masked_logit);
            for (uint64_t key_index = 0; key_index < actual_end; ++key_index)
            {
                const uint64_t key_position = key_index < cache.token_count ? cache.start_position + key_index : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0 && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_value : 0;
            }
            if (use_attention_sink)
                row[actual_end] = float_to_bfloat16(sinks[0]);
            std::fill(row + actual_end + (use_attention_sink ? 1 : 0), row + destination_count, masked_value);
        }
        else
        {
            float* row = first_head.row<float>(static_cast<int>(query_index));
            for (uint64_t key_index = 0; key_index < actual_end; ++key_index)
            {
                const uint64_t key_position = key_index < cache.token_count ? cache.start_position + key_index : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0 && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_logit : 0.0f;
            }
            if (use_attention_sink)
                row[actual_end] = sinks[0];
            std::fill(row + actual_end + (use_attention_sink ? 1 : 0), row + destination_count, masked_logit);
        }
    }
    for (uint32_t head = 1; head < config.head_count; ++head)
    {
        ncnn::Mat head_mask = mapped.channel(static_cast<int>(head));
        for (size_t query_index = 0; query_index < token_count; ++query_index)
        {
            if (bfloat16_storage)
            {
                const uint16_t* source = first_head.row<uint16_t>(static_cast<int>(query_index));
                uint16_t* row = head_mask.row<uint16_t>(static_cast<int>(query_index));
                std::copy_n(source, destination_count, row);
                if (use_attention_sink)
                    row[actual_end] = float_to_bfloat16(sinks[head]);
            }
            else
            {
                const float* source = first_head.row<float>(static_cast<int>(query_index));
                float* row = head_mask.row<float>(static_cast<int>(query_index));
                std::copy_n(source, destination_count, row);
                if (use_attention_sink)
                    row[actual_end] = sinks[head];
            }
        }
    }
    return true;
}

static bool fill_attention_cache_promotion_staging(
    ncnn::VkMat& key_staging,
    ncnn::VkMat& value_staging,
    const CpuLayerCache& cache,
    const NcnnVulkanAttentionConfig& config,
    ncnn::VkAllocator* allocator,
    NcnnVulkanRuntimeState& runtime_state)
{
    const uint32_t columns =
        config.kv_head_count * config.head_dimension;
    if (cache.token_count == 0
        || columns == 0
        || cache.token_count
               > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || config.kv_head_count
               > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || cache.capacity_tokens == 0
        || cache.first_slot >= cache.capacity_tokens
        || cache.token_count > cache.capacity_tokens
        || cache.columns != columns
        || (cache.dtype != DType::Float32
            && cache.dtype != DType::BFloat16)
        || cache.capacity_tokens
               > static_cast<uint64_t>(
                     std::numeric_limits<size_t>::max() / columns))
    {
        return false;
    }

    const size_t capacity_elements =
        static_cast<size_t>(cache.capacity_tokens) * columns;
    const bool bfloat16 = cache.dtype == DType::BFloat16;
    if ((bfloat16
         && (cache.bfloat16_keys.size() < capacity_elements
             || cache.bfloat16_values.size() < capacity_elements))
        || (!bfloat16
            && (cache.keys.size() < capacity_elements
                || cache.values.size() < capacity_elements))
        || !prepare_staging_tensor(
            key_staging,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            sizeof(float),
            allocator,
            runtime_state)
        || !prepare_staging_tensor(
            value_staging,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            sizeof(float),
            allocator,
            runtime_state))
    {
        return false;
    }

    ncnn::Mat key_mapped = key_staging.mapped();
    ncnn::Mat value_mapped = value_staging.mapped();
    if (key_mapped.empty() || value_mapped.empty())
        return false;

    for (uint32_t head = 0; head < config.kv_head_count; ++head)
    {
        ncnn::Mat key_channel = key_mapped.channel(static_cast<int>(head));
        ncnn::Mat value_channel =
            value_mapped.channel(static_cast<int>(head));
        const size_t head_offset =
            static_cast<size_t>(head) * config.head_dimension;
        for (uint64_t token = 0; token < cache.token_count; ++token)
        {
            const uint64_t slot =
                (cache.first_slot + token) % cache.capacity_tokens;
            const size_t source_offset =
                static_cast<size_t>(slot) * columns + head_offset;
            float* key_row =
                key_channel.row<float>(static_cast<int>(token));
            float* value_row =
                value_channel.row<float>(static_cast<int>(token));
            if (bfloat16)
            {
                for (uint32_t column = 0;
                     column < config.head_dimension;
                     ++column)
                {
                    key_row[column] = bfloat16_to_float(
                        cache.bfloat16_keys[source_offset + column]);
                    value_row[column] = bfloat16_to_float(
                        cache.bfloat16_values[source_offset + column]);
                }
            }
            else
            {
                std::copy_n(
                    cache.keys.data() + source_offset,
                    config.head_dimension,
                    key_row);
                std::copy_n(
                    cache.values.data() + source_offset,
                    config.head_dimension,
                    value_row);
            }
        }
    }
    return true;
}

static constexpr char attention_qkv_norm_rope_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer fused_qkv_blob
{
    float fused_qkv_data[];
};
layout(binding = 1) readonly buffer cosine_blob
{
    float cosine_data[];
};
layout(binding = 2) readonly buffer sine_blob
{
    float sine_data[];
};
layout(binding = 3) readonly buffer query_norm_blob
{
    float query_norm_data[];
};
layout(binding = 4) readonly buffer key_norm_blob
{
    float key_norm_data[];
};
layout(binding = 5) writeonly buffer query_blob
{
    float query_data[];
};
layout(binding = 6) writeonly buffer key_blob
{
    sfp key_data[];
};
layout(binding = 7) writeonly buffer value_blob
{
    sfp value_data[];
};
layout(binding = 8) writeonly buffer gate_blob
{
    float gate_data[];
};

layout(push_constant) uniform parameter
{
    uint total_columns;
    uint query_columns;
    uint key_value_columns;
    uint head_dimension;
    uint query_head_count;
    uint key_value_head_count;
    uint token_count;
    uint query_cstep;
    uint key_cstep;
    uint value_cstep;
    uint work_items;
    uint direct_ring;
    uint ring_capacity;
    uint destination_start;
    uint rope_dimension;
    float epsilon;
    uint device_rope;
    uint position_offset;
    float rope_concentration;
}
p;

shared float partial[32];

void main()
{
    const uint lane = gl_LocalInvocationID.x;
    const uint unit = gl_WorkGroupID.x;
    const uint heads_per_token = p.query_head_count + p.key_value_head_count;
    if (heads_per_token == 0 || unit >= p.work_items)
        return;

    const uint token = unit / heads_per_token;
    const uint head = unit % heads_per_token;
    if (token >= p.token_count)
        return;

    const bool is_query = head < p.query_head_count;
    const uint local_head = is_query ? head : head - p.query_head_count;
    const uint source_base = token * p.total_columns
        + (is_query ? local_head * p.head_dimension
                    : p.query_columns + local_head * p.head_dimension);
    float square_sum = 0.0;
    for (uint dimension = lane; dimension < p.head_dimension; dimension += 32)
    {
        const float value = fused_qkv_data[source_base + dimension];
        square_sum += value * value;
    }
    partial[lane] = square_sum;
    barrier();
    for (uint stride = 16; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial[lane] += partial[lane + stride];
        barrier();
    }
    const float inverse_rms = inversesqrt(
        partial[0] / float(p.head_dimension) + p.epsilon);
    barrier();

    const uint rope_half = p.rope_dimension / 2;
    if (is_query)
    {
        const uint destination_base = local_head * p.query_cstep
            + token * p.head_dimension;
        const uint gate_base = token * p.query_columns
            + local_head * p.head_dimension;
        for (uint pair = lane; pair < rope_half; pair += 32)
        {
            const float first = fused_qkv_data[source_base + pair]
                * inverse_rms * query_norm_data[pair];
            const float second = fused_qkv_data[source_base + rope_half + pair]
                * inverse_rms * query_norm_data[rope_half + pair];
            const float angle = float(p.position_offset + token)
                * cosine_data[pair];
            const float cosine = p.device_rope != 0
                ? cos(angle) * p.rope_concentration
                : cosine_data[token * rope_half + pair];
            const float sine = p.device_rope != 0
                ? sin(angle) * p.rope_concentration
                : sine_data[token * rope_half + pair];
            query_data[destination_base + pair] = first * cosine - second * sine;
            query_data[destination_base + rope_half + pair] = first * sine + second * cosine;
        }
        for (uint dimension = lane; dimension < p.head_dimension; dimension += 32)
        {
            if (dimension >= p.rope_dimension)
            {
                query_data[destination_base + dimension] =
                    fused_qkv_data[source_base + dimension]
                    * inverse_rms * query_norm_data[dimension];
            }
            const uint gate_source = token * p.total_columns
                + p.query_columns + p.key_value_columns * 2
                + local_head * p.head_dimension + dimension;
            gate_data[gate_base + dimension] =
                1.0 / (1.0 + exp(-fused_qkv_data[gate_source]));
        }
    }
    else
    {
        const uint destination_row = p.direct_ring != 0
            ? (p.destination_start + token) % p.ring_capacity
            : token;
        const uint destination_base = local_head * p.key_cstep
            + destination_row * p.head_dimension;
        const uint value_destination_base = local_head * p.value_cstep
            + destination_row * p.head_dimension;
        for (uint pair = lane; pair < rope_half; pair += 32)
        {
            const float first = fused_qkv_data[source_base + pair]
                * inverse_rms * key_norm_data[pair];
            const float second = fused_qkv_data[source_base + rope_half + pair]
                * inverse_rms * key_norm_data[rope_half + pair];
            const float angle = float(p.position_offset + token)
                * cosine_data[pair];
            const float cosine = p.device_rope != 0
                ? cos(angle) * p.rope_concentration
                : cosine_data[token * rope_half + pair];
            const float sine = p.device_rope != 0
                ? sin(angle) * p.rope_concentration
                : sine_data[token * rope_half + pair];
            const float rotated_first = first * cosine - second * sine;
            const float rotated_second = first * sine + second * cosine;
            buffer_st1(key_data, destination_base + pair, rotated_first);
            buffer_st1(key_data, destination_base + rope_half + pair, rotated_second);
            if (p.direct_ring != 0)
            {
                const uint duplicate = destination_base
                    + p.ring_capacity * p.head_dimension;
                buffer_st1(key_data, duplicate + pair, rotated_first);
                buffer_st1(key_data, duplicate + rope_half + pair, rotated_second);
            }
        }
        for (uint dimension = lane; dimension < p.head_dimension; dimension += 32)
        {
            if (dimension >= p.rope_dimension)
            {
                const float normalized =
                    fused_qkv_data[source_base + dimension]
                    * inverse_rms * key_norm_data[dimension];
                buffer_st1(key_data, destination_base + dimension, normalized);
                if (p.direct_ring != 0)
                {
                    buffer_st1(
                        key_data,
                        destination_base
                            + p.ring_capacity * p.head_dimension
                            + dimension,
                        normalized);
                }
            }
            const uint value_source = token * p.total_columns
                + p.query_columns + p.key_value_columns
                + local_head * p.head_dimension + dimension;
            const float value = fused_qkv_data[value_source];
            buffer_st1(value_data, value_destination_base + dimension, value);
            if (p.direct_ring != 0)
            {
                buffer_st1(
                    value_data,
                    value_destination_base
                        + p.ring_capacity * p.head_dimension
                        + dimension,
                    value);
            }
        }
    }
}
)glsl";

static constexpr char attention_output_gate_shader[] = R"glsl(
#version 450

layout(binding = 0) buffer attention_blob
{
    float attention_data[];
};
layout(binding = 1) readonly buffer gate_blob
{
    float gate_data[];
};

layout(push_constant) uniform parameter
{
    uint elements;
}
p;

void main()
{
    const uint index = gl_GlobalInvocationID.x;
    if (index >= p.elements)
        return;
    attention_data[index] *= gate_data[index];
}
)glsl";

static constexpr char attention_qkv_rope_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer fused_qkv_blob
{
    float fused_qkv_data[];
};
layout(binding = 1) readonly buffer cosine_blob
{
    float cosine_data[];
};
layout(binding = 2) readonly buffer sine_blob
{
    float sine_data[];
};
layout(binding = 3) writeonly buffer query_blob
{
    float query_data[];
};
layout(binding = 4) writeonly buffer key_blob
{
    sfp key_data[];
};
layout(binding = 5) writeonly buffer value_blob
{
    sfp value_data[];
};

layout(push_constant) uniform parameter
{
    uint total_columns;
    uint query_columns;
    uint key_value_columns;
    uint head_dimension;
    uint token_count;
    uint query_cstep;
    uint key_cstep;
    uint value_cstep;
    uint work_items;
    uint direct_ring;
    uint ring_capacity;
    uint destination_start;
    uint device_rope;
    uint position_offset;
    float rope_concentration;
}
p;

void main()
{
    const uint index = gl_GlobalInvocationID.x;
    if (index >= p.work_items)
        return;

    const uint half_dimension = p.head_dimension / 2;
    const uint query_heads = p.query_columns / p.head_dimension;
    const uint key_value_heads = p.key_value_columns / p.head_dimension;
    const uint query_pairs = p.token_count * query_heads * half_dimension;
    const uint key_pairs = p.token_count * key_value_heads * half_dimension;
    const uint value_elements = p.token_count * p.key_value_columns;

    if (index < query_pairs)
    {
        const uint pair = index % half_dimension;
        const uint head_token = index / half_dimension;
        const uint head = head_token % query_heads;
        const uint token = head_token / query_heads;
        const uint source = token * p.total_columns + head * p.head_dimension + pair;
        const uint destination = head * p.query_cstep + token * p.head_dimension + pair;
        const uint cache = token * half_dimension + pair;
        const float angle = float(p.position_offset + token)
            * cosine_data[pair];
        const float cosine = p.device_rope != 0
            ? cos(angle) * p.rope_concentration
            : cosine_data[cache];
        const float sine = p.device_rope != 0
            ? sin(angle) * p.rope_concentration
            : sine_data[cache];
        const float first = fused_qkv_data[source];
        const float second = fused_qkv_data[source + half_dimension];
        query_data[destination] = first * cosine - second * sine;
        query_data[destination + half_dimension] = first * sine + second * cosine;
        return;
    }

    const uint key_index = index - query_pairs;
    if (key_index < key_pairs)
    {
        const uint pair = key_index % half_dimension;
        const uint head_token = key_index / half_dimension;
        const uint head = head_token % key_value_heads;
        const uint token = head_token / key_value_heads;
        const uint source = token * p.total_columns + p.query_columns + head * p.head_dimension + pair;
        const uint destination_row = p.direct_ring != 0 ? (p.destination_start + token) % p.ring_capacity : token;
        const uint destination = head * p.key_cstep + destination_row * p.head_dimension + pair;
        const uint cache = token * half_dimension + pair;
        const float angle = float(p.position_offset + token)
            * cosine_data[pair];
        const float cosine = p.device_rope != 0
            ? cos(angle) * p.rope_concentration
            : cosine_data[cache];
        const float sine = p.device_rope != 0
            ? sin(angle) * p.rope_concentration
            : sine_data[cache];
        const float first = fused_qkv_data[source];
        const float second = fused_qkv_data[source + half_dimension];
        const float rotated_first = first * cosine - second * sine;
        const float rotated_second = first * sine + second * cosine;
        buffer_st1(key_data, destination, rotated_first);
        buffer_st1(key_data, destination + half_dimension, rotated_second);
        if (p.direct_ring != 0)
        {
            const uint duplicate = destination + p.ring_capacity * p.head_dimension;
            buffer_st1(key_data, duplicate, rotated_first);
            buffer_st1(key_data, duplicate + half_dimension, rotated_second);
        }
        return;
    }

    const uint value_index = key_index - key_pairs;
    if (value_index >= value_elements)
        return;
    const uint dimension = value_index % p.head_dimension;
    const uint head_token = value_index / p.head_dimension;
    const uint head = head_token % key_value_heads;
    const uint token = head_token / key_value_heads;
    const uint source = token * p.total_columns + p.query_columns + p.key_value_columns + head * p.head_dimension + dimension;
    const uint destination_row = p.direct_ring != 0 ? (p.destination_start + token) % p.ring_capacity : token;
    const uint destination = head * p.value_cstep + destination_row * p.head_dimension + dimension;
    buffer_st1(value_data, destination, fused_qkv_data[source]);
    if (p.direct_ring != 0)
    {
        buffer_st1(
            value_data,
            destination + p.ring_capacity * p.head_dimension,
            fused_qkv_data[source]);
    }
}
)glsl";

static constexpr char attention_decode_sdpa_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer query_blob
{
    float query_data[];
};
layout(binding = 1) readonly buffer key_blob
{
    sfp key_data[];
};
layout(binding = 2) readonly buffer value_blob
{
    sfp value_data[];
};
layout(binding = 3) readonly buffer sink_blob
{
    float sink_data[];
};
layout(binding = 4) writeonly buffer output_blob
{
    float output_data[];
};

layout(push_constant) uniform parameter
{
    uint head_dimension;
    uint destination_count;
    uint head_count;
    uint key_value_head_count;
    uint query_cstep;
    uint key_cstep;
    uint value_cstep;
    uint use_attention_sink;
    float scale;
}
p;

shared float reduction[128];
shared float online_maximum;
shared float online_sum;
shared float previous_scale;
shared float current_weight;

void main()
{
    const uint lane = gl_LocalInvocationID.x;
    const uint head = gl_WorkGroupID.z;
    if (head >= p.head_count || lane >= 128)
        return;

    const uint heads_per_group = p.head_count / p.key_value_head_count;
    const uint key_value_head = head / heads_per_group;
    const uint query_base = head * p.query_cstep;
    const uint key_base = key_value_head * p.key_cstep;
    const uint value_base = key_value_head * p.value_cstep;
    float numerator = 0.0;

    if (lane == 0)
    {
        online_maximum = -3.402823466e+38;
        online_sum = 0.0;
    }
    barrier();

    for (uint token = 0; token < p.destination_count; ++token)
    {
        const bool sink_token = p.use_attention_sink != 0 && token + 1 == p.destination_count;
        float partial = 0.0;
        if (lane < p.head_dimension && !sink_token)
        {
            partial = query_data[query_base + lane]
                      * buffer_ld1(key_data, key_base + token * p.head_dimension + lane);
        }
        reduction[lane] = partial;
        barrier();
        for (uint stride = 64; stride > 0; stride >>= 1)
        {
            if (lane < stride)
                reduction[lane] += reduction[lane + stride];
            barrier();
        }

        if (lane == 0)
        {
            const float score = reduction[0] * p.scale
                                + (sink_token ? sink_data[head] : 0.0);
            const float next_maximum = max(online_maximum, score);
            previous_scale = online_sum == 0.0 ? 0.0 : exp(online_maximum - next_maximum);
            current_weight = exp(score - next_maximum);
            online_sum = online_sum * previous_scale + current_weight;
            online_maximum = next_maximum;
        }
        barrier();

        if (lane < p.head_dimension)
        {
            numerator = numerator * previous_scale
                        + (sink_token
                               ? 0.0
                               : current_weight
                                     * buffer_ld1(
                                         value_data,
                                         value_base
                                             + token * p.head_dimension
                                             + lane));
        }
    }

    if (lane < p.head_dimension)
    {
        output_data[head * p.head_dimension + lane] = numerator / online_sum;
    }
}
)glsl";

static constexpr char attention_ring_append_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer source_key_blob
{
    sfp source_key_data[];
};
layout(binding = 1) readonly buffer source_value_blob
{
    sfp source_value_data[];
};
layout(binding = 2) buffer destination_key_blob
{
    sfp destination_key_data[];
};
layout(binding = 3) buffer destination_value_blob
{
    sfp destination_value_data[];
};

layout(push_constant) uniform parameter
{
    uint width;
    uint rows;
    uint channels;
    uint source_cstep;
    uint destination_cstep;
    uint capacity;
    uint destination_start;
}
p;

void main()
{
    const uint column = gl_GlobalInvocationID.x;
    const uint row = gl_GlobalInvocationID.y;
    const uint channel = gl_GlobalInvocationID.z;
    if (column >= p.width || row >= p.rows || channel >= p.channels)
        return;

    const uint source_index = channel * p.source_cstep + row * p.width + column;
    const uint slot = (p.destination_start + row) % p.capacity;
    const uint destination_index = channel * p.destination_cstep + slot * p.width + column;
    const uint duplicate_index = destination_index + p.capacity * p.width;
    const float key = buffer_ld1(source_key_data, source_index);
    const float value = buffer_ld1(source_value_data, source_index);
    buffer_st1(destination_key_data, destination_index, key);
    buffer_st1(destination_key_data, duplicate_index, key);
    buffer_st1(destination_value_data, destination_index, value);
    buffer_st1(destination_value_data, duplicate_index, value);
}
)glsl";

static constexpr char attention_ring_zero_shader[] = R"glsl(
#version 450

layout(binding = 0) buffer destination_key_blob
{
    sfp destination_key_data[];
};
layout(binding = 1) buffer destination_value_blob
{
    sfp destination_value_data[];
};

layout(push_constant) uniform parameter
{
    uint width;
    uint channels;
    uint destination_cstep;
    uint destination_row;
}
p;

void main()
{
    const uint column = gl_GlobalInvocationID.x;
    const uint channel = gl_GlobalInvocationID.z;
    if (column >= p.width || channel >= p.channels)
        return;

    const uint destination_index = channel * p.destination_cstep + p.destination_row * p.width + column;
    buffer_st1(destination_key_data, destination_index, 0.0);
    buffer_st1(destination_value_data, destination_index, 0.0);
}
)glsl";

static bool create_attention_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    const char* shader,
    int shader_size,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t storage_variant =
        vulkan_activation_storage_variant(option);
    const std::shared_ptr<const std::vector<uint32_t>> spirv =
        context->shader_binary(shader, shader_size, option, storage_variant);
    if (!spirv || spirv->empty())
        return false;

    destination = context->find_pipeline(shader, storage_variant);
    if (destination)
        return true;

    ncnn::VulkanDevice* vkdev = context->device();
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(vkdev));
    if (shader == attention_qkv_norm_rope_shader)
        pipeline->set_local_size_xyz(32, 1, 1);
    else if (shader == attention_output_gate_shader)
        pipeline->set_optimal_local_size_xyz(128, 1, 1);
    else if (shader == attention_qkv_rope_shader)
        pipeline->set_optimal_local_size_xyz(64, 1, 1);
    else if (shader == attention_decode_sdpa_shader)
        pipeline->set_local_size_xyz(128, 1, 1);
    else
        pipeline->set_optimal_local_size_xyz(8, 8, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(shader, storage_variant, destination);
    return true;
}

enum class AttentionQkvRopeFailureStage : uint8_t
{
    None,
    Pipeline,
    Shape,
    Source,
    Norm,
    Ring,
    Allocation,
};

static bool record_attention_qkv_rope(const ncnn::Pipeline* pipeline, const ncnn::VkMat& fused_qkv, const ncnn::VkMat& cosine, const ncnn::VkMat& sine,
                                      const NcnnVulkanAttentionConfig& config, size_t token_count, uint64_t position_offset, float rope_concentration,
                                      bool device_rope, size_t key_value_element_size,
                                      const ncnn::VkMat* direct_ring_key,
                                      const ncnn::VkMat* direct_ring_value, uint64_t ring_capacity, uint64_t destination_start, ncnn::VkMat& query,
                                      ncnn::VkMat& key, ncnn::VkMat& value, ncnn::VkCompute& command, ncnn::VkAllocator* allocator,
                                      AttentionQkvRopeFailureStage* failure_stage = nullptr)
{
    const auto fail = [failure_stage](AttentionQkvRopeFailureStage stage) {
        if (failure_stage)
            *failure_stage = stage;
        return false;
    };
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns) + static_cast<uint64_t>(key_value_columns) * 2;
    const uint64_t half_dimension = config.head_dimension / 2;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count) * half_dimension
                                   + static_cast<uint64_t>(config.kv_head_count) * half_dimension + key_value_columns);
    const bool direct_ring = direct_ring_key || direct_ring_value;
    const bool valid_rope_source = device_rope
                                       ? token_count != 0 && cosine.dims == 1 && cosine.w >= static_cast<int>(half_dimension)
                                             && position_offset <= std::numeric_limits<uint32_t>::max()
                                             && token_count - 1 <= std::numeric_limits<uint32_t>::max() - position_offset
                                       : cosine.dims == 2 && sine.dims == 2
                                             && cosine.w >= static_cast<int>(half_dimension)
                                             && sine.w >= static_cast<int>(half_dimension)
                                             && cosine.h >= static_cast<int>(token_count)
                                             && sine.h >= static_cast<int>(token_count);
    if (!pipeline || fused_qkv.empty() || cosine.empty() || sine.empty())
        return fail(AttentionQkvRopeFailureStage::Pipeline);
    if (fused_qkv.elempack != 1
        || fused_qkv.elemsize != sizeof(float)
        || (key_value_element_size != sizeof(float)
            && key_value_element_size != sizeof(uint16_t))
        || fused_qkv.dims != 2 || fused_qkv.w != static_cast<int>(total_columns)
        || fused_qkv.h != static_cast<int>(token_count))
        return fail(AttentionQkvRopeFailureStage::Shape);
    if (cosine.elempack != 1 || cosine.elemsize != sizeof(float)
        || sine.elempack != 1 || sine.elemsize != sizeof(float)
        || token_count == 0 || !valid_rope_source)
        return fail(AttentionQkvRopeFailureStage::Source);
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || total_columns > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || work_items > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return fail(AttentionQkvRopeFailureStage::Shape);
    if ((direct_ring_key == nullptr) != (direct_ring_value == nullptr)
        || (direct_ring
            && (ring_capacity == 0
                || ring_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                || destination_start >= ring_capacity
                || token_count > ring_capacity)))
        return fail(AttentionQkvRopeFailureStage::Ring);

    query.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.head_count), sizeof(float), 1, allocator);
    if (direct_ring)
    {
        key = *direct_ring_key;
        value = *direct_ring_value;
    }
    else
    {
        key.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.kv_head_count), key_value_element_size, 1, allocator);
        value.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.kv_head_count), key_value_element_size, 1,
                     allocator);
    }
    if (query.empty() || key.empty() || value.empty()
        || (direct_ring
            && (key.dims != 3 || value.dims != 3 || key.w != static_cast<int>(config.head_dimension) || value.w != key.w
                || key.h != static_cast<int>(ring_capacity * 2) || value.h != key.h || key.c != static_cast<int>(config.kv_head_count) || value.c != key.c
                || key.elemsize != key_value_element_size || value.elemsize != key_value_element_size || key.elempack != 1 || value.elempack != 1))
        || query.cstep > std::numeric_limits<uint32_t>::max() || key.cstep > std::numeric_limits<uint32_t>::max()
        || value.cstep > std::numeric_limits<uint32_t>::max())
    {
        return fail(AttentionQkvRopeFailureStage::Allocation);
    }

    const std::vector<ncnn::VkMat> bindings = {
        fused_qkv,
        cosine,
        sine,
        query,
        key,
        value,
    };
    std::vector<ncnn::vk_constant_type> constants(15);
    constants[0].u32 = static_cast<uint32_t>(total_columns);
    constants[1].u32 = query_columns;
    constants[2].u32 = key_value_columns;
    constants[3].u32 = config.head_dimension;
    constants[4].u32 = static_cast<uint32_t>(token_count);
    constants[5].u32 = static_cast<uint32_t>(query.cstep);
    constants[6].u32 = static_cast<uint32_t>(key.cstep);
    constants[7].u32 = static_cast<uint32_t>(value.cstep);
    constants[8].u32 = static_cast<uint32_t>(work_items);
    constants[9].u32 = direct_ring ? 1 : 0;
    constants[10].u32 = direct_ring ? static_cast<uint32_t>(ring_capacity) : 0;
    constants[11].u32 = direct_ring ? static_cast<uint32_t>(destination_start) : 0;
    constants[12].u32 = device_rope ? 1 : 0;
    constants[13].u32 = device_rope ? static_cast<uint32_t>(position_offset) : 0;
    constants[14].f = rope_concentration;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_qkv_norm_rope(
    const ncnn::Pipeline* pipeline,
    const ncnn::VkMat& fused_qkv,
    const ncnn::VkMat& cosine,
    const ncnn::VkMat& sine,
    const ncnn::VkMat& query_norm,
    const ncnn::VkMat& key_norm,
    const NcnnVulkanAttentionConfig& config,
    size_t token_count,
    uint64_t position_offset,
    float rope_concentration,
    bool device_rope,
    size_t key_value_element_size,
    const ncnn::VkMat* direct_ring_key,
    const ncnn::VkMat* direct_ring_value,
    uint64_t ring_capacity,
    uint64_t destination_start,
    ncnn::VkMat& query,
    ncnn::VkMat& key,
    ncnn::VkMat& value,
    ncnn::VkMat& gate,
    ncnn::VkCompute& command,
    ncnn::VkAllocator* allocator,
    AttentionQkvRopeFailureStage* failure_stage = nullptr)
{
    const auto fail = [failure_stage](AttentionQkvRopeFailureStage stage) {
        if (failure_stage)
            *failure_stage = stage;
        return false;
    };
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns)
                                   + static_cast<uint64_t>(key_value_columns) * 2
                                   + query_columns;
    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count)
                                   + config.kv_head_count);
    const bool direct_ring = direct_ring_key || direct_ring_value;
    const bool valid_rope_source = device_rope
                                       ? token_count != 0 && cosine.dims == 1 && cosine.w >= static_cast<int>(rope_dimension / 2)
                                             && position_offset <= std::numeric_limits<uint32_t>::max()
                                             && token_count - 1 <= std::numeric_limits<uint32_t>::max() - position_offset
                                       : cosine.dims == 2 && sine.dims == 2
                                             && cosine.w >= static_cast<int>(rope_dimension / 2)
                                             && sine.w >= static_cast<int>(rope_dimension / 2)
                                             && cosine.h >= static_cast<int>(token_count)
                                             && sine.h >= static_cast<int>(token_count);
    if (!pipeline || fused_qkv.empty() || cosine.empty() || sine.empty()
        || query_norm.empty() || key_norm.empty())
        return fail(AttentionQkvRopeFailureStage::Pipeline);
    if (token_count == 0
        || token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || total_columns > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || work_items > static_cast<uint64_t>(std::numeric_limits<int>::max() / 32)
        || config.head_count == 0 || config.kv_head_count == 0
        || config.head_count % config.kv_head_count != 0)
        return fail(AttentionQkvRopeFailureStage::Shape);
    if (fused_qkv.dims != 2 || fused_qkv.elempack != 1
        || fused_qkv.elemsize != sizeof(float)
        || (key_value_element_size != sizeof(float)
            && key_value_element_size != sizeof(uint16_t))
        || fused_qkv.w != static_cast<int>(total_columns)
        || fused_qkv.h != static_cast<int>(token_count))
        return fail(AttentionQkvRopeFailureStage::Shape);
    if (cosine.elempack != 1 || sine.elempack != 1
        || cosine.elemsize != sizeof(float)
        || sine.elemsize != sizeof(float)
        || (rope_dimension != 0
            && (rope_dimension > config.head_dimension
                || (rope_dimension & 1) != 0))
        || !valid_rope_source)
        return fail(AttentionQkvRopeFailureStage::Source);
    if (query_norm.dims != 1 || key_norm.dims != 1
        || query_norm.w != static_cast<int>(config.head_dimension)
        || key_norm.w != static_cast<int>(config.head_dimension)
        || query_norm.elempack != 1 || key_norm.elempack != 1
        || query_norm.elemsize != sizeof(float)
        || key_norm.elemsize != sizeof(float))
        return fail(AttentionQkvRopeFailureStage::Norm);
    if ((direct_ring_key == nullptr) != (direct_ring_value == nullptr)
        || (direct_ring
            && (ring_capacity == 0
                || ring_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                || destination_start >= ring_capacity
                || token_count > ring_capacity)))
        return fail(AttentionQkvRopeFailureStage::Ring);

    query.create(
        static_cast<int>(config.head_dimension),
        static_cast<int>(token_count),
        static_cast<int>(config.head_count),
        sizeof(float),
        1,
        allocator);
    gate.create(
        static_cast<int>(query_columns),
        static_cast<int>(token_count),
        sizeof(float),
        1,
        allocator);
    if (direct_ring)
    {
        key = *direct_ring_key;
        value = *direct_ring_value;
    }
    else
    {
        key.create(
            static_cast<int>(config.head_dimension),
            static_cast<int>(token_count),
            static_cast<int>(config.kv_head_count),
            key_value_element_size,
            1,
            allocator);
        value.create(
            static_cast<int>(config.head_dimension),
            static_cast<int>(token_count),
            static_cast<int>(config.kv_head_count),
            key_value_element_size,
            1,
            allocator);
    }
    if (query.empty() || key.empty() || value.empty() || gate.empty()
        || query.cstep > std::numeric_limits<uint32_t>::max()
        || key.cstep > std::numeric_limits<uint32_t>::max()
        || value.cstep > std::numeric_limits<uint32_t>::max()
        || (direct_ring
            && (key.dims != 3 || value.dims != 3
                || key.w != static_cast<int>(config.head_dimension)
                || value.w != key.w
                || key.h != static_cast<int>(ring_capacity * 2)
                || value.h != key.h
                || key.c != static_cast<int>(config.kv_head_count)
                || value.c != key.c
                || key.elemsize != key_value_element_size
                || value.elemsize != key_value_element_size
                || key.elempack != 1 || value.elempack != 1)))
    {
        return fail(AttentionQkvRopeFailureStage::Allocation);
    }

    const std::vector<ncnn::VkMat> bindings = {
        fused_qkv,
        cosine,
        sine,
        query_norm,
        key_norm,
        query,
        key,
        value,
        gate,
    };
    std::vector<ncnn::vk_constant_type> constants(19);
    constants[0].u32 = static_cast<uint32_t>(total_columns);
    constants[1].u32 = query_columns;
    constants[2].u32 = key_value_columns;
    constants[3].u32 = config.head_dimension;
    constants[4].u32 = config.head_count;
    constants[5].u32 = config.kv_head_count;
    constants[6].u32 = static_cast<uint32_t>(token_count);
    constants[7].u32 = static_cast<uint32_t>(query.cstep);
    constants[8].u32 = static_cast<uint32_t>(key.cstep);
    constants[9].u32 = static_cast<uint32_t>(value.cstep);
    constants[10].u32 = static_cast<uint32_t>(work_items);
    constants[11].u32 = direct_ring ? 1 : 0;
    constants[12].u32 = direct_ring ? static_cast<uint32_t>(ring_capacity) : 0;
    constants[13].u32 = direct_ring ? static_cast<uint32_t>(destination_start) : 0;
    constants[14].u32 = rope_dimension;
    constants[15].f = config.norm_epsilon;
    constants[16].u32 = device_rope ? 1 : 0;
    constants[17].u32 = device_rope ? static_cast<uint32_t>(position_offset) : 0;
    constants[18].f = rope_concentration;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items * 32);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static void record_attention_qkv_rope_failure(
    NcnnVulkanRuntimeState& runtime_state,
    AttentionQkvRopeFailureStage failure_stage) noexcept
{
    switch (failure_stage)
    {
    case AttentionQkvRopeFailureStage::Pipeline:
        ++runtime_state.attention_qkv_rope_pipeline_failures;
        break;
    case AttentionQkvRopeFailureStage::Shape:
        ++runtime_state.attention_qkv_rope_shape_failures;
        break;
    case AttentionQkvRopeFailureStage::Source:
        ++runtime_state.attention_qkv_rope_source_failures;
        break;
    case AttentionQkvRopeFailureStage::Norm:
        ++runtime_state.attention_qkv_rope_norm_failures;
        break;
    case AttentionQkvRopeFailureStage::Ring:
        ++runtime_state.attention_qkv_rope_ring_failures;
        break;
    case AttentionQkvRopeFailureStage::Allocation:
        ++runtime_state.attention_qkv_rope_allocation_failures;
        break;
    case AttentionQkvRopeFailureStage::None:
        break;
    }
}

static bool record_attention_output_gate(
    const ncnn::Pipeline* pipeline,
    ncnn::VkMat& attention,
    const ncnn::VkMat& gate,
    size_t token_count,
    uint32_t columns,
    ncnn::VkCompute& command)
{
    const uint64_t elements = static_cast<uint64_t>(token_count) * columns;
    if (!pipeline || attention.empty() || gate.empty()
        || elements == 0
        || elements > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || gate.dims != 2 || gate.w != static_cast<int>(columns)
        || gate.h != static_cast<int>(token_count)
        || gate.elempack != 1 || gate.elemsize != sizeof(float)
        || attention.elempack != 1
        || attention.elemsize != sizeof(float)
        || attention.total() * attention.elempack < elements)
    {
        return false;
    }
    const std::vector<ncnn::VkMat> bindings = {attention, gate};
    std::vector<ncnn::vk_constant_type> constants(1);
    constants[0].u32 = static_cast<uint32_t>(elements);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(elements);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

enum class DecodeSdpaMode
{
    Auto,
    Disabled,
    Forced
};

static DecodeSdpaMode decode_sdpa_mode(uint64_t optimization_flags) noexcept
{
    if (!runtime_optimization_enabled(optimization_flags, RuntimeOptimizationVulkanDecodeSdpa))
        return DecodeSdpaMode::Disabled;
    return runtime_optimization_enabled(
               optimization_flags,
               RuntimeOptimizationVulkanDecodeSdpaForce)
               ? DecodeSdpaMode::Forced
               : DecodeSdpaMode::Auto;
}

static bool qkv_ring_fusion_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationVulkanQkvRing);
}

static bool record_attention_decode_sdpa(const ncnn::Pipeline* pipeline, const ncnn::VkMat& query, const ncnn::VkMat& key, const ncnn::VkMat& value,
                                         const ncnn::VkMat& sinks, const NcnnVulkanAttentionConfig& config, uint64_t destination_count, ncnn::VkMat& output,
                                         ncnn::VkCompute& command, ncnn::VkAllocator* allocator)
{
    const uint64_t query_columns = static_cast<uint64_t>(config.head_count) * config.head_dimension;
    if (!pipeline || query.empty() || key.empty() || value.empty() || sinks.empty() || query.dims != 3 || query.w != static_cast<int>(config.head_dimension)
        || query.h != 1 || query.c != static_cast<int>(config.head_count) || key.dims != 3 || value.dims != 3
        || key.w != static_cast<int>(config.head_dimension) || value.w != key.w || key.h != static_cast<int>(destination_count) || value.h != key.h
        || key.c != static_cast<int>(config.kv_head_count) || value.c != key.c || sinks.dims != 1 || sinks.w != static_cast<int>(config.head_count)
        || query.elempack != 1 || key.elempack != 1 || value.elempack != 1 || sinks.elempack != 1
        || query.elemsize != sizeof(float)
        || (key.elemsize != sizeof(float)
            && key.elemsize != sizeof(uint16_t))
        || value.elemsize != key.elemsize
        || sinks.elemsize != sizeof(float) || config.head_dimension == 0
        || config.head_dimension > 128 || config.head_count == 0 || config.kv_head_count == 0 || config.head_count % config.kv_head_count != 0
        || destination_count == 0 || destination_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        || query_columns > static_cast<uint64_t>(std::numeric_limits<int>::max()) || query.cstep > std::numeric_limits<uint32_t>::max()
        || key.cstep > std::numeric_limits<uint32_t>::max() || value.cstep > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }

    output.create(static_cast<int>(query_columns), 1, sizeof(float), 1, allocator);
    if (output.empty())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        query,
        key,
        value,
        sinks,
        output,
    };
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = config.head_dimension;
    constants[1].u32 = static_cast<uint32_t>(destination_count);
    constants[2].u32 = config.head_count;
    constants[3].u32 = config.kv_head_count;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = has_flag(config.flags, NcnnAttentionSink) ? 1 : 0;
    constants[8].f = 1.0f / std::sqrt(static_cast<float>(config.head_dimension));
    ncnn::VkMat dispatcher;
    dispatcher.w = 128;
    dispatcher.h = 1;
    dispatcher.c = static_cast<int>(config.head_count);
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static uint64_t next_attention_ring_capacity(uint64_t current, uint64_t required)
{
    uint64_t capacity = current == 0 ? 16 : current;
    while (capacity < required)
    {
        if (capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / 4)
            return required;
        capacity *= 2;
    }
    return capacity;
}

static bool checked_multiply_u64(
    uint64_t left,
    uint64_t right,
    uint64_t& product)
{
    if (left != 0
        && right > std::numeric_limits<uint64_t>::max() / left)
    {
        return false;
    }
    product = left * right;
    return true;
}

static bool attention_promotion_within_budget(
    const NcnnVulkanContext& context,
    uint64_t ring_capacity,
    uint64_t kv_columns,
    size_t element_size,
    uint64_t transfer_bytes)
{
    uint64_t ring_elements = 0;
    uint64_t ring_bytes = 0;
    if (!checked_multiply_u64(
            ring_capacity,
            kv_columns,
            ring_elements)
        || !checked_multiply_u64(
            ring_elements,
            element_size * 4,
            ring_bytes)
        || transfer_bytes
               > std::numeric_limits<uint64_t>::max() - ring_bytes)
    {
        return false;
    }

    const ncnn::VulkanDevice* device = context.device();
    if (!device)
        return false;
    const uint64_t heap_budget_bytes =
        static_cast<uint64_t>(device->get_heap_budget()) * 1024 * 1024;
    // Promotion temporarily owns the FP32 staging payload and a double-written
    // key/value ring. Keep this opportunistic, model-neutral path to a small
    // per-layer share of the device budget; larger caches remain on the CPU.
    constexpr uint64_t maximum_per_layer_working_set =
        32ull * 1024 * 1024;
    const uint64_t admission_bytes = std::min(
        maximum_per_layer_working_set,
        heap_budget_bytes / 256);
    return admission_bytes != 0
           && ring_bytes + transfer_bytes <= admission_bytes;
}

class AttentionPromotionAttempt
{
public:
    explicit AttentionPromotionAttempt(CpuLayerCache* cache) noexcept
        : cache_(cache)
    {
    }

    ~AttentionPromotionAttempt()
    {
        if (cache_ && !completed_)
            cache_->vulkan_attention_promotion_disabled = true;
    }

    void complete() noexcept
    {
        completed_ = true;
    }

private:
    CpuLayerCache* cache_ = nullptr;
    bool completed_ = false;
};

static bool create_attention_ring_storage(
    NcnnVulkanAttentionCache& cache,
    uint32_t width,
    uint32_t channels,
    uint64_t capacity,
    size_t element_size,
    ncnn::VkAllocator* allocator)
{
    if (capacity == 0
        || capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / 2
        || (element_size != sizeof(float)
            && element_size != sizeof(uint16_t)))
        return false;
    cache.key.create(static_cast<int>(width), static_cast<int>(capacity * 2), static_cast<int>(channels), element_size, 1, allocator);
    cache.value.create(static_cast<int>(width), static_cast<int>(capacity * 2), static_cast<int>(channels), element_size, 1, allocator);
    return !cache.key.empty() && !cache.value.empty();
}

static ncnn::VkMat attention_ring_view(const ncnn::VkMat& storage, uint64_t first_slot, uint64_t rows)
{
#if NCNN_BATCH
    if (storage.empty() || rows == 0 || first_slot + rows > static_cast<uint64_t>(storage.h))
        return {};
    ncnn::VkMat view = storage;
    view.h = static_cast<int>(rows);
    view.offset += static_cast<size_t>(first_slot) * storage.w * storage.elemsize;
    view.n = 1;
    view.nstep = static_cast<size_t>(view.c - 1) * view.cstep + static_cast<size_t>(view.h) * view.w;
    return view;
#else
    (void)storage;
    (void)first_slot;
    (void)rows;
    return {};
#endif
}

static bool record_attention_ring_append(const ncnn::Pipeline* pipeline, const ncnn::VkMat& source_key, const ncnn::VkMat& source_value,
                                         ncnn::VkMat& destination_key, ncnn::VkMat& destination_value, uint64_t capacity, uint64_t destination_start,
                                         ncnn::VkCompute& command)
{
    if (!pipeline || source_key.empty() || source_value.empty() || destination_key.empty() || destination_value.empty() || source_key.dims != 3
        || source_key.elempack != 1
        || (source_key.elemsize != sizeof(float)
            && source_key.elemsize != sizeof(uint16_t))
        || source_key.elemsize != source_value.elemsize
        || source_key.w != source_value.w || source_key.h != source_value.h
        || source_key.c != source_value.c || destination_key.w != source_key.w || destination_key.c != source_key.c || destination_value.w != source_key.w
        || destination_value.c != source_key.c
        || destination_key.elemsize != source_key.elemsize
        || destination_value.elemsize != source_key.elemsize
        || capacity == 0 || destination_start >= capacity || source_key.cstep > std::numeric_limits<uint32_t>::max()
        || destination_key.cstep > std::numeric_limits<uint32_t>::max())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        source_key,
        source_value,
        destination_key,
        destination_value,
    };
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].u32 = static_cast<uint32_t>(source_key.w);
    constants[1].u32 = static_cast<uint32_t>(source_key.h);
    constants[2].u32 = static_cast<uint32_t>(source_key.c);
    constants[3].u32 = static_cast<uint32_t>(source_key.cstep);
    constants[4].u32 = static_cast<uint32_t>(destination_key.cstep);
    constants[5].u32 = static_cast<uint32_t>(capacity);
    constants[6].u32 = static_cast<uint32_t>(destination_start);
    ncnn::VkMat dispatcher;
    dispatcher.w = source_key.w;
    dispatcher.h = source_key.h;
    dispatcher.c = source_key.c;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_ring_zero(const ncnn::Pipeline* pipeline, ncnn::VkMat& destination_key, ncnn::VkMat& destination_value, uint64_t destination_row,
                                       ncnn::VkCompute& command)
{
    if (!pipeline || destination_key.empty() || destination_value.empty() || destination_key.w != destination_value.w
        || destination_key.c != destination_value.c || destination_key.cstep != destination_value.cstep
        || (destination_key.elemsize != sizeof(float)
            && destination_key.elemsize != sizeof(uint16_t))
        || destination_value.elemsize != destination_key.elemsize
        || destination_row >= static_cast<uint64_t>(destination_key.h) || destination_key.cstep > std::numeric_limits<uint32_t>::max())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        destination_key,
        destination_value,
    };
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = static_cast<uint32_t>(destination_key.w);
    constants[1].u32 = static_cast<uint32_t>(destination_key.c);
    constants[2].u32 = static_cast<uint32_t>(destination_key.cstep);
    constants[3].u32 = static_cast<uint32_t>(destination_row);
    ncnn::VkMat dispatcher;
    dispatcher.w = destination_key.w;
    dispatcher.h = 1;
    dispatcher.c = destination_key.c;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}
#endif

#if NCNN_MOE_WITH_VULKAN
#endif

#if NCNN_MOE_WITH_VULKAN
static bool create_vulkan_layer(int type, const ncnn::ParamDict& parameters, ncnn::VulkanDevice* device, const ncnn::Option& option,
                                std::vector<ncnn::Layer*>& layers, ncnn::Layer*& destination)
{
    ncnn::Layer* layer = ncnn::create_layer_vulkan(type);
    if (!layer)
        return false;
    layer->vkdev = device;
    if (layer->load_param(parameters) != 0 || layer->create_pipeline(option) != 0)
    {
        delete layer;
        return false;
    }
    layers.push_back(layer);
    destination = layer;
    return true;
}
#endif

std::shared_ptr<NcnnVulkanAttentionOperator> NcnnVulkanAttentionOperator::create(const TensorData& norm_weight, const TensorData* sinks,
                                                                                 std::shared_ptr<NcnnLinearOperator> fused_qkv,
                                                                                 std::shared_ptr<NcnnLinearOperator> output_projection,
                                                                                 const NcnnVulkanAttentionConfig& config)
{
#if NCNN_MOE_WITH_VULKAN
#if !NCNN_BATCH
    // Wrapped KV rings require VkMat offset views.
    return {};
#endif
    if (!fused_qkv || !output_projection || !fused_qkv->uses_vulkan() || !output_projection->uses_vulkan() || config.hidden_size == 0 || config.head_count == 0
        || config.kv_head_count == 0 || config.head_dimension == 0 || config.head_dimension % 2 != 0 || config.head_count % config.kv_head_count != 0
        || config.activation_dtype != config.kv_cache_dtype || (!has_flag(config.flags, NcnnAttentionSink) && config.sliding_window > 0)
        || norm_weight.shape != std::vector<uint32_t>{config.hidden_size}
        || (has_flag(config.flags, NcnnAttentionSink) && (!sinks || sinks->shape != std::vector<uint32_t>{config.head_count})))
        return {};

    const NcnnLinearOperator::Implementation& fused_implementation = *fused_qkv->implementation_;
    const NcnnLinearOperator::Implementation& output_implementation = *output_projection->implementation_;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    if (!fused_implementation.layer || !output_implementation.layer || fused_implementation.input_columns != config.hidden_size
        || fused_implementation.output_columns != query_columns + 2 * key_value_columns || output_implementation.input_columns != query_columns
        || output_implementation.output_columns != config.hidden_size || fused_implementation.vulkan_context != output_implementation.vulkan_context)
        return {};

    std::shared_ptr<NcnnVulkanAttentionOperator> attention(new NcnnVulkanAttentionOperator);
    Implementation& implementation = *attention->implementation_;
    implementation.fused_qkv = std::move(fused_qkv);
    implementation.output_projection = std::move(output_projection);
    implementation.config = config;
    if (has_flag(config.flags, NcnnAttentionSink) && !tensor_to_float_vector(*sinks, implementation.sinks))
        return {};
    implementation.vulkan_context = fused_implementation.vulkan_context;
    implementation.option = fused_implementation.option;
    // SDPA uses unpacked [head, token, dimension] tensors.
    implementation.option.use_packing_layout = false;
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    implementation.kv_option = implementation.option;

    const uint32_t rotary_dimension = config.rope_head_dimension == 0
                                          ? config.head_dimension
                                          : config.rope_head_dimension;
    const uint32_t half_dimension = config.head_dimension / 2;
    const uint32_t rotary_half_dimension = rotary_dimension / 2;
    implementation.rope_inverse_frequencies.resize(half_dimension);
    float rope_low = 0.0f;
    float rope_high = 0.0f;
    if (config.rope_scaling_factor > 1.0f)
    {
        implementation.rope_concentration = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(rotary_half_dimension);
        rope_low = half * std::log(static_cast<float>(config.initial_context_length) / (config.rope_ntk_beta * 2.0f * std::acos(-1.0f)))
                   / std::log(config.rope_theta);
        rope_high = half * std::log(static_cast<float>(config.initial_context_length) / (config.rope_ntk_alpha * 2.0f * std::acos(-1.0f)))
                    / std::log(config.rope_theta);
    }
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float frequency = std::pow(config.rope_theta, static_cast<float>(2 * index) / static_cast<float>(config.head_dimension));
        float inverse_frequency = 1.0f / frequency;
        if (config.rope_scaling_factor > 1.0f)
        {
            const float ramp = std::clamp((static_cast<float>(index) - rope_low) / (rope_high - rope_low), 0.0f, 1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation = 1.0f / (config.rope_scaling_factor * frequency);
            inverse_frequency = interpolation * (1.0f - mask) + inverse_frequency * mask;
        }
        implementation.rope_inverse_frequencies[index] = inverse_frequency;
    }

    std::vector<float> norm_values;
    if (!tensor_to_float_vector(norm_weight, norm_values))
        return {};
    for (float& value : norm_values)
        value += config.norm_weight_offset;
    ncnn::Layer* norm = ncnn::create_layer_vulkan(ncnn::LayerType::RMSNorm);
    if (!norm)
        return {};
    norm->vkdev = vkdev;
    ncnn::ParamDict norm_parameters;
    norm_parameters.set(0, static_cast<int>(config.hidden_size));
    norm_parameters.set(1, config.norm_epsilon);
    norm_parameters.set(2, 1);
    ncnn::Mat norm_model[1] = {ncnn::Mat(static_cast<int>(norm_values.size()), norm_values.data(), sizeof(float))};
    if (norm->load_param(norm_parameters) != 0 || norm->load_model(ncnn::ModelBinFromMatArray(norm_model)) != 0
        || norm->create_pipeline(implementation.option) != 0)
    {
        delete norm;
        return {};
    }
    implementation.layers.push_back(norm);
    implementation.norm = norm;

    ncnn::ParamDict slice_parameters;
    ncnn::Mat slice_sizes(3, sizeof(int));
    int* slice_values = static_cast<int*>(slice_sizes.data);
    slice_values[0] = static_cast<int>(query_columns);
    slice_values[1] = static_cast<int>(key_value_columns);
    slice_values[2] = static_cast<int>(key_value_columns);
    slice_parameters.set(0, slice_sizes);
    slice_parameters.set(1, 1);
    if (!create_vulkan_layer(ncnn::LayerType::Slice, slice_parameters, vkdev, implementation.option, implementation.layers, implementation.slice_qkv))
        return {};

    ncnn::ParamDict reshape_query_parameters;
    reshape_query_parameters.set(0, static_cast<int>(config.head_dimension));
    reshape_query_parameters.set(1, static_cast<int>(config.head_count));
    reshape_query_parameters.set(2, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_query_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.reshape_query))
        return {};

    ncnn::ParamDict reshape_key_value_parameters;
    reshape_key_value_parameters.set(0, static_cast<int>(config.head_dimension));
    reshape_key_value_parameters.set(1, static_cast<int>(config.kv_head_count));
    reshape_key_value_parameters.set(2, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_key_value_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.reshape_key_value))
        return {};

    ncnn::ParamDict permute_parameters;
    permute_parameters.set(0, 2);
    if (!create_vulkan_layer(ncnn::LayerType::Permute, permute_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.permute_heads_tokens))
        return {};

    ncnn::ParamDict rotary_parameters;
    rotary_parameters.set(0, 0);
    if (!create_vulkan_layer(ncnn::LayerType::RotaryEmbed, rotary_parameters, vkdev, implementation.option, implementation.layers, implementation.rotary))
        return {};

    ncnn::ParamDict sdpa_parameters;
    sdpa_parameters.set(5, 1);
    sdpa_parameters.set(6, 1.0f / std::sqrt(static_cast<float>(config.head_dimension)));
    sdpa_parameters.set(7, 0);
    if (!create_vulkan_layer(ncnn::LayerType::SDPA, sdpa_parameters, vkdev, implementation.option, implementation.layers, implementation.sdpa))
        return {};

    ncnn::ParamDict reshape_attention_parameters;
    reshape_attention_parameters.set(0, static_cast<int>(query_columns));
    reshape_attention_parameters.set(1, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_attention_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.reshape_attention))
        return {};

    ncnn::ParamDict add_parameters;
    add_parameters.set(0, 0);
    if (!create_vulkan_layer(ncnn::LayerType::BinaryOp, add_parameters, vkdev, implementation.option, implementation.layers, implementation.add))
        return {};

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    if (!create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_qkv_rope_shader, static_cast<int>(sizeof(attention_qkv_rope_shader) - 1),
                                   implementation.qkv_rope_pipeline)
        || !create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_decode_sdpa_shader, static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1),
                                      implementation.decode_sdpa_pipeline)
        || !create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_ring_append_shader, static_cast<int>(sizeof(attention_ring_append_shader) - 1),
                                      implementation.ring_append_pipeline)
        || !create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_ring_zero_shader, static_cast<int>(sizeof(attention_ring_zero_shader) - 1),
                                      implementation.ring_zero_pipeline))
        return {};
    ncnn::VkTransfer command(vkdev);
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    ncnn::Mat rope_inverse_model(
        static_cast<int>(implementation.rope_inverse_frequencies.size()),
        sizeof(float));
    ncnn::Mat sink_model(static_cast<int>(config.head_count), sizeof(float));
    if (rope_inverse_model.empty() || sink_model.empty())
        return {};
    std::copy(
        implementation.rope_inverse_frequencies.begin(),
        implementation.rope_inverse_frequencies.end(),
        static_cast<float*>(rope_inverse_model.data));
    float* sink_values = static_cast<float*>(sink_model.data);
    std::fill_n(sink_values, config.head_count, 0.0f);
    if (has_flag(config.flags, NcnnAttentionSink))
    {
        std::copy(implementation.sinks.begin(), implementation.sinks.end(), sink_values);
    }
    command.record_upload(
        rope_inverse_model,
        implementation.rope_inverse_frequencies_gpu,
        upload_option);
    command.record_upload(sink_model, implementation.attention_sinks, upload_option);
    if (implementation.norm->upload_model(command, upload_option) != 0
        || implementation.rope_inverse_frequencies_gpu.empty()
        || implementation.attention_sinks.empty()
        || command.submit_and_wait() != 0)
        return {};
    return attention;
#else
    (void)norm_weight;
    (void)sinks;
    (void)fused_qkv;
    (void)output_projection;
    (void)config;
    return {};
#endif
}

std::shared_ptr<NcnnVulkanAttentionOperator>
NcnnVulkanAttentionOperator::create_with_query_key_norm_and_gate(
    const TensorData& norm_weight,
    const TensorData& query_norm_weight,
    const TensorData& key_norm_weight,
    const TensorData* sinks,
    std::shared_ptr<NcnnVulkanBfloat16Operator> fused_qkv_gate,
    std::shared_ptr<NcnnVulkanBfloat16Operator> output_projection,
    const NcnnVulkanAttentionConfig& config)
{
#if NCNN_MOE_WITH_VULKAN
    const auto fail = [](const char* reason) {
        (void)reason;
        return std::shared_ptr<NcnnVulkanAttentionOperator>();
    };
#if !NCNN_BATCH
    return fail("NCNN_BATCH is disabled");
#endif
    if (!fused_qkv_gate || !output_projection
        || config.hidden_size == 0 || config.head_count == 0
        || config.kv_head_count == 0 || config.head_dimension == 0
        || (config.head_dimension & 1) != 0
        || config.head_count % config.kv_head_count != 0
        || config.activation_dtype != config.kv_cache_dtype
        || !has_flag(config.flags, NcnnAttentionQueryKeyNorm)
        || !has_flag(config.flags, NcnnAttentionOutputGate)
        || (!has_flag(config.flags, NcnnAttentionSink)
            && config.sliding_window > 0)
        || norm_weight.shape != std::vector<uint32_t>{config.hidden_size}
        || query_norm_weight.shape
               != std::vector<uint32_t>{config.head_dimension}
        || key_norm_weight.shape
               != std::vector<uint32_t>{config.head_dimension}
        || (has_flag(config.flags, NcnnAttentionSink)
            && (!sinks
                || sinks->shape
                       != std::vector<uint32_t>{config.head_count})))
    {
        return fail("input validation");
    }

    const NcnnVulkanBfloat16Operator::Implementation& fused =
        *fused_qkv_gate->implementation_;
    const NcnnVulkanBfloat16Operator::Implementation& projection =
        *output_projection->implementation_;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns =
        config.kv_head_count * config.head_dimension;
    const uint64_t expected_fused_columns =
        static_cast<uint64_t>(query_columns)
        + static_cast<uint64_t>(key_value_columns) * 2
        + query_columns;
    if (!fused.pipeline || !projection.pipeline
        || fused.input_columns != config.hidden_size
        || fused.output_columns != expected_fused_columns
        || projection.input_columns != query_columns
        || projection.output_columns != config.hidden_size
        || fused.vulkan_context != projection.vulkan_context)
    {
        return fail("projection operator validation");
    }

    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    if (rope_dimension > config.head_dimension
        || (rope_dimension & 1) != 0)
    {
        return fail("rotary dimension validation");
    }

    std::shared_ptr<NcnnVulkanAttentionOperator> attention(
        new NcnnVulkanAttentionOperator);
    Implementation& implementation = *attention->implementation_;
    implementation.fused_qkv_gate = std::move(fused_qkv_gate);
    implementation.output_projection_bfloat16 = std::move(output_projection);
    implementation.config = config;
    if (has_flag(config.flags, NcnnAttentionSink)
        && !tensor_to_float_vector(*sinks, implementation.sinks))
    {
        return fail("create Vulkan Permute layer");
    }
    implementation.vulkan_context = fused.vulkan_context;
    implementation.option = fused.option;
    implementation.option.use_packing_layout = false;
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    implementation.kv_option = implementation.option;

    const uint32_t rotary_dimension = config.rope_head_dimension == 0
                                          ? config.head_dimension
                                          : config.rope_head_dimension;
    const uint32_t half_dimension = config.head_dimension / 2;
    const uint32_t rotary_half_dimension = rotary_dimension / 2;
    implementation.rope_inverse_frequencies.resize(half_dimension);
    float rope_low = 0.0f;
    float rope_high = 0.0f;
    if (config.rope_scaling_factor > 1.0f)
    {
        implementation.rope_concentration =
            0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(rotary_half_dimension);
        rope_low = half
                   * std::log(static_cast<float>(config.initial_context_length)
                              / (config.rope_ntk_beta * 2.0f
                                 * std::acos(-1.0f)))
                   / std::log(config.rope_theta);
        rope_high = half
                    * std::log(static_cast<float>(config.initial_context_length)
                               / (config.rope_ntk_alpha * 2.0f
                                  * std::acos(-1.0f)))
                    / std::log(config.rope_theta);
    }
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const uint32_t frequency_dimension = index < rotary_half_dimension
                                                 ? rotary_dimension
                                                 : config.head_dimension;
        const float frequency = std::pow(
            config.rope_theta,
            static_cast<float>(2 * index)
                / static_cast<float>(frequency_dimension));
        float inverse_frequency = 1.0f / frequency;
        if (config.rope_scaling_factor > 1.0f)
        {
            const float ramp = std::clamp(
                (static_cast<float>(index) - rope_low)
                    / (rope_high - rope_low),
                0.0f,
                1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation =
                1.0f / (config.rope_scaling_factor * frequency);
            inverse_frequency =
                interpolation * (1.0f - mask) + inverse_frequency * mask;
        }
        implementation.rope_inverse_frequencies[index] = inverse_frequency;
    }

    std::vector<float> norm_values;
    if (!tensor_to_float_vector(norm_weight, norm_values))
        return {};
    for (float& value : norm_values)
        value += config.norm_weight_offset;
    ncnn::Layer* norm = ncnn::create_layer_vulkan(ncnn::LayerType::RMSNorm);
    if (!norm)
        return {};
    norm->vkdev = vkdev;
    ncnn::ParamDict norm_parameters;
    norm_parameters.set(0, static_cast<int>(config.hidden_size));
    norm_parameters.set(1, config.norm_epsilon);
    norm_parameters.set(2, 1);
    ncnn::Mat norm_model[1] = {
        ncnn::Mat(static_cast<int>(norm_values.size()), norm_values.data(), sizeof(float))};
    if (norm->load_param(norm_parameters) != 0
        || norm->load_model(ncnn::ModelBinFromMatArray(norm_model)) != 0
        || norm->create_pipeline(implementation.option) != 0)
    {
        delete norm;
        return {};
    }
    implementation.layers.push_back(norm);
    implementation.norm = norm;

    ncnn::ParamDict permute_parameters;
    permute_parameters.set(0, 2);
    if (!create_vulkan_layer(
            ncnn::LayerType::Permute,
            permute_parameters,
            vkdev,
            implementation.kv_option,
            implementation.layers,
            implementation.permute_heads_tokens))
    {
        return fail("create Vulkan SDPA layer");
    }

    ncnn::ParamDict sdpa_parameters;
    sdpa_parameters.set(5, 1);
    sdpa_parameters.set(
        6, 1.0f / std::sqrt(static_cast<float>(config.head_dimension)));
    sdpa_parameters.set(7, 0);
    if (!create_vulkan_layer(
            ncnn::LayerType::SDPA,
            sdpa_parameters,
            vkdev,
            implementation.option,
            implementation.layers,
            implementation.sdpa))
    {
        return fail("create Vulkan Attention reshape layer");
    }

    ncnn::ParamDict reshape_attention_parameters;
    reshape_attention_parameters.set(0, static_cast<int>(query_columns));
    reshape_attention_parameters.set(1, -1);
    if (!create_vulkan_layer(
            ncnn::LayerType::Reshape,
            reshape_attention_parameters,
            vkdev,
            implementation.kv_option,
            implementation.layers,
            implementation.reshape_attention))
    {
        return fail("create Vulkan residual add layer");
    }

    ncnn::ParamDict add_parameters;
    add_parameters.set(0, 0);
    if (!create_vulkan_layer(
            ncnn::LayerType::BinaryOp,
            add_parameters,
            vkdev,
            implementation.kv_option,
            implementation.layers,
            implementation.add))
    {
        return fail("create QKV/norm/rope attention pipelines");
    }

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    implementation.weight_staging_allocator.reset(
        new ncnn::VkWeightStagingAllocator(vkdev));
    const std::lock_guard<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    if (!create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_qkv_norm_rope_shader,
            static_cast<int>(sizeof(attention_qkv_norm_rope_shader) - 1),
            implementation.qkv_norm_rope_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.option,
            attention_output_gate_shader,
            static_cast<int>(sizeof(attention_output_gate_shader) - 1),
            implementation.output_gate_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_decode_sdpa_shader,
            static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1),
            implementation.decode_sdpa_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_ring_append_shader,
            static_cast<int>(sizeof(attention_ring_append_shader) - 1),
            implementation.ring_append_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_ring_zero_shader,
            static_cast<int>(sizeof(attention_ring_zero_shader) - 1),
            implementation.ring_zero_pipeline))
    {
        return fail("convert query/key norm weights");
    }

    ncnn::Mat query_norm_model;
    ncnn::Mat key_norm_model;
    if (!prepare_float_tensor_upload(query_norm_weight, query_norm_model)
        || !prepare_float_tensor_upload(key_norm_weight, key_norm_model))
    {
        return {};
    }
    if (config.norm_weight_offset != 0.0f)
    {
        float* query_values = static_cast<float*>(query_norm_model.data);
        float* key_values = static_cast<float*>(key_norm_model.data);
        for (uint32_t index = 0; index < config.head_dimension; ++index)
        {
            query_values[index] += config.norm_weight_offset;
            key_values[index] += config.norm_weight_offset;
        }
    }
    ncnn::Mat sink_model(static_cast<int>(config.head_count), sizeof(float));
    if (sink_model.empty())
        return fail("allocate attention sink weights");
    float* sink_values = static_cast<float*>(sink_model.data);
    std::fill_n(sink_values, config.head_count, 0.0f);
    if (has_flag(config.flags, NcnnAttentionSink))
    {
        std::copy(implementation.sinks.begin(), implementation.sinks.end(), sink_values);
    }
    ncnn::VkTransfer command(vkdev);
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator =
        implementation.weight_staging_allocator.get();
    ncnn::Mat rope_inverse_model(
        static_cast<int>(implementation.rope_inverse_frequencies.size()),
        sizeof(float));
    if (rope_inverse_model.empty())
        return fail("allocate RoPE inverse frequencies");
    std::copy(
        implementation.rope_inverse_frequencies.begin(),
        implementation.rope_inverse_frequencies.end(),
        static_cast<float*>(rope_inverse_model.data));
    command.record_upload(
        query_norm_model,
        implementation.query_norm_weight,
        upload_option);
    command.record_upload(
        key_norm_model,
        implementation.key_norm_weight,
        upload_option);
    command.record_upload(
        rope_inverse_model,
        implementation.rope_inverse_frequencies_gpu,
        upload_option);
    command.record_upload(sink_model, implementation.attention_sinks, upload_option);
    if (implementation.norm->upload_model(command, upload_option) != 0
        || implementation.query_norm_weight.empty()
        || implementation.key_norm_weight.empty()
        || implementation.rope_inverse_frequencies_gpu.empty()
        || implementation.attention_sinks.empty()
        || command.submit_and_wait() != 0)
    {
        return fail("upload QKV norm attention weights");
    }
    implementation.weight_staging_allocator.reset();
    return attention;
#else
    (void)norm_weight;
    (void)query_norm_weight;
    (void)key_norm_weight;
    (void)sinks;
    (void)fused_qkv_gate;
    (void)output_projection;
    (void)config;
    return {};
#endif
}

bool NcnnVulkanAttentionOperator::forward(uint64_t position_offset, CpuLayerCache& cache, const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    const NcnnVulkanAttentionConfig& config = implementation.config;
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();
    const size_t activation_element_size =
        vulkan_activation_element_size(implementation.kv_option);
    const bool low_precision_kv =
        vulkan_activation_storage_variant(implementation.kv_option) != 0;
    const bool has_device_cache = cache.vulkan_attention_cache != nullptr;
    const bool promote_host_cache =
        cache.token_count != 0 && !has_device_cache;
    if (!vulkan_attention_enabled(config.optimization_flags)
        || cache.vulkan_attention_state_unknown
        || input.rows() == 0 || input.columns() != config.hidden_size || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || (cache.dtype != config.kv_cache_dtype && cache.token_count != 0)
        || (cache.token_count != 0
            && (cache.capacity_tokens == 0
                || cache.first_slot >= cache.capacity_tokens
                || cache.token_count > cache.capacity_tokens))
        || (cache.transaction.active && !has_device_cache)
        || (promote_host_cache
            && (cache.vulkan_attention_promotion_disabled
                || !vulkan_attention_kv_promotion_enabled(config.optimization_flags)))
        || (has_device_cache
            && (cache.vulkan_attention_cache->key.empty()
                || cache.vulkan_attention_cache->value.empty()
                || cache.vulkan_attention_cache->key.dims != 3
                || cache.vulkan_attention_cache->value.dims != 3
                || cache.vulkan_attention_cache->key.w
                       != static_cast<int>(config.head_dimension)
                || cache.vulkan_attention_cache->value.w
                       != static_cast<int>(config.head_dimension)
                || cache.vulkan_attention_cache->key.c
                       != static_cast<int>(config.kv_head_count)
                || cache.vulkan_attention_cache->value.c
                       != static_cast<int>(config.kv_head_count)
                || static_cast<uint64_t>(
                       cache.vulkan_attention_cache->key.h)
                       != cache.capacity_tokens * 2
                || static_cast<uint64_t>(
                       cache.vulkan_attention_cache->value.h)
                       != cache.capacity_tokens * 2
                || cache.vulkan_attention_cache->key.elemsize
                       != activation_element_size
                || cache.vulkan_attention_cache->value.elemsize
                       != activation_element_size
                || cache.vulkan_attention_cache->key.elempack != 1
                || cache.vulkan_attention_cache->value.elempack != 1)))
    {
        ++runtime_state.attention_precondition_failures;
        return false;
    }

    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16 && implementation.option.use_bf16_storage;
    const bool device_rope = vulkan_attention_device_rope_enabled(config.optimization_flags)
                             && !implementation.rope_inverse_frequencies_gpu.empty()
                             && position_offset <= std::numeric_limits<uint32_t>::max()
                             && input.rows() - 1 <= std::numeric_limits<uint32_t>::max() - position_offset;
    const uint64_t query_columns_u64 =
        static_cast<uint64_t>(config.head_count)
        * config.head_dimension;
    const uint64_t kv_columns =
        static_cast<uint64_t>(config.kv_head_count)
        * config.head_dimension;
    if (query_columns_u64 > std::numeric_limits<uint32_t>::max()
        || kv_columns > std::numeric_limits<uint32_t>::max())
    {
        {
            ++runtime_state.attention_precondition_failures;
            return false;
        }
    }
    const uint32_t query_columns =
        static_cast<uint32_t>(query_columns_u64);
    uint64_t promotion_elements = 0;
    uint64_t promotion_transfer_bytes = 0;
    if (promote_host_cache
        && (!checked_multiply_u64(
                cache.token_count,
                kv_columns,
                promotion_elements)
            || !checked_multiply_u64(
                promotion_elements,
                sizeof(float) * 2,
                promotion_transfer_bytes)))
    {
        cache.vulkan_attention_promotion_disabled = true;
        {
            ++runtime_state.attention_precondition_failures;
            return false;
        }
    }
    const uint64_t actual_token_count = cache.token_count + input.rows();
    const uint64_t sink_token_count = has_flag(config.flags, NcnnAttentionSink) ? 1 : 0;
    const uint64_t destination_count = actual_token_count + sink_token_count;
    if (actual_token_count < cache.token_count || destination_count < actual_token_count
        || destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            ++runtime_state.attention_precondition_failures;
            return false;
        }

    if (promote_host_cache
        && !attention_promotion_within_budget(
            *implementation.vulkan_context,
            next_attention_ring_capacity(0, actual_token_count),
            kv_columns,
            activation_element_size,
            promotion_transfer_bytes))
    {
        cache.vulkan_attention_promotion_disabled = true;
        ++runtime_state.attention_cache_failures;
        return false;
    }
    AttentionPromotionAttempt promotion_attempt(
        promote_host_cache ? &cache : nullptr);

    const DecodeSdpaMode selected_decode_sdpa_mode = decode_sdpa_mode(config.optimization_flags);
    const bool adaptive_decode_sdpa = input.rows() == 1
                                      && config.head_dimension <= 128
                                      && destination_count <= 4096
                                      && selected_decode_sdpa_mode
                                             == DecodeSdpaMode::Auto;
    const bool try_decode_sdpa = input.rows() == 1
                                 && selected_decode_sdpa_mode
                                        != DecodeSdpaMode::Disabled
                                 && (selected_decode_sdpa_mode
                                         == DecodeSdpaMode::Forced
                                     || (adaptive_decode_sdpa
                                         && implementation.choose_decode_sdpa(
                                                config.head_dimension,
                                                config.head_count,
                                                config.kv_head_count,
                                                destination_count)));

    const bool query_key_norm_and_gate =
        implementation.fused_qkv_gate != nullptr;
    const NcnnLinearOperator::Implementation* fused =
        implementation.fused_qkv
            ? implementation.fused_qkv->implementation_.get()
            : nullptr;
    const NcnnVulkanBfloat16Operator::Implementation* fused_gate =
        implementation.fused_qkv_gate
            ? implementation.fused_qkv_gate->implementation_.get()
            : nullptr;
    const NcnnLinearOperator::Implementation* projection =
        implementation.output_projection
            ? implementation.output_projection->implementation_.get()
            : nullptr;
    const NcnnVulkanBfloat16Operator::Implementation* projection_bfloat16 =
        implementation.output_projection_bfloat16
            ? implementation.output_projection_bfloat16->implementation_.get()
            : nullptr;
    NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = input.rows() == 1
                                   && vulkan_activation_storage_variant(implementation.option) == 0
                                   && direct_host_input_enabled(
                                       *implementation.vulkan_context,
                                       static_cast<size_t>(config.hidden_size)
                                           * sizeof(float),
                                       input.dtype());
     if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
         || (!device_rope
             && !fill_rope_staging_pair(transfer_slot.rope_cosine, transfer_slot.rope_sine, input.rows(), position_offset,
                                        implementation.rope_inverse_frequencies, implementation.rope_concentration, bfloat16_storage,
                                        transfer_slot.staging_allocator, runtime_state))
        || (!try_decode_sdpa
            && !fill_attention_mask_staging(transfer_slot.attention_mask, input.rows(), destination_count, position_offset, cache, config, implementation.sinks,
                                             bfloat16_storage, transfer_slot.staging_allocator, runtime_state))
        || (promote_host_cache
            && !fill_attention_cache_promotion_staging(
                transfer_slot.attention_cache_key,
                transfer_slot.attention_cache_value,
                cache,
                config,
                 transfer_slot.staging_allocator, runtime_state))
         || !prepare_staging_batch(transfer_slot.download, input.rows(), config.hidden_size, transfer_slot.staging_allocator, runtime_state))
    {
        ++runtime_state.attention_staging_failures;
        return false;
    }

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            ++runtime_state.attention_cache_failures;
            return false;
        }
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    ncnn::VkMat cosine_gpu = device_rope
                                 ? implementation.rope_inverse_frequencies_gpu
                                 : ncnn::VkMat();
    ncnn::VkMat sine_gpu = device_rope
                               ? implementation.rope_inverse_frequencies_gpu
                               : ncnn::VkMat();
    ncnn::VkMat mask_gpu;
    ncnn::VkMat promoted_key_gpu;
    ncnn::VkMat promoted_value_gpu;
    bool mask_uploaded = false;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), input_gpu, command, vkdev, implementation.option, input.dtype()))
    {
        ++runtime_state.attention_staging_failures;
        return false;
    }
    if (!device_rope
        && (!record_mapped_upload(transfer_slot.rope_cosine, cosine_gpu, command, implementation.option)
            || !record_mapped_upload(transfer_slot.rope_sine, sine_gpu, command, implementation.option)))
    {
        ++runtime_state.attention_staging_failures;
        return false;
    }
    if (promote_host_cache
        && (!record_mapped_activation_upload(
                transfer_slot.attention_cache_key,
                promoted_key_gpu,
                command,
                vkdev,
                implementation.kv_option)
            || !record_mapped_activation_upload(
                transfer_slot.attention_cache_value,
                promoted_value_gpu,
                command,
                vkdev,
                implementation.kv_option)))
    {
        ++runtime_state.attention_staging_failures;
        return false;
    }
    if (!try_decode_sdpa)
    {
        if (!record_mapped_upload(transfer_slot.attention_mask, mask_gpu, command, implementation.option))
        {
            ++runtime_state.attention_staging_failures;
            return false;
        }
        mask_uploaded = true;
    }
    ncnn::VkMat normalized_gpu;
    if (implementation.norm->forward(input_gpu, normalized_gpu, command, implementation.option) != 0)
    {
        ++runtime_state.attention_norm_failures;
        return false;
    }

    ncnn::VkMat fused_gpu;
    if (query_key_norm_and_gate)
    {
        ncnn::VkMat normalized_unpacked = normalized_gpu;
        if (normalized_unpacked.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                normalized_unpacked,
                unpacked,
                1,
                command,
                implementation.option);
            normalized_unpacked = unpacked;
        }
        if (!fused_gate || !fused_gate->pipeline
            || normalized_unpacked.empty()
            || normalized_unpacked.dims != 2
            || normalized_unpacked.w != static_cast<int>(fused_gate->input_columns)
            || normalized_unpacked.h != static_cast<int>(input.rows())
            || normalized_unpacked.elempack != 1
            || normalized_unpacked.elemsize != sizeof(float))
        {
            ++runtime_state.attention_qkv_failures;
            return false;
        }
        fused_gpu.create(
            static_cast<int>(fused_gate->output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            implementation.vulkan_context->blob_allocator());
        if (fused_gpu.empty())
        {
            ++runtime_state.attention_qkv_failures;
            return false;
        }
        const std::vector<ncnn::VkMat> bindings = {
            normalized_unpacked,
            fused_gate->packed,
            fused_gate->bias,
            fused_gpu};
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].u32 = fused_gate->input_columns;
        constants[1].u32 = fused_gate->output_columns;
        constants[2].u32 = fused_gate->block_count;
        constants[3].u32 = static_cast<uint32_t>(input.rows());
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(fused_gate->output_columns * 32);
        dispatcher.h = static_cast<int>(input.rows());
        dispatcher.c = 1;
        command.record_pipeline(
            fused_gate->pipeline.get(),
            bindings,
            constants,
            dispatcher);
    }
    else
    {
        if (!fused || !fused->layer
            || fused->layer->forward(
                   normalized_gpu,
                   fused_gpu,
                   command,
                   implementation.option)
                   != 0)
        {
            ++runtime_state.attention_qkv_failures;
            return false;
        }
    }

    uint64_t ring_capacity = cache.capacity_tokens;
    uint64_t ring_first_slot = cache.first_slot;
    const bool ring_resized =
        !promote_host_cache && actual_token_count > ring_capacity;
    const bool allocate_ring = !has_device_cache || ring_resized;
    std::shared_ptr<NcnnVulkanAttentionCache> next_cache = std::make_shared<NcnnVulkanAttentionCache>();
    if (allocate_ring)
    {
        ring_capacity = next_attention_ring_capacity(
            has_device_cache ? ring_capacity : 0,
            actual_token_count);
        if (!create_attention_ring_storage(
                *next_cache,
                config.head_dimension,
                config.kv_head_count,
                ring_capacity,
                activation_element_size,
                implementation.option.blob_vkallocator))
        {
            ++runtime_state.attention_cache_failures;
            return false;
        }
        if (cache.token_count != 0)
        {
            const ncnn::VkMat previous_key = promote_host_cache
                                                 ? promoted_key_gpu
                                                 : attention_ring_view(
                                                       cache.vulkan_attention_cache->key,
                                                       cache.first_slot,
                                                       cache.token_count);
            const ncnn::VkMat previous_value = promote_host_cache
                                                   ? promoted_value_gpu
                                                   : attention_ring_view(
                                                         cache.vulkan_attention_cache->value,
                                                         cache.first_slot,
                                                         cache.token_count);
            if (!record_attention_ring_append(implementation.ring_append_pipeline.get(), previous_key, previous_value, next_cache->key, next_cache->value,
                                              ring_capacity, 0, command))
            {
                ++runtime_state.attention_cache_failures;
                return false;
            }
        }
        ring_first_slot = 0;
    }
    else
    {
        if (!cache.vulkan_attention_cache || cache.vulkan_attention_cache->key.empty() || cache.vulkan_attention_cache->value.empty())
        {
            ++runtime_state.attention_cache_failures;
            return false;
        }
        next_cache->key = cache.vulkan_attention_cache->key;
        next_cache->value = cache.vulkan_attention_cache->value;
    }
    const uint64_t append_slot = (ring_first_slot + cache.token_count) % ring_capacity;

    ncnn::VkMat query_rope;
    ncnn::VkMat key_rope;
    ncnn::VkMat value_heads;
    ncnn::VkMat output_gate;
    ncnn::VkMat fused_qkv_unpacked = fused_gpu;
    AttentionQkvRopeFailureStage qkv_rope_failure =
        AttentionQkvRopeFailureStage::None;
    if (fused_qkv_unpacked.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(fused_qkv_unpacked, unpacked, 1, command, implementation.option);
        fused_qkv_unpacked = unpacked;
    }
    const bool fused_qkv_ring = input.rows() == 1
                                && qkv_ring_fusion_enabled(config.optimization_flags)
                                && (query_key_norm_and_gate
                                    ? record_attention_qkv_norm_rope(
                                          implementation.qkv_norm_rope_pipeline.get(),
                                          fused_qkv_unpacked,
                                          cosine_gpu,
                                          sine_gpu,
                                          implementation.query_norm_weight,
                                          implementation.key_norm_weight,
                                          config,
                                          input.rows(),
                                          position_offset,
                                          implementation.rope_concentration,
                                          device_rope,
                                          activation_element_size,
                                          &next_cache->key,
                                          &next_cache->value,
                                          ring_capacity,
                                          append_slot,
                                          query_rope,
                                          key_rope,
                                          value_heads,
                                          output_gate,
                                          command,
                                           implementation.option.blob_vkallocator,
                                           &qkv_rope_failure)
                                    : record_attention_qkv_rope(
                                          implementation.qkv_rope_pipeline.get(),
                                          fused_qkv_unpacked,
                                          cosine_gpu,
                                          sine_gpu,
                                          config,
                                          input.rows(),
                                          position_offset,
                                          implementation.rope_concentration,
                                          device_rope,
                                          activation_element_size,
                                          &next_cache->key,
                                          &next_cache->value,
                                          ring_capacity,
                                          append_slot,
                                          query_rope,
                                           key_rope,
                                           value_heads,
                                           command,
                                           implementation.option.blob_vkallocator,
                                           &qkv_rope_failure));
    const bool fused_qkv_rope = fused_qkv_ring
                                || (query_key_norm_and_gate
                                    ? record_attention_qkv_norm_rope(
                                          implementation.qkv_norm_rope_pipeline.get(),
                                          fused_qkv_unpacked,
                                          cosine_gpu,
                                          sine_gpu,
                                          implementation.query_norm_weight,
                                          implementation.key_norm_weight,
                                          config,
                                          input.rows(),
                                          position_offset,
                                          implementation.rope_concentration,
                                          device_rope,
                                          activation_element_size,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          query_rope,
                                          key_rope,
                                          value_heads,
                                          output_gate,
                                          command,
                                           implementation.option.blob_vkallocator,
                                           &qkv_rope_failure)
                                    : record_attention_qkv_rope(
                                          implementation.qkv_rope_pipeline.get(),
                                          fused_qkv_unpacked,
                                          cosine_gpu,
                                          sine_gpu,
                                          config,
                                          input.rows(),
                                          position_offset,
                                          implementation.rope_concentration,
                                          device_rope,
                                          activation_element_size,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          query_rope,
                                           key_rope,
                                           value_heads,
                                           command,
                                           implementation.option.blob_vkallocator,
                                           &qkv_rope_failure));
    if (!fused_qkv_rope)
    {
        record_attention_qkv_rope_failure(runtime_state, qkv_rope_failure);
        if (device_rope)
            return false;
        if (query_key_norm_and_gate)
            return false;
        std::vector<ncnn::VkMat> qkv_input(1, fused_gpu);
        std::vector<ncnn::VkMat> qkv(3);
        if (implementation.slice_qkv->forward(qkv_input, qkv, command, implementation.option) != 0)
        {
            return false;
        }

        ncnn::VkMat query_shaped;
        ncnn::VkMat key_shaped;
        ncnn::VkMat value_shaped;
        if (implementation.reshape_query->forward(qkv[0], query_shaped, command, implementation.option) != 0
            || implementation.reshape_key_value->forward(qkv[1], key_shaped, command, implementation.option) != 0
            || implementation.reshape_key_value->forward(qkv[2], value_shaped, command, implementation.option) != 0)
        {
            return false;
        }

        ncnn::VkMat query_heads;
        ncnn::VkMat key_heads;
        if (implementation.permute_heads_tokens->forward(query_shaped, query_heads, command, implementation.option) != 0
            || implementation.permute_heads_tokens->forward(key_shaped, key_heads, command, implementation.option) != 0
            || implementation.permute_heads_tokens->forward(value_shaped, value_heads, command, implementation.option) != 0)
        {
            return false;
        }
        if (query_heads.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(query_heads, unpacked, 1, command, implementation.option);
            query_heads = unpacked;
        }
        if (key_heads.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(key_heads, unpacked, 1, command, implementation.option);
            key_heads = unpacked;
        }
        if (value_heads.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(value_heads, unpacked, 1, command, implementation.option);
            value_heads = unpacked;
        }

        std::vector<ncnn::VkMat> query_rope_input = {
            query_heads,
            cosine_gpu,
            sine_gpu,
        };
        std::vector<ncnn::VkMat> key_rope_input = {
            key_heads,
            cosine_gpu,
            sine_gpu,
        };
        std::vector<ncnn::VkMat> query_rope_output(1);
        std::vector<ncnn::VkMat> key_rope_output(1);
        if (implementation.rotary->forward(query_rope_input, query_rope_output, command, implementation.option) != 0
            || implementation.rotary->forward(key_rope_input, key_rope_output, command, implementation.option) != 0)
        {
            ++runtime_state.attention_qkv_failures;
            return false;
        }
        query_rope = query_rope_output[0];
        key_rope = key_rope_output[0];
    }

    if (query_rope.elempack != 1 || query_rope.dims != 3 || query_rope.w != static_cast<int>(config.head_dimension)
        || query_rope.h != static_cast<int>(input.rows()) || query_rope.c != static_cast<int>(config.head_count)
        || (!fused_qkv_ring
            && (key_rope.elempack != 1 || value_heads.elempack != 1 || key_rope.dims != 3 || key_rope.w != static_cast<int>(config.head_dimension)
                || key_rope.h != static_cast<int>(input.rows()) || key_rope.c != static_cast<int>(config.kv_head_count) || value_heads.dims != 3
                || value_heads.w != static_cast<int>(config.head_dimension) || value_heads.h != static_cast<int>(input.rows())
                 || value_heads.c != static_cast<int>(config.kv_head_count) || key_rope.elemsize != activation_element_size
                 || value_heads.elemsize != activation_element_size)))
    {
        ++runtime_state.attention_qkv_failures;
        return false;
    }

    if (!fused_qkv_ring
        && !record_attention_ring_append(implementation.ring_append_pipeline.get(), key_rope, value_heads, next_cache->key, next_cache->value, ring_capacity,
                                         append_slot, command))
    {
        ++runtime_state.attention_cache_failures;
        return false;
    }

    bool sink_zero_recorded = false;
    if (sink_token_count != 0 && !fused_qkv_ring)
    {
        const uint64_t sink_row = ring_first_slot + actual_token_count;
        if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(), next_cache->key, next_cache->value, sink_row, command))
        {
            ++runtime_state.attention_cache_failures;
            return false;
        }
        sink_zero_recorded = true;
    }

    ncnn::VkMat combined_key = attention_ring_view(next_cache->key, ring_first_slot, destination_count);
    ncnn::VkMat combined_value = attention_ring_view(next_cache->value, ring_first_slot, destination_count);
    if (combined_key.empty() || combined_value.empty())
    {
        ++runtime_state.attention_cache_failures;
        return false;
    }
    ncnn::VkMat attention_matrix;
    const bool fused_decode_sdpa = try_decode_sdpa
                                   && record_attention_decode_sdpa(implementation.decode_sdpa_pipeline.get(), query_rope, combined_key, combined_value, implementation.attention_sinks, config,
                                                                   destination_count, attention_matrix, command, implementation.option.blob_vkallocator);
    if (!fused_decode_sdpa)
    {
        if (sink_token_count != 0 && !sink_zero_recorded)
        {
            const uint64_t sink_row = ring_first_slot + actual_token_count;
            if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(), next_cache->key, next_cache->value, sink_row, command))
            {
                ++runtime_state.attention_cache_failures;
                return false;
            }
            sink_zero_recorded = true;
        }
        if (!mask_uploaded)
        {
            if (!fill_attention_mask_staging(transfer_slot.attention_mask, input.rows(), destination_count, position_offset, cache, config,
                                             implementation.sinks, bfloat16_storage, transfer_slot.staging_allocator, runtime_state)
                || !record_mapped_upload(transfer_slot.attention_mask, mask_gpu, command, implementation.option))
            {
                ++runtime_state.attention_sdpa_failures;
                return false;
            }
            mask_uploaded = true;
        }
        ncnn::VkMat sdpa_key = combined_key;
        ncnn::VkMat sdpa_value = combined_value;
        if (low_precision_kv)
        {
            vkdev->convert_packing(
                combined_key,
                sdpa_key,
                1,
                1,
                command,
                implementation.kv_option);
            vkdev->convert_packing(
                combined_value,
                sdpa_value,
                1,
                1,
                command,
                implementation.kv_option);
            if (sdpa_key.empty() || sdpa_value.empty()
                || sdpa_key.elemsize != sizeof(float)
                || sdpa_value.elemsize != sizeof(float))
            {
                ++runtime_state.attention_sdpa_failures;
                return false;
            }
        }
        std::vector<ncnn::VkMat> sdpa_input = {
            query_rope,
            sdpa_key,
            sdpa_value,
            mask_gpu,
        };
        std::vector<ncnn::VkMat> sdpa_output(1);
        if (implementation.sdpa->forward(sdpa_input, sdpa_output, command, implementation.option) != 0)
        {
            ++runtime_state.attention_sdpa_failures;
            return false;
        }

        ncnn::VkMat attention_token_major;
        if (implementation.permute_heads_tokens->forward(sdpa_output[0], attention_token_major, command, implementation.option) != 0
            || implementation.reshape_attention->forward(attention_token_major, attention_matrix, command, implementation.option) != 0)
        {
            ++runtime_state.attention_sdpa_failures;
            return false;
        }
    }

    if (query_key_norm_and_gate)
    {
        if (attention_matrix.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                attention_matrix,
                unpacked,
                1,
                command,
                implementation.option);
            attention_matrix = unpacked;
        }
        if (!record_attention_output_gate(
                implementation.output_gate_pipeline.get(),
                attention_matrix,
                output_gate,
                input.rows(),
                query_columns,
                command))
        {
            ++runtime_state.attention_projection_failures;
            return false;
        }
    }

    ncnn::VkMat projected_gpu;
    if (query_key_norm_and_gate)
    {
        if (!projection_bfloat16 || !projection_bfloat16->pipeline
            || attention_matrix.empty()
            || attention_matrix.dims != 2
            || attention_matrix.w != static_cast<int>(projection_bfloat16->input_columns)
            || attention_matrix.h != static_cast<int>(input.rows())
            || attention_matrix.elempack != 1
            || attention_matrix.elemsize != sizeof(float)
            || projection_bfloat16->output_columns != config.hidden_size)
        {
            return false;
        }
        projected_gpu.create(
            static_cast<int>(projection_bfloat16->output_columns),
            static_cast<int>(input.rows()),
            sizeof(float),
            implementation.vulkan_context->blob_allocator());
        if (projected_gpu.empty())
        {
            ++runtime_state.attention_projection_failures;
            return false;
        }
        const std::vector<ncnn::VkMat> bindings = {
            attention_matrix,
            projection_bfloat16->packed,
            projection_bfloat16->bias,
            projected_gpu};
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].u32 = projection_bfloat16->input_columns;
        constants[1].u32 = projection_bfloat16->output_columns;
        constants[2].u32 = projection_bfloat16->block_count;
        constants[3].u32 = static_cast<uint32_t>(input.rows());
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(projection_bfloat16->output_columns * 32);
        dispatcher.h = static_cast<int>(input.rows());
        dispatcher.c = 1;
        command.record_pipeline(
            projection_bfloat16->pipeline.get(),
            bindings,
            constants,
            dispatcher);
    }
    else
    {
        if (!projection || !projection->layer
            || projection->layer->forward(
                   attention_matrix,
                   projected_gpu,
                   command,
                   implementation.option)
                   != 0)
        {
            ++runtime_state.attention_projection_failures;
            return false;
        }
    }
    std::vector<ncnn::VkMat> add_input = {input_gpu, projected_gpu};
    std::vector<ncnn::VkMat> add_output(1);
    if (implementation.add->forward(add_input, add_output, command, implementation.option) != 0)
    {
        ++runtime_state.attention_output_failures;
        return false;
    }

    ncnn::VkMat download_gpu = add_output[0];
    if (download_gpu.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(download_gpu, unpacked, 1, command, implementation.option);
        download_gpu = unpacked;
    }
    if (!record_prepared_activation_staging_download(
            download_gpu,
            input.rows(),
            config.hidden_size,
            transfer_slot.download,
            command,
            vkdev,
            implementation.option,
            output.dtype()))
    {
        ++runtime_state.attention_output_failures;
        return false;
    }

    const uint64_t total_actual_tokens = cache.token_count + input.rows();
    const uint64_t retained_tokens = config.sliding_window == 0 ? total_actual_tokens
                                                                : std::min<uint64_t>(total_actual_tokens, config.sliding_window > 1 ? config.sliding_window - 1 : 0);
    const uint64_t dropped_tokens = total_actual_tokens - retained_tokens;
    const uint64_t next_first_slot = retained_tokens == 0 ? 0 : (ring_first_slot + dropped_tokens) % ring_capacity;
    const uint64_t allocated_cache_bytes = retained_tokens == 0
                                               ? 0
                                               : static_cast<uint64_t>(next_cache->key.cstep) * next_cache->key.c * next_cache->key.elemsize
                                                     + static_cast<uint64_t>(next_cache->value.cstep) * next_cache->value.c * next_cache->value.elemsize;

    output = ActivationBuffer(input.rows(), config.hidden_size);
    const auto execution_started = std::chrono::steady_clock::now();
    const int submit_result = submit_compute_and_wait(command, runtime_state);
    if (submit_result != 0)
    {
        ++runtime_state.attention_submit_failures;
        if (has_device_cache && !ring_resized)
            cache.vulkan_attention_state_unknown = true;
        return false;
    }
    if (!copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        ++runtime_state.attention_submit_failures;
        if (has_device_cache && !ring_resized)
            cache.vulkan_attention_state_unknown = true;
        return false;
    }
    if (adaptive_decode_sdpa)
    {
        implementation.observe_decode_sdpa(
            config.head_dimension, config.head_count, config.kv_head_count, destination_count, fused_decode_sdpa,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - execution_started).count()));
    }
    for (size_t row_index = 0; row_index < output.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            if (!std::isfinite(output.row(row_index)[column]))
            {
                ++runtime_state.attention_submit_failures;
                if (has_device_cache && !ring_resized)
                    cache.vulkan_attention_state_unknown = true;
                return false;
            }
        }
    }

    const uint64_t previous_start = cache.token_count == 0 ? position_offset : cache.start_position;
    std::vector<float>{}.swap(cache.keys);
    std::vector<float>{}.swap(cache.values);
    std::vector<uint16_t>{}.swap(cache.bfloat16_keys);
    std::vector<uint16_t>{}.swap(cache.bfloat16_values);
    cache.start_position = previous_start + dropped_tokens;
    cache.token_count = retained_tokens;
    cache.first_slot = next_first_slot;
    cache.capacity_tokens = retained_tokens == 0 ? 0 : ring_capacity;
    cache.columns = config.kv_head_count * config.head_dimension;
    cache.dtype = config.kv_cache_dtype;
    cache.vulkan_attention_cache = retained_tokens == 0 ? nullptr : std::move(next_cache);
    cache.device_allocated_bytes = allocated_cache_bytes;
    cache.vulkan_attention_state_unknown = false;
    record_standard_cache_transaction_rows(cache, input.rows());
        runtime_state.dispatches += 2;
        ++runtime_state.attention_blocks;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    runtime_state.auxiliary_uploads +=
        (mask_uploaded ? 1 : 0) + (device_rope ? 0 : 2)
        + (promote_host_cache ? 2 : 0);
    if (fused_qkv_rope)
        ++runtime_state.attention_qkv_rope_fusions;
    if (device_rope)
        ++runtime_state.attention_device_rope_fusions;
    if (fused_qkv_ring)
        ++runtime_state.attention_qkv_ring_fusions;
    if (fused_decode_sdpa)
        ++runtime_state.attention_decode_sdpa_fusions;
    ++runtime_state.kv_ring_appends;
    if (ring_resized)
        ++runtime_state.kv_ring_resizes;
    if (ring_first_slot + destination_count > ring_capacity)
        ++runtime_state.kv_ring_wrapped_views;
    if (promote_host_cache)
    {
        ++runtime_state.kv_cache_promotions;
        runtime_state.kv_cache_promotion_bytes +=
            promotion_transfer_bytes;
    }
    runtime_state.auxiliary_upload_bytes += (device_rope
                                                                   ? 0
                                                                   : transfer_slot.rope_cosine.buffer_capacity()
                                                                         + transfer_slot.rope_sine.buffer_capacity())
                                                              + (mask_uploaded ? transfer_slot.attention_mask.buffer_capacity() : 0)
                                                              + (promote_host_cache
                                                                     ? transfer_slot.attention_cache_key.buffer_capacity()
                                                                           + transfer_slot.attention_cache_value.buffer_capacity()
                                                                     : 0);
    promotion_attempt.complete();
    return true;
#else
    (void)position_offset;
    (void)cache;
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanAttentionOperator::materialize_device_cache(
    CpuLayerCache& cache) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    const NcnnVulkanAttentionConfig& config = implementation.config;
    if (!implementation.vulkan_context
        || cache.vulkan_attention_state_unknown
        || !cache.vulkan_attention_cache
        || cache.token_count == 0
        || cache.capacity_tokens == 0
        || cache.first_slot >= cache.capacity_tokens
        || cache.token_count > cache.capacity_tokens
        || cache.dtype != config.kv_cache_dtype
        || config.head_dimension == 0
        || config.kv_head_count == 0
        || cache.token_count
               > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    const size_t element_size =
        vulkan_activation_element_size(implementation.kv_option);
    const NcnnVulkanAttentionCache& device_cache =
        *cache.vulkan_attention_cache;
    if (device_cache.key.empty()
        || device_cache.value.empty()
        || device_cache.key.dims != 3
        || device_cache.value.dims != 3
        || device_cache.key.w
               != static_cast<int>(config.head_dimension)
        || device_cache.value.w
               != static_cast<int>(config.head_dimension)
        || device_cache.key.c
               != static_cast<int>(config.kv_head_count)
        || device_cache.value.c
               != static_cast<int>(config.kv_head_count)
        || static_cast<uint64_t>(device_cache.key.h)
               != cache.capacity_tokens * 2
        || static_cast<uint64_t>(device_cache.value.h)
               != cache.capacity_tokens * 2
        || device_cache.key.elemsize != element_size
        || device_cache.value.elemsize != element_size
        || device_cache.key.elempack != 1
        || device_cache.value.elempack != 1)
    {
        return false;
    }

    const ncnn::VkMat source_key = attention_ring_view(
        device_cache.key,
        cache.first_slot,
        cache.token_count);
    const ncnn::VkMat source_value = attention_ring_view(
        device_cache.value,
        cache.first_slot,
        cache.token_count);
    if (source_key.empty() || source_value.empty())
        return false;

    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();
    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_tensor(
            transfer_slot.attention_cache_key,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            element_size,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_tensor(
            transfer_slot.attention_cache_value,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            element_size,
            transfer_slot.staging_allocator,
            runtime_state))
    {
        return false;
    }

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::Option copy_option = implementation.option;
    copy_option.blob_vkallocator = transfer_slot.staging_allocator;
    copy_option.workspace_vkallocator = transfer_slot.staging_allocator;
    copy_option.staging_vkallocator = transfer_slot.staging_allocator;
    ncnn::VkMat& key_staging = transfer_slot.attention_cache_key;
    ncnn::VkMat& value_staging = transfer_slot.attention_cache_value;
    command.record_clone(source_key, key_staging, copy_option);
    command.record_clone(source_value, value_staging, copy_option);
    if (key_staging.empty() || value_staging.empty())
        return false;

    if (submit_compute_and_wait(command, runtime_state) != 0)
        return false;
    key_staging.allocator->invalidate(key_staging.data);
    value_staging.allocator->invalidate(value_staging.data);
    const ncnn::Mat key_mapped = key_staging.mapped();
    const ncnn::Mat value_mapped = value_staging.mapped();
    if (key_mapped.empty()
        || value_mapped.empty()
        || key_mapped.dims != 3
        || value_mapped.dims != 3
        || key_mapped.w != static_cast<int>(config.head_dimension)
        || value_mapped.w != static_cast<int>(config.head_dimension)
        || key_mapped.h != static_cast<int>(cache.token_count)
        || value_mapped.h != static_cast<int>(cache.token_count)
        || key_mapped.c != static_cast<int>(config.kv_head_count)
        || value_mapped.c != static_cast<int>(config.kv_head_count)
        || key_mapped.elemsize != element_size
        || value_mapped.elemsize != element_size
        || key_mapped.elempack != 1
        || value_mapped.elempack != 1)
    {
        return false;
    }

    const size_t storage_variant =
        vulkan_activation_storage_variant(implementation.kv_option);
    ncnn::Mat key_float_storage;
    ncnn::Mat value_float_storage;
    const ncnn::Mat* key_float_source = &key_mapped;
    const ncnn::Mat* value_float_source = &value_mapped;
    if (storage_variant == 1
        || (storage_variant == 2 && cache.dtype == DType::Float32))
    {
        ncnn::Option cast_option = implementation.option;
        if (storage_variant == 1)
        {
            ncnn::cast_float16_to_float32(
                key_mapped,
                key_float_storage,
                cast_option);
            ncnn::cast_float16_to_float32(
                value_mapped,
                value_float_storage,
                cast_option);
        }
        else
        {
            ncnn::cast_bfloat16_to_float32(
                key_mapped,
                key_float_storage,
                cast_option);
            ncnn::cast_bfloat16_to_float32(
                value_mapped,
                value_float_storage,
                cast_option);
        }
        if (key_float_storage.empty() || value_float_storage.empty())
            return false;
        key_float_source = &key_float_storage;
        value_float_source = &value_float_storage;
    }

    const uint64_t columns_u64 =
        static_cast<uint64_t>(config.kv_head_count)
        * config.head_dimension;
    if (columns_u64 == 0
        || columns_u64 > std::numeric_limits<uint32_t>::max()
        || cache.token_count
               > static_cast<uint64_t>(
                     std::numeric_limits<size_t>::max() / columns_u64))
    {
        return false;
    }
    const uint32_t columns = static_cast<uint32_t>(columns_u64);
    const size_t element_count =
        static_cast<size_t>(cache.token_count) * columns;
    if (cache.dtype == DType::BFloat16)
    {
        std::vector<uint16_t> keys(element_count);
        std::vector<uint16_t> values(element_count);
        for (uint32_t head = 0; head < config.kv_head_count; ++head)
        {
            const ncnn::Mat key_channel = key_mapped.channel(head);
            const ncnn::Mat value_channel = value_mapped.channel(head);
            const ncnn::Mat key_float_channel =
                key_float_source->channel(head);
            const ncnn::Mat value_float_channel =
                value_float_source->channel(head);
            for (uint64_t token = 0; token < cache.token_count; ++token)
            {
                uint16_t* key_destination =
                    keys.data()
                    + static_cast<size_t>(token) * columns
                    + static_cast<size_t>(head) * config.head_dimension;
                uint16_t* value_destination =
                    values.data()
                    + static_cast<size_t>(token) * columns
                    + static_cast<size_t>(head) * config.head_dimension;
                if (storage_variant == 2)
                {
                    std::copy_n(
                        key_channel.row<uint16_t>(static_cast<int>(token)),
                        config.head_dimension,
                        key_destination);
                    std::copy_n(
                        value_channel.row<uint16_t>(static_cast<int>(token)),
                        config.head_dimension,
                        value_destination);
                }
                else
                {
                    const float* key_source =
                        key_float_channel.row<float>(static_cast<int>(token));
                    const float* value_source =
                        value_float_channel.row<float>(static_cast<int>(token));
                    for (uint32_t column = 0;
                         column < config.head_dimension;
                         ++column)
                    {
                        key_destination[column] =
                            float_to_bfloat16(key_source[column]);
                        value_destination[column] =
                            float_to_bfloat16(value_source[column]);
                    }
                }
            }
        }
        cache.bfloat16_keys = std::move(keys);
        cache.bfloat16_values = std::move(values);
        std::vector<float>{}.swap(cache.keys);
        std::vector<float>{}.swap(cache.values);
    }
    else if (cache.dtype == DType::Float32)
    {
        std::vector<float> keys(element_count);
        std::vector<float> values(element_count);
        for (uint32_t head = 0; head < config.kv_head_count; ++head)
        {
            const ncnn::Mat key_channel = key_float_source->channel(head);
            const ncnn::Mat value_channel = value_float_source->channel(head);
            for (uint64_t token = 0; token < cache.token_count; ++token)
            {
                std::copy_n(
                    key_channel.row<float>(static_cast<int>(token)),
                    config.head_dimension,
                    keys.data()
                        + static_cast<size_t>(token) * columns
                        + static_cast<size_t>(head) * config.head_dimension);
                std::copy_n(
                    value_channel.row<float>(static_cast<int>(token)),
                    config.head_dimension,
                    values.data()
                        + static_cast<size_t>(token) * columns
                        + static_cast<size_t>(head) * config.head_dimension);
            }
        }
        cache.keys = std::move(keys);
        cache.values = std::move(values);
        std::vector<uint16_t>{}.swap(cache.bfloat16_keys);
        std::vector<uint16_t>{}.swap(cache.bfloat16_values);
    }
    else
    {
        return false;
    }

    cache.first_slot = 0;
    cache.capacity_tokens = cache.token_count;
    cache.vulkan_attention_cache.reset();
    cache.device_allocated_bytes = 0;
    cache.vulkan_attention_state_unknown = false;
    // The failed device path has now been converted to a valid CPU path.  Do
    // not immediately re-promote the same cache and repeat the failure.
    cache.vulkan_attention_promotion_disabled = true;
    ++runtime_state.attention_cache_materializations;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)cache;
    return false;
#endif
}

void NcnnVulkanAttentionOperator::record_cpu_fallback() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (implementation_ && implementation_->vulkan_context)
        ++implementation_->vulkan_context->runtime_state().attention_cpu_fallbacks;
#endif
}

NcnnVulkanAttentionBatchResult
NcnnVulkanAttentionOperator::forward_batch(
    std::span<const NcnnVulkanAttentionBatchEntry> entries) const
{
#if NCNN_MOE_WITH_VULKAN
    const uint64_t optimization_flags = implementation_->config.optimization_flags;
    if (entries.size() < 2
        || !vulkan_attention_enabled(optimization_flags)
        || !vulkan_attention_batch_enabled(optimization_flags))
        return NcnnVulkanAttentionBatchResult::NotExecuted;

    const Implementation& implementation = *implementation_;
    NcnnVulkanRuntimeState& runtime_state =
        implementation.vulkan_context->runtime_state();
    const NcnnVulkanAttentionConfig& config = implementation.config;
    const size_t activation_element_size =
        vulkan_activation_element_size(implementation.kv_option);
    const bool low_precision_kv =
        vulkan_activation_storage_variant(implementation.kv_option) != 0;
    const bool bfloat16_storage =
        config.activation_dtype == DType::BFloat16
        && implementation.option.use_bf16_storage;
    const uint64_t query_columns_u64 =
        static_cast<uint64_t>(config.head_count)
        * config.head_dimension;
    if (query_columns_u64 > std::numeric_limits<uint32_t>::max())
        return NcnnVulkanAttentionBatchResult::NotExecuted;
    const uint32_t query_columns =
        static_cast<uint32_t>(query_columns_u64);
    const uint64_t sink_token_count =
        has_flag(config.flags, NcnnAttentionSink) ? 1 : 0;
    const bool query_key_norm_and_gate =
        implementation.fused_qkv_gate != nullptr;
    const NcnnLinearOperator::Implementation* fused =
        implementation.fused_qkv
            ? implementation.fused_qkv->implementation_.get()
            : nullptr;
    const NcnnVulkanBfloat16Operator::Implementation* fused_gate =
        implementation.fused_qkv_gate
            ? implementation.fused_qkv_gate->implementation_.get()
            : nullptr;
    const NcnnLinearOperator::Implementation* projection =
        implementation.output_projection
            ? implementation.output_projection->implementation_.get()
            : nullptr;
    const NcnnVulkanBfloat16Operator::Implementation* projection_bfloat16 =
        implementation.output_projection_bfloat16
            ? implementation.output_projection_bfloat16->implementation_.get()
            : nullptr;

    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input =
        vulkan_activation_storage_variant(implementation.option) == 0
        && direct_host_input_enabled(
            *implementation.vulkan_context,
            static_cast<size_t>(config.hidden_size) * sizeof(float),
            entries.front().input->dtype());

    struct PreparedAttentionEntry
    {
        const NcnnVulkanAttentionBatchEntry* entry = nullptr;
        ncnn::VkMat upload;
        ncnn::VkMat download;
        ncnn::VkMat rope_cosine;
        ncnn::VkMat rope_sine;
        ncnn::VkMat attention_mask;
        ncnn::VkMat input_gpu;
        ncnn::VkMat cosine_gpu;
        ncnn::VkMat sine_gpu;
        ncnn::VkMat mask_gpu;
        ncnn::VkMat normalized_gpu;
        ncnn::VkMat normalized_unpacked_gpu;
        ncnn::VkMat fused_gpu;
        ncnn::VkMat fused_qkv_unpacked_gpu;
        std::vector<ncnn::VkMat> qkv;
        ncnn::VkMat query_shaped_gpu;
        ncnn::VkMat key_shaped_gpu;
        ncnn::VkMat value_shaped_gpu;
        ncnn::VkMat query_heads_gpu;
        ncnn::VkMat key_heads_gpu;
        ncnn::VkMat query_rope;
        ncnn::VkMat key_rope;
        ncnn::VkMat value_heads;
        ncnn::VkMat output_gate;
        std::vector<ncnn::VkMat> query_rope_output;
        std::vector<ncnn::VkMat> key_rope_output;
        ncnn::VkMat combined_key;
        ncnn::VkMat combined_value;
        ncnn::VkMat attention_matrix;
        std::vector<ncnn::VkMat> sdpa_output;
        ncnn::VkMat attention_token_major;
        ncnn::VkMat projected_gpu;
        std::vector<ncnn::VkMat> add_output;
        ncnn::VkMat download_gpu;
        std::vector<ncnn::VkMat> retained_gpu;
        std::shared_ptr<NcnnVulkanAttentionCache> next_cache;
        uint64_t actual_token_count = 0;
        uint64_t destination_count = 0;
        uint64_t retained_tokens = 0;
        uint64_t dropped_tokens = 0;
        uint64_t ring_first_slot = 0;
        uint64_t next_first_slot = 0;
        uint64_t allocated_cache_bytes = 0;
        bool adaptive_decode_sdpa = false;
        bool try_decode_sdpa = false;
        bool mask_uploaded = false;
        bool device_rope = false;
        bool fused_qkv_rope = false;
        bool fused_qkv_ring = false;
        bool fused_decode_sdpa = false;
    };

    std::vector<PreparedAttentionEntry> prepared;
    prepared.reserve(entries.size());
    const DecodeSdpaMode selected_decode_sdpa_mode = decode_sdpa_mode(config.optimization_flags);
    for (const NcnnVulkanAttentionBatchEntry& entry : entries)
    {
        if (!entry.cache || !entry.input || !entry.output)
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        CpuLayerCache& cache = *entry.cache;
        const ActivationBuffer& input = *entry.input;
        if (cache.vulkan_attention_state_unknown)
            return NcnnVulkanAttentionBatchResult::Failed;
        if (input.rows() != 1
            || input.columns() != config.hidden_size
            || cache.transaction.active
            || cache.token_count == 0
            || cache.dtype != config.kv_cache_dtype
            || cache.capacity_tokens == 0
            || cache.first_slot >= cache.capacity_tokens
            || cache.token_count > cache.capacity_tokens
            || !cache.vulkan_attention_cache
            || cache.vulkan_attention_cache->key.empty()
            || cache.vulkan_attention_cache->value.empty()
            || cache.vulkan_attention_cache->key.dims != 3
            || cache.vulkan_attention_cache->value.dims != 3
            || cache.vulkan_attention_cache->key.w
                   != static_cast<int>(config.head_dimension)
            || cache.vulkan_attention_cache->value.w
                   != static_cast<int>(config.head_dimension)
            || cache.vulkan_attention_cache->key.c
                   != static_cast<int>(config.kv_head_count)
            || cache.vulkan_attention_cache->value.c
                   != static_cast<int>(config.kv_head_count)
            || static_cast<uint64_t>(
                   cache.vulkan_attention_cache->key.h)
                   != cache.capacity_tokens * 2
            || static_cast<uint64_t>(
                   cache.vulkan_attention_cache->value.h)
                   != cache.capacity_tokens * 2
            || cache.vulkan_attention_cache->key.elemsize
                   != activation_element_size
            || cache.vulkan_attention_cache->value.elemsize
                   != activation_element_size
            || cache.vulkan_attention_cache->key.elempack != 1
            || cache.vulkan_attention_cache->value.elempack != 1)
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }

        const uint64_t actual_token_count = cache.token_count + 1;
        const uint64_t destination_count =
            actual_token_count + sink_token_count;
        if (actual_token_count <= cache.token_count
            || actual_token_count > cache.capacity_tokens
            || destination_count < actual_token_count
            || destination_count
                   > static_cast<uint64_t>(
                         std::numeric_limits<int>::max()))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }

        prepared.emplace_back();
        PreparedAttentionEntry& work = prepared.back();
        work.entry = &entry;
        work.device_rope = vulkan_attention_device_rope_enabled(config.optimization_flags)
                           && !implementation.rope_inverse_frequencies_gpu.empty()
                           && entry.position_offset
                                  <= std::numeric_limits<uint32_t>::max();
        work.actual_token_count = actual_token_count;
        work.destination_count = destination_count;
        work.adaptive_decode_sdpa =
            config.head_dimension <= 128
            && destination_count <= 4096
            && selected_decode_sdpa_mode == DecodeSdpaMode::Auto;
        work.try_decode_sdpa =
            selected_decode_sdpa_mode != DecodeSdpaMode::Disabled
            && (selected_decode_sdpa_mode == DecodeSdpaMode::Forced
                || (work.adaptive_decode_sdpa
                    && implementation.choose_decode_sdpa(
                        config.head_dimension,
                        config.head_count,
                        config.kv_head_count,
                        destination_count)));
        work.next_cache = cache.vulkan_attention_cache;
        work.retained_tokens =
            config.sliding_window == 0
                ? actual_token_count
                : std::min<uint64_t>(
                      actual_token_count,
                      config.sliding_window > 1
                          ? config.sliding_window - 1
                          : 0);
        work.dropped_tokens =
            actual_token_count - work.retained_tokens;
        work.ring_first_slot = cache.first_slot;
        work.next_first_slot =
            work.retained_tokens == 0
                ? 0
                : (cache.first_slot + work.dropped_tokens)
                      % cache.capacity_tokens;
        work.allocated_cache_bytes =
            work.retained_tokens == 0
                ? 0
                : static_cast<uint64_t>(work.next_cache->key.cstep)
                          * work.next_cache->key.c
                          * work.next_cache->key.elemsize
                      + static_cast<uint64_t>(work.next_cache->value.cstep)
                          * work.next_cache->value.c
                          * work.next_cache->value.elemsize;

        if (!fill_staging_upload(
                input,
                work.upload,
                transfer_slot.staging_allocator,
                runtime_state)
            || (!work.device_rope
                && !fill_rope_staging_pair(
                    work.rope_cosine,
                    work.rope_sine,
                    1,
                    entry.position_offset,
                    implementation.rope_inverse_frequencies,
                    implementation.rope_concentration,
                    bfloat16_storage,
                    transfer_slot.staging_allocator,
                    runtime_state))
            || !fill_attention_mask_staging(
                work.attention_mask,
                1,
                destination_count,
                entry.position_offset,
                cache,
                config,
                implementation.sinks,
                bfloat16_storage,
                transfer_slot.staging_allocator,
                runtime_state)
            || !prepare_staging_batch(
                work.download,
                1,
                config.hidden_size,
                transfer_slot.staging_allocator,
                runtime_state))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }
        entry.output->reset(1, config.hidden_size, false);
    }

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev =
        implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    for (PreparedAttentionEntry& work : prepared)
    {
        CpuLayerCache& cache = *work.entry->cache;
        if (work.device_rope)
        {
            work.cosine_gpu = implementation.rope_inverse_frequencies_gpu;
            work.sine_gpu = implementation.rope_inverse_frequencies_gpu;
        }
        if (direct_host_input)
            work.input_gpu = bind_direct_host_input(work.upload, runtime_state);
        else if (!record_prepared_staging_upload(
                     work.upload,
                     1,
                     work.input_gpu,
                     command,
                     vkdev,
                     implementation.option,
                     work.entry->input->dtype()))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }
        if (!work.device_rope
                && (!record_mapped_activation_upload(
                        work.rope_cosine,
                        work.cosine_gpu,
                        command,
                        vkdev,
                        implementation.option)
                    || !record_mapped_activation_upload(
                        work.rope_sine,
                        work.sine_gpu,
                        command,
                        vkdev,
                        implementation.option)))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }
        if (!work.try_decode_sdpa)
        {
            if (!record_mapped_activation_upload(
                    work.attention_mask,
                    work.mask_gpu,
                    command,
                    vkdev,
                    implementation.option))
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            work.mask_uploaded = true;
        }
        if (implementation.norm->forward(
                work.input_gpu,
                work.normalized_gpu,
                command,
                implementation.option)
            != 0)
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }

        if (query_key_norm_and_gate)
        {
            work.normalized_unpacked_gpu = work.normalized_gpu;
            if (work.normalized_unpacked_gpu.elempack != 1)
            {
                work.retained_gpu.push_back(
                    work.normalized_unpacked_gpu);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.normalized_unpacked_gpu,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.normalized_unpacked_gpu = unpacked;
            }
            if (!fused_gate || !fused_gate->pipeline
                || work.normalized_unpacked_gpu.empty()
                || work.normalized_unpacked_gpu.dims != 2
                || work.normalized_unpacked_gpu.w
                       != static_cast<int>(fused_gate->input_columns)
                 || work.normalized_unpacked_gpu.h != 1
                 || work.normalized_unpacked_gpu.elempack != 1
                 || work.normalized_unpacked_gpu.elemsize
                        != sizeof(float))
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            work.fused_gpu.create(
                static_cast<int>(fused_gate->output_columns),
                1,
                sizeof(float),
                implementation.vulkan_context->blob_allocator());
            if (work.fused_gpu.empty())
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            const std::vector<ncnn::VkMat> bindings = {
                work.normalized_unpacked_gpu,
                fused_gate->packed,
                fused_gate->bias,
                work.fused_gpu};
            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].u32 = fused_gate->input_columns;
            constants[1].u32 = fused_gate->output_columns;
            constants[2].u32 = fused_gate->block_count;
            constants[3].u32 = 1;
            ncnn::VkMat dispatcher;
            dispatcher.w =
                static_cast<int>(fused_gate->output_columns * 32);
            dispatcher.h = 1;
            dispatcher.c = 1;
            command.record_pipeline(
                fused_gate->pipeline.get(),
                bindings,
                constants,
                dispatcher);
        }
        else if (!fused || !fused->layer
                 || fused->layer->forward(
                        work.normalized_gpu,
                        work.fused_gpu,
                        command,
                        implementation.option)
                        != 0)
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }

        const uint64_t ring_capacity = cache.capacity_tokens;
        const uint64_t ring_first_slot = cache.first_slot;
        const uint64_t append_slot =
            (ring_first_slot + cache.token_count) % ring_capacity;
        work.fused_qkv_unpacked_gpu = work.fused_gpu;
        if (work.fused_qkv_unpacked_gpu.elempack != 1)
        {
            work.retained_gpu.push_back(work.fused_qkv_unpacked_gpu);
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                work.fused_qkv_unpacked_gpu,
                unpacked,
                1,
                command,
                implementation.option);
            work.fused_qkv_unpacked_gpu = unpacked;
        }
        work.fused_qkv_ring =
            qkv_ring_fusion_enabled(config.optimization_flags)
            && (query_key_norm_and_gate
                ? record_attention_qkv_norm_rope(
                      implementation.qkv_norm_rope_pipeline.get(),
                      work.fused_qkv_unpacked_gpu,
                      work.cosine_gpu,
                      work.sine_gpu,
                      implementation.query_norm_weight,
                      implementation.key_norm_weight,
                      config,
                      1,
                      work.entry->position_offset,
                      implementation.rope_concentration,
                      work.device_rope,
                      activation_element_size,
                      &work.next_cache->key,
                      &work.next_cache->value,
                      ring_capacity,
                      append_slot,
                      work.query_rope,
                      work.key_rope,
                      work.value_heads,
                      work.output_gate,
                      command,
                      implementation.option.blob_vkallocator)
                : record_attention_qkv_rope(
                      implementation.qkv_rope_pipeline.get(),
                      work.fused_qkv_unpacked_gpu,
                      work.cosine_gpu,
                      work.sine_gpu,
                      config,
                      1,
                      work.entry->position_offset,
                      implementation.rope_concentration,
                      work.device_rope,
                      activation_element_size,
                      &work.next_cache->key,
                      &work.next_cache->value,
                      ring_capacity,
                      append_slot,
                      work.query_rope,
                      work.key_rope,
                      work.value_heads,
                      command,
                      implementation.option.blob_vkallocator));
        work.fused_qkv_rope =
            work.fused_qkv_ring
            || (query_key_norm_and_gate
                ? record_attention_qkv_norm_rope(
                      implementation.qkv_norm_rope_pipeline.get(),
                      work.fused_qkv_unpacked_gpu,
                      work.cosine_gpu,
                      work.sine_gpu,
                      implementation.query_norm_weight,
                      implementation.key_norm_weight,
                      config,
                      1,
                      work.entry->position_offset,
                      implementation.rope_concentration,
                      work.device_rope,
                      activation_element_size,
                      nullptr,
                      nullptr,
                      0,
                      0,
                      work.query_rope,
                      work.key_rope,
                      work.value_heads,
                      work.output_gate,
                      command,
                      implementation.option.blob_vkallocator)
                : record_attention_qkv_rope(
                      implementation.qkv_rope_pipeline.get(),
                      work.fused_qkv_unpacked_gpu,
                      work.cosine_gpu,
                      work.sine_gpu,
                      config,
                      1,
                      work.entry->position_offset,
                      implementation.rope_concentration,
                      work.device_rope,
                      activation_element_size,
                      nullptr,
                      nullptr,
                      0,
                      0,
                      work.query_rope,
                      work.key_rope,
                      work.value_heads,
                      command,
                      implementation.option.blob_vkallocator));
        if (!work.fused_qkv_rope)
        {
            if (work.device_rope)
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            if (query_key_norm_and_gate)
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            std::vector<ncnn::VkMat> qkv_input(1, work.fused_gpu);
            work.qkv.resize(3);
            if (implementation.slice_qkv->forward(
                    qkv_input,
                    work.qkv,
                    command,
                    implementation.option)
                != 0)
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            if (implementation.reshape_query->forward(
                    work.qkv[0],
                    work.query_shaped_gpu,
                    command,
                    implementation.option)
                    != 0
                || implementation.reshape_key_value->forward(
                    work.qkv[1],
                    work.key_shaped_gpu,
                    command,
                    implementation.option)
                    != 0
                || implementation.reshape_key_value->forward(
                    work.qkv[2],
                    work.value_shaped_gpu,
                    command,
                    implementation.option)
                    != 0
                || implementation.permute_heads_tokens->forward(
                    work.query_shaped_gpu,
                    work.query_heads_gpu,
                    command,
                    implementation.option)
                    != 0
                || implementation.permute_heads_tokens->forward(
                    work.key_shaped_gpu,
                    work.key_heads_gpu,
                    command,
                    implementation.option)
                    != 0
                || implementation.permute_heads_tokens->forward(
                    work.value_shaped_gpu,
                    work.value_heads,
                    command,
                    implementation.option)
                    != 0)
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            if (work.query_heads_gpu.elempack != 1)
            {
                work.retained_gpu.push_back(work.query_heads_gpu);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.query_heads_gpu,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.query_heads_gpu = unpacked;
            }
            if (work.key_heads_gpu.elempack != 1)
            {
                work.retained_gpu.push_back(work.key_heads_gpu);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.key_heads_gpu,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.key_heads_gpu = unpacked;
            }
            if (work.value_heads.elempack != 1)
            {
                work.retained_gpu.push_back(work.value_heads);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.value_heads,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.value_heads = unpacked;
            }
            std::vector<ncnn::VkMat> query_rope_input = {
                work.query_heads_gpu,
                work.cosine_gpu,
                work.sine_gpu};
            std::vector<ncnn::VkMat> key_rope_input = {
                work.key_heads_gpu,
                work.cosine_gpu,
                work.sine_gpu};
            work.query_rope_output.resize(1);
            work.key_rope_output.resize(1);
            if (implementation.rotary->forward(
                    query_rope_input,
                    work.query_rope_output,
                    command,
                    implementation.option)
                    != 0
                || implementation.rotary->forward(
                    key_rope_input,
                    work.key_rope_output,
                    command,
                    implementation.option)
                    != 0)
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            work.query_rope = work.query_rope_output[0];
            work.key_rope = work.key_rope_output[0];
        }

        if (work.query_rope.elempack != 1
            || work.query_rope.dims != 3
            || work.query_rope.w
                   != static_cast<int>(config.head_dimension)
            || work.query_rope.h != 1
            || work.query_rope.c
                   != static_cast<int>(config.head_count)
            || (!work.fused_qkv_ring
                && (work.key_rope.elempack != 1
                    || work.value_heads.elempack != 1
                    || work.key_rope.dims != 3
                    || work.key_rope.w
                           != static_cast<int>(config.head_dimension)
                    || work.key_rope.h != 1
                    || work.key_rope.c
                           != static_cast<int>(config.kv_head_count)
                    || work.value_heads.dims != 3
                    || work.value_heads.w
                           != static_cast<int>(config.head_dimension)
                    || work.value_heads.h != 1
                    || work.value_heads.c
                           != static_cast<int>(config.kv_head_count)
                    || work.key_rope.elemsize != activation_element_size
                    || work.value_heads.elemsize != activation_element_size)))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }
        if (!work.fused_qkv_ring
            && !record_attention_ring_append(
                implementation.ring_append_pipeline.get(),
                work.key_rope,
                work.value_heads,
                work.next_cache->key,
                work.next_cache->value,
                ring_capacity,
                append_slot,
                command))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }

        bool sink_zero_recorded = false;
        if (sink_token_count != 0 && !work.fused_qkv_ring)
        {
            const uint64_t sink_row =
                ring_first_slot + work.actual_token_count;
            if (!record_attention_ring_zero(
                    implementation.ring_zero_pipeline.get(),
                    work.next_cache->key,
                    work.next_cache->value,
                    sink_row,
                    command))
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            sink_zero_recorded = true;
        }
        work.combined_key = attention_ring_view(
            work.next_cache->key,
            ring_first_slot,
            work.destination_count);
        work.combined_value = attention_ring_view(
            work.next_cache->value,
            ring_first_slot,
            work.destination_count);
        if (work.combined_key.empty() || work.combined_value.empty())
            return NcnnVulkanAttentionBatchResult::NotExecuted;

        work.fused_decode_sdpa =
            work.try_decode_sdpa
            && record_attention_decode_sdpa(
                implementation.decode_sdpa_pipeline.get(),
                work.query_rope,
                work.combined_key,
                work.combined_value,
                implementation.attention_sinks,
                config,
                work.destination_count,
                work.attention_matrix,
                command,
                implementation.option.blob_vkallocator);
        if (!work.fused_decode_sdpa)
        {
            if (sink_token_count != 0 && !sink_zero_recorded)
            {
                const uint64_t sink_row =
                    ring_first_slot + work.actual_token_count;
                if (!record_attention_ring_zero(
                        implementation.ring_zero_pipeline.get(),
                        work.next_cache->key,
                        work.next_cache->value,
                        sink_row,
                        command))
                {
                    return NcnnVulkanAttentionBatchResult::NotExecuted;
                }
            }
            if (!work.mask_uploaded)
            {
                if (!record_mapped_activation_upload(
                        work.attention_mask,
                        work.mask_gpu,
                        command,
                        vkdev,
                        implementation.option))
                {
                    return NcnnVulkanAttentionBatchResult::NotExecuted;
                }
                work.mask_uploaded = true;
            }
            ncnn::VkMat sdpa_key = work.combined_key;
            ncnn::VkMat sdpa_value = work.combined_value;
            if (low_precision_kv)
            {
                ncnn::VkMat converted_key;
                ncnn::VkMat converted_value;
                vkdev->convert_packing(
                    work.combined_key,
                    converted_key,
                    1,
                    1,
                    command,
                    implementation.kv_option);
                vkdev->convert_packing(
                    work.combined_value,
                    converted_value,
                    1,
                    1,
                    command,
                    implementation.kv_option);
                if (converted_key.empty() || converted_value.empty()
                    || converted_key.elemsize != sizeof(float)
                    || converted_value.elemsize != sizeof(float))
                {
                    return NcnnVulkanAttentionBatchResult::NotExecuted;
                }
                work.retained_gpu.push_back(converted_key);
                work.retained_gpu.push_back(converted_value);
                sdpa_key = converted_key;
                sdpa_value = converted_value;
            }
            std::vector<ncnn::VkMat> sdpa_input = {
                work.query_rope,
                sdpa_key,
                sdpa_value,
                work.mask_gpu};
            work.sdpa_output.resize(1);
            if (implementation.sdpa->forward(
                    sdpa_input,
                    work.sdpa_output,
                    command,
                    implementation.option)
                    != 0
                || implementation.permute_heads_tokens->forward(
                    work.sdpa_output[0],
                    work.attention_token_major,
                    command,
                    implementation.option)
                    != 0
                || implementation.reshape_attention->forward(
                    work.attention_token_major,
                    work.attention_matrix,
                    command,
                    implementation.option)
                    != 0)
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
        }

        if (query_key_norm_and_gate)
        {
            if (work.attention_matrix.elempack != 1)
            {
                work.retained_gpu.push_back(work.attention_matrix);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.attention_matrix,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.attention_matrix = unpacked;
            }
            if (!record_attention_output_gate(
                    implementation.output_gate_pipeline.get(),
                    work.attention_matrix,
                    work.output_gate,
                    1,
                    query_columns,
                    command))
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
        }

        if (query_key_norm_and_gate)
        {
            if (!projection_bfloat16
                || !projection_bfloat16->pipeline
                || work.attention_matrix.empty()
                || work.attention_matrix.dims != 2
                || work.attention_matrix.w
                       != static_cast<int>(
                           projection_bfloat16->input_columns)
                 || work.attention_matrix.h != 1
                 || work.attention_matrix.elempack != 1
                 || work.attention_matrix.elemsize != sizeof(float)
                || projection_bfloat16->output_columns
                       != config.hidden_size)
            {
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            }
            work.projected_gpu.create(
                static_cast<int>(projection_bfloat16->output_columns),
                1,
                sizeof(float),
                implementation.vulkan_context->blob_allocator());
            if (work.projected_gpu.empty())
                return NcnnVulkanAttentionBatchResult::NotExecuted;
            const std::vector<ncnn::VkMat> bindings = {
                work.attention_matrix,
                projection_bfloat16->packed,
                projection_bfloat16->bias,
                work.projected_gpu};
            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].u32 = projection_bfloat16->input_columns;
            constants[1].u32 = projection_bfloat16->output_columns;
            constants[2].u32 = projection_bfloat16->block_count;
            constants[3].u32 = 1;
            ncnn::VkMat dispatcher;
            dispatcher.w = static_cast<int>(
                projection_bfloat16->output_columns * 32);
            dispatcher.h = 1;
            dispatcher.c = 1;
            command.record_pipeline(
                projection_bfloat16->pipeline.get(),
                bindings,
                constants,
                dispatcher);
        }
        else if (!projection || !projection->layer
                 || projection->layer->forward(
                        work.attention_matrix,
                        work.projected_gpu,
                        command,
                        implementation.option)
                        != 0)
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }

        std::vector<ncnn::VkMat> add_input = {
            work.input_gpu,
            work.projected_gpu};
        work.add_output.resize(1);
        if (implementation.add->forward(
                add_input,
                work.add_output,
                command,
                implementation.option)
            != 0)
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }
        work.download_gpu = work.add_output[0];
        if (work.download_gpu.elempack != 1)
        {
            work.retained_gpu.push_back(work.download_gpu);
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                work.download_gpu,
                unpacked,
                1,
                command,
                implementation.option);
            work.download_gpu = unpacked;
        }
        if (!record_prepared_activation_staging_download(
                work.download_gpu,
                1,
                config.hidden_size,
                work.download,
                command,
                vkdev,
                implementation.option,
                work.entry->output->dtype()))
        {
            return NcnnVulkanAttentionBatchResult::NotExecuted;
        }
    }

    const auto mark_device_states_unknown = [&]() noexcept {
        for (const PreparedAttentionEntry& work : prepared)
            work.entry->cache->vulkan_attention_state_unknown = true;
    };
    const auto execution_started = std::chrono::steady_clock::now();
    if (submit_compute_and_wait(command, runtime_state) != 0)
    {
        mark_device_states_unknown();
        return NcnnVulkanAttentionBatchResult::Failed;
    }
    const uint64_t execution_microseconds = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    const uint64_t per_entry_execution_microseconds =
        execution_microseconds / prepared.size();
    for (PreparedAttentionEntry& work : prepared)
    {
        if (!copy_staging_to_cpu_batch(
                work.download,
                *work.entry->output))
        {
            mark_device_states_unknown();
            return NcnnVulkanAttentionBatchResult::Failed;
        }
        for (uint32_t column = 0;
             column < work.entry->output->columns();
             ++column)
        {
            if (!std::isfinite(work.entry->output->row(0)[column]))
            {
                mark_device_states_unknown();
                return NcnnVulkanAttentionBatchResult::Failed;
            }
        }
        if (work.adaptive_decode_sdpa)
        {
            implementation.observe_decode_sdpa(
                config.head_dimension,
                config.head_count,
                config.kv_head_count,
                work.destination_count,
                work.fused_decode_sdpa,
                per_entry_execution_microseconds);
        }
    }

    for (PreparedAttentionEntry& work : prepared)
    {
        CpuLayerCache& cache = *work.entry->cache;
        const uint64_t previous_start = cache.start_position;
        cache.start_position = previous_start + work.dropped_tokens;
        cache.token_count = work.retained_tokens;
        cache.first_slot = work.next_first_slot;
        cache.capacity_tokens =
            work.retained_tokens == 0 ? 0 : cache.capacity_tokens;
        cache.columns = config.kv_head_count * config.head_dimension;
        cache.dtype = config.kv_cache_dtype;
        cache.vulkan_attention_cache =
            work.retained_tokens == 0 ? nullptr : work.next_cache;
        cache.device_allocated_bytes = work.allocated_cache_bytes;
        cache.vulkan_attention_state_unknown = false;
    }
    const PreparedAttentionEntry& representative = prepared.front();
    runtime_state.dispatches += 2;
    ++runtime_state.attention_blocks;
    if (representative.fused_qkv_rope)
        ++runtime_state.attention_qkv_rope_fusions;
    if (representative.device_rope)
        ++runtime_state.attention_device_rope_fusions;
    if (representative.fused_qkv_ring)
        ++runtime_state.attention_qkv_ring_fusions;
    if (representative.fused_decode_sdpa)
        ++runtime_state.attention_decode_sdpa_fusions;
    ++runtime_state.kv_ring_appends;
    if (representative.ring_first_slot
            + representative.destination_count
        > representative.entry->cache->capacity_tokens)
    {
        ++runtime_state.kv_ring_wrapped_views;
    }
    runtime_state.auxiliary_uploads +=
        (representative.mask_uploaded ? 1 : 0)
        + (representative.device_rope ? 0 : 2);
    runtime_state.auxiliary_upload_bytes +=
        (representative.device_rope
             ? 0
             : representative.rope_cosine.buffer_capacity()
                   + representative.rope_sine.buffer_capacity())
        + (representative.mask_uploaded
               ? representative.attention_mask.buffer_capacity()
               : 0);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return NcnnVulkanAttentionBatchResult::Executed;
#else
    (void)entries;
    return NcnnVulkanAttentionBatchResult::NotExecuted;
#endif
}

} // namespace moe
} // namespace ncnn
