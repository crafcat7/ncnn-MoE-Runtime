#ifndef NCNN_MOE_BACKENDS_NCNN_CORE_CONTEXT_H
#define NCNN_MOE_BACKENDS_NCNN_CORE_CONTEXT_H

#include "backends/ncnn/vulkan.h"
#include "ncnn/moe/option.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <option.h>
#include <pipeline.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#endif

namespace ncnn {
class Layer;
namespace moe {

inline constexpr uint64_t vulkan_weight_block_size = 64 * 1024;

[[nodiscard]] inline uint64_t align_up(uint64_t value, uint64_t align) noexcept
{
    return (value + align - 1) / align * align;
}

[[nodiscard]] inline uint64_t vulkan_kv_size(DType dtype, uint32_t device_index) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (dtype == DType::BFloat16 || dtype == DType::Float16)
    {
        const int count = ncnn::get_gpu_count();
        if (count <= 0 || device_index >= static_cast<uint32_t>(count))
            return UINT64_MAX;
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(static_cast<int>(device_index));
        if (dtype == DType::BFloat16 ? info.support_bf16_storage() : info.support_fp16_storage())
            return sizeof(uint16_t);
    }
#else
    (void)dtype;
    (void)device_index;
#endif
    return sizeof(float);
}

// Devices without BF16 storage use FP16 instead of widening KV cache to FP32.
[[nodiscard]] inline DType kv_device_dtype(DType dtype, uint32_t device_index) noexcept
{
    if (dtype == DType::BFloat16 && vulkan_kv_size(dtype, device_index) != sizeof(uint16_t))
        return vulkan_kv_size(DType::Float16, device_index) == sizeof(uint16_t) ? DType::Float16 : DType::Float32;
    return dtype;
}

#if NCNN_MOE_WITH_VULKAN

class ActivationBuffer;
class VkGraphAllocator;
class VkTransferPool;
class VkComputeBatch;
class VulkanContext;
struct TensorData;

struct VulkanContextKey
{
    uint32_t device_index = 0;
    uint64_t opt_flags = 0;

    [[nodiscard]] bool operator==(const VulkanContextKey& other) const noexcept
    {
        return device_index == other.device_index && opt_flags == other.opt_flags;
    }
};

struct VulkanContextKeyHash
{
    [[nodiscard]] size_t operator()(const VulkanContextKey& key) const noexcept
    {
        const size_t h1 = std::hash<uint32_t>{}(key.device_index);
        const size_t h2 = std::hash<uint64_t>{}(key.opt_flags);
        return h1 ^ (h2 + static_cast<size_t>(0x9e3779b9u) + (h1 * 64u) + (h1 >> 2));
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
    AtomicCounter attention_cpu_fallbacks;
    AtomicCounter shared_expert_swiglu_fusions;
    AtomicCounter gated_delta_fusions;
    AtomicCounter gated_delta_submissions;
    AtomicCounter rms_norm_linear_fusions;
    AtomicCounter kv_ring_appends;
    AtomicCounter kv_ring_resizes;
    AtomicCounter kv_ring_wrapped_views;
    AtomicCounter dispatches;
    AtomicCounter int4_dispatches;
    AtomicCounter attention_blocks;
};

class VulkanRuntime
{
public:
    VulkanRuntime() noexcept = default;

private:
    friend class VulkanContext;
    friend VulkanStatistics get_vulkan_statistics(const VulkanRuntimePtr& runtime) noexcept;

    mutable std::mutex init_mutex;
    bool init_attempted = false;
    bool instance_ready = false;
    VulkanRuntimeState state;
    mutable std::mutex ctx_mutex;
    std::unordered_map<VulkanContextKey, std::weak_ptr<VulkanContext>, VulkanContextKeyHash> contexts;
};

struct VkTransferSlot
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

class VkTransferLease
{
public:
    VkTransferLease(VkTransferSlot& slot, std::unique_lock<std::mutex> lock);

    VkTransferLease(const VkTransferLease&) = delete;
    VkTransferLease& operator=(const VkTransferLease&) = delete;
    VkTransferLease(VkTransferLease&&) noexcept = default;
    VkTransferLease& operator=(VkTransferLease&&) noexcept = default;

    [[nodiscard]] VkTransferSlot& slot() const noexcept
    {
        return *transfer_slot;
    }

private:
    VkTransferSlot* transfer_slot = nullptr;
    std::unique_lock<std::mutex> lock;
};

class VulkanContext
{
public:
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    ~VulkanContext();

    [[nodiscard]] static std::shared_ptr<VulkanContext> acquire(uint32_t device_index,
                                                                const VulkanRuntimePtr& runtime,
                                                                uint64_t opt_flags);
    [[nodiscard]] static std::shared_ptr<VulkanContext> find(uint32_t device_index,
                                                             const VulkanRuntimePtr& runtime,
                                                             uint64_t opt_flags);

    [[nodiscard]] ncnn::VulkanDevice* device() const noexcept
    { return vkdev; }
    [[nodiscard]] ncnn::VkAllocator* blob_allocator() const noexcept
    { return blob_alloc; }
    [[nodiscard]] ncnn::VkAllocator* staging_allocator() const noexcept
    { return staging_alloc; }
    [[nodiscard]] ncnn::VkAllocator* kv_allocator() const noexcept
    { return kv_alloc.get(); }
    [[nodiscard]] uint64_t optimization_flags() const noexcept
    { return flags; }
    [[nodiscard]] const VulkanRuntimePtr& runtime() const noexcept
    { return vulkan_runtime; }
    [[nodiscard]] VulkanRuntimeState& runtime_state() noexcept
    { return vulkan_runtime->state; }
    [[nodiscard]] std::mutex& command_mutex() noexcept
    { return command_lock; }
    [[nodiscard]] VkTransferPool& transfer_pool() noexcept
    { return *weight_transfer_pool; }

    [[nodiscard]] bool support_direct_host_buffer(size_t size, DType dtype) const noexcept
    {
        // UMA direct binding avoids staging copies for small FP32 inputs.
        return dtype == DType::Float32 && size <= 64 * 1024 && vkdev && vkdev->info.type() > 0;
    }

    [[nodiscard]] bool graph_arena_sealed() const noexcept;
    [[nodiscard]] bool reserve_graph_compute(uint64_t bytes);
    [[nodiscard]] VulkanGraphArenaSize graph_compute_physical_size() const;

    // Persistent device-local scratch reused under command_mutex().
    // h == 0 yields a 1D view of w elements; h > 0 yields a 2D (w, h) view.
    [[nodiscard]] ncnn::VkMat cm2_scratch(int w, int h, size_t elemsize);

    [[nodiscard]] std::shared_ptr<const std::vector<uint32_t>> shader_binary(const char* src,
                                                                             int len,
                                                                             const ncnn::Option& opt,
                                                                             uint64_t variant);
    [[nodiscard]] std::shared_ptr<ncnn::Pipeline> find_pipeline(const char* src, uint64_t variant) const;
    [[nodiscard]] std::shared_ptr<ncnn::Pipeline> cache_pipeline(const char* src, uint64_t variant,
                                                                 std::shared_ptr<ncnn::Pipeline> pipe);

    [[nodiscard]] VkTransferLease acquire_transfer_slot();

private:
    struct ShaderKey
    {
        const char* src = nullptr;
        uint64_t variant = 0;

        [[nodiscard]] bool operator==(const ShaderKey& other) const noexcept
        {
            return src == other.src && variant == other.variant;
        }
    };

    struct ShaderKeyHash
    {
        [[nodiscard]] size_t operator()(const ShaderKey& key) const noexcept
        {
            const size_t h1 = std::hash<const void*>{}(static_cast<const void*>(key.src));
            const size_t h2 = std::hash<uint64_t>{}(key.variant);
            return h1 ^ (h2 + static_cast<size_t>(0x9e3779b9u) + (h1 * 64u) + (h1 >> 2));
        }
    };

    friend class VkComputeBatch;
    friend std::shared_ptr<ncnn::Layer> own_vk_layer(const std::shared_ptr<VulkanContext>& ctx,
                                                     ncnn::Layer* layer,
                                                     const ncnn::Option& opt);

    VulkanContext(ncnn::VulkanDevice* device, VulkanRuntimePtr runtime, uint64_t opt_flags);
    void release_resources() noexcept;
    void retire(std::function<void()> fn) noexcept;
    void drain_retired() noexcept;

    // Reused under command_mutex(); each batch drains before releasing the lock.
    std::vector<std::unique_ptr<ncnn::VkCompute>> ordered_commands;

    ncnn::VulkanDevice* vkdev = nullptr;
    VulkanRuntimePtr vulkan_runtime;
    uint64_t flags = OptimizationDefaultFlags;
    ncnn::VkAllocator* blob_alloc = nullptr;
    std::unique_ptr<VkGraphAllocator> graph_allocator;
    ncnn::VkAllocator* staging_alloc = nullptr;
    // Growable device-local pool for long-lived KV rings and CM2 scratch.
    std::unique_ptr<ncnn::VkBlobAllocator> kv_alloc;
    ncnn::VkMat cm2_scratch_buffer;

    std::mutex command_lock;
    std::array<VkTransferSlot, 2> transfer_slots;
    std::atomic<size_t> next_transfer_slot{0};
    std::unique_ptr<VkTransferPool> weight_transfer_pool;

    mutable std::mutex pipeline_cache_mutex;
    std::unordered_map<ShaderKey, std::shared_ptr<const std::vector<uint32_t>>, ShaderKeyHash> shader_binaries;
    std::unordered_map<ShaderKey, std::shared_ptr<ncnn::Pipeline>, ShaderKeyHash> pipelines;

    std::mutex retire_lock;
    std::atomic<bool> has_retired{false};
    std::vector<std::function<void()>> retired;
};

// --- Common inline primitives ------------------------------------------------

[[nodiscard]] inline size_t activation_variant(const ncnn::Option& opt) noexcept
{
    if (opt.use_fp16_storage)
        return 1;
    if (opt.use_bf16_storage)
        return 2;
    return 0;
}

[[nodiscard]] inline size_t act_element_size(const ncnn::Option& opt) noexcept
{
    return activation_variant(opt) == 0 ? sizeof(float) : sizeof(uint16_t);
}

[[nodiscard]] inline ncnn::Option make_vk_opt(const VulkanContext& ctx)
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_int8_storage = false;
    opt.use_int8_packed = false;
    opt.use_int8_arithmetic = false;
    opt.use_subgroup_ops = ctx.device()->info.support_subgroup_ops() != 0;
    opt.blob_vkallocator = ctx.blob_allocator();
    opt.workspace_vkallocator = ctx.blob_allocator();
    opt.staging_vkallocator = ctx.staging_allocator();
    return opt;
}

inline void rearm_read(const ncnn::VkMat& mat) noexcept
{
    if (!mat.empty() && mat.data)
    {
        mat.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        mat.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
}

[[nodiscard]] inline ncnn::VkMat row_view(const ncnn::VkMat& src, size_t first_row, size_t rows)
{
    if (src.empty() || src.dims != 2 || rows == 0
#if NCNN_BATCH
        || src.n != 1
#endif
        || first_row > static_cast<size_t>(src.h)
        || rows > static_cast<size_t>(src.h) - first_row)
    {
        return {};
    }

    ncnn::VkMat view = src;
    view.h = static_cast<int>(rows);
    view.offset += first_row * static_cast<size_t>(src.w) * src.elemsize;
    view.cstep = static_cast<size_t>(src.w) * rows;
#if NCNN_BATCH
    view.nstep = view.cstep;
#endif
    return view;
}

inline ncnn::VkMat bind_host_in(ncnn::VkMat& staging)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return staging;
}

inline ncnn::VkMat bind_host_out(ncnn::VkMat& staging)
{
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return staging;
}

// --- Command wait, staging, and pipeline utilities ---------------------------

// Retries VkCompute::wait() until completion or confirmed VK_ERROR_DEVICE_LOST.
int wait_cmd(ncnn::VkCompute& cmd, const ncnn::VulkanDevice* probe_device,
             VkQueue borrowed_queue = VK_NULL_HANDLE);

int submit_wait(ncnn::VkCompute& cmd, const ncnn::VulkanDevice* probe_device);

// Drains every queue in a family during legacy VkTransfer error recovery.
void drain_queue_family(const ncnn::VulkanDevice* device, uint32_t family,
                        std::span<VkQueue> queues) noexcept;

bool prepare_staging_batch(ncnn::VkMat& buf, size_t rows, uint32_t cols,
                           ncnn::VkAllocator* alloc, size_t elem_size = sizeof(float));

bool record_mapped_upload(ncnn::VkMat& staging, ncnn::VkMat& dst,
                          ncnn::VkCompute& cmd, const ncnn::Option& opt);

bool record_mapped_act_upload(ncnn::VkMat& staging, ncnn::VkMat& dst,
                              ncnn::VkCompute& cmd, ncnn::VulkanDevice* device,
                              const ncnn::Option& opt, DType src_dtype = DType::Float32);

bool fill_staging_upload(const ActivationBuffer& input, ncnn::VkMat& staging,
                         ncnn::VkAllocator* alloc);

bool fill_staging_values(const void* src, size_t count, size_t elem_size,
                         ncnn::VkMat& staging, ncnn::VkAllocator* alloc);

bool record_prepared_staging_upload(const ncnn::VkMat& staging, size_t rows,
                                    ncnn::VkMat& dst, ncnn::VkCompute& cmd,
                                    ncnn::VulkanDevice* device, const ncnn::Option& opt,
                                    DType src_dtype = DType::Float32);

bool record_staging_download(const ncnn::VkMat& src, size_t rows, uint32_t cols,
                             ncnn::VkMat& staging, ncnn::VkCompute& cmd,
                             ncnn::VulkanDevice* device, const ncnn::Option& opt,
                             DType dst_dtype = DType::Float32);

bool copy_staging_to_cpu_batch(ncnn::VkMat& staging, ActivationBuffer& output);

// Distinct outputs required; only same-index input/output alias is permitted.
bool copy_staging_to_cpu_batches(ncnn::VkMat& staging,
                                 std::span<const ActivationBuffer*> inputs,
                                 std::span<ActivationBuffer*> outputs,
                                 uint32_t cols);

bool prepare_float_upload(const TensorData& src, ncnn::Mat& dst);

[[nodiscard]] std::string prefix_shader(const char* src, int len, const char* defines);

// Compiles and caches a compute pipeline owned by VulkanContext.
[[nodiscard]] bool create_pipeline(const std::shared_ptr<VulkanContext>& ctx,
                                   const char* src,
                                   int len,
                                   const ncnn::Option& opt,
                                   uint64_t shader_variant,
                                   uint64_t pipe_variant,
                                   int lx,
                                   std::shared_ptr<ncnn::Pipeline>& dst,
                                   const std::vector<ncnn::vk_specialization_type>& specs = {},
                                   int subgroup_size = 0,
                                   int ly = 1,
                                   int lz = 1,
                                   bool lock_create = false);

// Variant that compiles from prefixed GLSL source while keying cache on base_src.
[[nodiscard]] bool create_pipeline_prefixed(const std::shared_ptr<VulkanContext>& ctx,
                                            const char* base_src,
                                            int base_len,
                                            const char* defines,
                                            const ncnn::Option& opt,
                                            uint64_t shader_variant,
                                            uint64_t pipe_variant,
                                            int lx,
                                            std::shared_ptr<ncnn::Pipeline>& dst,
                                            const std::vector<ncnn::vk_specialization_type>& specs = {},
                                            int subgroup_size = 0,
                                            int ly = 1,
                                            int lz = 1);

// Wraps a Vulkan ncnn::Layer so destruction defers to VulkanContext teardown
// or idle batch reset instead of locking command_mutex() in a deleter.
[[nodiscard]] std::shared_ptr<ncnn::Layer> own_vk_layer(const std::shared_ptr<VulkanContext>& ctx,
                                                        ncnn::Layer* layer,
                                                        const ncnn::Option& opt);

void compute_yarn_inv_freq(uint32_t rope_dim, uint32_t active_dim, float theta,
                           float scale, float ntk_alpha, float ntk_beta,
                           uint32_t orig_ctx, float* out) noexcept;

#else

class VulkanRuntime
{
public:
    VulkanRuntime() noexcept = default;
};

#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_BACKENDS_NCNN_CORE_CONTEXT_H
