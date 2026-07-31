#include "ncnn_linear.h"

#include "engine/expert_backend.h"
#include "engine/cpu_session_state.h"
#include "kernels/cpu_float8.h"
#include "kernels/cpu_ops.h"
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

#if NCNN_MOE_USE_NCNN
static constexpr uint64_t max_ncnn_linear_weight_bytes = 64ull * 1024ull * 1024ull;
#endif

#if NCNN_MOE_WITH_VULKAN
static thread_local uint64_t current_vulkan_dispatch_count = 0;
static thread_local uint64_t current_vulkan_attention_block_count = 0;
static thread_local NcnnVulkanRuntimeCounters current_vulkan_runtime_counters;

static int submit_compute_and_wait(ncnn::VkCompute& command)
{
    const auto started = std::chrono::steady_clock::now();
    const int result = command.submit_and_wait();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started);
    current_vulkan_runtime_counters.submit_wait_time_microseconds +=
        static_cast<uint64_t>(elapsed.count());
    return result;
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
    ncnn::VkMat rope_cosine;
    ncnn::VkMat rope_sine;
    ncnn::VkMat attention_mask;
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
    NcnnVulkanContext(const NcnnVulkanContext&) = delete;
    NcnnVulkanContext& operator=(const NcnnVulkanContext&) = delete;

    ~NcnnVulkanContext() = default;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanContext> acquire(uint32_t requested_device_index = automatic_vulkan_device_index)
    {
        static std::mutex creation_mutex;
        static bool creation_attempted = false;
        static bool instance_ready = false;
        static std::vector<std::shared_ptr<NcnnVulkanContext>> contexts;
        const std::lock_guard<std::mutex> lock(creation_mutex);
        if (!creation_attempted)
        {
            creation_attempted = true;
#if defined(__APPLE__) && defined(NCNN_MOE_MOLTENVK_LIBRARY_PATH)
            instance_ready = ncnn::create_gpu_instance(NCNN_MOE_MOLTENVK_LIBRARY_PATH) == 0 && ncnn::get_gpu_count() > 0;
#else
            instance_ready = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
#endif
        }
        if (!instance_ready)
            return {};

        const uint32_t device_index = requested_device_index == automatic_vulkan_device_index ? static_cast<uint32_t>(ncnn::get_default_gpu_index()) : requested_device_index;
        if (device_index >= static_cast<uint32_t>(ncnn::get_gpu_count()))
        {
            return {};
        }
        if (contexts.size() < static_cast<size_t>(ncnn::get_gpu_count()))
        {
            contexts.resize(static_cast<size_t>(ncnn::get_gpu_count()));
        }
        if (contexts[device_index])
            return contexts[device_index];

        ncnn::VulkanDevice* device = ncnn::get_gpu_device(static_cast<int>(device_index));
        if (!device)
            return {};
        contexts[device_index].reset(new NcnnVulkanContext(device));
        return contexts[device_index];
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
            ++current_vulkan_runtime_counters.staging_slot_contentions;
            lock.lock();
        }
        ++current_vulkan_runtime_counters.staging_slot_acquisitions;
        return NcnnVulkanTransferLease(slot, std::move(lock));
    }

    [[nodiscard]] bool choose_decode_sdpa(uint32_t head_dimension, uint32_t head_count, uint32_t key_value_head_count, uint64_t destination_count)
    {
        const uint64_t key = decode_sdpa_key(head_dimension, head_count, key_value_head_count, destination_count);
        const std::lock_guard<std::mutex> lock(decode_sdpa_mutex_);
        DecodeSdpaPolicy& policy = decode_sdpa_policies_[key];
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

    void observe_decode_sdpa(uint32_t head_dimension, uint32_t head_count, uint32_t key_value_head_count, uint64_t destination_count, bool fused,
                             uint64_t elapsed_microseconds)
    {
        const uint64_t key = decode_sdpa_key(head_dimension, head_count, key_value_head_count, destination_count);
        const std::lock_guard<std::mutex> lock(decode_sdpa_mutex_);
        DecodeSdpaPolicy& policy = decode_sdpa_policies_[key];
        double& estimate = fused ? policy.fused_microseconds : policy.ncnn_microseconds;
        uint64_t& samples = fused ? policy.fused_samples : policy.ncnn_samples;
        estimate = samples == 0 ? static_cast<double>(elapsed_microseconds) : estimate * 0.75 + static_cast<double>(elapsed_microseconds) * 0.25;
        ++samples;
        const bool initial_comparison = !policy.preference_initialized && policy.fused_samples >= 2 && policy.ncnn_samples >= 2;
        const bool probe_comparison = policy.probe_pending && policy.probe_fused == fused;
        if (initial_comparison || probe_comparison)
        {
            if (policy.prefer_fused)
            {
                if (policy.fused_microseconds > policy.ncnn_microseconds * 1.02)
                {
                    policy.prefer_fused = false;
                }
            }
            else if (policy.fused_microseconds * 1.02 < policy.ncnn_microseconds)
            {
                policy.prefer_fused = true;
            }
            policy.preference_initialized = true;
            policy.probe_pending = false;
        }
    }

private:
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

    [[nodiscard]] static uint64_t decode_sdpa_key(uint32_t head_dimension, uint32_t head_count, uint32_t key_value_head_count,
                                                  uint64_t destination_count) noexcept
    {
        uint64_t context_bucket = 0;
        uint64_t upper_bound = 16;
        while (upper_bound < destination_count && context_bucket < 15)
        {
            upper_bound <<= 1;
            ++context_bucket;
        }
        return static_cast<uint64_t>(head_dimension) << 40 | static_cast<uint64_t>(head_count) << 24 | static_cast<uint64_t>(key_value_head_count) << 8
               | context_bucket;
    }

    explicit NcnnVulkanContext(ncnn::VulkanDevice* device)
        : device_(device), blob_allocator_(device->acquire_blob_allocator()), staging_allocator_(device->acquire_staging_allocator())
    {
        for (NcnnVulkanTransferSlot& slot : transfer_slots_)
            slot.staging_allocator = device->acquire_staging_allocator();
        for (NcnnVulkanTransferSlot& slot : transfer_slots_)
            slot.command = new ncnn::VkCompute(device);
    }

    // ncnn owns Vulkan teardown through atexit.

    ncnn::VulkanDevice* device_ = nullptr;
    ncnn::VkAllocator* blob_allocator_ = nullptr;
    ncnn::VkAllocator* staging_allocator_ = nullptr;
    std::mutex command_mutex_;
    // Staging slots require independent allocators while commands are in flight.
    std::array<NcnnVulkanTransferSlot, 2> transfer_slots_;
    std::atomic<size_t> next_transfer_slot_{0};
    std::mutex decode_sdpa_mutex_;
    std::unordered_map<uint64_t, DecodeSdpaPolicy> decode_sdpa_policies_;
};

class VulkanExpertVictimCache final : public IExpertVictimCache
{
public:
    VulkanExpertVictimCache(std::shared_ptr<NcnnVulkanContext> context, uint64_t capacity_bytes)
        : context_(std::move(context)),
          capacity_bytes_(capacity_bytes),
          maximum_pending_bytes_(std::min(capacity_bytes, UINT64_C(256) * 1024 * 1024)),
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
            static_cast<uint32_t>(context_->device()->info.device_index()), entry->data, entry->execution.activation);
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
        ncnn::VkCompute command(context_->device());
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
            ncnn::VkCompute command(context_->device());
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

static bool has_batch_shape(const ncnn::VkMat& buffer, size_t rows, uint32_t columns)
{
    return buffer.dims == 2 && buffer.w == static_cast<int>(columns) && buffer.h == static_cast<int>(rows) && buffer.elemsize == sizeof(float)
           && buffer.elempack == 1;
}

static bool prepare_staging_batch(ncnn::VkMat& buffer, size_t rows, uint32_t columns, ncnn::VkAllocator* allocator)
{
    const bool reused = has_batch_shape(buffer, rows, columns);
    buffer.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++current_vulkan_runtime_counters.staging_slot_reuses;
    else
        ++current_vulkan_runtime_counters.staging_slot_resizes;
    return true;
}

static bool prepare_staging_matrix(ncnn::VkMat& buffer, int width, int height, size_t element_size, ncnn::VkAllocator* allocator)
{
    const bool reused = buffer.dims == 2 && buffer.w == width && buffer.h == height && buffer.elemsize == element_size && buffer.elempack == 1;
    buffer.create(width, height, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++current_vulkan_runtime_counters.staging_slot_reuses;
    else
        ++current_vulkan_runtime_counters.staging_slot_resizes;
    return true;
}

static bool prepare_staging_tensor(ncnn::VkMat& buffer, int width, int height, int channels, size_t element_size, ncnn::VkAllocator* allocator)
{
    const bool reused = buffer.dims == 3 && buffer.w == width && buffer.h == height && buffer.c == channels && buffer.elemsize == element_size && buffer.elempack == 1;
    buffer.create(width, height, channels, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++current_vulkan_runtime_counters.staging_slot_reuses;
    else
        ++current_vulkan_runtime_counters.staging_slot_resizes;
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

static bool fill_staging_upload(const CpuBatch& input, ncnn::VkMat& staging, ncnn::VkAllocator* allocator)
{
    if (!prepare_staging_batch(staging, input.rows(), input.columns(), allocator))
        return false;

    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty())
        return false;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        std::copy_n(input.row(row_index), input.columns(), mapped.row<float>(static_cast<int>(row_index)));
    }
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static bool record_prepared_staging_upload(const ncnn::VkMat& staging, size_t rows, ncnn::VkMat& destination, ncnn::VkCompute& command,
                                           ncnn::VulkanDevice* device, const ncnn::Option& option)
{
    if (staging.empty() || !staging.mapped_ptr())
        return false;
    const int packed_rows = static_cast<int>(rows);
    const int destination_elempack = packed_rows % 4 == 0 ? 4 : 1;
    int cast_type = 0;
    if (device->info.type() != 0)
    {
        if (option.use_bf16_storage || option.use_bf16_packed)
            cast_type = 5;
        else if (option.use_fp16_storage || option.use_fp16_packed)
            cast_type = 2;
        else
            cast_type = 1;
    }
    device->convert_packing(staging, destination, destination_elempack, cast_type, command, option);
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

static bool copy_staging_to_cpu_batch(ncnn::VkMat& staging, CpuBatch& output)
{
    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.dims != 2 || mapped.w != static_cast<int>(output.columns()) || mapped.h != static_cast<int>(output.rows())
        || mapped.elempack != 1 || mapped.elembits() != 32)
        return false;
    for (size_t row_index = 0; row_index < output.rows(); ++row_index)
    {
        std::copy_n(mapped.row<float>(static_cast<int>(row_index)), output.columns(), output.row(row_index));
    }
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
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
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
#endif
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
    ~Implementation()
    {
        if (vulkan_context)
        {
            const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
            delete qkv_rope_pipeline;
            delete decode_sdpa_pipeline;
            delete ring_append_pipeline;
            delete ring_zero_pipeline;
            qkv_rope_pipeline = nullptr;
            decode_sdpa_pipeline = nullptr;
            ring_append_pipeline = nullptr;
            ring_zero_pipeline = nullptr;
            for (ncnn::Layer* layer : layers)
            {
                layer->destroy_pipeline(option);
                delete layer;
            }
        }
        layers.clear();
        attention_sinks = ncnn::VkMat();
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
    ncnn::Pipeline* qkv_rope_pipeline = nullptr;
    ncnn::Pipeline* decode_sdpa_pipeline = nullptr;
    ncnn::Pipeline* ring_append_pipeline = nullptr;
    ncnn::Pipeline* ring_zero_pipeline = nullptr;
    std::vector<ncnn::Layer*> layers;
    ncnn::Option option;
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    ncnn::VkMat attention_sinks;
#endif
    std::shared_ptr<NcnnLinearOperator> fused_qkv;
    std::shared_ptr<NcnnLinearOperator> output_projection;
    NcnnVulkanAttentionConfig config;
    std::vector<float> sinks;
    std::vector<float> rope_inverse_frequencies;
    float rope_concentration = 1.0f;
};

NcnnLinearOperator::NcnnLinearOperator()
    : implementation_(new Implementation)
{
}

NcnnLinearOperator::~NcnnLinearOperator() = default;

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create(const TensorData& matrix, const TensorData* bias, NcnnLinearDevice device,
                                                               uint32_t vulkan_device_index)
{
#if NCNN_MOE_USE_NCNN
    if (matrix.shape.size() != 2 || (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16))
        return {};

    const uint64_t element_size = matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
    if (device == NcnnLinearDevice::Cpu && matrix.element_count() > max_ncnn_linear_weight_bytes / element_size)
        return {};
    if (matrix.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};

    std::shared_ptr<NcnnLinearOperator> linear(new NcnnLinearOperator);
    Implementation& implementation = *linear->implementation_;
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
        implementation.vulkan_context = NcnnVulkanContext::acquire(vulkan_device_index);
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
    return {};
#endif
}

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create_fused(const std::vector<const TensorData*>& matrices,
                                                                     const std::vector<const TensorData*>& biases, NcnnLinearDevice device,
                                                                     uint32_t vulkan_device_index)
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
    return create(fused_matrix, has_bias ? &fused_bias : nullptr, device, vulkan_device_index);
}

std::shared_ptr<IExpertVictimCache> create_vulkan_victim_cache(uint64_t capacity_bytes, uint32_t vulkan_device_index)
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context = NcnnVulkanContext::acquire(vulkan_device_index);
    if (!context || capacity_bytes == 0)
        return {};
    return std::make_shared<VulkanExpertVictimCache>(context, capacity_bytes);
#else
    (void)capacity_bytes;
    (void)vulkan_device_index;
    return {};
#endif
}

uint32_t NcnnLinearOperator::vulkan_device_count() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context = NcnnVulkanContext::acquire();
    return context ? static_cast<uint32_t>(ncnn::get_gpu_count()) : 0;
#else
    return 0;
#endif
}

uint64_t NcnnLinearOperator::vulkan_heap_budget_bytes() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context = NcnnVulkanContext::acquire();
    if (!context)
        return 0;
    return static_cast<uint64_t>(context->device()->get_heap_budget()) * 1024 * 1024;
#else
    return 0;
#endif
}

std::vector<VulkanDeviceCapabilities> NcnnLinearOperator::vulkan_device_capabilities()
{
    std::vector<VulkanDeviceCapabilities> result;
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context = NcnnVulkanContext::acquire();
    if (!context)
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

uint64_t NcnnLinearOperator::current_thread_vulkan_dispatches() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return current_vulkan_dispatch_count;
#else
    return 0;
#endif
}

NcnnVulkanRuntimeCounters NcnnLinearOperator::current_thread_vulkan_runtime_counters() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return current_vulkan_runtime_counters;
#else
    return {};
#endif
}

bool NcnnLinearOperator::forward(const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_USE_NCNN
    const Implementation& implementation = *implementation_;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context)
    {
        NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
        NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
        if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
            || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator))
            return false;

        output.reset(input.rows(), implementation.output_columns, false);
        std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        ncnn::VkCompute& command = *transfer_slot.command;
        if (transfer_slot.command_used)
        {
            if (command.reset() != 0)
                return false;
            ++current_vulkan_runtime_counters.command_buffer_reuses;
        }
        transfer_slot.command_used = true;
        ncnn::VkMat bottom_gpu;
        if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), bottom_gpu, command, vkdev, implementation.option))
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
        if (!record_prepared_staging_download(download_gpu, input.rows(), implementation.output_columns, transfer_slot.download, command,
                                              implementation.option))
            return false;
        if (submit_compute_and_wait(command) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
            return false;
        ++current_vulkan_dispatch_count;
        ++current_vulkan_runtime_counters.compute_submissions;
        ++current_vulkan_runtime_counters.batch_uploads;
        ++current_vulkan_runtime_counters.batch_downloads;
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
    ncnn::VkMat packed;
    ncnn::VkMat bias;
    ncnn::Option option;
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
            const uint first = packed_words[(weight_row + column) >> 1];
            const uint second = packed_words[(weight_row + column + 2) >> 1];
            sum += decode_bfloat16(first & 65535) * input_data[input_row + column];
            sum += decode_bfloat16(first >> 16) * input_data[input_row + column + 1];
            sum += decode_bfloat16(second & 65535) * input_data[input_row + column + 2];
            sum += decode_bfloat16(second >> 16) * input_data[input_row + column + 3];
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

static bool create_bfloat16_projection_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant = option.use_subgroup_ops ? 1 : 0;
    static std::array<std::once_flag, 2> compile_once;
    static std::array<std::vector<uint32_t>, 2> spirv;
    static std::array<bool, 2> compiled = {false, false};
    std::call_once(compile_once[shader_variant], [&] {
        compiled[shader_variant] =
            ncnn::compile_spirv_module(
                bfloat16_projection_shader,
                static_cast<int>(sizeof(bfloat16_projection_shader) - 1),
                option,
                spirv[shader_variant])
                == 0
            && !spirv[shader_variant].empty();
    });
    if (!compiled[shader_variant])
        return false;

    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<
        const ncnn::VulkanDevice*,
        std::weak_ptr<ncnn::Pipeline>>
        cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv[shader_variant].data(),
            spirv[shader_variant].size() * sizeof(uint32_t),
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
    cache[device] = destination;
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

std::shared_ptr<NcnnVulkanBfloat16Operator>
NcnnVulkanBfloat16Operator::create(
    const TensorData& matrix,
    const TensorData* bias,
    uint32_t vulkan_device_index)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::BFloat16
        || matrix.shape.size() != 2
        || matrix.shape[0] == 0
        || matrix.shape[1] == 0
        || matrix.shape[1] % 128 != 0
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
    implementation.block_count = input_columns / 128;
    implementation.vulkan_context =
        NcnnVulkanContext::acquire(vulkan_device_index);
    if (!implementation.vulkan_context)
        return {};
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    if (device->info.subgroup_size() != 32)
        return {};
    implementation.option.use_vulkan_compute = true;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
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
        + static_cast<uint64_t>(output_columns) * sizeof(float);
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
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator =
        implementation.weight_allocator.get();
    upload_option.workspace_vkallocator =
        implementation.weight_allocator.get();
    upload_option.staging_vkallocator =
        implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(
            packed,
            implementation.packed,
            upload_option);
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
    return {};
#endif
}

std::shared_ptr<NcnnVulkanBfloat16Operator>
NcnnVulkanBfloat16Operator::create_fused(
    const std::vector<const TensorData*>& matrices,
    const std::vector<const TensorData*>& biases,
    uint32_t vulkan_device_index)
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
        vulkan_device_index);
}

bool NcnnVulkanBfloat16Operator::forward(
    const CpuBatch& input,
    CpuBatch& output) const
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

    NcnnVulkanTransferLease transfer_lease =
        implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            implementation.output_columns,
            transfer_slot.staging_allocator))
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
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(
            transfer_slot.upload,
            input_gpu,
            command,
            implementation.option))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    output_gpu.create(
        static_cast<int>(implementation.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = input_gpu;
    bindings[1] = implementation.packed;
    bindings[2] = implementation.bias;
    bindings[3] = output_gpu;
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat dispatcher;
    dispatcher.w =
        static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(
        implementation.pipeline.get(),
        bindings,
        constants,
        dispatcher);
    if (!record_prepared_staging_download(
            output_gpu,
            input.rows(),
            implementation.output_columns,
            transfer_slot.download,
            command,
            implementation.option)
        || submit_compute_and_wait(command) != 0
        || !copy_staging_to_cpu_batch(
            transfer_slot.download,
            output))
    {
        return false;
    }
    ++current_vulkan_dispatch_count;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

class NcnnVulkanFloat8Operator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> swiglu_quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> rms_norm_quantize_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    ncnn::Option option;
#endif
    uint32_t matrix_input_columns = 0;
    uint32_t logical_input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t output_columns_per_group = 0;
    uint32_t block_count = 0;
    uint32_t input_group_count = 1;
    float rms_norm_epsilon = 0.0f;
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
    static std::once_flag compile_once;
    static std::vector<uint32_t> spirv;
    static bool compiled = false;
    std::call_once(compile_once, [&] {
        compiled = ncnn::compile_spirv_module(float8_projection_shader, static_cast<int>(sizeof(float8_projection_shader) - 1), option, spirv) == 0
                   && !spirv.empty();
    });
    if (!compiled)
        return false;

    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<const ncnn::VulkanDevice*, std::weak_ptr<ncnn::Pipeline>> cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    cache[device] = destination;
    return true;
}

static bool create_float8_quantize_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    static std::once_flag compile_once;
    static std::vector<uint32_t> spirv;
    static bool compiled = false;
    std::call_once(compile_once, [&] {
        compiled = ncnn::compile_spirv_module(float8_quantize_shader, static_cast<int>(sizeof(float8_quantize_shader) - 1), option, spirv) == 0
                   && !spirv.empty();
    });
    if (!compiled)
        return false;

    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<const ncnn::VulkanDevice*, std::weak_ptr<ncnn::Pipeline>> cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    cache[device] = destination;
    return true;
}

static bool create_float8_rms_norm_quantize_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    static std::once_flag compile_once;
    static std::vector<uint32_t> spirv;
    static bool compiled = false;
    std::call_once(compile_once, [&] {
        compiled = ncnn::compile_spirv_module(
                       float8_rms_norm_quantize_shader,
                       static_cast<int>(sizeof(float8_rms_norm_quantize_shader) - 1),
                       option,
                       spirv)
                       == 0
                   && !spirv.empty();
    });
    if (!compiled)
        return false;

    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<const ncnn::VulkanDevice*, std::weak_ptr<ncnn::Pipeline>> cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    cache[device] = destination;
    return true;
}

static bool create_float8_swiglu_quantize_pipeline(
    const std::shared_ptr<NcnnVulkanContext>& context,
    const ncnn::Option& option,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    static std::once_flag compile_once;
    static std::vector<uint32_t> spirv;
    static bool compiled = false;
    std::call_once(compile_once, [&] {
        compiled = ncnn::compile_spirv_module(
                       float8_swiglu_quantize_shader,
                       static_cast<int>(sizeof(float8_swiglu_quantize_shader) - 1),
                       option,
                       spirv)
                       == 0
                   && !spirv.empty();
    });
    if (!compiled)
        return false;

    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<const ncnn::VulkanDevice*, std::weak_ptr<ncnn::Pipeline>> cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    cache[device] = destination;
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
    uint32_t vulkan_device_index)
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
    implementation.vulkan_context = NcnnVulkanContext::acquire(vulkan_device_index);
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
    return {};
#endif
}

bool NcnnVulkanFloat8Operator::prepare_rms_norm(const TensorData& weight, float epsilon)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context || epsilon <= 0.0f
        || weight.shape != std::vector<uint32_t>{implementation.output_columns}
        || (weight.dtype != DType::Float32 && weight.dtype != DType::BFloat16))
    {
        return false;
    }
    ncnn::Mat values;
    values.create(static_cast<int>(implementation.output_columns), sizeof(float));
    if (values.empty())
        return false;
    float* destination = static_cast<float*>(values.data);
    if (weight.dtype == DType::Float32)
    {
        const std::span<const float> source = weight.float32_values();
        std::copy(source.begin(), source.end(), destination);
    }
    else
    {
        const std::span<const uint16_t> source = weight.bfloat16_values();
        for (uint32_t index = 0; index < implementation.output_columns; ++index)
            destination[index] = bfloat16_to_float(source[index]);
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
        command.record_upload(values, implementation.rms_norm_weight, upload_option);
        uploaded = !implementation.rms_norm_weight.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return false;
    implementation.rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward(const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context || !implementation.pipeline || input.rows() == 0
        || input.columns() != implementation.logical_input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    CpuBatch quantized_input = input;
    for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
        quantize_float8_e4m3_inplace(
            quantized_input.row(token_index),
            implementation.logical_input_columns,
            128,
            true);
    if (!fill_staging_upload(quantized_input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), implementation.output_columns);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
        return false;
    ncnn::VkMat output_gpu;
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
    command.record_pipeline(implementation.pipeline.get(), bindings, constants, dispatcher);
    if (!record_prepared_staging_download(
            output_gpu,
            input.rows(),
            implementation.output_columns,
            transfer_slot.download,
            command,
            implementation.option)
        || submit_compute_and_wait(command) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++current_vulkan_dispatch_count;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_chain(const CpuBatch& input, const NcnnVulkanFloat8Operator& next, CpuBatch& output) const
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
    NcnnVulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    CpuBatch quantized_input = input;
    for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
        quantize_float8_e4m3_inplace(quantized_input.row(token_index), first.logical_input_columns, 128, true);
    if (!fill_staging_upload(quantized_input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), second.output_columns);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
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
    command.record_pipeline(first.pipeline.get(), first_bindings, first_constants, first_dispatcher);

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
    command.record_pipeline(second.pipeline.get(), second_bindings, second_constants, second_dispatcher);

    if (!record_prepared_staging_download(output_gpu, input.rows(), second.output_columns, transfer_slot.download, command, first.option)
        || submit_compute_and_wait(command) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    current_vulkan_dispatch_count += 2;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
    return true;
#else
    (void)input;
    (void)next;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_rms_norm_chain(
    const CpuBatch& input,
    const NcnnVulkanFloat8Operator& next,
    CpuBatch& output) const
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

    NcnnVulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    CpuBatch quantized_input = input;
    for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
        quantize_float8_e4m3_inplace(quantized_input.row(token_index), first.logical_input_columns, 128, true);
    if (!fill_staging_upload(quantized_input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), second.output_columns);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
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
    command.record_pipeline(first.pipeline.get(), first_bindings, first_constants, first_dispatcher);

    std::vector<ncnn::VkMat> norm_bindings = {intermediate_gpu, first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline(first.rms_norm_quantize_pipeline.get(), norm_bindings, norm_constants, norm_dispatcher);

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
    command.record_pipeline(second.pipeline.get(), second_bindings, second_constants, second_dispatcher);

    if (!record_prepared_staging_download(output_gpu, input.rows(), second.output_columns, transfer_slot.download, command, first.option)
        || submit_compute_and_wait(command) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    current_vulkan_dispatch_count += 2;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
    return true;
#else
    (void)input;
    (void)next;
    (void)output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_rms_norm_chain_parallel(
    const CpuBatch& input,
    const NcnnVulkanFloat8Operator& next,
    const NcnnVulkanFloat8Operator& parallel_operator,
    CpuBatch& output,
    CpuBatch& parallel_output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *implementation_;
    const Implementation& second = *next.implementation_;
    const Implementation& parallel = *parallel_operator.implementation_;
    if (!first.vulkan_context || !second.vulkan_context || !parallel.vulkan_context
        || first.vulkan_context.get() != second.vulkan_context.get()
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !parallel.pipeline
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

    NcnnVulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    CpuBatch quantized_input = input;
    for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
        quantize_float8_e4m3_inplace(quantized_input.row(token_index), first.logical_input_columns, 128, true);
    ncnn::VkMat parallel_download;
    if (!fill_staging_upload(quantized_input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator)
        || !prepare_staging_batch(parallel_download, input.rows(), parallel.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), second.output_columns);
    parallel_output = CpuBatch(input.rows(), parallel.output_columns);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(
        static_cast<int>(first.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    output_gpu.create(
        static_cast<int>(second.output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
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
        command.record_pipeline(projection.pipeline.get(), bindings, constants, dispatcher);
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
    command.record_pipeline(first.rms_norm_quantize_pipeline.get(), norm_bindings, norm_constants, norm_dispatcher);

    record_projection(second, intermediate_gpu, output_gpu);
    record_projection(parallel, input_gpu, parallel_gpu);
    if (!record_prepared_staging_download(output_gpu, input.rows(), second.output_columns, transfer_slot.download, command, first.option)
        || !record_prepared_staging_download(parallel_gpu, input.rows(), parallel.output_columns, parallel_download, command, first.option)
        || submit_compute_and_wait(command) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    current_vulkan_dispatch_count += 3;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    current_vulkan_runtime_counters.batch_downloads += 2;
    return true;
#else
    (void)input;
    (void)next;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    return false;
#endif
}

bool NcnnVulkanFloat8Operator::forward_swiglu_chain(
    const CpuBatch& input,
    const NcnnVulkanFloat8Operator& up_operator,
    const NcnnVulkanFloat8Operator& down_operator,
    ExpertActivation activation,
    float activation_limit,
    CpuBatch& output) const
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

    NcnnVulkanTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    CpuBatch quantized_input = input;
    for (size_t token_index = 0; token_index < quantized_input.rows(); ++token_index)
        quantize_float8_e4m3_inplace(quantized_input.row(token_index), gate.logical_input_columns, 128, true);
    if (!fill_staging_upload(quantized_input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), down.output_columns);

    std::unique_lock<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
        return false;
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
        command.record_pipeline(projection.pipeline.get(), bindings, constants, dispatcher);
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
    if (!record_prepared_staging_download(output_gpu, input.rows(), down.output_columns, transfer_slot.download, command, gate.option)
        || submit_compute_and_wait(command) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    current_vulkan_dispatch_count += 3;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
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
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::Option option;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

#if NCNN_MOE_WITH_VULKAN
static constexpr char mxfp4_projection_shader[] = R"glsl(
#version 450

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

shared float partial_sum[32];

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
            sum += decode_mxfp4(nibble) * input_data[input_row + block * 32 + lane] * scale;
        }
    }
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
        output_data[token * p.output_columns + output_column] = partial_sum[0] + bias_data[output_column];
    }
}
)glsl";

static bool create_mxfp4_projection_pipeline(const std::shared_ptr<NcnnVulkanContext>& context, const ncnn::Option& option,
                                             std::shared_ptr<ncnn::Pipeline>& destination)
{
    static std::once_flag compile_once;
    static std::vector<uint32_t> spirv;
    static bool compiled = false;
    std::call_once(compile_once, [&] {
        compiled = ncnn::compile_spirv_module(mxfp4_projection_shader, static_cast<int>(sizeof(mxfp4_projection_shader) - 1), option, spirv) == 0
                   && !spirv.empty();
    });
    if (!compiled)
        return false;

    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<const ncnn::VulkanDevice*, std::weak_ptr<ncnn::Pipeline>> cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    cache[device] = destination;
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

std::shared_ptr<NcnnVulkanMxfp4Operator> NcnnVulkanMxfp4Operator::create(const TensorData& matrix, const TensorData* bias, uint32_t vulkan_device_index)
{
    return create_with_allocator(matrix, bias, vulkan_device_index, nullptr);
}

std::shared_ptr<NcnnVulkanMxfp4Operator> NcnnVulkanMxfp4Operator::create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                                        uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator)
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
    implementation.vulkan_context = NcnnVulkanContext::acquire(vulkan_device_index);
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
        ncnn::VkTransfer command(device);
        command.record_upload(packed, implementation.packed, upload_option);
        command.record_upload(scales, implementation.scales, upload_option);
        command.record_upload(biases, implementation.bias, upload_option);
        uploaded = !implementation.packed.empty() && !implementation.scales.empty() && !implementation.bias.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
    {
        return {};
    }
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    return {};
#endif
}

bool NcnnVulkanMxfp4Operator::forward(const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    if (!implementation.vulkan_context || !implementation.pipeline || input.rows() == 0 || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), implementation.output_columns);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
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
    if (!record_prepared_staging_download(output_gpu, input.rows(), implementation.output_columns, transfer_slot.download, command, implementation.option)
        || submit_compute_and_wait(command) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++current_vulkan_dispatch_count;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
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
};

#if NCNN_MOE_WITH_VULKAN
static constexpr char mxfp4_gate_up_shader[] = R"glsl(
#version 450

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

shared float gate_partial[32];
shared float up_partial[32];

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
            gate_sum += decode_mxfp4(gate_nibble) * input_value * decode_scale(scale_byte(gate_scale_row + block));
            up_sum += decode_mxfp4(up_nibble) * input_value * decode_scale(scale_byte(up_scale_row + block));
        }
    }
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
        if (p.activation == 1)
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
}
)glsl";

static bool create_mxfp4_gate_up_pipeline(const std::shared_ptr<NcnnVulkanContext>& context, const ncnn::Option& option,
                                          std::shared_ptr<ncnn::Pipeline>& destination)
{
    static std::once_flag compile_once;
    static std::vector<uint32_t> spirv;
    static bool compiled = false;
    std::call_once(compile_once, [&] {
        compiled = ncnn::compile_spirv_module(mxfp4_gate_up_shader, static_cast<int>(sizeof(mxfp4_gate_up_shader) - 1), option, spirv) == 0 && !spirv.empty();
    });
    if (!compiled)
        return false;
    ncnn::VulkanDevice* device = context->device();
    static std::mutex cache_mutex;
    static std::unordered_map<const ncnn::VulkanDevice*, std::weak_ptr<ncnn::Pipeline>> cache;
    const std::lock_guard<std::mutex> cache_lock(cache_mutex);
    const auto cached = cache.find(device);
    if (cached != cache.end())
    {
        destination = cached->second.lock();
        if (destination)
            return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    cache[device] = destination;
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
                                                                                     ExpertActivation activation)
{
    return create_with_allocator(gate_up, gate_up_bias, down, down_bias, activation_limit, vulkan_device_index, nullptr, activation);
}

std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> NcnnVulkanMxfp4ExpertOperator::create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                                    const TensorData& down, const TensorData* down_bias,
                                                                                                    float activation_limit, uint32_t vulkan_device_index,
                                                                                                    ncnn::VkAllocator* weight_allocator,
                                                                                                    ExpertActivation activation)
{
#if NCNN_MOE_WITH_VULKAN
    if (gate_up.shape.size() != 2 || gate_up.shape[0] % 2 != 0 || down.shape.size() != 2 || down.shape[1] != gate_up.shape[0] / 2 || activation_limit < 0.0f)
    {
        return {};
    }
    std::shared_ptr<NcnnVulkanMxfp4Operator> gate_up_projection = NcnnVulkanMxfp4Operator::create_with_allocator(
        gate_up, gate_up_bias, vulkan_device_index, weight_allocator);
    std::shared_ptr<NcnnVulkanMxfp4Operator> down_projection = NcnnVulkanMxfp4Operator::create_with_allocator(
        down, down_bias, vulkan_device_index, weight_allocator);
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
    return {};
#endif
}

std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> NcnnVulkanMxfp4ExpertOperator::create_from_device_storage(
    const NcnnVulkanMxfp4DeviceMatrixView& gate_up, const TensorData* gate_up_bias, const NcnnVulkanMxfp4DeviceMatrixView& down,
    const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage,
    ExpertActivation activation)
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
        implementation.vulkan_context = NcnnVulkanContext::acquire(vulkan_device_index);
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
    return {};
#endif
}

bool NcnnVulkanMxfp4ExpertOperator::forward(const CpuBatch& input, CpuBatch& output) const
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

    NcnnVulkanTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    output = CpuBatch(input.rows(), down.output_columns);
    std::unique_lock<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
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
    gate_constants[4].u32 = implementation.activation == ExpertActivation::DeepSeekSwiGlu ? 1 : 0;
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

    if (!record_prepared_staging_download(output_gpu, input.rows(), down.output_columns, transfer_slot.download, command, down.option)
        || submit_compute_and_wait(command) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    current_vulkan_dispatch_count += 2;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
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
    VulkanMxfp4ExpertBackend(uint64_t capacity_bytes, uint32_t vulkan_device_index, std::shared_ptr<VulkanExpertVictimCache> device_weight_source)
        : capacity_bytes_(capacity_bytes),
          vulkan_device_index_(vulkan_device_index),
          maximum_pending_bytes_(std::min(capacity_bytes, UINT64_C(256) * 1024 * 1024)),
          device_weight_source_(std::move(device_weight_source)),
          worker_(&VulkanMxfp4ExpertBackend::worker_loop, this),
          execution_worker_(&VulkanMxfp4ExpertBackend::execution_loop, this)
    {
        vulkan_context_ = NcnnVulkanContext::acquire(vulkan_device_index_);
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
        admission.timing_bucket = timing_bucket(token_count);
        admission.bytes = bytes;

        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || entries_.find(admission.key) != entries_.end() || pending_keys_.find(admission.key) != pending_keys_.end())
        {
            return;
        }
        const PhaseTiming& phase = phase_timings_[admission.timing_bucket];
        if (phase.cpu_samples >= 3 && phase.hybrid_samples >= 8 && !phase.hybrid_enabled)
        {
            return;
        }
        if (phase.hybrid_samples < 8)
        {
            const uint64_t calibration_capacity = std::min<uint64_t>(capacity_bytes_, bytes > UINT64_MAX / 4 ? UINT64_MAX : bytes * 4);
            if (bytes > calibration_capacity || resident_bytes_ + pending_bytes_ > calibration_capacity - bytes)
            {
                return;
            }
        }
        const bool ghost = recent_ghost_index_.find(admission.key) != recent_ghost_index_.end()
                           || frequent_ghost_index_.find(admission.key) != frequent_ghost_index_.end();
        if (!ghost && admission_candidates_.insert(admission.key).second)
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

    ExpertBackendExecutionResult try_execute(const std::string& key, const CpuBatch& input, CpuBatch& output) override
    {
        const ExpertBackendRequest request{key, &input, &output};
        std::vector<ExpertBackendExecutionResult> results = try_execute_batch(std::span<const ExpertBackendRequest>(&request, 1));
        return results.empty() ? ExpertBackendExecutionResult ::Failed : results.front();
    }

    std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests) override
    {
        auto submission = submit_batch(requests);
        return submission ? submission->wait() : std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult ::Failed);
    }

    std::unique_ptr<IExpertBackendBatchSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        auto work = std::make_shared<WorkItem>();
        work->requests.assign(requests.begin(), requests.end());
        work->planned.assign(requests.size(), ExpertBackendExecutionResult ::NotResident);
        work->selected.reserve(requests.size());
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work->uncontended_sample = pending_bytes_ == 0;
            std::vector<Selection>& candidates = work->selected;
            double cpu_baseline_microseconds = 0.0;
            bool cpu_prediction_ready = !requests.empty();
            size_t phase_bucket = 0;
            for (size_t request_index = 0; request_index < requests.size(); ++request_index)
            {
                const ExpertBackendRequest& request = requests[request_index];
                if (!request.input || !request.output || request.input->rows() == 0)
                {
                    work->planned[request_index] = ExpertBackendExecutionResult ::Failed;
                    cpu_prediction_ready = false;
                    continue;
                }
                const size_t bucket = timing_bucket(request.input->rows());
                phase_bucket = std::max(phase_bucket, bucket);
                const uint64_t requested_bytes = request.weight_bytes;
                const Timing& timing = timings_[bucket];
                if (requested_bytes == 0 || timing.cpu_samples < 3)
                {
                    cpu_prediction_ready = false;
                }
                else
                {
                    cpu_baseline_microseconds += timing.cpu_microseconds_per_byte * static_cast<double>(requested_bytes);
                }
                auto existing = entries_.find(request.key);
                if (existing == entries_.end())
                {
                    const bool source_allowed = request.weight_bytes == 0 || allow_device_source_locked(bucket);
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
                            bucket,
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
                    bucket,
                });
            }

            size_t selected_count = 0;
            const bool phase_allows_hybrid = candidates.empty() || allow_hybrid_phase_locked(phase_bucket);
            if (!phase_allows_hybrid)
            {
                selected_count = 0;
            }
            else if (!candidates.empty() && !cpu_prediction_ready)
            {
                // Keep Vulkan until a host baseline exists.
                selected_count = candidates.size();
            }
            else if (!candidates.empty())
            {
                auto cold = candidates.end();
                for (auto candidate = candidates.begin(); candidate != candidates.end(); ++candidate)
                {
                    if (timings_[candidate->bucket].gpu_samples < 3)
                    {
                        cold = candidate;
                        break;
                    }
                }
                if (cold != candidates.end())
                {
                    std::iter_swap(candidates.begin(), cold);
                    selected_count = 1;
                }
                else
                {
                    std::sort(candidates.begin(), candidates.end(), SelectionOrder{this, requests});
                    double cpu_microseconds = cpu_baseline_microseconds;
                    double gpu_microseconds = 0.0;
                    double best_phase_microseconds = cpu_baseline_microseconds;
                    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index)
                    {
                        const Selection& selection = candidates[candidate_index];
                        const ExpertBackendRequest& request = requests[selection.request_index];
                        const uint64_t bytes = request.weight_bytes == 0 ? selection.entry->bytes : request.weight_bytes;
                        const Timing& timing = timings_[selection.bucket];
                        cpu_microseconds = std::max(0.0, cpu_microseconds - timing.cpu_microseconds_per_byte * static_cast<double>(bytes));
                        gpu_microseconds += timing.gpu_microseconds_per_byte * static_cast<double>(bytes);
                        const double phase_microseconds = std::max(cpu_microseconds, gpu_microseconds);
                        if (phase_microseconds < best_phase_microseconds * 0.98)
                        {
                            best_phase_microseconds = phase_microseconds;
                            selected_count = candidate_index + 1;
                        }
                    }

                    if (selected_count == 0)
                    {
                        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index)
                        {
                            Timing& timing = timings_[candidates[candidate_index].bucket];
                            ++timing.cpu_decisions;
                            if (timing.cpu_decisions % 64 == 0)
                            {
                                std::iter_swap(candidates.begin(), candidates.begin() + candidate_index);
                                selected_count = 1;
                                break;
                            }
                        }
                    }
                }
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
            uint64_t source_accelerated_bytes = 0;
            for (const Selection& selection : work->selected)
            {
                if (selection.entry->device_source)
                {
                    source_accelerated_bytes += selection.entry->bytes;
                }
            }
            if (source_accelerated_bytes == 0)
            {
                thread_source_accelerated_bytes_.erase(this);
            }
            else
            {
                thread_source_accelerated_bytes_.insert_or_assign(this, source_accelerated_bytes);
            }
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
        return std::unique_ptr<IExpertBackendBatchSubmission>(new Submission(std::move(work)));
    }

    void observe_cpu(uint32_t token_count, uint64_t weight_bytes, uint64_t elapsed_microseconds) override
    {
        if (token_count == 0 || weight_bytes == 0 || elapsed_microseconds == 0)
        {
            return;
        }
        const std::lock_guard<std::mutex> lock(mutex_);
        Timing& timing = timings_[timing_bucket(token_count)];
        observe_timing(timing.cpu_microseconds_per_byte, timing.cpu_samples, elapsed_microseconds, weight_bytes);
        update_preference_locked(timing_bucket(token_count));
    }

    void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) override
    {
        if (token_count == 0 || total_weight_bytes == 0 || elapsed_microseconds == 0)
        {
            return;
        }
        const size_t bucket = timing_bucket(token_count);
        const std::lock_guard<std::mutex> lock(mutex_);
        PhaseTiming& phase = phase_timings_[bucket];
        if (accelerated_weight_bytes == 0)
        {
            observe_timing(phase.cpu_microseconds_per_byte, phase.cpu_samples, elapsed_microseconds, total_weight_bytes);
        }
        else
        {
            observe_timing(phase.hybrid_microseconds_per_byte, phase.hybrid_samples, elapsed_microseconds, total_weight_bytes);
        }
        const auto source_observation = thread_source_accelerated_bytes_.find(this);
        if (source_observation != thread_source_accelerated_bytes_.end())
        {
            SourcePhaseTiming& source_phase = source_phase_timings_[bucket];
            observe_timing(source_phase.hybrid_microseconds_per_byte, source_phase.hybrid_samples, elapsed_microseconds, total_weight_bytes);
            thread_source_accelerated_bytes_.erase(source_observation);
            if (phase.cpu_samples >= 3 && source_phase.hybrid_samples >= 3)
            {
                if (source_phase.hybrid_enabled && source_phase.hybrid_microseconds_per_byte > phase.cpu_microseconds_per_byte * 1.02)
                {
                    source_phase.hybrid_enabled = false;
                    source_phase.cpu_decisions = 0;
                }
                else if (!source_phase.hybrid_enabled && source_phase.hybrid_microseconds_per_byte < phase.cpu_microseconds_per_byte * 0.90)
                {
                    source_phase.hybrid_enabled = true;
                    source_phase.cpu_decisions = 0;
                }
            }
        }
        if (phase.cpu_samples < 3 || phase.hybrid_samples < 8)
        {
            return;
        }
        if (phase.hybrid_enabled && phase.hybrid_microseconds_per_byte > phase.cpu_microseconds_per_byte * 1.02)
        {
            phase.hybrid_enabled = false;
            phase.cpu_decisions = 0;
            for (auto pending = pending_.begin(); pending != pending_.end();)
            {
                if (pending->timing_bucket != bucket)
                {
                    ++pending;
                    continue;
                }
                pending_bytes_ -= pending->bytes;
                pending_keys_.erase(pending->key);
                pending = pending_.erase(pending);
                ++dropped_admissions_;
            }
            if (pending_.empty() && active_admissions_ == 0)
            {
                admission_idle_.notify_all();
            }
        }
        else if (!phase.hybrid_enabled && phase.hybrid_microseconds_per_byte < phase.cpu_microseconds_per_byte * 0.90)
        {
            phase.hybrid_enabled = true;
            phase.cpu_decisions = 0;
        }
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
        size_t bucket = 0;
    };

    struct WorkItem
    {
        std::vector<ExpertBackendRequest> requests;
        std::vector<Selection> selected;
        std::vector<ExpertBackendExecutionResult> planned;
        std::vector<ExpertBackendExecutionResult> final;
        bool uncontended_sample = false;
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
        }

        std::span<const ExpertBackendExecutionResult> planned_results() const noexcept override
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

    private:
        std::shared_ptr<WorkItem> work_;
        bool waited_ = false;
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
        size_t timing_bucket = 0;
        uint64_t bytes = 0;
    };

    struct Ghost
    {
        std::string key;
        uint64_t bytes = 0;
    };
    using GhostList = std::list<Ghost>;
    using GhostIndex = std::unordered_map<std::string, GhostList::iterator, TransparentStringHash, std::equal_to<>>;

    struct Timing
    {
        double cpu_microseconds_per_byte = 0.0;
        double gpu_microseconds_per_byte = 0.0;
        uint64_t cpu_samples = 0;
        uint64_t gpu_samples = 0;
        uint64_t cpu_decisions = 0;
        bool gpu_preferred = true;
    };

    struct SelectionOrder
    {
        const VulkanMxfp4ExpertBackend* backend = nullptr;
        std::span<const ExpertBackendRequest> requests;

        bool operator()(const Selection& left, const Selection& right) const
        {
            return backend->benefit_ratio(left, requests) > backend->benefit_ratio(right, requests);
        }
    };

    struct PhaseTiming
    {
        double cpu_microseconds_per_byte = 0.0;
        double hybrid_microseconds_per_byte = 0.0;
        uint64_t cpu_samples = 0;
        uint64_t hybrid_samples = 0;
        uint64_t cpu_decisions = 0;
        bool hybrid_enabled = false;
    };

    struct SourcePhaseTiming
    {
        double hybrid_microseconds_per_byte = 0.0;
        uint64_t hybrid_samples = 0;
        uint64_t cpu_decisions = 0;
        bool hybrid_enabled = false;
    };

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

    static size_t timing_bucket(size_t token_count)
    {
        if (token_count <= 1)
            return 0;
        if (token_count <= 3)
            return 1;
        if (token_count <= 7)
            return 2;
        return 3;
    }

    double benefit_ratio(const Selection& selection, std::span<const ExpertBackendRequest> requests) const
    {
        const ExpertBackendRequest& request = requests[selection.request_index];
        const uint64_t bytes = request.weight_bytes == 0 ? selection.entry->bytes : request.weight_bytes;
        const Timing& timing = timings_[selection.bucket];
        const double gpu_cost = timing.gpu_microseconds_per_byte * static_cast<double>(bytes);
        const double cpu_cost = timing.cpu_microseconds_per_byte * static_cast<double>(bytes);
        return cpu_cost / std::max(gpu_cost, 1e-9);
    }

    static void observe_timing(double& average, uint64_t& samples, uint64_t elapsed_microseconds, uint64_t bytes)
    {
        if (bytes == 0)
            return;
        const double observation = static_cast<double>(elapsed_microseconds) / static_cast<double>(bytes);
        if (samples == 0)
            average = observation;
        else
            average = average * 0.8 + observation * 0.2;
        ++samples;
    }

    bool prefer_gpu_locked(size_t bucket)
    {
        Timing& timing = timings_[bucket];
        if (timing.gpu_samples < 3 || timing.cpu_samples < 3)
        {
            return true;
        }
        if (!timing.gpu_preferred)
        {
            ++timing.cpu_decisions;
            // Sparse probes adapt to changing device conditions.
            return timing.cpu_decisions % 64 == 0;
        }
        return true;
    }

    void update_preference_locked(size_t bucket)
    {
        Timing& timing = timings_[bucket];
        if (timing.gpu_samples < 3 || timing.cpu_samples < 3)
        {
            return;
        }
        if (timing.gpu_preferred)
        {
            if (timing.gpu_microseconds_per_byte > timing.cpu_microseconds_per_byte * 1.1)
            {
                timing.gpu_preferred = false;
                timing.cpu_decisions = 0;
            }
        }
        else if (timing.gpu_microseconds_per_byte < timing.cpu_microseconds_per_byte * 0.9)
        {
            timing.gpu_preferred = true;
            timing.cpu_decisions = 0;
        }
    }

    bool allow_hybrid_phase_locked(size_t bucket)
    {
        PhaseTiming& phase = phase_timings_[bucket];
        if (phase.cpu_samples < 3)
        {
            // Direct clients may not provide a host baseline.
            return phase.cpu_samples == 0 && phase.hybrid_samples == 0;
        }
        if (phase.hybrid_samples < 8)
            return true;
        if (phase.hybrid_enabled)
            return true;
        ++phase.cpu_decisions;
        return phase.cpu_decisions % 64 == 0;
    }

    bool allow_device_source_locked(size_t bucket)
    {
        const PhaseTiming& phase = phase_timings_[bucket];
        SourcePhaseTiming& source = source_phase_timings_[bucket];
        if (phase.cpu_samples < 3)
            return false;
        if (source.hybrid_enabled)
            return true;
        ++source.cpu_decisions;
        return source.cpu_decisions % 64 == 0;
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
        const uint32_t input_columns = first_gate.input_columns;
        const uint32_t output_columns = first_down.output_columns;
        size_t total_rows = 0;
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
        }
        if (total_rows == 0 || total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            return false;
        }

        CpuBatch combined_input;
        combined_input.reset(total_rows, input_columns, false);
        size_t row_offset = 0;
        for (const Selection& selection : selected)
        {
            const CpuBatch& input = *requests[selection.request_index].input;
            for (size_t row = 0; row < input.rows(); ++row)
            {
                std::copy_n(input.row(row), input_columns, combined_input.row(row_offset + row));
            }
            row_offset += input.rows();
        }

        NcnnVulkanTransferLease transfer_lease = first_gate.vulkan_context->acquire_transfer_slot();
        NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
        if (!fill_staging_upload(combined_input, transfer_slot.upload, transfer_slot.staging_allocator)
            || !prepare_staging_batch(transfer_slot.download, total_rows, output_columns, transfer_slot.staging_allocator))
        {
            return false;
        }
        CpuBatch combined_output;
        combined_output.reset(total_rows, output_columns, false);
        std::unique_lock<std::mutex> lock(first_gate.vulkan_context->command_mutex());
        ncnn::VkCompute& command = *transfer_slot.command;
        if (transfer_slot.command_used)
        {
            if (command.reset() != 0)
                return false;
            ++current_vulkan_runtime_counters.command_buffer_reuses;
        }
        transfer_slot.command_used = true;
        ncnn::VkMat input_gpu;
        if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first_gate.option))
        {
            return false;
        }
        ncnn::VkMat output_gpu;
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
            gate_constants[4].u32 = expert.activation == ExpertActivation::DeepSeekSwiGlu ? 1 : 0;
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
        if (!record_prepared_staging_download(output_gpu, total_rows, output_columns, transfer_slot.download, command, first_down.option)
            || submit_compute_and_wait(command) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, combined_output))
        {
            return false;
        }
        lock.unlock();

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
        current_vulkan_dispatch_count += selected.size() * 2;
        ++current_vulkan_runtime_counters.compute_submissions;
        ++current_vulkan_runtime_counters.batch_uploads;
        ++current_vulkan_runtime_counters.batch_downloads;
        return true;
    }

    void execute_work_item(const std::shared_ptr<WorkItem>& work)
    {
        const auto started = std::chrono::steady_clock::now();
        const bool executed = forward_batch(work->requests, work->selected);
        const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
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
                uint64_t selected_bytes = 0;
                for (const Selection& selection : work->selected)
                {
                    if (selection.entry->device_source)
                    {
                        ++device_source_executions_;
                    }
                    selected_bytes += selection.entry->bytes;
                }
                if (work->uncontended_sample && selected_bytes != 0)
                {
                    std::array<bool, 4> observed = {};
                    for (const Selection& selection : work->selected)
                    {
                        if (observed[selection.bucket])
                        {
                            continue;
                        }
                        observed[selection.bucket] = true;
                        Timing& timing = timings_[selection.bucket];
                        observe_timing(timing.gpu_microseconds_per_byte, timing.gpu_samples, elapsed, selected_bytes);
                        update_preference_locked(selection.bucket);
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
        if (pending_.empty() && active_admissions_ == 0)
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
                work_available_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
                if (stopping_)
                    return;
                admission = std::move(pending_.front());
                pending_.pop_front();
                ++active_admissions_;
            }

            auto operation = NcnnVulkanMxfp4ExpertOperator ::create_with_allocator(*admission.gate_up, admission.gate_up_bias.get(), *admission.down,
                                                                                   admission.down_bias.get(), admission.activation_limit, vulkan_device_index_,
                                                                                   expert_weight_allocator_.get(), admission.activation);
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
    const uint64_t maximum_pending_bytes_;
    std::shared_ptr<NcnnVulkanContext> vulkan_context_;
    std::unique_ptr<ncnn::VkBlobAllocator> expert_weight_allocator_;
    std::shared_ptr<VulkanExpertVictimCache> device_weight_source_;
    mutable std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable execution_available_;
    std::condition_variable admission_idle_;
    bool stopping_ = false;
    std::deque<PendingAdmission> pending_;
    std::deque<std::shared_ptr<WorkItem>> execution_pending_;
    std::unordered_set<std::string, TransparentStringHash, std::equal_to<>> pending_keys_;
    std::unordered_set<std::string, TransparentStringHash, std::equal_to<>> admission_candidates_;
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
    std::array<Timing, 4> timings_;
    std::array<PhaseTiming, 4> phase_timings_;
    std::array<SourcePhaseTiming, 4> source_phase_timings_;
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
    static thread_local std::unordered_map<const VulkanMxfp4ExpertBackend*, uint64_t> thread_source_accelerated_bytes_;
    std::thread worker_;
    std::thread execution_worker_;
};

thread_local std::unordered_map<const VulkanMxfp4ExpertBackend*, uint64_t> VulkanMxfp4ExpertBackend::thread_source_accelerated_bytes_;
#endif

std::shared_ptr<IExpertExecutionBackend> create_vulkan_mxfp4_expert_backend(uint64_t capacity_bytes, uint32_t vulkan_device_index,
                                                                            std::shared_ptr<IExpertVictimCache> device_weight_source)
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
    return std::make_shared<VulkanMxfp4ExpertBackend>(capacity_bytes, vulkan_device_index, std::move(source));
#else
    (void)capacity_bytes;
    (void)vulkan_device_index;
    (void)device_weight_source;
    return {};
#endif
}

NcnnVulkanAttentionOperator::NcnnVulkanAttentionOperator()
    : implementation_(new Implementation)
{
}

NcnnVulkanAttentionOperator::~NcnnVulkanAttentionOperator() = default;

uint64_t NcnnVulkanAttentionOperator::current_thread_blocks() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return current_vulkan_attention_block_count;
#else
    return 0;
#endif
}

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
                                   const std::vector<float>& inverse_frequencies, float concentration, bool bfloat16_storage, ncnn::VkAllocator* allocator)
{
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max()) || inverse_frequencies.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_matrix(cosine_staging, static_cast<int>(inverse_frequencies.size()), static_cast<int>(token_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator)
        || !prepare_staging_matrix(sine_staging, static_cast<int>(inverse_frequencies.size()), static_cast<int>(token_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator))
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
                                        bool bfloat16_storage, ncnn::VkAllocator* allocator)
{
    if (destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_tensor(staging, static_cast<int>(destination_count), static_cast<int>(token_count), static_cast<int>(config.head_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator))
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
    float key_data[];
};
layout(binding = 5) writeonly buffer value_blob
{
    float value_data[];
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
        const float cosine = cosine_data[cache];
        const float sine = sine_data[cache];
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
        const float cosine = cosine_data[cache];
        const float sine = sine_data[cache];
        const float first = fused_qkv_data[source];
        const float second = fused_qkv_data[source + half_dimension];
        const float rotated_first = first * cosine - second * sine;
        const float rotated_second = first * sine + second * cosine;
        key_data[destination] = rotated_first;
        key_data[destination + half_dimension] = rotated_second;
        if (p.direct_ring != 0)
        {
            const uint duplicate = destination + p.ring_capacity * p.head_dimension;
            key_data[duplicate] = rotated_first;
            key_data[duplicate + half_dimension] = rotated_second;
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
    value_data[destination] = fused_qkv_data[source];
    if (p.direct_ring != 0)
    {
        value_data[destination + p.ring_capacity * p.head_dimension] = fused_qkv_data[source];
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
    float key_data[];
};
layout(binding = 2) readonly buffer value_blob
{
    float value_data[];
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
            partial = query_data[query_base + lane] * key_data[key_base + token * p.head_dimension + lane];
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
            const float score = reduction[0] * p.scale + (sink_token ? sink_data[head] : 0.0);
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
                        + (sink_token ? 0.0 : current_weight * value_data[value_base + token * p.head_dimension + lane]);
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
    float source_key_data[];
};
layout(binding = 1) readonly buffer source_value_blob
{
    float source_value_data[];
};
layout(binding = 2) buffer destination_key_blob
{
    float destination_key_data[];
};
layout(binding = 3) buffer destination_value_blob
{
    float destination_value_data[];
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
    const float key = source_key_data[source_index];
    const float value = source_value_data[source_index];
    destination_key_data[destination_index] = key;
    destination_key_data[duplicate_index] = key;
    destination_value_data[destination_index] = value;
    destination_value_data[duplicate_index] = value;
}
)glsl";

static constexpr char attention_ring_zero_shader[] = R"glsl(
#version 450

layout(binding = 0) buffer destination_key_blob
{
    float destination_key_data[];
};
layout(binding = 1) buffer destination_value_blob
{
    float destination_value_data[];
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
    destination_key_data[destination_index] = 0.0;
    destination_value_data[destination_index] = 0.0;
}
)glsl";

static bool create_attention_pipeline(ncnn::VulkanDevice* vkdev, const ncnn::Option& option, const char* shader, int shader_size, ncnn::Pipeline*& destination)
{
    static std::once_flag qkv_compile_once;
    static std::once_flag decode_sdpa_compile_once;
    static std::once_flag append_compile_once;
    static std::once_flag zero_compile_once;
    static std::vector<uint32_t> qkv_spirv;
    static std::vector<uint32_t> decode_sdpa_spirv;
    static std::vector<uint32_t> append_spirv;
    static std::vector<uint32_t> zero_spirv;
    static bool qkv_compiled = false;
    static bool decode_sdpa_compiled = false;
    static bool append_compiled = false;
    static bool zero_compiled = false;
    std::once_flag& compile_once = shader == attention_qkv_rope_shader      ? qkv_compile_once
                                   : shader == attention_decode_sdpa_shader ? decode_sdpa_compile_once
                                   : shader == attention_ring_append_shader ? append_compile_once
                                                                            : zero_compile_once;
    std::vector<uint32_t>& spirv = shader == attention_qkv_rope_shader      ? qkv_spirv
                                   : shader == attention_decode_sdpa_shader ? decode_sdpa_spirv
                                   : shader == attention_ring_append_shader ? append_spirv
                                                                            : zero_spirv;
    bool& compiled = shader == attention_qkv_rope_shader      ? qkv_compiled
                     : shader == attention_decode_sdpa_shader ? decode_sdpa_compiled
                     : shader == attention_ring_append_shader ? append_compiled
                                                              : zero_compiled;
    std::call_once(compile_once, [&] { compiled = ncnn::compile_spirv_module(shader, shader_size, option, spirv) == 0 && !spirv.empty(); });
    if (!compiled)
        return false;

    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(vkdev));
    if (shader == attention_qkv_rope_shader)
        pipeline->set_optimal_local_size_xyz(64, 1, 1);
    else if (shader == attention_decode_sdpa_shader)
        pipeline->set_local_size_xyz(128, 1, 1);
    else
        pipeline->set_optimal_local_size_xyz(8, 8, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = pipeline.release();
    return true;
}

static bool record_attention_qkv_rope(const ncnn::Pipeline* pipeline, const ncnn::VkMat& fused_qkv, const ncnn::VkMat& cosine, const ncnn::VkMat& sine,
                                      const NcnnVulkanAttentionConfig& config, size_t token_count, const ncnn::VkMat* direct_ring_key,
                                      const ncnn::VkMat* direct_ring_value, uint64_t ring_capacity, uint64_t destination_start, ncnn::VkMat& query,
                                      ncnn::VkMat& key, ncnn::VkMat& value, ncnn::VkCompute& command, ncnn::VkAllocator* allocator)
{
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns) + static_cast<uint64_t>(key_value_columns) * 2;
    const uint64_t half_dimension = config.head_dimension / 2;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count) * half_dimension
                                   + static_cast<uint64_t>(config.kv_head_count) * half_dimension + key_value_columns);
    const bool direct_ring = direct_ring_key || direct_ring_value;
    if (!pipeline || fused_qkv.empty() || cosine.empty() || sine.empty() || fused_qkv.elempack != 1 || fused_qkv.elemsize != sizeof(float)
        || cosine.elempack != 1 || cosine.elemsize != sizeof(float) || sine.elempack != 1 || sine.elemsize != sizeof(float) || token_count == 0
        || token_count > static_cast<size_t>(std::numeric_limits<int>::max()) || total_columns > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || work_items > static_cast<uint64_t>(std::numeric_limits<int>::max()) || fused_qkv.dims != 2 || fused_qkv.w != static_cast<int>(total_columns)
        || fused_qkv.h != static_cast<int>(token_count) || (direct_ring_key == nullptr) != (direct_ring_value == nullptr)
        || (direct_ring
            && (ring_capacity == 0 || ring_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) || destination_start >= ring_capacity
                || token_count > ring_capacity)))
    {
        return false;
    }

    query.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.head_count), sizeof(float), 1, allocator);
    if (direct_ring)
    {
        key = *direct_ring_key;
        value = *direct_ring_value;
    }
    else
    {
        key.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.kv_head_count), sizeof(float), 1, allocator);
        value.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.kv_head_count), sizeof(float), 1,
                     allocator);
    }
    if (query.empty() || key.empty() || value.empty()
        || (direct_ring
            && (key.dims != 3 || value.dims != 3 || key.w != static_cast<int>(config.head_dimension) || value.w != key.w
                || key.h != static_cast<int>(ring_capacity * 2) || value.h != key.h || key.c != static_cast<int>(config.kv_head_count) || value.c != key.c
                || key.elemsize != sizeof(float) || value.elemsize != sizeof(float) || key.elempack != 1 || value.elempack != 1))
        || query.cstep > std::numeric_limits<uint32_t>::max() || key.cstep > std::numeric_limits<uint32_t>::max()
        || value.cstep > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }

    const std::vector<ncnn::VkMat> bindings = {
        fused_qkv,
        cosine,
        sine,
        query,
        key,
        value,
    };
    std::vector<ncnn::vk_constant_type> constants(12);
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
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items);
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

static DecodeSdpaMode decode_sdpa_mode() noexcept
{
    const char* value = nullptr;
#if defined(_MSC_VER)
    std::array<char, 16> value_storage = {};
    size_t value_length = 0;
    if (getenv_s(&value_length, value_storage.data(), value_storage.size(), "NCNN_MOE_VULKAN_DECODE_SDPA") == 0 && value_length > 1
        && value_length <= value_storage.size())
    {
        value = value_storage.data();
    }
#else
    value = std::getenv("NCNN_MOE_VULKAN_DECODE_SDPA");
#endif
    if (!value)
        return DecodeSdpaMode::Auto;
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0 || std::strcmp(value, "false") == 0)
    {
        return DecodeSdpaMode::Disabled;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0 || std::strcmp(value, "true") == 0)
    {
        return DecodeSdpaMode::Forced;
    }
    return DecodeSdpaMode::Auto;
}

static bool qkv_ring_fusion_enabled() noexcept
{
    const char* value = nullptr;
#if defined(_MSC_VER)
    std::array<char, 16> value_storage = {};
    size_t value_length = 0;
    if (getenv_s(&value_length, value_storage.data(), value_storage.size(), "NCNN_MOE_VULKAN_QKV_RING") == 0 && value_length > 1
        && value_length <= value_storage.size())
    {
        value = value_storage.data();
    }
#else
    value = std::getenv("NCNN_MOE_VULKAN_QKV_RING");
#endif
    return !value || (std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 && std::strcmp(value, "false") != 0);
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
        || query.elempack != 1 || key.elempack != 1 || value.elempack != 1 || sinks.elempack != 1 || query.elemsize != sizeof(float)
        || key.elemsize != sizeof(float) || value.elemsize != sizeof(float) || sinks.elemsize != sizeof(float) || config.head_dimension == 0
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

static bool create_attention_ring_storage(NcnnVulkanAttentionCache& cache, uint32_t width, uint32_t channels, uint64_t capacity, ncnn::VkAllocator* allocator)
{
    if (capacity == 0 || capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / 2)
        return false;
    cache.key.create(static_cast<int>(width), static_cast<int>(capacity * 2), static_cast<int>(channels), sizeof(float), 1, allocator);
    cache.value.create(static_cast<int>(width), static_cast<int>(capacity * 2), static_cast<int>(channels), sizeof(float), 1, allocator);
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
        || source_key.elempack != 1 || source_key.elemsize != sizeof(float) || source_key.w != source_value.w || source_key.h != source_value.h
        || source_key.c != source_value.c || destination_key.w != source_key.w || destination_key.c != source_key.c || destination_value.w != source_key.w
        || destination_value.c != source_key.c || capacity == 0 || destination_start >= capacity || source_key.cstep > std::numeric_limits<uint32_t>::max()
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
        || destination_key.c != destination_value.c || destination_key.cstep != destination_value.cstep || destination_key.elemsize != sizeof(float)
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

    const uint32_t half_dimension = config.head_dimension / 2;
    implementation.rope_inverse_frequencies.resize(half_dimension);
    float rope_low = 0.0f;
    float rope_high = 0.0f;
    if (config.rope_scaling_factor > 1.0f)
    {
        implementation.rope_concentration = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(half_dimension);
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
    if (!create_attention_pipeline(vkdev, implementation.option, attention_qkv_rope_shader, static_cast<int>(sizeof(attention_qkv_rope_shader) - 1),
                                   implementation.qkv_rope_pipeline)
        || !create_attention_pipeline(vkdev, implementation.option, attention_decode_sdpa_shader, static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1),
                                      implementation.decode_sdpa_pipeline)
        || !create_attention_pipeline(vkdev, implementation.option, attention_ring_append_shader, static_cast<int>(sizeof(attention_ring_append_shader) - 1),
                                      implementation.ring_append_pipeline)
        || !create_attention_pipeline(vkdev, implementation.option, attention_ring_zero_shader, static_cast<int>(sizeof(attention_ring_zero_shader) - 1),
                                      implementation.ring_zero_pipeline))
        return {};
    ncnn::VkTransfer command(vkdev);
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    ncnn::Mat sink_model(static_cast<int>(config.head_count), sizeof(float));
    if (sink_model.empty())
        return {};
    float* sink_values = static_cast<float*>(sink_model.data);
    std::fill_n(sink_values, config.head_count, 0.0f);
    if (has_flag(config.flags, NcnnAttentionSink))
    {
        std::copy(implementation.sinks.begin(), implementation.sinks.end(), sink_values);
    }
    command.record_upload(sink_model, implementation.attention_sinks, upload_option);
    if (implementation.norm->upload_model(command, upload_option) != 0 || implementation.attention_sinks.empty() || command.submit_and_wait() != 0)
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

bool NcnnVulkanAttentionOperator::forward(uint64_t position_offset, CpuLayerCache& cache, const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    const NcnnVulkanAttentionConfig& config = implementation.config;
    if (input.rows() == 0 || input.columns() != config.hidden_size || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || (cache.dtype != config.kv_cache_dtype && cache.token_count != 0)
        || (cache.token_count != 0
            && (!cache.vulkan_attention_cache || cache.capacity_tokens == 0 || cache.first_slot >= cache.capacity_tokens
                || cache.token_count > cache.capacity_tokens || static_cast<uint64_t>(cache.vulkan_attention_cache->key.h) != cache.capacity_tokens * 2
                || static_cast<uint64_t>(cache.vulkan_attention_cache->value.h) != cache.capacity_tokens * 2)))
        return false;

    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16 && implementation.option.use_bf16_storage;
    const uint64_t actual_token_count = cache.token_count + input.rows();
    const uint64_t sink_token_count = has_flag(config.flags, NcnnAttentionSink) ? 1 : 0;
    const uint64_t destination_count = actual_token_count + sink_token_count;
    if (actual_token_count < cache.token_count || destination_count < actual_token_count
        || destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;

    const DecodeSdpaMode selected_decode_sdpa_mode = decode_sdpa_mode();
    const bool adaptive_decode_sdpa = input.rows() == 1 && config.head_dimension <= 128 && destination_count <= 4096 && selected_decode_sdpa_mode == DecodeSdpaMode::Auto;
    const bool try_decode_sdpa = input.rows() == 1 && selected_decode_sdpa_mode != DecodeSdpaMode::Disabled
                                 && (selected_decode_sdpa_mode == DecodeSdpaMode::Forced
                                     || (adaptive_decode_sdpa
                                         && implementation.vulkan_context->choose_decode_sdpa(config.head_dimension, config.head_count, config.kv_head_count, destination_count)));

    const NcnnLinearOperator::Implementation& fused = *implementation.fused_qkv->implementation_;
    const NcnnLinearOperator::Implementation& projection = *implementation.output_projection->implementation_;
    NcnnVulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !fill_rope_staging_pair(transfer_slot.rope_cosine, transfer_slot.rope_sine, input.rows(), position_offset, implementation.rope_inverse_frequencies,
                                   implementation.rope_concentration, bfloat16_storage, transfer_slot.staging_allocator)
        || (!try_decode_sdpa
            && !fill_attention_mask_staging(transfer_slot.attention_mask, input.rows(), destination_count, position_offset, cache, config, implementation.sinks,
                                            bfloat16_storage, transfer_slot.staging_allocator))
        || !prepare_staging_batch(transfer_slot.download, input.rows(), config.hidden_size, transfer_slot.staging_allocator))
        return false;

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++current_vulkan_runtime_counters.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    ncnn::VkMat cosine_gpu;
    ncnn::VkMat sine_gpu;
    ncnn::VkMat mask_gpu;
    bool mask_uploaded = false;
    if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), input_gpu, command, vkdev, implementation.option))
        return false;
    if (!record_mapped_upload(transfer_slot.rope_cosine, cosine_gpu, command, implementation.option)
        || !record_mapped_upload(transfer_slot.rope_sine, sine_gpu, command, implementation.option))
        return false;
    if (!try_decode_sdpa)
    {
        if (!record_mapped_upload(transfer_slot.attention_mask, mask_gpu, command, implementation.option))
            return false;
        mask_uploaded = true;
    }
    ncnn::VkMat normalized_gpu;
    if (implementation.norm->forward(input_gpu, normalized_gpu, command, implementation.option) != 0)
        return false;

    ncnn::VkMat fused_gpu;
    if (fused.layer->forward(normalized_gpu, fused_gpu, command, implementation.option) != 0)
        return false;

    uint64_t ring_capacity = cache.capacity_tokens;
    uint64_t ring_first_slot = cache.first_slot;
    const bool ring_resized = actual_token_count > ring_capacity;
    std::shared_ptr<NcnnVulkanAttentionCache> next_cache = std::make_shared<NcnnVulkanAttentionCache>();
    if (ring_resized)
    {
        ring_capacity = next_attention_ring_capacity(ring_capacity, actual_token_count);
        if (!create_attention_ring_storage(*next_cache, config.head_dimension, config.kv_head_count, ring_capacity, implementation.option.blob_vkallocator))
        {
            return false;
        }
        if (cache.token_count != 0)
        {
            const ncnn::VkMat previous_key = attention_ring_view(cache.vulkan_attention_cache->key, cache.first_slot, cache.token_count);
            const ncnn::VkMat previous_value = attention_ring_view(cache.vulkan_attention_cache->value, cache.first_slot, cache.token_count);
            if (!record_attention_ring_append(implementation.ring_append_pipeline, previous_key, previous_value, next_cache->key, next_cache->value,
                                              ring_capacity, 0, command))
            {
                return false;
            }
        }
        ring_first_slot = 0;
    }
    else
    {
        if (!cache.vulkan_attention_cache || cache.vulkan_attention_cache->key.empty() || cache.vulkan_attention_cache->value.empty())
        {
            return false;
        }
        next_cache->key = cache.vulkan_attention_cache->key;
        next_cache->value = cache.vulkan_attention_cache->value;
    }
    const uint64_t append_slot = (ring_first_slot + cache.token_count) % ring_capacity;

    ncnn::VkMat query_rope;
    ncnn::VkMat key_rope;
    ncnn::VkMat value_heads;
    ncnn::VkMat fused_qkv_unpacked = fused_gpu;
    if (fused_qkv_unpacked.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(fused_qkv_unpacked, unpacked, 1, command, implementation.option);
        fused_qkv_unpacked = unpacked;
    }
    const bool fused_qkv_ring = input.rows() == 1 && qkv_ring_fusion_enabled()
                                && record_attention_qkv_rope(implementation.qkv_rope_pipeline, fused_qkv_unpacked, cosine_gpu, sine_gpu, config, input.rows(),
                                                             &next_cache->key, &next_cache->value, ring_capacity, append_slot, query_rope, key_rope,
                                                             value_heads, command, implementation.option.blob_vkallocator);
    const bool fused_qkv_rope = fused_qkv_ring
                                || record_attention_qkv_rope(implementation.qkv_rope_pipeline, fused_qkv_unpacked, cosine_gpu, sine_gpu, config, input.rows(),
                                                             nullptr, nullptr, 0, 0, query_rope, key_rope, value_heads, command,
                                                             implementation.option.blob_vkallocator);
    if (!fused_qkv_rope)
    {
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
                || value_heads.c != static_cast<int>(config.kv_head_count) || key_rope.elemsize != sizeof(float) || value_heads.elemsize != sizeof(float))))
        return false;

    if (!fused_qkv_ring
        && !record_attention_ring_append(implementation.ring_append_pipeline, key_rope, value_heads, next_cache->key, next_cache->value, ring_capacity,
                                         append_slot, command))
    {
        return false;
    }

    bool sink_zero_recorded = false;
    if (sink_token_count != 0 && !fused_qkv_ring)
    {
        const uint64_t sink_row = ring_first_slot + actual_token_count;
        if (!record_attention_ring_zero(implementation.ring_zero_pipeline, next_cache->key, next_cache->value, sink_row, command))
            return false;
        sink_zero_recorded = true;
    }

    ncnn::VkMat combined_key = attention_ring_view(next_cache->key, ring_first_slot, destination_count);
    ncnn::VkMat combined_value = attention_ring_view(next_cache->value, ring_first_slot, destination_count);
    if (combined_key.empty() || combined_value.empty())
        return false;
    ncnn::VkMat attention_matrix;
    const bool fused_decode_sdpa = try_decode_sdpa
                                   && record_attention_decode_sdpa(implementation.decode_sdpa_pipeline, query_rope, combined_key, combined_value, implementation.attention_sinks, config,
                                                                   destination_count, attention_matrix, command, implementation.option.blob_vkallocator);
    if (!fused_decode_sdpa)
    {
        if (sink_token_count != 0 && !sink_zero_recorded)
        {
            const uint64_t sink_row = ring_first_slot + actual_token_count;
            if (!record_attention_ring_zero(implementation.ring_zero_pipeline, next_cache->key, next_cache->value, sink_row, command))
            {
                return false;
            }
            sink_zero_recorded = true;
        }
        if (!mask_uploaded)
        {
            if (!fill_attention_mask_staging(transfer_slot.attention_mask, input.rows(), destination_count, position_offset, cache, config,
                                             implementation.sinks, bfloat16_storage, transfer_slot.staging_allocator)
                || !record_mapped_upload(transfer_slot.attention_mask, mask_gpu, command, implementation.option))
            {
                return false;
            }
            mask_uploaded = true;
        }
        std::vector<ncnn::VkMat> sdpa_input = {
            query_rope,
            combined_key,
            combined_value,
            mask_gpu,
        };
        std::vector<ncnn::VkMat> sdpa_output(1);
        if (implementation.sdpa->forward(sdpa_input, sdpa_output, command, implementation.option) != 0)
            return false;

        ncnn::VkMat attention_token_major;
        if (implementation.permute_heads_tokens->forward(sdpa_output[0], attention_token_major, command, implementation.option) != 0
            || implementation.reshape_attention->forward(attention_token_major, attention_matrix, command, implementation.option) != 0)
            return false;
    }

    ncnn::VkMat projected_gpu;
    if (projection.layer->forward(attention_matrix, projected_gpu, command, implementation.option) != 0)
        return false;
    std::vector<ncnn::VkMat> add_input = {input_gpu, projected_gpu};
    std::vector<ncnn::VkMat> add_output(1);
    if (implementation.add->forward(add_input, add_output, command, implementation.option) != 0)
        return false;

    ncnn::VkMat download_gpu = add_output[0];
    if (download_gpu.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(download_gpu, unpacked, 1, command, implementation.option);
        download_gpu = unpacked;
    }
    if (!record_prepared_staging_download(download_gpu, input.rows(), config.hidden_size, transfer_slot.download, command, implementation.option))
        return false;

    const uint64_t total_actual_tokens = cache.token_count + input.rows();
    const uint64_t retained_tokens = config.sliding_window == 0 ? total_actual_tokens
                                                                : std::min<uint64_t>(total_actual_tokens, config.sliding_window > 1 ? config.sliding_window - 1 : 0);
    const uint64_t dropped_tokens = total_actual_tokens - retained_tokens;
    const uint64_t next_first_slot = retained_tokens == 0 ? 0 : (ring_first_slot + dropped_tokens) % ring_capacity;
    const uint64_t allocated_cache_bytes = retained_tokens == 0
                                               ? 0
                                               : static_cast<uint64_t>(next_cache->key.cstep) * next_cache->key.c * next_cache->key.elemsize
                                                     + static_cast<uint64_t>(next_cache->value.cstep) * next_cache->value.c * next_cache->value.elemsize;

    output = CpuBatch(input.rows(), config.hidden_size);
    const auto execution_started = std::chrono::steady_clock::now();
    if (submit_compute_and_wait(command) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
        return false;
    if (adaptive_decode_sdpa)
    {
        implementation.vulkan_context->observe_decode_sdpa(
            config.head_dimension, config.head_count, config.kv_head_count, destination_count, fused_decode_sdpa,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - execution_started).count()));
    }
    for (size_t row_index = 0; row_index < output.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            if (!std::isfinite(output.row(row_index)[column]))
            {
                return false;
            }
        }
    }

    const uint64_t previous_start = cache.token_count == 0 ? position_offset : cache.start_position;
    cache.keys.clear();
    cache.values.clear();
    cache.bfloat16_keys.clear();
    cache.bfloat16_values.clear();
    cache.start_position = previous_start + dropped_tokens;
    cache.token_count = retained_tokens;
    cache.first_slot = next_first_slot;
    cache.capacity_tokens = retained_tokens == 0 ? 0 : ring_capacity;
    cache.columns = config.kv_head_count * config.head_dimension;
    cache.dtype = config.kv_cache_dtype;
    cache.vulkan_attention_cache = retained_tokens == 0 ? nullptr : std::move(next_cache);
    cache.device_allocated_bytes = allocated_cache_bytes;
    current_vulkan_dispatch_count += 2;
    ++current_vulkan_attention_block_count;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
    current_vulkan_runtime_counters.auxiliary_uploads += mask_uploaded ? 3 : 2;
    if (fused_qkv_rope)
        ++current_vulkan_runtime_counters.attention_qkv_rope_fusions;
    if (fused_qkv_ring)
        ++current_vulkan_runtime_counters.attention_qkv_ring_fusions;
    if (fused_decode_sdpa)
        ++current_vulkan_runtime_counters.attention_decode_sdpa_fusions;
    ++current_vulkan_runtime_counters.kv_ring_appends;
    if (ring_resized)
        ++current_vulkan_runtime_counters.kv_ring_resizes;
    if (ring_first_slot + destination_count > ring_capacity)
        ++current_vulkan_runtime_counters.kv_ring_wrapped_views;
    current_vulkan_runtime_counters.auxiliary_upload_bytes += transfer_slot.rope_cosine.buffer_capacity() + transfer_slot.rope_sine.buffer_capacity()
                                                              + (mask_uploaded ? transfer_slot.attention_mask.buffer_capacity() : 0);
    return true;
#else
    (void)position_offset;
    (void)cache;
    (void)input;
    (void)output;
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
