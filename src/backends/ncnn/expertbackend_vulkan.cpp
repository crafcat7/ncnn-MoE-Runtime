#include "expertbackend_vulkan.h"

#include "linear.h"
#include "vulkandevice.h"
#include "kernels/qnk.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <option.h>

#include <array>
#include <iterator>
#include <optional>
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
VulkanExpertVictimCache::VulkanExpertVictimCache(std::shared_ptr<NcnnVulkanContext> _context, uint64_t _cache_size)
    : context(std::move(_context)),
      cache_size(_cache_size)
{
    try
    {
        upload_staging_allocator = context->device()->acquire_staging_allocator();
        for (DownloadSlot& slot : download_slots)
            slot.staging_allocator = context->device()->acquire_staging_allocator();
        worker = std::thread(&VulkanExpertVictimCache::worker_loop, this);
    }
    catch (...)
    {
        if (upload_staging_allocator)
            context->device()->reclaim_staging_allocator(upload_staging_allocator);
        for (DownloadSlot& slot : download_slots)
        {
            if (slot.staging_allocator)
                context->device()->reclaim_staging_allocator(slot.staging_allocator);
        }
        throw;
    }
}

VulkanExpertVictimCache::~VulkanExpertVictimCache()
{
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        while (!pending.empty())
        {
            pending_size -= pending.front().size;
            pending_keys.erase(pending.front().key);
            pending.pop_front();
            ++dropped_admissions;
        }
    }
    work_available.notify_all();
    if (worker.joinable())
        worker.join();
    upload_staging = ncnn::VkMat();
    context->device()->reclaim_staging_allocator(upload_staging_allocator);
    for (DownloadSlot& slot : download_slots)
    {
        slot.staging = ncnn::VkMat();
        context->device()->reclaim_staging_allocator(slot.staging_allocator);
    }
}

void VulkanExpertVictimCache::admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down,
                                    ExpertVictimExecutionMetadata execution)
{
    if (!gate_up || !down)
        return;
    const uint64_t gate_blocks_size = gate_up->mxfp4_blocks.size();
    const uint64_t gate_scales_size = gate_up->mxfp4_scales.size();
    const uint64_t down_blocks_size = down->mxfp4_blocks.size();
    const uint64_t down_scales_size = down->mxfp4_scales.size();
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    uint64_t cursor = 0;
    uint64_t gate_blocks_offset = 0;
    uint64_t gate_scales_offset = 0;
    uint64_t down_blocks_offset = 0;
    uint64_t down_scales_offset = 0;
    if (!append_segment(gate_blocks_size, alignment, cursor, gate_blocks_offset) || !append_segment(gate_scales_size, alignment, cursor, gate_scales_offset)
        || !append_segment(down_blocks_size, alignment, cursor, down_blocks_offset) || !append_segment(down_scales_size, alignment, cursor, down_scales_offset))
    {
        return;
    }
    const uint64_t size = cursor;
    if (size == 0 || size > cache_size || size > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return;
    }

    const std::lock_guard<std::mutex> lock(mutex);
    if (stopping || entries.find(key) != entries.end() || pending_keys.find(key) != pending_keys.end())
    {
        return;
    }
    while (!pending.empty() && pending_size > cache_size - size)
    {
        pending_size -= pending.front().size;
        pending_keys.erase(pending.front().key);
        pending.pop_front();
        ++dropped_admissions;
    }
    if (pending_size > cache_size - size)
    {
        ++dropped_admissions;
        return;
    }

    PendingAdmission admission;
    admission.key = std::move(key);
    admission.gate_up = std::move(gate_up);
    admission.down = std::move(down);
    admission.size = size;
    admission.gate_blocks_offset = gate_blocks_offset;
    admission.gate_scales_offset = gate_scales_offset;
    admission.down_blocks_offset = down_blocks_offset;
    admission.down_scales_offset = down_scales_offset;
    admission.execution = execution;
    pending_size += size;
    pending_keys.insert(admission.key);
    pending.push_back(std::move(admission));
    ++admissions;
    work_available.notify_one();
}

std::optional<VulkanExpertVictimCache::DeviceOperationLease> VulkanExpertVictimCache::find_device_operation(std::string_view key)
{
    std::shared_ptr<DeviceEntry> entry;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto existing = entries.find(key);
        if (existing == entries.end() || !existing->second->execution.enabled)
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
        entry->gate_blocks_size,
        entry->gate_scales_size,
        static_cast<size_t>(entry->gate_blocks_offset),
        static_cast<size_t>(entry->gate_scales_offset),
    };
    const NcnnVulkanMxfp4DeviceMatrixView down{
        entry->down_output_columns,
        entry->down_input_columns,
        entry->down_blocks_size,
        entry->down_scales_size,
        static_cast<size_t>(entry->down_blocks_offset),
        static_cast<size_t>(entry->down_scales_offset),
    };
    auto operation = NcnnVulkanMxfp4ExpertOperator ::create_from_device_storage(
        gate_up, entry->execution.gate_up_bias, down, entry->execution.down_bias, entry->execution.activation_limit,
        static_cast<uint32_t>(context->device()->info.device_index()), entry->data, entry->execution.activation,
        context->instance(),
        context->optimization_flags());
    if (!operation)
        return std::nullopt;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        entry->operation = operation;
    }
    return DeviceOperationLease{
        std::move(operation),
        std::move(entry),
    };
}

void VulkanExpertVictimCache::touch_device_operations(std::span<const std::string_view> keys)
{
    const std::lock_guard<std::mutex> lock(mutex);
    for (std::string_view key : keys)
    {
        const auto existing = entries.find(key);
        if (existing != entries.end())
            existing->second->used_at = ++clock;
    }
}

std::optional<ExpertVictimPair> VulkanExpertVictimCache::restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source)
{
    std::shared_ptr<DeviceEntry> entry;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto existing = entries.find(key);
        if (existing == entries.end())
        {
            ++misses;
            return std::nullopt;
        }
        entry = existing->second;
        entry->used_at = ++clock;
    }

    ExpertVictimPair restored;
    const bool mapped_restore = entry->data.mapped_ptr() != nullptr;
    const auto restore_started = std::chrono::steady_clock::now();
    if (!download(*entry, gate_up_source, down_source, restored))
    {
        const uint64_t restore_microseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - restore_started).count());
        const std::lock_guard<std::mutex> lock(mutex);
        restore_time_microseconds += restore_microseconds;
        ++restore_failures;
        const auto existing = entries.find(key);
        if (existing != entries.end() && existing->second == entry)
        {
            resident_size -= entry->size;
            entries.erase(existing);
        }
        return std::nullopt;
    }
    const uint64_t restore_microseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - restore_started).count());

    {
        const std::lock_guard<std::mutex> lock(mutex);
        ++hits;
        bytes_downloaded += entry->size;
        restore_time_microseconds += restore_microseconds;
        if (mapped_restore)
            ++mapped_restores;
    }
    return restored;
}

void VulkanExpertVictimCache::wait_for_background_work()
{
    std::unique_lock<std::mutex> lock(mutex);
    idle.wait(lock, [this] { return pending.empty() && active_admissions == 0; });
}

ExpertVictimCacheStatistics VulkanExpertVictimCache::statistics() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    return {hits,
            misses,
            admissions,
            0,
            0,
            0,
            stores,
            evictions,
            dropped_admissions,
            restore_failures,
            bytes_uploaded,
            bytes_downloaded,
            restore_time_microseconds,
            mapped_stores,
            mapped_restores,
            resident_size,
            pending_size};
}

uint64_t VulkanExpertVictimCache::capacity() const noexcept
{
    return cache_size;
}

bool VulkanExpertVictimCache::append_segment(uint64_t size, uint64_t alignment, uint64_t& cursor, uint64_t& offset)
{
    if (size == 0 || alignment == 0)
        return false;
    const uint64_t remainder = cursor % alignment;
    const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (cursor > std::numeric_limits<uint64_t>::max() - padding)
    {
        return false;
    }
    cursor += padding;
    offset = cursor;
    if (cursor > std::numeric_limits<uint64_t>::max() - size)
    {
        return false;
    }
    cursor += size;
    return true;
}

void VulkanExpertVictimCache::copy_payload(const PendingAdmission& admission, uint8_t* destination)
{
    std::memcpy(destination + admission.gate_blocks_offset, admission.gate_up->mxfp4_blocks.data(), admission.gate_up->mxfp4_blocks.size());
    std::memcpy(destination + admission.gate_scales_offset, admission.gate_up->mxfp4_scales.data(), admission.gate_up->mxfp4_scales.size());
    std::memcpy(destination + admission.down_blocks_offset, admission.down->mxfp4_blocks.data(), admission.down->mxfp4_blocks.size());
    std::memcpy(destination + admission.down_scales_offset, admission.down->mxfp4_scales.data(), admission.down->mxfp4_scales.size());
}

MxFp4ByteBuffer VulkanExpertVictimCache::copy_bytes(const uint8_t* source, uint64_t offset, uint64_t byte_count)
{
    MxFp4ByteBuffer result;
    result.assign(source + offset, static_cast<size_t>(byte_count));
    return result;
}

void VulkanExpertVictimCache::materialize(const DeviceEntry& entry, const TensorData& gate_up_source, const TensorData& down_source, const uint8_t* source,
                                          ExpertVictimPair& restored)
{
    restored.gate_up = std::make_shared<TensorData>();
    restored.gate_up->dtype = DType::MxFp4;
    restored.gate_up->shape = gate_up_source.shape;
    restored.gate_up->mxfp4_blocks = copy_bytes(source, entry.gate_blocks_offset, entry.gate_blocks_size);
    restored.gate_up->mxfp4_scales = copy_bytes(source, entry.gate_scales_offset, entry.gate_scales_size);
    restored.down = std::make_shared<TensorData>();
    restored.down->dtype = DType::MxFp4;
    restored.down->shape = down_source.shape;
    restored.down->mxfp4_blocks = copy_bytes(source, entry.down_blocks_offset, entry.down_blocks_size);
    restored.down->mxfp4_scales = copy_bytes(source, entry.down_scales_offset, entry.down_scales_size);
}

std::shared_ptr<VulkanExpertVictimCache::DeviceEntry> VulkanExpertVictimCache::upload(const PendingAdmission& admission)
{
    auto entry = std::make_shared<DeviceEntry>();
    entry->size = admission.size;
    entry->gate_blocks_size = admission.gate_up->mxfp4_blocks.size();
    entry->gate_scales_size = admission.gate_up->mxfp4_scales.size();
    entry->down_blocks_size = admission.down->mxfp4_blocks.size();
    entry->down_scales_size = admission.down->mxfp4_scales.size();
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
        const std::lock_guard<std::mutex> command_lock(context->command_mutex());
        entry->data.create(static_cast<int>(admission.size), sizeof(uint8_t), context->blob_allocator());
    }
    if (entry->data.empty())
        return {};
    if (entry->data.mapped_ptr())
    {
        std::memset(entry->data.mapped_ptr(), 0, static_cast<size_t>(admission.size));
        copy_payload(admission, static_cast<uint8_t*>(entry->data.mapped_ptr()));
        entry->data.allocator->flush(entry->data.data);
        entry->data.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        entry->data.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        return entry;
    }

    upload_staging.create(static_cast<int>(admission.size), sizeof(uint8_t), upload_staging_allocator);
    if (upload_staging.empty() || !upload_staging.mapped_ptr())
        return {};

    uint8_t* destination = static_cast<uint8_t*>(upload_staging.mapped_ptr());
    std::memset(destination, 0, static_cast<size_t>(admission.size));
    copy_payload(admission, destination);
    upload_staging.allocator->flush(upload_staging.data);
    upload_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    upload_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    const std::lock_guard<std::mutex> command_lock(context->command_mutex());
    ncnn::Option option;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = upload_staging_allocator;
    ncnn::VkCompute command(context->device(), context->command_optimization_flags());
    command.record_clone(upload_staging, entry->data, option);
    if (entry->data.empty() || command.submit_and_wait() != 0)
        return {};
    return entry;
}

bool VulkanExpertVictimCache::download(const DeviceEntry& entry, const TensorData& gate_up_source, const TensorData& down_source, ExpertVictimPair& restored)
{
    if (!gate_up_source.mxfp4_file_storage || !down_source.mxfp4_file_storage)
    {
        return false;
    }
    const MxFp4FileStorage& gate_file = *gate_up_source.mxfp4_file_storage;
    const MxFp4FileStorage& down_file = *down_source.mxfp4_file_storage;
    if (gate_file.blocks_size != entry.gate_blocks_size || gate_file.scales_size != entry.gate_scales_size || down_file.blocks_size != entry.down_blocks_size
        || down_file.scales_size != entry.down_scales_size)
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

    DownloadSlot& slot = download_slots[next_download_slot.fetch_add(1, std::memory_order_relaxed) % download_slots.size()];
    const std::lock_guard<std::mutex> slot_lock(slot.mutex);
    slot.staging.create(static_cast<int>(entry.size), sizeof(uint8_t), slot.staging_allocator);
    if (slot.staging.empty() || !slot.staging.mapped_ptr())
        return false;

    {
        const std::lock_guard<std::mutex> command_lock(context->command_mutex());
        ncnn::Option option;
        option.blob_vkallocator = slot.staging_allocator;
        option.workspace_vkallocator = slot.staging_allocator;
        option.staging_vkallocator = slot.staging_allocator;
        ncnn::VkCompute command(context->device(), context->command_optimization_flags());
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

void VulkanExpertVictimCache::worker_loop()
{
    for (;;)
    {
        PendingAdmission admission;
        {
            std::unique_lock<std::mutex> lock(mutex);
            work_available.wait(lock, [this] { return stopping || !pending.empty(); });
            if (stopping && pending.empty())
                return;
            admission = std::move(pending.front());
            pending.pop_front();
            ++active_admissions;
        }

        std::shared_ptr<DeviceEntry> entry = upload(admission);
        {
            const std::lock_guard<std::mutex> lock(mutex);
            pending_size -= admission.size;
            pending_keys.erase(admission.key);
            if (entry)
            {
                while (resident_size > cache_size - entry->size)
                {
                    auto victim = entries.end();
                    for (auto iterator = entries.begin(); iterator != entries.end(); ++iterator)
                    {
                        if (iterator->second.use_count() != 1)
                        {
                            continue;
                        }
                        if (victim == entries.end() || iterator->second->used_at < victim->second->used_at)
                        {
                            victim = iterator;
                        }
                    }
                    if (victim == entries.end())
                        break;
                    resident_size -= victim->second->size;
                    entries.erase(victim);
                    ++evictions;
                }
                if (resident_size <= cache_size - entry->size)
                {
                    entry->used_at = ++clock;
                    resident_size += entry->size;
                    bytes_uploaded += entry->size;
                    if (entry->data.mapped_ptr())
                        ++mapped_stores;
                    ++stores;
                    entries[admission.key] = std::move(entry);
                }
                else
                {
                    ++dropped_admissions;
                }
            }
            --active_admissions;
            if (pending.empty() && active_admissions == 0)
            {
                idle.notify_all();
            }
        }
    }
}
#endif // NCNN_MOE_WITH_VULKAN

std::shared_ptr<ExpertVictimCache> create_vulkan_victim_cache(uint64_t cache_size, uint32_t device_index,
                                                              const NcnnVulkanContextInstancePtr& context_instance,
                                                              uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context = NcnnVulkanContext::acquire(
        device_index,
        context_instance,
        optimization_flags);
    if (!context || cache_size == 0)
        return {};
    return std::make_shared<VulkanExpertVictimCache>(context, cache_size);
#else
    (void)cache_size;
    (void)device_index;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
VulkanExpertBackend::VulkanExpertBackend(
    uint64_t _cache_size,
    uint32_t _vulkan_device_index,
    std::shared_ptr<VulkanExpertVictimCache> _device_weight_source,
    NcnnVulkanContextInstancePtr _context_instance,
    uint64_t _optimization_flags)
    : cache_size(_cache_size),
      vulkan_device_index(_vulkan_device_index),
      context_instance(std::move(_context_instance)),
      optimization_flags(_optimization_flags),
      device_weight_source(std::move(_device_weight_source))
{
    vulkan_context = NcnnVulkanContext::acquire(
        vulkan_device_index,
        context_instance,
        optimization_flags);
    if (vulkan_context && cache_size != 0)
    {
        const uint64_t allocator_block_size = std::min<uint64_t>(cache_size, UINT64_C(64) * 1024 * 1024);
        expert_weight_allocator = std::make_shared<ncnn::VkBlobAllocator>(vulkan_context->device(), static_cast<size_t>(allocator_block_size));
    }
    try
    {
        worker = std::thread(&VulkanExpertBackend::worker_loop, this);
        execution_worker = std::thread(&VulkanExpertBackend::execution_loop, this);
    }
    catch (...)
    {
        stop_workers();
        throw;
    }
}

VulkanExpertBackend::~VulkanExpertBackend()
{
    stop_workers();
    retired_entries.clear();
    entries.clear();
}

void VulkanExpertBackend::stop_workers()
{
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        dropped_admissions += pending.size();
        pending.clear();
        pending_keys.clear();
        pending_size = 0;
    }
    work_available.notify_all();
    execution_available.notify_all();
    if (worker.joinable())
        worker.join();
    if (execution_worker.joinable())
        execution_worker.join();
}

void VulkanExpertBackend::set_foreground_active(bool active) noexcept
{
    std::unique_lock<std::mutex> lock(mutex);
    if (active)
    {
        if (foreground_depth++ == 0)
        {
            admission_idle.wait(lock, [this] {
                return active_admissions == 0;
            });
        }
    }
    else if (foreground_depth != 0)
    {
        if (--foreground_depth == 0)
            work_available.notify_all();
    }
}

void VulkanExpertBackend::admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down,
                                const TensorData* down_bias, uint32_t residency_group, float activation_limit,
                                ExpertActivation activation)
{
    const bool bfloat16_expert = gate_up
                                 && down
                                 && gate_up->dtype == DType::BFloat16
                                 && down->dtype == DType::BFloat16
                                 && activation == ExpertActivation::Silu
                                 && gate_up->shape.size() == 2
                                 && down->shape.size() == 2
                                 && gate_up->shape[0] % 2 == 0
                                 && gate_up->shape[0] / 2 % 128 == 0
                                 && down->shape[1] == gate_up->shape[0] / 2
                                 && gate_up->bfloat16_values().size() == gate_up->element_count()
                                 && down->bfloat16_values().size() == down->element_count();
    const bool mxfp4_expert = gate_up
                              && down
                              && gate_up->dtype == DType::MxFp4
                              && down->dtype == DType::MxFp4
                              && gate_up->shape.size() == 2
                              && down->shape.size() == 2
                              && gate_up->shape[0] % 2 == 0
                              && down->shape[1] == gate_up->shape[0] / 2;
    const bool qnk_expert = gate_up
                            && down
                            && is_qnk_dtype(gate_up->dtype)
                            && gate_up->dtype == down->dtype
                            && gate_up->shape.size() == 2
                            && down->shape.size() == 2
                            && gate_up->shape[0] % 2 == 0
                            && down->shape[1] == gate_up->shape[0] / 2
                            && qnk_shape_supported(gate_up->dtype, gate_up->shape[0], gate_up->shape[1])
                            && qnk_shape_supported(down->dtype, down->shape[0], down->shape[1]);
    if (key.empty()
        || (!mxfp4_expert && !qnk_expert && !bfloat16_expert)
        || (activation != ExpertActivation::Silu
            && activation != ExpertActivation::GptOssSwiGlu
            && activation != ExpertActivation::DeepSeekSwiGlu)
        || activation_limit < 0.0f)
    {
        return;
    }
    const uint64_t size = expert_matrix_bytes(*gate_up) + expert_matrix_bytes(*down) + tensor_bytes(gate_up_bias) + tensor_bytes(down_bias);
    if (size == 0 || size > cache_size)
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
    admission.size = size;
    const std::lock_guard<std::mutex> lock(mutex);
    if (stopping || entries.find(admission.key) != entries.end() || pending_keys.find(admission.key) != pending_keys.end())
    {
        return;
    }
    while (!pending.empty() && pending_size > cache_size - size)
    {
        pending_size -= pending.front().size;
        pending_keys.erase(pending.front().key);
        pending.pop_front();
        ++dropped_admissions;
    }
    if (pending_size > cache_size - size)
    {
        ++dropped_admissions;
        return;
    }
    pending_size += size;
    pending_keys.insert(admission.key);
    pending.push_back(std::move(admission));
    ++admissions;
    work_available.notify_one();
}

ExpertBackendExecutionResult VulkanExpertBackend::try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output)
{
    const ExpertBackendRequest request{key, &input, &output};
    std::vector<ExpertBackendExecutionResult> results = try_execute_batch(std::span<const ExpertBackendRequest>(&request, 1));
    return results.empty() ? ExpertBackendExecutionResult ::Failed : results.front();
}

std::vector<ExpertBackendExecutionResult> VulkanExpertBackend::try_execute_batch(std::span<const ExpertBackendRequest> requests)
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

std::unique_ptr<ExpertSubmission> VulkanExpertBackend::submit_batch(std::span<const ExpertBackendRequest> requests)
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
        std::unique_lock<std::mutex> lock(mutex);
        // Admission is asynchronous; resident selection is fixed.
        std::vector<Selection>& candidates = work->selected;
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            const ExpertBackendRequest& request = requests[request_index];
            if (!request.input || !request.output || request.input->rows() == 0)
            {
                work->planned[request_index] = ExpertBackendExecutionResult ::Failed;
                continue;
            }
            auto existing = entries.find(request.key);
            if (existing == entries.end())
            {
                std::optional<VulkanExpertVictimCache::DeviceOperationLease> device_lease;
                const std::shared_ptr<VulkanExpertVictimCache> victim_cache = device_weight_source;
                if (victim_cache)
                {
                    // Vulkan allocation must stay outside the scheduler lock.
                    lock.unlock();
                    device_lease = victim_cache->find_device_operation(request.key);
                    lock.lock();
                    existing = entries.find(request.key);
                }
                if (existing != entries.end())
                {
                    // Prefer an admission completed during victim lookup.
                    device_lease.reset();
                }
                else if (!device_lease)
                {
                    ++misses;
                    if (device_weight_source)
                    {
                        ++device_source_misses;
                    }
                    continue;
                }
                else
                {
                    std::shared_ptr<Entry> entry = std::make_shared<Entry>();
                    entry->key.assign(request.key);
                    entry->operation = std::move(device_lease->operation);
                    entry->device_source_pin = std::move(device_lease->pin);
                    entry->size = request.weight_size;
                    ++hits;
                    ++device_source_hits;
                    candidates.push_back({
                        request_index,
                        std::move(entry),
                    });
                    continue;
                }
            }
            std::shared_ptr<Entry> entry = existing->second;
            touch_locked(*entry, true);
            ++hits;
            candidates.push_back({
                request_index,
                std::move(entry),
            });
        }

        // This backend is created only for mixed Vulkan execution. Once
        // a resident Expert is available, execute it on the device even
        // for a single-token wave; the caller keeps CPU fallback for
        // non-resident or failed requests.
        for (const Selection& selection : candidates)
        {
            work->planned[selection.request_index] = ExpertBackendExecutionResult ::Executed;
        }
        if (work->selected.empty())
        {
            work->final = work->planned;
            work->done = true;
        }
        else
        {
            execution_pending.push_back(work);
            execution_available.notify_one();
        }
    }
    return std::make_unique<Submission>(std::move(work));
}

void VulkanExpertBackend::observe_cpu(uint32_t token_count, uint64_t weight_size, uint64_t elapsed_microseconds)
{
    // Placement is deterministic and independent of runtime timing.
    (void)token_count;
    (void)weight_size;
    (void)elapsed_microseconds;
}

void VulkanExpertBackend::observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds)
{
    // Placement does not use timing samples.
    (void)token_count;
    (void)total_weight_bytes;
    (void)accelerated_weight_bytes;
    (void)elapsed_microseconds;
}

void VulkanExpertBackend::wait_for_background_work()
{
    std::unique_lock<std::mutex> lock(mutex);
    admission_idle.wait(lock, [this] { return pending.empty() && active_admissions == 0; });
}

ExpertBackendStatistics VulkanExpertBackend::statistics() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    ExpertBackendStatistics result;
    result.hits = hits;
    result.misses = misses;
    result.admissions = admissions;
    result.stores = stores;
    result.evictions = evictions;
    result.dropped_admissions = dropped_admissions;
    result.executions = executions;
    result.execution_failures = execution_failures;
    result.bytes_uploaded = bytes_uploaded;
    result.resident_size = resident_size;
    result.pending_size = pending_size;
    result.execution_time_microseconds = execution_time_microseconds;
    result.arc_recent_size = recent_size;
    result.arc_frequent_size = frequent_size;
    result.arc_recent_target_size = recent_target_size;
    result.arc_recent_ghost_size = recent_ghost_size;
    result.arc_frequent_ghost_size = frequent_ghost_size;
    result.device_source_hits = device_source_hits;
    result.device_source_misses = device_source_misses;
    result.device_source_executions = device_source_executions;
    result.device_source_execution_failures = device_source_execution_failures;
    result.route_aggregation_batches = route_aggregation_batches;
    result.route_aggregation_routes = route_aggregation_routes;
    result.route_aggregation_bytes_saved = route_aggregation_bytes_saved;
    return result;
}

std::vector<ExpertBackendDeviceStatistics> VulkanExpertBackend::device_statistics() const
{
    return {{
        vulkan_device_index,
        cache_size,
        statistics(),
    }};
}

uint64_t VulkanExpertBackend::capacity() const noexcept
{
    return cache_size;
}

VulkanExpertBackend::Submission::Submission(std::shared_ptr<WorkItem> _work)
    : work(std::move(_work))
{
}

VulkanExpertBackend::Submission::~Submission()
{
    if (work && !waited)
        (void)wait();
    if (work && !committed && !aborted)
        abort();
}

std::span<const ExpertBackendExecutionResult> VulkanExpertBackend::Submission::reservations() const noexcept
{
    return work->planned;
}

std::vector<ExpertBackendExecutionResult> VulkanExpertBackend::Submission::wait()
{
    std::unique_lock<std::mutex> lock(work->mutex);
    work->completed.wait(lock, [this] { return work->done; });
    waited = true;
    return work->final;
}

bool VulkanExpertBackend::Submission::commit()
{
    if (!waited)
        (void)wait();
    if (committed || aborted)
        return committed;
    // Validate the complete publication set before touching any
    // caller-owned buffer. A failed reservation must be a safe CPU
    // fallback, never a partially published batch.
    ActivationBuffer* route_output = nullptr;
    size_t route_requests = 0;
    size_t executed_route_requests = 0;
    size_t completed_route_requests = 0;
    bool require_all_route_requests = false;
    for (size_t index = 0; index < work->final.size(); ++index)
    {
        ExpertBackendRequest& client = work->client_requests[index];
        const bool executed = work->final[index] == ExpertBackendExecutionResult::Executed;
        if (executed && !client.output)
        {
            aborted = true;
            return false;
        }
        if (!client.route_aggregation.output)
        {
            if (work->private_route_completed[index] != 0)
            {
                aborted = true;
                return false;
            }
            continue;
        }
        ++route_requests;
        require_all_route_requests = require_all_route_requests
                                     || client.route_aggregation.require_all_requests;
        if (executed)
            ++executed_route_requests;
        if (work->private_route_completed[index] != 0)
        {
            if (!executed
                || (route_output
                    && route_output != client.route_aggregation.output))
            {
                aborted = true;
                return false;
            }
            route_output = client.route_aggregation.output;
            ++completed_route_requests;
        }
    }
    if (completed_route_requests != 0)
    {
        const size_t expected_route_requests = require_all_route_requests
                                                   ? route_requests
                                                   : executed_route_requests;
        if (completed_route_requests != expected_route_requests)
        {
            aborted = true;
            return false;
        }
    }
    bool route_published = false;
    for (size_t index = 0; index < work->final.size(); ++index)
    {
        if (work->final[index] != ExpertBackendExecutionResult::Executed)
            continue;
        ExpertBackendRequest& client = work->client_requests[index];
        client.output->swap(work->private_outputs[index]);
        if (work->private_route_completed[index] != 0)
        {
            if (!route_published)
            {
                client.route_aggregation.output->swap(work->private_aggregation);
                route_published = true;
            }
            if (client.route_aggregation.completed)
                *client.route_aggregation.completed = 1;
        }
    }
    committed = true;
    return true;
}

void VulkanExpertBackend::Submission::abort() noexcept
{
    if (committed || aborted)
        return;
    aborted = true;
}

uint64_t VulkanExpertBackend::mxfp4_bytes(const TensorData& tensor)
{
    return tensor.mxfp4_blocks.size() + tensor.mxfp4_scales.size();
}

uint64_t VulkanExpertBackend::qnk_bytes(const TensorData& tensor)
{
    return tensor.qnk_values().size();
}

uint64_t VulkanExpertBackend::expert_matrix_bytes(const TensorData& tensor)
{
    if (tensor.dtype == DType::MxFp4)
        return mxfp4_bytes(tensor);
    if (tensor.dtype == DType::BFloat16)
        return tensor.bfloat16_values().size() * sizeof(uint16_t);
    return qnk_bytes(tensor);
}

uint64_t VulkanExpertBackend::tensor_bytes(const TensorData* tensor)
{
    if (!tensor)
        return 0;
    if (tensor->dtype == DType::Float32)
        return tensor->float32_values().size() * sizeof(float);
    if (tensor->dtype == DType::BFloat16)
        return tensor->bfloat16_values().size() * sizeof(uint16_t);
    return 0;
}

void VulkanExpertBackend::touch_locked(Entry& entry, bool repeated)
{
    if (entry.list == ArcList::Recent && repeated)
    {
        frequent.splice(frequent.end(), recent, entry.position);
        recent_size -= entry.size;
        entry.list = ArcList::Frequent;
        frequent_size += entry.size;
        return;
    }
    std::list<std::string>& list = entry.list == ArcList::Recent ? recent : frequent;
    list.splice(list.end(), list, entry.position);
    entry.position = std::prev(list.end());
}

void VulkanExpertBackend::erase_ghost_entry(GhostIndex& index, GhostList& list, uint64_t& size, const std::string& key)
{
    const auto existing = index.find(key);
    if (existing == index.end())
        return;
    size -= existing->second->size;
    list.erase(existing->second);
    index.erase(existing);
}

void VulkanExpertBackend::erase_ghost_locked(const std::string& key)
{
    erase_ghost_entry(recent_ghost_index, recent_ghost, recent_ghost_size, key);
    erase_ghost_entry(frequent_ghost_index, frequent_ghost, frequent_ghost_size, key);
}

void VulkanExpertBackend::add_ghost_locked(const Entry& entry)
{
    erase_ghost_locked(entry.key);
    Ghost ghost{entry.key, entry.size};
    if (entry.list == ArcList::Recent)
    {
        recent_ghost.push_back(std::move(ghost));
        const auto position = std::prev(recent_ghost.end());
        recent_ghost_index[position->key] = position;
        recent_ghost_size += entry.size;
    }
    else
    {
        frequent_ghost.push_back(std::move(ghost));
        const auto position = std::prev(frequent_ghost.end());
        frequent_ghost_index[position->key] = position;
        frequent_ghost_size += entry.size;
    }
    trim_ghosts_locked();
}

void VulkanExpertBackend::trim_ghost_front(GhostIndex& index, GhostList& list, uint64_t& size)
{
    if (list.empty())
        return;
    size -= list.front().size;
    index.erase(list.front().key);
    list.pop_front();
}

void VulkanExpertBackend::trim_ghosts_locked()
{
    while (recent_ghost_size + frequent_ghost_size > cache_size)
    {
        if (recent_ghost_size >= frequent_ghost_size && !recent_ghost.empty())
        {
            trim_ghost_front(recent_ghost_index, recent_ghost, recent_ghost_size);
        }
        else
        {
            trim_ghost_front(frequent_ghost_index, frequent_ghost, frequent_ghost_size);
        }
    }
}

uint64_t VulkanExpertBackend::arc_delta(uint64_t required, uint64_t numerator, uint64_t denominator) const
{
    if (denominator == 0 || numerator <= denominator)
        return required;
    const uint64_t ratio = numerator / denominator;
    if (required != 0 && ratio > cache_size / required)
        return cache_size;
    return std::min(cache_size, std::max(required, required * ratio));
}

bool VulkanExpertBackend::consume_ghost_locked(const std::string& key, uint64_t required, bool& promote, bool& from_frequent)
{
    promote = false;
    from_frequent = false;
    const auto recent_entry = recent_ghost_index.find(key);
    if (recent_entry != recent_ghost_index.end())
    {
        const uint64_t adjustment = arc_delta(required, frequent_ghost_size, recent_ghost_size);
        recent_ghost_size -= recent_entry->second->size;
        recent_ghost.erase(recent_entry->second);
        recent_ghost_index.erase(recent_entry);
        recent_target_size = std::min(cache_size, recent_target_size + std::min(adjustment, cache_size - recent_target_size));
        promote = true;
        return true;
    }
    const auto frequent_entry = frequent_ghost_index.find(key);
    if (frequent_entry == frequent_ghost_index.end())
    {
        return false;
    }
    const uint64_t adjustment = arc_delta(required, recent_ghost_size, frequent_ghost_size);
    frequent_ghost_size -= frequent_entry->second->size;
    frequent_ghost.erase(frequent_entry->second);
    frequent_ghost_index.erase(frequent_entry);
    recent_target_size -= std::min(adjustment, recent_target_size);
    promote = true;
    from_frequent = true;
    return true;
}

std::shared_ptr<VulkanExpertBackend::Entry> VulkanExpertBackend::find_victim_locked(const std::list<std::string>& list, uint32_t residency_group)
{
    for (const std::string& key : list)
    {
        const auto existing = entries.find(key);
        if (existing == entries.end() || existing->second.use_count() != 1
            || (residency_group != std::numeric_limits<uint32_t>::max() && existing->second->residency_group != residency_group))
        {
            continue;
        }
        return existing->second;
    }
    return {};
}

bool VulkanExpertBackend::evict_one_locked(bool incoming_from_frequent, uint32_t incoming_group, uint64_t required)
{
    const bool prefer_recent = recent_size > recent_target_size || (incoming_from_frequent && recent_size == recent_target_size);
    uint32_t preferred_group = std::numeric_limits<uint32_t>::max();
    if (!residency_group_sizes.empty())
    {
        const uint64_t fair_share = cache_size / residency_group_sizes.size();
        if (incoming_group < residency_group_sizes.size() && residency_group_sizes[incoming_group] + required > fair_share)
        {
            preferred_group = incoming_group;
        }
        else
        {
            uint64_t maximum_excess = 0;
            for (uint32_t group = 0; group < residency_group_sizes.size(); ++group)
            {
                const uint64_t size = residency_group_sizes[group];
                const uint64_t excess = size > fair_share ? size - fair_share : 0;
                if (excess > maximum_excess)
                {
                    maximum_excess = excess;
                    preferred_group = group;
                }
            }
        }
    }
    std::shared_ptr<Entry> victim = prefer_recent ? find_victim_locked(recent, preferred_group) : find_victim_locked(frequent, preferred_group);
    if (!victim)
    {
        victim = prefer_recent ? find_victim_locked(frequent, preferred_group) : find_victim_locked(recent, preferred_group);
    }
    if (!victim && preferred_group != std::numeric_limits<uint32_t>::max())
    {
        victim = prefer_recent ? find_victim_locked(recent, std::numeric_limits<uint32_t>::max())
                               : find_victim_locked(frequent, std::numeric_limits<uint32_t>::max());
        if (!victim)
        {
            victim = prefer_recent ? find_victim_locked(frequent, std::numeric_limits<uint32_t>::max())
                                   : find_victim_locked(recent, std::numeric_limits<uint32_t>::max());
        }
    }
    if (!victim)
        return false;
    add_ghost_locked(*victim);
    if (victim->list == ArcList::Recent)
    {
        recent.erase(victim->position);
        recent_size -= victim->size;
    }
    else
    {
        frequent.erase(victim->position);
        frequent_size -= victim->size;
    }
    resident_size -= victim->size;
    if (victim->residency_group < residency_group_sizes.size())
    {
        residency_group_sizes[victim->residency_group] -= victim->size;
    }
    entries.erase(victim->key);
    retired_entries.push_back(std::move(victim));
    ++evictions;
    return true;
}

void VulkanExpertBackend::execute_work_item(const std::shared_ptr<WorkItem>& work)
{
    const auto started = std::chrono::steady_clock::now();
    const IndexedBatchResult indexed_result = has_flag(
                                                  optimization_flags,
                                                  OptimizationVulkanIndexedExperts)
                                                  ? forward_indexed_batch(work->requests, work->selected)
                                                  : IndexedBatchResult::NotSupported;
    const bool executed = indexed_result == IndexedBatchResult::Executed
                          || forward_batch(work->requests, work->selected)
                          || forward_bfloat16_batch(work->requests, work->selected)
                          || forward_qnk_batch(work->requests, work->selected);
    const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
    uint64_t aggregated_route_count = 0;
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
            aggregated_route_count += request.route_aggregation.routes.size();
            route_aggregated = true;
        }
    }
    uint64_t saved_transfer_size = 0;
    if (route_aggregated && aggregated_route_count > route_aggregation_token_count && route_aggregation_columns != 0)
    {
        const uint64_t saved_rows = aggregated_route_count - route_aggregation_token_count;
        if (saved_rows <= std::numeric_limits<uint64_t>::max() / route_aggregation_columns / sizeof(float))
        {
            saved_transfer_size = saved_rows * route_aggregation_columns * sizeof(float);
        }
    }
    std::vector<ExpertBackendExecutionResult> final = work->planned;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        execution_time_microseconds += elapsed;
        if (!executed)
        {
            execution_failures += work->selected.size();
            for (const Selection& selection : work->selected)
            {
                if (selection.entry->device_source_pin)
                {
                    ++device_source_execution_failures;
                }
                final[selection.request_index] = ExpertBackendExecutionResult ::Failed;
            }
        }
        else
        {
            executions += work->selected.size();
            if (route_aggregated)
            {
                ++route_aggregation_batches;
                route_aggregation_routes += aggregated_route_count;
                route_aggregation_bytes_saved += saved_transfer_size;
            }
            for (const Selection& selection : work->selected)
            {
                if (selection.entry->device_source_pin)
                {
                    ++device_source_executions;
                }
            }
        }
    }
    if (executed && device_weight_source)
    {
        static constexpr size_t touch_batch_size = 256;
        std::array<std::string_view, touch_batch_size> keys;
        size_t key_count = 0;
        for (const Selection& selection : work->selected)
        {
            if (!selection.entry->device_source_pin)
                continue;
            keys[key_count++] = selection.entry->key;
            if (key_count == keys.size())
            {
                device_weight_source->touch_device_operations(std::span<const std::string_view>(keys.data(), key_count));
                key_count = 0;
            }
        }
        if (key_count != 0)
            device_weight_source->touch_device_operations(std::span<const std::string_view>(keys.data(), key_count));
    }
    {
        const std::lock_guard<std::mutex> lock(work->mutex);
        work->final = std::move(final);
        work->done = true;
    }
    work->completed.notify_all();
}

void VulkanExpertBackend::execution_loop()
{
    while (true)
    {
        std::shared_ptr<WorkItem> work;
        {
            std::unique_lock<std::mutex> lock(mutex);
            execution_available.wait(lock, [this] { return stopping || !execution_pending.empty(); });
            if (execution_pending.empty())
            {
                if (stopping)
                    return;
                continue;
            }
            work = std::move(execution_pending.front());
            execution_pending.pop_front();
        }
        execute_work_item(work);
    }
}

void VulkanExpertBackend::finish_admission_locked()
{
    --active_admissions;
    if (active_admissions == 0)
        admission_idle.notify_all();
}

void VulkanExpertBackend::worker_loop()
{
    static constexpr size_t maximum_upload_batch = 8;
    while (true)
    {
        std::vector<std::shared_ptr<Entry>> retired;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            retired.swap(retired_entries);
        }
        retired.clear();
        std::vector<PendingAdmission> batch;
        {
            std::unique_lock<std::mutex> lock(mutex);
            work_available.wait(lock, [this] {
                // Keep admission moving while the foreground session runs.
                return stopping || !pending.empty();
            });
            if (stopping)
                return;
            batch.push_back(std::move(pending.front()));
            pending.pop_front();
            ++active_admissions;
            if (has_flag(
                    optimization_flags,
                    OptimizationVulkanExpertBatchAdmission)
                && batch.front().gate_up
                && batch.front().gate_up->dtype == DType::MxFp4)
            {
                // Keep the batch homogeneous.  MXFP4 admissions can
                // share one VkTransfer command; BF16 and QnK creation
                // paths still use their existing upload implementation.
                while (batch.size() < maximum_upload_batch
                       && !pending.empty()
                       && pending.front().gate_up
                       && pending.front().gate_up->dtype == DType::MxFp4)
                {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();
                    ++active_admissions;
                }
            }
        }

        std::vector<std::shared_ptr<Entry>> loaded(batch.size());
        {
            std::optional<NcnnVulkanWeightUploadBatch> upload_batch;
            if (batch.size() > 1)
                upload_batch.emplace(vulkan_context);
            for (size_t admission_index = 0; admission_index < batch.size(); ++admission_index)
            {
                const PendingAdmission& admission = batch[admission_index];
                std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
                std::shared_ptr<NcnnVulkanQnkExpertOperator> qnk_operation;
                std::shared_ptr<NcnnVulkanBfloat16ExpertOperator> bfloat16_operation;
                if (admission.gate_up->dtype == DType::MxFp4)
                {
                    operation = NcnnVulkanMxfp4ExpertOperator::create_with_allocator(
                        *admission.gate_up,
                        admission.gate_up_bias.get(),
                        *admission.down,
                        admission.down_bias.get(),
                        admission.activation_limit,
                        vulkan_device_index,
                        expert_weight_allocator.get(),
                        admission.activation,
                        context_instance,
                        optimization_flags,
                        upload_batch ? &*upload_batch : nullptr);
                }
                else if (admission.gate_up->dtype == DType::BFloat16)
                {
                    bfloat16_operation = NcnnVulkanBfloat16ExpertOperator::create_with_allocator(
                        *admission.gate_up,
                        admission.gate_up_bias.get(),
                        *admission.down,
                        admission.down_bias.get(),
                        admission.activation_limit,
                        vulkan_device_index,
                        expert_weight_allocator.get(),
                        admission.activation,
                        context_instance,
                        optimization_flags);
                }
                else
                {
                    qnk_operation = NcnnVulkanQnkExpertOperator::create_with_allocator(
                        *admission.gate_up,
                        admission.gate_up_bias.get(),
                        *admission.down,
                        admission.down_bias.get(),
                        admission.activation_limit,
                        vulkan_device_index,
                        expert_weight_allocator.get(),
                        admission.activation,
                        context_instance,
                        optimization_flags);
                }
                if (operation || qnk_operation || bfloat16_operation)
                {
                    loaded[admission_index] = std::make_shared<Entry>();
                    loaded[admission_index]->key = admission.key;
                    loaded[admission_index]->weight_allocator = expert_weight_allocator;
                    loaded[admission_index]->operation = std::move(operation);
                    loaded[admission_index]->qnk_operation = std::move(qnk_operation);
                    loaded[admission_index]->bfloat16_operation = std::move(bfloat16_operation);
                    loaded[admission_index]->size = admission.size;
                    loaded[admission_index]->residency_group = admission.residency_group;
                }
            }
            if (upload_batch && !upload_batch->submit())
                std::fill(loaded.begin(), loaded.end(), std::shared_ptr<Entry>());
        }

        const std::lock_guard<std::mutex> lock(mutex);
        for (size_t admission_index = 0; admission_index < batch.size(); ++admission_index)
        {
            const PendingAdmission& admission = batch[admission_index];
            pending_size -= admission.size;
            pending_keys.erase(admission.key);
            std::shared_ptr<Entry>& entry = loaded[admission_index];
            if (stopping)
            {
                finish_admission_locked();
                continue;
            }
            if (!entry || entries.find(entry->key) != entries.end())
            {
                if (!entry)
                    ++dropped_admissions;
                finish_admission_locked();
                continue;
            }
            bool promote = false;
            bool from_frequent = false;
            (void)consume_ghost_locked(entry->key, entry->size, promote, from_frequent);
            while (resident_size > cache_size - entry->size)
            {
                if (!evict_one_locked(from_frequent, entry->residency_group, entry->size))
                {
                    ++dropped_admissions;
                    entry.reset();
                    break;
                }
            }
            if (!entry)
            {
                finish_admission_locked();
                continue;
            }
            if (promote)
            {
                frequent.push_back(entry->key);
                entry->position = std::prev(frequent.end());
                entry->list = ArcList::Frequent;
                frequent_size += entry->size;
            }
            else
            {
                recent.push_back(entry->key);
                entry->position = std::prev(recent.end());
                entry->list = ArcList::Recent;
                recent_size += entry->size;
            }
            resident_size += entry->size;
            if (entry->residency_group >= residency_group_sizes.size())
            {
                residency_group_sizes.resize(static_cast<size_t>(entry->residency_group) + 1, 0);
            }
            residency_group_sizes[entry->residency_group] += entry->size;
            bytes_uploaded += entry->size;
            const std::string entry_key = entry->key;
            entries.emplace(entry_key, std::move(entry));
            ++stores;
            finish_admission_locked();
        }
    }
}

#endif // NCNN_MOE_WITH_VULKAN

std::shared_ptr<ExpertBackend> create_vulkan_expert_backend(uint64_t cache_size, uint32_t device_index,
                                                            std::shared_ptr<ExpertVictimCache> device_weight_source,
                                                            const NcnnVulkanContextInstancePtr& context_instance,
                                                            uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    auto source = std::dynamic_pointer_cast<VulkanExpertVictimCache>(std::move(device_weight_source));
    if ((cache_size == 0 && !source) || get_gpu_count() == 0)
    {
        return {};
    }
    if (device_index == automatic_vulkan_device_index)
    {
        device_index = static_cast<uint32_t>(ncnn::get_default_gpu_index());
    }
    if (device_index >= get_gpu_count())
    {
        return {};
    }
    return std::make_shared<VulkanExpertBackend>(
        cache_size,
        device_index,
        std::move(source),
        context_instance,
        optimization_flags);
#else
    (void)cache_size;
    (void)device_index;
    (void)device_weight_source;
    (void)context_instance;
    (void)optimization_flags;
    return {};
#endif
}

} // namespace moe
} // namespace ncnn
