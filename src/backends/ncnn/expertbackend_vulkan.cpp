#include "expertbackend_vulkan.h"

#include "linear.h"
#include "vulkancontext.h"
#include "experttransfer_vulkan.h"
#include "storage/expertresidency.h"
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
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
VulkanExpertVictimCache::VulkanExpertVictimCache(std::shared_ptr<VulkanContext> _context, uint64_t _cache_size)
    : context(std::move(_context)),
      cache_size(_cache_size)
{
    try
    {
        weight_allocator = std::make_shared<VulkanExpertWeightAllocator>(context->device(), static_cast<size_t>(std::min<uint64_t>(cache_size, UINT64_C(64) * 1024 * 1024)));
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
    {
        const std::lock_guard<std::mutex> lock(mutex);
        for (auto& [key, entry] : entries) entry->residency_token.reset();
        entries.clear();
    }
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
    // Victim admission is best effort. Publish accounting only after both
    // allocations succeed, and roll back the key if deque growth fails.
    try
    {
        const auto [pending_key, inserted] = pending_keys.insert(admission.key);
        if (!inserted) return;
        try
        {
            pending.push_back(std::move(admission));
        }
        catch (...)
        {
            pending_keys.erase(pending_key);
            throw;
        }
    }
    catch (...)
    {
        ++dropped_admissions;
        return;
    }
    pending_size += size;
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
            // The returned operator itself retains the entry's allocator,
            // even when a caller keeps it after releasing the explicit pin.
            return DeviceOperationLease{
                std::shared_ptr<Mxfp4Expert_vulkan>(entry, entry->operation.get()),
                entry,
            };
        }
        if (entry->operation_attempted)
            return std::nullopt;
        entry->operation_attempted = true;
    }
    const Mxfp4DeviceMatrixView_vulkan gate_up{
        entry->gate_output_columns,
        entry->gate_input_columns,
        entry->gate_blocks_size,
        entry->gate_scales_size,
        static_cast<size_t>(entry->gate_blocks_offset),
        static_cast<size_t>(entry->gate_scales_offset),
    };
    const Mxfp4DeviceMatrixView_vulkan down{
        entry->down_output_columns,
        entry->down_input_columns,
        entry->down_blocks_size,
        entry->down_scales_size,
        static_cast<size_t>(entry->down_blocks_offset),
        static_cast<size_t>(entry->down_scales_offset),
    };
    auto operation = Mxfp4Expert_vulkan ::create_from_device_storage(gate_up, entry->execution.gate_up_bias, down, entry->execution.down_bias, entry->execution.activation_limit,
                                                                     static_cast<uint32_t>(context->device()->info.device_index()), entry->data, entry->execution.activation,
                                                                     context->runtime(),
                                                                     context->optimization_flags());
    if (!operation)
        return std::nullopt;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        entry->operation = operation;
    }
    return DeviceOperationLease{
        std::shared_ptr<Mxfp4Expert_vulkan>(entry, operation.get()),
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
    if (!download(*entry, gate_up_source, down_source, restored))
    {
        const std::lock_guard<std::mutex> lock(mutex);
        ++restore_failures;
        const auto existing = entries.find(key);
        if (existing != entries.end() && existing->second == entry)
        {
            resident_size -= entry->size;
            entry->residency_token.reset();
            entries.erase(existing);
        }
        return std::nullopt;
    }

    {
        const std::lock_guard<std::mutex> lock(mutex);
        ++hits;
        bytes_downloaded += entry->size;
        if (mapped_restore)
            ++mapped_restores;
    }
    return restored;
}

void VulkanExpertVictimCache::set_residency_coordinator(std::shared_ptr<ExpertResidencyCoordinator> coordinator)
{
    const std::lock_guard<std::mutex> lock(mutex);
    residency_coordinator = std::move(coordinator);
    for (auto& [key, entry] : entries)
    {
        entry->residency_token.reset();
        if (residency_coordinator) entry->residency_token = residency_coordinator->register_device(key);
    }
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
    entry->weight_allocator = weight_allocator;
    ncnn::Mat payload(static_cast<int>(admission.size), sizeof(uint8_t));
    if (payload.empty()) return {};
    std::memset(payload.data, 0, payload.total() * payload.elemsize);
    copy_payload(admission, static_cast<uint8_t*>(payload.data));
    auto* independent_allocator = static_cast<VulkanExpertWeightAllocator*>(weight_allocator.get());
    VulkanIndependentWeightTransfer command(context, upload_staging_allocator);
    if (!command.record(payload, entry->data, independent_allocator) || !command.submit_and_wait()) return {};
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
        if (slot.staging.empty() || submit_compute_and_wait(command, context->device()) != 0)
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

        std::shared_ptr<DeviceEntry> entry;
        try
        {
            entry = upload(admission);
        }
        catch (...)
        {
            entry.reset();
        }
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
                    victim->second->residency_token.reset();
                    entries.erase(victim);
                    ++evictions;
                }
                if (resident_size <= cache_size - entry->size)
                {
                    bool installed = false;
                    try
                    {
                        installed = entries.emplace(admission.key, entry).second;
                    }
                    catch (...)
                    {
                        installed = false;
                    }
                    if (installed)
                    {
                        if (residency_coordinator) entry->residency_token = residency_coordinator->register_device(admission.key);
                        entry->used_at = ++clock;
                        resident_size += entry->size;
                        bytes_uploaded += entry->size;
                        if (entry->data.mapped_ptr()) ++mapped_stores;
                        ++stores;
                    }
                    else
                        ++dropped_admissions;
                }
                else
                {
                    ++dropped_admissions;
                }
            }
            else
                ++dropped_admissions;
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
                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                              uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<VulkanContext> context = VulkanContext::acquire(device_index,
                                                                          vulkan_runtime,
                                                                          optimization_flags);
    if (!context || cache_size == 0)
        return {};
    return std::make_shared<VulkanExpertVictimCache>(context, cache_size);
#else
    (void)cache_size;
    (void)device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
VulkanExpertBackend::VulkanExpertBackend(uint64_t _cache_size,
                                         uint32_t _vulkan_device_index,
                                         std::shared_ptr<VulkanExpertVictimCache> _device_weight_source,
                                         VulkanRuntimePtr _vulkan_runtime,
                                         uint64_t _optimization_flags)
    : cache_size(_cache_size),
      vulkan_device_index(_vulkan_device_index),
      vulkan_runtime(std::move(_vulkan_runtime)),
      optimization_flags(_optimization_flags),
      device_weight_source(std::move(_device_weight_source))
{
    vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                            vulkan_runtime,
                                            optimization_flags);
    if (vulkan_context && cache_size != 0)
    {
        const uint64_t allocator_block_size = std::min<uint64_t>(cache_size, UINT64_C(64) * 1024 * 1024);
        expert_weight_allocator = std::make_shared<VulkanExpertWeightAllocator>(vulkan_context->device(), static_cast<size_t>(allocator_block_size));
        admission_staging_allocator = std::make_unique<VulkanUploadStagingAllocator>(vulkan_context->device(), allocator_block_size);
    }
    try
    {
        worker = std::thread(&VulkanExpertBackend::worker_loop, this);
        execution_worker = std::thread(&VulkanExpertBackend::execution_loop, this);
        demand_worker = std::thread(&VulkanExpertBackend::demand_loop, this);
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
        for (const PendingAdmission& admission : pending)
            pending_size -= admission.size;
        pending.clear();
        pending_keys.clear();
        // Active uploads retain their own reservation until they finish.
        // Remove only queued references from pending_size.
        // (The worker may already have removed a batch from pending.)
    }
    work_available.notify_all();
    execution_available.notify_all();
    demand_available.notify_all();
    if (demand_worker.joinable())
        demand_worker.join();
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
        // Background staging uses a distinct command/allocator domain. Stop
        // speculative admissions at the next queue boundary without waiting
        // for an already submitted independent transfer.
        ++foreground_depth;
    }
    else if (foreground_depth != 0)
    {
        if (--foreground_depth == 0)
            work_available.notify_all();
    }
}

bool VulkanExpertBackend::make_admission(std::string key, std::shared_ptr<const TensorData> gate_up,
                                         const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down,
                                         const TensorData* down_bias, uint32_t residency_group,
                                         float activation_limit, ExpertActivation activation,
                                         PendingAdmission& admission) const
{
    const auto valid_mxfp4_matrix = [](const TensorData& matrix) {
        if (matrix.shape.size() != 2 || matrix.shape[0] == 0 || matrix.shape[1] == 0
            || matrix.shape[1] % 32 != 0
            || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 64)
            || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            return false;
        const uint64_t scales = static_cast<uint64_t>(matrix.shape[0]) * (matrix.shape[1] / 32);
        const uint64_t blocks = scales * 16;
        return blocks <= static_cast<uint64_t>(std::numeric_limits<int>::max() - 3)
               && scales <= static_cast<uint64_t>(std::numeric_limits<int>::max() - 3)
               && matrix.mxfp4_blocks.size() == blocks
               && matrix.mxfp4_scales.size() == scales;
    };
    const auto valid_bias = [](const TensorData* bias, uint32_t columns) {
        if (!bias)
            return true;
        if (bias->shape.size() != 1 || bias->shape[0] != columns)
            return false;
        if (bias->dtype == DType::Float32)
            return bias->float32_values().size() == columns;
        return bias->dtype == DType::BFloat16 && bias->bfloat16_values().size() == columns;
    };
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
                              && down->shape[1] == gate_up->shape[0] / 2
                              && valid_mxfp4_matrix(*gate_up)
                              && valid_mxfp4_matrix(*down);
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
        || !valid_bias(gate_up_bias, gate_up->shape[0])
        || !valid_bias(down_bias, down->shape[0])
        || (activation != ExpertActivation::Silu
            && activation != ExpertActivation::GptOssSwiGlu
            && activation != ExpertActivation::DeepSeekSwiGlu)
        || !std::isfinite(activation_limit) || activation_limit < 0.0f)
    {
        return false;
    }
    uint64_t size = 0;
    const std::array<uint64_t, 4> parts = {expert_matrix_bytes(*gate_up), expert_matrix_bytes(*down), tensor_bytes(gate_up_bias), tensor_bytes(down_bias)};
    for (uint64_t part : parts)
    {
        if (part > cache_size - size)
            return false;
        size += part;
    }
    if (size == 0)
    {
        return false;
    }

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
    return true;
}

void VulkanExpertBackend::admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down,
                                const TensorData* down_bias, uint32_t residency_group, float activation_limit,
                                ExpertActivation activation)
{
    PendingAdmission admission;
    if (!make_admission(std::move(key), std::move(gate_up), gate_up_bias, std::move(down), down_bias,
                        residency_group, activation_limit, activation, admission))
        return;
    const uint64_t size = admission.size;
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

bool VulkanExpertBackend::prepare_demand(std::string key,
                                         std::shared_ptr<const TensorData> gate_up,
                                         const TensorData* gate_up_bias,
                                         std::shared_ptr<const TensorData> down,
                                         const TensorData* down_bias,
                                         uint32_t residency_group,
                                         float activation_limit,
                                         ExpertActivation activation,
                                         std::shared_ptr<const void>& pin)
{
    pin.reset();
    PendingAdmission admission;
    try
    {
        if (!vulkan_context || !expert_weight_allocator
            || !make_admission(std::move(key), std::move(gate_up), gate_up_bias,
                               std::move(down), down_bias, residency_group,
                               activation_limit, activation, admission))
            return false;
    }
    catch (...)
    {
        return false;
    }
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto existing = entries.find(admission.key);
        if (!stopping && existing != entries.end())
            pin = existing->second;
    }
    if (pin)
        return true;

    // Never wait for the background queue: foreground execution may suspend
    // that queue. Serializing admission makes this request's upload and its
    // cache reservation independent of background progress.
    const std::unique_lock<std::mutex> admission_lock(admission_mutex);
    std::vector<std::shared_ptr<Entry>> retired;
    bool reserved = false;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto existing = entries.find(admission.key);
        if (!stopping && existing != entries.end())
        {
            pin = existing->second;
        }
        else if (!stopping)
        {
            bool already_queued = false;
            for (auto queued = pending.begin(); queued != pending.end(); ++queued)
            {
                if (queued->key != admission.key)
                    continue;
                pending_size -= queued->size;
                pending_keys.erase(queued->key);
                pending.erase(queued);
                already_queued = true;
                break;
            }
            // The demand's host references share the same bound as queued
            // uploads. Discard older background work before exceeding it.
            while (!pending.empty() && pending_size > cache_size - admission.size)
            {
                pending_size -= pending.front().size;
                pending_keys.erase(pending.front().key);
                pending.pop_front();
                ++dropped_admissions;
            }
            bool capacity_available = pending_size <= cache_size - admission.size;
            const bool from_frequent = frequent_ghost_index.find(admission.key) != frequent_ghost_index.end();
            while (capacity_available && resident_size > cache_size - admission.size)
                capacity_available = evict_one_locked(from_frequent, admission.residency_group, admission.size);
            if (capacity_available)
            {
                // Free retired Vulkan resources before creating the incoming
                // pair; an upload cannot transiently double a full cache.
                retired.swap(retired_entries);
                pending_size += admission.size;
                pending_keys.insert(admission.key);
                ++active_admissions;
                if (!already_queued)
                    ++admissions;
                reserved = true;
            }
            else if (already_queued)
            {
                ++dropped_admissions;
            }
        }
    }
    if (pin)
        return true;
    if (!reserved)
        return false;
    retired.clear();

    std::shared_ptr<Entry> loaded;
    bool upload_succeeded = true;
    try
    {
        std::optional<VulkanWeightUploadBatch> upload_batch;
        if (admission.gate_up->dtype == DType::MxFp4)
            upload_batch.emplace(vulkan_context, admission_staging_allocator.get(), &admission_upload_scratch);
        loaded = create_entry(admission, upload_batch ? &*upload_batch : nullptr);
        if (upload_batch)
        {
            upload_succeeded = upload_batch->submit();
        }
    }
    catch (...)
    {
        upload_succeeded = false;
    }
    // Batch destruction releases the context lock before the last pipeline
    // owners can be destroyed by an unsuccessful admission.
    if (!upload_succeeded)
        loaded.reset();
    bool success = false;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        pending_size -= admission.size;
        pending_keys.erase(admission.key);
        if (!stopping && install_entry_locked(loaded))
        {
            pin = entries.find(admission.key)->second;
            success = true;
        }
        else
        {
            ++dropped_admissions;
        }
        finish_admission_locked();
    }
    work_available.notify_all();
    return success;
}

size_t VulkanExpertBackend::prepare_demand_batch(std::span<const ExpertDemandRequest> requests,
                                                 std::span<std::shared_ptr<const void>> pins)
{
    for (auto& pin : pins) pin.reset();
    if (requests.empty() || requests.size() != pins.size()) return 0;
    if (!requests.front().gate_up || requests.front().gate_up->dtype != DType::MxFp4)
        return ExpertBackend::prepare_demand_batch(requests, pins);

    static constexpr size_t maximum_requests = 8;
    static constexpr uint64_t maximum_upload_bytes = UINT64_C(64) * 1024 * 1024;
    const size_t limit = std::min(requests.size(), maximum_requests);
    std::vector<PendingAdmission> candidates;
    candidates.reserve(limit);
    for (size_t index = 0; index < limit; ++index)
    {
        const auto& request = requests[index];
        if (!request.gate_up || request.gate_up->dtype != DType::MxFp4) break;
        PendingAdmission admission;
        try
        {
            if (!vulkan_context || !expert_weight_allocator
                || !make_admission(std::string(request.key), request.gate_up, request.gate_up_bias,
                                   request.down, request.down_bias, request.residency_group,
                                   request.activation_limit, request.activation, admission))
            {
                break;
            }
        }
        catch (...)
        {
            break;
        }
        candidates.push_back(std::move(admission));
    }

    // Each matrix's device layout also contains aligned FP32 bias, even when
    // no bias was provided. Bound in-flight staging by that complete layout.
    const uint64_t alignment = vulkan_context
                                   ? std::max<uint64_t>(4, vulkan_context->device()->info.buffer_offset_alignment())
                                   : 4;
    const auto upload_size = [alignment](const PendingAdmission& admission) {
        const auto matrix_size = [alignment](const TensorData& matrix) {
            const auto align = [alignment](uint64_t bytes) {
                const uint64_t remainder = bytes % alignment;
                return remainder == 0 ? bytes : bytes + alignment - remainder;
            };
            return align(matrix.mxfp4_blocks.size()) + align(matrix.mxfp4_scales.size())
                   + align(static_cast<uint64_t>(matrix.shape[0]) * sizeof(float));
        };
        return matrix_size(*admission.gate_up) + matrix_size(*admission.down);
    };
    std::vector<std::shared_ptr<Entry>> loaded(candidates.size());
    const std::unique_lock<std::mutex> admission_lock(admission_mutex);
    std::vector<std::shared_ptr<Entry>> retired;
    std::vector<size_t> new_indices;
    std::vector<size_t> duplicate_of(candidates.size(), candidates.size());
    new_indices.reserve(candidates.size());
    size_t settled_admissions = 0;
    const auto rollback_reservations = [&] {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            for (size_t index = settled_admissions; index < new_indices.size(); ++index)
            {
                const auto& admission = candidates[new_indices[index]];
                pending_size -= admission.size;
                pending_keys.erase(admission.key);
                ++dropped_admissions;
                finish_admission_locked();
            }
        }
        work_available.notify_all();
    };
    struct ReservationGuard
    {
        const decltype(rollback_reservations)& rollback;
        bool armed = true;
        ~ReservationGuard()
        {
            if (armed) rollback();
        }
    } reservation_guard{rollback_reservations};
    uint64_t reserved_bytes = 0;
    uint64_t transfer_bytes = 0;
    size_t prefix = 0;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        // Protect this round's cache hits while choosing victims. On a tiny
        // cache, release later hits only if they block the current prefix.
        if (!stopping)
            for (size_t index = 0; index < candidates.size(); ++index)
            {
                const auto existing = entries.find(candidates[index].key);
                if (existing != entries.end()) pins[index] = existing->second;
            }
        for (; prefix < candidates.size(); ++prefix)
        {
            auto& admission = candidates[prefix];
            if (stopping)
            {
                break;
            }
            // A previous reservation may have released later hit pins to
            // reclaim capacity. Some of those entries can still be resident.
            if (!pins[prefix])
            {
                const auto existing = entries.find(admission.key);
                if (existing != entries.end()) pins[prefix] = existing->second;
            }
            if (pins[prefix])
            {
                continue;
            }
            for (size_t index : new_indices)
                if (candidates[index].key == admission.key)
                {
                    duplicate_of[prefix] = index;
                    break;
                }
            if (duplicate_of[prefix] != candidates.size())
            {
                continue;
            }
            const uint64_t current_transfer_bytes = upload_size(admission);
            // A legal pair larger than the batching cap retains the original
            // single-demand path; it is never joined by a second upload.
            if (!new_indices.empty()
                && (transfer_bytes > maximum_upload_bytes
                    || current_transfer_bytes > maximum_upload_bytes - transfer_bytes))
                break;
            bool already_queued = false;
            for (auto queued = pending.begin(); queued != pending.end(); ++queued)
            {
                if (queued->key != admission.key) continue;
                pending_size -= queued->size;
                pending_keys.erase(queued->key);
                pending.erase(queued);
                already_queued = true;
                break;
            }
            while (!pending.empty() && pending_size > cache_size - admission.size)
            {
                pending_size -= pending.front().size;
                pending_keys.erase(pending.front().key);
                pending.pop_front();
                ++dropped_admissions;
            }
            bool capacity_available = pending_size <= cache_size - admission.size
                                      && reserved_bytes <= cache_size - admission.size;
            const bool from_frequent = frequent_ghost_index.find(admission.key) != frequent_ghost_index.end();
            if (capacity_available)
            {
                const uint64_t available_resident = cache_size - reserved_bytes - admission.size;
                while (resident_size > available_resident)
                {
                    if (evict_one_locked(from_frequent, admission.residency_group, admission.size)) continue;
                    for (size_t index = prefix + 1; index < candidates.size(); ++index) pins[index].reset();
                    if (!evict_one_locked(from_frequent, admission.residency_group, admission.size))
                    {
                        capacity_available = false;
                        break;
                    }
                }
            }
            if (!capacity_available)
            {
                if (already_queued) ++dropped_admissions;
                break;
            }
            // A failure in a later key allocation still completes the
            // earlier reservations through the normal prefix upload path.
            try
            {
                pending_keys.insert(admission.key);
            }
            catch (...)
            {
                if (already_queued) ++dropped_admissions;
                break;
            }
            pending_size += admission.size;
            ++active_admissions;
            if (!already_queued) ++admissions;
            reserved_bytes += admission.size;
            transfer_bytes += current_transfer_bytes;
            new_indices.push_back(prefix);
        }
        for (size_t index = prefix; index < pins.size(); ++index) pins[index].reset();
        retired.swap(retired_entries);
    }
    retired.clear();

    bool upload_succeeded = true;
    if (!new_indices.empty())
    {
        try
        {
            VulkanWeightUploadBatch upload_batch(vulkan_context, admission_staging_allocator.get(), &admission_upload_scratch);
            for (size_t index = 0; index < new_indices.size(); ++index)
            {
                loaded[index] = create_entry(candidates[new_indices[index]], &upload_batch);
                if (!loaded[index])
                {
                    upload_succeeded = false;
                    break;
                }
            }
            if (upload_succeeded)
            {
                upload_succeeded = upload_batch.submit();
            }
        }
        catch (...)
        {
            upload_succeeded = false;
        }
        // Destroy the transfer command and release its context mutex before
        // clearing any last projection/pipeline owners on failure.
        if (!upload_succeeded) loaded.clear();
    }
    size_t completed = prefix;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        for (size_t index = 0; index < new_indices.size(); ++index)
        {
            const size_t request_index = new_indices[index];
            const auto& admission = candidates[request_index];
            bool installed = false;
            try
            {
                installed = upload_succeeded && !stopping && install_entry_locked(loaded[index]);
            }
            catch (...)
            {
                // Installation keeps list/map/size accounting unchanged
                // when its allocation fails. Release this reservation below.
            }
            pending_size -= admission.size;
            pending_keys.erase(admission.key);
            if (installed)
            {
                pins[request_index] = entries.find(admission.key)->second;
            }
            else
            {
                ++dropped_admissions;
                completed = std::min(completed, request_index);
            }
            finish_admission_locked();
            ++settled_admissions;
        }
        for (size_t index = 0; index < completed; ++index)
            if (duplicate_of[index] != candidates.size()) pins[index] = pins[duplicate_of[index]];
        for (size_t index = completed; index < pins.size(); ++index) pins[index].reset();
    }
    reservation_guard.armed = false;
    work_available.notify_all();
    return completed;
}

VulkanExpertBackend::DemandSubmission::DemandSubmission(std::shared_ptr<DemandWork> _work)
    : work(std::move(_work))
{
}

VulkanExpertBackend::DemandSubmission::~DemandSubmission()
{
    if (!work) return;
    std::unique_lock<std::mutex> lock(work->mutex);
    work->completed.wait(lock, [this] { return work->done; });
    work->aborted = true;
    work->pins.clear();
}

size_t VulkanExpertBackend::DemandSubmission::wait(std::span<std::shared_ptr<const void>> pins)
{
    for (auto& pin : pins) pin.reset();
    if (!work || pins.size() != work->requests.size()) return 0;
    std::unique_lock<std::mutex> lock(work->mutex);
    work->completed.wait(lock, [this] { return work->done; });
    if (work->aborted) return 0;
    std::copy(work->pins.begin(), work->pins.end(), pins.begin());
    return work->prefix;
}

void VulkanExpertBackend::DemandSubmission::abort() noexcept
{
    if (!work) return;
    const std::lock_guard<std::mutex> lock(work->mutex);
    work->aborted = true;
    if (work->done) work->pins.clear();
}

std::unique_ptr<ExpertDemandSubmission> VulkanExpertBackend::begin_demand_batch(std::span<const ExpertDemandRequest> requests)
{
    auto work = std::make_shared<DemandWork>();
    work->keys.reserve(requests.size());
    work->biases.reserve(requests.size() * 2);
    work->requests.assign(requests.begin(), requests.end());
    work->pins.resize(requests.size());
    for (size_t index = 0; index < requests.size(); ++index)
    {
        work->keys.emplace_back(requests[index].key);
        work->requests[index].key = work->keys.back();
        for (bool gate : {true, false})
        {
            const TensorData* source = gate ? requests[index].gate_up_bias : requests[index].down_bias;
            if (!source) continue;
            work->biases.push_back(std::make_shared<const TensorData>(*source));
            if (gate)
                work->requests[index].gate_up_bias = work->biases.back().get();
            else
                work->requests[index].down_bias = work->biases.back().get();
        }
    }
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (stopping || demand_pending.size() >= 2)
            work->done = true;
        else
            demand_pending.push_back(work);
    }
    demand_available.notify_one();
    // Construct the draining handle only after enqueue succeeds. If handle
    // allocation fails, the queued work still owns and settles its resources.
    return std::make_unique<DemandSubmission>(std::move(work));
}

void VulkanExpertBackend::demand_loop()
{
    for (;;)
    {
        std::shared_ptr<DemandWork> work;
        bool stopped = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            demand_available.wait(lock, [this] { return stopping || !demand_pending.empty(); });
            if (demand_pending.empty()) return;
            work = std::move(demand_pending.front());
            demand_pending.pop_front();
            ++demand_active;
            stopped = stopping;
        }
        bool cancelled = false;
        {
            const std::lock_guard<std::mutex> lock(work->mutex);
            cancelled = work->aborted;
        }
        size_t prefix = 0;
        if (!stopped && !cancelled)
        {
            try
            {
                prefix = prepare_demand_batch(work->requests, work->pins);
            }
            catch (...)
            {
                for (auto& pin : work->pins) pin.reset();
            }
        }
        {
            const std::lock_guard<std::mutex> lock(work->mutex);
            work->prefix = prefix;
            if (work->aborted) work->pins.clear();
            work->done = true;
        }
        work->completed.notify_all();
        {
            const std::lock_guard<std::mutex> lock(mutex);
            --demand_active;
            if (demand_pending.empty() && demand_active == 0) admission_idle.notify_all();
        }
    }
}

void VulkanExpertBackend::set_residency_coordinator(std::shared_ptr<ExpertResidencyCoordinator> coordinator)
{
    const std::lock_guard<std::mutex> lock(mutex);
    residency_coordinator = std::move(coordinator);
    for (auto& [key, entry] : entries)
    {
        entry->residency_token.reset();
        if (residency_coordinator) entry->residency_token = residency_coordinator->register_device(key);
    }
}

std::shared_ptr<VulkanExpertBackend::Entry> VulkanExpertBackend::create_entry(const PendingAdmission& admission,
                                                                              VulkanWeightUploadBatch* upload_batch)
{
    std::shared_ptr<Entry> entry = std::make_shared<Entry>();
    if (admission.gate_up->dtype == DType::MxFp4)
    {
        entry->operation = Mxfp4Expert_vulkan::create_with_allocator(*admission.gate_up,
                                                                     admission.gate_up_bias.get(),
                                                                     *admission.down,
                                                                     admission.down_bias.get(),
                                                                     admission.activation_limit,
                                                                     vulkan_device_index,
                                                                     expert_weight_allocator.get(),
                                                                     admission.activation,
                                                                     vulkan_runtime,
                                                                     optimization_flags,
                                                                     upload_batch);
    }
    else if (admission.gate_up->dtype == DType::BFloat16)
    {
        entry->bfloat16_operation = Bfloat16Expert_vulkan::create_with_allocator(*admission.gate_up,
                                                                                 admission.gate_up_bias.get(),
                                                                                 *admission.down,
                                                                                 admission.down_bias.get(),
                                                                                 admission.activation_limit,
                                                                                 vulkan_device_index,
                                                                                 expert_weight_allocator.get(),
                                                                                 admission.activation,
                                                                                 vulkan_runtime,
                                                                                 optimization_flags);
    }
    else
    {
        entry->qnk_operation = QnkExpert_vulkan::create_with_allocator(*admission.gate_up,
                                                                       admission.gate_up_bias.get(),
                                                                       *admission.down,
                                                                       admission.down_bias.get(),
                                                                       admission.activation_limit,
                                                                       vulkan_device_index,
                                                                       expert_weight_allocator.get(),
                                                                       admission.activation,
                                                                       vulkan_runtime,
                                                                       optimization_flags);
    }
    if (!entry->operation && !entry->bfloat16_operation && !entry->qnk_operation)
        return {};
    entry->key = admission.key;
    entry->weight_allocator = expert_weight_allocator;
    entry->size = admission.size;
    entry->residency_group = admission.residency_group;
    return entry;
}

bool VulkanExpertBackend::install_entry_locked(std::shared_ptr<Entry>& entry)
{
    if (!entry || entries.find(entry->key) != entries.end())
        return false;
    // Complete potentially allocating metadata changes before publishing
    // resident size or upload accounting for the incoming entry.
    if (entry->residency_group >= residency_group_sizes.size())
        residency_group_sizes.resize(static_cast<size_t>(entry->residency_group) + 1, 0);
    bool promote = false;
    bool from_frequent = false;
    (void)consume_ghost_locked(entry->key, entry->size, promote, from_frequent);
    while (resident_size > cache_size - entry->size)
    {
        if (!evict_one_locked(from_frequent, entry->residency_group, entry->size))
        {
            entry.reset();
            return false;
        }
    }
    auto& destination = promote ? frequent : recent;
    const auto position = destination.insert(destination.end(), entry->key);
    try
    {
        if (!entries.emplace(entry->key, entry).second)
        {
            destination.erase(position);
            return false;
        }
    }
    catch (...)
    {
        destination.erase(position);
        throw;
    }
    if (residency_coordinator)
        entry->residency_token = residency_coordinator->register_device(entry->key);
    entry->position = position;
    entry->list = promote ? ArcList::Frequent : ArcList::Recent;
    if (promote)
        frequent_size += entry->size;
    else
        recent_size += entry->size;
    resident_size += entry->size;
    residency_group_sizes[entry->residency_group] += entry->size;
    bytes_uploaded += entry->size;
    entry.reset();
    ++stores;
    return true;
}

void VulkanExpertBackend::WorkItem::clear_references() noexcept
{
    selected.clear();
    private_aggregation.clear();
    for (auto& output : private_outputs)
        output.clear();
    requests.clear();
    client_requests.clear();
    for (auto& output : private_device_outputs)
        output.reset();
    uint64_t size = private_aggregation.allocated_bytes()
                    + client_requests.capacity() * sizeof(ExpertBackendRequest)
                    + requests.capacity() * sizeof(ExpertBackendRequest)
                    + private_outputs.capacity() * sizeof(ActivationBuffer)
                    + private_device_outputs.capacity() * sizeof(std::shared_ptr<DeviceTensor_vulkan>)
                    + private_route_completed.capacity() * sizeof(uint8_t)
                    + selected.capacity() * sizeof(Selection)
                    + planned.capacity() * sizeof(ExpertBackendExecutionResult)
                    + final.capacity() * sizeof(ExpertBackendExecutionResult);
    for (const auto& output : private_outputs)
        size += output.allocated_bytes();
    // Large prefill storage is reclaimed once its submission is released.
    static constexpr uint64_t maximum_retained_host_size = 64 * 1024 * 1024;
    if (size > maximum_retained_host_size)
    {
        decltype(private_outputs)().swap(private_outputs);
        private_aggregation = {};
        decltype(client_requests)().swap(client_requests);
        decltype(requests)().swap(requests);
        decltype(private_device_outputs)().swap(private_device_outputs);
        decltype(private_route_completed)().swap(private_route_completed);
        decltype(selected)().swap(selected);
        decltype(planned)().swap(planned);
        decltype(final)().swap(final);
    }
}

std::shared_ptr<VulkanExpertBackend::WorkItem> VulkanExpertBackend::acquire_work_item()
{
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& work : execution_scratch)
        if (work.use_count() == 1)
        {
            return work;
        }
    auto work = std::make_shared<WorkItem>();
    static constexpr size_t maximum_scratch_count = 4;
    if (execution_scratch.size() < maximum_scratch_count)
        execution_scratch.push_back(work);
    return work;
}

std::unique_ptr<ExpertSubmission> VulkanExpertBackend::submit_batch(std::span<const ExpertBackendRequest> requests)
{
    auto work = acquire_work_item();
    std::unique_ptr<ExpertSubmission> submission;
    bool queued = false;
    try
    {
        // Failed preparation can leave a sole-owned item without a ticket.
        work->clear_references();
        work->client_requests.assign(requests.begin(), requests.end());
        work->requests = work->client_requests;
        if (work->private_outputs.size() < requests.size())
            work->private_outputs.resize(requests.size());
        if (work->private_device_outputs.size() < requests.size())
            work->private_device_outputs.resize(requests.size());
        work->private_aggregation.clear();
        work->done = false;
        work->private_route_completed.assign(requests.size(), 0);
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            ExpertBackendRequest& private_request = work->requests[request_index];
            private_request.output = &work->private_outputs[request_index];
            if (private_request.device_output)
                private_request.device_output = &work->private_device_outputs[request_index];
            if (private_request.route_aggregation.output)
            {
                const ActivationBuffer* client_aggregation = work->client_requests[request_index].route_aggregation.output;
                if (!client_aggregation)
                    continue;
                if (work->private_aggregation.rows() == 0)
                {
                    work->private_aggregation.reset(client_aggregation->rows(),
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
                if (!request.output || request.rows() == 0 || request.columns() == 0
                    || (!request.has_host_input() && !request.device_input))
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
            // Prepare the public handle before borrowed inputs become visible
            // to the worker. A failed queue allocation completes the private item
            // before its handle unwinds, so destruction never waits on unqueued work.
            submission = std::make_unique<Submission>(work);
            if (work->selected.empty())
            {
                work->final = work->planned;
                work->done = true;
            }
            else
            {
                execution_pending.push_back(work);
                queued = true;
                execution_available.notify_one();
            }
        }
        return submission;
    }
    catch (...)
    {
        if (!queued)
        {
            work->clear_references();
            {
                const std::lock_guard<std::mutex> lock(work->mutex);
                work->done = true;
            }
            work->completed.notify_all();
        }
        throw;
    }
}

void VulkanExpertBackend::wait_for_background_work()
{
    std::unique_lock<std::mutex> lock(mutex);
    // Existing block execution explicitly drains queued admissions while
    // holding a foreground scope. Permit only that explicit wait to resume
    // the worker; ordinary demand uploads remain independent of the queue.
    ++admission_drain_waiters;
    work_available.notify_all();
    admission_idle.wait(lock, [this] { return pending.empty() && active_admissions == 0 && demand_pending.empty() && demand_active == 0; });
    --admission_drain_waiters;
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

VulkanExpertBackend::Submission::Submission(std::shared_ptr<WorkItem> _work)
    : work(std::move(_work))
{
}

VulkanExpertBackend::Submission::~Submission()
{
    if (work && !waited)
        wait_completion();
    if (work && !committed && !aborted)
        abort();
    if (work)
        work->clear_references();
}

std::span<const ExpertBackendExecutionResult> VulkanExpertBackend::Submission::reservations() const noexcept
{
    return work->planned;
}

void VulkanExpertBackend::Submission::wait_completion()
{
    std::unique_lock<std::mutex> lock(work->mutex);
    work->completed.wait(lock, [this] { return work->done; });
    waited = true;
}

std::vector<ExpertBackendExecutionResult> VulkanExpertBackend::Submission::wait()
{
    std::vector<ExpertBackendExecutionResult> results;
    wait(results);
    return results;
}

void VulkanExpertBackend::Submission::wait(std::vector<ExpertBackendExecutionResult>& results)
{
    wait_completion();
    results.assign(work->final.begin(), work->final.end());
}

bool VulkanExpertBackend::Submission::commit()
{
    if (!waited)
        wait_completion();
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
        if (executed && work->private_device_outputs[index]
            && (!client.device_output || work->private_device_outputs[index]->empty()
                || work->private_device_outputs[index]->rows() != client.rows()))
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
        if (client.device_output)
            client.device_output->swap(work->private_device_outputs[index]);
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

bool VulkanExpertBackend::route_aggregation_enabled(std::span<const ExpertBackendRequest> requests,
                                                    std::span<const Selection> selected,
                                                    uint32_t output_columns,
                                                    ActivationBuffer*& output,
                                                    uint32_t& token_count,
                                                    uint64_t optimization_flags)
{
    output = nullptr;
    token_count = 0;
    size_t route_count = 0;
    if (selected.empty())
        return false;

    const bool enabled = has_flag(optimization_flags,
                                  OptimizationVulkanRouteAggregation);
    if (!enabled)
        return false;

    bool require_all_requests = false;
    for (const ExpertBackendRequest& request : requests)
    {
        require_all_requests = require_all_requests
                               || request.route_aggregation.require_all_requests;
    }
    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size())
            return false;
        const ExpertBackendRequest& request = requests[selection.request_index];
        if (request.device_output)
            return false;
        const ExpertBackendRequest::RouteAggregation& aggregation = request.route_aggregation;
        if (request.rows() == 0 || !aggregation.output || !aggregation.completed || aggregation.token_count == 0
            || aggregation.routes.size() != request.rows()
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

    if (require_all_requests && selected.size() != requests.size())
        return false;

    // Keep single-token decode in canonical route order.  Device-side
    // reduction remains useful for multi-token waves, where it amortizes the
    // metadata upload and output transfer without changing the next-token
    // decision at every layer.
    if (token_count == 1 || route_count <= token_count)
        return false;
    return true;
}

bool VulkanExpertBackend::build_route_aggregation_metadata(std::span<const ExpertBackendRequest> requests,
                                                           std::span<const Selection> selected,
                                                           std::vector<uint32_t>& offsets,
                                                           std::vector<uint32_t>& rows,
                                                           std::vector<float>& weights)
{
    if (selected.empty())
        return false;
    const ExpertBackendRequest::RouteAggregation& first = requests[selected.front().request_index].route_aggregation;
    const uint32_t token_count = first.token_count;
    offsets.assign(static_cast<size_t>(token_count) + 1, 0);
    size_t total_rows = 0;
    for (const Selection& selection : selected)
    {
        const ExpertBackendRequest& request = requests[selection.request_index];
        const auto& aggregation = request.route_aggregation;
        if (aggregation.token_count != token_count || aggregation.routes.size() != request.rows())
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
    for (const Selection& selection : selected)
    {
        const ExpertBackendRequest& request = requests[selection.request_index];
        for (size_t batch_index = 0; batch_index < request.route_aggregation.routes.size(); ++batch_index)
        {
            const ExpertRoute& route = request.route_aggregation.routes[batch_index];
            const uint32_t destination = cursors[route.token_index]++;
            if (destination >= rows.size() || row_offset + batch_index > std::numeric_limits<uint32_t>::max())
                return false;
            rows[destination] = static_cast<uint32_t>(row_offset + batch_index);
            weights[destination] = route.weight;
        }
        row_offset += request.rows();
    }
    return true;
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
    ExpertKeySize ghost{entry.key, entry.size};
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
    // Keep duplicate recovery cheap only among the oldest eligible entries.
    // Scanning the full ARC list would sacrifice newer GPU weights whenever
    // their host copies remain warm, increasing demand uploads under churn.
    static constexpr size_t joint_eviction_candidate_limit = 4;
    const bool use_joint_eviction = residency_coordinator
                                    && has_flag(optimization_flags, OptimizationVulkanJointCacheEviction);
    size_t candidate_count = 0;
    std::shared_ptr<Entry> oldest;
    for (const std::string& key : list)
    {
        const auto existing = entries.find(key);
        if (existing == entries.end() || existing->second.use_count() != 1
            || (residency_group != std::numeric_limits<uint32_t>::max() && existing->second->residency_group != residency_group))
            continue;
        if (!oldest) oldest = existing->second;
        if (!use_joint_eviction || residency_coordinator->host_resident(key))
            return existing->second;
        if (++candidate_count == joint_eviction_candidate_limit)
            break;
    }
    return oldest;
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
    // Keep ARC's reuse preference ahead of the layer fair-share tie-break.
    // A cold admission must not evict a hot entry while a recent entry in
    // another layer remains available.
    if (!victim && preferred_group != std::numeric_limits<uint32_t>::max())
    {
        victim = prefer_recent ? find_victim_locked(recent, std::numeric_limits<uint32_t>::max())
                               : find_victim_locked(frequent, std::numeric_limits<uint32_t>::max());
    }
    if (!victim)
    {
        victim = prefer_recent ? find_victim_locked(frequent, preferred_group) : find_victim_locked(recent, preferred_group);
    }
    if (!victim && preferred_group != std::numeric_limits<uint32_t>::max())
    {
        victim = prefer_recent ? find_victim_locked(frequent, std::numeric_limits<uint32_t>::max())
                               : find_victim_locked(recent, std::numeric_limits<uint32_t>::max());
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
    victim->residency_token.reset();
    entries.erase(victim->key);
    retired_entries.push_back(std::move(victim));
    ++evictions;
    return true;
}

void VulkanExpertBackend::execute_work_item(const std::shared_ptr<WorkItem>& work)
{
    bool executed = work->selected.empty();
    if (!work->selected.empty())
    {
        const std::shared_ptr<Entry>& entry = work->selected.front().entry;
        if (entry && entry->operation)
            executed = forward_batch(work->requests, work->selected);
        else if (entry && entry->bfloat16_operation)
        {
            executed = forward_bfloat16_batch(work->requests, work->selected);
        }
        else if (entry && entry->qnk_operation)
        {
            executed = forward_qnk_batch(work->requests, work->selected);
        }
    }
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
    work->final.assign(work->planned.begin(), work->planned.end());
    auto& final = work->final;
    {
        const std::lock_guard<std::mutex> lock(mutex);
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
        work->selected.clear();
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
        {
            const std::unique_lock<std::mutex> admission_lock(admission_mutex);
            std::vector<std::shared_ptr<Entry>> retired;
            {
                const std::lock_guard<std::mutex> lock(mutex);
                retired.swap(retired_entries);
            }
            retired.clear();
        }
        {
            std::unique_lock<std::mutex> lock(mutex);
            work_available.wait(lock, [this] {
                return stopping || ((foreground_depth == 0 || admission_drain_waiters != 0) && !pending.empty());
            });
            if (stopping)
                return;
        }
        // Acquire this outside the scheduler mutex. A foreground demand can
        // remove a queued pair while the worker waits, so recheck the queue.
        const std::unique_lock<std::mutex> admission_lock(admission_mutex);
        std::vector<PendingAdmission> batch;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            if (stopping)
                return;
            if ((foreground_depth != 0 && admission_drain_waiters == 0) || pending.empty())
                continue;
            batch.push_back(std::move(pending.front()));
            pending.pop_front();
            ++active_admissions;
            if (has_flag(optimization_flags, OptimizationVulkanExpertBatchAdmission)
                && batch.front().gate_up && batch.front().gate_up->dtype == DType::MxFp4)
            {
                while (batch.size() < maximum_upload_batch && !pending.empty()
                       && pending.front().gate_up && pending.front().gate_up->dtype == DType::MxFp4)
                {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();
                    ++active_admissions;
                }
            }
        }
        std::vector<std::shared_ptr<Entry>> loaded(batch.size());
        bool upload_succeeded = true;
        try
        {
            std::optional<VulkanWeightUploadBatch> upload_batch;
            if (batch.front().gate_up->dtype == DType::MxFp4)
                upload_batch.emplace(vulkan_context, admission_staging_allocator.get(), &admission_upload_scratch);
            for (size_t index = 0; index < batch.size(); ++index)
                loaded[index] = create_entry(batch[index], upload_batch ? &*upload_batch : nullptr);
            if (upload_batch)
            {
                upload_succeeded = upload_batch->submit();
            }
        }
        catch (...)
        {
            upload_succeeded = false;
        }
        if (!upload_succeeded)
            std::fill(loaded.begin(), loaded.end(), std::shared_ptr<Entry>());
        const std::lock_guard<std::mutex> lock(mutex);
        for (size_t index = 0; index < batch.size(); ++index)
        {
            pending_size -= batch[index].size;
            pending_keys.erase(batch[index].key);
            if (!stopping && !install_entry_locked(loaded[index]))
                ++dropped_admissions;
            finish_admission_locked();
        }
    }
}

bool VulkanExpertBackend::forward_bfloat16_batch(std::span<const ExpertBackendRequest> requests,
                                                 std::span<const Selection> selected)
{
    if (selected.empty())
        return true;
    if (!selected.front().entry
        || !selected.front().entry->bfloat16_operation)
    {
        return false;
    }

    std::vector<const Bfloat16Expert_vulkan*> experts;
    std::vector<const ActivationBuffer*> inputs;
    std::vector<ActivationBuffer*> outputs;
    experts.reserve(selected.size());
    inputs.reserve(selected.size());
    outputs.reserve(selected.size());
    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size()
            || !selection.entry
            || !selection.entry->bfloat16_operation)
        {
            return false;
        }
        const ExpertBackendRequest& request = requests[selection.request_index];
        if (!request.has_host_input() || !request.output)
            return false;
        experts.push_back(selection.entry->bfloat16_operation.get());
        inputs.push_back(request.input);
        outputs.push_back(request.output);
    }

    const uint32_t output_columns = selected.front().entry->bfloat16_operation->output_columns();
    ActivationBuffer* aggregated_output = nullptr;
    uint32_t aggregated_token_count = 0;
    const bool use_route_aggregation = route_aggregation_enabled(requests,
                                                                 selected,
                                                                 output_columns,
                                                                 aggregated_output,
                                                                 aggregated_token_count,
                                                                 optimization_flags);

    if (!Bfloat16Expert_vulkan::forward_batch(experts,
                                              inputs,
                                              outputs))
    {
        return false;
    }

    if (use_route_aggregation)
    {
        if (!aggregated_output)
            return false;
        aggregated_output->reset(aggregated_token_count, output_columns, true);
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            if (request.route_aggregation.routes.size()
                != request.rows())
            {
                return false;
            }
            for (size_t row = 0; row < request.rows(); ++row)
            {
                const ExpertRoute& route = request.route_aggregation.routes[row];
                if (route.token_index >= aggregated_output->rows())
                    return false;
                float* destination = aggregated_output->row(route.token_index);
                const float* source = request.output->row(row);
                for (uint32_t column = 0; column < output_columns; ++column)
                    destination[column] += route.weight * source[column];
            }
            if (request.route_aggregation.completed)
                *request.route_aggregation.completed = 1;
        }
    }
    return true;
}

bool VulkanExpertBackend::forward_qnk_batch(std::span<const ExpertBackendRequest> requests, std::span<const Selection> selected)
{
    if (selected.empty())
        return true;
    if (!selected.front().entry || !selected.front().entry->qnk_operation)
        return false;

    const uint32_t output_columns = selected.front().entry->qnk_operation->output_columns();
    if (output_columns == 0)
        return false;

    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size()
            || !selection.entry
            || !selection.entry->qnk_operation)
        {
            return false;
        }
        const ExpertBackendRequest& request = requests[selection.request_index];
        if (!request.has_host_input() || !request.output)
        {
            return false;
        }
    }

    ActivationBuffer* aggregated_output = nullptr;
    uint32_t aggregated_token_count = 0;
    const bool use_route_aggregation = route_aggregation_enabled(requests,
                                                                 selected,
                                                                 output_columns,
                                                                 aggregated_output,
                                                                 aggregated_token_count,
                                                                 optimization_flags);
    if (selected.size() == 1)
    {
        const Selection& selection = selected.front();
        const ExpertBackendRequest& request = requests[selection.request_index];
        if (!selection.entry->qnk_operation->forward(*request.input, *request.output))
            return false;
    }
    else
    {
        std::vector<const QnkExpert_vulkan*> experts;
        std::vector<const ActivationBuffer*> inputs;
        std::vector<ActivationBuffer*> outputs;
        experts.reserve(selected.size());
        inputs.reserve(selected.size());
        outputs.reserve(selected.size());
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            experts.push_back(selection.entry->qnk_operation.get());
            inputs.push_back(request.input);
            outputs.push_back(request.output);
        }
        if (!QnkExpert_vulkan::forward_batch(experts, inputs, outputs))
            return false;
    }

    if (use_route_aggregation)
    {
        if (!aggregated_output)
            return false;
        aggregated_output->reset(aggregated_token_count, output_columns, true);
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            if (request.route_aggregation.routes.size() != request.rows())
                return false;
            for (size_t row = 0; row < request.rows(); ++row)
            {
                const ExpertRoute& route = request.route_aggregation.routes[row];
                if (route.token_index >= aggregated_output->rows())
                    return false;
                float* destination = aggregated_output->row(route.token_index);
                const float* source = request.output->row(row);
                for (uint32_t column = 0; column < output_columns; ++column)
                    destination[column] += route.weight * source[column];
            }
            if (request.route_aggregation.completed)
                *request.route_aggregation.completed = 1;
        }
    }
    return true;
}

#endif // NCNN_MOE_WITH_VULKAN

std::shared_ptr<ExpertBackend> create_vulkan_expert_backend(uint64_t cache_size, uint32_t device_index,
                                                            std::shared_ptr<ExpertVictimCache> device_weight_source,
                                                            const VulkanRuntimePtr& vulkan_runtime,
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
    return std::make_shared<VulkanExpertBackend>(cache_size,
                                                 device_index,
                                                 std::move(source),
                                                 vulkan_runtime,
                                                 optimization_flags);
#else
    (void)cache_size;
    (void)device_index;
    (void)device_weight_source;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

} // namespace moe
} // namespace ncnn
