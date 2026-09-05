#ifndef NCNN_MOE_NCNN_EXPERTBACKEND_VULKAN_H
#define NCNN_MOE_NCNN_EXPERTBACKEND_VULKAN_H

#include "engine/expertbackend.h"
#include "vulkancontext.h"
#include "storage/expertcache_victim.h"

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ncnn {

class Pipeline;
class VkBlobAllocator;

namespace moe {

class NcnnVulkanContext;
class NcnnVulkanMxfp4ExpertOperator;
class NcnnVulkanQnkExpertOperator;
class NcnnVulkanBfloat16ExpertOperator;

class VulkanExpertVictimCache final : public ExpertVictimCache
{
public:
    VulkanExpertVictimCache(std::shared_ptr<NcnnVulkanContext> _context, uint64_t _cache_size);

    ~VulkanExpertVictimCache() override;

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down,
               ExpertVictimExecutionMetadata execution) override;

    struct DeviceOperationLease
    {
        std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
        std::shared_ptr<const void> pin;
    };

    std::optional<DeviceOperationLease> find_device_operation(std::string_view key);

    void touch_device_operations(std::span<const std::string_view> keys);

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source) override;

    void wait_for_background_work() override;

    ExpertVictimCacheStatistics statistics() const override;

    uint64_t capacity() const noexcept override;

private:
    struct DeviceEntry
    {
        ncnn::VkMat data;
        std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
        uint64_t size = 0;
        uint64_t gate_blocks_size = 0;
        uint64_t gate_scales_size = 0;
        uint64_t down_blocks_size = 0;
        uint64_t down_scales_size = 0;
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
    };

    struct PendingAdmission
    {
        std::string key;
        std::shared_ptr<const TensorData> gate_up;
        std::shared_ptr<const TensorData> down;
        uint64_t size = 0;
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

    static bool append_segment(uint64_t size, uint64_t alignment, uint64_t& cursor, uint64_t& offset);

    static void copy_payload(const PendingAdmission& admission, uint8_t* destination);

    static MxFp4ByteBuffer copy_bytes(const uint8_t* source, uint64_t offset, uint64_t byte_count);

    static void materialize(const DeviceEntry& entry, const TensorData& gate_up_source, const TensorData& down_source, const uint8_t* source,
                            ExpertVictimPair& restored);

    std::shared_ptr<DeviceEntry> upload(const PendingAdmission& admission);

    bool download(const DeviceEntry& entry, const TensorData& gate_up_source, const TensorData& down_source, ExpertVictimPair& restored);

    void worker_loop();

    std::shared_ptr<NcnnVulkanContext> context;
    // Bounds both resident data and queued host-weight references.
    uint64_t cache_size = 0;
    ncnn::VkAllocator* upload_staging_allocator = nullptr;
    mutable std::mutex mutex;
    std::condition_variable work_available;
    std::condition_variable idle;
    std::deque<PendingAdmission> pending;
    std::unordered_set<std::string, ExpertKeyHash, std::equal_to<>> pending_keys;
    std::unordered_map<std::string, std::shared_ptr<DeviceEntry>, ExpertKeyHash, std::equal_to<>> entries;
    uint64_t pending_size = 0;
    uint32_t active_admissions = 0;
    uint64_t resident_size = 0;
    uint64_t clock = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t stores = 0;
    uint64_t evictions = 0;
    uint64_t dropped_admissions = 0;
    uint64_t restore_failures = 0;
    uint64_t bytes_uploaded = 0;
    uint64_t bytes_downloaded = 0;
    uint64_t restore_time_microseconds = 0;
    uint64_t mapped_stores = 0;
    uint64_t mapped_restores = 0;
    bool stopping = false;
    ncnn::VkMat upload_staging;
    std::array<DownloadSlot, 2> download_slots;
    std::atomic<size_t> next_download_slot{0};
    std::thread worker;
};

class VulkanExpertBackend final : public ExpertBackend
{
public:
    VulkanExpertBackend(
        uint64_t _cache_size,
        uint32_t _vulkan_device_index,
        std::shared_ptr<VulkanExpertVictimCache> _device_weight_source,
        NcnnVulkanContextInstancePtr _context_instance,
        uint64_t _optimization_flags);

    ~VulkanExpertBackend() override;

    void set_foreground_active(bool active) noexcept override;

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down,
               const TensorData* down_bias, uint32_t residency_group, float activation_limit,
               ExpertActivation activation) override;

    ExpertBackendExecutionResult try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output) override;

    std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests) override;

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override;

    void observe_cpu(uint32_t token_count, uint64_t weight_size, uint64_t elapsed_microseconds) override;

    void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) override;

    void wait_for_background_work() override;

    ExpertBackendStatistics statistics() const override;

    std::vector<ExpertBackendDeviceStatistics> device_statistics() const override;

    uint64_t capacity() const noexcept override;

private:
    enum class ArcList
    {
        Recent,
        Frequent
    };

    struct Entry
    {
        std::string key;
        // Submissions may outlive the backend; release operators before their allocator.
        std::shared_ptr<ncnn::VkBlobAllocator> weight_allocator;
        std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> operation;
        std::shared_ptr<NcnnVulkanQnkExpertOperator> qnk_operation;
        std::shared_ptr<NcnnVulkanBfloat16ExpertOperator> bfloat16_operation;
        std::shared_ptr<const void> device_source_pin;
        uint64_t size = 0;
        uint32_t residency_group = 0;
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

    class Submission final : public ExpertSubmission
    {
    public:
        explicit Submission(std::shared_ptr<WorkItem> _work);

        ~Submission() override;

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override;

        std::vector<ExpertBackendExecutionResult> wait() override;

        bool commit() override;

        void abort() noexcept override;

    private:
        std::shared_ptr<WorkItem> work;
        bool waited = false;
        bool committed = false;
        bool aborted = false;
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
        uint64_t size = 0;
    };

    using GhostList = std::list<ExpertKeySize>;
    using GhostIndex = std::unordered_map<std::string, GhostList::iterator, ExpertKeyHash, std::equal_to<>>;

    static uint64_t mxfp4_bytes(const TensorData& tensor);

    static uint64_t qnk_bytes(const TensorData& tensor);

    static uint64_t expert_matrix_bytes(const TensorData& tensor);

    static uint64_t tensor_bytes(const TensorData* tensor);

    void touch_locked(Entry& entry, bool repeated);

    static void erase_ghost_entry(GhostIndex& index, GhostList& list, uint64_t& size, const std::string& key);

    void erase_ghost_locked(const std::string& key);

    void add_ghost_locked(const Entry& entry);

    static void trim_ghost_front(GhostIndex& index, GhostList& list, uint64_t& size);

    void trim_ghosts_locked();

    uint64_t arc_delta(uint64_t required, uint64_t numerator, uint64_t denominator) const;

    bool consume_ghost_locked(const std::string& key, uint64_t required, bool& promote, bool& from_frequent);

    std::shared_ptr<Entry> find_victim_locked(const std::list<std::string>& list, uint32_t residency_group);

    bool evict_one_locked(bool incoming_from_frequent, uint32_t incoming_group, uint64_t required);

    enum class IndexedBatchResult
    {
        NotSupported,
        Executed,
        Failed
    };

    IndexedBatchResult forward_indexed_batch(
        std::span<const ExpertBackendRequest> requests,
        std::span<const Selection> selected);

    bool forward_batch(std::span<const ExpertBackendRequest> requests, std::span<const Selection> selected);

    bool forward_bfloat16_batch(
        std::span<const ExpertBackendRequest> requests,
        std::span<const Selection> selected);

    bool forward_qnk_batch(std::span<const ExpertBackendRequest> requests, std::span<const Selection> selected);

    void execute_work_item(const std::shared_ptr<WorkItem>& work);

    void execution_loop();

    void finish_admission_locked();

    void worker_loop();
    void stop_workers();

    // Bounds both resident data and queued host-weight references.
    const uint64_t cache_size;
    const uint32_t vulkan_device_index;
    NcnnVulkanContextInstancePtr context_instance;
    const uint64_t optimization_flags;
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::shared_ptr<ncnn::VkBlobAllocator> expert_weight_allocator;
    std::shared_ptr<ncnn::Pipeline> indexed_pipeline;
    std::shared_ptr<ncnn::Pipeline> route_aggregation_pipeline;
    std::shared_ptr<VulkanExpertVictimCache> device_weight_source;
    mutable std::mutex mutex;
    std::condition_variable work_available;
    std::condition_variable execution_available;
    std::condition_variable admission_idle;
    bool stopping = false;
    uint32_t foreground_depth = 0;
    std::deque<PendingAdmission> pending;
    std::deque<std::shared_ptr<WorkItem>> execution_pending;
    std::unordered_set<std::string, ExpertKeyHash, std::equal_to<>> pending_keys;
    uint64_t pending_size = 0;
    uint32_t active_admissions = 0;
    std::unordered_map<std::string, std::shared_ptr<Entry>, ExpertKeyHash, std::equal_to<>> entries;
    std::vector<std::shared_ptr<Entry>> retired_entries;
    std::list<std::string> recent;
    std::list<std::string> frequent;
    uint64_t recent_size = 0;
    uint64_t frequent_size = 0;
    uint64_t recent_target_size = 0;
    GhostList recent_ghost;
    GhostList frequent_ghost;
    GhostIndex recent_ghost_index;
    GhostIndex frequent_ghost_index;
    uint64_t recent_ghost_size = 0;
    uint64_t frequent_ghost_size = 0;
    std::vector<uint64_t> residency_group_sizes;
    uint64_t resident_size = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t stores = 0;
    uint64_t evictions = 0;
    uint64_t dropped_admissions = 0;
    uint64_t executions = 0;
    uint64_t execution_failures = 0;
    uint64_t bytes_uploaded = 0;
    uint64_t execution_time_microseconds = 0;
    uint64_t device_source_hits = 0;
    uint64_t device_source_misses = 0;
    uint64_t device_source_executions = 0;
    uint64_t device_source_execution_failures = 0;
    uint64_t route_aggregation_batches = 0;
    uint64_t route_aggregation_routes = 0;
    uint64_t route_aggregation_bytes_saved = 0;
    std::thread worker;
    std::thread execution_worker;
};

} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_WITH_VULKAN

namespace ncnn {
namespace moe {

[[nodiscard]] std::shared_ptr<ExpertBackend> create_vulkan_expert_backend(
    uint64_t cache_size, uint32_t device_index,
    std::shared_ptr<ExpertVictimCache> device_weight_source,
    const NcnnVulkanContextInstancePtr& context_instance, uint64_t optimization_flags);

[[nodiscard]] std::shared_ptr<ExpertVictimCache> create_vulkan_victim_cache(
    uint64_t cache_size, uint32_t device_index,
    const NcnnVulkanContextInstancePtr& context_instance, uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_EXPERTBACKEND_VULKAN_H
