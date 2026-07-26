#ifndef NCNN_MOE_EXPERT_H
#define NCNN_MOE_EXPERT_H

#include "ncnn/moe/types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

class ModelCompiler;

struct ExpertKey
{
    uint32_t layer_id = 0;
    uint32_t expert_id = 0;
};

enum class ExpertCacheState
{
    Unloaded,
    Loading,
    Resident,
    Failed
};

enum class ExpertKernel
{
    PortableCpu,
    Mxfp4Scalar,
    Mxfp4ArmNeon,
    Mxfp4ArmSve2,
    Mxfp4X86Avx2,
    Mxfp4X86Avx512
};

struct ExpertStatistics
{
    ExpertKey key;
    ExpertCacheState cache_state = ExpertCacheState::Unloaded;
    TensorLocation memory_location = TensorLocation::Automatic;
    ExpertKernel kernel = ExpertKernel::PortableCpu;
    uint64_t weight_bytes = 0;
    uint64_t dispatch_count = 0;
    uint64_t token_count = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t last_used = 0;
};

class Expert
{
private:
    friend class ExpertStore;
    friend class ModelCompiler;

    Expert(ExpertKey key, uint64_t weight_bytes, ExpertCacheState cache_state, TensorLocation memory_location, ExpertKernel kernel);

    [[nodiscard]] ExpertStatistics statistics() const;

    ExpertKey key_;
    uint64_t weight_bytes_ = 0;
    ExpertKernel kernel_ = ExpertKernel::Mxfp4Scalar;
    std::atomic<ExpertCacheState> cache_state_{ExpertCacheState::Unloaded};
    std::atomic<TensorLocation> memory_location_{TensorLocation::Automatic};
    std::atomic<uint64_t> dispatch_count_{0};
    std::atomic<uint64_t> token_count_{0};
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
    std::atomic<uint64_t> last_used_{0};

public:
    void record_dispatch(uint64_t token_count, uint64_t clock);
    void record_cache_hit();
    void record_cache_miss();
    void set_residency(ExpertCacheState state, TensorLocation location);
};

struct ExpertStoreStatistics
{
    uint64_t expert_count = 0;
    uint64_t unloaded_experts = 0;
    uint64_t loading_experts = 0;
    uint64_t resident_experts = 0;
    uint64_t failed_experts = 0;
    uint64_t registered_weight_bytes = 0;
    uint64_t dispatch_count = 0;
    uint64_t token_count = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
};

struct ExpertHotsetEstimate
{
    uint64_t capacity_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t active_expert_count = 0;
    uint64_t resident_expert_count = 0;
    uint64_t requested_batch_weight_bytes = 0;
    uint64_t covered_batch_weight_bytes = 0;
    uint64_t requested_route_weight_bytes = 0;
    uint64_t covered_route_weight_bytes = 0;
};

class ExpertStore
{
private:
    friend class ModelCompiler;

    void add(std::shared_ptr<Expert> expert);

    std::vector<std::shared_ptr<Expert>> experts_;

public:
    [[nodiscard]] ExpertStoreStatistics statistics() const;
    [[nodiscard]] ExpertHotsetEstimate estimate_hotset(uint64_t capacity_bytes) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_H
