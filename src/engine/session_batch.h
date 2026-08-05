#ifndef NCNN_MOE_SESSION_BATCH_H
#define NCNN_MOE_SESSION_BATCH_H

#include "engine/cpu_executor.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/session.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct StagedDecodeBatchMetrics
{
    uint64_t logical_expert_batches = 0;
    uint64_t physical_expert_batches = 0;
    uint64_t coalesced_expert_routes = 0;
    uint64_t max_expert_batch_size = 0;
    uint64_t vulkan_attention_batch_submissions = 0;
    uint64_t vulkan_attention_batch_rows = 0;
    uint64_t vulkan_attention_batch_avoided_submissions = 0;
    uint64_t expert_time_microseconds = 0;
    uint64_t expert_cache_wait_time_microseconds = 0;
};

struct SessionDecodePhaseSnapshot
{
    uint64_t expert_time_microseconds = 0;
    uint64_t expert_cache_wait_time_microseconds = 0;
};

class SessionBatchAccess
{
public:
    [[nodiscard]] static bool compatible(std::span<Session* const> sessions) noexcept;
    [[nodiscard]] static Result<std::vector<PrefillResult>> prefill(
        std::span<Session* const> sessions,
        std::span<const std::vector<int32_t>> input_ids,
        StagedDecodeBatchMetrics& metrics);
    [[nodiscard]] static Result<std::vector<DecodeResult>> decode(std::span<Session* const> sessions, std::span<const int32_t> input_ids, StagedDecodeBatchMetrics& metrics);
    [[nodiscard]] static SessionDecodePhaseSnapshot phase_snapshot(Session& session);
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSION_BATCH_H
