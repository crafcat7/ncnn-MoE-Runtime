#ifndef NCNN_MOE_SCHEDULER_H
#define NCNN_MOE_SCHEDULER_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/session.h"

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

struct DecodeBatchRequest
{
    SessionPtr session;
    int32_t input_id = -1;
};

struct PrefillBatchRequest
{
    SessionPtr session;
    std::vector<int32_t> input_ids;
};

struct SchedulerOptions
{
    // Zero uses an automatic thread count.
    uint32_t num_threads = 0;
    // Compatible multi-Session Decode batches use one staged execution.
    bool use_staged_decode = true;
};

struct SchedulerStatistics
{
    uint64_t prefill_batches = 0;
    uint64_t decode_batches = 0;
    uint64_t staged_prefill_batches = 0;
    uint64_t staged_decode_batches = 0;
    uint32_t num_threads = 0;
};

class BatchSchedulerPrivate;

class BatchScheduler
{
public:
    ~BatchScheduler();

    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    [[nodiscard]] std::future<std::vector<Result<PrefillResult>>> submit_prefill(std::vector<PrefillBatchRequest> requests);
    [[nodiscard]] std::future<std::vector<Result<DecodeResult>>> submit_decode(std::vector<DecodeBatchRequest> requests);
    [[nodiscard]] SchedulerStatistics statistics() const noexcept;

private:
    explicit BatchScheduler(const SchedulerOptions& opt);

    std::unique_ptr<BatchSchedulerPrivate> d;

    friend class Runtime;
};

using BatchSchedulerPtr = std::shared_ptr<BatchScheduler>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SCHEDULER_H
