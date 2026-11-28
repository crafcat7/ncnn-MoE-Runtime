#ifndef NCNN_MOE_GRAPHRESOURCES_H
#define NCNN_MOE_GRAPHRESOURCES_H

#include "graph/memoryplan.h"
#include "ncnn/moe/result.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace ncnn {
namespace moe {

struct CompiledModel;

// Shape growth retains the selected backends and existing device allocations.
// The shared state belongs to the model, including all of its Sessions.
class GraphResources
{
public:
    std::mutex mutex;
    ModelGraphMemoryPlan plan;
    std::atomic<uint32_t> resident_submission_rows{0};
    std::vector<std::shared_ptr<void>> contexts;
};

[[nodiscard]] Result<void> prepare_graph(CompiledModel& model, uint32_t prefill_rows);

// Static capability/capacity rejection retains the existing GPU streaming
// plan. Once full banks are selected, allocation/upload failures are errors.
[[nodiscard]] Result<void> prepare_model_expert_banks(CompiledModel& model, uint32_t prefill_rows,
                                                      uint64_t expert_budget);

// Returns the admitted host output slot size.
[[nodiscard]] Result<uint64_t> ensure_graph(const CompiledModel& model,
                                            uint64_t rows, uint64_t context_tokens, uint32_t sequence_count = 1);

// Steps down prefill chunk rows if the requested row count cannot fit in the
// remaining device memory budget. Returns the admitted chunk row count (>= 1).
[[nodiscard]] Result<size_t> admit_prefill_chunk(const CompiledModel& model,
                                                 size_t requested_rows,
                                                 uint64_t required_context);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPHRESOURCES_H
