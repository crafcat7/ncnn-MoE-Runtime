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
    std::atomic<uint32_t> prefill_rows{0};
    std::atomic<uint32_t> context_tokens{0};
    std::atomic<uint32_t> sequence_count{1};
    ModelGraphMemoryPlan plan;
    std::vector<std::shared_ptr<void>> contexts;
};

[[nodiscard]] Result<void> prepare_model_graph_resources(CompiledModel& model, uint32_t prefill_rows);

// Static capability/capacity rejection retains the existing GPU streaming
// plan. Once full banks are selected, allocation/upload failures are errors.
[[nodiscard]] Result<void> prepare_model_expert_banks(CompiledModel& model, uint32_t prefill_rows,
                                                      uint64_t expert_budget, uint64_t victim_budget = 0);

[[nodiscard]] Result<void> ensure_model_graph_resources(const CompiledModel& model,
                                                        uint64_t rows, uint64_t context_tokens, uint32_t sequence_count = 1);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPHRESOURCES_H
