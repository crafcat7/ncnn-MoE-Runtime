#ifndef NCNN_MOE_GRAPHRESOURCES_H
#define NCNN_MOE_GRAPHRESOURCES_H

#include "graph/memoryplan.h"
#include "ncnn/moe/result.h"

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

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPHRESOURCES_H
