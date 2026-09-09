#ifndef NCNN_MOE_GATEDDELTANET_H
#define NCNN_MOE_GATEDDELTANET_H

#include "backends/ncnn/gateddeltanet_vulkan.h"
#include "kernels/activationbuffer.h"
#include "graph/layerplan.h"
#include "graph/graph.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"
#include "storage/weightstore.h"

#include <span>
#include <vector>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
struct GatedDeltaScratch
{
    ActivationBuffer normalized;
    ActivationBuffer fused_input;
    ActivationBuffer qkv;
    ActivationBuffer z;
    ActivationBuffer beta;
    ActivationBuffer alpha;
    ActivationBuffer recurrent_output;
    ActivationBuffer projected;
    ActivationBuffer output;
    std::vector<float> recurrent_memory;
    std::vector<float> recurrent_delta;
};

struct LayerCache;

struct GatedDeltaBatchEntry
{
    const ActivationBuffer* hidden = nullptr;
    GatedDeltaScratch* scratch = nullptr;
    LayerCache* cache = nullptr;
    ActivationBuffer* output = nullptr;
};

[[nodiscard]] Result<void> forward_gated_delta(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    LayerCache& cache,
    GatedDeltaScratch& scratch,
    const ActivationBuffer& hidden,
    ActivationBuffer& output,
    uint64_t optimization_flags);

bool forward_gated_delta_batch(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<GatedDeltaBatchEntry> entries,
    std::vector<GatedDeltaBatchEntry_vulkan>& device_entries,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GATEDDELTANET_H
