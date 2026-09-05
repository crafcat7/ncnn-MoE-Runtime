#ifndef NCNN_MOE_GRAPH_H
#define NCNN_MOE_GRAPH_H

#include "ncnn/moe/result.h"
#include "layerplan.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

struct CompiledModel;

using ExecutionNodeId = uint32_t;
using ExecutionTensorId = uint32_t;
inline constexpr ExecutionNodeId invalid_execution_node_id = std::numeric_limits<ExecutionNodeId>::max();
inline constexpr ExecutionTensorId invalid_execution_tensor_id = std::numeric_limits<ExecutionTensorId>::max();
inline constexpr uint32_t invalid_execution_layer_id = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t invalid_execution_expert_id = std::numeric_limits<uint32_t>::max();

enum class ExecutionBackend
{
    Cpu,
    Vulkan
};

enum class ExecutionNodeType
{
    TokenEmbedding,
    Attention,
    Router,
    ExpertDispatch,
    Expert,
    ExpertGroup,
    SharedExpertGroup,
    Combine,
    FinalNorm,
    LmHead
};

#define NCNN_MOE_BACKEND_CPU_BIT    0
#define NCNN_MOE_BACKEND_VULKAN_BIT 1

enum ExecutionBackendMask : uint32_t
{
    ExecutionBackendCpu = UINT32_C(1) << NCNN_MOE_BACKEND_CPU_BIT,
    ExecutionBackendVulkan = UINT32_C(1) << NCNN_MOE_BACKEND_VULKAN_BIT
};

#define NCNN_MOE_NODE_CONDITIONAL_BIT  0
#define NCNN_MOE_NODE_CPU_PREFETCH_BIT 1

enum ExecutionNodeFlag : uint32_t
{
    ExecutionNodeConditional = UINT32_C(1) << NCNN_MOE_NODE_CONDITIONAL_BIT,
    ExecutionNodeCpuPrefetch = UINT32_C(1) << NCNN_MOE_NODE_CPU_PREFETCH_BIT
};

struct ExecutionTensor
{
    ExecutionTensorId id = invalid_execution_tensor_id;
    std::string name;
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    ExecutionNodeId producer = invalid_execution_node_id;
    std::vector<ExecutionNodeId> consumers;
};

struct ExecutionNode
{
    ExecutionNodeId id = invalid_execution_node_id;
    ExecutionNodeType type = ExecutionNodeType::TokenEmbedding;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    uint32_t backend_mask = ExecutionBackendCpu;
    // Index into ExecutionGraph::layer_plans.  The graph owns both execution
    // order and the immutable layer payload consumed by the executor.
    uint32_t layer_plan_index = invalid_execution_layer_id;
    uint32_t expert_id = invalid_execution_expert_id;
    // WeightStore handles consumed by non-layer nodes.  Layer nodes resolve
    // their complete operator payload through layer_plan_index instead.
    std::vector<TensorHandle> weight_inputs;
    uint32_t flags = 0;
    std::string name;
    std::vector<ExecutionNodeId> dependencies;
    std::vector<ExecutionTensorId> inputs;
    std::vector<ExecutionTensorId> outputs;
};

struct ExecutionGraph
{
    std::vector<ExecutionTensor> tensors;
    std::vector<ExecutionNode> nodes;
    std::vector<CompiledLayerPlan> layer_plans;

    [[nodiscard]] Result<void> validate() const;
};

// A contiguous backend run in the immutable execution reservation.  The
// scheduler owns the order; executors must not reconstruct it from the graph
// or from layer metadata at execution time.
struct ExecutionBackendRun
{
    ExecutionBackend backend = ExecutionBackend::Cpu;
    uint32_t first_node = 0;
    uint32_t node_count = 0;
};

struct ExecutionSchedule
{
    std::vector<ExecutionNodeId> node_order;
    std::vector<ExecutionBackendRun> backend_runs;

    [[nodiscard]] Result<void> validate(const ExecutionGraph& graph) const;
};

struct GraphOption
{
    uint32_t available_backends = ExecutionBackendCpu;
    bool prefer_vulkan_dense = true;
};

// Selects backends in place; discard the local graph on failure.
[[nodiscard]] Result<ExecutionSchedule> schedule_graph(
    ExecutionGraph& graph,
    const GraphOption& options);

// Build and schedule the target and speculative graph regions.
[[nodiscard]] Result<void> build_graph(CompiledModel& compiled, bool use_vulkan_experts);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPH_H
