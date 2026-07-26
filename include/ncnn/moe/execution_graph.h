#ifndef NCNN_MOE_EXECUTION_GRAPH_H
#define NCNN_MOE_EXECUTION_GRAPH_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

using ExecutionNodeId = uint32_t;
using ExecutionTensorId = uint32_t;
using ExecutionEventId = uint32_t;
inline constexpr ExecutionNodeId invalid_execution_node_id = std::numeric_limits<ExecutionNodeId>::max();
inline constexpr ExecutionTensorId invalid_execution_tensor_id = std::numeric_limits<ExecutionTensorId>::max();
inline constexpr ExecutionEventId invalid_execution_event_id = std::numeric_limits<ExecutionEventId>::max();
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

#define NCNN_MOE_NODE_CONDITIONAL_BIT 0
#define NCNN_MOE_NODE_ASYNC_BIT       1

enum ExecutionNodeFlag : uint32_t
{
    ExecutionNodeConditional = UINT32_C(1) << NCNN_MOE_NODE_CONDITIONAL_BIT,
    ExecutionNodeAsync = UINT32_C(1) << NCNN_MOE_NODE_ASYNC_BIT
};

#define NCNN_MOE_TENSOR_PERSISTENT_BIT 0
#define NCNN_MOE_TENSOR_DYNAMIC_BIT    1
#define NCNN_MOE_TENSOR_TRANSFER_BIT   2

enum ExecutionTensorFlag : uint32_t
{
    ExecutionTensorPersistent = UINT32_C(1) << NCNN_MOE_TENSOR_PERSISTENT_BIT,
    ExecutionTensorDynamic = UINT32_C(1) << NCNN_MOE_TENSOR_DYNAMIC_BIT,
    ExecutionTensorTransferBoundary = UINT32_C(1) << NCNN_MOE_TENSOR_TRANSFER_BIT
};

struct ExecutionTensor
{
    ExecutionTensorId id = invalid_execution_tensor_id;
    std::string name;
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    TensorLocation location = TensorLocation::Cpu;
    ExecutionNodeId producer = invalid_execution_node_id;
    std::vector<ExecutionNodeId> consumers;
    uint64_t estimated_bytes = 0;
    uint32_t flags = 0;
};

struct ExecutionEvent
{
    ExecutionEventId id = invalid_execution_event_id;
    ExecutionNodeId producer = invalid_execution_node_id;
    std::vector<ExecutionNodeId> consumers;
};

struct ExecutionNode
{
    ExecutionNodeId id = invalid_execution_node_id;
    ExecutionNodeType type = ExecutionNodeType::TokenEmbedding;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    uint32_t backend_mask = ExecutionBackendCpu;
    uint32_t layer_id = invalid_execution_layer_id;
    uint32_t expert_id = invalid_execution_expert_id;
    uint32_t flags = 0;
    std::string name;
    std::vector<ExecutionNodeId> dependencies;
    std::vector<ExecutionTensorId> inputs;
    std::vector<ExecutionTensorId> outputs;
    std::vector<ExecutionEventId> wait_events;
    ExecutionEventId signal_event = invalid_execution_event_id;
};

struct ExecutionGraph
{
    std::vector<ExecutionTensor> tensors;
    std::vector<ExecutionNode> nodes;
    std::vector<ExecutionEvent> events;

    [[nodiscard]] Result<void> validate() const;
    [[nodiscard]] const ExecutionNode* find(ExecutionNodeId id) const noexcept;
    [[nodiscard]] const ExecutionTensor* find_tensor(ExecutionTensorId id) const noexcept;
};

struct ExecutionWave
{
    std::vector<ExecutionNodeId> nodes;
    std::vector<ExecutionNodeId> cpu_nodes;
    std::vector<ExecutionNodeId> vulkan_nodes;
};

struct ExecutionSchedule
{
    std::vector<ExecutionWave> waves;
    std::vector<ExecutionEvent> events;
    uint32_t cpu_parallelism = 1;
    uint32_t vulkan_queue_count = 0;
};

class MoeScheduler
{
public:
    [[nodiscard]] Result<ExecutionSchedule> schedule(const ExecutionGraph& graph) const;
};

#define NCNN_MOE_SCHEDULING_ASYNC_EVENT_BIT  0
#define NCNN_MOE_SCHEDULING_VULKAN_DENSE_BIT 1

enum RuntimeSchedulingOptionFlag : uint32_t
{
    RuntimeSchedulingEnableAsyncEvents = UINT32_C(1) << NCNN_MOE_SCHEDULING_ASYNC_EVENT_BIT,
    RuntimeSchedulingPreferVulkanDense = UINT32_C(1) << NCNN_MOE_SCHEDULING_VULKAN_DENSE_BIT
};

struct RuntimeSchedulingOptions
{
    uint32_t available_backends = ExecutionBackendCpu;
    uint32_t cpu_parallelism = 1;
    uint32_t vulkan_queue_count = 0;
    uint32_t flags = RuntimeSchedulingEnableAsyncEvents | RuntimeSchedulingPreferVulkanDense;
};

struct ScheduledExecutionGraph
{
    ExecutionGraph graph;
    ExecutionSchedule schedule;
};

class RuntimeScheduler
{
public:
    [[nodiscard]] Result<ScheduledExecutionGraph> compile(ExecutionGraph graph, const RuntimeSchedulingOptions& options) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTION_GRAPH_H
