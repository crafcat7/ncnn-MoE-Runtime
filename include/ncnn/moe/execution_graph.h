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
inline constexpr ExecutionNodeId invalid_execution_node_id
    = std::numeric_limits<ExecutionNodeId>::max();
inline constexpr uint32_t invalid_execution_layer_id
    = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t invalid_execution_expert_id
    = std::numeric_limits<uint32_t>::max();

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
    Combine,
    FinalNorm,
    LmHead
};

enum ExecutionNodeFlag : uint32_t
{
    ExecutionNodeConditional = 1u << 0
};

struct ExecutionNode
{
    ExecutionNodeId id = invalid_execution_node_id;
    ExecutionNodeType type = ExecutionNodeType::TokenEmbedding;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    uint32_t layer_id = invalid_execution_layer_id;
    uint32_t expert_id = invalid_execution_expert_id;
    uint32_t flags = 0;
    std::string name;
    std::vector<ExecutionNodeId> dependencies;
};

struct ExecutionGraph
{
    std::vector<ExecutionNode> nodes;

    [[nodiscard]] Result<void> validate() const;
    [[nodiscard]] const ExecutionNode* find(ExecutionNodeId id) const noexcept;
};

struct ExecutionWave
{
    // Every node in a wave is dependency-ready at the same time. Backends may
    // execute these nodes concurrently; routed Expert nodes are conditional.
    std::vector<ExecutionNodeId> nodes;
};

struct ExecutionSchedule
{
    std::vector<ExecutionWave> waves;
};

class MoeScheduler
{
public:
    [[nodiscard]] Result<ExecutionSchedule> schedule(
        const ExecutionGraph& graph) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTION_GRAPH_H
