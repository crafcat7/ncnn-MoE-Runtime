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

using LayerId = uint32_t;
using ExecutionTensorId = uint32_t;
inline constexpr LayerId invalid_layer_id = std::numeric_limits<LayerId>::max();
inline constexpr ExecutionTensorId invalid_execution_tensor_id = std::numeric_limits<ExecutionTensorId>::max();
inline constexpr uint32_t invalid_execution_layer_id = std::numeric_limits<uint32_t>::max();

enum class ExecutionBackend
{
    Cpu,
    Vulkan
};

enum class LayerKind
{
    TokenEmbedding,
    Attention,
    Router,
    ExpertDispatch,
    ExpertGroup,
    SharedExpertGroup,
    Combine,
    DenseFfn,
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

#define NCNN_MOE_LAYER_CPU_PREFETCH_BIT    0
#define NCNN_MOE_LAYER_REQUEST_EXPERTS_BIT 1

enum LayerFlag : uint32_t
{
    LayerCpuPrefetch = UINT32_C(1) << NCNN_MOE_LAYER_CPU_PREFETCH_BIT,
    // Submit exact reads during dispatch, before independent shared CPU work.
    LayerRequestExperts = UINT32_C(1) << NCNN_MOE_LAYER_REQUEST_EXPERTS_BIT
};

// One scheduled execution step. Planning-only fields (dependencies, tensor
// edges and names) stay on Layer until P3 moves them into the Blob graph.
struct Layer
{
    LayerId id = invalid_layer_id;
    LayerKind kind = LayerKind::TokenEmbedding;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    uint32_t backend_mask = ExecutionBackendCpu;
    uint32_t flags = 0;
    uint32_t plan_index = invalid_execution_layer_id;
    std::vector<TensorHandle> weights;
    // Planning-only fields, retained on the layer until P3 moves the tensor
    // edges fully into the Blob graph.
    std::string name;
    std::vector<LayerId> dependencies;
    std::vector<ExecutionTensorId> inputs;
    std::vector<ExecutionTensorId> outputs;
};

struct ExecutionTensor
{
    ExecutionTensorId id = invalid_execution_tensor_id;
    std::string name;
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    LayerId producer = invalid_layer_id;
    std::vector<LayerId> consumers;
};

struct ExecutionGraph
{
    std::vector<ExecutionTensor> tensors;
    std::vector<Layer> layers;
    std::vector<CompiledLayerPlan> layer_plans;
};

struct ExecutionSchedule
{
    std::vector<LayerId> node_order;

    [[nodiscard]] Result<void> validate(const ExecutionGraph& graph) const;
};

// Selects backends in place and fills caller-owned storage.  Callers provide
// fresh, discardable graph and schedule objects when constructing a model;
// discard the owning model on failure.
[[nodiscard]] Result<void> schedule_graph(ExecutionGraph& graph,
                                          ExecutionSchedule& schedule,
                                          uint32_t available_backends);

// Build and schedule the target and speculative graph regions.
[[nodiscard]] Result<void> build_graph(CompiledModel& compiled, bool use_vulkan_experts);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPH_H
