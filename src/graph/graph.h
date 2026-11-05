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
using BlobId = uint32_t;
inline constexpr LayerId invalid_layer_id = std::numeric_limits<LayerId>::max();
inline constexpr BlobId invalid_blob_id = std::numeric_limits<BlobId>::max();
inline constexpr uint32_t invalid_plan_index = std::numeric_limits<uint32_t>::max();

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

// One scheduled execution step. All graph edges live in Blob; the
// layer only carries the runtime placement and compiled-plan binding.
struct Layer
{
    LayerId id = invalid_layer_id;
    LayerKind kind = LayerKind::TokenEmbedding;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    uint32_t backend_mask = ExecutionBackendCpu;
    uint32_t flags = 0;
    uint32_t plan_index = invalid_plan_index;
    std::vector<TensorHandle> weights;
};

enum class BlobLifetime : uint8_t
{
    Activation,
    // KV/session state is owned outside the reusable activation arena.
    Session,
    // Ordering-only edge; never materialized or copied.
    Control,
};

struct Blob
{
    BlobId id = invalid_blob_id;
    std::string name;
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    BlobLifetime lifetime = BlobLifetime::Activation;
    LayerId producer = invalid_layer_id;
    std::vector<LayerId> consumers;
};

struct Net
{
    std::vector<Blob> blobs;
    std::vector<Layer> layers;
    std::vector<CompiledLayerPlan> layer_plans;
    std::vector<LayerId> order;

    [[nodiscard]] Result<void> validate_order() const;
};

// Selects backends in place and fills net.order. Callers provide a fresh,
// discardable net when constructing a model; discard the owning model on
// failure.
[[nodiscard]] Result<void> schedule_graph(Net& net, uint32_t available_backends);

// Build and schedule the target and speculative graph regions.
[[nodiscard]] Result<void> build_graph(CompiledModel& compiled, bool use_vulkan_experts);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPH_H
