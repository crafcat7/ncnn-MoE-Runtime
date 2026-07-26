#ifndef NCNN_MOE_MOE_IR_H
#define NCNN_MOE_MOE_IR_H

#include "ncnn/moe/model_descriptor.h"
#include "ncnn/moe/result.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

using MoeIRValueId = uint32_t;
using MoeIRNodeId = uint32_t;
inline constexpr MoeIRValueId invalid_moe_ir_value_id = std::numeric_limits<MoeIRValueId>::max();
inline constexpr MoeIRNodeId invalid_moe_ir_node_id = std::numeric_limits<MoeIRNodeId>::max();
inline constexpr uint32_t invalid_moe_ir_layer_id = std::numeric_limits<uint32_t>::max();

enum class MoeIROperator
{
    TokenEmbedding,
    Attention,
    Router,
    ExpertGroup,
    SharedExpertGroup,
    DenseFfn,
    Combine,
    KvCache,
    FinalNorm,
    LmHead
};

enum class QuantizationScheme
{
    None,
    PerTensor,
    PerChannel,
    BlockWise
};

struct QuantConfig
{
    DType storage_dtype = DType::Float32;
    DType compute_dtype = DType::Float32;
    QuantizationScheme scheme = QuantizationScheme::None;
    uint32_t block_size = 0;
    uint32_t group_size = 0;
};

#define NCNN_MOE_IR_VALUE_INPUT_BIT      0
#define NCNN_MOE_IR_VALUE_OUTPUT_BIT     1
#define NCNN_MOE_IR_VALUE_PERSISTENT_BIT 2
#define NCNN_MOE_IR_VALUE_MUTABLE_BIT    3
#define NCNN_MOE_IR_VALUE_DYNAMIC_BIT    4

enum MoeIRValueFlag : uint32_t
{
    MoeIRValueGraphInput = UINT32_C(1) << NCNN_MOE_IR_VALUE_INPUT_BIT,
    MoeIRValueGraphOutput = UINT32_C(1) << NCNN_MOE_IR_VALUE_OUTPUT_BIT,
    MoeIRValuePersistent = UINT32_C(1) << NCNN_MOE_IR_VALUE_PERSISTENT_BIT,
    MoeIRValueMutableState = UINT32_C(1) << NCNN_MOE_IR_VALUE_MUTABLE_BIT,
    MoeIRValueDynamicShape = UINT32_C(1) << NCNN_MOE_IR_VALUE_DYNAMIC_BIT
};

struct MoeIRValue
{
    MoeIRValueId id = invalid_moe_ir_value_id;
    std::string name;
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    TensorLocation preferred_location = TensorLocation::Automatic;
    uint32_t flags = 0;
};

#define NCNN_MOE_IR_NODE_STATEFUL_BIT 0

enum MoeIRNodeFlag : uint32_t
{
    MoeIRNodeStateful = UINT32_C(1) << NCNN_MOE_IR_NODE_STATEFUL_BIT
};

struct MoeIRNode
{
    MoeIRNodeId id = invalid_moe_ir_node_id;
    MoeIROperator operation = MoeIROperator::TokenEmbedding;
    std::string name;
    uint32_t layer_id = invalid_moe_ir_layer_id;
    std::vector<MoeIRValueId> inputs;
    std::vector<MoeIRValueId> outputs;
    uint32_t intermediate_size = 0;
    AttentionDescriptor attention;
    MoeDescriptor experts;
    QuantConfig quantization;
    uint32_t flags = 0;
};

struct MoeGraph
{
    std::vector<MoeIRValue> values;
    std::vector<MoeIRNode> nodes;
    std::vector<MoeIRValueId> outputs;

    [[nodiscard]] Result<void> validate() const;
};

struct MoeIR : MoeModelDescriptor
{
    MoeGraph graph;
    QuantConfig activation_quantization;
    QuantConfig kv_cache_quantization;
    QuantConfig expert_quantization;

    [[nodiscard]] Result<void> validate() const;
};

class MoeGraphBuilder
{
public:
    [[nodiscard]] Result<MoeGraph> build(const MoeModelDescriptor& descriptor) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MOE_IR_H
