#include "backends/ncnn/vulkan.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/expert/backend.h"
#include "backends/ncnn/expert/bank.h"
#include "backends/ncnn/expert/ffn_region.h"
#include "backends/ncnn/expert/shared_ffn.h"
#include "backends/ncnn/layer/attention_vulkan.h"
#include "backends/ncnn/layer/attentiondecode_vulkan.h"
#include "backends/ncnn/layer/gateddeltanet_vulkan.h"
#include "backends/ncnn/layer/gatedresidual_vulkan.h"
#include "backends/ncnn/layer/hyperconnection_vulkan.h"
#include "backends/ncnn/layer/latentattention_vulkan.h"
#include "backends/ncnn/layer/layernorm_vulkan.h"
#include "backends/ncnn/layer/moecombine_vulkan.h"
#include "backends/ncnn/layer/qsaattention_vulkan.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/layer/router_vulkan.h"
#include "backends/ncnn/linear/linear.h"
#include "backends/ncnn/net/layer_head.h"
#include "backends/ncnn/net/latent_layer.h"
#include "backends/ncnn/net/memory.h"
#include "backends/ncnn/net/model.h"
#include "backends/ncnn/net/residual_layer.h"
#include "backends/ncnn/net/segment.h"
#include "engine/expert.h"
#include "graph/compiledmodel.h"
#include "storage/weightstore.h"

#include <algorithm>
#include <cstdint>
#include <memory>

namespace ncnn {
namespace moe {

// --- Public vulkan.h facade --------------------------------------------------

VulkanRuntimePtr create_vulkan_runtime()
{
    return std::make_shared<VulkanRuntime>();
}

std::shared_ptr<void> retain_vulkan_graph_context(const VulkanRuntimePtr&, uint32_t, uint64_t) noexcept
{
    return {};
}

Result<VulkanGraphArenaSize> reserve_vulkan_graph_compute(const VulkanRuntimePtr&, uint32_t,
                                                          uint64_t, uint64_t)
{
    return Error{ErrorCode::UnsupportedModel, "Vulkan support is disabled in this build"};
}

VulkanGraphArenaSize query_vulkan_graph_compute(const VulkanRuntimePtr&, uint32_t, uint64_t) noexcept
{
    return {};
}

uint32_t get_gpu_count() noexcept
{
    return 0;
}

uint32_t get_default_gpu_index() noexcept
{
    return 0;
}

std::vector<GpuInfo> get_gpu_infos()
{
    return {};
}

VulkanStatistics get_vulkan_statistics(const VulkanRuntimePtr&) noexcept
{
    return {};
}

// --- Core --------------------------------------------------------------------

class DeviceTensor_vulkan::Impl
{
};

DeviceTensor_vulkan::DeviceTensor_vulkan() = default;
DeviceTensor_vulkan::~DeviceTensor_vulkan() = default;
DeviceTensor_vulkan::DeviceTensor_vulkan(DeviceTensor_vulkan&&) noexcept = default;
DeviceTensor_vulkan& DeviceTensor_vulkan::operator=(DeviceTensor_vulkan&&) noexcept = default;

bool DeviceTensor_vulkan::empty() const noexcept
{
    return true;
}

size_t DeviceTensor_vulkan::rows() const noexcept
{
    return 0;
}

uint32_t DeviceTensor_vulkan::columns() const noexcept
{
    return 0;
}

// --- Linear ------------------------------------------------------------------

class Linear::Implementation
{
};

Linear::Linear()
    : d(new Implementation)
{
}

Linear::~Linear() = default;

std::shared_ptr<Linear> Linear::create(const TensorData&, const TensorData*, uint32_t,
                                       const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<Linear> Linear::create_fused(const std::vector<const TensorData*>&,
                                             const std::vector<const TensorData*>&,
                                             uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

bool Linear::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

int Linear::forward(const ncnn::VkMat&, ncnn::VkMat&, ncnn::VkCompute&, const ncnn::Option&) const
{
    return -1;
}

uint32_t Linear::input_columns() const noexcept
{
    return 0;
}

uint32_t Linear::output_columns() const noexcept
{
    return 0;
}

class CommandGraph_vulkan::Implementation
{
};

CommandGraph_vulkan::CommandGraph_vulkan(std::unique_ptr<Implementation> implementation)
    : d(std::move(implementation))
{
}

CommandGraph_vulkan::~CommandGraph_vulkan() = default;
CommandGraph_vulkan::CommandGraph_vulkan(CommandGraph_vulkan&&) noexcept = default;
CommandGraph_vulkan& CommandGraph_vulkan::operator=(CommandGraph_vulkan&&) noexcept = default;

std::unique_ptr<CommandGraph_vulkan> CommandGraph_vulkan::create(const Linear&)
{
    return {};
}

bool CommandGraph_vulkan::upload(const ActivationBuffer&, DeviceTensor_vulkan&)
{
    return false;
}

bool CommandGraph_vulkan::linear(const Linear&, const DeviceTensor_vulkan&, DeviceTensor_vulkan&)
{
    return false;
}

bool CommandGraph_vulkan::download(const DeviceTensor_vulkan&, ActivationBuffer&)
{
    return false;
}

bool CommandGraph_vulkan::submit()
{
    return false;
}

bool CommandGraph_vulkan::wait()
{
    return false;
}

class Bfloat16Linear_vulkan::Implementation
{
};

Bfloat16Linear_vulkan::Bfloat16Linear_vulkan()
    : d(new Implementation)
{
}

Bfloat16Linear_vulkan::~Bfloat16Linear_vulkan() = default;

uint64_t Bfloat16Linear_vulkan::storage_bytes(const TensorData&, uint64_t) noexcept
{
    return 0;
}

std::shared_ptr<Bfloat16Linear_vulkan> Bfloat16Linear_vulkan::create(const TensorData&, const TensorData*,
                                                                     uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<Bfloat16Linear_vulkan> Bfloat16Linear_vulkan::create_fused(const std::vector<const TensorData*>&,
                                                                           const std::vector<const TensorData*>&,
                                                                           uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<Bfloat16Linear_vulkan> Bfloat16Linear_vulkan::create_with_allocator(const TensorData&, const TensorData*,
                                                                                    uint32_t, ncnn::VkAllocator*,
                                                                                    const VulkanRuntimePtr&, uint64_t,
                                                                                    VkWeightBatch*)
{
    return {};
}

std::vector<uint64_t> Bfloat16Linear_vulkan::swiglu_scratch_elements(uint32_t, bool, uint64_t)
{
    return {};
}

bool Bfloat16Linear_vulkan::prepare_rms_norm(const TensorData&, float, float)
{
    return false;
}

bool Bfloat16Linear_vulkan::has_rms_norm_chain() const noexcept
{
    return false;
}

bool Bfloat16Linear_vulkan::prepare_input_rms_norm(const TensorData&, float, float)
{
    return false;
}

bool Bfloat16Linear_vulkan::prepare_output_rms_norm(const TensorData&, float)
{
    return false;
}

bool Bfloat16Linear_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

int Bfloat16Linear_vulkan::forward(const ncnn::VkMat&, ncnn::VkMat&, ncnn::VkCompute&, const ncnn::Option&) const
{
    return -1;
}

bool Bfloat16Linear_vulkan::forward_parallel(const ActivationBuffer&, const Bfloat16Linear_vulkan&,
                                             ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

bool Bfloat16Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

bool Bfloat16Linear_vulkan::forward_swiglu_chain(const ActivationBuffer&, const Bfloat16Linear_vulkan&,
                                                 uint32_t, ExpertActivation, float, bool, ActivationBuffer&) const
{
    return false;
}

bool Bfloat16Linear_vulkan::forward_swiglu_chain_device(const DeviceTensor_vulkan&, const Bfloat16Linear_vulkan&,
                                                        uint32_t, ExpertActivation, float, bool,
                                                        DeviceTensor_vulkan&) const
{
    return false;
}

bool Bfloat16Linear_vulkan::materialize(const DeviceTensor_vulkan&, ActivationBuffer&) const
{
    return false;
}

uint32_t Bfloat16Linear_vulkan::input_columns() const noexcept
{
    return 0;
}

uint32_t Bfloat16Linear_vulkan::output_columns() const noexcept
{
    return 0;
}

class Bfloat16Expert_vulkan::Implementation
{
};

Bfloat16Expert_vulkan::Bfloat16Expert_vulkan()
    : d(new Implementation)
{
}

Bfloat16Expert_vulkan::~Bfloat16Expert_vulkan() = default;

std::shared_ptr<Bfloat16Expert_vulkan> Bfloat16Expert_vulkan::create(const TensorData&, const TensorData*,
                                                                     const TensorData&, const TensorData*,
                                                                     float, uint32_t, ExpertActivation,
                                                                     const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::vector<uint64_t> Bfloat16Expert_vulkan::scratch_elements(uint32_t, uint64_t)
{
    return {};
}

bool Bfloat16Expert_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

class Float8Linear_vulkan::Implementation
{
};

Float8Linear_vulkan::Float8Linear_vulkan()
    : d(new Implementation)
{
}

Float8Linear_vulkan::~Float8Linear_vulkan() = default;

std::shared_ptr<Float8Linear_vulkan> Float8Linear_vulkan::create(const TensorData&, const TensorData*, uint32_t,
                                                                 uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::vector<uint64_t> Float8Linear_vulkan::scratch_elements(uint32_t, uint64_t)
{
    return {};
}

std::vector<uint64_t> Float8Linear_vulkan::swiglu_scratch_elements(uint32_t, uint32_t, uint64_t, bool)
{
    return {};
}

bool Float8Linear_vulkan::prepare_rms_norm(const TensorData&, float)
{
    return false;
}

bool Float8Linear_vulkan::prepare_input_rms_norm(const TensorData&, float)
{
    return false;
}

bool Float8Linear_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

bool Float8Linear_vulkan::forward_chain(const ActivationBuffer&, const Float8Linear_vulkan&, ActivationBuffer&) const
{
    return false;
}

bool Float8Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer&, const Float8Linear_vulkan&,
                                                 ActivationBuffer&) const
{
    return false;
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel(const ActivationBuffer&, const Float8Linear_vulkan&,
                                                          const Float8Linear_vulkan&, ActivationBuffer&,
                                                          ActivationBuffer&, DeviceTensor_vulkan*) const
{
    return false;
}

bool Float8Linear_vulkan::forward_input_rms_norm_chain_parallel(const ActivationBuffer&, const Float8Linear_vulkan&,
                                                                const Float8Linear_vulkan&, ActivationBuffer&,
                                                                ActivationBuffer&, DeviceTensor_vulkan*) const
{
    return false;
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel_bfloat16(const ActivationBuffer&, const Float8Linear_vulkan&,
                                                                   const Float8Linear_vulkan&,
                                                                   std::span<const Bfloat16Linear_vulkan*>,
                                                                   std::span<ActivationBuffer*>,
                                                                   ActivationBuffer&, ActivationBuffer&,
                                                                   DeviceTensor_vulkan*) const
{
    return false;
}

bool Float8Linear_vulkan::forward_swiglu_chain(const ActivationBuffer&, const Float8Linear_vulkan&,
                                               const Float8Linear_vulkan&, ExpertActivation,
                                               float, ActivationBuffer&) const
{
    return false;
}

bool Float8Linear_vulkan::forward_swiglu_chain_device(const DeviceTensor_vulkan&, const Float8Linear_vulkan&,
                                                      const Float8Linear_vulkan&, ExpertActivation,
                                                      float, DeviceTensor_vulkan&) const
{
    return false;
}

bool Float8Linear_vulkan::materialize(const DeviceTensor_vulkan&, ActivationBuffer&) const
{
    return false;
}

uint32_t Float8Linear_vulkan::input_columns() const noexcept
{
    return 0;
}

uint32_t Float8Linear_vulkan::output_columns() const noexcept
{
    return 0;
}

class Mxfp4Linear_vulkan::Implementation
{
};

Mxfp4Linear_vulkan::Mxfp4Linear_vulkan()
    : d(new Implementation)
{
}

Mxfp4Linear_vulkan::~Mxfp4Linear_vulkan() = default;

std::shared_ptr<Mxfp4Linear_vulkan> Mxfp4Linear_vulkan::create(const TensorData&, const TensorData*, uint32_t,
                                                               const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

bool Mxfp4Linear_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

uint32_t Mxfp4Linear_vulkan::input_columns() const noexcept
{
    return 0;
}

uint32_t Mxfp4Linear_vulkan::output_columns() const noexcept
{
    return 0;
}

class Mxfp4Expert_vulkan::Implementation
{
};

Mxfp4Expert_vulkan::Mxfp4Expert_vulkan()
    : d(new Implementation)
{
}

Mxfp4Expert_vulkan::~Mxfp4Expert_vulkan() = default;

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create(const TensorData&, const TensorData*,
                                                               const TensorData&, const TensorData*,
                                                               float, uint32_t, ExpertActivation,
                                                               const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create_from_device_storage(const Mxfp4DeviceMatrixView_vulkan&,
                                                                                   const TensorData*,
                                                                                   const Mxfp4DeviceMatrixView_vulkan&,
                                                                                   const TensorData*, float, uint32_t,
                                                                                   const ncnn::VkMat&, ExpertActivation,
                                                                                   const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

bool Mxfp4Expert_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

TensorData qnk_view(const TensorData&)
{
    return {};
}

class QnkLinear_vulkan::Implementation
{
};

QnkLinear_vulkan::QnkLinear_vulkan()
    : d(new Implementation)
{
}

QnkLinear_vulkan::~QnkLinear_vulkan() = default;

uint64_t QnkLinear_vulkan::storage_bytes(const TensorData&, uint64_t) noexcept
{
    return 0;
}

uint64_t QnkLinear_vulkan::projection_storage_bytes(const TensorData&, uint64_t) noexcept
{
    return 0;
}

bool QnkLinear_vulkan::can_fuse(std::span<const TensorData* const>, uint64_t, uint64_t) noexcept
{
    return false;
}

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create_fused(std::span<const TensorData* const>,
                                                                 std::span<const TensorData* const>,
                                                                 uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create(const TensorData&, const TensorData*, uint32_t,
                                                           const VulkanRuntimePtr&, uint64_t, uint32_t)
{
    return {};
}

bool QnkLinear_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

DType QnkLinear_vulkan::dtype() const noexcept
{
    return DType::Q4K;
}

uint32_t QnkLinear_vulkan::input_columns() const noexcept
{
    return 0;
}

uint32_t QnkLinear_vulkan::output_columns() const noexcept
{
    return 0;
}

uint64_t QnkLinear_vulkan::device_bytes() const noexcept
{
    return 0;
}

class QnkExpert_vulkan::Implementation
{
};

QnkExpert_vulkan::QnkExpert_vulkan()
    : d(new Implementation)
{
}

QnkExpert_vulkan::~QnkExpert_vulkan() = default;

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create(const TensorData&, const TensorData*,
                                                           const TensorData&, const TensorData*,
                                                           float, uint32_t, ExpertActivation,
                                                           const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create_separate(const TensorData&, const TensorData*,
                                                                    const TensorData&, const TensorData*,
                                                                    const TensorData&, const TensorData*,
                                                                    float, uint32_t, ExpertActivation,
                                                                    const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::vector<uint64_t> QnkExpert_vulkan::scratch_elements(uint32_t, uint32_t, bool, bool, uint64_t)
{
    return {};
}

bool QnkExpert_vulkan::forward(const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

bool QnkExpert_vulkan::forward_device(const DeviceTensor_vulkan&, DeviceTensor_vulkan&) const
{
    return false;
}

bool QnkExpert_vulkan::prepare_router_gate(const TensorData&)
{
    return false;
}

bool QnkExpert_vulkan::has_router_gate() const noexcept
{
    return false;
}

bool QnkExpert_vulkan::materialize(const DeviceTensor_vulkan&, ActivationBuffer&) const
{
    return false;
}

uint64_t QnkExpert_vulkan::device_bytes() const noexcept
{
    return 0;
}

uint32_t QnkExpert_vulkan::input_columns() const noexcept
{
    return 0;
}

uint32_t QnkExpert_vulkan::output_columns() const noexcept
{
    return 0;
}

uint32_t QnkExpert_vulkan::int4_operations() const noexcept
{
    return 0;
}

uint32_t QnkExpert_vulkan::record_dispatches() const noexcept
{
    return 0;
}

bool QnkExpert_vulkan::use_cm2() const noexcept
{
    return false;
}

// --- Layer -------------------------------------------------------------------

class RmsNorm_vulkan::Implementation
{
};

RmsNorm_vulkan::RmsNorm_vulkan()
    : d(new Implementation)
{
}

RmsNorm_vulkan::~RmsNorm_vulkan() = default;

std::shared_ptr<RmsNorm_vulkan> RmsNorm_vulkan::create(const TensorData&, float, float,
                                                       std::shared_ptr<VulkanContext>,
                                                       const ncnn::Option&, ncnn::VkAllocator*, bool)
{
    return {};
}

bool RmsNorm_vulkan::record(const ncnn::VkMat&, ncnn::VkMat&, ncnn::VkCompute&,
                            const ncnn::VkMat*, ncnn::VkMat*, bool) const
{
    return false;
}

class LayerNorm_vulkan::Implementation
{
};

LayerNorm_vulkan::LayerNorm_vulkan()
    : d(new Implementation)
{
}

LayerNorm_vulkan::~LayerNorm_vulkan() = default;

std::shared_ptr<LayerNorm_vulkan> LayerNorm_vulkan::create(const TensorData&, const TensorData&, float,
                                                           std::shared_ptr<VulkanContext>,
                                                           const ncnn::Option&, ncnn::VkAllocator*)
{
    return {};
}

bool LayerNorm_vulkan::record(const ncnn::VkMat&, ncnn::VkMat&, ncnn::VkCompute&) const
{
    return false;
}

class Router_vulkan::Implementation
{
};

Router_vulkan::Router_vulkan()
    : d(new Implementation)
{
}

Router_vulkan::~Router_vulkan() = default;

uint32_t Router_vulkan::selected_columns(uint32_t top_k) noexcept
{
    return top_k > 0 && top_k <= 64 ? top_k * 2 + 1 : 0;
}

std::shared_ptr<Router_vulkan> Router_vulkan::create(const TensorData&, const TensorData*, const TensorData*,
                                                     uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

Result<void> Router_vulkan::decode_selected(const ActivationBuffer& selected,
                                            const ExpertDispatchOptions& options,
                                            ExpertDispatchPlan& result)
{
    if (result.route_scratch.size() < options.expert_count)
        result.route_scratch.resize(options.expert_count);
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
        result.route_scratch[expert].clear();

    for (size_t row = 0; row < selected.rows(); ++row)
    {
        const float* values = selected.row(row);
        for (uint32_t rank = 0; rank < options.top_k; ++rank)
        {
            const uint32_t expert = static_cast<uint32_t>(values[rank]);
            auto& entries = result.route_scratch[expert];
            entries.push_back({static_cast<uint32_t>(row), rank, values[options.top_k + rank]});
        }
    }

    size_t batch_count = 0;
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
        batch_count += !result.route_scratch[expert].empty();
    if (result.batches.size() < batch_count)
        result.batches.resize(batch_count);

    size_t batch_index = 0;
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
    {
        if (result.route_scratch[expert].empty())
            continue;
        ExpertBatch& batch = result.batches[batch_index++];
        batch.expert_id = expert;
        batch.routes.swap(result.route_scratch[expert]);
    }

    for (size_t i = batch_count; i < result.batches.size(); ++i)
    {
        ExpertBatch& batch = result.batches[i];
        if (batch.expert_id < result.route_scratch.size()
            && batch.routes.capacity() > result.route_scratch[batch.expert_id].capacity())
            batch.routes.swap(result.route_scratch[batch.expert_id]);
    }
    result.batches.resize(batch_count);
    result.assignment_count = selected.rows() * options.top_k;
    return {};
}

class MoeCombinePending_vulkan::Implementation
{
};

MoeCombinePending_vulkan::MoeCombinePending_vulkan()
    : d(new Implementation)
{
}

MoeCombinePending_vulkan::~MoeCombinePending_vulkan() = default;

size_t MoeCombinePending_vulkan::rows() const noexcept
{
    return 0;
}

uint32_t MoeCombinePending_vulkan::columns() const noexcept
{
    return 0;
}

class MoeCombine_vulkan::Implementation
{
};

MoeCombine_vulkan::MoeCombine_vulkan()
    : d(new Implementation)
{
}

MoeCombine_vulkan::~MoeCombine_vulkan() = default;

std::shared_ptr<MoeCombine_vulkan> MoeCombine_vulkan::create(std::shared_ptr<VulkanContext>)
{
    return {};
}

bool MoeCombine_vulkan::supports_grouped_single_row() const noexcept
{
    return false;
}

bool MoeCombine_vulkan::combine(std::span<const MoeCombineInput_vulkan>, size_t, uint32_t,
                                const ActivationBuffer*, std::shared_ptr<const DeviceTensor_vulkan>,
                                std::shared_ptr<DeviceTensor_vulkan>&) const
{
    return false;
}

std::shared_ptr<MoeCombinePending_vulkan> MoeCombine_vulkan::prepare(std::span<const MoeCombineInput_vulkan>,
                                                                     size_t, uint32_t, const ActivationBuffer*,
                                                                     std::shared_ptr<const DeviceTensor_vulkan>) const
{
    return {};
}

bool MoeCombine_vulkan::forward(std::span<const MoeCombineInput_vulkan>, size_t, uint32_t,
                                const ActivationBuffer*, std::shared_ptr<const DeviceTensor_vulkan>,
                                std::shared_ptr<DeviceTensor_vulkan>&)
{
    return false;
}

bool MoeCombine_vulkan::materialize(const DeviceTensor_vulkan&, ActivationBuffer&)
{
    return false;
}

AttnSplitRange split_range(uint64_t destination_count, bool available) noexcept
{
    AttnSplitRange range;
    if (!available || destination_count < 256)
        return range;
    const uint64_t hint = std::min((destination_count + 63) / 64, uint64_t{256});
    if (hint < 2)
        return range;
    const uint64_t per_split = (destination_count + hint - 1) / hint;
    range.tokens = (per_split + 15) / 16 * 16;
    range.splits = (destination_count + range.tokens - 1) / range.tokens;
    if (range.splits < 2)
        range = AttnSplitRange{};
    return range;
}

class AttentionWork_vulkan::Implementation
{
};

AttentionWork_vulkan::AttentionWork_vulkan()
    : d(new Implementation)
{
}

AttentionWork_vulkan::~AttentionWork_vulkan() = default;

bool AttentionWork_vulkan::commit() noexcept
{
    return false;
}

void AttentionWork_vulkan::mark_submit_failed() noexcept
{
}

void AttentionWork_vulkan::retire() noexcept
{
}

class Attention_vulkan::Implementation
{
};

Attention_vulkan::Attention_vulkan()
    : d(new Implementation)
{
}

Attention_vulkan::~Attention_vulkan() = default;

std::vector<uint64_t> Attention_vulkan::scratch_elements(const AttentionConfig_vulkan&, uint32_t, uint32_t,
                                                         uint32_t, uint64_t, bool, bool)
{
    return {};
}

bool Attention_vulkan::flash_ok() const noexcept
{
    return false;
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData&, const TensorData*,
                                                           std::shared_ptr<Linear>, std::shared_ptr<Linear>,
                                                           const AttentionConfig_vulkan&, const TensorData*)
{
    return {};
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData&, const TensorData*,
                                                           std::shared_ptr<Bfloat16Linear_vulkan>,
                                                           std::shared_ptr<Bfloat16Linear_vulkan>,
                                                           const AttentionConfig_vulkan&, const TensorData*)
{
    return {};
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData&, const TensorData&,
                                                           const TensorData&, const TensorData*,
                                                           std::shared_ptr<Bfloat16Linear_vulkan>,
                                                           std::shared_ptr<Bfloat16Linear_vulkan>,
                                                           const AttentionConfig_vulkan&, const TensorData*)
{
    return {};
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create_native(const TensorData&, const TensorData*,
                                                                  const TensorData*, const TensorData*,
                                                                  const std::array<CompiledOperator, 5>&,
                                                                  const AttentionConfig_vulkan&,
                                                                  std::shared_ptr<QnkLinear_vulkan>,
                                                                  const TensorData*, const TensorData*,
                                                                  const std::array<CompiledOperator, 5>*,
                                                                  std::shared_ptr<QnkLinear_vulkan>)
{
    return {};
}

bool Attention_vulkan::forward(uint64_t, LayerCache&, const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

bool Attention_vulkan::materialize_device_cache(LayerCache&) const
{
    return false;
}

AttentionBatchResult_vulkan Attention_vulkan::forward_batch(std::span<const AttentionBatchEntry_vulkan>) const
{
    return AttentionBatchResult_vulkan::NotExecuted;
}

uint64_t LatentCache_vulkan::allocated_bytes() const noexcept
{
    return 0;
}

class LatentValueWork_vulkan::Implementation
{
};

LatentValueWork_vulkan::LatentValueWork_vulkan()
    : d(new Implementation)
{
}

LatentValueWork_vulkan::~LatentValueWork_vulkan() = default;

bool LatentValueWork_vulkan::commit() noexcept
{
    return false;
}

void LatentValueWork_vulkan::abort() noexcept
{
}

class LatentAttentionWork_vulkan::Implementation
{
};

LatentAttentionWork_vulkan::LatentAttentionWork_vulkan()
    : d(new Implementation)
{
}

LatentAttentionWork_vulkan::~LatentAttentionWork_vulkan() = default;

bool LatentAttentionWork_vulkan::commit()
{
    return false;
}

class LatentAttention_vulkan::Implementation
{
};

LatentAttention_vulkan::LatentAttention_vulkan()
    : d(new Implementation)
{
}

LatentAttention_vulkan::~LatentAttention_vulkan() = default;

std::vector<uint64_t> LatentAttention_vulkan::scratch_elements(const AttentionBlockPlan&, uint32_t,
                                                               uint32_t, uint32_t)
{
    return {};
}

std::shared_ptr<LatentAttention_vulkan> LatentAttention_vulkan::create(const AttentionBlockPlan&, const TensorData*,
                                                                       std::shared_ptr<Float8Linear_vulkan>,
                                                                       std::shared_ptr<Float8Linear_vulkan>)
{
    return {};
}

std::shared_ptr<LatentAttention_vulkan> LatentAttention_vulkan::create(const AttentionBlockPlan&, const TensorData*,
                                                                       const CompiledOperator&, const CompiledOperator&)
{
    return {};
}

bool LatentAttention_vulkan::forward_batch(std::span<const uint64_t>, std::span<LayerCache* const>,
                                           std::span<const LatentAttentionRowContext>,
                                           const ActivationBuffer&, const DeviceTensor_vulkan*,
                                           ActivationBuffer&) const
{
    return false;
}

bool LatentAttention_vulkan::forward_value_batch(std::span<const uint64_t>, std::span<LayerCache* const>,
                                                 const ActivationBuffer&, const ActivationBuffer&,
                                                 const ActivationBuffer&, ActivationBuffer&) const
{
    return false;
}

bool LatentAttention_vulkan::forward_projected_batch(const ActivationBuffer&, std::span<const uint64_t>,
                                                     std::span<LayerCache* const>, ActivationBuffer&) const
{
    return false;
}

bool LatentAttention_vulkan::prepare(const WeightStore&, const CompiledOperatorTable&, uint64_t)
{
    return false;
}

bool LatentAttention_vulkan::prepare_value(const WeightStore&, const CompiledOperatorTable&)
{
    return false;
}

uint64_t QsaCache_vulkan::allocated_bytes() const noexcept
{
    return 0;
}

class QsaAttentionWork_vulkan::Implementation
{
};

QsaAttentionWork_vulkan::QsaAttentionWork_vulkan()
    : d(new Implementation)
{
}

QsaAttentionWork_vulkan::~QsaAttentionWork_vulkan() = default;

bool QsaAttentionWork_vulkan::commit()
{
    return false;
}

class QsaAttention_vulkan::Implementation
{
};

QsaAttention_vulkan::QsaAttention_vulkan()
    : d(new Implementation)
{
}

QsaAttention_vulkan::~QsaAttention_vulkan() = default;

std::shared_ptr<QsaAttention_vulkan> QsaAttention_vulkan::create(const WeightStore&, const AttentionBlockPlan&,
                                                                 uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::vector<uint64_t> QsaAttention_vulkan::scratch_elements(const AttentionBlockPlan&, uint32_t, uint32_t, uint32_t)
{
    return {};
}

uint64_t QsaAttention_vulkan::initial_state_bytes() const noexcept
{
    return 0;
}

class GatedDeltaState_vulkan::Implementation
{
};

GatedDeltaState_vulkan::GatedDeltaState_vulkan()
    : d(new Implementation)
{
}

GatedDeltaState_vulkan::~GatedDeltaState_vulkan() = default;

std::shared_ptr<GatedDeltaState_vulkan> GatedDeltaState_vulkan::create(const std::shared_ptr<VulkanContext>&,
                                                                       uint32_t, uint32_t, uint32_t,
                                                                       uint32_t, uint32_t, const ncnn::Option&)
{
    return {};
}

bool GatedDeltaState_vulkan::begin_transaction(size_t) noexcept
{
    return false;
}

bool GatedDeltaState_vulkan::prepare_transaction_finish(size_t, size_t) noexcept
{
    return false;
}

void GatedDeltaState_vulkan::complete_transaction() noexcept
{
}

bool GatedDeltaState_vulkan::prepare_cpu_state(const std::vector<float>&, const std::vector<float>&)
{
    return false;
}

bool GatedDeltaState_vulkan::download(std::vector<float>&, std::vector<float>&) const
{
    return false;
}

uint64_t GatedDeltaState_vulkan::allocated_bytes() const noexcept
{
    return 0;
}

class GatedDeltaWork_vulkan::Implementation
{
};

GatedDeltaWork_vulkan::GatedDeltaWork_vulkan()
    : d(new Implementation)
{
}

GatedDeltaWork_vulkan::~GatedDeltaWork_vulkan() = default;

bool GatedDeltaWork_vulkan::commit() noexcept
{
    return false;
}

void GatedDeltaWork_vulkan::mark_submit_failed() noexcept
{
}

class GatedDeltaNet_vulkan::Implementation
{
};

GatedDeltaNet_vulkan::GatedDeltaNet_vulkan()
    : d(new Implementation)
{
}

GatedDeltaNet_vulkan::~GatedDeltaNet_vulkan() = default;

std::shared_ptr<GatedDeltaNet_vulkan> GatedDeltaNet_vulkan::create(const std::shared_ptr<Bfloat16Linear_vulkan>&,
                                                                   const TensorData&, const TensorData&,
                                                                   const TensorData&, const TensorData&,
                                                                   const std::shared_ptr<Bfloat16Linear_vulkan>&,
                                                                   uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                                                   float, bool, uint32_t, float, uint32_t,
                                                                   const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

std::shared_ptr<GatedDeltaNet_vulkan> GatedDeltaNet_vulkan::create_quantized(std::span<const std::shared_ptr<QnkLinear_vulkan>>,
                                                                             const TensorData&, const TensorData&,
                                                                             const TensorData&, const TensorData&,
                                                                             const std::shared_ptr<QnkLinear_vulkan>&,
                                                                             uint32_t, uint32_t, uint32_t, uint32_t,
                                                                             uint32_t, float, bool, uint32_t, float,
                                                                             uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

bool GatedDeltaNet_vulkan::forward(const ActivationBuffer&, LayerCache&, ActivationBuffer&) const
{
    return false;
}

bool GatedDeltaNet_vulkan::forward_input_rms_norm(const ActivationBuffer&, LayerCache&, ActivationBuffer&) const
{
    return false;
}

bool GatedDeltaNet_vulkan::has_input_rms_norm() const noexcept
{
    return false;
}

GatedDeltaBatchResult_vulkan GatedDeltaNet_vulkan::forward_batch(std::span<const GatedDeltaBatchEntry_vulkan>) const
{
    return GatedDeltaBatchResult_vulkan::NotExecuted;
}

std::vector<uint64_t> GatedDeltaNet_vulkan::scratch_elements(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
{
    return {};
}

class HyperConnection_vulkan::Implementation
{
};

HyperConnection_vulkan::HyperConnection_vulkan()
    : d(new Implementation)
{
}

HyperConnection_vulkan::~HyperConnection_vulkan() = default;

uint64_t HyperConnection_vulkan::device_bytes() const noexcept
{
    return 0;
}

std::shared_ptr<HyperConnection_vulkan> HyperConnection_vulkan::create(const TensorData&, const TensorData&,
                                                                       const TensorData&, uint32_t, uint32_t,
                                                                       float, float, uint32_t,
                                                                       const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

class GatedResidual_vulkan::Implementation
{
};

GatedResidual_vulkan::GatedResidual_vulkan()
    : d(new Implementation)
{
}

GatedResidual_vulkan::~GatedResidual_vulkan() = default;

std::shared_ptr<GatedResidual_vulkan> GatedResidual_vulkan::create(const WeightStore&, const GatedResidualPlan&,
                                                                   const MoeModelDescriptor&, uint32_t,
                                                                   const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

// --- Expert ------------------------------------------------------------------

class ResidentFfnRegion_vulkan::Implementation
{
};

ResidentFfnRegion_vulkan::ResidentFfnRegion_vulkan()
    : d(new Implementation)
{
}

ResidentFfnRegion_vulkan::~ResidentFfnRegion_vulkan() = default;

std::shared_ptr<ResidentFfnRegion_vulkan> ResidentFfnRegion_vulkan::create(const ActivationBuffer&, size_t, uint32_t)
{
    return {};
}

std::shared_ptr<ResidentFfnRegion_vulkan> ResidentFfnRegion_vulkan::create(std::shared_ptr<const DeviceTensor_vulkan>,
                                                                           size_t, uint32_t,
                                                                           ResidentFfnOutput_vulkan)
{
    return {};
}

bool ResidentFfnRegion_vulkan::take_output(ActivationBuffer&)
{
    return false;
}

bool ResidentFfnRegion_vulkan::take_device_output(std::shared_ptr<const DeviceTensor_vulkan>&)
{
    return false;
}

bool ResidentFfnRegion_vulkan::complete() const noexcept
{
    return false;
}

size_t ResidentFfnRegion_vulkan::rows() const noexcept
{
    return 0;
}

uint32_t ResidentFfnRegion_vulkan::columns() const noexcept
{
    return 0;
}

uint64_t ResidentFfnRegion_vulkan::pending_combine_dispatches() const noexcept
{
    return 0;
}

uint64_t ResidentFfnRegion_vulkan::completed_combine_dispatches() const noexcept
{
    return 0;
}

bool ResidentFfnRegion_vulkan::set_continuation(std::shared_ptr<ResidentFfnContinuation_vulkan>)
{
    return false;
}

void ResidentFfnRegion_vulkan::mark_submit_attempted() noexcept
{
}

bool support_vulkan_shared_experts(const CompiledOperatorTable&, const MoeBlockPlan&) noexcept
{
    return false;
}

bool support_vulkan_ffn(const CompiledOperatorTable&, const ExpertPlan&) noexcept
{
    return false;
}

class SharedFfn_vulkan::Implementation
{
};

SharedFfn_vulkan::SharedFfn_vulkan()
    : d(new Implementation)
{
}

SharedFfn_vulkan::~SharedFfn_vulkan() = default;

std::shared_ptr<SharedFfn_vulkan> SharedFfn_vulkan::create(const CompiledModel&, const MoeBlockPlan&)
{
    return {};
}

std::shared_ptr<SharedFfn_vulkan> SharedFfn_vulkan::create(const CompiledModel&, const ExpertPlan&)
{
    return {};
}

const std::shared_ptr<VulkanContext>& SharedFfn_vulkan::context() const noexcept
{
    static const std::shared_ptr<VulkanContext> empty_context;
    return empty_context;
}

class ExpertBank_vulkan::Implementation
{
};

ExpertBank_vulkan::ExpertBank_vulkan()
    : d(new Implementation)
{
}

ExpertBank_vulkan::~ExpertBank_vulkan() = default;

Result<ExpertBankPlan_vulkan> ExpertBank_vulkan::plan(std::span<const ExpertBankWeights_vulkan>,
                                                      const std::shared_ptr<VulkanContext>&)
{
    return Error{ErrorCode::UnsupportedModel, "Vulkan Expert banks are unavailable in this build"};
}

std::vector<uint64_t> ExpertBank_vulkan::scratch_elements(const ExpertBankPlan_vulkan&, uint64_t, uint32_t)
{
    return {};
}

Result<std::shared_ptr<ExpertBank_vulkan>> ExpertBank_vulkan::create(const ExpertBankPlan_vulkan&,
                                                                     const std::shared_ptr<VulkanContext>&,
                                                                     const ExpertBankLoader_vulkan&)
{
    return Error{ErrorCode::UnsupportedModel, "Vulkan Expert banks are unavailable in this build"};
}

const std::shared_ptr<VulkanContext>& ExpertBank_vulkan::context() const noexcept
{
    static const std::shared_ptr<VulkanContext> empty_context;
    return empty_context;
}

const ExpertBankPlan_vulkan& ExpertBank_vulkan::description() const noexcept
{
    static const ExpertBankPlan_vulkan empty_plan{};
    return empty_plan;
}

uint64_t ExpertBank_vulkan::physical_size() const noexcept
{
    return 0;
}

uint64_t quant_expert_bytes(const TensorData&, const TensorData*, const TensorData&, uint32_t,
                            const VulkanRuntimePtr&, uint64_t)
{
    return 0;
}

uint64_t quant_expert_bytes(const TensorData&, const TensorData*, const TensorData&, uint32_t,
                            const VulkanRuntimePtr&, uint64_t, const TensorData*, const TensorData*,
                            const TensorData*, ExpertActivation, float)
{
    return 0;
}

uint64_t quant_cache_size(uint64_t available_size, uint64_t total_size, uint64_t largest_expert_size) noexcept
{
    if (total_size < largest_expert_size || largest_expert_size == 0)
        return 0;
    const uint64_t available = std::min(available_size, total_size);
    return available < largest_expert_size ? 0 : available;
}

std::shared_ptr<ExpertBackend> create_vulkan_expert_backend(uint64_t, uint32_t,
                                                            std::shared_ptr<ExpertVictimCache>,
                                                            const VulkanRuntimePtr&, uint64_t, uint64_t)
{
    return {};
}

std::shared_ptr<ExpertVictimCache> create_vulkan_victim_cache(uint64_t, uint32_t, const VulkanRuntimePtr&,
                                                              uint64_t, uint64_t)
{
    return {};
}

// --- Net ---------------------------------------------------------------------

class LayerHead_vulkan::Implementation
{
};

LayerHead_vulkan::LayerHead_vulkan()
    : d(new Implementation)
{
}

LayerHead_vulkan::~LayerHead_vulkan() = default;

std::shared_ptr<LayerHead_vulkan> LayerHead_vulkan::create(const TensorData&, const TensorData&, const TensorData&,
                                                           const TensorData&, const CompiledOperator&, uint32_t,
                                                           float, float, uint32_t, const VulkanRuntimePtr&, uint64_t)
{
    return {};
}

class LatentLayerWorkspace_vulkan::Implementation
{
};

LatentLayerWorkspace_vulkan::LatentLayerWorkspace_vulkan()
    : d(new Implementation)
{
}

LatentLayerWorkspace_vulkan::~LatentLayerWorkspace_vulkan() = default;

void LatentLayerWorkspace_vulkan::reset()
{
}

void LatentLayerWorkspace_vulkan::release()
{
}

uint64_t LatentLayerWorkspace_vulkan::allocated_bytes() const noexcept
{
    return 0;
}

bool LatentLayerWorkspace_vulkan::active() const noexcept
{
    return false;
}

const std::shared_ptr<const DeviceTensor_vulkan>& LatentLayerWorkspace_vulkan::device_input() const noexcept
{
    static const std::shared_ptr<const DeviceTensor_vulkan> empty_tensor;
    return empty_tensor;
}

std::shared_ptr<LatentLayer_vulkan> LatentLayer_vulkan::create(std::shared_ptr<LatentAttention_vulkan>,
                                                               std::shared_ptr<HyperConnection_vulkan>,
                                                               std::shared_ptr<HyperConnection_vulkan>,
                                                               std::shared_ptr<RmsNorm_vulkan>,
                                                               std::shared_ptr<Router_vulkan>)
{
    return {};
}

bool LatentLayer_vulkan::forward(const ActivationBuffer&, std::span<const uint64_t>,
                                 std::span<LayerCache* const>, const ExpertDispatchOptions&,
                                 ActivationBuffer&, ExpertDispatchPlan&, LatentLayerWorkspace_vulkan&) const
{
    return false;
}

bool LatentLayer_vulkan::combine_device(std::span<const MoeCombineInput_vulkan>, size_t, uint32_t,
                                        const ActivationBuffer*, std::shared_ptr<const DeviceTensor_vulkan>,
                                        std::shared_ptr<DeviceTensor_vulkan>&, const LatentLayerWorkspace_vulkan&)
{
    return false;
}

bool LatentLayer_vulkan::defer_combine(std::span<const MoeCombineInput_vulkan>, size_t, uint32_t,
                                       const ActivationBuffer*, std::shared_ptr<const DeviceTensor_vulkan>,
                                       LatentLayerWorkspace_vulkan&)
{
    return false;
}

bool LatentLayer_vulkan::defer_combine(ActivationBuffer&, LatentLayerWorkspace_vulkan&)
{
    return false;
}

bool LatentLayer_vulkan::defer_combine(std::shared_ptr<const DeviceTensor_vulkan>, LatentLayerWorkspace_vulkan&)
{
    return false;
}

bool LatentLayer_vulkan::materialize_hidden(LatentLayerWorkspace_vulkan&, ActivationBuffer&)
{
    return false;
}

bool LatentLayer_vulkan::can_finish(const LatentLayerWorkspace_vulkan&, const LayerHead_vulkan&) noexcept
{
    return false;
}

bool LatentLayer_vulkan::finish(LatentLayerWorkspace_vulkan&, const LayerHead_vulkan&, bool, bool,
                                ActivationBuffer&, ActivationBuffer&)
{
    return false;
}

class PleState_vulkan::Implementation
{
};

PleState_vulkan::PleState_vulkan()
    : d(new Implementation)
{
}

PleState_vulkan::~PleState_vulkan() = default;

uint64_t PleState_vulkan::allocated_bytes() const noexcept
{
    return 0;
}

class GatedResidualWorkspace_vulkan::Implementation
{
};

GatedResidualWorkspace_vulkan::GatedResidualWorkspace_vulkan()
    : d(new Implementation)
{
}

GatedResidualWorkspace_vulkan::~GatedResidualWorkspace_vulkan() = default;

void GatedResidualWorkspace_vulkan::reset()
{
}

bool GatedResidualWorkspace_vulkan::active() const noexcept
{
    return false;
}

const std::shared_ptr<const DeviceTensor_vulkan>& GatedResidualWorkspace_vulkan::device_input() const noexcept
{
    static const std::shared_ptr<const DeviceTensor_vulkan> empty_tensor;
    return empty_tensor;
}

class GatedResidualLayer_vulkan::Implementation
{
};

GatedResidualLayer_vulkan::GatedResidualLayer_vulkan()
    : d(new Implementation)
{
}

GatedResidualLayer_vulkan::~GatedResidualLayer_vulkan() = default;

std::shared_ptr<GatedResidualLayer_vulkan> GatedResidualLayer_vulkan::create(const CompiledModel&,
                                                                             const CompiledLayerPlan&)
{
    return {};
}

std::shared_ptr<GatedResidualLayer_vulkan> GatedResidualLayer_vulkan::create_head(const CompiledModel&)
{
    return {};
}

bool GatedResidualLayer_vulkan::forward(const ActivationBuffer&, std::span<const int32_t>, uint64_t, LayerCache&,
                                        const ExpertDispatchOptions&, ExpertDispatchPlan&,
                                        GatedResidualWorkspace_vulkan&, std::string&) const
{
    return false;
}

bool GatedResidualLayer_vulkan::defer_combine(std::span<const MoeCombineInput_vulkan>, size_t, uint32_t,
                                              std::shared_ptr<const DeviceTensor_vulkan>,
                                              GatedResidualWorkspace_vulkan&)
{
    return false;
}

bool GatedResidualLayer_vulkan::finish(const CompiledModel&, bool, bool, ActivationBuffer&,
                                       GatedResidualWorkspace_vulkan&)
{
    return false;
}

uint32_t resident_chunk_layers(const CompiledModel&, uint32_t) noexcept
{
    return 0;
}

class GpuSegmentWorkspace_vulkan::Implementation
{
};

GpuSegmentWorkspace_vulkan::GpuSegmentWorkspace_vulkan()
    : d(new Implementation)
{
}

GpuSegmentWorkspace_vulkan::~GpuSegmentWorkspace_vulkan() = default;

void GpuSegmentWorkspace_vulkan::reset()
{
}

void GpuSegmentWorkspace_vulkan::reserve_host_output(uint64_t) noexcept
{
}

bool GpuSegmentWorkspace_vulkan::active() const noexcept
{
    return false;
}

uint64_t GpuSegmentWorkspace_vulkan::allocated_bytes() const noexcept
{
    return 0;
}

const std::shared_ptr<const DeviceTensor_vulkan>& GpuSegmentWorkspace_vulkan::device_input() const noexcept
{
    static const std::shared_ptr<const DeviceTensor_vulkan> empty_tensor;
    return empty_tensor;
}

class GpuSegment_vulkan::Implementation
{
};

GpuSegment_vulkan::GpuSegment_vulkan()
    : d(new Implementation)
{
}

GpuSegment_vulkan::~GpuSegment_vulkan() = default;

bool GpuSegment_vulkan::supports(const CompiledModel&, const CompiledLayerPlan&) noexcept
{
    return false;
}

bool GpuSegment_vulkan::supports_head(const CompiledModel&) noexcept
{
    return false;
}

std::shared_ptr<GpuSegment_vulkan> GpuSegment_vulkan::create(const CompiledModel&, const CompiledLayerPlan&)
{
    return {};
}

std::shared_ptr<GpuSegment_vulkan> GpuSegment_vulkan::create_head(const CompiledModel&)
{
    return {};
}

const std::shared_ptr<VulkanContext>& GpuSegment_vulkan::context() const noexcept
{
    static const std::shared_ptr<VulkanContext> empty_context;
    return empty_context;
}

Result<uint32_t> GpuSegment_vulkan::continuation_target(const CompiledModel&, uint32_t, uint32_t& fence_id)
{
    fence_id = invalid_layer_id;
    return invalid_layer_id;
}

bool GpuSegment_vulkan::forward_resident(const CompiledModel&, const ActivationBuffer&,
                                         std::span<const uint64_t>, std::span<LayerCache* const>,
                                         std::span<const ExpertDispatchOptions>, bool, bool,
                                         ActivationBuffer&, std::string&, GpuSegmentWorkspace_vulkan*,
                                         uint32_t)
{
    return false;
}

std::shared_ptr<GpuSegmentContinuation_vulkan> GpuSegment_vulkan::prepare_continuation(std::span<const uint64_t>,
                                                                                       std::span<LayerCache* const>,
                                                                                       const ExpertDispatchOptions&,
                                                                                       const GpuSegmentWorkspace_vulkan&,
                                                                                       std::string&) const
{
    return {};
}

std::shared_ptr<GpuSegmentContinuation_vulkan> GpuSegment_vulkan::prepare_head_continuation(bool, bool,
                                                                                            const GpuSegmentWorkspace_vulkan&,
                                                                                            std::string&) const
{
    return {};
}

bool GpuSegment_vulkan::forward(const ActivationBuffer&, std::span<const uint64_t>, std::span<LayerCache* const>,
                                const ExpertDispatchOptions&, ExpertDispatchPlan&, GpuSegmentWorkspace_vulkan&,
                                std::string&) const
{
    return false;
}

bool GpuSegment_vulkan::defer_combine(std::shared_ptr<const DeviceTensor_vulkan>, GpuSegmentWorkspace_vulkan&)
{
    return false;
}

bool GpuSegment_vulkan::materialize_hidden(GpuSegmentWorkspace_vulkan&, ActivationBuffer&, std::string&)
{
    return false;
}

bool GpuSegment_vulkan::finish(bool, bool, ActivationBuffer&, ActivationBuffer&, GpuSegmentWorkspace_vulkan&,
                               std::string&) const
{
    return false;
}

class GpuSegmentContinuation_vulkan::Implementation
{
};

GpuSegmentContinuation_vulkan::GpuSegmentContinuation_vulkan()
    : d(new Implementation)
{
}

GpuSegmentContinuation_vulkan::~GpuSegmentContinuation_vulkan() = default;

const std::shared_ptr<VulkanContext>& GpuSegmentContinuation_vulkan::context() const noexcept
{
    static const std::shared_ptr<VulkanContext> empty_context;
    return empty_context;
}

void GpuSegmentContinuation_vulkan::mark_submit_attempted() noexcept
{
}

void GpuSegmentContinuation_vulkan::mark_completed() noexcept
{
}

void GpuSegmentContinuation_vulkan::abort() noexcept
{
}

bool GpuSegmentContinuation_vulkan::publish(ExpertDispatchPlan&, GpuSegmentWorkspace_vulkan&, std::string&)
{
    return false;
}

bool GpuSegmentContinuation_vulkan::publish_head(ActivationBuffer&, GpuSegmentWorkspace_vulkan&, std::string&)
{
    return false;
}

uint64_t attn_vk_size(const CompiledModel&, const CompiledLayerPlan&, uint64_t, uint32_t) noexcept
{
    return 0;
}

uint64_t dense_weight_size(const CompiledModel&, const TensorData&, uint64_t, uint32_t, bool) noexcept
{
    return 0;
}

uint64_t dense_pending_size(const CompiledModel&, uint32_t) noexcept
{
    return 0;
}

uint64_t vulkan_pending_reserve(const CompiledModel&, uint32_t) noexcept
{
    return 0;
}

Result<ModelGraphMemoryPlan> plan_model_graph_memory(const CompiledModel&,
                                                     uint32_t prefill_rows,
                                                     uint32_t sequence_count,
                                                     uint32_t context_tokens,
                                                     uint32_t)
{
    if (prefill_rows == 0 || sequence_count == 0)
        return Error{ErrorCode::InvalidArgument, "graph memory planning requires nonzero prefill rows and sequence count"};
    ModelGraphMemoryPlan result;
    result.prefill_rows = prefill_rows;
    result.sequence_count = sequence_count;
    result.context_tokens = std::max(prefill_rows, context_tokens);
    return result;
}

Result<void> prepare_model_pipeline(CompiledModel& compiled, const CompilerOption&)
{
    compiled.operators.bind_weight_count(compiled.weights.size());
    return {};
}

void release_host_copies(CompiledModel&)
{
}

bool support_vulkan_experts(const WeightStore& weights,
                            const MoeBlockPlan& moe,
                            uint64_t optimization_flags) noexcept
{
    if (moe.experts.empty())
        return false;
    for (const ExpertPlan& expert : moe.experts)
    {
        const TensorHandle gate = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight
                                                                                 : expert.gate_weight;
        if (gate == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle)
            return false;
        const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                   ? &weights.at(expert.up_weight)
                                   : nullptr;
        if (!support_vulkan_expert(expert, weights.at(gate), weights.at(expert.down_weight), optimization_flags, up))
            return false;
    }
    return true;
}

bool support_vulkan_attention(const CompiledOperatorTable&, const AttentionBlockPlan&) noexcept
{
    return false;
}

} // namespace moe
} // namespace ncnn
