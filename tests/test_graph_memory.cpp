#include "graph/graphmemory.h"
#include "graph/compiledmodel.h"
#include "graph/submissionbudget.h"
#include "backends/ncnn/modelpipeline.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace ncnn {
namespace moe {

static void memory_check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static void test_submission_budget()
{
    for (double estimate : {0.0, 4000.0, std::numeric_limits<double>::infinity()})
    {
        GpuSubmissionBudget budget(estimate);
        memory_check(budget.append(std::numeric_limits<double>::max()),
                     "recorded work must make progress even with an empty or unbounded estimate");
    }
}

static const GraphMemoryAllocation& allocation(const GraphMemoryPlan& plan, uint32_t id, uint32_t device = 0)
{
    const auto found = std::find_if(plan.allocations.begin(), plan.allocations.end(), [&](const GraphMemoryAllocation& item) {
        return item.id == id && item.device_index == device;
    });
    memory_check(found != plan.allocations.end(), "planned allocation is missing");
    return *found;
}

static uint64_t peak(const GraphMemoryPlan& plan, uint32_t device = 0)
{
    const auto found = std::find_if(plan.devices.begin(), plan.devices.end(), [&](const GraphMemoryDevice& item) {
        return item.device_index == device;
    });
    memory_check(found != plan.devices.end(), "planned device is missing");
    return found->peak_size;
}

static void check_nonoverlap(const GraphMemoryPlan& plan, std::span<const GraphMemoryBuffer> buffers)
{
    memory_check(plan.allocations.size() == buffers.size(), "one allocation is required for every buffer");
    for (size_t index = 0; index < buffers.size(); ++index)
    {
        const GraphMemoryBuffer& source = buffers[index];
        const GraphMemoryAllocation& item = plan.allocations[index];
        memory_check(item.id == source.id && item.device_index == source.device_index
                         && item.size == source.size && item.first_node == source.first_node && item.last_node == source.last_node,
                     "allocations must retain input order and lifetime metadata");
        memory_check(item.offset % source.alignment == 0, "an allocation violates its buffer alignment");
        memory_check(item.size <= peak(plan, item.device_index)
                         && item.offset <= peak(plan, item.device_index) - item.size,
                     "an allocation exceeds its device arena peak");
        for (size_t previous = 0; previous < index; ++previous)
        {
            const GraphMemoryAllocation& other = plan.allocations[previous];
            if (item.device_index != other.device_index || item.size == 0 || other.size == 0
                || item.last_node < other.first_node || other.last_node < item.first_node)
                continue;
            memory_check(item.offset >= other.offset + other.size || other.offset >= item.offset + item.size,
                         "simultaneously live graph buffers overlap in device memory");
        }
    }
}

static void test_closed_lifetimes_and_branches()
{
    const std::array<GraphMemoryBuffer, 4> diamond = {{
        {0, 0, 16, 1, 0, 2},
        {1, 0, 8, 1, 1, 3},
        {2, 0, 8, 1, 2, 3},
        {3, 0, 16, 1, 3, 4},
    }};
    const auto plan = plan_graph_memory(diamond);
    memory_check(static_cast<bool>(plan), "branched graph planning failed");
    check_nonoverlap(plan.value(), diamond);
    memory_check(peak(plan.value()) == 32, "branched consumers must retain their source through its last use");
    memory_check(allocation(plan.value(), 3).offset == allocation(plan.value(), 0).offset,
                 "a merged result should reuse its expired source storage");

    const std::array<GraphMemoryBuffer, 3> same_step = {{
        {0, 0, 16, 8, 0, 1},
        {1, 0, 16, 8, 1, 1},
        {2, 0, 32, 8, 2, 2},
    }};
    const auto closed = plan_graph_memory(same_step);
    memory_check(static_cast<bool>(closed), "closed lifetime planning failed");
    check_nonoverlap(closed.value(), same_step);
    memory_check(allocation(closed.value(), 0).offset != allocation(closed.value(), 1).offset,
                 "same-node inputs and outputs must not alias");
    memory_check(peak(closed.value()) == 32 && allocation(closed.value(), 2).offset == 0,
                 "adjacent expired buffers must coalesce for a larger successor");
}

static void test_submission_fences()
{
    const std::array<GraphMemoryBuffer, 7> buffers = {{
        {0, 0, 64, 16, 0, 1},
        {1, 0, 32, 16, 2, 2},
        {2, 0, 16, 16, 3, 3},
        {3, 0, 64, 16, 4, 4},
        {4, 0, 0, 16, 2, 2},
        {0, 1, 64, 16, 0, 1},
        {1, 1, 64, 16, 2, 2},
    }};
    const auto separate = plan_graph_memory(buffers);
    memory_check(separate && peak(separate.value(), 0) == 64 && peak(separate.value(), 1) == 64,
                 "separate submissions must reuse expired temporary storage");
    const std::array<GraphMemorySubmission, 1> submission = {{{0, 1, 3}}};
    const auto joined = plan_graph_memory(buffers, submission);
    memory_check(joined && peak(joined.value(), 0) == 112 && peak(joined.value(), 1) == 64,
                 "a common fence must retain inputs and all recorded temporaries on that device only");
    auto retained = buffers;
    for (GraphMemoryBuffer& buffer : retained)
        if (buffer.device_index == 0 && buffer.first_node <= 3 && buffer.last_node >= 1)
            buffer.last_node = 3;
    check_nonoverlap(joined.value(), retained);
    memory_check(allocation(joined.value(), 0).first_node == 0 && allocation(joined.value(), 0).last_node == 3
                     && allocation(joined.value(), 1).last_node == 3 && allocation(joined.value(), 2).last_node == 3
                     && allocation(joined.value(), 4).last_node == 3,
                 "fence retention must include earlier-produced inputs, private temporaries, and zero-sized descriptors");
    memory_check(allocation(joined.value(), 3).offset == 0 && allocation(joined.value(), 3).last_node == 4
                     && buffers[0].last_node == 1,
                 "completion must release the submission storage for its successor without mutating caller facts");
    memory_check(allocation(joined.value(), 0, 1).last_node == 1 && allocation(joined.value(), 1, 1).offset == 0,
                 "another device must not inherit this submission fence");

    // Unsorted ranges and closed boundary nodes are accepted without merging
    // neighboring submissions into an artificial whole-graph lifetime.
    const std::array<GraphMemorySubmission, 3> ranges = {{{0, 3, 5}, {1, 1, 2}, {0, 0, 1}}};
    const auto multiple = plan_graph_memory(buffers, ranges);
    memory_check(multiple && allocation(multiple.value(), 1).last_node == 2
                     && allocation(multiple.value(), 2).last_node == 5 && allocation(multiple.value(), 3).last_node == 5
                     && peak(multiple.value(), 1) == 128,
                 "independent submission intervals must preserve gaps and per-device boundaries");
    const std::array<GraphMemorySubmission, 1> reversed = {{{0, 4, 3}}};
    const std::array<GraphMemorySubmission, 2> overlap = {{{0, 1, 3}, {0, 3, 4}}};
    memory_check(!plan_graph_memory(buffers, reversed) && !plan_graph_memory(buffers, overlap)
                     && !plan_graph_memory({}, reversed),
                 "invalid or overlapping closed submission intervals must be rejected even for an empty graph");
    const std::array<GraphMemorySubmission, 1> maximum = {{{0, 1, UINT32_MAX}}};
    const auto last_node = plan_graph_memory(buffers, maximum);
    memory_check(last_node && allocation(last_node.value(), 0).last_node == UINT32_MAX
                     && peak(last_node.value(), 0) == 176,
                 "a maximal fence position must extend lifetimes without overflowing a node increment");
}

static void test_alignment_and_reuse()
{
    const std::array<GraphMemoryBuffer, 4> aligned = {{
        {0, 0, 3, 1, 0, 0},
        {1, 0, 4, 8, 0, 1},
        {2, 0, 4, 4, 0, 1},
        {3, 0, 8, 8, 2, 2},
    }};
    const auto padded = plan_graph_memory(aligned);
    memory_check(static_cast<bool>(padded), "aligned graph planning failed");
    check_nonoverlap(padded.value(), aligned);

    const std::array<GraphMemoryBuffer, 2> growing = {{{0, 0, 16, 1, 0, 0}, {1, 0, 32, 1, 1, 1}}};
    const auto tail = plan_graph_memory(growing);
    memory_check(tail && peak(tail.value()) == 32,
                 "expired storage must remain reusable rather than accumulating every working set");
    check_nonoverlap(tail.value(), growing);
}

static void test_devices_empty_buffers_and_validation()
{
    const std::array<GraphMemoryBuffer, 4> buffers = {{
        {5, 7, 64, 16, 0, 2},
        {5, 2, 24, 8, 0, 2},
        {6, 7, 32, 16, 1, 2},
        {7, 9, 0, 256, 0, std::numeric_limits<uint32_t>::max()},
    }};
    const auto plan = plan_graph_memory(buffers);
    memory_check(static_cast<bool>(plan), "independent device planning failed");
    check_nonoverlap(plan.value(), buffers);
    memory_check(plan.value().devices.size() == 3 && plan.value().devices.front().device_index == 2
                     && peak(plan.value(), 2) == 24 && peak(plan.value(), 7) == 96 && peak(plan.value(), 9) == 0,
                 "device peaks and equal buffer IDs on different devices must remain independent");
    const auto empty = plan_graph_memory({});
    memory_check(empty && empty.value().allocations.empty() && empty.value().devices.empty(),
                 "an empty graph must not allocate device arenas");
    for (const GraphMemoryBuffer invalid : {GraphMemoryBuffer{0, 0, 1, 0, 0, 1},
                                            GraphMemoryBuffer{0, 0, 1, 3, 0, 1},
                                            GraphMemoryBuffer{0, 0, 0, 1, 2, 1}})
    {
        const auto rejected = plan_graph_memory({&invalid, 1});
        memory_check(!rejected && rejected.error().code == ErrorCode::InvalidArgument,
                     "invalid alignment or lifetime must return a validation error");
    }
    const std::array<GraphMemoryBuffer, 2> duplicate = {{{0, 2, 8, 1, 0, 1}, {0, 2, 8, 1, 2, 3}}};
    memory_check(!plan_graph_memory(duplicate), "duplicate buffer identity must not produce ambiguous allocations");
}

static void test_overflow()
{
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    const std::array<GraphMemoryBuffer, 2> size_overflow = {{{0, 0, maximum, 1, 0, 1}, {1, 0, 1, 1, 0, 1}}};
    const auto rejected_size = plan_graph_memory(size_overflow);
    memory_check(!rejected_size && rejected_size.error().code == ErrorCode::InvalidArgument,
                 "simultaneously live buffer sizes cannot wrap the device peak");
    const std::array<GraphMemoryBuffer, 2> alignment_overflow = {{{0, 0, maximum - 3, 1, 0, 1}, {1, 0, 1, 8, 0, 1}}};
    memory_check(!plan_graph_memory(alignment_overflow), "alignment rounding cannot wrap a device offset");
    const std::array<GraphMemoryBuffer, 2> expired_large = {{{0, 0, maximum, 1, 0, 0}, {1, 0, maximum, 1, 1, 1}}};
    const auto reused = plan_graph_memory(expired_large);
    memory_check(reused && peak(reused.value()) == maximum && allocation(reused.value(), 1).offset == 0,
                 "valid maximal ranges must remain reusable without addition overflow");
}

static void append_model_memory_node(CompiledModel& model, ExecutionNodeType type, ExecutionBackend backend,
                                     uint32_t layer, std::initializer_list<uint32_t> inputs,
                                     std::initializer_list<uint32_t> outputs)
{
    ExecutionNode node;
    node.id = static_cast<uint32_t>(model.graph.nodes.size());
    node.type = type;
    node.backend = backend;
    node.backend_mask = backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu;
    node.layer_plan_index = layer;
    node.inputs = inputs;
    node.outputs = outputs;
    if (node.id != 0) node.dependencies.push_back(node.id - 1);
    for (uint32_t input : node.inputs) model.graph.tensors[input].consumers.push_back(node.id);
    for (uint32_t output : node.outputs) model.graph.tensors[output].producer = node.id;
    model.schedule.node_order.push_back(node.id);
    model.graph.nodes.push_back(std::move(node));
}

static CompiledModel model_memory_fixture()
{
    CompiledModel model;
    model.descriptor.hidden_size = 16;
    model.descriptor.vocabulary_size = 64;
    model.descriptor.layers.resize(2);
    model.opt.hybrid_mode = HybridMode::HybridExperts;
    model.opt.vulkan_device_index = 2;
    model.graph.layer_plans.resize(2);
    for (uint32_t index = 0; index < 2; ++index)
    {
        CompiledLayerPlan& layer = model.graph.layer_plans[index];
        layer.layer_id = index;
        layer.vulkan_device_index = index == 0 ? 2 : 7;
        layer.attention.kind = index == 0 ? AttentionKind::GatedDeltaNet : AttentionKind::Standard;
        layer.attention.head_count = 2;
        layer.attention.kv_head_count = 1;
        layer.attention.head_dimension = 4;
        layer.attention.value_head_dimension = 4;
        layer.attention.convolution_kernel_size = 2;
        layer.attention.sliding_window = 4096;
    }
    for (uint32_t id = 0; id < 8; ++id)
    {
        ExecutionTensor tensor;
        tensor.id = id;
        tensor.dtype = DType::Float32;
        tensor.shape = {0, id == 7 ? model.descriptor.vocabulary_size : model.descriptor.hidden_size};
        model.graph.tensors.push_back(std::move(tensor));
    }
    append_model_memory_node(model, ExecutionNodeType::TokenEmbedding, ExecutionBackend::Cpu, invalid_execution_layer_id, {}, {0});
    append_model_memory_node(model, ExecutionNodeType::Attention, ExecutionBackend::Vulkan, 0, {0}, {1, 2});
    append_model_memory_node(model, ExecutionNodeType::FinalNorm, ExecutionBackend::Cpu, invalid_execution_layer_id, {1}, {3});
    append_model_memory_node(model, ExecutionNodeType::Attention, ExecutionBackend::Vulkan, 1, {3}, {4, 5});
    append_model_memory_node(model, ExecutionNodeType::FinalNorm, ExecutionBackend::Cpu, invalid_execution_layer_id, {4}, {6});
    append_model_memory_node(model, ExecutionNodeType::LmHead, ExecutionBackend::Vulkan, invalid_execution_layer_id, {6}, {7});
    return model;
}

static uint64_t total_model_peak(const GraphMemoryPlan& plan)
{
    uint64_t total = 0;
    for (const GraphMemoryDevice& device : plan.devices) total += device.peak_size;
    return total;
}

static void test_model_graph_lifetimes_and_shapes()
{
    CompiledModel model = model_memory_fixture();
    const auto planned = plan_model_graph_memory(model, 8, 1, 64);
    memory_check(static_cast<bool>(planned), "selected GDN/Attention graph memory planning failed");
    memory_check(!planned.value().prefill.allocations.empty() && !planned.value().decode.allocations.empty()
                     && planned.value().prefill.devices.size() == 2 && planned.value().session_state.size() == 2,
                 "selected graph backends must publish both execution phases and per-device Session state");
    const uint64_t hidden_bytes = UINT64_C(8) * model.descriptor.hidden_size * sizeof(float);
    const GraphMemoryAllocation* retained_output = nullptr;
    const GraphMemoryAllocation* destination_input = nullptr;
    for (const GraphMemoryAllocation& item : planned.value().prefill.allocations)
    {
        if (item.device_index == 7 && item.size == hidden_bytes && item.first_node == 3 && item.last_node == 4)
            retained_output = &item;
        if (item.device_index == 7 && item.size == hidden_bytes && item.first_node == 3 && item.last_node == 3)
            destination_input = &item;
    }
    memory_check(retained_output && destination_input,
                 "GPU outputs must live through CPU consumers and CPU inputs need destination-device copies");
    memory_check(retained_output->offset >= destination_input->offset + destination_input->size
                     || destination_input->offset >= retained_output->offset + retained_output->size,
                 "same-node uploaded inputs and produced outputs cannot share the destination range");
    memory_check(total_model_peak(planned.value().prefill) > total_model_peak(planned.value().decode),
                 "prefill rows must produce a larger working set than single-sequence decode");
    const auto larger_rows = plan_model_graph_memory(model, 16, 3, 64);
    memory_check(larger_rows && total_model_peak(larger_rows.value().prefill) > total_model_peak(planned.value().prefill)
                     && total_model_peak(larger_rows.value().decode) > total_model_peak(planned.value().decode),
                 "prefill and batched decode working sets must follow their actual row counts");
    const auto larger_context = plan_model_graph_memory(model, 8, 1, 1024);
    memory_check(larger_context && larger_context.value().session_state.back().peak_size > planned.value().session_state.back().peak_size,
                 "longer contexts must grow the selected Standard Attention Session state");

    CompiledModel cross_device = model;
    cross_device.graph.nodes[3].inputs = {1};
    cross_device.graph.tensors[1].consumers.push_back(3);
    cross_device.graph.tensors[3].consumers.clear();
    const auto transferred = plan_model_graph_memory(cross_device, 8, 1, 64);
    memory_check(static_cast<bool>(transferred), "direct cross-device graph memory planning failed");
    bool source_retained = false;
    bool destination_copied = false;
    for (const GraphMemoryAllocation& item : transferred.value().prefill.allocations)
    {
        source_retained = source_retained || (item.device_index == 2 && item.size == hidden_bytes && item.first_node == 1 && item.last_node == 3);
        destination_copied = destination_copied || (item.device_index == 7 && item.size == hidden_bytes && item.first_node == 3 && item.last_node == 3);
    }
    memory_check(source_retained && destination_copied,
                 "a cross-device edge needs separate source and destination instances through the transfer consumer");

    CompiledModel cpu = model;
    cpu.opt.hybrid_mode = HybridMode::CpuOnly;
    for (ExecutionNode& node : cpu.graph.nodes)
    {
        node.backend = ExecutionBackend::Cpu;
        node.backend_mask = ExecutionBackendCpu;
    }
    const auto host = plan_model_graph_memory(cpu, 8, 1, 64);
    memory_check(host && host.value().prefill.allocations.empty() && host.value().decode.allocations.empty()
                     && host.value().session_state.empty(),
                 "CPU-only graphs must not reserve GPU activation or Session state arenas");
    memory_check(!plan_model_graph_memory(model, 0, 1, 64) && !plan_model_graph_memory(model, 8, 0, 64),
                 "zero-sized execution variants must be rejected before planning their graph");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::test_closed_lifetimes_and_branches();
        ncnn::moe::test_submission_budget();
        ncnn::moe::test_submission_fences();
        ncnn::moe::test_alignment_and_reuse();
        ncnn::moe::test_devices_empty_buffers_and_validation();
        ncnn::moe::test_overflow();
        ncnn::moe::test_model_graph_lifetimes_and_shapes();
        std::cout << "Graph memory lifetime tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Graph memory lifetime test failed: " << error.what() << '\n';
        return 1;
    }
}
