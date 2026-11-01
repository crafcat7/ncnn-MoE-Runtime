#include "backends/ncnn/graphallocator_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "engine/graphresources.h"
#include "graph/compiledmodel.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <option.h>
#endif

namespace ncnn {
namespace moe {

static void arena_check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
static void check_physical_size(const VulkanGraphArenaSize& size)
{
    arena_check(size.preparation_physical_size <= size.total_physical_size
                    && size.compute_physical_size == size.total_physical_size - size.preparation_physical_size,
                "graph arena physical preparation and compute sizes must sum to the total");
}

static ncnn::Option graph_option(VulkanGraphAllocator& arena, const VulkanContext& context)
{
    ncnn::Option option;
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = &arena;
    option.workspace_vkallocator = &arena;
    option.staging_vkallocator = context.staging_allocator();
    return option;
}

static void verify_content(const std::shared_ptr<VulkanContext>& context, VulkanGraphAllocator& arena,
                           const ncnn::VkMat& device, const ncnn::Mat& expected)
{
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute command(context->device());
    ncnn::Mat actual;
    command.record_download(device, actual, graph_option(arena, *context));
    arena_check(command.submit_and_wait() == 0 && !actual.empty() && actual.total() == expected.total(),
                "graph arena payload download failed");
    const float* reference = expected;
    const float* values = actual;
    for (size_t index = 0; index < expected.total(); ++index)
        arena_check(values[index] == reference[index], "graph arena growth or reuse invalidated a live GPU payload");
}

static void test_reservation_reuse_and_growth(uint32_t device_index)
{
    auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(device_index, runtime, OptimizationDefaultFlags);
    arena_check(static_cast<bool>(context), "graph arena context creation failed");
    VulkanGraphAllocator arena(context->device());
    const auto release = [&](ncnn::VkBufferMemory* buffer) { arena.fastFree(buffer); };
    std::unique_ptr<ncnn::VkBufferMemory, decltype(release)> preparation(arena.fastMalloc(4096), release);
    arena_check(preparation && preparation->buffer != VK_NULL_HANDLE && preparation->memory != VK_NULL_HANDLE,
                "graph arena did not physically allocate preparation storage");
    const uint64_t preparation_size = arena.physical_size().total_physical_size;
    constexpr uint64_t initial_compute = 64 * 1024;
    arena_check(arena.reserve_graph_compute(initial_compute), "initial graph compute reservation failed");
    const VulkanGraphArenaSize reserved = arena.physical_size();
    check_physical_size(reserved);
    arena_check(reserved.preparation_physical_size == preparation_size && reserved.total_physical_size >= preparation_size,
                "first graph reservation did not snapshot persistent preparation storage");
    std::unique_ptr<ncnn::VkBufferMemory, decltype(release)> first(arena.fastMalloc(initial_compute), release);
    arena_check(first != nullptr, "reserved graph compute range is not physically available");
    const VkBuffer reused_buffer = first->buffer;
    const size_t reused_offset = first->offset;
    first.reset();
    first.reset(arena.fastMalloc(initial_compute));
    arena_check(first && first->buffer == reused_buffer && first->offset == reused_offset,
                "sequential graph execution did not reuse its physical buffer range");
    arena_check(arena.physical_size().total_physical_size == reserved.total_physical_size,
                "ordinary graph allocation grew physical backing after reservation");

    const size_t alignment = std::max<size_t>(4, context->device()->info.buffer_offset_alignment());
    arena_check(reserved.total_physical_size <= std::numeric_limits<size_t>::max() - alignment,
                "graph arena exhaustion fixture cannot represent its allocation size");
    std::unique_ptr<ncnn::VkBufferMemory, decltype(release)> rejected(arena.fastMalloc(static_cast<size_t>(reserved.total_physical_size) + alignment), release);
    arena_check(!rejected && arena.physical_size().total_physical_size == reserved.total_physical_size,
                "a sealed graph arena exceeded its physically reserved ceiling");
    arena_check(arena.reserve_graph_compute(initial_compute / 2)
                    && arena.reserve_graph_compute(initial_compute)
                    && arena.physical_size().total_physical_size == reserved.total_physical_size,
                "repeated graph reservations allocated a second working set");
    first.reset();

    ncnn::Mat expected(128, static_cast<size_t>(sizeof(float)));
    float* values = expected;
    for (int index = 0; index < expected.w; ++index)
        values[index] = static_cast<float>(index % 19 - 9) * 0.25f;
    ncnn::VkMat live;
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        command.record_upload(expected, live, graph_option(arena, *context));
        arena_check(command.submit_and_wait() == 0 && !live.empty(), "graph arena payload upload failed");
    }
    verify_content(context, arena, live, expected);
    const VkBuffer live_buffer = live.buffer();
    const size_t live_offset = live.buffer_offset();
    constexpr uint64_t larger_compute = initial_compute * 4;
    arena_check(arena.reserve_graph_compute(larger_compute), "a larger graph working set could not grow its arena between executions");
    const VulkanGraphArenaSize grown = arena.physical_size();
    check_physical_size(grown);
    arena_check(grown.total_physical_size > reserved.total_physical_size
                    && grown.preparation_physical_size == reserved.preparation_physical_size,
                "graph arena growth changed preparation ownership or failed to add compute backing");
    arena_check(live.buffer() == live_buffer && live.buffer_offset() == live_offset,
                "graph arena growth relocated an existing live tensor");
    verify_content(context, arena, live, expected);
    std::unique_ptr<ncnn::VkBufferMemory, decltype(release)> larger(arena.fastMalloc(larger_compute), release);
    arena_check(larger != nullptr && arena.physical_size().total_physical_size == grown.total_physical_size,
                "larger graph reservation did not back the required contiguous allocation");
    live.release();
}

static void test_cross_thread_allocation(uint32_t device_index)
{
    auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(device_index, runtime, OptimizationDefaultFlags);
    arena_check(static_cast<bool>(context), "cross-thread graph arena context creation failed");
    VulkanGraphAllocator arena(context->device());
    arena_check(arena.reserve_graph_compute(64 * 1024), "cross-thread graph arena reservation failed");
    const uint64_t physical_size = arena.physical_size().total_physical_size;
    auto* handoff = arena.fastMalloc(1024);
    arena_check(handoff != nullptr, "cross-thread graph arena handoff allocation failed");
    const VkBuffer handoff_buffer = handoff->buffer;
    const size_t handoff_offset = handoff->offset;
    std::thread receiver([&] { arena.fastFree(handoff); });
    receiver.join();
    auto* recovered = arena.fastMalloc(1024);
    const bool reused = recovered && recovered->buffer == handoff_buffer && recovered->offset == handoff_offset;
    if (recovered) arena.fastFree(recovered);
    arena_check(reused, "a buffer freed by another thread was not reusable by the caller");

    constexpr size_t worker_count = 2;
    std::array<ncnn::VkBufferMemory*, worker_count> buffers{};
    std::barrier phase(static_cast<std::ptrdiff_t>(worker_count + 1));
    std::vector<std::thread> workers;
    for (size_t index = 0; index < worker_count; ++index)
    {
        workers.emplace_back([&, index] {
            buffers[index] = arena.fastMalloc(1024);
            phase.arrive_and_wait();
            phase.arrive_and_wait();
            if (buffers[index]) arena.fastFree(buffers[index]);
        });
    }
    phase.arrive_and_wait();
    bool valid = true;
    for (size_t index = 0; index < worker_count; ++index)
    {
        valid = valid && buffers[index] != nullptr;
        if (!buffers[index]) continue;
        for (size_t previous = 0; previous < index; ++previous)
        {
            if (!buffers[previous] || buffers[index]->buffer != buffers[previous]->buffer) continue;
            valid = valid && (buffers[index]->offset >= buffers[previous]->offset + buffers[previous]->capacity || buffers[previous]->offset >= buffers[index]->offset + buffers[index]->capacity);
        }
    }
    phase.arrive_and_wait();
    for (std::thread& worker : workers) worker.join();
    arena_check(valid, "concurrent graph allocations failed or returned overlapping live ranges");
    arena_check(arena.physical_size().total_physical_size == physical_size,
                "cross-thread graph execution escaped its shared physical ceiling");
}

static void test_model_shape_growth(uint32_t device_index)
{
    CompiledModel model;
    model.vulkan_runtime = create_vulkan_runtime();
    model.opt.hybrid_mode = HybridMode::HybridExperts;
    model.opt.vulkan_device_index = device_index;
    model.descriptor.hidden_size = 64;
    model.descriptor.vocabulary_size = 1024;
    model.descriptor.layers.resize(1);
    model.graph.layer_plans.resize(1);
    CompiledLayerPlan& layer = model.graph.layer_plans.front();
    layer.vulkan_device_index = device_index;
    layer.attention.kind = AttentionKind::Standard;
    layer.attention.head_count = 1;
    layer.attention.kv_head_count = 1;
    layer.attention.head_dimension = 64;
    layer.attention.value_head_dimension = 64;
    layer.attention.sliding_window = 4096;
    model.memory_plan.gpu_devices.push_back(make_gpu_resource_plan({device_index, 0, 0}, 1));
    for (uint32_t index = 0; index < 4; ++index)
    {
        ExecutionTensor tensor;
        tensor.id = index;
        tensor.shape = {0, index == 3 ? model.descriptor.vocabulary_size : model.descriptor.hidden_size};
        tensor.producer = index;
        if (index != 3) tensor.consumers.push_back(index + 1);
        model.graph.tensors.push_back(std::move(tensor));
        ExecutionNode node;
        node.id = index;
        node.type = std::array{ExecutionNodeType::TokenEmbedding, ExecutionNodeType::Attention,
                               ExecutionNodeType::FinalNorm, ExecutionNodeType::LmHead}[index];
        node.backend = index % 2 ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
        node.backend_mask = index % 2 ? ExecutionBackendVulkan : ExecutionBackendCpu;
        if (index == 1) node.layer_plan_index = 0;
        if (index != 0)
        {
            node.inputs.push_back(index - 1);
            node.dependencies.push_back(index - 1);
        }
        node.outputs.push_back(index);
        model.graph.nodes.push_back(std::move(node));
        model.schedule.node_order.push_back(index);
    }
    arena_check(static_cast<bool>(prepare_model_graph_resources(model, 512)), "model graph prefill reservation failed");
    const auto physical_size = [&] {
        return query_vulkan_graph_compute(model.vulkan_runtime, device_index, model.opt.optimization_flags).total_physical_size;
    };
    const uint64_t initial = physical_size();
    arena_check(initial != 0 && model.graph_resources->plan.prefill_rows == 512,
                "model graph preparation did not physically reserve its planned shape");
    arena_check(static_cast<bool>(ensure_model_graph_resources(model, 1, 1024)), "long-context decode planning failed");
    arena_check(model.graph_resources->plan.prefill_rows == 1 && model.graph_resources->plan.context_tokens == 1024,
                "decode growth must not reserve the Cartesian product of previous shape maxima");
    arena_check(physical_size() == initial, "long-context decode failed to reuse the larger prefill allocation");
    arena_check(static_cast<bool>(ensure_model_graph_resources(model, 512, 1024)), "later long-context prefill planning failed");
    arena_check(model.graph_resources->plan.prefill_rows == 512 && model.graph_resources->plan.context_tokens == 1024,
                "a later prefill must replan its own requested shape");
    const uint64_t expanded = physical_size();
    arena_check(expanded >= initial, "graph shape growth discarded its existing allocation");
    arena_check(static_cast<bool>(ensure_model_graph_resources(model, 512, 1024)) && physical_size() == expanded,
                "repeated execution of the planned shape allocated another working set");
}
#endif

} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0)
    {
        std::cout << "SKIP: Vulkan GPU unavailable\n";
        return 77;
    }
    try
    {
        for (uint32_t device = 0; device < ncnn::moe::get_gpu_count(); ++device)
        {
            ncnn::moe::test_reservation_reuse_and_growth(device);
            ncnn::moe::test_cross_thread_allocation(device);
            ncnn::moe::test_model_shape_growth(device);
        }
        std::cout << "Vulkan graph arena allocation, reuse, growth and cross-thread tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan graph arena test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
