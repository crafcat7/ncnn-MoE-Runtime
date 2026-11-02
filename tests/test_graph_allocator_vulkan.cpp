#include "backends/ncnn/graphallocator_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "engine/graphresources.h"
#include "graph/compiledmodel.h"
#include "kernels/activationbuffer.h"

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
    arena_check(size.preparation_physical_size <= size.total_physical_size,
                "graph arena preparation physical size must not exceed total size");
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
    larger.reset();
    for (uint64_t capacity : {larger_compute * 2, larger_compute * 4, larger_compute * 8})
    {
        arena_check(arena.reserve_graph_compute(capacity), "successive graph shape growth failed");
        arena_check(arena.physical_size().total_physical_size <= preparation_size + capacity + alignment,
                    "graph replanning accumulated obsolete fully free working sets");
        arena_check(live.buffer() == live_buffer && live.buffer_offset() == live_offset,
                    "reclaiming obsolete graph arenas relocated a live tensor");
        verify_content(context, arena, live, expected);
    }
    if (std::numeric_limits<size_t>::max() == std::numeric_limits<uint64_t>::max())
    {
        // Alignment overflow rejects this request after trimming, without
        // depending on the driver or exhausting the device's physical heap.
        arena_check(!arena.reserve_graph_compute(std::numeric_limits<uint64_t>::max()),
                    "an unrepresentable graph reservation unexpectedly succeeded");
        verify_content(context, arena, live, expected);
        arena_check(arena.reserve_graph_compute(initial_compute),
                    "a failed replan prevented a later smaller reservation");
        larger.reset(arena.fastMalloc(initial_compute));
        arena_check(larger != nullptr, "a failed replan left a stale advertised capacity");
    }
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

static void test_allocator_descriptor_retirement(uint32_t device_index)
{
    auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(device_index, runtime, OptimizationDefaultFlags);
    arena_check(static_cast<bool>(context), "descriptor retirement context creation failed");
    VulkanGraphAllocator arena(context->device());
    const size_t unit = std::max<size_t>(256, context->device()->info.buffer_offset_alignment());
    arena_check(arena.reserve_graph_compute(unit * 16), "descriptor retirement reservation failed");
    const uint64_t physical = arena.physical_size().total_physical_size;
    auto* first = arena.fastMalloc(unit);
    arena_check(first != nullptr, "descriptor retirement allocation failed");
    first->access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    first->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    arena.fastFree(first);
    auto* reused = arena.fastMalloc(unit * 2);
    arena_check(reused && reused->capacity >= unit * 2 && reused->access_flags == 0
                    && reused->stage_flags == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    && arena.physical_size().total_physical_size == physical,
                "range reuse retained stale synchronization state or grew the reserved backing");

    // A pre-clear handle must not free a replacement allocation's live range.
    arena.clear();
    arena_check(arena.reserve_graph_compute(unit * 16), "graph arena reservation after clear failed");
    auto* current = arena.fastMalloc(unit * 3);
    arena_check(current != nullptr, "graph arena allocation after clear failed");
    arena.fastFree(reused);
    auto* adjacent = arena.fastMalloc(unit);
    arena_check(adjacent && (adjacent->buffer != current->buffer || adjacent->offset >= current->offset + current->capacity || current->offset >= adjacent->offset + adjacent->capacity)
                    && arena.physical_size().total_physical_size == physical,
                "disposing a stale descriptor exposed a live range or exceeded the arena ceiling");
    arena.fastFree(adjacent);
    arena.fastFree(current);
}

static void write_host_output(const std::shared_ptr<VulkanContext>& context, VulkanGraphAllocator& arena,
                              ncnn::VkMat& destination, const ActivationBuffer& pending, float base)
{
    ncnn::Mat expected(static_cast<int>(pending.columns()), static_cast<int>(pending.rows()), sizeof(float));
    float* values = expected;
    for (size_t index = 0; index < pending.rows() * pending.columns(); ++index)
        values[index] = base + static_cast<float>(index) * 0.125f;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const void* mapped = pending.bytes().data();
    ncnn::VkCompute command(context->device());
    ncnn::VkMat device;
    const ncnn::Option option = graph_option(arena, *context);
    command.record_upload(expected, device, option);
    arena_check(!device.empty()
                    && record_prepared_activation_staging_download(device, pending.rows(), pending.columns(), destination,
                                                                   command, context->device(), option),
                "planned host output transfer recording failed");
    arena_check(destination.mapped_ptr() == mapped, "host output recording replaced the leased mapped allocation");
    arena_check(command.submit_and_wait() == 0 && VulkanHostOutput::complete(destination, pending),
                "planned host output transfer completion failed");
    for (size_t index = 0; index < pending.rows() * pending.columns(); ++index)
        arena_check(pending.values()[index] == values[index], "mapped host output lost a full-logits value");
}

static void test_host_output_leases(uint32_t device_index)
{
    auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(device_index, runtime, OptimizationDefaultFlags);
    arena_check(static_cast<bool>(context), "host output context creation failed");
    VulkanGraphAllocator arena(context->device());
    arena_check(arena.reserve_graph_compute(64 * 1024), "host output transfer reservation failed");
    const auto small_plan = plan_host_output_slot_size(2, 17);
    const auto large_plan = plan_host_output_slot_size(3, 33);
    arena_check(small_plan && large_plan, "host output fixture planning failed");
    auto owner = std::make_unique<VulkanHostOutput>();
    auto& pool = *owner;
    ActivationBuffer first, second;
    ncnn::VkMat first_device, second_device;
    const auto prepare = [&](VulkanHostOutput& target, size_t rows, uint32_t columns, uint64_t limit,
                             ncnn::VkMat& destination, ActivationBuffer& pending) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        return target.prepare(context, rows, columns, limit, destination, pending);
    };
    arena_check(prepare(pool, 2, 17, small_plan.value(), first_device, first)
                    && prepare(pool, 1, 19, small_plan.value(), second_device, second),
                "two simultaneously leased host output slots could not be prepared");
    arena_check(first.bytes().data() == first_device.mapped_ptr()
                    && second.bytes().data() == second_device.mapped_ptr()
                    && first.bytes().data() != second.bytes().data()
                    && pool.allocated_bytes() <= 2 * small_plan.value(),
                "host output slots overlap, copy through owned storage, or exceed their planned bound");
    write_host_output(context, arena, first_device, first, -3.0f);
    write_host_output(context, arena, second_device, second, 7.0f);
    const ActivationBuffer first_snapshot(first), second_snapshot(second);
    arena_check(!VulkanHostOutput::complete(first_device, second),
                "host output completion accepted a lease for a different mapped slot");

    ActivationBuffer rejected;
    rejected.reset(1, 1, true);
    rejected.row(0)[0] = 123.0f;
    const std::byte* rejected_pointer = rejected.bytes().data();
    ncnn::VkMat rejected_device = first_device;
    const uint64_t initial_bytes = pool.allocated_bytes();
    arena_check(!prepare(pool, 1, 1, small_plan.value(), rejected_device, rejected)
                    && rejected.bytes().data() == rejected_pointer && rejected.row(0)[0] == 123.0f
                    && rejected_device.data == first_device.data && pool.allocated_bytes() == initial_bytes,
                "exhausted host output preparation changed the previous destination or published output");
    rejected_device.release();

    const std::byte* first_pointer = first.bytes().data();
    first_device.release();
    first.clear();
    arena_check(prepare(pool, 1, 13, small_plan.value(), first_device, first)
                    && first.bytes().data() == first_pointer && pool.allocated_bytes() == initial_bytes,
                "released host output storage was not reused for a smaller logical shape");
    write_host_output(context, arena, first_device, first, 31.0f);
    arena_check(std::equal(second.values().begin(), second.values().end(), second_snapshot.values().begin())
                    && first_snapshot.row(0)[0] == -3.0f,
                "slot reuse changed another live lease or a deep-copy snapshot");

    first_device.release();
    first.clear();
    arena_check(!prepare(pool, 3, 33, small_plan.value(), rejected_device, rejected)
                    && rejected.row(0)[0] == 123.0f && rejected_device.empty()
                    && pool.allocated_bytes() == initial_bytes,
                "an unplanned larger host output allocated storage or changed its destination");
    arena_check(prepare(pool, 3, 33, large_plan.value(), first_device, first)
                    && pool.allocated_bytes() <= 2 * large_plan.value(),
                "a larger planned host output could not replace a free slot within its bound");
    write_host_output(context, arena, first_device, first, 43.0f);
    arena_check(std::equal(second.values().begin(), second.values().end(), second_snapshot.values().begin()),
                "growing a free host output slot invalidated a live smaller output");

    // The external owner, rather than the workspace or the command, keeps the
    // mapped allocation and its dedicated allocator alive after publication.
    first_device.release();
    second_device.release();
    owner.reset();
    arena_check(first.row(2)[32] == 43.0f + 98.0f * 0.125f
                    && std::equal(second.values().begin(), second.values().end(), second_snapshot.values().begin()),
                "destroying a workspace invalidated its already published full logits");
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
    const uint64_t initial_host_slot = model.graph_resources->plan.host_output_slot_size;
    const uint64_t initial_host_workspace = model.graph_resources->plan.host_output_workspace_size;
    arena_check(initial != 0 && model.graph_resources->plan.prefill_rows == 512,
                "model graph preparation did not physically reserve its planned shape");
    arena_check(static_cast<bool>(ensure_model_graph_resources(model, 1, 1024)), "long-context decode planning failed");
    arena_check(model.graph_resources->plan.prefill_rows == 1 && model.graph_resources->plan.context_tokens == 1024,
                "decode growth must not reserve the Cartesian product of previous shape maxima");
    arena_check(initial_host_slot != 0 && initial_host_workspace == 2 * initial_host_slot
                    && model.graph_resources->plan.host_output_slot_size == initial_host_slot
                    && model.graph_resources->plan.host_output_workspace_size == initial_host_workspace,
                "smaller-row context replanning forgot the capacity of retained host output slots");
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
            ncnn::moe::test_allocator_descriptor_retirement(device);
            ncnn::moe::test_host_output_leases(device);
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
