#include "linear.h"
#include "sharedffn_vulkan.h"
#include "mxfp4_weightpacking.h"
#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "experttransfer_vulkan.h"
#include "expertbackend_vulkan.h"
#include "innerproductweights_vulkan.h"
#include "moecombine_vulkan.h"
#include "residentffn_vulkan.h"

#include "ncnn/moe/runtime.h"
#include "engine/expertbackend.h"
#include "kernels/float8.h"
#include "kernels/int4.h"
#include "kernels/mxfp4.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"

#if NCNN_MOE_WITH_VULKAN
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#endif

#include "linear_impl.h"
#include "linear_internal.h"

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
bool VulkanExpertBackend::forward_batch(std::span<const ExpertRequest> requests,
                                        std::span<const Selection> selected)
{
    if (selected.empty())
        return true;
    if (selected.front().request_index >= requests.size()
        || !selected.front().entry || !selected.front().entry->operation)
        return false;
    const Mxfp4Expert_vulkan::Implementation& first_expert = *selected.front().entry->operation->d;
    if (!first_expert.gate_up || !first_expert.down || !first_expert.gate_up_pipeline)
    {
        return false;
    }
    const Mxfp4Linear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& first_down = *first_expert.down->d;
    if (!first_gate.vulkan_context || first_gate.vulkan_context != first_down.vulkan_context)
    {
        return false;
    }

    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;
    const auto region = requests[selected.front().request_index].region;
    if (region)
    {
        if (region->columns() != input_columns || region->rows() == 0
            || activation_variant(first_gate.option) != 0)
            return false;
        if (!resident_combiner) resident_combiner = MoeCombine_vulkan::create(first_gate.vulkan_context);
        if (!region->bind_context(first_gate.vulkan_context, resident_combiner)) return false;
    }
    bool use_indexed = has_flag(optimization_flags, OptimizationVulkanIndexedExperts)
                       && selected.size() >= 4
                       && selected.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    if (use_indexed
        && (input_columns == 0 || first_gate.output_columns / 2 == 0 || output_columns == 0
            || !first_gate.indexed_storage || !first_down.indexed_storage
            || first_gate.storage.empty() || first_down.storage.empty()))
    {
        use_indexed = false;
    }

    VulkanRuntimeState& runtime_state = first_gate.vulkan_context->runtime_state();
    bool retain_output = false;
    bool download_output = false;
    size_t total_rows = 0;
    size_t maximum_request_rows = 0;
    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size()
            || !selection.entry
            || !selection.entry->operation)
        {
            return false;
        }
        const ExpertRequest& request = requests[selection.request_index];
        const auto& expert = *selection.entry->operation->d;
        if (!expert.gate_up || !expert.down)
            return false;
        const auto& gate = *expert.gate_up->d;
        const auto& down = *expert.down->d;
        if (!request.output || request.rows() == 0 || request.columns() != input_columns
            || request.region != region || (region && !request.device_output)
            || gate.vulkan_context != first_gate.vulkan_context || down.vulkan_context != first_gate.vulkan_context
            || gate.input_columns != input_columns
            || gate.output_columns % 2 != 0 || down.input_columns != gate.output_columns / 2
            || down.output_columns != output_columns
            || request.rows() > static_cast<size_t>(std::numeric_limits<int>::max()) - total_rows)
        {
            return false;
        }

        if (use_indexed)
        {
            if (expert.activation != first_expert.activation
                || expert.activation_limit != first_expert.activation_limit
                || !gate.indexed_storage || !down.indexed_storage
                || gate.storage.empty() || down.storage.empty()
                || gate.output_columns != first_gate.output_columns
                || gate.block_count != first_gate.block_count
                || down.block_count != first_down.block_count
                || gate.indexed_scales_word_offset != first_gate.indexed_scales_word_offset
                || gate.indexed_bias_word_offset != first_gate.indexed_bias_word_offset
                || down.indexed_scales_word_offset != first_down.indexed_scales_word_offset
                || down.indexed_bias_word_offset != first_down.indexed_bias_word_offset)
            {
                use_indexed = false;
            }
            else
            {
                maximum_request_rows = std::max(maximum_request_rows, request.rows());
            }
        }
        retain_output = retain_output || request.device_output != nullptr;
        download_output = download_output || request.device_output == nullptr;
        total_rows += request.rows();
    }
    bool use_tiled_indexed = false;
    size_t indexed_chunk_count = 0;
    if (use_indexed)
    {
        // Tile only when the largest selected Expert has row reuse.
        use_tiled_indexed = maximum_request_rows >= 2;
        indexed_chunk_count = (selected.size() + indexed_max_experts - 1)
                              / indexed_max_experts;
        // The indexed shader has a generic slot switch, while the regular
        // path uses a specialized projection pipeline. Keep the dynamic
        // capability model-neutral, but only select it when the dispatch
        // reduction is large enough to amortize that selector overhead.
        // This is intentionally an internal policy (not a model option):
        // top-k=10 decode uses the regular path, whereas top-k=16/24 can
        // benefit from 8+8/8+8+8 chunking.
        const uint64_t regular_dispatches = static_cast<uint64_t>(selected.size()) * 2;
        const uint64_t indexed_dispatches = static_cast<uint64_t>(indexed_chunk_count) * 2;
        const uint64_t selector_overhead_factor = use_tiled_indexed ? 4 : 6;
        if (indexed_dispatches == 0
            || regular_dispatches <= indexed_dispatches * selector_overhead_factor)
        {
            use_indexed = false;
            use_tiled_indexed = false;
            indexed_chunk_count = 0;
        }
    }

    if (use_indexed
        && !indexed_pipeline
        && !create_indexed_pipeline(first_gate.vulkan_context,
                                    first_gate.option,
                                    indexed_pipeline))
    {
        use_indexed = false;
        use_tiled_indexed = false;
        indexed_chunk_count = 0;
    }

    if (use_indexed && indexed_pipeline)
    {
        const uint64_t indexed_local_size_x = indexed_pipeline->local_size_x();
        const uint64_t max_dispatch_width = static_cast<uint64_t>(std::numeric_limits<int>::max());
        if (indexed_local_size_x == 0 || indexed_local_size_x > max_dispatch_width
            || static_cast<uint64_t>(first_gate.output_columns / 2)
                   > (max_dispatch_width - (indexed_local_size_x - 1)) / indexed_local_size_x
            || static_cast<uint64_t>(output_columns)
                   > (max_dispatch_width - (indexed_local_size_x - 1)) / indexed_local_size_x)
        {
            use_indexed = false;
            use_tiled_indexed = false;
            indexed_chunk_count = 0;
        }
    }

    std::vector<uint32_t> expert_row_offsets;
    std::vector<uint32_t> row_slots;
    if (use_indexed)
    {
        expert_row_offsets.resize(selected.size() + 1);
        row_slots.resize(total_rows);
    }

    ActivationBuffer* aggregated_output = nullptr;
    uint32_t aggregated_token_count = 0;
    const bool use_route_aggregation = !retain_output && route_aggregation_enabled(requests, selected, output_columns, aggregated_output, aggregated_token_count, optimization_flags);
    std::vector<uint32_t> route_offsets;
    std::vector<uint32_t> route_rows;
    std::vector<float> route_weights;
    if (use_route_aggregation
        && !build_ra_meta(requests,
                          selected,
                          route_offsets,
                          route_rows,
                          route_weights))
    {
        return false;
    }

    const std::shared_ptr<const DeviceTensor_vulkan> retained_input = requests[selected.front().request_index].device_input;
    const ncnn::VkMat* retained_value = retained_input ? retained_input->value() : nullptr;
    const uint32_t input_pack = region ? 1 : retained_value ? static_cast<uint32_t>(retained_value->elempack)
                                                            : 0;
    const uint64_t input_rows_available = region ? region->rows()
                                          : retained_value && retained_value->h > 0
                                              ? static_cast<uint64_t>(retained_value->h) * input_pack
                                              : 0;
    bool use_device_input = region || (retained_value && retained_input->vulkan_context() == first_gate.vulkan_context && retained_value->dims == 2 && retained_value->w == static_cast<int>(input_columns) && (input_pack == 1 || input_pack == 4) && retained_value->elemsize == sizeof(float) * input_pack && retained_input->columns() == input_columns && retained_input->rows() == input_rows_available);
    std::vector<uint32_t> input_rows;
    if (use_device_input)
    {
        input_rows.reserve(total_rows);
        for (const Selection& selection : selected)
        {
            const ExpertRequest& request = requests[selection.request_index];
            if ((!region && request.device_input != retained_input) || request.device_routes.size() != request.rows())
            {
                use_device_input = false;
                break;
            }
            for (const ExpertRoute& route : request.device_routes)
            {
                if (route.token_index >= input_rows_available)
                {
                    use_device_input = false;
                    break;
                }
                input_rows.push_back(route.token_index);
            }
            if (!use_device_input)
                break;
        }
    }
    const bool shared_single_row = use_device_input && !use_indexed
                                   && can_share_row(input_rows_available, selected.size(), input_rows);
    if (use_device_input && !shared_single_row && !input_gather_pipeline)
    {
        const std::lock_guard<std::mutex> lock(first_gate.vulkan_context->command_mutex());
        use_device_input = create_gather_pipeline(first_gate.vulkan_context, first_gate.option, input_gather_pipeline);
    }

    if (region && !use_device_input) return false;

    // A complete Host operand may supply this GPU upload when no valid
    // device operand was supplied. Device-only and region requests fail.
    if (!use_device_input)
        for (const Selection& selection : selected)
            if (!requests[selection.request_index].has_host_input()) return false;

    VulkanTransferLease transfer_lease = first_gate.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    ncnn::VkMat input_rows_staging;
    if (use_device_input && !shared_single_row)
    {
        if (!fill_staging_values(input_rows.data(), input_rows.size(), sizeof(uint32_t), input_rows_staging,
                                 transfer_slot.staging_allocator))
            return false;
    }
    if (!use_device_input && !prepare_staging_batch(transfer_slot.upload, total_rows, input_columns, transfer_slot.staging_allocator, sizeof(float)))
    {
        return false;
    }
    ncnn::Mat mapped_input;
    if (!use_device_input)
        mapped_input = transfer_slot.upload.mapped();
    if (!use_device_input && (mapped_input.empty() || mapped_input.dims != 2 || mapped_input.w != static_cast<int>(input_columns) || mapped_input.h != static_cast<int>(total_rows) || mapped_input.elemsize != sizeof(float) || mapped_input.elempack != 1))
    {
        return false;
    }

    size_t row_offset = 0;
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    for (size_t slot = 0; slot < selected.size(); ++slot)
    {
        const ExpertRequest& request = requests[selected[slot].request_index];
        if (use_indexed)
            expert_row_offsets[slot] = static_cast<uint32_t>(row_offset);
        if (!use_device_input)
        {
            const ActivationBuffer& input = *request.input;
            std::memcpy(mapped_input_data + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                        input.bytes().data(), input.bytes().size());
        }
        if (use_indexed)
        {
            std::fill(row_slots.begin() + static_cast<std::ptrdiff_t>(row_offset),
                      row_slots.begin() + static_cast<std::ptrdiff_t>(row_offset + request.rows()),
                      static_cast<uint32_t>(slot));
        }
        row_offset += request.rows();
    }
    if (use_indexed)
        expert_row_offsets[selected.size()] = static_cast<uint32_t>(row_offset);
    if (!use_device_input)
    {
        transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
        transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }

    const size_t descriptor_alignment = std::max<size_t>(4, first_gate.vulkan_context->device()->info.buffer_offset_alignment());
    const bool align_output_rows = selected.size() > 1
                                   && (static_cast<size_t>(output_columns) * sizeof(float)) % descriptor_alignment != 0;
    const bool direct_host_input = !use_device_input && first_gate.vulkan_context->support_direct_host_buffer(total_rows * input_columns * sizeof(float), DType::Float32);
    const bool direct_host_output = download_output && !retain_output && !use_route_aggregation && !align_output_rows
                                    && first_gate.vulkan_context->support_direct_host_buffer(total_rows * output_columns * sizeof(float),
                                                                                             DType::Float32);
    const size_t download_rows = use_route_aggregation ? aggregated_token_count : total_rows;
    if ((download_output || use_route_aggregation) && !prepare_staging_batch(transfer_slot.download, download_rows, output_columns, transfer_slot.staging_allocator))
        return false;

    if (use_indexed)
    {
        if (!fill_staging_values(row_slots.data(),
                                 row_slots.size(),
                                 sizeof(uint32_t),
                                 transfer_slot.expert_slots,
                                 transfer_slot.staging_allocator))
            return false;
        if (use_tiled_indexed
            && !fill_staging_values(expert_row_offsets.data(),
                                    expert_row_offsets.size(),
                                    sizeof(uint32_t),
                                    transfer_slot.expert_row_offsets,
                                    transfer_slot.staging_allocator))
            return false;
    }

    if (use_route_aggregation)
    {
        if (!fill_staging_values(route_offsets.data(),
                                 route_offsets.size(),
                                 sizeof(uint32_t),
                                 transfer_slot.route_offsets,
                                 transfer_slot.staging_allocator)
            || !fill_staging_values(route_rows.data(),
                                    route_rows.size(),
                                    sizeof(uint32_t),
                                    transfer_slot.route_rows,
                                    transfer_slot.staging_allocator)
            || !fill_staging_values(route_weights.data(),
                                    route_weights.size(),
                                    sizeof(float),
                                    transfer_slot.route_weights,
                                    transfer_slot.staging_allocator))
            return false;
    }

    const bool use_direct_output = download_output && !use_route_aggregation && selected.size() == 1;
    ActivationBuffer combined_output;
    // A single request already owns a private output until Submission::commit().
    ActivationBuffer& host_output = use_direct_output ? *requests[selected.front().request_index].output : combined_output;
    if (download_output && !use_route_aggregation && !use_direct_output)
        host_output.reset(total_rows, output_columns, false);

    std::unique_lock<std::mutex> lock(first_gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    ncnn::VkMat input_rows_gpu;
    ncnn::VkMat region_input;
    ncnn::VkMat output_gpu;
    ncnn::VkMat expert_ids_gpu;
    ncnn::VkMat expert_row_offsets_gpu;
    ncnn::VkMat intermediate_gpu;
    std::vector<ncnn::VkMat> intermediates;
    std::vector<ncnn::VkMat> aligned_inputs;
    std::vector<ncnn::VkMat> aligned_outputs(align_output_rows ? selected.size() : 0);
    std::vector<ResidentFfnRegion_vulkan::Output> region_outputs;
    bool region_uploaded = false;
    bool region_downloaded = false;
    bool successful = false;
    // Abort recording before operands or the region's retained outputs can
    // be released. The existing wave submission owns all GPU work.
    const auto cleanup = [&](void*) {
        if (!successful)
        {
            (void)command.reset();
            if (region) region->abort_wave();
        }
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);
    if (use_device_input)
    {
        if (region)
        {
            if (!region->record_input(region_input, command, first_gate.option, region_uploaded)) return false;
            retained_value = &region_input;
        }
        else
            retained_input->prepare_read();
        if (shared_single_row)
        {
            input_gpu = aligned_row_view(*retained_value, 0, 1, command, first_gate.option,
                                         first_gate.vulkan_context, aligned_inputs, selected.size());
            if (input_gpu.empty()) return false;
        }
        else
        {
            if (!record_mapped_upload(input_rows_staging, input_rows_gpu, command, first_gate.option))
                return false;
            input_gpu.create(static_cast<int>(input_columns), static_cast<int>(total_rows), sizeof(float), first_gate.vulkan_context->blob_allocator());
            if (input_gpu.empty())
                return false;
            std::vector<ncnn::vk_constant_type> constants(3);
            constants[0].u32 = input_columns;
            constants[1].u32 = static_cast<uint32_t>(total_rows);
            constants[2].u32 = input_pack;
            ncnn::VkMat dispatcher;
            dispatcher.w = static_cast<int>(input_columns);
            dispatcher.h = static_cast<int>(std::min(total_rows, size_t(65535)));
            dispatcher.c = static_cast<int>((total_rows + 65534) / 65535);
            command.record_pipeline_readonly(input_gather_pipeline.get(), {*retained_value, input_rows_gpu, input_gpu}, {1, 1, 0}, constants, dispatcher);
        }
    }
    else if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first_gate.option))
    {
        return false;
    }

    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(output_columns),
                          static_cast<int>(total_rows),
                          sizeof(float),
                          first_gate.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    if (use_indexed)
    {
        if (!record_mapped_upload(transfer_slot.expert_slots,
                                  expert_ids_gpu,
                                  command,
                                  first_gate.option))
        {
            return false;
        }
        if (use_tiled_indexed
            && !record_mapped_upload(transfer_slot.expert_row_offsets,
                                     expert_row_offsets_gpu,
                                     command,
                                     first_gate.option))
        {
            return false;
        }

        intermediate_gpu.create(static_cast<int>(first_gate.output_columns / 2),
                                static_cast<int>(total_rows),
                                sizeof(float),
                                first_gate.vulkan_context->blob_allocator());
        if (intermediate_gpu.empty())
            return false;

        std::vector<ncnn::VkMat> bindings(indexed_max_experts + 3);
        std::vector<unsigned char> readonly_bindings(bindings.size(), 1);
        readonly_bindings.back() = 0;
        std::vector<ncnn::vk_constant_type> constants(20);
        constants[0].u32 = first_gate.input_columns;
        constants[1].u32 = first_gate.output_columns;
        constants[2].u32 = first_down.input_columns;
        constants[3].u32 = first_down.output_columns;
        constants[4].u32 = first_gate.block_count;
        constants[5].u32 = first_down.block_count;
        constants[6].u32 = static_cast<uint32_t>(total_rows);
        constants[7].u32 = 0;
        constants[8].u32 = 0;
        constants[9].u32 = first_gate.indexed_scales_word_offset;
        constants[10].u32 = first_gate.indexed_bias_word_offset;
        constants[11].u32 = 0;
        constants[12].u32 = first_down.indexed_scales_word_offset;
        constants[13].u32 = first_down.indexed_bias_word_offset;
        constants[14].u32 = 0;
        constants[15].u32 = first_expert.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
        constants[16].f = first_expert.activation_limit;
        constants[17].u32 = 0;
        constants[18].u32 = 0;
        constants[19].u32 = 0;

        for (size_t chunk_begin = 0;
             chunk_begin < selected.size();
             chunk_begin += indexed_max_experts)
        {
            const size_t chunk_end = std::min(selected.size(),
                                              chunk_begin + static_cast<size_t>(indexed_max_experts));
            const uint32_t chunk_count = static_cast<uint32_t>(chunk_end - chunk_begin);
            const uint32_t row_base = expert_row_offsets[chunk_begin];
            const uint32_t row_end = expert_row_offsets[chunk_end];
            const uint32_t chunk_rows = row_end - row_base;
            uint32_t chunk_maximum_request_rows = 0;
            for (size_t slot = chunk_begin; slot < chunk_end; ++slot)
            {
                const size_t request_index = selected[slot].request_index;
                chunk_maximum_request_rows = std::max(chunk_maximum_request_rows,
                                                      static_cast<uint32_t>(requests[request_index].rows()));
            }
            const uint32_t chunk_tile_mode = use_tiled_indexed && chunk_maximum_request_rows >= 2 ? 1u : 0u;

            bindings[0] = input_gpu;
            for (uint32_t slot = 0; slot < indexed_max_experts; ++slot)
            {
                if (slot < chunk_count)
                {
                    bindings[slot + 1] = selected[chunk_begin + slot].entry->operation->d->gate_up->d->storage;
                }
                else
                {
                    bindings[slot + 1].release();
                }
            }
            bindings[indexed_max_experts + 1] = chunk_tile_mode != 0 ? expert_row_offsets_gpu : expert_ids_gpu;
            bindings.back() = intermediate_gpu;
            constants[7].u32 = chunk_count;
            constants[14].u32 = 0;
            constants[17].u32 = chunk_tile_mode;
            constants[18].u32 = row_base;
            constants[19].u32 = static_cast<uint32_t>(chunk_begin);
            ncnn::VkMat gate_dispatcher;
            gate_dispatcher.w = static_cast<int>(static_cast<uint64_t>(first_gate.output_columns / 2)
                                                 * indexed_pipeline->local_size_x());
            gate_dispatcher.h = static_cast<int>(chunk_tile_mode != 0 ? (chunk_maximum_request_rows + 1) / 2 : chunk_rows);
            gate_dispatcher.c = static_cast<int>(chunk_tile_mode != 0 ? chunk_count : 1);
            command.record_pipeline_readonly(indexed_pipeline.get(),
                                             bindings,
                                             readonly_bindings,
                                             constants,
                                             gate_dispatcher);

            bindings[0] = intermediate_gpu;
            for (uint32_t slot = 0; slot < indexed_max_experts; ++slot)
            {
                if (slot < chunk_count)
                {
                    bindings[slot + 1] = selected[chunk_begin + slot].entry->operation->d->down->d->storage;
                }
                else
                {
                    bindings[slot + 1].release();
                }
            }
            bindings.back() = output_gpu;
            constants[14].u32 = 1;
            ncnn::VkMat down_dispatcher;
            down_dispatcher.w = static_cast<int>(static_cast<uint64_t>(output_columns)
                                                 * indexed_pipeline->local_size_x());
            down_dispatcher.h = static_cast<int>(chunk_tile_mode != 0 ? (chunk_maximum_request_rows + 1) / 2 : chunk_rows);
            down_dispatcher.c = static_cast<int>(chunk_tile_mode != 0 ? chunk_count : 1);
            command.record_pipeline_readonly(indexed_pipeline.get(),
                                             bindings,
                                             readonly_bindings,
                                             constants,
                                             down_dispatcher);
        }
    }
    else
    {
        intermediates.reserve(selected.size());
        std::vector<ncnn::VkMat> bindings(5);
        std::vector<ncnn::vk_constant_type> constants(7);
        row_offset = 0;
        for (size_t selected_slot = 0; selected_slot < selected.size(); ++selected_slot)
        {
            const Selection& selection = selected[selected_slot];
            const ExpertRequest& request = requests[selection.request_index];
            const auto& expert = *selection.entry->operation->d;
            const auto& gate = *expert.gate_up->d;
            const uint32_t intermediate_columns = gate.output_columns / 2;
            ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_single_row ? 0 : row_offset, request.rows(), command, first_gate.option,
                                                      first_gate.vulkan_context, aligned_inputs, selected.size());
            ncnn::VkMat output_view = row_view(output_gpu, row_offset, request.rows());
            if (input_view.empty() || output_view.empty())
            {
                return false;
            }
            if (align_output_rows)
            {
                // Every output in this batch uses the same transfer-write
                // assembly path, preserving shared buffer access tracking.
                aligned_outputs[selected_slot].create(output_view.w, output_view.h, sizeof(float),
                                                      first_gate.vulkan_context->blob_allocator());
                output_view = aligned_outputs[selected_slot];
                if (output_view.empty()) return false;
            }
            ncnn::VkMat& intermediate = intermediates.emplace_back();
            intermediate.create(static_cast<int>(intermediate_columns),
                                static_cast<int>(request.rows()),
                                sizeof(float),
                                first_gate.vulkan_context->blob_allocator());
            if (intermediate.empty())
                return false;

            selection.entry->operation->record(input_view, intermediate, output_view, command, bindings, constants);
            if (align_output_rows)
            {
                ncnn::VkMat destination = row_view(output_gpu, row_offset, request.rows());
                if (!record_row_copy(output_view, destination, command, first_gate.option)) return false;
            }
            row_offset += request.rows();
        }
    }

    if (align_output_rows && retain_output && use_indexed)
    {
        row_offset = 0;
        for (size_t selected_slot = 0; selected_slot < selected.size(); ++selected_slot)
        {
            const auto& request = requests[selected[selected_slot].request_index];
            if (request.device_output)
            {
                ncnn::VkMat source = row_view(output_gpu, row_offset, request.rows());
                command.record_clone(source, aligned_outputs[selected_slot], first_gate.option);
                if (aligned_outputs[selected_slot].empty()) return false;
            }
            row_offset += request.rows();
        }
    }

    if (region)
    {
        region_outputs.reserve(selected.size());
        row_offset = 0;
        for (size_t selected_slot = 0; selected_slot < selected.size(); ++selected_slot)
        {
            const auto& request = requests[selected[selected_slot].request_index];
            const ncnn::VkMat value = align_output_rows ? aligned_outputs[selected_slot]
                                                        : row_view(output_gpu, row_offset, request.rows());
            if (value.empty()) return false;
            region_outputs.push_back({request.region_index, value, request.device_routes});
            row_offset += request.rows();
        }
        if (!region->record_outputs(region_outputs, command, first_gate.option, region_downloaded)) return false;
    }

    if (use_route_aggregation)
    {
        ncnn::VkMat route_offsets_gpu;
        ncnn::VkMat route_rows_gpu;
        ncnn::VkMat route_weights_gpu;
        if (!record_mapped_upload(transfer_slot.route_offsets,
                                  route_offsets_gpu,
                                  command,
                                  first_gate.option)
            || !record_mapped_upload(transfer_slot.route_rows,
                                     route_rows_gpu,
                                     command,
                                     first_gate.option)
            || !record_mapped_upload(transfer_slot.route_weights,
                                     route_weights_gpu,
                                     command,
                                     first_gate.option))
        {
            return false;
        }
        if (!route_aggregation_pipeline
            && !create_route_agg_pipeline(first_gate.vulkan_context,
                                          first_gate.option,
                                          route_aggregation_pipeline))
        {
            return false;
        }
        ncnn::VkMat aggregated_output_gpu;
        aggregated_output_gpu.create(static_cast<int>(output_columns),
                                     static_cast<int>(aggregated_token_count),
                                     sizeof(float),
                                     first_gate.vulkan_context->blob_allocator());
        if (aggregated_output_gpu.empty())
            return false;
        std::vector<ncnn::VkMat> aggregation_bindings = {
            output_gpu,
            route_offsets_gpu,
            route_rows_gpu,
            route_weights_gpu,
            aggregated_output_gpu,
        };
        std::vector<ncnn::vk_constant_type> aggregation_constants(2);
        aggregation_constants[0].u32 = output_columns;
        aggregation_constants[1].u32 = aggregated_token_count;
        ncnn::VkMat aggregation_dispatcher;
        aggregation_dispatcher.w = static_cast<int>(output_columns * 128);
        aggregation_dispatcher.h = static_cast<int>(aggregated_token_count);
        aggregation_dispatcher.c = 1;
        command.record_pipeline(route_aggregation_pipeline.get(),
                                aggregation_bindings,
                                aggregation_constants,
                                aggregation_dispatcher);
        if (!record_staging_download(aggregated_output_gpu,
                                     aggregated_token_count,
                                     output_columns,
                                     transfer_slot.download,
                                     command,
                                     first_down.vulkan_context->device(),
                                     first_down.option,
                                     aggregated_output->dtype())
            || submit_compute_and_wait(command, first_gate.vulkan_context->device()) != 0
            || !copy_staging_to_cpu_batch(transfer_slot.download, *aggregated_output))
        {
            return false;
        }
    }
    else
    {
        if (download_output && !direct_host_output
            && !record_staging_download(output_gpu,
                                        total_rows,
                                        output_columns,
                                        transfer_slot.download,
                                        command,
                                        first_down.vulkan_context->device(),
                                        first_down.option,
                                        host_output.dtype()))
        {
            return false;
        }
        if (region) region->mark_submit_attempted();
        if (submit_compute_and_wait(command, first_gate.vulkan_context->device()) != 0)
            return false;
        if (use_direct_output)
            host_output.reset(total_rows, output_columns, false);
        if (download_output && !copy_staging_to_cpu_batch(transfer_slot.download, host_output))
            return false;
    }
    if (region && !region->complete_wave()) return false;
    if (retain_output)
    {
        row_offset = 0;
        for (size_t selected_slot = 0; selected_slot < selected.size(); ++selected_slot)
        {
            const auto& request = requests[selected[selected_slot].request_index];
            if (request.device_output)
            {
                auto completed = std::make_shared<DeviceTensor_vulkan>();
                const ncnn::VkMat value = align_output_rows ? aligned_outputs[selected_slot]
                                                            : row_view(output_gpu, row_offset, request.rows());
                if (!completed->assign_completed(value, first_gate.vulkan_context)) return false;
                *request.device_output = std::move(completed);
            }
            row_offset += request.rows();
        }
    }
    // VkBlobAllocator is shared and has no internal lock. Release every
    // temporary GPU owner before leaving the command/allocator lock.
    region_outputs.clear();
    aligned_outputs.clear();
    aligned_inputs.clear();
    intermediates.clear();
    intermediate_gpu.release();
    expert_row_offsets_gpu.release();
    expert_ids_gpu.release();
    output_gpu.release();
    input_rows_gpu.release();
    input_gpu.release();
    region_input.release();
    successful = true;
    lock.unlock();

    if (use_route_aggregation)
    {
        for (const Selection& selection : selected)
        {
            const ExpertRequest& request = requests[selection.request_index];
            if (request.route_aggregation.completed)
                *request.route_aggregation.completed = 1;
        }
    }
    else if (download_output && !use_direct_output)
    {
        row_offset = 0;
        for (const Selection& selection : selected)
        {
            const ExpertRequest& request = requests[selection.request_index];
            if (request.device_output)
            {
                row_offset += request.rows();
                continue;
            }
            request.output->reset(request.rows(), output_columns, false);
            for (size_t row = 0; row < request.rows(); ++row)
            {
                std::copy_n(combined_output.row(row_offset + row),
                            output_columns,
                            request.output->row(row));
            }
            row_offset += request.rows();
        }
    }
    const uint64_t projection_dispatches = use_indexed
                                               ? static_cast<uint64_t>(indexed_chunk_count) * 2
                                               : static_cast<uint64_t>(selected.size()) * 2;
    runtime_state.dispatches += projection_dispatches + static_cast<uint64_t>(use_route_aggregation)
                                + static_cast<uint64_t>(use_device_input && !shared_single_row)
                                + (region && region->complete() ? region->completed_combine_dispatches() : 0);
    ++runtime_state.compute_submissions;
    if (!use_device_input || region_uploaded)
        ++runtime_state.batch_uploads;
    if (download_output || use_route_aggregation || region_downloaded)
        ++runtime_state.batch_downloads;
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
