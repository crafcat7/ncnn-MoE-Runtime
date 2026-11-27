#include "backends/ncnn/expert/backend.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/expert/ffn_region.h"
#include "backends/ncnn/layer/moecombine_vulkan.h"
#include "backends/ncnn/linear/internal.h"
#include "kernels/vulkan/expert_gather.comp.hex.h"
#include "kernels/vulkan/mxfp4_indexed.comp.hex.h"
#include "kernels/vulkan/mxfp4_route_aggregation.comp.hex.h"

#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

static bool can_share_row(size_t available_rows, size_t request_count,
                          std::span<const uint32_t> input_rows)
{
    return available_rows == 1 && request_count != 0 && input_rows.size() == request_count
           && std::all_of(input_rows.begin(), input_rows.end(), [](uint32_t row) { return row == 0; });
}

static bool create_indexed_pipeline(const std::shared_ptr<VulkanContext>& context,
                                    const ncnn::Option& option,
                                    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const auto spirv = context->shader_binary(mxfp4_indexed_shader,
                                              static_cast<int>(sizeof(mxfp4_indexed_shader) - 1),
                                              option, 0);
    if (!spirv || spirv->empty())
        return false;

    const uint64_t key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_indexed_shader, key);
    if (destination)
        return true;

    std::unique_ptr<ncnn::Pipeline> pipe(new ncnn::Pipeline(context->device()));
    pipe->set_optimal_local_size_xyz(32, 1, 1);
    std::vector<ncnn::vk_specialization_type> specs(1);
    specs[0].u32 = pipe->local_size_x() * pipe->local_size_y() * pipe->local_size_z();
    if (pipe->create(spirv->data(), spirv->size() * sizeof(uint32_t), specs) != 0)
        return false;

    destination = std::shared_ptr<ncnn::Pipeline>(pipe.release(),
                                                  [context = context](ncnn::Pipeline* value) mutable {
                                                      {
                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                          delete value;
                                                      }
                                                      context.reset();
                                                  });
    context->cache_pipeline(mxfp4_indexed_shader, key, destination);
    return true;
}

bool VulkanExpertBackend::forward_batch(std::span<const ExpertRequest> requests,
                                        std::span<const Selection> selected)
{
    if (selected.empty())
        return true;

    const Mxfp4Expert_vulkan::Implementation& first_expert = *selected.front().entry->operation->d;
    const Mxfp4Linear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& first_down = *first_expert.down->d;

    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;
    const auto region = requests[selected.front().request_index].region;
    if (region)
    {
        if (!resident_combiner)
            resident_combiner = MoeCombine_vulkan::create(first_gate.vulkan_context);
        if (!region->bind_context(first_gate.vulkan_context, resident_combiner))
            return false;
    }

    bool use_indexed = has_flag(optimization_flags, OptimizationVulkanIndexedExperts)
                       && selected.size() >= 4
                       && selected.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())
                       && input_columns != 0 && first_gate.output_columns / 2 != 0 && output_columns != 0
                       && first_gate.indexed_storage && first_down.indexed_storage
                       && !first_gate.storage.empty() && !first_down.storage.empty();

    VulkanRuntimeState& state = first_gate.vulkan_context->runtime_state();
    bool retain_output = false;
    bool download_output = false;
    size_t total_rows = 0;
    size_t maximum_request_rows = 0;

    for (const Selection& selection : selected)
    {
        const ExpertRequest& request = requests[selection.request_index];
        const auto& expert = *selection.entry->operation->d;
        const auto& gate = *expert.gate_up->d;
        const auto& down = *expert.down->d;
        if (request.rows() > static_cast<size_t>(std::numeric_limits<int>::max()) - total_rows)
            return false;

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
        use_tiled_indexed = maximum_request_rows >= 2;
        indexed_chunk_count = (selected.size() + indexed_max_experts - 1) / indexed_max_experts;

        // Only enable indexed dispatch when batch chunking offsets its slot-switch overhead.
        const uint64_t regular_dispatches = static_cast<uint64_t>(selected.size()) * 2;
        const uint64_t indexed_dispatches = static_cast<uint64_t>(indexed_chunk_count) * 2;
        const uint64_t overhead_factor = use_tiled_indexed ? 4 : 6;
        if (indexed_dispatches == 0 || regular_dispatches <= indexed_dispatches * overhead_factor)
        {
            use_indexed = false;
            use_tiled_indexed = false;
            indexed_chunk_count = 0;
        }
    }

    if (use_indexed && !indexed_pipeline)
    {
        const std::lock_guard<std::mutex> lock(first_gate.vulkan_context->command_mutex());
        if (!create_indexed_pipeline(first_gate.vulkan_context, first_gate.option, indexed_pipeline))
        {
            use_indexed = false;
            use_tiled_indexed = false;
            indexed_chunk_count = 0;
        }
    }

    if (use_indexed && indexed_pipeline)
    {
        const uint64_t lx = indexed_pipeline->local_size_x();
        const uint64_t max_w = static_cast<uint64_t>(std::numeric_limits<int>::max());
        if (lx == 0 || lx > max_w
            || static_cast<uint64_t>(first_gate.output_columns / 2) > (max_w - (lx - 1)) / lx
            || static_cast<uint64_t>(output_columns) > (max_w - (lx - 1)) / lx)
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
    const bool use_route_agg = !retain_output
                               && route_aggregation_enabled(requests, selected, output_columns,
                                                            aggregated_output, aggregated_token_count,
                                                            optimization_flags);
    std::vector<uint32_t> route_offsets;
    std::vector<uint32_t> route_rows;
    std::vector<float> route_weights;
    if (use_route_agg && !build_ra_meta(requests, selected, route_offsets, route_rows, route_weights))
        return false;

    const auto retained_input = requests[selected.front().request_index].device_input;
    const ncnn::VkMat* retained_value = retained_input ? retained_input->value() : nullptr;
    const uint32_t input_pack = region ? 1 : (retained_value ? static_cast<uint32_t>(retained_value->elempack) : 0);
    const uint64_t input_rows_available = region ? region->rows()
                                                 : (retained_value && retained_value->h > 0
                                                        ? static_cast<uint64_t>(retained_value->h) * input_pack
                                                        : 0);
    bool use_device_input = region
                            || (retained_value && retained_input->context() == first_gate.vulkan_context
                                && retained_value->dims == 2
                                && retained_value->w == static_cast<int>(input_columns)
                                && (input_pack == 1 || input_pack == 4)
                                && retained_value->elemsize == sizeof(float) * input_pack
                                && retained_input->columns() == input_columns
                                && retained_input->rows() == input_rows_available);
    std::vector<uint32_t> input_rows;
    if (use_device_input)
    {
        input_rows.reserve(total_rows);
        for (const Selection& selection : selected)
        {
            const ExpertRequest& request = requests[selection.request_index];
            if ((!region && request.device_input != retained_input)
                || request.device_routes.size() != request.rows())
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
        use_device_input = create_pipeline(first_gate.vulkan_context, expert_gather_shader,
                                           static_cast<int>(sizeof(expert_gather_shader) - 1),
                                           first_gate.option, 0, 0, 128, input_gather_pipeline,
                                           {}, 0, 1, 1, true);
    }

    if (region && !use_device_input)
        return false;

    if (!use_device_input)
    {
        for (const Selection& selection : selected)
        {
            if (!requests[selection.request_index].has_host_input())
                return false;
        }
    }

    VkTransferLease transfer_lease = first_gate.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& slot = transfer_lease.slot();
    ncnn::VkMat input_rows_staging;
    if (use_device_input && !shared_single_row
        && !fill_staging_values(input_rows.data(), input_rows.size(), sizeof(uint32_t),
                                input_rows_staging, slot.staging_allocator))
        return false;

    if (!use_device_input
        && !prepare_staging_batch(slot.upload, total_rows, input_columns,
                                  slot.staging_allocator, sizeof(float)))
        return false;

    ncnn::Mat mapped_input;
    if (!use_device_input)
        mapped_input = slot.upload.mapped();

    size_t row_offset = 0;
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    for (size_t i = 0; i < selected.size(); ++i)
    {
        const ExpertRequest& request = requests[selected[i].request_index];
        if (use_indexed)
            expert_row_offsets[i] = static_cast<uint32_t>(row_offset);
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
                      static_cast<uint32_t>(i));
        }
        row_offset += request.rows();
    }
    if (use_indexed)
        expert_row_offsets[selected.size()] = static_cast<uint32_t>(row_offset);

    const size_t descriptor_alignment = std::max<size_t>(4, first_gate.vulkan_context->device()->info.buffer_offset_alignment());
    const bool align_output_rows = selected.size() > 1
                                   && (static_cast<size_t>(output_columns) * sizeof(float)) % descriptor_alignment != 0;
    const bool direct_in = !use_device_input
                           && first_gate.vulkan_context->support_direct_host_buffer(total_rows * input_columns * sizeof(float), DType::Float32);
    const bool direct_out = download_output && !retain_output && !use_route_agg && !align_output_rows
                            && first_gate.vulkan_context->support_direct_host_buffer(total_rows * output_columns * sizeof(float), DType::Float32);
    const size_t download_rows = use_route_agg ? aggregated_token_count : total_rows;
    if ((download_output || use_route_agg)
        && !prepare_staging_batch(slot.download, download_rows, output_columns, slot.staging_allocator))
        return false;

    if (use_indexed)
    {
        if (!fill_staging_values(row_slots.data(), row_slots.size(), sizeof(uint32_t),
                                 slot.expert_slots, slot.staging_allocator))
            return false;
        if (use_tiled_indexed
            && !fill_staging_values(expert_row_offsets.data(), expert_row_offsets.size(), sizeof(uint32_t),
                                    slot.expert_row_offsets, slot.staging_allocator))
            return false;
    }

    if (use_route_agg)
    {
        if (!fill_staging_values(route_offsets.data(), route_offsets.size(), sizeof(uint32_t),
                                 slot.route_offsets, slot.staging_allocator)
            || !fill_staging_values(route_rows.data(), route_rows.size(), sizeof(uint32_t),
                                    slot.route_rows, slot.staging_allocator)
            || !fill_staging_values(route_weights.data(), route_weights.size(), sizeof(float),
                                    slot.route_weights, slot.staging_allocator))
            return false;
    }

    const bool use_direct_output = download_output && !use_route_agg && selected.size() == 1;
    ActivationBuffer combined_output;
    ActivationBuffer& host_output = use_direct_output ? *requests[selected.front().request_index].output
                                                      : combined_output;
    if (download_output && !use_route_agg && !use_direct_output)
        host_output.reset(total_rows, output_columns, false);

    std::unique_lock<std::mutex> lock(first_gate.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *slot.command;
    if (slot.command_used && cmd.reset() != 0)
        return false;
    slot.command_used = true;

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

    const auto cleanup = [&](void*) {
        if (!successful)
        {
            (void)cmd.reset();
            if (region)
                region->abort_wave();
        }
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);

    if (use_device_input)
    {
        if (region)
        {
            if (!region->record_input(region_input, cmd, first_gate.option, region_uploaded))
                return false;
            retained_value = &region_input;
        }
        else
        {
            retained_input->prepare_read();
        }

        if (shared_single_row)
        {
            input_gpu = aligned_row_view(*retained_value, 0, 1, cmd, first_gate.option,
                                         first_gate.vulkan_context, aligned_inputs, selected.size());
            if (input_gpu.empty())
                return false;
        }
        else
        {
            if (!record_mapped_upload(input_rows_staging, input_rows_gpu, cmd, first_gate.option))
                return false;
            input_gpu.create(static_cast<int>(input_columns), static_cast<int>(total_rows),
                             sizeof(float), first_gate.vulkan_context->blob_allocator());
            if (input_gpu.empty())
                return false;

            std::vector<ncnn::vk_constant_type> constants(3);
            constants[0].u32 = input_columns;
            constants[1].u32 = static_cast<uint32_t>(total_rows);
            constants[2].u32 = input_pack;

            ncnn::VkMat dispatch;
            dispatch.w = static_cast<int>(input_columns);
            dispatch.h = static_cast<int>(std::min(total_rows, size_t(65535)));
            dispatch.c = static_cast<int>((total_rows + 65534) / 65535);
            cmd.record_pipeline_readonly(input_gather_pipeline.get(),
                                         {*retained_value, input_rows_gpu, input_gpu},
                                         {1, 1, 0}, constants, dispatch);
        }
    }
    else if (direct_in)
    {
        input_gpu = bind_host_in(slot.upload);
    }
    else
    {
        slot.upload.allocator->flush(slot.upload.data);
        slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        if (!record_mapped_upload(slot.upload, input_gpu, cmd, first_gate.option))
            return false;
    }

    if (direct_out)
        output_gpu = bind_host_out(slot.download);
    else
        output_gpu.create(static_cast<int>(output_columns), static_cast<int>(total_rows),
                          sizeof(float), first_gate.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    if (use_indexed)
    {
        if (!record_mapped_upload(slot.expert_slots, expert_ids_gpu, cmd, first_gate.option))
            return false;
        if (use_tiled_indexed
            && !record_mapped_upload(slot.expert_row_offsets, expert_row_offsets_gpu, cmd, first_gate.option))
            return false;

        intermediate_gpu.create(static_cast<int>(first_gate.output_columns / 2),
                                static_cast<int>(total_rows),
                                sizeof(float), first_gate.vulkan_context->blob_allocator());
        if (intermediate_gpu.empty())
            return false;

        std::vector<ncnn::VkMat> bindings(indexed_max_experts + 3);
        std::vector<unsigned char> readonly(bindings.size(), 1);
        readonly.back() = 0;

        std::vector<ncnn::vk_constant_type> constants(20);
        constants[0].u32 = first_gate.input_columns;
        constants[1].u32 = first_gate.output_columns;
        constants[2].u32 = first_down.input_columns;
        constants[3].u32 = first_down.output_columns;
        constants[4].u32 = first_gate.block_count;
        constants[5].u32 = first_down.block_count;
        constants[6].u32 = static_cast<uint32_t>(total_rows);
        constants[9].u32 = first_gate.indexed_scales_word_offset;
        constants[10].u32 = first_gate.indexed_bias_word_offset;
        constants[12].u32 = first_down.indexed_scales_word_offset;
        constants[13].u32 = first_down.indexed_bias_word_offset;
        constants[15].u32 = first_expert.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
        constants[16].f = first_expert.activation_limit;

        for (size_t chunk_begin = 0; chunk_begin < selected.size(); chunk_begin += indexed_max_experts)
        {
            const size_t chunk_end = std::min(selected.size(), chunk_begin + static_cast<size_t>(indexed_max_experts));
            const uint32_t chunk_count = static_cast<uint32_t>(chunk_end - chunk_begin);
            const uint32_t row_base = expert_row_offsets[chunk_begin];
            const uint32_t row_end = expert_row_offsets[chunk_end];
            const uint32_t chunk_rows = row_end - row_base;
            uint32_t chunk_max_rows = 0;
            for (size_t s = chunk_begin; s < chunk_end; ++s)
                chunk_max_rows = std::max(chunk_max_rows, static_cast<uint32_t>(requests[selected[s].request_index].rows()));

            const uint32_t chunk_tile_mode = use_tiled_indexed && chunk_max_rows >= 2 ? 1u : 0u;
            bindings[0] = input_gpu;
            for (uint32_t s = 0; s < indexed_max_experts; ++s)
            {
                if (s < chunk_count)
                    bindings[s + 1] = selected[chunk_begin + s].entry->operation->d->gate_up->d->storage;
                else
                    bindings[s + 1].release();
            }
            bindings[indexed_max_experts + 1] = chunk_tile_mode != 0 ? expert_row_offsets_gpu : expert_ids_gpu;
            bindings.back() = intermediate_gpu;
            constants[7].u32 = chunk_count;
            constants[14].u32 = 0;
            constants[17].u32 = chunk_tile_mode;
            constants[18].u32 = row_base;
            constants[19].u32 = static_cast<uint32_t>(chunk_begin);

            ncnn::VkMat gate_dispatch;
            gate_dispatch.w = static_cast<int>(static_cast<uint64_t>(first_gate.output_columns / 2)
                                               * indexed_pipeline->local_size_x());
            gate_dispatch.h = static_cast<int>(chunk_tile_mode != 0 ? (chunk_max_rows + 1) / 2 : chunk_rows);
            gate_dispatch.c = static_cast<int>(chunk_tile_mode != 0 ? chunk_count : 1);
            cmd.record_pipeline_readonly(indexed_pipeline.get(), bindings, readonly, constants, gate_dispatch);

            bindings[0] = intermediate_gpu;
            for (uint32_t s = 0; s < indexed_max_experts; ++s)
            {
                if (s < chunk_count)
                    bindings[s + 1] = selected[chunk_begin + s].entry->operation->d->down->d->storage;
                else
                    bindings[s + 1].release();
            }
            bindings.back() = output_gpu;
            constants[14].u32 = 1;

            ncnn::VkMat down_dispatch;
            down_dispatch.w = static_cast<int>(static_cast<uint64_t>(output_columns)
                                               * indexed_pipeline->local_size_x());
            down_dispatch.h = gate_dispatch.h;
            down_dispatch.c = gate_dispatch.c;
            cmd.record_pipeline_readonly(indexed_pipeline.get(), bindings, readonly, constants, down_dispatch);
        }
    }
    else
    {
        intermediates.reserve(selected.size());
        std::vector<ncnn::VkMat> bindings(5);
        std::vector<ncnn::vk_constant_type> constants(7);
        row_offset = 0;

        for (size_t s = 0; s < selected.size(); ++s)
        {
            const Selection& selection = selected[s];
            const ExpertRequest& request = requests[selection.request_index];
            const auto& expert = *selection.entry->operation->d;
            const auto& gate = *expert.gate_up->d;
            const uint32_t intermediate_columns = gate.output_columns / 2;

            ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_single_row ? 0 : row_offset,
                                                      request.rows(), cmd, first_gate.option,
                                                      first_gate.vulkan_context, aligned_inputs, selected.size());
            ncnn::VkMat output_view = row_view(output_gpu, row_offset, request.rows());
            if (input_view.empty() || output_view.empty())
                return false;

            if (align_output_rows)
            {
                aligned_outputs[s].create(output_view.w, output_view.h, sizeof(float),
                                          first_gate.vulkan_context->blob_allocator());
                output_view = aligned_outputs[s];
                if (output_view.empty())
                    return false;
            }

            ncnn::VkMat& intermediate = intermediates.emplace_back();
            intermediate.create(static_cast<int>(intermediate_columns), static_cast<int>(request.rows()),
                                sizeof(float), first_gate.vulkan_context->blob_allocator());
            if (intermediate.empty())
                return false;

            selection.entry->operation->record(input_view, intermediate, output_view, cmd, bindings, constants);
            if (align_output_rows)
            {
                ncnn::VkMat dst = row_view(output_gpu, row_offset, request.rows());
                if (!record_row_copy(output_view, dst, cmd, first_gate.option))
                    return false;
            }
            row_offset += request.rows();
        }
    }

    if (align_output_rows && retain_output && use_indexed)
    {
        row_offset = 0;
        for (size_t s = 0; s < selected.size(); ++s)
        {
            const auto& request = requests[selected[s].request_index];
            if (request.device_output)
            {
                ncnn::VkMat src = row_view(output_gpu, row_offset, request.rows());
                cmd.record_clone(src, aligned_outputs[s], first_gate.option);
            }
            row_offset += request.rows();
        }
    }

    if (region)
    {
        region_outputs.reserve(selected.size());
        row_offset = 0;
        for (size_t s = 0; s < selected.size(); ++s)
        {
            const auto& request = requests[selected[s].request_index];
            const ncnn::VkMat val = align_output_rows ? aligned_outputs[s]
                                                      : row_view(output_gpu, row_offset, request.rows());
            if (val.empty())
                return false;
            region_outputs.push_back({request.region_index, val, request.device_routes});
            row_offset += request.rows();
        }
        if (!region->record_outputs(region_outputs, cmd, first_gate.option, region_downloaded))
            return false;
    }

    if (use_route_agg)
    {
        ncnn::VkMat route_offsets_gpu;
        ncnn::VkMat route_rows_gpu;
        ncnn::VkMat route_weights_gpu;
        if (!record_mapped_upload(slot.route_offsets, route_offsets_gpu, cmd, first_gate.option)
            || !record_mapped_upload(slot.route_rows, route_rows_gpu, cmd, first_gate.option)
            || !record_mapped_upload(slot.route_weights, route_weights_gpu, cmd, first_gate.option))
            return false;

        if (!route_aggregation_pipeline
            && !create_pipeline(first_gate.vulkan_context, mxfp4_route_aggregation_shader,
                                static_cast<int>(sizeof(mxfp4_route_aggregation_shader) - 1),
                                first_gate.option, 0, 0, 128, route_aggregation_pipeline))
            return false;

        ncnn::VkMat aggregated_gpu(static_cast<int>(output_columns), static_cast<int>(aggregated_token_count),
                                   sizeof(float), first_gate.vulkan_context->blob_allocator());
        if (aggregated_gpu.empty())
            return false;

        std::vector<ncnn::VkMat> agg_bindings = {output_gpu, route_offsets_gpu, route_rows_gpu,
                                                 route_weights_gpu, aggregated_gpu};
        std::vector<ncnn::vk_constant_type> agg_constants(2);
        agg_constants[0].u32 = output_columns;
        agg_constants[1].u32 = aggregated_token_count;

        ncnn::VkMat agg_dispatch;
        agg_dispatch.w = static_cast<int>(output_columns * 128);
        agg_dispatch.h = static_cast<int>(aggregated_token_count);
        agg_dispatch.c = 1;
        cmd.record_pipeline(route_aggregation_pipeline.get(), agg_bindings, agg_constants, agg_dispatch);

        if (!record_staging_download(aggregated_gpu, aggregated_token_count, output_columns,
                                     slot.download, cmd, first_down.vulkan_context->device(),
                                     first_down.option, aggregated_output->dtype())
            || submit_wait(cmd, first_gate.vulkan_context->device()) != 0
            || !copy_staging_to_cpu_batch(slot.download, *aggregated_output))
            return false;
    }
    else
    {
        if (download_output && !direct_out
            && !record_staging_download(output_gpu, total_rows, output_columns,
                                        slot.download, cmd, first_down.vulkan_context->device(),
                                        first_down.option, host_output.dtype()))
            return false;

        if (region)
            region->mark_submit_attempted();
        if (submit_wait(cmd, first_gate.vulkan_context->device()) != 0)
            return false;
        if (use_direct_output)
            host_output.reset(total_rows, output_columns, false);
        if (download_output && !copy_staging_to_cpu_batch(slot.download, host_output))
            return false;
    }

    if (region && !region->complete_wave())
        return false;

    if (retain_output)
    {
        row_offset = 0;
        for (size_t s = 0; s < selected.size(); ++s)
        {
            const auto& request = requests[selected[s].request_index];
            if (request.device_output)
            {
                auto completed = std::make_shared<DeviceTensor_vulkan>();
                const ncnn::VkMat val = align_output_rows ? aligned_outputs[s]
                                                          : row_view(output_gpu, row_offset, request.rows());
                if (!completed->assign_completed(val, first_gate.vulkan_context))
                    return false;
                *request.device_output = std::move(completed);
            }
            row_offset += request.rows();
        }
    }

    // Release temporary device buffers while still holding the shared blob allocator lock.
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

    if (use_route_agg)
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
                std::copy_n(combined_output.row(row_offset + row), output_columns, request.output->row(row));
            row_offset += request.rows();
        }
    }

    const uint64_t proj_dispatches = use_indexed ? static_cast<uint64_t>(indexed_chunk_count) * 2
                                                 : static_cast<uint64_t>(selected.size()) * 2;
    state.dispatches += proj_dispatches + static_cast<uint64_t>(use_route_agg)
                        + static_cast<uint64_t>(use_device_input && !shared_single_row)
                        + (region && region->complete() ? region->completed_combine_dispatches() : 0);
    ++state.compute_submissions;
    if (!use_device_input || region_uploaded)
        ++state.batch_uploads;
    if (download_output || use_route_agg || region_downloaded)
        ++state.batch_downloads;
    return true;
}

bool VulkanExpertBackend::forward_device_batch(std::span<const ExpertRequest> requests,
                                               std::span<const Selection> selected)
{
    if (selected.empty())
        return true;

    auto valid_entry = [](const Entry* entry) {
        return entry && (entry->qnk_operation || entry->bfloat16_operation || entry->operation);
    };
    if (!valid_entry(selected.front().entry.get()) || selected.front().request_index >= requests.size())
        return false;

    const auto& first = *selected.front().entry;
    const auto context = first.qnk_operation        ? first.qnk_operation->d->vulkan_context
                         : first.bfloat16_operation ? first.bfloat16_operation->d->vulkan_context
                                                    : first.operation->d->gate_up->d->vulkan_context;
    const auto& option = first.qnk_operation        ? first.qnk_operation->d->option
                         : first.bfloat16_operation ? first.bfloat16_operation->d->option
                                                    : first.operation->d->gate_up->d->option;
    if (!context || activation_variant(option) != 0)
        return false;

    const uint32_t input_columns = first.qnk_operation        ? first.qnk_operation->d->gate_up->input_columns()
                                   : first.bfloat16_operation ? first.bfloat16_operation->d->input_columns
                                                              : first.operation->d->gate_up->d->input_columns;
    const uint32_t output_columns = first.qnk_operation        ? first.qnk_operation->d->output_columns
                                    : first.bfloat16_operation ? first.bfloat16_operation->d->output_columns
                                                               : first.operation->d->down->d->output_columns;

    const auto& first_request = requests[selected.front().request_index];
    const auto region = first_request.region;
    const auto retained_input = first_request.device_input;
    const ncnn::VkMat* source = retained_input ? retained_input->value() : nullptr;
    size_t available_rows = 0;

    if (region)
    {
        if (region->columns() != input_columns || region->rows() == 0)
            return false;
        available_rows = region->rows();
        if (!resident_combiner)
            resident_combiner = MoeCombine_vulkan::create(context);
        if (!region->bind_context(context, resident_combiner))
            return false;
    }
    else
    {
        if (!source || source->dims != 2 || source->w != static_cast<int>(input_columns)
            || source->h <= 0 || (source->elempack != 1 && source->elempack != 4)
            || source->elemsize != sizeof(float) * static_cast<size_t>(source->elempack)
            || retained_input->context() != context)
            return false;
        available_rows = static_cast<size_t>(source->h) * source->elempack;
        if (retained_input->rows() != available_rows || retained_input->columns() != input_columns)
            return false;
    }

    size_t total_rows = 0;
    uint64_t dispatches = 0;
    uint64_t int4_dispatches = 0;
    std::vector<uint32_t> input_rows;

    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size() || !valid_entry(selection.entry.get()))
            return false;

        const auto& request = requests[selection.request_index];
        const auto* qnk = selection.entry->qnk_operation.get();
        const auto* bf16 = selection.entry->bfloat16_operation.get();
        const auto* mxfp4 = selection.entry->operation.get();
        const auto& expert_ctx = qnk    ? qnk->d->vulkan_context
                                 : bf16 ? bf16->d->vulkan_context
                                        : mxfp4->d->gate_up->d->vulkan_context;
        const uint32_t expert_in = qnk    ? qnk->d->gate_up->input_columns()
                                   : bf16 ? bf16->d->input_columns
                                          : mxfp4->d->gate_up->d->input_columns;
        const uint32_t expert_out = qnk    ? qnk->d->output_columns
                                    : bf16 ? bf16->d->output_columns
                                           : mxfp4->d->down->d->output_columns;

        if (!request.output || !request.device_output || request.region != region
            || (!region && request.device_input != retained_input)
            || request.rows() == 0 || request.rows() != request.device_routes.size()
            || request.columns() != input_columns || expert_ctx != context
            || expert_in != input_columns || expert_out != output_columns
            || request.rows() > context->device()->info.max_workgroup_count_y()
            || request.rows() > static_cast<size_t>(std::numeric_limits<int>::max()) - total_rows)
            return false;

        for (const ExpertRoute& route : request.device_routes)
        {
            if (route.token_index >= available_rows)
                return false;
            input_rows.push_back(route.token_index);
        }

        total_rows += request.rows();
        if (qnk)
        {
            dispatches += (qnk->d->up ? 4u : 3u) + (qnk->d->router_gate ? 2u : 0u);
            int4_dispatches += qnk->d->gate_up->dtype() == DType::Int4 ? 1u : 0u;
            int4_dispatches += qnk->d->down->dtype() == DType::Int4 ? 1u : 0u;
            int4_dispatches += qnk->d->up && qnk->d->up->dtype() == DType::Int4 ? 1u : 0u;
        }
        else
        {
            dispatches += 2;
        }
    }

    const bool shared_single_row = can_share_row(available_rows, selected.size(), input_rows);
    dispatches += !shared_single_row;

    VkTransferLease transfer_lease = context->acquire_transfer_slot();
    VkTransferSlot& slot = transfer_lease.slot();
    ncnn::VkMat rows_staging;
    if (!shared_single_row
        && !fill_staging_values(input_rows.data(), input_rows.size(), sizeof(uint32_t),
                                rows_staging, slot.staging_allocator))
        return false;

    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (!shared_single_row && !input_gather_pipeline
        && !create_pipeline(context, expert_gather_shader,
                            static_cast<int>(sizeof(expert_gather_shader) - 1),
                            option, 0, 0, 128, input_gather_pipeline))
        return false;

    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;
    ncnn::VkCompute& cmd = *slot.command;

    ncnn::VkMat region_input;
    ncnn::VkMat rows_gpu;
    ncnn::VkMat input_gpu;
    std::vector<ncnn::VkMat> workspace;
    std::vector<ncnn::VkMat> aligned_inputs;
    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::VkMat> outputs;
    std::vector<ncnn::vk_constant_type> constants(3);
    std::vector<ResidentFfnRegion_vulkan::Output> region_outputs;
    outputs.reserve(selected.size());
    region_outputs.reserve(selected.size());

    bool successful = false;
    const auto cleanup = [&](void*) {
        if (!successful)
        {
            (void)cmd.reset();
            if (region)
                region->abort_wave();
        }
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);

    bool uploaded = false;
    bool downloaded = false;
    if (region)
    {
        if (!region->record_input(region_input, cmd, option, uploaded))
            return false;
        source = &region_input;
    }
    else
    {
        retained_input->prepare_read();
    }

    if (shared_single_row)
    {
        input_gpu = aligned_row_view(*source, 0, 1, cmd, option, context, aligned_inputs, selected.size());
        if (input_gpu.empty())
            return false;
    }
    else
    {
        if (!record_mapped_upload(rows_staging, rows_gpu, cmd, option))
            return false;
        input_gpu.create(static_cast<int>(input_columns), static_cast<int>(total_rows),
                         sizeof(float), context->blob_allocator());
        if (input_gpu.empty())
            return false;

        constants[0].u32 = input_columns;
        constants[1].u32 = static_cast<uint32_t>(total_rows);
        constants[2].u32 = static_cast<uint32_t>(source->elempack);
        ncnn::VkMat dispatch;
        dispatch.w = static_cast<int>(input_columns);
        dispatch.h = static_cast<int>(std::min(total_rows, size_t(65535)));
        dispatch.c = static_cast<int>((total_rows + 65534) / 65535);
        cmd.record_pipeline_readonly(input_gather_pipeline.get(), {*source, rows_gpu, input_gpu},
                                     {1, 1, 0}, constants, dispatch);
    }

    size_t offset = 0;
    for (const Selection& selection : selected)
    {
        const auto& request = requests[selection.request_index];
        ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_single_row ? 0 : offset,
                                                  request.rows(), cmd, option, context,
                                                  aligned_inputs, selected.size());
        ncnn::VkMat output_gpu(static_cast<int>(output_columns), static_cast<int>(request.rows()),
                               sizeof(float), context->blob_allocator());
        if (input_view.empty() || output_gpu.empty())
            return false;
        outputs.push_back(output_gpu);

        if (selection.entry->qnk_operation)
        {
            if (!selection.entry->qnk_operation->record(input_view, output_gpu, cmd, workspace, bindings, constants))
                return false;
        }
        else if (selection.entry->bfloat16_operation)
        {
            if (!selection.entry->bfloat16_operation->record(input_view, output_gpu, cmd, workspace, bindings, constants))
                return false;
        }
        else
        {
            auto& intermediate = workspace.emplace_back();
            intermediate.create(static_cast<int>(selection.entry->operation->d->gate_up->d->output_columns / 2),
                                static_cast<int>(request.rows()), sizeof(float), context->blob_allocator());
            if (intermediate.empty())
                return false;
            selection.entry->operation->record(input_view, intermediate, output_gpu, cmd, bindings, constants);
        }

        if (region)
            region_outputs.push_back({request.region_index, output_gpu, request.device_routes});
        offset += request.rows();
    }

    if (region && !region->record_outputs(region_outputs, cmd, option, downloaded))
        return false;
    if (region)
        region->mark_submit_attempted();
    if (submit_wait(cmd, context->device()) != 0)
        return false;
    if (region && !region->complete_wave())
        return false;

    std::vector<std::shared_ptr<DeviceTensor_vulkan>> completed;
    completed.reserve(selected.size());
    for (const ncnn::VkMat& output : outputs)
    {
        auto val = std::make_shared<DeviceTensor_vulkan>();
        if (!val->assign_completed(output, context))
            return false;
        completed.push_back(std::move(val));
    }
    for (size_t i = 0; i < selected.size(); ++i)
        *requests[selected[i].request_index].device_output = std::move(completed[i]);

    auto& state = context->runtime_state();
    state.dispatches += dispatches + (region && region->complete() ? region->completed_combine_dispatches() : 0);
    state.int4_dispatches += int4_dispatches;
    state.batch_uploads += uploaded ? 1 : 0;
    state.batch_downloads += downloaded ? 1 : 0;
    ++state.compute_submissions;
    successful = true;
    return true;
}

} // namespace moe
} // namespace ncnn
