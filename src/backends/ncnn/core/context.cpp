#include "backends/ncnn/core/context.h"

#include "backends/ncnn/core/memory.h"
#include "backends/ncnn/core/transfer.h"
#include "kernels/activationbuffer.h"

#include <layer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

namespace ncnn {
namespace moe {

static ncnn::Option source_act_opt(const ncnn::Option& opt, DType src_dtype)
{
    ncnn::Option src_opt = opt;
    if (src_dtype == DType::BFloat16)
    {
        src_opt.use_fp16_storage = false;
        src_opt.use_fp16_packed = false;
        src_opt.use_bf16_storage = true;
        src_opt.use_bf16_packed = true;
    }
    else if (src_dtype == DType::Float16)
    {
        src_opt.use_bf16_storage = false;
        src_opt.use_bf16_packed = false;
        src_opt.use_fp16_storage = true;
        src_opt.use_fp16_packed = true;
    }
    return src_opt;
}

bool prepare_staging_batch(ncnn::VkMat& buf, size_t rows, uint32_t cols,
                           ncnn::VkAllocator* alloc, size_t elem_size)
{
    buf.create(static_cast<int>(cols), static_cast<int>(rows), elem_size, alloc);
    return !buf.empty() && buf.mapped_ptr() != nullptr;
}

bool record_mapped_upload(ncnn::VkMat& staging, ncnn::VkMat& dst,
                          ncnn::VkCompute& cmd, const ncnn::Option& opt)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    cmd.record_clone(staging, dst, opt);
    return !dst.empty();
}

bool record_mapped_act_upload(ncnn::VkMat& staging, ncnn::VkMat& dst,
                              ncnn::VkCompute& cmd, ncnn::VulkanDevice* device,
                              const ncnn::Option& opt, DType src_dtype)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    if (activation_variant(opt) == 0)
    {
        if (src_dtype == DType::Float32)
        {
            device->convert_packing(staging, dst, 1, 1, cmd, opt);
            return !dst.empty();
        }

        const ncnn::Option src_opt = source_act_opt(opt, src_dtype);
        const int dst_pack = staging.h % 4 == 0 ? 4 : 1;
        device->convert_packing(staging, dst, dst_pack, 1, cmd, src_opt);
        return !dst.empty();
    }

    const int cast_type = opt.use_bf16_storage ? 5 : 2;
    const ncnn::Option src_opt = source_act_opt(opt, src_dtype);
    device->convert_packing(staging, dst, 1, cast_type, cmd, src_opt);
    return !dst.empty();
}

bool fill_staging_upload(const ActivationBuffer& input, ncnn::VkMat& staging,
                         ncnn::VkAllocator* alloc)
{
    const size_t elem_size = input.element_size();
    if (!prepare_staging_batch(staging, input.rows(), input.columns(), alloc, elem_size))
        return false;

    ncnn::Mat mapped = staging.mapped();
    std::memcpy(mapped.data, input.bytes().data(), input.bytes().size());

    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool fill_staging_values(const void* src, size_t count, size_t elem_size,
                         ncnn::VkMat& staging, ncnn::VkAllocator* alloc)
{
    if (!prepare_staging_batch(staging, 1, static_cast<uint32_t>(count), alloc, elem_size))
        return false;

    ncnn::Mat mapped = staging.mapped();
    std::memcpy(mapped.data, src, count * elem_size);

    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool record_prepared_staging_upload(const ncnn::VkMat& staging, size_t rows,
                                    ncnn::VkMat& dst, ncnn::VkCompute& cmd,
                                    ncnn::VulkanDevice* device, const ncnn::Option& opt,
                                    DType src_dtype)
{
    const int packed_rows = static_cast<int>(rows);
    const int dst_pack = packed_rows % 4 == 0 && (opt.use_packing_layout || packed_rows <= 4) ? 4 : 1;

    int cast_type = src_dtype == DType::Float32 ? 0 : 1;
    if (opt.use_bf16_storage || opt.use_bf16_packed)
        cast_type = 5;
    else if (opt.use_fp16_storage || opt.use_fp16_packed)
        cast_type = 2;
    else if (device->info.type() != 0)
        cast_type = 1;

    const ncnn::Option src_opt = source_act_opt(opt, src_dtype);
    device->convert_packing(staging, dst, dst_pack, cast_type, cmd, src_opt);
    return !dst.empty();
}

bool record_staging_download(const ncnn::VkMat& src, size_t rows, uint32_t cols,
                             ncnn::VkMat& staging, ncnn::VkCompute& cmd,
                             ncnn::VulkanDevice* device, const ncnn::Option& opt,
                             DType dst_dtype)
{
    const size_t dst_elem_size = dst_dtype == DType::Float32 ? sizeof(float) : sizeof(uint16_t);
    const bool direct_fp32 = activation_variant(opt) == 0
                             && dst_dtype == DType::Float32
                             && src.dims == 2
                             && src.w == static_cast<int>(cols)
                             && src.h == static_cast<int>(rows)
                             && src.elempack == 1
                             && src.elemsize == sizeof(float);

    ncnn::Option staging_opt = opt;
    staging_opt.blob_vkallocator = staging.allocator;
    staging_opt.workspace_vkallocator = staging.allocator;
    staging_opt.staging_vkallocator = staging.allocator;

    if (direct_fp32)
    {
        cmd.record_clone(src, staging, staging_opt);
        return !staging.empty();
    }

    staging.create(static_cast<int>(cols), static_cast<int>(rows), dst_elem_size, staging.allocator);
    if (staging.empty() || !staging.mapped_ptr())
        return false;

    const int cast_type = dst_dtype == DType::Float32 ? 1 : (dst_dtype == DType::Float16 ? 2 : 5);
    device->convert_packing(src, staging, 1, cast_type, cmd, staging_opt);
    return !staging.empty();
}

bool copy_staging_to_cpu_batch(ncnn::VkMat& staging, ActivationBuffer& output)
{
    const size_t elem_size = output.element_size();
    if (staging.empty() || !staging.allocator || !staging.data || elem_size == 0
        || staging.elemsize != elem_size || staging.w <= 0 || staging.h <= 0
        || output.columns() != static_cast<uint32_t>(staging.w)
        || output.rows() != static_cast<size_t>(staging.h))
        return false;

    staging.allocator->invalidate(staging.data);

    const ncnn::Mat mapped = staging.mapped();
    const size_t row_bytes = static_cast<size_t>(output.columns()) * elem_size;
    if (mapped.empty() || !mapped.data || output.bytes().size() != row_bytes * output.rows())
        return false;

    const auto* src = static_cast<const std::byte*>(mapped.data);
    for (size_t r = 0; r < output.rows(); ++r)
        std::memcpy(output.mutable_row_bytes(r).data(), src + r * row_bytes, row_bytes);

    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool copy_staging_to_cpu_batches(ncnn::VkMat& staging,
                                 std::span<const ActivationBuffer*> inputs,
                                 std::span<ActivationBuffer*> outputs,
                                 uint32_t cols)
{
    if (inputs.empty() || inputs.size() != outputs.size() || cols == 0
        || staging.empty() || !staging.allocator || !staging.data || !outputs.front())
    {
        return false;
    }

    const DType dst_dtype = outputs.front()->dtype();
    const size_t elem_size = outputs.front()->element_size();
    if (elem_size == 0 || static_cast<size_t>(cols) > std::numeric_limits<size_t>::max() / elem_size)
        return false;

    const size_t row_bytes = static_cast<size_t>(cols) * elem_size;
    size_t total_rows = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const ActivationBuffer* in = inputs[i];
        ActivationBuffer* out = outputs[i];
        if (!in || !out || in->rows() == 0 || out->dtype() != dst_dtype
            || in->rows() > std::numeric_limits<size_t>::max() / row_bytes
            || total_rows > std::numeric_limits<size_t>::max() - in->rows())
        {
            return false;
        }

        // Reject duplicate outputs and cross-index input/output aliasing before mutating outputs.
        for (size_t j = 0; j < i; ++j)
            if (out == outputs[j])
                return false;

        for (size_t j = 0; j < inputs.size(); ++j)
            if (j != i && out == inputs[j])
                return false;

        total_rows += in->rows();
    }

    if (total_rows > static_cast<size_t>(std::numeric_limits<int>::max())
        || cols > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || staging.dims != 2
        || staging.w != static_cast<int>(cols)
        || staging.h != static_cast<int>(total_rows)
        || staging.elemsize != elem_size
        || staging.elempack != 1)
    {
        return false;
    }

    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();

    for (size_t i = 0; i < outputs.size(); ++i)
        outputs[i]->reset(inputs[i]->rows(), cols, false);

    const auto* src = static_cast<const std::byte*>(mapped.data);
    size_t row_offset = 0;
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        ActivationBuffer& out = *outputs[i];
        const size_t bytes = out.rows() * row_bytes;
        std::memcpy(out.mutable_bytes().data(), src + row_offset * row_bytes, bytes);
        row_offset += out.rows();
    }

    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool prepare_float_upload(const TensorData& src, ncnn::Mat& dst)
{
    const size_t count = src.element_count();
    if (count == 0 || count > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;

    dst.create(static_cast<int>(count), sizeof(float));
    if (dst.empty())
        return false;

    float* out = static_cast<float*>(dst.data);
    if (src.dtype == DType::Float32)
    {
        const auto vals = src.float32_values();
        if (vals.size() != count)
            return false;
        std::copy(vals.begin(), vals.end(), out);
        return true;
    }

    if (src.dtype == DType::BFloat16)
    {
        const auto vals = src.bfloat16_values();
        if (vals.size() != count)
            return false;
        for (size_t i = 0; i < count; ++i)
            out[i] = ncnn::bfloat16_to_float32(vals[i]);
        return true;
    }

    return false;
}

int wait_cmd(ncnn::VkCompute& cmd, const ncnn::VulkanDevice* probe_device,
             VkQueue borrowed_queue)
{
    // ncnn collapses fence wait errors to -1; retry until completion or confirmed device loss.
    for (;;)
    {
        const int ret = cmd.wait();
        if (ret == 0)
            return 0;

        VkResult probe = VK_SUCCESS;
        if (borrowed_queue)
        {
            probe = vkQueueWaitIdle(borrowed_queue);
        }
        else if (probe_device)
        {
            const uint32_t family = probe_device->info.compute_queue_family_index();
            const VkQueue queue = probe_device->acquire_queue(family);
            if (queue)
            {
                probe = vkQueueWaitIdle(queue);
                probe_device->reclaim_queue(family, queue);
            }
        }

        if (probe == VK_ERROR_DEVICE_LOST)
            return ret;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int submit_wait(ncnn::VkCompute& cmd, const ncnn::VulkanDevice* probe_device)
{
    const int ret = cmd.submit();
    return ret == 0 ? wait_cmd(cmd, probe_device) : ret;
}

void drain_queue_family(const ncnn::VulkanDevice* device, uint32_t family,
                        std::span<VkQueue> queues) noexcept
{
    for (VkQueue& q : queues)
    {
        while (!(q = device->acquire_queue(family)))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    for (VkQueue q : queues)
    {
        for (VkResult res = vkQueueWaitIdle(q);
             res != VK_SUCCESS && res != VK_ERROR_DEVICE_LOST;
             res = vkQueueWaitIdle(q))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    for (VkQueue& q : queues)
    {
        device->reclaim_queue(family, q);
        q = VK_NULL_HANDLE;
    }
}

std::string prefix_shader(const char* src, int len, const char* defines)
{
    std::string out(src, static_cast<size_t>(len));
    const size_t pos = out.find('\n');
    if (pos == std::string::npos)
        return {};
    out.insert(pos + 1, defines);
    return out;
}

bool create_pipeline(const std::shared_ptr<VulkanContext>& ctx,
                     const char* src,
                     int len,
                     const ncnn::Option& opt,
                     uint64_t shader_variant,
                     uint64_t pipe_variant,
                     int lx,
                     std::shared_ptr<ncnn::Pipeline>& dst,
                     const std::vector<ncnn::vk_specialization_type>& specs,
                     int subgroup_size,
                     int ly,
                     int lz,
                     bool lock_create)
{
    dst = ctx->find_pipeline(src, pipe_variant);
    if (dst)
        return true;

    const auto spirv = ctx->shader_binary(src, len, opt, shader_variant);
    if (!spirv || spirv->empty())
        return false;

    std::unique_ptr<ncnn::Pipeline> pipe(new ncnn::Pipeline(ctx->device()));
    if (subgroup_size > 0)
        pipe->set_subgroup_size(subgroup_size);
    pipe->set_local_size_xyz(lx, ly, lz);

    if (lock_create)
    {
        const std::lock_guard<std::mutex> lock(ctx->command_mutex());
        dst = ctx->find_pipeline(src, pipe_variant);
        if (dst)
            return true;
        if (pipe->create(spirv->data(), spirv->size() * sizeof(uint32_t), specs) != 0)
            return false;
    }
    else
    {
        if (pipe->create(spirv->data(), spirv->size() * sizeof(uint32_t), specs) != 0)
            return false;
    }

    dst = ctx->cache_pipeline(src, pipe_variant, std::shared_ptr<ncnn::Pipeline>(pipe.release()));
    return true;
}

bool create_pipeline_prefixed(const std::shared_ptr<VulkanContext>& ctx,
                              const char* base_src,
                              int base_len,
                              const char* defines,
                              const ncnn::Option& opt,
                              uint64_t shader_variant,
                              uint64_t pipe_variant,
                              int lx,
                              std::shared_ptr<ncnn::Pipeline>& dst,
                              const std::vector<ncnn::vk_specialization_type>& specs,
                              int subgroup_size,
                              int ly,
                              int lz)
{
    dst = ctx->find_pipeline(base_src, pipe_variant);
    if (dst)
        return true;

    const std::string prefixed = prefix_shader(base_src, base_len, defines);
    if (prefixed.empty())
        return false;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(prefixed.c_str(), static_cast<int>(prefixed.size()), opt, spirv) != 0
        || spirv.empty())
    {
        return false;
    }

    std::unique_ptr<ncnn::Pipeline> pipe(new ncnn::Pipeline(ctx->device()));
    if (subgroup_size > 0)
        pipe->set_subgroup_size(subgroup_size);
    pipe->set_local_size_xyz(lx, ly, lz);

    if (pipe->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs) != 0)
        return false;

    (void)shader_variant;
    dst = ctx->cache_pipeline(base_src, pipe_variant, std::shared_ptr<ncnn::Pipeline>(pipe.release()));
    return true;
}

std::shared_ptr<ncnn::Layer> own_vk_layer(const std::shared_ptr<VulkanContext>& ctx,
                                          ncnn::Layer* layer,
                                          const ncnn::Option& opt)
{
    if (!layer)
        return {};
    return std::shared_ptr<ncnn::Layer>(layer, [ctx = ctx, opt](ncnn::Layer* l) mutable {
        ctx->retire([l, opt]() {
            l->destroy_pipeline(opt);
            delete l;
        });
        ctx.reset();
    });
}

void compute_yarn_inv_freq(uint32_t rope_dim, uint32_t active_dim, float theta,
                           float scale, float ntk_alpha, float ntk_beta,
                           uint32_t orig_ctx, float* out) noexcept
{
    const uint32_t half = rope_dim / 2;
    const uint32_t active_half = (active_dim > 0 && active_dim < rope_dim) ? (active_dim / 2) : half;
    const bool yarn = scale > 1.0f && orig_ctx > 0;

    float low = 0.0f;
    float high = 0.0f;
    if (yarn)
    {
        const float two_pi = 2.0f * 3.14159265358979323846f;
        const float log_theta = std::log(theta);
        low = std::floor(static_cast<float>(rope_dim)
                         * std::log(static_cast<float>(orig_ctx) / (ntk_beta * two_pi))
                         / (2.0f * log_theta));
        high = std::ceil(static_cast<float>(rope_dim)
                         * std::log(static_cast<float>(orig_ctx) / (ntk_alpha * two_pi))
                         / (2.0f * log_theta));
        low = std::max(low, 0.0f);
        high = std::min(high, static_cast<float>(rope_dim - 1));
        if (low == high)
            high += 0.001f;
    }

    for (uint32_t i = 0; i < half; ++i)
    {
        if (i >= active_half)
        {
            out[i] = 0.0f;
            continue;
        }

        const float exponent = static_cast<float>(2 * i) / static_cast<float>(rope_dim);
        const float freq_extra = 1.0f / std::pow(theta, exponent);
        if (!yarn)
        {
            out[i] = freq_extra;
            continue;
        }

        const float freq_inter = freq_extra / scale;
        const float linear = (static_cast<float>(i) - low) / (high - low);
        const float ramp = std::clamp(linear, 0.0f, 1.0f);
        out[i] = freq_inter * ramp + freq_extra * (1.0f - ramp);
    }
}

// --- VkTransferLease / VulkanContext -----------------------------------------

VkTransferLease::VkTransferLease(VkTransferSlot& slot, std::unique_lock<std::mutex> _lock)
    : transfer_slot(&slot),
      lock(std::move(_lock))
{
}

VulkanContext::VulkanContext(ncnn::VulkanDevice* device, VulkanRuntimePtr _runtime, uint64_t opt_flags)
    : vkdev(device),
      vulkan_runtime(std::move(_runtime)),
      flags(opt_flags)
{
    try
    {
        weight_transfer_pool = std::make_unique<VkTransferPool>(device);
        graph_allocator = std::make_unique<VkGraphAllocator>(device);
        blob_alloc = graph_allocator.get();
        kv_alloc = std::make_unique<ncnn::VkBlobAllocator>(device);
        staging_alloc = device->acquire_staging_allocator();

        for (VkTransferSlot& slot : transfer_slots)
            slot.staging_allocator = device->acquire_staging_allocator();
        for (VkTransferSlot& slot : transfer_slots)
            slot.command = new ncnn::VkCompute(device);
    }
    catch (...)
    {
        release_resources();
        throw;
    }
}

VulkanContext::~VulkanContext()
{
    const std::lock_guard<std::mutex> lock(command_lock);
    release_resources();
}

void VulkanContext::release_resources() noexcept
{
    // Destroy commands before tensors, and tensors before their allocators.
    ordered_commands.clear();
    weight_transfer_pool.reset();

    for (VkTransferSlot& slot : transfer_slots)
    {
        delete slot.command;
        slot.command = nullptr;
    }

    // Destroy deferred resources and cached pipelines before releasing allocators.
    drain_retired();
    {
        const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
        pipelines.clear();
    }

    for (VkTransferSlot& slot : transfer_slots)
    {
        slot.upload.release();
        slot.download.release();
        slot.expert_slots.release();
        slot.expert_row_offsets.release();
        slot.route_offsets.release();
        slot.route_rows.release();
        slot.route_weights.release();
        slot.rope_cosine.release();
        slot.rope_sine.release();
        slot.attention_mask.release();
        slot.attention_cache_key.release();
        slot.attention_cache_value.release();
    }

    for (VkTransferSlot& slot : transfer_slots)
    {
        if (slot.staging_allocator)
        {
            vkdev->reclaim_staging_allocator(slot.staging_allocator);
            slot.staging_allocator = nullptr;
        }
    }

    if (staging_alloc)
    {
        vkdev->reclaim_staging_allocator(staging_alloc);
        staging_alloc = nullptr;
    }

    blob_alloc = nullptr;
    cm2_scratch_buffer.release();
    kv_alloc.reset();
    graph_allocator.reset();
}

ncnn::VkMat VulkanContext::cm2_scratch(int w, int h, size_t elemsize)
{
    const uint64_t elements = h > 0 ? static_cast<uint64_t>(w) * static_cast<uint64_t>(h)
                                    : static_cast<uint64_t>(w);
    const uint64_t bytes = elements * elemsize;
    if (elements > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }

    if (cm2_scratch_buffer.empty() || cm2_scratch_buffer.buffer_capacity() < bytes)
    {
        cm2_scratch_buffer.release();
        cm2_scratch_buffer.create(static_cast<int>(bytes), sizeof(uint8_t), 1, kv_alloc.get());
        if (cm2_scratch_buffer.empty())
            return {};
    }

    ncnn::VkMat view = cm2_scratch_buffer;
    view.dims = h > 0 ? 2 : 1;
    view.w = w;
    view.h = h > 0 ? h : 1;
    view.d = 1;
    view.c = 1;
    view.elemsize = elemsize;
    view.elempack = 1;
    view.cstep = static_cast<size_t>(elements);
#if NCNN_BATCH
    view.n = 1;
    view.nstep = static_cast<size_t>(elements);
#endif
    return view;
}

bool VulkanContext::reserve_graph_compute(uint64_t bytes)
{
    return graph_allocator->reserve(bytes);
}

VulkanGraphArenaSize VulkanContext::graph_compute_physical_size() const
{
    return graph_allocator->physical_size();
}

bool VulkanContext::graph_arena_sealed() const noexcept
{
    return graph_allocator->sealed();
}

std::shared_ptr<VulkanContext> VulkanContext::find(uint32_t requested_device,
                                                   const VulkanRuntimePtr& runtime,
                                                   uint64_t opt_flags)
{
    if (!runtime)
        return {};

    const uint32_t device_index = requested_device == automatic_vulkan_device_index
                                      ? static_cast<uint32_t>(ncnn::get_default_gpu_index())
                                      : requested_device;

    const std::lock_guard<std::mutex> lock(runtime->ctx_mutex);
    const auto it = runtime->contexts.find({device_index, opt_flags});
    return it == runtime->contexts.end() ? std::shared_ptr<VulkanContext>{} : it->second.lock();
}

std::shared_ptr<VulkanContext> VulkanContext::acquire(uint32_t requested_device,
                                                      const VulkanRuntimePtr& runtime,
                                                      uint64_t opt_flags)
{
    if (!runtime)
        return {};

    {
        const std::lock_guard<std::mutex> lock(runtime->init_mutex);
        if (!runtime->init_attempted)
        {
            runtime->init_attempted = true;
#if defined(__APPLE__) && defined(NCNN_MOE_MOLTENVK_LIBRARY_PATH)
            runtime->instance_ready = ncnn::create_gpu_instance(NCNN_MOE_MOLTENVK_LIBRARY_PATH) == 0 && ncnn::get_gpu_count() > 0;
#else
            runtime->instance_ready = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
#endif
        }
    }

    if (!runtime->instance_ready)
        return {};

    const uint32_t device_index = requested_device == automatic_vulkan_device_index
                                      ? static_cast<uint32_t>(ncnn::get_default_gpu_index())
                                      : requested_device;
    if (device_index >= static_cast<uint32_t>(ncnn::get_gpu_count()))
        return {};

    ncnn::VulkanDevice* device = ncnn::get_gpu_device(static_cast<int>(device_index));
    if (!device)
        return {};

    const VulkanContextKey key{device_index, opt_flags};
    const std::lock_guard<std::mutex> lock(runtime->ctx_mutex);

    const auto it = runtime->contexts.find(key);
    if (it != runtime->contexts.end())
    {
        if (auto existing = it->second.lock())
            return existing;
        runtime->contexts.erase(it);
    }

    auto ctx = std::shared_ptr<VulkanContext>(new VulkanContext(device, runtime, opt_flags));
    runtime->contexts.emplace(key, ctx);
    return ctx;
}

std::shared_ptr<const std::vector<uint32_t>> VulkanContext::shader_binary(const char* src,
                                                                          int len,
                                                                          const ncnn::Option& opt,
                                                                          uint64_t variant)
{
    const ShaderKey key{src, variant};
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);

    const auto it = shader_binaries.find(key);
    if (it != shader_binaries.end())
        return it->second;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(src, len, opt, spirv) != 0 || spirv.empty())
    {
        auto empty = std::make_shared<const std::vector<uint32_t>>();
        shader_binaries.emplace(key, empty);
        return empty;
    }

    auto binary = std::make_shared<const std::vector<uint32_t>>(std::move(spirv));
    shader_binaries.emplace(key, binary);
    return binary;
}

std::shared_ptr<ncnn::Pipeline> VulkanContext::find_pipeline(const char* src, uint64_t variant) const
{
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
    const auto it = pipelines.find({src, variant});
    return it == pipelines.end() ? std::shared_ptr<ncnn::Pipeline>{} : it->second;
}

std::shared_ptr<ncnn::Pipeline> VulkanContext::cache_pipeline(const char* src, uint64_t variant,
                                                              std::shared_ptr<ncnn::Pipeline> pipe)
{
    std::shared_ptr<ncnn::Pipeline> out;
    {
        const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
        auto [it, inserted] = pipelines.try_emplace(ShaderKey{src, variant}, std::move(pipe));
        out = it->second;
    }
    if (pipe)
        retire([dup = std::move(pipe)]() mutable { dup.reset(); });
    return out;
}

void VulkanContext::retire(std::function<void()> fn) noexcept
{
    if (!fn)
        return;
    try
    {
        const std::lock_guard<std::mutex> lock(retire_lock);
        retired.push_back(std::move(fn));
        has_retired.store(true, std::memory_order_release);
    }
    catch (...)
    {
    }
}

void VulkanContext::drain_retired() noexcept
{
    if (!has_retired.load(std::memory_order_acquire))
        return;
    std::vector<std::function<void()>> pending;
    {
        const std::lock_guard<std::mutex> lock(retire_lock);
        pending.swap(retired);
        has_retired.store(false, std::memory_order_release);
    }
    for (auto& fn : pending)
    {
        try
        {
            fn();
        }
        catch (...)
        {
        }
    }
}

VkTransferLease VulkanContext::acquire_transfer_slot()
{
    const size_t idx = next_transfer_slot.fetch_add(1, std::memory_order_relaxed) % transfer_slots.size();
    VkTransferSlot& slot = transfer_slots[idx];

    std::unique_lock<std::mutex> lock(slot.mutex, std::try_to_lock);
    if (!lock.owns_lock())
        lock.lock();

    return VkTransferLease(slot, std::move(lock));
}

// --- Public vulkan.h entry points --------------------------------------------

VulkanRuntimePtr create_vulkan_runtime()
{
    return std::make_shared<VulkanRuntime>();
}

std::shared_ptr<void> retain_vulkan_graph_context(const VulkanRuntimePtr& runtime,
                                                  uint32_t device_index,
                                                  uint64_t opt_flags) noexcept
{
    try
    {
        return VulkanContext::acquire(device_index, runtime, opt_flags);
    }
    catch (...)
    {
        return {};
    }
}

Result<VulkanGraphArenaSize> reserve_vulkan_graph_compute(const VulkanRuntimePtr& runtime,
                                                          uint32_t device_index,
                                                          uint64_t opt_flags,
                                                          uint64_t compute_bytes)
{
    try
    {
        const auto ctx = VulkanContext::find(device_index, runtime, opt_flags);
        if (!ctx)
            return Error{ErrorCode::InternalError,
                         "Vulkan graph compute reservation has no prepared Context for device " + std::to_string(device_index)};

        if (!ctx->reserve_graph_compute(compute_bytes))
            return Error{ErrorCode::InternalError,
                         "Vulkan graph compute reservation failed for device " + std::to_string(device_index)
                             + ": " + std::to_string(compute_bytes) + " bytes"};

        return ctx->graph_compute_physical_size();
    }
    catch (const std::exception& err)
    {
        return Error{ErrorCode::InternalError,
                     "Vulkan graph compute reservation failed for device " + std::to_string(device_index)
                         + ": " + err.what()};
    }
}

VulkanGraphArenaSize query_vulkan_graph_compute(const VulkanRuntimePtr& runtime,
                                                uint32_t device_index,
                                                uint64_t opt_flags) noexcept
{
    try
    {
        const auto ctx = VulkanContext::find(device_index, runtime, opt_flags);
        return ctx ? ctx->graph_compute_physical_size() : VulkanGraphArenaSize{};
    }
    catch (...)
    {
        return {};
    }
}

uint32_t get_gpu_count() noexcept
{
    if (ncnn::create_gpu_instance() != 0)
        return 0;
    return static_cast<uint32_t>(ncnn::get_gpu_count());
}

uint32_t get_default_gpu_index() noexcept
{
    if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() == 0)
        return 0;
    return static_cast<uint32_t>(ncnn::get_default_gpu_index());
}

std::vector<GpuInfo> get_gpu_infos()
{
    std::vector<GpuInfo> infos;
    if (ncnn::create_gpu_instance() != 0)
        return infos;

    const int count = ncnn::get_gpu_count();
    infos.reserve(static_cast<size_t>(count));

    for (int i = 0; i < count; ++i)
    {
        const ncnn::GpuInfo& src = ncnn::get_gpu_info(i);
        GpuInfo info;
        info.device_index = static_cast<uint32_t>(i);
        info.vendor_id = src.vendor_id();
        info.device_id = src.device_id();
        info.device_name = src.device_name();
        info.rough_score = src.rough_score();
        info.compute_queue_count = src.compute_queue_count();
        info.transfer_queue_count = src.transfer_queue_count();

        switch (src.type())
        {
        case 0: info.type = VulkanDeviceType::Discrete; break;
        case 1: info.type = VulkanDeviceType::Integrated; break;
        case 2: info.type = VulkanDeviceType::Virtual; break;
        case 3: info.type = VulkanDeviceType::Cpu; break;
        default: info.type = VulkanDeviceType::Unknown; break;
        }

        if (src.support_fp16_storage()) info.flags |= VulkanDeviceFp16Storage;
        if (src.support_fp16_arithmetic()) info.flags |= VulkanDeviceFp16Arithmetic;
        if (src.support_bf16_storage()) info.flags |= VulkanDeviceBf16Storage;
        if (src.support_int8_storage()) info.flags |= VulkanDeviceInt8Storage;
        if (src.support_int8_arithmetic()) info.flags |= VulkanDeviceInt8Arithmetic;
        if (src.support_VK_KHR_shader_integer_dot_product()) info.flags |= VulkanDeviceIntegerDotProduct;
        if (src.support_subgroup_ops() != 0) info.flags |= VulkanDeviceSubgroupOperations;
        if (src.support_cooperative_matrix()) info.flags |= VulkanDeviceCooperativeMatrix;
        if (src.support_int8_cooperative_matrix()) info.flags |= VulkanDeviceInt8CooperativeMatrix;
        if (src.unified_compute_transfer_queue()) info.flags |= VulkanDeviceUnifiedComputeTransfer;
        if (src.resizable_bar_enabled()) info.flags |= VulkanDeviceResizableBar;

        if (ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(i))
        {
            info.heap_budget = static_cast<uint64_t>(vkdev->get_heap_budget()) * 1024 * 1024;
            info.heap_usage = static_cast<uint64_t>(vkdev->get_heap_usage()) * 1024 * 1024;
            info.heap_available = info.heap_usage < info.heap_budget ? info.heap_budget - info.heap_usage : 0;
        }

        infos.push_back(std::move(info));
    }

    return infos;
}

VulkanStatistics get_vulkan_statistics(const VulkanRuntimePtr& runtime) noexcept
{
    if (!runtime)
        return {};

    const auto& s = runtime->state;
    VulkanStatistics out;
    out.dispatches = s.dispatches.load();
    out.int4_dispatches = s.int4_dispatches.load();
    out.attention_blocks = s.attention_blocks.load();
    out.compute_submissions = s.compute_submissions.load();
    out.batch_uploads = s.batch_uploads.load();
    out.batch_downloads = s.batch_downloads.load();
    out.weight_upload_bytes = s.weight_upload_bytes.load();
    out.weight_transfer_submissions = s.weight_transfer_submissions.load();
    out.weight_visibility_submissions = s.weight_visibility_submissions.load();
    out.weight_transfer_fence_waits = s.weight_transfer_fence_waits.load();
    out.weight_transfer_wait_microseconds = s.weight_transfer_wait_microseconds.load();
    out.resident_graph_submissions = s.resident_graph_submissions.load();
    out.resident_graph_activation_upload_bytes = s.resident_graph_activation_upload_bytes.load();
    out.resident_graph_activation_download_bytes = s.resident_graph_activation_download_bytes.load();
    out.resident_graph_route_download_bytes = s.resident_graph_route_download_bytes.load();
    out.attention_qkv_rope_fusions = s.attention_qkv_rope_fusions.load();
    out.attention_device_rope_fusions = s.attention_device_rope_fusions.load();
    out.attention_qkv_ring_fusions = s.attention_qkv_ring_fusions.load();
    out.attention_decode_sdpa_fusions = s.attention_decode_sdpa_fusions.load();
    out.attention_cpu_fallbacks = s.attention_cpu_fallbacks.load();
    out.shared_expert_swiglu_fusions = s.shared_expert_swiglu_fusions.load();
    out.gated_delta_fusions = s.gated_delta_fusions.load();
    out.gated_delta_submissions = s.gated_delta_submissions.load();
    out.rms_norm_linear_fusions = s.rms_norm_linear_fusions.load();
    out.kv_ring_appends = s.kv_ring_appends.load();
    out.kv_ring_resizes = s.kv_ring_resizes.load();
    out.kv_ring_wrapped_views = s.kv_ring_wrapped_views.load();
    return out;
}

} // namespace moe
} // namespace ncnn
