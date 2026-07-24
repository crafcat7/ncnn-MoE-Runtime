#include "ncnn_linear.h"

#include "cpu_session_state.h"
#include "cpu_ops.h"
#include "ncnn_attention.h"

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#endif
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_USE_NCNN
static constexpr uint64_t max_ncnn_linear_weight_bytes = 64ull * 1024ull * 1024ull;
#endif

#if NCNN_MOE_WITH_VULKAN
static thread_local uint64_t current_vulkan_dispatch_count = 0;
static thread_local uint64_t current_vulkan_attention_block_count = 0;
static thread_local NcnnVulkanRuntimeCounters current_vulkan_runtime_counters;
#endif

#if NCNN_MOE_WITH_VULKAN
struct NcnnVulkanTransferSlot
{
    std::mutex mutex;
    ncnn::VkAllocator* staging_allocator = nullptr;
    ncnn::VkMat upload;
    ncnn::VkMat download;
    ncnn::VkMat rope_cosine;
    ncnn::VkMat rope_sine;
    ncnn::VkMat attention_mask;
};

class NcnnVulkanTransferLease
{
public:
    NcnnVulkanTransferLease(
        NcnnVulkanTransferSlot& slot,
        std::unique_lock<std::mutex> lock)
        : slot_(&slot),
          lock_(std::move(lock))
    {
    }

    NcnnVulkanTransferLease(const NcnnVulkanTransferLease&) = delete;
    NcnnVulkanTransferLease& operator=(const NcnnVulkanTransferLease&) = delete;
    NcnnVulkanTransferLease(NcnnVulkanTransferLease&&) noexcept = default;
    NcnnVulkanTransferLease& operator=(NcnnVulkanTransferLease&&) noexcept = default;

    [[nodiscard]] NcnnVulkanTransferSlot& slot() const noexcept
    {
        return *slot_;
    }

private:
    NcnnVulkanTransferSlot* slot_ = nullptr;
    std::unique_lock<std::mutex> lock_;
};

class NcnnVulkanContext
{
public:
    NcnnVulkanContext(const NcnnVulkanContext&) = delete;
    NcnnVulkanContext& operator=(const NcnnVulkanContext&) = delete;

    ~NcnnVulkanContext() = default;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanContext> acquire()
    {
        static std::mutex creation_mutex;
        static std::shared_ptr<NcnnVulkanContext> context;
        const std::lock_guard<std::mutex> lock(creation_mutex);
        if (context)
            return context;
#if defined(__APPLE__) && defined(NCNN_MOE_MOLTENVK_LIBRARY_PATH)
        if (ncnn::create_gpu_instance(NCNN_MOE_MOLTENVK_LIBRARY_PATH) != 0
            || ncnn::get_gpu_count() <= 0)
#else
        if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() <= 0)
#endif
            return {};

        ncnn::VulkanDevice* device = ncnn::get_gpu_device();
        if (!device)
            return {};
        context.reset(new NcnnVulkanContext(device));
        return context;
    }

    [[nodiscard]] ncnn::VulkanDevice* device() const noexcept
    {
        return device_;
    }

    [[nodiscard]] ncnn::VkAllocator* blob_allocator() const noexcept
    {
        return blob_allocator_;
    }

    [[nodiscard]] ncnn::VkAllocator* staging_allocator() const noexcept
    {
        return staging_allocator_;
    }

    [[nodiscard]] std::mutex& command_mutex() noexcept
    {
        return command_mutex_;
    }

    [[nodiscard]] NcnnVulkanTransferLease acquire_transfer_slot()
    {
        const size_t slot_index
            = next_transfer_slot_.fetch_add(1, std::memory_order_relaxed)
              % transfer_slots_.size();
        NcnnVulkanTransferSlot& slot = transfer_slots_[slot_index];
        std::unique_lock<std::mutex> lock(slot.mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            ++current_vulkan_runtime_counters.staging_slot_contentions;
            lock.lock();
        }
        ++current_vulkan_runtime_counters.staging_slot_acquisitions;
        return NcnnVulkanTransferLease(slot, std::move(lock));
    }

private:
    explicit NcnnVulkanContext(ncnn::VulkanDevice* device)
        : device_(device),
          blob_allocator_(device->acquire_blob_allocator()),
          staging_allocator_(device->acquire_staging_allocator())
    {
        for (NcnnVulkanTransferSlot& slot : transfer_slots_)
            slot.staging_allocator = device->acquire_staging_allocator();
    }

    // ncnn registers global Vulkan teardown with atexit. These device-owned
    // allocators must not be reclaimed here after that teardown has started.

    ncnn::VulkanDevice* device_ = nullptr;
    ncnn::VkAllocator* blob_allocator_ = nullptr;
    ncnn::VkAllocator* staging_allocator_ = nullptr;
    std::mutex command_mutex_;
    // Each slot owns a separate allocator so the next request can fill mapped
    // host memory while the other slot is referenced by a Vulkan command.
    std::array<NcnnVulkanTransferSlot, 2> transfer_slots_;
    std::atomic<size_t> next_transfer_slot_{0};
};

static bool has_batch_shape(const ncnn::VkMat& buffer, size_t rows, uint32_t columns)
{
    return buffer.dims == 2
           && buffer.w == static_cast<int>(columns)
           && buffer.h == static_cast<int>(rows)
           && buffer.elemsize == sizeof(float)
           && buffer.elempack == 1;
}

static bool prepare_staging_batch(
    ncnn::VkMat& buffer,
    size_t rows,
    uint32_t columns,
    ncnn::VkAllocator* allocator)
{
    const bool reused = has_batch_shape(buffer, rows, columns);
    buffer.create(
        static_cast<int>(columns),
        static_cast<int>(rows),
        sizeof(float),
        allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++current_vulkan_runtime_counters.staging_slot_reuses;
    else
        ++current_vulkan_runtime_counters.staging_slot_resizes;
    return true;
}

static bool prepare_staging_matrix(
    ncnn::VkMat& buffer,
    int width,
    int height,
    size_t element_size,
    ncnn::VkAllocator* allocator)
{
    const bool reused = buffer.dims == 2
                        && buffer.w == width
                        && buffer.h == height
                        && buffer.elemsize == element_size
                        && buffer.elempack == 1;
    buffer.create(width, height, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++current_vulkan_runtime_counters.staging_slot_reuses;
    else
        ++current_vulkan_runtime_counters.staging_slot_resizes;
    return true;
}

static bool prepare_staging_tensor(
    ncnn::VkMat& buffer,
    int width,
    int height,
    int channels,
    size_t element_size,
    ncnn::VkAllocator* allocator)
{
    const bool reused = buffer.dims == 3
                        && buffer.w == width
                        && buffer.h == height
                        && buffer.c == channels
                        && buffer.elemsize == element_size
                        && buffer.elempack == 1;
    buffer.create(width, height, channels, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++current_vulkan_runtime_counters.staging_slot_reuses;
    else
        ++current_vulkan_runtime_counters.staging_slot_resizes;
    return true;
}

static bool record_mapped_upload(
    ncnn::VkMat& staging,
    ncnn::VkMat& destination,
    ncnn::VkCompute& command,
    const ncnn::Option& option)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    command.record_clone(staging, destination, option);
    return !destination.empty();
}

static bool fill_staging_upload(
    const CpuBatch& input,
    ncnn::VkMat& staging,
    ncnn::VkAllocator* allocator)
{
    if (!prepare_staging_batch(
            staging,
            input.rows(),
            input.columns(),
            allocator))
        return false;

    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty())
        return false;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
        std::copy_n(
            input.row(row_index),
            input.columns(),
            mapped.row<float>(static_cast<int>(row_index)));
    }
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static bool record_prepared_staging_upload(
    const ncnn::VkMat& staging,
    size_t rows,
    ncnn::VkMat& destination,
    ncnn::VkCompute& command,
    ncnn::VulkanDevice* device,
    const ncnn::Option& option)
{
    if (staging.empty() || !staging.mapped_ptr())
        return false;
    const int packed_rows = static_cast<int>(rows);
    const int destination_elempack = packed_rows % 4 == 0 ? 4 : 1;
    int cast_type = 0;
    if (device->info.type() != 0) {
        if (option.use_bf16_storage || option.use_bf16_packed)
            cast_type = 5;
        else if (option.use_fp16_storage || option.use_fp16_packed)
            cast_type = 2;
        else
            cast_type = 1;
    }
    device->convert_packing(
        staging,
        destination,
        destination_elempack,
        cast_type,
        command,
        option);
    return !destination.empty();
}

static bool record_prepared_staging_download(
    const ncnn::VkMat& source,
    size_t rows,
    uint32_t columns,
    ncnn::VkMat& staging,
    ncnn::VkCompute& command,
    const ncnn::Option& option)
{
    if (!has_batch_shape(staging, rows, columns))
        return false;
    ncnn::Option staging_option = option;
    staging_option.blob_vkallocator = staging.allocator;
    staging_option.workspace_vkallocator = staging.allocator;
    staging_option.staging_vkallocator = staging.allocator;
    command.record_clone(source, staging, staging_option);
    return !staging.empty();
}

static bool copy_staging_to_cpu_batch(
    ncnn::VkMat& staging,
    CpuBatch& output)
{
    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();
    if (mapped.empty()
        || mapped.dims != 2
        || mapped.w != static_cast<int>(output.columns())
        || mapped.h != static_cast<int>(output.rows())
        || mapped.elempack != 1
        || mapped.elembits() != 32)
        return false;
    for (size_t row_index = 0; row_index < output.rows(); ++row_index) {
        std::copy_n(
            mapped.row<float>(static_cast<int>(row_index)),
            output.columns(),
            output.row(row_index));
    }
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}
#endif

class NcnnLinearOperator::Implementation
{
public:
#if NCNN_MOE_USE_NCNN
    ~Implementation()
    {
        if (layer) {
#if NCNN_MOE_WITH_VULKAN
            if (vulkan_context) {
                const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
                if (pipeline_created)
                    layer->destroy_pipeline(option);
            }
            else
#endif
            if (pipeline_created) {
                layer->destroy_pipeline(option);
            }
            delete layer;
        }
#if NCNN_MOE_WITH_VULKAN
        weight_staging_allocator.reset();
        weight_allocator.reset();
#endif
    }

    ncnn::Layer* layer = nullptr;
    ncnn::Option option;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool pipeline_created = false;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
#endif
#endif
};

class NcnnVulkanAttentionCache
{
public:
#if NCNN_MOE_WITH_VULKAN
    ncnn::VkMat key;
    ncnn::VkMat value;
#endif
};

class NcnnVulkanAttentionOperator::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    ~Implementation()
    {
        if (vulkan_context) {
            const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
            for (ncnn::Layer* layer : layers) {
                layer->destroy_pipeline(option);
                delete layer;
            }
        }
        layers.clear();
        zero_key_value.release();
        weight_staging_allocator.reset();
        weight_allocator.reset();
    }

    ncnn::Layer* norm = nullptr;
    ncnn::Layer* slice_qkv = nullptr;
    ncnn::Layer* reshape_query = nullptr;
    ncnn::Layer* reshape_key_value = nullptr;
    ncnn::Layer* permute_heads_tokens = nullptr;
    ncnn::Layer* rotary = nullptr;
    ncnn::Layer* concat_sequence = nullptr;
    ncnn::Layer* compact_sliding_cache = nullptr;
    ncnn::Layer* sdpa = nullptr;
    ncnn::Layer* reshape_attention = nullptr;
    ncnn::Layer* add = nullptr;
    std::vector<ncnn::Layer*> layers;
    ncnn::Option option;
    std::shared_ptr<NcnnVulkanContext> vulkan_context;
    ncnn::VkMat zero_key_value;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
#endif
    std::shared_ptr<NcnnLinearOperator> fused_qkv;
    std::shared_ptr<NcnnLinearOperator> output_projection;
    NcnnVulkanAttentionConfig config;
    std::vector<float> sinks;
    std::vector<float> rope_inverse_frequencies;
    float rope_concentration = 1.0f;
};

NcnnLinearOperator::NcnnLinearOperator()
    : implementation_(new Implementation)
{
}

NcnnLinearOperator::~NcnnLinearOperator() = default;

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create(
    const TensorData& matrix,
    const TensorData* bias,
    NcnnLinearDevice device)
{
#if NCNN_MOE_USE_NCNN
    if (matrix.shape.size() != 2
        || (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16))
        return {};

    const uint64_t element_size = matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
    if (device == NcnnLinearDevice::Cpu
        && matrix.element_count() > max_ncnn_linear_weight_bytes / element_size)
        return {};
    if (matrix.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};

    std::shared_ptr<NcnnLinearOperator> linear(new NcnnLinearOperator);
    Implementation& implementation = *linear->implementation_;
    implementation.input_columns = matrix.shape[1];
    implementation.output_columns = matrix.shape[0];
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;

#if NCNN_MOE_WITH_VULKAN
    if (device == NcnnLinearDevice::Vulkan) {
        implementation.vulkan_context = NcnnVulkanContext::acquire();
        if (!implementation.vulkan_context)
            return {};

        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        implementation.option.use_vulkan_compute = true;
        implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
        implementation.option.use_cooperative_matrix = vkdev->info.support_cooperative_matrix();
        implementation.option.use_subgroup_ops = vkdev->info.support_subgroup_ops();
        implementation.layer = ncnn::create_layer_vulkan(ncnn::LayerType::InnerProduct);
        if (implementation.layer)
            implementation.layer->vkdev = vkdev;
    }
    else
#endif
    {
        implementation.layer = ncnn::create_layer_cpu(ncnn::LayerType::InnerProduct);
    }
    if (!implementation.layer)
        return {};

    ncnn::ParamDict parameters;
    parameters.set(0, static_cast<int>(implementation.output_columns));
    parameters.set(1, bias ? 1 : 0);
    parameters.set(2, static_cast<int>(matrix.element_count()));
    if (implementation.layer->load_param(parameters) != 0)
        return {};

    ncnn::Mat model_data[2];
    model_data[0].create(static_cast<int>(matrix.element_count()), sizeof(float));
    if (model_data[0].empty())
        return {};
    float* weight_data = static_cast<float*>(model_data[0].data);
    for (size_t index = 0; index < matrix.element_count(); ++index)
        weight_data[index] = matrix.dtype == DType::Float32
                                 ? matrix.float32_data[index]
                                 : bfloat16_to_float(matrix.bfloat16_data[index]);

    if (bias) {
        model_data[1].create(static_cast<int>(implementation.output_columns), sizeof(float));
        if (model_data[1].empty())
            return {};
        float* bias_data = static_cast<float*>(model_data[1].data);
        for (uint32_t column = 0; column < implementation.output_columns; ++column) {
            bias_data[column] = bias->dtype == DType::Float32
                                    ? bias->float32_data[column]
                                    : bfloat16_to_float(bias->bfloat16_data[column]);
        }
    }

    if (implementation.layer->load_model(ncnn::ModelBinFromMatArray(model_data)) != 0
        || implementation.layer->create_pipeline(implementation.option) != 0)
        return {};
    implementation.pipeline_created = true;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context) {
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(vkdev);
        ncnn::Option upload_option = implementation.option;
        upload_option.blob_vkallocator = implementation.weight_allocator.get();
        upload_option.workspace_vkallocator = implementation.weight_allocator.get();
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        if (implementation.layer->upload_model(command, upload_option) != 0
            || command.submit_and_wait() != 0)
            return {};
    }
#endif
    return linear;
#else
    (void)matrix;
    (void)bias;
    (void)device;
    return {};
#endif
}

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create_fused(
    const std::vector<const TensorData*>& matrices,
    const std::vector<const TensorData*>& biases,
    NcnnLinearDevice device)
{
    if (matrices.empty() || matrices.size() != biases.size() || !matrices.front())
        return {};

    const DType dtype = matrices.front()->dtype;
    if (matrices.front()->shape.size() != 2
        || (dtype != DType::Float32 && dtype != DType::BFloat16))
        return {};
    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index) {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix || matrix->dtype != dtype || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns || (bias != nullptr) != has_bias)
            return {};
        if (bias
            && (bias->dtype != dtype || bias->shape.size() != 1
                || bias->shape[0] != matrix->shape[0]))
            return {};
        output_columns += matrix->shape[0];
        element_count += matrix->element_count();
    }
    if (output_columns > std::numeric_limits<uint32_t>::max())
        return {};

    TensorData fused_matrix;
    fused_matrix.dtype = dtype;
    fused_matrix.shape = {static_cast<uint32_t>(output_columns), input_columns};
    TensorData fused_bias;
    fused_bias.dtype = dtype;
    fused_bias.shape = {static_cast<uint32_t>(output_columns)};
    if (dtype == DType::Float32) {
        fused_matrix.float32_data.reserve(element_count);
        if (has_bias)
            fused_bias.float32_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index) {
            fused_matrix.float32_data.insert(
                fused_matrix.float32_data.end(),
                matrices[index]->float32_data.begin(),
                matrices[index]->float32_data.end());
            if (has_bias) {
                fused_bias.float32_data.insert(
                    fused_bias.float32_data.end(),
                    biases[index]->float32_data.begin(),
                    biases[index]->float32_data.end());
            }
        }
    }
    else {
        fused_matrix.bfloat16_data.reserve(element_count);
        if (has_bias)
            fused_bias.bfloat16_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index) {
            fused_matrix.bfloat16_data.insert(
                fused_matrix.bfloat16_data.end(),
                matrices[index]->bfloat16_data.begin(),
                matrices[index]->bfloat16_data.end());
            if (has_bias) {
                fused_bias.bfloat16_data.insert(
                    fused_bias.bfloat16_data.end(),
                    biases[index]->bfloat16_data.begin(),
                    biases[index]->bfloat16_data.end());
            }
        }
    }
    return create(fused_matrix, has_bias ? &fused_bias : nullptr, device);
}

uint32_t NcnnLinearOperator::vulkan_device_count() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    const std::shared_ptr<NcnnVulkanContext> context = NcnnVulkanContext::acquire();
    return context ? static_cast<uint32_t>(ncnn::get_gpu_count()) : 0;
#else
    return 0;
#endif
}

uint64_t NcnnLinearOperator::current_thread_vulkan_dispatches() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return current_vulkan_dispatch_count;
#else
    return 0;
#endif
}

NcnnVulkanRuntimeCounters NcnnLinearOperator::current_thread_vulkan_runtime_counters() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return current_vulkan_runtime_counters;
#else
    return {};
#endif
}

bool NcnnLinearOperator::forward(const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_USE_NCNN
    const Implementation& implementation = *implementation_;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context) {
        NcnnVulkanTransferLease transfer_lease
            = implementation.vulkan_context->acquire_transfer_slot();
        NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
        if (!fill_staging_upload(
                input,
                transfer_slot.upload,
                transfer_slot.staging_allocator)
            || !prepare_staging_batch(
                transfer_slot.download,
                input.rows(),
                implementation.output_columns,
                transfer_slot.staging_allocator))
            return false;

        output = CpuBatch(input.rows(), implementation.output_columns);
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        ncnn::VkCompute command(vkdev);
        ncnn::VkMat bottom_gpu;
        if (!record_prepared_staging_upload(
                transfer_slot.upload,
                input.rows(),
                bottom_gpu,
                command,
                vkdev,
                implementation.option))
            return false;

        ncnn::VkMat top_gpu;
        if (implementation.layer->forward(
                bottom_gpu,
                top_gpu,
                command,
                implementation.option)
            != 0)
            return false;
        ncnn::VkMat download_gpu = top_gpu;
        if (top_gpu.elempack != 1) {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                top_gpu,
                unpacked,
                1,
                command,
                implementation.option);
            download_gpu = unpacked;
        }
        if (!record_prepared_staging_download(
                download_gpu,
                input.rows(),
                implementation.output_columns,
                transfer_slot.download,
                command,
                implementation.option))
            return false;
        if (command.submit_and_wait() != 0
            || !copy_staging_to_cpu_batch(transfer_slot.download, output))
            return false;
        ++current_vulkan_dispatch_count;
        ++current_vulkan_runtime_counters.compute_submissions;
        ++current_vulkan_runtime_counters.batch_uploads;
        ++current_vulkan_runtime_counters.batch_downloads;
        return true;
    }
#endif

    ncnn::Mat bottom(
        static_cast<int>(input.columns()),
        static_cast<int>(input.rows()),
        sizeof(float));
    if (bottom.empty())
        return false;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        std::copy_n(input.row(row_index), input.columns(), bottom.row<float>(static_cast<int>(row_index)));

    ncnn::Mat top;
    if (implementation.layer->forward(bottom, top, implementation.option) != 0 || top.empty()
        || top.total() * top.elempack != input.rows() * implementation.output_columns)
        return false;
    output = CpuBatch(input.rows(), implementation.output_columns);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
        const float* source = top.row<float>(static_cast<int>(row_index));
        std::copy_n(source, implementation.output_columns, output.row(row_index));
    }
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool NcnnLinearOperator::uses_vulkan() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return implementation_->vulkan_context != nullptr;
#else
    return false;
#endif
}

NcnnVulkanAttentionOperator::NcnnVulkanAttentionOperator()
    : implementation_(new Implementation)
{
}

NcnnVulkanAttentionOperator::~NcnnVulkanAttentionOperator() = default;

uint64_t NcnnVulkanAttentionOperator::current_thread_blocks() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return current_vulkan_attention_block_count;
#else
    return 0;
#endif
}

#if NCNN_MOE_WITH_VULKAN
static bool tensor_to_float_vector(const TensorData& tensor, std::vector<float>& values)
{
    if (tensor.dtype != DType::Float32 && tensor.dtype != DType::BFloat16)
        return false;
    values.resize(tensor.element_count());
    if (tensor.dtype == DType::Float32) {
        if (tensor.float32_data.size() != values.size())
            return false;
        values = tensor.float32_data;
    }
    else {
        if (tensor.bfloat16_data.size() != values.size())
            return false;
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = bfloat16_to_float(tensor.bfloat16_data[index]);
    }
    return true;
}

static bool fill_rope_staging_pair(
    ncnn::VkMat& cosine_staging,
    ncnn::VkMat& sine_staging,
    size_t token_count,
    uint64_t position_offset,
    const std::vector<float>& inverse_frequencies,
    float concentration,
    bool bfloat16_storage,
    ncnn::VkAllocator* allocator)
{
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || inverse_frequencies.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_matrix(
            cosine_staging,
            static_cast<int>(inverse_frequencies.size()),
            static_cast<int>(token_count),
            bfloat16_storage ? sizeof(uint16_t) : sizeof(float),
            allocator)
        || !prepare_staging_matrix(
            sine_staging,
            static_cast<int>(inverse_frequencies.size()),
            static_cast<int>(token_count),
            bfloat16_storage ? sizeof(uint16_t) : sizeof(float),
            allocator))
        return false;

    ncnn::Mat cosine_mapped = cosine_staging.mapped();
    ncnn::Mat sine_mapped = sine_staging.mapped();
    if (cosine_mapped.empty() || sine_mapped.empty())
        return false;
    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        float* cosine_float_row = bfloat16_storage
                                      ? nullptr
                                      : cosine_mapped.row<float>(static_cast<int>(token_index));
        float* sine_float_row = bfloat16_storage
                                    ? nullptr
                                    : sine_mapped.row<float>(static_cast<int>(token_index));
        uint16_t* cosine_bfloat16_row = bfloat16_storage
                                           ? cosine_mapped.row<uint16_t>(static_cast<int>(token_index))
                                           : nullptr;
        uint16_t* sine_bfloat16_row = bfloat16_storage
                                         ? sine_mapped.row<uint16_t>(static_cast<int>(token_index))
                                         : nullptr;
        for (size_t index = 0; index < inverse_frequencies.size(); ++index) {
            const float angle = static_cast<float>(position_offset + token_index)
                                * inverse_frequencies[index];
            const float cosine = std::cos(angle) * concentration;
            const float sine = std::sin(angle) * concentration;
            if (bfloat16_storage) {
                cosine_bfloat16_row[index] = float_to_bfloat16(cosine);
                sine_bfloat16_row[index] = float_to_bfloat16(sine);
            }
            else {
                cosine_float_row[index] = cosine;
                sine_float_row[index] = sine;
            }
        }
    }
    return true;
}

static bool fill_attention_mask_staging(
    ncnn::VkMat& staging,
    size_t token_count,
    uint64_t destination_count,
    uint64_t position_offset,
    const CpuLayerCache& cache,
    const NcnnVulkanAttentionConfig& config,
    const std::vector<float>& sinks,
    bool use_attention_sink,
    bool bfloat16_storage,
    ncnn::VkAllocator* allocator)
{
    if (destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_tensor(
            staging,
            static_cast<int>(destination_count),
            static_cast<int>(token_count),
            static_cast<int>(config.head_count),
            bfloat16_storage ? sizeof(uint16_t) : sizeof(float),
            allocator))
        return false;

    // A finite sentinel avoids NaNs in BF16 flash-attention implementations
    // while still underflowing to an exact zero probability after softmax.
    constexpr float masked_logit = -10000.0f;
    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty())
        return false;
    const uint64_t actual_end = cache.token_count + token_count;
    ncnn::Mat first_head = mapped.channel(0);
    for (size_t query_index = 0; query_index < token_count; ++query_index) {
        const uint64_t query_position = position_offset + query_index;
        if (bfloat16_storage) {
            uint16_t* row = first_head.row<uint16_t>(static_cast<int>(query_index));
            const uint16_t masked_value = float_to_bfloat16(masked_logit);
            for (uint64_t key_index = 0; key_index < actual_end; ++key_index) {
                const uint64_t key_position = key_index < cache.token_count
                                                  ? cache.start_position + key_index
                                                  : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0
                                     && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_value : 0;
            }
            if (use_attention_sink)
                row[actual_end] = float_to_bfloat16(sinks[0]);
            std::fill(
                row + actual_end + (use_attention_sink ? 1 : 0),
                row + destination_count,
                masked_value);
        }
        else {
            float* row = first_head.row<float>(static_cast<int>(query_index));
            for (uint64_t key_index = 0; key_index < actual_end; ++key_index) {
                const uint64_t key_position = key_index < cache.token_count
                                                  ? cache.start_position + key_index
                                                  : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0
                                     && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_logit : 0.0f;
            }
            if (use_attention_sink)
                row[actual_end] = sinks[0];
            std::fill(
                row + actual_end + (use_attention_sink ? 1 : 0),
                row + destination_count,
                masked_logit);
        }
    }
    for (uint32_t head = 1; head < config.head_count; ++head) {
        ncnn::Mat head_mask = mapped.channel(static_cast<int>(head));
        for (size_t query_index = 0; query_index < token_count; ++query_index) {
            if (bfloat16_storage) {
                const uint16_t* source
                    = first_head.row<uint16_t>(static_cast<int>(query_index));
                uint16_t* row = head_mask.row<uint16_t>(static_cast<int>(query_index));
                std::copy_n(source, destination_count, row);
                if (use_attention_sink)
                    row[actual_end] = float_to_bfloat16(sinks[head]);
            }
            else {
                const float* source = first_head.row<float>(static_cast<int>(query_index));
                float* row = head_mask.row<float>(static_cast<int>(query_index));
                std::copy_n(source, destination_count, row);
                if (use_attention_sink)
                    row[actual_end] = sinks[head];
            }
        }
    }
    return true;
}
#endif

std::shared_ptr<NcnnVulkanAttentionOperator> NcnnVulkanAttentionOperator::create(
    const TensorData& norm_weight,
    const TensorData* sinks,
    std::shared_ptr<NcnnLinearOperator> fused_qkv,
    std::shared_ptr<NcnnLinearOperator> output_projection,
    const NcnnVulkanAttentionConfig& config)
{
#if NCNN_MOE_WITH_VULKAN
    if (!fused_qkv || !output_projection || !fused_qkv->uses_vulkan()
        || !output_projection->uses_vulkan() || config.hidden_size == 0
        || config.head_count == 0 || config.kv_head_count == 0
        || config.head_dimension == 0 || config.head_dimension % 2 != 0
        || config.head_count % config.kv_head_count != 0
        || config.activation_dtype != config.kv_cache_dtype
        || (!config.use_attention_sink && config.sliding_window > 0)
        || norm_weight.shape != std::vector<uint32_t>{config.hidden_size}
        || (config.use_attention_sink
            && (!sinks
                || sinks->shape != std::vector<uint32_t>{config.head_count})))
        return {};

    const NcnnLinearOperator::Implementation& fused_implementation = *fused_qkv->implementation_;
    const NcnnLinearOperator::Implementation& output_implementation = *output_projection->implementation_;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    if (!fused_implementation.layer || !output_implementation.layer
        || fused_implementation.input_columns != config.hidden_size
        || fused_implementation.output_columns != query_columns + 2 * key_value_columns
        || output_implementation.input_columns != query_columns
        || output_implementation.output_columns != config.hidden_size
        || fused_implementation.vulkan_context != output_implementation.vulkan_context)
        return {};

    std::shared_ptr<NcnnVulkanAttentionOperator> attention(new NcnnVulkanAttentionOperator);
    Implementation& implementation = *attention->implementation_;
    implementation.fused_qkv = std::move(fused_qkv);
    implementation.output_projection = std::move(output_projection);
    implementation.config = config;
    if (config.use_attention_sink
        && !tensor_to_float_vector(*sinks, implementation.sinks))
        return {};
    implementation.vulkan_context = fused_implementation.vulkan_context;
    implementation.option = fused_implementation.option;
    // SDPA consumes unpacked [head, token, dimension] tensors. Keeping this
    // subgraph at pack1 avoids GQA head channels being silently packed as four.
    implementation.option.use_packing_layout = false;
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();

    const uint32_t half_dimension = config.head_dimension / 2;
    implementation.rope_inverse_frequencies.resize(half_dimension);
    float rope_low = 0.0f;
    float rope_high = 0.0f;
    if (config.rope_scaling_factor > 1.0f) {
        implementation.rope_concentration
            = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(half_dimension);
        rope_low = half * std::log(
                              static_cast<float>(config.initial_context_length)
                              / (config.rope_ntk_beta * 2.0f * std::acos(-1.0f)))
                   / std::log(config.rope_theta);
        rope_high = half * std::log(
                               static_cast<float>(config.initial_context_length)
                               / (config.rope_ntk_alpha * 2.0f * std::acos(-1.0f)))
                    / std::log(config.rope_theta);
    }
    for (uint32_t index = 0; index < half_dimension; ++index) {
        const float frequency = std::pow(
            config.rope_theta,
            static_cast<float>(2 * index) / static_cast<float>(config.head_dimension));
        float inverse_frequency = 1.0f / frequency;
        if (config.rope_scaling_factor > 1.0f) {
            const float ramp = std::clamp(
                (static_cast<float>(index) - rope_low) / (rope_high - rope_low),
                0.0f,
                1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation = 1.0f / (config.rope_scaling_factor * frequency);
            inverse_frequency = interpolation * (1.0f - mask) + inverse_frequency * mask;
        }
        implementation.rope_inverse_frequencies[index] = inverse_frequency;
    }

    auto create_layer = [&](int type, const ncnn::ParamDict& parameters, ncnn::Layer*& destination) {
        ncnn::Layer* layer = ncnn::create_layer_vulkan(type);
        if (!layer)
            return false;
        layer->vkdev = vkdev;
        if (layer->load_param(parameters) != 0
            || layer->create_pipeline(implementation.option) != 0) {
            delete layer;
            return false;
        }
        implementation.layers.push_back(layer);
        destination = layer;
        return true;
    };

    std::vector<float> norm_values;
    if (!tensor_to_float_vector(norm_weight, norm_values))
        return {};
    ncnn::Layer* norm = ncnn::create_layer_vulkan(ncnn::LayerType::RMSNorm);
    if (!norm)
        return {};
    norm->vkdev = vkdev;
    ncnn::ParamDict norm_parameters;
    norm_parameters.set(0, static_cast<int>(config.hidden_size));
    norm_parameters.set(1, config.norm_epsilon);
    norm_parameters.set(2, 1);
    ncnn::Mat norm_model[1] = {
        ncnn::Mat(static_cast<int>(norm_values.size()), norm_values.data(), sizeof(float))};
    if (norm->load_param(norm_parameters) != 0
        || norm->load_model(ncnn::ModelBinFromMatArray(norm_model)) != 0
        || norm->create_pipeline(implementation.option) != 0) {
        delete norm;
        return {};
    }
    implementation.layers.push_back(norm);
    implementation.norm = norm;

    ncnn::ParamDict slice_parameters;
    ncnn::Mat slice_sizes(3, sizeof(int));
    int* slice_values = static_cast<int*>(slice_sizes.data);
    slice_values[0] = static_cast<int>(query_columns);
    slice_values[1] = static_cast<int>(key_value_columns);
    slice_values[2] = static_cast<int>(key_value_columns);
    slice_parameters.set(0, slice_sizes);
    slice_parameters.set(1, 1);
    if (!create_layer(ncnn::LayerType::Slice, slice_parameters, implementation.slice_qkv))
        return {};

    ncnn::ParamDict reshape_query_parameters;
    reshape_query_parameters.set(0, static_cast<int>(config.head_dimension));
    reshape_query_parameters.set(1, static_cast<int>(config.head_count));
    reshape_query_parameters.set(2, -1);
    if (!create_layer(
            ncnn::LayerType::Reshape,
            reshape_query_parameters,
            implementation.reshape_query))
        return {};

    ncnn::ParamDict reshape_key_value_parameters;
    reshape_key_value_parameters.set(0, static_cast<int>(config.head_dimension));
    reshape_key_value_parameters.set(1, static_cast<int>(config.kv_head_count));
    reshape_key_value_parameters.set(2, -1);
    if (!create_layer(
            ncnn::LayerType::Reshape,
            reshape_key_value_parameters,
            implementation.reshape_key_value))
        return {};

    ncnn::ParamDict permute_parameters;
    permute_parameters.set(0, 2);
    if (!create_layer(
            ncnn::LayerType::Permute,
            permute_parameters,
            implementation.permute_heads_tokens))
        return {};

    ncnn::ParamDict rotary_parameters;
    rotary_parameters.set(0, 0);
    if (!create_layer(ncnn::LayerType::RotaryEmbed, rotary_parameters, implementation.rotary))
        return {};

    ncnn::ParamDict concat_parameters;
    concat_parameters.set(0, 1);
    if (!create_layer(
            ncnn::LayerType::Concat,
            concat_parameters,
            implementation.concat_sequence))
        return {};

    if (config.sliding_window > 1) {
        ncnn::ParamDict compact_cache_parameters;
        ncnn::Mat compact_cache_indices(2, sizeof(int));
        int* compact_cache_values = static_cast<int*>(compact_cache_indices.data);
        compact_cache_values[0] = -static_cast<int>(config.sliding_window);
        compact_cache_values[1] = -1;
        compact_cache_parameters.set(1, 1);
        compact_cache_parameters.set(2, compact_cache_indices);
        if (!create_layer(
                ncnn::LayerType::Slice,
                compact_cache_parameters,
                implementation.compact_sliding_cache))
            return {};
    }

    ncnn::ParamDict sdpa_parameters;
    sdpa_parameters.set(5, 1);
    sdpa_parameters.set(6, 1.0f / std::sqrt(static_cast<float>(config.head_dimension)));
    sdpa_parameters.set(7, 0);
    if (!create_layer(ncnn::LayerType::SDPA, sdpa_parameters, implementation.sdpa))
        return {};

    ncnn::ParamDict reshape_attention_parameters;
    reshape_attention_parameters.set(0, static_cast<int>(query_columns));
    reshape_attention_parameters.set(1, -1);
    if (!create_layer(
            ncnn::LayerType::Reshape,
            reshape_attention_parameters,
            implementation.reshape_attention))
        return {};

    ncnn::ParamDict add_parameters;
    add_parameters.set(0, 0);
    if (!create_layer(ncnn::LayerType::BinaryOp, add_parameters, implementation.add))
        return {};

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkTransfer command(vkdev);
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16
                                  && implementation.option.use_bf16_storage;
    if (config.use_attention_sink) {
        ncnn::Mat zero_key_value(
            static_cast<int>(config.head_dimension),
            1,
            static_cast<int>(config.kv_head_count),
            bfloat16_storage ? sizeof(uint16_t) : sizeof(float));
        if (zero_key_value.empty())
            return {};
        if (bfloat16_storage)
            zero_key_value.fill<uint16_t>(0);
        else
            zero_key_value.fill(0.0f);
        command.record_upload(
            zero_key_value,
            implementation.zero_key_value,
            upload_option,
            false);
    }
    if ((config.use_attention_sink && implementation.zero_key_value.empty())
        || implementation.norm->upload_model(command, upload_option) != 0
        || command.submit_and_wait() != 0)
        return {};
    return attention;
#else
    (void)norm_weight;
    (void)sinks;
    (void)fused_qkv;
    (void)output_projection;
    (void)config;
    return {};
#endif
}

bool NcnnVulkanAttentionOperator::forward(
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input,
    CpuBatch& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *implementation_;
    const NcnnVulkanAttentionConfig& config = implementation.config;
    if (input.rows() == 0 || input.columns() != config.hidden_size
        || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || (cache.dtype != config.kv_cache_dtype && cache.token_count != 0)
        || (cache.token_count != 0 && !cache.vulkan_attention_cache))
        return false;

    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16
                                  && implementation.option.use_bf16_storage;
    const uint64_t actual_token_count = cache.token_count + input.rows();
    const uint64_t sink_token_count = config.use_attention_sink ? 1 : 0;
    const uint64_t destination_count = actual_token_count + sink_token_count;

    const NcnnLinearOperator::Implementation& fused = *implementation.fused_qkv->implementation_;
    const NcnnLinearOperator::Implementation& projection = *implementation.output_projection->implementation_;
    NcnnVulkanTransferLease transfer_lease
        = implementation.vulkan_context->acquire_transfer_slot();
    NcnnVulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator)
        || !fill_rope_staging_pair(
            transfer_slot.rope_cosine,
            transfer_slot.rope_sine,
            input.rows(),
            position_offset,
            implementation.rope_inverse_frequencies,
            implementation.rope_concentration,
            bfloat16_storage,
            transfer_slot.staging_allocator)
        || !fill_attention_mask_staging(
            transfer_slot.attention_mask,
            input.rows(),
            destination_count,
            position_offset,
            cache,
            config,
            implementation.sinks,
            config.use_attention_sink,
            bfloat16_storage,
            transfer_slot.staging_allocator)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            config.hidden_size,
            transfer_slot.staging_allocator))
        return false;

    const std::lock_guard<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute command(vkdev);
    ncnn::VkMat input_gpu;
    ncnn::VkMat cosine_gpu;
    ncnn::VkMat sine_gpu;
    ncnn::VkMat mask_gpu;
    if (!record_prepared_staging_upload(
            transfer_slot.upload,
            input.rows(),
            input_gpu,
            command,
            vkdev,
            implementation.option))
        return false;
    if (!record_mapped_upload(
            transfer_slot.rope_cosine,
            cosine_gpu,
            command,
            implementation.option)
        || !record_mapped_upload(
            transfer_slot.rope_sine,
            sine_gpu,
            command,
            implementation.option)
        || !record_mapped_upload(
            transfer_slot.attention_mask,
            mask_gpu,
            command,
            implementation.option))
        return false;
    ncnn::VkMat normalized_gpu;
    if (implementation.norm->forward(
            input_gpu, normalized_gpu, command, implementation.option)
        != 0)
        return false;

    ncnn::VkMat fused_gpu;
    if (fused.layer->forward(
            normalized_gpu, fused_gpu, command, implementation.option)
        != 0)
        return false;
    std::vector<ncnn::VkMat> qkv_input(1, fused_gpu);
    std::vector<ncnn::VkMat> qkv(3);
    if (implementation.slice_qkv->forward(
            qkv_input, qkv, command, implementation.option)
        != 0)
        return false;

    ncnn::VkMat query_shaped;
    ncnn::VkMat key_shaped;
    ncnn::VkMat value_shaped;
    if (implementation.reshape_query->forward(
            qkv[0], query_shaped, command, implementation.option)
            != 0
        || implementation.reshape_key_value->forward(
               qkv[1], key_shaped, command, implementation.option)
               != 0
        || implementation.reshape_key_value->forward(
               qkv[2], value_shaped, command, implementation.option)
               != 0)
        return false;

    ncnn::VkMat query_heads;
    ncnn::VkMat key_heads;
    ncnn::VkMat value_heads;
    if (implementation.permute_heads_tokens->forward(
            query_shaped, query_heads, command, implementation.option)
            != 0
        || implementation.permute_heads_tokens->forward(
               key_shaped, key_heads, command, implementation.option)
               != 0
        || implementation.permute_heads_tokens->forward(
               value_shaped, value_heads, command, implementation.option)
               != 0)
        return false;
    if (query_heads.elempack != 1) {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(query_heads, unpacked, 1, command, implementation.option);
        query_heads = unpacked;
    }
    if (key_heads.elempack != 1) {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(key_heads, unpacked, 1, command, implementation.option);
        key_heads = unpacked;
    }
    if (value_heads.elempack != 1) {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(value_heads, unpacked, 1, command, implementation.option);
        value_heads = unpacked;
    }

    std::vector<ncnn::VkMat> query_rope_input = {query_heads, cosine_gpu, sine_gpu};
    std::vector<ncnn::VkMat> key_rope_input = {key_heads, cosine_gpu, sine_gpu};
    std::vector<ncnn::VkMat> query_rope_output(1);
    std::vector<ncnn::VkMat> key_rope_output(1);
    if (implementation.rotary->forward(
            query_rope_input, query_rope_output, command, implementation.option)
            != 0
        || implementation.rotary->forward(
               key_rope_input, key_rope_output, command, implementation.option)
               != 0)
        return false;

    std::vector<ncnn::VkMat> key_with_sink_output = {key_rope_output[0]};
    std::vector<ncnn::VkMat> value_with_sink_output = {value_heads};
    if (config.use_attention_sink) {
        std::vector<ncnn::VkMat> key_with_sink_input = {
            key_rope_output[0],
            implementation.zero_key_value,
        };
        std::vector<ncnn::VkMat> value_with_sink_input = {
            value_heads,
            implementation.zero_key_value,
        };
        if (implementation.concat_sequence->forward(
                key_with_sink_input,
                key_with_sink_output,
                command,
                implementation.option)
                != 0
            || implementation.concat_sequence->forward(
                   value_with_sink_input,
                   value_with_sink_output,
                   command,
                   implementation.option)
                   != 0)
            return false;
    }
    if (query_rope_output[0].elempack != 1
        || key_with_sink_output[0].elempack != 1
        || value_with_sink_output[0].elempack != 1
        || query_rope_output[0].dims != 3
        || query_rope_output[0].w != static_cast<int>(config.head_dimension)
        || query_rope_output[0].h != static_cast<int>(input.rows())
        || query_rope_output[0].c != static_cast<int>(config.head_count)
        || key_with_sink_output[0].dims != 3
        || key_with_sink_output[0].w != static_cast<int>(config.head_dimension)
        || key_with_sink_output[0].h != static_cast<int>(input.rows() + sink_token_count)
        || key_with_sink_output[0].c != static_cast<int>(config.kv_head_count)
        || value_with_sink_output[0].dims != 3
        || value_with_sink_output[0].w != static_cast<int>(config.head_dimension)
        || value_with_sink_output[0].h != static_cast<int>(input.rows() + sink_token_count)
        || value_with_sink_output[0].c != static_cast<int>(config.kv_head_count))
        return false;

    ncnn::VkMat combined_key = key_with_sink_output[0];
    ncnn::VkMat combined_value = value_with_sink_output[0];
    if (cache.token_count != 0) {
        std::vector<ncnn::VkMat> combined_key_input = {
            cache.vulkan_attention_cache->key,
            combined_key,
        };
        std::vector<ncnn::VkMat> combined_value_input = {
            cache.vulkan_attention_cache->value,
            combined_value,
        };
        std::vector<ncnn::VkMat> combined_key_output(1);
        std::vector<ncnn::VkMat> combined_value_output(1);
        if (implementation.concat_sequence->forward(
                combined_key_input,
                combined_key_output,
                command,
                implementation.option)
                != 0
            || implementation.concat_sequence->forward(
                   combined_value_input,
                   combined_value_output,
                   command,
                   implementation.option)
                   != 0)
            return false;
        combined_key = combined_key_output[0];
        combined_value = combined_value_output[0];
    }
    std::vector<ncnn::VkMat> sdpa_input = {
        query_rope_output[0],
        combined_key,
        combined_value,
        mask_gpu,
    };
    std::vector<ncnn::VkMat> sdpa_output(1);
    if (implementation.sdpa->forward(
            sdpa_input, sdpa_output, command, implementation.option)
        != 0)
        return false;

    ncnn::VkMat attention_token_major;
    ncnn::VkMat attention_matrix;
    if (implementation.permute_heads_tokens->forward(
            sdpa_output[0],
            attention_token_major,
            command,
            implementation.option)
            != 0
        || implementation.reshape_attention->forward(
               attention_token_major,
               attention_matrix,
               command,
               implementation.option)
               != 0)
        return false;

    ncnn::VkMat projected_gpu;
    if (projection.layer->forward(
            attention_matrix,
            projected_gpu,
            command,
            implementation.option)
        != 0)
        return false;
    std::vector<ncnn::VkMat> add_input = {input_gpu, projected_gpu};
    std::vector<ncnn::VkMat> add_output(1);
    if (implementation.add->forward(
            add_input, add_output, command, implementation.option)
        != 0)
        return false;

    ncnn::VkMat download_gpu = add_output[0];
    if (download_gpu.elempack != 1) {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(
            download_gpu,
            unpacked,
            1,
            command,
            implementation.option);
        download_gpu = unpacked;
    }
    if (!record_prepared_staging_download(
            download_gpu,
            input.rows(),
            config.hidden_size,
            transfer_slot.download,
            command,
            implementation.option))
        return false;

    const uint64_t total_actual_tokens = cache.token_count + input.rows();
    const uint64_t retained_tokens = config.sliding_window == 0
                                         ? total_actual_tokens
                                         : std::min<uint64_t>(
                                               total_actual_tokens,
                                               config.sliding_window > 1
                                                   ? config.sliding_window - 1
                                                   : 0);
    const uint64_t dropped_tokens = total_actual_tokens - retained_tokens;
    std::shared_ptr<NcnnVulkanAttentionCache> next_cache;
    uint64_t allocated_cache_bytes = 0;
    if (retained_tokens != 0) {
        next_cache = std::make_shared<NcnnVulkanAttentionCache>();
        if (dropped_tokens != 0) {
            if (!implementation.compact_sliding_cache)
                return false;
            std::vector<ncnn::VkMat> combined_key_input(1, combined_key);
            std::vector<ncnn::VkMat> combined_value_input(1, combined_value);
            std::vector<ncnn::VkMat> retained_key_parts(3);
            std::vector<ncnn::VkMat> retained_value_parts(3);
            if (implementation.compact_sliding_cache->forward(
                    combined_key_input,
                    retained_key_parts,
                    command,
                    implementation.option)
                    != 0
                || implementation.compact_sliding_cache->forward(
                       combined_value_input,
                       retained_value_parts,
                       command,
                       implementation.option)
                    != 0)
                return false;
            next_cache->key = retained_key_parts[1];
            next_cache->value = retained_value_parts[1];
        }
        else {
            // When present, the final sequence row is the learned attention
            // sink. retained_tokens counts only real tokens, so the same
            // shallow view excludes that row without copying the full history.
            // cstep remains unchanged because channels share the combined
            // buffer strides.
            next_cache->key = combined_key;
            next_cache->value = combined_value;
            next_cache->key.h = static_cast<int>(retained_tokens);
            next_cache->value.h = static_cast<int>(retained_tokens);
        }
        allocated_cache_bytes = static_cast<uint64_t>(next_cache->key.cstep)
                                    * next_cache->key.c * next_cache->key.elemsize
                                + static_cast<uint64_t>(next_cache->value.cstep)
                                      * next_cache->value.c * next_cache->value.elemsize;
    }

    output = CpuBatch(input.rows(), config.hidden_size);
    if (command.submit_and_wait() != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
        return false;
    for (size_t row_index = 0; row_index < output.rows(); ++row_index) {
        for (uint32_t column = 0; column < output.columns(); ++column) {
            if (!std::isfinite(output.row(row_index)[column])) {
                return false;
            }
        }
    }

    const uint64_t previous_start = cache.token_count == 0 ? position_offset : cache.start_position;
    cache.keys.clear();
    cache.values.clear();
    cache.bfloat16_keys.clear();
    cache.bfloat16_values.clear();
    cache.start_position = previous_start + dropped_tokens;
    cache.token_count = retained_tokens;
    cache.first_slot = 0;
    cache.capacity_tokens = 0;
    cache.columns = config.kv_head_count * config.head_dimension;
    cache.dtype = config.kv_cache_dtype;
    cache.vulkan_attention_cache = std::move(next_cache);
    cache.device_allocated_bytes = allocated_cache_bytes;
    current_vulkan_dispatch_count += 2;
    ++current_vulkan_attention_block_count;
    ++current_vulkan_runtime_counters.compute_submissions;
    ++current_vulkan_runtime_counters.batch_uploads;
    ++current_vulkan_runtime_counters.batch_downloads;
    current_vulkan_runtime_counters.auxiliary_uploads += 3;
    current_vulkan_runtime_counters.auxiliary_upload_bytes
        += transfer_slot.rope_cosine.buffer_capacity()
           + transfer_slot.rope_sine.buffer_capacity()
           + transfer_slot.attention_mask.buffer_capacity();
    return true;
#else
    (void)position_offset;
    (void)cache;
    (void)input;
    (void)output;
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
