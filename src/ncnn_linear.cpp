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
#include <cstring>
#include <limits>
#include <mutex>
#include <numbers>
#include <thread>
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
#endif

#if NCNN_MOE_WITH_VULKAN
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
        if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() <= 0)
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

private:
    explicit NcnnVulkanContext(ncnn::VulkanDevice* device)
        : device_(device),
          blob_allocator_(device->acquire_blob_allocator()),
          staging_allocator_(device->acquire_staging_allocator())
    {
    }

    // ncnn registers global Vulkan teardown with atexit. These device-owned
    // allocators must not be reclaimed here after that teardown has started.

    ncnn::VulkanDevice* device_ = nullptr;
    ncnn::VkAllocator* blob_allocator_ = nullptr;
    ncnn::VkAllocator* staging_allocator_ = nullptr;
    std::mutex command_mutex_;
};
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
    std::vector<float> bias;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool bfloat16 = false;
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
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
#endif
    std::shared_ptr<NcnnLinearOperator> fused_qkv;
    std::shared_ptr<NcnnLinearOperator> output_projection;
    NcnnVulkanAttentionConfig config;
    std::vector<float> sinks;
};

NcnnLinearOperator::NcnnLinearOperator()
    : implementation_(new Implementation)
{
}

NcnnLinearOperator::~NcnnLinearOperator() = default;

#if NCNN_MOE_USE_NCNN
static uint16_t float_to_bfloat16_storage(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}
#endif

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
    implementation.bfloat16 = matrix.dtype == DType::BFloat16;
    implementation.option.lightmode = true;
    implementation.option.use_packing_layout = true;
    implementation.option.num_threads = std::max(1u, std::thread::hardware_concurrency());

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
        implementation.option.use_fp16_packed = false;
        implementation.option.use_fp16_storage = false;
        implementation.option.use_fp16_arithmetic = false;
        implementation.option.use_bf16_packed = implementation.bfloat16 && vkdev->info.support_bf16_packed();
        implementation.option.use_bf16_storage = implementation.bfloat16 && vkdev->info.support_bf16_storage();
        implementation.option.use_cooperative_matrix = vkdev->info.support_cooperative_matrix();
        implementation.option.use_subgroup_ops = vkdev->info.support_subgroup_ops();
        implementation.layer = ncnn::create_layer_vulkan(ncnn::LayerType::InnerProduct);
        if (implementation.layer)
            implementation.layer->vkdev = vkdev;
    }
    else
#endif
    {
        implementation.option.use_bf16_storage = implementation.bfloat16;
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

    std::vector<float> converted_weight;
    ncnn::Mat model_data[2];
    if (implementation.bfloat16) {
        converted_weight.resize(matrix.element_count());
        for (size_t index = 0; index < converted_weight.size(); ++index)
            converted_weight[index] = bfloat16_to_float(matrix.bfloat16_data[index]);
        model_data[0] = ncnn::Mat(
            static_cast<int>(matrix.element_count()),
            converted_weight.data(),
            sizeof(float));
    }
    else {
        model_data[0] = ncnn::Mat(
            static_cast<int>(matrix.element_count()),
            const_cast<float*>(matrix.float32_data.data()),
            sizeof(float));
    }

    if (bias) {
        implementation.bias.resize(implementation.output_columns);
        for (uint32_t column = 0; column < implementation.output_columns; ++column) {
            implementation.bias[column] = bias->dtype == DType::Float32
                                              ? bias->float32_data[column]
                                              : bfloat16_to_float(bias->bfloat16_data[column]);
        }
        model_data[1] = ncnn::Mat(
            static_cast<int>(implementation.output_columns),
            implementation.bias.data(),
            sizeof(float));
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
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index) {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix || !bias || matrix->dtype != dtype || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns || bias->dtype != dtype
            || bias->shape.size() != 1 || bias->shape[0] != matrix->shape[0])
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
        fused_bias.float32_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index) {
            fused_matrix.float32_data.insert(
                fused_matrix.float32_data.end(),
                matrices[index]->float32_data.begin(),
                matrices[index]->float32_data.end());
            fused_bias.float32_data.insert(
                fused_bias.float32_data.end(),
                biases[index]->float32_data.begin(),
                biases[index]->float32_data.end());
        }
    }
    else {
        fused_matrix.bfloat16_data.reserve(element_count);
        fused_bias.bfloat16_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index) {
            fused_matrix.bfloat16_data.insert(
                fused_matrix.bfloat16_data.end(),
                matrices[index]->bfloat16_data.begin(),
                matrices[index]->bfloat16_data.end());
            fused_bias.bfloat16_data.insert(
                fused_bias.bfloat16_data.end(),
                biases[index]->bfloat16_data.begin(),
                biases[index]->bfloat16_data.end());
        }
    }
    return create(fused_matrix, &fused_bias, device);
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

bool NcnnLinearOperator::forward(const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_USE_NCNN
    const Implementation& implementation = *implementation_;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context) {
        ncnn::Mat bottom(
            static_cast<int>(input.columns()),
            static_cast<int>(input.rows()),
            sizeof(float));
        if (bottom.empty())
            return false;
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            std::copy_n(input.row(row_index), input.columns(), bottom.row<float>(static_cast<int>(row_index)));

        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        ncnn::VkCompute command(vkdev);
        ncnn::VkMat bottom_gpu;
        command.record_upload(bottom, bottom_gpu, implementation.option);

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
        ncnn::Mat top;
        ncnn::Option download_option = implementation.option;
        download_option.use_packing_layout = false;
        command.record_download(download_gpu, top, download_option);
        if (command.submit_and_wait() != 0 || top.empty())
            return false;

        const size_t output_size = input.rows() * implementation.output_columns;
        if (top.total() * top.elempack != output_size || top.elembits() != 32)
            return false;
        output = CpuBatch(input.rows(), implementation.output_columns);
        const float* source = static_cast<const float*>(top.data);
        for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
            std::copy_n(
                source + row_index * implementation.output_columns,
                implementation.output_columns,
                output.row(row_index));
        }
        ++current_vulkan_dispatch_count;
        return true;
    }
#endif

    ncnn::Mat bottom;
    if (implementation.bfloat16) {
        bottom.create(static_cast<int>(input.columns()), static_cast<int>(input.rows()), sizeof(uint16_t));
        if (bottom.empty())
            return false;
        for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
            uint16_t* destination = bottom.row<uint16_t>(static_cast<int>(row_index));
            const float* source = input.row(row_index);
            for (uint32_t column = 0; column < input.columns(); ++column)
                destination[column] = float_to_bfloat16_storage(source[column]);
        }
    }
    else {
        bottom.create(static_cast<int>(input.columns()), static_cast<int>(input.rows()), sizeof(float));
        if (bottom.empty())
            return false;
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            std::copy_n(input.row(row_index), input.columns(), bottom.row<float>(static_cast<int>(row_index)));
    }

    ncnn::Mat top;
    if (implementation.layer->forward(bottom, top, implementation.option) != 0 || top.empty())
        return false;
    output = CpuBatch(input.rows(), implementation.output_columns);
    if (implementation.bfloat16) {
        const uint16_t* source = static_cast<const uint16_t*>(top.data);
        for (size_t index = 0; index < input.rows() * implementation.output_columns; ++index)
            output.row(index / implementation.output_columns)[index % implementation.output_columns] = bfloat16_to_float(source[index]);
    }
    else {
        const float* source = static_cast<const float*>(top.data);
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            std::copy_n(source + row_index * implementation.output_columns, implementation.output_columns, output.row(row_index));
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

#if NCNN_MOE_WITH_VULKAN && NCNN_BATCH
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

static ncnn::Mat make_rope_cache(
    size_t token_count,
    uint64_t position_offset,
    const NcnnVulkanAttentionConfig& config,
    bool sine,
    bool bfloat16_storage)
{
    const uint32_t half_dimension = config.head_dimension / 2;
    ncnn::Mat cache(
        static_cast<int>(half_dimension),
        static_cast<int>(token_count),
        bfloat16_storage ? sizeof(uint16_t) : sizeof(float));
    if (cache.empty())
        return {};

    float concentration = 1.0f;
    float low = 0.0f;
    float high = 0.0f;
    if (config.rope_scaling_factor > 1.0f) {
        concentration = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(half_dimension);
        low = half * std::log(
                         static_cast<float>(config.initial_context_length)
                         / (config.rope_ntk_beta * 2.0f * std::numbers::pi_v<float>))
              / std::log(config.rope_theta);
        high = half * std::log(
                          static_cast<float>(config.initial_context_length)
                          / (config.rope_ntk_alpha * 2.0f * std::numbers::pi_v<float>))
               / std::log(config.rope_theta);
    }

    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        float* float_row = bfloat16_storage
                               ? nullptr
                               : cache.row<float>(static_cast<int>(token_index));
        uint16_t* bfloat16_row = bfloat16_storage
                                    ? cache.row<uint16_t>(static_cast<int>(token_index))
                                    : nullptr;
        for (uint32_t index = 0; index < half_dimension; ++index) {
            const float frequency = std::pow(
                config.rope_theta,
                static_cast<float>(2 * index) / static_cast<float>(config.head_dimension));
            float inverse_frequency = 1.0f / frequency;
            if (config.rope_scaling_factor > 1.0f) {
                const float ramp = std::clamp(
                    (static_cast<float>(index) - low) / (high - low),
                    0.0f,
                    1.0f);
                const float mask = 1.0f - ramp;
                const float interpolation = 1.0f / (config.rope_scaling_factor * frequency);
                inverse_frequency = interpolation * (1.0f - mask) + inverse_frequency * mask;
            }
            const float angle = static_cast<float>(position_offset + token_index) * inverse_frequency;
            const float value = (sine ? std::sin(angle) : std::cos(angle)) * concentration;
            if (bfloat16_storage)
                bfloat16_row[index] = float_to_bfloat16_storage(value);
            else
                float_row[index] = value;
        }
    }
    return cache;
}

static ncnn::Mat make_attention_mask(
    size_t token_count,
    uint64_t destination_count,
    uint64_t position_offset,
    const CpuLayerCache& cache,
    const NcnnVulkanAttentionConfig& config,
    const std::vector<float>& sinks,
    bool bfloat16_storage)
{
    if (destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};
    ncnn::Mat mask(
        static_cast<int>(destination_count),
        static_cast<int>(token_count),
        static_cast<int>(config.head_count),
        bfloat16_storage ? sizeof(uint16_t) : sizeof(float));
    if (mask.empty())
        return {};

    // A finite sentinel avoids NaNs in BF16 flash-attention implementations
    // while still underflowing to an exact zero probability after softmax.
    constexpr float masked_logit = -10000.0f;
    for (uint32_t head = 0; head < config.head_count; ++head) {
        ncnn::Mat head_mask = mask.channel(static_cast<int>(head));
        for (size_t query_index = 0; query_index < token_count; ++query_index) {
            const uint64_t query_position = position_offset + query_index;
            const uint64_t actual_end = cache.token_count + token_count;
            if (bfloat16_storage) {
                uint16_t* row = head_mask.row<uint16_t>(static_cast<int>(query_index));
                const uint16_t masked_value = float_to_bfloat16_storage(masked_logit);
                for (uint64_t key_index = 0; key_index < actual_end; ++key_index) {
                    const uint64_t key_position = key_index < cache.token_count
                                                      ? cache.start_position + key_index
                                                      : position_offset + key_index - cache.token_count;
                    const bool future = key_position > query_position;
                    const bool too_old = config.sliding_window > 0
                                         && key_position + config.sliding_window <= query_position;
                    row[key_index] = future || too_old ? masked_value : 0;
                }
                row[actual_end] = float_to_bfloat16_storage(sinks[head]);
                std::fill(row + actual_end + 1, row + destination_count, masked_value);
            }
            else {
                float* row = head_mask.row<float>(static_cast<int>(query_index));
                for (uint64_t key_index = 0; key_index < actual_end; ++key_index) {
                    const uint64_t key_position = key_index < cache.token_count
                                                      ? cache.start_position + key_index
                                                      : position_offset + key_index - cache.token_count;
                    const bool future = key_position > query_position;
                    const bool too_old = config.sliding_window > 0
                                         && key_position + config.sliding_window <= query_position;
                    row[key_index] = future || too_old ? masked_logit : 0.0f;
                }
                row[actual_end] = sinks[head];
                std::fill(row + actual_end + 1, row + destination_count, masked_logit);
            }
        }
    }
    return mask;
}
#endif

std::shared_ptr<NcnnVulkanAttentionOperator> NcnnVulkanAttentionOperator::create(
    const TensorData& norm_weight,
    const TensorData& sinks,
    std::shared_ptr<NcnnLinearOperator> fused_qkv,
    std::shared_ptr<NcnnLinearOperator> output_projection,
    const NcnnVulkanAttentionConfig& config)
{
#if NCNN_MOE_WITH_VULKAN && NCNN_BATCH
    if (!fused_qkv || !output_projection || !fused_qkv->uses_vulkan()
        || !output_projection->uses_vulkan() || config.hidden_size == 0
        || config.head_count == 0 || config.kv_head_count == 0
        || config.head_dimension == 0 || config.head_dimension % 2 != 0
        || config.head_count % config.kv_head_count != 0
        || config.activation_dtype != config.kv_cache_dtype
        || norm_weight.shape != std::vector<uint32_t>{config.hidden_size}
        || sinks.shape != std::vector<uint32_t>{config.head_count})
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
    if (!tensor_to_float_vector(sinks, implementation.sinks))
        return {};
    implementation.vulkan_context = fused_implementation.vulkan_context;
    implementation.option = fused_implementation.option;
    // SDPA consumes unpacked [head, token, dimension] tensors. Keeping this
    // subgraph at pack1 avoids GQA head channels being silently packed as four.
    implementation.option.use_packing_layout = false;
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();

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
    if (implementation.norm->upload_model(command, upload_option) != 0
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
#if NCNN_MOE_WITH_VULKAN && NCNN_BATCH
    const Implementation& implementation = *implementation_;
    const NcnnVulkanAttentionConfig& config = implementation.config;
    if (input.rows() == 0 || input.columns() != config.hidden_size
        || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || (cache.dtype != config.kv_cache_dtype && cache.token_count != 0)
        || (cache.token_count != 0 && !cache.vulkan_attention_cache))
        return false;

    ncnn::Mat input_mat(
        static_cast<int>(input.columns()),
        static_cast<int>(input.rows()),
        sizeof(float));
    if (input_mat.empty())
        return false;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        std::copy_n(input.row(row_index), input.columns(), input_mat.row<float>(static_cast<int>(row_index)));

    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16
                                  && implementation.option.use_bf16_storage;
    ncnn::Mat cosine = make_rope_cache(
        input.rows(), position_offset, config, false, bfloat16_storage);
    ncnn::Mat sine = make_rope_cache(
        input.rows(), position_offset, config, true, bfloat16_storage);
    const uint64_t actual_token_count = cache.token_count + input.rows();
    constexpr uint64_t sink_token_count = 1;
    const uint64_t destination_count = actual_token_count + sink_token_count;
    ncnn::Mat mask = make_attention_mask(
        input.rows(),
        destination_count,
        position_offset,
        cache,
        config,
        implementation.sinks,
        bfloat16_storage);
    ncnn::Mat zero_key_value(
        static_cast<int>(config.head_dimension),
        static_cast<int>(sink_token_count),
        static_cast<int>(config.kv_head_count),
        bfloat16_storage ? sizeof(uint16_t) : sizeof(float));
    if (cosine.empty() || sine.empty() || mask.empty() || zero_key_value.empty())
        return false;
    if (bfloat16_storage)
        zero_key_value.fill<uint16_t>(0);
    else
        zero_key_value.fill(0.0f);

    const NcnnLinearOperator::Implementation& fused = *implementation.fused_qkv->implementation_;
    const NcnnLinearOperator::Implementation& projection = *implementation.output_projection->implementation_;
    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute command(vkdev);
    ncnn::VkMat input_gpu;
    ncnn::VkMat cosine_gpu;
    ncnn::VkMat sine_gpu;
    ncnn::VkMat mask_gpu;
    ncnn::VkMat zero_gpu;
    command.record_upload(input_mat, input_gpu, implementation.option);
    command.record_upload(cosine, cosine_gpu, implementation.option);
    command.record_upload(sine, sine_gpu, implementation.option);
    command.record_upload(mask, mask_gpu, implementation.option);
    command.record_upload(zero_key_value, zero_gpu, implementation.option);
    auto unpack_gpu = [&](ncnn::VkMat& tensor) {
        if (tensor.elempack == 1)
            return;
        ncnn::VkMat unpacked;
        vkdev->convert_packing(
            tensor,
            unpacked,
            1,
            command,
            implementation.option);
        tensor = unpacked;
    };
    unpack_gpu(cosine_gpu);
    unpack_gpu(sine_gpu);
    unpack_gpu(mask_gpu);
    unpack_gpu(zero_gpu);

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

    std::vector<ncnn::VkMat> key_with_sink_input = {key_rope_output[0], zero_gpu};
    std::vector<ncnn::VkMat> value_with_sink_input = {value_heads, zero_gpu};
    std::vector<ncnn::VkMat> key_with_sink_output(1);
    std::vector<ncnn::VkMat> value_with_sink_output(1);
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
    ncnn::Mat output_mat;
    ncnn::Option download_option = implementation.option;
    download_option.use_packing_layout = false;
    command.record_download(download_gpu, output_mat, download_option);

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
            // The final sequence row is the learned attention sink. A shallow
            // shape view keeps it out of the logical KV cache without copying
            // the full history. cstep deliberately remains unchanged because
            // channels still use the strides of the shared combined buffer.
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

    if (command.submit_and_wait() != 0 || output_mat.empty())
        return false;
    if (output_mat.dims != 2 || output_mat.w != static_cast<int>(config.hidden_size)
        || output_mat.h != static_cast<int>(input.rows())
        || output_mat.elempack != 1 || output_mat.elembits() != 32)
        return false;
    output = CpuBatch(input.rows(), config.hidden_size);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        std::copy_n(
            output_mat.row<float>(static_cast<int>(row_index)),
            config.hidden_size,
            output.row(row_index));
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
