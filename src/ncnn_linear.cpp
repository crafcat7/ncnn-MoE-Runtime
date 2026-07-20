#include "ncnn_linear.h"

#include "cpu_ops.h"

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_USE_NCNN
static constexpr uint64_t max_ncnn_linear_weight_bytes = 64ull * 1024ull * 1024ull;
#endif

class NcnnLinearOperator::Implementation
{
public:
#if NCNN_MOE_USE_NCNN
    ~Implementation()
    {
        if (layer) {
            layer->destroy_pipeline(option);
            delete layer;
        }
    }

    ncnn::Layer* layer = nullptr;
    ncnn::Option option;
    std::vector<float> bias;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool bfloat16 = false;
#endif
};

NcnnLinearOperator::NcnnLinearOperator()
    : implementation_(new Implementation)
{
}

NcnnLinearOperator::~NcnnLinearOperator() = default;

#if NCNN_MOE_USE_NCNN
static uint16_t float_to_bfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}
#endif

std::shared_ptr<NcnnLinearOperator> NcnnLinearOperator::create(
    const TensorData& matrix,
    const TensorData* bias)
{
#if NCNN_MOE_USE_NCNN
    if (matrix.shape.size() != 2
        || (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16))
        return {};

    const uint64_t element_size = matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
    if (matrix.element_count() > max_ncnn_linear_weight_bytes / element_size)
        return {};

    std::shared_ptr<NcnnLinearOperator> linear(new NcnnLinearOperator);
    Implementation& implementation = *linear->implementation_;
    implementation.input_columns = matrix.shape[1];
    implementation.output_columns = matrix.shape[0];
    implementation.bfloat16 = matrix.dtype == DType::BFloat16;
    implementation.option.lightmode = true;
    implementation.option.use_packing_layout = true;
    implementation.option.use_bf16_storage = implementation.bfloat16;
    implementation.option.num_threads = std::max(1u, std::thread::hardware_concurrency());

    implementation.layer = ncnn::create_layer_cpu(ncnn::LayerType::InnerProduct);
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
    return linear;
#else
    (void)matrix;
    (void)bias;
    return {};
#endif
}

bool NcnnLinearOperator::forward(const CpuBatch& input, CpuBatch& output) const
{
#if NCNN_MOE_USE_NCNN
    const Implementation& implementation = *implementation_;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

    ncnn::Mat bottom;
    if (implementation.bfloat16) {
        bottom.create(static_cast<int>(input.columns()), static_cast<int>(input.rows()), sizeof(uint16_t));
        if (bottom.empty())
            return false;
        for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
            uint16_t* destination = bottom.row<uint16_t>(static_cast<int>(row_index));
            const float* source = input.row(row_index);
            for (uint32_t column = 0; column < input.columns(); ++column)
                destination[column] = float_to_bfloat16(source[column]);
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

} // namespace moe
} // namespace ncnn
