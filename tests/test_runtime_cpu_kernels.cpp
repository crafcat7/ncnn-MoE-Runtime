#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_float_scaled_add()
{
    std::array<float, 17> output = {};
    std::array<float, 17> input = {};
    std::array<float, 17> expected = {};
    constexpr float scale = -0.375f;
    for (size_t index = 0; index < output.size(); ++index)
    {
        output[index] = static_cast<float>(index) * 0.125f - 0.75f;
        input[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25f;
        expected[index] = output[index] + scale * input[index];
    }
    float_scaled_add(output.data(),
                     input.data(),
                     scale,
                     static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
        check_near(output[index], expected[index], 1e-6f);
}

void test_float_scale_inplace_and_scaled_add()
{
    std::array<float, 19> values = {};
    std::array<float, 19> output = {};
    std::array<float, 19> expected_values = {};
    std::array<float, 19> expected_output = {};
    constexpr float value_scale = 0.625f;
    constexpr float output_scale = -0.375f;
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index] = static_cast<float>(static_cast<int>(index % 11) - 5)
                        * 0.125f;
        output[index] = static_cast<float>(static_cast<int>(index % 7) - 3)
                        * 0.25f;
        expected_values[index] = values[index] * value_scale;
        expected_output[index] = output[index]
                                 + output_scale * expected_values[index];
    }
    scale_add(values.data(),
              value_scale,
              output.data(),
              output_scale,
              static_cast<uint32_t>(values.size()));
    for (size_t index = 0; index < values.size(); ++index)
    {
        check_near(values[index], expected_values[index], 1e-6f);
        check_near(output[index], expected_output[index], 1e-6f);
    }

    std::array<float, 19> recurrent = {};
    std::array<float, 19> delta = {};
    std::array<float, 19> accumulated = {};
    std::array<float, 19> expected_recurrent = {};
    std::array<float, 19> expected_accumulated = {};
    constexpr float input_scale = 0.375f;
    constexpr float accumulate_scale = -0.625f;
    for (size_t index = 0; index < recurrent.size(); ++index)
    {
        recurrent[index] = static_cast<float>(static_cast<int>(index % 13) - 6)
                           * 0.125f;
        delta[index] = static_cast<float>(static_cast<int>(index % 7) - 3)
                       * 0.25f;
        accumulated[index] = static_cast<float>(static_cast<int>(index % 11) - 5)
                             * 0.0625f;
        const float product = delta[index] * input_scale;
        expected_recurrent[index] = recurrent[index] + product;
        expected_accumulated[index] = accumulated[index]
                                      + accumulate_scale * expected_recurrent[index];
    }
    scaled_acc(recurrent.data(),
               delta.data(),
               input_scale,
               accumulated.data(),
               accumulate_scale,
               static_cast<uint32_t>(recurrent.size()));
    for (size_t index = 0; index < recurrent.size(); ++index)
    {
        check_near(recurrent[index], expected_recurrent[index], 1e-6f);
        check_near(accumulated[index], expected_accumulated[index], 1e-6f);
    }

    // Multiplication must round before the state update; fusing it would
    // leave 2^-46 instead of zero in every SIMD lane and the scalar tail.
    recurrent.fill(-0x1.000004p0f);
    delta.fill(0x1.000002p0f);
    accumulated.fill(0.0f);
    scaled_acc(recurrent.data(),
               delta.data(),
               0x1.000002p0f,
               accumulated.data(),
               1.0f,
               static_cast<uint32_t>(recurrent.size()));
    for (size_t index = 0; index < recurrent.size(); ++index)
    {
        check(recurrent[index] == 0.0f);
        check(accumulated[index] == 0.0f);
    }

    std::array<float, 7> special_recurrent = {
        0.0f,
        -0.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::denorm_min(),
        1.25f};
    const std::array<float, 7> special_delta = {
        0.0f,
        -0.0f,
        -0.0f,
        0.0f,
        0.5f,
        std::numeric_limits<float>::denorm_min(),
        -0.5f};
    std::array<float, 7> special_output = {0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.75f};
    scaled_acc(special_recurrent.data(),
               special_delta.data(),
               0.5f,
               special_output.data(),
               1.0f,
               static_cast<uint32_t>(special_recurrent.size()));
    check(special_recurrent[0] == 0.0f && !std::signbit(special_recurrent[0]));
    check(special_recurrent[1] == 0.0f && std::signbit(special_recurrent[1]));
    check(std::isinf(special_recurrent[2]) && !std::signbit(special_recurrent[2]));
    check(std::isinf(special_recurrent[3]) && std::signbit(special_recurrent[3]));
    check(std::isnan(special_recurrent[4]));
    check(special_recurrent[5] == 0.0f
          || special_recurrent[5] == std::numeric_limits<float>::denorm_min());
    check(special_recurrent[6] == 1.0f);
    check(special_output[0] == 0.0f && !std::signbit(special_output[0]));
    check(special_output[1] == 0.0f && std::signbit(special_output[1]));
    check(std::isinf(special_output[2]) && !std::signbit(special_output[2]));
    check(std::isinf(special_output[3]) && std::signbit(special_output[3]));
    check(std::isnan(special_output[4]));
    check(special_output[5] == 0.0f
          || special_output[5] == std::numeric_limits<float>::denorm_min());
    check(special_output[6] == 0.25f);
}

void test_float_scale_add()
{
    std::array<float, 23> output = {};
    std::array<float, 23> input = {};
    constexpr float output_scale = -0.375f;
    constexpr float input_scale = 0.625f;
    for (size_t index = 0; index < output.size(); ++index)
    {
        output[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0625f;
    }
    std::array<float, 23> expected = output;
    for (size_t index = 0; index < expected.size(); ++index)
        expected[index] = expected[index] * output_scale + input[index] * input_scale;
    float_scale_add(output.data(),
                    output_scale,
                    input.data(),
                    input_scale,
                    static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
        check_near(output[index], expected[index], 1e-6f);
}

void test_float_dot()
{
    std::array<float, 137> left = {};
    std::array<float, 137> right = {};
    float expected = 0.0f;
    for (size_t index = 0; index < left.size(); ++index)
    {
        left[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.125f;
        right[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.0625f;
        expected += left[index] * right[index];
    }
    check_near(float_dot(left.data(), right.data(), static_cast<uint32_t>(left.size())),
               expected,
               1e-4f);
    check_near(float_dot(left.data(), right.data(), 0), 0.0f, 1e-6f);

    // Exercise unaligned inputs, SIMD boundaries and long reductions against
    // a double-precision reference instead of another float kernel.
    for (uint32_t count : {0u, 1u, 3u, 4u, 5u, 15u, 16u, 17u, 31u, 32u, 33u, 127u, 129u, 1024u, 1027u, 4099u})
    {
        std::vector<float> a(count + 1, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> b(count + 1, std::numeric_limits<float>::quiet_NaN());
        double reference = 0.0;
        double magnitude = 0.0;
        for (uint32_t index = 0; index < count; ++index)
        {
            a[index + 1] = static_cast<float>(static_cast<int>((index * 17 + 3) % 101) - 50) / 57.0f;
            b[index + 1] = static_cast<float>(static_cast<int>((index * 23 + 7) % 97) - 48) / 61.0f;
            const double product = static_cast<double>(a[index + 1]) * b[index + 1];
            reference += product;
            magnitude += std::abs(product);
        }
        const float tolerance = static_cast<float>(8.0 * std::numeric_limits<float>::epsilon()
                                                   * std::max(1.0, magnitude));
        check_near(float_dot(a.data() + 1, b.data() + 1, count),
                   static_cast<float>(reference), tolerance);
    }
}

void test_dense_gemm_tails()
{
    constexpr float sentinel = 12345.5f;
    for (uint32_t kernel = 0; kernel < 3; ++kernel)
    {
        const bool bfloat16 = kernel == 2;
        const uint32_t tile_size = kernel == 0 ? 4 : 8;
        for (uint32_t columns : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 15u, 16u, 17u, 31u, 33u, 129u, 1026u, 1027u})
        {
            for (uint32_t tokens = 0; tokens <= 4; ++tokens)
            {
                for (uint32_t outputs = 0; outputs <= tile_size; ++outputs)
                {
                    const size_t weight_stride = columns + 3;
                    const size_t input_stride = columns + 5;
                    const size_t output_stride = tile_size + 3;
                    std::vector<float> weights(tile_size * weight_stride + 1, std::numeric_limits<float>::quiet_NaN());
                    std::vector<uint16_t> bfloat16_weights(weights.size(), float_to_bfloat16(std::numeric_limits<float>::quiet_NaN()));
                    std::vector<float> input(4 * input_stride + 1, std::numeric_limits<float>::quiet_NaN());
                    std::vector<float> output(4 * output_stride + 2, sentinel);
                    for (uint32_t row = 0; row < outputs; ++row)
                        for (uint32_t column = 0; column < columns; ++column)
                        {
                            weights[1 + row * weight_stride + column] = static_cast<float>(static_cast<int>((row * 11 + column * 7) % 37) - 18) / 23.0f;
                            if (bfloat16)
                            {
                                const size_t index = 1 + row * weight_stride + column;
                                bfloat16_weights[index] = float_to_bfloat16(weights[index]);
                                weights[index] = bfloat16_to_float(bfloat16_weights[index]);
                            }
                        }
                    for (uint32_t token = 0; token < tokens; ++token)
                        for (uint32_t column = 0; column < columns; ++column)
                            input[1 + token * input_stride + column] = static_cast<float>(static_cast<int>((token * 13 + column * 17) % 41) - 20) / 29.0f;
                    if (bfloat16)
                        bfloat16_gemm_4x8(bfloat16_weights.data() + 1, weight_stride, input.data() + 1, input_stride,
                                          columns, outputs, tokens, output.data() + 1, output_stride);
                    else if (tile_size == 4)
                        float_gemm_4x4(weights.data() + 1, weight_stride, input.data() + 1, input_stride,
                                       columns, outputs, tokens, output.data() + 1, output_stride);
                    else
                        float_gemm_4x8(weights.data() + 1, weight_stride, input.data() + 1, input_stride,
                                       columns, outputs, tokens, output.data() + 1, output_stride);
                    check(output.front() == sentinel && output.back() == sentinel);
                    for (uint32_t token = 0; token < 4; ++token)
                    {
                        for (size_t row = 0; row < output_stride; ++row)
                        {
                            const float actual = output[1 + token * output_stride + row];
                            if (token >= tokens || row >= outputs)
                            {
                                check(actual == sentinel);
                                continue;
                            }
                            double reference = 0.0;
                            double magnitude = 0.0;
                            for (uint32_t column = 0; column < columns; ++column)
                            {
                                const double product = static_cast<double>(weights[1 + row * weight_stride + column])
                                                       * input[1 + token * input_stride + column];
                                reference += product;
                                magnitude += std::abs(product);
                            }
                            const float tolerance = static_cast<float>(8.0 * std::numeric_limits<float>::epsilon()
                                                                       * std::max(1.0, magnitude));
                            check_near(actual, static_cast<float>(reference), tolerance);
                        }
                    }
                }
            }
        }
    }

    struct Shape
    {
        uint32_t rows;
        uint32_t input_columns;
        uint32_t output_columns;
    };
    constexpr std::array shapes = {
        Shape{31, 33, 65},
        Shape{32, 255, 129},
        Shape{32, 256, 64},
        Shape{63, 257, 257},
        Shape{64, 257, 257},
        Shape{65, 257, 257},
        Shape{33, 256, 63},
        Shape{33, 257, 64},
        Shape{33, 511, 129},
        Shape{33, 512, 65},
        Shape{32, 513, 65},
        Shape{128, 1025, 129},
        Shape{128, 1027, 257},
        Shape{128, 2047, 129},
        Shape{128, 2048, 65},
        Shape{65, 2048, 513},
        Shape{128, 33, 129},
        Shape{31, 2048, 129},
        Shape{33, 255, 129},
        Shape{1, 513, 1025},
        Shape{1, 1023, 1025},
        Shape{1, 1024, 2048},
        Shape{1, 1024, 2049},
        Shape{32, 513, 257},
        Shape{33, 1025, 513},
        Shape{32, 2049, 257},
        Shape{32, 4097, 257},
        Shape{31, 4097, 257},
    };
    ActivationBuffer output;
    ActivationBuffer quantized_input_scratch(2, 7);
    for (size_t row = 0; row < quantized_input_scratch.rows(); ++row)
    {
        for (uint32_t column = 0; column < quantized_input_scratch.columns(); ++column)
            quantized_input_scratch.row(row)[column] = static_cast<float>(row * 7 + column) + 0.5f;
    }
    const auto check_scratch = [&] {
        check(quantized_input_scratch.rows() == 2);
        check(quantized_input_scratch.columns() == 7);
        for (size_t row = 0; row < quantized_input_scratch.rows(); ++row)
        {
            for (uint32_t column = 0; column < quantized_input_scratch.columns(); ++column)
            {
                check(quantized_input_scratch.row(row)[column]
                      == static_cast<float>(row * 7 + column) + 0.5f);
            }
        }
    };
    const auto check_reference = [](const TensorData& matrix,
                                    const TensorData* bias,
                                    const ActivationBuffer& input,
                                    const ActivationBuffer& actual) {
        check(actual.rows() == input.rows());
        check(actual.columns() == matrix.shape[0]);
        for (size_t row = 0; row < input.rows(); ++row)
        {
            for (uint32_t output_column = 0; output_column < matrix.shape[0]; ++output_column)
            {
                double reference = 0.0;
                double magnitude = 0.0;
                if (bias)
                {
                    const double bias_value = bias->dtype == DType::Float32
                                                  ? bias->float32_values()[output_column]
                                                  : bfloat16_to_float(bias->bfloat16_values()[output_column]);
                    reference = bias_value;
                    magnitude = std::abs(bias_value);
                }
                for (uint32_t input_column = 0; input_column < matrix.shape[1]; ++input_column)
                {
                    const size_t weight_index = static_cast<size_t>(output_column) * matrix.shape[1]
                                                + input_column;
                    const double weight = matrix.dtype == DType::Float32
                                              ? matrix.float32_values()[weight_index]
                                              : bfloat16_to_float(matrix.bfloat16_values()[weight_index]);
                    const double product = weight * input.row(row)[input_column];
                    reference += product;
                    magnitude += std::abs(product);
                }
                const float tolerance = static_cast<float>(8.0 * std::numeric_limits<float>::epsilon()
                                                           * std::max(1.0, magnitude));
                check_near(actual.row(row)[output_column], static_cast<float>(reference), tolerance);
            }
        }
    };

    for (uint32_t requested_threads : {1u, 2u, 8u})
    {
        CpuOpenMpThreadLimitScope thread_limit;
        thread_limit.set(requested_threads);
        for (size_t shape_index = 0; shape_index < shapes.size(); ++shape_index)
        {
            const Shape shape = shapes[shape_index];
            ActivationBuffer input(shape.rows, shape.input_columns);
            for (size_t row = 0; row < input.rows(); ++row)
            {
                for (uint32_t column = 0; column < input.columns(); ++column)
                {
                    const int value = static_cast<int>((row * 19 + column * 7 + shape_index * 11) % 43) - 21;
                    input.row(row)[column] = static_cast<float>(value) * 0.00390625f;
                }
            }
            for (uint32_t dtype_index = 0; dtype_index < 2; ++dtype_index)
            {
                const bool bfloat16 = dtype_index == 1;
                TensorData matrix;
                matrix.dtype = bfloat16 ? DType::BFloat16 : DType::Float32;
                matrix.shape = {shape.output_columns, shape.input_columns};
                if (bfloat16)
                    matrix.bfloat16_data.resize(matrix.element_count());
                else
                    matrix.float32_data.resize(matrix.element_count());
                for (uint32_t output_column = 0; output_column < shape.output_columns; ++output_column)
                {
                    for (uint32_t input_column = 0; input_column < shape.input_columns; ++input_column)
                    {
                        const size_t index = static_cast<size_t>(output_column) * shape.input_columns
                                             + input_column;
                        const int value = static_cast<int>((index * 13 + shape_index * 17) % 61) - 30;
                        const float weight = static_cast<float>(value) * 0.003125f;
                        if (bfloat16)
                            matrix.bfloat16_data[index] = float_to_bfloat16(weight);
                        else
                            matrix.float32_data[index] = weight;
                    }
                }

                TensorData bias;
                const bool use_bias = (shape_index + dtype_index) % 2 == 0;
                const TensorData* bias_pointer = nullptr;
                if (use_bias)
                {
                    bias.dtype = shape_index % 2 == 0 ? DType::Float32 : DType::BFloat16;
                    bias.shape = {shape.output_columns};
                    if (bias.dtype == DType::Float32)
                        bias.float32_data.resize(shape.output_columns);
                    else
                        bias.bfloat16_data.resize(shape.output_columns);
                    for (uint32_t column = 0; column < shape.output_columns; ++column)
                    {
                        const int value = static_cast<int>((column * 11 + shape_index * 5) % 19) - 9;
                        const float bias_value = static_cast<float>(value) * 0.0078125f;
                        if (bias.dtype == DType::Float32)
                            bias.float32_data[column] = bias_value;
                        else
                            bias.bfloat16_data[column] = float_to_bfloat16(bias_value);
                    }
                    bias_pointer = &bias;
                }

                if (requested_threads == 2
                    && shape.rows == 128
                    && shape.input_columns == 1025
                    && shape.output_columns == 129)
                {
                    const uint64_t operation_count = static_cast<uint64_t>(shape.rows)
                                                     * shape.input_columns * shape.output_columns;
                    check(cpu_linear_team_size(operation_count, matrix.dtype)
                          == static_cast<int>(cpu_linear_num_threads()));
                }

                const auto run_linear = [&] {
                    if (bias_pointer)
                    {
                        forward_linear(matrix,
                                       *bias_pointer,
                                       input,
                                       output,
                                       g_test_optimization_flags,
                                       nullptr,
                                       ExecutionBackend::Cpu,
                                       &quantized_input_scratch);
                    }
                    else
                    {
                        forward_linear(matrix,
                                       input,
                                       output,
                                       g_test_optimization_flags,
                                       nullptr,
                                       ExecutionBackend::Cpu,
                                       &quantized_input_scratch);
                    }
                    check_reference(matrix, bias_pointer, input, output);
                    check_scratch();
                };

#if defined(__APPLE__) && TARGET_OS_OSX && defined(__MAC_15_0)
                if (bfloat16
                    && ((requested_threads == 1
                         && shape.rows == 128
                         && shape.input_columns == 1025
                         && shape.output_columns == 129)
                        || (shape.rows == 65
                            && shape.input_columns == 2048
                            && shape.output_columns == 513)))
                {
                    if (__builtin_available(macOS 15.0, *))
                    {
                        const BLAS_THREADING previous_threading = BLASGetThreading();
                        for (BLAS_THREADING initial_threading : {BLAS_THREADING_MULTI_THREADED,
                                                                 BLAS_THREADING_SINGLE_THREADED})
                        {
                            const BLAS_THREADING previous_mode = BLASGetThreading();
                            const int set_status = BLASSetThreading(initial_threading);
                            const BLAS_THREADING expected_mode = set_status == 0
                                                                     ? initial_threading
                                                                     : previous_mode;
                            run_linear();
                            check(BLASGetThreading() == expected_mode);
                        }
                        (void)BLASSetThreading(previous_threading);
                        check(BLASGetThreading() == previous_threading);
                    }
                    else
                    {
                        run_linear();
                    }
                }
                else
#endif
                {
                    run_linear();
                }
            }
        }
    }
}

void test_float_exp_inplace()
{
    std::array<float, 37> values = {};
    std::array<float, 37> expected = {};
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index] = -6.0f + static_cast<float>(index) * 0.25f;
        expected[index] = std::exp(values[index]);
    }
    float_exp_inplace(values.data(), static_cast<uint32_t>(values.size()));
    for (size_t index = 0; index < values.size(); ++index)
        check_near(values[index], expected[index], 2e-4f);
    float_exp_inplace(values.data(), 0);

    for (uint32_t count : {4u, 5u, 7u, 8u, 9u})
    {
        std::array<float, 9> short_values = {};
        for (uint32_t index = 0; index < count; ++index)
            short_values[index] = -3.0f + static_cast<float>(index) * 0.75f;
        float_exp_inplace(short_values.data(), count);
        for (uint32_t index = 0; index < count; ++index)
            check_near(short_values[index], std::exp(-3.0f + static_cast<float>(index) * 0.75f), 2e-4f);
    }

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float positive_104 = 104.0f;
    const float negative_104 = -104.0f;
    const float positive_88 = 88.72f;
    const float negative_88 = -88.72f;
    const std::array<float, 17> edge_values = {
        0.0f,
        -0.0f,
        infinity,
        -infinity,
        nan,
        positive_104,
        std::nextafter(positive_104, -infinity),
        std::nextafter(positive_104, infinity),
        negative_104,
        std::nextafter(negative_104, 0.0f),
        std::nextafter(negative_104, -infinity),
        positive_88,
        std::nextafter(positive_88, 0.0f),
        std::nextafter(positive_88, infinity),
        negative_88,
        std::nextafter(negative_88, 0.0f),
        std::nextafter(negative_88, -infinity),
    };
    const auto check_exp = [](float actual, float expected) {
        if (std::isnan(expected))
        {
            check(std::isnan(actual));
        }
        else if (std::isinf(expected))
        {
            check(std::isinf(actual) && std::signbit(actual) == std::signbit(expected));
        }
        else if (expected == 0.0f)
        {
            check(actual == 0.0f);
        }
        else
        {
            const float tolerance = std::max(std::numeric_limits<float>::denorm_min() * 4.0f,
                                             std::abs(expected) * 5e-4f);
            check_near(actual, expected, tolerance);
        }
    };
    std::array<float, 17> edge_actual = edge_values;
    float_exp_inplace(edge_actual.data(), static_cast<uint32_t>(edge_actual.size()));
    for (size_t index = 0; index < edge_actual.size(); ++index)
        check_exp(edge_actual[index], float_approximate_exp(edge_values[index]));
}

void test_float_approximate_exp()
{
    for (int step = -80; step <= 80; ++step)
    {
        const float value = static_cast<float>(step) * 0.25f;
        const float expected = std::exp(value);
        const float tolerance = std::max(1e-6f, std::abs(expected) * 5e-4f);
        check_near(float_approximate_exp(value), expected, tolerance);
    }
    check(std::isinf(float_approximate_exp(104.0f)));
    check(float_approximate_exp(-104.0f) == 0.0f);
    check(std::isnan(float_approximate_exp(std::numeric_limits<float>::quiet_NaN())));
}

void test_int8_float_dot()
{
    std::array<int8_t, 137> left = {};
    std::array<float, 137> right = {};
    float expected = 0.0f;
    for (size_t index = 0; index < left.size(); ++index)
    {
        left[index] = static_cast<int8_t>(static_cast<int>(index % 23) - 11);
        right[index] = static_cast<float>(static_cast<int>(index % 17) - 8)
                       * 0.0625f;
        expected += static_cast<float>(left[index]) * right[index];
    }
    check_near(int8_float_dot(left.data(), right.data(), static_cast<uint32_t>(left.size())),
               expected,
               1e-4f);
    check_near(int8_float_dot(left.data(), right.data(), 0), 0.0f, 1e-6f);

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    if ((cpu_isa_flags() & CpuIsaX86Avx2Fma) != 0)
    {
        // Exercise AVX2 directly so AVX512-capable hosts cannot mask its tail path.
        check_near(avx2_int8_float_dot(left.data(), right.data(), static_cast<uint32_t>(left.size())),
                   expected,
                   1e-4f);
        constexpr std::array<uint32_t, 23> counts = {
            0,
            1,
            4,
            7,
            8,
            9,
            12,
            15,
            16,
            23,
            24,
            31,
            32,
            33,
            39,
            40,
            47,
            48,
            55,
            56,
            63,
            64,
            137,
        };
        std::array<int8_t, 137> avx2_left = {};
        std::array<float, 137> avx2_right = {};
        for (size_t index = 0; index < avx2_left.size(); ++index)
        {
            avx2_left[index] = static_cast<int8_t>(index % 13 + 1);
            avx2_right[index] = static_cast<float>(index % 7 + 1) * 0.125f;
        }
        for (const uint32_t count : counts)
        {
            float avx2_expected = 0.0f;
            for (uint32_t index = 0; index < count; ++index)
            {
                avx2_expected += static_cast<float>(avx2_left[index])
                                 * avx2_right[index];
            }
            check_near(avx2_int8_float_dot(avx2_left.data(), avx2_right.data(), count),
                       avx2_expected,
                       1e-4f);
        }
    }
#endif
}

void test_weighted_scale()
{
    std::array<float, 35> input = {};
    std::array<float, 35> float_weight = {};
    std::array<uint16_t, 35> bfloat16_weight = {};
    std::array<float, 35> float_output = {};
    std::array<float, 35> bfloat16_output = {};
    constexpr float scale = 0.3125f;
    constexpr float offset = 0.125f;
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 9) - 4)
                       * 0.1875f;
        float_weight[index] = static_cast<float>(static_cast<int>(index % 7) - 3)
                              * 0.25f;
        bfloat16_weight[index] = float_to_bfloat16(float_weight[index]);
    }
    float_weighted_scale(float_output.data(), input.data(), float_weight.data(), scale, offset,
                         static_cast<uint32_t>(input.size()));
    bfloat16_weighted_scale(bfloat16_output.data(), input.data(), bfloat16_weight.data(), scale,
                            offset, static_cast<uint32_t>(input.size()));
    for (size_t index = 0; index < input.size(); ++index)
    {
        check_near(float_output[index],
                   input[index] * scale * (float_weight[index] + offset),
                   1e-6f);
        check_near(bfloat16_output[index],
                   input[index] * scale
                       * (bfloat16_to_float(bfloat16_weight[index]) + offset),
                   1e-6f);
    }

    float_weighted_scale(input.data(), input.data(), float_weight.data(), scale, offset,
                         static_cast<uint32_t>(input.size()));
    for (size_t index = 0; index < input.size(); ++index)
        check_near(input[index], float_output[index], 1e-6f);
}

void test_rms_vector_kernels()
{
    constexpr uint32_t count = 137;
    constexpr float epsilon = 1e-5f;
    constexpr float weight_offset = 0.125f;
    std::array<float, count> input = {};
    std::array<float, count> float_weight = {};
    std::array<uint16_t, count> bfloat16_weight = {};
    std::array<float, count> float_output = {};
    std::array<float, count> bfloat16_output = {};
    for (uint32_t index = 0; index < count; ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0625f;
        float_weight[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125f;
        bfloat16_weight[index] = float_to_bfloat16(float_weight[index]);
    }
    float square_sum = 0.0f;
    for (float value : input)
        square_sum += value * value;
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    float_rms_norm(float_output.data(), input.data(), float_weight.data(), epsilon,
                   weight_offset, count);
    bfloat16_rms_norm(bfloat16_output.data(), input.data(), bfloat16_weight.data(), epsilon,
                      weight_offset, count);
    for (uint32_t index = 0; index < count; ++index)
    {
        check_near(float_output[index],
                   input[index] * inverse_rms * (float_weight[index] + weight_offset),
                   2e-4f);
        check_near(bfloat16_output[index],
                   input[index] * inverse_rms
                       * (bfloat16_to_float(bfloat16_weight[index]) + weight_offset),
                   2e-4f);
    }

    for (uint32_t long_count : {2049u, 4097u})
    {
        std::vector<float> long_input(long_count, 1e-4f);
        long_input.front() = 1.0f;
        long_input.back() = 0.5f;
        std::vector<float> long_float_weight(long_count);
        std::vector<uint16_t> long_bfloat16_weight(long_count);
        for (uint32_t index = 0; index < long_count; ++index)
        {
            long_float_weight[index] = 0.5f + static_cast<float>(index % 7) * 0.0625f;
            long_bfloat16_weight[index] = float_to_bfloat16(long_float_weight[index]);
        }

        double long_square_sum = 0.0;
        for (float value : long_input)
            long_square_sum += static_cast<double>(value) * value;
        const double long_inverse_rms = 1.0 / std::sqrt(long_square_sum / static_cast<double>(long_count) + static_cast<double>(epsilon));
        std::vector<float> long_float_output(long_count);
        std::vector<float> long_bfloat16_output(long_count);
        float_rms_norm(long_float_output.data(), long_input.data(), long_float_weight.data(),
                       epsilon, weight_offset, long_count);
        bfloat16_rms_norm(long_bfloat16_output.data(), long_input.data(),
                          long_bfloat16_weight.data(), epsilon, weight_offset, long_count);

        std::vector<float> long_float_alias = long_input;
        std::vector<float> long_bfloat16_alias = long_input;
        float_rms_norm(long_float_alias.data(), long_float_alias.data(), long_float_weight.data(),
                       epsilon, weight_offset, long_count);
        bfloat16_rms_norm(long_bfloat16_alias.data(), long_bfloat16_alias.data(),
                          long_bfloat16_weight.data(), epsilon, weight_offset, long_count);

        for (uint32_t index = 0; index < long_count; ++index)
        {
            const double float_expected = static_cast<double>(long_input[index]) * long_inverse_rms
                                          * (static_cast<double>(long_float_weight[index])
                                             + static_cast<double>(weight_offset));
            const double bfloat16_weight_value = static_cast<double>(bfloat16_to_float(long_bfloat16_weight[index]));
            const double bfloat16_expected = static_cast<double>(long_input[index]) * long_inverse_rms
                                             * (bfloat16_weight_value
                                                + static_cast<double>(weight_offset));
            check(std::abs(static_cast<double>(long_float_output[index]) - float_expected)
                  <= 3e-5 * std::abs(float_expected));
            check(std::abs(static_cast<double>(long_bfloat16_output[index]) - bfloat16_expected)
                  <= 3e-5 * std::abs(bfloat16_expected));
            check(long_float_alias[index] == long_float_output[index]);
            check(long_bfloat16_alias[index] == long_bfloat16_output[index]);
        }
    }

    std::array<float, count> normalized = input;
    float_rms_scale_inplace(normalized.data(), epsilon, count);
    for (uint32_t index = 0; index < count; ++index)
        check_near(normalized[index], input[index] * inverse_rms, 2e-4f);
    float_rms_scale_inplace(normalized.data(), epsilon, 0);

    // Batch normalization can reuse its input, including SIMD tail elements.
    ActivationBuffer batch(3, count);
    for (size_t row = 0; row < batch.rows(); ++row)
        for (uint32_t column = 0; column < count; ++column)
            batch.row(row)[column] = input[column] * static_cast<float>(row + 1);
    for (DType dtype : {DType::Float32, DType::BFloat16})
    {
        TensorData weight;
        weight.dtype = dtype;
        weight.shape = {count};
        if (dtype == DType::Float32)
            weight.float32_data.assign(float_weight.begin(), float_weight.end());
        else
            weight.bfloat16_data.assign(bfloat16_weight.begin(), bfloat16_weight.end());
        const ActivationBuffer expected = forward_rms_norm(batch, weight, epsilon, weight_offset);
        ActivationBuffer inplace = batch;
        const std::byte* storage = inplace.bytes().data();
        forward_rms_norm(inplace, weight, epsilon, inplace, weight_offset);
        check(inplace.bytes().data() == storage);
        check(inplace.rows() == batch.rows());
        check(inplace.columns() == batch.columns());
        for (size_t row = 0; row < batch.rows(); ++row)
            for (uint32_t column = 0; column < count; ++column)
                check_near(inplace.row(row)[column], expected.row(row)[column], 0.0f);
    }
}

void test_l2_vector_kernel()
{
    constexpr uint32_t count = 137;
    constexpr float epsilon = 1e-6f;
    std::array<float, count> values = {};
    for (uint32_t index = 0; index < count; ++index)
        values[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.125f;
    float square_sum = 0.0f;
    for (float value : values)
        square_sum += value * value;
    const float inverse_norm = 1.0f / std::sqrt(square_sum + epsilon);
    const std::array<float, count> input = values;
    float_l2_scale_inplace(values.data(), epsilon, count);
    for (uint32_t index = 0; index < count; ++index)
        check_near(values[index], input[index] * inverse_norm, 2e-5f);
    float_l2_scale_inplace(values.data(), epsilon, 0);
}

void test_hyper_connection_vector_kernels()
{
    constexpr uint32_t hidden_size = 19;
    std::array<float, hidden_size * 4> input = {};
    std::array<float, hidden_size> reduced = {};
    constexpr std::array<float, 4> pre = {0.25f, -0.5f, 0.75f, 1.25f};
    for (size_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
    float_hc_pre_4(reduced.data(), input.data(), pre[0], pre[1], pre[2], pre[3], hidden_size);
    for (uint32_t index = 0; index < hidden_size; ++index)
    {
        float expected = 0.0f;
        for (uint32_t copy = 0; copy < 4; ++copy)
            expected += input[static_cast<size_t>(copy) * hidden_size + index] * pre[copy];
        check_near(reduced[index], expected, 2e-5f);
    }

    std::array<float, hidden_size> branch = {};
    std::array<float, hidden_size * 4> residual = {};
    std::array<float, 4> post = {0.5f, 1.0f, -0.75f, 1.5f};
    std::array<float, 16> combine = {};
    std::array<float, hidden_size * 4> output = {};
    for (uint32_t index = 0; index < hidden_size; ++index)
        branch[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25f;
    for (size_t index = 0; index < residual.size(); ++index)
        residual[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.125f;
    for (size_t index = 0; index < combine.size(); ++index)
        combine[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.0625f;
    float_hc_post_4(output.data(), branch.data(), residual.data(), post.data(), combine.data(), hidden_size);
    for (uint32_t output_index = 0; output_index < 4; ++output_index)
    {
        for (uint32_t index = 0; index < hidden_size; ++index)
        {
            float expected = branch[index] * post[output_index];
            for (uint32_t residual_index = 0; residual_index < 4; ++residual_index)
                expected += residual[static_cast<size_t>(residual_index) * hidden_size + index]
                            * combine[residual_index * 4 + output_index];
            check_near(output[static_cast<size_t>(output_index) * hidden_size + index],
                       expected,
                       2e-5f);
        }
    }
}

void test_float_rope_kernel()
{
    constexpr uint32_t dimension = 34;
    constexpr uint32_t half_dimension = dimension / 2;
    std::array<float, dimension> values = {};
    std::array<float, half_dimension> cosine = {};
    std::array<float, half_dimension> sine = {};
    std::array<float, dimension> expected = {};
    for (uint32_t index = 0; index < dimension; ++index)
        values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        cosine[index] = 0.75f + static_cast<float>(index % 5) * 0.03125f;
        sine[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.0625f;
    }
    expected = values;
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float first = values[index];
        const float second = values[half_dimension + index];
        expected[index] = first * cosine[index] - second * sine[index];
        expected[half_dimension + index] = second * cosine[index] + first * sine[index];
    }
    float_rope_inplace(values.data(), cosine.data(), sine.data(), dimension);
    for (uint32_t index = 0; index < dimension; ++index)
        check_near(values[index], expected[index], 2e-5f);
}

void test_float_silu_mul()
{
    std::array<float, 97> gate = {};
    std::array<float, 97> up = {};
    std::array<float, 97> output = {};
    for (size_t index = 0; index < gate.size(); ++index)
    {
        gate[index] = -7.0f + static_cast<float>(index % 29) * 0.5f;
        up[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    }
    constexpr float sigmoid_scale = 1.702f;
    constexpr float up_offset = 1.0f;
    float_silu_mul(output.data(), gate.data(), up.data(), sigmoid_scale, up_offset,
                   static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
    {
        const float gate_value = gate[index];
        const float expected = gate_value / (1.0f + std::exp(-sigmoid_scale * gate_value))
                               * (up[index] + up_offset);
        check_near(output[index], expected, 5e-4f);
    }

    constexpr std::array<uint32_t, 6> counts = {4, 5, 7, 8, 9, 17};
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float positive_104 = 104.0f;
    const float negative_104 = -104.0f;
    const float positive_88 = 88.72f;
    const float negative_88 = -88.72f;
    const std::array<float, 17> edge_gate = {
        0.0f,
        -0.0f,
        infinity,
        -infinity,
        nan,
        positive_104,
        std::nextafter(positive_104, -infinity),
        std::nextafter(positive_104, infinity),
        negative_104,
        std::nextafter(negative_104, 0.0f),
        std::nextafter(negative_104, -infinity),
        positive_88,
        std::nextafter(positive_88, 0.0f),
        std::nextafter(positive_88, infinity),
        negative_88,
        std::nextafter(negative_88, 0.0f),
        std::nextafter(negative_88, -infinity),
    };
    std::array<float, 17> edge_up = {};
    for (size_t index = 0; index < edge_up.size(); ++index)
        edge_up[index] = index % 2 == 0 ? 1.25f : -0.75f;
    const auto expected_silu_mul = [](float gate_value, float up_value) {
        return gate_value / (1.0f + float_approximate_exp(-gate_value)) * up_value;
    };
    const auto check_silu_mul = [](float actual, float expected) {
        if (std::isnan(expected))
        {
            check(std::isnan(actual));
        }
        else if (std::isinf(expected))
        {
            check(std::isinf(actual) && std::signbit(actual) == std::signbit(expected));
        }
        else if (expected == 0.0f)
        {
            check(actual == 0.0f);
        }
        else
        {
            const float tolerance = std::max(std::numeric_limits<float>::denorm_min() * 4.0f,
                                             std::abs(expected) * 5e-4f);
            check_near(actual, expected, tolerance);
        }
    };
    for (uint32_t count : counts)
    {
        std::array<float, 17> separate_output = {};
        float_silu_mul(separate_output.data(), edge_gate.data(), edge_up.data(), 1.0f, 0.0f, count);
        std::array<float, 17> aliased_output = edge_gate;
        float_silu_mul(aliased_output.data(), aliased_output.data(), edge_up.data(), 1.0f, 0.0f, count);
        for (uint32_t index = 0; index < count; ++index)
        {
            const float expected = expected_silu_mul(edge_gate[index], edge_up[index]);
            check_silu_mul(separate_output[index], expected);
            check_silu_mul(aliased_output[index], expected);
        }
    }

    for (uint32_t count : counts)
    {
        std::array<float, 17> in_place = edge_gate;
        float_silu_inplace(in_place.data(), count);
        for (uint32_t index = 0; index < count; ++index)
            check_silu_mul(in_place[index], expected_silu_mul(edge_gate[index], 1.0f));
    }

    std::array<float, 9> positive_gate = {};
    std::array<float, 9> negative_zero_up = {};
    std::array<float, 9> signed_zero_output = {};
    positive_gate.fill(1.0f);
    negative_zero_up.fill(-0.0f);
    float_silu_mul(signed_zero_output.data(),
                   positive_gate.data(),
                   negative_zero_up.data(),
                   1.0f,
                   0.0f,
                   static_cast<uint32_t>(signed_zero_output.size()));
    for (float value : signed_zero_output)
        check(value == 0.0f && std::signbit(value));

    constexpr uint32_t compact_rows = 3;
    constexpr uint32_t compact_columns = 9;
    std::array<float, compact_rows * compact_columns * 2> packed_gate_up = {};
    std::array<float, compact_rows * compact_columns> compact_expected = {};
    for (uint32_t row = 0; row < compact_rows; ++row)
    {
        for (uint32_t column = 0; column < compact_columns; ++column)
        {
            const float gate_value = static_cast<float>(static_cast<int>((row * 7 + column * 3) % 17) - 8)
                                     * 0.25f;
            const float up_value = static_cast<float>(static_cast<int>((row * 5 + column * 11) % 13) - 6)
                                   * 0.125f;
            packed_gate_up[static_cast<size_t>(row) * compact_columns * 2 + column] = gate_value;
            packed_gate_up[static_cast<size_t>(row) * compact_columns * 2 + compact_columns + column] = up_value;
            compact_expected[static_cast<size_t>(row) * compact_columns + column] = expected_silu_mul(gate_value, up_value);
        }
    }
    for (uint32_t row = 0; row < compact_rows; ++row)
    {
        const float* source = packed_gate_up.data() + static_cast<size_t>(row) * compact_columns * 2;
        float* destination = packed_gate_up.data() + static_cast<size_t>(row) * compact_columns;
        float_silu_mul(destination, source, source + compact_columns, 1.0f, 0.0f, compact_columns);
    }
    for (size_t index = 0; index < compact_expected.size(); ++index)
        check_near(packed_gate_up[index], compact_expected[index], 5e-4f);
}

void test_float_sigmoid_mul()
{
    std::array<float, 97> gate = {};
    std::array<float, 97> input = {};
    std::array<float, 97> output = {};
    for (size_t index = 0; index < gate.size(); ++index)
    {
        gate[index] = -7.0f + static_cast<float>(index % 29) * 0.5f;
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    }
    float_sigmoid_mul(output.data(),
                      gate.data(),
                      input.data(),
                      static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
        check_near(output[index],
                   input[index] / (1.0f + std::exp(-gate[index])),
                   5e-4f);

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 17> edges = {
        -infinity, -104.0f, std::nextafter(-104.0f, 0.0f), -90.0f,
        -88.0f, -80.0f, -1.0f, -0.0f, 0.0f, 1.0f, 80.0f, 88.0f,
        std::nextafter(104.0f, 0.0f), 104.0f, infinity, nan, -infinity};
    for (size_t index = 0; index < gate.size(); ++index)
    {
        gate[index] = edges[index % edges.size()];
        input[index] = 0.75f;
        if (index % 11 == 0)
            input[index] = -0.0f;
        else if (index % 13 == 0)
            input[index] = infinity;
        else if (index % 19 == 0)
            input[index] = nan;
    }
    for (uint32_t count : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 96u, 97u})
    {
        output.fill(123.0f);
        std::array<float, 97> gate_alias = gate;
        std::array<float, 97> input_alias = input;
        float_sigmoid_mul(output.data(), gate.data(), input.data(), count);
        float_sigmoid_mul(gate_alias.data(), gate_alias.data(), input.data(), count);
        float_sigmoid_mul(input_alias.data(), gate.data(), input_alias.data(), count);
        for (size_t index = 0; index < gate.size(); ++index)
        {
            if (index >= count)
            {
                check(output[index] == 123.0f);
                check(std::memcmp(&gate_alias[index], &gate[index], sizeof(float)) == 0);
                check(std::memcmp(&input_alias[index], &input[index], sizeof(float)) == 0);
                continue;
            }
            // Preserve scalar exp overflow and non-finite behavior in SIMD lanes and tails.
            const float expected = input[index] / (1.0f + float_approximate_exp(-gate[index]));
            for (float actual : {output[index], gate_alias[index], input_alias[index]})
            {
                if (std::isnan(expected))
                    check(std::isnan(actual));
                else if (std::isinf(expected) || expected == 0.0f)
                    check(actual == expected && std::signbit(actual) == std::signbit(expected));
                else
                    check_near(actual, expected, std::max(std::numeric_limits<float>::denorm_min() * 4.0f, std::abs(expected) * 5e-4f));
            }
        }
    }
}

void test_dense_expert_thread_budget()
{
    // Compare the same Expert outputs across outer Expert teams, inner
    // matrix teams and serial execution, including reused uneven batches.
    for (DType dtype : {DType::Float32, DType::BFloat16})
    {
        for (uint32_t hidden : {32u, 1024u, 256u, 257u, 1025u})
        {
            if (dtype == DType::Float32 && hidden != 32 && hidden != 1024)
                continue;
            const bool grouped_bfloat16_case = dtype == DType::BFloat16
                                               && (hidden == 256 || hidden == 257 || hidden == 1025);
            const uint32_t intermediate = hidden == 1025 ? 2051u : hidden * 2;
            CompiledModel model;
            model.descriptor.hidden_size = hidden;
            model.opt.hybrid_mode = HybridMode::CpuOnly;
            model.opt.optimization_flags = g_test_optimization_flags;
            MoeBlockPlan moe;
            const size_t expert_count = hidden == 32 ? 5 : 2;
            moe.experts.resize(expert_count);
            const auto add_weight = [&](const std::string& name, uint32_t rows, uint32_t columns, uint32_t seed) {
                std::vector<float> values(static_cast<size_t>(rows) * columns);
                for (size_t index = 0; index < values.size(); ++index)
                    values[index] = static_cast<float>(static_cast<int>((index * 17 + seed) % 37) - 18) * 0.003f;
                if (dtype == DType::BFloat16)
                    return add_bfloat16_tensor(model.weights, name, {rows, columns}, values);
                return add_float_tensor(model.weights, name, {rows, columns}, std::move(values));
            };
            for (uint32_t expert_id = 0; expert_id < moe.experts.size(); ++expert_id)
            {
                ExpertPlan& expert = moe.experts[expert_id];
                const std::string prefix = "expert." + std::to_string(expert_id) + ".";
                expert.layout = ExpertLayout::PackedGateUpDown;
                expert.activation = ExpertActivation::Silu;
                if (grouped_bfloat16_case && hidden == 1025 && expert_id == 0)
                    expert.activation_limit = 0.125f;
                expert.gate_up_weight = add_weight(prefix + "gate_up", intermediate * 2, hidden, expert_id + 1);
                expert.down_weight = add_weight(prefix + "down", hidden, intermediate, expert_id + 9);
                if (!grouped_bfloat16_case)
                {
                    expert.gate_up_bias = add_float_tensor(model.weights, prefix + "gate_up_bias", {intermediate * 2}, std::vector<float>(intermediate * 2, 0.01f));
                    expert.down_bias = add_float_tensor(model.weights, prefix + "down_bias", {hidden}, std::vector<float>(hidden, -0.02f));
                }
            }

            LayerState state;
            state.resize_experts(moe.experts.size());
            ExpertScratch scratch;
            CpuOpenMpThreadLimitScope thread_limit;
            size_t retained_workspace_count = 0;
            std::vector<const std::byte*> retained_workspace_storage;
            std::vector<uint64_t> retained_workspace_capacity;
            bool checked_cold_backend_scratch = false;
            bool poisoned_backend_scratch = false;
            for (uint32_t token_count : {1u, 3u, 8u, 1u})
            {
                // Keep fast SiLU for the first decode, then test scalar activation on reused shapes.
                if (grouped_bfloat16_case && hidden == 256 && token_count > 1)
                    model.opt.optimization_flags = g_test_optimization_flags & ~OptimizationCpuFastSilu;
                state.normalized.reset(token_count, hidden, false);
                for (uint32_t row = 0; row < token_count; ++row)
                    for (uint32_t column = 0; column < hidden; ++column)
                        state.normalized.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 5) % 31) - 15) * 0.03f;
                for (uint32_t expert_id = 0; expert_id < moe.experts.size(); ++expert_id)
                {
                    ExpertState& active = state.active_experts()[expert_id];
                    active.batch.expert_id = expert_id;
                    active.batch.routes.clear();
                    const uint32_t rows = expert_id == 0 ? token_count : 1;
                    for (uint32_t row = 0; row < rows; ++row)
                        active.batch.routes.push_back({token_count - row - 1, expert_id, 0.5f});
                }
                if (hidden == 32 && token_count == 1 && scratch.expert_workspaces.size() >= 4 && retained_workspace_count == 0)
                {
                    retained_workspace_count = scratch.expert_workspaces.size();
                    retained_workspace_storage.resize(retained_workspace_count);
                    retained_workspace_capacity.resize(retained_workspace_count);
                    for (size_t index = 0; index < retained_workspace_count; ++index)
                    {
                        retained_workspace_storage[index] = scratch.expert_workspaces[index].projection.bytes().data();
                        retained_workspace_capacity[index] = scratch.expert_workspaces[index].projection.allocated_bytes();
                    }
                }
                std::vector<ActivationBuffer> reference;
                for (uint32_t threads : {1u, 2u, 4u, 8u})
                {
                    if (threads == 8 && !grouped_bfloat16_case)
                        continue;
                    thread_limit.set(threads);
                    state.reset();
                    scratch.decode_tasks.push_back({});
                    SessionStatistics statistics;
                    const bool check_cold_backend_scratch = hidden == 32
                                                            && token_count == 1
                                                            && threads == 1
                                                            && !checked_cold_backend_scratch;
                    if (check_cold_backend_scratch)
                    {
                        check(scratch.backend_status.capacity() == 0);
                        check(scratch.backend_aggregated.capacity() == 0);
                        check(scratch.backend_indices.capacity() == 0);
                        check(scratch.backend_requests.capacity() == 0);
                    }
                    if (hidden == 32 && token_count == 1 && threads == 2 && !poisoned_backend_scratch)
                    {
                        const size_t stale_count = state.active_experts().size() + 3;
                        scratch.backend_status.assign(stale_count, ExpertStatus::Executed);
                        scratch.backend_aggregated.assign(stale_count, 1);
                        scratch.backend_indices.assign(stale_count, std::numeric_limits<size_t>::max());
                        scratch.backend_requests.resize(stale_count);
                        scratch.backend_aggregated_output_valid = true;
                        for (ExpertState& active : state.active_experts())
                        {
                            std::fill_n(active.output.row(0), active.output.values().size(), -91.0f);
                        }
                        poisoned_backend_scratch = true;
                    }
                    const bool prefetch = !grouped_bfloat16_case
                                          && threads == 1
                                          && token_count == 1;
                    check(static_cast<bool>(forward_moe(model, moe, state, &statistics, scratch, 0, ExecutionBackend::Cpu, prefetch)));
                    check(state.experts_executed);
                    check(scratch.decode_tasks.empty());
                    check(scratch.backend_status.empty());
                    check(scratch.backend_aggregated.empty());
                    check(scratch.backend_indices.empty());
                    check(scratch.backend_requests.empty());
                    check(!scratch.backend_aggregated_output_valid);
                    if (check_cold_backend_scratch)
                    {
                        check(scratch.backend_status.capacity() == 0);
                        check(scratch.backend_aggregated.capacity() == 0);
                        check(scratch.backend_indices.capacity() == 0);
                        check(scratch.backend_requests.capacity() == 0);
                        checked_cold_backend_scratch = true;
                    }
                    if (threads == 1 && retained_workspace_count != 0)
                    {
                        check(scratch.expert_workspaces.size() == retained_workspace_count);
                        for (size_t index = 1; index < retained_workspace_count; ++index)
                        {
                            check(scratch.expert_workspaces[index].projection.bytes().data() == retained_workspace_storage[index]);
                            check(scratch.expert_workspaces[index].projection.allocated_bytes() == retained_workspace_capacity[index]);
                        }
                    }
                    for (const ExpertState& active : state.active_experts())
                    {
                        check(active.output.rows() == active.batch.routes.size());
                        check(active.output.columns() == hidden);
                        if (threads == 1)
                        {
                            reference.push_back(active.output);
                            continue;
                        }
                        const ActivationBuffer& expected = reference[active.batch.expert_id];
                        for (size_t index = 0; index < active.output.values().size(); ++index)
                        {
                            check_near(active.output.values()[index], expected.values()[index], 1e-5f + 1e-5f * std::abs(expected.values()[index]));
#if defined(_OPENMP) && (defined(__aarch64__) || defined(_M_ARM64))
                            if (grouped_bfloat16_case && token_count == 1)
                                check(std::bit_cast<uint32_t>(active.output.values()[index]) == std::bit_cast<uint32_t>(expected.values()[index]));
#endif
                        }
                    }
                }
            }
            if (hidden == 32)
            {
                std::vector<float> previous_output;
                for (uint32_t pass = 0; pass < 3; ++pass)
                {
                    const size_t route_count = pass == 1 ? 3 : 1;
                    state.normalized.reset(1, hidden, false);
                    for (uint32_t column = 0; column < hidden; ++column)
                    {
                        const int sample = static_cast<int>((column * 11 + pass * 7) % 29) - 14;
                        state.normalized.row(0)[column] = static_cast<float>(sample) * 0.03125f;
                    }
                    state.reset();
                    for (uint32_t expert_id = 0; expert_id < moe.experts.size(); ++expert_id)
                    {
                        ExpertState& active = state.active_experts()[expert_id];
                        active.batch.expert_id = expert_id;
                        active.batch.routes.assign(route_count, {0, expert_id, 1.0f});
                    }
                    thread_limit.set(1);
                    SessionStatistics statistics;
                    const auto status = forward_moe(model,
                                                    moe,
                                                    state,
                                                    &statistics,
                                                    scratch,
                                                    0,
                                                    ExecutionBackend::Cpu,
                                                    false);
                    check(static_cast<bool>(status));
                    check(state.experts_executed && !state.normalized_device);
                    for (const ExpertState& active : state.active_experts())
                    {
                        check(active.input.rows() == (route_count == 1 ? 0 : route_count));
                        if (route_count != 1)
                        {
                            check(active.input.columns() == hidden);
                            for (size_t row = 0; row < route_count; ++row)
                                for (uint32_t column = 0; column < hidden; ++column)
                                    check(active.input.row(row)[column] == state.normalized.row(0)[column]);
                        }
                        check(active.output.rows() == route_count && active.output.columns() == hidden);

                        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                        const ActivationBuffer projected = forward_linear(model.weights.at(expert.gate_up_weight),
                                                                          model.weights.at(expert.gate_up_bias),
                                                                          state.normalized,
                                                                          model.opt.optimization_flags);
                        ActivationBuffer activated(1, intermediate);
                        for (uint32_t column = 0; column < intermediate; ++column)
                            activated.row(0)[column] = scaled_silu(projected.row(0)[column], 1.0f, model.opt.optimization_flags)
                                                       * projected.row(0)[intermediate + column];
                        const ActivationBuffer expected = forward_linear(model.weights.at(expert.down_weight),
                                                                         model.weights.at(expert.down_bias),
                                                                         activated,
                                                                         model.opt.optimization_flags);
                        for (size_t row = 0; row < route_count; ++row)
                            for (uint32_t column = 0; column < hidden; ++column)
                                check_near(active.output.row(row)[column], expected.row(0)[column],
                                           1e-5f + 1e-5f * std::abs(expected.row(0)[column]));
                    }
                    const ExpertState& first = state.active_experts().front();
                    if (!previous_output.empty())
                    {
                        bool changed = false;
                        for (uint32_t column = 0; column < hidden; ++column)
                            changed = changed || std::abs(first.output.row(0)[column] - previous_output[column]) > 1e-6f;
                        check(changed);
                    }
                    previous_output.assign(first.output.row(0), first.output.row(0) + hidden);
                }
            }
        }
    }
}

void test_bfloat16_vector_kernels()
{
    std::array<uint16_t, 129> weights = {};
    std::array<float, 129> input = {};
    for (size_t index = 0; index < weights.size(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.375f;
        weights[index] = float_to_bfloat16(value);
        input[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.125f;
    }

    for (uint32_t count : {
             0u, 1u, 7u, 8u, 15u, 16u, 17u, 32u, 33u,
             63u, 64u, 65u, 127u, 128u, 129u})
    {
        float expected_dot = 0.0f;
        float expected_pair_dot = 0.0f;
        std::array<uint16_t, 129> pair_input = {};
        for (uint32_t index = 0; index < count; ++index)
        {
            expected_dot += bfloat16_to_float(weights[index]) * input[index];
            pair_input[index] = float_to_bfloat16(input[index]);
            expected_pair_dot += bfloat16_to_float(weights[index])
                                 * bfloat16_to_float(pair_input[index]);
        }
        check_near(bfloat16_dot(weights.data(), input.data(), count), expected_dot, 1e-4f);
        check_near(bfloat16_pair_dot(weights.data(),
                                     pair_input.data(),
                                     count),
                   expected_pair_dot,
                   1e-4f);

        std::array<float, 129> output = {};
        std::array<float, 129> expected_output = {};
        constexpr float scale = -0.625f;
        for (uint32_t index = 0; index < count; ++index)
        {
            output[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.25f;
            expected_output[index] = output[index] + scale * bfloat16_to_float(weights[index]);
        }
        bfloat16_scaled_add(output.data(), weights.data(), scale, count);
        for (uint32_t index = 0; index < count; ++index)
            check_near(output[index], expected_output[index], 1e-5f);
    }
}

void test_bfloat16_dot_reduction()
{
    const auto next_u32 = [](uint32_t& state) noexcept {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (uint32_t count : {
             0u, 1u, 3u, 4u, 5u, 15u, 16u, 17u, 31u, 32u, 33u,
             63u, 64u, 65u, 127u, 128u, 129u, 1023u, 1024u,
             1025u, 4099u})
    {
        std::vector<uint16_t> weights(count + 8, 0x7fc0u);
        std::vector<float> input(count + 8, std::numeric_limits<float>::quiet_NaN());
        uint32_t weight_state = 0x1234abcdu ^ count;
        uint32_t input_state = 0x9e3779b9u ^ count;
        bool weight_positive = false;
        bool weight_negative = false;
        bool input_positive = false;
        bool input_negative = false;
        double reference = 0.0;
        for (uint32_t index = 0; index < count; ++index)
        {
            const int weight_sample = static_cast<int>(next_u32(weight_state) % 2001u) - 1000;
            const int input_sample = static_cast<int>(next_u32(input_state) % 2001u) - 1000;
            const float weight_value = static_cast<float>(weight_sample) * (1.0f / 256.0f);
            const float input_value = static_cast<float>(input_sample) * (1.0f / 512.0f);
            weights[index + 1] = float_to_bfloat16(weight_value);
            input[index + 1] = input_value;
            weight_positive = weight_positive || weight_sample > 0;
            weight_negative = weight_negative || weight_sample < 0;
            input_positive = input_positive || input_sample > 0;
            input_negative = input_negative || input_sample < 0;
            reference += static_cast<double>(bfloat16_to_float(weights[index + 1]))
                         * static_cast<double>(input[index + 1]);
        }
        if (count >= 16)
        {
            check(weight_positive && weight_negative);
            check(input_positive && input_negative);
        }

        const float actual = bfloat16_dot(weights.data() + 1,
                                          input.data() + 1,
                                          count);
        const double error = std::abs(static_cast<double>(actual) - reference);
        const double tolerance = 2.0e-5 + 2.0e-5 * std::abs(reference);
        check(std::isfinite(actual));
        check(error <= tolerance);
    }
}

void test_float_to_bfloat16_array()
{
    std::array<float, 37> input = {};
    std::array<uint16_t, 37> output = {};
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8)
                           * 0.03125f
                       + static_cast<float>(index) * 1e-5f;
    }
    float_to_bfloat16_array(output.data(),
                            input.data(),
                            static_cast<uint32_t>(input.size()));
    for (size_t index = 0; index < input.size(); ++index)
        check(static_cast<bool>(output[index] == float_to_bfloat16(input[index])));
}

void test_activation_buffer_reset()
{
    ActivationBuffer buffer(2, 3, DType::Int8);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0});
    std::fill(buffer.mutable_bytes().begin(), buffer.mutable_bytes().end(), std::byte{0x5a});
    bool rejected = false;
    try
    {
        // Exceed vector's size limit without attempting a large allocation.
        buffer.reset(std::vector<std::byte>().max_size() + 1, 1, false);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    check(rejected);
    check(buffer.rows() == 2);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 6);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0x5a});

    rejected = false;
    try
    {
        // This product would wrap to zero before vector::resize sees it.
        buffer.reset(std::numeric_limits<size_t>::max() / 2 + 1, 2, false);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    check(rejected);
    check(buffer.rows() == 2);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 6);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0x5a});

    ActivationBuffer typed(1, 2, DType::BFloat16);
    rejected = false;
    try
    {
        typed.reset(std::numeric_limits<size_t>::max() / sizeof(uint16_t) + 1, 1, false);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    check(rejected);
    check(typed.rows() == 1);
    check(typed.columns() == 2);
    check(typed.dtype() == DType::BFloat16);
    check(typed.element_size() == sizeof(uint16_t));
    check(typed.bytes().size() == 2 * sizeof(uint16_t));

    rejected = false;
    try
    {
        ActivationBuffer packed(1, 2, DType::MxFp4);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    check(rejected);

    buffer.reset(1, 3, false);
    check(buffer.rows() == 1);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 3);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0x5a});

    buffer.reset(3, 3, true);
    check(buffer.rows() == 3);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 9);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0});

    std::fill(buffer.mutable_bytes().begin(), buffer.mutable_bytes().end(), std::byte{0x5a});
    buffer.reset(4, 3, false);
    for (size_t index = 0; index < buffer.bytes().size(); ++index)
        check(buffer.bytes()[index] == (index < 9 ? std::byte{0x5a} : std::byte{0}));

    buffer.reset(2, 3, true);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0});

    std::fill(buffer.mutable_bytes().begin(), buffer.mutable_bytes().end(), std::byte{0x5a});
    buffer.reset(2, 3, true);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0});

    buffer.clear();
    check(buffer.bytes().empty());
    buffer.reset(2, 3, false);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0});

    ActivationBuffer batch(2, 3);
    const std::array<float, 6> batch_values = {1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f};
    for (size_t row = 0; row < batch.rows(); ++row)
        std::copy_n(batch_values.data() + row * batch.columns(), batch.columns(), batch.row(row));
    auto vectors = batch_to_vectors(batch);
    check(vectors.size() == 2);
    check(vectors[0].size() == 3);
    check(vectors[1].size() == 3);
    for (size_t index = 0; index < batch_values.size(); ++index)
        check(vectors[index / batch.columns()][index % batch.columns()] == batch_values[index]);
    batch.row(0)[0] = 99.0f;
    check(vectors[0][0] == batch_values[0]);
    vectors[1][1] = -42.0f;
    check(batch.row(1)[1] == batch_values[4]);

    ActivationBuffer empty_rows(0, 137);
    auto empty_row_vectors = batch_to_vectors(empty_rows);
    check(empty_row_vectors.empty());

    ActivationBuffer zero_columns(2, 0);
    auto zero_column_vectors = batch_to_vectors(zero_columns);
    check(zero_column_vectors.size() == 2);
    for (const std::vector<float>& row : zero_column_vectors)
        check(row.empty());
}

void test_bfloat16_batched_linear_kernel()
{
    constexpr size_t token_count = 5;
    constexpr uint32_t input_columns = 512;
    constexpr uint32_t output_columns = 2048;
    std::vector<uint16_t> weights(static_cast<size_t>(output_columns) * input_columns);
    std::vector<float> input(token_count * input_columns);
    std::vector<float> output(token_count * output_columns, -7.0f);
    std::vector<float> expected(token_count * output_columns);
    for (size_t index = 0; index < weights.size(); ++index)
    {
        weights[index] = float_to_bfloat16(static_cast<float>(static_cast<int>((index * 17 + 5) % 67) - 33)
                                           * 0.0009765625f);
    }
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>((index * 13 + 7) % 29) - 14)
                           * 0.03125f
                       + static_cast<float>(static_cast<int>(index % 11) - 5) * 1e-5f;
    }
    for (size_t token = 0; token < token_count; ++token)
    {
        for (uint32_t output_column = 0;
             output_column < output_columns;
             ++output_column)
        {
            expected[token * output_columns + output_column] = bfloat16_dot(weights.data()
                                                                                + static_cast<size_t>(output_column) * input_columns,
                                                                            input.data() + token * input_columns,
                                                                            input_columns);
        }
    }

    const uint64_t batched_flags = OptimizationCpuBfloat16Batched;
    const uint64_t disabled_flags = g_test_optimization_flags & ~batched_flags;
    const uint64_t enabled_flags = g_test_optimization_flags | batched_flags;
    check(!forward_linear_bf16(weights.data(),
                               input.data(),
                               input_columns,
                               token_count,
                               output_columns,
                               input_columns,
                               output.data(),
                               output_columns,
                               4,
                               disabled_flags));
    check(static_cast<bool>(std::all_of(output.begin(), output.end(), [](float value) {
        return value == -7.0f;
    })));

    bool dispatched = false;
    dispatched = forward_linear_bf16(weights.data(),
                                     input.data(),
                                     input_columns,
                                     token_count,
                                     output_columns,
                                     input_columns,
                                     output.data(),
                                     output_columns,
                                     4,
                                     enabled_flags);
    if (dispatched)
    {
        for (size_t index = 0; index < output.size(); ++index)
            check_near(output[index], expected[index], 5e-5f);

        std::vector<float> single_output(output_columns, -9.0f);
        check(static_cast<bool>(forward_linear_bf16(weights.data(),
                                                    input.data(),
                                                    input_columns,
                                                    1,
                                                    output_columns,
                                                    input_columns,
                                                    single_output.data(),
                                                    output_columns,
                                                    4,
                                                    enabled_flags)));
        for (uint32_t output_column = 0;
             output_column < output_columns;
             ++output_column)
        {
            check(static_cast<bool>(single_output[output_column] == expected[output_column]));
        }
    }
    else
    {
        check(std::string(bfloat16_batched_linear_kernel_name(enabled_flags)) == "unavailable");
    }
}

void benchmark_bfloat16_attention_kernels()
{
    constexpr uint32_t head_dimension = 128;
    constexpr uint32_t token_count = 4096;
    constexpr uint32_t iterations = 16;
    constexpr float probability = 0.03125f;
    std::vector<uint16_t> keys(static_cast<size_t>(token_count) * head_dimension);
    std::vector<uint16_t> values(static_cast<size_t>(token_count) * head_dimension);
    std::vector<float> key_float(keys.size());
    std::vector<float> value_float(values.size());
    std::array<float, head_dimension> query = {};
    std::array<float, head_dimension> output = {};
    for (uint32_t column = 0; column < head_dimension; ++column)
        query[column] = static_cast<float>(static_cast<int>(column % 17) - 8) * 0.03125f;
    for (size_t index = 0; index < keys.size(); ++index)
    {
        const float key = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.015625f;
        const float value = static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0234375f;
        keys[index] = float_to_bfloat16(key);
        values[index] = float_to_bfloat16(value);
    }

    float baseline_checksum = 0.0f;
    float direct_checksum = 0.0f;
    const auto baseline_start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        for (size_t index = 0; index < keys.size(); ++index)
        {
            key_float[index] = bfloat16_to_float(keys[index]);
            value_float[index] = bfloat16_to_float(values[index]);
        }
        output.fill(0.0f);
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const size_t offset = static_cast<size_t>(token) * head_dimension;
            baseline_checksum += float_dot(query.data(), key_float.data() + offset, head_dimension);
            float_scaled_add(output.data(), value_float.data() + offset, probability, head_dimension);
        }
    }
    const auto baseline_end = std::chrono::steady_clock::now();

    const auto direct_start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        output.fill(0.0f);
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const size_t offset = static_cast<size_t>(token) * head_dimension;
            direct_checksum += bfloat16_dot(keys.data() + offset, query.data(), head_dimension);
            bfloat16_scaled_add(output.data(), values.data() + offset, probability, head_dimension);
        }
    }
    const auto direct_end = std::chrono::steady_clock::now();

    const double baseline_ms = std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();
    const double direct_ms = std::chrono::duration<double, std::milli>(direct_end - direct_start).count();
    std::cout << std::fixed << std::setprecision(3)
              << "bfloat16_attention_ab baseline_ms=" << baseline_ms
              << " direct_ms=" << direct_ms
              << " speedup=" << (baseline_ms / direct_ms)
              << " checksum_delta=" << std::abs(baseline_checksum - direct_checksum) << '\n';
}

void benchmark_qnk_gemm()
{
    constexpr uint32_t columns = 4096;
    constexpr size_t rows = 32;
    constexpr size_t token_count = 8;
    constexpr uint32_t iterations = 24;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };

    ActivationBuffer input(token_count, columns);
    for (size_t token = 0; token < token_count; ++token)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            input.row(token)[column] = static_cast<float>(static_cast<int>((column + token * 3) % 31) - 15)
                                       * 0.03125f;
        }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "qnk_gemm_kernel=" << mxfp4_kernel_name() << '\n';
    for (const DType dtype : dtypes)
    {
        const size_t raw_bytes = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
        std::vector<uint8_t> raw(raw_bytes, 0);
        const size_t row_bytes = raw_bytes / rows;
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + row * row_bytes + static_cast<size_t>(block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                    for (uint32_t index = 0; index < 64; ++index)
                        encoded[16 + index] = static_cast<uint8_t>(index * 7u + row + block);
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                    for (uint32_t index = 0; index < 32; ++index)
                        encoded[index] = static_cast<uint8_t>(index * 5u + row);
                    for (uint32_t index = 0; index < 64; ++index)
                        encoded[32 + index] = static_cast<uint8_t>(index * 11u + row + block);
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 8, uint8_t{1});
                    for (uint32_t index = 0; index < 128; ++index)
                        encoded[(dtype == DType::Q4K ? 16u : 48u) + index] = static_cast<uint8_t>(index * 13u + row + block);
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                    for (uint32_t index = 0; index < 128; ++index)
                        encoded[index] = static_cast<uint8_t>(index * 7u + row + block);
                    for (uint32_t index = 0; index < 64; ++index)
                        encoded[128 + index] = static_cast<uint8_t>(index * 5u + row);
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                    for (uint32_t index = 0; index < qnk_block_elements; ++index)
                        encoded[4 + index] = static_cast<uint8_t>(index * 9u + row + block);
                }
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = std::move(raw);
        ActivationBuffer direct_output;
        forward_linear(matrix, input, direct_output, 0);
        for (uint32_t iteration = 0; iteration < 2; ++iteration)
            forward_linear(matrix, input, direct_output, 0);

        const std::span<const uint8_t> raw_view = matrix.qnk_values();
        float reference_checksum = 0.0f;
        std::vector<float> reference_output(token_count * rows, 0.0f);
        alignas(64) float decoded[qnk_block_elements];
        const auto baseline_start = std::chrono::steady_clock::now();
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t token = 0; token < token_count; ++token)
                {
                    float sum = 0.0f;
                    for (uint32_t block = 0; block < block_count; ++block)
                    {
                        const uint8_t* encoded = raw_view.data()
                                                 + (row * static_cast<size_t>(block_count) + block) * block_bytes;
                        qnk_dequantize_block(dtype, encoded, decoded);
                        sum += float_dot(decoded,
                                         input.row(token) + static_cast<size_t>(block) * qnk_block_elements,
                                         qnk_block_elements);
                    }
                    reference_output[token * rows + row] = sum;
                }
            }
            reference_checksum += reference_output[(iteration % token_count) * rows + iteration % rows];
        }
        const auto baseline_end = std::chrono::steady_clock::now();

        float direct_checksum = 0.0f;
        const auto direct_start = std::chrono::steady_clock::now();
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            forward_linear(matrix, input, direct_output, 0);
            direct_checksum += direct_output.row(iteration % token_count)[iteration % rows];
        }
        const auto direct_end = std::chrono::steady_clock::now();
        const double baseline_ms = std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();
        const double direct_ms = std::chrono::duration<double, std::milli>(direct_end - direct_start).count();
        std::cout << "qnk_gemm dtype=" << static_cast<int>(dtype)
                  << " rows=" << rows
                  << " columns=" << columns
                  << " tokens=" << token_count
                  << " iterations=" << iterations
                  << " baseline_ms=" << baseline_ms
                  << " direct_ms=" << direct_ms
                  << " ms=" << direct_ms
                  << " per_iter_ms=" << (direct_ms / iterations)
                  << " speedup=" << (baseline_ms / direct_ms)
                  << " checksum_delta=" << std::abs(reference_checksum - direct_checksum)
                  << " checksum=" << direct_checksum << '\n';
    }
}

void benchmark_vulkan_qnk()
{
    constexpr uint32_t columns = 4096;
    constexpr size_t rows = 256;
    constexpr size_t token_count = 64;
    constexpr uint32_t iterations = 8;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    if (get_gpu_count() == 0)
    {
        std::cout << "vulkan_qnk unavailable\n";
        return;
    }

    ActivationBuffer input(token_count, columns);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((column * 7 + token * 13) % 29) - 14) * 0.03125f;
    }

    std::cout << std::fixed << std::setprecision(3);
    for (const DType dtype : dtypes)
    {
        const size_t raw_bytes = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
        const size_t row_bytes = raw_bytes / rows;
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        std::vector<uint8_t> raw(raw_bytes);
        for (size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<uint8_t>((index * 37u + static_cast<uint32_t>(dtype) * 11u + 5u) & 0xffu);
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + row * row_bytes + static_cast<size_t>(block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 8, uint8_t{1});
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                }
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = std::move(raw);
        ActivationBuffer cpu_output;
        forward_linear(matrix, input, cpu_output, 0);
        auto vulkan = QnkLinear_vulkan::create(matrix,
                                               nullptr,
                                               automatic_vulkan_device_index,
                                               vulkan_runtime,
                                               g_test_optimization_flags | OptimizationVulkanQnK);
        if (!vulkan)
        {
            std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype) << " create=failed\n";
            continue;
        }
        ActivationBuffer gpu_output;
        for (uint32_t iteration = 0; iteration < 2; ++iteration)
        {
            forward_linear(matrix, input, cpu_output, 0);
            if (!vulkan->forward(input, gpu_output))
            {
                std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype) << " forward=failed\n";
                gpu_output.clear();
                break;
            }
        }
        if (gpu_output.rows() == 0)
            continue;

        const auto cpu_start = std::chrono::steady_clock::now();
        float cpu_checksum = 0.0f;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            forward_linear(matrix, input, cpu_output, 0);
            cpu_checksum += cpu_output.row(iteration % token_count)[iteration % rows];
        }
        const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count();

        const auto gpu_start = std::chrono::steady_clock::now();
        float gpu_checksum = 0.0f;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            if (!vulkan->forward(input, gpu_output))
            {
                gpu_output.clear();
                break;
            }
            gpu_checksum += gpu_output.row(iteration % token_count)[iteration % rows];
        }
        const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_start).count();
        if (gpu_output.rows() == 0)
        {
            std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype) << " forward=failed\n";
            continue;
        }
        float max_error = 0.0f;
        for (size_t token = 0; token < token_count; ++token)
        {
            for (size_t row = 0; row < rows; ++row)
                max_error = std::max(max_error, std::fabs(gpu_output.row(token)[row] - cpu_output.row(token)[row]));
        }
        std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype)
                  << " rows=" << rows
                  << " columns=" << columns
                  << " tokens=" << token_count
                  << " iterations=" << iterations
                  << " cpu_ms=" << cpu_ms
                  << " gpu_ms=" << gpu_ms
                  << " speedup=" << (cpu_ms / gpu_ms)
                  << " max_error=" << max_error
                  << " checksum_delta=" << std::fabs(cpu_checksum - gpu_checksum)
                  << '\n';
    }
}

void benchmark_vulkan_qnk_expert()
{
    constexpr uint32_t hidden_columns = 512;
    constexpr uint32_t intermediate_columns = 256;
    constexpr size_t token_count = 64;
    constexpr uint32_t iterations = 8;
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    if (get_gpu_count() == 0)
    {
        std::cout << "vulkan_qnk_expert unavailable\n";
        return;
    }

    const auto make_q4k = [](size_t rows, uint32_t columns, uint32_t seed) {
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(DType::Q4K, rows, columns)));
        const size_t block_bytes = qnk_block_bytes(DType::Q4K);
        const uint32_t block_count = columns / qnk_block_elements;
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                encoded[0] = 0x00;
                encoded[1] = 0x3c;
                encoded[2] = 0x00;
                encoded[3] = 0x3c;
                std::fill_n(encoded + 4, 8, uint8_t{1});
                for (uint32_t index = 0; index < 64; ++index)
                    encoded[16 + index] = static_cast<uint8_t>(0x11u + ((seed + row + block + index) & 0x0fu));
            }
        }
        return raw;
    };

    TensorData gate_up;
    gate_up.dtype = DType::Q4K;
    gate_up.shape = {intermediate_columns * 2, hidden_columns};
    gate_up.quantized_data = make_q4k(gate_up.shape[0], hidden_columns, 7);
    TensorData down;
    down.dtype = DType::Q4K;
    down.shape = {hidden_columns, intermediate_columns};
    down.quantized_data = make_q4k(down.shape[0], intermediate_columns, 19);

    ActivationBuffer input(token_count, hidden_columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * 17 + column * 5) % 31) - 15) * 0.015625f;
    }

    auto vulkan = QnkExpert_vulkan::create(gate_up,
                                           nullptr,
                                           down,
                                           nullptr,
                                           0.0f,
                                           automatic_vulkan_device_index,
                                           ExpertActivation::GptOssSwiGlu,
                                           vulkan_runtime,
                                           g_test_optimization_flags | OptimizationVulkanQnK);
    if (!vulkan)
    {
        std::cout << "vulkan_qnk_expert create=failed\n";
        return;
    }

    const auto cpu_forward = [&]() {
        const ActivationBuffer gate_up_output = forward_linear(gate_up, input, 0);
        ActivationBuffer activated(token_count, intermediate_columns);
        for (size_t row = 0; row < token_count; ++row)
        {
            for (uint32_t column = 0; column < intermediate_columns; ++column)
            {
                const float gate = gate_up_output.row(row)[column * 2];
                const float up = gate_up_output.row(row)[column * 2 + 1];
                const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                activated.row(row)[column] = silu * (up + 1.0f);
            }
        }
        return forward_linear(down, activated, 0);
    };

    ActivationBuffer cpu_output = cpu_forward();
    ActivationBuffer gpu_output;
    for (uint32_t iteration = 0; iteration < 2; ++iteration)
    {
        cpu_output = cpu_forward();
        check(static_cast<bool>(vulkan->forward(input, gpu_output)));
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    float cpu_checksum = 0.0f;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        cpu_output = cpu_forward();
        cpu_checksum += cpu_output.row(iteration % token_count)[iteration % hidden_columns];
    }
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count();

    const auto gpu_start = std::chrono::steady_clock::now();
    float gpu_checksum = 0.0f;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        check(static_cast<bool>(vulkan->forward(input, gpu_output)));
        gpu_checksum += gpu_output.row(iteration % token_count)[iteration % hidden_columns];
    }
    const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_start).count();
    float max_error = 0.0f;
    for (size_t row = 0; row < token_count; ++row)
    {
        for (uint32_t column = 0; column < hidden_columns; ++column)
            max_error = std::max(max_error, std::fabs(gpu_output.row(row)[column] - cpu_output.row(row)[column]));
    }
    std::cout << std::fixed << std::setprecision(3)
              << "vulkan_qnk_expert rows=" << hidden_columns
              << " intermediate=" << intermediate_columns
              << " tokens=" << token_count
              << " iterations=" << iterations
              << " cpu_ms=" << cpu_ms
              << " gpu_ms=" << gpu_ms
              << " speedup=" << (cpu_ms / gpu_ms)
              << " max_error=" << max_error
              << " checksum_delta=" << std::fabs(cpu_checksum - gpu_checksum)
              << '\n';
}

void test_cpu_resource_coordination()
{
    const CpuThreadBudget budget = resolve_cpu_thread_budget(1);
#if defined(__APPLE__)
    uint32_t physical_cores = 0;
    size_t size = sizeof(physical_cores);
    check(sysctlbyname("hw.physicalcpu", &physical_cores, &size, nullptr, 0) == 0);
    check(physical_cores != 0);
    check(get_physical_cpu_count() == physical_cores);
    check(budget.max_threads == physical_cores);
#endif
    check(budget.num_threads >= 1);
    check(budget.num_threads <= budget.max_threads);
    check(budget.num_io_threads == 1);
    check(choose_cpu_team_size(0, 8, 4, budget.num_threads) == 1);
    check(choose_cpu_team_size(1024, 8, 4, budget.num_threads) <= budget.num_threads);
    check(choose_cpu_team_size(std::numeric_limits<uint64_t>::max(), 2, 4, 8) == 8);

#if defined(_OPENMP)
    const uint32_t original_thread_limit = cpu_openmp_thread_limit();
    {
        CpuOpenMpThreadLimitScope outer;
        outer.set(3);
        check(cpu_openmp_thread_limit() == 3);
        {
            CpuOpenMpThreadLimitScope inner;
            inner.set(2);
            check(cpu_openmp_thread_limit() == 2);
        }
        check(cpu_openmp_thread_limit() == 3);
    }
    check(cpu_openmp_thread_limit() == original_thread_limit);

    try
    {
        CpuOpenMpThreadLimitScope scope;
        scope.set(2);
        check(cpu_openmp_thread_limit() == 2);
        throw 1;
    }
    catch (int)
    {
    }
    check(cpu_openmp_thread_limit() == original_thread_limit);
#endif

#if defined(NCNN_MOE_WITH_NCNN) && NCNN_MOE_WITH_NCNN \
    && defined(_OPENMP)                               \
    && (defined(__clang__) || defined(_OPENMP_LLVM_RUNTIME))
    const bool explicit_wait_policy = std::getenv("KMP_BLOCKTIME")
                                      || std::getenv("OMP_WAIT_POLICY")
                                      || std::getenv("KMP_LIBRARY");
    const int original_blocktime = ncnn::get_kmp_blocktime();
    ncnn::set_kmp_blocktime(13);
    // ncnn may be built without LLVM OpenMP even when this test uses it.
    if (ncnn::get_kmp_blocktime() == 13)
    {
        const int scoped_blocktime = explicit_wait_policy || cpu_openmp_thread_limit() <= 1 ? 13 : 1;
        {
            CpuOpenMpThreadLimitScope outer;
            check(ncnn::get_kmp_blocktime() == scoped_blocktime);
            {
                CpuOpenMpThreadLimitScope inner;
                check(ncnn::get_kmp_blocktime() == scoped_blocktime);
            }
            check(ncnn::get_kmp_blocktime() == scoped_blocktime);
        }
        check(ncnn::get_kmp_blocktime() == 13);
        try
        {
            CpuOpenMpThreadLimitScope scope;
            check(ncnn::get_kmp_blocktime() == scoped_blocktime);
            throw 1;
        }
        catch (int)
        {
        }
        check(ncnn::get_kmp_blocktime() == 13);
        {
            CpuOpenMpThreadLimitScope outer;
            outer.set(1);
            ncnn::set_kmp_blocktime(13);
            {
                const CpuOpenMpThreadLimitScope inner;
                check(ncnn::get_kmp_blocktime() == 13);
            }
            check(ncnn::get_kmp_blocktime() == 13);
        }
        check(ncnn::get_kmp_blocktime() == 13);
    }
    ncnn::set_kmp_blocktime(original_blocktime);
    check(ncnn::get_kmp_blocktime() == original_blocktime);
#endif

    // Test permit accounting independently of the host topology.
    CpuThreadBudgetController controller({2, 1, 4});
    check(controller.try_acquire_compute(0).empty());
    auto base = controller.acquire_compute(2, false);
    check(base.size() == 2);
    check(controller.try_acquire_compute(1, false).empty());
    auto extra = controller.try_acquire_compute(8);
    check(extra.size() == 2);
    check(controller.available() == 0);
    check(controller.try_acquire_compute(1).empty());

    // Extra permits must not prevent reuse of released base permits.
    base = {};
    check(controller.available(false) == 2);
    auto next = controller.try_acquire_compute(2, false);
    check(next.size() == 2);
    auto moved = std::move(extra);
    check(extra.empty());
    check(moved.size() == 2);
    check(controller.available() == 0);
    next = std::move(moved);
    check(moved.empty());
    check(controller.available(false) == 2);
    next = {};
    check(controller.available() == 4);
    check(controller.available(false) == 2);
    {
        auto scoped = controller.try_acquire_compute(4);
        check(scoped.size() == 4);
    }
    check(controller.available() == 4);

    const CpuThreadBudget scheduler_budget = resolve_cpu_thread_budget();
    constexpr std::array<uint32_t, 5> scheduler_thread_counts = {0, 1, 2, 4, std::numeric_limits<uint32_t>::max()};
    TestRuntime runtime;
    for (uint32_t requested_threads : scheduler_thread_counts)
    {
        SchedulerOptions options;
        options.num_threads = requested_threads;
        auto scheduler = runtime.create_scheduler(options);
        check(static_cast<bool>(scheduler));
        const SchedulerStatistics statistics = scheduler.value()->statistics();
        const uint32_t expected_threads = std::min(requested_threads == 0 ? 4u : requested_threads, scheduler_budget.num_threads);
        check(statistics.num_threads == expected_threads);
    }
}

} // namespace moe
} // namespace ncnn
