#include "kernels/int4.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "graph/compiledoperator.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void int4_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("INT4 test failed: ") + message);
}

static void int4_near(float actual, float expected)
{
    int4_check(std::isfinite(actual) && std::abs(actual - expected) <= 2e-5f * std::max(1.0f, std::abs(expected)), "direct scalar reference mismatch");
}

static uint8_t int4_test_code(uint32_t row, uint32_t column)
{
    return static_cast<uint8_t>((row * 11 + column * 7 + 3) % 16);
}

static TensorData int4_test_matrix(uint32_t rows, uint32_t columns, uint32_t group_size, bool indexed)
{
    TensorData matrix;
    matrix.dtype = DType::Int4;
    matrix.shape = {rows, columns};
    matrix.quantization_group_size = group_size;
    const size_t row_bytes = (static_cast<size_t>(columns) + 1) / 2;
    const uint32_t groups = (columns + group_size - 1) / group_size;
    // An odd final column must ignore the high nibble rather than read the next row.
    matrix.quantized_data.resize(rows * row_bytes, 0xf0);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            uint8_t& packed = matrix.quantized_data[row * row_bytes + column / 2];
            const uint32_t shift = (column & 1u) * 4;
            packed = static_cast<uint8_t>((packed & ~(15u << shift)) | (int4_test_code(row, column) << shift));
        }
        for (uint32_t group = 0; group < groups; ++group)
        {
            matrix.quantization_scales.push_back(static_cast<float>((row * 3 + group * 5) % 9 + 1) / 32.0f);
            matrix.quantization_zero_points.push_back(static_cast<uint8_t>((row * 7 + group * 11 + 16) % 17));
        }
    }
    if (indexed)
    {
        for (uint32_t column = 0; column < columns; ++column)
            matrix.quantization_group_indices.push_back((column * 7 + column / 3) % groups);
    }
    return matrix;
}

static void check_int4_reference(const TensorData& matrix, size_t tokens)
{
    int4_check(static_cast<bool>(validate_int4_tensor(matrix)), "reference weight validation failed");
    ActivationBuffer input(tokens, matrix.shape[1]);
    for (size_t token = 0; token < tokens; ++token)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((token * 13 + column * 3) % 23) - 11) / 16.0f;
    }
    const size_t packed_bytes = matrix.int4_values().size();
    const uint8_t* packed_pointer = matrix.int4_values().data();
    ActivationBuffer output;
    int4_check(forward_linear_int4(matrix, input, output), "valid native forward rejected");
    int4_check(output.rows() == tokens && output.columns() == matrix.shape[0], "output shape mismatch");
    const uint32_t groups = (matrix.shape[1] + matrix.quantization_group_size - 1) / matrix.quantization_group_size;
    for (size_t token = 0; token < tokens; ++token)
    {
        for (uint32_t row = 0; row < matrix.shape[0]; ++row)
        {
            float expected = 0.0f;
            for (uint32_t column = 0; column < matrix.shape[1]; ++column)
            {
                const uint32_t group = matrix.quantization_group_indices.empty()
                                           ? column / matrix.quantization_group_size
                                           : matrix.quantization_group_indices[column];
                const size_t metadata = static_cast<size_t>(row) * groups + group;
                const float weight = (static_cast<int>(int4_test_code(row, column)) - matrix.quantization_zero_points[metadata]) * matrix.quantization_scales[metadata];
                expected += weight * input.row(token)[column];
            }
            int4_near(output.row(token)[row], expected);
        }
    }
    ActivationBuffer dispatched;
    forward_linear(matrix, input, dispatched, 0);
    for (size_t index = 0; index < output.values().size(); ++index)
        int4_near(dispatched.values()[index], output.values()[index]);
    int4_check(matrix.int4_values().size() == packed_bytes && matrix.int4_values().data() == packed_pointer, "forward replaced packed storage");
    int4_check(matrix.float32_data.empty() && matrix.bfloat16_data.empty(), "forward materialized dense weight storage");
    int4_check(int4_storage_bytes(matrix) == packed_bytes + matrix.quantization_scales.size() * sizeof(float) + matrix.quantization_zero_points.size() + matrix.quantization_group_indices.size() * sizeof(uint32_t),
               "packed byte accounting mismatch");
}

static void test_int4_known_values()
{
    TensorData matrix;
    matrix.dtype = DType::Int4;
    matrix.shape = {2, 5};
    matrix.quantization_group_size = 2;
    matrix.quantized_data = {0x40, 0x8f, 0xf2, 0xf1, 0x03, 0xbe};
    matrix.quantization_scales = {0.5f, 2.0f, 0.25f, 1.0f, 0.5f, 2.0f};
    matrix.quantization_zero_points = {0, 8, 16, 16, 3, 8};
    int4_check(static_cast<bool>(validate_int4_tensor(matrix)), "known-value weight validation failed");
    ActivationBuffer input(1, 5);
    const std::array<float, 5> values = {1.0f, -2.0f, 0.5f, 3.0f, -1.0f};
    std::copy(values.begin(), values.end(), input.row(0));
    ActivationBuffer output;
    int4_check(forward_linear_int4(matrix, input, output), "known-value forward rejected");
    int4_near(output.row(0)[0], 6.5f);
    int4_near(output.row(0)[1], -29.5f);
}

static void test_int4_invalid_tensors()
{
    const TensorData valid = int4_test_matrix(3, 5, 2, true);
    std::vector<TensorData> invalid(25, valid);
    invalid[0].dtype = DType::Int8;
    invalid[1].shape = {15};
    invalid[2].shape = {0, 5};
    invalid[3].shape = {3, 0};
    invalid[4].quantization_group_size = 0;
    invalid[5].quantized_data.pop_back();
    invalid[6].quantized_data.push_back(0);
    invalid[7].quantization_scales.pop_back();
    invalid[8].quantization_zero_points.pop_back();
    invalid[9].quantization_zero_points[0] = 17;
    invalid[10].quantization_scales[0] = std::numeric_limits<float>::quiet_NaN();
    invalid[11].quantization_scales[0] = std::numeric_limits<float>::infinity();
    invalid[12].quantization_scales[0] = -1.0f;
    invalid[13].quantization_group_indices.pop_back();
    invalid[14].quantization_group_indices[0] = 3;
    invalid[15].quantization_scales[0] = 0.0f;
    invalid[16].float32_data = {1.0f};
    invalid[17].bfloat16_data = {0x3f80};
    invalid[18].int8_data = {1};
    invalid[19].int64_data = {1};
    invalid[20].mxfp4_blocks = {0};
    invalid[21].mxfp4_scales = {0};
    invalid[22].mxfp4_file_storage = std::make_shared<MxFp4FileStorage>();
    auto mapped = std::make_shared<std::vector<uint8_t>>(valid.quantized_data);
    invalid[23].mapped_data = std::shared_ptr<const uint8_t>(mapped, mapped->data());
    invalid[23].mapped_size = mapped->size();
    invalid[24].mapped_size = 1;
    ActivationBuffer input(1, 5);
    ActivationBuffer output(1, 1);
    output.row(0)[0] = 123.0f;
    for (size_t index = 0; index < invalid.size(); ++index)
    {
        const TensorData& matrix = invalid[index];
        int4_check(!validate_int4_tensor(matrix), "malformed tensor accepted");
        // Numeric metadata is validated once by the loader/compiler. The
        // execution contract checks layout without rescanning those values.
        const bool invalid_values = index == 9 || index == 10 || index == 11
                                    || index == 12 || index == 14 || index == 15;
        if (!invalid_values)
        {
            int4_check(!forward_linear_int4(matrix, input, output), "malformed tensor layout executed");
            int4_check(output.columns() == 1 && output.row(0)[0] == 123.0f, "failed forward changed output");
        }
    }
    ActivationBuffer wrong_width(1, 4);
    int4_check(!forward_linear_int4(valid, wrong_width, output), "incompatible activation width accepted");
    ActivationBuffer wrong_dtype(1, 5, DType::BFloat16);
    int4_check(!forward_linear_int4(valid, wrong_dtype, output), "non-float activation accepted");
    int4_check(!forward_linear_int4(valid, input, input), "aliased input/output accepted");
    ActivationBuffer wrong_output(1, 3, DType::BFloat16);
    int4_check(!forward_linear_int4(valid, input, wrong_output), "non-float output accepted");
}

static float int4_test_weight(const TensorData& matrix, uint32_t row, uint32_t column)
{
    const uint32_t groups = (matrix.shape[1] + matrix.quantization_group_size - 1) / matrix.quantization_group_size;
    const uint32_t group = matrix.quantization_group_indices.empty() ? column / matrix.quantization_group_size : matrix.quantization_group_indices[column];
    const size_t metadata = static_cast<size_t>(row) * groups + group;
    return (static_cast<int>(int4_test_code(row, column)) - matrix.quantization_zero_points[metadata]) * matrix.quantization_scales[metadata];
}

static float q8k_test_weight(uint32_t row, uint32_t column)
{
    const float scale = static_cast<float>((row + 1) * (column / qnk_block_elements + 1)) / 32.0f;
    const int code = static_cast<int>((row * 11 + column * 7 + 3) % 31) - 15;
    return scale * static_cast<float>(code);
}

static TensorData qnk_test_matrix(DType dtype, uint32_t rows, uint32_t columns)
{
    TensorData matrix;
    matrix.dtype = dtype;
    matrix.shape = {rows, columns};
    matrix.quantized_data.resize(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)));
    const uint32_t blocks = columns / qnk_block_elements;
    const size_t block_bytes = qnk_block_bytes(dtype);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t block = 0; block < blocks; ++block)
        {
            uint8_t* encoded = matrix.quantized_data.data() + (static_cast<size_t>(row) * blocks + block) * block_bytes;
            std::fill_n(encoded, block_bytes, static_cast<uint8_t>(0x11 + row + block));
            if (dtype == DType::Q8K)
            {
                const float scale = static_cast<float>((row + 1) * (block + 1)) / 32.0f;
                std::memcpy(encoded, &scale, sizeof(scale));
                for (uint32_t lane = 0; lane < qnk_block_elements; ++lane)
                {
                    const int code = static_cast<int>((row * 11 + (block * qnk_block_elements + lane) * 7 + 3) % 31) - 15;
                    encoded[4 + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
                }
            }
            else
            {
                const uint16_t scale = 0x3800;
                const uint16_t minimum = 0x3400;
                const size_t offset = dtype == DType::Q2K ? 80 : dtype == DType::Q3K ? 108
                                                             : dtype == DType::Q6K   ? 208
                                                                                     : 0;
                std::memcpy(encoded + offset, &scale, sizeof(scale));
                if (dtype == DType::Q2K || dtype == DType::Q4K || dtype == DType::Q5K)
                    std::memcpy(encoded + offset + sizeof(scale), &minimum, sizeof(minimum));
            }
        }
    }
    return matrix;
}

static void test_qnk_paired_linear()
{
#if defined(__aarch64__) || defined(_M_ARM64)
    for (DType dtype : {DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K})
    {
        for (uint32_t columns : {256u, 512u})
        {
            for (size_t tokens : {size_t{1}, size_t{5}, size_t{9}})
            {
                const bool has_parallel_decode = dtype == DType::Q4K && columns == 512 && tokens == 1;
                for (size_t row_case = 0; row_case < (has_parallel_decode ? 2u : 1u); ++row_case)
                {
                    const bool parallel_decode = row_case != 0;
                    for (bool packed : {false, true})
                    {
                        TensorData first = qnk_test_matrix(dtype, parallel_decode ? 257 : 9, columns);
                        TensorData second = qnk_test_matrix(dtype, parallel_decode ? 129 : 17, columns);
                        ActivationBuffer input(tokens, columns);
                        for (size_t row = 0; row < input.rows(); ++row)
                        {
                            for (uint32_t column = 0; column < columns; ++column)
                            {
                                input.row(row)[column] = static_cast<float>(static_cast<int>((row * 7 + column * 3) % 29) - 14) / 32.0f;
                            }
                        }

                        const uint64_t flags = packed ? OptimizationCpuPackedWeights : 0;
                        CompiledOperator first_owner;
                        CompiledOperator second_owner;
                        ActivationBuffer first_scalar;
                        ActivationBuffer second_scalar;
                        forward_linear(first, input, first_scalar, flags, &first_owner);
                        forward_linear(second, input, second_scalar, flags, &second_owner);
                        const auto first_pack = first_owner.qnk_packed;
                        const auto second_pack = second_owner.qnk_packed;
                        int4_check(!packed || (first_pack && second_pack), "packed scalar Qn_K owner did not retain its sidecar");

                        ActivationBuffer first_paired;
                        ActivationBuffer second_paired;
                        const auto forward_pair = [&] {
                            return forward_linear({&first, &second}, input,
                                                  {&first_paired, &second_paired}, flags,
                                                  {&first_owner, &second_owner});
                        };
                        int4_check(forward_pair(), "paired Qn_K forward rejected valid matrices");
                        int4_check(first_paired.rows() == input.rows() && first_paired.columns() == first.shape[0]
                                       && second_paired.rows() == input.rows() && second_paired.columns() == second.shape[0],
                                   "paired Qn_K output shape mismatch");
                        for (size_t index = 0; index < first_scalar.values().size(); ++index)
                            int4_near(first_paired.values()[index], first_scalar.values()[index]);
                        for (size_t index = 0; index < second_scalar.values().size(); ++index)
                            int4_near(second_paired.values()[index], second_scalar.values()[index]);

                        int4_check(forward_pair(), "repeated paired Qn_K forward rejected valid matrices");
                        int4_check(first_owner.qnk_packed == first_pack && second_owner.qnk_packed == second_pack,
                                   "paired Qn_K forward did not reuse weight sidecars");
                        for (size_t index = 0; index < first_scalar.values().size(); ++index)
                            int4_near(first_paired.values()[index], first_scalar.values()[index]);
                        for (size_t index = 0; index < second_scalar.values().size(); ++index)
                            int4_near(second_paired.values()[index], second_scalar.values()[index]);
                    }
                }
            }
        }
    }
#else
    {
        TensorData first = qnk_test_matrix(DType::Q8K, 9, 256);
        TensorData second = qnk_test_matrix(DType::Q8K, 17, 256);
        ActivationBuffer input(1, 256);
        ActivationBuffer first_output(1, 1);
        ActivationBuffer second_output(1, 2);
        first_output.row(0)[0] = 17.0f;
        second_output.row(0)[0] = 23.0f;
        second_output.row(0)[1] = 29.0f;
        int4_check(!forward_linear({&first, &second}, input, {&first_output, &second_output}, 0),
                   "paired Qn_K forward unexpectedly ran outside ARM");
        int4_check(first_output.rows() == 1 && first_output.columns() == 1 && first_output.row(0)[0] == 17.0f
                       && second_output.rows() == 1 && second_output.columns() == 2
                       && second_output.row(0)[0] == 23.0f && second_output.row(0)[1] == 29.0f,
                   "unsupported paired Qn_K forward changed outputs");
    }
#endif

    TensorData first = qnk_test_matrix(DType::Q8K, 9, 256);
    TensorData second = qnk_test_matrix(DType::Q8K, 17, 256);
    ActivationBuffer input(1, 256);
    ActivationBuffer first_output;
    ActivationBuffer second_output;
    const auto reset_outputs = [&] {
        first_output.reset(1, 2, false);
        second_output.reset(1, 2, false);
        first_output.row(0)[0] = 17.0f;
        first_output.row(0)[1] = 19.0f;
        second_output.row(0)[0] = 23.0f;
        second_output.row(0)[1] = 29.0f;
    };
    const auto outputs_unchanged = [&] {
        int4_check(first_output.rows() == 1 && first_output.columns() == 2
                       && first_output.row(0)[0] == 17.0f && first_output.row(0)[1] == 19.0f
                       && second_output.rows() == 1 && second_output.columns() == 2
                       && second_output.row(0)[0] == 23.0f && second_output.row(0)[1] == 29.0f,
                   "rejected paired Qn_K forward changed outputs");
    };
    const auto check_rejected = [&](const TensorData& gate, const TensorData& up,
                                    const ActivationBuffer& activation, const char* message) {
        reset_outputs();
        int4_check(!forward_linear({&gate, &up}, activation,
                                   {&first_output, &second_output}, 0),
                   message);
        outputs_unchanged();
    };
    ActivationBuffer wrong_width(1, 128);
    check_rejected(first, second, wrong_width, "paired Qn_K forward accepted a mismatched input shape");
    TensorData wrong_dtype = second;
    wrong_dtype.dtype = DType::Q4K;
    check_rejected(first, wrong_dtype, input, "paired Qn_K forward accepted mixed dtypes");
    TensorData wrong_raw = second;
    wrong_raw.quantized_data.pop_back();
    check_rejected(first, wrong_raw, input, "paired Qn_K forward accepted truncated raw bytes");
    TensorData wrong_columns = second;
    wrong_columns.input_column_indices.push_back(0);
    check_rejected(first, wrong_columns, input, "paired Qn_K forward accepted a column map");

    reset_outputs();
    int4_check(!forward_linear({&first, &second}, input, {&first_output, &first_output}, 0),
               "paired Qn_K forward accepted aliased outputs");
    outputs_unchanged();
    ActivationBuffer alias_input(1, 256);
    std::fill_n(alias_input.row(0), alias_input.columns(), 31.0f);
    reset_outputs();
    int4_check(!forward_linear({&first, &second}, alias_input, {&alias_input, &second_output}, 0),
               "paired Qn_K forward accepted input/output aliasing");
    int4_check(alias_input.rows() == 1 && alias_input.columns() == 256
                   && std::all_of(alias_input.values().begin(), alias_input.values().end(), [](float value) { return value == 31.0f; }),
               "rejected paired Qn_K alias changed its input");
    int4_check(second_output.rows() == 1 && second_output.columns() == 2
                   && second_output.row(0)[0] == 23.0f && second_output.row(0)[1] == 29.0f,
               "rejected paired Qn_K alias changed the other output");
}

static void test_quantized_column_map()
{
    for (bool int4 : {false, true})
    {
        const uint32_t columns = int4 ? 7 : 512;
        TensorData matrix = int4 ? int4_test_matrix(3, columns, 3, true) : qnk_test_matrix(DType::Q8K, 3, columns);
        for (uint32_t column = 0; column < columns; ++column)
            matrix.input_column_indices.push_back((column * (int4 ? 3 : 5) + 1) % columns);
        int4_check(static_cast<bool>(validate_input_column_indices(matrix)), "valid input-column permutation rejected");
        if (int4)
            int4_check(static_cast<bool>(validate_int4_tensor(matrix)), "permuted INT4 validation failed");
        ActivationBuffer input(5, columns);
        for (size_t row = 0; row < input.rows(); ++row)
            for (uint32_t column = 0; column < columns; ++column)
                input.row(row)[column] = static_cast<float>(static_cast<int>((row * 3 + column * 5) % 17) - 8) / 16.0f;
        ActivationBuffer local_output;
        forward_linear(matrix, input, local_output, 0);
        ActivationBuffer scratch(1, 1);
        ActivationBuffer scratch_output;
        forward_linear(matrix, input, scratch_output, 0, nullptr, ExecutionBackend::Cpu, &scratch);
        int4_check(scratch.rows() == input.rows() && scratch.columns() == columns, "column gather did not reuse float scratch");
        const std::byte* scratch_storage = scratch.bytes().data();
        forward_linear(matrix, input, scratch_output, 0, nullptr, ExecutionBackend::Cpu, &scratch);
        int4_check(scratch.bytes().data() == scratch_storage, "column gather reallocated reusable scratch");
        ActivationBuffer packed_output;
        forward_linear(matrix, input, packed_output, OptimizationCpuPackedWeights);
        ActivationBuffer wrong_scratch(1, 1, DType::BFloat16);
        ActivationBuffer fallback_output;
        forward_linear(matrix, input, fallback_output, 0, nullptr, ExecutionBackend::Cpu, &wrong_scratch);
        int4_check(wrong_scratch.dtype() == DType::BFloat16 && wrong_scratch.rows() == 1 && wrong_scratch.columns() == 1, "column gather modified incompatible scratch");
        TensorData bias;
        bias.shape = {3};
        bias.float32_data = {0.125f, -0.25f, 0.5f};
        ActivationBuffer bias_output;
        forward_linear(matrix, bias, input, bias_output, 0);
        for (size_t token = 0; token < input.rows(); ++token)
        {
            for (uint32_t row = 0; row < matrix.shape[0]; ++row)
            {
                float expected = 0.0f;
                for (uint32_t column = 0; column < columns; ++column)
                {
                    const float weight = int4 ? int4_test_weight(matrix, row, column) : q8k_test_weight(row, column);
                    const uint32_t logical = matrix.input_column_indices[column];
                    expected += weight * input.row(token)[logical];
                    int4_near(scratch.row(token)[column], input.row(token)[logical]);
                }
                int4_near(local_output.row(token)[row], expected);
                int4_near(scratch_output.row(token)[row], expected);
                int4_near(packed_output.row(token)[row], expected);
                int4_near(fallback_output.row(token)[row], expected);
                int4_near(bias_output.row(token)[row], expected + bias.float32_data[row]);
            }
        }
        TensorData invalid = matrix;
        invalid.input_column_indices.pop_back();
        int4_check(!validate_input_column_indices(invalid), "short column map accepted");
        invalid = matrix;
        invalid.input_column_indices[0] = columns;
        int4_check(!validate_input_column_indices(invalid), "out-of-range column map accepted");
        invalid = matrix;
        invalid.input_column_indices[0] = invalid.input_column_indices[1];
        int4_check(!validate_input_column_indices(invalid), "duplicate column map accepted");
        invalid = matrix;
        invalid.dtype = DType::BFloat16;
        int4_check(!validate_input_column_indices(invalid), "column map accepted on dense weights");
    }
}

static void test_input_column_map_boundaries()
{
    for (uint32_t columns : {63u, 64u, 65u, 8192u, 8193u})
    {
        TensorData matrix;
        matrix.dtype = DType::Q8K;
        matrix.shape = {1, columns};
        matrix.input_column_indices.resize(columns);
        for (uint32_t column = 0; column < columns; ++column)
            matrix.input_column_indices[column] = column;
        int4_check(static_cast<bool>(validate_input_column_indices(matrix)), "valid column map at boundary rejected");

        TensorData invalid = matrix;
        invalid.input_column_indices.back() = invalid.input_column_indices.front();
        int4_check(!validate_input_column_indices(invalid), "duplicate final column accepted");

        invalid = matrix;
        invalid.input_column_indices.back() = columns;
        int4_check(!validate_input_column_indices(invalid), "out-of-range final column accepted");

        invalid = matrix;
        invalid.input_column_indices.pop_back();
        int4_check(!validate_input_column_indices(invalid), "short column map accepted");

        invalid = matrix;
        invalid.shape = {columns};
        int4_check(!validate_input_column_indices(invalid), "non-matrix column map accepted");

        invalid = matrix;
        invalid.dtype = DType::Float32;
        int4_check(!validate_input_column_indices(invalid), "dense column map accepted");

        if (columns > 64)
        {
            invalid = matrix;
            invalid.input_column_indices[64] = invalid.input_column_indices[63];
            int4_check(!validate_input_column_indices(invalid), "duplicate across columns 63 and 64 accepted");
        }
    }

    TensorData empty_map;
    empty_map.dtype = DType::Float32;
    empty_map.shape = {1};
    int4_check(static_cast<bool>(validate_input_column_indices(empty_map)), "empty column map did not bypass validation");
}

static void test_quantized_embeddings()
{
    const std::array<int32_t, 4> tokens = {3, 1, 3, 0};
    for (DType dtype : {DType::Int4, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K})
    {
        for (bool permuted : {false, true})
        {
            const uint32_t columns = dtype == DType::Int4 ? 7 : 512;
            TensorData matrix = dtype == DType::Int4 ? int4_test_matrix(4, columns, 3, true) : qnk_test_matrix(dtype, 4, columns);
            if (permuted)
            {
                for (uint32_t column = 0; column < columns; ++column)
                    matrix.input_column_indices.push_back((column * (dtype == DType::Int4 ? 3 : 5) + 1) % columns);
            }
            int4_check(static_cast<bool>(validate_input_column_indices(matrix)), "embedding column map rejected");
            if (dtype == DType::Int4)
                int4_check(static_cast<bool>(validate_int4_tensor(matrix)), "INT4 embedding validation failed");
            const uint8_t* storage = matrix.quantized_data.data();
            ActivationBuffer output;
            forward_embedding(matrix, tokens, output);
            int4_check(output.rows() == tokens.size() && output.columns() == columns, "quantized embedding shape mismatch");
            std::array<float, qnk_block_elements> decoded;
            for (size_t token = 0; token < tokens.size(); ++token)
            {
                for (uint32_t column = 0; column < columns; ++column)
                {
                    if (dtype != DType::Int4 && column % qnk_block_elements == 0)
                    {
                        const size_t block = static_cast<size_t>(tokens[token]) * (columns / qnk_block_elements) + column / qnk_block_elements;
                        qnk_dequantize_block(dtype, storage + block * qnk_block_bytes(dtype), decoded.data());
                    }
                    const float expected = dtype == DType::Int4 ? int4_test_weight(matrix, static_cast<uint32_t>(tokens[token]), column) : decoded[column % qnk_block_elements];
                    const uint32_t destination = permuted ? matrix.input_column_indices[column] : column;
                    int4_near(output.row(token)[destination], expected);
                }
            }
            int4_check(matrix.quantized_data.data() == storage && matrix.float32_data.empty() && matrix.bfloat16_data.empty(), "embedding expanded resident weights");
            auto mapped = std::make_shared<std::vector<uint8_t>>(matrix.quantized_data);
            matrix.mapped_data = std::shared_ptr<const uint8_t>(mapped, mapped->data());
            matrix.mapped_size = mapped->size();
            matrix.quantized_data.clear();
            ActivationBuffer mapped_output;
            forward_embedding(matrix, tokens, mapped_output);
            for (size_t index = 0; index < output.values().size(); ++index)
                int4_near(mapped_output.values()[index], output.values()[index]);
            forward_embedding(matrix, {}, mapped_output);
            int4_check(mapped_output.rows() == 0 && mapped_output.columns() == columns, "empty quantized embedding input failed");
            for (int32_t invalid : {-1, 4})
            {
                bool rejected = false;
                try
                {
                    forward_embedding(matrix, std::span<const int32_t>(&invalid, 1), mapped_output);
                }
                catch (const std::runtime_error&)
                {
                    rejected = true;
                }
                int4_check(rejected, "invalid embedding token ID accepted");
            }
        }
    }
}

void test_int4()
{
    test_int4_known_values();
    test_int4_invalid_tensors();
    test_qnk_paired_linear();
    test_quantized_column_map();
    test_input_column_map_boundaries();
    test_quantized_embeddings();
    const std::array<std::array<uint32_t, 3>, 6> shapes = {{{1, 1, 1}, {3, 5, 2}, {5, 39, 17}, {7, 64, 16}, {9, 97, 32}, {5, 7, 128}}};
    for (const auto& shape : shapes)
    {
        for (bool indexed : {false, true})
        {
            const TensorData matrix = int4_test_matrix(shape[0], shape[1], shape[2], indexed);
            int4_check(static_cast<bool>(validate_int4_tensor(matrix)), "generated matrix rejected");
            for (size_t tokens : {0, 1, 2, 4, 5, 9})
                check_int4_reference(matrix, tokens);
            TensorData mapped = matrix;
            auto storage = std::make_shared<std::vector<uint8_t>>(matrix.quantized_data);
            mapped.mapped_data = std::shared_ptr<const uint8_t>(storage, storage->data());
            mapped.mapped_size = storage->size();
            mapped.quantized_data.clear();
            check_int4_reference(mapped, 5);
        }
    }
    // Cross the normal linear OpenMP threshold with a partial final token tile.
    check_int4_reference(int4_test_matrix(257, 257, 32, false), 33);
    check_int4_reference(int4_test_matrix(257, 257, 32, true), 33);
}

} // namespace moe
} // namespace ncnn
