#include "backends/ncnn/gatedresidual_vulkan.h"
#include "backends/ncnn/qwen_vulkan_internal.h"
#include "graph/compiledoperator.h"
#include "kernels/attention.h"
#include "kernels/gatedresidual.h"
#include "kernels/ops.h"
#include "kernels/ple.h"
#include "kernels/qnk.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void qwen_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorHandle qwen_add(WeightStore& weights, TensorData tensor)
{
    auto result = weights.add("qwen_test_" + std::to_string(weights.size()), std::move(tensor));
    qwen_check(static_cast<bool>(result), "add Qwen fixture weight");
    return result.value();
}

static TensorData qwen_matrix(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData value;
    value.dtype = DType::BFloat16;
    value.shape = {rows, columns};
    value.bfloat16_data.resize(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 19 + column * 13 + seed * 17 + (row ^ column)) % 43) - 21;
            value.bfloat16_data[static_cast<size_t>(row) * columns + column] = float_to_bfloat16(static_cast<float>(sample) * 0.001953125f);
        }
    return value;
}

static TensorData qwen_norm(uint32_t columns)
{
    TensorData value;
    value.dtype = DType::BFloat16;
    value.shape = {columns};
    value.bfloat16_data.resize(columns);
    for (uint32_t column = 0; column < columns; ++column)
        value.bfloat16_data[column] = float_to_bfloat16(-0.1f + static_cast<float>((column * 7) % 17) * 0.0125f);
    return value;
}

static ActivationBuffer qwen_input(uint32_t rows, uint32_t columns, uint32_t position)
{
    ActivationBuffer value(rows, columns);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
            value.row(row)[column] = 0.7f * std::sin(static_cast<float>((row + position) * 19 + column * 7) * 0.091f)
                                     + 0.2f * std::cos(static_cast<float>(column * 3 + row * 5) * 0.113f);
    return value;
}

static void qwen_equal(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label,
                       float absolute = 0.002f, float relative = 0.004f)
{
    qwen_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + " shape");
    for (size_t index = 0; index < actual.values().size(); ++index)
        qwen_check(std::isfinite(actual.values()[index])
                       && std::abs(actual.values()[index] - expected.values()[index]) <= absolute + relative * std::abs(expected.values()[index]),
                   label + " element " + std::to_string(index) + " actual=" + std::to_string(actual.values()[index])
                       + " expected=" + std::to_string(expected.values()[index]));
}

#if NCNN_MOE_WITH_VULKAN
static bool qwen_upload(const ActivationBuffer& host, ncnn::VkMat& gpu, ncnn::VkMat& staging,
                        ncnn::VkCompute& cmd, const std::shared_ptr<VulkanContext>& context, const ncnn::Option& opt)
{
    return fill_staging_upload(host, staging, context->staging_allocator()) && record_mapped_upload(staging, gpu, cmd, opt);
}

static void qwen_gr_case(uint32_t device, uint32_t rows, bool head)
{
    WeightStore weights;
    MoeModelDescriptor descriptor;
    descriptor.hidden_size = 32;
    descriptor.hyper_connection_multiplier = 4;
    descriptor.norm_epsilon = 1e-5f;
    descriptor.norm_weight_offset = 1.0f;
    GatedResidualPlan plan;
    plan.norm_weight = qwen_add(weights, qwen_norm(128));
    plan.mix_down_weight = qwen_add(weights, qwen_matrix(16, 128, 1));
    plan.mix_up_weight = qwen_add(weights, qwen_matrix(128, 16, 2));
    if (!head)
        plan.inject_weight = qwen_add(weights, qwen_matrix(4, 128, 3));
    auto runtime = create_vulkan_runtime();
    auto op = GatedResidual_vulkan::create(weights, plan, descriptor, device, runtime, OptimizationDefaultFlags);
    qwen_check(static_cast<bool>(op), "create GatedResidual device operator");
    const auto input = qwen_input(rows, 128, 0);
    const auto branch = qwen_input(rows, 32, 11);
    CompiledOperatorTable cpu_operators;
    HyperConnectionMix mix;
    HyperConnectionScratch scratch;
    ActivationBuffer expected;
    ActivationBuffer expected_reduced;
    if (head)
        qwen_check(static_cast<bool>(forward_gated_residual_head(input, weights, cpu_operators, plan, descriptor,
                                                                 expected, scratch, OptimizationDefaultFlags)),
                   "CPU GatedResidual head");
    else
    {
        qwen_check(static_cast<bool>(forward_gated_residual_pre(input, weights, cpu_operators, plan, descriptor,
                                                                mix, scratch, OptimizationDefaultFlags)),
                   "CPU GatedResidual pre");
        expected_reduced = mix.reduced;
        qwen_check(static_cast<bool>(forward_gated_residual_post(branch, input, mix, 4, expected)), "CPU GatedResidual post");
    }
    const auto& context = op->context();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute cmd(context->device());
    ncnn::VkMat source, source_staging, branch_gpu, branch_staging, reduced, injection, output, downloaded, reduced_staging;
    std::vector<ncnn::VkMat> storage;
    qwen_check(qwen_upload(input, source, source_staging, cmd, context, op->option())
                   && op->record_pre(source, reduced, injection, cmd, storage),
               "record GatedResidual pre");
    if (head)
        output = reduced;
    else
    {
        qwen_check(qwen_upload(branch, branch_gpu, branch_staging, cmd, context, op->option())
                       && op->record_post(branch_gpu, source, injection, output, cmd)
                       && prepare_staging_batch(reduced_staging, rows, 32, context->staging_allocator())
                       && record_prepared_activation_staging_download(reduced, rows, 32, reduced_staging, cmd,
                                                                      context->device(), op->option()),
                   "record GatedResidual post");
    }
    qwen_check(prepare_staging_batch(downloaded, rows, output.w, context->staging_allocator())
                   && record_prepared_activation_staging_download(output, rows, output.w, downloaded, cmd,
                                                                  context->device(), op->option())
                   && submit_compute_and_wait(cmd, context->device()) == 0,
               "submit GatedResidual");
    ActivationBuffer actual(rows, static_cast<uint32_t>(output.w));
    qwen_check(copy_staging_to_cpu_batch(downloaded, actual), "download GatedResidual");
    qwen_equal(actual, expected, head ? "GatedResidual head" : "GatedResidual pre/post");
    if (!head)
    {
        ActivationBuffer reduced_host(rows, 32);
        qwen_check(copy_staging_to_cpu_batch(reduced_staging, reduced_host), "download GatedResidual pre");
        qwen_equal(reduced_host, expected_reduced, "GatedResidual reduction");
    }
}

static TensorData qwen_ple_table(DType dtype)
{
    TensorData table;
    table.dtype = dtype;
    const uint32_t dimension = dtype == DType::IQ4XS ? 256 : 160;
    table.shape = {4, dimension};
    const uint32_t block_bytes = static_cast<uint32_t>(qnk_block_bytes(dtype));
    const uint32_t blocks = dimension / qnk_block_size(dtype);
    table.quantized_data.resize(4 * blocks * block_bytes);
    for (uint32_t row = 0; row < 4; ++row)
        for (uint32_t block = 0; block < blocks; ++block)
        {
            uint8_t* data = table.quantized_data.data() + (row * blocks + block) * block_bytes;
            data[0] = 0;
            data[1] = static_cast<uint8_t>(0x14 + (row & 1) * 0x80); // +/- 2^-10.
            for (uint32_t byte = 2; byte < block_bytes; ++byte)
                data[byte] = static_cast<uint8_t>(row * 67 + block * 31 + byte * 13);
            if (dtype == DType::Q4_1 || dtype == DType::Q5_1)
            {
                data[2] = 0;
                data[3] = 0x18;
            }
        }
    return table;
}

static void qwen_ple_case(uint32_t device, DType dtype)
{
    WeightStore weights;
    PleBlockPlan plan;
    const uint32_t dimension = dtype == DType::IQ4XS ? 256 : 160;
    plan.embedding_dimension = dimension * 4;
    const uint32_t encoding = dtype == DType::Q4_0    ? 2
                              : dtype == DType::Q4_1  ? 3
                              : dtype == DType::Q5_0  ? 6
                              : dtype == DType::Q5_1  ? 7
                              : dtype == DType::Q8_0  ? 8
                              : dtype == DType::IQ4NL ? 20
                                                      : 23;
    plan.ngram_size = 3;
    plan.heads_per_ngram = 2;
    plan.convolution_kernel_size = 3;
    plan.eos_token_id = 0;
    plan.embedding_shards.push_back(qwen_add(weights, qwen_ple_table(dtype)));
    plan.key_weight = qwen_add(weights, qwen_matrix(128, plan.embedding_dimension, 4));
    plan.value_weight = qwen_add(weights, qwen_matrix(32, plan.embedding_dimension, 5));
    plan.key_norm_weight = qwen_add(weights, qwen_norm(128));
    plan.query_norm_weight = qwen_add(weights, qwen_norm(128));
    plan.convolution_norm_weight = qwen_add(weights, qwen_norm(128));
    plan.convolution_weight = qwen_add(weights, qwen_matrix(128, 3, 6));
    const auto integers = [&](std::vector<int64_t> values) {
        TensorData value;
        value.dtype = DType::Int64;
        value.shape = {static_cast<uint32_t>(values.size())};
        value.int64_data = std::move(values);
        return qwen_add(weights, std::move(value));
    };
    plan.hash_multipliers = integers({1, 17, 31});
    // Each head has one row, so expected selected native rows are independent
    // of the production host hash helper. Numerical kernels remain nontrivial.
    plan.head_vocabulary_sizes = integers({1, 1, 1, 1});
    plan.head_offsets = integers({0, 1, 2, 3});
    MoeModelDescriptor descriptor;
    descriptor.hidden_size = 32;
    descriptor.hyper_connection_multiplier = 4;
    descriptor.norm_epsilon = 1e-5f;
    descriptor.norm_weight_offset = 1.0f;
    auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    qwen_check(static_cast<bool>(context), "PLE Vulkan context");
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = opt.use_packing_layout = false;
    opt.blob_vkallocator = opt.workspace_vkallocator = context->blob_allocator();
    opt.staging_vkallocator = context->staging_allocator();
    LayerCache cache;
    AttentionScratch scratch;
    CompiledOperatorTable operators;
    ncnn::VkMat key_norm, query_norm, conv_norm, conv_weight, state;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        pipeline = qwen_pipeline(context, opt);
        qwen_check(static_cast<bool>(pipeline), "create PLE shader");
        {
            ncnn::VkMat probe;
            qwen_check(qwen_allocate(probe, 128, 2, context), "allocate Qwen refcount probe");
            ncnn::VkCompute probe_cmd(context->device());
            qwen_record(pipeline.get(), 0, 32, 4, 2, 128, descriptor.norm_epsilon, descriptor.norm_weight_offset,
                        {probe, probe, probe, probe, probe, probe}, probe_cmd);
            qwen_check(probe.refcount && *probe.refcount == 1, "Qwen record releases binding references");
        }
        ncnn::VkCompute cmd(context->device());
        ncnn::Mat zero(6 * 128, sizeof(float));
        zero.fill(0.0f);
        cmd.record_upload(zero, state, opt);
        qwen_check(qwen_float_weight(weights.at(plan.key_norm_weight), key_norm, cmd, opt)
                       && qwen_float_weight(weights.at(plan.query_norm_weight), query_norm, cmd, opt)
                       && qwen_float_weight(weights.at(plan.convolution_norm_weight), conv_norm, cmd, opt)
                       && qwen_float_weight(weights.at(plan.convolution_weight), conv_weight, cmd, opt)
                       && submit_compute_and_wait(cmd, context->device()) == 0,
                   "upload PLE constants");
    }
    uint32_t position = 0;
    uint32_t slot = 0;
    for (uint32_t rows : {3u, 1u, 5u})
    {
        const auto input = qwen_input(rows, 128, position);
        ActivationBuffer expected = input;
        std::vector<int32_t> tokens(rows);
        for (uint32_t row = 0; row < rows; ++row)
            tokens[row] = static_cast<int32_t>((position + row) % 7);
        qwen_check(static_cast<bool>(forward_ple(weights, operators, plan, descriptor, tokens, cache, scratch,
                                                 expected, OptimizationDefaultFlags)),
                   "CPU PLE oracle");
        const auto& table = weights.at(plan.embedding_shards.front());
        const uint32_t row_bytes = static_cast<uint32_t>(qnk_storage_bytes(dtype, 1, dimension));
        const uint32_t stride = (row_bytes + 3) / 4 * 4;
        ActivationBuffer packed(rows * 4, stride / 4);
        for (uint32_t row = 0; row < rows; ++row)
            for (uint32_t head = 0; head < 4; ++head)
                std::memcpy(packed.mutable_row_bytes(row * 4 + head).data(), table.qnk_values().data() + head * row_bytes, row_bytes);
        // CPU oracle supplies only the projection stage inputs. The native row
        // decoder and all PLE normalization/gate/recurrence phases below run
        // on GPU. Generic quantized projection parity is tested separately.
        ActivationBuffer key, value;
        forward_linear(weights.at(plan.key_weight), scratch.normalized, key, OptimizationDefaultFlags);
        forward_linear(weights.at(plan.value_weight), scratch.normalized, value, OptimizationDefaultFlags);
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute cmd(context->device());
        ncnn::VkMat input_gpu, input_staging, key_gpu, key_staging, value_gpu, value_staging, raw_gpu, raw_staging;
        ncnn::VkMat decoded, key_normalized, query_normalized, gated, convolution_input, output, output_staging, decode_staging;
        qwen_check(qwen_upload(input, input_gpu, input_staging, cmd, context, opt)
                       && qwen_upload(key, key_gpu, key_staging, cmd, context, opt)
                       && qwen_upload(value, value_gpu, value_staging, cmd, context, opt)
                       && qwen_upload(packed, raw_gpu, raw_staging, cmd, context, opt)
                       && qwen_allocate(decoded, plan.embedding_dimension, rows, context)
                       && qwen_allocate(key_normalized, 128, rows, context)
                       && qwen_allocate(query_normalized, 128, rows, context)
                       && qwen_allocate(gated, 128, rows, context)
                       && qwen_allocate(convolution_input, 128, rows, context)
                       && qwen_allocate(output, 128, rows, context),
                   "PLE stage allocations");
        for (const auto& constant : {key_norm, query_norm, conv_norm, conv_weight, state})
            qwen_prepare_read(constant);
        qwen_record(pipeline.get(), 7, plan.embedding_dimension, 4, rows, plan.embedding_dimension, 0, 0,
                    {raw_gpu, raw_gpu, raw_gpu, raw_gpu, raw_gpu, decoded}, cmd,
                    encoding, dimension, stride);
        qwen_record(pipeline.get(), 0, 32, 4, rows, 128, descriptor.norm_epsilon, 1,
                    {key_gpu, key_norm, key_gpu, key_gpu, key_gpu, key_normalized}, cmd);
        qwen_record(pipeline.get(), 0, 32, 4, rows, 128, descriptor.norm_epsilon, 1,
                    {input_gpu, query_norm, input_gpu, input_gpu, input_gpu, query_normalized}, cmd);
        qwen_record(pipeline.get(), 5, 32, 4, rows, 128, 0, 0,
                    {key_normalized, query_normalized, value_gpu, value_gpu, value_gpu, gated}, cmd);
        qwen_record(pipeline.get(), 0, 32, 4, rows, 128, descriptor.norm_epsilon, 1,
                    {gated, conv_norm, gated, gated, gated, convolution_input}, cmd);
        qwen_record(pipeline.get(), 6, 32, 4, rows, 128, 0, 0,
                    {convolution_input, gated, conv_weight, input_gpu, state, output}, cmd, 3, 3, 6, slot);
        qwen_check(prepare_staging_batch(output_staging, rows, 128, context->staging_allocator())
                       && prepare_staging_batch(decode_staging, rows, plan.embedding_dimension, context->staging_allocator())
                       && record_prepared_activation_staging_download(output, rows, 128, output_staging, cmd, context->device(), opt)
                       && record_prepared_activation_staging_download(decoded, rows, plan.embedding_dimension, decode_staging, cmd, context->device(), opt)
                       && submit_compute_and_wait(cmd, context->device()) == 0,
                   "submit PLE phases");
        ActivationBuffer actual(rows, 128);
        ActivationBuffer embeddings(rows, plan.embedding_dimension);
        qwen_check(copy_staging_to_cpu_batch(output_staging, actual) && copy_staging_to_cpu_batch(decode_staging, embeddings), "download PLE phases");
        qwen_equal(embeddings, scratch.normalized, "native PLE row decode", 0, 0);
        qwen_equal(actual, expected, "PLE grouped norm/gate/dilated convolution", 0.0005f, 0.002f);
        slot = (slot + rows) % 6;
        qwen_check(slot == cache.ple_first_slot, "PLE GPU/CPU cursor across wrap");
        position += rows;
    }
    std::cout << "PLE device=" << device << " dtype=" << static_cast<int>(dtype) << " phases passed\n";
}
#endif

} // namespace moe
} // namespace ncnn

int main()
{
    using namespace ncnn;
    using namespace moe;
    try
    {
#if NCNN_MOE_WITH_VULKAN
        if (ncnn::moe::get_gpu_count() == 0)
            return 77;
        for (uint32_t device = 0; device < ncnn::moe::get_gpu_count(); ++device)
        {
            qwen_gr_case(device, 1, false);
            qwen_gr_case(device, 3, false);
            qwen_gr_case(device, 3, true);
            for (DType dtype : {DType::IQ4NL, DType::Q8_0, DType::Q5_1, DType::Q4_0,
                                DType::Q4_1, DType::Q5_0, DType::IQ4XS})
                qwen_ple_case(device, dtype);
        }
        std::cout << "Qwen device numerical tests passed\n";
        return 0;
#else
        return 77;
#endif
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
