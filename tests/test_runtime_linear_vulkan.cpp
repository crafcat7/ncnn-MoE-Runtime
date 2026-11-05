#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_linear_operator()
{
    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {4, 3};
    matrix.float32_data = {
        0.1234f,
        -0.9876f,
        1.2345f,
        -0.2222f,
        0.3333f,
        -0.4444f,
        1.1111f,
        -1.2222f,
        0.5555f,
        -0.8765f,
        0.7654f,
        -0.6543f,
    };
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {4};
    bias.float32_data = {0.1357f, -0.2468f, 0.3579f, -0.4680f};

    ActivationBuffer input(2, 3);
    input.row(0)[0] = 0.2345f;
    input.row(0)[1] = -1.3456f;
    input.row(0)[2] = 2.4567f;
    input.row(1)[0] = -0.5678f;
    input.row(1)[1] = 1.6789f;
    input.row(1)[2] = -2.7891f;
    ActivationBuffer output = forward_linear(matrix, input, g_test_optimization_flags);
    add_bias_inplace(output, bias);
    check(static_cast<bool>(output.rows() == 2));
    check(static_cast<bool>(output.columns() == 4));
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            float expected = bias.float32_data[column];
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column)
            {
                expected += matrix.float32_data[column * input.columns() + input_column] * input.row(row_index)[input_column];
            }
            check_near(output.row(row_index)[column], expected, 1e-5f);
        }
    }
    const ActivationBuffer expected_output = output;
    std::atomic<bool> concurrent_linear_valid{true};
    std::array<std::thread, 4> linear_workers;
    for (std::thread& worker : linear_workers)
    {
        worker = std::thread([&] {
            for (uint32_t iteration = 0; iteration < 64; ++iteration)
            {
                ActivationBuffer concurrent_output = forward_linear(matrix, input, g_test_optimization_flags);
                add_bias_inplace(concurrent_output, bias);
                for (size_t row = 0; row < concurrent_output.rows(); ++row)
                {
                    for (uint32_t column = 0; column < concurrent_output.columns(); ++column)
                    {
                        if (!std::isfinite(concurrent_output.row(row)[column])
                            || std::abs(concurrent_output.row(row)[column] - expected_output.row(row)[column]) > 1e-5f)
                        {
                            concurrent_linear_valid.store(false, std::memory_order_relaxed);
                            return;
                        }
                    }
                }
            }
        });
    }
    for (std::thread& worker : linear_workers)
        worker.join();
    check(concurrent_linear_valid.load(std::memory_order_relaxed));

    TensorData bfloat_matrix;
    bfloat_matrix.dtype = DType::BFloat16;
    bfloat_matrix.shape = matrix.shape;
    for (float value : matrix.float32_data)
        bfloat_matrix.bfloat16_data.push_back(float_to_bfloat16(value));
    TensorData bfloat_bias;
    bfloat_bias.dtype = DType::BFloat16;
    bfloat_bias.shape = bias.shape;
    for (float value : bias.float32_data)
        bfloat_bias.bfloat16_data.push_back(float_to_bfloat16(value));

    output = forward_linear(bfloat_matrix, input, g_test_optimization_flags);
    add_bias_inplace(output, bfloat_bias);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            float expected = bfloat16_to_float(bfloat_bias.bfloat16_data[column]);
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column)
            {
                expected += bfloat16_to_float(bfloat_matrix.bfloat16_data[column * input.columns()
                                                                          + input_column])
                            * input.row(row_index)[input_column];
            }
            check_near(output.row(row_index)[column], expected, 1e-5f);
        }
    }

#if NCNN_MOE_WITH_NCNN && NCNN_MOE_WITH_VULKAN
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    if (get_gpu_count() > 0)
    {
        const auto vulkan_linear = Linear::create(matrix,
                                                  &bias,
                                                  automatic_vulkan_device_index,
                                                  vulkan_runtime,
                                                  g_test_optimization_flags);
        check(static_cast<bool>(vulkan_linear));
        const VulkanStatistics initial_counters = get_vulkan_statistics(vulkan_runtime);
        for (uint32_t iteration = 0; iteration < 4; ++iteration)
        {
            check(static_cast<bool>(vulkan_linear->forward(input, output)));
            for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            {
                for (uint32_t column = 0; column < output.columns(); ++column)
                {
                    float expected = bias.float32_data[column];
                    for (uint32_t input_column = 0; input_column < input.columns(); ++input_column)
                    {
                        expected += matrix.float32_data[column * input.columns() + input_column] * input.row(row_index)[input_column];
                    }
                    check_near(output.row(row_index)[column], expected, 1e-4f);
                }
            }
        }
        const VulkanStatistics final_counters = get_vulkan_statistics(vulkan_runtime);
        check(static_cast<bool>(final_counters.compute_submissions - initial_counters.compute_submissions == 4));
        check(static_cast<bool>(final_counters.batch_uploads - initial_counters.batch_uploads == 4));
        check(static_cast<bool>(final_counters.batch_downloads - initial_counters.batch_downloads == 4));
        TensorData chain_matrix;
        chain_matrix.dtype = DType::Float32;
        chain_matrix.shape = {5, 4};
        chain_matrix.float32_data.resize(chain_matrix.element_count());
        for (size_t index = 0; index < chain_matrix.float32_data.size(); ++index)
        {
            chain_matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 7) % 23) - 11)
                                               * 0.03125f;
        }
        TensorData parallel_matrix;
        parallel_matrix.dtype = DType::Float32;
        parallel_matrix.shape = {6, 3};
        parallel_matrix.float32_data.resize(parallel_matrix.element_count());
        for (size_t index = 0;
             index < parallel_matrix.float32_data.size();
             ++index)
        {
            parallel_matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 11) % 29) - 14)
                                                  * 0.0234375f;
        }
        const auto chain_operator = Linear::create(chain_matrix,
                                                   nullptr,
                                                   automatic_vulkan_device_index,
                                                   vulkan_runtime,
                                                   g_test_optimization_flags);
        const auto parallel_operator = Linear::create(parallel_matrix,
                                                      nullptr,
                                                      automatic_vulkan_device_index,
                                                      vulkan_runtime,
                                                      g_test_optimization_flags);
        check(static_cast<bool>(chain_operator));
        check(static_cast<bool>(parallel_operator));
        const ActivationBuffer expected_chain = forward_linear(chain_matrix,
                                                               expected_output,
                                                               g_test_optimization_flags);
        const ActivationBuffer expected_parallel = forward_linear(parallel_matrix,
                                                                  input,
                                                                  g_test_optimization_flags);
        const VulkanStatistics graph_before = get_vulkan_statistics(vulkan_runtime);
        auto graph = CommandGraph_vulkan::create(*vulkan_linear);
        DeviceTensor_vulkan graph_input;
        DeviceTensor_vulkan graph_first;
        DeviceTensor_vulkan graph_chain;
        DeviceTensor_vulkan graph_parallel;
        ActivationBuffer graph_chain_output;
        ActivationBuffer graph_parallel_output;
        check(static_cast<bool>(graph
                                && graph->upload(input, graph_input)
                                && graph->linear(*vulkan_linear, graph_input, graph_first)
                                && graph->linear(*chain_operator, graph_first, graph_chain)
                                && graph->linear(*parallel_operator,
                                                 graph_input,
                                                 graph_parallel)
                                && graph->download(graph_chain, graph_chain_output)
                                && graph->download(graph_parallel, graph_parallel_output)
                                && graph->submit()
                                && graph->wait()));
        for (size_t row = 0; row < input.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < expected_chain.columns();
                 ++column)
            {
                check_near(graph_chain_output.row(row)[column],
                           expected_chain.row(row)[column],
                           1e-4f);
            }
            for (uint32_t column = 0;
                 column < expected_parallel.columns();
                 ++column)
            {
                check_near(graph_parallel_output.row(row)[column],
                           expected_parallel.row(row)[column],
                           1e-4f);
            }
        }
        const VulkanStatistics graph_after = get_vulkan_statistics(vulkan_runtime);
        check(static_cast<bool>(graph_after.compute_submissions
                                    - graph_before.compute_submissions
                                == 1));
        check(static_cast<bool>(graph_after.batch_uploads - graph_before.batch_uploads == 1));
        check(static_cast<bool>(graph_after.batch_downloads - graph_before.batch_downloads == 2));

        ActivationBuffer typed_input(input.rows(), input.columns(), DType::BFloat16);
        for (size_t row = 0; row < input.rows(); ++row)
        {
            const std::span<std::byte> destination = typed_input.mutable_row_bytes(row);
            for (uint32_t column = 0; column < input.columns(); ++column)
            {
                const uint16_t value = float_to_bfloat16(input.row(row)[column]);
                std::memcpy(destination.data() + column * sizeof(uint16_t),
                            &value,
                            sizeof(value));
            }
        }
        ActivationBuffer typed_output(0, 0, DType::BFloat16);
        auto typed_graph = CommandGraph_vulkan::create(*vulkan_linear);
        DeviceTensor_vulkan typed_graph_input;
        DeviceTensor_vulkan typed_graph_output;
        check(static_cast<bool>(typed_graph));
        check(typed_graph->upload(typed_input, typed_graph_input));
        check(typed_graph->linear(*vulkan_linear,
                                  typed_graph_input,
                                  typed_graph_output));
        check(typed_graph->download(typed_graph_output, typed_output));
        check(typed_graph->submit());
        check(typed_graph->wait());
        check(static_cast<bool>(typed_output.dtype() == DType::BFloat16));
        for (size_t row = 0; row < typed_input.rows(); ++row)
        {
            const std::span<const std::byte> source = typed_input.row_bytes(row);
            for (uint32_t column = 0;
                 column < typed_input.columns();
                 ++column)
            {
                float expected = bias.float32_data[column];
                for (uint32_t input_column = 0;
                     input_column < typed_input.columns();
                     ++input_column)
                {
                    uint16_t input_value = 0;
                    std::memcpy(&input_value,
                                source.data()
                                    + input_column * sizeof(uint16_t),
                                sizeof(input_value));
                    expected += matrix.float32_data[column * typed_input.columns()
                                                    + input_column]
                                * bfloat16_to_float(input_value);
                }
                const std::span<const std::byte> output_bytes = typed_output.row_bytes(row);
                uint16_t output_value = 0;
                std::memcpy(&output_value,
                            output_bytes.data() + column * sizeof(uint16_t),
                            sizeof(output_value));
                check_near(bfloat16_to_float(output_value),
                           expected,
                           bfloat16_storage_tolerance(expected));
            }
        }
    }
#endif
}

void test_vulkan_device_tensor_residency()
{
    if (get_gpu_count() == 0)
        return;

    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {16, 16};
    matrix.float32_data.resize(matrix.element_count());
    for (size_t index = 0; index < matrix.float32_data.size(); ++index)
        matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 7 + 3) % 13) - 6) / 32.0f;
    ActivationBuffer input(3, 16);
    for (size_t index = 0; index < input.rows() * input.columns(); ++index)
        input.row(index / input.columns())[index % input.columns()] = static_cast<float>(static_cast<int>((index * 5 + 1) % 17) - 8) / 16.0f;
    const ActivationBuffer expected_first = forward_linear(matrix, input, 0);
    const ActivationBuffer expected_second = forward_linear(matrix, expected_first, 0);
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    const auto op = Linear::create(matrix, nullptr, get_default_gpu_index(), runtime,
                                   OptimizationDefaultFlags);
    check(static_cast<bool>(op));
    auto producer = CommandGraph_vulkan::create(*op);
    auto consumer = CommandGraph_vulkan::create(*op);
    check(static_cast<bool>(producer));
    check(static_cast<bool>(consumer));
    DeviceTensor_vulkan source;
    DeviceTensor_vulkan retained;
    DeviceTensor_vulkan output;
    ActivationBuffer downloaded;
    check(producer->upload(input, source));
    check(producer->linear(*op, source, retained));
    check(!consumer->linear(*op, retained, output));
    check(!consumer->download(retained, downloaded));
    check(producer->submit());
    // Host completion must be explicit even if the queue finishes immediately.
    check(!consumer->linear(*op, retained, output));
    check(!consumer->download(retained, downloaded));
    check(producer->wait());
    producer.reset();
    check(!retained.empty());
    check(retained.rows() == input.rows());
    check(retained.columns() == matrix.shape[0]);

    const VulkanRuntimePtr other_runtime = create_vulkan_runtime();
    const auto other_op = Linear::create(matrix, nullptr, get_default_gpu_index(), other_runtime,
                                         OptimizationDefaultFlags);
    check(static_cast<bool>(other_op));
    auto other_graph = CommandGraph_vulkan::create(*other_op);
    check(static_cast<bool>(other_graph));
    check(!other_graph->linear(*other_op, retained, output));
    check(!other_graph->download(retained, downloaded));

    DeviceTensor_vulkan abandoned;
    {
        auto unsubmitted = CommandGraph_vulkan::create(*op);
        check(static_cast<bool>(unsubmitted));
        check(unsubmitted->upload(input, abandoned));
    }
    check(!consumer->linear(*op, abandoned, output));
    check(!consumer->download(abandoned, downloaded));

    const VulkanStatistics before = get_vulkan_statistics(runtime);
    ActivationBuffer retained_host;
    check(consumer->linear(*op, retained, output));
    check(consumer->download(retained, retained_host));
    check(consumer->download(output, downloaded));
    check(consumer->submit());
    check(consumer->wait());
    const VulkanStatistics after = get_vulkan_statistics(runtime);
    check(after.batch_uploads == before.batch_uploads);
    check(after.compute_submissions == before.compute_submissions + 1);
    check(after.batch_downloads == before.batch_downloads + 2);
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < matrix.shape[0]; ++column)
        {
            check_near(retained_host.row(row)[column], expected_first.row(row)[column], 1e-4f);
            check_near(downloaded.row(row)[column], expected_second.row(row)[column], 1e-4f);
        }

    consumer.reset();
    DeviceTensor_vulkan uploaded;
    {
        auto upload_only = CommandGraph_vulkan::create(*op);
        check(static_cast<bool>(upload_only));
        check(upload_only->upload(input, uploaded));
        check(upload_only->submit());
        check(upload_only->wait());
    }
    const VulkanStatistics cancellation_before = get_vulkan_statistics(runtime);
    ActivationBuffer canceled_download;
    ActivationBuffer canceled_upload_download;
    DeviceTensor_vulkan abandoned_projection;
    {
        auto canceled = CommandGraph_vulkan::create(*op);
        check(static_cast<bool>(canceled));
        check(canceled->download(output, canceled_download));
        check(canceled->download(uploaded, canceled_upload_download));
        // Recording changes shared buffer flags even though this consumer
        // never submits its shader/transfer read barriers.
    }
    {
        auto canceled = CommandGraph_vulkan::create(*op);
        check(static_cast<bool>(canceled));
        check(canceled->linear(*op, output, abandoned_projection));
    }
    const VulkanStatistics cancellation_after = get_vulkan_statistics(runtime);
    check(cancellation_after.compute_submissions == cancellation_before.compute_submissions);
    check(cancellation_after.batch_uploads == cancellation_before.batch_uploads);
    check(cancellation_after.batch_downloads == cancellation_before.batch_downloads);

    auto materializer = CommandGraph_vulkan::create(*op);
    check(static_cast<bool>(materializer));
    ActivationBuffer materialized;
    ActivationBuffer materialized_upload;
    const VulkanStatistics materialize_before = get_vulkan_statistics(runtime);
    check(!materializer->download(abandoned_projection, materialized));
    check(materializer->download(output, materialized));
    check(materializer->download(uploaded, materialized_upload));
    check(materializer->submit());
    check(materializer->wait());
    const VulkanStatistics materialize_after = get_vulkan_statistics(runtime);
    check(materialize_after.batch_uploads == materialize_before.batch_uploads);
    check(materialize_after.batch_downloads == materialize_before.batch_downloads + 2);
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < matrix.shape[0]; ++column)
        {
            check_near(materialized.row(row)[column], expected_second.row(row)[column], 1e-4f);
            check_near(materialized_upload.row(row)[column], input.row(row)[column], 1e-4f);
        }

    materializer.reset(); // Both bounded command slots are needed below.

    // Replacing a submitted Graph must finish its old CB and pending host
    // download before releasing the old transfer slot/owners.
    auto move_target = CommandGraph_vulkan::create(*op);
    auto move_source = CommandGraph_vulkan::create(*op);
    check(static_cast<bool>(move_target) && static_cast<bool>(move_source));
    DeviceTensor_vulkan retired_input;
    DeviceTensor_vulkan retired_output;
    DeviceTensor_vulkan replacement_input;
    DeviceTensor_vulkan replacement_output;
    ActivationBuffer retired_download;
    ActivationBuffer replacement_download;
    check(move_target->upload(input, retired_input));
    check(move_target->linear(*op, retired_input, retired_output));
    check(move_target->download(retired_output, retired_download));
    check(move_target->submit());
    check(move_source->upload(input, replacement_input));
    check(move_source->linear(*op, replacement_input, replacement_output));
    check(move_source->download(replacement_output, replacement_download));
    *move_target = std::move(*move_target);
    *move_target = std::move(*move_source);
    check(!move_source->submit());
    check(move_target->submit());
    check(move_target->wait());
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < matrix.shape[0]; ++column)
        {
            check_near(retired_download.row(row)[column], expected_first.row(row)[column], 1e-4f);
            check_near(replacement_download.row(row)[column], expected_first.row(row)[column], 1e-4f);
        }
    move_target.reset();
    move_source.reset();

    // The fence can succeed even when copying its staging result into the
    // caller's host buffer fails. Keep GPU completion separate from logical
    // Graph completion; destruction must not wait the cleared native CB again.
    auto failed_download_graph = CommandGraph_vulkan::create(*op);
    check(static_cast<bool>(failed_download_graph));
    DeviceTensor_vulkan failed_input;
    DeviceTensor_vulkan failed_output;
    ActivationBuffer failed_download;
    check(failed_download_graph->upload(input, failed_input));
    check(failed_download_graph->linear(*op, failed_input, failed_output));
    check(failed_download_graph->download(failed_output, failed_download));
    check(failed_download_graph->submit());
    failed_download.clear(); // Deliberate host shape mismatch after recording.
    check(!failed_download_graph->wait());
    check(!failed_download_graph->wait());
    failed_download_graph.reset();

    auto recovered_graph = CommandGraph_vulkan::create(*op);
    check(static_cast<bool>(recovered_graph));
    DeviceTensor_vulkan recovered_input;
    DeviceTensor_vulkan recovered_output;
    ActivationBuffer recovered_download;
    check(!recovered_graph->download(failed_output, recovered_download));
    check(recovered_graph->upload(input, recovered_input));
    check(recovered_graph->linear(*op, recovered_input, recovered_output));
    check(recovered_graph->download(recovered_output, recovered_download));
    check(recovered_graph->submit());
    check(recovered_graph->wait());
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < matrix.shape[0]; ++column)
            check_near(recovered_download.row(row)[column], expected_first.row(row)[column], 1e-4f);
}

void test_dense_mxn_tiles()
{
    TensorData float_matrix;
    float_matrix.dtype = DType::Float32;
    float_matrix.shape = {8, 32};
    float_matrix.float32_data.resize(float_matrix.element_count());
    for (size_t index = 0; index < float_matrix.float32_data.size(); ++index)
    {
        float_matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 13 + 3) % 41) - 20)
                                           * 0.03125f;
    }
    ActivationBuffer input(8, 32);
    for (size_t index = 0; index < input.rows() * input.columns(); ++index)
    {
        input.row(index / input.columns())[index % input.columns()] = static_cast<float>(static_cast<int>((index * 7 + 5) % 37) - 18)
                                                                      * 0.015625f;
    }
    const ActivationBuffer float_output = forward_linear(float_matrix,
                                                         input,
                                                         g_test_optimization_flags);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t output_column = 0;
             output_column < float_matrix.shape[0];
             ++output_column)
        {
            float expected = 0.0f;
            for (uint32_t input_column = 0;
                 input_column < input.columns();
                 ++input_column)
            {
                expected += float_matrix.float32_data[static_cast<size_t>(output_column) * input.columns()
                                                      + input_column]
                            * input.row(token)[input_column];
            }
            check_near(float_output.row(token)[output_column],
                       expected,
                       1e-4f);
        }
    }

    TensorData bfloat_matrix;
    bfloat_matrix.dtype = DType::BFloat16;
    bfloat_matrix.shape = float_matrix.shape;
    bfloat_matrix.bfloat16_data.reserve(float_matrix.float32_data.size());
    for (float value : float_matrix.float32_data)
        bfloat_matrix.bfloat16_data.push_back(float_to_bfloat16(value));
    ActivationBuffer single(1, input.columns());
    std::copy_n(input.row(0), input.columns(), single.row(0));
    const ActivationBuffer single_output = forward_linear(bfloat_matrix,
                                                          single,
                                                          g_test_optimization_flags);
    const ActivationBuffer bfloat_output = forward_linear(bfloat_matrix,
                                                          input,
                                                          g_test_optimization_flags);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t output_column = 0;
             output_column < bfloat_matrix.shape[0];
             ++output_column)
        {
            float expected = 0.0f;
            for (uint32_t input_column = 0;
                 input_column < input.columns();
                 ++input_column)
            {
                expected += bfloat16_to_float(bfloat_matrix.bfloat16_data[static_cast<size_t>(output_column) * input.columns()
                                                                          + input_column])
                            * input.row(token)[input_column];
            }
            check_near(bfloat_output.row(token)[output_column],
                       expected,
                       1e-4f);
            if (token == 0)
                check_near(single_output.row(0)[output_column], expected, 1e-4f);
        }
    }

    constexpr uint32_t reuse_tokens = 9;
    constexpr uint32_t reuse_outputs = 65;
    constexpr uint32_t reuse_columns = 35;
    TensorData reused_bfloat_matrix;
    reused_bfloat_matrix.dtype = DType::BFloat16;
    reused_bfloat_matrix.shape = {reuse_outputs, reuse_columns};
    reused_bfloat_matrix.bfloat16_data.resize(reused_bfloat_matrix.element_count());
    for (size_t index = 0; index < reused_bfloat_matrix.bfloat16_data.size(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 19 + 11) % 53) - 26)
                            * 0.0234375f;
        reused_bfloat_matrix.bfloat16_data[index] = float_to_bfloat16(value);
    }
    ActivationBuffer reused_input(reuse_tokens, reuse_columns);
    for (size_t token = 0; token < reused_input.rows(); ++token)
    {
        for (uint32_t column = 0; column < reuse_columns; ++column)
        {
            reused_input.row(token)[column] = static_cast<float>(static_cast<int>((token * 17 + column * 7 + 3) % 47) - 23)
                                              * 0.01953125f;
        }
    }
    const ActivationBuffer reused_output = forward_linear(reused_bfloat_matrix,
                                                          reused_input,
                                                          g_test_optimization_flags);
    for (uint32_t token = 0; token < reuse_tokens; ++token)
    {
        for (uint32_t output_column = 0; output_column < reuse_outputs; ++output_column)
        {
            float expected = 0.0f;
            for (uint32_t input_column = 0; input_column < reuse_columns; ++input_column)
            {
                expected += bfloat16_to_float(reused_bfloat_matrix.bfloat16_data[static_cast<size_t>(output_column) * reuse_columns + input_column])
                            * reused_input.row(token)[input_column];
            }
            check_near(reused_output.row(token)[output_column], expected, 1e-4f);
        }
    }

    {
        // N=1025 yields eight full 128-column panels and a one-output tail.
        constexpr uint32_t M = 33;
        constexpr uint32_t N = 1025;
        constexpr uint32_t K = 513;
        TensorData wide_matrix;
        wide_matrix.dtype = DType::BFloat16;
        wide_matrix.shape = {N, K};
        wide_matrix.bfloat16_data.resize(wide_matrix.element_count());
        for (uint32_t row = 0; row < N; ++row)
        {
            for (uint32_t column = 0; column < K; ++column)
            {
                const int value = static_cast<int>((static_cast<uint64_t>(row) * 13
                                                    + static_cast<uint64_t>(column) * 7)
                                                   % 9)
                                  - 4;
                wide_matrix.bfloat16_data[static_cast<size_t>(row) * K + column] = float_to_bfloat16(static_cast<float>(value) * 0.03125f);
            }
        }
        ActivationBuffer wide_input(M, K);
        for (uint32_t row = 0; row < M; ++row)
        {
            for (uint32_t column = 0; column < K; ++column)
            {
                const int value = static_cast<int>((static_cast<uint64_t>(row) * 19
                                                    + static_cast<uint64_t>(column) * 11)
                                                   % 129)
                                  - 64;
                wide_input.row(row)[column] = static_cast<float>(value) * 0.015317f;
            }
        }
        const std::array<uint32_t, 2> thread_counts = {1, 8};
        for (uint32_t threads : thread_counts)
        {
            CpuOpenMpThreadLimitScope thread_limit;
            thread_limit.set(threads);
            const ActivationBuffer wide_output = forward_linear(wide_matrix,
                                                                wide_input,
                                                                g_test_optimization_flags);
            for (uint32_t row = 0; row < M; ++row)
            {
                for (uint32_t output_column = 0; output_column < N; ++output_column)
                {
                    double expected = 0.0;
                    const uint16_t* weights = wide_matrix.bfloat16_data.data()
                                              + static_cast<size_t>(output_column) * K;
                    for (uint32_t column = 0; column < K; ++column)
                        expected += static_cast<double>(bfloat16_to_float(weights[column]))
                                    * static_cast<double>(wide_input.row(row)[column]);
                    check(std::abs(static_cast<double>(wide_output.row(row)[output_column])
                                   - expected)
                          <= 1e-4);
                }
            }
        }
    }
}

void test_released_dense_host_storage_guard()
{
    ActivationBuffer input(1, 2);
    input.row(0)[0] = 1.0f;
    input.row(0)[1] = -2.0f;

    TensorData released_matrix;
    released_matrix.dtype = DType::Float32;
    released_matrix.shape = {2, 2};
    ActivationBuffer output;
    bool matrix_failure_reported = false;
    try
    {
        forward_linear(released_matrix,
                       input,
                       output,
                       g_test_optimization_flags);
    }
    catch (const std::runtime_error& error)
    {
        matrix_failure_reported = std::string(error.what()).find("host storage was released")
                                  != std::string::npos;
    }
    check(matrix_failure_reported);

    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {2, 2};
    matrix.float32_data = {1.0f, 0.0f, 0.0f, 1.0f};
    TensorData released_bias;
    released_bias.dtype = DType::Float32;
    released_bias.shape = {2};
    bool bias_failure_reported = false;
    try
    {
        forward_linear(matrix,
                       released_bias,
                       input,
                       output,
                       g_test_optimization_flags);
    }
    catch (const std::runtime_error& error)
    {
        bias_failure_reported = std::string(error.what()).find("host storage was released")
                                != std::string::npos;
    }
    check(bias_failure_reported);

    TensorData released_norm;
    released_norm.dtype = DType::BFloat16;
    released_norm.shape = {2};
    bool norm_failure_reported = false;
    try
    {
        forward_rms_norm(input,
                         released_norm,
                         1e-5f,
                         output,
                         0.0f);
    }
    catch (const std::runtime_error& error)
    {
        norm_failure_reported = std::string(error.what()).find("host storage was released")
                                != std::string::npos;
    }
    check(norm_failure_reported);
}

void test_ncnn_vulkan_float8_operator()
{
#if NCNN_MOE_WITH_NCNN
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    if (get_gpu_count() == 0)
        return;

    TensorData matrix;
    matrix.dtype = DType::Float8E4M3;
    matrix.shape = {256, 128};
    const size_t element_count = static_cast<size_t>(matrix.shape[0]) * matrix.shape[1];
    std::shared_ptr<uint8_t[]> storage(new uint8_t[element_count], std::default_delete<uint8_t[]>());
    for (size_t index = 0; index < element_count; ++index)
    {
        const float value = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.03125f;
        storage[index] = float_to_float8_e4m3(value);
    }
    matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    matrix.mapped_size = element_count;
    matrix.quantization_scales = {0.5f, 2.0f};

    ActivationBuffer input(2, 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * input.columns() + column) % 37) - 18) * 0.015625f;
        }
    }
    const ActivationBuffer cpu_output = forward_linear(matrix,
                                                       input,
                                                       g_test_optimization_flags);
    const auto vulkan = Float8Linear_vulkan::create(matrix,
                                                    nullptr,
                                                    1,
                                                    automatic_vulkan_device_index,
                                                    vulkan_runtime,
                                                    g_test_optimization_flags);
    check(static_cast<bool>(vulkan));
    {
        CompiledModel accounting_model;
        auto embedding = accounting_model.weights.add("token_embedding.weight", matrix);
        auto head = accounting_model.weights.add("lm_head.weight", matrix);
        check(static_cast<bool>(embedding) && static_cast<bool>(head));
        accounting_model.operators.bind_weight_count(accounting_model.weights.size());
        accounting_model.operators.at_weight_mutable(head.value()).float8 = vulkan;
        release_host_copies(accounting_model);
        check(accounting_model.released_dense_host_size == matrix.quantization_scales.size() * sizeof(float));
        check(accounting_model.weights.at(head.value()).float8_values().empty());
        check(!accounting_model.weights.at(embedding.value()).float8_values().empty());
        accounting_model.operators.at_weight_mutable(embedding.value()).float8 = vulkan;
        release_host_copies(accounting_model);
        check(accounting_model.released_dense_host_size == matrix.mapped_size + matrix.quantization_scales.size() * sizeof(float) * 2);
        const uint64_t released_size = accounting_model.released_dense_host_size;
        release_host_copies(accounting_model);
        check(accounting_model.released_dense_host_size == released_size);
    }
    {
        TensorData owned_matrix = matrix;
        owned_matrix.quantized_data.assign(matrix.float8_values().begin(), matrix.float8_values().end());
        owned_matrix.mapped_data.reset();
        owned_matrix.mapped_size = 0;
        CompiledModel accounting_model;
        auto head = accounting_model.weights.add("lm_head.weight", std::move(owned_matrix));
        check(static_cast<bool>(head));
        accounting_model.operators.bind_weight_count(accounting_model.weights.size());
        accounting_model.operators.at_weight_mutable(head.value()).float8 = vulkan;
        release_host_copies(accounting_model);
        check(accounting_model.weights.at(head.value()).quantized_data.empty());
        check(accounting_model.released_dense_host_size == matrix.mapped_size + matrix.quantization_scales.size() * sizeof(float));
    }
    ActivationBuffer vulkan_output;
    check(static_cast<bool>(vulkan->forward(input, vulkan_output)));
    const std::byte* output_storage = vulkan_output.bytes().data();
    check(static_cast<bool>(vulkan->forward(input, vulkan_output)));
    check(vulkan_output.bytes().data() == output_storage);
    check(static_cast<bool>(vulkan_output.rows() == cpu_output.rows()));
    check(static_cast<bool>(vulkan_output.columns() == cpu_output.columns()));
    for (size_t row = 0; row < cpu_output.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_output.columns(); ++column)
            check_near(vulkan_output.row(row)[column], cpu_output.row(row)[column], 1e-4f);
    }

    TensorData second_matrix;
    second_matrix.dtype = DType::Float8E4M3;
    second_matrix.shape = {128, 256};
    const size_t second_element_count = static_cast<size_t>(second_matrix.shape[0]) * second_matrix.shape[1];
    std::shared_ptr<uint8_t[]> second_storage(new uint8_t[second_element_count], std::default_delete<uint8_t[]>());
    for (size_t index = 0; index < second_element_count; ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 5) % 29) - 14) * 0.015625f;
        second_storage[index] = float_to_float8_e4m3(value);
    }
    second_matrix.mapped_data = std::shared_ptr<const uint8_t>(second_storage, second_storage.get());
    second_matrix.mapped_size = second_element_count;
    second_matrix.quantization_scales = {0.5f, 1.5f};
    const ActivationBuffer cpu_chain = forward_linear(second_matrix,
                                                      cpu_output,
                                                      g_test_optimization_flags);
    const auto second_vulkan = Float8Linear_vulkan::create(second_matrix,
                                                           nullptr,
                                                           1,
                                                           automatic_vulkan_device_index,
                                                           vulkan_runtime,
                                                           g_test_optimization_flags);
    check(static_cast<bool>(second_vulkan));
    ActivationBuffer vulkan_chain;
    check(static_cast<bool>(vulkan->forward_chain(input, *second_vulkan, vulkan_chain)));
    const std::byte* chain_storage = vulkan_chain.bytes().data();
    check(static_cast<bool>(vulkan->forward_chain(input, *second_vulkan, vulkan_chain)));
    check(vulkan_chain.bytes().data() == chain_storage);
    check(static_cast<bool>(vulkan_chain.rows() == cpu_chain.rows()));
    check(static_cast<bool>(vulkan_chain.columns() == cpu_chain.columns()));
    for (size_t row = 0; row < cpu_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_chain.columns(); ++column)
            check_near(vulkan_chain.row(row)[column], cpu_chain.row(row)[column], 1e-3f);
    }
    TensorData norm_weight;
    norm_weight.dtype = DType::Float32;
    norm_weight.shape = {256};
    norm_weight.float32_data.resize(256);
    for (uint32_t column = 0; column < 256; ++column)
        norm_weight.float32_data[column] = 0.75f + static_cast<float>(column % 9) * 0.03125f;
    constexpr float norm_epsilon = 1e-6f;
    const ActivationBuffer normalized_cpu_output = forward_rms_norm(cpu_output,
                                                                    norm_weight,
                                                                    norm_epsilon,
                                                                    0.0f);
    const ActivationBuffer cpu_norm_chain = forward_linear(second_matrix,
                                                           normalized_cpu_output,
                                                           g_test_optimization_flags);
    check(static_cast<bool>(vulkan->prepare_rms_norm(norm_weight, norm_epsilon)));
    ActivationBuffer vulkan_norm_chain;
    check(static_cast<bool>(vulkan->forward_rms_norm_chain(input, *second_vulkan, vulkan_norm_chain)));
    const std::byte* norm_chain_storage = vulkan_norm_chain.bytes().data();
    check(static_cast<bool>(vulkan->forward_rms_norm_chain(input, *second_vulkan, vulkan_norm_chain)));
    check(vulkan_norm_chain.bytes().data() == norm_chain_storage);
    for (size_t row = 0; row < cpu_norm_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_norm_chain.columns(); ++column)
            check_near(vulkan_norm_chain.row(row)[column], cpu_norm_chain.row(row)[column], 2e-3f);
    }
    ActivationBuffer vulkan_parallel_chain;
    ActivationBuffer vulkan_parallel_output;
    check(static_cast<bool>(vulkan->forward_rms_norm_chain_parallel(input,
                                                                    *second_vulkan,
                                                                    *vulkan,
                                                                    vulkan_parallel_chain,
                                                                    vulkan_parallel_output)));
    for (size_t row = 0; row < cpu_norm_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_norm_chain.columns(); ++column)
            check_near(vulkan_parallel_chain.row(row)[column], cpu_norm_chain.row(row)[column], 2e-3f);
        for (uint32_t column = 0; column < cpu_output.columns(); ++column)
            check_near(vulkan_parallel_output.row(row)[column], cpu_output.row(row)[column], 1e-4f);
    }
    ActivationBuffer cpu_activated = cpu_output;
    for (size_t row = 0; row < cpu_activated.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_activated.columns(); ++column)
        {
            const float gate = cpu_output.row(row)[column];
            cpu_activated.row(row)[column] = gate / (1.0f + std::exp(-gate)) * gate;
        }
    }
    const ActivationBuffer cpu_swiglu_chain = forward_linear(second_matrix,
                                                             cpu_activated,
                                                             g_test_optimization_flags);
    ActivationBuffer vulkan_swiglu_chain;
    check(static_cast<bool>(vulkan->forward_swiglu_chain(input,
                                                         *vulkan,
                                                         *second_vulkan,
                                                         ExpertActivation::DeepSeekSwiGlu,
                                                         0.0f,
                                                         vulkan_swiglu_chain)));
    for (size_t row = 0; row < cpu_swiglu_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_swiglu_chain.columns(); ++column)
            check_near(vulkan_swiglu_chain.row(row)[column], cpu_swiglu_chain.row(row)[column], 2e-3f);
    }
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            const float expected = static_cast<float>(static_cast<int>((row * input.columns() + column) % 37) - 18) * 0.015625f;
            check(input.row(row)[column] == expected);
        }
    }

    auto make_shared_float8 = [](uint32_t rows, uint32_t columns) {
        TensorData tensor;
        tensor.dtype = DType::Float8E4M3;
        tensor.shape = {rows, columns};
        const size_t element_count = static_cast<size_t>(rows) * columns;
        std::shared_ptr<uint8_t[]> storage(new uint8_t[element_count], std::default_delete<uint8_t[]>());
        std::fill_n(storage.get(), element_count, float_to_float8_e4m3(0.5f));
        tensor.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
        tensor.mapped_size = element_count;
        tensor.quantization_scales.assign(static_cast<size_t>((rows + 127) / 128) * ((columns + 127) / 128),
                                          1.0f);
        return tensor;
    };
    CompiledModel shared_model;
    shared_model.descriptor.hidden_size = 128;
    shared_model.descriptor.layers.resize(2);
    shared_model.vulkan_runtime = vulkan_runtime;
    shared_model.opt.hybrid_mode = HybridMode::HybridExperts;
    shared_model.opt.vulkan_device_index = automatic_vulkan_device_index;
    shared_model.opt.optimization_flags = g_test_optimization_flags;
    shared_model.net.layer_plans.resize(2);
    for (CompiledLayerPlan& layer_plan : shared_model.net.layer_plans)
        layer_plan.vulkan_device_index = automatic_vulkan_device_index;
    MoeBlockPlan& shared_moe = shared_model.net.layer_plans[0].moe;
    MoeBlockPlan& routed_shared_moe = shared_model.net.layer_plans[1].moe;
    shared_moe.has_shared_expert = true;
    routed_shared_moe.has_shared_expert = true;
    auto add_shared_weight = [&](const char* name, TensorData tensor) {
        auto handle = shared_model.weights.add(name, std::move(tensor));
        check(static_cast<bool>(handle));
        return handle.value();
    };
    shared_model.lm_head_weight = add_shared_weight("lm_head.weight", make_shared_float8(256, 128));
    shared_moe.shared_expert.gate_weight = add_shared_weight("shared.gate.weight", make_shared_float8(128, 128));
    shared_moe.shared_expert.up_weight = add_shared_weight("shared.up.weight", make_shared_float8(128, 128));
    shared_moe.shared_expert.down_weight = add_shared_weight("shared.down.weight", make_shared_float8(128, 128));
    routed_shared_moe.shared_expert.gate_weight = add_shared_weight("routed_shared.gate.weight", make_shared_float8(128, 128));
    routed_shared_moe.shared_expert.up_weight = add_shared_weight("routed_shared.up.weight", make_shared_float8(128, 128));
    routed_shared_moe.shared_expert.down_weight = add_shared_weight("routed_shared.down.weight", make_shared_float8(128, 128));
    routed_shared_moe.shared_expert_gate_weight = add_shared_weight("routed_shared.router_gate.weight", make_shared_float8(1, 128));
    CompilerOption pipeline_options;
    pipeline_options.flags |= BackendVulkanDense;
    auto prepared_shared = prepare_model_pipeline(shared_model, pipeline_options);
    check(static_cast<bool>(prepared_shared));
    shared_model.opt.optimization_flags = g_test_optimization_flags;
    check(shared_moe.fused_shared_input_bfloat16_operator == invalid_compiled_operator_handle);
    check(routed_shared_moe.fused_shared_input_bfloat16_operator == invalid_compiled_operator_handle);
    check(!support_vulkan_shared_experts(shared_model.operators, shared_moe));
    check(!support_vulkan_shared_experts(shared_model.operators, routed_shared_moe));
    const std::array<TensorHandle, 7> shared_handles = {
        shared_moe.shared_expert.gate_weight,
        shared_moe.shared_expert.up_weight,
        shared_moe.shared_expert.down_weight,
        routed_shared_moe.shared_expert.gate_weight,
        routed_shared_moe.shared_expert.up_weight,
        routed_shared_moe.shared_expert.down_weight,
        routed_shared_moe.shared_expert_gate_weight,
    };
    for (TensorHandle handle : shared_handles)
        check(!shared_model.operators.at_weight(handle).float8);

    ActivationBuffer shared_before_release;
    ActivationBuffer routed_shared_before_release;
    ExpertWorkspace shared_workspace;
    ExpertWorkspace routed_shared_workspace;
    forward_shared_expert(shared_model,
                          shared_moe,
                          input,
                          shared_before_release,
                          shared_workspace);
    forward_shared_expert(shared_model,
                          routed_shared_moe,
                          input,
                          routed_shared_before_release,
                          routed_shared_workspace);
    const auto check_finite_nonzero = [](const ActivationBuffer& output) {
        return std::all_of(output.values().begin(),
                           output.values().end(),
                           [](float value) { return std::isfinite(value); })
               && std::any_of(output.values().begin(),
                              output.values().end(),
                              [](float value) { return value != 0.0f; });
    };
    check(check_finite_nonzero(shared_before_release));
    check(check_finite_nonzero(routed_shared_before_release));

    release_host_copies(shared_model);
    for (TensorHandle handle : shared_handles)
    {
        const TensorData& tensor = shared_model.weights.at(handle);
        check(tensor.float8_values().size() == tensor.element_count());
        check(!tensor.quantization_scales.empty());
    }

    ActivationBuffer shared_after_release;
    ActivationBuffer routed_shared_after_release;
    forward_shared_expert(shared_model,
                          shared_moe,
                          input,
                          shared_after_release,
                          shared_workspace);
    forward_shared_expert(shared_model,
                          routed_shared_moe,
                          input,
                          routed_shared_after_release,
                          routed_shared_workspace);
    check(check_finite_nonzero(shared_after_release));
    check(check_finite_nonzero(routed_shared_after_release));
    check(shared_after_release.rows() == shared_before_release.rows());
    check(shared_after_release.columns() == shared_before_release.columns());
    check(routed_shared_after_release.rows() == routed_shared_before_release.rows());
    check(routed_shared_after_release.columns() == routed_shared_before_release.columns());
    for (size_t row = 0; row < shared_before_release.rows(); ++row)
    {
        for (uint32_t column = 0; column < shared_before_release.columns(); ++column)
        {
            check_near(shared_after_release.row(row)[column],
                       shared_before_release.row(row)[column],
                       1e-5f);
            check_near(routed_shared_after_release.row(row)[column],
                       routed_shared_before_release.row(row)[column],
                       1e-5f);
        }
    }
#endif
}

void test_ncnn_vulkan_bfloat16_operator()
{
#if NCNN_MOE_WITH_NCNN && NCNN_MOE_WITH_VULKAN
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    if (get_gpu_count() == 0)
    {
        std::cout << "SKIP BF16 Vulkan operator: no Vulkan device is available\n";
        return;
    }
    const uint32_t device_index = get_default_gpu_index();
    const auto context = VulkanContext::acquire(device_index, vulkan_runtime, g_test_optimization_flags);
    check(static_cast<bool>(context));
    const uint32_t subgroup_size = context->device()->info.subgroup_size();
    if (subgroup_size != 32)
    {
        std::cout << "SKIP BF16 Vulkan operator: device subgroup size " << subgroup_size
                  << " is not 32\n";
        return;
    }
    TensorData first;
    first.dtype = DType::BFloat16;
    first.shape = {192, 128};
    first.bfloat16_data.resize(first.element_count());
    for (size_t index = 0; index < first.element_count(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 17) % 97) - 48)
                            * 0.0013f;
        first.bfloat16_data[index] = float_to_bfloat16(value);
    }
    TensorData second;
    second.dtype = DType::BFloat16;
    second.shape = {64, 128};
    second.bfloat16_data.resize(second.element_count());
    for (size_t index = 0; index < second.element_count(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 11) % 71) - 35)
                            * 0.0017f;
        second.bfloat16_data[index] = float_to_bfloat16(value);
    }
    TensorData first_bias;
    first_bias.dtype = DType::BFloat16;
    first_bias.shape = {192};
    first_bias.bfloat16_data.resize(192);
    for (uint32_t index = 0; index < 192; ++index)
    {
        first_bias.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>(index % 13) - 6)
                                                            * 0.0021f);
    }
    TensorData second_bias;
    second_bias.dtype = DType::Float32;
    second_bias.shape = {64};
    second_bias.float32_data.resize(64);
    for (uint32_t index = 0; index < 64; ++index)
    {
        second_bias.float32_data[index] = static_cast<float>(static_cast<int>(index % 9) - 4)
                                          * 0.0019f;
    }

    ActivationBuffer input(3, 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * input.columns() + column * 7) % 89)
                                                        - 44)
                                     * 0.0031f;
        }
    }
    const ActivationBuffer first_cpu = forward_linear(first, first_bias, input, g_test_optimization_flags);
    const auto first_vulkan = Bfloat16Linear_vulkan::create(first,
                                                            &first_bias,
                                                            device_index,
                                                            vulkan_runtime,
                                                            g_test_optimization_flags);
    check(static_cast<bool>(first_vulkan));
    ActivationBuffer first_output;
    check(static_cast<bool>(first_vulkan->forward(input, first_output)));
    for (size_t row = 0; row < first_cpu.rows(); ++row)
    {
        for (uint32_t column = 0;
             column < first_cpu.columns();
             ++column)
        {
            check_near(first_output.row(row)[column],
                       first_cpu.row(row)[column],
                       2e-4f);
        }
    }

    TensorData norm_weight;
    norm_weight.dtype = DType::Float32;
    norm_weight.shape = {128};
    norm_weight.float32_data.resize(128);
    for (uint32_t column = 0; column < 128; ++column)
    {
        norm_weight.float32_data[column] = 0.75f + static_cast<float>(column % 9) * 0.03125f;
    }
    constexpr float norm_epsilon = 1e-6f;
    const ActivationBuffer normalized_cpu = forward_rms_norm(input,
                                                             norm_weight,
                                                             norm_epsilon,
                                                             0.0f);
    const ActivationBuffer norm_chain_cpu = forward_linear(first,
                                                           first_bias,
                                                           normalized_cpu,
                                                           g_test_optimization_flags);
    check(static_cast<bool>(first_vulkan->prepare_rms_norm(norm_weight, norm_epsilon)));
    ActivationBuffer norm_chain_vulkan;
    check(static_cast<bool>(first_vulkan->forward_rms_norm_chain(input,
                                                                 norm_chain_vulkan)));
    for (size_t row = 0; row < norm_chain_cpu.rows(); ++row)
    {
        for (uint32_t column = 0;
             column < norm_chain_cpu.columns();
             ++column)
        {
            check_near(norm_chain_vulkan.row(row)[column],
                       norm_chain_cpu.row(row)[column],
                       2e-3f);
        }
    }

    ActivationBuffer one_row(1, input.columns());
    std::copy_n(input.row(1),
                input.columns(),
                one_row.row(0));
    ActivationBuffer one_row_output;
    check(static_cast<bool>(first_vulkan->forward(one_row, one_row_output)));
    for (uint32_t column = 0;
         column < first_output.columns();
         ++column)
    {
        check_near(one_row_output.row(0)[column],
                   first_output.row(1)[column],
                   1e-6f);
    }

    {
        const uint64_t cooperative_flags = g_test_optimization_flags | OptimizationVulkanBfloat16CoopMatrix;
        const auto cooperative_operator = Bfloat16Linear_vulkan::create(first,
                                                                        &first_bias,
                                                                        device_index,
                                                                        vulkan_runtime,
                                                                        cooperative_flags);
        check(static_cast<bool>(cooperative_operator));
        ActivationBuffer cooperative_input(16, input.columns());
        for (size_t row = 0; row < cooperative_input.rows(); ++row)
        {
            std::copy_n(input.row(row % input.rows()),
                        input.columns(),
                        cooperative_input.row(row));
        }
        const ActivationBuffer cooperative_reference = forward_linear(first,
                                                                      first_bias,
                                                                      cooperative_input,
                                                                      cooperative_flags);
        ActivationBuffer cooperative_output;
        check(static_cast<bool>(cooperative_operator->forward(cooperative_input,
                                                              cooperative_output)));
        for (size_t row = 0; row < cooperative_reference.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < cooperative_reference.columns();
                 ++column)
            {
                check_near(cooperative_output.row(row)[column],
                           cooperative_reference.row(row)[column],
                           3e-3f);
            }
        }
    }

    {
        TensorData tail_matrix;
        tail_matrix.dtype = DType::BFloat16;
        tail_matrix.shape = {64, 192};
        tail_matrix.bfloat16_data.resize(tail_matrix.element_count());
        for (size_t index = 0;
             index < tail_matrix.bfloat16_data.size();
             ++index)
        {
            tail_matrix.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>((index * 11) % 53) - 26)
                                                                 * 0.0017f);
        }
        ActivationBuffer tail_input(3, 192);
        for (size_t row = 0; row < tail_input.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < tail_input.columns();
                 ++column)
            {
                tail_input.row(row)[column] = static_cast<float>(static_cast<int>((row * 17 + column * 5) % 71)
                                                                 - 35)
                                              * 0.0023f;
            }
        }
        const ActivationBuffer tail_reference = forward_linear(tail_matrix,
                                                               tail_input,
                                                               g_test_optimization_flags);
        const auto tail_operator = Bfloat16Linear_vulkan::create(tail_matrix,
                                                                 nullptr,
                                                                 device_index,
                                                                 vulkan_runtime,
                                                                 g_test_optimization_flags);
        check(static_cast<bool>(tail_operator));
        ActivationBuffer tail_output;
        check(static_cast<bool>(tail_operator->forward(tail_input,
                                                       tail_output)));
        for (size_t row = 0; row < tail_reference.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < tail_reference.columns();
                 ++column)
            {
                check_near(tail_output.row(row)[column],
                           tail_reference.row(row)[column],
                           3e-4f);
            }
        }
    }

    const auto fused = Bfloat16Linear_vulkan::create_fused({&first, &second},
                                                           {&first_bias, &second_bias},
                                                           device_index,
                                                           vulkan_runtime,
                                                           g_test_optimization_flags);
    check(static_cast<bool>(fused));
    ActivationBuffer fused_output;
    check(static_cast<bool>(fused->forward(input, fused_output)));
    const ActivationBuffer second_cpu = forward_linear(second,
                                                       second_bias,
                                                       input,
                                                       g_test_optimization_flags);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < 192; ++column)
        {
            check_near(fused_output.row(row)[column],
                       first_cpu.row(row)[column],
                       2e-4f);
        }
        for (uint32_t column = 0; column < 64; ++column)
        {
            check_near(fused_output.row(row)[192 + column],
                       second_cpu.row(row)[column],
                       2e-4f);
        }
    }

    TensorData gate;
    gate.dtype = DType::BFloat16;
    gate.shape = {128, 128};
    gate.bfloat16_data.resize(gate.element_count());
    TensorData up;
    up.dtype = DType::BFloat16;
    up.shape = {128, 128};
    up.bfloat16_data.resize(up.element_count());
    TensorData down;
    down.dtype = DType::BFloat16;
    down.shape = {64, 128};
    down.bfloat16_data.resize(down.element_count());
    TensorData router_gate;
    router_gate.dtype = DType::BFloat16;
    router_gate.shape = {1, 128};
    router_gate.bfloat16_data.resize(router_gate.element_count());
    for (size_t index = 0; index < gate.element_count(); ++index)
    {
        gate.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>((index * 13) % 61) - 30) * 0.0023f);
        up.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>((index * 7) % 47) - 23) * 0.0027f);
    }
    for (size_t index = 0; index < down.element_count(); ++index)
    {
        down.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>((index * 19) % 73) - 36) * 0.0019f);
    }
    for (size_t index = 0; index < router_gate.element_count(); ++index)
    {
        router_gate.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0031f);
    }
    const auto fused_swiglu = Bfloat16Linear_vulkan::create_fused({&gate, &up, &router_gate},
                                                                  {nullptr, nullptr, nullptr},
                                                                  device_index,
                                                                  vulkan_runtime,
                                                                  g_test_optimization_flags);
    const auto down_vulkan = Bfloat16Linear_vulkan::create(down,
                                                           nullptr,
                                                           device_index,
                                                           vulkan_runtime,
                                                           g_test_optimization_flags);
    check(static_cast<bool>(fused_swiglu));
    check(static_cast<bool>(down_vulkan));
    const ActivationBuffer gate_cpu = forward_linear(gate,
                                                     input,
                                                     g_test_optimization_flags);
    const ActivationBuffer up_cpu = forward_linear(up,
                                                   input,
                                                   g_test_optimization_flags);
    const ActivationBuffer router_cpu = forward_linear(router_gate,
                                                       input,
                                                       g_test_optimization_flags);
    ActivationBuffer activated(input.rows(), 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < 128; ++column)
        {
            const float gate_value = gate_cpu.row(row)[column];
            activated.row(row)[column] = gate_value / (1.0f + std::exp(-gate_value))
                                         * up_cpu.row(row)[column];
        }
    }
    ActivationBuffer expected_swiglu = forward_linear(down,
                                                      activated,
                                                      g_test_optimization_flags);
    for (size_t row = 0; row < expected_swiglu.rows(); ++row)
    {
        const float router_scale = 1.0f / (1.0f + std::exp(-router_cpu.row(row)[0]));
        for (uint32_t column = 0; column < expected_swiglu.columns(); ++column)
            expected_swiglu.row(row)[column] *= router_scale;
    }
    ActivationBuffer actual_swiglu;
    const VulkanStatistics before_swiglu = get_vulkan_statistics(vulkan_runtime);
    check(static_cast<bool>(fused_swiglu->forward_swiglu_chain(input,
                                                               *down_vulkan,
                                                               128,
                                                               ExpertActivation::Silu,
                                                               0.0f,
                                                               true,
                                                               actual_swiglu)));
    const VulkanStatistics after_swiglu = get_vulkan_statistics(vulkan_runtime);
    check(static_cast<bool>(after_swiglu.shared_expert_swiglu_fusions
                            == before_swiglu.shared_expert_swiglu_fusions + 1));
    for (size_t row = 0; row < expected_swiglu.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected_swiglu.columns(); ++column)
            check_near(actual_swiglu.row(row)[column], expected_swiglu.row(row)[column], 3e-3f);
    }

    {
        TensorData expert_gate_up;
        expert_gate_up.dtype = DType::BFloat16;
        expert_gate_up.shape = {256, 128};
        expert_gate_up.bfloat16_data.resize(expert_gate_up.element_count());
        std::copy(gate.bfloat16_data.begin(),
                  gate.bfloat16_data.end(),
                  expert_gate_up.bfloat16_data.begin());
        std::copy(up.bfloat16_data.begin(),
                  up.bfloat16_data.end(),
                  expert_gate_up.bfloat16_data.begin() + gate.bfloat16_data.size());

        const auto expected_expert = [&](const ActivationBuffer& expert_input) {
            const ActivationBuffer gate_output = forward_linear(gate,
                                                                expert_input,
                                                                g_test_optimization_flags);
            const ActivationBuffer up_output = forward_linear(up,
                                                              expert_input,
                                                              g_test_optimization_flags);
            ActivationBuffer activated(expert_input.rows(), 128);
            for (size_t row = 0; row < expert_input.rows(); ++row)
            {
                for (uint32_t column = 0; column < 128; ++column)
                {
                    const float gate_value = gate_output.row(row)[column];
                    activated.row(row)[column] = gate_value / (1.0f + std::exp(-gate_value))
                                                 * up_output.row(row)[column];
                }
            }
            return forward_linear(down, activated, g_test_optimization_flags);
        };

        const auto bfloat_backend = create_vulkan_expert_backend(1024 * 1024,
                                                                 device_index,
                                                                 nullptr,
                                                                 vulkan_runtime,
                                                                 g_test_optimization_flags);
        check(static_cast<bool>(bfloat_backend));
        const auto backend_gate_up = std::make_shared<TensorData>(expert_gate_up);
        const auto backend_down = std::make_shared<TensorData>(down);
        bfloat_backend->admit(ExpertDemandRequest{"bfloat16-batch-first", backend_gate_up, nullptr,
                                                  backend_down, nullptr, 0, 0.0f, ExpertActivation::Silu});
        bfloat_backend->admit(ExpertDemandRequest{"bfloat16-batch-second", backend_gate_up, nullptr,
                                                  backend_down, nullptr, 0, 0.0f, ExpertActivation::Silu});
        bfloat_backend->wait_for_background_work();
        check(bfloat_backend->statistics().stores == 2);

        ActivationBuffer first_batch_input(1, input.columns());
        std::copy_n(input.row(0), input.columns(), first_batch_input.row(0));
        ActivationBuffer second_batch_input(2, input.columns());
        for (size_t row = 0; row < second_batch_input.rows(); ++row)
        {
            std::copy_n(input.row(row + 1),
                        input.columns(),
                        second_batch_input.row(row));
        }
        const ActivationBuffer first_batch_expected = expected_expert(first_batch_input);
        const ActivationBuffer second_batch_expected = expected_expert(second_batch_input);
        ActivationBuffer first_batch_output(4, 64);
        ActivationBuffer second_batch_output(4, 64);
        std::fill_n(first_batch_output.row(0),
                    first_batch_output.rows() * first_batch_output.columns(),
                    73.0f);
        std::fill_n(second_batch_output.row(0),
                    second_batch_output.rows() * second_batch_output.columns(),
                    74.0f);
        const std::array<ExpertRequest, 2> bfloat_batch_requests = {{
            {"bfloat16-batch-first", &first_batch_input, &first_batch_output},
            {"bfloat16-batch-second", &second_batch_input, &second_batch_output},
        }};
        const auto bfloat_batch_results = submit_backend_batch(*bfloat_backend, bfloat_batch_requests);
        check(bfloat_batch_results.size() == 2);
        check(bfloat_batch_results[0] == ExpertStatus::Executed);
        check(bfloat_batch_results[1] == ExpertStatus::Executed);
        check(first_batch_output.rows() == first_batch_expected.rows());
        check(second_batch_output.rows() == second_batch_expected.rows());
        for (size_t row = 0; row < first_batch_expected.rows(); ++row)
        {
            for (uint32_t column = 0; column < first_batch_expected.columns(); ++column)
                check_near(first_batch_output.row(row)[column], first_batch_expected.row(row)[column], 3e-3f);
        }
        for (size_t row = 0; row < second_batch_expected.rows(); ++row)
        {
            for (uint32_t column = 0; column < second_batch_expected.columns(); ++column)
                check_near(second_batch_output.row(row)[column], second_batch_expected.row(row)[column], 3e-3f);
        }
    }

#if NCNN_MOE_WITH_VULKAN
    {
        const auto context = VulkanContext::acquire(device_index,
                                                    vulkan_runtime,
                                                    g_test_optimization_flags);
        check(static_cast<bool>(context));
        ncnn::VkMat staging;
        check(static_cast<bool>(prepare_staging_batch(staging,
                                                      3,
                                                      4,
                                                      context->staging_allocator(),
                                                      sizeof(uint16_t))));
        const std::array<uint16_t, 12> staged_values = {
            float_to_bfloat16(1.0f),
            float_to_bfloat16(-2.0f),
            float_to_bfloat16(3.0f),
            float_to_bfloat16(-4.0f),
            float_to_bfloat16(5.0f),
            float_to_bfloat16(-6.0f),
            float_to_bfloat16(7.0f),
            float_to_bfloat16(-8.0f),
            float_to_bfloat16(9.0f),
            float_to_bfloat16(-10.0f),
            float_to_bfloat16(11.0f),
            float_to_bfloat16(-12.0f),
        };
        ncnn::Mat mapped = staging.mapped();
        check(!mapped.empty());
        std::memcpy(mapped.data, staged_values.data(), sizeof(staged_values));
        staging.allocator->flush(staging.data);
        staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

        ActivationBuffer first_input(1, 4, DType::BFloat16);
        ActivationBuffer second_input(2, 4, DType::BFloat16);
        ActivationBuffer first_output(3, 4, DType::BFloat16);
        ActivationBuffer second_output(3, 4, DType::BFloat16);
        std::fill(first_output.mutable_bytes().begin(), first_output.mutable_bytes().end(), std::byte{0xa5});
        std::fill(second_output.mutable_bytes().begin(), second_output.mutable_bytes().end(), std::byte{0x5a});
        std::array<const ActivationBuffer*, 2> inputs = {&first_input, &second_input};
        std::array<ActivationBuffer*, 2> outputs = {&first_output, &second_output};
        check(copy_staging_to_cpu_batches(staging, inputs, outputs, 4));
        check(first_output.rows() == 1 && second_output.rows() == 2);
        check(std::memcmp(first_output.bytes().data(), staged_values.data(), 4 * sizeof(uint16_t)) == 0);
        check(std::memcmp(second_output.bytes().data(),
                          staged_values.data() + 4,
                          8 * sizeof(uint16_t))
              == 0);

        first_output.reset(3, 4, true);
        std::fill(first_output.mutable_bytes().begin(), first_output.mutable_bytes().end(), std::byte{0x3c});
        const std::vector<std::byte> first_guard(first_output.bytes().begin(), first_output.bytes().end());
        std::array<ActivationBuffer*, 2> invalid_later_outputs = {&first_output, nullptr};
        check(!copy_staging_to_cpu_batches(staging, inputs, invalid_later_outputs, 4));
        check(first_output.rows() == 3);
        check(std::equal(first_output.bytes().begin(), first_output.bytes().end(), first_guard.begin()));

        ActivationBuffer duplicate_output(3, 4, DType::BFloat16);
        std::fill(duplicate_output.mutable_bytes().begin(), duplicate_output.mutable_bytes().end(), std::byte{0x4d});
        const std::vector<std::byte> duplicate_guard(duplicate_output.bytes().begin(), duplicate_output.bytes().end());
        std::array<ActivationBuffer*, 2> duplicate_outputs = {&duplicate_output, &duplicate_output};
        check(!copy_staging_to_cpu_batches(staging, inputs, duplicate_outputs, 4));
        check(std::equal(duplicate_output.bytes().begin(), duplicate_output.bytes().end(), duplicate_guard.begin()));

        ActivationBuffer cross_output(3, 4, DType::BFloat16);
        std::fill(second_input.mutable_bytes().begin(), second_input.mutable_bytes().end(), std::byte{0x6e});
        const std::vector<std::byte> cross_guard(second_input.bytes().begin(), second_input.bytes().end());
        std::array<ActivationBuffer*, 2> cross_outputs = {&second_input, &cross_output};
        check(!copy_staging_to_cpu_batches(staging, inputs, cross_outputs, 4));
        check(second_input.rows() == 2);
        check(std::equal(second_input.bytes().begin(), second_input.bytes().end(), cross_guard.begin()));

        ActivationBuffer self_second_output(3, 4, DType::BFloat16);
        ActivationBuffer self_first_input(1, 4, DType::BFloat16);
        ActivationBuffer self_second_input(2, 4, DType::BFloat16);
        std::fill(self_first_input.mutable_bytes().begin(), self_first_input.mutable_bytes().end(), std::byte{0x21});
        std::array<const ActivationBuffer*, 2> self_inputs = {&self_first_input, &self_second_input};
        std::array<ActivationBuffer*, 2> self_outputs = {&self_first_input, &self_second_output};
        check(copy_staging_to_cpu_batches(staging, self_inputs, self_outputs, 4));
        check(self_first_input.rows() == 1);
        check(std::memcmp(self_first_input.bytes().data(), staged_values.data(), 4 * sizeof(uint16_t)) == 0);
    }
#endif
#else
    std::cout << "SKIP BF16 Vulkan operator: NCNN/Vulkan backend is unavailable\n";
#endif
}
void test_ncnn_vulkan_qnk_operator()
{
    constexpr uint32_t columns = 512;
    constexpr size_t rows = 5;
    constexpr size_t input_rows = 3;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    ActivationBuffer input(input_rows, columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 13) % 29) - 14) * 0.03125f;
    }

    for (const DType dtype : dtypes)
    {
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)));
        for (size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<uint8_t>((index * 37u + static_cast<uint32_t>(dtype) * 11u + 5u) & 0xffu);
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
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
                    std::fill_n(encoded + 4, 4, uint8_t{1});
                    std::fill_n(encoded + 8, 4, uint8_t{0});
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
        const ActivationBuffer expected = forward_linear(matrix, input, 0);
        auto vulkan = QnkLinear_vulkan::create(matrix,
                                               nullptr,
                                               automatic_vulkan_device_index,
                                               vulkan_runtime,
                                               g_test_optimization_flags | OptimizationVulkanQnK);
        if (get_gpu_count() == 0)
        {
            check(static_cast<bool>(!vulkan));
            continue;
        }
        check(static_cast<bool>(vulkan));
        check(vulkan->dtype() == dtype);
        check(vulkan->input_columns() == columns);
        check(vulkan->output_columns() == rows);
        ActivationBuffer actual;
        check(static_cast<bool>(vulkan->forward(input, actual)));
        check(actual.rows() == expected.rows());
        check(actual.columns() == expected.columns());
        for (size_t token = 0; token < input_rows; ++token)
        {
            for (size_t row = 0; row < rows; ++row)
                check_near(actual.row(token)[row], expected.row(token)[row], 0.2f);
        }
    }
}

void test_ncnn_vulkan_qnk_expert_operator()
{
    const uint32_t hidden_columns = 2048;
    const uint32_t intermediate_columns = 256;
    const size_t input_rows = 3;
    const uint32_t qnk_columns = hidden_columns;
    const DType dtype = DType::Q4K;
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
                for (uint32_t index = 0; index < 8; ++index)
                    encoded[4 + index] = static_cast<uint8_t>(1u + ((seed + row + block + index) & 1u));
                for (uint32_t index = 0; index < 64; ++index)
                    encoded[16 + index] = static_cast<uint8_t>(0x11u + ((seed + row + block + index) & 0x22u));
            }
        }
        return raw;
    };

    TensorData gate_up;
    gate_up.dtype = dtype;
    gate_up.shape = {intermediate_columns * 2, qnk_columns};
    gate_up.quantized_data = make_q4k(gate_up.shape[0], qnk_columns, 7);
    TensorData down;
    down.dtype = dtype;
    down.shape = {hidden_columns, intermediate_columns};
    down.quantized_data = make_q4k(down.shape[0], intermediate_columns, 19);

    ActivationBuffer input(input_rows, hidden_columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * 17 + column * 5) % 31) - 15) * 0.015625f;
    }

    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    const std::array<ExpertActivation, 3> activations = {
        ExpertActivation::Silu,
        ExpertActivation::DeepSeekSwiGlu,
        ExpertActivation::GptOssSwiGlu,
    };
    for (const ExpertActivation activation : activations)
    {
        const ActivationBuffer gate_up_output = forward_linear(gate_up, input, 0);
        ActivationBuffer activated(input_rows, intermediate_columns);
        for (size_t row = 0; row < input_rows; ++row)
        {
            for (uint32_t column = 0; column < intermediate_columns; ++column)
            {
                const float gate = gate_up_output.row(row)[column * 2];
                const float up = gate_up_output.row(row)[column * 2 + 1];
                if (activation == ExpertActivation::GptOssSwiGlu)
                {
                    const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                    activated.row(row)[column] = silu * (up + 1.0f);
                }
                else
                {
                    const float silu = gate / (1.0f + std::exp(-gate));
                    activated.row(row)[column] = silu * up;
                }
            }
        }
        const ActivationBuffer expected = forward_linear(down, activated, 0);
        auto vulkan = QnkExpert_vulkan::create(gate_up,
                                               nullptr,
                                               down,
                                               nullptr,
                                               0.0f,
                                               automatic_vulkan_device_index,
                                               activation,
                                               vulkan_runtime,
                                               g_test_optimization_flags | OptimizationVulkanQnK);
        if (get_gpu_count() == 0)
        {
            check(static_cast<bool>(!vulkan));
            continue;
        }
        check(static_cast<bool>(vulkan));
        ActivationBuffer actual;
        check(static_cast<bool>(vulkan->forward(input, actual)));
        check(actual.rows() == expected.rows());
        check(actual.columns() == expected.columns());
        for (size_t row = 0; row < actual.rows(); ++row)
        {
            for (uint32_t column = 0; column < actual.columns(); ++column)
                check_near(actual.row(row)[column], expected.row(row)[column], 0.2f);
        }
    }

    TensorData packed_gate_up = gate_up;
    packed_gate_up.qnk_interleave_rows = false;
    const size_t qnk_block_bytes_value = qnk_block_bytes(dtype);
    const uint32_t qnk_block_count = qnk_columns / qnk_block_elements;
    const uint32_t packed_intermediate_rows = intermediate_columns;
    for (uint32_t packed_row = 0; packed_row < packed_intermediate_rows * 2; ++packed_row)
    {
        const uint32_t source_row = packed_row < packed_intermediate_rows
                                        ? packed_row * 2
                                        : (packed_row - packed_intermediate_rows) * 2 + 1;
        for (uint32_t block = 0; block < qnk_block_count; ++block)
        {
            const size_t destination_offset = (static_cast<size_t>(packed_row) * qnk_block_count + block) * qnk_block_bytes_value;
            const size_t source_offset = (static_cast<size_t>(source_row) * qnk_block_count + block) * qnk_block_bytes_value;
            std::memcpy(packed_gate_up.quantized_data.data() + destination_offset,
                        gate_up.quantized_data.data() + source_offset,
                        qnk_block_bytes_value);
        }
    }
    const ActivationBuffer packed_gate_output = forward_linear(packed_gate_up, input, 0);
    ActivationBuffer packed_activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = packed_gate_output.row(row)[column];
            const float up = packed_gate_output.row(row)[intermediate_columns + column];
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            packed_activated.row(row)[column] = silu * (up + 1.0f);
        }
    }
    const ActivationBuffer packed_expected = forward_linear(down, packed_activated, 0);
    auto packed_vulkan = QnkExpert_vulkan::create(packed_gate_up,
                                                  nullptr,
                                                  down,
                                                  nullptr,
                                                  0.0f,
                                                  automatic_vulkan_device_index,
                                                  ExpertActivation::GptOssSwiGlu,
                                                  vulkan_runtime,
                                                  g_test_optimization_flags | OptimizationVulkanQnK);
    if (get_gpu_count() == 0)
    {
        check(static_cast<bool>(!packed_vulkan));
    }
    else
    {
        check(static_cast<bool>(packed_vulkan));
        ActivationBuffer packed_actual;
        check(static_cast<bool>(packed_vulkan->forward(input, packed_actual)));
        check(packed_actual.rows() == packed_expected.rows());
        check(packed_actual.columns() == packed_expected.columns());
        for (size_t row = 0; row < packed_actual.rows(); ++row)
        {
            for (uint32_t column = 0; column < packed_actual.columns(); ++column)
            {
                check_near(packed_actual.row(row)[column], packed_expected.row(row)[column], 0.2f);
            }
        }
    }

    if (get_gpu_count() == 0)
        return;

#if NCNN_MOE_WITH_VULKAN
    TensorData separate_gate;
    separate_gate.dtype = dtype;
    separate_gate.shape = {intermediate_columns, qnk_columns};
    separate_gate.quantized_data = make_q4k(intermediate_columns, qnk_columns, 29);
    TensorData separate_up;
    separate_up.dtype = dtype;
    separate_up.shape = {intermediate_columns, qnk_columns};
    separate_up.quantized_data = make_q4k(intermediate_columns, qnk_columns, 41);

    const ActivationBuffer separate_gate_output = forward_linear(separate_gate, input, 0);
    const ActivationBuffer separate_up_output = forward_linear(separate_up, input, 0);
    ActivationBuffer separate_activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = separate_gate_output.row(row)[column];
            const float silu = gate / (1.0f + std::exp(-gate));
            separate_activated.row(row)[column] = silu * separate_up_output.row(row)[column];
        }
    }
    const ActivationBuffer separate_expected = forward_linear(down, separate_activated, 0);
    const auto check_batch = [&vulkan_runtime](const TensorData& gate, const TensorData* up, const TensorData& down,
                                               const ActivationBuffer& batch_input, const ActivationBuffer& expected, uint64_t flags,
                                               uint64_t dispatches, uint64_t int4_dispatches, float tolerance) {
        const uint64_t size = quant_expert_bytes(gate, up, down, automatic_vulkan_device_index, vulkan_runtime, flags);
        check(size != 0);
        auto backend = create_vulkan_expert_backend(size + 4096, automatic_vulkan_device_index, nullptr, vulkan_runtime, flags);
        check(static_cast<bool>(backend));
        ExpertDemandRequest admission{"batch", std::make_shared<TensorData>(gate), nullptr,
                                      std::make_shared<TensorData>(down), nullptr, 0, 0.0f,
                                      ExpertActivation::Silu};
        if (up)
            admission.up = std::make_shared<TensorData>(*up);
        backend->admit(admission);
        backend->wait_for_background_work();
        check(backend->statistics().stores == 1);
        ActivationBuffer first_input(1, batch_input.columns());
        std::copy_n(batch_input.row(0), batch_input.columns(), first_input.row(0));
        std::array<ActivationBuffer, 2> actual;
        std::array<ExpertRequest, 2> requests = {{{"batch", &first_input, &actual[0]}, {"batch", &batch_input, &actual[1]}}};
        for (uint32_t input_case = 0; input_case < 3; ++input_case)
        {
            requests[0].input = input_case == 2 ? &batch_input : &first_input;
            requests[1].input = input_case == 1 ? &first_input : &batch_input;
            const VulkanStatistics before = get_vulkan_statistics(vulkan_runtime);
            const auto results = submit_backend_batch(*backend, requests);
            check(results.size() == 2 && results[0] == ExpertStatus::Executed && results[1] == ExpertStatus::Executed);
            const VulkanStatistics after = get_vulkan_statistics(vulkan_runtime);
            check(after.dispatches - before.dispatches == dispatches);
            check(after.int4_dispatches - before.int4_dispatches == int4_dispatches);
            check(after.compute_submissions - before.compute_submissions == 1);
            check(after.batch_uploads - before.batch_uploads == 1);
            check(after.batch_downloads - before.batch_downloads == 1);
            for (size_t index = 0; index < actual.size(); ++index)
            {
                check(actual[index].rows() == requests[index].input->rows() && actual[index].columns() == expected.columns());
                for (size_t row = 0; row < actual[index].rows(); ++row)
                    for (uint32_t column = 0; column < actual[index].columns(); ++column)
                        check_near(actual[index].row(row)[column], expected.row(row)[column], tolerance);
            }
        }
        const VulkanStatistics after = get_vulkan_statistics(vulkan_runtime);
        for (auto& request : requests) request.output = nullptr;
        const auto invalid = submit_backend_batch(*backend, requests);
        check(invalid.size() == 2 && invalid[0] != ExpertStatus::Executed && invalid[1] != ExpertStatus::Executed);
        const VulkanStatistics failed = get_vulkan_statistics(vulkan_runtime);
        check(failed.dispatches == after.dispatches && failed.int4_dispatches == after.int4_dispatches);
        check(failed.compute_submissions == after.compute_submissions);
        check(failed.batch_uploads == after.batch_uploads && failed.batch_downloads == after.batch_downloads);
    };
    for (const bool enable_fusion : {true, false})
    {
        uint64_t optimization_flags = g_test_optimization_flags;
        if (enable_fusion)
            optimization_flags |= OptimizationVulkanQnK;
        else
            optimization_flags &= ~OptimizationVulkanQnK;
        const auto context = VulkanContext::acquire(automatic_vulkan_device_index,
                                                    vulkan_runtime,
                                                    optimization_flags);
        check(static_cast<bool>(context));
        const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
        const uint64_t expected_storage = QnkLinear_vulkan::storage_bytes(separate_gate, alignment)
                                          + QnkLinear_vulkan::storage_bytes(separate_up, alignment)
                                          + QnkLinear_vulkan::storage_bytes(down, alignment);
        auto separate_vulkan = QnkExpert_vulkan::create_separate(separate_gate,
                                                                 nullptr,
                                                                 separate_up,
                                                                 nullptr,
                                                                 down,
                                                                 nullptr,
                                                                 0.0f,
                                                                 automatic_vulkan_device_index,
                                                                 ExpertActivation::Silu,
                                                                 vulkan_runtime,
                                                                 optimization_flags);
        check(static_cast<bool>(separate_vulkan));
        check(separate_vulkan->device_bytes() == expected_storage);
        const VulkanStatistics before = get_vulkan_statistics(vulkan_runtime);
        ActivationBuffer separate_actual;
        check(static_cast<bool>(separate_vulkan->forward(input, separate_actual)));
        const VulkanStatistics after = get_vulkan_statistics(vulkan_runtime);
        check(after.dispatches - before.dispatches == (enable_fusion ? 3u : 4u));
        check(separate_actual.rows() == separate_expected.rows());
        check(separate_actual.columns() == separate_expected.columns());
        for (size_t row = 0; row < separate_actual.rows(); ++row)
        {
            for (uint32_t column = 0; column < separate_actual.columns(); ++column)
                check_near(separate_actual.row(row)[column], separate_expected.row(row)[column], 0.2f);
        }
        if (enable_fusion)
            check_batch(separate_gate, &separate_up, down, input, separate_expected, optimization_flags, 6, 0, 0.2f);
    }

    const auto file_backed_qnk = [](const TensorData& matrix, const char* path) {
        TensorData result = matrix;
        result.quantized_data.clear();
        result.mapped_data.reset();
        result.mapped_size = 0;
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = path;
        storage->blocks_size = qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1]);
        result.mxfp4_file_storage = std::move(storage);
        return result;
    };
    const TensorData metadata_gate_up = file_backed_qnk(gate_up, "unused-qnk-gate-up");
    const TensorData metadata_down = file_backed_qnk(down, "unused-qnk-down");
    const auto context = VulkanContext::acquire(automatic_vulkan_device_index,
                                                vulkan_runtime,
                                                g_test_optimization_flags | OptimizationVulkanQnK);
    check(static_cast<bool>(context));
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const auto make_int4 = [](uint32_t rows, uint32_t columns) {
        TensorData result;
        result.dtype = DType::Int4;
        result.shape = {rows, columns};
        result.quantized_data.assign(static_cast<size_t>(rows) * ((columns + 1) / 2), 0x99);
        result.quantization_group_size = 3;
        result.quantization_scales.assign(static_cast<size_t>(rows) * ((columns + 2) / 3), 1.0f / 16.0f);
        result.quantization_zero_points.assign(result.quantization_scales.size(), 8);
        return result;
    };
    const TensorData int4_gate = make_int4(3, 5);
    const TensorData int4_gate_up = make_int4(6, 5);
    const TensorData int4_down = make_int4(5, 3);
    ActivationBuffer int4_input(3, 5);
    for (size_t row = 0; row < int4_input.rows(); ++row)
        for (uint32_t column = 0; column < int4_input.columns(); ++column)
            int4_input.row(row)[column] = static_cast<float>(row * 5 + column + 1) * 0.125f;
    ActivationBuffer int4_activated = forward_linear(int4_gate, int4_input, 0);
    for (size_t row = 0; row < int4_activated.rows(); ++row)
        for (uint32_t column = 0; column < int4_activated.columns(); ++column)
        {
            float& value = int4_activated.row(row)[column];
            value = value * value / (1.0f + std::exp(-value));
        }
    const ActivationBuffer int4_expected = forward_linear(int4_down, int4_activated, 0);
    const uint64_t int4_flags = g_test_optimization_flags | OptimizationVulkanQnK;
    check_batch(int4_gate_up, nullptr, int4_down, int4_input, int4_expected, int4_flags, 6, 4, 1e-5f);
    check_batch(int4_gate, &int4_gate, int4_down, int4_input, int4_expected, int4_flags, 8, 6, 1e-5f);
    if (alignment <= 4)
        std::cout << "INT4 batch row offsets are aligned on this device; misaligned-input branch is not exercised\n";
    check(QnkLinear_vulkan::storage_bytes(metadata_gate_up, alignment)
          == QnkLinear_vulkan::storage_bytes(gate_up, alignment));
    check(QnkLinear_vulkan::storage_bytes(metadata_down, alignment)
          == QnkLinear_vulkan::storage_bytes(down, alignment));
    const uint64_t metadata_size = quant_expert_bytes(metadata_gate_up,
                                                      nullptr,
                                                      metadata_down,
                                                      automatic_vulkan_device_index,
                                                      vulkan_runtime,
                                                      g_test_optimization_flags | OptimizationVulkanQnK);
    check(metadata_size == quant_expert_bytes(gate_up, nullptr, down, automatic_vulkan_device_index, vulkan_runtime, g_test_optimization_flags | OptimizationVulkanQnK));
    check(static_cast<bool>(!QnkExpert_vulkan::create(metadata_gate_up,
                                                      nullptr,
                                                      metadata_down,
                                                      nullptr,
                                                      0.0f,
                                                      automatic_vulkan_device_index,
                                                      ExpertActivation::GptOssSwiGlu,
                                                      vulkan_runtime,
                                                      g_test_optimization_flags | OptimizationVulkanQnK)));
    TensorData invalid_metadata = metadata_gate_up;
    auto invalid_storage = std::make_shared<MxFp4FileStorage>(*metadata_gate_up.mxfp4_file_storage);
    --invalid_storage->blocks_size;
    invalid_metadata.mxfp4_file_storage = std::move(invalid_storage);
    check(QnkLinear_vulkan::storage_bytes(invalid_metadata, alignment) == 0);
    check(quant_expert_bytes(invalid_metadata,
                             nullptr,
                             metadata_down,
                             automatic_vulkan_device_index,
                             vulkan_runtime,
                             g_test_optimization_flags | OptimizationVulkanQnK)
          == 0);

    const auto metadata_backend = create_vulkan_expert_backend(metadata_size + 4096,
                                                               automatic_vulkan_device_index,
                                                               nullptr,
                                                               vulkan_runtime,
                                                               g_test_optimization_flags | OptimizationVulkanQnK);
    check(static_cast<bool>(metadata_backend));
    const ExpertBackendStatistics before_metadata_admission = metadata_backend->statistics();
    metadata_backend->admit(ExpertDemandRequest{"qnk-file-metadata",
                                                std::make_shared<TensorData>(metadata_gate_up), nullptr,
                                                std::make_shared<TensorData>(metadata_down), nullptr,
                                                0, 0.0f, ExpertActivation::GptOssSwiGlu});
    const ExpertBackendStatistics after_metadata_admission = metadata_backend->statistics();
    check(after_metadata_admission.admissions == before_metadata_admission.admissions);
    check(after_metadata_admission.stores == before_metadata_admission.stores);
    check(after_metadata_admission.pending_size == before_metadata_admission.pending_size);
    check(after_metadata_admission.bytes_uploaded == before_metadata_admission.bytes_uploaded);
    metadata_backend->wait_for_background_work();
    const ExpertBackendStatistics after_metadata_wait = metadata_backend->statistics();
    check(after_metadata_wait.admissions == before_metadata_admission.admissions);
    check(after_metadata_wait.stores == before_metadata_admission.stores);
    check(after_metadata_wait.pending_size == before_metadata_admission.pending_size);
    check(after_metadata_wait.bytes_uploaded == before_metadata_admission.bytes_uploaded);
#endif

    ActivationBuffer gate_up_output = forward_linear(gate_up, input, 0);
    ActivationBuffer activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = gate_up_output.row(row)[column * 2];
            const float up = gate_up_output.row(row)[column * 2 + 1];
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            activated.row(row)[column] = silu * (up + 1.0f);
        }
    }
    const ActivationBuffer expected = forward_linear(down, activated, 0);
    const uint64_t qnk_pair_bytes = packed_vulkan->device_bytes();
    const uint64_t backend_flags = g_test_optimization_flags
                                   | OptimizationVulkanQnK;
    const auto backend = create_vulkan_expert_backend(qnk_pair_bytes * 2 + 4096,
                                                      automatic_vulkan_device_index,
                                                      nullptr,
                                                      vulkan_runtime,
                                                      backend_flags);
    check(static_cast<bool>(backend));
    const auto backend_gate_up = std::make_shared<TensorData>(gate_up);
    const auto backend_down = std::make_shared<TensorData>(down);
    backend->admit(ExpertDemandRequest{"qnk-expert", backend_gate_up, nullptr, backend_down, nullptr,
                                       0, 0.0f, ExpertActivation::GptOssSwiGlu});
    backend->wait_for_background_work();
    check(backend->statistics().stores == 1);
    ActivationBuffer actual;
    const ExpertRequest request{"qnk-expert", &input, &actual};
    const auto results = submit_backend_batch(*backend, std::span<const ExpertRequest>(&request, 1));
    check(results.size() == 1);
    check(results.front() == ExpertStatus::Executed);
    check(actual.rows() == expected.rows());
    check(actual.columns() == expected.columns());
    for (size_t row = 0; row < actual.rows(); ++row)
    {
        for (uint32_t column = 0; column < actual.columns(); ++column)
            check_near(actual.row(row)[column], expected.row(row)[column], 0.2f);
    }

    auto backend_gate_up_second = std::make_shared<TensorData>(gate_up);
    auto backend_down_second = std::make_shared<TensorData>(down);
    backend_gate_up_second->quantized_data = make_q4k(gate_up.shape[0], qnk_columns, 8);
    backend_down_second->quantized_data = make_q4k(down.shape[0], intermediate_columns, 20);
    const ActivationBuffer second_gate_up_output = forward_linear(*backend_gate_up_second, input, 0);
    ActivationBuffer second_activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = second_gate_up_output.row(row)[column * 2];
            const float up = second_gate_up_output.row(row)[column * 2 + 1];
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            second_activated.row(row)[column] = silu * (up + 1.0f);
        }
    }
    const ActivationBuffer second_expected = forward_linear(*backend_down_second, second_activated, 0);

    const std::vector<uint8_t> packed_second_gate_up = std::move(backend_gate_up_second->quantized_data);
    const uint32_t second_intermediate_columns = backend_gate_up_second->shape[0] / 2;
    const size_t second_row_bytes = static_cast<size_t>(qnk_block_count) * qnk_block_bytes_value;
    backend_gate_up_second->shape[0] = second_intermediate_columns;
    backend_gate_up_second->quantized_data.resize(static_cast<size_t>(second_intermediate_columns) * second_row_bytes);
    auto backend_up_second = std::make_shared<TensorData>();
    backend_up_second->dtype = backend_gate_up_second->dtype;
    backend_up_second->shape = backend_gate_up_second->shape;
    backend_up_second->quantized_data.resize(static_cast<size_t>(second_intermediate_columns) * second_row_bytes);
    for (uint32_t row = 0; row < second_intermediate_columns; ++row)
    {
        std::memcpy(backend_gate_up_second->quantized_data.data() + static_cast<size_t>(row) * second_row_bytes,
                    packed_second_gate_up.data() + static_cast<size_t>(row * 2) * second_row_bytes,
                    second_row_bytes);
        std::memcpy(backend_up_second->quantized_data.data() + static_cast<size_t>(row) * second_row_bytes,
                    packed_second_gate_up.data() + static_cast<size_t>(row * 2 + 1) * second_row_bytes,
                    second_row_bytes);
    }

    auto direct_second = QnkExpert_vulkan::create_separate(*backend_gate_up_second,
                                                           nullptr,
                                                           *backend_up_second,
                                                           nullptr,
                                                           *backend_down_second,
                                                           nullptr,
                                                           0.0f,
                                                           automatic_vulkan_device_index,
                                                           ExpertActivation::GptOssSwiGlu,
                                                           vulkan_runtime,
                                                           g_test_optimization_flags | OptimizationVulkanQnK);
    ActivationBuffer direct_second_output;
    check(static_cast<bool>(direct_second));
    check(static_cast<bool>(direct_second->forward(input, direct_second_output)));
    ExpertDemandRequest second_admission{"qnk-expert-second", backend_gate_up_second, nullptr,
                                         backend_down_second, nullptr, 0, 0.0f,
                                         ExpertActivation::GptOssSwiGlu};
    second_admission.up = backend_up_second;
    backend->admit(second_admission);
    backend->wait_for_background_work();
    check(backend->statistics().stores == 2);

    ActivationBuffer second_single_output;
    const ExpertRequest second_request{"qnk-expert-second", &input, &second_single_output};
    const auto second_results = submit_backend_batch(*backend, std::span<const ExpertRequest>(&second_request, 1));
    check(second_results.size() == 1);
    check(second_results.front() == ExpertStatus::Executed);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            check_near(second_single_output.row(row)[column], second_expected.row(row)[column], 0.2f);
        }
    }

    ActivationBuffer batch_output_first;
    ActivationBuffer batch_output_second;
    const std::array<ExpertRequest, 2> batch_requests = {{
        {"qnk-expert", &input, &batch_output_first},
        {"qnk-expert-second", &input, &batch_output_second},
    }};
    const auto batch_results = submit_backend_batch(*backend, batch_requests);
    check(batch_results.size() == batch_requests.size());
    check(batch_results[0] == ExpertStatus::Executed);
    check(batch_results[1] == ExpertStatus::Executed);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            check_near(batch_output_first.row(row)[column], expected.row(row)[column], 0.2f);
            check_near(batch_output_second.row(row)[column], second_expected.row(row)[column], 0.2f);
        }
    }

    {
        ActivationBuffer uneven_input_first(1, hidden_columns);
        std::copy_n(input.row(0), hidden_columns, uneven_input_first.row(0));
        ActivationBuffer uneven_input_second(2, hidden_columns);
        for (size_t row = 0; row < uneven_input_second.rows(); ++row)
        {
            std::copy_n(input.row(row + 1),
                        hidden_columns,
                        uneven_input_second.row(row));
        }
        ActivationBuffer uneven_output_first(4, expected.columns());
        ActivationBuffer uneven_output_second(4, expected.columns());
        std::fill_n(uneven_output_first.row(0),
                    uneven_output_first.rows() * uneven_output_first.columns(),
                    81.0f);
        std::fill_n(uneven_output_second.row(0),
                    uneven_output_second.rows() * uneven_output_second.columns(),
                    82.0f);
        const std::array<ExpertRequest, 2> uneven_requests = {{
            {"qnk-expert", &uneven_input_first, &uneven_output_first},
            {"qnk-expert-second", &uneven_input_second, &uneven_output_second},
        }};
        const auto uneven_results = submit_backend_batch(*backend, uneven_requests);
        check(uneven_results.size() == uneven_requests.size());
        check(uneven_results[0] == ExpertStatus::Executed);
        check(uneven_results[1] == ExpertStatus::Executed);
        check(uneven_output_first.rows() == 1);
        check(uneven_output_second.rows() == 2);
        for (uint32_t column = 0; column < expected.columns(); ++column)
            check_near(uneven_output_first.row(0)[column], expected.row(0)[column], 0.2f);
        for (size_t row = 0; row < uneven_output_second.rows(); ++row)
        {
            for (uint32_t column = 0; column < expected.columns(); ++column)
            {
                check_near(uneven_output_second.row(row)[column],
                           second_expected.row(row + 1)[column],
                           0.2f);
            }
        }
    }

    ActivationBuffer aggregated_output(2, expected.columns());
    std::vector<ExpertRoute> first_routes = {
        {0, 0, 0.15f},
        {0, 1, 0.25f},
        {1, 2, 0.10f},
    };
    std::vector<ExpertRoute> second_routes = {
        {0, 0, 0.20f},
        {1, 1, 0.15f},
        {1, 2, 0.15f},
    };
    ActivationBuffer aggregation_output_first(1, 1);
    ActivationBuffer aggregation_output_second(1, 1);
    aggregation_output_first.row(0)[0] = 71.0f;
    aggregation_output_second.row(0)[0] = 72.0f;
    std::fill_n(aggregated_output.row(0), aggregated_output.rows() * aggregated_output.columns(), 123.0f);
    uint8_t first_completed = 0;
    uint8_t second_completed = 0;
    std::array<ExpertRequest, 2> aggregation_requests = {{
        {"qnk-expert", &input, &aggregation_output_first},
        {"qnk-expert-second", &input, &aggregation_output_second},
    }};
    aggregation_requests[0].route_aggregation = {
        &aggregated_output,
        first_routes,
        2,
        &first_completed,
        true,
    };
    aggregation_requests[1].route_aggregation = {
        &aggregated_output,
        second_routes,
        2,
        &second_completed,
        true,
    };
    const auto check_unpublished = [&] {
        check(first_completed == 0);
        check(second_completed == 0);
        check(aggregation_output_first.rows() == 1 && aggregation_output_first.columns() == 1);
        check(aggregation_output_second.rows() == 1 && aggregation_output_second.columns() == 1);
        check(aggregation_output_first.row(0)[0] == 71.0f);
        check(aggregation_output_second.row(0)[0] == 72.0f);
        check(aggregated_output.rows() == 2 && aggregated_output.columns() == expected.columns());
        check(std::all_of(aggregated_output.values().begin(), aggregated_output.values().end(), [](float value) { return value == 123.0f; }));
    };

    // Aborting before or after wait must leave every caller buffer untouched.
    for (bool abort_before_wait : {false, true})
    {
        auto submission = backend->submit_batch(aggregation_requests);
        check(static_cast<bool>(submission));
        if (abort_before_wait)
            submission->abort();
        const auto results = submission->wait();
        check(results.size() == aggregation_requests.size());
        check(results[0] == ExpertStatus::Executed);
        check(results[1] == ExpertStatus::Executed);
        check_unpublished();
        submission->abort();
        submission->abort();
        check(!submission->commit());
        check_unpublished();
    }

    // Conflicting aggregate destinations must reject the entire publication.
    {
        ActivationBuffer other_aggregate(2, expected.columns());
        std::fill_n(other_aggregate.row(0), other_aggregate.rows() * other_aggregate.columns(), 456.0f);
        auto invalid_requests = aggregation_requests;
        invalid_requests[1].route_aggregation.output = &other_aggregate;
        auto submission = backend->submit_batch(invalid_requests);
        check(static_cast<bool>(submission));
        const auto results = submission->wait();
        check(results.size() == invalid_requests.size());
        check(results[0] == ExpertStatus::Executed);
        check(results[1] == ExpertStatus::Executed);
        check_unpublished();
        check(!submission->commit());
        check_unpublished();
        check(other_aggregate.rows() == 2 && other_aggregate.columns() == expected.columns());
        check(std::all_of(other_aggregate.values().begin(), other_aggregate.values().end(), [](float value) { return value == 456.0f; }));
        submission->abort();
        submission->abort();
        check(!submission->commit());
        check_unpublished();
    }

    auto aggregation_submission = backend->submit_batch(aggregation_requests);
    check(static_cast<bool>(aggregation_submission));
    const auto aggregation_results = aggregation_submission->wait();
    check_unpublished();
    check(aggregation_submission->commit());
    check(aggregation_results.size() == aggregation_requests.size());
    check(aggregation_results[0] == ExpertStatus::Executed);
    check(aggregation_results[1] == ExpertStatus::Executed);
    check(first_completed == 1);
    check(second_completed == 1);
    for (size_t token = 0; token < aggregated_output.rows(); ++token)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            float expected_aggregation = 0.0f;
            for (size_t row = 0; row < input.rows(); ++row)
            {
                if (first_routes[row].token_index == token)
                    expected_aggregation += first_routes[row].weight * expected.row(row)[column];
                if (second_routes[row].token_index == token)
                    expected_aggregation += second_routes[row].weight * second_expected.row(row)[column];
            }
            check_near(aggregated_output.row(token)[column], expected_aggregation, 0.2f);
        }
    }
    check(backend->statistics().route_aggregation_batches >= 1);

    const uint64_t aggregation_batches = backend->statistics().route_aggregation_batches;
    std::fill_n(aggregated_output.row(0),
                aggregated_output.rows() * aggregated_output.columns(),
                123.0f);
    ActivationBuffer partial_output_first;
    ActivationBuffer partial_output_second;
    ActivationBuffer missing_output(input.rows(), expected.columns());
    std::fill_n(missing_output.row(0),
                missing_output.rows() * missing_output.columns(),
                321.0f);
    uint8_t partial_first_completed = 0;
    uint8_t partial_second_completed = 0;
    uint8_t missing_completed = 0;
    std::array<ExpertRequest, 3> partial_requests = {{
        {"qnk-expert", &input, &partial_output_first},
        {"qnk-expert-second", &input, &partial_output_second},
        {"qnk-expert-missing", &input, &missing_output},
    }};
    partial_requests[0].route_aggregation = {
        &aggregated_output,
        first_routes,
        2,
        &partial_first_completed,
        false,
    };
    partial_requests[1].route_aggregation = {
        &aggregated_output,
        second_routes,
        2,
        &partial_second_completed,
        false,
    };
    partial_requests[2].route_aggregation = {
        &aggregated_output,
        first_routes,
        2,
        &missing_completed,
        true,
    };
    auto partial_submission = backend->submit_batch(partial_requests);
    check(static_cast<bool>(partial_submission));
    const auto partial_results = partial_submission->wait();
    check(partial_submission->commit());
    check(partial_results.size() == partial_requests.size());
    check(partial_results[0] == ExpertStatus::Executed);
    check(partial_results[1] == ExpertStatus::Executed);
    check(partial_results[2] == ExpertStatus::NotResident);
    check(partial_first_completed == 0);
    check(partial_second_completed == 0);
    check(missing_completed == 0);
    check(aggregated_output.row(0)[0] == 123.0f);
    check(missing_output.row(0)[0] == 321.0f);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            check_near(partial_output_first.row(row)[column], expected.row(row)[column], 0.2f);
            check_near(partial_output_second.row(row)[column], second_expected.row(row)[column], 0.2f);
        }
    }
    check(backend->statistics().route_aggregation_batches
          == aggregation_batches);
}

} // namespace moe
} // namespace ncnn
