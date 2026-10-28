#include "backends/ncnn/qsaattention_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "graph/compiledoperator.h"
#include "kernels/attention.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void qsa_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData qsa_weight(uint32_t rows, uint32_t columns, uint32_t seed, bool quantized)
{
    TensorData tensor;
    tensor.shape = {rows, columns};
    if (quantized)
    {
        tensor.dtype = DType::Q8_0;
        tensor.quantized_data.resize(static_cast<size_t>(rows) * (columns / 32) * 34);
        for (uint32_t row = 0; row < rows; ++row)
            for (uint32_t block = 0; block < columns / 32; ++block)
            {
                uint8_t* output = tensor.quantized_data.data() + (static_cast<size_t>(row) * (columns / 32) + block) * 34;
                output[0] = 0;
                output[1] = 0x18; // FP16 2^-9, native 34-byte Q8_0 block.
                for (uint32_t lane = 0; lane < 32; ++lane)
                {
                    const uint32_t column = block * 32 + lane;
                    const int sample = static_cast<int>((row * 17 + column * 23 + seed * 19 + (row ^ column) * 3) % 43) - 21;
                    output[2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(sample));
                }
            }
    }
    else
    {
        tensor.dtype = DType::BFloat16;
        tensor.bfloat16_data.resize(static_cast<size_t>(rows) * columns);
        for (uint32_t row = 0; row < rows; ++row)
            for (uint32_t column = 0; column < columns; ++column)
            {
                const int sample = static_cast<int>((row * 11 + column * 19 + seed * 7 + (row ^ column)) % 37) - 18;
                tensor.bfloat16_data[static_cast<size_t>(row) * columns + column] = float_to_bfloat16(static_cast<float>(sample) * 0.001953125f);
            }
    }
    return tensor;
}

static TensorData qsa_norm(uint32_t columns, uint32_t seed)
{
    TensorData tensor;
    tensor.dtype = DType::Float32;
    tensor.shape = {columns};
    tensor.float32_data.resize(columns);
    for (uint32_t column = 0; column < columns; ++column)
        tensor.float32_data[column] = 0.9f + static_cast<float>((column * 7 + seed) % 11) * 0.025f;
    return tensor;
}

struct QsaFixture
{
    WeightStore weights;
    AttentionBlockPlan plan;
    uint32_t hidden = 64;

    QsaFixture(uint32_t head_dimension, DType cache_dtype, bool quantized, bool tied_scores, bool zero_norm_no_input_norm = false)
    {
        if (zero_norm_no_input_norm)
            hidden = 256;
        plan.kind = AttentionKind::Standard;
        plan.flags = AttentionBlockQsa | AttentionBlockQueryKeyNorm | AttentionBlockOutputGate | AttentionBlockExternalResidual;
        plan.head_count = 4;
        plan.kv_head_count = 2;
        plan.head_dimension = head_dimension;
        plan.value_head_dimension = head_dimension;
        plan.rope_head_dimension = 64;
        plan.index_head_count = zero_norm_no_input_norm ? 2 : 4;
        plan.index_head_dimension = zero_norm_no_input_norm ? 64 : 128;
        plan.compression_ratio = 4;
        plan.index_top_k = 2; // Forces sparse selection at a short test prefix.
        plan.initial_context_length = 2048;
        plan.kv_cache_dtype = cache_dtype;
        plan.norm_weight_offset = zero_norm_no_input_norm ? 1.0f : 0.125f;
        const auto add = [&](TensorData tensor) {
            auto result = weights.add("qsa_" + std::to_string(weights.size()), std::move(tensor));
            qsa_check(static_cast<bool>(result), "add QSA fixture tensor");
            return result.value();
        };
        const auto norm = [&](uint32_t columns, uint32_t seed) {
            if (!zero_norm_no_input_norm)
                return qsa_norm(columns, seed);
            TensorData tensor;
            tensor.dtype = DType::BFloat16;
            tensor.shape = {columns};
            tensor.bfloat16_data.assign(columns, 0);
            return tensor;
        };
        if (!zero_norm_no_input_norm)
            plan.pre_attention_norm_weight = add(qsa_norm(hidden, 1));
        plan.query_weight = add(qsa_weight(plan.head_count * head_dimension, hidden, 2, quantized));
        plan.key_weight = add(qsa_weight(plan.kv_head_count * head_dimension, hidden, 3, quantized));
        plan.value_weight = add(qsa_weight(plan.kv_head_count * head_dimension, hidden, 4, quantized));
        plan.output_gate_weight = add(qsa_weight(plan.head_count * head_dimension, hidden, 5, quantized));
        plan.output_weight = add(qsa_weight(hidden, plan.head_count * head_dimension, 6, quantized));
        TensorData index = qsa_weight((plan.index_head_count + 1) * plan.index_head_dimension, hidden, 7, quantized);
        if (tied_scores)
        {
            std::fill(index.bfloat16_data.begin(), index.bfloat16_data.end(), 0);
            if (quantized)
                for (size_t offset = 0; offset < index.quantized_data.size(); offset += 34)
                    std::fill_n(index.quantized_data.data() + offset + 2, 32, 0);
        }
        plan.qsa_query_key_weight = add(std::move(index));
        plan.query_norm_weight = add(norm(head_dimension, 8));
        plan.key_norm_weight = add(norm(head_dimension, 9));
        plan.qsa_query_norm_weight = add(norm(plan.index_head_dimension, 10));
        plan.qsa_key_norm_weight = add(norm(plan.index_head_dimension, 11));
    }
};

static ActivationBuffer qsa_input(uint32_t rows, uint32_t hidden, uint32_t first)
{
    ActivationBuffer input(rows, hidden);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < hidden; ++column)
        {
            const float phase = static_cast<float>((first + row) * 13 + column * 5);
            input.row(row)[column] = std::sin(phase * 0.071f) + 0.3f * std::cos(phase * 0.113f);
        }
    return input;
}

static void qsa_parity(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label)
{
    qsa_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + " shape");
    float largest = 0.0f;
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        largest = std::max(largest, std::abs(expected.values()[index]));
        const float tolerance = 0.0025f + std::abs(expected.values()[index]) * 0.004f;
        qsa_check(std::isfinite(actual.values()[index])
                      && std::abs(actual.values()[index] - expected.values()[index]) <= tolerance,
                  label + " element " + std::to_string(index) + " actual=" + std::to_string(actual.values()[index])
                      + " expected=" + std::to_string(expected.values()[index]));
    }
    qsa_check(largest > 0.0001f, label + " nontrivial oracle");
}

#if NCNN_MOE_WITH_VULKAN
static bool qsa_forward(const std::shared_ptr<QsaAttention_vulkan>& op, LayerCache& cache,
                        uint64_t position, const ActivationBuffer& input, ActivationBuffer& output,
                        bool abandon = false)
{
    ncnn::VkMat output_probe;
    {
        const auto& context = op->vulkan_context();
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input_staging;
        ncnn::VkMat input_gpu;
        ncnn::VkMat output_gpu;
        ncnn::VkMat output_staging;
        QsaAttentionWork_vulkan work;
        if (!fill_staging_upload(input, input_staging, context->staging_allocator())
            || !record_mapped_activation_upload(input_staging, input_gpu, command, context->device(), op->option(),
                                                DType::Float32)
            || !op->record_forward(input_gpu, cache, position, output_gpu, command, work))
            return false;
        if (abandon)
        {
            output_probe = output_gpu;
        }
        else
        {
            if (!prepare_staging_batch(output_staging, input.rows(), input.columns(), context->staging_allocator())
                || !record_prepared_activation_staging_download(output_gpu, input.rows(), input.columns(), output_staging,
                                                                command, context->device(), op->option(), DType::Float32)
                || submit_compute_and_wait(command, context->device()) != 0)
                return false;
            ActivationBuffer completed(input.rows(), input.columns());
            if (!copy_staging_to_cpu_batch(output_staging, completed) || !work.commit() || work.commit())
                return false;
            output_probe = output_gpu;
            output.swap(completed);
            // QSA record itself does not upload, download or submit activations.
            ++context->runtime_state().compute_submissions;
            ++context->runtime_state().batch_uploads;
            ++context->runtime_state().batch_downloads;
        }
    }
    qsa_check(output_probe.refcount && *output_probe.refcount == 1,
              "discarded or completed QSA work releases recorded output bindings");
    return true;
}

static void qsa_download(const std::shared_ptr<QsaAttention_vulkan>& op, const ncnn::VkMat& input,
                         uint32_t rows, uint32_t columns, ActivationBuffer& output)
{
    const auto& context = op->vulkan_context();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute command(context->device());
    ncnn::VkMat staging;
    output.reset(rows, columns, false);
    ncnn::VkMat visible = input;
    visible.h = static_cast<int>(rows);
    visible.cstep = static_cast<size_t>(columns) * rows;
#if NCNN_BATCH
    visible.nstep = visible.cstep;
#endif
    qsa_check(prepare_staging_batch(staging, rows, columns, context->staging_allocator())
                  && record_prepared_activation_staging_download(visible, rows, columns, staging, command,
                                                                 context->device(), op->option(), DType::Float32)
                  && submit_compute_and_wait(command, context->device()) == 0
                  && copy_staging_to_cpu_batch(staging, output),
              "download QSA cache oracle probe");
}

static void qsa_case(uint32_t device, uint32_t head_dimension, DType dtype, bool quantized, bool tied_scores,
                     bool zero_norm_no_input_norm = false)
{
    QsaFixture fixture(head_dimension, dtype, quantized, tied_scores, zero_norm_no_input_norm);
    WeightStore reference_weights = fixture.weights;
    auto runtime = create_vulkan_runtime();
    auto op = QsaAttention_vulkan::create(fixture.weights, fixture.plan, device, runtime, OptimizationDefaultFlags);
    qsa_check(static_cast<bool>(op), "create native QSA operator");
    // Reference weights have independent owned vectors; execution must survive
    // destruction of every matrix payload in the GPU fixture.
    for (TensorHandle handle = 0; handle < fixture.weights.size(); ++handle)
    {
        auto& tensor = fixture.weights.at_mutable(handle);
        tensor.quantized_data.clear();
        tensor.bfloat16_data.clear();
        tensor.float32_data.clear();
    }
    CompiledOperatorTable cpu_operators;
    AttentionScratch scratch;
    LayerCache reference;
    LayerCache actual;
    LayerCache peer_reference;
    LayerCache peer_actual;
    const uint64_t start_position = 7;
    const uint64_t peer_start_position = 101;
    uint32_t consumed = 0;
    uint32_t peer_consumed = 0;
    uint64_t calls = 0;
    AttentionScratch peer_scratch;
    const auto advance_peer = [&](uint32_t rows) {
        const auto peer_input = qsa_input(rows, fixture.hidden, 37 + peer_consumed);
        ActivationBuffer peer_expected;
        ActivationBuffer peer_output;
        auto peer_cpu = forward_attention(reference_weights, cpu_operators, fixture.plan, ExecutionBackend::Cpu,
                                          peer_start_position + peer_consumed, peer_reference, peer_scratch,
                                          peer_input, peer_expected, OptimizationDefaultFlags);
        qsa_check(static_cast<bool>(peer_cpu), "interleaved CPU QSA oracle execution");
        qsa_check(qsa_forward(op, peer_actual, peer_start_position + peer_consumed, peer_input, peer_output),
                  "interleaved GPU QSA cache chunk");
        ++calls;
        qsa_parity(peer_output, peer_expected, "interleaved QSA chunk " + std::to_string(peer_consumed));
        peer_consumed += rows;
        qsa_check(peer_actual.token_count == peer_consumed && peer_actual.start_position == peer_start_position
                      && peer_actual.capacity_tokens >= peer_consumed && peer_actual.qsa_device_state
                      && peer_actual.qsa_device_state != actual.qsa_device_state
                      && peer_actual.device_allocated_size == peer_actual.qsa_device_state->allocated_bytes(),
                  "interleaved QSA Session keeps an independent device cache");
        const auto& peer_state = *peer_actual.qsa_device_state;
        qsa_check(peer_state.keys.refcount && *peer_state.keys.refcount == 1
                      && peer_state.values.refcount && *peer_state.values.refcount == 1
                      && peer_state.index_raw.refcount && *peer_state.index_raw.refcount == 1
                      && peer_state.pooled_keys.refcount && *peer_state.pooled_keys.refcount == 1,
                  "interleaved QSA cache has no retained command bindings");
    };
    const auto before = get_vulkan_statistics(runtime);
    for (uint32_t rows : {3u, 1u, 7u, 2u, 8u, 1u})
    {
        const auto input = qsa_input(rows, fixture.hidden, consumed);
        ActivationBuffer expected;
        ActivationBuffer output;
        auto cpu = forward_attention(reference_weights, cpu_operators, fixture.plan, ExecutionBackend::Cpu,
                                     start_position + consumed, reference, scratch, input, expected, OptimizationDefaultFlags);
        qsa_check(static_cast<bool>(cpu), "CPU QSA oracle execution");
        {
            const auto old_state = actual.qsa_device_state;
            const uint64_t old_bytes = actual.device_allocated_size;
            qsa_check(qsa_forward(op, actual, start_position + consumed, input, output, true), "abandon recorded QSA");
            qsa_check(actual.qsa_device_state == old_state && actual.token_count == consumed
                          && actual.device_allocated_size == old_bytes,
                      "abandoned work leaves authoritative state unchanged");
        }
        qsa_check(qsa_forward(op, actual, start_position + consumed, input, output), "GPU QSA chunk");
        ++calls;
        qsa_parity(output, expected, "QSA chunk " + std::to_string(consumed));
        consumed += rows;
        qsa_check(actual.token_count == consumed && actual.start_position == start_position
                      && actual.capacity_tokens >= consumed && actual.qsa_device_state
                      && actual.device_allocated_size == actual.qsa_device_state->allocated_bytes(),
                  "committed QSA cache metadata and memory");
        qsa_check(actual.keys.empty() && actual.bfloat16_keys.empty() && actual.qsa_block_keys.empty()
                      && actual.qsa_index_key_tail.empty(),
                  "device QSA has no CPU cache shadow");
        const auto& state = *actual.qsa_device_state;
        qsa_check(state.keys.refcount && *state.keys.refcount == 1
                      && state.values.refcount && *state.values.refcount == 1
                      && state.index_raw.refcount && *state.index_raw.refcount == 1
                      && state.pooled_keys.refcount && *state.pooled_keys.refcount == 1,
                  "published QSA cache has no retained command bindings");
        if (consumed == 3)
            advance_peer(2);
        else if (consumed == 4)
            advance_peer(1);
    }
    const auto after = get_vulkan_statistics(runtime);
    qsa_check(after.attention_blocks - before.attention_blocks == calls
                  && after.compute_submissions - before.compute_submissions == calls
                  && after.batch_uploads - before.batch_uploads == calls
                  && after.batch_downloads - before.batch_downloads == calls
                  && after.attention_cpu_fallbacks == before.attention_cpu_fallbacks,
              "one external boundary per QSA call, no internal transfers or CPU fallback");
    ActivationBuffer pooled;
    qsa_download(op, actual.qsa_device_state->pooled_keys, consumed / fixture.plan.compression_ratio,
                 fixture.plan.index_head_dimension, pooled);
    qsa_check(pooled.values().size() == reference.qsa_block_keys.size(), "pooled cache shape");
    for (size_t index = 0; index < reference.qsa_block_keys.size(); ++index)
        qsa_check(std::abs(pooled.values()[index] - reference.qsa_block_keys[index]) <= 0.0035f,
                  "BF16 average before QSA key normalization and partial RoPE");
    auto stable = actual.qsa_device_state;
    const auto input = qsa_input(1, fixture.hidden, consumed);
    ActivationBuffer ignored;
    qsa_check(!qsa_forward(op, actual, start_position + consumed + 1, input, ignored), "reject discontinuous QSA prefix");
    qsa_check(actual.qsa_device_state == stable, "prefix rejection preserves state");
    auto transaction = begin_state_cache_transaction(std::span<LayerCache>(&actual, 1), 1);
    qsa_check(!transaction && !actual.transaction.active, "device QSA transaction rejected before mutation");
    std::weak_ptr<QsaCache_vulkan> old_owner = actual.qsa_device_state;
    actual = {};
    qsa_check(actual.device_allocated_size == 0 && !actual.qsa_device_state, "reset clears QSA ownership and accounting");
    // An explicit lease keeps the old cache alive across Session reset.
    qsa_check(!old_owner.expired(), "held device cache lease survives Session reset");
    stable.reset();
    qsa_check(old_owner.expired(), "released device cache lease expires after Session reset");
    auto fresh = qsa_input(3, fixture.hidden, 0);
    LayerCache fresh_cpu;
    AttentionScratch fresh_scratch;
    ActivationBuffer expected;
    qsa_check(static_cast<bool>(forward_attention(reference_weights, cpu_operators, fixture.plan, ExecutionBackend::Cpu,
                                                  0, fresh_cpu, fresh_scratch, fresh, expected, OptimizationDefaultFlags)),
              "reset CPU oracle");
    qsa_check(qsa_forward(op, actual, 0, fresh, ignored), "reset QSA starts clean prefix");
    qsa_parity(ignored, expected, "reset QSA parity");
    std::cout << "QSA device=" << device << " head=" << head_dimension << " dtype=" << static_cast<int>(dtype)
              << " nativeQ8=" << quantized << " tied=" << tied_scores << " passed\n";
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
        {
            std::cout << "SKIP: Vulkan device unavailable\n";
            return 77;
        }
        for (uint32_t device = 0; device < ncnn::moe::get_gpu_count(); ++device)
        {
            for (uint32_t round = 0; round < 4; ++round)
            {
                std::cout << "QSA stress round=" << (round + 1) << "/4 device=" << device << '\n';
                qsa_case(device, 64, DType::BFloat16, false, false, true);
                qsa_case(device, 64, DType::BFloat16, true, false, true);
                qsa_case(device, 128, DType::Float32, false, false);
                qsa_case(device, 256, DType::BFloat16, true, false);
                qsa_case(device, 128, DType::BFloat16, true, true);
            }
        }
        return 0;
#else
        std::cout << "SKIP: Vulkan disabled\n";
        return 77;
#endif
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
