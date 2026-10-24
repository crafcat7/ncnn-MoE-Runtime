#include "memoryplan.h"

#include "ncnn/moe/runtime.h"

#include "kernels/mxfp4.h"
#include "kernels/qnk.h"
#include "kernels/int4.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace ncnn {
namespace moe {

static constexpr uint64_t gibibyte = 1024ull * 1024ull * 1024ull;
static constexpr uint64_t mebibyte = 1024ull * 1024ull;

static Result<uint64_t> checked_add(uint64_t left, uint64_t right, const char* name)
{
    if (right > std::numeric_limits<uint64_t>::max() - left)
        return Error{ErrorCode::InvalidModel, std::string(name) + " byte estimate overflows"};
    return left + right;
}

static Result<uint64_t> checked_multiply(uint64_t left, uint64_t right, const char* name)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return Error{ErrorCode::InvalidModel, std::string(name) + " byte estimate overflows"};
    return left * right;
}

static Result<uint64_t> matrix_storage_size(uint64_t rows, uint64_t columns, DType dtype, const char* name)
{
    auto elements = checked_multiply(rows, columns, name);
    if (!elements)
        return elements.error();
    if (dtype == DType::Float8E4M3)
    {
        const uint64_t scale_count = ((rows + 127) / 128) * ((columns + 127) / 128);
        return checked_add(elements.value(), scale_count * sizeof(float), name);
    }
    if (is_qnk_dtype(dtype))
    {
        if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, std::string(name) + " Qn_K dimensions exceed uint32"};
        const uint64_t size = qnk_storage_bytes(dtype, static_cast<size_t>(rows), static_cast<uint32_t>(columns));
        if (size == 0)
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid Qn_K dimensions"};
        return size;
    }
    const uint64_t element_size = dtype == DType::Int64 ? 8 : dtype == DType::BFloat16 ? 2
                                                                                       : 4;
    return checked_multiply(elements.value(), element_size, name);
}

static Result<void> add_matrix_size(uint64_t rows, uint64_t columns, DType dtype, const char* name, uint64_t& total)
{
    auto size = matrix_storage_size(rows, columns, dtype, name);
    if (!size)
        return size.error();
    auto added = checked_add(total, size.value(), name);
    if (!added)
        return added.error();
    total = added.value();
    return {};
}

static Result<void> add_vector_size(uint64_t count, uint64_t element_size, const char* name, uint64_t& total)
{
    auto size = checked_multiply(count, element_size, name);
    if (!size)
        return size.error();
    auto added = checked_add(total, size.value(), name);
    if (!added)
        return added.error();
    total = added.value();
    return {};
}

static Result<uint64_t> latent_dense_size(const MoeModelDescriptor& descriptor)
{
    uint64_t total = 0;
    const uint64_t element_size = descriptor.activation_dtype == DType::BFloat16 ? 2 : 4;
    Result<void> status = add_matrix_size(descriptor.vocabulary_size, descriptor.hidden_size, descriptor.activation_dtype, "DeepSeek embedding", total);
    if (!status)
        return status.error();
    status = add_matrix_size(descriptor.vocabulary_size, descriptor.hidden_size, descriptor.activation_dtype, "DeepSeek LM head", total);
    if (!status)
        return status.error();
    status = add_vector_size(descriptor.hidden_size, element_size, "DeepSeek final norm", total);
    if (!status)
        return status.error();

    const uint64_t multiplier = descriptor.hyper_connection_multiplier;
    status = add_matrix_size(multiplier, multiplier * descriptor.hidden_size, DType::Float32, "DeepSeek hyper head", total);
    if (!status)
        return status.error();
    status = add_vector_size(multiplier + 1, 4, "DeepSeek hyper head parameters", total);
    if (!status)
        return status.error();

    const bool has_dspark = descriptor.speculative_kind == SpeculativeModelKind::DSpark
                            && descriptor.speculative_layer_count != 0;
    if (has_dspark)
    {
        auto projection_columns = checked_multiply(descriptor.hidden_size,
                                                   descriptor.speculative_target_layer_ids.size(),
                                                   "DeepSeek speculative main projection");
        if (!projection_columns)
            return projection_columns.error();
        status = add_matrix_size(descriptor.hidden_size, projection_columns.value(), DType::Float8E4M3,
                                 "DeepSeek speculative main projection", total);
        if (!status)
            return status.error();
        status = add_vector_size(static_cast<uint64_t>(descriptor.hidden_size) * 2,
                                 element_size, "DeepSeek speculative norms", total);
        if (!status)
            return status.error();
        auto hyper_columns = checked_multiply(multiplier, descriptor.hidden_size,
                                              "DeepSeek speculative hyper head");
        if (!hyper_columns)
            return hyper_columns.error();
        status = add_matrix_size(multiplier, hyper_columns.value(), DType::Float32,
                                 "DeepSeek speculative hyper head", total);
        if (!status)
            return status.error();
        status = add_vector_size(multiplier + 1, 4, "DeepSeek speculative hyper head parameters", total);
        if (!status)
            return status.error();
        status = add_matrix_size(descriptor.vocabulary_size, descriptor.speculative_markov_rank,
                                 descriptor.activation_dtype, "DeepSeek speculative Markov embedding", total);
        if (!status)
            return status.error();
        status = add_matrix_size(descriptor.vocabulary_size, descriptor.speculative_markov_rank,
                                 descriptor.activation_dtype, "DeepSeek speculative Markov head", total);
        if (!status)
            return status.error();
        auto confidence_columns = checked_add(descriptor.hidden_size, descriptor.speculative_markov_rank,
                                              "DeepSeek speculative confidence");
        if (!confidence_columns)
            return confidence_columns.error();
        status = add_matrix_size(1, confidence_columns.value(), descriptor.activation_dtype,
                                 "DeepSeek speculative confidence", total);
        if (!status)
            return status.error();
    }

    const uint64_t main_layer_count = descriptor.layers.size();
    auto layer_count = checked_add(main_layer_count,
                                   has_dspark ? descriptor.speculative_layer_count : 0,
                                   "DeepSeek layer count");
    if (!layer_count)
        return layer_count.error();
    for (uint64_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        const bool draft = layer_id >= main_layer_count;
        const LayerDescriptor& layer = descriptor.layers[std::min(layer_id, main_layer_count - 1)];
        const AttentionDescriptor& attention = layer.attention;
        const MoeDescriptor& moe = layer.moe;
        if (attention.kind != AttentionKind::MultiHeadLatent
            || (attention.projection_weight_dtype != DType::Float8E4M3
                && attention.projection_weight_dtype != DType::BFloat16
                && attention.projection_weight_dtype != DType::Q8_0)
            || attention.output_group_count == 0
            || attention.head_count % attention.output_group_count != 0)
        {
            return Error{ErrorCode::InvalidModel, "invalid DeepSeek latent attention dimensions"};
        }
        const uint64_t mix_count = (2 + multiplier) * multiplier;
        for (uint32_t block = 0; block < 2; ++block)
        {
            status = add_matrix_size(mix_count, multiplier * descriptor.hidden_size, DType::Float32, "DeepSeek hyper function", total);
            if (!status)
                return status.error();
            status = add_vector_size(mix_count + 3, 4, "DeepSeek hyper parameters", total);
            if (!status)
                return status.error();
        }
        status = add_vector_size(descriptor.hidden_size * 2ull, element_size, "DeepSeek layer norms", total);
        if (!status)
            return status.error();

        status = add_matrix_size(attention.query_lora_rank, descriptor.hidden_size, attention.projection_weight_dtype, "DeepSeek query A", total);
        if (!status)
            return status.error();
        status = add_matrix_size(static_cast<uint64_t>(attention.head_count) * attention.head_dimension, attention.query_lora_rank, attention.projection_weight_dtype, "DeepSeek query B", total);
        if (!status)
            return status.error();
        status = add_matrix_size(attention.head_dimension, descriptor.hidden_size, attention.projection_weight_dtype, "DeepSeek latent KV", total);
        if (!status)
            return status.error();
        const uint64_t group_input = static_cast<uint64_t>(attention.head_count) * attention.head_dimension / attention.output_group_count;
        status = add_matrix_size(static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, group_input, attention.projection_weight_dtype, "DeepSeek output A", total);
        if (!status)
            return status.error();
        status = add_matrix_size(descriptor.hidden_size, static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, attention.projection_weight_dtype, "DeepSeek output B", total);
        if (!status)
            return status.error();
        status = add_vector_size(attention.query_lora_rank + attention.head_dimension, element_size,
                                 "DeepSeek attention norms", total);
        if (!status)
            return status.error();
        status = add_vector_size(attention.head_count, 4, "DeepSeek attention sinks", total);
        if (!status)
            return status.error();

        if (!draft && attention.compression_ratio != 0)
        {
            const uint64_t compressor_multiplier = attention.compression_ratio == 4 ? 2 : 1;
            status = add_matrix_size(compressor_multiplier * attention.head_dimension, descriptor.hidden_size,
                                     descriptor.activation_dtype, "DeepSeek compressor KV", total);
            if (!status)
                return status.error();
            status = add_matrix_size(compressor_multiplier * attention.head_dimension, descriptor.hidden_size,
                                     descriptor.activation_dtype, "DeepSeek compressor gate", total);
            if (!status)
                return status.error();
            status = add_matrix_size(attention.compression_ratio, compressor_multiplier * attention.head_dimension, DType::Float32, "DeepSeek compressor position", total);
            if (!status)
                return status.error();
            status = add_vector_size(attention.head_dimension, element_size, "DeepSeek compressor norm", total);
            if (!status)
                return status.error();
            if (attention.compression_ratio == 4)
            {
                status = add_matrix_size(2ull * attention.index_head_dimension, descriptor.hidden_size,
                                         descriptor.activation_dtype, "DeepSeek index compressor KV", total);
                if (!status)
                    return status.error();
                status = add_matrix_size(2ull * attention.index_head_dimension, descriptor.hidden_size,
                                         descriptor.activation_dtype, "DeepSeek index compressor gate", total);
                if (!status)
                    return status.error();
                status = add_matrix_size(attention.compression_ratio, 2ull * attention.index_head_dimension, DType::Float32, "DeepSeek index compressor position", total);
                if (!status)
                    return status.error();
                status = add_matrix_size(static_cast<uint64_t>(attention.index_head_count) * attention.index_head_dimension, attention.query_lora_rank, attention.projection_weight_dtype, "DeepSeek index query", total);
                if (!status)
                    return status.error();
                status = add_matrix_size(attention.index_head_count, descriptor.hidden_size,
                                         descriptor.activation_dtype, "DeepSeek index weights", total);
                if (!status)
                    return status.error();
                status = add_vector_size(attention.index_head_dimension, element_size,
                                         "DeepSeek index norm", total);
                if (!status)
                    return status.error();
            }
        }

        status = add_matrix_size(moe.expert_count, descriptor.hidden_size, descriptor.activation_dtype,
                                 "DeepSeek router", total);
        if (!status)
            return status.error();
        if (!draft && layer_id < descriptor.hash_routing_layer_count)
            status = add_matrix_size(descriptor.vocabulary_size, moe.top_k, DType::Int64, "DeepSeek hash router", total);
        else
            status = add_vector_size(moe.expert_count, 4, "DeepSeek router bias", total);
        if (!status)
            return status.error();
        if (draft || moe.shared_expert_count != 0)
        {
            status = add_matrix_size(moe.intermediate_size, descriptor.hidden_size, moe.shared_expert_weight_dtype,
                                     "DeepSeek shared Expert input", total);
            if (!status)
                return status.error();
            status = add_matrix_size(moe.intermediate_size, descriptor.hidden_size, moe.shared_expert_weight_dtype,
                                     "DeepSeek shared Expert input", total);
            if (!status)
                return status.error();
            status = add_matrix_size(descriptor.hidden_size, moe.intermediate_size, moe.shared_expert_weight_dtype,
                                     "DeepSeek shared Expert output", total);
            if (!status)
                return status.error();
        }
        if (!draft && moe.shared_expert_count != 0
            && has_flag(moe.flags, MoeDescriptorSharedExpertGate))
        {
            status = add_vector_size(descriptor.hidden_size, element_size,
                                     "DeepSeek shared Expert router gate", total);
            if (!status)
                return status.error();
        }
    }
    return total;
}

static Result<uint64_t> latent_vulkan_releasable_dense_size(const MoeModelDescriptor& descriptor)
{
    if (descriptor.layers.empty()
        || descriptor.hyper_connection_multiplier <= 1
        || descriptor.layers.front().attention.kind != AttentionKind::MultiHeadLatent
        || (descriptor.layers.front().attention.projection_weight_dtype != DType::Float8E4M3
            && descriptor.layers.front().attention.projection_weight_dtype != DType::BFloat16
            && descriptor.layers.front().attention.projection_weight_dtype != DType::Q8_0))
    {
        return uint64_t{0};
    }

    uint64_t total = 0;
    Result<void> status = add_matrix_size(descriptor.vocabulary_size,
                                          descriptor.hidden_size,
                                          descriptor.activation_dtype,
                                          "Vulkan LM head",
                                          total);
    if (!status)
        return status.error();

    for (const LayerDescriptor& layer : descriptor.layers)
    {
        const AttentionDescriptor& attention = layer.attention;
        if (attention.kind != AttentionKind::MultiHeadLatent
            || (attention.projection_weight_dtype != DType::Float8E4M3
                && attention.projection_weight_dtype != DType::BFloat16
                && attention.projection_weight_dtype != DType::Q8_0)
            || attention.output_group_count == 0
            || attention.head_count % attention.output_group_count != 0)
        {
            return uint64_t{0};
        }
        const uint64_t group_input = static_cast<uint64_t>(attention.head_count) * attention.head_dimension / attention.output_group_count;
        const std::pair<uint64_t, uint64_t> matrices[] = {
            {attention.query_lora_rank, descriptor.hidden_size},
            {static_cast<uint64_t>(attention.head_count) * attention.head_dimension, attention.query_lora_rank},
            {attention.head_dimension, descriptor.hidden_size},
            {static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, group_input},
            {descriptor.hidden_size, static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank},
        };
        for (const auto& matrix : matrices)
        {
            status = add_matrix_size(matrix.first,
                                     matrix.second,
                                     attention.projection_weight_dtype,
                                     "Vulkan latent projection",
                                     total);
            if (!status)
                return status.error();
        }
        if (attention.compression_ratio == 4)
        {
            status = add_matrix_size(static_cast<uint64_t>(attention.index_head_count) * attention.index_head_dimension,
                                     attention.query_lora_rank,
                                     attention.projection_weight_dtype,
                                     "Vulkan index query",
                                     total);
            if (!status)
                return status.error();
        }

        // FP8 shared projections retain their CPU fallback host weights.
        if (layer.moe.shared_expert_count != 0
            && layer.moe.shared_expert_weight_dtype != DType::Float8E4M3)
        {
            status = add_matrix_size(layer.moe.intermediate_size,
                                     descriptor.hidden_size,
                                     layer.moe.shared_expert_weight_dtype,
                                     "Vulkan shared Expert input",
                                     total);
            if (!status)
                return status.error();
            status = add_matrix_size(layer.moe.intermediate_size,
                                     descriptor.hidden_size,
                                     layer.moe.shared_expert_weight_dtype,
                                     "Vulkan shared Expert input",
                                     total);
            if (!status)
                return status.error();
            status = add_matrix_size(descriptor.hidden_size,
                                     layer.moe.intermediate_size,
                                     layer.moe.shared_expert_weight_dtype,
                                     "Vulkan shared Expert output",
                                     total);
            if (!status)
                return status.error();
        }
    }
    return total;
}

static Result<uint64_t> dense_size(const MoeModelDescriptor& descriptor)
{
    const bool latent = !descriptor.layers.empty()
                        && descriptor.hyper_connection_multiplier > 1
                        && descriptor.layers.front().attention.kind == AttentionKind::MultiHeadLatent
                        && (descriptor.layers.front().attention.projection_weight_dtype == DType::Float8E4M3
                            || descriptor.layers.front().attention.projection_weight_dtype == DType::BFloat16
                            || descriptor.layers.front().attention.projection_weight_dtype == DType::Q8_0);
    if (latent)
        return latent_dense_size(descriptor);

    const uint64_t element_size = descriptor.activation_dtype == DType::BFloat16 ? 2 : 4;
    auto embedding_elements = checked_multiply(descriptor.vocabulary_size, descriptor.hidden_size, "embedding");
    if (!embedding_elements)
        return embedding_elements.error();
    auto embedding_size = checked_multiply(embedding_elements.value(), element_size * 2, "embedding and LM head");
    if (!embedding_size)
        return embedding_size.error();
    uint64_t total = embedding_size.value();

    for (const LayerDescriptor& layer : descriptor.layers)
    {
        uint64_t layer_elements = descriptor.hidden_size;

        if (layer.attention.kind != AttentionKind::None)
        {
            auto with_attention_norm = checked_add(layer_elements, descriptor.hidden_size, "layer norms");
            if (!with_attention_norm)
                return with_attention_norm.error();
            layer_elements = with_attention_norm.value();
            const AttentionDescriptor& attention = layer.attention;
            const uint64_t query_size = static_cast<uint64_t>(attention.head_count) * attention.head_dimension;
            const uint64_t key_value_size = static_cast<uint64_t>(attention.kv_head_count) * attention.head_dimension;
            Result<uint64_t> query = uint64_t{0};
            Result<uint64_t> key_value = uint64_t{0};
            Result<uint64_t> output = uint64_t{0};
            if (attention.kind == AttentionKind::Standard)
            {
                query = checked_multiply(query_size, descriptor.hidden_size, "query projection");
                key_value = checked_multiply(key_value_size, descriptor.hidden_size * 2ull, "key/value projections");
                output = checked_multiply(descriptor.hidden_size, query_size, "attention output projection");
            }
            else if (attention.kind == AttentionKind::MultiHeadLatent)
            {
                const uint64_t qk_size = static_cast<uint64_t>(attention.head_count) * (attention.qk_nope_head_dimension + attention.qk_rope_head_dimension);
                const uint64_t kv_a_size = attention.kv_lora_rank + attention.qk_rope_head_dimension;
                const uint64_t kv_b_size = static_cast<uint64_t>(attention.head_count) * (attention.qk_nope_head_dimension + attention.value_head_dimension);
                const uint64_t query_a_elements = attention.query_lora_rank == 0
                                                      ? 0
                                                      : static_cast<uint64_t>(descriptor.hidden_size) * attention.query_lora_rank + attention.query_lora_rank;
                const uint64_t query_b_elements = attention.query_lora_rank == 0
                                                      ? static_cast<uint64_t>(descriptor.hidden_size) * qk_size
                                                      : static_cast<uint64_t>(attention.query_lora_rank) * qk_size;
                query = checked_add(query_a_elements, query_b_elements, "MLA query projections");
                key_value = checked_add(static_cast<uint64_t>(descriptor.hidden_size) * kv_a_size + attention.kv_lora_rank, static_cast<uint64_t>(attention.kv_lora_rank) * kv_b_size, "MLA key/value projections");
                output = checked_multiply(descriptor.hidden_size, static_cast<uint64_t>(attention.head_count) * attention.value_head_dimension, "MLA output projection");
            }
            else
            {
                const uint64_t linear_key_size = static_cast<uint64_t>(attention.kv_head_count) * attention.head_dimension;
                const uint64_t linear_value_size = static_cast<uint64_t>(attention.head_count) * attention.value_head_dimension;
                const uint64_t convolution_size = linear_key_size * 2 + linear_value_size;
                query = checked_multiply(convolution_size + linear_value_size + attention.head_count * 2ull, descriptor.hidden_size, "gated DeltaNet projections");
                key_value = checked_multiply(convolution_size, attention.convolution_kernel_size, "gated DeltaNet convolution");
                if (key_value)
                {
                    key_value = checked_add(key_value.value(),
                                            attention.head_count * 2ull + attention.value_head_dimension,
                                            "gated DeltaNet parameters");
                }
                output = checked_multiply(descriptor.hidden_size, linear_value_size, "gated DeltaNet output projection");
            }
            if (!query)
                return query.error();
            if (!key_value)
                return key_value.error();
            if (!output)
                return output.error();
            auto projections = checked_add(query.value(), key_value.value(), "attention projections");
            if (!projections)
                return projections.error();
            projections = checked_add(projections.value(), output.value(), "attention projections");
            if (!projections)
                return projections.error();
            auto with_projections = checked_add(layer_elements, projections.value(), "layer dense weights");
            if (!with_projections)
                return with_projections.error();
            layer_elements = with_projections.value();
            if (has_flag(layer.attention.flags, AttentionDescriptorBias))
            {
                const uint64_t bias_elements = attention.kind == AttentionKind::Standard
                                                   ? query_size + key_value_size * 2 + descriptor.hidden_size
                                                   : static_cast<uint64_t>(attention.query_lora_rank) + attention.kv_lora_rank + attention.qk_rope_head_dimension + descriptor.hidden_size;
                auto with_bias = checked_add(layer_elements, bias_elements, "attention biases");
                if (!with_bias)
                    return with_bias.error();
                layer_elements = with_bias.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorSinks))
            {
                auto with_sinks = checked_add(layer_elements, layer.attention.head_count, "attention sinks");
                if (!with_sinks)
                    return with_sinks.error();
                layer_elements = with_sinks.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorQueryKeyNorm))
            {
                auto with_qk_norm = checked_add(layer_elements, layer.attention.head_dimension * 2ull, "query/key norms");
                if (!with_qk_norm)
                    return with_qk_norm.error();
                layer_elements = with_qk_norm.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorOutputGate))
            {
                auto output_gate = checked_multiply(query_size, descriptor.hidden_size, "attention output gate");
                if (!output_gate)
                    return output_gate.error();
                auto with_output_gate = checked_add(layer_elements, output_gate.value(), "attention output gate");
                if (!with_output_gate)
                    return with_output_gate.error();
                layer_elements = with_output_gate.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorQsa))
            {
                const uint64_t index_columns = (static_cast<uint64_t>(attention.index_head_count) + 1)
                                               * attention.index_head_dimension;
                auto index_projection = checked_multiply(index_columns, descriptor.hidden_size, "QSA index projection");
                if (!index_projection)
                    return index_projection.error();
                auto index_weights = checked_add(index_projection.value(),
                                                 static_cast<uint64_t>(attention.index_head_dimension) * 2,
                                                 "QSA index weights");
                if (!index_weights)
                    return index_weights.error();
                auto with_index = checked_add(layer_elements, index_weights.value(),
                                              "layer dense weights");
                if (!with_index)
                    return with_index.error();
                layer_elements = with_index.value();
            }
        }

        uint64_t layer_auxiliary_size = 0;
        if (layer.dense.enabled())
        {
            auto status = add_matrix_size(layer.dense.intermediate_size, descriptor.hidden_size,
                                          layer.dense.gate_weight_dtype, "dense FFN gate", layer_auxiliary_size);
            if (!status)
                return status.error();
            status = add_matrix_size(layer.dense.intermediate_size, descriptor.hidden_size,
                                     layer.dense.up_weight_dtype, "dense FFN up", layer_auxiliary_size);
            if (!status)
                return status.error();
            status = add_matrix_size(descriptor.hidden_size, layer.dense.intermediate_size,
                                     layer.dense.down_weight_dtype, "dense FFN down", layer_auxiliary_size);
            if (!status)
                return status.error();
        }
        if (layer.ple.enabled())
        {
            if (layer.ple.ngram_size < 2 || layer.ple.heads_per_ngram == 0)
                return Error{ErrorCode::InvalidModel, "invalid PLE dense-memory configuration"};
            const uint64_t head_count = static_cast<uint64_t>(layer.ple.ngram_size - 1)
                                        * layer.ple.heads_per_ngram;
            if (head_count == 0
                || layer.ple.embedding_dimension != descriptor.hidden_size
                || layer.ple.embedding_dimension % head_count != 0)
            {
                return Error{ErrorCode::InvalidModel, "invalid PLE dense-memory configuration"};
            }
            const uint64_t expanded_size = static_cast<uint64_t>(descriptor.hyper_connection_multiplier)
                                           * descriptor.hidden_size;
            auto key_projection = checked_multiply(expanded_size, descriptor.hidden_size, "PLE key projection");
            if (!key_projection)
                return key_projection.error();
            auto value_projection = checked_multiply(descriptor.hidden_size, descriptor.hidden_size, "PLE value projection");
            if (!value_projection)
                return value_projection.error();
            auto ple_elements = checked_add(key_projection.value(), value_projection.value(),
                                            "PLE projections");
            if (!ple_elements)
                return ple_elements.error();
            auto norm_and_convolution = checked_multiply(expanded_size,
                                                         3ull + layer.ple.convolution_kernel_size,
                                                         "PLE norms and convolution");
            if (!norm_and_convolution)
                return norm_and_convolution.error();
            ple_elements = checked_add(ple_elements.value(), norm_and_convolution.value(),
                                       "PLE dense weights");
            if (!ple_elements)
                return ple_elements.error();
            auto with_ple = checked_add(layer_elements, ple_elements.value(),
                                        "layer dense weights");
            if (!with_ple)
                return with_ple.error();
            layer_elements = with_ple.value();
            auto head_metadata_count = checked_multiply(head_count, 2, "PLE metadata");
            if (!head_metadata_count)
                return head_metadata_count.error();
            auto metadata_count = checked_add(layer.ple.ngram_size, head_metadata_count.value(),
                                              "PLE metadata");
            if (!metadata_count)
                return metadata_count.error();
            auto metadata_size = checked_multiply(metadata_count.value(), sizeof(int64_t),
                                                  "PLE metadata");
            if (!metadata_size)
                return metadata_size.error();
            auto auxiliary_size = checked_add(layer_auxiliary_size, metadata_size.value(), "layer auxiliary weights");
            if (!auxiliary_size)
                return auxiliary_size.error();
            layer_auxiliary_size = auxiliary_size.value();
        }

        if (descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        {
            const uint64_t expanded_size = static_cast<uint64_t>(descriptor.hyper_connection_multiplier)
                                           * descriptor.hidden_size;
            auto projections = checked_multiply(expanded_size, descriptor.hyper_connection_low_rank * 2ull,
                                                "gated-residual projections");
            if (!projections)
                return projections.error();
            auto block_elements = checked_add(expanded_size, projections.value(),
                                              "gated-residual block");
            if (!block_elements)
                return block_elements.error();
            auto injection = checked_multiply(descriptor.hyper_connection_multiplier, expanded_size,
                                              "gated-residual injection");
            if (!injection)
                return injection.error();
            block_elements = checked_add(block_elements.value(), injection.value(),
                                         "gated-residual block");
            if (!block_elements)
                return block_elements.error();
            auto layer_gated_residual = checked_multiply(block_elements.value(), 2, "gated-residual layer");
            if (!layer_gated_residual)
                return layer_gated_residual.error();
            auto with_gated_residual = checked_add(layer_elements, layer_gated_residual.value(),
                                                   "layer dense weights");
            if (!with_gated_residual)
                return with_gated_residual.error();
            layer_elements = with_gated_residual.value();
        }

        const MoeDescriptor& moe = layer.moe;
        auto router = checked_multiply(moe.expert_count, descriptor.hidden_size, "router");
        if (!router)
            return router.error();
        auto with_router = checked_add(layer_elements, router.value(), "layer dense weights");
        if (!with_router)
            return with_router.error();
        layer_elements = with_router.value();
        if (has_flag(moe.flags, MoeDescriptorRouterBias))
        {
            auto with_router_bias = checked_add(layer_elements, moe.expert_count, "router bias");
            if (!with_router_bias)
                return with_router_bias.error();
            layer_elements = with_router_bias.value();
        }
        if (has_flag(moe.flags, MoeDescriptorProjectionBias))
        {
            const uint64_t per_expert_bias = static_cast<uint64_t>(moe.intermediate_size) * 2 + descriptor.hidden_size;
            auto expert_biases = checked_multiply(moe.expert_count, per_expert_bias, "expert biases");
            if (!expert_biases)
                return expert_biases.error();
            auto with_expert_biases = checked_add(layer_elements, expert_biases.value(), "layer dense weights");
            if (!with_expert_biases)
                return with_expert_biases.error();
            layer_elements = with_expert_biases.value();
        }
        if (moe.shared_expert_count != 0)
        {
            auto shared_elements = checked_multiply(static_cast<uint64_t>(descriptor.hidden_size) * moe.intermediate_size * 3, moe.shared_expert_count, "shared Expert weights");
            if (!shared_elements)
                return shared_elements.error();
            auto with_shared = checked_add(layer_elements, shared_elements.value(), "layer dense weights");
            if (!with_shared)
                return with_shared.error();
            layer_elements = with_shared.value();
            if (has_flag(moe.flags, MoeDescriptorSharedExpertGate))
            {
                auto with_shared_gate = checked_add(layer_elements, descriptor.hidden_size, "shared Expert gate");
                if (!with_shared_gate)
                    return with_shared_gate.error();
                layer_elements = with_shared_gate.value();
            }
        }

        auto layer_size = checked_multiply(layer_elements, element_size, "layer dense weights");
        if (!layer_size)
            return layer_size.error();
        auto with_layer = checked_add(total, layer_size.value(), "dense weights");
        if (!with_layer)
            return with_layer.error();
        with_layer = checked_add(with_layer.value(), layer_auxiliary_size,
                                 "dense weights");
        if (!with_layer)
            return with_layer.error();
        total = with_layer.value();
    }

    if (descriptor.speculative_kind == SpeculativeModelKind::Mtp)
    {
        if (descriptor.speculative_layer_count != 1
            || descriptor.layers.empty()
            || descriptor.layers.back().attention.kind != AttentionKind::Standard)
        {
            return Error{ErrorCode::InvalidModel, "invalid MTP dense-memory configuration"};
        }
        LayerDescriptor draft = descriptor.layers.back();
        draft.attention.flags &= ~AttentionDescriptorQsa;
        const AttentionDescriptor& attention = draft.attention;
        const MoeDescriptor& moe = draft.moe;
        const uint64_t query_size = static_cast<uint64_t>(attention.head_count) * attention.head_dimension;
        const uint64_t key_value_size = static_cast<uint64_t>(attention.kv_head_count) * attention.head_dimension;
        uint64_t mtp_elements = 0;
        auto add_elements = [&mtp_elements](uint64_t elements, const char* name) -> Result<void> {
            auto added = checked_add(mtp_elements, elements, name);
            if (!added)
                return added.error();
            mtp_elements = added.value();
            return {};
        };
        auto matrix_elements = [](uint64_t rows, uint64_t columns, const char* name) {
            return checked_multiply(rows, columns, name);
        };

        Result<void> status = add_elements(descriptor.hidden_size * 5ull, "MTP norms");
        if (!status)
            return status.error();
        const std::pair<uint64_t, uint64_t> matrices[] = {
            {descriptor.hidden_size, descriptor.hidden_size * 2ull},
            {query_size * 2ull, descriptor.hidden_size},
            {key_value_size * 2ull, descriptor.hidden_size},
            {descriptor.hidden_size, query_size},
            {moe.expert_count, descriptor.hidden_size},
        };
        for (const auto& matrix : matrices)
        {
            auto elements = matrix_elements(matrix.first, matrix.second, "MTP matrix");
            if (!elements)
                return elements.error();
            status = add_elements(elements.value(), "MTP matrix");
            if (!status)
                return status.error();
        }
        status = add_elements(attention.head_dimension * 2ull, "MTP query/key norms");
        if (!status)
            return status.error();
        if (moe.shared_expert_count != 0)
        {
            auto shared = matrix_elements(static_cast<uint64_t>(descriptor.hidden_size) * moe.intermediate_size * 3ull,
                                          moe.shared_expert_count,
                                          "MTP shared Expert");
            if (!shared)
                return shared.error();
            status = add_elements(shared.value(), "MTP shared Expert");
            if (!status)
                return status.error();
            if (has_flag(moe.flags, MoeDescriptorSharedExpertGate))
            {
                status = add_elements(descriptor.hidden_size, "MTP shared Expert gate");
                if (!status)
                    return status.error();
            }
        }
        auto mtp_size = checked_multiply(mtp_elements, element_size, "MTP dense weights");
        if (!mtp_size)
            return mtp_size.error();
        auto with_mtp = checked_add(total, mtp_size.value(), "dense weights");
        if (!with_mtp)
            return with_mtp.error();
        total = with_mtp.value();
    }

    if (descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
    {
        const uint64_t expanded_size = static_cast<uint64_t>(descriptor.hyper_connection_multiplier)
                                       * descriptor.hidden_size;
        auto head_projections = checked_multiply(expanded_size, descriptor.hyper_connection_low_rank * 2ull,
                                                 "gated-residual head projections");
        if (!head_projections)
            return head_projections.error();
        auto head_elements = checked_add(expanded_size, head_projections.value(),
                                         "gated-residual head");
        if (!head_elements)
            return head_elements.error();
        auto head_size = checked_multiply(head_elements.value(), element_size,
                                          "gated-residual head");
        if (!head_size)
            return head_size.error();
        auto with_head = checked_add(total, head_size.value(), "dense weights");
        if (!with_head)
            return with_head.error();
        total = with_head.value();
    }

    const uint64_t final_norm_elements = descriptor.final_norm == NormType::None
                                             ? 0
                                             : descriptor.hidden_size;
    auto final_norm = checked_multiply(final_norm_elements, element_size, "final norm");
    if (!final_norm)
        return final_norm.error();
    auto dense_total = checked_add(total, final_norm.value(), "dense weights");
    if (!dense_total)
        return dense_total.error();
    return dense_total.value();
}

static Result<uint64_t> mxfp4_pair_size(const MoeModelDescriptor& descriptor)
{
    auto elements = checked_multiply(descriptor.hidden_size, descriptor.intermediate_size, "expert pair");
    if (!elements)
        return elements.error();
    elements = checked_multiply(elements.value(), 3, "expert pair");
    if (!elements)
        return elements.error();
    auto encoded = checked_multiply(elements.value(), 17, "MXFP4 expert pair");
    if (!encoded)
        return encoded.error();
    return encoded.value() / 32;
}

static Result<uint64_t> expert_pair_size(const MoeModelDescriptor& descriptor, DType dtype)
{
    if (dtype == DType::MxFp4)
        return mxfp4_pair_size(descriptor);
    auto gate_up = matrix_storage_size(static_cast<uint64_t>(descriptor.intermediate_size) * 2,
                                       descriptor.hidden_size,
                                       dtype,
                                       "expert gate/up");
    if (!gate_up)
        return gate_up.error();
    auto down = matrix_storage_size(descriptor.hidden_size,
                                    descriptor.intermediate_size,
                                    dtype,
                                    "expert down");
    if (!down)
        return down.error();
    return checked_add(gate_up.value(), down.value(), "expert pair");
}

static Result<uint64_t> packed_matrix_storage_size(uint64_t rows,
                                                   uint64_t columns,
                                                   DType dtype,
                                                   const char* name)
{
    if (dtype == DType::MxFp4)
    {
        if (rows < 4)
            return uint64_t{0};
        if (rows > std::numeric_limits<size_t>::max()
            || columns == 0
            || columns > std::numeric_limits<uint32_t>::max()
            || columns % 32 != 0)
        {
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid MXFP4 dimensions"};
        }
        const uint64_t size = mxfp4_q8_packed_storage_bytes(static_cast<size_t>(rows),
                                                            static_cast<uint32_t>(columns / 32));
        if (size == 0)
            return Error{ErrorCode::InvalidModel, std::string(name) + " packed byte estimate overflows"};
        return size;
    }
    if (is_qnk_dtype(dtype))
    {
        if (rows > std::numeric_limits<size_t>::max()
            || columns > std::numeric_limits<uint32_t>::max())
        {
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid Qn_K dimensions"};
        }
        const uint64_t size = qnk_packed_storage_bytes(dtype,
                                                       static_cast<size_t>(rows),
                                                       static_cast<uint32_t>(columns));
        if (size == 0)
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid Qn_K dimensions"};
        return size;
    }
    return uint64_t{0};
}

static Result<uint64_t> packed_expert_pair_size(const MoeModelDescriptor& descriptor,
                                                const MoeDescriptor& moe)
{
    const DType dtype = moe.expert_weight_dtype;
    auto gate_up = packed_matrix_storage_size(static_cast<uint64_t>(descriptor.intermediate_size) * 2,
                                              descriptor.hidden_size,
                                              dtype,
                                              "packed expert gate/up");
    if (!gate_up)
        return gate_up.error();
    auto down = packed_matrix_storage_size(descriptor.hidden_size,
                                           descriptor.intermediate_size,
                                           dtype,
                                           "packed expert down");
    if (!down)
        return down.error();
    return checked_add(gate_up.value(), down.value(), "packed expert pair");
}

enum class QnkExpertEncoding
{
    None,
    FileBacked,
    Unsupported
};

static QnkExpertEncoding qnk_expert_encoding(const WeightMapping& weights) noexcept
{
    bool found = false;
    bool qnk_only = true;
    bool file_backed = true;
    for (const auto& [name, tensor] : weights)
    {
        if (name.find(".experts.") == std::string::npos || !name.ends_with(".weight"))
            continue;
        if (is_qnk_dtype(tensor.dtype))
        {
            found = true;
            file_backed = file_backed && static_cast<bool>(tensor.mxfp4_file_storage);
        }
        else
        {
            qnk_only = false;
        }
    }
    if (!found)
        return QnkExpertEncoding::None;
    return qnk_only && file_backed ? QnkExpertEncoding::FileBacked
                                   : QnkExpertEncoding::Unsupported;
}

static bool mapped_ple_embedding_shard(const std::string& name, const TensorData& tensor,
                                       const MoeModelDescriptor& descriptor) noexcept
{
    if (!tensor.mapped_data || tensor.mapped_size == 0 || tensor.shape.size() != 2 || tensor.shape[0] == 0)
        return false;
    const std::string_view canonical = name;
    constexpr std::string_view prefix = "layers.";
    constexpr std::string_view marker = ".ple.embedding_shard.";
    if (!canonical.starts_with(prefix))
        return false;
    const size_t separator = canonical.find(marker, prefix.size());
    if (separator == std::string_view::npos)
        return false;
    const std::string_view layer_name = canonical.substr(prefix.size(), separator - prefix.size());
    const std::string_view shard_name = canonical.substr(separator + marker.size());
    if (layer_name.empty() || shard_name.empty()
        || (layer_name.size() > 1 && layer_name.front() == '0')
        || (shard_name.size() > 1 && shard_name.front() == '0'))
        return false;
    uint32_t layer_index = 0;
    uint32_t shard_index = 0;
    const auto layer = std::from_chars(layer_name.data(), layer_name.data() + layer_name.size(), layer_index);
    const auto shard = std::from_chars(shard_name.data(), shard_name.data() + shard_name.size(), shard_index);
    if (layer.ec != std::errc{} || layer.ptr != layer_name.data() + layer_name.size()
        || shard.ec != std::errc{} || shard.ptr != shard_name.data() + shard_name.size()
        || layer_index >= descriptor.layers.size())
        return false;
    const PleDescriptor& ple = descriptor.layers[layer_index].ple;
    if (!ple.enabled() || ple.ngram_size < 2 || ple.heads_per_ngram == 0
        || shard_index >= ple.embedding_shard_count || tensor.shape[0] > ple.embedding_row_count)
        return false;
    const uint64_t heads = static_cast<uint64_t>(ple.ngram_size - 1) * ple.heads_per_ngram;
    return ple.embedding_dimension % heads == 0 && tensor.shape[1] == ple.embedding_dimension / heads;
}

static Result<uint64_t> qwen_vulkan_releasable_dense_size(const MoeModelDescriptor& descriptor,
                                                          const Option& opt, const WeightMapping& weights)
{
    const uint64_t required_flags = OptimizationVulkanAttention | OptimizationVulkanQnK
                                    | OptimizationVulkanExpertGpuPriority | OptimizationVulkanCommandGraph;
    if (opt.hybrid_mode != HybridMode::HybridExperts
        || descriptor.hyper_connection_kind != HyperConnectionKind::GatedResidual
        || descriptor.layers.empty()
        || has_flag(opt.flags, OptionDisableGpuExpertExecution)
        || (opt.optimization_flags & required_flags) != required_flags)
        return uint64_t{0};
    for (const LayerDescriptor& layer : descriptor.layers)
    {
        if (layer.attention.kind != AttentionKind::GatedDeltaNet
            && (layer.attention.kind != AttentionKind::Standard
                || !has_flag(layer.attention.flags, AttentionDescriptorQsa)
                || !has_flag(layer.attention.flags, AttentionDescriptorQueryKeyNorm)))
            return uint64_t{0};
    }
    // The strict continuous chain rejects FP8 rather than retaining a CPU
    // execution path. Do not predict its release for that unsupported case.
    for (const auto& [name, tensor] : weights)
    {
        (void)name;
        if (tensor.dtype == DType::Float8E4M3)
            return uint64_t{0};
    }

    std::set<std::string> released_names;
    const auto add = [&](const std::string& name) {
        const auto found = weights.find(name);
        if (found != weights.end()
            && (found->second.dtype == DType::Float32 || found->second.dtype == DType::BFloat16
                || found->second.dtype == DType::Q8_0))
            released_names.insert(name);
    };
    add("lm_head.weight");
    for (size_t index = 0; index < descriptor.layers.size(); ++index)
    {
        const LayerDescriptor& layer = descriptor.layers[index];
        const std::string prefix = "layers." + std::to_string(index) + ".";
        if (layer.attention.kind == AttentionKind::GatedDeltaNet)
        {
            for (const char* name : {"attention.delta.qkv.weight", "attention.delta.z.weight",
                                     "attention.delta.beta.weight", "attention.delta.alpha.weight",
                                     "attention.output.weight"})
                add(prefix + name);
        }
        else
        {
            // QSA owns its projections internally, so their host copies stay.
            // Its public attention operator releases only these two norms.
            add(prefix + "attention.query_norm.weight");
            add(prefix + "attention.key_norm.weight");
        }
        if (layer.moe.shared_expert_count != 0)
        {
            const std::string shared = prefix + "shared_expert.";
            const auto gate = weights.find(shared + "gate.weight");
            const auto up = weights.find(shared + "up.weight");
            const auto down = weights.find(shared + "down.weight");
            if (gate != weights.end() && up != weights.end() && down != weights.end()
                && gate->second.dtype == DType::Q8_0 && up->second.dtype == DType::Q8_0
                && down->second.dtype == DType::Q8_0)
            {
                add(shared + "gate.weight");
                add(shared + "up.weight");
                add(shared + "down.weight");
            }
        }
    }

    struct MappedOwner
    {
        uint64_t bytes = 0;
        bool retained = false;
    };
    std::map<std::shared_ptr<const uint8_t>, MappedOwner, std::owner_less<std::shared_ptr<const uint8_t>>> owners;
    uint64_t released = 0;
    for (const auto& [name, tensor] : weights)
    {
        const bool release = released_names.contains(name);
        if (tensor.mapped_data)
        {
            auto& owner = owners[tensor.mapped_data];
            owner.bytes = std::max(owner.bytes, tensor.mapped_size);
            owner.retained = owner.retained || !release;
        }
        if (!release)
            continue;
        // Match release_tensor_host_storage, not logical tensor dimensions:
        // auxiliary metadata and native non-Q8 blocks are never reclaimed.
        const uint64_t count = tensor.dtype == DType::Float32    ? tensor.float32_data.size()
                               : tensor.dtype == DType::BFloat16 ? tensor.bfloat16_data.size()
                                                                 : tensor.quantized_data.size();
        const uint64_t width = tensor.dtype == DType::Float32    ? sizeof(float)
                               : tensor.dtype == DType::BFloat16 ? sizeof(uint16_t)
                                                                 : 1;
        auto bytes = checked_multiply(count, width, "resident Qwen owned dense release");
        if (!bytes)
            return bytes.error();
        auto total = checked_add(released, bytes.value(), "resident Qwen dense release");
        if (!total)
            return total.error();
        released = total.value();
    }
    for (const auto& [owner, state] : owners)
    {
        (void)owner;
        if (!state.retained)
        {
            // A tied embedding or any other surviving alias retains the
            // complete owner, exactly as final host-release accounting does.
            auto total = checked_add(released, state.bytes, "resident Qwen mapped dense release");
            if (!total)
                return total.error();
            released = total.value();
        }
    }
    return released;
}

static Result<void> mapped_weight_sizes(const WeightMapping& weights, const MoeModelDescriptor& descriptor,
                                        ModelMemoryPlan& plan, uint64_t& expert_fixed_size,
                                        uint64_t& max_qnk_cache_pair_size)
{
    struct MappedExpertSize
    {
        uint64_t total = 0;
        uint64_t raw = 0;
        uint64_t metadata = 0;
    };
    std::map<std::string, MappedExpertSize> expert_sizes;
    for (const auto& [name, tensor] : weights)
    {
        uint64_t bytes = tensor.dtype == DType::Int4
                             ? int4_storage_bytes(tensor)
                             : tensor.mapped_size
                                   + tensor.float32_data.size() * sizeof(float)
                                   + tensor.bfloat16_data.size() * sizeof(uint16_t)
                                   + tensor.int64_data.size() * sizeof(int64_t)
                                   + tensor.int8_data.size()
                                   + tensor.quantized_data.size()
                                   + tensor.quantization_scales.size() * sizeof(float)
                                   + tensor.input_column_indices.size() * sizeof(uint32_t)
                                   + tensor.mxfp4_blocks.size() + tensor.mxfp4_scales.size();
        // PLE accesses mapped table rows on demand. Its immutable file pages
        // are reclaimable rather than fixed resident storage. Keep every owned
        // allocation and metadata byte, including when a mapped view coexists.
        if (mapped_ple_embedding_shard(name, tensor, descriptor) && tensor.mapped_size <= bytes)
            bytes -= tensor.mapped_size;
        if (tensor.mxfp4_file_storage)
        {
            const auto& storage = *tensor.mxfp4_file_storage;
            auto stored = checked_add(storage.blocks_size, storage.secondary_blocks_size, "mapped Expert blocks");
            if (!stored)
                return stored.error();
            stored = checked_add(stored.value(), storage.scales_size, "mapped Expert scales");
            if (!stored)
                return stored.error();
            stored = checked_add(stored.value(), storage.secondary_scales_size, "mapped Expert scales");
            if (!stored)
                return stored.error();
            auto total = checked_add(bytes, stored.value(), "mapped Expert storage");
            if (!total)
                return total.error();
            bytes = total.value();
        }
        const size_t marker = name.find(".experts.");
        if (marker != std::string::npos)
        {
            const size_t end = name.find('.', marker + 9);
            if (end == std::string::npos)
                return Error{ErrorCode::InvalidModel, "invalid canonical Expert tensor name: " + name};
            MappedExpertSize& expert_size = expert_sizes[name.substr(0, end)];
            auto total = checked_add(expert_size.total, bytes, "mapped Expert weights");
            if (!total)
                return total.error();
            expert_size.total = total.value();
            if (!is_qnk_dtype(tensor.dtype))
            {
                auto resident = checked_add(expert_fixed_size, bytes,
                                            "resident non-Qn_K Expert weights");
                if (!resident)
                    return resident.error();
                expert_fixed_size = resident.value();
            }
            if (is_qnk_dtype(tensor.dtype))
            {
                if (tensor.input_column_indices.size()
                    > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t))
                {
                    return Error{ErrorCode::InvalidModel, "Qn_K column-map byte count overflows"};
                }
                const uint64_t metadata = static_cast<uint64_t>(tensor.input_column_indices.size())
                                          * sizeof(uint32_t);
                auto fixed_metadata = checked_add(expert_fixed_size, metadata,
                                                  "fixed Qn_K Expert metadata");
                if (!fixed_metadata)
                    return fixed_metadata.error();
                expert_fixed_size = fixed_metadata.value();
                auto expert_total = checked_add(expert_size.metadata, metadata,
                                                "Qn_K Expert metadata");
                if (!expert_total)
                    return expert_total.error();
                expert_size.metadata = expert_total.value();

                if (tensor.mxfp4_file_storage)
                {
                    const MxFp4FileStorage& storage = *tensor.mxfp4_file_storage;
                    auto raw_size = checked_add(storage.blocks_size,
                                                storage.secondary_blocks_size,
                                                "Qn_K Expert blocks");
                    if (!raw_size)
                        return raw_size.error();
                    auto expert_raw_size = checked_add(expert_size.raw, raw_size.value(), "Qn_K Expert blocks");
                    if (!expert_raw_size)
                        return expert_raw_size.error();
                    expert_size.raw = expert_raw_size.value();
                }
            }
        }
        else
        {
            auto total = checked_add(plan.estimated_dense_size, bytes, "mapped dense weights");
            if (!total)
                return total.error();
            plan.estimated_dense_size = total.value();
        }
    }
    for (const auto& [name, expert_size] : expert_sizes)
    {
        (void)name;
        auto total = checked_add(plan.estimated_expert_size, expert_size.total, "mapped Expert weights");
        if (!total)
            return total.error();
        plan.estimated_expert_size = total.value();
        plan.expert_pair_size = std::max(plan.expert_pair_size, expert_size.total);

        auto resident_metadata = checked_multiply(expert_size.metadata, 2,
                                                  "resident Qn_K Expert metadata");
        if (!resident_metadata)
            return resident_metadata.error();
        auto resident_pair = checked_add(expert_size.raw, resident_metadata.value(),
                                         "resident Qn_K Expert pair");
        if (!resident_pair)
            return resident_pair.error();
        max_qnk_cache_pair_size = std::max(max_qnk_cache_pair_size, resident_pair.value());
    }
    plan.expert_pair_resident_size = plan.expert_pair_size;
    plan.estimated_expert_resident_size = plan.estimated_expert_size;
    auto active = checked_multiply(plan.expert_pair_size, descriptor.experts_per_token, "mapped active Experts");
    if (!active)
        return active.error();
    plan.minimum_active_expert_size = active.value();
    return {};
}

Result<ModelMemoryPlan> plan_model_memory(const MoeModelDescriptor& descriptor, const Option& opt, uint64_t physical_memory_size,
                                          bool use_vulkan_dense,
                                          uint64_t available_memory_size,
                                          bool reserve_cpu_packed_weights,
                                          const WeightMapping* weights,
                                          bool use_vulkan_attention)
{
    if (opt.expert_memory_mode != ExpertMemoryMode::Auto
        && opt.expert_memory_mode != ExpertMemoryMode::Eager
        && opt.expert_memory_mode != ExpertMemoryMode::OnDemand)
    {
        return Error{ErrorCode::InvalidArgument, "expert memory mode is invalid"};
    }

    ModelMemoryPlan plan;
    bool budget_limited_by_available_memory = false;
    uint64_t expert_fixed_size = 0;
    uint64_t max_qnk_cache_pair_size = 0;
    plan.requested_mode = opt.expert_memory_mode;
    plan.physical_memory_size = physical_memory_size;
    plan.available_memory_size = available_memory_size;
    if (opt.host_memory_budget != 0)
    {
        if (physical_memory_size != 0 && opt.host_memory_budget > physical_memory_size)
        {
            return Error{ErrorCode::InvalidArgument, "host memory budget exceeds detected physical memory"};
        }
        plan.host_memory_budget = opt.host_memory_budget;
    }
    else if (physical_memory_size != 0)
    {
        plan.host_memory_budget = physical_memory_size / 4 * 3;
        if (available_memory_size != 0)
        {
            const uint64_t system_reserve = 2 * gibibyte;
            const uint64_t available_budget = available_memory_size > system_reserve
                                                  ? available_memory_size - system_reserve
                                                  : available_memory_size / 2;
            if (available_budget < plan.host_memory_budget)
            {
                plan.host_memory_budget = available_budget;
                budget_limited_by_available_memory = true;
            }
        }
    }
    else
    {
        plan.host_memory_budget = 8 * gibibyte;
    }

    if (weights)
    {
        if (reserve_cpu_packed_weights)
            return Error{ErrorCode::UnsupportedModel, "native imported weights require CpuPackedWeightMode::Disabled"};
        auto sizes = mapped_weight_sizes(*weights, descriptor, plan,
                                         expert_fixed_size,
                                         max_qnk_cache_pair_size);
        if (!sizes)
            return sizes.error();
    }
    else
    {
        auto estimated_dense = dense_size(descriptor);
        if (!estimated_dense)
            return estimated_dense.error();
        plan.estimated_dense_size = estimated_dense.value();
    }
    // PLE embedding shards remain file-backed and are paged on demand.  The
    // dense estimate covers only their resident projections and state.

    if (descriptor.layers.empty())
        return Error{ErrorCode::InvalidModel, "memory planner requires at least one layer"};
    if (std::all_of(descriptor.layers.begin(), descriptor.layers.end(), [](const LayerDescriptor& layer) {
            return layer.dense.enabled();
        }))
    {
        plan.selected_mode = ExpertMemoryMode::Eager;
        return plan;
    }
    const auto first_moe = std::find_if(descriptor.layers.begin(), descriptor.layers.end(), [](const LayerDescriptor& layer) {
        return !layer.dense.enabled();
    });
    const MoeDescriptor& moe = first_moe->moe;
    const QnkExpertEncoding qnk_encoding = weights ? qnk_expert_encoding(*weights)
                                                   : QnkExpertEncoding::None;
    const bool has_qnk_experts = qnk_encoding != QnkExpertEncoding::None;
    const bool file_backed_qnk_experts = qnk_encoding == QnkExpertEncoding::FileBacked;
    const auto add_expert_layers = [&](const MoeDescriptor& layer_moe,
                                       uint64_t layer_count) -> Result<void> {
        auto pair_size = expert_pair_size(descriptor, layer_moe.expert_weight_dtype);
        if (!pair_size)
            return pair_size.error();
        uint64_t packed_pair_size = 0;
        if (reserve_cpu_packed_weights)
        {
            auto packed_size = packed_expert_pair_size(descriptor, layer_moe);
            if (!packed_size)
                return packed_size.error();
            packed_pair_size = packed_size.value();
        }
        auto resident_pair_size = checked_add(pair_size.value(), packed_pair_size,
                                              "resident expert pair");
        if (!resident_pair_size)
            return resident_pair_size.error();

        plan.expert_pair_size = std::max(plan.expert_pair_size, pair_size.value());
        plan.expert_pair_resident_size = std::max(plan.expert_pair_resident_size,
                                                  resident_pair_size.value());

        auto pair_count = checked_multiply(layer_count, descriptor.expert_count,
                                           "expert count");
        if (!pair_count)
            return pair_count.error();
        auto expert_size = checked_multiply(pair_size.value(), pair_count.value(),
                                            "expert weights");
        if (!expert_size)
            return expert_size.error();
        auto total_expert_size = checked_add(plan.estimated_expert_size,
                                             expert_size.value(), "expert weights");
        if (!total_expert_size)
            return total_expert_size.error();
        plan.estimated_expert_size = total_expert_size.value();

        auto packed_expert_size = checked_multiply(packed_pair_size, pair_count.value(),
                                                   "packed expert weights");
        if (!packed_expert_size)
            return packed_expert_size.error();
        auto total_packed_expert_size = checked_add(plan.estimated_cpu_packed_expert_size,
                                                    packed_expert_size.value(),
                                                    "packed expert weights");
        if (!total_packed_expert_size)
            return total_packed_expert_size.error();
        plan.estimated_cpu_packed_expert_size = total_packed_expert_size.value();
        return {};
    };
    if (!weights)
    {
        for (const LayerDescriptor& layer : descriptor.layers)
        {
            if (layer.dense.enabled())
                continue;
            auto status = add_expert_layers(layer.moe, 1);
            if (!status)
                return status.error();
        }
        if (descriptor.speculative_layer_count != 0)
        {
            // MTP and D-Spark bind speculative gate/up weights as one 2I x H matrix.
            MoeDescriptor speculative_moe = descriptor.layers.back().moe;
            speculative_moe.layout = ExpertLayout::PackedGateUpDown;
            auto status = add_expert_layers(speculative_moe, descriptor.speculative_layer_count);
            if (!status)
                return status.error();
        }
        auto active_size = checked_multiply(plan.expert_pair_resident_size, moe.top_k, "active experts");
        if (!active_size)
            return active_size.error();
        plan.minimum_active_expert_size = active_size.value();
        auto resident_expert_size = checked_add(plan.estimated_expert_size,
                                                plan.estimated_cpu_packed_expert_size,
                                                "resident expert weights");
        if (!resident_expert_size)
            return resident_expert_size.error();
        plan.estimated_expert_resident_size = resident_expert_size.value();
    }

    const uint64_t safety_reserve = budget_limited_by_available_memory
                                        ? 0
                                        : std::max(2 * gibibyte, physical_memory_size == 0 ? 2 * gibibyte : physical_memory_size / 8);
    uint64_t eager_capacity = 0;
    if (plan.host_memory_budget > plan.estimated_dense_size && plan.host_memory_budget - plan.estimated_dense_size > safety_reserve)
    {
        eager_capacity = plan.host_memory_budget - plan.estimated_dense_size - safety_reserve;
    }
    const auto supports_file_backed_experts = [has_qnk_experts, file_backed_qnk_experts](const MoeDescriptor& descriptor) {
        if (has_qnk_experts)
            return file_backed_qnk_experts;
        return descriptor.expert_weight_dtype == DType::MxFp4
               || (descriptor.expert_weight_dtype == DType::BFloat16
                   && has_flag(descriptor.flags, MoeDescriptorFileBackedExperts));
    };
    if (!supports_file_backed_experts(moe))
    {
        plan.selected_mode = ExpertMemoryMode::Eager;
        if (opt.expert_memory_mode == ExpertMemoryMode::OnDemand || opt.expert_cache_size != 0)
        {
            return Error{ErrorCode::UnsupportedModel, "on-demand expert storage requires an explicitly file-backed Expert encoding"};
        }
        if (reserve_cpu_packed_weights
            && plan.estimated_expert_resident_size > eager_capacity)
        {
            return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold the requested CPU packed Expert weights"};
        }
        return plan;
    }

    if (opt.expert_cache_size != 0)
    {
        if (opt.expert_memory_mode == ExpertMemoryMode::Eager)
        {
            return Error{ErrorCode::InvalidArgument, "an explicit expert cache conflicts with eager expert mode"};
        }
        plan.selected_mode = ExpertMemoryMode::OnDemand;
    }
    else if (opt.expert_memory_mode == ExpertMemoryMode::Auto)
    {
        plan.selected_mode = plan.estimated_expert_resident_size <= eager_capacity ? ExpertMemoryMode::Eager : ExpertMemoryMode::OnDemand;
    }
    else
    {
        plan.selected_mode = opt.expert_memory_mode;
    }

    if (plan.selected_mode == ExpertMemoryMode::Eager)
    {
        if (reserve_cpu_packed_weights
            && plan.estimated_expert_resident_size > eager_capacity)
        {
            return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold the requested CPU packed Expert weights"};
        }
        return plan;
    }

    // Reserve host budget for the idle staging pool, not peak upload memory.
    uint64_t staging_reserve = 0;
    if (opt.hybrid_mode == HybridMode::HybridExperts
        && !has_flag(opt.flags, OptionDisableGpuExpertExecution))
    {
        const uint64_t device_count = opt.vulkan_device_indices.empty()
                                          ? 1
                                          : static_cast<uint64_t>(opt.vulkan_device_indices.size());
        auto staging_capacity = checked_multiply(device_count, 64 * mebibyte,
                                                 "GPU Expert staging reserve");
        if (!staging_capacity)
            return staging_capacity.error();
        staging_reserve = opt.expert_gpu_cache_size != 0
                              ? std::min(opt.expert_gpu_cache_size, staging_capacity.value())
                              : staging_capacity.value();
    }
    const uint64_t cache_safety_reserve = std::max(safety_reserve, staging_reserve);
    eager_capacity = plan.host_memory_budget > plan.estimated_dense_size
                             && plan.host_memory_budget - plan.estimated_dense_size > cache_safety_reserve
                         ? plan.host_memory_budget - plan.estimated_dense_size - cache_safety_reserve
                         : 0;
    for (const LayerDescriptor& layer : descriptor.layers)
    {
        if (layer.dense.enabled())
            continue;
        if (!supports_file_backed_experts(layer.moe))
        {
            return Error{ErrorCode::UnsupportedModel, "on-demand mode requires a file-backed Expert encoding"};
        }
    }

    if (use_vulkan_dense)
    {
        Result<uint64_t> releasable = uint64_t{0};
        if (weights && descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        {
            if (use_vulkan_attention)
                releasable = qwen_vulkan_releasable_dense_size(descriptor, opt, *weights);
        }
        else
            releasable = latent_vulkan_releasable_dense_size(descriptor);
        if (!releasable)
            return releasable.error();
        plan.estimated_vulkan_dense_host_release_size = std::min(releasable.value(), plan.estimated_dense_size);
        plan.estimated_dense_size -= plan.estimated_vulkan_dense_host_release_size;
        eager_capacity = 0;
        if (plan.host_memory_budget > plan.estimated_dense_size
            && plan.host_memory_budget - plan.estimated_dense_size > cache_safety_reserve)
        {
            eager_capacity = plan.host_memory_budget - plan.estimated_dense_size - cache_safety_reserve;
        }
    }
    if (plan.selected_mode == ExpertMemoryMode::OnDemand && file_backed_qnk_experts)
    {
        auto fixed_resident = checked_add(plan.estimated_dense_size, expert_fixed_size,
                                          "fixed Qn_K Expert weights");
        if (!fixed_resident)
            return fixed_resident.error();
        plan.estimated_dense_size = fixed_resident.value();

        plan.expert_pair_resident_size = max_qnk_cache_pair_size;
        auto active = checked_multiply(plan.expert_pair_resident_size,
                                       descriptor.experts_per_token,
                                       "mapped active Experts");
        if (!active)
            return active.error();
        plan.minimum_active_expert_size = active.value();

        eager_capacity = 0;
        if (plan.host_memory_budget > plan.estimated_dense_size
            && plan.host_memory_budget - plan.estimated_dense_size > cache_safety_reserve)
        {
            eager_capacity = plan.host_memory_budget - plan.estimated_dense_size - cache_safety_reserve;
        }
    }
    if (eager_capacity < plan.minimum_active_expert_size)
    {
        return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold one layer's active Expert set"};
    }

    const uint64_t automatic_target = physical_memory_size == 0 ? 2 * gibibyte : eager_capacity;
    plan.expert_cache_size = opt.expert_cache_size != 0 ? opt.expert_cache_size : std::min(automatic_target, eager_capacity);
    const uint64_t host_capacity = plan.host_memory_budget > plan.estimated_dense_size
                                       ? plan.host_memory_budget - plan.estimated_dense_size
                                       : 0;
    const uint64_t cache_limit = opt.expert_cache_size != 0
                                     ? (host_capacity > staging_reserve ? host_capacity - staging_reserve : 0)
                                     : eager_capacity;
    if (plan.expert_cache_size > cache_limit)
    {
        return Error{ErrorCode::InvalidArgument, "expert cache and dense weights exceed the host memory budget"};
    }
    if (plan.expert_cache_size < plan.minimum_active_expert_size)
    {
        return Error{ErrorCode::InvalidArgument, "expert cache is smaller than one layer's active Expert set"};
    }
    return plan;
}

Result<void> finalize_model_memory_plan(ModelMemoryPlan& plan, const Option& opt,
                                        uint64_t released_dense_host_size)
{
    auto retained_size = checked_add(plan.estimated_dense_size, plan.estimated_vulkan_dense_host_release_size,
                                     "dense host storage before Vulkan reclamation");
    if (!retained_size)
        return retained_size.error();
    plan.estimated_dense_size = retained_size.value() - std::min(retained_size.value(), released_dense_host_size);
    plan.estimated_vulkan_dense_host_release_size = std::min(retained_size.value(), released_dense_host_size);
    if (plan.selected_mode != ExpertMemoryMode::OnDemand)
        return {};

    const bool budget_limited_by_available_memory = opt.host_memory_budget == 0
                                                    && plan.physical_memory_size != 0
                                                    && plan.available_memory_size != 0
                                                    && plan.host_memory_budget < plan.physical_memory_size / 4 * 3;
    const uint64_t safety_reserve = budget_limited_by_available_memory
                                        ? 0
                                        : std::max(2 * gibibyte, plan.physical_memory_size == 0 ? 2 * gibibyte : plan.physical_memory_size / 8);
    uint64_t staging_reserve = 0;
    if (opt.hybrid_mode == HybridMode::HybridExperts
        && !has_flag(opt.flags, OptionDisableGpuExpertExecution))
    {
        const uint64_t device_count = opt.vulkan_device_indices.empty()
                                          ? 1
                                          : static_cast<uint64_t>(opt.vulkan_device_indices.size());
        auto staging_capacity = checked_multiply(device_count, 64 * mebibyte,
                                                 "GPU Expert staging reserve");
        if (!staging_capacity)
            return staging_capacity.error();
        staging_reserve = opt.expert_gpu_cache_size != 0
                              ? std::min(opt.expert_gpu_cache_size, staging_capacity.value())
                              : staging_capacity.value();
    }
    const uint64_t cache_safety_reserve = std::max(safety_reserve, staging_reserve);
    const uint64_t host_capacity = plan.host_memory_budget > plan.estimated_dense_size
                                       ? plan.host_memory_budget - plan.estimated_dense_size
                                       : 0;
    const uint64_t cache_capacity = host_capacity > cache_safety_reserve
                                        ? host_capacity - cache_safety_reserve
                                        : 0;
    if (cache_capacity < plan.minimum_active_expert_size)
        return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold one layer's active Expert set"};
    if (opt.expert_cache_size == 0)
    {
        const uint64_t automatic_target = plan.physical_memory_size == 0 ? 2 * gibibyte : cache_capacity;
        plan.expert_cache_size = std::min(automatic_target, cache_capacity);
    }
    const uint64_t explicit_cache_capacity = host_capacity > staging_reserve
                                                 ? host_capacity - staging_reserve
                                                 : 0;
    if (plan.expert_cache_size > (opt.expert_cache_size != 0 ? explicit_cache_capacity : cache_capacity))
        return Error{ErrorCode::InvalidArgument, "expert cache and dense weights exceed the host memory budget"};
    if (plan.expert_cache_size < plan.minimum_active_expert_size)
        return Error{ErrorCode::InvalidArgument, "expert cache is smaller than one layer's active Expert set"};
    return {};
}

} // namespace moe
} // namespace ncnn
