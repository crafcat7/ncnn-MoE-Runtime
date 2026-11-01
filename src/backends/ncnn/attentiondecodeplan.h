#ifndef NCNN_MOE_ATTENTIONDECODEPLAN_H
#define NCNN_MOE_ATTENTIONDECODEPLAN_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ncnn {
namespace moe {

enum class AttentionDecodePath
{
    Scalar,
    Matrix,
};

// Device facts only. No model identity or allocation results participate in
// selection. Bit (D / 16 - 1) describes supported matrix head dimensions.
struct AttentionDecodeCapabilities
{
    bool bfloat16_matrix = false;
    bool workgroup_matrix = false;
    uint32_t supported_head_dimensions = 0;
    uint32_t query_tile = 32;
    uint32_t kv_tile = 32;
    uint32_t subgroup_size = 32;
    uint32_t workgroup_size = 256;
    uint32_t device_parallelism = 32;
    uint32_t max_workgroups_x = 65535;
    uint32_t max_workgroups_y = 65535;
    uint32_t max_workgroups_z = 65535;
};

struct AttentionDecodeShape
{
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dimension = 0;
    uint32_t kv_tokens = 0;
    bool bfloat16_kv = false;
    // Logical BF16 cache values can live in native BF16 or FP32 storage.
    bool kv_storage_bfloat16 = true;
};

struct AttentionDecodePlan
{
    AttentionDecodePath path = AttentionDecodePath::Scalar;
    AttentionDecodeShape shape;
    uint32_t group_size = 0;
    uint32_t query_tile = 0;
    uint32_t kv_tile = 0;
    uint32_t query_groups = 0;
    uint32_t split_count = 0;
    uint32_t split_tokens = 0;
    uint32_t scratch_split_capacity = 0;
    uint32_t subgroup_size = 0;
    uint32_t workgroup_size = 0;
    uint64_t scratch_elements = 0;
    uint64_t scratch_bytes = 0;
};

// The workspace bound is monotonic in kv_tokens. Actual splits can decrease
// at tile boundaries, so allocation uses scratch_split_capacity instead.
// Each split/head stores D numerator floats followed by maximum and sum.
[[nodiscard]] inline bool make_attention_decode_plan(const AttentionDecodeCapabilities& capabilities,
                                                     const AttentionDecodeShape& shape,
                                                     AttentionDecodePlan& destination) noexcept
{
    destination = {};
    destination.shape = shape;
    if (shape.query_heads == 0 || shape.kv_heads == 0
        || shape.head_dimension == 0 || shape.kv_tokens == 0
        || shape.query_heads % shape.kv_heads != 0
        || static_cast<uint64_t>(shape.query_heads) * shape.head_dimension
               > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;

    destination.group_size = shape.query_heads / shape.kv_heads;
    if (!capabilities.bfloat16_matrix || !capabilities.workgroup_matrix
        || !shape.bfloat16_kv || destination.group_size < 2
        || shape.head_dimension > 128 || shape.head_dimension % 16 != 0
        || (capabilities.supported_head_dimensions & (1u << (shape.head_dimension / 16 - 1))) == 0
        || capabilities.query_tile != 32 || capabilities.kv_tile != 32
        || capabilities.subgroup_size == 0 || capabilities.workgroup_size != 256
        || capabilities.workgroup_size % capabilities.subgroup_size != 0
        || capabilities.device_parallelism == 0)
        return true;

    const auto divide_up = [](uint64_t numerator, uint64_t denominator) {
        return numerator / denominator + (numerator % denominator != 0);
    };
    const uint64_t query_groups = divide_up(destination.group_size, capabilities.query_tile);
    if (query_groups > capabilities.max_workgroups_y
        || shape.kv_heads > capabilities.max_workgroups_z
        || shape.query_heads > capabilities.max_workgroups_x
        || capabilities.max_workgroups_x == 0)
        return true;

    const uint64_t base_groups = query_groups * shape.kv_heads;
    const uint64_t blocks = divide_up(shape.kv_tokens, capabilities.kv_tile);
    const uint64_t desired_splits = divide_up(capabilities.device_parallelism, base_groups);
    const uint64_t split_capacity = std::min({blocks, desired_splits,
                                              static_cast<uint64_t>(capabilities.max_workgroups_x)});
    const uint64_t blocks_per_split = divide_up(blocks, split_capacity);
    const uint64_t split_count = divide_up(blocks, blocks_per_split);
    const uint64_t split_tokens = blocks_per_split * capabilities.kv_tile;
    const uint64_t row_elements = static_cast<uint64_t>(shape.head_dimension) + 2;
    // VkMat dimensions and shader offsets use 32-bit indexing. Invalid sizes
    // fail planning rather than creating a partially allocated execution plan.
    if (split_tokens > std::numeric_limits<uint32_t>::max()
        || split_count * split_tokens > std::numeric_limits<uint32_t>::max()
        || split_capacity > std::numeric_limits<uint64_t>::max() / shape.query_heads / row_elements)
        return false;
    const uint64_t scratch_elements = split_capacity * shape.query_heads * row_elements;
    if (scratch_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || static_cast<uint64_t>(shape.query_heads) * shape.head_dimension
               > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;

    destination.path = AttentionDecodePath::Matrix;
    destination.query_tile = capabilities.query_tile;
    destination.kv_tile = capabilities.kv_tile;
    destination.query_groups = static_cast<uint32_t>(query_groups);
    destination.split_count = static_cast<uint32_t>(split_count);
    destination.split_tokens = static_cast<uint32_t>(split_tokens);
    destination.scratch_split_capacity = static_cast<uint32_t>(split_capacity);
    destination.subgroup_size = capabilities.subgroup_size;
    destination.workgroup_size = capabilities.workgroup_size;
    destination.scratch_elements = scratch_elements;
    destination.scratch_bytes = scratch_elements * sizeof(float);
    return true;
}

} // namespace moe
} // namespace ncnn

#endif
