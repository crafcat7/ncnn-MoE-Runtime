#include "backends/ncnn/attentiondecodeplan.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

static void plan_check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static AttentionDecodeCapabilities matrix_capabilities()
{
    AttentionDecodeCapabilities result;
    result.bfloat16_matrix = result.workgroup_matrix = true;
    result.supported_head_dimensions = 0xff;
    result.device_parallelism = 64;
    return result;
}

static AttentionDecodePlan planned(const AttentionDecodeCapabilities& capabilities,
                                   const AttentionDecodeShape& shape)
{
    AttentionDecodePlan result;
    plan_check(make_attention_decode_plan(capabilities, shape, result), "valid shape failed planning");
    return result;
}

static void rejection_cases()
{
    const auto capabilities = matrix_capabilities();
    const AttentionDecodeShape valid{16, 2, 64, 513, true};
    const std::array<AttentionDecodeShape, 5> invalid = {
        AttentionDecodeShape{0, 2, 64, 513, true},
        {16, 0, 64, 513, true},
        {16, 2, 0, 513, true},
        {16, 2, 64, 0, true},
        {15, 2, 64, 513, true}};
    for (const auto& shape : invalid)
    {
        auto result = planned(capabilities, valid);
        plan_check(!make_attention_decode_plan(capabilities, shape, result), "malformed shape accepted");
        plan_check(result.path == AttentionDecodePath::Scalar && result.scratch_bytes == 0,
                   "failed planning retained an executable matrix plan");
    }
    auto wide = capabilities;
    wide.max_workgroups_y = wide.max_workgroups_z = std::numeric_limits<uint32_t>::max();
    AttentionDecodePlan result;
    plan_check(!make_attention_decode_plan(wide, {0x80000000u, 1, 128, 1, true}, result),
               "32-bit output/scratch indexing overflow accepted");
    plan_check(!make_attention_decode_plan(wide, {16, 2, 64, std::numeric_limits<uint32_t>::max(), true}, result),
               "rounded split length overflow accepted");
}

static void capability_selection()
{
    const auto supported = matrix_capabilities();
    const AttentionDecodeShape shape{16, 2, 64, 513, true};
    plan_check(planned(supported, shape).path == AttentionDecodePath::Matrix,
               "complete capabilities did not select matrix execution");
    auto float_ring = shape;
    float_ring.kv_storage_bfloat16 = false;
    const auto float_plan = planned(supported, float_ring);
    plan_check(float_plan.path == AttentionDecodePath::Matrix
                   && float_plan.scratch_bytes == planned(supported, shape).scratch_bytes,
               "physical FP32 storage changed the BF16-valued matrix plan");
    for (uint32_t disabled = 0; disabled < 11; ++disabled)
    {
        auto capabilities = supported;
        switch (disabled)
        {
        case 0: capabilities.bfloat16_matrix = false; break;
        case 1: capabilities.workgroup_matrix = false; break;
        case 2: capabilities.supported_head_dimensions &= ~(1u << (shape.head_dimension / 16 - 1)); break;
        case 3: capabilities.query_tile = 0; break;
        case 4: capabilities.kv_tile = 0; break;
        case 5: capabilities.subgroup_size = 0; break;
        case 6: capabilities.subgroup_size = 48; break;
        case 7: capabilities.workgroup_size = 128; break;
        case 8: capabilities.device_parallelism = 0; break;
        case 9: capabilities.max_workgroups_x = 0; break;
        case 10: capabilities.max_workgroups_z = 1; break;
        }
        const auto result = planned(capabilities, shape);
        plan_check(result.path == AttentionDecodePath::Scalar && result.scratch_bytes == 0,
                   "unsupported device did not select scalar before recording");
    }
    for (const auto& unsupported : {AttentionDecodeShape{16, 2, 64, 513, false},
                                    AttentionDecodeShape{2, 2, 64, 513, true},
                                    AttentionDecodeShape{16, 2, 10, 513, true},
                                    AttentionDecodeShape{16, 2, 144, 513, true}})
        plan_check(planned(supported, unsupported).path == AttentionDecodePath::Scalar,
                   "unsupported matrix shape did not select scalar before recording");
}

static void shape_and_memory_cases()
{
    const auto capabilities = matrix_capabilities();
    const std::array<std::array<uint32_t, 2>, 6> shapes = {{{32, 2}, {64, 8}, {128, 4}, {64, 20}, {128, 40}, {32, 3}}};
    for (const auto& shape : shapes)
    {
        const uint32_t dimension = shape[0], ratio = shape[1];
        uint64_t previous_bytes = 0;
        for (uint32_t tokens : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 513u, 4097u})
        {
            const auto result = planned(capabilities, {ratio * 2, 2, dimension, tokens, true});
            plan_check(result.path == AttentionDecodePath::Matrix, "supported shape did not select matrix");
            plan_check(result.group_size == ratio && result.query_groups == (ratio + result.query_tile - 1) / result.query_tile,
                       "GQA query groups include heads from another KV group");
            plan_check(result.split_count > 0 && result.split_tokens % result.kv_tile == 0
                           && static_cast<uint64_t>(result.split_count) * result.split_tokens >= tokens
                           && static_cast<uint64_t>(result.split_count - 1) * result.split_tokens < tokens,
                       "split plan overlaps, leaves a gap, or includes an empty trailing split");
            plan_check(result.split_count <= result.scratch_split_capacity,
                       "actual splits exceed reserved workspace");
            plan_check(result.scratch_bytes >= previous_bytes,
                       "growing KV length shrank the graph workspace bound");
            previous_bytes = result.scratch_bytes;
        }
    }
}

static void hardware_parallelism_cases()
{
    auto capabilities = matrix_capabilities();
    capabilities.max_workgroups_x = 3;
    for (uint32_t parallelism : {1u, 256u})
    {
        capabilities.device_parallelism = parallelism;
        const auto result = planned(capabilities, {2, 1, 64, 513, true});
        plan_check(result.path == AttentionDecodePath::Matrix && result.split_count > 0
                       && result.split_count <= result.scratch_split_capacity
                       && result.scratch_split_capacity <= capabilities.max_workgroups_x,
                   "split planner exceeded the device dispatch limit");
    }
    plan_check(planned(capabilities, {40, 2, 64, 513, true}).path == AttentionDecodePath::Scalar,
               "reducer exceeded the device dispatch limit");
    capabilities.max_workgroups_x = 65535;
    capabilities.max_workgroups_y = 1;
    plan_check(planned(capabilities, {80, 2, 64, 513, true}).path == AttentionDecodePath::Scalar,
               "query tiles exceeded the device dispatch limit");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::rejection_cases();
        ncnn::moe::capability_selection();
        ncnn::moe::shape_and_memory_cases();
        ncnn::moe::hardware_parallelism_cases();
        std::cout << "Attention decode capability selection, GQA grouping, split planning and monotonic workspace passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
