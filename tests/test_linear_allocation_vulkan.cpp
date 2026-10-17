#include "backends/ncnn/innerproductweights_vulkan.h"
#include "backends/ncnn/linear.h"

#include <allocator.h>

#include <iostream>
#include <limits>
#include <stdexcept>

static void allocation_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int main()
{
    try
    {
        ncnn::moe::TensorData matrix;
        matrix.dtype = ncnn::moe::DType::BFloat16;
        matrix.shape = {248320, 5120};
        const uint64_t payload = uint64_t{248320} * 5120 * sizeof(uint16_t);
        const uint64_t storage = ncnn::moe::Bfloat16Linear_vulkan::storage_bytes(matrix, 256);
        allocation_check(storage > payload && storage - payload < 2 * 1024 * 1024,
                         "LM head auxiliary buffers must not reserve additional weight-sized blocks");
        matrix.shape = {4, 4};
        allocation_check(ncnn::moe::Bfloat16Linear_vulkan::storage_bytes(matrix, 256) == 3 * 64 * 1024,
                         "small projections must include ordinary block slack");
        allocation_check(ncnn::moe::Bfloat16Linear_vulkan::storage_bytes(matrix, std::numeric_limits<uint64_t>::max()) == 0,
                         "overflowing allocation alignment was accepted");
        matrix.shape = {4, 6};
        allocation_check(ncnn::moe::Bfloat16Linear_vulkan::storage_bytes(matrix) == 0,
                         "unsupported projection input alignment was accepted");
        matrix.shape = {std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
        allocation_check(ncnn::moe::Bfloat16Linear_vulkan::storage_bytes(matrix) == 0,
                         "unsupported projection extent was accepted");
        matrix.shape = {4, 4};
        matrix.dtype = ncnn::moe::DType::Float32;
        allocation_check(ncnn::moe::Bfloat16Linear_vulkan::storage_bytes(matrix) == 0,
                         "non-BF16 projection storage was accepted");

        ncnn::VkBufferMemory weight_storage{};
        ncnn::VkBufferMemory bias_storage{};
        ncnn::InnerProduct_vulkan layer;
        allocation_check(!ncnn::moe::inner_product_weights_available(layer, false), "missing weights were accepted");
        allocation_check(!ncnn::moe::inner_product_weights_available(layer, true), "missing weights and bias were accepted");

        // External storage exercises allocation publication without a GPU or a
        // command submission. These buffers are never accessed by a shader.
        layer.weight_data_gpu = ncnn::VkMat(1, &weight_storage, sizeof(float), nullptr);
        allocation_check(ncnn::moe::inner_product_weights_available(layer, false), "uploaded bias-free weights were rejected");
        allocation_check(!ncnn::moe::inner_product_weights_available(layer, true), "missing required bias was accepted");

        layer.bias_data_gpu = ncnn::VkMat(1, &bias_storage, sizeof(float), nullptr);
        allocation_check(ncnn::moe::inner_product_weights_available(layer, true), "uploaded weights and bias were rejected");
        layer.weight_data_gpu.release();
        allocation_check(!ncnn::moe::inner_product_weights_available(layer, true), "bias-only allocation was accepted");
        std::cout << "Vulkan linear weight allocation validation passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan linear allocation test failed: " << error.what() << '\n';
        return 1;
    }
}
