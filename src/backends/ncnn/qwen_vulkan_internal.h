#ifndef NCNN_MOE_QWEN_VULKAN_INTERNAL_H
#define NCNN_MOE_QWEN_VULKAN_INTERNAL_H
#include "vulkancontext.h"
#if NCNN_MOE_WITH_VULKAN
#include <initializer_list>
#include <mat.h>
#include <memory>
namespace ncnn {
class Pipeline;
class VkCompute;
class Option;
namespace moe {
struct TensorData;
std::shared_ptr<ncnn::Pipeline> qwen_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& opt);
void qwen_record(const ncnn::Pipeline* pipeline, uint32_t mode, uint32_t hidden, uint32_t copies,
                 uint32_t rows, uint32_t columns, float epsilon, float offset, std::initializer_list<ncnn::VkMat> values,
                 ncnn::VkCompute& cmd, uint32_t kernel = 0, uint32_t ngram = 0, uint32_t length = 0, uint32_t slot = 0);
bool qwen_allocate(ncnn::VkMat& value, uint32_t columns, int rows, const std::shared_ptr<VulkanContext>& context);
bool qwen_float_weight(const TensorData& tensor, ncnn::VkMat& value, ncnn::VkCompute& command, const ncnn::Option& opt);
void qwen_prepare_read(const ncnn::VkMat& value);
} // namespace moe
} // namespace ncnn
#endif
#endif
