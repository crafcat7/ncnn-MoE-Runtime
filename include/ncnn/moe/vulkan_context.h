#ifndef NCNN_MOE_VULKAN_CONTEXT_H
#define NCNN_MOE_VULKAN_CONTEXT_H

#include <memory>

namespace ncnn {
namespace moe {

// Opaque ownership token for one model/runtime Vulkan resource domain.
// Operators created with the same token share command resources; different
// tokens never share mutable backend state.
class NcnnVulkanContextInstance;
using NcnnVulkanContextInstancePtr =
    std::shared_ptr<NcnnVulkanContextInstance>;

[[nodiscard]] NcnnVulkanContextInstancePtr
create_ncnn_vulkan_context_instance();

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKAN_CONTEXT_H
