#include "ncnn/moe/runtime.h"

#include "ncnn/moe/execution_plan.h"
#include "moe_adapter.h"
#include "ncnn_linear.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

static Result<std::string> read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open manifest: " + path.string()};

    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof())
        return Error{ErrorCode::IoError, "cannot read manifest: " + path.string()};
    return contents.str();
}

static Result<std::string> required_json_string(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "manifest is missing string field: " + key};
    return match[1].str();
}

Runtime::Runtime()
{
#if defined(NCNN_MOE_USE_NCNN) && NCNN_MOE_USE_NCNN
    capabilities_.ncnn_cpu_linear = true;
#endif
#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    capabilities_.vulkan_device_count = NcnnLinearOperator::vulkan_device_count();
    capabilities_.vulkan_execution = capabilities_.vulkan_device_count > 0;
#endif
    capabilities_.vulkan_cpu_mix = capabilities_.vulkan_execution;
    capabilities_.vulkan_cpu_prefetch = capabilities_.vulkan_cpu_mix;
    capabilities_.vulkan_attention = capabilities_.vulkan_cpu_mix;
    register_adapter(std::make_shared<MoeAdapter>());
}

void Runtime::register_adapter(std::shared_ptr<IMoeModelAdapter> adapter)
{
    if (adapter)
        adapters_.push_back(std::move(adapter));
}

Result<ModelPtr> Runtime::load_model(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options)
{
    HybridMode resolved_mode = options.hybrid_mode;
    if (resolved_mode == HybridMode::Auto)
        resolved_mode = capabilities_.vulkan_cpu_mix ? HybridMode::HybridExperts : HybridMode::CpuOnly;
    if (resolved_mode == HybridMode::VulkanOnly)
        return Error{ErrorCode::UnsupportedModel, "Vulkan-only execution is not implemented; use HybridExperts"};
    if (resolved_mode == HybridMode::HybridExperts && !capabilities_.vulkan_cpu_mix)
        return Error{ErrorCode::UnsupportedModel, "HybridExperts requires a Vulkan device and Vulkan-enabled ncnn"};
    if (resolved_mode == HybridMode::VulkanWithCpuPrefetch && !capabilities_.vulkan_cpu_prefetch)
        return Error{ErrorCode::UnsupportedModel, "VulkanWithCpuPrefetch requires a Vulkan device and Vulkan-enabled ncnn"};

    std::filesystem::path root = model_path;
    std::filesystem::path manifest_path;
    std::error_code filesystem_error;
    if (std::filesystem::is_directory(model_path, filesystem_error)) {
        manifest_path = model_path / "model.ncnnmoe.json";
        if (!std::filesystem::exists(manifest_path, filesystem_error))
            manifest_path = model_path / "config.json";
    }
    else {
        manifest_path = model_path;
        root = model_path.parent_path();
    }

    auto manifest_text = read_text_file(manifest_path);
    if (!manifest_text)
        return manifest_text.error();

    auto model_type = required_json_string(manifest_text.value(), "model_type");
    if (!model_type)
        return model_type.error();

    ModelPackage package{
        root,
        ModelManifest{std::move(model_type).value(), std::move(manifest_text).value()}};

    const IMoeModelAdapter* selected_adapter = nullptr;
    for (const auto& adapter : adapters_) {
        if (adapter->can_load(package.manifest)) {
            selected_adapter = adapter.get();
            break;
        }
    }
    if (!selected_adapter)
        return Error{ErrorCode::UnsupportedModel, "no adapter registered for model_type: " + package.manifest.model_type};

    auto descriptor = selected_adapter->parse_model(package);
    if (!descriptor)
        return descriptor.error();

    auto weights = selected_adapter->map_weights(package, descriptor.value());
    if (!weights)
        return weights.error();

    ModelCompiler compiler;
    auto compiled = compiler.compile(
        std::move(descriptor).value(),
        std::move(weights).value(),
        resolved_mode);
    if (!compiled)
        return compiled.error();

    auto immutable = std::make_shared<const CompiledModel>(std::move(compiled).value());
    return ModelPtr(new Model(std::move(immutable)));
}

Result<SessionPtr> Runtime::create_session(const ModelPtr& model, const SessionOptions& options)
{
    if (!model)
        return Error{ErrorCode::InvalidArgument, "model cannot be null"};
    if (options.logits_output_mode != LogitsOutputMode::FullLogits)
        return Error{ErrorCode::UnsupportedModel, "the current executor supports full logits output only"};
    return SessionPtr(new Session(model, options));
}

} // namespace moe
} // namespace ncnn
