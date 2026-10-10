#include "expertresidency.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace ncnn {
namespace moe {

struct ExpertResidencyCoordinator::State
{
    struct CopyCounts
    {
        uint64_t host = 0;
        uint64_t device = 0;
    };

    struct StringHash
    {
        using is_transparent = void;

        size_t operator()(std::string_view value) const noexcept
        {
            return std::hash<std::string_view>{}(value);
        }
    };

    struct Registration
    {
        Registration(std::shared_ptr<State> _owner, std::string_view _key, bool _device)
            : owner(std::move(_owner)), key(_key), device(_device)
        {
        }

        ~Registration()
        {
            if (!active)
                return;
            const std::lock_guard<std::mutex> lock(owner->mutex);
            const auto existing = owner->copies.find(key);
            if (existing == owner->copies.end())
                return;
            uint64_t& count = device ? existing->second.device : existing->second.host;
            --count;
            if (existing->second.host == 0 && existing->second.device == 0)
                owner->copies.erase(existing);
        }

        std::shared_ptr<State> owner;
        std::string key;
        bool device = false;
        bool active = false;
    };

    mutable std::mutex mutex;
    std::unordered_map<std::string, CopyCounts, StringHash, std::equal_to<>> copies;
};

ExpertResidencyCoordinator::ExpertResidencyCoordinator()
    : state(std::make_shared<State>())
{
}

ExpertResidencyCoordinator::~ExpertResidencyCoordinator() = default;

std::shared_ptr<const void> ExpertResidencyCoordinator::register_copy(std::string_view key, bool device) noexcept
{
    if (key.empty())
        return {};
    try
    {
        auto registration = std::make_shared<State::Registration>(state, key, device);
        const std::lock_guard<std::mutex> lock(state->mutex);
        auto existing = state->copies.try_emplace(registration->key);
        uint64_t& count = device ? existing.first->second.device : existing.first->second.host;
        ++count;
        registration->active = true;
        return registration;
    }
    catch (...)
    {
        return {};
    }
}

std::shared_ptr<const void> ExpertResidencyCoordinator::register_host(std::string_view key) noexcept
{
    return register_copy(key, false);
}

std::shared_ptr<const void> ExpertResidencyCoordinator::register_device(std::string_view key) noexcept
{
    return register_copy(key, true);
}

bool ExpertResidencyCoordinator::host_resident(std::string_view key) const
{
    const std::lock_guard<std::mutex> lock(state->mutex);
    const auto existing = state->copies.find(key);
    return existing != state->copies.end() && existing->second.host != 0;
}

bool ExpertResidencyCoordinator::device_resident(std::string_view key) const
{
    const std::lock_guard<std::mutex> lock(state->mutex);
    const auto existing = state->copies.find(key);
    return existing != state->copies.end() && existing->second.device != 0;
}

} // namespace moe
} // namespace ncnn
