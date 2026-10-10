#ifndef NCNN_MOE_EXPERTRESIDENCY_H
#define NCNN_MOE_EXPERTRESIDENCY_H

#include <memory>
#include <string_view>

namespace ncnn {
namespace moe {

// Advisory ready-copy registry shared by the host and device caches. It
// never calls either cache; all cache locks may safely precede this registry's
// mutex. Queries are snapshots, so neither cache relies on another copy for
// correctness and ordinary file-backed reload remains available.
class ExpertResidencyCoordinator
{
public:
    ExpertResidencyCoordinator();
    ~ExpertResidencyCoordinator();

    ExpertResidencyCoordinator(const ExpertResidencyCoordinator&) = delete;
    ExpertResidencyCoordinator& operator=(const ExpertResidencyCoordinator&) = delete;

    // Publish only successfully loaded/uploaded ready copies. A registration
    // is removed when the last owner of its token releases it; tokens retain
    // their registry state safely beyond this coordinator's lifetime. Empty
    // tokens on allocation failure disable only the advisory eviction hint.
    [[nodiscard]] std::shared_ptr<const void> register_host(std::string_view key) noexcept;
    [[nodiscard]] std::shared_ptr<const void> register_device(std::string_view key) noexcept;
    [[nodiscard]] bool host_resident(std::string_view key) const;
    [[nodiscard]] bool device_resident(std::string_view key) const;

private:
    struct State;
    [[nodiscard]] std::shared_ptr<const void> register_copy(std::string_view key, bool device) noexcept;
    std::shared_ptr<State> state;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERTRESIDENCY_H
