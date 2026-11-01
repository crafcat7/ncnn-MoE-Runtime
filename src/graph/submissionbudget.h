#ifndef NCNN_MOE_SUBMISSIONBUDGET_H
#define NCNN_MOE_SUBMISSIONBUDGET_H

#include <algorithm>
#include <cmath>

namespace ncnn {
namespace moe {

// Start execution early, then amortize queue submissions over larger batches.
// Work is estimated from compiled tensor dimensions, independently of model
// identity. A boundary never implies a fence or a shorter resource lifetime.
class GpuSubmissionBudget
{
public:
    explicit GpuSubmissionBudget(double total_work) noexcept
        : target(std::clamp(std::isfinite(total_work) ? total_work / 40.0 : max_work, 1.0, max_work))
    {
    }

    [[nodiscard]] bool append(double work) noexcept
    {
        pending += std::max(1.0, work);
        if (pending < target) return false;
        pending = 0;
        if (growth_steps < 3)
        {
            target = std::min(max_work, target * 2.0);
            ++growth_steps;
        }
        return true;
    }

private:
    static constexpr double max_work = 200e9;
    double target, pending = 0;
    unsigned growth_steps = 0;
};

} // namespace moe
} // namespace ncnn
#endif
