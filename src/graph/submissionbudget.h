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

    // Ask before recording the next indivisible stage. An oversized first stage
    // still has to be recorded: submitting an empty command cannot make progress.
    [[nodiscard]] bool would_flush_before(double next_work) const noexcept
    {
        return pending > 0 && pending + bounded_work(next_work) >= target;
    }

    void record(double work) noexcept
    {
        pending = std::min(max_work, pending + bounded_work(work));
    }

    // Call only after a successful queue submission. Merely considering a
    // boundary, or choosing to retain the tail, must not advance the budget.
    void submitted() noexcept
    {
        if (pending == 0) return;
        pending = 0;
        if (growth_steps < 3)
        {
            target = std::min(max_work, target * 2.0);
            ++growth_steps;
        }
    }

    // Compatibility for callers that decide boundaries after recording work.
    // Such callers historically acknowledge the boundary immediately.
    [[nodiscard]] bool append(double work) noexcept
    {
        record(work);
        if (pending < target) return false;
        submitted();
        return true;
    }

private:
    static constexpr double max_work = 200e9;

    static double bounded_work(double work) noexcept
    {
        // Keep pending finite even when an estimate overflows. A missing,
        // negative, or NaN estimate still represents one recorded stage.
        return std::min(max_work, std::max(1.0, work));
    }

    double target, pending = 0;
    unsigned growth_steps = 0;
};

} // namespace moe
} // namespace ncnn
#endif
