#include "ncnn/moe/modeladapter.h"
#include "ncnn/moe/modeldescriptor.h"
#include "ncnn/moe/model.h"
#include "ncnn/moe/option.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/runtime.h"
#include "ncnn/moe/scheduler.h"
#include "ncnn/moe/session.h"
#include "ncnn/moe/types.h"

#include <iostream>
#include <type_traits>
#include <utility>

namespace ncnn {
namespace moe {

static_assert(std::is_same_v<
              decltype(std::declval<ModelAdapter&>().parse_model(
                  std::declval<const ModelPackage&>())),
              Result<MoeModelDescriptor>>);
static_assert(std::is_same_v<
              decltype(std::declval<ModelAdapter&>().map_weights(
                  std::declval<const ModelPackage&>(),
                  std::declval<const MoeModelDescriptor&>())),
              Result<WeightMapping>>);

static int test_public_api()
{
    Runtime runtime;
    ModelPtr model;
    auto session = runtime.create_session(model);
    if (session || session.error().code != ErrorCode::InvalidArgument)
    {
        std::cerr << "creating a session without a model must fail\n";
        return 1;
    }

    auto sync = runtime.synchronize_model_caches(model);
    if (sync || sync.error().code != ErrorCode::InvalidArgument)
    {
        std::cerr << "synchronizing a null model must fail\n";
        return 1;
    }

    SchedulerOptions opt;
    opt.num_threads = 1;
    auto ret = runtime.create_scheduler(opt);
    if (!ret)
    {
        std::cerr << ret.error().message << '\n';
        return 1;
    }

    BatchSchedulerPtr scheduler = std::move(ret).value();
    auto prefill = scheduler->submit_prefill({}).get();
    if (!prefill.empty())
    {
        std::cerr << "an empty prefill batch must return no results\n";
        return 1;
    }

    auto decode = scheduler->submit_decode({DecodeBatchRequest{nullptr, 0}}).get();
    if (decode.size() != 1 || decode.front() || decode.front().error().code != ErrorCode::InvalidArgument)
    {
        std::cerr << "a decode request without a session must fail\n";
        return 1;
    }

    const SchedulerStatistics stats = scheduler->statistics();
    if (stats.num_threads != 1 || stats.prefill_batches != 1 || stats.decode_batches != 1)
    {
        std::cerr << "scheduler statistics must reflect the submitted batches\n";
        return 1;
    }
    return 0;
}

} // namespace moe
} // namespace ncnn

int main()
{
    return ncnn::moe::test_public_api();
}
