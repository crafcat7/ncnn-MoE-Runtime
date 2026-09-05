#ifndef NCNN_MOE_MODELRUNNER_H
#define NCNN_MOE_MODELRUNNER_H

#include <cstdint>

namespace ncnn {
namespace moe {

struct ExampleRunnerOptions
{
    const char* executable_name = nullptr;
    const char* expected_model_type = nullptr;
    int32_t default_stop_token = -1;
    int32_t secondary_default_stop_token = -1;
    bool use_speculative = true;
};

int run_model_example(int argc, char** argv, const ExampleRunnerOptions& runner_options);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELRUNNER_H
