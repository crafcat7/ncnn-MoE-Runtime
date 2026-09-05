#include "internal/modelrunner.h"

int main(int argc, char** argv)
{
    return ncnn::moe::run_model_example(
        argc,
        argv,
        {"ncnn_moe_deepseek_v4", "deepseek_v4", 1});
}
