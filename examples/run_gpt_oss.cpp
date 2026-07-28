#include "internal/run_model.h"

int main(int argc, char** argv)
{
    return ncnn::moe::run_model_example(
        argc,
        argv,
        {"ncnn_moe_gpt_oss", "gpt_oss"});
}
