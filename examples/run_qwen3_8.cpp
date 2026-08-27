#include "internal/run_model.h"

int main(int argc, char** argv)
{
    return ncnn::moe::run_model_example(
        argc,
        argv,
        {"ncnn_moe_qwen3_8", "qwen4_exp", 248046, 248044, false});
}
