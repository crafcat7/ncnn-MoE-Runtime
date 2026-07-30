# Model Catalog

`models/` is the public entry point for model compatibility and execution
instructions. Each model family owns its checkpoint requirements, supported
execution paths, runtime options, and validation commands. Downloaded model
files stay under the ignored model-family
directories in `models/` and are not version-controlled.

## Supported models

| Model family | Package | Weights | Execution and performance |
| --- | --- | --- | --- |
| GPT-OSS-20B | Official Hugging Face Safetensors | BF16 dense, native MXFP4 Experts | [GPT-OSS](gpt-oss/README.md#reference-performance) |
| GPT-OSS-120B | Official Hugging Face Safetensors | BF16 dense, native MXFP4 Experts | [GPT-OSS](gpt-oss/README.md#reference-performance) |
| DeepSeek-V4-Flash | Official Hugging Face Safetensors | FP8 dense/shared, FP4 routed Experts | [DeepSeek V4 Flash](deepseek-v4/README.md#deepseek-v4-flash-matrix) |
| DeepSeek-V4-Flash-DSpark | Official Hugging Face Safetensors | FP8 dense/shared, FP4 routed Experts | [DeepSeek V4 Flash DSpark](deepseek-v4/README.md#deepseek-v4-flash-dspark-matrix) |

## Execution capability

| Capability | GPT-OSS | DeepSeek V4 Flash | DeepSeek V4 Flash DSpark |
| --- | ---: | ---: | ---: |
| Direct model loading | Verified | Verified | Verified |
| Portable CPU backend | Yes | Yes | Yes |
| ncnn/Vulkan mixed backend | Dense projections and Attention | FP8 Dense projections; CPU Attention cache logic and Experts | Same |
| Native low-bit routed Experts | MXFP4 | FP4 E2M1 | FP4 E2M1 |
| Attention | Full/sliding GQA | CSA/MLA, window + compressed history | Same |
| Hyper-connections | No | mHC | mHC |
| Hash routing and shared Expert | No | Yes | Yes |
| Bounded on-demand Expert cache | Yes | Yes | Yes |
| Dense mmap | Yes | Yes | Yes |
| Optional on-demand Expert mmap | Yes | Yes | Yes |
| Text prompt wrapper | Harmony | Official checkpoint encoding and tokenizer | Same |
| DSpark speculative execution | n/a | No | Implemented; matrix requires proposal and draft activity |

`MoeIR` is model-neutral. Family-specific recognition and tensor names remain
inside the production adapters; Prefill, Decode, routing, Expert storage, and
scheduling consume compiled plans.
