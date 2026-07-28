# Model Catalog

`models/` is the public entry point for model compatibility and execution
instructions. Each model family owns its checkpoint requirements, supported
execution paths, runtime options, and validation commands. Downloaded model
files stay under the ignored model-family
directories in `models/` and are not version-controlled.

## Supported models

| Model family | Package | Weights | Execution guide |
| --- | --- | --- | --- |
| GPT-OSS-20B | Official Hugging Face Safetensors | BF16 dense, native MXFP4 Experts | [GPT-OSS](gpt-oss/README.md) |
| GPT-OSS-120B | Official Hugging Face Safetensors | BF16 dense, native MXFP4 Experts | [GPT-OSS](gpt-oss/README.md) |
| DeepSeek-V4-Flash-DSpark | Official Hugging Face Safetensors | FP8 dense/shared, FP4 routed Experts | [DeepSeek V4](deepseek-v4/README.md) |

## Execution capability

| Capability | GPT-OSS | DeepSeek V4 Flash DSpark |
| --- | ---: | ---: |
| Direct model loading | Verified | Verified |
| Portable CPU backend | Yes | Yes |
| ncnn/Vulkan mixed backend | Dense projections and Attention | FP8 Dense projections; CPU Attention cache logic and Experts |
| Native low-bit routed Experts | MXFP4 | FP4 E2M1 |
| Attention | Full/sliding GQA | CSA/MLA, window + compressed history |
| Hyper-connections | No | mHC |
| Hash routing and shared Expert | No | Yes |
| Bounded on-demand Expert cache | Yes | Yes |
| Dense mmap | Yes | Yes |
| Optional on-demand Expert mmap | Yes | Yes |
| Text prompt wrapper | Harmony | Official checkpoint encoding and tokenizer |
| DSpark speculative execution | n/a | Implemented; measured speedup remains workload-dependent |

`MoeIR` is model-neutral. Family-specific recognition and tensor names remain
inside the production adapters; Prefill, Decode, routing, Expert storage, and
scheduling consume compiled plans.
