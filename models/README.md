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
| Qwen3.6-35B-A3B text backbone | Official Hugging Face Safetensors | BF16 dense/shared weights; BF16 routed Experts or optional compiled MXFP4 Artifact | [Execution and performance](qwen3.6/README.md) |

## Execution capability

| Capability | GPT-OSS | DeepSeek V4 Flash | DeepSeek V4 Flash DSpark | Qwen3.6-35B-A3B |
| --- | ---: | ---: | ---: | ---: |
| Direct model loading | Verified | Verified | Verified | Verified, text weights only |
| Portable CPU backend | Yes | Yes | Yes | Yes |
| ncnn/Vulkan mixed backend | Dense projections and Attention | FP8 Dense projections; CPU Attention cache logic and Experts | Same | Dense projections; CPU recurrent/Attention state and Experts |
| Native low-bit routed Experts | MXFP4 | FP4 E2M1 | FP4 E2M1 | Optional compiled MXFP4 CPU path |
| Attention | Full/sliding GQA | CSA/MLA, window + compressed history | Same | Gated DeltaNet + gated full GQA |
| Hyper-connections | No | mHC | mHC | No |
| Hash routing and shared Expert | No | Yes | Yes | Gated shared Expert; no hash routing |
| Bounded on-demand Expert cache | Yes | Yes | Yes | With the optional Artifact; BF16 uses OS-managed mappings |
| Dense mmap | Yes | Yes | Yes | Yes |
| Optional on-demand Expert mmap | Yes | Yes | Yes | Artifact supports the Expert cache path; BF16 mappings are direct |
| Text prompt wrapper | Harmony | Official checkpoint encoding and tokenizer | Same | Official tokenizer and chat template |
| Model-provided speculative execution | No | No | DSpark | One-layer MTP with the compiled Artifact |
| Vision input | No | No | No | Not admitted |

`MoeIR` is model-neutral. Family-specific recognition and tensor names remain
inside the production adapters; Prefill, Decode, routing, Expert storage, and
scheduling consume compiled plans.
