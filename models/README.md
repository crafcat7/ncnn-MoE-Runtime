# Model Catalog

`models/` is the public entry point for model compatibility and execution
instructions. Each model family owns its checkpoint requirements, supported
execution paths, runtime options, validation commands, and measured
performance. Downloaded model files stay under the ignored model-family
directories in `models/` and are not version-controlled.

## Supported models

| Model family | Package | Weights | Execution guide |
| --- | --- | --- | --- |
| GPT-OSS-20B | Official Hugging Face Safetensors | BF16 dense, native MXFP4 Experts | [GPT-OSS](gpt-oss/README.md) |
| GPT-OSS-120B | Official Hugging Face Safetensors | BF16 dense, native MXFP4 Experts | [GPT-OSS](gpt-oss/README.md) |

## Execution capability

| Capability | GPT-OSS |
| --- | ---: |
| Direct model loading | Yes |
| Portable CPU backend | Yes |
| ncnn/Vulkan mixed backend | Dense projections and Attention |
| Native MXFP4 Experts | Yes |
| Full/sliding Attention and KV cache | Yes |
| Eager Expert residency | Yes |
| Bounded on-demand Expert cache | Yes |
| Dense/eager mmap | Yes |
| Optional on-demand Expert mmap | Yes |
| Text prompt wrapper | Harmony |
| Repeatable performance benchmark | Yes |
