# ncnn MoE Runtime

A compact C++20 Mixture-of-Experts inference runtime built on
[Tencent/ncnn](https://github.com/Tencent/ncnn). It loads official GPT-OSS
Safetensors directly, keeps dense operators eligible for Vulkan, and executes
routed MXFP4 Experts with optimized CPU kernels.

The release path targets practical local inference for
`openai/gpt-oss-20b` and `openai/gpt-oss-120b` without GGUF conversion or a
private checkpoint format.

## GPT-OSS-120B on 32 GB RAM + 16 GB VRAM

The official 60.7678 GiB checkpoint is verified on a Ryzen 7 9800X3D,
31.14 GiB RAM, and RTX 5070 Ti 16 GiB Windows host. Dense weights are
memory-mapped, while routed Expert pairs enter a byte-bounded host cache from
their original Safetensors shards.

| Metric | Result |
| --- | ---: |
| Decode throughput | **1.902 token/s** |
| 32-token generation | **16.822 s** |
| Expert-cache hit rate | 70.29% |
| Peak process working set | 19,815 MiB |
| Peak total NVIDIA memory | 12,029 MiB |

Protocol: Windows Release build, Vulkan dense/Attention, CPU AVX-512 MXFP4
Experts, 24 GiB host budget, 16 GiB Expert cache, four I/O workers, one warm-up,
and the median of three measured runs with identical token output.

## What works

| Area | Release capability |
| --- | --- |
| Models | Official GPT-OSS-20B/120B Safetensors |
| Graph | Model-neutral IR, validation, backend placement, and dependency scheduling |
| Dense path | Portable CPU, optional ncnn CPU operators, and Vulkan mixed execution |
| Attention | RMSNorm, GQA, learned sinks, full/sliding Attention, YaRN RoPE, BF16/FP32 KV cache |
| Experts | Float32, INT8, and native MXFP4 with NEON, AVX2/FMA, AVX-512, and scalar kernels |
| Storage | Automatic dense/eager mmap and bounded asynchronous Expert caching |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, and streaming |
| Sessions | Independent KV state with bounded CPU/OpenMP decode concurrency |

## Quick start

Requirements: a C++20 compiler, CMake 3.20 or newer, Python 3, and Git.

```powershell
git clone --recurse-submodules https://github.com/crafcat7/ncnn-MoE-Runtime.git
cd ncnn-MoE-Runtime
cmake -S . -B build-ncnn
cmake --build build-ncnn --config Release --parallel
ctest --test-dir build-ncnn -C Release --output-on-failure
```

For a single-config generator:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The pinned ncnn submodule is built with the runtime. Restore it with
`git submodule update --init --recursive` when necessary. A CPU-only build can
be selected with `-DNCNN_MOE_USE_VULKAN=OFF`.

## Runtime design

The architecture follows four explicit responsibility boundaries:

| Boundary | Ownership |
| --- | --- |
| Runtime | Model lifetime, Sessions, generation, sampling, and scheduling |
| Graph | Adapter IR, validation, memory planning, backend placement, and execution waves |
| Backends | Portable CPU kernels and ncnn CPU/Vulkan operator blocks |
| Model storage | Family adapters, Safetensors metadata, mmap, and Expert caches |

```text
official model package
        |
        v
 model adapter --> MoeIR --> ModelCompiler --> CompiledModel
                                                   |
                                             Session state
                                                   |
                                      +------------+------------+
                                      |                         |
                                CPU execution              ncnn / Vulkan
                           routing + MXFP4 Experts       dense + Attention
```

Adapters translate model-family metadata and tensor names into `MoeIR`.
`ModelCompiler` resolves weights, validates the graph, and creates an immutable
compiled plan. `Session` owns mutable KV cache, sampling state, and statistics.
Configuration switches use typed, domain-specific `uint32_t` flag groups.

## Models

Model execution guides live with their model-family definitions:

- [Model catalog and capability matrix](models/README.md)
- [GPT-OSS-20B/120B execution and performance](models/gpt-oss/README.md)

## Project layout

```text
include/ncnn/moe/  Installed API, IR, graph, plans, and model descriptors
src/engine/        Runtime, Sessions, scheduling, and CPU execution lifecycle
src/graph/         IR lowering, validation, memory planning, and execution graph
src/models/        GPT-OSS adapter, Safetensors loading, and canonical tensor names
src/storage/       mmap, system-memory discovery, and Expert cache residency
src/kernels/       Portable CPU Attention, Linear, and MXFP4 kernels
src/backends/ncnn/ ncnn CPU/Vulkan packaging and mixed Attention backend
models/            Model catalog and model-family execution guides
examples/          Model command-line runner
tools/             Fixture, Harmony, and benchmark utilities
tests/             Deterministic, parity, concurrency, and error-path tests
third_party/ncnn/  Pinned ncnn source submodule
```

Private source files are grouped by owning responsibility and retain direct,
ncnn-style names within each directory.

## License

See [LICENSE](LICENSE). This project builds on the pinned
[Tencent/ncnn](https://github.com/Tencent/ncnn) submodule.
