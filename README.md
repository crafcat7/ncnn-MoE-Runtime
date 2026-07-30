# ncnn MoE Runtime

>❗This software is in the early development stage

ncnn MoE Runtime is a lightweight, high-performance C++20 inference engine for
sparse Mixture-of-Experts models, built on
[Tencent/ncnn](https://github.com/Tencent/ncnn). Model adapters lower external
packages into a model-neutral `MoeIR`; the Runtime compiles that IR into an
execution graph and schedules Dense Transformer work, routing, and sparse
Experts across CPU, Vulkan, memory, and storage backends.

The runtime targets local inference on Desktop and edge-class systems where the
complete model may be larger than available RAM. The public API is independent
of a model family, device vendor, and fixed cache size. The distributed release
includes validated GPT-OSS, DeepSeek-V4-Flash, and
DeepSeek-V4-Flash-DSpark support for their official Safetensors packages.

[Runtime capabilities](#runtime-capabilities) | [Architecture](#architecture) |
[Supported models and data](#supported-models-and-reference-data) |
[Quick start](#quick-start)

## Supported models and reference data

The table is a compact index of the current reference measurements. Values
are three-run medians on a Ryzen 7 9800X3D, 31.14 GiB RAM, and an RTX 5070 Ti
16 GiB. Each model report owns the complete cold/warm, short/long,
multi-Session, memory, cache, and I/O matrix.

| Model | Reference execution | Warm single Session, 32 tokens | Warm service, 32 tokens | Full report |
| --- | --- | ---: | ---: | --- |
| GPT-OSS-20B | Hybrid, eager Experts | **14.855 token/s** | n/a | [GPT-OSS performance](models/gpt-oss/README.md#reference-performance) |
| GPT-OSS-120B | Hybrid, 16 GiB host ARC | **4.718 token/s** | **10.916 aggregate token/s** (2 Sessions) | [GPT-OSS performance](models/gpt-oss/README.md#reference-performance) |
| DeepSeek-V4-Flash | Hybrid, 16 GiB host ARC | **1.254 token/s** | **3.809 aggregate token/s** (4 staged Sessions) | [Flash performance](models/deepseek-v4/README.md#deepseek-v4-flash-matrix) |
| DeepSeek-V4-Flash-DSpark | Hybrid, DSpark enabled, 16 GiB host ARC | **1.254 token/s** | **3.594 aggregate token/s** (4 independent Sessions) | [DSpark performance](models/deepseek-v4/README.md#deepseek-v4-flash-dspark-matrix) |

Model-specific prompts, kernels, quantization formats, and service policies
differ, so these rows are reference points rather than a cross-model ranking.

## Sparse models beyond resident memory

Large MoE models contain many Expert weights while each token activates only a
small routed subset. ncnn MoE Runtime treats Experts as schedulable,
cacheable Runtime objects instead of requiring every Expert to remain resident.
Dense tensors can stay memory-mapped, while active Expert pairs move through
bounded host and device cache tiers backed by asynchronous range I/O.

| Runtime concern | Mechanism |
| --- | --- |
| Model integration | `IMoeModelAdapter` lowers package metadata and weights into `MoeIR` |
| Compilation | Validation, normalization, weight resolution, graph construction, and backend placement |
| Dense Transformer | Portable CPU, ncnn CPU operators, or ncnn Vulkan |
| Sparse Experts | Stable Top-K grouping, batched execution, and runtime-selected kernels |
| Weight residency | Automatic eager or byte-bounded on-demand Expert storage |
| Cache policy | Host ARC with independent optional Vulkan execution and victim tiers |
| Scheduling | Dependency waves, backend events, and cross-Session micro-batching |

The Runtime therefore scales with the active Expert working set rather than
requiring a resident copy of every routed weight.

## Runtime capabilities

| Area | Release capability |
| --- | --- |
| Runtime API | `Runtime`, immutable shared `Model`, per-request `Session`, and cross-Session `BatchScheduler` |
| Model integration | Public `IMoeModelAdapter` contract and model-neutral `MoeIR`; built-in GPT-OSS and DeepSeek V4 adapters |
| IR and compiler | Model-neutral `MoeIR` for Attention, Router, ExpertGroup, Combine, KV Cache, and quantization metadata; validation, normalization, weight resolution, and immutable compilation |
| Execution graph | Tensor and node dependencies, backend candidates and placement, cross-backend events, and topological execution waves |
| Dense path | Portable CPU with runtime-dispatched FP8 E4M3 scalar/AVX2/AVX-512 Linear, optional ncnn CPU operators, and mixed ncnn Vulkan Dense/Attention execution |
| Attention | RMSNorm, GQA, full/sliding Attention, latent Attention with learned compressed history, RoPE/YaRN variants, sinks, persistent KV state, fused QKV+RoPE, and adaptive online Decode SDPA |
| Experts | Stable Top-K regrouping, Softmax/Sigmoid/square-root-Softplus scoring, hash routes, shared Experts, float32/BF16/FP8/INT8 execution, and fused-decode FP4 kernels selected at runtime for scalar, NEON, SVE2, AVX2/FMA, or AVX-512 |
| Memory and storage | Automatic eager/on-demand planning, per-Session Tensor residency accounting, Expert lifecycle/hotset statistics, byte-bounded host ARC, mmap or asynchronous direct/buffered reads, optional packed Expert storage, and optional Vulkan cache tiers |
| Heterogeneous execution | CPU Experts by default, optional calibrated native Vulkan MXFP4 Experts, and capability-weighted multi-Vulkan layer placement |
| Scheduling | Independent Session state, ragged staged Prefill, mHC/Attention/Expert Decode batching, same-Expert and exact-input coalescing, adaptive staged/independent execution, and bounded cross-call micro-batching |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, streaming, and model-provided speculative plans |

## Why it is fast

- **Heterogeneous placement.** Dense projections and Attention run through
  ncnn Vulkan while routing and sparse Expert work use the CPU; native Vulkan
  Experts are admitted only when phase-level calibration measures a benefit.
- **Fused MXFP4 compute.** Expert kernels decode MXFP4 blocks inside the
  compute loop instead of materializing complete FP32 weights. Runtime dispatch
  selects scalar, NEON, SVE2, AVX2/FMA, or AVX-512 implementations.
- **Vectorized FP8 Linear.** E4M3 block-scale projections select scalar,
  AVX2/FMA, or AVX-512 kernels at runtime, share input loads across adjacent
  output rows, and use a data-type-aware CPU thread limit.
- **Expert-oriented batching.** Stable Top-K regrouping turns routed tokens
  into contiguous per-Expert batches and shares work across concurrent
  Sessions. Bitwise-identical MXFP4 input rows reuse one exact result after a
  hash and equality check.
- **Adaptive weight residency.** A byte-aware ARC, asynchronous exact-range
  reads, optional aligned direct I/O, and the packed Expert sidecar keep the
  active route working set close to compute.
- **Reusable execution state.** Persistent KV rings, reusable scratch buffers,
  direct QKV-to-ring writes, command reuse, and online Decode SDPA reduce
  per-token allocation and transfer overhead.
- **Workload-aware scheduling.** Ragged Prefill, cross-Session collection,
  mHC/Attention/Expert batching, and adaptive staged/independent execution
  improve service throughput without fixing policy choices to one device.

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

## Runtime API

`Runtime` selects a registered adapter for the supplied model package, compiles
its `MoeIR`, creates independent Session state, and exposes Prefill, Decode, or
complete generation:

```cpp
#include "ncnn/moe/runtime.h"

ncnn::moe::Result<ncnn::moe::GenerationResult> run(
    const std::filesystem::path& model_path,
    std::span<const int32_t> input_ids)
{
    ncnn::moe::Runtime runtime;
    ncnn::moe::RuntimeOptions runtime_options;
    runtime_options.hybrid_mode = ncnn::moe::HybridMode::Auto;

    auto model = runtime.load_model(model_path, runtime_options);
    if (!model)
        return model.error();

    auto session = runtime.create_session(model.value());
    if (!session)
        return session.error();

    ncnn::moe::GenerationOptions generation_options;
    generation_options.max_new_tokens = 64;
    return session.value()->generate(input_ids, generation_options);
}
```

Applications add model families that map to the supported `MoeIR` node set
through `IMoeModelAdapter::can_load`, `parse_model`, and `map_weights`;
execution code consumes only the compiled model-neutral representation. The
generic token-ID runner, GPT-OSS Harmony text wrapper, and model-specific
commands are documented in the model guides.

## Architecture

`Runtime Core` is a responsibility layer rather than one monolithic class. The
public API owns application-facing lifetime and request state; adapters and the
compiler produce immutable model state; the execution and memory subsystems
consume that state through explicit contracts.

| Boundary | Responsibility |
| --- | --- |
| MoE Runtime API | Hardware capabilities, model loading, immutable `Model` lifetime, mutable `Session` state, generation, sampling, cache synchronization, and `BatchScheduler` creation |
| Adapter and compiler | Model-package parsing through `IMoeModelAdapter`, model-neutral `MoeIR`, validation and normalization, weight resolution, memory planning, backend placement, and `CompiledModel` construction |
| Execution | `ExecutionGraph` dependencies and Tensor locations, `RuntimeScheduler` backend lanes/events, `MoeScheduler` topological waves, routing, and Expert dispatch |
| Memory | `ModelMemoryPlan`, per-Session `MemoryManager`, runtime `Expert`/`ExpertStore` state, host ARC residency, and optional Vulkan cache tiers |
| Backends | Portable CPU kernels, ncnn CPU/Vulkan Dense and Attention blocks, CPU Expert execution, and the optional native Vulkan MXFP4 Expert backend |
| Model storage | Package metadata and mappings, asynchronous range I/O, packed Expert storage, and cache lifetime ownership |

```text
Application
    |
MoE Runtime API
Runtime / Model / Session / BatchScheduler
    |
Runtime Core
    |
Model Adapter -> MoeIR -> ModelCompiler -> CompiledModel
                                                |
                         +----------------------+----------------------+
                         |                                             |
             ExecutionGraph + RuntimeScheduler          Memory plan + MemoryManager
                         |                               Expert + ExpertStore + ARC
                         +----------------------+----------------------+
                                                |
                            +-------------------+-------------------+
                            |                                       |
                    ncnn CPU/Vulkan                        CPU Expert Backend
                 Dense + Attention + KV              Router + Dispatch + MXFP4
                            |                                       |
                            +--- optional native Vulkan Experts ----+
```

Model adapters translate family-specific metadata and tensor names into
`MoeIR`. `ModelCompiler` resolves weights, validates the graph, and creates an
immutable compiled plan. `ExecutionGraph` records data dependencies, placement
and event metadata; `RuntimeScheduler` turns it into CPU/Vulkan lanes and
waves. `Session` owns mutable KV cache, sampling state, reusable execution
scratch, Tensor-residency accounting, and statistics. `ExpertStore` exposes
Expert lifecycle and hotset information, while the byte-bounded Expert cache
implements ARC recent/frequent resident lists and ghost histories.

Autoregressive dependencies are preserved within each Session. Independent
Sessions can overlap through the batch scheduler. Expert cache admission and
range I/O retain explicit asynchronous lifetimes, while Vulkan commands use
the public upstream ncnn submission API.

Cross-layer Router prediction, forward-aware cache policy, Rank-adaptive
prefetch, a bounded prediction worker, and exact-adjacency cross-Expert reads
are available as experimental opt-ins. They remain disabled by default. In the
current 24-token Direct-I/O triplets, synchronous prediction improves DeepSeek
throughput by 5.93%, while async prediction improves GPT-OSS by 0.88% but loses
to synchronous prediction on DeepSeek because it competes with CPU Experts.
See the
[Palm-Infra transfer report](memories/repo/investigation-results/palm-infra-transfer-experiment.md)
and the historical
[Router/cache/I/O A/B report](memories/repo/investigation-results/router-prefetch-cache-io-ab.md)
for protocols, per-Rank metrics, and acceptance boundaries.

Kernel, scheduling, memory-management, and fallback behavior are provided by
the Runtime API and its public model guides.

## Supported scope

| Capability | Public behavior |
| --- | --- |
| Adapter interface | Public `IMoeModelAdapter` to `MoeIR` lowering contract |
| Built-in reference adapters | GPT-OSS-20B/120B, DeepSeek-V4-Flash, and DeepSeek-V4-Flash-DSpark Safetensors |
| CPU execution | Complete portable path |
| Heterogeneous execution | Vulkan Dense/Attention with CPU routing and Experts |
| Native Vulkan Experts | Optional MXFP4 cache and execution with runtime calibration and CPU fallback |
| Multiple Vulkan devices | Capability-weighted whole-layer placement |
| Expert memory | Automatic eager or byte-bounded on-demand residency |
| KV cache | CPU FP32/BF16 or mixed-backend FP32 ring |
| Output | Full logits with native sampling and streaming |

The distributed release includes validated GPT-OSS and DeepSeek V4 production
adapters. Additional adapters use the same public IR and compiler boundary
without adding model-family checks to Prefill, Decode, scheduling, or Expert
execution.
Multi-device placement is layer placement rather than Tensor Parallelism, and
Vulkan-only execution is not a supported public mode.

## Model adapters

The Runtime core is model-neutral; production package support is supplied by
registered adapters. Model execution guides live with their adapter
definitions:

- [Model catalog and capability matrix](models/README.md)
- [GPT-OSS-20B/120B execution and performance](models/gpt-oss/README.md)
- [DeepSeek V4 Flash and DSpark execution and performance](models/deepseek-v4/README.md)

## Project layout

```text
include/ncnn/moe/  Installed Runtime API, MoeIR, graph, plan, Expert, memory, and scheduler contracts
src/compiler/      MoeIR validation, normalization, descriptor conversion, and graph construction
src/graph/         Model compilation, execution-graph scheduling, Expert dispatch, and memory planning
src/engine/        Runtime Core, Sessions, batch scheduling, MemoryManager, CPU execution, and Expert-backend coordination
src/models/        Built-in model adapters, package loading, packed-sidecar selection, and canonical Tensor names
src/storage/       Mapped files, range I/O, host ARC, Vulkan victim cache, system-memory discovery, and Expert lifecycle
src/kernels/       Portable CPU Attention/Linear, BF16 helpers, and runtime-selected FP8/MXFP4 SIMD kernels
src/backends/ncnn/ ncnn CPU/Vulkan operator packaging, mixed Attention, Vulkan contexts, and native MXFP4 Experts
models/            Model catalog and model-family execution guides
assets/            Published benchmark visualizations used by model reports
examples/          GPT-OSS and DeepSeek reference runners, model inspection, and MXFP4 microbenchmark
tools/             Fixture, GPT-OSS/DeepSeek text, packed-Expert, and benchmark utilities
tests/             Deterministic, parity, cache, scheduling, concurrency, and error-path tests
third_party/ncnn/  Pinned ncnn source submodule
```

Private source files are grouped by owning responsibility and retain direct,
ncnn-style names within each directory.

## License

See [LICENSE](LICENSE). This project builds on the pinned
[Tencent/ncnn](https://github.com/Tencent/ncnn) submodule.
