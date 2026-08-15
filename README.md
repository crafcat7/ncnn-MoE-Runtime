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
includes validated GPT-OSS, DeepSeek-V4-Flash, DeepSeek-V4-Flash-DSpark, and
Qwen3.6-35B-A3B text-backbone support for their official Safetensors packages.

[Runtime capabilities](#runtime-capabilities) | [Architecture](#architecture) |
[Supported models and data](#supported-models-and-reference-data) |
[Quick start](#quick-start)

## Supported models and reference data

The table is a compact index of the current reference measurements and
admission status. Published values are three-run medians on a Ryzen 7 9800X3D,
31.14 GiB RAM, and an RTX 5070 Ti 16 GiB. Each benchmarked model report owns
the complete cold/warm, short/long, multi-Session, memory, cache, and I/O
matrix.

| Model | Reference execution | Reference single Session, 32 tokens | Reference service, 32 tokens | Full report |
| --- | --- | ---: | ---: | --- |
| GPT-OSS-20B | Hybrid, eager Experts | **14.855 token/s** | n/a | [GPT-OSS performance](models/gpt-oss/README.md#reference-performance) |
| GPT-OSS-120B | Hybrid, 16 GiB host ARC | **4.718 token/s** | **10.916 aggregate token/s** (2 Sessions) | [GPT-OSS performance](models/gpt-oss/README.md#reference-performance) |
| DeepSeek-V4-Flash | Hybrid, 16 GiB host ARC | **1.254 token/s** | **3.809 aggregate token/s** (4 staged Sessions) | [Flash performance](models/deepseek-v4/README.md#deepseek-v4-flash-matrix) |
| DeepSeek-V4-Flash-DSpark | Hybrid, DSpark enabled, 16 GiB host ARC | **1.254 token/s** | **3.594 aggregate token/s** (4 independent Sessions) | [DSpark performance](models/deepseek-v4/README.md#deepseek-v4-flash-dspark-matrix) |
| Qwen3.6-35B-A3B | Hybrid, compiled MXFP4 Artifact, target only | **8.383 token/s** | **22.435 aggregate token/s** (4 independent Sessions) | [Qwen3.6 performance](models/qwen3.6/README.md#reference-performance) |

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
| Model integration | Public `IMoeModelAdapter` contract and model-neutral `MoeIR`; built-in GPT-OSS, DeepSeek V4, and Qwen3.6 text adapters |
| IR and compiler | Model-neutral `MoeIR` for Attention, Router, ExpertGroup, Combine, KV Cache, and quantization metadata; validation, normalization, weight resolution, and immutable compilation |
| Execution graph | Tensor and node dependencies, backend candidates and placement, cross-backend events, and topological execution waves |
| Dense path | Portable CPU with runtime-dispatched FP8 E4M3 scalar/AVX2/AVX-512 Linear, optional ncnn CPU operators, and mixed ncnn Vulkan Dense/Attention execution |
| Attention | RMSNorm, GQA, full/sliding Attention, Gated DeltaNet, latent Attention with learned compressed history, RoPE/YaRN variants, output gates, sinks, persistent KV/recurrent state, fused QKV+RoPE, and adaptive online Decode SDPA |
| Experts | Stable Top-K regrouping, Softmax/Sigmoid/square-root-Softplus scoring, hash routes, gated shared Experts, float32/BF16/FP8/INT8/Q2_K-Q6_K execution, and fused-decode FP4 kernels selected at runtime for scalar, NEON, SVE2, AVX2/FMA, or AVX-512 |
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
  selects scalar, NEON, SVE2, AVX2/FMA, or AVX-512 implementations. Qwen3.6 can
  optionally compile its routed BF16 banks into the same fused storage profile
  without modifying the official shards.
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
- **Model-provided speculation.** DSpark and experimental Qwen MTP use
  transactional Attention/recurrent state and exact fallback commits. DSpark
  supports batched verification; Qwen currently uses exact sequential target
  verification and remains opt-in.
- **Workload-aware scheduling.** Ragged Prefill, cross-Session collection,
  mHC/Attention/Expert batching, and adaptive staged/independent execution
  improve service throughput without fixing policy choices to one device.

## Quick start

Requirements: a C++20 compiler, CMake 3.21 or newer, Python 3.10+, and Git.

Install the complete Python environment for the unified examples from the
repository root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs the GPT-OSS Harmony adapter, Hugging Face-backed adapters,
DeepSeek/Qwen model-download and artifact-build tools, the optional
Rich/prompt-toolkit UI, and CPU/GPU telemetry providers. The file delegates to
the extras in `pyproject.toml`, so the dependency groups have a single source
of truth. For a smaller installation, use only the required extra, for example
`python -m pip install -e ".[gpt-oss]"`.

```powershell
git clone --recurse-submodules https://github.com/crafcat7/ncnn-MoE-Runtime.git
cd ncnn-MoE-Runtime
cmake -S . -B build-ncnn `
  -DNCNN_MOE_BUILD_TESTS=OFF `
  -DNCNN_MOE_BUILD_REFERENCE_RUNNERS=OFF `
  -DNCNN_MOE_BUILD_MXFP4_BENCHMARK=OFF
cmake --build build-ncnn --config Release --target ncnn_moe_worker --parallel
```

The worker build is the fast default for the unified CLI. CMake does not build
the model-specific reference runners or the MXFP4 microbenchmark unless they
are explicitly enabled. MSVC builds use multiprocessor compilation by default;
set `-DNCNN_MOE_ENABLE_MSVC_MP=OFF` only when a toolchain does not support it.

For the full development and CTest targets, use a separate build directory:

```powershell
cmake -S . -B build-test `
  -DNCNN_MOE_BUILD_TESTS=ON `
  -DNCNN_MOE_BUILD_REFERENCE_RUNNERS=ON `
cmake --build build-test --config Release `
  --target ncnn_moe_tests ncnn_moe_worker ncnn_moe_gpt_oss --parallel
ctest --test-dir build-test -C Release --output-on-failure
```

For a single-config generator:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_MOE_BUILD_TESTS=ON
cmake --build build --target ncnn_moe_tests ncnn_moe_worker --parallel
ctest --test-dir build --output-on-failure
```

The pinned ncnn submodule is built with the runtime. Restore it with
`git submodule update --init --recursive` when necessary. A CPU-only build can
be selected with `-DNCNN_MOE_USE_VULKAN=OFF`.

## Unified examples

The supported user-facing example entry point is the Python CLI:

```powershell
python tools\ncnn_moe.py inspect --model .\models\gpt-oss\gpt-oss-20b
python tools\ncnn_moe.py run --model .\models\gpt-oss\gpt-oss-20b `
  --prompt "Reply with exactly: OK"
python tools\ncnn_moe.py chat --model .\models\gpt-oss\gpt-oss-20b
```

The CLI uses `build-ncnn` as its fixed default build directory. On the
multi-config Windows generator the default worker is
`build-ncnn/Release/ncnn_moe_worker.exe`; on single-config platforms it is
`build-ncnn/ncnn_moe_worker`. It does not scan other build directories or
`PATH`. Use `--worker PATH` or `NCNN_MOE_WORKER=PATH` only when selecting a
different build explicitly.

Build `ncnn_moe_worker` with the examples. The worker owns Runtime/Model and
Session lifetime and communicates through token-ID JSONL; the Python adapters
own tokenizers, chat templates, reasoning channels, persistent history, and
TUI presentation. `inspect` reports the detected hardware and the effective
backend, memory, Expert-cache, and I/O plan. `chat` adds resumable sessions,
context budgeting, compaction, and live token/CPU/GPU/I/O metrics. Install
`openai-harmony` for GPT-OSS or `transformers` and the Hugging Face packages
for DeepSeek/Qwen; `rich` and `prompt-toolkit` are optional TUI upgrades.
While the worker is loading, the CLI renders native `init` lifecycle progress
on stderr so JSONL stdout remains machine-readable.
Human-readable resource sizes in `inspect` and metrics use decimal `GB`; the
machine-readable JSONL fields retain their exact byte values.
Persistent session history, user configuration, and tuning profiles are stored
under `.ncnn-moe/` in the project root; use `--config-dir` when another
location is required. Native KV and runtime cache state remain in memory.
Text generation streams by default, reasoning is shown by default, and the
periodic metrics trace is disabled by default. Use `--no-stream`,
`--hide-reasoning`, or `--metrics` to override these choices for one command.

See [examples/README.md](examples/README.md) for the protocol boundary and
common commands. The model guides document model-specific package preparation
and adapter options.

## Qn_K CPU weights and read sidecar

The CPU path accepts the portable Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and
Q8_K blocks. Q2_K-Q6_K use the exact 256-element super-block layouts; Q8_K is
available for activation/block interchange. A `TensorData` loaded through
`SafetensorsArchive::load_qnk_tensor()` or `load_qnk_expert()` keeps the raw
bytes lossless. By default, CPU linear execution reads those blocks directly
and does not create a second persistent in-memory weight layout.
The x86 build dispatches Q2_K-Q8_K to direct SIMD bit-decode dot kernels: AVX2
handles all formats, while AVX-512 uses 16-lane Q4_K/Q5_K/Q6_K kernels and the
validated AVX2 kernels for the remaining formats. The scalar decoder remains
the portability and correctness fallback.

`--cpu-packed-weights on` explicitly enables lazy in-memory repacking for
supported Qn_K and MXFP4-Q8 CPU Expert weights. The default is `off`; there is
no automatic mode. When enabled, the memory planner and on-demand Expert cache
reserve the additional sidecar bytes before loading. Repack can help some
batched shapes but increases resident memory and is not assumed to improve
single-token decode, so benchmark it on the target machine before enabling it.

The deterministic Qn_K reference comparison can be run from the CPU test
binary:

```powershell
build-cpu\Release\ncnn_moe_tests.exe --benchmark-qnk
```

It reports the scalar-decode/SIMD-float-dot reference time, direct SIMD time,
speedup, and checksum delta for all six formats.

For Expert banks stored as one contiguous U8 tensor, create an optional read-
optimized sidecar without modifying the source shards:

```powershell
python tools\pack_qnk_experts.py D:\Models\my-model `
  --tensor model.layers.0.mlp.experts.gate.weight `
  --tensor model.layers.0.mlp.experts.up.weight `
  --dtype q5_k --rows 4096 --columns 4096 --expert-count 32 `
  --output D:\Models\my-model\ncnn-moe-packed-qnk.safetensors
```

The source tensors must contain `expert_count * rows * (columns / 256) *
block_bytes` bytes, where `block_bytes` is the canonical Qn_K block size. The
runtime prefers sidecar names of the form
`__ncnn_moe_packed__.{expert}.{tensor}`. Packing only copies/reorders storage;
it does not requantize weights or introduce an additional numerical error.

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
unified Python CLI is the text and conversation entry point, while
`ncnn_moe_worker` is the tokenizer-neutral token-ID protocol boundary.
Model-specific tokenization, chat templates, reasoning/final-channel decoding,
and stop-token policy stay in Python adapters.

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
| Built-in reference adapters | GPT-OSS-20B/120B, DeepSeek-V4-Flash/DSpark, and the Qwen3.6-35B-A3B text backbone |
| CPU execution | Complete portable path |
| Heterogeneous execution | Vulkan Dense/Attention with CPU routing and Experts |
| Native Vulkan Experts | Optional MXFP4 cache and execution with runtime calibration and CPU fallback |
| Multiple Vulkan devices | Capability-weighted whole-layer placement |
| Expert memory | Automatic eager or byte-bounded on-demand residency |
| KV cache | CPU FP32/BF16 or mixed-backend FP32 ring |
| Output | Full logits with native sampling and streaming |

The distributed release includes validated GPT-OSS, DeepSeek V4, and Qwen3.6
text production adapters. The Qwen admission excludes its vision encoder;
its one-layer MTP payload is admitted with the checkpoint-bound compiled
Artifact. Additional adapters use the same public IR and
compiler boundary without adding model-family checks to Prefill, Decode,
scheduling, or Expert execution.
Multi-device placement is layer placement rather than Tensor Parallelism, and
Vulkan-only execution is not a supported public mode.

## Model adapters

The Runtime core is model-neutral; production package support is supplied by
registered adapters. Model execution guides live with their adapter
definitions:

- [Model catalog and capability matrix](models/README.md)
- [GPT-OSS-20B/120B execution and performance](models/gpt-oss/README.md)
- [DeepSeek V4 Flash and DSpark execution and performance](models/deepseek-v4/README.md)
- [Qwen3.6-35B-A3B text admission and execution](models/qwen3.6/README.md)

## Project layout

```text
include/ncnn/moe/  Installed Runtime API, MoeIR, graph, plan, Expert, memory, and scheduler contracts
src/compiler/      MoeIR validation, normalization, descriptor conversion, and graph construction
src/graph/         Model compilation, execution-graph scheduling, Expert dispatch, and memory planning
src/engine/        Runtime Core, Sessions, batch scheduling, MemoryManager, CPU execution, and Expert-backend coordination
src/models/        Built-in model adapters, package loading, packed-sidecar selection, and canonical Tensor names
src/storage/       Mapped files, range I/O, host ARC, Vulkan victim cache, system-memory discovery, and Expert lifecycle
src/kernels/       Portable CPU Attention/Linear, BF16 helpers, Qn_K pack/decode, and runtime-selected FP8/MXFP4 SIMD kernels
src/backends/ncnn/ ncnn CPU/Vulkan operator packaging, mixed Attention, Vulkan contexts, and native MXFP4 Experts
models/            Model catalog and model-family execution guides
assets/            Published benchmark visualizations used by model reports
examples/          Unified worker, benchmark/reference runners, protocol helpers, and MXFP4 microbenchmark
tools/             Unified CLI/adapters, fixture, packed-Expert/Qn_K, and benchmark utilities
tests/             Deterministic, parity, cache, scheduling, concurrency, and error-path tests
third_party/ncnn/  Pinned ncnn source submodule
```

Private source files are grouped by owning responsibility and retain direct,
ncnn-style names within each directory.

## License

See [LICENSE](LICENSE). This project builds on the pinned
[Tencent/ncnn](https://github.com/Tencent/ncnn) submodule.
