# ncnn MoE Runtime

> ❗ ❗ ❗**This project is in the early stages of development.**

`ncnn_moe` is a general model-inference framework built around ncnn. Its current
implementation is a C++20 Mixture-of-Experts Transformer runtime. Its verified
baseline is portable CPU execution, with optional ncnn CPU operators. The
ncnn/MoltenVK mixed backend remains experimental on macOS. The project is
designed to grow from an embedded inference core into reusable model adapters,
execution plans, session lifecycle management, and device backends.

> Its current model adapters support deterministic Tiny MoE packages and the
> official GPT-OSS Safetensors packages. GPT-OSS loads directly from the original
> Hugging Face package: there is no GGUF conversion step and no private checkpoint
> format.

## Framework architecture

```text
model packages
    |
    v
model adapters  --->  model-neutral descriptor + compiled execution plan
    |                                      |
    |                                      v
    +----------------------------> runtime core and per-inference sessions
                                               |
                                               v
                                  application, CLI, or external service host
```

The runtime core separates model-family parsing from execution. Adapters emit
an explicit operator graph (`RmsNorm`, `FusedQKV`, `RoPE`, optional
`AttentionSink`, `SDPA`, `Projection`, `Router`, `TopK`, `ExpertGroup`, and
`Combine`). The compiler validates that graph and selects implementations from
backend capabilities; it does not branch on a model-family name or use the
legacy adapter package flags as execution truth. The compiled plan records the
selected CPU/Vulkan backend for every node and is available for introspection.

## Current MoE runtime path

A mixed backend is useful when the GPU has enough VRAM for attention state and
dense activations, but not for the complete model. The experimental mixed path
keeps eligible dense operators on Vulkan and executes routing and routed expert
groups on CPU. Each attention block forms one explicit heterogeneous boundary:
the hidden state is uploaded once, all ncnn layers are recorded into one
`VkCompute`, and the result is downloaded once for CPU routing and experts.
The boundary uses two reusable host-visible Vulkan staging slots instead of
allocating intermediate pageable `ncnn::Mat` buffers for every transfer.
Each slot owns a separate staging allocator and is leased independently; the
next session can fill its mapped upload, RoPE, and mask buffers before entering
the serialized Vulkan command-recording section while the other slot remains
owned by the preceding command.
Attention cosine, sine, and mask tensors are generated directly in those
mapped slots; RoPE frequencies are precomputed and the invariant zero KV sink
is uploaded once when the attention operator is created.
GPT-OSS is the current production-scale adapter and exercises the same
execution plan with BF16 dense tensors and MXFP4 experts.

The asynchronous `BatchScheduler` accepts decode work from independent
sessions. It preserves submission order within each session while allowing a
CPU expert phase from one session to overlap a Vulkan dense phase from another.
The scheduler divides the available OpenMP expert threads across its workers to
avoid multiplying full-size expert teams. Vulkan command recording remains
serialized per device context, so reusable staging slots and shared model
operators remain race-free.

```text
supported model package
            |
            v
       adapter + compiler
            |
            +-- Vulkan (experimental): eligible dense operators and KV cache
            |
            +-- CPU: Top-K routing, native MXFP4 expert groups, sampling
```

This design reduces the **VRAM requirement** relative to a full-GPU model
placement. GPT-OSS can also opt into a byte-bounded, file-backed MXFP4 expert
cache. In that mode, the loader records Safetensors shard ranges instead of
making every expert resident, and routed Gate/Up plus Down pairs are read into
the shared host cache on demand. Dense weights and expert biases remain eager.

## What works today

| Area | Status |
| --- | --- |
| Tiny MoE adapter | Deterministic native package for runtime and integration testing |
| GPT-OSS adapter | Direct multi-shard Safetensors loading; RMSNorm, GQA, attention sinks, sliding/full attention, YaRN RoPE, Top-4 routing, and clamped SwiGLU |
| Execution core | Explicit adapter operator graphs and backend-capability-selected compiled plans |
| Dense backend | Portable CPU baseline; optional ncnn CPU `InnerProduct`; experimental Vulkan dense path with CPU fallback |
| Expert weights | Native MXFP4 with runtime-dispatched ARM NEON, GCC/Clang x86 AVX2/FMA, x86 AVX-512, and scalar fallback kernels; Float32 and INT8 paths |
| Expert execution | Fused MXFP4 unpack/dequant/FMA, fused interleaved Gate/Up activation, decode GEMV, weight-reusing prefill GEMM rows, and parallel active experts |
| Expert residency | Optional byte-bounded, file-backed GPT-OSS MXFP4 expert-pair cache with LRU eviction, fixed async I/O workers, exact/speculative priority queues, and compute-time leases |
| Scheduling | Future-based cross-session decode batches, per-session ordering, bounded worker/OpenMP concurrency, and Linux allowed-CPU/NUMA-topology affinity |
| KV cache | BF16 or FP32; GPU-resident compact cache in mixed mode, ring cache on CPU |
| Generation | Greedy and temperature / Top-K / Top-P / Min-P sampling; token streaming |
| Prompting | Optional OpenAI Harmony wrapper for GPT-OSS text prompts |
| Tests | Deterministic runtime tests plus a generated GPT-OSS-layout fixture |

### Current limits

- MXFP4 experts remain CPU-only. The SIMD kernels remove the scalar-only path,
  but they still need model-scale benchmarking and further cache/blocking work;
  native Vulkan MXFP4 experts are not implemented.
- The scheduler pipelines independent sessions, but it does not yet merge
  multiple sessions into one shape-compatible Vulkan dispatch.
- Linux scheduler workers can discover the process allowed-CPU mask and sysfs
  NUMA node CPU lists, choose at least one default worker per detected node,
  form disjoint node-local partitions, and size default OpenMP teams from the
  allowed mask; callers may still provide explicit CPU sets. Per-node expert
  placement, memory policy, and first-touch weight replication are not yet
  implemented.
- File-backed expert misses use a bounded worker pool. All exact routes are
  queued before compute; ready experts execute first while cold experts read in
  parallel. Decode also submits one low-priority, replaceable next-layer route
  prediction. The exact router remains authoritative.
- Dedicated expert handles use `F_NOCACHE` on macOS, offset `pread` plus
  `POSIX_FADV_DONTNEED` on Linux, and overlapped random-access reads on Windows.
  The runtime does not require repacking the official Safetensors shards.
- In constrained CPU mode, dense tensors keep their original BF16/FP32 storage
  and skip optional transformed ncnn CPU copies. Vulkan mode retains the
  original dense tensors for correctness-preserving CPU fallback while ncnn
  releases its transient upload-side weights.
- Current adapter coverage is Tiny MoE and GPT-OSS. Shared experts and other
  model families are rejected explicitly.
- On macOS, the MoltenVK backend is experimental. Attention operations may fall
  back to CPU; inspect the reported Vulkan dispatch and attention-block counts
  instead of assuming a requested hybrid placement was fully offloaded.

## Performance and memory reporting

The project reports Vulkan dispatch counts, compute submissions, host/device
batch transfers, auxiliary upload counts/bytes, staging-slot resizes/reuses,
staging-slot acquisitions/contentions,
attention-block counts, expert assignments, parallel expert tasks, MXFP4
decode-GEMV/prefill-GEMM/paired-output/fused-Gate-Up rows, expert-cache
hits/misses/evictions/queued and speculative reads/bytes read/resident bytes,
KV-cache bytes, and
attention/router/expert timings per session. Scheduler statistics separately
report submitted/completed/rejected work, maximum in-flight requests,
per-session serialization, and effective worker/team sizes.
This makes mixed-backend bottlenecks and heterogeneous transfer boundaries
visible instead of treating the GPU as a black box.

A useful comparison should report at least:

- the exact model package and revision;
- hardware, OS, driver, backend, and CPU thread count;
- prompt/prefill tokens per second and decode tokens per second separately;
- warm-up count, repeated-run median, peak RSS, and peak VRAM; and
- the requested hybrid placement and context length.

The included runner exposes the timing breakdown needed to build that benchmark
instead of making an unverified cross-runtime performance claim.

## Quick start

Clone the runtime and initialize the pinned ncnn dependency:

```powershell
git clone --recurse-submodules https://github.com/crafcat7/ncnn-MoE-Runtime.git
cd ncnn-MoE-Runtime
cmake -S . -B build-ncnn
cmake --build build-ncnn --config Release --parallel
ctest --test-dir build-ncnn -C Release --output-on-failure
```

The build compiles the pinned `third_party/ncnn` submodule together with the
runtime. Initialize it with `git submodule update --init --recursive` if the
checkout was not cloned with `--recurse-submodules`.

The commands above use a multi-config generator such as Visual Studio. With a
single-config generator, add `-DCMAKE_BUILD_TYPE=Release`, omit `-C Release`
from `ctest`, and remove `Release` from the executable paths below.

### macOS

macOS builds use the pinned `third_party/ncnn` submodule. Vulkan is enabled by
default and uses the experimental MoltenVK hybrid backend. Its requirements
are:

- Xcode or the Xcode Command Line Tools, including the Metal framework;
- a Metal-capable Mac; and
- `libMoltenVK.dylib` installed locally. `brew install molten-vk` is one way to
  provide this system Vulkan implementation; it does **not** provide ncnn.

The configuration searches the standard Homebrew prefixes automatically. For
a manually installed MoltenVK, pass its absolute library path with
`NCNN_MOE_MOLTENVK_LIBRARY` instead. A missing library stops configuration with
an actionable error.

```sh
brew install molten-vk
cmake -S . -B build-macos -G Xcode \
  -DCMAKE_OSX_ARCHITECTURES="$(uname -m)"
cmake --build build-macos --config Release --parallel
ctest --test-dir build-macos -C Release --output-on-failure
```

To use a manually installed MoltenVK:

```sh
cmake -S . -B build-macos -G Xcode \
  -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
  -DNCNN_MOE_MOLTENVK_LIBRARY=/absolute/path/to/libMoltenVK.dylib
```

The backend is experimental: dense and attention operators may fall back to
CPU, and the resulting static library embeds an absolute MoltenVK path. Build
it on the machine that will run it; do not redistribute it as a general CMake
package. To make a CPU-only build, set `-DNCNN_MOE_USE_VULKAN=OFF` explicitly.

#### Run GPT-OSS on macOS

Pass an official `openai/gpt-oss-20b` model directory followed by one or more
token IDs. The runner chooses the available backend automatically; use `--cpu`
or `--hybrid` to select one explicitly:

```sh
MODEL_DIR="/absolute/path/to/gpt-oss-20b"
./build-macos/Release/ncnn_moe_gpt_oss \
  "$MODEL_DIR" 0 \
  --max-new-tokens 64
```

The runner accepts token IDs, not prompt text. To run an OpenAI Harmony text
prompt, install the optional wrapper:

```sh
python3 -m pip install openai-harmony
python3 tools/run_gpt_oss_prompt.py \
  ./build-macos/Release/ncnn_moe_gpt_oss \
  "$MODEL_DIR" \
  "Reply with exactly: OK" \
  --max-new-tokens 64 --stream
```

When using a custom ncnn/MoltenVK installation, pass
`-DNCNN_MOE_MOLTENVK_LIBRARY=/absolute/path/to/libMoltenVK.dylib`; the runtime
uses that path directly instead of relying on the system Vulkan loader.

For a universal binary, replace `$(uname -m)` with `arm64;x86_64` and use a
universal ncnn build with the same architectures.

## Run GPT-OSS

Pass an official `openai/gpt-oss-20b` model directory followed by token IDs:

```powershell
.\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  D:\Models\gpt-oss-20b 0 `
  --max-new-tokens 1 --hybrid
```

`Auto` selects the standard Vulkan-dense/CPU-expert backend when Vulkan is
available. The explicit backend modes are:

| Flag | Placement |
| --- | --- |
| `--cpu` | Portable CPU execution |
| `--hybrid` | Vulkan dense path with CPU experts |
| `--hybrid-prefetch` | Hybrid mode with bounded CPU cache hints before selected experts |

`--expert-cache-mb N` changes MXFP4 expert residency independently of the
compute backend. A non-zero value keeps the official Safetensors expert slices
on disk until routing selects them, and bounds cached expert pairs to `N` MiB.
For the 48 GB `gpt-oss-120b` target, start with a 16 GiB cache:

```powershell
.\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  D:\Models\gpt-oss-120b 0 `
  --max-new-tokens 1 --hybrid --expert-cache-mb 16384
```

For a real text prompt, install OpenAI's Harmony package and use the wrapper:

```powershell
python -m pip install openai-harmony
python tools\run_gpt_oss_prompt.py `
  .\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  D:\Models\gpt-oss-20b `
  "Reply with exactly: OK" `
  --max-new-tokens 64 --stream --backend hybrid
```

The wrapper renders the official Harmony prompt, passes token IDs to the C++
runner, stops on assistant action tokens, and parses the generated IDs back into
structured messages. With `--stream`, decoded assistant text is flushed to
standard output while runner status and timing stay on standard error. Harmony
channel labels such as `[analysis]` and `[final]` remain visible; pass
`--stream-final-only` to suppress analysis text. The C++ runtime itself stays
independent of a tokenizer library.

## Runtime core API

```cpp
#include <ncnn/moe/runtime.h>

#include <vector>

ncnn::moe::Runtime runtime;
ncnn::moe::RuntimeOptions options;
options.expert_cache_bytes = UINT64_C(16) * 1024 * 1024 * 1024;
auto model = runtime.load_model("model-directory", options);
if (!model)
    return;

auto session = runtime.create_session(model.value());
std::vector<int32_t> prompt = {1, 7, 42};
ncnn::moe::GenerationOptions options;
options.max_new_tokens = 64;
options.sampling.temperature = 0.7f;
options.sampling.top_p = 0.9f;
auto generated = session.value()->generate(prompt, options);
```

Independent decode sessions can be submitted without blocking the caller:

```cpp
ncnn::moe::SchedulerOptions scheduler_options;
scheduler_options.worker_count = 2;
auto scheduler = runtime.create_scheduler(scheduler_options);
auto pending = scheduler.value()->submit_decode({
    {first_session, first_token},
    {second_session, second_token},
});
auto results = pending.get();
```

`Model` is immutable and shared; `Session` owns the request-local KV cache,
sampling state, and timing statistics. `SessionOptions::prefill_chunk_size`
defaults to 256 to bound prompt working memory. Set it to zero for one-shot
prefill when memory headroom is available.

## Tiny model smoke test

```powershell
python tools\create_tiny_model.py out\tiny
.\build-ncnn\Release\ncnn_moe_tiny.exe out\tiny 1 7 42
```

An INT8 expert fixture is also available:

```powershell
python tools\create_tiny_model.py out\tiny-int8 --expert-weight-dtype int8
.\build-ncnn\Release\ncnn_moe_tiny.exe out\tiny-int8 1 7 42
```

## Project layout

```text
include/ncnn/moe/  Public API, descriptors, adapters, and execution plans
src/               Safetensors loader, compiler, CPU/ncnn operators, executor
tests/             Deterministic and error-path verification
examples/          Tiny and GPT-OSS command-line runners
tools/             Fixtures and the Harmony prompt wrapper
third_party/ncnn/  Pinned Tencent ncnn source submodule
models/            Model catalog and setup notes
```

## Roadmap

1. Model-scale MXFP4 kernel benchmarks, cache blocking, and NUMA-aware expert
   placement.
2. Shape-compatible cross-session Vulkan batching beyond asynchronous
   pipelining.
3. Native Vulkan MXFP4 expert GEMM with CPU/Vulkan numerical parity tests.
4. Official `gpt-oss-120b` 48 GB/8K model-scale residency validation, followed
   by measured cache-size, I/O-worker, and route-prediction tuning.
5. Broader adapter coverage, shared experts, tokenizer integration, and paged
   KV allocation.

## License and acknowledgments

`ncnn_moe` builds on the pinned [Tencent/ncnn](https://github.com/Tencent/ncnn)
submodule. Consult this repository's license and the dependency notices before
redistributing a binary.
