# ncnn MoE Runtime

> ❗ ❗ ❗**This project is in the early stages of development.**

`ncnn_moe` is a general model-inference framework built around ncnn. Its current
implementation is a C++20 Mixture-of-Experts Transformer runtime with an
ncnn-powered Vulkan/CPU mixed backend. It is designed to grow from an embedded
inference core into reusable model adapters, execution plans, session lifecycle
management, and device backends.

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

The runtime core already separates model-family parsing from execution. New
model adapters therefore feed the same compiled-plan and session boundary rather
than adding model-specific branches to the executor.

## Current MoE runtime path

A mixed backend is useful when the GPU has enough VRAM for attention state and
dense activations, but not for the complete model. `ncnn_moe` keeps the
latency-sensitive dense path on Vulkan and executes only the routed expert groups
on CPU. GPT-OSS is the current production-scale adapter and exercises this path
with BF16 dense tensors and MXFP4 experts.

```text
supported model package
            |
            v
       adapter + compiler
            |
            +-- Vulkan: RMSNorm, fused QKV, RoPE, GQA attention,
            |           output projection, Router, LM head, BF16 KV cache
            |
            +-- CPU: Top-K routing, native MXFP4 expert groups, sampling
```

This design reduces the **VRAM requirement** relative to a full-GPU model
placement. It does not currently reduce total host-RAM use: weights are eagerly
loaded, experts are resident in memory, and there is no mmap or SSD expert
offload path yet.

## What works today

| Area | Status |
| --- | --- |
| Tiny MoE adapter | Deterministic native package for runtime and integration testing |
| GPT-OSS adapter | Direct multi-shard Safetensors loading; RMSNorm, GQA, attention sinks, sliding/full attention, YaRN RoPE, Top-4 routing, and clamped SwiGLU |
| Execution core | Model-neutral descriptors and compiled execution plans behind the adapter boundary |
| Dense backend | ncnn Vulkan operators for attention blocks, Router, and LM head; portable CPU fallback |
| Expert weights | Native MXFP4 storage and scalar CPU execution; Float32 and INT8 test paths |
| KV cache | BF16 or FP32; GPU-resident compact cache in mixed mode, ring cache on CPU |
| Generation | Greedy and temperature / Top-K / Top-P / Min-P sampling; token streaming |
| Prompting | Optional OpenAI Harmony wrapper for GPT-OSS text prompts |
| Tests | Deterministic runtime tests plus a generated GPT-OSS-layout fixture |

### Current limits

- MXFP4 expert GEMM is CPU-only. It is the primary decode bottleneck in the
  current mixed backend; native Vulkan MXFP4 experts are planned work.
- Weights are eagerly loaded. There is no memory mapping, demand paging, or
  SSD-backed expert cache.
- Current adapter coverage is Tiny MoE and GPT-OSS. Shared experts and other
  model families are rejected explicitly.

## Performance and memory reporting

The project reports Vulkan dispatch counts, attention-block counts, expert
assignments, KV-cache bytes, and attention/router/expert timings per session.
This makes mixed-backend bottlenecks visible instead of treating the GPU as a
black box.

Do not compare a single `ncnn_moe` run with another runtime's published number:
model format, prompt length, generated length, warm-up policy, CPU threads,
context length, and GPU placement all materially affect the result. A useful
comparison should report at least:

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
cmake -S . -B build-ncnn `
  -DNCNN_MOE_BUILD_BUNDLED_NCNN=ON `
  -DNCNN_MOE_USE_VULKAN=ON
cmake --build build-ncnn --config Release --parallel
ctest --test-dir build-ncnn -C Release --output-on-failure
```

The bundled ncnn build enables Vulkan and BF16 by default. With a separately
installed ncnn package, set `NCNN_MOE_BUILD_BUNDLED_NCNN=OFF` and pass
`-Dncnn_DIR=<path-to-ncnn-config>`.

The commands above use a multi-config generator such as Visual Studio. With a
single-config generator, add `-DCMAKE_BUILD_TYPE=Release`, omit `-C Release`
from `ctest`, and remove `Release` from the executable paths below.

For a portable CPU-only build, disable ncnn explicitly:

```powershell
cmake -S . -B build-cpu `
  -DNCNN_MOE_USE_NCNN=OFF `
  -DNCNN_MOE_USE_VULKAN=OFF
cmake --build build-cpu --config Release --parallel
ctest --test-dir build-cpu -C Release --output-on-failure
```

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
auto model = runtime.load_model("model-directory");
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

1. Native Vulkan MXFP4 expert GEMM with CPU/Vulkan numerical parity tests.
2. Reproducible same-hardware benchmarks against comparable runtimes, including
   decode, prefill, RSS, and VRAM.
3. Lazy/mmap and SSD-backed expert loading with a bounded cache for models that
   exceed host memory.
4. Broader model-family support, shared experts, and batched inference.
5. Better inference-core memory management, including paged KV allocation.

## License and acknowledgments

`ncnn_moe` builds on the pinned [Tencent/ncnn](https://github.com/Tencent/ncnn)
submodule. Consult this repository's license and the dependency notices before
redistributing a binary.
