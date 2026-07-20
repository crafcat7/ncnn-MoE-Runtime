# ncnn MoE Runtime

`ncnn_moe` is a C++20 MoE Transformer inference runtime built around ncnn CPU
operators. It can load the official OpenAI GPT-OSS Safetensors packages directly
and run CPU Prefill/Decode without converting the checkpoint into a private
format.

## Implemented

- Public `ncnn::moe::Runtime`, immutable shared `Model`, and per-request
  `Session` APIs.
- A single `MoeAdapter` for the deterministic `tiny_moe` package and official
  `gpt_oss` Hugging Face packages.
- Multi-shard Safetensors loading for BF16 dense tensors and native MXFP4 Expert
  tensors.
- GPT-OSS RMSNorm, learned attention sinks, grouped-query attention, alternating
  sliding/full attention, YaRN RoPE, persistent KV cache, Top-4 routing, and the
  GPT-OSS clamped SwiGLU Expert function.
- ncnn InnerProduct acceleration for eligible FP32/BF16 dense projections, with
  portable CPU kernels as a fallback and direct MXFP4 Expert execution.
- Greedy token generation plus an optional Harmony text-prompt wrapper.
- Deterministic Tiny MoE tests and a generated miniature checkpoint using the
  official GPT-OSS file and tensor layout.

## Build and test

The portable CPU build has no required external library:

```powershell
cmake -S . -B build -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DNCNN_MOE_USE_NCNN=OFF
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

To enable ncnn operators, point CMake at an installed ncnn package. The ncnn
build should enable BF16 and OpenMP:

```powershell
cmake -S . -B build-ncnn -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -Dncnn_DIR=D:\path\to\ncnn\install\lib\cmake\ncnn
cmake --build build-ncnn --parallel 4
ctest --test-dir build-ncnn --output-on-failure
```

MSVC and other C++20 toolchains may use their normal CMake generator. Optional
targets are controlled by `NCNN_MOE_BUILD_TESTS` and
`NCNN_MOE_BUILD_EXAMPLES`.

## Run GPT-OSS

Pass the official `openai/gpt-oss-20b` model directory followed by token IDs:

```powershell
.\build-ncnn\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-20b 0 `
  --max-new-tokens 1
```

For a real text prompt, install OpenAI's Harmony package and use the wrapper:

```powershell
python -m pip install openai-harmony
python tools\run_gpt_oss_prompt.py `
  .\build-ncnn\ncnn_moe_gpt_oss.exe `
  D:\Models\gpt-oss-20b `
  "Reply with exactly: OK" `
  --max-new-tokens 64
```

The wrapper renders the system/user conversation with the official Harmony
encoding, passes token IDs to the C++ runtime, stops on assistant action tokens,
and parses generated IDs back into structured Harmony messages.

## Run the Tiny model

Generate a deterministic Tiny MoE package and run Prefill:

```powershell
python tools\create_tiny_model.py out\tiny
.\build\ncnn_moe_tiny.exe out\tiny 1 7 42
```

An INT8 Expert variant is also available:

```powershell
python tools\create_tiny_model.py out\tiny-int8 --expert-weight-dtype int8
.\build\ncnn_moe_tiny.exe out\tiny-int8 1 7 42
```

## Public API

```cpp
#include <ncnn/moe/runtime.h>

ncnn::moe::Runtime runtime;
auto model = runtime.load_model("model-directory");
if (!model)
    return;

auto session = runtime.create_session(model.value());
std::vector<int32_t> prompt = {1, 7, 42};
auto prefill = session.value()->prefill(prompt);
auto next = session.value()->decode(5);
```

`Runtime::load_model` accepts an official GPT-OSS directory, a Tiny model
directory, or the corresponding manifest/config path. Adapters normalize model
names and tensor layouts during loading; execution only consumes the compiled,
model-neutral plan.

## Current limitations

- CPU execution and full-logits output only; Vulkan scheduling is not yet
  implemented.
- Greedy generation is provided by the example; sampling is not yet part of the
  public runtime API.
- Weights are eagerly loaded, and the KV cache currently stores float32 values.
- Shared Experts and model families other than Tiny MoE and GPT-OSS are rejected.

## Source layout

```text
include/ncnn/moe/    Public API, descriptors, adapters, and execution plans
src/                 Loader, compiler, CPU/ncnn operators, executor, and session
tests/               Deterministic runtime and error-path verification
examples/            Tiny and GPT-OSS command-line runners
tools/               Tiny/GPT fixtures and Harmony prompt wrapper
memories/repo/       Versioned contributor knowledge
```

The architectural rule is stable: adapters absorb model-family differences at
load time, while executors consume a model-neutral compiled plan.
