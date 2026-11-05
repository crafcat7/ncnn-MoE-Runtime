# ncnn MoE Runtime

A C++20 runtime for sparse Mixture-of-Experts transformer inference on CPU and
Vulkan. It compiles a model package into an immutable execution graph, keeps
Expert weights outside the dense working set, and exposes independent Sessions
for prefill, decode, generation, batching, and speculative decoding.

## Supported

| Model | Package `model_type` | Notes |
| --- | --- | --- |
| GPT-OSS-20B / 120B | `gpt_oss` | Routed + shared Experts, native text interface |
| DeepSeek-V4-Flash / DSpark | `deepseek_v4` | MLA/GatedDelta paths, DSpark decoding |
| Qwen3 dense | `qwen3` | Dense Transformer |
| Qwen3.5 / Qwen3.6 | `qwen3_5`, `qwen3_5_moe` | Dense and MoE, optional MXFP4 Expert sidecar |
| Qwen3.8-Flash-Next | `qwen4_exp` | MoE text backbone, optional MXFP4 Expert sidecar |
| Phi-mini-MoE-instruct | `phimoe` | CPU and Hybrid |
| Ling-3.0-tiny | `bailing_hybrid` | Hybrid attention/FFN layout |

Supported weight formats:

```text
BF16, FP8 E4M3, INT4, INT8, MxFp4,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS
```

Optional package tools:

```text
tools/import_quantized.py         AWQ/GPTQ/GGUF -> native Safetensors
tools/compile_qwen_artifact.py    Qwen routed Experts -> MXFP4 sidecar
```

Native text requires tokenizer assets from the model package. Token-ID input
works without them.

## Quick Start

Build the worker first; see [Build](#build). The default worker location is
`build-ncnn/Release/ncnn_moe_worker.exe` on Windows and
`build-ncnn/ncnn_moe_worker` on Linux/macOS.

A model path is a Safetensors directory or a GGUF file supported by the
built-in adapters.

### Python CLI

```bash
python -m pip install -r requirements.txt

python tools/ncnn_moe.py inspect --model PATH/TO/MODEL
python tools/ncnn_moe.py run --model PATH/TO/MODEL --prompt "Hello" --max-new-tokens 32
python tools/ncnn_moe.py chat --model PATH/TO/MODEL
```

Common options:

```text
--cpu / --hybrid
--backend auto|cpu|hybrid
--host-memory-mb N
--expert-memory auto|eager|on-demand
--expert-cache-mb N
--expert-gpu-cache-mb N
--vulkan-device N
```

### Worker JSONL

```bash
build-ncnn/Release/ncnn_moe_worker.exe PATH/TO/MODEL --hybrid
```

Then send one JSON object per line:

```json
{"op":"create_session","session_id":"demo"}
{"op":"generate","session_id":"demo","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_new_tokens":8}
{"op":"shutdown"}
```

The worker emits `ready`, `token`, `metrics`, `done`, and `error` events. A
request may use either `messages` or `prompt_tokens`.

### C++ API

```cpp
#include "ncnn/moe/runtime.h"

ncnn::moe::Runtime runtime;
ncnn::moe::Option opt;
opt.hybrid_mode = ncnn::moe::HybridMode::Auto;

auto model = runtime.load_model("PATH/TO/MODEL", opt);
if (!model) return;
auto session = runtime.create_session(model.value());
if (!session) return;

auto prompt = model.value()->encode(
    R"([{"role":"user","content":"Hello"}])");
if (!prompt) return;

ncnn::moe::GenerationOptions options;
options.max_new_tokens = 32;
options.stop_tokens = model.value()->stop_tokens();
auto generated = session.value()->generate(prompt.value(), options);
```

`Model` is immutable and shared. Each `Session` owns KV/recurrent state,
sampling state, and reusable execution scratch.

## Build

Requirements: CMake 3.21+, a C++20 compiler, Git, and Python 3.10+ for the CLI.
Vulkan execution also requires a Vulkan driver.

```bash
git clone --recurse-submodules https://github.com/crafcat7/ncnn-MoE-Runtime.git
cd ncnn-MoE-Runtime

cmake -S . -B build-ncnn -DNCNN_MOE_BUILD_TESTS=OFF
cmake --build build-ncnn --config Release --target ncnn_moe_worker --parallel
```

Presets:

```bash
cmake --preset worker-release
cmake --build --preset worker-release
```

Useful options:

| Option | Meaning |
| --- | --- |
| `NCNN_MOE_USE_VULKAN` | Enable Vulkan |
| `NCNN_MOE_USE_NCNN` | Use ncnn CPU operators |
| `NCNN_MOE_ENABLE_OPENMP` | Enable OpenMP |
| `NCNN_MOE_ENABLE_X86_SIMD` | Build AVX2/AVX512 kernels |
| `NCNN_MOE_ENABLE_ARM_SVE2` | Build ARM SVE2 MXFP4 kernels |
| `NCNN_MOE_ENABLE_LTO` | Enable Release LTO |
| `NCNN_MOE_OPT_LEVEL` | `O0`, `O1`, `O2`, `O3`, or `Os` |

Tests:

```bash
cmake -S . -B build-test -DNCNN_MOE_BUILD_TESTS=ON
cmake --build build-test --config Release --parallel
ctest --test-dir build-test -C Release --output-on-failure
```

## Features

- Compile once: `Net` owns `Layer[]`, `Blob[]`, compiled plans, and the final
  execution order. Decode does not rebuild or reschedule a graph per token.
- Hybrid CPU/Vulkan placement at plan time, including persistent Vulkan
  segments for supported Attention/FFN layer ranges.
- MoE-aware Expert management: router prediction, route grouping, host ARC
  cache, GPU cache, victim cache, resident Expert banks, and on-demand streaming.
- Shared Expert overlap and device-resident routed aggregation where supported.
- Native Vulkan paths for standard Attention, latent Attention, GatedDeltaNet,
  dense projections, and supported Experts.
- CPU SIMD dispatch for AVX2, AVX-512, BF16, VNNI, Arm NEON, and SVE/SVE2, with
  workload-aware OpenMP scheduling.
- Direct quantized execution for FP8, MXFP4, Qn_K, INT4/INT8, and GGUF-native
  Q4/Q5/Q8/IQ formats without expanding every weight to BF16.
- Host-memory budgeting, mmap/direct/buffered Expert I/O, graph arena planning
  from Blob lifetimes, and separate KV/recurrent Session state.
- `BatchScheduler` for multi-Session decode and DSpark/MTP speculative decoding
  with transactional KV commit/rollback.
- Native tokenizer/chat-template support and a strict JSONL worker protocol;
  Python is the control plane, not the inference data path.

## License

BSD 2-Clause License. See [LICENSE](LICENSE).