# ncnn MoE Runtime

`ncnn-MoE-Runtime` (NMR) is a C++20 inference runtime for sparse
Mixture-of-Experts transformer models. It runs on CPU and Vulkan, compiles a
model package into an immutable execution graph, and exposes independent
sessions for prefill, decode, generation, batching, and speculative decoding.

The runtime is designed around three ideas:

- compile the model graph once, then execute a flat ordered `Layer[]`;
- keep Expert weights out of the dense working set and stream or cache them;
- place every layer on CPU or Vulkan at plan time instead of guessing at runtime.

The Python CLI is a control plane. Tokenization, text decoding, model loading,
graph compilation, and inference all run in the native worker.

---

## Supported models

| Model family | Package `model_type` | Notes |
| --- | --- | --- |
| GPT-OSS-20B / GPT-OSS-120B | `gpt_oss` | Routed + shared Experts, Harmony text interface |
| DeepSeek-V4-Flash / DSpark | `deepseek_v4` | MLA/GatedDelta paths, DSpark speculative decoding |
| Qwen3 dense | `qwen3` | Dense Transformer, native Qwen tokenizer |
| Qwen3.5 / Qwen3.6 | `qwen3_5`, `qwen3_5_moe` | Dense and MoE variants, optional MXFP4 Expert sidecar |
| Qwen3.8-Flash-Next | `qwen4_exp` | MoE text backbone, optional MXFP4 Expert sidecar |
| Phi-mini-MoE-instruct | `phimoe` | CPU and Hybrid execution |
| Ling-3.0-tiny | `bailing_hybrid` | Hybrid attention/FFN layout |

Supported package inputs include native Safetensors directories and GGUF
packages where the model adapter provides a mapping. The runtime supports
FP32/BF16 activations and a range of dense and Expert weight formats:

```text
BF16, FP8 E4M3, INT4, INT8, MxFp4,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS
```

Optional tools:

```text
tools/import_quantized.py         AWQ/GPTQ/GGUF checkpoint -> native Safetensors
tools/compile_qwen_artifact.py    Qwen routed Experts -> optional MXFP4 sidecar
```

Native text input requires the tokenizer assets shipped with the model package.
Token-ID requests remain available when native text assets are absent.

---

## Quick start

NMR does not load PyTorch or pnnx graphs directly. A supported model is a
Safetensors directory or GGUF package parsed by a built-in model adapter. The
runtime path is:

```text
model package -> ModelAdapter -> CompiledModel -> Net -> Session
```

The commands below assume the native worker has been built. See
[Build and demo the worker](#build-and-demo-the-worker) for the build step.

### 1. Get the source and Python CLI

```bash
git clone --recurse-submodules https://github.com/crafcat7/ncnn-MoE-Runtime.git
cd ncnn-MoE-Runtime

python -m pip install -r requirements.txt
```

Python 3.10+ is required for the CLI. The native runtime and tokenizer do not
require `transformers`, `tokenizers`, or a Python Unicode package.

### 2. Prepare a model package

A native package is either:

```text
model-directory/
├── config.json
├── tokenizer assets for native text input
└── *.safetensors
```

or a GGUF file supported by its model adapter.

For quantized checkpoints, use the optional importers before running:

```bash
python tools/import_quantized.py --help
python tools/compile_qwen_artifact.py --help
```

Models without tokenizer assets can still be driven by token IDs.

### 3. Build the native worker

Build once with the instructions in
[Build and demo the worker](#build-and-demo-the-worker). The default worker
location is:

```text
Windows: build-ncnn/Release/ncnn_moe_worker.exe
Linux/macOS: build-ncnn/ncnn_moe_worker
```

### 4. Run with the Python CLI

Inspect the resolved backend, memory plan, Expert cache, and placement:

```bash
python tools/ncnn_moe.py inspect --model PATH/TO/MODEL
```

Run one prompt:

```bash
python tools/ncnn_moe.py run --model PATH/TO/MODEL --prompt "Hello" --max-new-tokens 32
```

Interactive chat with persisted history:

```bash
python tools/ncnn_moe.py chat --model PATH/TO/MODEL --session demo
```

Useful variants:

```bash
# stream runtime metrics
python tools/ncnn_moe.py run --model PATH/TO/MODEL --prompt "Hello" --metrics

# CPU-only comparison
python tools/ncnn_moe.py run --model PATH/TO/MODEL --prompt "Hello" --cpu

# explicit worker or device
python tools/ncnn_moe.py run \
  --model PATH/TO/MODEL \
  --worker build-worker/Release/ncnn_moe_worker.exe \
  --vulkan-device 0 \
  --prompt "Hello"
```

Common runtime controls:

```text
--backend auto|cpu|hybrid
--cpu / --hybrid
--host-memory-mb N
--expert-memory auto|eager|on-demand
--expert-cache-mb N
--expert-gpu-cache-mb N
--expert-io-workers N
--vulkan-device N
--vulkan-devices 0,1
--expected-concurrency N
```

### 5. Run with the C++ API

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

`Model` is immutable and shared by Sessions. `Session` owns KV/recurrent state,
sampling state, and reusable execution scratch. `Runtime::synchronize_model_caches`
is available as an explicit synchronization point before shutdown or after
warm-up.

---

## Build and demo the worker

Requirements:

- CMake 3.21+
- C++20 compiler
- Git and the `third_party/ncnn` submodule
- a Vulkan-capable driver for GPU execution (optional)

The worker is a long-lived native process that owns one `Model` and multiple
`Session` objects. It speaks strict UTF-8 JSONL over stdin/stdout. Generation is
asynchronous inside the worker, so tokens can stream while the parent process
reads events.

### Build the worker

From the repository root:

```powershell
cmake -S . -B build-ncnn `
  -DNCNN_MOE_BUILD_TESTS=OFF `
  -DNCNN_MOE_BUILD_EXAMPLES=ON
cmake --build build-ncnn --config Release --target ncnn_moe_worker --parallel
```

Or use the checked-in presets:

```powershell
cmake --preset worker-release
cmake --build --preset worker-release
```

`worker-release` builds with Vulkan enabled. `worker-release-cpu` builds with
`NCNN_MOE_USE_VULKAN=OFF`.

Useful build switches:

| Option | Meaning |
| --- | --- |
| `NCNN_MOE_BUILD_TESTS` | Build the CTest suite |
| `NCNN_MOE_BUILD_EXAMPLES` | Build `ncnn_moe_worker` |
| `NCNN_MOE_USE_VULKAN` | Enable the Vulkan backend |
| `NCNN_MOE_USE_NCNN` | Use ncnn optimized CPU operators |
| `NCNN_MOE_ENABLE_OPENMP` | Enable OpenMP CPU parallelism |
| `NCNN_MOE_ENABLE_X86_SIMD` | Build AVX2/AVX512 kernels |
| `NCNN_MOE_ENABLE_ARM_SVE2` | Build ARM SVE2 MXFP4 kernels |
| `NCNN_MOE_ENABLE_MSVC_MP` | Use MSVC `/MP` |
| `NCNN_MOE_ENABLE_LTO` | Enable Release LTO |
| `NCNN_MOE_OPT_LEVEL` | Release optimization: `O0`, `O1`, `O2`, `O3`, `Os` |
| `NCNN_MOE_SIZE_OPTIMIZE_COLD_PATHS` | Build loader/tokenizer paths with size optimization |

### Run the worker directly

```bash
build-ncnn/Release/ncnn_moe_worker.exe PATH/TO/MODEL --hybrid
```

The worker prints a `ready` event first. Then send one JSON object per line:

```json
{"op":"create_session","session_id":"demo"}
{"op":"generate","session_id":"demo","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_new_tokens":8}
{"op":"shutdown"}
```

Typical generation events:

```text
{"event":"token","request_id":"generate","session_id":"demo","token_id":...,"text":"OK"}
{"event":"metrics", ...}
{"event":"done","ok":true,...}
```

Supported operations:

| Operation | Purpose |
| --- | --- |
| `create_session` | Create a native `Session` |
| `generate` | Token-ID or message-based generation |
| `reset` | Clear Session state |
| `compact` | Replay messages/tokens into a compacted context |
| `stats` | Query prompt/context counts |
| `cancel` | Request cancellation of the active generation |
| `shutdown` | Stop the worker |

`generate` accepts exactly one of:

```json
{"prompt_tokens":[1,2,3]}
```

or:

```json
{"messages":[{"role":"user","content":"Hello"}]}
```

The message path uses the C++ tokenizer/chat template. The token path is useful
for benchmarks and applications that already own a tokenizer.

The Python CLI wraps this protocol; use the commands in
[Quick start](#quick-start) for normal text and chat usage.

### Tests and local validation

```bash
cmake -S . -B build-test -DNCNN_MOE_BUILD_TESTS=ON
cmake --build build-test --config Release --parallel
ctest --test-dir build-test -C Release --output-on-failure
```

Build all targets before running CTest; building only `ncnn_moe_tests` leaves
other test executables unavailable. The style test requires `clang-format` on
`PATH` in addition to Python 3.10+.

---

## Why it is fast

### Compile once, execute a flat graph

Model loading produces one immutable `Net`:

```text
Net
├── Layer[]        backend, kind, plan index, special weight bindings
├── Blob[]         producer/consumer edges and lifetimes
├── CompiledLayerPlan[]
└── order          topo/FFN-grouped execution order
```

Backend placement, Expert mode, tensor lifetimes, and memory reservations are
decided before generation starts. The decode loop walks `Net::order` and does not
build or schedule a graph per token.

### MoE-aware Expert management

The runtime treats Expert weights differently from dense weights:

- routed Expert routing is grouped before execution;
- Expert weights can live in a host ARC cache, GPU cache, GPU victim cache, or
  resident GPU Expert bank;
- on-demand mode streams Expert pairs from mapped checkpoint storage;
- shared Experts and routed Experts can overlap where the backend supports it;
- router prediction prefetches the next Experts before they are needed;
- native MXFP4/Qn_K Expert kernels avoid expanding weights into BF16.

This keeps the dense working set small while still allowing GPU-resident Experts
when the device has enough memory.

### Hybrid CPU / Vulkan placement

Each Layer is placed independently at compile time:

- CPU for unsupported or cheap operations;
- Vulkan for Attention, latent Attention, GatedDeltaNet, dense projections, and
  supported Experts;
- persistent Vulkan segments and command graphs for selected layer ranges;
- separate device budgets for weights, Session state, workspace, and Expert
  caches.

The runtime does not silently replay failed GPU work on the CPU. Unsupported
operator combinations are moved to the CPU during planning, with explicit graph
boundaries.

### CPU kernels and quantization

- runtime ISA dispatch: AVX2, AVX-512, BF16, VNNI, Arm NEON, SVE/SVE2;
- OpenMP parallelism with workload-aware thread budgets;
- direct Qn_K block execution without a persistent expanded copy;
- GGUF-native Q4/Q5/Q8/IQ formats;
- MXFP4 and FP8 kernels;
- FP32/BF16 activation paths with vectorized normalization, RoPE, Attention, and
  Expert arithmetic.

### Memory planning

- host-memory budget and automatic Expert cache sizing;
- mmap, direct I/O, and buffered Expert I/O modes;
- graph arena planning from Blob lifetimes;
- separate host output slots, device arenas, Session KV/recurrent state, and
  Expert staging;
- shape growth replans the graph arena without rebuilding the model.

### Batching and speculative decoding

- `BatchScheduler` batches decode across Sessions;
- one shared Router/Expert/Combine workspace is used for staged rows;
- DSpark and MTP speculative paths share the same compiled execution model;
- speculative verification commits or rolls back Session KV state together.

### Control plane without Python in the data path

The Python CLI handles history, sessions, profiles, and display. The worker owns:

- tokenizer and chat template;
- generation;
- KV state;
- Expert cache and GPU state;
- metrics and telemetry.

This keeps text generation reproducible across Python, direct JSONL, and C++
embeddings.

### Reference measurements

Reference points from the project benchmarks, measured on a Ryzen 7 9800X3D,
31.14 GiB RAM, and an RTX 5070 Ti 16 GiB:

| Model | Execution | Single Session | Multi-Session |
| --- | --- | ---: | ---: |
| GPT-OSS-20B | Hybrid, eager Experts | 14.855 token/s | n/a |
| GPT-OSS-120B | Hybrid, 16 GiB host ARC | 4.718 token/s | 10.916 token/s aggregate (2 Sessions) |
| DeepSeek-V4-Flash | Hybrid, 16 GiB host ARC | 1.254 token/s | 3.809 token/s aggregate (4 staged Sessions) |
| Qwen3.6-35B-A3B | Hybrid, MXFP4 Expert artifact | 8.383 token/s | 22.435 token/s aggregate (4 Sessions) |

These are reference points, not a cross-model ranking. Model size, quantization,
context length, Expert residency, and Session count dominate the result.

### Project layout

```text
include/ncnn/moe/  public Runtime / Model / Session / Scheduler API
src/graph/         Net, Layer, Blob, compiler, memory and resource plans
src/engine/        Runtime, Session, Scheduler, executor, Expert orchestration
src/backends/ncnn/ CPU/Vulkan operators, GpuSegment, Expert banks
src/kernels/       portable and SIMD kernels
src/models/        GGUF/Safetensors readers and model adapters
src/storage/       mapped files, WeightStore, Expert cache/residency
examples/          JSONL worker and telemetry
tools/             Python CLI, package importer, Qwen artifact compiler
tests/             kernel, graph, model, and parity tests
```

---

## License

This project is licensed under the **BSD 2-Clause License**.

```text
Copyright (c) 2026, Lune
```

See [LICENSE](LICENSE) for the full text.

Third-party components retain their own licenses, including the pinned ncnn
submodule under `third_party/ncnn`.
