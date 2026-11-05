# DeepSeek V4 Flash and DSpark

The built-in `deepseek_v4` adapter runs both official Safetensors packages
directly:

- [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)

No GGUF conversion or private checkpoint format is required. Both packages
share the same 43-layer target model. The DSpark package additionally exposes
the draft layers and prediction heads used by the optional speculative path.

AWQ/GPTQ INT4 Safetensors also load directly, including complete DSpark
draft layers and prediction heads. The [shared importer](../README.md#quantized-package-import)
accepts target `deepseek4` GGUF Qn_K/MXFP4, including standard split files;
its output omits DSpark draft metadata because target GGUF has no draft
weights. FP8 source weights and precise scale values remain native. Generated
models validate CPU/Vulkan execution and actual DSpark proposals; full-size
quantized quality and performance have not been measured.

## Model support

| Area | Implementation |
| --- | --- |
| Package | Official multi-shard Hugging Face Safetensors and `config.json` |
| Dense weights | Native FP8 E4M3 with exact scales, INT4 or Qn_K |
| Attention | Latent Attention, 128-token window, ratio-4/128 compression, learned indexing, sinks, and YaRN RoPE |
| Hyper-Connections | Four mHC streams, Sinkhorn mixing, and learned head reduction |
| Experts | Official FP4 E2M1 routed / FP8 shared weights or native INT4/Qn_K variants, square-root Softplus routing, selection bias, route scaling, and token-hash layers |
| Mixed execution | Vulkan Dense projections and eligible native Experts; CPU cache logic, routing and fallback |
| Expert residency | Automatic, eager, or byte-bounded on-demand |
| Expert I/O | Asynchronous range reads, Windows aligned direct I/O, mmap, and byte-aware ARC |
| Scheduling | Ragged Prefill and staged cross-Session mHC, Attention, routing, and Expert batching |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, streaming, and optional DSpark speculative decoding |

## Download

Run commands from the repository root and keep either checkpoint in its
ignored model directory:

```powershell
hf download deepseek-ai/DeepSeek-V4-Flash `
  --local-dir .\models\deepseek-v4\DeepSeek-V4-Flash

hf download deepseek-ai/DeepSeek-V4-Flash-DSpark `
  --local-dir .\models\deepseek-v4\DeepSeek-V4-Flash-DSpark
```

The directory must contain `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and every Safetensors shard referenced by the
index. DSpark metadata and prediction tensors are required only for the DSpark
package. DeepSeek-V4-Flash's conventional `mtp.0` payload is not used by the
target-model runtime. Text formatting and tokenization run in C++ with built-in
Unicode tables; the CLI does not execute `encoding/encoding_dsv4.py` or require the official
PyTorch inference dependencies such as `torch`, `tilelang`, or
`fast_hadamard_transform`. Build the examples by following the root
[Quick start](../../README.md#quick-start), then build `ncnn_moe_worker` for
normal text and chat usage.

## Run text or chat

Runtime applies the checkpoint's text-message encoding and tokenizer. The
unified CLI sends messages to the native worker:

```powershell
python -m pip install -e .
python tools\ncnn_moe.py run `
  --model .\models\deepseek-v4\DeepSeek-V4-Flash `
  --prompt "Briefly introduce yourself." `
  --max-new-tokens 1024 --stream --backend hybrid
python tools\ncnn_moe.py chat `
  --model .\models\deepseek-v4\DeepSeek-V4-Flash
```

`run` streams tokens and metrics; `chat` keeps a persistent conversation and
supports `/context`, `/compact`, `/reset`, `/new`, `/stats`, and `/tune`.
`--no-thinking` requests a direct answer, while
`--show-reasoning` or `/reasoning` expands the hidden reasoning channel. The
CLI locates `ncnn_moe_worker` in common build directories, or accepts an
explicit path with `--worker`. Scripted runs can add `--no-metrics` to suppress
periodic metrics events while retaining final statistics.

## Run token IDs with the reference runner

Applications that already own tokenization can use the model-named native
reference target. It accepts a model directory followed by prompt token IDs:

```powershell
cmake -S . -B build-ncnn -DNCNN_MOE_BUILD_REFERENCE_RUNNERS=ON
cmake --build build-ncnn --config Release --target ncnn_moe_deepseek_v4 --parallel
.\build-ncnn\Release\ncnn_moe_deepseek_v4.exe `
  .\models\deepseek-v4\DeepSeek-V4-Flash 0 `
  --max-new-tokens 64 --hybrid --report-throughput
```

Use `--prompt-token-file PATH` for long whitespace-separated sequences and
`--stream-token-ids` for incremental machine-readable IDs. These reference
targets are retained for the benchmark harness and CTest fixture; normal text
and chat usage should use `ncnn_moe.py`.

## Sampling

| Option | Meaning |
| --- | --- |
| `--max-new-tokens N` | Maximum generated token count |
| `--temperature T` | `0` selects deterministic greedy decoding |
| `--top-k K` | Keep the highest `K` logits; `0` keeps all |
| `--top-p P` | Nucleus sampling probability threshold |
| `--min-p P` | Remove tokens below a fraction of the highest probability |
| `--seed N` | Sampling seed |
| `--speculative-confidence P` | Override the DSpark confidence threshold |
| `--speculative-max-draft N` | Bound the number of draft tokens per proposal |

The native runner stops on DeepSeek EOS token `1` and also accepts repeated
`--stop-token ID` options.

## Execution modes

| Option | Execution |
| --- | --- |
| `--cpu` | Portable CPU path |
| `--hybrid` | Vulkan FP8 Dense projections with CPU Attention cache logic, routing, and Experts |

`HybridMode::Auto` selects mixed execution for a hardware Vulkan device and
falls back to CPU-only for a software CPU Vulkan implementation.

## Expert memory and storage

The unified CLI can override the automatically sized Expert residency and I/O
plan for a known host:

```powershell
python tools\ncnn_moe.py run `
  --model .\models\deepseek-v4\DeepSeek-V4-Flash `
  --prompt "Briefly introduce yourself." `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 28672 --expert-cache-mb 16384 `
  --expert-io-workers 4 --buffered-expert-io `
  --max-new-tokens 1024 --stream
```

| Option | Effect |
| --- | --- |
| `--expert-memory auto\|eager\|on-demand` | Select Expert residency |
| `--cpu-packed-weights on\|off` | Opt into the CPU MXFP4-Q8 in-memory repack; default `off` |
| `--host-memory-mb N` | Override the detected host-memory budget |
| `--expert-cache-mb N` | Bound resident Expert pairs in RAM |
| `--expert-io-workers N` | Set asynchronous read concurrency |
| `--mmap-experts` | Map on-demand Expert ranges |
| `--buffered-expert-io` | Force conventional buffered reads |
| `--expert-gpu-cache-mb N` | Add an executable Vulkan Expert cache |
| `--expert-gpu-victim-cache-mb N` | Add a compressed-weight Vulkan cache behind the host ARC |
| `--vulkan-device N` | Select one Vulkan device |

Begin with CPU Expert execution and calibrate the intended prompt, context,
and Session mix before assigning memory to Vulkan Expert tiers.

## DSpark speculative decoding

DeepSeek-V4-Flash-DSpark uses its draft layers, Markov/confidence heads, and
transactional latent-cache rollback. It is enabled by default when the package
contains the complete DSpark metadata and tensor set. DeepSeek-V4-Flash has no
DSpark plan and always executes the target model directly. The reference
benchmark's `--no-speculative` control provides a target-only baseline for
correctness or performance comparisons. Acceptance rate and net speedup remain
workload- and cache-dependent, so deployment decisions require end-to-end
measurement on the target machine.

For throughput tests with multiple DSpark Sessions, add
`--parallel-speculative` together with `--parallel-sessions N`. Each Session
then executes its own speculative transaction concurrently. The staged
`BatchScheduler` path remains target-only because it submits Prefill and Decode
operations directly rather than calling `Session::generate`.
