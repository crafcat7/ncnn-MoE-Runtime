# GPT-OSS

The built-in `gpt_oss` adapter runs the official
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) and
[`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)
Safetensors packages directly. No GGUF conversion or private checkpoint format
is required.

AWQ/GPTQ INT4 Safetensors also load directly. GGUF Qn_K/MXFP4 checkpoints
use the [shared importer](../README.md#quantized-package-import), including
standard split files. These paths retain packed weights and use the existing
CPU/Vulkan execution and Expert residency policy. Generated-model format and
numerical tests cover these variants; full-size quantized quality and
performance have not been measured.

## Model support

| Area | Implementation |
| --- | --- |
| Package | Official multi-shard Hugging Face Safetensors and `config.json` |
| Dense weights | BF16/F32, native INT4 or imported Qn_K with automatic mmap and buffered fallback |
| Attention | RMSNorm, fused QKV+RoPE, option-controlled fused Decode SDPA, GQA, learned sinks, full/sliding Attention, and YaRN RoPE |
| KV cache | Persistent per-Session BF16 or FP32 CPU cache; FP32 mixed-backend ring |
| Experts | Native MXFP4, INT4 and Qn_K with runtime-selected CPU kernels and eligible Vulkan execution |
| Mixed execution | Vulkan Dense/Attention and eligible Experts, with CPU routing and fallback |
| Expert residency | Automatic, eager, or byte-bounded on-demand |
| Expert I/O | Asynchronous range reads, Windows aligned direct I/O, packed sidecar support, and byte-aware ARC |
| Scheduling | Independent Sessions, explicit batch submission, and Expert coalescing |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, and streaming |

GPT-OSS-20B can use eager Expert residency when memory permits. GPT-OSS-120B
is designed to run with on-demand Expert residency on hosts that cannot keep
the complete checkpoint in RAM.

## Download

Run commands from the repository root and keep checkpoints in the ignored
model directories:

```powershell
hf download openai/gpt-oss-20b --local-dir .\models\gpt-oss\gpt-oss-20b
hf download openai/gpt-oss-120b --local-dir .\models\gpt-oss\gpt-oss-120b
```

Each directory must contain `config.json`, `model.safetensors.index.json`, and
every shard referenced by the index. Text prompts also require the official
`tokenizer.json` and `tokenizer_config.json` assets. Build the examples by following the root
[Quick start](../../README.md#quick-start), then build `ncnn_moe_worker` for
normal text and chat usage.

## Run text or chat

Runtime applies Harmony formatting and token decoding from the local tokenizer
assets. The unified CLI sends messages to the native worker:

```powershell
python -m pip install -e .
python tools\ncnn_moe.py run `
  --model .\models\gpt-oss\gpt-oss-20b `
  --prompt "Reply with exactly: OK" `
  --max-new-tokens 1024 --stream --backend hybrid
python tools\ncnn_moe.py chat `
  --model .\models\gpt-oss\gpt-oss-20b
```

`run` streams tokens and metrics; `chat` keeps a persistent conversation and
supports `/context`, `/compact`, `/reset`, `/new`, `/stats`, and `/tune`. Use
`--show-reasoning` or `/reasoning` when the hidden reasoning channel is needed.
The CLI locates `ncnn_moe_worker` in common build directories, or accepts an
explicit path with `--worker`. Scripted runs can add `--no-metrics` to suppress
periodic metrics events while retaining final statistics.

## Sampling

| Option | Meaning |
| --- | --- |
| `--max-new-tokens N` | Maximum generated token count |
| `--temperature T` | `0` selects deterministic greedy decoding |
| `--top-k K` | Keep the highest `K` logits; `0` keeps all |
| `--top-p P` | Nucleus sampling probability threshold |
| `--min-p P` | Remove tokens below a fraction of the highest probability |
| `--seed N` | Sampling seed |

The native runner also accepts repeated `--stop-token ID` options.

## Execution modes

| Option | Execution |
| --- | --- |
| `--cpu` | Portable CPU path |
| `--hybrid` | Vulkan Dense/Attention and CPU MXFP4 Experts |

`HybridMode::Auto` selects mixed execution for a hardware Vulkan device and
falls back to CPU-only for a software CPU Vulkan implementation. MXFP4 Experts
use CPU arithmetic by default. A non-zero executable Expert GPU cache enables
native Vulkan MXFP4 kernels only when runtime calibration measures an
end-to-end benefit; losing workload buckets return to CPU automatically.

## Expert memory and storage

`Auto` estimates dense and MXFP4 storage before loading weights. It selects
eager Expert residency when the host budget has safe headroom and otherwise
uses a byte-bounded cache backed by exact ranges in the original shards.

| Option | Effect |
| --- | --- |
| `--expert-memory auto\|eager\|on-demand` | Select Expert residency |
| `--cpu-packed-weights on\|off` | Opt into the CPU MXFP4-Q8 in-memory repack; default `off` |
| `--host-memory-mb N` | Override the detected host-memory budget |
| `--expert-cache-mb N` | Bound resident Expert pairs in RAM |
| `--expert-io-workers N` | Set asynchronous read concurrency; `0` derives it from Top-K and physical cores |
| `--mmap-experts` | Map on-demand Expert ranges |
| `--buffered-expert-io` | Force conventional buffered reads |
| `--expert-gpu-cache-mb N` | Add an executable Vulkan MXFP4 Expert cache |
| `--expert-gpu-victim-cache-mb N` | Add a compressed-weight Vulkan cache behind the host ARC |
| `--vulkan-device N` | Select one Vulkan device |
| `--vulkan-devices N[,N...]` | Supply candidates for capability-weighted layer placement |
| `--parallel-sessions N` | Run independent sequences through the batch scheduler |

Dense tensors and eager MXFP4 ranges are mapped automatically. On-demand
Experts are read directly into final cache storage without allocating or
zero-filling complete Expert tensors.

The host cache uses byte-aware Adaptive Replacement Cache lists: T1 tracks
recent pairs, T2 tracks repeatedly used pairs, and the B1/B2 ghost histories
adapt the resident split without retaining weight bytes. Cache accounting and
the benchmark report expose resident bytes, ghost hits, reads, cancellations,
and speculative admissions.

The shipped kernel and Vulkan optimization profile is enabled by default. The
experimental prediction/cache controls remain native A/B hooks and are not
exposed by the unified worker. They remain disabled by default; benchmark them
against the target workload before enabling them through native callers.

## GPT-OSS-120B on a constrained-memory host

This configuration is suitable as a starting point for a machine with about
32 GiB RAM and a 16 GiB Vulkan device. The same resource overrides are accepted
by the unified CLI:

```powershell
python tools\ncnn_moe.py run `
  --model .\models\gpt-oss\gpt-oss-120b `
  --prompt "Reply with exactly: OK" `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 24576 --expert-cache-mb 16384 `
  --expert-io-workers 4 `
  --max-new-tokens 128 --stream
```

Begin with CPU Expert execution and measure the intended prompt, context, and
Session mix before assigning memory to Vulkan Expert tiers. Larger host ARC
budgets improve route reuse but must leave enough memory for dense mappings,
KV state, execution scratch, and the operating system.
