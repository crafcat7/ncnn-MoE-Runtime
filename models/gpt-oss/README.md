# GPT-OSS

The built-in `gpt_oss` adapter loads the official
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) and
[`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)
Safetensors packages directly. No conversion, GGUF export, or weight repacking
is required.

## Execution capability

| Area | Implementation |
| --- | --- |
| Package | Official multi-shard Hugging Face Safetensors and `config.json` |
| Dense weights | BF16/F32 with automatic mmap and buffered fallback |
| Attention | RMSNorm, fused QKV, GQA, learned sinks, full/sliding Attention, YaRN RoPE |
| KV cache | Persistent per-Session BF16 or FP32 cache |
| Experts | Native MXFP4 block/scales with CPU SIMD kernels |
| CPU backend | Complete portable execution |
| Mixed backend | Vulkan dense/Attention with CPU routing and Experts |
| Expert residency | Automatic, eager, or byte-bounded on-demand |
| Expert I/O | Fixed worker pool, exact-route priority, speculative prefetch, LRU eviction |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, and streaming |

## Download

Run every command in this guide from the repository root. Download checkpoints
into the model-specific directories below this guide:

```powershell
hf download openai/gpt-oss-20b --local-dir .\models\gpt-oss\gpt-oss-20b
hf download openai/gpt-oss-120b --local-dir .\models\gpt-oss\gpt-oss-120b
```

The repository `.gitignore` excludes both model directories, including
configuration, index, tokenizer, and weight files downloaded into them.
Each directory must contain `config.json`, `model.safetensors.index.json`, and
every shard referenced by the index. Build the Release runner by following the
root [Quick start](../../README.md#quick-start).

## Run token IDs

The native runner accepts a model directory followed by one or more prompt
token IDs:

```powershell
.\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-20b 0 `
  --max-new-tokens 64 --hybrid
```

Use `--stream-token-ids` to print each generated ID as it becomes available.
The final report includes backend dispatch counts, Attention/Router/Expert
timings, cache statistics, and the complete generated token sequence.

## Run text prompts

The optional Harmony wrapper performs prompt formatting and token decoding
without adding a tokenizer dependency to the C++ runtime:

```powershell
python -m pip install openai-harmony
python tools\run_gpt_oss_prompt.py `
  .\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-20b `
  "Reply with exactly: OK" `
  --max-new-tokens 64 --stream --backend hybrid
```

Sampling controls shared by both entry points:

| Option | Meaning |
| --- | --- |
| `--max-new-tokens N` | Maximum generated token count |
| `--temperature T` | `0` selects deterministic greedy decoding |
| `--top-k K` | Keep the highest `K` logits; `0` keeps all |
| `--top-p P` | Nucleus sampling probability threshold |
| `--min-p P` | Remove tokens below a fraction of the highest probability |
| `--seed N` | Sampling seed |

The native runner also accepts repeated `--stop-token ID` options.

## Select a backend

| Option | Execution |
| --- | --- |
| `--cpu` | Portable CPU path |
| `--hybrid` | Vulkan dense/Attention and CPU MXFP4 Experts |
| `--hybrid-prefetch` | Mixed path with explicit CPU cache hints |

`HybridMode::Auto` selects the mixed path when a compatible Vulkan device is
available. MXFP4 Expert arithmetic remains CPU-owned in every mixed mode.

## Control Expert memory

`Auto` estimates dense and MXFP4 storage before loading weights. It selects
eager Expert residency when the host budget has safe headroom and otherwise
uses a byte-bounded cache backed by exact ranges in the original shards.

| Option | Effect |
| --- | --- |
| `--expert-memory auto\|eager\|on-demand` | Select the residency policy |
| `--host-memory-mb N` | Override the detected host-memory budget |
| `--expert-cache-mb N` | Bound resident Expert pairs in RAM |
| `--expert-io-workers N` | Set asynchronous read concurrency |
| `--expert-gpu-cache-mb N` | Add an opt-in packed-Expert Vulkan victim cache |
| `--mmap-experts` | Map on-demand Expert ranges instead of buffered reads |

Dense tensors and eager MXFP4 ranges are mapped automatically. The default
on-demand path reads each requested Expert pair directly into final cache
storage, so it does not allocate or zero-fill complete Expert tensors.

`--mmap-experts` affects only on-demand Experts. Buffered overlapped reads are
the default because they were faster than page-fault-driven Expert access on
the verified Windows/NVMe system.

## Run GPT-OSS-120B with constrained memory

The following configuration is verified with the official 60.7678 GiB
checkpoint on 31.14 GiB RAM and an RTX 5070 Ti 16 GiB:

```powershell
python tools\run_gpt_oss_prompt.py `
  .\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-120b `
  "Reply with exactly: OK" `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 24576 --expert-cache-mb 16384 `
  --expert-io-workers 4 `
  --max-new-tokens 128 --stream
```

Add `--expert-gpu-cache-mb 3072` when a repeated or sustained workload benefits
from additional VRAM residency.

## Performance

### Repeated GPT-OSS-120B decode

Protocol: Windows Release build, Ryzen 7 9800X3D, RTX 5070 Ti 16 GiB, Vulkan
dense/Attention, CPU AVX-512 MXFP4 Experts, 24 GiB host budget, 16 GiB Expert
cache, four I/O workers, one process warm-up, and three measured 32-token runs
with identical generated token IDs.

| Metric | Median |
| --- | ---: |
| Decode throughput | **1.902 token/s** |
| 32-token generation | **16.822 s** |
| Expert execution | 14.443 s |
| Expert-cache hit rate | 70.29% |
| Peak process working set | 19,815 MiB |
| Peak total NVIDIA memory | 12,029 MiB |

### mmap comparison

The following 128-token runs used the same host and generated the same token
sequence:

| Storage path | Load | Generation | Throughput | Expert time | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buffered baseline before dense mmap | 11.281 s | 40.202 s | 3.18 token/s | 32.416 s | 20,773 MiB |
| **Dense mmap + buffered Experts** | **7.223 s** | **40.295 s** | **3.18 token/s** | **32.222 s** | **19,597 MiB** |
| Dense mmap + Expert mmap | 6.235 s | 49.757 s | 2.57 token/s | 41.792 s | 19,693 MiB |

A separate 128-token B/A/B/A comparison measured 42.290 seconds without the
3 GiB GPU victim cache and 40.049 seconds with it. Token IDs were identical and
SSD reads fell by 7.35 GB.

These measurements are a reproducible local baseline, not a cross-platform
guarantee.

### Repeat the benchmark

`tools/benchmark_gpt_oss.py` checks token parity, reports medians, samples peak
process RSS, and records NVIDIA memory when `nvidia-smi` is available.

```powershell
python tools\benchmark_gpt_oss.py `
  .\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-120b `
  --model-revision "Hugging-Face-commit" `
  --max-new-tokens 32 --warmup 1 --repeats 3 `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 24576 --expert-cache-mb 16384 `
  --expert-io-workers 4 `
  --json-output .\build-ncnn\gpt-oss-120b-benchmark.json
```
