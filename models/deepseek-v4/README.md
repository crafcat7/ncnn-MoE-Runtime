# DeepSeek V4 Flash DSpark

The built-in `deepseek_v4` adapter runs the official
[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
Safetensors package directly. No GGUF conversion or private checkpoint format
is required.

## Model support

| Area | Implementation |
| --- | --- |
| Package | Official multi-shard Hugging Face Safetensors and `config.json` |
| Dense weights | Blockwise FP8 E4M3 values with E8M0 scales |
| Attention | Latent Attention, 128-token window, ratio-4/128 compression, learned indexing, sinks, and YaRN RoPE |
| Hyper-Connections | Four mHC streams, Sinkhorn mixing, and learned head reduction |
| Experts | FP4 E2M1 routed Experts, FP8 shared Expert, square-root Softplus routing, selection bias, route scaling, and token-hash layers |
| Mixed execution | Vulkan FP8 Dense projections with CPU Attention cache logic, routing, and Experts |
| Expert residency | Automatic, eager, or byte-bounded on-demand |
| Expert I/O | Asynchronous range reads, Windows aligned direct I/O, mmap, and byte-aware ARC |
| Scheduling | Ragged Prefill and staged cross-Session mHC, Attention, routing, and Expert batching |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, streaming, and DSpark speculative decoding |

## Download

Run commands from the repository root and keep the checkpoint in the ignored
model directory:

```powershell
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark `
  --local-dir .\models\deepseek-v4\DeepSeek-V4-Flash-DSpark
```

The directory must contain `config.json`, `tokenizer.json`,
`encoding/encoding_dsv4.py`, every Safetensors shard referenced by the index,
and the `mtp.*` DSpark tensors. Build the Release runner by following the root
[Quick start](../../README.md#quick-start).

## Run a text prompt

The text wrapper applies the checkpoint's official DeepSeek message encoding
and tokenizer without adding a tokenizer dependency to the C++ runtime:

```powershell
python -m pip install -U transformers
python tools\run_deepseek_v4_prompt.py `
  .\build-ncnn\Release\ncnn_moe_deepseek_v4.exe `
  .\models\deepseek-v4\DeepSeek-V4-Flash-DSpark `
  "请简短介绍一下你自己。" `
  --max-new-tokens 1024 --stream --backend hybrid
```

By default, the wrapper prints only the human-readable `[reasoning]` and
`[answer]` sections. Native runner diagnostics, token IDs, timings, and cache
statistics are suppressed unless `--verbose` is present. The reply limit
defaults to 1024 tokens; reaching it before EOS produces a warning.
`--thinking-mode chat` requests a direct answer, and `--stream-final-only`
suppresses reasoning during streaming.

## Run token IDs

The native runner accepts a model directory followed by one or more prompt
token IDs:

```powershell
.\build-ncnn\Release\ncnn_moe_deepseek_v4.exe `
  .\models\deepseek-v4\DeepSeek-V4-Flash-DSpark 0 `
  --max-new-tokens 64 --hybrid
```

Use `--prompt-token-file PATH` for long whitespace-separated token sequences.
Use `--stream-token-ids` to print generated IDs as they become available. The
final machine-readable line is `generated token ids:`.
`ncnn_moe_deepseek_v4` and `ncnn_moe_gpt_oss` use the same native runner
implementation and option parser; their entry points only select the expected
adapter and model-specific default EOS.

## Sampling

| Option | Meaning |
| --- | --- |
| `--max-new-tokens N` | Maximum generated token count |
| `--temperature T` | `0` selects deterministic greedy decoding |
| `--top-k K` | Keep the highest `K` logits; `0` keeps all |
| `--top-p P` | Nucleus sampling probability threshold |
| `--min-p P` | Remove tokens below a fraction of the highest probability |
| `--seed N` | Sampling seed |
| `--no-speculative` | Disable DSpark speculative decoding |

The native runner stops on DeepSeek EOS token `1` and also accepts repeated
`--stop-token ID` options.

## Execution modes

| Option | Execution |
| --- | --- |
| `--cpu` | Portable CPU path |
| `--hybrid` | Vulkan FP8 Dense projections with CPU Attention cache logic, routing, and Experts |
| `--hybrid-prefetch` | Mixed path with explicit CPU cache hints |

`HybridMode::Auto` selects mixed execution for a hardware Vulkan device and
falls back to CPU-only for a software CPU Vulkan implementation.

## Expert memory and storage

The DeepSeek text wrapper defaults to on-demand Expert residency with four I/O
workers. Runtime auto-sizing can be overridden for a known host:

```powershell
python tools\run_deepseek_v4_prompt.py `
  .\build-ncnn\Release\ncnn_moe_deepseek_v4.exe `
  .\models\deepseek-v4\DeepSeek-V4-Flash-DSpark `
  "请简短介绍一下你自己。" `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 28672 --expert-cache-mb 16384 `
  --expert-io-workers 4 --buffered-expert-io `
  --max-new-tokens 1024 --stream
```

| Option | Effect |
| --- | --- |
| `--expert-memory auto\|eager\|on-demand` | Select Expert residency |
| `--host-memory-mb N` | Override the detected host-memory budget |
| `--expert-cache-mb N` | Bound resident Expert pairs in RAM |
| `--expert-io-workers N` | Set asynchronous read concurrency |
| `--mmap-experts` | Map on-demand Expert ranges |
| `--direct-expert-io` | Force aligned direct reads when supported |
| `--buffered-expert-io` | Force conventional buffered reads |
| `--expert-gpu-cache-mb N` | Add an executable Vulkan Expert cache |
| `--expert-gpu-victim-cache-mb N` | Add a compressed-weight Vulkan cache behind the host ARC |
| `--vulkan-device N` | Select one Vulkan device |

Begin with CPU Expert execution and calibrate the intended prompt, context,
and Session mix before assigning memory to Vulkan Expert tiers.

## DSpark speculative decoding

DSpark uses the checkpoint's draft layers, Markov/confidence heads, and
transactional latent-cache rollback. It is enabled by default when the package
contains the required tensors. `--no-speculative` provides a deterministic
baseline for correctness or performance comparisons. Acceptance rate and net
speedup remain workload- and cache-dependent, so deployment decisions require
end-to-end measurement on the target machine.
