# Qwen3.5 / Qwen3.6

## Dense text models

The `qwen3_5` adapter also admits the dense text architecture used by
[`Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) and the
[`Qwen3.5` dense series](https://huggingface.co/Qwen/Qwen3.5-4B).
It reads `text_config.intermediate_size` and ordinary
`mlp.{gate_proj,up_proj,down_proj}.weight` tensors. Tied output embeddings are
supported. Each dense FFN executes directly, without routing, Expert cache
allocation, or Expert statistics. Gated DeltaNet and full Attention use the
existing CPU/mixed backends, including Vulkan dense FFNs.

Use the same `run` and `chat` commands below with the dense model directory.
The dense path is text-only and target-only; vision and MTP tensors are not
executed. Native text accepts the validated Qwen3.5 and Qwen3.6 tokenizer and
string-message templates; Qwen3.5-0.8B generation was validated on the CPU and
mixed Vulkan backends. Automated tests cover generated tiny packages and the
official tokenizer assets; end-to-end throughput has not been benchmarked for
these dense models.

Dense FFN projections may retain native affine INT4 or raw Q2_K–Q8_K blocks.
Each gate/up/down projection has its own dtype, including mixed Q4_K/Q6_K
layers. CPU and Vulkan paths retain packed weights without persistent expanded copies.
AWQ/GPTQ scales, zero points and GPTQ group indices are preserved. Attention
and the output head can also use quantized projections. Keep
`--cpu-packed-weights off` for native imported weights; resident packed storage
is not admitted for this path. See the [model catalog](../README.md) for the
quantized-package import boundary.

## Qwen3.6-35B-A3B

The built-in `qwen3_5_moe` adapter runs the text backbone from the official
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
Safetensors package directly. The package name is Qwen3.6, while its published
`config.json` identifies the architecture as `qwen3_5_moe`.

No conversion, GGUF file, or generated sidecar is required for the default
BF16 path. An optional compiled MXFP4 routed-Expert Artifact is available as a
separate performance/quality profile. Artifact v3 also admits the package's
one-layer MTP payload as an experimental speculative-decoding option.

## Admitted scope

| Area | Implementation |
| --- | --- |
| Package | Official multi-shard Hugging Face BF16 Safetensors and `config.json` |
| Text architecture | 40 layers in a repeated 3 Gated DeltaNet + 1 gated full-Attention pattern |
| Gated DeltaNet | Depthwise convolution, normalized Q/K, FP32 recurrent state, learned decay/update gates, gated RMSNorm, and persistent per-Session state |
| Full Attention | GQA, partial RoPE, shifted Q/K RMSNorm, output gate, and persistent KV state |
| Experts | 256 BF16 routed Experts by default, or an optional compiled MXFP4 routed-Expert Artifact; normalized Softmax Top-8, one gated BF16 shared Expert per layer, and SiLU activation |
| Norms | Qwen shifted RMSNorm weights for the text backbone |
| Mixed execution | Vulkan Dense projections and device-resident Gated DeltaNet when Vulkan Attention is available, with CPU fallback, routing, and BF16 Experts |
| Speculation | One experimental Qwen MTP layer with sequential target verification and transactional Gated DeltaNet/KV state; opt-in and available only with Artifact v3 |
| Generation | Greedy, temperature, Top-K, Top-P, Min-P, stop tokens, token-ID streaming, and optional MTP |
| Text input | Official tokenizer and text chat template in Runtime |

This admission is text-only. The vision encoder and image/video token path are
not compiled or executed. The package's one-layer MTP payload is compiled only
when the checkpoint-bound Artifact is present; the zero-conversion BF16 profile
remains target-only.

## Download

Run from the repository root and keep the checkpoint in its ignored model
directory:

```powershell
hf download Qwen/Qwen3.6-35B-A3B `
  --local-dir .\models\qwen3.6\Qwen3.6-35B-A3B
```

The directory must contain `config.json`, `tokenizer.json`,
`chat_template.jinja`, the Safetensors index, and every referenced shard.

## Optional compiled Expert Artifact

The default package remains a zero-conversion BF16 model. For higher local
throughput, compile the 40 target-model routed-Expert banks plus the one MTP
routed-Expert bank into a read-optimized OCP MXFP4 sidecar:

```powershell
python tools\compile_qwen_artifact.py `
  .\models\qwen3.6\Qwen3.6-35B-A3B
```

The compiler requires NumPy, leaves every official shard unchanged, and writes
`ncnn-moe-qwen3.6-mxfp4.safetensors` atomically in the model directory. It
interleaves Gate/Up rows for the fused CPU kernel and stores FP4 E2M1 values
with one E8M0 power-of-two scale per 32 values. Use `--workers N` to control
parallel conversion and `--overwrite` to rebuild an existing Artifact.

The adapter automatically selects the fixed sidecar when present. Before
selection it verifies the Artifact version, its binding to the exact local
`config.json` and Safetensors index, all routed layer tensor names, shapes,
dtypes, byte counts, and file ranges. A stale, truncated, or copied Artifact
is rejected with an instruction to rebuild it; removing the sidecar selects
the original BF16 routed Experts again. Verbose runner output and benchmark
JSON identify the active profile as `mxfp4-compiled-artifact-v3` or
`bfloat16-source`.

This Artifact is an explicit weight-quantization profile, not a lossless cache:
it reduces the routed-Expert payload to about 16.34 GiB and
does not promise token parity with the BF16 path. Validate task quality before
deployment. Shared Experts and all non-routed weights remain in their official
storage formats.

## Build

The first-class user-facing native target is the long-lived worker:

```powershell
cmake -S . -B build-ncnn
cmake --build build-ncnn --config Release `
  --target ncnn_moe_worker --parallel
```

Use `-DNCNN_MOE_USE_VULKAN=OFF` for a portable CPU-only build.

## Run text or chat

Runtime applies the checkpoint's official tokenizer and text chat template.
The unified CLI sends messages and streams text through `ncnn_moe_worker`:

```powershell
python -m pip install -e .
python tools\ncnn_moe.py run `
  --model .\models\qwen3.6\Qwen3.6-35B-A3B `
  --prompt "Briefly explain mixture-of-experts models." `
  --max-new-tokens 1024 --stream --backend hybrid
python tools\ncnn_moe.py chat `
  --model .\models\qwen3.6\Qwen3.6-35B-A3B
```

Thinking mode uses the checkpoint defaults: temperature 1.0, Top-P 0.95,
Top-K 20, and Min-P 0.0. `--no-thinking` requests the official template's
direct-answer mode. `--show-reasoning` or `/reasoning` expands the hidden
reasoning channel. The runtime automatically uses the model-provided speculative
plan when supported. `--speculative-max-draft N` caps the draft window when the
checkpoint exposes one. The CLI locates `ncnn_moe_worker` automatically or
accepts `--worker PATH`. Scripted runs can add `--no-metrics` to suppress
periodic metrics events while retaining final statistics.

The Runtime does not currently expose the checkpoint's recommended presence
penalty. Applications that require exact sampling-policy parity should account
for that difference.

## Execution and memory

| Option | Execution |
| --- | --- |
| `--cpu` | Portable CPU path |
| `--hybrid` | Vulkan Dense projections with CPU stateful Attention and Experts |
| `--vulkan-device N` | Select one Vulkan device |
| `--host-memory-mb N` | Override the automatic host-memory planning budget |
| `--cpu-packed-weights on\|off` | Opt into MXFP4-Q8 repack when the compiled Artifact is active; default `off` |

The official package contains about 60 GiB of routed BF16 Expert tensors.
Without the optional Artifact, these tensors are mapped directly from
Safetensors rather than copied into an additional resident buffer. The
operating system therefore owns physical page residency for the BF16 weights.

With the compiled Artifact, routed Experts use the existing MXFP4 fused CPU
kernels and are eligible for eager mapping or the byte-bounded Expert ARC and
range-I/O path. Qwen's SiLU routed Experts are not admitted to the native
Vulkan MXFP4 Expert backend, so the Artifact does not silently enable a
different GPU arithmetic path. Artifact v3 also contains the MTP routed bank;
each speculative target verification uses transactional standard KV and
Gated DeltaNet state so rejected trailing rows are rolled back exactly.

When Vulkan Attention is available, Gated DeltaNet uses the Vulkan path by
default. The implementation uses fused BF16 input projection,
device-resident FP32 convolution/recurrent state, one workgroup-owned
sequential state transition, and Vulkan output projection. GPU transcendental
functions and reduction order can drift from the CPU reference during long
greedy generations, so validate token parity on the target GPU.

## Validation boundary

Admission validation covers configuration recognition, graph compilation,
all 40 text layers, state continuation across Gated DeltaNet calls, official
chat-template tokenization, and real-checkpoint CPU and mixed-backend
generation, including the native 32/256-token matrix above. Artifact
validation additionally covers its checkpoint identity, the target and MTP
routed banks, and repeated-process token stability for the stated 32/256-token
greedy protocol. Experimental MTP validation covers
speculative KV/recurrent-state transactions and exact sequential verification;
it is not a current performance recommendation. Validation does not establish:

- vision encoder, image, or video support;
- exact parity with a Transformers reference implementation;
- exact token parity between the BF16 and quantized Artifact profiles;
- the published 262,144-token context limit under long-context workloads; or
- throughput or memory behavior on hardware other than the stated reference
  host and protocol.

The accepted matrices are device-specific native Runtime measurements, not
portable deployment-performance guarantees.
