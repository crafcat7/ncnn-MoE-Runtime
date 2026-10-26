# Model Catalog

`models/` is the public entry point for model compatibility and execution
instructions. Each model family owns its checkpoint requirements, supported
execution paths, runtime options, and validation commands. Downloaded model
files stay under the ignored model-family
directories in `models/` and are not version-controlled.

## Supported models

| Model family | Package | Weights | Execution and performance |
| --- | --- | --- | --- |
| GPT-OSS-20B | Hugging Face Safetensors / native GGUF | BF16, native MXFP4, INT4 or Qn_K | [GPT-OSS](gpt-oss/README.md#reference-performance) |
| GPT-OSS-120B | Hugging Face Safetensors / native GGUF | BF16, native MXFP4, INT4 or Qn_K | [GPT-OSS](gpt-oss/README.md#reference-performance) |
| DeepSeek-V4-Flash | Hugging Face Safetensors / native target GGUF | FP8/FP4, native INT4 or Qn_K | [DeepSeek V4 Flash](deepseek-v4/README.md#deepseek-v4-flash-matrix) |
| DeepSeek-V4-Flash-DSpark | Hugging Face Safetensors; GGUF target-only | FP8/FP4, native INT4 or Qn_K | [DeepSeek V4 Flash DSpark](deepseek-v4/README.md#deepseek-v4-flash-dspark-matrix) |
| Qwen3.6-35B-A3B text backbone | Hugging Face Safetensors / native GGUF | BF16, native INT4/Qn_K, or optional compiled MXFP4 Artifact | [Execution and performance](qwen3.6/README.md) |
| Qwen3.5 / Qwen3.6 / Qwen3.8-27B dense text architecture | Hugging Face Safetensors / native GGUF (`qwen3_5`) | BF16, native AWQ/GPTQ INT4, or GGUF Qn_K | [Dense admission and validation boundary](qwen3.6/README.md#dense-text-models) |
| Qwen3.8-Flash-Next text backbone | Hugging Face Safetensors / native GGUF | BF16, native INT4/Qn_K, or optional compiled MXFP4 Artifact | [Admission and execution](qwen3.8/README.md) |

## Execution capability

| Capability | GPT-OSS | DeepSeek V4 Flash | DeepSeek V4 Flash DSpark | Qwen3.6-35B-A3B | Qwen3.8-Flash-Next |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct model loading | Verified | Verified | Verified | Verified, text weights only | Verified, text weights only |
| Portable CPU backend | Yes | Yes | Yes | Yes | Yes |
| ncnn/Vulkan mixed backend | Dense projections, Attention and eligible native Experts | Dense projections and eligible native Experts; CPU cache logic | Same | Dense projections, recurrent state and eligible native Experts | Dense projections, Gated DeltaNet state and eligible native Experts |
| Native low-bit routed Experts | MXFP4, INT4/Qn_K | FP4 E2M1, INT4/Qn_K | Same | INT4/Qn_K; optional compiled MXFP4 | INT4/Qn_K; optional compiled MXFP4 |
| Attention | Full/sliding GQA | CSA/MLA, window + compressed history | Same | Gated DeltaNet + gated full GQA | Gated DeltaNet + QSA full Attention |
| Hyper-connections | No | mHC | mHC | No | mHC |
| Hash routing and shared Expert | No | Yes | Yes | Gated shared Expert; no hash routing | PLE n-gram input and gated shared Expert |
| Bounded on-demand Expert cache | Yes | Yes | Yes | With the optional Artifact; BF16 uses OS-managed mappings | With the optional Artifact; BF16 uses file-backed mappings |
| Dense mmap | Yes | Yes | Yes | Yes | Yes |
| Optional on-demand Expert mmap | Yes | Yes | Yes | Artifact supports the Expert cache path; BF16 mappings are direct | Artifact and BF16 Experts use the bounded storage path |
| Text prompt wrapper | Harmony | Official checkpoint encoding and tokenizer | Same | Official tokenizer and chat template | Official tokenizer and chat template |
| Model-provided speculative execution | No | No | DSpark | One-layer MTP with the compiled Artifact | Not admitted |
| Vision input | No | No | No | Not admitted | Not admitted |

`MoeModelDescriptor` is the public model-neutral descriptor. Family-specific
recognition and tensor names remain inside the production adapters; Prefill,
Decode, routing, Expert storage, and scheduling consume compiled plans.
`LayerDescriptor::dense` describes an unconditional gated FFN. Its gate, up,
and down dtypes are independent; layers with a dense FFN do not declare MoE
Experts. A fully dense model sets the global Expert count and Top-K to zero.

Qwen3.8 also accepts raw U8 Q2_K–Q8_K routed-Expert banks in runtime
Safetensors. Formats may differ between layers; gate/up and down within an
Expert bank must share a dtype. This is distinct from directly loading GGUF.

## Quantized package import

The CPU and Vulkan Linear paths read affine grouped INT4 weights directly. Packed integer
codes, FP32 scales, zero points, and optional GPTQ `g_idx` remain compressed;
Prefill and Decode do not create complete floating-point weight matrices.
Portable scalar and AArch64 NEON kernels share the existing Linear dispatch.
Attention, dense/shared FFNs, independent routed Experts, and the output head
can use different weight dtypes. INT4 and Qn_K share the existing Vulkan
projection, Expert admission, cache, and device-selection paths. Quantized FFNs
keep intermediate projections and activation on the device in one submission;
quantized Gated DeltaNet reuses the existing device recurrence and state cache.
CPU execution remains available through the existing backend policy.

AWQ/GPTQ Safetensors directories for all built-in families load directly through
`Runtime::load_model` and the existing CLI: GPT-OSS-20B/120B, DeepSeek V4 Flash
and DSpark, Qwen3.5/3.6 dense and MoE text backbones, Qwen3.8-27B, and
Qwen3.8-Flash-Next. AWQ asymmetric GEMM 4-bit and legacy GPTQ 4-bit are
admitted; `group_size=-1`, grouped scales, and GPTQ activation-order `g_idx`
are supported. Source integer weights are repacked without floating-point
expansion. Ordinary F16/F32 source tensors normalize to BF16; DeepSeek's
FP32 hyper-connection parameters, routing selection biases, sinks, and
compressor positions retain FP32.

`tools/import_quantized.py` produces a new Safetensors model directory with
its tokenizer assets, leaving the source unchanged. The same adapters load
the result. Matching canonical HF configuration and tokenizer assets are
required for offline GGUF import.

All built-in model families accept native GGUF through the existing
`--model` option, using a GGUF file or a directory containing a single package.
Directory selection matches only files directly in that directory; it does
not search subdirectories. Standard split shards are resolved in the same
directory. Multiple packages require an explicit file or package directory.
GGUF architecture names `deepseek4`, `gpt-oss`,
`qwen35`, `qwen35moe`, and `qwen4exp` select the existing model adapters.
Model configuration, vocabulary, merges, and supported tokenizer profiles
come from GGUF metadata. BF16/F32 projections and native Qn_K blocks use their
existing kernels; MXFP4 routed Experts are read from the original GGUF ranges
through the existing on-demand host cache and Hybrid GPU Expert cache. Only
selected Expert cache entries unpack the GGUF block layout in memory; no
converted weight files or tokenizer sidecars are written. Qwen normalization,
recurrent-head ordering, PLE, and indexer layouts are restored at the tensor
boundary. Packed recurrent output weights retain their codes and carry an
activation column map. Unknown tokenizer profiles or chat templates fail
explicitly. Target packages containing MTP tensors run their target layers;
draft-only packages and missing DSpark draft weights do not enable speculation.

```powershell
python tools/ncnn_moe.py run --model "./models/DeepSeek-GGUF" `
  --backend hybrid --expert-memory on-demand --cpu-packed-weights off `
  --prompt "Reply OK" --max-new-tokens 32
```

| Source | Imported storage and current boundary |
| --- | --- |
| Plain BF16/F16/F32 Safetensors | BF16 floating-point weights; required DeepSeek FP32 auxiliary parameters retained |
| AWQ | Asymmetric GEMM 4-bit packing; integer codes, scales and zero points retained as INT4 |
| GPTQ | Legacy 4-bit packing, including `g_idx`/`desc_act`; retained as INT4 |
| FP8 Safetensors | Native E4M3 weights with 128×128 scale blocks; F32/F16/BF16/E8M0 scales retain their values |
| Native GPT-OSS / DeepSeek MXFP4 Safetensors | Original packed Expert blocks and scales |
| GGUF | Little-endian `qwen35`, `qwen35moe`, `qwen4exp`, `gpt-oss`, and `deepseek4`; single or standard split files |

GGUF Embedding, Attention, output-head, FFN, shared-Expert and routed-Expert
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS and Q2_K–Q8_K blocks remain
lossless and use native CPU/Vulkan kernels. GPT-OSS
and DeepSeek MXFP4 Expert blocks are also retained. Mixed projection formats
are supported. Embedding and PLE decode only selected rows; recurrent output
projections permute activations while retaining packed weights. Flash-Next
PLE, QSA, and gated-residual matrices, and DeepSeek hyper-connection and DSpark
matrices use their actual supported encoding. No complete quantized matrix is
silently expanded to BF16. Small floating-point normalization and convolution
tensors keep the existing representation. Q8_0 retains its FP16 scale and
32 signed values in each 34-byte block, including DeepSeek MLA, continuous
device execution, and Shared/routed Expert caching. Other unsupported GGUF
quantized encodings are explicitly rejected.

CPU/GPU placement uses the existing device and residency policy. INT4/Qn_K
Experts, grouped latent output projections, and eligible dense projections use
native Vulkan operators. FP8 dense/shared projections reuse their existing
Vulkan path; routed FP8 Expert matrices use the existing CPU fallback.
Memory planning counts actual packed weights and quantization metadata.
GPU residency counts aligned packed payloads, scales, zero points, group indices,
column maps, and biases. Runtime metrics expose actual `int4_dispatches` and
routed Expert GPU executions, so mixed-mode tests verify device execution.
The existing GPU cache budget follows live device memory and is capped by the
eligible packed weights. `--expert-gpu-cache-mb` also applies to native quantized
Experts with CPU weights in eager mode. Warm-cache tests reject repeated stores
and evictions across identical sessions.
Keep `--cpu-packed-weights off` for native imported packages; no second
persistent packed layout is required.

```sh
python -m pip install -e '.[quantized]'
python tools/ncnn_moe.py run --model /models/Qwen-AWQ --prompt 'Reply OK'
python tools/import_quantized.py /models/Qwen-AWQ /models/Qwen-imported
python tools/import_quantized.py /models/Qwen-Q4_K_M.gguf /models/Qwen-imported \
  --config /models/Qwen-original-config
python tools/ncnn_moe.py run --model /models/Qwen-imported --prompt 'Reply OK'
```

Choose a fresh output directory for each import. Offline GGUF import requires matching
original `config.json`, `tokenizer.json`, `tokenizer_config.json`, and chat
template assets. Imported target GGUF files skip MTP tensors for target-only
execution. For DeepSeek target GGUF, the output configuration omits DSpark
draft fields to accurately describe the tensors present. DeepSeek `deepseek4` target weights do not contain DSpark draft
weights: full DSpark execution requires its HF package and `mtp.*` weights.
Draft-only `dflash` GGUF is not a complete target model. AWQ GEMV, GPTQ v2 /
Marlin, compressed-tensors variants, Hadamard-transformed GGUF, and unknown
architectures are rejected explicitly.

Generated-model tests compare CPU Prefill/Decode and Vulkan execution against
independent floating-point references for all model families, including
nonzero GPT-OSS biases, DeepSeek latent Attention and hyper-connections,
Flash-Next PLE/QSA/DeltaNet, source/imported INT4, mixed Q4_K/Q6_K weights,
packed-byte accounting, and warm-cache reuse. FP8 end-to-end tests compare
source CPU execution with imported CPU/Vulkan execution; independent projection
oracles check FP8 decoding, activation quantization, scale groups, and row
selection. FP8 activation rounding means decoded BF16 is an observational
quality baseline, rather than an equality oracle. Import tests use real GGUF
reader/writer round trips. Official Qwen3.8-27B and Flash-Next tokenizers pass
172 parity cases each; Qwen3.5/3.6 pass 150 each. Full-size quantized checkpoint
generation quality and performance have not been validated. These checks
establish architecture, format, and execution compatibility.
Run the importer tests with `python -m unittest tools.test_import_quantized`.
Run native source/import inference and packed-memory checks with
`python tools/test_native_quantized.py --worker build/ncnn_moe_worker`.
