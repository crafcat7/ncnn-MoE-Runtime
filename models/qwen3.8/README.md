# Qwen3.8-Flash-Next

The built-in `qwen4_exp` adapter runs the text backbone from an official
Qwen3.8-Flash-Next Safetensors package. It loads the source BF16 weights
directly, so conversion is optional.

## Admitted scope

| Area | Implementation |
| --- | --- |
| Text architecture | 48 layers in a repeated 3 Gated DeltaNet + 1 full-Attention pattern |
| Attention | Persistent Gated DeltaNet state and QSA full Attention with partial RoPE, Q/K normalization, and a sigmoid output gate |
| Experts | 512 routed Experts, normalized Top-10 routing, one gated shared Expert per layer, and SiLU activation |
| Input features | PLE n-gram embedding on the checkpoint-selected layer and four-stream hyper-connections |
| Expert weights | Official file-backed BF16 or an optional checkpoint-bound MXFP4 Artifact |
| Mixed execution | Vulkan Dense projections, Gated DeltaNet state, and resident compatible MXFP4 Experts; CPU fallback remains available |
| Text input | Official tokenizer and chat template through the Python wrapper |

The vision encoder and multimodal token path are not admitted. The checkpoint's
MTP tensors are also not executed, so the optional Artifact contains only the
48 target-model Expert banks.

## Build and run

Build the long-lived worker from the repository root:

```powershell
cmake -S . -B build-ncnn
cmake --build build-ncnn --config Release `
  --target ncnn_moe_worker --parallel
```

Inspect the automatically selected memory and device plan before generation:

```powershell
python tools\ncnn_moe.py inspect `
  --model D:\Models\Qwen3.8-Flash-Next --hybrid
python tools\ncnn_moe.py chat `
  --model D:\Models\Qwen3.8-Flash-Next --hybrid
```

Use `--cpu` for a portable comparison. `--host-memory-mb`,
`--expert-cache-mb`, and `--expert-gpu-cache-mb` remain general Runtime
controls; the model does not require a Qwen-specific scheduling option.

## Optional MXFP4 Artifact

Compile the routed BF16 Expert banks when lower Expert storage and bounded GPU
residency are desired:

```powershell
python tools\compile_qwen3_8_artifact.py `
  D:\Models\Qwen3.8-Flash-Next
```

The compiler requires NumPy and about 60 GiB of free space. It writes
`ncnn-moe-qwen3.8-mxfp4.safetensors` atomically beside the official shards and
does not modify them. Use `--workers N` to bound conversion concurrency and
`--overwrite` to replace an existing Artifact.

The adapter binds the Artifact to the exact local `config.json` and
Safetensors index, then validates every target Expert bank. Existing v1
Artifacts that also contain an unused MTP bank remain readable; newly compiled
Artifacts omit that bank. Remove the Artifact to return to the BF16 source
profile.

MXFP4 is a lossy weight profile and does not promise token parity with BF16.
Validate task quality on the target workload. It is also not proof of an
end-to-end speedup by itself: compare prompt rate, generation rate, cache
hits, GPU Expert executions, and submit/wait time under the same prompt and
memory plan.

Hybrid execution is not bitwise identical to CPU execution. A cold Expert may
fall back to CPU before the same weight becomes GPU-resident, and the two
floating-point reduction paths can select different greedy tokens when logits
are nearly tied. Use the CPU profile or a fixed warmed residency plan when
cross-process token reproducibility is required.
