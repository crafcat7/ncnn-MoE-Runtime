# Ling-3.0-tiny

The built-in `bailing_hybrid` adapter runs the text backbone from the official
[`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
Safetensors package directly. The package identifies the architecture as
`model_type: "bailing_hybrid"` (`BailingMoeV3ForCausalLM`).

No conversion or generated sidecar is required. Native text input uses the
package's ByteLevel BPE tokenizer (`tokenizer.json`) and the inline
`tokenizer_config.json` chat template.

## Admitted scope

| Area | Implementation |
| --- | --- |
| Package | Official multi-shard Hugging Face BF16 Safetensors and `config.json` |
| Text architecture | 24 decoder layers in a repeated 3 KDA + 1 MLA pattern |
| KDA | Short depthwise convolution, normalized Q/K, per-channel bounded decay gate, FP32 recurrent state, gated RMSNorm, and per-Session state |
| MLA | Query/compressed-KV LoRA with expanded value heads, interleaved partial RoPE, and a head-wise output gate |
| Experts | 128 routed BF16 Experts, group-limited sigmoid Top-8 routing with a selection bias, one BF16 shared Expert per layer, and SiLU activation |
| Dense layer | The first layer uses a dense BF16 SiLU gate/up/down FFN |
| Norms | RMSNorm for the input, post-attention, and final norms |
| Execution | CPU, or Hybrid with Vulkan KDA, MLA, group-limited routing, routed/shared Experts, and LM head |
| Text input | Native tokenizer and Ling chat template in Runtime |

This admission is text-only. The package's MTP/next-token payload is not
present in the distributed checkpoint and is not executed.

## Download

Run from the repository root and keep the checkpoint in its ignored model
directory:

```powershell
hf download inclusionAI/Ling-3.0-tiny `
  --local-dir .\models\ling\Ling-3.0-tiny
```

The directory must contain `config.json`, `tokenizer.json`,
`tokenizer_config.json`, the Safetensors index, and every referenced shard.

## Run

```powershell
python tools/ncnn_moe.py run --model .\models\ling\Ling-3.0-tiny `
  --prompt "Hi 介绍下你的能力"
```

`--backend auto` selects Hybrid when a Vulkan device is available. KDA, MLA,
group-limited routing, routed/shared Experts, and the LM head use the Vulkan
paths. Each KDA/MLA layer is recorded as a resident Vulkan segment, and the
segments chain through the Expert wave. Use `--cpu` for a fully portable path.
Native text emits the package's `<|role_end|>` stop token and preserves the
chat template's thinking boundary.

## Validation boundary

The adapter maps `input_layernorm`, `post_attention_layernorm`,
`mlp.experts.*`, `mlp.shared_experts.*`, `mlp.gate.*`, and the KDA/MLA
attention tensors to the canonical runtime layout. The three depthwise
convolutions and the three Q/K/V projections are fused row-wise. Greedy
decoding has been checked against the official modeling code for the same
prompt; the runtime reproduces its continuation after the reference MLP,
group-limited router, shared Expert, KDA recurrence, and absorbed MLA path.
CPU and Hybrid greedy tokens are identical on the reference prompt. Hybrid
uses the existing ncnn Vulkan Gated DeltaNet and latent-Attention cores with
an extended decay-gate/expanded-value shader contract, the existing Vulkan
router extended with group-limited Top-K, and the existing Vulkan Linear and
Qn_K grouped projection operators. The absorbed query and grouped value
projection stay on the device across the resident segment chain.
Batched Expert execution still crosses the host ExpertStore wave once per
layer, so the BF16 Expert working set is the remaining decode bottleneck.
The Runtime automatically selects the largest contiguous layer prefix whose
resident Expert bank fits the live GPU budget and streams the remaining tail
through the bounded ExpertCache. No CLI flag is required. Ling is eligible
for this selection because the dense first layer, MLA segments, and device
routing are all admitted. On a 16 GiB device the BF16 package banks 21 of the
24 layers (20 routed layers plus the dense layer) and streams the last three;
the resident chain removes most per-layer ExpertStore waves. The remaining
decode gap to a fully resident MXFP4 bank comes from the BF16 native bank
kernel and the streamed tail. Compile-time INT4/MXFP4 Experts make the whole
bank resident and switch the same code to the packed fast path. Native tokenizer chat and raw encoding match the official
tokenizer on ASCII, CJK, emoji, newline, multi-turn, and reasoning-history
prompts.
