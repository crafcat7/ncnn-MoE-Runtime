# Phi-mini-MoE-instruct

The built-in `phimoe` adapter runs the text backbone from the official
[`microsoft/Phi-mini-MoE-instruct`](https://huggingface.co/microsoft/Phi-mini-MoE-instruct)
Safetensors package directly. The package uses the SlimMoE `PhiMoEForCausalLM`
architecture identified by `model_type: "phimoe"`.

No conversion, GGUF file, or generated sidecar is required. Native text input
uses the package's SentencePiece-style BPE tokenizer (`tokenizer.json`) and the
inline `tokenizer_config.json` chat template.

## Admitted scope

| Area | Implementation |
| --- | --- |
| Package | Multi-shard Hugging Face BF16 Safetensors and `config.json` |
| Text architecture | 32 decoder layers, 4096 hidden size, 32 attention heads, 8 KV heads (GQA), 128 head dimension |
| Attention | Full causal attention, query/key/value/output biases, unscaled RoPE (`rope_theta=10000`) |
| Experts | 16 routed BF16 Experts per layer, Top-2 sparse-mixer routing, 960 intermediate size, SiLU gate/up activation and down projection |
| Norms | Standard LayerNorm with weight and bias for the input, post-attention, and final norms |
| Output head | Separate `lm_head.weight` and `lm_head.bias` |
| Execution | CPU, or Hybrid with Vulkan Attention, LayerNorm, sparse-mixer router, BF16 Experts, final norm, and LM head |
| Text input | Native tokenizer and Phi chat template in Runtime |

The model's `config.json` also contains `sliding_window: 2047`, but the official
eager/SDPA implementations use full causal attention; the runtime follows that
behavior. The FlashAttention-only sliding-window path is not admitted.

This admission is text-only. The package's vision, audio, and tool-call object
paths are not exposed by the native text API.

## Download

Run from the repository root and keep the checkpoint in its ignored model
directory:

```powershell
hf download microsoft/Phi-mini-MoE-instruct `
  --local-dir .\models\phimoe\Phi-mini-MoE-instruct
```

The directory must contain `config.json`, `tokenizer.json`,
`tokenizer_config.json`, the Safetensors index, and every referenced shard.

## Run

```powershell
python tools/ncnn_moe.py run --model .\models\phimoe\Phi-mini-MoE-instruct `
  --prompt "Hi 介绍下你自己"
```

`--backend auto` selects Hybrid when a Vulkan device is available. The Vulkan
path applies LayerNorm, attention bias, the sparse-mixer router, BF16 Experts,
final norm, and the LM head bias on device; CPU remains available with
`--cpu`. Native text emits the prompt's stop-token metadata (`32000`, `32001`,
`32007`) and decodes the ByteFallback/SentencePiece token stream.

## Validation boundary

The adapter maps the package's `input_layernorm`, `post_attention_layernorm`,
`block_sparse_moe.gate`, `experts.{w1,w2,w3}`, `self_attn.*`, `model.norm`, and
`lm_head` tensors to the canonical runtime layout. Greedy decoding has been
checked against the official `modeling_slimmoe.py` implementation, including
the sparse-mixer router and LayerNorm bias. CPU and Hybrid greedy tokens are
identical on the reference prompt.

ncnn provides the Vulkan `LayerNorm` layer and its shaders. nmr's execution
graph instantiates that ncnn layer through a thin recorder, so LayerNorm
semantics and GPU shaders stay owned by ncnn. RMSNorm keeps the existing nmr
recorder; residual addition is composed with ncnn's Vulkan `BinaryOp`.
