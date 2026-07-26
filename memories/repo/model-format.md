# Model Package Format

## Supported families

`BuiltinModelAdapter` recognizes official `gpt_oss` packages and maps
model-family tensor names to canonical handles before execution.

## Official GPT-OSS

The runtime reads the Hugging Face package directly:

- `config.json` supplies GPT-OSS dimensions, Attention/RoPE settings, routing,
  activation limits, and model type.
- `model.safetensors.index.json` maps tensors to shards.
- `model-*-of-*.safetensors` stores BF16 embedding, norms, Attention, router,
  biases, and LM Head tensors plus MXFP4 Expert blocks/scales.

GPT-OSS-20B uses 24 layers, hidden size 2880, 64 query heads, 8 KV heads,
head dimension 64, 32 Experts, and Top-4 routing. Even-numbered layers use a
128-token sliding window and odd-numbered layers use full causal Attention.
Expert gate/up values are interleaved, while MXFP4 scales cover groups of 32
input values. The adapter exposes each Expert as canonical `gate_up` and `down`
matrices without expanding MXFP4 to float32.

Dense Safetensors BF16 data remains BF16 in the model. ncnn InnerProduct expects
float32 model weights even when `use_bf16_storage` is enabled, so the optional
ncnn packaging path converts eligible matrices temporarily during pipeline
creation and then keeps ncnn's packed BF16 representation. Matrices over 64 MiB
remain on the portable kernel to bound temporary/prepacked memory.
