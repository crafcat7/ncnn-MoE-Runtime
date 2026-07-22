# GPT-OSS

## Provenance

- Model: [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
- Publisher: OpenAI
- Checkpoint format: original Hugging Face Safetensors shards
- Runtime adapter: `gpt_oss`
- Compatibility: verified with the 20B checkpoint on the mixed ncnn
  Vulkan-Attention / CPU-Expert backend

## Required checkpoint layout

Download the official checkpoint to a directory outside this repository. The
runtime expects `config.json`, `model.safetensors.index.json`, and every
Safetensors shard referenced by the index.

For example, after installing `huggingface_hub`:

```powershell
huggingface-cli download openai/gpt-oss-20b --local-dir D:\Models\gpt-oss-20b
```

## Run

```powershell
.\build\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-20b 0 --max-new-tokens 1 --hybrid
```

Use `tools/run_gpt_oss_prompt.py` for Harmony-formatted text prompts. See the
root README for build prerequisites and complete command-line options.
