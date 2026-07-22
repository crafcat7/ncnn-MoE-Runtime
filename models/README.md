# MoE Model Catalog

This directory is the version-controlled catalog for model families supported by
`ncnn_moe`. It records model provenance, checkpoint layout, compatibility, and
reproducible run instructions. It does not contain model weights.

Add one directory per model family. Weight files remain excluded by the
repository-wide ignore rules.

## Current entries

- [`gpt-oss`](gpt-oss/README.md): official OpenAI GPT-OSS checkpoints in their
  original Hugging Face Safetensors layout.
