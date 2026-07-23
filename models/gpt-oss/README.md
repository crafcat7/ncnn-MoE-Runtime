# GPT-OSS

## Provenance

- Model: [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
- Publisher: OpenAI
- Checkpoint format: original Hugging Face Safetensors shards
- Runtime adapter: `gpt_oss`
- Compatibility: direct loading from the official checkpoint layout; choose the
  execution backend through the runner options described below

## Required checkpoint layout

Download the official checkpoint to a directory outside this repository. The
runtime expects `config.json`, `model.safetensors.index.json`, and every
Safetensors shard referenced by the index.

For example, after installing the Hugging Face `hf` CLI:

```powershell
hf download openai/gpt-oss-20b --local-dir D:\Models\gpt-oss-20b
```

## Run

```powershell
.\build\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-20b 0 --max-new-tokens 1
```

The runner selects the available backend automatically unless one is requested:

| Flag | Backend |
| --- | --- |
| `--cpu` | CPU execution |
| `--hybrid` | Vulkan dense operators with CPU experts, when available |
| `--hybrid-prefetch` | Hybrid execution with CPU expert-cache hints |

`--hybrid` requires a Vulkan-enabled build. On macOS, also enable the
experimental MoltenVK configuration described in the root README. Operations
that the experimental backend cannot execute fall back to CPU.

### macOS

The default macOS build uses the experimental MoltenVK hybrid backend. It
requires Xcode or its Command Line Tools, a Metal-capable Mac, and a local
`libMoltenVK.dylib`; see the root README for the build command and custom
library-path configuration.

Download the model outside the repository, then export its absolute path. The
check confirms that the download contains the manifest the runtime needs before
starting a long model load:

```sh
brew install hf
export MODEL_DIR="$HOME/Models/gpt-oss-20b"
hf download openai/gpt-oss-20b --local-dir "$MODEL_DIR"
test -f "$MODEL_DIR/config.json" && echo "GPT-OSS model found"
```

After the macOS build from the root README, run the model with the backend that
matches that build:

```sh
./build-macos/Release/ncnn_moe_gpt_oss \
  "$MODEL_DIR" 0 \
  --max-new-tokens 64
```

For a Harmony-formatted text prompt:

```sh
python3 -m pip install openai-harmony
python3 tools/run_gpt_oss_prompt.py \
  ./build-macos/Release/ncnn_moe_gpt_oss \
  "$MODEL_DIR" \
  "Reply with exactly: OK" \
  --max-new-tokens 64 --stream
```

For local MoltenVK experimentation, add `--hybrid` (or
`--backend hybrid` to the Harmony wrapper) to the relevant command. Inspect the
runner's dispatch counters to see which operations ran on Vulkan. This build
embeds its local MoltenVK path and is not intended for redistribution or
production inference.

See the root README for build prerequisites and complete command-line options.
