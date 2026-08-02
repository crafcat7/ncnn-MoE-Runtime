# Example entry points

The supported user-facing example workflow is the unified Python CLI from the
repository root:

```powershell
python tools\ncnn_moe.py inspect --model PATH
python tools\ncnn_moe.py run --model PATH --prompt "Hello" --stream
python tools\ncnn_moe.py chat --model PATH
```

Build the long-lived native worker with the examples target:

```powershell
cmake -S . -B build-ncnn `
  -DNCNN_MOE_BUILD_TESTS=OFF `
  -DNCNN_MOE_BUILD_REFERENCE_RUNNERS=OFF `
  -DNCNN_MOE_BUILD_MXFP4_BENCHMARK=OFF
cmake --build build-ncnn --config Release --target ncnn_moe_worker --parallel
```

`ncnn_moe_worker` loads Runtime/Model once and accepts one JSON object per line
on stdin. It emits `ready`, `token`, `metrics`, `done`, and `error` events. Its
native sessions contain token IDs and KV state only:

- Python adapters provide the official tokenizer and chat template for each
  model family.
- Python session state provides messages, context budgeting, prefix reuse,
  compaction, persistence, and continuous chat.
- The worker provides model loading, native Session state, generation, reset,
  cancellation, and runtime statistics.

Use `inspect` before a run to see the detected CPU/Vulkan devices and effective
resource plan. `Auto` selects Hybrid only when a real Vulkan device is
available; otherwise it falls back to CPU. Use `--cpu`, `--hybrid`, or the
advanced memory, Expert-cache, I/O, and device options when an explicit plan is
needed. A plan change that requires a new worker is reported instead of being
silently applied to the current session.

The CLI is intentionally the only public text entry point. The three
model-named native executables (`ncnn_moe_gpt_oss`,
`ncnn_moe_deepseek_v4`, and `ncnn_moe_qwen3_6`) remain reference/benchmark
targets because the benchmark harness and CTest fixture use their positional
token-ID, prompt-file, and multi-session controls. They are not built by the
fast default configuration and are not required for normal text or chat usage.
Enable them at configure time when needed:

```powershell
cmake -S . -B build-reference -DNCNN_MOE_BUILD_REFERENCE_RUNNERS=ON
cmake --build build-reference --config Release `
  --target ncnn_moe_gpt_oss ncnn_moe_deepseek_v4 ncnn_moe_qwen3_6 --parallel
```

## Metrics trace

The CLI emits periodic `metrics` events by default. Scripted generation can
disable that live trace without losing the final `done` statistics:

```powershell
python tools\ncnn_moe.py run --model PATH --prompt "Hello" --no-metrics
python tools\ncnn_moe.py run --model PATH --prompt "Hello" --metrics-interval-ms 0
```

The native JSONL request accepts `"metrics_enabled": false` as the equivalent
protocol-level switch. `stats`, cache counters, and final telemetry remain
available.

## Optional Python dependencies

```powershell
python -m pip install -e .
python -m pip install -e ".[tui]"
python -m pip install -e ".[gpt-oss]"
python -m pip install -e ".[hf]"
```

The `ncnn-moe` console script is equivalent to
`python tools\ncnn_moe.py`. Without the optional TUI packages, `chat` falls back
to a plain Python REPL.

## Local state

The CLI stores persistent session history, user configuration, and tuning
profiles under `.ncnn-moe/` in the project root:

```text
.ncnn-moe/
  config.json
  tuning_profiles.json
  sessions/<session-id>.json
```

Use `--config-dir PATH` or `NCNN_MOE_CONFIG_DIR` to select another location.
`--ephemeral` keeps a chat out of persistent storage. Native KV state and
runtime Expert/GPU caches remain process-local and are not written to disk.

For the JSONL operation schema, see
`tools/ncnn_moe_protocol.py` and the native implementation in
`examples/ncnn_moe_worker.cpp`. The native boundary deliberately does not
contain tokenizer or model-family chat logic.
