# Example entry points

The supported user-facing example workflow is the unified Python CLI from the
repository root:

```powershell
python tools\ncnn_moe.py inspect --model PATH
python tools\ncnn_moe.py run --model PATH --prompt "Hello"
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

The unified CLI uses `build-ncnn` by default and does not search other build
directories or executables on `PATH`. On Windows its default worker path is
`build-ncnn/Release/ncnn_moe_worker.exe`; pass `--worker PATH` or set
`NCNN_MOE_WORKER=PATH` when a different build is intentionally selected.

`ncnn_moe_worker` loads Runtime/Model once and accepts one JSON object per line
on stdin. During startup it emits `init` progress events before `ready`, then
emits `token`, `metrics`, `done`, and `error` events. Its
native sessions contain token IDs and KV state only:

- Python adapters provide the official tokenizer and chat template for each
  model family.
- Python session state provides messages, context budgeting, prefix reuse,
  compaction, persistence, and continuous chat.
- The worker provides model loading, native Session state, generation, reset,
  cancellation, and runtime statistics.

The Python CLI consumes the `init` events and renders a progress bar such as:

```text
Init [====================----]  83.3% · Preparing Expert storage and caches · 2.4s
```

The progress stages are public lifecycle descriptions (`hardware`, `manifest`,
`architecture`, `memory`, `weights`, `compile`, `cache`, and `finalize`); they
are not a promise about the exact number of internal operations. If loading
fails, the last stage remains visible and the following `error` event contains
the actionable failure.

Use `inspect` before a run to see the detected CPU/Vulkan devices and effective
resource plan; human-readable memory and I/O sizes are shown as decimal `GB`.
The JSONL protocol still reports exact `*_bytes` fields. `Auto` selects Hybrid only when a real Vulkan device is
available; otherwise it falls back to CPU. Use `--cpu`, `--hybrid`, or the
advanced memory, Expert-cache, I/O, and device options when an explicit plan is
needed. Hybrid automatically issues bounded cache-prefetch hints before CPU
Expert kernels; there is no separate prefetch backend. A plan change that
requires a new worker is reported instead of being
silently applied to the current session.

Hybrid execution automatically sends resident, backend-compatible routed
Experts to Vulkan, including single-token decode waves; non-resident or
unsupported requests retain the CPU fallback. The policy is shared by all
adapters and does not require a model-specific GPU-decode option. The Qwen3.8-
Flash-Next `qwen4_exp` adapter can use its checkpoint-bound MXFP4 artifact to
reduce routed-Expert storage and improve GPU residency; without it, the
official file-backed BF16 Experts remain supported. Use `--cpu` for a CPU-only
comparison.

CPU Expert weight repacking is an explicit experiment. Pass
`--cpu-packed-weights on` to enable it for supported MXFP4-Q8 or Qn_K weights;
when the option is absent, the worker keeps repack off and does not reserve an
in-memory packed sidecar.

The CLI is intentionally the only public text entry point. The four
model-named native executables (`ncnn_moe_gpt_oss`,
`ncnn_moe_deepseek_v4`, `ncnn_moe_qwen3_6`, and `ncnn_moe_qwen3_8`) remain
reference/benchmark
targets because the benchmark harness and CTest fixture use their positional
token-ID, prompt-file, and multi-session controls. They are not built by the
fast default configuration and are not required for normal text or chat usage.
Enable them at configure time when needed:

```powershell
cmake -S . -B build-reference -DNCNN_MOE_BUILD_REFERENCE_RUNNERS=ON
cmake --build build-reference --config Release `
  --target ncnn_moe_gpt_oss ncnn_moe_deepseek_v4 ncnn_moe_qwen3_6 ncnn_moe_qwen3_8 --parallel
```

## Metrics trace

The CLI streams text and shows reasoning by default, while periodic `metrics`
events are disabled by default. Enable the live trace without losing the final
`done` statistics when needed:

```powershell
python tools\ncnn_moe.py run --model PATH --prompt "Hello" --metrics
python tools\ncnn_moe.py run --model PATH --prompt "Hello" --metrics-interval-ms 0
python tools\ncnn_moe.py run --model PATH --prompt "Hello" --no-stream --hide-reasoning
```

The native JSONL request accepts `"metrics_enabled": false` as the equivalent
protocol-level switch. `stats`, cache counters, and final telemetry remain
available.

The human-readable trace is intentionally a summary rather than a dump of
Runtime internals:

```text
metrics
  Prompt: 3.70 t/s | Generation: 7.30 t/s · TTFT 1.20 s · TPOT 250.00 ms
  Expert: cache hit 12 · cache miss 3 · IO 1.23 GB
  CPU: expert compute 890.00 ms · process 759.1%
  GPU: submit 42 · wait 12.00 ms · kernel 8.00 ms · utilization 4.0%
```

The JSONL payload also exposes `prompt_tokens_per_second` and
`generation_tokens_per_second` (plus the shorter `prompt_tok_per_second` and
`generation_tok_per_second` aliases) for internal benchmark tooling. It keeps
exact microseconds and bytes for tooling. `IO` in the
CLI is the Runtime's logical Expert-cache read volume, while process-level
read/write counters remain under `process` for diagnostics. `GPU kernel time`
is the Expert GPU backend wall time; Vulkan device timestamp queries are not
assumed, so it is not presented as a complete device-wide kernel timeline.
When the selected backend is CPU-only, GPU runtime counters are emitted as
`null` with a reason rather than as zero-valued measurements.
`Prompt` measures input-token prefill throughput until the first generated token
is ready. `Generation` measures steady-state output throughput after that first
token; it is also exposed under the compatibility field
`decode_tokens_per_second`. `TTFT` uses the same prompt boundary and `TPOT`
is the average interval for the remaining output tokens.

The native public boundary is `RuntimeConfig`, token IDs, token-ID generation
config, Session state, and the stable `SessionMetrics` view. The generic
`SamplingOptions` controls distribution selection over token IDs; it does not
own a tokenizer or chat policy. Tokenizers, chat templates, stop-policy
selection, reasoning/final-channel decoding, conversation history, and
human-readable formatting remain in the Python adapters and CLI.

## Optional Python dependencies

For a complete local Example setup, run this from the repository root:

```powershell
python -m pip install -r requirements.txt
```

This installs all supported adapter, model-download, Qwen artifact-build, TUI,
and telemetry dependencies. The DeepSeek-V4 `encoding/encoding_dsv4.py` file
is supplied by the checkpoint and is self-contained; the ncnn Runtime CLI does
not require the model repository's PyTorch inference stack. The minimal and
grouped installations remain available when a smaller environment is
preferred:

```powershell
python -m pip install -e .
python -m pip install -e ".[tui]"
python -m pip install -e ".[gpt-oss]"
python -m pip install -e ".[hf]"
python -m pip install -e ".[telemetry]"
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
