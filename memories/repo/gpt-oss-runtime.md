# GPT-OSS Runtime Milestone

## Verified checkpoint

- Model: official `openai/gpt-oss-20b` Hugging Face package.
- Checkpoint payload: three Safetensors shards totaling 13,761,264,768 bytes.
- Validation host: Ryzen 7 9800X3D, 32 GB system RAM, Windows, MinGW, OpenMP,
  and an ncnn Release build with BF16 enabled.

## Constrained-memory target

- Minimum target: run the official `openai/gpt-oss-120b` checkpoint on a host
  with 32 GB RAM and 16 GB VRAM without converting or repacking Safetensors.
  These resources are a support floor; residency scales upward on larger
  machines.
- Official architecture used by the planner: 36 layers, 128 routed Experts,
  Top-4 routing, hidden/intermediate width 2880, and MXFP4 Expert weights.
- The Expert payload estimate is approximately 56.73 GiB. A 32 GiB host selects
  file-backed Expert residency automatically and derives an approximately
  16 GiB maximum-effective cache while retaining operating-system headroom.
- Dense Attention/KV/LM Head operators remain eligible for Vulkan; routing and
  active MXFP4 Expert pairs execute on CPU.
- The official 120B checkpoint has been verified locally on the target host.
  The 15 Safetensors shards contain 687 indexed tensors and total
  65,248,893,184 bytes (60.7678 GiB). Every header range was checked against
  its physical shard boundary before execution.

## End-to-end verification

The ncnn-enabled Release runner loaded the checkpoint directly, executed all 24
Attention/MoE layers, generated logits, and greedily decoded a Harmony prompt.
The smoke prompt `Reply with exactly: OK` rendered to 61 prompt tokens. Harmony
parsing recovered an assistant analysis message followed by a final message whose
content was exactly `OK`; generation stopped on token 200002 (`<|return|>`).

Observed local timings are diagnostic, not performance guarantees:

- Model load and compile: approximately 10.7-11.0 seconds.
- 61-token prefill: approximately 8.6-9.7 seconds.
- Cached greedy decode: approximately 0.21-0.29 seconds per token.

### GPT-OSS-120B verified baseline

- Host: Ryzen 7 9800X3D, 31.14 GiB visible RAM, Windows, RTX 5070 Ti 16 GiB,
  NVIDIA driver 610.62.
- Runtime envelope: Vulkan dense/Attention, CPU MXFP4 Experts, 24 GiB host
  budget, approximately 16 GiB file-backed Expert cache, and four Expert I/O
  workers.
- Cold one-token run: 12.76-second load and 1.49-second generation, token `279`.
- Monitored warm repeat: 9.16-second load, 0.84-second generation, token `279`,
  10,935.2 MiB peak process working set, and 6,781 MiB peak NVIDIA memory
  increase over baseline.
- The one-token execution completed 36 Vulkan Attention blocks, 144 exact
  Expert misses, 144 parallel Expert tasks, and read 2,048,976,000 Expert
  bytes. Resident Expert bytes were the same and no eviction occurred.
- The Harmony prompt `Reply with exactly: OK` completed successfully and
  returned `OK` in the final channel. A mid-run sample recorded 12.29 GiB
  process working set, 9.45 GiB total NVIDIA memory use, and 52.88 GiB process
  reads under Prefill/cache pressure.
- The original MSVC scalar build reported 756.6 ms of Expert time and
  0.838 seconds for the warm first token. Runtime-dispatched AVX-512, one
  horizontal reduction per decode row, and bounded OpenMP teams produced the
  same token in warm samples of 0.266 and 0.271 seconds, with 195.2 and
  200.0 ms of Expert time. The improvement is approximately 3.1x end to end
  and 3.8x in Expert execution.
- A monitored 32-token AVX-512 run using the 16 GiB performance cache completed
  generation in 12.74 seconds (2.51 token/s), peaked at 20,710 MiB process
  working set, and increased NVIDIA memory use by 6,755 MiB. The cache recorded
  3,240 hits, 1,368 misses, 280 evictions, and 20,873,116,800 bytes read.
- Automatic dense Safetensors mmap plus zero-fill-free buffered Experts loaded
  the 120B checkpoint in 7.223 seconds and generated the established 128-token
  sequence in 40.295 seconds, versus 11.281 and 40.202 seconds for the final
  pre-mmap baseline. Peak RSS fell from 20,773 to 19,597 MiB. Explicit
  on-demand Expert mmap preserved output but took 49.757 seconds, so it remains
  an opt-in lower-allocation mode rather than the throughput default.
- A later process-level benchmark using one warm-up and three measured
  32-token runs reported a 16.822-second median (1.902 token/s), with measured
  runs ranging from 16.400 to 17.288 seconds after sustained checkpoint I/O.
  The median Expert time was 14.443 seconds and the cache hit rate was 70.29%.
  Peak measured working set was 19,815 MiB; total NVIDIA memory peaked at
  12,029 MiB, 7,368 MiB above the per-run baseline. The earlier 12.74-second
  sample remains a valid best local observation, but the repeated median is the
  more conservative sustained baseline.
- A 20 GiB cache with a 28 GiB host budget eliminated eviction and peaked at
  23,456 MiB working set, but slowed generation to 13.07 seconds. Additional
  residency beyond 16 GiB was therefore not useful on this workload.
- Four and eight I/O workers performed equivalently, and a 16 GiB cache did
  not need additional I/O concurrency. The selected performance configuration
  is four workers plus an approximately 16 GiB cache.
- The Vulkan heap budget probe reported 15,227 MiB. A 3 GiB packed-MXFP4
  Expert victim cache is verified as an opt-in L2 behind the 16 GiB RAM cache.
  In a 128-token B/A/B/A comparison, no-cache runs took 42.077 and 42.503
  seconds while GPU-cache runs took 39.580 and 40.519 seconds. The 5.3% mean
  improvement preserved identical token sequences, reduced SSD reads by
  7,349,875,200 bytes, and peaked at 14.0-14.5 GiB NVIDIA memory.
- Treat all timings as local smoke data rather than general performance
  guarantees.
- The Release cleanup and typed-flag refactor was revalidated on the same 120B
  checkpoint: the buffered on-demand path loaded in 8.375 seconds, generated
  parity token `279` in 0.819 seconds from a cold Expert cache, completed 36
  Vulkan Attention blocks and 144 exact Expert misses, and reported zero
  on-demand Expert mappings.

The shared file-backed path was also forced on for the official 20B checkpoint
with a 24 GiB host budget, 8 GiB Expert cache, and four I/O workers. The model
loaded in 6.81 seconds and generated greedy token ID `623` in 0.92 seconds.
The first token produced 96 exact cache misses, read 1,321,920,000 bytes, and
left the same number of bytes resident. This verifies the Windows shard reader,
route scheduling, MXFP4 materialization, and cache leases used by the 120B plan.
A monitored repeat peaked at 10,224.5 MiB process working set and increased
reported NVIDIA memory use by 5,288 MiB over the pre-run baseline.

Windows overlapped reads must use a separate manual-reset event for each
operation and must retrieve byte counts through `GetOverlappedResult`, including
when `ReadFile` completes immediately. Reading `lpNumberOfBytesRead` directly
caused valid multi-megabyte Expert ranges to be reported as truncated. Each I/O
worker now retains and resets its own event across reads, avoiding repeated
kernel-handle creation while preserving per-operation isolation.

## Reproduction

```powershell
python -m pip install openai-harmony
python tools\run_gpt_oss_prompt.py `
  .\build-ncnn\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-20b `
  "Reply with exactly: OK" `
  --max-new-tokens 64
```

For the constrained-memory path:

```powershell
python tools\run_gpt_oss_prompt.py `
  .\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-120b `
  "Reply with exactly: OK" `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 24576 --expert-cache-mb 16384 `
  --expert-gpu-cache-mb 3072 --expert-io-workers 4 `
  --max-new-tokens 128
```

For repeatable performance and memory measurements:

```powershell
python tools\benchmark_gpt_oss.py `
  .\build-ncnn\Release\ncnn_moe_gpt_oss.exe `
  .\models\gpt-oss\gpt-oss-120b `
  --model-revision "replace-with-Hugging-Face-commit" `
  --max-new-tokens 128 --warmup 1 --repeats 3 `
  --backend hybrid --expert-memory on-demand `
  --host-memory-mb 24576 --expert-cache-mb 16384 `
  --expert-gpu-cache-mb 3072 --expert-io-workers 4 `
  --json-output .\build-ncnn\gpt-oss-120b-benchmark.json
```

The C++ runner accepts token IDs and repeated `--stop-token` options. The Python
wrapper is intentionally separate so the C++ runtime remains independent of a
tokenizer library.

## 2026-07-25 resident 120B throughput acceptance

- The model compiler now retains a precomputed file-backed Expert-pair cache
  key. Readiness checks and acquisitions use heterogeneous string lookup and
  no longer rebuild shard-path/offset identities in the decode hot path.
- The host ARC applies a fair byte share per transformer layer before its
  normal T1/T2 victim selection. This prevents the cyclic layer scan from
  evicting the next layer's working set while retaining pure ARC behavior;
  dispatch frequency is not part of victim selection.
- Portable BF16 Router projection now dispatches its dot product to AVX2,
  AVX-512, ARM NEON, or scalar fallback. On the verified 120B host this reduced
  32-token Router time from 232.4 ms to roughly 46 ms without changing tokens.
- The old cross-layer speculative prefetch was removed because it evaluated
  the next Router before the next Attention block and therefore used an
  invalid hidden state. When the Router became faster, that path issued 11
  wrong speculative reads (145,411,200 bytes). Same-layer previous-route
  prefetch remains as a low-priority heuristic.
- Before the dynamic ExpertGroup and serial decode-regroup changes, a 28 GiB
  host budget, 20 GiB layer-aware ARC, and one same-model cache warm-up
  produced a one-Session 32-token median of 3.738 seconds, or
  **8.56 token/s**, with a 100% measured cache hit rate.
- The runner and benchmark accept `--parallel-sessions`. Two deterministic
  Sessions use the public `BatchScheduler`, share the immutable model and ARC,
  retain independent KV state, and divide physical CPU cores across workers.
  The final 64-token aggregate samples were 13.356, 13.473, and 13.015 token/s,
  for a **13.356 token/s median**. Peak RSS was 21,821.5 MiB and peak NVIDIA
  memory was 10,724 MiB.
- A later single-Session optimization replaced per-Expert conditional
  execution nodes with one dynamic `ExpertGroup` node and made token regroup
  parallelism shape-driven. The final default 32-token samples were 9.747,
  10.238, and 10.407 token/s for a **10.238 token/s median**. The established
  token sequence was identical in all runs. The report is
  `build-ncnn/gpt-oss-120b-default-32x3.json`.
- Vulkan command recording and execution are serialized through fence
  completion for shared ncnn layer/allocator safety. An earlier version
  released the lock after submit and produced a rare divergence at flattened
  token 20 under repeated two-Session tests. CPU Expert work remains
  concurrent with the other Session's GPU phase.
- The acceptance report is
  `build-ncnn/gpt-oss-120b-parallel2-safe-final-64x3.json`. Its command uses
  `D:\Models\gpt-oss-120b`, 32 tokens per Session, one cache warm-up, a
  20 GiB cache, a 28 GiB host budget, Hybrid execution, and no GPU Expert
  victim cache.
- Vulkan Attention now stores KV in a geometrically grown double-written ring.
  Each logical row is written at `slot` and `slot + capacity`, making wrapped
  sliding windows a single VkMat offset view without historical concat/Slice.
  Counters expose append, resize, and wrapped-view behavior.
- `BatchScheduler` now has a staged path for compatible idle Sessions. It
  batches shared dense stages and merges same-layer/same-ID Expert routes into
  one multi-row MXFP4 call before scattering results. Automatic two-Session
  staging requires matching input IDs; three or more stage by default, and
  flags can force or disable the path.
- The latest default one-Session 120B report is
  `build-ncnn/gpt-oss-120b-runtime-final2-32x3.json`: 10.574, 10.945, and
  11.815 token/s, or **10.945 token/s median**, with exact token parity.

## 2026-07-26 near-20 token/s service-throughput milestone

- A native Vulkan MXFP4 Expert kernel now fuses packed-block decoding and
  computation. On the RTX 5070 Ti microbenchmark, a complete Expert pair took
  0.1261 ms for one token versus 0.1603 ms before, and 0.3128 ms for four
  tokens versus 0.3972 ms, with maximum error below `2e-6`.
- The executable Vulkan Expert path owns a byte-bounded device ARC and is
  calibrated by phase against the CPU path. It self-disabled on the measured
  120B end-to-end workload because CPU/GPU contention erased the isolated
  kernel gain. This is an intentional portable policy decision, not a
  device-name special case.
- The portable CPU MXFP4 kernel groups two output rows. On this host that
  changed a 16-token one-session run from 10.868 to 11.609 token/s and reduced
  Expert time from 787.6 to 723.9 ms.
- ARM SVE2 is an optional translation unit selected by runtime measurement
  against NEON. The x64 build cannot validate its ARM code generation.
  VNNI/AMX capability detection is present, but exact MXFP4-by-FP32 arithmetic
  cannot use integer dot-product instructions without adding activation
  quantization and an explicit accuracy contract.
- The Vulkan Attention path now fuses QKV layout conversion and RoPE. It hit
  576 of 576 eligible layers and changed the 16-token one-session median from
  11.609 to **11.941 token/s**, while Attention time fell from 549.1 to
  512.8 ms.
- Multi-Vulkan placement is concurrency-aware. A static 29/7 split across the
  RTX GPU and the slower integrated GPU regressed aggregate throughput to
  7.979 token/s. The corrected objective selected `device 0: 36 layers` and
  recovered 18.599 token/s; candidates are no longer used merely because they
  exist.
- The multi-session scheduler measures staged and independent execution by
  context/batch bucket, explores both policies, and switches with 5%
  hysteresis. Its default Expert-worker budget divides detected logical CPUs
  across concurrent sessions.
- The accepted short-window two-session report is
  `build-ncnn/gpt-oss-120b-adaptive-parallel2-threads8-16x3.json`: 19.707,
  19.977, and 20.302 aggregate token/s, for a **19.977 token/s median**.
  Peak RSS was 15,430.6 MiB and peak NVIDIA allocation was 10,644 MiB.
- A later identical-policy default run varied from 17.615 to 19.585 token/s.
  Therefore 20 token/s is a demonstrated short-window aggregate operating
  point, not a stable floor and not one-session throughput.
- A 32-token-per-session attempt with a 20 GiB Expert cache and 28 GiB host
  budget entered severe memory pressure on the 32 GiB test machine and was
  terminated without an acceptance report. Do not extrapolate the short
  resident-window result to sustained or long-context service; use a smaller
  cache or more physical memory and remeasure.

## 2026-07-26 adaptive online Decode SDPA

- Single-token FP32 Attention now has a native online-Softmax shader. It reads
  the double-written KV ring, combines QK/Softmax/PV, and writes the
  output-projection input layout directly, eliminating the score matrix,
  Permute, and Reshape.
- Learned sink logits live in an immutable Vulkan buffer. Fused Decode does
  not build or upload the expanded mask; Prefill and the ncnn fallback retain
  the existing mask path.
- A device/shape/context-bucket policy compares two samples per path, requires
  a 2% fused advantage, and probes every 256 blocks. It has no model or device
  name branches.
- Forced same-build 120B A/B reduced Attention from 506.537 to 494.455 ms and
  increased throughput from 12.234 to 12.397 token/s. Tokens matched exactly.
- The final automatic one-Session report reached 12.359 token/s median and
  493.526 ms Attention with 538/576 fused blocks. Its report is
  `build-ncnn/gpt-oss-120b-decode-sdpa-maskless-auto-final-16x3.json`.
- A no-warm-up counter run reduced auxiliary transfers from 1,728 uploads /
  1,769,472 bytes to 1,190 uploads / 337,920 bytes.
- The corresponding two-Session samples were 19.868, 19.847, and 19.597
  token/s, for a 19.847 median. The best historical short-window median
  remains 19.977 token/s; neither result proves a stable 20 token/s floor.

## 2026-07-26 direct QKV-to-KV-ring Decode fusion

- One-row ncnn Vulkan outputs include allocation-tail padding. The QKV+RoPE
  helper previously compared allocation capacity with logical element count,
  so Decode silently fell back even when the logical `dims/w/h` shape was
  valid. Validation now uses the logical layout.
- Single-token QKV+RoPE writes rotated K and V directly to both physical copies
  of the double-written ring. It removes intermediate K/V VkMats and the
  separate append dispatch. This path is independent of online versus ncnn
  SDPA; the ncnn fallback writes the learned-sink row lazily.
- `NCNN_MOE_VULKAN_QKV_RING=0` disables only direct ring output for diagnostic
  A/B. It is not a deployment recommendation.
- With online SDPA forced in both arms, the three-run 120B A/B changed median
  Attention from 494.368 to 487.553 ms and throughput from 12.380 to
  12.555 token/s. All 576 blocks used each selected path and token IDs matched.
- The best new short-window two-Session report is
  `build-ncnn/gpt-oss-120b-qkv-ring-auto-parallel2-threads8-16x3.json`:
  20.039, 21.101, and 19.863 token/s, or **20.039 token/s median**.
- After direct ring output was decoupled from SDPA selection, the final report
  used it in 576/576 blocks. Median Attention fell from 597.442 to 580.588 ms,
  but median Expert time rose from 697.505 to 760.813 ms and aggregate
  throughput varied to 19.481 token/s. This confirms a local Attention win and
  leaves CPU Expert variance as the dominant stable-20 gap.
- `tools/benchmark_gpt_oss.py` now captures child output in temporary files.
  The former `stdout=PIPE`/`stderr=PIPE` implementation sampled memory without
  draining either pipe and could deadlock verbose Windows runs after the pipe
  filled.

## 2026-07-26 near-20 stabilization follow-up

- Scheduler cold buckets now delay their first alternative staged/independent
  sample until the configured probe interval. This removes an immediate cold
  probe from short generations while retaining long-running adaptation.
- Uniform active-Expert shapes use O(1) flattened row-group lookup; mixed
  shapes retain the generic scan. The MSVC AVX-512 batch-two path also shares
  input loads across four adjacent rows, guarded by numerical tests and an
  opt-out diagnostic override.
- The accepted report
  `build-ncnn/gpt-oss-120b-uniform-expert-index-auto-16x3.json` measured
  20.315, 19.953, and 19.982 token/s for two 16-token Sessions. Median
  generation time gives **19.982 token/s**, with 716.1-740.5 ms Expert time,
  no staged batches in the short window, and exact token parity.
- Eight-row AVX-512 grouping, forced staging, aggressive OpenMP wait policies,
  parallel staged Attention, and concurrent Vulkan Attention submissions were
  all measured and rejected rather than enabled as device-specific defaults.

## 2026-07-26 lazy device-resident victim Expert execution

- A Vulkan victim entry can now lend its aligned compressed MXFP4 storage
  directly to the native Expert operator. Gate/Up and Down block/scale views
  share the resident `VkMat`; this removes a victim download and a subsequent
  executable-cache upload when the online policy selects the source.
- The first long 120B run exposed `0xC0000374`. Lookup retained the operation
  but not its owning `DeviceEntry`, so churn could move final allocator release
  onto the execution thread. `DeviceOperationLease` now pins the whole entry;
  Vulkan/portable tests and the repaired long run pass.
- Lookup no longer changes victim recency. Only a successful source execution
  touches LRU state.
- Eager operation creation was rejected. An eager pinned run reached only
  4.856 token/s, made 663 source executions, and raised Expert compute to
  24.19 seconds despite reducing downloads. Building an operation for every
  victim admission also contended on the Vulkan command mutex even when the
  scheduler chose CPU.
- The accepted infrastructure is lazy and conservative: entries store compact
  shape/offset/bias metadata, operation/bias creation occurs only for an
  admitted probe, CPU phase data is required first, probes occur once per 64
  eligible misses, and 0.90/1.02 hysteresis controls broad use. The policy has
  no model or device-name branch.
- Device-resident victim execution defaults to true. Its compatibility
  fallback is represented by
  `RuntimeOptionDisableGpuVictimExecution` in
  `RuntimeOptions::flags`; `--disable-gpu-victim-execution`
  provides same-binary A/B. Session/runner/JSON metrics expose source hits,
  misses, executions, and failures.
- A high-background-load paired 96-token/two-Session screen measured 4.294
  token/s enabled versus 4.245 disabled. The enabled arm made 10 direct
  executions, had zero failures, and reduced victim restore time from 3.107 to
  2.710 seconds with identical tokens. This validates bounded behavior, not a
  statistically significant speedup. The earlier 9.546 token/s three-run
  staging-off median remains the near-10 long-window reference.
- Verification after the final lazy change: MSVC Vulkan 3/3, MinGW Vulkan
  3/3, portable 2/2, Python benchmark parser compilation, and `git diff
  --check` all pass.

## 2026-07-26 runtime-overhead audit

- Configuration switches use domain-specific `uint32_t` flags. Do not pack
  mutable states protected by different mutexes into one non-atomic word.
- `CpuSessionState` owns reusable Router, Combine, staged, Attention, and
  Expert backend scratch. Overwrite-only buffers skip zero-fill; accumulation
  buffers still clear. Large tensors remain heap-backed and reuse capacity.
- Session statistics use two reusable buffers and commit with a swap, keeping
  transactional failure behavior without reallocating Expert counters each
  token. Greedy sampling has no one-element candidate allocation.
- Single-token top-k up to 16 uses a stack candidate array and persistent
  dispatch/route storage. The generic multi-token and large-top-k path remains
  available.
- Expert pair request/prefetch returns whether the pair was ready at the
  enqueue linearization point. Pending exact reads are acquired in completion
  order; the diagnostic `NCNN_MOE_EXPERT_READY_FIRST=0` restores front waiting.
- Vulkan victim operation construction occurs outside the executable backend
  mutex and rechecks normal residency after relocking. A single-device
  submission remembers that it was waited, avoiding a destructor-time second
  wait/result-vector copy.
- Staged scheduling releases Session ownership before publishing future
  completion. This closes a transient busy window that caused immediately
  following batches to fall back to independent scheduling.
- Same-binary 96-token/two-Session completion-order A/B with direct source
  disabled measured 8.946 token/s for any-ready versus 8.715 for forced-front
  (+2.65%, identical tokens). A later full-audit repeat measured 8.802 token/s:
  Router -11.4%, regroup -24.1%, Combine -8.5%, and cache wait -2.6% versus
  the 8.946 reference, while Expert compute/LM Head variance kept end-to-end
  throughput 1.6% lower.
- Keep ARC's hash + T1/T2/B1/B2 lists and mutex until profiling proves lock
  contention. A generic lock-free queue cannot atomically maintain capacity,
  ghosts, eviction, pin/lease, and lower-tier state; red-black trees and Tries
  are worse fits for stable compiled Expert keys.
