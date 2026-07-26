# Performance Notes

## Vulkan mixed execution

- Vulkan mixed modes fuse Attention Q, K, and V into one InnerProduct created during model compilation. The temporary concatenated CPU tensor is released after synchronous Vulkan weight upload, so the model does not retain a second QKV weight copy.
- For GPT-OSS-20B, one model execution now submits 73 Vulkan linear dispatches instead of 121: 24 layers each run fused QKV, Attention output, and Router, followed by one LM Head projection.
- Each Attention block uploads hidden activations once and records RMSNorm,
  fused QKV, YaRN RoPE, GQA SDPA, output projection, residual, and compact
  KV-cache updates in one Vulkan command. Decode can replace QK, Softmax, PV,
  Permute, and Reshape with one online-Softmax shader.
- ncnn upload packing must be undone explicitly for Q/K/V and all auxiliary mask/RoPE/sink tensors because SDPA consumes pack1. The generated multi-head GQA fixture guards this requirement by comparing automatic and CPU generation across Prefill and repeated Decode.
- Vulkan KV caches expose BF16/FP32 logical state while the current ncnn
  compute storage remains unpacked FP32. The backend uses a double-written
  device ring: every KV row is written to `slot` and `slot + capacity`, so
  wrapped full/sliding windows remain one contiguous offset view. Capacity
  grows geometrically; normal Decode no longer concatenates or slices
  historical KV.
- Prefill/ncnn fallback masks and RoPE tables use FP32 staging. Prefill
  processes at most 256 tokens per execution by default, reducing the peak
  expanded mask from `O(heads * prompt^2)` to
  `O(heads * chunk * prompt)`; `SessionOptions::prefill_chunk_size = 0`
  restores one-shot batching. A fused Decode block keeps learned sinks on the
  device and skips mask construction and upload entirely.
- Activation residency ends after the Attention residual because routing and MXFP4 Experts remain on CPU.

## CPU Expert prefetch

- `VulkanWithCpuPrefetch` issues cache hints immediately before each selected Expert matrix executes. Each backing buffer is capped at 4 KiB of hints; MXFP4 blocks and scales are hinted separately.
- `Auto` remains `HybridExperts`. Prefetch behavior is CPU-cache dependent and must be selected explicitly with `--hybrid-prefetch` or the public `HybridMode::VulkanWithCpuPrefetch` value.
- Verified on an NVIDIA GeForce RTX 5070 Ti with GPT-OSS-20B, prompt token IDs `0 1 2 3`, four greedy output tokens: both mixed modes produced `4 5 14 416`, 292 linear dispatches, and 96 completed Vulkan Attention blocks. The post-cache-optimization paired smoke run measured 1.496 seconds for standard mixed execution and 1.378 seconds with CPU Expert prefetch; treat these timings as smoke data, not a benchmark.

## File-backed Expert residency

- `RuntimeOptions::expert_memory_mode` defaults to `Auto`. The pre-load memory
  planner derives a host budget from detected physical RAM, estimates dense and
  Expert bytes from `MoeIR`, and selects eager or on-demand Expert residency.
- On a nominal 32 GiB host, the official GPT-OSS-120B dimensions select an
  approximately 16 GiB file-backed performance cache: the largest capacity
  inside the default host budget after dense-weight and safety reserves. The
  estimated MXFP4 Expert payload is approximately 56.73 GiB, so eager Expert
  loading is rejected.
- The official 60.7678 GiB GPT-OSS-120B checkpoint is verified on a Ryzen 7
  9800X3D / 31.14 GiB RAM / RTX 5070 Ti 16 GiB Windows host. The original
  MSVC scalar warm run generated token `279` in 0.838 seconds, of which
  756.6 ms was Expert time.
- MSVC compiles AVX2/FMA and AVX-512 MXFP4 implementations in separate
  translation units and selects them with CPUID/XGETBV from a baseline
  translation unit. Decode kernels accumulate scaled block products in vector
  lanes and perform one horizontal reduction per complete row instead of one
  per 32-value block.
- OpenMP Expert teams are bounded to the active Expert count. Small dense
  Linear operations use up to four threads, avoiding repeated expansion from
  the Top-4 team to the default 16-thread team for each Router.
- With those changes, warm one-token samples produced token `279` in 0.266 and
  0.271 seconds with 195.2 and 200.0 ms of Expert time. This is approximately
  3.1x faster end to end and 3.8x faster in Expert execution than the original
  MSVC scalar baseline.
- A monitored 32-token run with a 16 GiB cache generated in 12.74 seconds
  (2.51 token/s), peaked at 20,710 MiB working set, and increased NVIDIA memory
  use by 6,755 MiB. It recorded 3,240 hits, 1,368 misses, 280 evictions, and
  20,873,116,800 bytes read.
- A 20 GiB cache with a 28 GiB host budget removed all eviction and peaked at
  23,456 MiB working set, but generation slowed to 13.07 seconds. The selected
  policy therefore maximizes useful residency rather than allocating memory
  after throughput has saturated.
- Increasing I/O workers from four to eight at a 12 GiB cache produced only a
  noise-level change. Four workers remain the selected configuration.
- Runtime Vulkan capability reporting exposes the device heap budget. The
  RTX 5070 Ti reported 15,227 MiB. An opt-in device-local victim cache now
  retains packed MXFP4 Expert pairs evicted from the RAM cache. It restores
  packed bytes to CPU memory before the existing AVX-512 Expert kernels run;
  it does not execute Expert arithmetic on Vulkan.
- One 120B token read 2,048,976,000 Expert bytes across 144 misses with no
  eviction. A Harmony prompt returned final-channel `OK`; a mid-run pressure
  sample observed 12.29 GiB working set, 9.45 GiB total NVIDIA memory use, and
  52.88 GiB process reads.
- A real GPT-OSS-20B forced-on-demand smoke run on Windows loaded in 6.81
  seconds and generated the parity token `623` in 0.92 seconds. It read
  1,321,920,000 Expert bytes for the first token with no eviction. A monitored
  repeat peaked at 10,224.5 MiB process working set and a 5,288 MiB increase in
  reported NVIDIA memory use over baseline.
- The file-backed cache reduces resident Expert memory; it does not reduce the
  checkpoint's SSD footprint or eliminate I/O-sensitive cold-token latency.

## GPT-OSS-120B optimization experiments

- Eight Expert I/O workers are counterproductive on the verified 9800X3D /
  RTX 5070 Ti host once the 16 GiB cache is under steady eviction pressure. A
  paired 32-token run took 16.51 seconds with eight workers and 14.93 seconds
  with four workers. Keep four as the default; more CPU threads do not imply
  more useful SSD concurrency.
- CPU cache hints showed an apparent 7.1% improvement across alternating
  eight-token smoke runs, but a 32-token sandwich comparison converged to
  13.995 versus 14.018 seconds. `VulkanWithCpuPrefetch` therefore remains
  explicit rather than becoming the `Auto` policy.
- A native packed-MXFP4 Vulkan prototype produced the correct token but did not
  beat the AVX-512 CPU Expert path. The first shared-memory kernel spent about
  0.99 seconds in Expert execution for one token; subgroup reduction reduced
  that to roughly 0.43-0.62 seconds. Four-expert batching took 6.35 seconds for
  eight tokens, per-Expert pipelining took 5.04 seconds, and a 4 GiB device LRU
  took 5.30-5.64 seconds with only a 32.2% hit rate. The prototype was removed.
  A viable GPU path needs a fused multi-Expert dispatch, persistent packed
  residency, and asynchronous storage/compute overlap rather than per-Expert
  submissions.
- Four-row MXFP4 CPU kernels remain rejected because they did not improve
  beyond measurement noise. The original flattened OpenMP experiment was also
  rejected because it mixed cache acquisition with compute and weakened cold
  I/O scheduling. The retained implementation first acquires Expert pairs in
  parallel, then flattens resident MXFP4 Gate/Up and Down row-pairs across
  Experts. It uses detected physical cores instead of all SMT processors.
- Windows Expert reads now reuse one manual-reset `OVERLAPPED` event per I/O
  worker thread instead of creating and closing an event for each blocks/scales
  range. This removes thousands of kernel-handle operations in a 32-token run
  without changing cache capacity, read ordering, or tensor values.
- A 3 GiB Vulkan Expert victim cache was neutral in paired 32-token runs, but a
  128-token B/A/B/A comparison showed a repeatable sustained benefit. Disabled
  runs took 42.077 and 42.503 seconds; enabled runs took 39.580 and 40.519
  seconds. Mean generation time improved by 5.3% from 42.290 to 40.049 seconds
  with identical token sequences.
- In each enabled 128-token run, the victim cache served approximately 574-575
  restores and reduced SSD bytes from 63,412,502,400 to 56,062,627,200. It
  uploaded about 46.3 GB and downloaded about 7.6 GB of packed MXFP4 data with
  zero restore failures. Peak NVIDIA memory was 14.0-14.5 GiB, leaving roughly
  1.8 GiB of physical headroom on the 16 GiB device.
- The cache copies directly when ncnn returns a host-mappable device
  allocation and records mapped store/restore counters. The RTX 5070 Ti
  validation run instead returned non-mappable device-local allocations:
  all 280 stores and 13 restores in the final 32-token smoke test used the
  Vulkan staging fallback. That run completed in 12.800 seconds with zero
  failures, 20,778 MiB peak RSS, and 13,998 MiB peak NVIDIA memory.
  The option remains disabled by default and is recommended for sustained
  sessions.

## Repeatable benchmark protocol

- `tools/benchmark_gpt_oss.py` runs fixed warm-up and measured repetitions,
  verifies identical generated token IDs, and writes raw per-run metrics plus
  medians to JSON.
- The tool samples peak process RSS and uses `nvidia-smi`, when available, to
  capture NVIDIA device-memory baseline, peak, and delta. The Windows/WDDM
  value is device-wide and should be measured on an otherwise idle GPU.
- Runner stdout/stderr are written to temporary files while memory is sampled.
  Do not replace them with unread `subprocess.PIPE` handles: verbose Windows
  runs can fill the anonymous pipe and deadlock the child before exit.
- Its official 120B one-token acceptance run generated token `279`, recorded
  144 misses and 2,048,976,000 Expert bytes read, peaked at 10,936.8 MiB RSS,
  and observed a 6,795 MiB NVIDIA-memory increase. This cold smoke sample
  validates the monitor and report schema; it is not a throughput result.
- The first full protocol run used one process-level warm-up and three measured
  32-token runs. It produced the same 32-token sequence every time, with a
  16.822-second median (1.902 token/s), 14.443-second median Expert time, and
  70.29% median cache hit rate. Peak RSS was 19,815 MiB and peak total NVIDIA
  memory was 12,029 MiB (+7,368 MiB). Individual generation times ranged from
  16.400 to 17.288 seconds, demonstrating why isolated samples are unsuitable
  for accepting small optimizations on this storage-bound workload.

## Zero-fill-free buffers and mmap

- `MxFp4ByteBuffer` allocates overwrite destinations without value-initializing
  every byte, while mapped buffers share a range owner and copies remain deep.
  This removes redundant zero writes before buffered Expert reads.
- Safetensors BF16/F32 dense tensors and eager MXFP4 ranges use copy-on-write
  file mappings automatically, with buffered fallback. The bounded on-demand
  Expert cache defaults to aligned overlapped direct reads on Windows and
  falls back to buffered reads when direct I/O is unavailable. Other
  platforms keep the portable buffered path. Explicit mmap remains available
  through `RuntimeOptionMemoryMapExperts` / `--mmap-experts`.
- On the verified GPT-OSS-120B host, the final zero-fill-free baseline loaded
  in 11.281 seconds, generated 128 tokens in 40.202 seconds, spent 32.416
  seconds in Experts, and peaked at 20,773 MiB RSS. Dense mmap plus buffered
  Experts loaded in 7.223 seconds, generated the identical sequence in 40.295
  seconds, spent 32.222 seconds in Experts, and peaked at 19,597 MiB RSS.
- Mapping on-demand Experts too preserved the sequence and peaked at 19,693
  MiB RSS, but 16,756 range mappings covering 55,375,228,800 logical bytes
  increased generation to 49.757 seconds and Expert time to 41.792 seconds.
  The measured regression is why Expert mmap is not selected by `Auto`.
- Releasing an I/O worker's completed Entry reference before notifying waiters
  is required when the cache holds exactly one Expert pair. Without this,
  a racing next acquire can mistake the worker's transient reference for an
  active compute lease and report capacity exhaustion.

## 2026-07-25 hardware-aware acceptance

- Test host: Ryzen 7 9800X3D (8 cores / 16 logical processors), RTX 5070 Ti
  16 GiB, Windows, MSVC Release, Vulkan dense/Attention, AVX-512 MXFP4.
- A 64-token GPT-OSS-20B eager run after the asynchronous command boundary,
  with no OpenMP environment override, completed in 4.4095 seconds, or
  **14.514 token/s**. It preserved the exact
  token sequence used by earlier baselines. Attention took 1.544 seconds,
  Expert wall time took 2.498 seconds, and LM Head took 0.315 seconds.
- The decisive changes were physical-core-aware grouped MXFP4 execution and a
  linear-time greedy argmax. The latter replaces a full stable sort of roughly
  200k logits per token when `temperature == 0`, while retaining the lower
  token ID as the deterministic tie-break.
- Before greedy argmax, the same automatic physical-core policy generated 64
  tokens in 5.7840 seconds (11.065 token/s). An explicit
  `OMP_NUM_THREADS=8` run generated them in 6.0472 seconds
  (10.583 token/s). Both already exceeded the requested near-10 token/s
  target; the final greedy path adds substantial margin.
- The current 120B short-run baseline disables the Vulkan victim cache:
  32 tokens completed in 15.5036 seconds (2.064 token/s), Expert wall time was
  12.497 seconds, cache hit rate was 70.51%, and 20,740,924,800 Expert bytes
  were read. Peak RSS was 19,553.6 MiB.
- Enabling a 3 GiB victim cache for the comparable current 32-token run
  completed at 1.608 token/s and served only 13 restores. Its short-session
  overhead and VRAM pressure outweighed saved reads. The option remains
  workload-dependent and disabled by default.
- Per-Expert accumulated timings on the 120B no-victim run recorded
  18.774 seconds waiting for cache leases and 10.791 seconds computing. These
  are sums across parallel tasks, not wall time; they demonstrate that storage
  residency dominates the 120B path.

## Asynchronous Vulkan boundary

- The pinned ncnn `VkCompute` API now exposes `submit_async()` and `wait()`;
  `submit_and_wait()` remains a compatibility wrapper. `reset()` and the
  destructor wait for an outstanding fence before releasing resources.
- Runtime Vulkan Linear and Attention keep the device command mutex through
  fence completion. Releasing it immediately after submission caused rare,
  deterministic-token divergence when two Sessions executed against shared
  ncnn layers and the shared blob allocator. Separate transfer slots may
  still prepare host staging concurrently, and one Session's CPU Expert work
  overlaps another Session's serialized GPU phase.
- Two reusable staging slots provide double-buffered host transfer storage and
  retain one `VkCompute` each. Linear and Attention reset/reuse those command
  objects after completed submissions and expose command-buffer reuse
  counters. Replaying static command recordings is still future work;
  asynchronous submission should not be described as single-Session
  autoregressive overlap because Attention, routing, and Experts have true
  data dependencies.

## 2026-07-25 byte-aware ARC and generic-policy acceptance

- The host Expert cache replaced its LFU/LRU combination with a byte-aware
  Adaptive Replacement Cache. Resident T1/T2 lists represent recent and
  repeated accesses; B1/B2 ghost lists retain only keys and conceptual bytes.
  B1 hits increase the byte target for T1 and B2 hits decrease it. Resident
  plus ghost bounds are enforced in bytes, so the policy supports model
  families whose Expert pairs differ in size.
- Speculative prefetch cannot evict an exact/demand-resident entry. Under
  pressure it may replace only an unpinned speculative entry; oversized or
  otherwise inadmissible predictions are dropped and counted. Superseded
  unstarted reads are cancelled. Exact requests remain high-priority and
  return a capacity error if active leases/reads pin the required space.
- ARC observability includes T1/T2 bytes, the adaptive T1 target, B1/B2 bytes
  and hits, cancelled speculative reads, and dropped speculative admissions.
  The native runner prints these fields and `benchmark_gpt_oss.py` records
  per-run values and medians.
- The final three-run GPT-OSS-120B acceptance used a 24 GiB host budget,
  16 GiB ARC, automatically selected four-worker I/O pool, and no GPU victim
  cache. Median generation was 13.1394 seconds for 32 tokens
  (**2.435 token/s**), median Expert wall time was 11.029 seconds, cache hit
  rate was 70.10%, and 20,661,609,600 Expert bytes were read. Median ARC state
  was 6,186,585,600 T1 bytes, 10,985,155,200 T2 bytes, and a 39,657,600-byte
  T1 target. Peak RSS was 19,574 MiB. The generated token sequence matched
  earlier baselines.
- Do not attribute the full improvement over the older 2.064 token/s baseline
  to ARC alone: grouped compute, persistent Vulkan command reuse, I/O changes,
  and Windows file-cache warmth all affect the result. A pre-ARC reuse-aware
  three-run sample and the ARC result were within run-to-run variance. ARC is
  retained for adaptive behavior and lower manual policy tuning.
- The final GPT-OSS-20B three-run acceptance generated 64 tokens in a
  4.4947-second median (**14.239 token/s**) with identical token IDs. Expert
  wall time was 2.567 seconds and peak RSS was 17,595 MiB. This exceeds the
  near-10 token/s goal on the validation host without a device-specific thread
  override.
- Generic default decisions now use runtime evidence: grouped MXFP4 thread
  count is bounded by matrix work, OpenMP capacity, and detected physical
  cores; Expert I/O concurrency is bounded by model Top-K and half the
  physical cores; Scaled-SiLU chooses between the exact libm and validated
  polynomial paths through a small alternating startup benchmark; and
  `HybridMode::Auto` avoids software CPU Vulkan devices. Every choice remains
  observable or explicitly overridable.

## 2026-07-25 single-Session 120B decode acceptance

- The Expert profiler now separates total Expert wall time, engine wall time,
  compute wall time, cache management/wait, regroup, combine, and residual
  orchestration. The earlier compute counter incorrectly counted the same
  parallel Expert wave once per Top-K assignment and was fixed before making
  optimization decisions.
- The compiled execution graph now lowers routed work to one dynamic
  `ExecutionNodeType::ExpertGroup` per layer. The previous graph emitted one
  conditional node per model Expert even though the executor computed the
  complete active Top-K wave at the first node.
- Decode-sized token regroup no longer starts an OpenMP region. Parallel
  regroup is selected only when the actual copy volume reaches 256 Ki
  elements. In the 120B decode profile, 16-token Expert orchestration fell
  from 179.249 ms to 2.362 ms, while token output stayed identical.
- MXFP4 startup dispatch benchmarks every supported ISA and can be overridden
  with `NCNN_MOE_MXFP4_KERNEL`. Contiguous row-pair grouping is benchmarked
  separately, requires a material 20% win, and can be overridden with
  `NCNN_MOE_MXFP4_DECODE_GROUP=1|2`. This avoids unstable, device-name-based
  policy.
- The default 120B acceptance used `D:\Models\gpt-oss-120b`, Vulkan
  Dense/Attention, CPU MXFP4 Experts, a 28 GiB host budget, a 20 GiB ARC, one
  same-model cache warm-up, 32 generated tokens, and three measured runs.
  Samples were 9.747, 10.238, and 10.407 token/s; the median was
  **10.238 token/s** (3.1256 seconds). Cache hit rate was 100%, measured
  Expert reads were zero, and peak RSS was 21,819 MiB.
- The selected default was AVX-512 with row-pair group size 1. Median stage
  times were 1,193.11 ms Attention, 38.083 ms Router, 1,707.96 ms Expert,
  1,691.06 ms Expert compute, 4.632 ms cache management, 4.887 ms residual
  Expert orchestration, and 169.11 ms LM Head.
- The machine-readable acceptance report is
  `build-ncnn/gpt-oss-120b-default-32x3.json`. The stronger forced-group-1
  diagnostic report reached an 11.291 token/s median, but the published
  acceptance uses the default autotuned configuration.

## 2026-07-25 Vulkan KV ring and staged cross-Session batching

- Vulkan Attention now appends K/V with one custom device dispatch into a
  geometrically grown double-written ring. A second small dispatch writes the
  synthetic sink outside the active logical interval. `VkMat` offset/cstep
  views feed SDPA directly across wraparound.
- A repeated sliding-window fixture advances beyond the initial 16 slots,
  observes wrapped views, and matches the CPU backend within `1e-4`.
- The public `BatchScheduler` can reserve compatible idle Sessions as one
  staged cohort. Dense embedding, Router, final norm, and LM Head are
  multi-row; Attention keeps independent KV state; routes with the same layer
  and Expert ID are gathered into one `CpuBatch`, executed once, and scattered
  back to the originating Session.
- Scheduler counters expose logical versus physical Expert batches,
  coalesced routes, maximum Expert batch rows, staged requests, and automatic
  bypasses. Two-request auto mode stages matching input tokens; three or more
  compatible requests stage by default, with explicit force/disable flags.
- AVX-512 MXFP4 row-pair bulk compute keeps 2–4 token accumulators in SIMD
  registers across all blocks before horizontal reduction; AVX2 specializes
  two tokens. This changed the 120B two-Session single sample from 10.594 to
  13.624 aggregate token/s while preserving identical output.
- The final default one-Session 120B acceptance samples were 10.574, 10.945,
  and 11.815 token/s, for a **10.945 token/s median** (2.9237 seconds).

## 2026-07-25 GPT-OSS-120B 20 token/s feasibility

- GPT-OSS-120B requests 1,903,564,800 packed Expert bytes per ordinary
  single-token Top-4 × 36-layer pass. Synchronously uploading every active
  Expert at 20 token/s would require 35.50 GiB/s, so a native Vulkan kernel
  must use an executable device cache and must not put upload-on-miss on the
  default critical path.
- The established route sequence produced 1,449 active layer/Expert keys
  across two identical 32-token passes. Greedy static hot-set coverage was
  38.0% at 1,903 MiB, 56.1% at 3,806 MiB, 64.8% at 5,075 MiB, and 76.7% at
  7,613 MiB. These are static upper estimates, not online ARC hit rates.
- The preferred heterogeneous policy is GPU execution only for resident
  Experts, CPU execution for misses, asynchronous post-use admission, and
  concurrent CPU/GPU route groups followed by Combine.
- Runtime capability reporting now exposes usable CPU ISA features and
  Vulkan INT8 storage/arithmetic, integer dot product, subgroup, and
  cooperative-matrix support. The measured Ryzen 7 9800X3D exposes
  AVX2/FMA, AVX-512, AVX/AVX-512 VNNI, and AVX-512 BF16 but no AMX. VNNI
  and AMX-INT8 require activation quantization and therefore cannot silently
  replace the FP32-activation MXFP4 path.
- The feasibility sample and parsed capability/traffic fields are in
  `build-ncnn/gpt-oss-120b-20tps-feasibility2-32x1.json`. The implementation
  plan is `gpt-oss-120b-20tps-roadmap.md` in this directory.
  Attention was 1,142.77 ms median and Expert compute 1,546.58 ms. The report
  is `build-ncnn/gpt-oss-120b-runtime-final2-32x3.json`.

## 2026-07-26 online Decode SDPA

- FP32 ncnn SDPA does not enter its BF16/FP16 flash-attention branch. For a
  single query it previously materialized QK, ran Softmax and PV, then
  Permute and Reshape.
- The native Decode shader assigns one 128-thread workgroup per query head,
  reduces the QK dot product, maintains an online maximum/sum, accumulates V,
  and writes the token-major projection input directly. It supports head
  dimensions through 128 and GQA, and falls back for Prefill or unsupported
  shapes.
- Learned sink logits are stored in an immutable per-model device buffer.
  The double-written ring already bounds the valid Decode history, so a fused
  block does not construct or upload an expanded mask.
- Selection is keyed by device, head shape, and power-of-two context bucket.
  It measures two ncnn and two fused blocks, requires a 2% advantage, holds
  the selected policy stable, and probes the alternative every 256 blocks.
  `NCNN_MOE_VULKAN_DECODE_SDPA=0|1` is a diagnostic override.
- A same-build forced three-run A/B on GPT-OSS-120B changed median Attention
  from 506.537 to 494.455 ms and throughput from 12.234 to 12.397 token/s.
  All generated token IDs were identical.
- The final automatic one-Session report is
  `build-ncnn/gpt-oss-120b-decode-sdpa-maskless-auto-final-16x3.json`:
  12.359 token/s median, 493.526 ms Attention, and 538/576 fused blocks.
- With no process warm-up, automatic selection reduced auxiliary transfers
  from 1,728 uploads / 1,769,472 bytes to 1,190 uploads / 337,920 bytes.
- The final two-Session report is
  `build-ncnn/gpt-oss-120b-decode-sdpa-maskless-auto-parallel2-threads8-16x3.json`:
  19.868, 19.847, and 19.597 token/s, or 19.847 median. This narrows but does
  not close the stable 20 token/s gap.

## 2026-07-26 direct QKV-to-ring fusion

- QKV+RoPE validation now checks logical matrix dimensions rather than
  `VkMat::total()`. The latter includes allocation-tail alignment for one-row
  ncnn Vulkan outputs and incorrectly rejected otherwise valid Decode tensors.
- For one token, the QKV shader writes rotated K and V to both copies of the
  ring destination. Intermediate K/V VkMats and the append dispatch disappear.
  This is valid with either native online SDPA or ncnn SDPA; learned-sink
  storage is synthesized only for the ncnn fallback.
- `NCNN_MOE_VULKAN_QKV_RING=0` is the isolated diagnostic override. Session
  statistics and JSON reports expose `vulkan_attention_qkv_ring_fusions`.
- With `NCNN_MOE_VULKAN_DECODE_SDPA=1` in both arms, ring off/on reports were
  `build-ncnn/gpt-oss-120b-qkv-ring-off-16x3.json` and
  `build-ncnn/gpt-oss-120b-qkv-ring-on-16x3.json`. Median Attention changed
  494.368 -> 487.553 ms and throughput 12.380 -> 12.555 token/s, with exact
  token parity and 576/576 direct writes.
- The best two-Session report reached a 20.039 token/s median from
  20.039/21.101/19.863 samples. A later independent-ring rerun reduced median
  Attention to 580.588 ms but returned 19.481 token/s as Expert time increased
  to 760.813 ms. The implementation is accepted on phase evidence; stable
  20 token/s still requires reducing CPU Expert time and variance.

## 2026-07-26 Expert scheduling follow-up

- A two-Session thread sweep on the 8-core/16-thread host measured 18.069,
  18.737, and 19.494 token/s with 4, 6, and 8 Expert threads per worker.
  Eight remains the shape/topology-derived choice for this host.
- Extending the single-token AVX-512 row group to eight output rows was
  rejected. The four-pair form caused register spilling and increased Expert
  time from 650.478 to 1,385.850 ms in the one-run screen. The code was
  reverted.
- `OMP_PROC_BIND=spread`/`OMP_PLACES=cores` showed a small isolated benefit,
  but passive wait/block-time combinations were unstable. No process-wide
  OpenMP environment default was added.
- Forced staged two-Session execution measured 15.747 token/s versus 18.716
  with staging disabled in the initial isolation. The staged path therefore
  remains adaptive rather than forced. The public runner and benchmark now
  accept `--scheduler-staging auto|force|off` for reproducible diagnostics.
- Adaptive buckets no longer probe the alternative immediately after their
  first preferred-path observation. They wait for the configured interval
  (32 decisions by default), preserving eventual adaptation without charging a
  short generation for a cold, predictably expensive sample.
- The MSVC AVX-512 batch-two row-pair kernel now shares input loads across four
  adjacent output rows. Alternating 101-repeat microbenchmarks improved median
  complete-Expert time from 0.4164 to 0.4098 ms at 8 threads and from 0.3759
  to 0.3624 ms at 16 threads. Other ISAs retain their normal dispatch.
- Single-token grouped execution previously linearly rescanned the active
  Expert list for every flattened row group. Uniform Expert shapes now use
  constant-time division/modulo mapping, while heterogeneous shapes keep the
  general scan. The final 120B report
  `build-ncnn/gpt-oss-120b-uniform-expert-index-auto-16x3.json` produced
  20.315, 19.953, and 19.982 token/s. Median generation time was 1.6014
  seconds for 32 aggregate tokens, or **19.982 token/s**; Expert time ranged
  from 716.1 to 740.5 ms and all output tokens matched.
- Parallel per-Session Attention inside staged execution and opt-in concurrent
  Vulkan Attention submissions were both rejected. They increased lock/device
  contention; the latter fell to 17.229 token/s and increased GPU memory.

## 2026-07-26 long-window scheduler confidence

- A 96-token x two-Session GPT-OSS-120B run proved that one alternative-path
  timing is insufficient for a scheduler switch under ARC and device
  contention. The former fixed-interval policy used staged execution for
  45/95 decisions, switched three times, and measured 7.366 token/s.
- The final policy confirms a candidate on the next batch, requires 10% gain
  below four candidate samples, 7.5% below eight samples, and 5% afterward.
  It also expands the non-preferred probe interval by 2x/4x/8x when the
  measured path gap exceeds 10%/25%/50%. Buckets remain keyed by request count
  and context length, so a new workload range recalibrates independently.
- The final one-run screen
  `build-ncnn/gpt-oss-120b-confidence-probe-96x1.json` used staged execution
  only for two probes, made no switch, and measured 8.026 token/s. Expert time
  was 18.276 seconds versus 19.719 seconds in the former screen; generated
  tokens were identical.
- The final run had a 90.88% ARC hit rate and read 30.56 GiB of Expert data.
  It is a long-window diagnostic, not a three-run acceptance result. The
  short resident-window 19.982 token/s result cannot be extrapolated through
  this storage-pressure regime.
- Exact CPU E8M0 bit construction was rejected for MSVC AVX-512. Seven
  alternating 2880 -> 2880 Expert microbench rounds regressed median latency
  from 0.5256 to 0.5352 ms and mean latency by about 3.1%. The 1 KiB lookup
  table stays hot in L1 on this CPU; the experimental code was reverted.

## 2026-07-26 compressed Vulkan victim L2 and warm-up quiescence

- A distinct `expert_gpu_victim_cache_bytes` option now places evicted
  compressed MXFP4 pairs in VRAM while CPU arithmetic remains selected.
- Multi-device capacity is assigned by runtime capability score, keys are
  deterministically sharded, every active device must hold one Expert pair,
  and each shard is capped at one quarter of the reported Vulkan heap budget.
- `Runtime::synchronize_model_caches()` waits for host Expert reads, sharded
  victim admissions, and executable Expert admissions at explicit warm-up or
  service-transition boundaries. The GPT-OSS runner calls it after internal
  cache warm-up. Normal inference does not call the barrier.
- Three synchronized 96-token x two-Session runs with a 20 GiB host ARC and
  3 GiB victim L2 measured 8.035, 8.193, and 8.798 token/s (8.193 median).
  Median Expert reads were 28,196,553,600 bytes, median cache wait was
  11.160 seconds, and median GPU restore traffic was 2,419,113,600 bytes in
  0.772 seconds. The earlier matching no-L2 diagnostic measured 8.026
  token/s, read 32,810,054,400 bytes, and waited 12.665 seconds on the host
  cache. This is a modest storage-pressure improvement, not a sustained
  10 token/s acceptance.
- A standard ARC implementation was tested and reverted for the victim tier.
  It measured 7.886 token/s and 197 hits, while the matched initial LRU screen
  measured 8.729 token/s and 340 hits. Since this tier sees the host ARC's
  eviction stream rather than original accesses, frequency promotion
  overprotected stale entries. Host residency and executable Expert caching
  retain ARC; the compressed victim tier uses LRU.
- A 3.5 GiB screen measured 8.493 token/s with 15,216 MiB peak NVIDIA memory
  and no clear Expert-time benefit over 3 GiB. A 4 GiB request was rejected
  by the per-device one-quarter heap safety limit. Keep 3 GiB as an explicit
  local test point rather than a cross-device default.
- Requiring four consecutive scheduler candidate confirmations was also
  tested and reverted. Its three-run median fell from 8.193 to 8.060 token/s;
  one run genuinely favored staged execution, so delaying the switch was not
  a general improvement.

## 2026-07-26 route prediction and packed ExpertStore

- Same-layer previous-token route prediction was instrumented before adding
  speculative I/O. In the 96-token/two-Session run it made 13,680 predictions
  and matched 3,619 selected Experts (26.45%). All 13,680 predicted pairs were
  already ready in the host ARC. Consequently, issuing those prefetches would
  only repeat cache locking and would not hide a miss. The executor now skips
  the redundant prefetch call when readiness is already established.
- The original GPT-OSS Safetensors layout stores all Experts' blocks followed
  by all Experts' scales. One pair therefore requires four random ranges and
  its blocks/scales can be hundreds of MiB apart. Expanding a range would
  create unacceptable read amplification.
- The accepted alternative is an optional standard-Safetensors sidecar with
  each Expert's Gate/Up blocks/scales and Down blocks/scales adjacent. The
  runtime validates contiguity, performs one read, and returns four aliasing
  views. The original checkpoint is unchanged and remains the transparent
  fallback. `tools/pack_mxfp4_experts.py` produced the official 120B 56.73 GiB
  sidecar in 62.4 seconds.
- With a 22 GiB host ARC, 3 GiB compressed Vulkan L2, synchronized warm-up,
  96 tokens, and two Sessions, packed samples were 9.833, 9.125, and
  9.371 token/s (**9.371 median**). Median generation was 20.490 seconds,
  Expert time 13.913 seconds, cache wait 8.127 seconds, compute 5.494 seconds,
  and Attention 3.995 seconds. Median direct traffic used 986 reads whose
  average logical size was 12.61 MiB. Peak RSS was 23,741 MiB and peak NVIDIA
  memory was 14,593 MiB. Tokens matched across all runs. Evidence:
  `build-ncnn/gpt-oss-120b-packed-arc22g-victim3g-96x3.json`.
- This improves the earlier 20 GiB ARC + 3 GiB L2 storage-pressure median from
  8.193 to 9.371 token/s (+14.4%), but remains 6.7% below 10 token/s. A 23 GiB
  ARC was rejected: its three-run median fell to 6.873 token/s and CPU Expert
  compute roughly doubled, despite reduced disk traffic. The 31.14 GiB host
  crossed its memory-pressure knee; larger ARC capacity is not a portable
  default.

## 2026-07-26 coexisting executable and victim GPU caches

- The executable Vulkan Expert ARC and compressed Vulkan victim L2 may now
  coexist. The Runtime distributes the two capacities independently by device
  capability and validates their sum against the existing per-device
  one-quarter heap limit. Each enabled tier must hold at least one Expert pair
  on every active device. Empty-device and integer-overflow paths fail
  explicitly.
- Session statistics, the native runner, and `benchmark_gpt_oss.py` now expose
  executable-cache and victim-cache traffic separately. The staged
  multi-Session path records victim deltas as well; earlier reports combined
  the two tiers under `expert_gpu_cache_*`.
- Pure 3 GiB executable caching measured 8.790 token/s because disk traffic
  rose to 23.33 GB and CPU/GPU contention outweighed 3,447 native executions.
  A 1 GiB executable + 2 GiB victim split measured 9.144 token/s once.
- A 512 MiB executable + 2.5 GiB victim split initially measured
  9.819 token/s median (9.819/8.768/9.956). The independent split-statistics
  validation measured 9.402/9.088/8.622 token/s, or 9.088 median. Its median
  generation was 21.126 seconds, Expert time 15.019 seconds, host-cache wait
  8.687 seconds, Expert compute 6.999 seconds, and Attention 4.690 seconds.
  The executable tier uploaded 8.62 GB, ran 311 native Vulkan batches in
  241.6 ms, and made 1,353 CPU-preferred decisions. The victim tier restored
  4.19 GB in 1.419 seconds. Evidence:
  `build-ncnn/gpt-oss-120b-packed-arc22g-exec512m-victim2560m-split-metrics-96x3.json`.
- A 256 MiB executable + 2.75 GiB victim split reached 9.994 token/s in one
  run but fell to an 8.494 token/s three-run median. The 512 MiB split is the
  better local exploration point, but neither split is a portable default or
  a sustained 10 token/s acceptance. The remaining long-window bottleneck is
  Expert cache wait and storage traffic, not native Vulkan kernel time.
- Explicit `--expert-io-workers 8` measured 9.441 token/s. This did not change
  the automatic configuration: Top-4 gives an exact-plus-speculative budget of
  eight workers on the test host. An adaptive read-policy run measured 9.787
  token/s once and had already selected direct I/O before the measured window;
  it recorded zero buffered ranges. The difference is ordinary run variance,
  not evidence for a worker-count or I/O-policy default change.

## 2026-07-26 adaptive cross-call micro-batching

- `BatchScheduler` now collects independent singleton `submit_decode()` calls
  before its existing staged/independent decision. It selects at most one
  request per Session, preserving FIFO and Session serialization. The default
  wait is 200 microseconds and maximum batch size defaults to `worker_count`.
- Four consecutive empty probes trigger bypass mode. Collection resumes on
  the normal adaptive probe interval, so a singleton workload pays four
  bounded probes rather than a permanent per-token delay. Deterministic tests
  cover successful two-Session collection and the four-probe/two-bypass
  sequence.
- The public statistics distinguish collected batches/requests, probes,
  timeouts, bypasses, cumulative wait, maximum batch size, and maximum pending
  depth. The native runner and JSON benchmark expose the same counters.
- Packed 120B, 16 tokens x two Sessions:
  `gpt-oss-120b-packed-cross-call-200us-16x3.json` measured
  20.230/20.369/13.674 token/s (20.230 median). All 15 measured rounds were
  collected into two-request batches; median cumulative collection wait was
  2 microseconds. This is API-path evidence, not a stable 20 token/s floor.
- Packed 120B, 96 tokens x two Sessions: cross-call plus staging disabled
  measured 9.751 token/s; the explicit-vector control measured 9.587 token/s.
  Cross-call collection accumulated only 22 microseconds of wait across
  95 batches. Automatic staging measured 8.014 token/s because timing noise
  led to 17 staged decisions and Expert compute rose from 5.881 to
  9.097 seconds. The accepted conclusion is that collection overhead is
  negligible; cache/phase-aware adaptive-policy stability remains open.
- A matched staging-off run with both GPU Expert tiers disabled measured
  8.871 token/s. Compared with the 512 MiB executable + 2.5 GiB victim run,
  Expert wait rose from 8.036 to 9.251 seconds and disk reads from 22.79 to
  26.68 GiB. The GPU tiers are net positive on this host, but the enabled run
  uploaded 26.65 GiB to obtain 314 victim restores / 3.87 GiB of downloads.
  Future work should add benefit- and Vulkan-queue-aware admission rather than
  disabling the cache or hard-coding this device.
- VNNI and AMX capability detection remains intentionally separate from exact
  MXFP4 dispatch. Integer dot-product paths require activation quantization,
  while BF16 paths change FP32 activation arithmetic. They need an explicit
  `QuantConfig` mode and numerical acceptance rather than a silent
  hardware-specific replacement.

## 2026-07-26 phase- and noise-aware staging policy

- Each context/request-count bucket is now split by the preceding Expert
  phase: resident at no more than 5% cache-wait share, mixed below 50%, and
  storage at 50% or higher. The signal uses two lightweight counters captured
  immediately around Decode and does not copy the full Session statistics or
  its per-Expert vector.
- Staged and independent paths maintain exponentially weighted mean and
  absolute deviation. A candidate must beat the existing sample-aware
  relative margin plus half the combined deviation. Phase decisions,
  observations, changes, and noisy switch rejections are public statistics
  and are included in benchmark JSON.
- `gpt-oss-120b-packed-phase-aware-cross-call-auto-96x3.json` measured
  9.565/9.003/9.453 token/s (9.453 median), with zero policy switches and only
  one or two staged probes. The matched staging-off cohort measured
  9.546/8.483/10.137 token/s (9.546 median). Auto was 0.9% slower by median
  but substantially less variable and removed the earlier 17-staged-decision
  failure.
- The 16-token regression measured 19.640/20.093/20.424 token/s
  (20.093 median), with zero staged probes/switches and identical output.
  This is the latest short-resident-window 20 token/s service point. The
  sustained 96-token 10 token/s gate remains open.
- Attention still ends with the ncnn output projection followed by a separate
  residual BinaryOp and host download. Fusing projection epilogue/residual
  without replacing ncnn's tuned matrix kernel is the next safe Attention
  boundary to investigate. The victim tier's 26.65 GiB upload for 3.87 GiB
  restore remains the other high-value general target.

## 2026-07-26 victim reuse-probe and Attention epilogue results

- Upper-cache ARC `Frequent` state was too broad as a lower-tier admission
  signal. Sampling only entries without that flag filtered 70/2,174
  evictions; additionally requiring more than one exact access filtered
  190/2,168. The latter measured 8.054 token/s and was rejected.
- The retained optional policy instead uses a payload-byte-bounded ghost
  doorkeeper over the actual host-ARC eviction stream. First observations are
  sampled at `1/N`; a repeated cross-residency key passes. `N=1` bypasses the
  ghost and preserves the old admit-all behavior.
- `gpt-oss-120b-packed-victim-ghost-probe2-staging-off-96x1.json` measured
  8.633 token/s. It halved victim uploads to 13.35 GiB while retaining the
  baseline-like 317 hits and 3.90 GiB of restores, but cache wait was 9.211
  seconds and it missed the 9.5 token/s screen.
- `gpt-oss-120b-packed-victim-ghost-probe4-staging-off-96x1.json` measured
  8.318 token/s, 6.62 GiB uploaded, 239 hits, and 2.94 GiB restored. Its ghost
  reuse count was zero because reuse distance exceeded the 2.5 GiB metadata
  window. Both probe settings remain diagnostic only.
- A projection-download/CPU-residual Attention path was numerically correct
  and hit all 3,456 measured blocks, but was reverted. Its three samples were
  9.151/9.510/8.640 token/s (9.151 median); median Attention was 4.760 seconds
  versus 4.731 seconds in the existing staging-off cohort. Shifting the add to
  CPU interfered with the other Session's Expert phase.
- The next general cache optimization should execute a victim-resident MXFP4
  pair directly on Vulkan, or promote it device-to-device into the executable
  ARC under the existing phase-benefit controller. Fixed admission thresholds
  cannot remove the victim upload/download round trip.

## 2026-07-26 code normalization and benchmark control

- The production adapter registry is GPT-OSS-only. Unsupported model parsers,
  mappings, tests, and capability claims were removed while retaining the
  model-neutral `MoeIR`/compiler boundary. The private `test_moe` fixture is
  linked only into the unit-test target.
- Project C++ was normalized with the repository's ncnn-style clang-format.
  The configuration uses Allman control braces and upstream ncnn's
  `ColumnLimit: 0`, with a C++11 parser to preserve raw strings. The production
  tree has 43 remaining comment locations, limited to contracts, invariants,
  platform constraints, and non-obvious performance decisions.
- `ExpertStore` no longer maintains a duplicate hash index, snapshot copy, or
  mutex after model publication. `Session::generate` and
  `MemoryManager::record_execution` each use one outer critical section.
  Shared file/mapping caches use reader locks for hits, perform open/mapping
  syscalls outside the writer lock, and double-check insertion. Windows
  direct-I/O events are reused per worker.
- Final validation passed MSVC Vulkan 3/3, MinGW Vulkan 3/3, and portable CPU
  2/2 tests. The benchmark Python modules byte-compiled successfully.
- `gpt-oss-120b-code-normalization-staging-off-96x3.json` used the existing
  96-token x two-Session protocol with a 22 GiB host ARC, 512 MiB executable
  GPU cache, 2.5 GiB victim tier, direct I/O, cross-call collection, and
  staging disabled. It measured 5.712/5.485/8.580 token/s (5.712 median).
  Median cache hit rate was 92.03%, reads were 22.95 GiB, cache wait was
  7.815 seconds, and Expert compute was 18.773 seconds.
- An immediately adjacent control using the pre-normalization binary and the
  same checkpoint/options measured 5.090 token/s. It retained a 92.00% hit
  rate and 23.02 GiB of reads but spent 21.467 seconds in Expert compute.
  Because the control regressed more than the new binary under the same system
  state, this pair is classified as background-load/frequency/memory-bandwidth
  contamination, not a normalization regression or throughput acceptance.
- At that point, the latest comparable idle cohort was
  9.546/8.483/10.137 token/s (9.546 median). It is superseded by the
  post-reboot clean baseline below.
- An adjacent resident-window control reduces the ambiguity.
  `gpt-oss-120b-code-normalization-auto-16x3.json` measured
  15.711/17.090/18.492 token/s (17.090 median), while
  `gpt-oss-120b-diagnostic-old-binary-auto-16x3.json` measured
  15.968/15.265/16.856 token/s (15.968 median). The normalized binary was 7.0%
  faster in this system state. Both had zero Expert-cache wait and identical
  output, but both remained below the historical idle 20.093 median. Treat the
  pair as non-regression evidence, not a new peak-throughput claim.

## 2026-07-26 post-reboot clean benchmark baseline

- The current working tree was rebuilt with
  `cmake --build build-gptoss-vulkan-msvc --config Release --clean-first
  --parallel`; MSVC Vulkan tests passed 3/3. The runner timestamp was
  2026-07-26 16:42:32 local time and its SHA-256 was
  `47AA3AC914D29858F090166F66C9FF58BF0AB65B2193B014A323C73B78FCF7C3`.
- The host was rebooted before the run. Hardware was Ryzen 7 9800X3D
  (8 cores/16 threads), 31.14 GiB RAM, RTX 5070 Ti 16 GiB, NVIDIA driver
  610.62, and the Windows balanced power plan. No model, Python, compiler, or
  MSBuild process remained. Ten pre-run CPU samples averaged 5.93% utilization.
- The local 120B package has no verifiable upstream revision and must remain
  recorded as unpinned. It contains the official checkpoint plus the
  60,916,773,944-byte packed Expert sidecar. Protocol inspection read the
  approximately 2.7 MiB sidecar header but not the 56.73 GiB payload, so the
  first result is a cold Runtime/Expert working set rather than an absolute
  zero-page-cache claim.
- `gpt-oss-120b-clean-cold-32x1.json` ran first with no outer or in-process
  warm-up, a 24 GiB host budget, 16 GiB ARC, Hybrid execution, and on-demand
  Experts. It measured **3.294 token/s** (32 tokens in 9.715 seconds), loaded
  in 8.247 seconds, read 17.86 GiB, hit 68.51% of Expert lookups, spent
  5.617 seconds waiting for the cache and 2.627 seconds in Expert compute,
  peaked at 19,650 MiB RSS and 8,441 MiB GPU memory, and produced the accepted
  token sequence.
- `gpt-oss-120b-clean-single-32x3.json` used one outer warm-up, one
  same-process ARC warm-up per run, a 28 GiB host budget, and 20 GiB ARC.
  Samples were 11.045/10.050/11.764 token/s, for an **11.045 median**.
  Median generation was 2.897 seconds, Expert time was 1.611 seconds, cache
  wait and measured Expert reads were zero, and peak RSS/GPU memory were
  21,554/8,748 MiB.
- `gpt-oss-120b-clean-service-2x16x3.json` used two Sessions, eight Expert
  threads per scheduler worker, cross-call collection, a 22 GiB host ARC,
  512 MiB executable Vulkan Expert cache, 2.5 GiB compressed victim tier, and
  direct I/O. Samples were 20.510/20.600/20.445 token/s, for a **20.510
  aggregate median** and 1.560-second median generation. The resident working
  set had 100% host-ARC hits and no reads. All 15 decisions were independent;
  there were no probes or switches. Peak RSS/GPU memory were
  15,263/9,096 MiB.
- `gpt-oss-120b-clean-service-2x96x3.json` used the same service settings.
  Samples were 10.142/9.010/9.778 token/s, for a **9.778 aggregate median** and
  19.637-second median generation. The host ARC hit rate was 91.96%, direct
  reads were 22.76 GiB, cache wait was 7.950 seconds, Expert compute was
  5.301 seconds, Attention was 5.091 seconds, and peak RSS/GPU memory were
  25,782/12,525 MiB.
- The long-window median made 94 independent decisions, one staged probe, no
  policy switch, and collected 190 requests into 95 cross-call batches. The
  executable GPU cache completed 323 Expert executions and marked 936
  observations CPU-preferred. The victim tier restored 317 entries in
  1.278 seconds; direct victim-source execution succeeded 8 times with zero
  failures.
- `gpt-oss-20b-clean-eager-64x3.json` was the final sanity check. Samples were
  16.898/16.918/16.809 token/s, for a **16.898 median**, 3.788-second median
  generation, zero cache wait, and 17,598/6,766 MiB peak RSS/GPU memory.
- All multi-run reports enforced exact output-token parity. The outer
  `--warmup` is a separate process and only warms OS/driver/frequency state;
  `--cache-warmup-runs 1` is the same-process Runtime ARC warm-up followed by
  `synchronize_model_caches()`.
- The sustained target remains workload-specific. Resident one-Session 120B
  clears 10 token/s, and the short two-Session service window clears 20
  aggregate token/s. The storage-pressure 2x96 median is 9.778 token/s with
  one sample above 10, so stable long-window 10 token/s is not yet accepted.
  Expert storage wait is the largest remaining general bottleneck.

Verified commands:

```powershell
python tools\benchmark_gpt_oss.py build-gptoss-vulkan-msvc\Release\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-120b --max-new-tokens 32 --warmup 0 --cache-warmup-runs 0 --repeats 1 --backend hybrid --expert-memory on-demand --host-memory-mb 24576 --expert-cache-mb 16384 --vulkan-device-index 0 --json-output build-ncnn\gpt-oss-120b-clean-cold-32x1.json
python tools\benchmark_gpt_oss.py build-gptoss-vulkan-msvc\Release\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-120b --max-new-tokens 32 --warmup 1 --cache-warmup-runs 1 --repeats 3 --backend hybrid --expert-memory on-demand --host-memory-mb 28672 --expert-cache-mb 20480 --vulkan-device-index 0 --json-output build-ncnn\gpt-oss-120b-clean-single-32x3.json
python tools\benchmark_gpt_oss.py build-gptoss-vulkan-msvc\Release\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-120b --max-new-tokens 16 --warmup 1 --cache-warmup-runs 1 --parallel-sessions 2 --scheduler-expert-threads 8 --scheduler-cross-call --repeats 3 --backend hybrid --expert-memory on-demand --host-memory-mb 28672 --expert-cache-mb 22528 --expert-gpu-cache-mb 512 --expert-gpu-victim-cache-mb 2560 --vulkan-device-index 0 --direct-expert-io --json-output build-ncnn\gpt-oss-120b-clean-service-2x16x3.json
python tools\benchmark_gpt_oss.py build-gptoss-vulkan-msvc\Release\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-120b --max-new-tokens 96 --warmup 1 --cache-warmup-runs 1 --parallel-sessions 2 --scheduler-expert-threads 8 --scheduler-cross-call --repeats 3 --backend hybrid --expert-memory on-demand --host-memory-mb 28672 --expert-cache-mb 22528 --expert-gpu-cache-mb 512 --expert-gpu-victim-cache-mb 2560 --vulkan-device-index 0 --direct-expert-io --json-output build-ncnn\gpt-oss-120b-clean-service-2x96x3.json
python tools\benchmark_gpt_oss.py build-gptoss-vulkan-msvc\Release\ncnn_moe_gpt_oss.exe D:\Models\gpt-oss-20b --max-new-tokens 64 --warmup 1 --cache-warmup-runs 0 --repeats 3 --backend hybrid --expert-memory eager --vulkan-device-index 0 --json-output build-ncnn\gpt-oss-20b-clean-eager-64x3.json
```

## 2026-07-26 unified warm/cold short/long matrix

- A clean MSVC Vulkan Release rebuild completed with
  `cmake --build build-gptoss-vulkan-msvc --config Release --clean-first
  --parallel`. The runner SHA-256 is
  `D4E00440CE3F31F9A584B9143A2CF6D7F05AF02CD001BB0232507DEE249B5AA4`.
  CTest passed 3/3; the benchmark modules passed Python byte compilation.
- `tools/benchmark_performance_matrix.py` fixes one 16-token prompt, greedy
  decoding, short output at 32 tokens, long output at 256 tokens, and three
  measured samples per case. Cold is `warmup=0` plus
  `cache-warmup-runs=0`. Warm adds one unreported outer warm-up run and one
  same-process ARC cache warm-up before the measured samples. Each case starts
  a fresh benchmark process and validates exact generated-token parity.
- The run covered 16 cases across GPT-OSS-20B and 120B, one and two Sessions,
  hybrid Vulkan Dense/Attention with CPU Experts, and CPU Expert storage
  controls at 1/10/16 GiB ARC. It completed in about 7,464 seconds with no
  report errors.

Hybrid medians on the reference host:

| Workload | Short cold | Short warm | Long cold | Long warm |
| --- | ---: | ---: | ---: | ---: |
| GPT-OSS-20B, one Session | 14.103 | 14.855 | 17.069 | 16.597 |
| GPT-OSS-120B, one Session | 2.885 | 4.718 | 4.971 | 5.171 |
| GPT-OSS-120B, two Sessions | 5.284 | 10.916 | 9.870 | 10.295 |

Values are token/s; the two-Session row is aggregate. The corresponding
120B hybrid ARC hit rates are 66.7/84.6/82.9/84.3% for single-Session
short-cold/short-warm/long-cold/long-warm and 81.6/96.2/91.9/93.1% for the
two-Session rows. Runtime logical reads are, respectively, 24.6/11.4/85.5/
78.7 GB and 24.4/2.6/70.0/55.3 GB.

CPU Expert storage-control medians and Runtime logical reads:

| ARC cache | Short cold | Short warm | Long cold | Long warm |
| --- | ---: | ---: | ---: | ---: |
| 1 GiB | 1.042 (73.9 GB) | 1.050 (73.8 GB) | 1.143 (500.0 GB) | 1.133 (500.0 GB) |
| 10 GiB | 1.812 (34.7 GB) | 2.006 (29.5 GB) | 2.145 (183.7 GB) | 2.160 (182.1 GB) |
| 16 GiB | 2.119 (26.6 GB) | 2.444 (20.8 GB) | 2.637 (115.0 GB) | 2.636 (115.4 GB) |

Sampled system physical reads for 1/10/16 GiB were 57.8/26.0/19.1 GB
(short cold), 114.4/50.6/37.6 GB (short warm), 392.1/140.8/88.4 GB
(long cold), and 783.5/285.5/174.8 GB (long warm). These are total-disk
samples, not process-attributed SSD traffic.

The parenthesized values are Runtime Expert-cache logical bytes. Peak RSS
was approximately 4.1/13.2/19.2 GiB for the 1/10/16 GiB rows. System
physical reads were sampled Windows CIM total-disk counters, include
unrelated traffic, and must not be interpreted as process-attributed SSD
bytes. The storage control is not directly comparable to the hybrid service
rows because it intentionally keeps Expert execution on the CPU.

The aggregate report is
`build-gptoss-vulkan-msvc/performance-matrix-unified/performance-matrix.json`.
The verified run used:

```powershell
python tools\benchmark_performance_matrix.py --runner build-gptoss-vulkan-msvc\Release\ncnn_moe_gpt_oss.exe --model-20b D:\Models\gpt-oss-20b --model-120b D:\Models\gpt-oss-120b --output-dir build-gptoss-vulkan-msvc\performance-matrix-unified --repeats 3 --short-tokens 32 --long-tokens 256 --vulkan-device-index 0
```
