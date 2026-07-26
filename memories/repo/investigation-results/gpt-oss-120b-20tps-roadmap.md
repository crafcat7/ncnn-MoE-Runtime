# GPT-OSS-120B 20 token/s feasibility and implementation roadmap

This document defines the evidence, performance budget, and implementation
order for pursuing 20 token/s on GPT-OSS-120B without introducing
model-name or device-name branches. Every choice must be driven by tensor
shape, runtime capabilities, measured cost, queue pressure, and residency.

## Implementation status (2026-07-26)

| Workstream | Current status |
| --- | --- |
| Native Vulkan MXFP4 Expert | Implemented with fused Gate/Up + activation + Down, asynchronous ARC admission, CPU overlap, and end-to-end self-disabling calibration |
| CPU ISA | Scalar/NEON/SVE2/AVX2/AVX-512 are executable and runtime-selected; VNNI/AMX are detected but intentionally not used by exact MXFP4×FP32 mode |
| Attention fusion | Native QKV+RoPE, direct double-written KV-ring output, and single-token online Decode SDPA shaders implemented with measured selection and transparent fallback |
| Multi-GPU | Candidate devices, per-layer device metadata, weighted/concurrency-aware placement, and multi-device Expert backends implemented |
| Adaptive multi-Session | Context/batch buckets learn staged versus independent ms/request with hysteresis, delayed cold exploration, and sparse steady-state probes |

The best accepted short-window service result is 20.039 token/s aggregate
for two Sessions generating 16 tokens each. The three samples were 20.039,
21.101, and 19.863 token/s. This is not a one-Session 20 token/s claim:
the latest one-Session 16-token median is 12.359 token/s, and the established
one-Session 32-token median remains 10.945 token/s.

The accepted report is
`build-ncnn/gpt-oss-120b-qkv-ring-auto-parallel2-threads8-16x3.json`. A later
identical-policy run with direct ring writes selected independently from SDPA
reduced median Attention from 597.442 to 580.588 ms but reached only
19.481 token/s because median Expert time increased to 760.813 ms. This
demonstrates that the remaining short-window variance is dominated by the CPU
Expert phase. A 32-token/two-Session run with a 20 GiB ARC was
terminated after the working set reached about 20.5 GiB RSS and 31.5 GiB
private commit; no long-context 20 token/s claim is made.

The latest default acceptance removes per-row-group linear Expert lookup and
does not spend the second Decode on an immediate cold staging probe. Its
two-Session samples are 20.315, 19.953, and 19.982 token/s; median generation
time is 1.6014 seconds for 32 aggregate tokens, or **19.982 token/s**. This
narrows the local short-window band but still does not establish a long-context
or cross-device 20 token/s floor.

## Measured starting point

The accepted resident one-Session result is 10.945 token/s median for 32
generated tokens. Its median critical-path components are:

| Stage | Total | Per generated token |
| --- | ---: | ---: |
| Vulkan Attention | 1,142.77 ms | 35.71 ms |
| CPU Expert compute | 1,546.58 ms | 48.33 ms |
| Router | 40.36 ms | 1.26 ms |
| Vulkan LM Head | 168.82 ms | 5.28 ms |
| End-to-end | 2,923.67 ms | 91.36 ms |

20 token/s requires at most 50 ms/token. Expert compute alone is close to
that limit, so optimizing only Attention or only Expert cannot reach the
goal. A working target budget is:

| Stage | Target |
| --- | ---: |
| Attention and dense projections | at most 22 ms/token |
| Heterogeneous Expert phase | at most 20 ms/token |
| Router, LM Head, transfers, scheduler | at most 8 ms/token |

The budget is a planning constraint, not a performance claim.

## Expert traffic lower bound

GPT-OSS-120B has hidden size 2,880, intermediate size 2,880, 36 layers,
128 Experts per layer, and Top-4 routing. One packed Gate/Up + Down MXFP4
Expert pair occupies about 13,219,200 bytes. A route with no device-resident
reuse therefore requests:

`13,219,200 × 4 × 36 = 1,903,564,800 bytes/token`

At 20 token/s, synchronously uploading every active Expert would require
35.50 GiB/s. The current test system reported PCIe Gen4 x8, so this design
is impossible even before compute and protocol overhead.

The new route profiler measured the following static greedy hot-set upper
estimates over the established deterministic 32-token sequence. These
values include two identical passes because the benchmark uses one cache
warm-up Session; coverage is unchanged by the duplicated frequency.

| Device Expert budget | Experts retained | Static batch-byte coverage | Miss traffic at 20 token/s |
| --- | ---: | ---: | ---: |
| 1,903 MiB | 150 / 1,449 active | 38.0% | 22.0 GiB/s |
| 3,806 MiB | 301 / 1,449 active | 56.1% | 15.6 GiB/s |
| 5,075 MiB | 402 / 1,449 active | 64.8% | 12.5 GiB/s |
| 7,613 MiB | 603 / 1,449 active | 76.7% | 8.3 GiB/s |

This is a static placement estimate, not an online ARC hit-rate result.
It nevertheless proves that a 4 GiB cache cannot make synchronous
upload-on-miss viable on the measured link.

## 1. Native Vulkan MXFP4 Expert engine

### Kernel shape

The first correctness-preserving kernel uses packed `uint32` storage and
FP32 accumulation, so it does not require 8-bit storage or activation
quantization:

1. Fused MXFP4 Gate/Up decode, dot product, bias, clamped SiLU, and multiply.
2. MXFP4 Down decode and dot product.
3. Weighted Combine, preferably on device when every route stays on Vulkan.

Packed bytes and exponent bytes are extracted from 32-bit words in the
shader. Later variants may use 8-bit storage, integer dot product,
cooperative matrices, or BF16, but only behind numerical validation and
runtime autotuning.

### Executable Expert cache

The Vulkan cache is now executable, byte-aware, and layer-fair:

- key: model, layer, Expert, quant layout, Vulkan device;
- value: packed Gate/Up, Down, scales, biases, compiled shape metadata;
- two-touch asynchronous admission after CPU execution;
- pin while a command buffer references the entry;
- no eviction of pinned or in-flight entries;
- per-device capacity is checked against a fraction of the reported heap.

Pipelines and the reclaimable weight arena are shared per device. The exact
E8M0 scale is constructed from IEEE exponent bits in shader code. On the
measured RTX 5070 Ti, full-Expert token-1 latency improved from 0.1603 to
0.1261 ms and token-4 from 0.3972 to 0.3128 ms.

### Dynamic CPU/GPU selection

Never place a cache miss on the synchronous GPU critical path by default.
For every active route group:

- GPU-resident Experts are candidates for Vulkan;
- non-resident Experts remain CPU candidates while an asynchronous
  admission is queued;
- CPU and GPU groups execute concurrently, then Combine waits for both;
- an online cost model tracks CPU time, resident-GPU time, upload time,
  batch rows, weight bytes, queue delay, and device index;
- switch only when the predicted gain exceeds a hysteresis margin;
- periodically explore a resident alternative to avoid a permanently stale
  estimate;
- explicit diagnostic overrides remain available, but `Auto` contains no
  model or device names.

This implemented policy turns cache misses into useful CPU work instead of PCIe stalls.
With roughly two or three Top-4 routes resident, it can also split one
layer across the CPU and GPU rather than choosing one backend for the whole
layer.

## 2. AMX, SVE2, and VNNI

Runtime capability reporting now distinguishes NEON, SVE/SVE2, AVX2/FMA,
AVX-512, AVX-VNNI, AVX-512 VNNI/BF16, and AMX Tile/INT8/BF16. The measured
Ryzen 7 9800X3D reports AVX2/FMA, AVX-512, AVX-VNNI, AVX-512 VNNI, and
AVX-512 BF16; it does not report AMX.

Backend rules:

- SVE2 now has a separately compiled, vector-length-agnostic implementation
  that preserves FP32 accumulation. Runtime microbenchmarking selects it only
  when it beats NEON.
- VNNI and AMX-INT8 require activation quantization. They are not exact
  replacements for MXFP4 weight × FP32 activation and must be an opt-in
  quantization mode with error validation.
- AVX-512 BF16 and AMX-BF16 require BF16 rounding and are most useful for
  multi-row Prefill or cross-Session batches. AMX setup cost makes it a poor
  unconditional single-row Decode choice.
- Every new kernel participates in the same shape-aware runtime
  microbenchmark and retains scalar/NEON/AVX fallback.

## 3. Deeper Attention fusion

The current Vulkan Attention block already fuses Q/K/V weights, keeps KV on
device, and submits one command sequence per layer. The remaining 35.71
ms/token is mostly dense projection work plus 36 CPU/GPU synchronization
boundaries, not historical KV copying.

Implemented steps and remaining order:

1. QKV Slice, three Reshapes, head/token Permutes, and Q/K RoPE are fused in
   one native FP32 shader. The 120B A/B hit 576/576 blocks and reduced
   Attention from 549.1 to 512.8 ms.
2. Single-token QKV+RoPE writes K/V directly to both physical copies of the KV
   ring. It removes the K/V intermediates and append dispatch independently of
   the later SDPA choice. With online SDPA fixed in both arms, median Attention
   fell from 494.368 to 487.553 ms and throughput rose from 12.380 to
   12.555 token/s. The ncnn fallback materializes a learned-sink row lazily.
3. A single-token Decode shader now consumes the double-written KV ring,
   performs online Softmax without materializing the score vector, and writes
   the output-projection input layout directly. Device-resident sink values
   remove the Decode mask upload. A device/shape/context policy samples both
   paths and retains ncnn when the fused path is not at least 2% faster.
   Same-build forced A/B reduced Attention from 506.5 to 494.5 ms; the final
   default report reached 12.359 token/s with 538/576 fused blocks.
4. Fuse RMSNorm with QKV input consumption.
5. Fuse output projection tail, residual, and the next normalization where
   dependency and precision permit.
6. When enough Experts remain on Vulkan, keep hidden state and Router on device.
   Download only Top-K metadata; download hidden activation only for routes
   selected for CPU execution.
7. Record several consecutive all-GPU layers into reusable command buffers
   when their Experts are resident.

Fusion is selected from shape and shader capability. Each stage must retain
the existing CPU/Vulkan logit parity tests.

## 4. Multi-GPU placement

The ncnn backend owns one context per enumerated Vulkan device. A compiled
model now accepts a candidate set and records a device on each layer:

- a device index in tensor residency, compiled layers, and execution lanes;
- a placement plan minimizing per-sequence latency plus the expected
  concurrent pipeline bottleneck;
- per-device queue, heap, rough capability score, and independent Expert
  execution calibration;
- Expert cache sharding by layer placement and byte budget;
- no assumption of Vulkan peer memory: host-staged transfer is the portable
  fallback, while external-memory/peer paths are optional capabilities;
- independent failure fallback to CPU or another compatible device.

On a single-GPU system this infrastructure remains dormant. With RTX 5070 Ti
plus a slower AMD iGPU, an initial rough-score 29/7 split regressed to
7.979 token/s. The concurrency-aware objective now leaves the slow candidate
unused for a two-Session workload and records `0:36`, recovering 18.599
token/s in the verification run.

## 5. Adaptive long-context and multi-Session micro-batching

Autoregressive dependencies prevent batching future tokens from one
ordinary Session. The scheduler can obtain larger rows from multiple
Sessions or from speculative decoding, not by violating that dependency.

The scheduler now:

- buckets decisions by context-length range and request count;
- separates each workload bucket into resident, mixed, and storage phases
  from the previous Expert cache-wait share, using 5% and 50% boundaries;
- measures staged and independent end-to-end microseconds per request;
- tracks exponentially weighted absolute timing deviation and requires a
  candidate to beat both the relative margin and half the combined deviation;
- uses confidence-aware hysteresis: a sparsely sampled candidate must lead by
  10%, then 7.5%, before converging to the 5% steady-state margin;
- waits one probe interval before the first alternative-path sample, so short
  requests are not dominated by cold exploration;
- immediately confirms a prospective switch with a second candidate sample;
- expands the probe interval by up to 8x when the measured paths are far
  apart, while a new context/request-count bucket starts from the base
  interval;
- collects independent singleton submissions for at most 200 microseconds,
  up to the worker count, before applying the existing execution policy;
- never selects the same Session twice, preserving per-Session FIFO order;
- backs off after four empty collection probes and periodically probes again,
  so low-concurrency traffic does not continuously pay the latency budget;
- divides detected logical processors across workers by default.

A 96-token, two-Session 120B screen exposed why confidence is necessary. The
previous fixed-probe policy switched 45/95 decisions to staged execution and
measured 7.366 aggregate token/s. The confidence-aware policy kept staged work
to two probes, made no switch, and measured 8.026 token/s with the identical
generated sequences. Expert time fell from 19.719 to 18.276 seconds. This is a
single-run long-window diagnostic rather than a three-run acceptance result;
its 90.88% ARC hit rate and 30.56 GiB of Expert reads show that storage
residency, not the short-window compute path, becomes the dominant limit.

The Runtime now exposes a second, compressed-weight Vulkan cache tier behind
the host ARC. It is independent from the executable Expert cache, distributes
capacity across selected devices by capability score, deterministically shards
keys, and enforces a one-quarter reported-heap safety limit per device. An
explicit cache synchronization API drains host reads and victim admissions at
warm-up/traffic-transition boundaries, and also covers executable Expert
admissions, without changing the asynchronous request path.

With a 3 GiB victim L2 and synchronized warm-up, three 96-token/two-Session
runs measured 8.035, 8.193, and 8.798 token/s (**8.193 median**). Median
Expert reads were 28.20 GB, versus 32.81 GB in the one-run no-L2 diagnostic.
Median Expert cache wait was 11.16 seconds and median GPU restore time was
0.772 seconds, so storage wait remains the dominant gap to a sustained
10 token/s. A standard ARC policy was rejected for this lower tier: the host
ARC's eviction stream caused stale frequency protection and reduced the test
to 7.886 token/s. Host residency and executable Expert caching retain ARC;
the compressed victim tier uses recency replacement.

The next accepted storage optimization is a packed ExpertStore sidecar. It is
a standard Safetensors file whose per-Expert Gate/Up blocks, Gate/Up scales,
Down blocks, and Down scales are adjacent. `Mxfp4ExpertCache` detects this
layout generically and returns four shared views backed by one range read; the
original layout still follows the four-read fallback. The GPT-OSS packer
creates the sidecar atomically and never rewrites the checkpoint.

With a 22 GiB host ARC and 3 GiB Vulkan victim L2, three 96-token/two-Session
runs measured 9.833, 9.125, and 9.371 token/s (**9.371 median**). Median
generation was 20.490 seconds, Expert wait 8.127 seconds, Expert compute
5.494 seconds, and Attention 3.995 seconds. A direct disk miss is now one
12.61 MiB range rather than four ranges. This closes most of the sustained
10 token/s gap without an ISA or device-specific branch. A 23 GiB cache
regressed to 6.873 token/s under memory pressure and is rejected for this
32 GiB host.

The native executable Expert cache and compressed victim L2 are no longer
mutually exclusive. The Runtime distributes each capacity independently by
device capability, validates their combined per-device footprint against one
quarter of the reported heap budget, and exposes separate execution-cache and
victim-cache counters. With the same 3 GiB total GPU budget, a 512 MiB
executable + 2.5 GiB victim screen first measured 9.819 token/s median. The
split-counter validation cohort measured 9.402, 9.088, and 8.622 token/s
(**9.088 median**), with median 8.687 seconds of host-cache wait. It recorded
311 native Vulkan executions in 241.6 ms and 4.19 GB of victim restores in
1.419 seconds. A 256 MiB executable + 2.75 GiB victim split achieved 9.994
token/s in one run but only 8.494 token/s over three runs. Coexistence is
accepted, but neither capacity split is a stable performance default and the
long-window 10 token/s gate remains open.

An explicit eight-worker I/O screen measured 9.441 token/s, effectively the
same as the 9.402 token/s first sample from the automatic configuration:
Top-4 already makes the shape-driven Runtime choose eight workers on this
host. A one-run adaptive read-policy screen measured 9.787 token/s and selected
direct I/O before the measured window, with zero measured buffered ranges.
Neither result justifies a fixed worker count or filesystem policy.

The cross-call collector was then validated without application-side vector
batching. On the 16-token/two-Session screen, three samples were
20.230/20.369/13.674 token/s (**20.230 median**); every measured round formed
a two-request batch and the median total collection wait was 2 microseconds.
This proves the Runtime can discover service concurrency, but the variance
does not establish a stable 20 token/s floor.

On the 96-token storage-pressure screen, cross-call collection with staging
disabled measured 9.751 token/s, while the explicit-vector control measured
9.587 token/s. The collector spent only 22 microseconds waiting across
95 batches. With automatic staging, however, the same cross-call path
measured 8.014 token/s after 17 staged decisions; Expert compute rose from
5.881 to 9.097 seconds. The collector is therefore accepted, while the next
general scheduler optimization is cache-pressure/phase-aware policy
confidence rather than a longer fixed collection delay.

Removing both GPU Expert tiers from the staging-off control reduced throughput
from 9.751 to 8.871 token/s. Expert wait rose from 8.036 to 9.251 seconds and
disk reads rose from 22.79 to 26.68 GiB. The caches therefore provide a real
net benefit on this workload. However, the enabled run admitted 26.65 GiB to
the victim tier for 314 restores (3.87 GiB downloaded), so a portable next
step is online admission/queue-pressure control that preserves useful restores
without letting low-yield uploads contend indefinitely with dense Vulkan work.

The phase-aware policy was accepted in a three-run 96-token/two-Session
cohort. It measured 9.565/9.003/9.453 token/s (**9.453 median**), made no
policy switch, and issued only one or two staged probes per run. The matched
staging-off cohort measured 9.546/8.483/10.137 token/s (**9.546 median**).

### Victim admission and Attention epilogue follow-up

An optional bounded ghost/doorkeeper now distinguishes first host-ARC
evictions from keys seen across separate host residencies. The policy is
model- and device-neutral, stores metadata rather than weights, and samples
first observations at `1/N`. `N=1` preserves the previous admit-all behavior.

On the packed 96-token/two-Session staging-off screen, `N=2` reduced victim
uploads from about 26.65 GiB to 13.35 GiB while retaining 317 restores, but
throughput was only 8.633 token/s. `N=4` reduced uploads to 6.62 GiB, retained
239 restores, and measured 8.318 token/s. Neither passed the 9.5 token/s
single-run gate, so no repeated acceptance run was justified and sampling is
not enabled by default.

An Attention epilogue experiment removed the Vulkan residual BinaryOp and
folded the add into the projection download loop on CPU. Although it removed
one device dispatch per block, its three-run median was 9.151 token/s and
median Attention time was 4.760 seconds, versus the existing 9.546 token/s
staging-off cohort. The change was reverted because it competes with the other
Session's CPU Expert phase. The next high-value boundary is therefore a
device-side projection epilogue or direct victim-to-executable-Expert handoff,
not host-side residual work or more fixed admission thresholds.

### Device-resident victim execution follow-up

The compressed victim tier can now lend an aligned Vulkan MXFP4 buffer directly
to the native Expert operator. Four aligned block/scale subviews share the
existing `VkMat`; only the small bias buffers are uploaded. A borrowed
operation pins its complete victim entry, so eviction cannot release allocator
state while an asynchronous Expert command still references the buffer.
Successful execution, rather than candidate lookup, updates victim recency.

The first eager version exposed two important failure modes on the 96-token,
two-Session screen:

- returning only the operation without its owning entry failed under cache
  churn with Windows `0xC0000374`; the entry lease fixed the lifetime error;
- eagerly creating an operation for every victim admission and broadly
  executing source hits reduced throughput to 4.856 token/s. It made 663
  source executions and raised Expert compute from 10.33 to 24.19 seconds in
  the adjacent control period, despite reducing victim downloads.

The final implementation is lazy and phase-adaptive. It stores only compact
matrix layout/bias metadata with each compressed entry, waits for a measured
CPU phase baseline, probes one of every 64 eligible misses, and enables broad
source use only after the end-to-end hybrid phase wins with hysteresis.
Device-resident victim execution defaults on. The Runtime represents its
compatibility escape hatch with
`RuntimeOptionDisableGpuVictimExecution` in
`RuntimeOptions::flags`; the runner and benchmark wrapper expose
`--disable-gpu-victim-execution` as an A/B and compatibility
escape hatch.

During a period where background desktop processes consumed roughly
12-15% of total CPU capacity, the same-binary paired screen measured 4.294
token/s enabled versus 4.245 disabled. The enabled arm made 10 lazy source
executions with zero failures and reduced restore time from 3.107 to 2.710
seconds. Identical greedy tokens were produced. This validates bounded default
overhead and fallback behavior, but the noisy single-run pair is not a
10-token/s acceptance. The existing 9.546 token/s three-run long-window median
remains the comparable near-10 reference until an idle-host repeat is run.

The remaining general bottleneck order is now:

1. reduce host ARC miss/cache-wait time without increasing CPU memory pressure;
2. make cross-Session CPU/GPU phase arbitration account for shared CPU
   contention, rather than isolated Expert-kernel speed alone;
3. retain lazy device-source execution for hardware/storage combinations where
   its end-to-end phase measurements actually win;
4. pursue fused native Expert work and longer-context microbatching only with
   token-parity and three-run acceptance gates.
The off path was 0.9% faster by median but much more variable, while the
automatic path eliminated the prior 17-staged-decision failure. This is a
scheduler-stability acceptance, not a sustained 10 token/s acceptance.

The corresponding 16-token cohort measured
19.640/20.093/20.424 token/s (**20.093 median**) with zero staged probes,
zero switches, and identical generated sequences. This is the current
short-resident-window 20 token/s service point.

## 2026-07-26 runtime-overhead and allocation audit

- Runtime and scheduler configuration switches are packed into domain-specific
  `uint32_t` flags. Mutable stop states protected by different mutexes remain
  separate to avoid unsafe packed-word read-modify-write.
- Session-owned scratch now reuses Router, Combine, staged execution,
  Attention, and Expert backend buffers. Fully overwritten outputs skip
  zero-fill; accumulation outputs continue to clear.
- Transactional Session statistics use two reusable buffers and swap only
  after success. Greedy sampling avoids its former one-element allocation,
  and generation reserves its requested token capacity.
- Single-token `top_k <= 16` routing uses a stack candidate array and a
  persistent per-layer dispatch plan. Large tensors remain heap-backed and
  capacity-reused; they are deliberately not placed on worker stacks.
- Exact Expert requests return their enqueue-time ready state, removing the
  preceding readiness lookup. Pending reads are consumed in completion order
  instead of blocking behind the oldest request.
- Lazy Vulkan victim operation creation runs outside the executable-backend
  scheduler lock and rechecks residency after relocking. Single-device
  submissions no longer wait and copy their result vector a second time from
  the destructor.
- The scheduler now releases staged Session ownership before fulfilling its
  result future. The previous order exposed a transient busy state and could
  prevent an immediately following batch from being coalesced.
- A same-binary completion-order A/B with direct victim execution disabled
  measured 8.946 token/s versus 8.715 token/s for forced front waiting
  (+2.65%, identical tokens). In the later full audit run, the repeatable
  96-token sample measured 8.802 token/s. Relative to the 8.946 reference,
  Router fell 11.4%, regroup 24.1%, Combine 8.5%, and cache wait 2.6%, while
  variable Expert compute and LM Head time kept end-to-end throughput 1.6%
  lower. An adjacent first run at 5.817 token/s and a short resident run at
  14.825 token/s showed whole-machine compute variance; neither is accepted
  as a code regression or a new performance floor.
- After the statistics/submission cleanup, the adjacent short-window median
  was 15.049 token/s versus 14.825 token/s before it. This 1.5% movement is
  directionally consistent but remains within the larger machine-state
  variance and is not presented as isolated attribution.
- Lock-free ARC/scheduler queues and red-black-tree/Trie replacements are not
  accepted without contention evidence. ARC requires atomic consistency
  across byte capacity, resident/ghost lists, eviction, and leases; its
  existing hash + list representation already provides average O(1)
  operations.

Admission budgets based on live KV/scratch pressure, speculative multi-token
Decode, and a single-Session cross-token pipeline remain future work. Direct
dense-device calibration and peer/external-memory transfer paths are also not
implemented.

## Verification gates

Each optimization is accepted only when it proves:

1. deterministic token parity for the established 32-token sequence;
2. CPU/Vulkan kernel numerical agreement on randomized shapes and tails;
3. no unsupported ISA execution on older x86 or ARM devices;
4. bounded cache capacity with pinned-entry safety;
5. no synchronous upload-on-miss regression in `Auto`;
6. three measured 120B runs with phase timings, RSS, VRAM, upload bytes,
   cache coverage, and generated token IDs;
7. portable CPU-only build and Vulkan-enabled build both pass.

Starting feasibility evidence is stored in
`build-ncnn/gpt-oss-120b-20tps-feasibility2-32x1.json`; current acceptance
evidence is stored in the reports named above.
