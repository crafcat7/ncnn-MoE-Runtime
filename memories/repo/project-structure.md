# Project Structure

- `include/ncnn/moe/` is the installed public C++ API.
- `include/ncnn/moe/moe_ir.h` names the model-neutral adapter/compiler boundary.
- `include/ncnn/moe/moe_ir.h` now defines a real value/node IR with Attention,
  Router, ExpertGroup, Combine, persistent KV Cache, and QuantConfig metadata.
- `include/ncnn/moe/expert.h` owns observable Expert lifecycle, residency,
  kernel selection, heat, and model-level `ExpertStore` statistics.
- `include/ncnn/moe/memory_manager.h` owns per-session Execution Tensor
  residency and transition statistics.
- `src/compiler/` validates and normalizes MoeIR and builds the compiled
  execution graph. Legacy adapters that populate descriptor layers are
  normalized automatically.
- `include/ncnn/moe/execution_graph.h` owns runtime node dependencies, backend
  placement, topological execution waves, and `MoeScheduler`.
- `include/ncnn/moe/expert_dispatcher.h` owns Top-K route grouping into stable,
  weighted per-Expert batches.
- `src/engine/` owns Runtime, Sessions, scheduling, CPU execution, per-session
  state, and CPU topology discovery.
- `src/graph/` owns Execution Graph validation, resource-aware scheduling,
  exact Top-K route grouping, memory planning, and model compilation.
- `src/models/` maps GPT-OSS into canonical descriptors, tensor names, and
  Safetensors metadata. GPT-OSS is the only built-in model adapter and is
  accepted against real 20B and 120B checkpoints. `MoeIR` remains
  model-neutral, but unsupported model-family stubs do not belong in the
  production adapter registry.
- `src/storage/` owns cross-platform, copy-on-write file-range mappings,
  shared shard handles, page pre-touch for opt-in Expert mappings, and mapped
  lifetime ownership, physical-memory discovery, byte-bounded Expert-pair
  residency, asynchronous exact/speculative reads, byte-aware ARC eviction,
  and platform-specific shard I/O.
- `src/engine/expert_backend.h/.cpp` defines the executable Expert-backend
  boundary and the composite multi-device backend. The Vulkan implementation
  lives beside the shared device context in `src/backends/ncnn/ncnn_linear.cpp`;
  it keeps packed MXFP4 pairs in a device-resident ARC, runs fused decode plus
  compute, and falls back to the CPU backend when admission, calibration, or
  capability checks reject the device path.
- `src/kernels/` owns portable dense, INT8, and MXFP4 Linear operations,
  runtime-dispatched BF16 Router dots (scalar/NEON/AVX2/AVX-512), shared
  tensor conversion helpers, and the portable GPT-OSS Attention path.
- `src/backends/ncnn/` owns ncnn CPU/Vulkan InnerProduct packaging, the native
  MXFP4 Expert shader, and the mixed-backend Vulkan Attention subgraph:
  RMSNorm, fused QKV layout conversion plus RoPE, direct double-written KV-ring
  output, adaptive single-token online SDPA, GQA, device-resident learned sinks,
  compact GPU KV caches, output projection, residual, synchronization, and
  dispatch accounting.
- `tests/` owns independent deterministic golden and error-path coverage.
- `tests/fixture_model_adapter.cpp` provides the private `test_moe` package
  adapter used by unit tests. It is compiled only into `ncnn_moe_tests` and is
  not a supported runtime model family.
- `models/README.md` is the public model catalog and execution-capability
  matrix. Each `models/<family>/README.md` owns that family's checkpoint,
  runtime, storage, and benchmark tutorial.
- `models/gpt-oss/README.md` owns official GPT-OSS download, token/Harmony
  execution, backend selection, Expert-memory controls, mmap behavior, the
  constrained-memory 120B recipe, and measured performance.
- `models/gpt-oss/gpt-oss-20b/` and `models/gpt-oss/gpt-oss-120b/` are ignored
  workspace-local directories for the downloaded official model packages.
- `tools/create_gpt_oss_fixture.py` creates a miniature checkpoint with official GPT-OSS config, Safetensors names, BF16 tensors, and MXFP4 Experts.
- `tools/run_gpt_oss_prompt.py` bridges official Harmony text encoding to the token-ID C++ runner.
- `tools/benchmark_performance_matrix.py` runs the unified cold/warm short/long
  reference matrix; `tools/generate_performance_chart.py` renders its published
  SVG chart into `assets/gpt-oss-performance.svg`.
- `assets/gpt-oss-performance.svg` is the public README benchmark visualization;
  it is generated from the matrix report and has a published-value fallback.

## Ownership boundaries

- A shared `Model` owns immutable descriptors, weights, and compiled plans.
- A `Session` owns sequence length, routing statistics, and persistent per-layer KV caches.
- `CpuExecutor` orchestrates the model-neutral plan. In `HybridExperts`,
  complete Attention blocks and eligible Router/LM Head projections dispatch
  through ncnn Vulkan, with per-session KV state remaining on the GPU. Routing
  remains CPU-owned. Experts use the CPU backend by default and may use the
  native Vulkan backend only after measured phase-level calibration says it is
  beneficial. `VulkanWithCpuPrefetch` remains an explicit compatibility mode.
- Candidate Vulkan devices are compiled into per-layer placements using
  capability scores and the requested concurrency. One-session plans prefer
  the fastest device; concurrent plans split only when the estimated
  makespan improves. The composite Expert backend dispatches each layer to its
  assigned device without forcing a slower GPU into the plan.
- Model adapters may parse configuration and map weight names into `MoeIR`, but
  may not make scheduling decisions.
- `ModelCompiler` lowers `MoeIR` and resolved weights into fused layer plans plus
  a tensor-aware `ExecutionGraph`. `RuntimeScheduler` assigns backend lanes and
  cross-backend events; `MoeScheduler` derives maximal topological waves. The
  current executor consumes the waves synchronously.
- `ExpertDispatcher` is the only exact-routing owner. Executors consume its
  active batches and may reorder whole batches for cache readiness, but must
  preserve route weights and token indices.
- Model-family checks must not enter Prefill or Decode.

## Current limitations

- CPU-only and Vulkan-dense/CPU-Expert mixed execution are implemented. `Auto`
  selects the mixed backend for a hardware Vulkan device and falls back to
  CPU-only for software CPU Vulkan; explicit mixed selection remains possible.
- Dense weights support float32 and BF16. Expert projections support float32, symmetric per-output-channel int8, and GPT-OSS MXFP4.
- GPT-OSS Attention and persistent KV caches are implemented. CPU mode stores
  float32 or BF16 ring buffers. Mixed mode currently uses a physical FP32
  double-written Vulkan ring; wrapped windows remain one offset view without
  historical concat or Slice.
- Full-logits output only.
- Dense weights and Expert biases are eager. GPT-OSS MXFP4 Expert blocks/scales
  can remain in their original Safetensors shards and enter a byte-bounded
  host cache on demand. An explicit Vulkan L2 may retain RAM-cache evictions.
  Dense BF16/F32 and eager MXFP4 tensors use mmap with buffered fallback.
  On-demand Experts default to aligned overlapped direct reads on Windows,
  with buffered fallback; explicit Expert mmap remains opt-in. Native Vulkan
  MXFP4 Expert compute and device-resident ARC caching are available but
  self-disable per phase when end-to-end calibration is unfavorable.
  Cross-layer GPU activation residency, peer-device transfers, paged GPU KV
  allocation, and a cross-token single-session pipeline are not implemented.
- Vulkan-only execution is rejected explicitly. Vulkan/CPU Expert prefetch is available as an explicit mode and is not selected by `Auto`.
- Shared Experts and unsupported model families/formats are rejected explicitly.
