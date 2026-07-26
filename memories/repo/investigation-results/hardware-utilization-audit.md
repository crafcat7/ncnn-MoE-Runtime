# Hardware Utilization Audit

## Verified baseline

- The MSVC Release build with bundled ncnn, Vulkan, and OpenMP passes
  `ncnn_moe_tests`, `ncnn_moe_style`, and `ncnn_moe_gpt_oss_fixture`.
- Runtime capability discovery reports physical/logical CPU counts, OpenMP
  team size, selected MXFP4 ISA, and all visible Vulkan device profiles.
  `RuntimeOptions::vulkan_device_indices` accepts candidate devices;
  automatic mode uses ncnn's default device.
- Compiled schedules retain detected CPU parallelism, expected concurrency,
  and a per-layer Vulkan placement. A slow candidate remains unused unless
  assigning it work improves the estimated concurrent makespan.

## Current utilization

- Vulkan owns complete dense Attention blocks, persistent Vulkan KV caches,
  eligible LM Head projections, and an adaptive single-token online SDPA
  shader. The shader is selected from measured device/shape/context buckets
  and removes Decode score materialization, Permute/Reshape, and mask upload.
- CPU owns Router, token grouping, and Combine. CPU Expert arithmetic remains
  the default; a native Vulkan MXFP4 backend may execute resident Expert
  batches after phase-level calibration proves it beneficial.
- Cache acquisition remains Top-K parallel. Resident MXFP4 decode flattens
  row-pair work across selected Experts and uses detected physical cores,
  avoiding SMT contention on bandwidth-bound kernels.
- Independent Sessions can Decode concurrently through `BatchScheduler`.
  Physical CPU cores are divided across active workers. Shared Vulkan commands
  execute through fence completion under the device lock, while a different
  Session can run CPU Experts or prepare a separate staging slot.
- MXFP4 dispatch is portable: scalar, ARM NEON/SVE2, x86 AVX2/FMA, and x86
  AVX-512 are selected by runtime benchmark without making a verified host
  ISA the binary baseline.

## Confirmed gaps

- `CpuExecutor` consumes dependency waves synchronously. Execution Graph events
  are a validated scheduling contract but do not yet overlap CPU Experts with
  later Vulkan work.
- The pinned ncnn command API now exposes `submit_async()` and `wait()`.
  Runtime Linear and Attention keep the shared-device mutex through fence
  completion because ncnn layers and the blob allocator are model-shared.
  Two staging slots still provide double-buffered host transfer preparation.
  Command objects are reset/reused; static command replay and single-Session
  cross-layer pipelining are not present.
- The Vulkan Expert cache is executable and byte-bounded, but the measured
  GPT-OSS-120B end-to-end calibration selects CPU Experts because shared
  CPU/GPU contention outweighs the isolated shader gain.
- One compiled model can place layers and Expert caches across candidate
  Vulkan devices. Cross-device activation transfer still uses host staging;
  peer/external-memory transport and tensor parallelism are not implemented.
- SVE2 and native Vulkan MXFP4 kernels are implemented. AMX/VNNI capability
  detection is available, but integer Expert kernels require an explicit
  activation-quantization and accuracy contract that does not yet exist.

## Durable decision

- Do not describe the runtime as maximizing every hardware resource by forcing
  every device to run. Describe it as measured heterogeneous placement:
  Vulkan Dense/Attention, calibrated CPU-or-Vulkan Experts, and portable CPU
  fallback.
- The next overlap implementation should pool command objects and connect
  graph Backend Events to persistent Vulkan fences for independent Sessions
  or micro-batches. Wrapping waits in host futures alone is not sufficient.
