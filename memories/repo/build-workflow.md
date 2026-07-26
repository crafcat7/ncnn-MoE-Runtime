# Build Workflow

## Configure and build

- Initialize dependencies: `git submodule update --init --recursive`
- MSVC/ncnn/Vulkan: `cmake -S . -B build-ncnn`
- MSVC Release build: `cmake --build build-ncnn --config Release --parallel`
- Portable Windows/MinGW: `cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DNCNN_MOE_USE_NCNN=OFF`
- Single-config Release build: `cmake --build build --parallel`
- The project requires a C++20 compiler and CMake 3.21 or newer.
- MSVC x64 builds compile the MXFP4 AVX2/FMA and AVX-512 kernels in separate
  translation units with `/arch:AVX2` and `/arch:AVX512`. The baseline dispatch
  translation unit checks CPUID and XGETBV before calling either implementation,
  preserving execution on older x86-64 hosts through the scalar fallback.
- Both the Visual Studio 18 2026 build and the MinGW/Vulkan build pass the
  mapped-storage, MXFP4 numerical, runtime, style, and GPT-OSS fixture tests.
  Windows handle sentinels must not require `constexpr` integer-to-pointer
  conversion because MinGW rejects it.
- ncnn is optional only for the explicitly portable build. The default build
  compiles the pinned `third_party/ncnn` submodule. With Vulkan enabled, `Auto`
  keeps complete Attention blocks and their KV caches on Vulkan, also dispatches
  Router and LM Head projections there, and leaves routing plus Experts on CPU.
- `NCNN_MOE_USE_VULKAN` defaults to `ON`. When `NCNN_MOE_USE_NCNN=ON`, ncnn discovery is required and a package built with `NCNN_VULKAN=OFF` is rejected. Set `NCNN_MOE_USE_NCNN=OFF` for the portable backend.

## Test

- MSVC Release: `ctest --test-dir build-ncnn -C Release --output-on-failure`
- Single-config: `ctest --test-dir build --output-on-failure`
- `ncnn_moe_tests` covers deterministic logits, chunked/one-shot Prefill parity, Attention/KV reset, BF16 ring-cache accounting, sampling/streaming, routing/grouping statistics, Session state, fused-QKV CPU/Vulkan parity, full Vulkan Attention with cache-view and sliding-compaction parity, exact dispatch accounting, CPU Expert prefetch statistics, transactional invalid-token behavior, adapter selection, and malformed weights.
- `ncnn_moe_gpt_oss_fixture` generates a multi-head GQA checkpoint in the official layout and compares multi-step automatic-backend generation with CPU generation, covering Attention, cache reuse, MXFP4 Experts, Prefill, and Decode end to end.
- `ncnn_moe_tests` also maps unaligned multi-page file ranges, verifies mapped
  Safetensors BF16/F32 tensors and slices, checks zero-fill-free MXFP4 shared
  buffer copy isolation, and validates mapped Expert-cache byte/range counters.
- `ncnn_moe_style` rejects anonymous/auxiliary namespaces, misplaced control-flow braces, pragma-once headers, and new templates outside the approved `Result<T>` container.
- After a unified benchmark, regenerate the README chart with
  `python tools/generate_performance_chart.py --report <matrix.json> --output assets/gpt-oss-performance.svg`.

## Example

- Run official GPT-OSS token IDs in mixed mode: `./build-vulkan/ncnn_moe_gpt_oss.exe ./models/gpt-oss/gpt-oss-20b 0 --max-new-tokens 1 --hybrid`
- Enable bounded CPU Expert cache hints with `--hybrid-prefetch`; `Auto` deliberately remains the standard mixed mode.
- Run a CPU comparison with the same binary: add `--cpu`; the runner reports its resolved backend, Vulkan linear dispatch count, and completed Vulkan Attention block count.
- Run a Harmony text smoke test: install `openai-harmony`, then `python tools/run_gpt_oss_prompt.py ./build-vulkan/ncnn_moe_gpt_oss.exe ./models/gpt-oss/gpt-oss-20b "Reply with exactly: OK" --max-new-tokens 64 --backend hybrid`

## Options

- `NCNN_BUILD_MOE_RUNTIME` enables the library and defaults to `ON` in this standalone repository.
- `NCNN_MOE_BUILD_TESTS` and `NCNN_MOE_BUILD_EXAMPLES` control optional targets.
- `NCNN_MOE_USE_NCNN` controls discovery of ncnn CPU/Vulkan operators.
- `NCNN_MOE_USE_VULKAN` requires Vulkan capability from a discovered ncnn package and defaults to `ON`.
