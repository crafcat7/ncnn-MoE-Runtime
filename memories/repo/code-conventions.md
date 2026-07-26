# Code Conventions

- Use C++20 and place every project C++ symbol in `ncnn::moe` using the explicit ncnn-style nesting `namespace ncnn { namespace moe { ... } }`.
- Do not create anonymous, `detail`, or other auxiliary namespaces. Give translation-unit helpers internal linkage with `static`.
- Avoid function templates when a concrete type is sufficient. `Result<T>` is the approved generic result container; do not introduce additional templates without a concrete need.
- Follow ncnn's direct implementation style: use concrete helpers, plain data structures, explicit control flow, and include guards. Avoid abstraction layers that are not required by a measured need.
- Store configuration switches in a domain-specific `uint32_t flags` field.
  Declare an unscoped `uint32_t` enum beside the owning structure and query it
  with `has_flag`. Define every numeric position once as an object-like
  `#define NCNN_MOE_<DOMAIN>_<FEATURE>_BIT <index>` immediately before the
  owning enum, then derive the typed enum value with
  `UINT32_C(1) << ..._BIT` or `UINT64_C(1) << ..._BIT`. Call sites use the enum
  value, never the position macro or a literal shift. Do not mix bits from
  Runtime, Scheduler, adapter, graph, cache, or backend-capability domains.
  Apply the same rule to CPUID and XSTATE feature positions; arithmetic data
  encoding is not a feature bitmap.
- Minimize macros outside named bitmap positions. Include guards,
  compiler/build capability checks, and object-like `NCNN_MOE_*_BIT` position
  macros are permitted. Do not introduce function-like macros when a C++
  construct is sufficient. Existing test assertion macros should be removed
  opportunistically when their surrounding tests are otherwise being edited.
- Follow the repository `.clang-format`: four spaces, left-aligned pointers,
  no namespace indentation, and Allman braces for functions, classes,
  namespaces, and multi-line control flow. `else`, `catch`, and a `do` block's
  `while` start on a new line. Match upstream ncnn's `ColumnLimit: 0`; do not
  force mechanical line wrapping. Use the C++11 formatter parser because the
  project uses raw strings and modern C++ syntax.
- Place private helpers and private data above the compact public API in
  project classes. Do not expose construction, mutation, lookup, or snapshot
  helpers that are only used by the owning compiler, store, or backend.
- Prefer named static/private helpers or explicit loops over local algorithm
  lambdas. Keep lambdas only where a closure is intrinsic: thread jobs,
  condition-variable predicates, asynchronous completion, one-time
  initialization, or resource deleters.
- Use include guards rather than `#pragma once`, matching ncnn public headers.
- Return expected user/model failures through `Result<T>`. Return successful values and `Error` objects directly; use assertions rather than exceptions to guard access to the wrong Result alternative.
- Validate model dimensions, tensor types, tensor shapes, and tensor lengths during compilation, not in the execution hot path.
- Keep canonical tensor names in `src/internal/tensor_names.h`.
- Keep model-family parsing inside adapters. Execution code must use descriptor fields and resolved handles only.
- Keep CPU activation batches in one contiguous row-major buffer. Gather every non-empty Expert group into one contiguous batch and call the batched Linear boundary once per projection; do not reintroduce per-token Expert execution or nested activation vectors inside the executor.
- Session operations must be transactional: invalid input must not advance sequence length or statistics.
- Keep the root README concise and release-facing: current capability and
  reproducible performance precede build details, isolated smoke timings stay
  out of the headline table, and every reported benchmark states its protocol.
- Treat every public README as product documentation, not a development log.
  Describe the project, supported capabilities, architecture, operating
  modes, reproducible representative performance, and usage. Do not include
  dated progress entries, optimization diaries, target-status commentary, or
  superseded benchmark series. Preserve detailed experiments under
  `memories/repo/investigation-results/` or a dedicated technical report.
- Lead the root README with the problem the Runtime solves and one verified
  constrained-hardware operating point. Follow with the support matrix,
  representative performance, concrete performance differentiators, an
  end-to-end quick start, and architecture. Keep benchmark protocols explicit
  and do not present different workloads as a same-protocol comparison.
- Keep the root README's subject model-neutral. Present built-in model families
  as validated adapters and benchmark targets, never as the definition of the
  Runtime. Model-specific downloads, commands, storage layouts, and tuning
  belong in `models/<family>/README.md`.
- Keep model download, execution, backend, storage, and benchmark tutorials
  under `models/<family>/README.md`. The root README owns only the project
  position, a compact capability/performance summary, shared build entry,
  architecture, and links into the model catalog.
- Public model guides must use workspace-relative checkpoint paths. Download
  official packages under the ignored `models/<family>/<model>/` directory;
  do not instruct users to maintain a separate external model workspace.
- Do not publish roadmaps, future plans, planned features, or roadmap-like
  limitation lists in external README files. Keep internal planning and
  unfinished-work tracking outside release-facing documentation.
- Keep comments only when they document an API contract, a correctness
  invariant, a platform/toolchain constraint, or a non-obvious performance
  decision. Do not annotate literal fixture data, repeat a function or field
  name in prose, narrate implementation steps, or preserve historical
  debugging notes in release code.
- Keep private implementation files inside the responsibility directories:
  `src/engine`, `src/graph`, `src/models`, `src/storage`, `src/kernels`, or
  `src/backends/ncnn`. Cross-directory private includes use a path relative to
  `src`, such as `storage/expert_cache.h`; do not add new files to the `src`
  root.
