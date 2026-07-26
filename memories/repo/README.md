# Repository memory

This directory contains durable, reviewable project knowledge. The root
README is the public capability and benchmark summary; these files preserve
the implementation evidence and operating decisions behind it.

## Workflow and structure

- [`build-workflow.md`](build-workflow.md): verified build, test, and validation commands.
- [`code-conventions.md`](code-conventions.md): formatting, naming, and source-boundary rules.
- [`project-structure.md`](project-structure.md): responsibility-based source layout.
- [`model-format.md`](model-format.md): supported model package and tensor-format facts.
- [`gpt-oss-runtime.md`](gpt-oss-runtime.md): GPT-OSS adapter and runtime behavior.

## Investigation results

- [`hardware-utilization-audit.md`](investigation-results/hardware-utilization-audit.md): hardware and backend utilization findings.
- [`performance-notes.md`](investigation-results/performance-notes.md): chronological benchmark evidence and accepted/rejected measurements.
- [`gpt-oss-120b-20tps-roadmap.md`](investigation-results/gpt-oss-120b-20tps-roadmap.md): performance budget, bottleneck analysis, and general optimization roadmap.
- [`runtime-architecture-and-optimization.md`](investigation-results/runtime-architecture-and-optimization.md): architecture, optimization, and runtime-degradation audit.

The two archived audit documents were moved from the former `docs/` folder so
there is one repository knowledge location. Historical measurements remain
labelled as evidence; current public benchmark values are maintained in the
root README and the unified matrix report.
