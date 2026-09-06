# Local Agent Evals

Local evaluation of language models and coding agents, starting with Qwen Code
and locally served models on two RTX 3090 GPUs.

## Status

Repository initialized. Planning and implementation are pending approval.
No evaluation runner or benchmark suite is implemented yet.

## Intended scope

- Configurable smoke suites with selected benchmarks, sample counts, and time budgets.
- Serial execution with progress logs, recovery, and bounded resource use.
- Markdown and JSON reports covering results, failures, and runtime.
- Established benchmark tools for prompting and scoring where practical.

Model building remains in [qwen38-int8-lab](https://github.com/gracee3/qwen38-int8-lab).
Evaluation reports should identify the exact checkpoint, build commit, datasets,
and runtime settings used.

## Data boundaries

This is a public repository. Keep model weights, dataset caches, raw responses,
generated code, logs, credentials, and private task material outside Git.
The proposed local run-data root is `/data/local-agent-evals/`.
Only reviewed, compact summaries should be published here.

## Planning topics

Confirm the first supported benchmarks, model/server ownership, time and retry
budgets, resume behavior, execution environments for generated code, and the
report format before implementation.
