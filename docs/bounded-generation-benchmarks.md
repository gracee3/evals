# Bounded generation benchmarks

The CLI now recognizes `ifbench`, `gpqa_diamond`, and `livecodebench_v6`. They
are generation protocols and use the same frozen selection, response records,
resume identity, stage budgets, and paired intersection reporting as existing
suites. Each record stores the raw response, separate reasoning and final text,
extraction provenance, invalid state, truncation state, and grading result.

IFBench is intended to use the official held-out data and upstream strict and
loose verifiers. GPQA uses the gated `Idavidrein/gpqa` Diamond subset, with the
option order frozen in the prepared data and final-choice grading. Preparation
must fail closed when the gated dataset is unavailable. LiveCodeBench uses its
upstream code-generation grader and CPU-only isolated grading container; the
dataset release, date window, grader commit, and pass@1 sampling details still
need to be pinned before a reference comparison is called protocol matched.

The example suite is [qwen3.8-bounded-comparison.yaml](../examples/qwen3.8-bounded-comparison.yaml).
Use `bench plan`, `bench prepare`, and then `bench run` as with existing suites.
`bench preflight` exercises synthetic fixtures only and never starts vLLM or
claims a model result. The published references are stored separately in
`references/qwen3.8-27b.json` and reports label differences as percentage points
from the published reference, not quantization loss.

Thinking settings are opt-in through `generation`. Existing suites retain their
non-thinking defaults. New suites can set `enable_thinking`, `reasoning_effort`,
sampling parameters, `preserve_thinking`, request deadlines, and benchmark
specific output-token limits up to the validated 65,536-token configuration
ceiling. MTP remains disabled by the runtime profiles.

## Later agent-benchmark design

Terminal-Bench 2.1, SWE-bench Pro, and NL2Repo-Bench need an adapter interface
with four explicit pieces: immutable task manifest, owned task environment,
agent event/trajectory capture, and an isolated grader that returns pass/fail,
timeout, invalid, and infrastructure-failure states. The environment must be
created from a pinned image or repository snapshot, expose only declared task
resources, and prevent network, credentials, model mounts, and benchmark-data
exfiltration. The supervisor should reserve the existing GPU and host budgets
for the agent process, persist events transactionally, and resume only from a
committed task boundary.

Terminal-Bench should record the Terminus harness and command policy. SWE-bench
Pro must record the refined task manifest and Claude Code harness settings shown
in the Qwen reference. NL2Repo-Bench must enforce the model-card repository
access restrictions and record its repository snapshot and grader. These three
names should remain documentation-only until execution and grading integration
passes CPU fixtures plus two real examples per model.

## GPU acceptance checklist

After the image, model mounts, and gated data are available, run the example with
`bench prepare examples/qwen3.8-bounded-comparison.yaml`, then run the resulting
suite with `bench run examples/qwen3.8-bounded-comparison.yaml`. For the first
acceptance, use a copied suite with `count: 2` for each benchmark. Confirm serial
INT8-v2 TP2 then INT4-v1 TP1 execution, thinking flags and `reasoning_effort`,
output limits below context, request deadlines, memory admission, final-answer
parsing, LiveCodeBench grading, paired reporting, and stop/resume reuse. Review
the frozen metadata and logs, verify no owned containers remain, both GPUs are
idle, and the protected secondary NVMe is still unmounted and read-only.
