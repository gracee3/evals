# Local Agent Evals

A small Python CLI for serial, resumable local smoke evaluations on two RTX 3090s.
Version 1 supports IFEval, HumanEval+, BBH, MMLU-Pro, IFBench, GPQA Diamond, and
LiveCodeBench v6. It defaults to the local
INT8 Agentic v2 checkpoint; the comparison suite adds the original INT8 checkpoint.
No weights are downloaded. Runs and reports are private and remain outside Git.

`examples/int4-acceptance.yaml` runs two examples per benchmark against the
completed expanded-400 INT4 v1 checkpoint. A suite containing only `int4-v1`
uses TP1 on physical GPU 0 by UUID, 96K context and a 3.5 GiB FP8 KV allocation.
The GPU choice and effective settings are frozen in each run. The shared lock
and whole-host idle checks remain in force. Two-model suites retain TP2;
INT4 paired runs are not covered by the single-GPU acceptance check.
HumanEval's owned server and the native harness both use the resolved TP setting.
The INT4 single-GPU profile is the validated maximum-context preset; 32K and 64K
remain lower-headroom alternatives. The quantization repository records the
validated TP2 comparison separately; this v1 acceptance suite remains TP1.

The completed INT4 v1 smoke is summarized in
[`docs/int4-v1-smoke-2026-09-07.md`](docs/int4-v1-smoke-2026-09-07.md).

`examples/paired-profiles-acceptance.yaml` demonstrates autonomous serial
switching: INT8-v2 uses the imported 262K FP8 TP2 profile, then INT4-v1 uses
the imported 96K FP8 TP1 GPU0 profile. Each model has its own active-hour cap.

Native lm-eval stages are grouped by model. IFEval, BBH, and MMLU-Pro share one
vLLM process and model allocation for each selected model; the process is torn
down before the next model. HumanEval+ keeps its separate owned HTTP server for
generation and grading. This removes repeated model loading and most repeated
Triton startup compilation while preserving the stage-scoped result caches and
the full teardown boundary between models.

## Setup

Use a checkout under `/home/emmy/workspace` and a project-local environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/bench list
.venv/bin/bench plan examples/smoke.yaml
.venv/bin/bench prepare examples/smoke.yaml
.venv/bin/bench run examples/smoke.yaml
```

`prepare` requires Docker access, the existing pinned
`qwen38-int8-lab/eval:0.1.0` base image, readable checkpoint metadata, and write
access to `/data/local-agent-evals`. It checks the base image's content identity,
builds the runtime overlay if absent, downloads pinned public benchmark data,
hashes the complete checkpoint, freezes deterministic sample selections, and
records package versions and task/dataset hashes. Model weights are mounted read-only
and read as container root because the existing shards are root-owned mode 0600.
Only the orchestration dependency PyYAML is installed on the host.

EvalPlus 0.3.1 is in a separate Python environment **inside the Docker image**.
Its resolved requirements are checked in; they do not change the inherited
lm-eval 0.4.12 / vLLM 0.27.1 environment. `pins.json` records the validated local
image identities. Docker builds can have different image identities even with
identical package versions; updating that pin requires explicit revalidation.
The local base image is a prerequisite, not a publicly downloadable image.

To reuse a pre-existing Hugging Face cache during initial preparation:

```sh
BENCH_IMPORT_HF_CACHE=/path/to/existing/huggingface .venv/bin/bench prepare examples/smoke.yaml
```

Existing prepared benchmark caches are reused automatically when available.
The cache is copied with copy-on-write when available, never hard-linked. No
original cache is mounted writable. Each run receives a separate cache copy.
`prepare` logs progress to the printed preparation directory's `prepare.log`.
Changed implementation or checkpoint identities require preparation again.

## Commands

```text
bench list                     supported benchmarks and local profiles
bench plan SUITE.yaml           validate; show total examples and deadline ceilings
bench prepare SUITE.yaml        prepare exact dependencies, data, and sample IDs
bench run SUITE.yaml            detach; print run ID and supervisor/status paths
bench status RUN_ID             stage progress, elapsed budgets, errors, liveness
bench stop RUN_ID               request owned-container cleanup; retain progress
bench resume RUN_ID             explicitly restart compatible saved work
bench report RUN_ID             regenerate local Markdown and JSON
```

Activate `.venv`, use `.venv/bin/bench`, or put that directory on your PATH.
There is no boot service, web server, external notification, or automatic result
publication. `run` is the only overnight launch; preparation does not start GPU
inference. `examples/acceptance.yaml` selects two examples per benchmark.

The bounded generation comparison example is
[`examples/qwen3.8-bounded-comparison.yaml`](examples/qwen3.8-bounded-comparison.yaml).
`bench preflight` runs synthetic CPU checks only: it does not discover GPUs,
load checkpoints, start vLLM, or report model scores. Published Qwen reference
values are stored separately in [`references/qwen3.8-27b.json`](references/qwen3.8-27b.json);
reports call local differences percentage-point differences from that reference,
not quantization loss.

## Suite semantics

`budgets: {run_to_completion: true}` disables stage, model and overall active
time cutoffs. Queue limits, explicit stop, RAM/swap protection, inference-error
handling, and generated-code grading timeouts still apply. Elapsed time is still
recorded. `examples/paired-16k-distributed-complete.yaml` uses this mode for a
fresh paired evaluation on the two local GPUs.

Opt in to independent model replicas with `--scale-gpus auto` on **both**
`prepare` and `run` (and optionally `plan`), or list GPU UUIDs after the flag.
The equivalent suite setting is `distribution: {gpus: auto}` or
`distribution: {gpus: [GPU-uuid1, GPU-uuid2]}`.

At launch, the selected devices are frozen into disjoint groups of the model's
tensor-parallel size. Four GPUs give four TP1 replicas or two TP2 replicas;
incomplete groups remain unused. The replica count is capped by the smallest
benchmark count. In distribution mode this pool overrides a profile's single
`gpu_device` pin, while preserving its TP size and inference settings.
The whole-host idle gate and shared exclusive lock remain required; `auto`
means all detected GPUs, not opportunistically sharing a busy host.

Replicas receive stable, disjoint slices of each benchmark's frozen IDs and
isolated response, compiler, and dataset caches under `replicas/`. Their committed
records are collected into the regular stage directories for live reporting.
HumanEval generation is parallel; grading remains serial after all GPU workers
stop. Budgets measure wall time, not summed replica time. Resume preserves
the frozen allocation and each replica's committed work. GPU replicas require
enough host RAM for simultaneous model loading; existing RAM/swap guards apply.
This option has automated orchestration coverage; multi-GPU replica inference
still requires a live acceptance run before treating throughput as validated.

See the commented [default suite](examples/smoke.yaml) and
[two-model comparison](examples/comparison.yaml). Counts are **total per benchmark**,
not per BBH task or MMLU subject. Seed 42 selects examples deterministically;
round-robin allocation spreads BBH and MMLU-Pro selections across categories.
Both models use exactly the same frozen IDs.

The default suite has 100 IFEval, all 164 HumanEval+, 60 BBH, and 60 MMLU-Pro
examples. Stage deadlines per model are 90, 150, 90, and 90 minutes. This gives a
seven-hour deadline ceiling for one model and fourteen for two; these are not
throughput predictions. Preparation and waiting for resources are separate from
the active budget (24 hours by default, configurable up to 48).

Runtime settings for the default INT8 suites are TP2, BF16 runtime dtype and KV cache, 16K context,
non-thinking, no MTP, no CPU offload, eager execution, no prefix cache, and one
model request at a time. The harness commits bounded batches (four examples by
default); that batch size does not enable model concurrency. Context length,
batch size, and KV cache allocation can be configured within validated limits.
The INT4-only acceptance suite overrides this with the validated GPU0 TP1
profile: 98,304-token context, FP8 KV, 3.5 GiB KV allocation, prefix caching,
non-eager execution, and 2,048-token chunked prefill.
All effective settings are saved in `frozen.json`.

Named native serving profiles are opt-in through `runtime_profiles`:
`int8-v2-16k-bf16-tp2`, `int8-v2-262k-fp8-tp2`,
`int4-v1-16k-fp8-tp1`, `int4-v1-96k-fp8-tp1`, and
`int4-v1-96k-fp8-tp2`. A suite may assign a different profile to each selected
model. `budgets.model_active_hours` bounds each model independently while
`budgets.active_hours` remains the whole-run cap.

The imported INT8-v2 profile retains the native 262K FP8 TP2 settings but uses
4.5 GiB of KV reservation per GPU. The native 3 GiB template admitted the
original INT8 profile, while v2's vLLM admission check required 4.16 GiB at
262,144 tokens; a real v2 smoke validated 283,236 available KV tokens.

IFEval uses the pinned leaderboard task with the local 1,024-token cap.
HumanEval+ uses EvalPlus 0.3.1's OpenAI chat prompt, greedy generation, sanitizer,
and full base/plus scoring on dataset v0.1.10, with one 2,048-token response.
The supervisor replaces EvalPlus's unbounded transport retry loop with its own
bounded policy. BBH and MMLU-Pro preserve the pinned leaderboard protocol's
**answer-choice likelihood scoring**, including native few-shot prompts.
IFEval and HumanEval+ retain vLLM finish reasons for truncation counts.
Inputs that would require context truncation are rejected. The known optional
DeepGEMM import probe warning on SM86 is retained in logs without treating it as
a failed W8A8 evaluation; OOMs, CUDA errors, and other tracebacks remain fatal.

These are protocol-qualified local smoke results, not official leaderboard
scores. No combined score is calculated. Reports separate planned, completed,
and scored counts, partial coverage, execution failures, infrastructure errors,
output truncation, and durations. Comparisons include paired pass/fail changes
only for examples scored by both models.

## Ownership, limits, and recovery

The supervisor queues for up to 24 hours until both GPUs are idle and it can hold
`/data/qwen38-int8-lab/quant-swappiness.lock`. It checks GPU availability again
after acquiring the lock and before each GPU stage. It never changes swappiness
or stops unrelated workloads. Containers have a unique ownership label; cleanup
checks that exact label and does not kill host processes by PID. A watchdog
inherits both the run-ownership lock and shared quantization lock; if the
supervisor is killed, it removes owned containers and writes crash reports before
releasing either lock. Swap-growth baselines persist across resumes within a boot.

A sustained ten-second breach of either 8 GiB available RAM or 32 GiB swap growth
stops the run. The baseline is captured when resources are acquired. Active and
stage time are saved every heartbeat and retained across retries and resumes.
A stopped run resumes only unfinished work. Deadline-exhausted stages retain
partial results and advance to the next stage; persistent runtime/resource
failures stop the supervisor. Only explicitly recognized transient connection
failures receive one infrastructure retry. Wrong answers, failing programs, and
timeouts never receive correctness retries.

Native lm-eval SQLite response caching commits responses transactionally. Result
records and HumanEval generations are written with fsync and atomic rename.
Resume reuses committed records; only uncommitted work repeats. Run and model
paths isolate caches, and frozen configuration, source, package/image, dataset,
and checkpoint identities prevent incompatible resumes. Full weight hashes are
recorded during preparation; startup/resume checks file names, sizes, inodes,
modification times, and non-weight content hashes. An unexpected host
reboot requires explicit `resume`; status reports stale supervisor state.

HumanEval generation uses an owned vLLM server bound only to the GPU container's
loopback interface. That container stops before grading. Each grading container
has a read-only root filesystem, no network, no GPUs, no model mounts, no
credentials or Docker socket, and runs as the host's unprivileged UID. It mounts
only its single example, its own output directory, and read-only runner code.
Limits are two CPUs, 4 GiB RAM with no extra swap, 128 processes, and a configurable
120-second outer wall deadline, in addition to EvalPlus's native test timeouts.
This is Docker isolation on a shared kernel, not a VM boundary.

All run artifacts are under `/data/local-agent-evals/runs/RUN_ID/` with private
permissions: frozen identities, live `status.json`, tail-able `supervisor.log`,
raw outputs/code, response caches, `report.md`, and `report.json`.

## Development and provenance

```sh
.venv/bin/pip install pytest==8.4.2
.venv/bin/pytest -q
BENCH_DOCKER_TESTS=1 .venv/bin/pytest -q tests/test_watchdog.py
```

Model construction remains in
[qwen38-int8-lab](https://github.com/gracee3/qwen38-int8-lab).
The base requirements, dataset revisions, native task choices, and RAM/swap guard
thresholds derive from that repository's pinned evaluation and quantization work.
This runner contains no dependency on its checkout location. Existing quantization
artifacts and historical evaluation caveats remain in that repository; a new
smoke score does not supersede its quality gates or establish equivalence to BF16.

Upstream protocols: [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12),
[EvalPlus 0.3.1](https://github.com/evalplus/evalplus/tree/v0.3.1).
See [validation and limits](docs/validation.md) for test and acceptance coverage.
Rust, long-context tests, Qwen Code tasks, and a web interface are outside v1.
