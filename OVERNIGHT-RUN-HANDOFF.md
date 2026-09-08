# Overnight paired evaluation handoff

Prepared 2026-09-08 after merging the grouped-worker changes.

The clean checkout is `~/projects/evals`, on `main` at commit
`d8b2173ada57e59fd03a2a81fb6f6de76b136daa`. The grouping implementation is in
PRs #5 and #6; both are merged. The local qwen38-lab checkout is
`~/projects/qwen38-lab` at merged commit
`b0c33924aa484558c223aed33e8544ce333ff5ab`.

## What changed

For each model, IFEval, BBH, and MMLU-Pro now run through one GPU worker and one
vLLM model allocation. HumanEval+ remains a separate owned HTTP server because
EvalPlus generation and grading use that interface. The worker writes a
`duration.json` per native benchmark; records and response caches remain in each
benchmark directory. The supervisor tears the worker down before changing models.

PR #6 fixed the first grouped-smoke setup error by creating the model stage
directory before opening its grouped log. The first smoke failed before the GPU
worker started and left both GPUs idle; it is not a model result.

The full test suite passed after both fixes: 25 passed, 2 skipped, in the existing
local eval image. No image was rebuilt or pulled. The old project checkout
`~/projects/local-agent-evals` was removed after verifying it was clean. The
separate worktree `~/worktrees/local-agent-evals` is retained and was not touched.

## Suite to run

Use [`examples/paired-16k-overnight-10h.yaml`](examples/paired-16k-overnight-10h.yaml).
It plans 2,440 model-example evaluations:

- 256 IFEval per model
- all 164 HumanEval+ problems per model
- 400 BBH per model
- 400 MMLU-Pro per model

The suite preserves the successful paired baseline's seed 42, 16K runtime
profiles, non-thinking generation, output limits, pinned datasets, and checkpoint
identities. It has a 10-hour total active cap, 5 hours per model, and 1 hour of
queue time. Prior steady-state timing extrapolated about 8.2 active hours;
expect variation, and treat stage/model deadlines as ceilings rather than a
throughput guarantee.

Preparation for the exact suite completed at:

`/data/local-agent-evals/prepared/96f598d52a87af087ade3fbe5d412ff08b7c0b4bbdbfd1737cc69a1b28682a60`

The preparation was rechecked after the grouped code change and matched the
successful baseline's dataset hashes, checkpoint metadata identities, profile
settings, generation limits, and first eight paired IDs. The preparation files
and raw runs remain private under `/data/local-agent-evals`.

## Launch

Before launch, confirm both GPUs are idle, the shared lock is available, and no
interactive server is using either card:

```sh
cd ~/projects/evals
python3 -B -m benchlib.cli plan examples/paired-16k-overnight-10h.yaml
python3 -B -m benchlib.cli prepare examples/paired-16k-overnight-10h.yaml
python3 -B -m benchlib.cli run examples/paired-16k-overnight-10h.yaml
```

`prepare` is safe to reuse when the prepared identity matches. `run` is the only
command that starts inference. Record the printed run ID, then monitor with:

```sh
python3 -B -m benchlib.cli status RUN_ID
tail -f /data/local-agent-evals/runs/RUN_ID/supervisor.log
```

Use explicit `stop` if the host needs the GPUs. Resume only with `resume RUN_ID`;
do not start a second run against the same prepared work while the first is live.

## Expected evidence and limits

The previous 8-example paired run completed 64 evaluations in about 35 minutes,
with no infrastructure errors. It scored INT4-v1 at 0.7500 IFEval, 0.8750
HumanEval+, 0.7500 BBH, and 0.6250 MMLU-Pro; INT8-v2 scored 0.8750, 1.0000,
0.7500, and 0.7500 respectively. Those values are directional only. The
overnight report must keep benchmark scores separate, report paired coverage,
truncation, execution failures, and infrastructure errors, and must not compute
a combined intelligence score.

The INT4 profile uses TP1 on physical GPU0 with FP8 KV; INT8-v2 uses TP2 across
both GPUs with BF16 KV. Teardown between models is the safety boundary. Within a
model, native stages share the model allocation and therefore avoid repeated
model load/Triton startup. Context/KV state is request-scoped; prefix caching is
disabled in these 16K profiles. HumanEval's server is stopped before grading.

The old `~/worktrees/local-agent-evals` feature checkout is preserved for history,
but it is not the launch checkout. Do not use the deleted `~/projects/local-agent-evals`
path or the old pre-grouped runner.
