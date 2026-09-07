# INT4-v1 vs INT8-v2 paired calibration baseline

Date: 2026-09-07  
Run: `20260907T204532Z-dd2ef2503084`  
Protocol: local, deterministic, non-thinking, 16K scored calibration  
Runner commit: `090f2c8`

This is the first compact, repository-tracked quality baseline for comparing
future local model artifacts. It contains reviewed aggregates only. Raw prompts,
responses, generated programs, caches, model weights, and host-private paths
remain outside Git.

## Hardware and serving envelope

- Two NVIDIA RTX 3090 GPUs, 24 GiB each.
- vLLM 0.27.1; one active model server at a time.
- INT4-v1: TP1 on GPU 0, 16,384 context, FP8 KV, 0.75 GiB KV reservation.
- INT8-v2: TP2 across both GPUs, 16,384 context, BF16 KV, 0.75 GiB KV reservation.
- Non-thinking, text-only, eager execution, one sequence, 1,024-token chunked
  prefill, no prefix cache.
- The supervisor automatically started/stopped each server and released both
  GPUs between model stages.

## Scores

Each benchmark used the same frozen seed-42 selection: 8 examples per model.
Scores are directional calibration results, not leaderboard estimates.

| Benchmark | Metric | INT4-v1 TP1 | INT8-v2 TP2 | Delta |
|---|---|---:|---:|---:|
| IFEval | prompt-level strict accuracy | 0.7500 (6/8) | 0.8750 (7/8) | +0.1250 |
| HumanEval+ | pass@1 | 0.8750 (7/8) | 1.0000 (8/8) | +0.1250 |
| BBH | answer-choice likelihood accuracy | 0.7500 (6/8) | 0.7500 (6/8) | +0.0000 |
| MMLU-Pro | answer-choice likelihood accuracy | 0.6250 (5/8) | 0.7500 (6/8) | +0.1250 |

All 64 planned examples were scored. There were no infrastructure errors. INT4
had one HumanEval+ execution failure and one IFEval truncation; INT8 had two
IFEval truncations. The report treats those outcomes according to the pinned
runner protocol and does not invent a combined intelligence score.

## Paired interpretation

- IFEval: 1 failure changed to a pass; 6 pass-to-pass, 1 fail-to-fail.
- HumanEval+: 1 failure changed to a pass; 7 pass-to-pass.
- BBH: no changes; 6 pass-to-pass, 2 fail-to-fail.
- MMLU-Pro: 1 failure changed to a pass; 5 pass-to-pass, 2 fail-to-fail.

The small denominator makes this a calibration anchor, not a stable ranking.
Future Q5/Q6 or other model runs should reuse the same suite, seed, task pins,
and profile family before changing the sample size.

## Reproduction and provenance

The reproducible suite is
[`examples/paired-16k-scored-calibration.yaml`](../examples/paired-16k-scored-calibration.yaml).
The complete private report and raw evidence are retained by the local runner;
this repository intentionally records only the reviewed aggregate and serving
configuration.
