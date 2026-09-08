# Validation — 2026-09-06

Bootstrap the project environment and locked test dependency with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The final implementation passed 20 automated tests, including the opt-in Docker
integration tests:

```sh
BENCH_DOCKER_TESTS=1 .venv/bin/pytest -q --basetemp=/data/local-agent-evals/delivery-tests-tmp
```

Coverage includes deterministic category-balanced sampling with suite-wide totals,
configuration limits, serial scheduling, shared-lock contention and queue expiry,
cumulative active budgets, one transient retry, changed-checkpoint rejection after
queuing, owned-container cleanup, partial/timeout reporting, interrupted generation
and grading, and incompatible preparation rejection.

The pinned lm-eval SQLite cache was tested directly inside Docker: a child exited
after committing one response, resume generated only the missing response, the
outputs matched uninterrupted execution, and another checkpoint used an isolated
cache. A real SIGKILL watchdog exercise verified that the shared and ownership
locks remained held until the owned container was removed and crash reports were
written.

A live grading-container probe confirmed non-root execution, no external network,
no NVIDIA devices, no model mount, no host credentials, and no Docker socket.
Docker inspection confirmed a read-only root filesystem, no added capabilities,
two CPUs, 4 GiB memory without extra swap, and 128 processes. The actual EvalPlus
base/plus grader was also exercised with a canonical solution.

The final model-backed acceptance run processed two examples each of IFEval,
HumanEval+, BBH, and MMLU-Pro using INT8 Agentic v2. All four stages completed with
no infrastructure errors or retries. The run froze implementation commit
`4d840a7`; its Python source hashes match the delivered implementation. Subsequent
commits add tests and documentation only. Raw evidence, generated code, model
scores, and reports remain private under `/data/local-agent-evals/`.

A separate real stop/resume exercise preserved committed IFEval results. Both
resumed responses and scores exactly matched the final uninterrupted acceptance
run. The diagnostic HumanEval run also resumed from saved generations into grading
without regenerating them. Diagnostic evidence is retained separately from the
successful final acceptance run.

Postflight checks confirmed that all owned containers were removed, both GPUs
were idle, SSH/Docker/containerd were active, remote-access routes were intact,
dpkg reported no audit issues, and the protected NVMe remained read-only and
unmounted. The runner did not change swappiness or host service configuration.

## Scope and limitations

- The full overnight suite was not launched. Tiny acceptance validates the runner,
  not checkpoint quality or an official leaderboard score.
- The real acceptance used INT8 Agentic v2. Two-model configuration, cache
  isolation, and paired reporting are tested; a real original-INT8 comparison was
  not run.
- Setup requires the existing local pinned evaluation base image. The repository
  does not download model weights or host ML dependencies.
- Full weight hashes are recorded during preparation. Startup/resume checks file
  names, sizes, inodes, modification times, and non-weight content hashes.
- Docker provides the documented container boundary on the shared host kernel;
  the grading environment is not a virtual machine.
