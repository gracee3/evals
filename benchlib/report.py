from collections import Counter
from pathlib import Path

from benchlib.core import BENCHMARKS, REFERENCES, REFERENCE_SOURCE, read_json, write_json
from benchlib.worker import record_path


def report(run):
    run = Path(run)
    frozen = read_json(run / 'frozen.json')
    status = read_json(run / 'status.json')
    rows, outcomes = [], {}
    for model in frozen['suite']['models']:
        for bench in frozen['suite']['benchmarks']:
            name = bench['name']
            key = model + '/' + name
            directory = run / 'stages' / model / name
            records = [read_json(record_path(directory, item)) for item in frozen['prepared']['selection'][name]
                       if record_path(directory, item).exists()]
            scored = [r for r in records if r.get('score') is not None]
            strict = [r['strict'] for r in records if isinstance(r.get('strict'), bool)]
            loose = [r['loose'] for r in records if isinstance(r.get('loose'), bool)]
            state = status.get('stages', {}).get(key, {})
            reference = REFERENCES.get(name)
            measured = sum(r['score'] for r in scored) / len(scored) if scored else None
            reference_score = reference['score'] if reference else None
            row = dict(model=model, benchmark=name, metric=BENCHMARKS[name]['metric'],
                completed=len(records), scored=len(scored), planned=bench['count'],
                score=measured, official_reference=reference_score,
                strict_score=sum(strict) / len(strict) if strict else None,
                loose_score=sum(loose) / len(loose) if loose else None,
                difference_from_published_reference=(measured * 100 - reference_score) if measured is not None and reference_score is not None else None,
                attempted=len(records), partial=len(scored) < bench['count'], duration_seconds=state.get('elapsed', 0),
                truncated=sum(r.get('truncated') is True for r in records),
                invalid=sum(r.get('invalid') is True for r in records),
                execution_failures=sum(bool(r.get('execution_failure')) for r in records),
                infrastructure_errors=[e for e in state.get('errors', []) if e != 'stage deadline exhausted'],
                stage_timeout=state.get('status') == 'timeout', status=state.get('status', 'pending'))
            outcomes[key] = {str(r['item']): r['score'] for r in scored}
            rows.append(row)
    pairs = []
    if len(frozen['suite']['models']) == 2:
        a, b = frozen['suite']['models']
        for bench in frozen['suite']['benchmarks']:
            name = bench['name']
            left, right = outcomes[a + '/' + name], outcomes[b + '/' + name]
            changes = Counter()
            for item in left.keys() & right.keys():
                changes[f'{"pass" if left[item] == 1 else "fail"}_to_{"pass" if right[item] == 1 else "fail"}'] += 1
            pairs.append(dict(benchmark=name, first=a, second=b, paired=len(left.keys() & right.keys()),
                              planned=bench['count'], changes=dict(changes)))
    payload = dict(run_id=run.name, status=status, results=rows, paired=pairs,
        identities=frozen, scope='Local protocol-qualified smoke evaluation; partial scores use scored examples only. No combined score.')
    payload['official_references'] = {name: dict(value, source_url=REFERENCE_SOURCE,
        retrieval_date='2026-09-08', comparison_qualification='Difference from published reference; not quantization loss.')
        for name, value in REFERENCES.items()}
    write_json(run / 'report.json', payload)
    lines = [f'# Local evaluation {run.name}', '', payload['scope'], '',
             f'Status: {status["status"]}; active seconds: {status.get("active_seconds", 0):.1f}; queue seconds: {status.get("queue_seconds", 0):.1f}', '',
             '| Model | Benchmark / metric | Score | Published ref. | Difference (pp) | Planned / attempted / scored | Coverage / stage | Seconds | Invalid | Truncated | Failures | Infra errors |',
             '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        score = '—' if r['score'] is None else f'{r["score"]:.4f}'
        ref = '—' if r['official_reference'] is None else f'{r["official_reference"]:.1f}%'
        diff = '—' if r['difference_from_published_reference'] is None else f'{r["difference_from_published_reference"]:+.1f}'
        lines.append(f'| {r["model"]} | {r["benchmark"]} / {r["metric"]} | {score} | {ref} | {diff} | {r["planned"]}/{r["attempted"]}/{r["scored"]} | {"partial" if r["partial"] else "complete"} / {r["status"]} | {r["duration_seconds"]:.1f} | {r["invalid"]} | {r["truncated"]} | {r["execution_failures"]} | {len(r["infrastructure_errors"])} |')
    if status.get('error'):
        lines += ['', 'Run error: ' + status['error']]
    if pairs:
        lines += ['', 'Paired changes (first model → second model):', '']
        for p in pairs:
            lines.append(f'- {p["benchmark"]}: {p["first"]} → {p["second"]}, {p["paired"]}/{p["planned"]} paired; {p["changes"]}')
    lines += ['', f'Published references: {REFERENCE_SOURCE}. Differences are percentage points from the published reference and are not quantization-loss measurements.',
              'IFEval and HumanEval+ truncation counts use vLLM finish reasons. BBH and MMLU-Pro use native answer-choice likelihood scoring.', '',
              f'Frozen configuration and identities: `{run / "frozen.json"}`',
              f'Raw records and code: `{run / "stages"}`', f'Supervisor log: `{run / "supervisor.log"}`', '']
    tmp = run / '.report.md.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.chmod(0o600)
    tmp.replace(run / 'report.md')
    return payload
