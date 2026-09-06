import fcntl
import json
from pathlib import Path

import pytest

from benchlib.core import write_json
from benchlib.supervisor import Halt, RuntimeFailure, Supervisor
from benchlib.worker import record_path
from benchlib.report import report


def fixture_run(tmp_path):
    config = dict(models=['a', 'b'], benchmarks=[dict(name='ifeval', count=2, seconds=100, tokens=1024),
        dict(name='bbh', count=2, seconds=100, tokens=None)], budgets=dict(active_hours=24, queue_hours=24, grade_seconds=1))
    selected = {s['name']: [{'task': s['name'], 'index': i} for i in range(2)] for s in config['benchmarks']}
    frozen = dict(suite=config, owner='owned-test', prepared=dict(selection=selected))
    write_json(tmp_path / 'frozen.json', frozen)
    write_json(tmp_path / 'status.json', dict(status='created', active_seconds=0, queue_seconds=0, stages={}))
    return frozen


def patch_host(monkeypatch, tmp_path):
    import benchlib.supervisor as mod
    monkeypatch.setattr(mod, 'cleanup', lambda owner: None)
    monkeypatch.setattr(mod, 'gpu_idle', lambda: True)
    monkeypatch.setattr(mod, 'memory', lambda: (16 * 1024**3, 0))
    monkeypatch.setattr(mod, 'LOCK', tmp_path / 'shared.lock')


def test_serial_resume_no_duplicate_scores(tmp_path, monkeypatch):
    frozen = fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    calls = []
    interrupted = [False]
    def stage(self, model, benchmark, directory):
        calls.append((model, benchmark))
        for item in frozen['prepared']['selection'][benchmark]:
            destination = record_path(directory, item)
            if not destination.exists():
                write_json(destination, dict(item=item, score=float(item['index'] == 0)))
                if not interrupted[0]:
                    interrupted[0] = True
                    raise Halt('stop requested')
    monkeypatch.setattr(Supervisor, 'gpu_stage', stage)
    first = Supervisor(tmp_path)
    first.work()
    committed = record_path(tmp_path / 'stages/a/ifeval', frozen['prepared']['selection']['ifeval'][0])
    before = committed.stat().st_mtime_ns
    assert first.state['status'] == 'stopped'
    second = Supervisor(tmp_path)
    second.work()
    assert second.state['status'] == 'complete'
    assert committed.stat().st_mtime_ns == before
    assert calls == [('a', 'ifeval'), ('a', 'ifeval'), ('a', 'bbh'), ('b', 'ifeval'), ('b', 'bbh')]
    result = report(tmp_path)
    assert all(r['score'] == .5 and not r['partial'] for r in result['results'])
    assert all(p['paired'] == 2 for p in result['paired'])


def test_one_transient_retry_persistent_failure_stops(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    calls = []
    def stage(*args):
        calls.append(1)
        raise RuntimeFailure('connection reset by peer')
    monkeypatch.setattr(Supervisor, 'gpu_stage', stage)
    supervisor = Supervisor(tmp_path)
    supervisor.work()
    assert len(calls) == 2
    assert supervisor.state['status'] == 'halted'
    assert supervisor.state['stages']['a/ifeval']['retries'] == 1
    assert all(r['partial'] for r in report(tmp_path)['results'])


def test_shared_lock_queue_deadline(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    supervisor = Supervisor(tmp_path)
    supervisor.config['budgets']['queue_hours'] = .00001
    with open(tmp_path / 'shared.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(Halt, match='queue deadline'):
            supervisor.acquire()
        supervisor.lock.close()
    assert not supervisor.active
    assert supervisor.state['active_seconds'] == 0


def test_timeout_continues_partial_reports(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    from benchlib.supervisor import Deadline
    def stage(*args):
        raise Deadline('stage deadline exhausted')
    monkeypatch.setattr(Supervisor, 'gpu_stage', stage)
    supervisor = Supervisor(tmp_path)
    supervisor.work()
    assert supervisor.state['status'] == 'partial'
    assert len(supervisor.state['stages']) == 4
    assert all(v['status'] == 'timeout' for v in supervisor.state['stages'].values())
    assert (tmp_path / 'report.md').is_file()


def test_cache_paths_isolate_checkpoints(tmp_path):
    fixture_run(tmp_path)
    item = dict(task='ifeval', index=1)
    assert record_path(tmp_path / 'stages/a/ifeval', item) != record_path(tmp_path / 'stages/b/ifeval', item)
