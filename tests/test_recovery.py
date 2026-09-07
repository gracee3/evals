import json
from pathlib import Path
import subprocess

import pytest

from benchlib.core import digest, read_json, write_json
from benchlib.host import cleanup, verify_prepared
from benchlib.supervisor import Halt, Supervisor
from benchlib.worker import record_path
from test_supervisor import fixture_run, patch_host


def test_cleanup_checks_exact_label(monkeypatch):
    import benchlib.host as host
    calls = []
    def command(args):
        calls.append(args)
        if args[:3] == ['docker', 'ps', '-aq']:
            return 'ours'
        if args[1] == 'inspect':
            return 'owner'
        return ''
    monkeypatch.setattr(host, 'command', command)
    cleanup('owner')
    assert calls[0][-1] == 'label=org.local-agent-evals.run=owner'
    assert calls[-1] == ['docker', 'rm', '-f', 'ours']
    monkeypatch.setattr(host, 'command', lambda args: 'ours' if args[1] == 'ps' else 'somebody-else')
    with pytest.raises(RuntimeError, match='ownership'):
        cleanup('owner')


def test_incompatible_preparation_rejected(tmp_path, monkeypatch):
    import benchlib.host as host
    config = {'models': ['a']}
    prep = dict(suite_digest=digest(config), source={'file': 'hash'}, host_models={'a': ['original']}, image='image')
    write_json(tmp_path / 'prepared.json', prep)
    monkeypatch.setattr(host, 'prepared_path', lambda c: tmp_path)
    monkeypatch.setattr(host, 'source_identity', lambda: {'file': 'hash'})
    monkeypatch.setattr(host, 'PROFILES', {'a': 'path'})
    monkeypatch.setattr(host, 'model_identity', lambda p: ['original'])
    monkeypatch.setattr(host, 'image_id', lambda i: i)
    assert verify_prepared(config) == prep
    monkeypatch.setattr(host, 'model_identity', lambda p: ['changed'])
    with pytest.raises(ValueError, match='checkpoint'):
        verify_prepared(config)
    monkeypatch.setattr(host, 'model_identity', lambda p: ['original'])
    monkeypatch.setattr(host, 'source_identity', lambda: {'file': 'changed'})
    with pytest.raises(ValueError, match='implementation'):
        verify_prepared(config)


def test_interrupted_grading_reuses_committed_generation_and_grade(tmp_path, monkeypatch):
    import benchlib.supervisor as mod
    frozen = fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    frozen['prepared']['image'] = 'image'
    items = [{'task': 'HumanEval/0', 'index': 0}, {'task': 'HumanEval/1', 'index': 1}]
    frozen['prepared']['selection']['humaneval_plus'] = items
    write_json(tmp_path / 'frozen.json', frozen)
    write_json(tmp_path / 'humaneval.json', {i['task']: {'task_id': i['task']} for i in items})
    monkeypatch.setattr(mod, 'prepared_path', lambda c: tmp_path)
    stage = tmp_path / 'stages/a/humaneval_plus'
    for item in items:
        write_json(record_path(stage, item, 'generation'), dict(item=item, solution='saved', truncated=False))
    calls = []
    def interrupted(args, log, **kwargs):
        calls.append(log)
        if len(calls) == 2:
            raise Halt('stop requested')
        value = read_json(log.parent / 'input/example.json')
        write_json(log.parent / 'output/result.json', dict(item=value['generation']['item'], score=1.0))
        return 0
    supervisor = Supervisor(tmp_path)
    monkeypatch.setattr(supervisor, 'execute', interrupted)
    with pytest.raises(Halt):
        supervisor.grade(stage)
    first = record_path(stage, items[0])
    before = first.read_bytes()
    def finish(args, log, **kwargs):
        calls.append(log)
        value = read_json(log.parent / 'input/example.json')
        write_json(log.parent / 'output/result.json', dict(item=value['generation']['item'], score=1.0))
        return 0
    monkeypatch.setattr(supervisor, 'execute', finish)
    supervisor.grade(stage)
    assert len(calls) == 3
    assert first.read_bytes() == before
    assert len(list((stage / 'result').glob('*.json'))) == 2


def test_active_budget_preserved_on_resume(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    status = read_json(tmp_path / 'status.json')
    status['active_seconds'] = 24 * 3600
    write_json(tmp_path / 'status.json', status)
    supervisor = Supervisor(tmp_path)
    supervisor.work()
    assert supervisor.state['status'] == 'halted'
    assert supervisor.state['active_seconds'] >= 24 * 3600
    assert 'budget' in supervisor.state['error']


def test_changed_checkpoint_while_queued_stops_before_stage(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    patch_host(monkeypatch, tmp_path)
    def changed(self):
        assert self.active
        raise Halt('checkpoint identity changed while queued')
    monkeypatch.setattr(Supervisor, 'verify_models', changed)
    monkeypatch.setattr(Supervisor, 'gpu_stage', lambda *args: pytest.fail('changed model must never launch'))
    supervisor = Supervisor(tmp_path)
    supervisor.work()
    assert supervisor.state['status'] == 'halted'
    assert 'checkpoint identity' in supervisor.state['error']
