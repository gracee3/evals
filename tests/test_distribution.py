import pytest

from benchlib import distribution as dist
from benchlib.core import read_json, write_json, suite
from benchlib.worker import record_path


def test_groups_respect_tp_and_inventory(monkeypatch):
    monkeypatch.setattr(dist, 'command', lambda args: 'GPU-a\nGPU-b\nGPU-c\nGPU-d')
    config = dict(models=['a', 'b'], runtimes={'a': {'tensor_parallel_size': 1},
                  'b': {'tensor_parallel_size': 2}}, distribution={'gpus': 'auto'},
                  benchmarks=[{'count': 10}])
    assert dist.allocate(config) == {'a': [['GPU-a'], ['GPU-b'], ['GPU-c'], ['GPU-d']],
                                    'b': [['GPU-a', 'GPU-b'], ['GPU-c', 'GPU-d']]}
    config['distribution']['gpus'] = ['GPU-d', 'GPU-b', 'GPU-a']
    assert dist.allocate(config)['b'] == [['GPU-d', 'GPU-b']]
    config['distribution']['gpus'] = ['GPU-missing']
    with pytest.raises(ValueError):
        dist.allocate(config)


def test_partition_and_collect_resume(tmp_path):
    items = [{'task': 'test', 'index': i} for i in range(7)]
    frozen = {'prepared': {'selection': {'ifeval': items}}}
    parts = [dist.shard(frozen, i, 3) for i in range(3)]
    assert sorted(i['index'] for p in parts for i in p['prepared']['selection']['ifeval']) == list(range(7))
    target = tmp_path / 'run'
    for n, part in enumerate(parts):
        root = tmp_path / str(n)
        for item in part['prepared']['selection']['ifeval']:
            write_json(record_path(root / 'stages/a/ifeval', item), {'item': item, 'score': 1})
        dist.collect(root, target, 'a', part['prepared']['selection'])
        saved = record_path(target / 'stages/a/ifeval', part['prepared']['selection']['ifeval'][0])
        stamp = saved.stat().st_mtime_ns
        dist.collect(root, target, 'a', part['prepared']['selection'])
        assert saved.stat().st_mtime_ns == stamp
    assert len(list((target / 'stages/a/ifeval/result').glob('*.json'))) == 7
    assert frozen['prepared']['selection']['ifeval'] == items


def test_distribution_config(tmp_path):
    p = tmp_path / 'suite.yaml'
    p.write_text('distribution: {gpus: auto}\n')
    assert suite(p)['distribution'] == {'gpus': 'auto'}
    p.write_text('distribution: {gpus: [GPU-a, GPU-a]}\n')
    with pytest.raises(ValueError):
        suite(p)


@pytest.mark.parametrize('fail', [False, True])
def test_sibling_completion_does_not_cleanup_early(tmp_path, monkeypatch, fail):
    from types import SimpleNamespace
    items = [{'task': 'test', 'index': i} for i in range(4)]
    frozen = {'gpu_groups': {'a': [['GPU-a'], ['GPU-b']]},
              'prepared': {'selection': {'ifeval': items}, 'image': 'image'}}
    supervisor = SimpleNamespace(run=tmp_path, frozen=frozen, current='a/ifeval', owner='owner')
    events = []
    supervisor.tick = lambda: events.append('tick')
    monkeypatch.setattr(dist, 'command', lambda args, **kw: 'GPU-a\nGPU-b')
    monkeypatch.setattr(dist, 'restore_permissions', lambda *args: None)
    monkeypatch.setattr(dist.time, 'sleep', lambda _: None)
    monkeypatch.setattr(dist, 'cleanup', lambda _: events.append('cleanup'))
    class Process:
        def __init__(self, args, **kw):
            events.append('start')
            self.polls = 0
            assert args[args.index('--gpus') + 1] in ('"device=GPU-a"', '"device=GPU-b"')
        def poll(self):
            self.polls += 1
            return (1 if fail else 0) if self.polls > 1 else None
        def wait(self, **kw):
            return 0
    monkeypatch.setattr(dist.subprocess, 'Popen', Process)
    args = ['docker', 'run', '--name', 'test', '--gpus', 'all'] + dist.mount(tmp_path, '/work')
    if fail:
        from benchlib.supervisor import RuntimeFailure
        with pytest.raises(RuntimeFailure, match='replica container failed'):
            dist.execute(supervisor, args, tmp_path / 'runtime.log')
    else:
        assert dist.execute(supervisor, args, tmp_path / 'runtime.log') == 0
    assert events[:2] == ['start', 'start']
    assert events[-1] == 'cleanup'
    for index in range(2):
        child = read_json(tmp_path / 'replicas/a' / str(index) / 'frozen.json')
        assert child['prepared']['selection']['ifeval'] == items[index::2]
