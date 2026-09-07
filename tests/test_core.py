from collections import Counter
import json
from pathlib import Path

import pytest

from benchlib.core import digest, sample, suite, write_json
from benchlib.host import ResourceGuard, container_args
from benchlib.supervisor import transient


def test_stratified_fixed_total():
    pools = {str(k): [{'task': str(k), 'index': i} for i in range(100)] for k in range(24)}
    a = sample(pools, 60, 42)
    assert a == sample(dict(reversed(list(pools.items()))), 60, 42)
    assert len(a) == len({digest(i) for i in a}) == 60
    assert set(Counter(i['task'] for i in a).values()) == {2, 3}
    assert a != sample(pools, 60, 43)
    assert len(sample({'small': [1], 'large': [2, 3, 4]}, 4, 42)) == 4


def test_defaults_and_limits(tmp_path):
    p = tmp_path / 'suite.yaml'
    p.write_text('version: 1\n')
    config = suite(p)
    assert sum(s['count'] for s in config['benchmarks']) == 384
    assert config['runtime']['concurrency'] == 1
    for text in ('models: []', 'models: [int8-v2, int8-v2]', 'budgets: {active_hours: 49}',
                 'benchmarks: [{name: bbh, count: 6000}]', 'runtime: {enable_thinking: true}',
                 'benchmarks: [{name: ifeval, count: true}]', 'benchmarks: [{name: bbh, tokens: 12}]',
                 'bogus: 1', 'benchmarks: [{name: bbh}, {name: bbh}]'):
        p.write_text(text)
        with pytest.raises(ValueError):
            suite(p)


def test_int4_single_gpu_config(tmp_path):
    from benchlib.worker import runtime_args
    p = tmp_path / 'suite.yaml'
    p.write_text('models: [int4-v1]\n')
    config = suite(p)
    assert config['runtime']['tensor_parallel_size'] == 1
    assert runtime_args(config)['tensor_parallel_size'] == 1
    assert 'gpu_device' not in runtime_args(config)
    args = container_args('test', 'owner', 'image', gpu=True,
        gpu_device=config['runtime']['gpu_device'])
    assert args[args.index('--gpus') + 1] == 'device=GPU-613c7d78-a76d-306b-05da-1db1f15a5032'
    assert config['runtime']['max_model_len'] == 98304
    assert config['runtime']['kv_cache_dtype'] == 'fp8'
    assert config['runtime']['kv_cache_memory_bytes'] == 3758096384
    p.write_text('models: [int8-v2]\n')
    assert suite(p)['runtime']['tensor_parallel_size'] == 2


def test_explicit_model_profiles_resolve_independently(tmp_path):
    from benchlib.worker import runtime_args
    p = tmp_path / 'suite.yaml'
    p.write_text('''
models: [int8-v2, int4-v1]
runtime_profiles:
  int8-v2: int8-v2-262k-fp8-tp2
  int4-v1: int4-v1-96k-fp8-tp1
budgets:
  active_hours: 8
  model_active_hours: {int8-v2: 4, int4-v1: 4}
''')
    config = suite(p)
    assert config['runtimes']['int8-v2']['max_model_len'] == 262144
    assert config['runtimes']['int8-v2']['tensor_parallel_size'] == 2
    assert config['runtimes']['int8-v2']['kv_cache_dtype'] == 'fp8'
    assert config['runtimes']['int4-v1']['max_model_len'] == 98304
    assert config['runtimes']['int4-v1']['tensor_parallel_size'] == 1
    assert config['runtimes']['int4-v1']['gpu_device'].startswith('GPU-')
    assert runtime_args(config, 'int8-v2')['max_model_len'] == 262144
    assert runtime_args(config, 'int4-v1')['tensor_parallel_size'] == 1
    assert config['budgets']['model_active_hours'] == {'int8-v2': 4, 'int4-v1': 4}


def test_int4_tp2_262k_profile(tmp_path):
    p = tmp_path / 'suite.yaml'
    p.write_text('''
models: [int4-v1]
runtime_profiles:
  int4-v1: int4-v1-262k-fp8-tp2
''')
    config = suite(p)
    runtime = config['runtime']
    assert runtime['tensor_parallel_size'] == 2
    assert runtime['max_model_len'] == 262144
    assert runtime['kv_cache_dtype'] == 'fp8'
    assert runtime['kv_cache_memory_bytes'] == 4831838208


def test_resource_guard_sustained():
    now = [0]
    guard = ResourceGuard(2 * 1024**3, lambda: now[0])
    assert not guard.check(7 * 1024**3, 2 * 1024**3)
    now[0] = 9
    assert not guard.check(7 * 1024**3, 2 * 1024**3)
    now[0] = 10
    assert guard.check(7 * 1024**3, 2 * 1024**3)
    assert not guard.check(9 * 1024**3, 2 * 1024**3)
    now[0] = 20
    assert not guard.check(9 * 1024**3, 35 * 1024**3)
    now[0] = 30
    assert guard.check(9 * 1024**3, 35 * 1024**3)


def test_cpu_isolation_flags():
    args = container_args('owned', 'abc', 'sha256:abc')
    for flag, value in [('--network', 'none'), ('--cap-drop', 'ALL'), ('--pids-limit', '128'),
                        ('--memory', '4g'), ('--memory-swap', '4g')]:
        assert args[args.index(flag) + 1] == value
    assert '--read-only' in args and '--gpus' not in args
    assert 'NVIDIA_VISIBLE_DEVICES=void' in args
    assert not any('docker.sock' in a or '/home/emmy/.ssh' in a or 'dst=/model' in a for a in args)


def test_transient_allowlist():
    assert transient('Connection reset by peer')
    assert not transient('CUDA out of memory')
    assert not transient('wrong answer')
    assert not transient('timeout')


def test_atomic_commit(tmp_path, monkeypatch):
    path = tmp_path / 'record.json'
    write_json(path, {'first': 1})
    import benchlib.core as core
    real = core.os.replace
    def fail(*args):
        raise OSError('injected interruption before commit')
    monkeypatch.setattr(core.os, 'replace', fail)
    with pytest.raises(OSError):
        write_json(path, {'second': 2})
    assert json.loads(path.read_text()) == {'first': 1}
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(core.os, 'replace', real)
    write_json(path, {'second': 2})
    assert path.stat().st_mode & 0o777 == 0o600


def test_runtime_diagnostic_classification():
    from benchlib.supervisor import runtime_error
    known = ('WARNING [import_utils.py:408] Module vllm.third_party.deep_gemm was found but failed to import\n'
             'WARNING [import_utils.py:408] Traceback (most recent call last):\nAssertionError')
    assert not runtime_error(known)
    assert runtime_error(known + '\nTraceback (most recent call last):\nRuntimeError')
    assert runtime_error('WARNING CUDA out of memory')
    assert runtime_error('Truncating context.')


def test_interleaved_optional_import_warnings():
    from benchlib.supervisor import runtime_error
    prefix = 'WARNING [import_utils.py:408] '
    text = prefix + 'Module vllm.third_party.deep_gemm was found but failed to import\n'
    text += prefix + 'Traceback (most recent call last):\n'
    text += prefix + '  File optional.py\n' * 15
    text += prefix + 'Traceback (most recent call last):\n'
    assert not runtime_error(text)
    assert runtime_error(text + '\nTraceback (most recent call last):')
    assert runtime_error(text + '\n' + prefix + 'Module unrelated was found but failed to import\n' + prefix + 'Traceback (most recent call last):')
