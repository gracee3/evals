from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

ROOT = Path('/data/local-agent-evals')
LOCK = Path('/data/qwen38-int8-lab/quant-swappiness.lock')
PROFILES = {
    'int8-v2': '/data/models/Qwen3.8-27B-W8A8-INT8-Agentic-v2',
    'int8-original': '/data/models/Qwen3.8-27B-W8A8-INT8',
}
BENCHMARKS = {
    'ifeval': {'count': 100, 'maximum': 541, 'minutes': 90, 'tokens': 1024, 'task': 'leaderboard_ifeval', 'metric': 'prompt_level_strict_acc'},
    'humaneval_plus': {'count': 164, 'maximum': 164, 'minutes': 150, 'tokens': 2048, 'task': None, 'metric': 'pass@1'},
    'bbh': {'count': 60, 'maximum': 5761, 'minutes': 90, 'tokens': None, 'task': 'leaderboard_bbh', 'metric': 'acc_norm'},
    'mmlu_pro': {'count': 60, 'maximum': 12032, 'minutes': 90, 'tokens': None, 'task': 'leaderboard_mmlu_pro', 'metric': 'acc'},
}
PINS = {
    'wis-k/instruction-following-eval': '5a5661c2a35488308556cf4453dc074d1eba91a0',
    'SaylorTwift/bbh': 'b5306be6f827cfafbb545ff5a51f96916029b0fd',
    'TIGER-Lab/MMLU-Pro': 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
}
RUNTIME = dict(tensor_parallel_size=2, max_model_len=16384, dtype='bfloat16',
    kv_cache_dtype='bfloat16', enable_thinking=False, speculative_decoding=False,
    concurrency=1, batch_size=4, enforce_eager=True, enable_prefix_caching=False,
    language_model_only=True, enable_chunked_prefill=True, max_num_batched_tokens=1024,
    kv_cache_memory_bytes=805306368, cpu_offload_gb=0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    """Durable rename; only a complete JSON record is ever visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix='.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, default=str)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if os.geteuid() == 0 and 'BENCH_UID' in os.environ:
            os.chown(tmp, int(os.environ['BENCH_UID']), int(os.environ['BENCH_GID']))
            os.chown(path.parent, int(os.environ['BENCH_UID']), int(os.environ['BENCH_GID']))
        os.replace(tmp, path)
        d = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer from 1 to {maximum}')
    return value


def suite(path):
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('suite must be a YAML mapping')
    unknown = set(raw) - {'version', 'models', 'benchmarks', 'seed', 'runtime', 'budgets'}
    if unknown:
        raise ValueError(f'unknown suite keys: {sorted(unknown)}')
    if raw.get('version', 1) != 1:
        raise ValueError('only suite version 1 is supported')
    models = raw.get('models', ['int8-v2'])
    if not isinstance(models, list) or not 1 <= len(models) <= 2 or any(m not in PROFILES for m in models) or len(set(models)) != len(models):
        raise ValueError('models must contain one or two distinct listed profiles')
    seed = raw.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    stages = raw.get('benchmarks', [{'name': k} for k in BENCHMARKS])
    if not isinstance(stages, list) or not stages:
        raise ValueError('benchmarks must be a nonempty ordered list')
    selected = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - {'name', 'count', 'minutes', 'tokens'} or stage.get('name') not in BENCHMARKS:
            raise ValueError(f'invalid benchmark: {stage!r}')
        name = stage['name']
        defaults = BENCHMARKS[name]
        row = {'name': name, 'count': positive(stage.get('count', defaults['count']), 'count', defaults['maximum']),
               'seconds': positive(stage.get('minutes', defaults['minutes']), 'minutes', 2880) * 60,
               'tokens': stage.get('tokens', defaults['tokens'])}
        if defaults['tokens'] is None and row['tokens'] is not None:
            raise ValueError(f'{name} uses native likelihood scoring, not output tokens')
        if row['tokens'] is not None:
            positive(row['tokens'], 'tokens', 8192)
        selected.append(row)
    if len({s['name'] for s in selected}) != len(selected):
        raise ValueError('duplicate benchmark stages are not supported')
    runtime = raw.get('runtime', {})
    if not isinstance(runtime, dict) or set(runtime) - {'max_model_len', 'batch_size', 'kv_cache_memory_bytes'}:
        raise ValueError('runtime supports max_model_len, batch_size, kv_cache_memory_bytes; other v1 settings are fixed')
    runtime = RUNTIME | runtime
    positive(runtime['batch_size'], 'batch_size', 16)
    positive(runtime['max_model_len'], 'max_model_len', 16384)
    positive(runtime['kv_cache_memory_bytes'], 'kv_cache_memory_bytes', 4 * 1024**3)
    if any(s['tokens'] and s['tokens'] >= runtime['max_model_len'] for s in selected):
        raise ValueError('output limits must be smaller than context length')
    budgets = raw.get('budgets', {})
    if not isinstance(budgets, dict) or set(budgets) - {'active_hours', 'queue_hours', 'grade_seconds'}:
        raise ValueError('unknown budgets key')
    budgets = dict(active_hours=24, queue_hours=24, grade_seconds=120) | budgets
    positive(budgets['active_hours'], 'active_hours', 48)
    positive(budgets['queue_hours'], 'queue_hours', 24)
    positive(budgets['grade_seconds'], 'grade_seconds', 600)
    return dict(version=1, models=models, benchmarks=selected, seed=seed, runtime=runtime, budgets=budgets)


def sample(categories, count, seed):
    """Seeded round robin across categories; count is suite-wide, never per category."""
    if count > sum(len(v) for v in categories.values()):
        raise ValueError('sample count exceeds dataset')
    rng = random.Random(seed)
    pools = {k: sorted(v, key=lambda x: json.dumps(x, sort_keys=True)) for k, v in sorted(categories.items())}
    for pool in pools.values():
        rng.shuffle(pool)
    keys = list(pools)
    rng.shuffle(keys)
    result = []
    while len(result) < count:
        for key in keys:
            if pools[key]:
                result.append(pools[key].pop())
                if len(result) == count:
                    break
    return result


def model_identity(path, full=False):
    path = Path(path)
    if not (path / 'config.json').is_file() or not list(path.glob('*.safetensors')):
        raise ValueError(f'checkpoint incomplete: {path}')
    records = []
    for p in sorted(path.iterdir()):
        if p.is_file():
            s = p.stat()
            row = dict(name=p.name, size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)
            if full or p.suffix != '.safetensors':
                row['sha256'] = file_hash(p)
            records.append(row)
    return records
