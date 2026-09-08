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
    'int4-v1': '/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1',
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

# Imported from qwen38-int4-full-run/inference/config/*.yaml.  Keep the
# conservative 16K runtime above as the backwards-compatible default; long
# context is opt-in because it has a materially different memory envelope.
RUNTIME_PROFILES = {
    'int8-v2-16k-bf16-tp2': RUNTIME | dict(
        tensor_parallel_size=2, max_model_len=16384, kv_cache_dtype='bfloat16',
        kv_cache_memory_bytes=805306368, enforce_eager=True,
        enable_prefix_caching=False, max_num_batched_tokens=1024),
    'int8-v2-262k-fp8-tp2': RUNTIME | dict(
        tensor_parallel_size=2, max_model_len=262144, kv_cache_dtype='fp8',
        # Native vllm.yaml starts at 3 GiB, but v2's 262K single-sequence
        # admission check requires 4.16 GiB; 4.5 GiB provides measured headroom.
        kv_cache_memory_bytes=4831838208, enforce_eager=False,
        enable_prefix_caching=True, max_num_batched_tokens=2048),
    'int4-v1-16k-fp8-tp1': RUNTIME | dict(
        tensor_parallel_size=1, gpu_device='GPU-613c7d78-a76d-306b-05da-1db1f15a5032',
        max_model_len=16384, kv_cache_dtype='fp8',
        kv_cache_memory_bytes=805306368, enforce_eager=True,
        enable_prefix_caching=False, max_num_batched_tokens=1024),
    'int4-v1-96k-fp8-tp1': RUNTIME | dict(
        tensor_parallel_size=1, gpu_device='GPU-613c7d78-a76d-306b-05da-1db1f15a5032',
        max_model_len=98304, kv_cache_dtype='fp8',
        kv_cache_memory_bytes=3758096384, enforce_eager=False,
        enable_prefix_caching=True, max_num_batched_tokens=2048),
    'int4-v1-96k-fp8-tp2': RUNTIME | dict(
        tensor_parallel_size=2, max_model_len=98304, kv_cache_dtype='fp8',
        kv_cache_memory_bytes=3758096384, enforce_eager=False,
        enable_prefix_caching=True, max_num_batched_tokens=2048),
    'int4-v1-262k-fp8-tp2': RUNTIME | dict(
        tensor_parallel_size=2, max_model_len=262144, kv_cache_dtype='fp8',
        # The native 96K profile reserves 3.5 GiB and measured 208K cache
        # tokens.  4.5 GiB is the initial 262K admission target; the native
        # smoke below remains the source of truth for the actual envelope.
        kv_cache_memory_bytes=4831838208, enforce_eager=False,
        enable_prefix_caching=True, max_num_batched_tokens=2048),
}


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
    unknown = set(raw) - {'version', 'models', 'benchmarks', 'seed', 'runtime', 'runtime_profiles', 'budgets', 'distribution'}
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
    requested_profiles = raw.get('runtime_profiles', {})
    if not isinstance(requested_profiles, dict) or set(requested_profiles) - set(models):
        raise ValueError('runtime_profiles must map selected model names to named profiles')
    runtimes = {}
    for model in models:
        if model in requested_profiles:
            profile = requested_profiles[model]
            if profile not in RUNTIME_PROFILES:
                raise ValueError(f'unknown runtime profile: {profile}')
            defaults = RUNTIME_PROFILES[profile]
        elif models == ['int4-v1']:
            defaults = RUNTIME_PROFILES['int4-v1-96k-fp8-tp1']
        else:
            defaults = RUNTIME
        resolved = defaults | runtime
        positive(resolved['batch_size'], 'batch_size', 16)
        positive(resolved['max_model_len'], 'max_model_len', 262144)
        positive(resolved['kv_cache_memory_bytes'], 'kv_cache_memory_bytes', 8 * 1024**3)
        if any(s['tokens'] and s['tokens'] >= resolved['max_model_len'] for s in selected):
            raise ValueError(f'output limits must be smaller than context length for {model}')
        runtimes[model] = resolved
    budgets = raw.get('budgets', {})
    if not isinstance(budgets, dict) or set(budgets) - {'active_hours', 'queue_hours', 'grade_seconds', 'model_active_hours'}:
        raise ValueError('unknown budgets key')
    budgets = dict(active_hours=24, queue_hours=24, grade_seconds=120) | budgets
    positive(budgets['active_hours'], 'active_hours', 48)
    positive(budgets['queue_hours'], 'queue_hours', 24)
    positive(budgets['grade_seconds'], 'grade_seconds', 600)
    model_hours = budgets.get('model_active_hours', {})
    if not isinstance(model_hours, dict) or set(model_hours) - set(models):
        raise ValueError('model_active_hours must map selected model names to hour limits')
    budgets['model_active_hours'] = {m: model_hours.get(m, budgets['active_hours']) for m in models}
    for model, hours in budgets['model_active_hours'].items():
        positive(hours, f'model_active_hours[{model}]', 48)
    # Retain runtime for older callers; new workers use runtimes[model].
    result = dict(version=1, models=models, benchmarks=selected, seed=seed,
                runtime=runtimes[models[0]], runtimes=runtimes,
                runtime_profiles=requested_profiles, budgets=budgets)
    if 'distribution' in raw:
        distribution = raw['distribution']
        if not isinstance(distribution, dict) or set(distribution) != {'gpus'}:
            raise ValueError('distribution requires gpus: auto or a list of GPU UUIDs')
        devices = distribution['gpus']
        if devices != 'auto' and (not isinstance(devices, list) or not devices
                or any(not isinstance(d, str) or not d.startswith('GPU-') or ',' in d for d in devices)
                or len(set(devices)) != len(devices)):
            raise ValueError('distribution.gpus must be auto or distinct GPU UUIDs')
        result['distribution'] = distribution
    return result


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
