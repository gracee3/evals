from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from benchlib.core import LOCK, PROFILES, ROOT, digest, file_hash, model_identity, read_json, write_json

PROJECT = Path(__file__).resolve().parent.parent
LABEL = 'org.local-agent-evals.run'
BASE_IMAGE = 'sha256:c314ecf8b7e2b5d0067d6703f5bf55a939d35965e0da33efb91a4ed7ee2e0804'
IMAGE = 'local-agent-evals/runtime:0.1.0'


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True, capture_output=True, timeout=kwargs.pop('timeout', 60), **kwargs).stdout.strip()


def source_identity(directory=PROJECT):
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted((directory / 'benchlib').glob('*.py'))}


def image_id(image):
    return command(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])


def mount(source, dest, readonly=False):
    if ',' in str(source):
        raise ValueError('commas are unsupported in mount paths')
    return ['--mount', f'type=bind,src={Path(source).resolve()},dst={dest}' + (',readonly' if readonly else '')]


def container_args(name, owner, image, *, gpu=False, code=PROJECT):
    args = ['docker', 'run', '--name', name, '--label', f'{LABEL}={owner}',
        '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--init', '--env', 'PYTHONPATH=/app', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'HF_HUB_DISABLE_TELEMETRY=1', '--env', 'TOKENIZERS_PARALLELISM=false',
        '--env', f'BENCH_UID={os.getuid()}', '--env', f'BENCH_GID={os.getgid()}']
    args += mount(code / 'benchlib', '/app/benchlib', True)
    if gpu:
        args += ['--cap-add', 'CHOWN', '--cap-add', 'DAC_OVERRIDE', '--gpus', 'all', '--shm-size', '2g', '--pids-limit', '1024',
                 '--env', 'VLLM_USE_FLASHINFER_SAMPLER=0']
    else:
        args += ['--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'CUDA_VISIBLE_DEVICES=',
                 '--cpus', '2', '--memory', '4g', '--memory-swap', '4g', '--pids-limit', '128',
                 '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
                 '--tmpfs', '/tmp:rw,nosuid,nodev,size=1073741824,mode=1777',
                 '--env', 'HOME=/tmp', '--env', 'XDG_CACHE_HOME=/tmp/cache']
    return args


def cleanup(owner):
    """Only select containers with our exact unguessable run label; never use host PIDs."""
    ids = command(['docker', 'ps', '-aq', '--filter', f'label={LABEL}={owner}']).split()
    for cid in ids:
        label = command(['docker', 'inspect', cid, '--format', '{{index .Config.Labels "' + LABEL + '"}}'])
        if label != owner:
            raise RuntimeError('container ownership changed')
        command(['docker', 'rm', '-f', cid])


def gpu_idle():
    apps = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    memory = command(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    used = [int(v.strip()) for v in memory.splitlines()]
    return not apps and len(used) == 2 and max(used) < 512


def memory():
    values = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable'] * 1024, (values['SwapTotal'] - values['SwapFree']) * 1024


class ResourceGuard:
    def __init__(self, swap, clock=time.monotonic):
        self.swap = swap
        self.clock = clock
        self.since = None

    def check(self, available, swap):
        bad = available < 8 * 1024**3 or swap - self.swap > 32 * 1024**3
        now = self.clock()
        self.since = (now if self.since is None else self.since) if bad else None
        return self.since is not None and now - self.since >= 10


def prepared_path(config):
    return ROOT / 'prepared' / digest(config)


def prepare(config):
    os.umask(0o077)
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = prepared_path(config)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The preparation lock also prevents two preparations mutating the same cache.
    with open(path / 'prepare.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pins = read_json(PROJECT / 'pins.json')
        if image_id('qwen38-int8-lab/eval:0.1.0') != BASE_IMAGE:
            raise ValueError('base image differs; restore the recorded lm-eval image')
        try:
            actual = image_id(IMAGE)
        except subprocess.CalledProcessError:
            command(['docker', 'build', '-f', PROJECT / 'docker/Dockerfile', '-t', IMAGE, PROJECT], timeout=1800)
            actual = image_id(IMAGE)
        if pins.get('runtime_image') and actual != pins['runtime_image']:
            raise ValueError('runtime image differs from pins.json; build the pinned Dockerfile')
        if (path / 'prepared.json').exists():
            try:
                verify_prepared(config)
                return path
            except (ValueError, subprocess.CalledProcessError):
                (path / 'prepared.json').rename(path / ('previous-' + uuid.uuid4().hex + '.json'))
        write_json(path / 'suite.json', config)
        (path / 'cache').mkdir(exist_ok=True, mode=0o700)
        # Explicit opt-in cache import. Copy-on-write when supported; never hardlink or write original.
        cache = os.environ.get('BENCH_IMPORT_HF_CACHE')
        if cache:
            command(['cp', '-a', '--reflink=auto', str(Path(cache).resolve()) + '/.', str(path / 'cache')], timeout=300)
        for m in config['models']:
            model_identity(PROFILES[m])
        owner = 'prepare-' + uuid.uuid4().hex
        args = container_args(owner, owner, actual)
        # Preparation needs network for pinned public datasets, more RAM for imports, and model read access.
        for key, val in [('--network', 'bridge'), ('--memory', '16g'), ('--memory-swap', '16g'), ('--user', '0:0')]:
            args[args.index(key) + 1] = val
        args += ['--cap-add', 'CHOWN', '--cap-add', 'DAC_OVERRIDE']
        args += mount(path, '/work')
        for m in config['models']:
            args += mount(PROFILES[m], '/models/' + m, True)
        args += ['--env', 'HF_HOME=/work/cache', '--env', 'XDG_CACHE_HOME=/work/cache',
                 '--entrypoint', 'python', actual, '-m', 'benchlib.worker', 'prepare']
        try:
            with open(path / 'prepare.log', 'a') as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            prep = read_json(path / 'prepared.json')
            prep.update(image=actual, source=source_identity(), suite_digest=digest(config),
                        host_models={m: model_identity(PROFILES[m]) for m in config['models']})
            write_json(path / 'prepared.json', prep)
        finally:
            cleanup(owner)
            restore_permissions(path, actual)
    return path


def verify_prepared(config, frozen=None):
    path = prepared_path(config)
    if not (path / 'prepared.json').exists():
        raise ValueError('missing preparation; run bench prepare with this suite first')
    prep = read_json(path / 'prepared.json')
    if prep.get('suite_digest') != digest(config) or prep.get('source') != source_identity():
        raise ValueError('suite or implementation changed; prepare this suite again after removing its stale prepared directory')
    if prep['host_models'] != {m: model_identity(PROFILES[m]) for m in config['models']}:
        raise ValueError('checkpoint identity changed; prepare a new run')
    if image_id(prep['image']) != prep['image']:
        raise ValueError('prepared runtime image unavailable')
    if frozen is not None and prep != frozen['prepared']:
        raise ValueError('prepared identity changed; incompatible resume')
    return prep


def restore_permissions(path, image):
    owner = 'permissions-' + uuid.uuid4().hex
    args = ['docker', 'run', '--rm', '--label', f'{LABEL}={owner}', '--network', 'none',
            '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--cap-drop', 'ALL', '--cap-add', 'CHOWN',
            '--cap-add', 'DAC_OVERRIDE', '--cap-add', 'FOWNER', '--security-opt', 'no-new-privileges']
    args += mount(path, '/owned')
    script = '''import os
for root, dirs, files in os.walk('/owned'):
    for p in [root] + [os.path.join(root, f) for f in files]:
        if not os.path.islink(p):
            os.chown(p, int(os.environ['U']), int(os.environ['G']))
            os.chmod(p, 0o700 if os.path.isdir(p) else 0o600)
'''
    command(args + ['--env', f'U={os.getuid()}', '--env', f'G={os.getgid()}', '--entrypoint', 'python', image, '-c', script], timeout=300)
