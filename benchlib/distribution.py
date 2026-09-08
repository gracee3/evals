"""Frozen GPU replica groups with isolated worker caches and canonical records."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import subprocess
import time
import uuid

from benchlib.core import read_json, write_json
from benchlib.host import cleanup, command, mount, restore_permissions
from benchlib.worker import record_path


def allocate(config):
    inventory = command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    requested = config['distribution']['gpus']
    devices = inventory if requested == 'auto' else requested
    if not devices or any(d not in inventory for d in devices):
        raise ValueError('requested GPU UUIDs are unavailable')
    groups = {}
    for model in config['models']:
        tp = config['runtimes'][model]['tensor_parallel_size']
        if len(devices) < tp:
            raise ValueError(f'{model} requires at least {tp} GPUs')
        groups[model] = [devices[i:i + tp] for i in range(0, len(devices) - tp + 1, tp)]
        groups[model] = groups[model][:min(s['count'] for s in config['benchmarks'])]
    return groups


def shard(frozen, index, count):
    child = deepcopy(frozen)
    child['prepared']['selection'] = {
        name: items[index::count] for name, items in frozen['prepared']['selection'].items()}
    return child


def collect(root, target, model, selection):
    """Publish only each shard's assigned committed records; preserve existing records."""
    for name, items in selection.items():
        for kind in ('result', 'generation'):
            for item in items:
                source = record_path(root / 'stages' / model / name, item, kind)
                dest = record_path(target / 'stages' / model / name, item, kind)
                if source.exists() and not dest.exists():
                    value = read_json(source)
                    if value.get('item') != item:
                        raise ValueError('replica returned an unexpected item')
                    write_json(dest, value)


def execute(supervisor, args, log):
    from benchlib.supervisor import RuntimeFailure, runtime_error
    model = supervisor.current.split('/')[0]
    groups = supervisor.frozen['gpu_groups'][model]
    inventory = command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    if any(device not in inventory for group in groups for device in group):
        raise RuntimeFailure('frozen replica GPU is unavailable')
    workers = []
    processes = []
    with ExitStack() as stack:
        try:
            for index, group in enumerate(groups):
                root = supervisor.run / 'replicas' / model / str(index)
                root.mkdir(parents=True, exist_ok=True)
                frozen = shard(supervisor.frozen, index, len(groups))
                path = root / 'frozen.json'
                if path.exists() and read_json(path) != frozen:
                    raise ValueError('replica identity changed; incompatible resume')
                write_json(path, frozen)
                if not (root / 'cache').exists():
                    command(['cp', '-a', '--reflink=auto', supervisor.run / 'cache', root / 'cache'], timeout=600)
                workers.append((root, frozen['prepared']['selection']))
                child = list(args)
                child[child.index('--name') + 1] = 'bench-replica-' + uuid.uuid4().hex
                # Docker parses --gpus as CSV; quote a multi-device value internally.
                child[child.index('--gpus') + 1] = '"device=' + ','.join(group) + '"'
                work_mount = mount(supervisor.run, '/work')[1]
                child[child.index(work_mount)] = mount(root, '/work')[1]
                child_log = root / (log.name + '.log')
                output = stack.enter_context(open(child_log, 'a'))
                processes.append((subprocess.Popen(child, stdout=output, stderr=subprocess.STDOUT), child_log))
            while True:
                for root, selection in workers:
                    collect(root, supervisor.run, model, selection)
                supervisor.tick()
                codes = [process.poll() for process, _ in processes]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeFailure('replica container failed; see replicas runtime logs')
                if all(code is not None for code in codes):
                    break
                time.sleep(1)
            for _, child_log in processes:
                if runtime_error(child_log.read_text(errors='replace')[-200000:]):
                    raise RuntimeFailure('replica runtime log contains an inference error')
            # Existing stage reporting expects one wall duration per benchmark.
            for name in supervisor.frozen['prepared']['selection']:
                durations = [root / 'stages' / model / name / 'duration.json' for root, _ in workers]
                if all(p.exists() for p in durations):
                    write_json(supervisor.run / 'stages' / model / name / 'duration.json',
                               {'seconds': max(read_json(p)['seconds'] for p in durations),
                                'replicas': len(groups)})
            return 0
        finally:
            # All siblings remain owned by the watchdog's single run label.
            cleanup(supervisor.owner)
            for process, _ in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=15)
            restore_permissions(supervisor.run, supervisor.frozen['prepared']['image'])
            for root, selection in workers:
                collect(root, supervisor.run, model, selection)
            with open(log, 'a') as output:
                output.write(f'Replica logs: {supervisor.run / "replicas" / model}\n')
