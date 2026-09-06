from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from benchlib.core import BENCHMARKS, PROFILES, ROOT, digest, read_json, suite, write_json
from benchlib.host import PROJECT, cleanup, command, prepare, prepared_path, source_identity, verify_prepared
from benchlib.report import report


def run_path(run_id):
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}', run_id):
        raise ValueError('invalid run ID')
    path = ROOT / 'runs' / run_id
    if not (path / 'frozen.json').is_file():
        raise ValueError(f'unknown run: {run_id}')
    return path


def live(path):
    with open(path / 'supervisor.lock', 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def launch(path, resume=False):
    lock = open(path / 'supervisor.lock', 'a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('supervisor already running') from None
        frozen = read_json(path / 'frozen.json')
        if digest({k: v for k, v in frozen.items() if k != 'identity'}) != frozen['identity']:
            raise ValueError('frozen configuration identity mismatch')
        if source_identity(path / 'code') != frozen['prepared']['source']:
            raise ValueError('saved implementation changed; incompatible resume')
        verify_prepared(frozen['suite'], frozen)
        if resume:
            state = read_json(path / 'status.json')
            if state['status'] == 'complete':
                raise ValueError('run is already complete')
            # Reserve one heartbeat for interrupted accounting; never grant extra budget on resume.
            if state['status'] in ('running', 'crashed'):
                state['active_seconds'] += 1
                for stage in state['stages'].values():
                    if stage['status'] == 'running':
                        stage['elapsed'] += 1
            state['resumes'] = state.get('resumes', 0) + 1
            state.pop('error', None)
            write_json(path / 'status.json', state)
        cleanup(frozen['owner'])
        (path / 'stop').unlink(missing_ok=True)
        env = dict(os.environ, PYTHONPATH=str(path / 'code'), PYTHONUNBUFFERED='1')
        with open(path / 'supervisor.log', 'a') as output:
            process = subprocess.Popen([sys.executable, '-m', 'benchlib.supervisor', str(path), str(lock.fileno())],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lock.fileno(),), cwd=path, env=env)
        write_json(path / 'launch.json', dict(pid=process.pid, launched_at=time.time()))
    finally:
        lock.close()
    print(f'Run ID: {path.name}\nLog: {path / "supervisor.log"}\nStatus: {path / "status.json"}')


def new_run(config):
    prep = verify_prepared(config)
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:12]
    path = ROOT / 'runs' / run_id
    path.mkdir(parents=True, mode=0o700)
    frozen = dict(suite=config, prepared=prep, owner=uuid.uuid4().hex,
                  created_at=time.time(), git_commit=command(['git', '-C', PROJECT, 'rev-parse', 'HEAD']))
    frozen['identity'] = digest(frozen)
    write_json(path / 'frozen.json', frozen)
    (path / 'code').mkdir(mode=0o700)
    shutil.copytree(PROJECT / 'benchlib', path / 'code/benchlib', ignore=shutil.ignore_patterns('__pycache__'))
    # Copy prepared caches; source remains read-only to every active worker.
    command(['cp', '-a', '--reflink=auto', prepared_path(config) / 'cache', path / 'cache'], timeout=600)
    write_json(path / 'status.json', dict(status='created', stages={}, active_seconds=0, queue_seconds=0, resumes=0))
    report(path)
    launch(path)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='bench', description='Local serial smoke evaluations with private, resumable evidence.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    for verb in ('plan', 'prepare', 'run'):
        sub.add_parser(verb).add_argument('suite')
    for verb in ('status', 'stop', 'resume', 'report'):
        sub.add_parser(verb).add_argument('run_id')
    args = parser.parse_args()
    try:
        if args.command == 'list':
            print(json.dumps(dict(benchmarks=BENCHMARKS, models=PROFILES), indent=2))
        elif args.command in ('plan', 'prepare', 'run'):
            config = suite(args.suite)
            if args.command == 'plan':
                print(json.dumps(dict(suite=config,
                    planned_examples=len(config['models']) * sum(s['count'] for s in config['benchmarks']),
                    maximum_stage_hours=len(config['models']) * sum(s['seconds'] for s in config['benchmarks']) / 3600,
                    estimate='Deadline ceiling, not a throughput prediction. Preparation and queue time are separate.',
                    preparation=str(prepared_path(config))), indent=2))
            elif args.command == 'prepare':
                print(prepare(config))
            else:
                new_run(config)
        else:
            path = run_path(args.run_id)
            if args.command == 'status':
                state = read_json(path / 'status.json')
                state['supervisor_alive'] = live(path)
                if not state['supervisor_alive'] and state['status'] in ('running', 'queued', 'created'):
                    state['status'] = 'interrupted; explicit resume required'
                print(json.dumps(state, indent=2))
            elif args.command == 'stop':
                (path / 'stop').touch(mode=0o600)
                if not live(path):
                    cleanup(read_json(path / 'frozen.json')['owner'])
                    state = read_json(path / 'status.json')
                    state['status'] = 'stopped'
                    write_json(path / 'status.json', state)
                    report(path)
                print(f'Stop requested; progress retained at {path}')
            elif args.command == 'resume':
                launch(path, resume=True)
            else:
                report(path)
                print(f'{path / "report.md"}\n{path / "report.json"}')
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'bench: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
