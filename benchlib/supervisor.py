from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from benchlib.core import LOCK, PROFILES, read_json, write_json
from benchlib.host import ResourceGuard, cleanup, command, container_args, gpu_idle, memory, mount, prepared_path, restore_permissions
from benchlib.report import report
from benchlib.worker import record_path


class Halt(Exception):
    pass


class Deadline(Exception):
    pass


class RuntimeFailure(Exception):
    pass


def transient(text):
    # Explicit allowlist. Wrong outputs, OOMs, exceptions, and timeouts are not correctness retries.
    return any(s in text.lower() for s in ('connection reset by peer', 'connection refused', 'temporarily unavailable', 'connection aborted'))


def runtime_error(text):
    lines = text.lower().splitlines()
    for i, line in enumerate(lines):
        if any(word in line for word in ('out of memory', 'cuda error:', 'truncating context', 'truncating input')):
            return True
        if 'traceback (most recent call last)' in line:
            # Known optional SM90 kernel import probe on SM86. Preserve it in the raw log.
            optional_deepgemm = (i > 0 and 'warning' in line and '[import_utils.py:' in line
                and 'module vllm.third_party.deep_gemm was found but failed to import' in lines[i - 1])
            if not optional_deepgemm:
                return True
    return False


class Supervisor:
    def __init__(self, run, lifetime_fd=None):
        self.run = Path(run)
        self.frozen = read_json(self.run / 'frozen.json')
        self.config = self.frozen['suite']
        self.state = read_json(self.run / 'status.json')
        self.active = False
        self.current = None
        self.last = time.monotonic()
        self.guard = None
        self.owner = self.frozen['owner']
        self.lock = None
        self.lifetime_fd = lifetime_fd
        self.watchdog = None
        self.watchdog_pipe = None

    def save(self):
        self.state['updated_at'] = time.time()
        write_json(self.run / 'status.json', self.state)

    def tick(self):
        now = time.monotonic()
        elapsed = now - self.last
        self.last = now
        field = 'active_seconds' if self.active else 'queue_seconds'
        self.state[field] = self.state.get(field, 0) + elapsed
        if self.active and self.current:
            stage = self.state['stages'][self.current]
            stage['elapsed'] += elapsed
            directory = self.run / 'stages' / self.current
            stage['completed'] = len(list((directory / 'result').glob('*.json')))
            stage['generated'] = len(list((directory / 'generation').glob('*.json')))
        self.save()
        if (self.run / 'stop').exists():
            raise Halt('stop requested')
        if self.active:
            available, swap = memory()
            if self.guard.check(available, swap):
                raise Halt('resource guard: available RAM below 8 GiB or swap growth above 32 GiB for ten seconds')
            if self.state['active_seconds'] >= self.config['budgets']['active_hours'] * 3600:
                raise Halt('overall active budget exhausted')
            if self.current:
                name = self.current.split('/')[1]
                limit = next(s['seconds'] for s in self.config['benchmarks'] if s['name'] == name)
                if self.state['stages'][self.current]['elapsed'] >= limit:
                    raise Deadline('stage deadline exhausted')
        elif self.state['queue_seconds'] >= self.config['budgets']['queue_hours'] * 3600:
            raise Halt('queue deadline exhausted')

    def acquire(self):
        self.state['status'] = 'queued'
        self.save()
        # Never replace/delete the shared inode. Opening an absent lock is allowed at its established path.
        self.lock = open(LOCK, 'a')
        while True:
            self.tick()
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(1)
                continue
            try:
                if gpu_idle():
                    break
            except Exception:
                fcntl.flock(self.lock, fcntl.LOCK_UN)
                raise
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            time.sleep(1)
        self.tick()
        self.active = True
        self.last = time.monotonic()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if self.state.get('swap_boot') != boot:
            self.state.update(swap_boot=boot, swap_baseline=memory()[1])
        self.guard = ResourceGuard(self.state['swap_baseline'])
        if self.lifetime_fd is not None:
            read_fd, self.watchdog_pipe = os.pipe()
            self.watchdog = subprocess.Popen([sys.executable, '-m', 'benchlib.watchdog', str(self.run),
                str(read_fd), str(self.lock.fileno()), str(self.lifetime_fd)],
                pass_fds=(read_fd, self.lock.fileno(), self.lifetime_fd))
            os.close(read_fd)
        self.state['status'] = 'running'
        self.save()

    def execute(self, args, log, *, grade_limit=None):
        start = time.monotonic()
        with open(log, 'a') as output:
            process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    self.tick()
                    if grade_limit and time.monotonic() - start >= grade_limit:
                        raise Deadline('generated-code container wall timeout')
                    time.sleep(1)
                self.tick()
                return process.returncode
            finally:
                # GPU teardown and failed Docker clients are handled by exact label, including nested server workers.
                cleanup(self.owner)
                if '--gpus' in args:
                    restore_permissions(self.run, self.frozen['prepared']['image'])
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=15)

    def gpu_stage(self, model, benchmark, stage):
        name = 'bench-' + uuid.uuid4().hex
        args = container_args(name, self.owner, self.frozen['prepared']['image'], gpu=True, code=self.run / 'code')
        args += mount(PROFILES[model], '/model', True) + mount(self.run, '/work')
        args += mount(prepared_path(self.config), '/prepared', True)
        args += ['--env', 'HF_HOME=/work/cache', '--env', 'HF_HUB_OFFLINE=1', '--env', 'HF_DATASETS_OFFLINE=1',
                 '--env', 'VLLM_CACHE_ROOT=/work/vllm-cache', '--env', 'HOME=/tmp',
                 '--entrypoint', '/opt/evalplus/bin/python' if benchmark == 'humaneval_plus' else 'python',
                 self.frozen['prepared']['image'], '-m', 'benchlib.worker',
                 'generate' if benchmark == 'humaneval_plus' else 'harness',
                 '--stage', '/work/stages/' + model + '/' + benchmark, '--benchmark', benchmark]
        rc = self.execute(args, stage / 'runtime.log')
        text = (stage / 'runtime.log').read_text(errors='replace')[-200000:]
        if rc:
            raise RuntimeFailure(f'container exit {rc}: {text[-5000:]}')
        if runtime_error(text):
            raise RuntimeFailure('runtime log contains OOM, exception, CUDA error, or input truncation')
        # Wait for teardown without touching anyone else's GPU work.
        for _ in range(30):
            self.tick()
            if gpu_idle():
                return
            time.sleep(1)
        raise Halt('GPUs remain occupied after owned-container cleanup')

    def grade(self, stage):
        problems = read_json(prepared_path(self.config) / 'humaneval.json')
        for item in self.frozen['prepared']['selection']['humaneval_plus']:
            destination = record_path(stage, item)
            if destination.exists():
                continue
            generation = read_json(record_path(stage, item, 'generation'))
            job = stage / 'grading' / destination.stem
            (job / 'input').mkdir(parents=True, exist_ok=True, mode=0o700)
            (job / 'output').mkdir(exist_ok=True, mode=0o700)
            write_json(job / 'input/example.json', dict(problem=problems[item['task']], generation=generation))
            name = 'bench-grade-' + uuid.uuid4().hex
            args = container_args(name, self.owner, self.frozen['prepared']['image'], code=self.run / 'code')
            args += mount(job / 'input', '/input', True) + mount(job / 'output', '/output')
            args += ['--entrypoint', '/opt/evalplus/bin/python', self.frozen['prepared']['image'], '-m', 'benchlib.worker', 'grade']
            try:
                rc = self.execute(args, job / 'grade.log', grade_limit=self.config['budgets']['grade_seconds'])
            except Deadline as e:
                if str(e) != 'generated-code container wall timeout':
                    raise
                write_json(destination, dict(item=item, score=0.0, execution_failure=True,
                    execution_error=str(e), truncated=generation['truncated']))
                continue
            result = job / 'output/result.json'
            if rc or not result.exists():
                raise RuntimeFailure(f'grading infrastructure exit {rc}; see {job / "grade.log"}')
            value = read_json(result)
            if value.get('item') != item or value.get('score') not in (0, 1):
                raise RuntimeFailure('invalid grading result')
            write_json(destination, value)

    def work(self):
        try:
            cleanup(self.owner)
            self.acquire()
            for model in self.config['models']:
                for benchmark in self.config['benchmarks']:
                    name = benchmark['name']
                    self.current = model + '/' + name
                    state = self.state['stages'].setdefault(self.current, dict(status='pending', elapsed=0, retries=0, errors=[]))
                    if state['status'] in ('complete', 'timeout', 'failed'):
                        continue
                    state['status'] = 'running'
                    self.save()
                    stage = self.run / 'stages' / self.current
                    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
                    while True:
                        try:
                            self.tick()
                            selected = self.frozen['prepared']['selection'][name]
                            kind = 'generation' if name == 'humaneval_plus' else 'result'
                            if any(not record_path(stage, i, kind).exists() for i in selected):
                                if not gpu_idle():
                                    raise Halt('GPU availability changed under shared lock')
                                self.gpu_stage(model, name, stage)
                            if name == 'humaneval_plus':
                                self.grade(stage)
                            state['status'] = 'complete'
                            break
                        except Deadline as e:
                            state['status'] = 'timeout'
                            state['errors'].append(str(e))
                            cleanup(self.owner)
                            break
                        except RuntimeFailure as e:
                            state['errors'].append(str(e))
                            if transient(str(e)) and state['retries'] < 1:
                                state['retries'] += 1
                                self.save()
                                continue
                            state['status'] = 'failed'
                            raise Halt('persistent runtime failure; see stage errors')
                    self.save()
                    report(self.run)
                    self.current = None
            self.state['status'] = 'complete' if all(s['status'] == 'complete' for s in self.state['stages'].values()) else 'partial'
        except Halt as e:
            self.state['status'] = 'stopped' if str(e) == 'stop requested' else 'halted'
            self.state['error'] = str(e)
        except Exception as e:
            self.state['status'] = 'crashed'
            self.state['error'] = f'{type(e).__name__}: {e}'
            import traceback
            traceback.print_exc()
        finally:
            try:
                cleanup(self.owner)
            finally:
                if self.watchdog_pipe is not None:
                    os.write(self.watchdog_pipe, b'done')
                    os.close(self.watchdog_pipe)
                    self.watchdog.wait()
                if self.active:
                    delta = time.monotonic() - self.last
                    self.state['active_seconds'] += delta
                    if self.current:
                        self.state['stages'][self.current]['elapsed'] += delta
                if self.lock:
                    self.lock.close()
                self.save()
                report(self.run)
                print(f'Status: {self.state["status"]}; reports: {self.run / "report.md"}, {self.run / "report.json"}', flush=True)


def main():
    os.umask(0o077)
    run = Path(sys.argv[1])
    inherited_lock = int(sys.argv[2])
    # Keep inherited flock descriptor alive for the complete supervisor lifetime.
    os.fstat(inherited_lock)
    def stop(signum, frame):
        (run / 'stop').touch(mode=0o600)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    Supervisor(run, inherited_lock).work()

if __name__ == '__main__':
    main()
