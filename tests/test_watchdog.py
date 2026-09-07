"""Optional real Docker recovery exercise: BENCH_DOCKER_TESTS=1 pytest tests/test_watchdog.py."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from benchlib.core import read_json, write_json
from benchlib.host import IMAGE, cleanup, command, image_id


@pytest.mark.skipif(os.environ.get('BENCH_DOCKER_TESTS') != '1', reason='explicit Docker integration test')
def test_sigkill_watchdog_keeps_lock_until_owned_cleanup(tmp_path):
    owner = 'watchdog-test-' + uuid.uuid4().hex
    write_json(tmp_path / 'frozen.json', dict(owner=owner, prepared={'image': image_id(IMAGE), 'selection': {}}, suite={'models': [], 'benchmarks': []}))
    write_json(tmp_path / 'status.json', dict(status='running', updated_at=time.time(), active_seconds=0, queue_seconds=0, stages={}))
    helper = r'''
import fcntl,os,subprocess,sys
from pathlib import Path
from benchlib.host import container_args,command,IMAGE
run=Path(sys.argv[1]); owner=sys.argv[2]
shared=open(run/'shared.lock','a'); owned=open(run/'supervisor.lock','a')
fcntl.flock(shared,fcntl.LOCK_EX); fcntl.flock(owned,fcntl.LOCK_EX)
args=container_args(owner,owner,IMAGE)
args.insert(2,'-d')
command(args+['--entrypoint','python',IMAGE,'-c','import time;time.sleep(120)'])
r,w=os.pipe()
subprocess.Popen([sys.executable,'-m','benchlib.watchdog',str(run),str(r),str(shared.fileno()),str(owned.fileno())],pass_fds=(r,shared.fileno(),owned.fileno()))
os.close(r)
print('ready',flush=True)
import time;time.sleep(120)
'''
    parent = subprocess.Popen([sys.executable, '-c', helper, str(tmp_path), owner], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert parent.stdout.readline().strip() == 'ready'
        with open(tmp_path / 'shared.lock', 'a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent.kill()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.1)
            else:
                pytest.fail('watchdog failed to clean and release shared lock')
        assert not command(['docker', 'ps', '-aq', '--filter', f'label=org.local-agent-evals.run={owner}'])
        assert read_json(tmp_path / 'status.json')['status'] == 'crashed'
        assert (tmp_path / 'report.md').exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=30)
        cleanup(owner)
