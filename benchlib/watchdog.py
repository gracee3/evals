"""Hold inherited ownership/shared locks until a dead supervisor's containers are gone."""
import os
from pathlib import Path
import sys
import time

from benchlib.core import read_json, write_json
from benchlib.host import cleanup, restore_permissions
from benchlib.report import report


def main():
    run = Path(sys.argv[1])
    pipe_fd, shared_fd, ownership_fd = map(int, sys.argv[2:5])
    os.fstat(shared_fd)
    os.fstat(ownership_fd)
    message = b''
    while True:
        chunk = os.read(pipe_fd, 32)
        if not chunk:
            break
        message += chunk
    frozen = read_json(run / 'frozen.json')
    # Do not release the shared lock while Docker cannot confirm owned cleanup.
    while True:
        try:
            cleanup(frozen['owner'])
            break
        except Exception as exc:
            print(f'watchdog cleanup pending: {type(exc).__name__}', flush=True)
            time.sleep(2)
    if message != b'done':
        try:
            restore_permissions(run, frozen['prepared']['image'])
        except Exception as exc:
            print(f'watchdog permissions: {type(exc).__name__}', flush=True)
        state = read_json(run / 'status.json')
        delta = max(0, time.time() - state.get('updated_at', time.time()))
        state['active_seconds'] += delta
        for stage in state['stages'].values():
            if stage['status'] == 'running':
                stage['elapsed'] += delta
        state.update(status='crashed', error='supervisor exited unexpectedly; watchdog cleaned owned containers; explicit resume required', updated_at=time.time())
        write_json(run / 'status.json', state)
        report(run)
    os.close(shared_fd)
    os.close(ownership_fd)

if __name__ == '__main__':
    main()
