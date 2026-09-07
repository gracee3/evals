"""Exercise the pinned harness's real transactional cache without GPU inference."""
import os
import uuid

import pytest

from benchlib.host import IMAGE, cleanup, command, container_args, mount


@pytest.mark.skipif(os.environ.get('BENCH_DOCKER_TESTS') != '1', reason='explicit Docker integration test')
def test_native_cache_crash_recovery_and_checkpoint_isolation(tmp_path):
    source = r'''
import os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from lm_eval.api.model import LM, CachingLM

class Fake(LM):
    def __init__(self, checkpoint, crash=False):
        super().__init__(); self.checkpoint=checkpoint; self.crash=crash; self.seen=[]
    def loglikelihood(self, requests): raise NotImplementedError
    def loglikelihood_rolling(self, requests): raise NotImplementedError
    def generate_until(self, requests):
        results=[]
        for req in requests:
            self.seen.append(req.args[0]); value=self.checkpoint+':'+req.args[0]
            self.cache_hook.add_partial('generate_until',req.args,value)
            self.cache_hook.dbdict.commit(blocking=True)
            if self.crash: os._exit(9)
            results.append(value)
        return results

requests=[SimpleNamespace(args=(s,{'do_sample':False})) for s in ('first','second')]
if len(sys.argv)>1:
    cache=CachingLM(Fake('a',True),'/cache/a.db'); cache.generate_until(requests)
else:
    child=subprocess.run([sys.executable,__file__,'crash'])
    assert child.returncode==9
    resumed=Fake('a'); cache=CachingLM(resumed,'/cache/a.db')
    actual=cache.generate_until(requests); cache.dbdict.close()
    assert resumed.seen==['second'], resumed.seen
    uninterrupted=Fake('a'); whole=CachingLM(uninterrupted,'/cache/whole.db')
    assert actual==whole.generate_until(requests); whole.dbdict.close()
    other=Fake('b'); isolated=CachingLM(other,'/cache/b.db')
    assert isolated.generate_until(requests)==['b:first','b:second']; isolated.dbdict.close()
    assert other.seen==['first','second']
    print('native transactional cache recovery and checkpoint isolation passed')
'''
    script = tmp_path / 'native.py'
    script.write_text(source)
    cache = tmp_path / 'cache'
    cache.mkdir()
    owner = 'cache-test-' + uuid.uuid4().hex
    args = container_args(owner, owner, IMAGE) + mount(script, '/native.py', True) + mount(cache, '/cache')
    try:
        output = command(args + ['--entrypoint', 'python', IMAGE, '/native.py'], timeout=120)
        assert 'native transactional cache recovery and checkpoint isolation passed' in output
    finally:
        cleanup(owner)
