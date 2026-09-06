"""Container entry points. Never executes generated code in a GPU container."""
from __future__ import annotations

import argparse
from collections import defaultdict
from importlib.metadata import distributions, version
import os
from pathlib import Path
import subprocess
import sys
import time

from benchlib.core import BENCHMARKS, PINS, digest, file_hash, model_identity, read_json, sample, write_json


def pin_datasets(offline):
    import datasets
    original = datasets.load_dataset
    def load(*args, **kwargs):
        path = kwargs.get('path') or args[0]
        if path not in PINS or kwargs.get('revision') not in (None, PINS[path]):
            raise RuntimeError(f'unpinned dataset request: {path}')
        kwargs['revision'] = PINS[path]
        if offline:
            kwargs['download_mode'] = datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
        return original(*args, **kwargs)
    datasets.load_dataset = load


def prepare():
    import lm_eval
    from lm_eval.tasks import TaskManager
    config = read_json('/work/suite.json')
    for package, expected in [('lm_eval', '0.4.12'), ('vllm', '0.27.1')]:
        if version(package) != expected:
            raise RuntimeError(f'{package} version mismatch')
    pin_datasets(False)
    tm = TaskManager()
    selection = {}
    dataset_hashes = {}
    task_hashes = {}
    for stage in config['benchmarks']:
        name = stage['name']
        print('Preparing', name, flush=True)
        pools = defaultdict(list)
        if name == 'humaneval_plus':
            subprocess.run(['/opt/evalplus/bin/python', '-m', 'benchlib.worker', 'export_he'], check=True)
            problems = read_json('/work/humaneval.json')
            dataset_hashes[name] = file_hash('/work/humaneval.json')
            for task_id in problems:
                pools['all'].append({'task': task_id, 'index': int(task_id.split('/')[1])})
            write_json('/work/humaneval.json', problems)
        else:
            tasks = tm.load([BENCHMARKS[name]['task']])['tasks']
            for task_name, task in sorted(tasks.items()):
                docs = task.eval_docs
                for i, doc in enumerate(docs):
                    category = doc.get('category', task_name)
                    pools[category].append({'task': task_name, 'index': i})
                dataset_hashes[task_name] = digest(list(docs))
        selection[name] = sample(pools, stage['count'], config['seed'])
    task_root = Path(lm_eval.__file__).parent / 'tasks' / 'leaderboard'
    for p in sorted(task_root.rglob('*')):
        if p.is_file() and p.suffix != '.pyc':
            task_hashes[str(p.relative_to(task_root))] = file_hash(p)
    print('Hashing checkpoints', flush=True)
    identity = {m: model_identity('/models/' + m, full=True) for m in config['models']}
    write_json('/work/prepared.json', dict(selection=selection, models=identity, dataset_hashes=dataset_hashes,
        dataset_revisions=PINS, humaneval_version='v0.1.10', task_hashes=task_hashes,
        software={d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']},
        evalplus_software=read_json('/work/evalplus-software.json') if Path('/work/evalplus-software.json').exists() else {}))


def runtime_args(config):
    excluded = {'concurrency', 'batch_size', 'speculative_decoding'}
    return {k: v for k, v in config['runtime'].items() if k not in excluded} | dict(
        pretrained='/model', seed=config['seed'], add_bos_token=False, batch_size=1, max_num_seqs=1)


def record_path(stage, item, kind='result'):
    return Path(stage) / kind / (digest(item) + '.json')


def harness(stage_path, benchmark):
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    config = read_json('/work/frozen.json')['suite']
    frozen = read_json('/work/frozen.json')
    stage_config = next(s for s in config['benchmarks'] if s['name'] == benchmark)
    pin_datasets(True)
    selected = frozen['prepared']['selection'][benchmark]
    pending = [s for s in selected if not record_path(stage_path, s).exists()]
    if not pending:
        return
    lm = VLLM(**runtime_args(config))
    original_generate = lm._model_generate
    def capture(*args, **kwargs):
        outputs = original_generate(*args, **kwargs)
        if kwargs.get('generate', True):
            for output in outputs:
                write_json(Path(stage_path) / 'finish' / (digest(output.prompt_token_ids) + '.json'),
                    dict(finish_reason=output.outputs[0].finish_reason, tokens=len(output.outputs[0].token_ids)))
        return outputs
    lm._model_generate = capture
    original_likelihood = lm._loglikelihood_tokens
    def bounded_likelihood(requests, **kwargs):
        if any(len(ctx) + len(cont) >= lm.max_length for _, ctx, cont in requests):
            raise RuntimeError('input exceeds configured context; refusing truncation')
        return original_likelihood(requests, **kwargs)
    lm._loglikelihood_tokens = bounded_likelihood
    import lm_eval.models.vllm_causallms as backend
    def bounded_generation(tokens, max_gen_toks, max_model_len, **kwargs):
        if len(tokens) + max_gen_toks > max_model_len:
            raise RuntimeError('input plus output exceeds context; refusing truncation')
        return tokens, max_gen_toks
    backend.maybe_truncate = bounded_generation
    # Native transactional response cache survives interruption between generation and scoring.
    batch_size = config['runtime']['batch_size']
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        samples = defaultdict(list)
        for item in batch:
            samples[item['task']].append(item['index'])
        result = simple_evaluate(model=lm, tasks=list(samples), samples=dict(samples),
            batch_size=1, use_cache=str(Path(stage_path) / 'responses'),
            apply_chat_template=True, fewshot_as_multiturn=True, log_samples=True,
            gen_kwargs={'max_gen_toks': stage_config['tokens']} if benchmark == 'ifeval' else None,
            random_seed=config['seed'], numpy_random_seed=config['seed'],
            torch_random_seed=config['seed'], fewshot_random_seed=config['seed'], bootstrap_iters=0)
        lm.cache_hook.dbdict.close()
        write_json(Path(stage_path) / 'raw' / (digest(batch) + '.json'), result)
        for task, rows in result['samples'].items():
            for row in rows:
                item = {'task': task, 'index': row['doc_id']}
                if item not in batch:
                    raise RuntimeError('harness returned an unselected sample')
                truncated = None
                if benchmark == 'ifeval':
                    prompt_tokens = lm.tok_encode(row['arguments'][0][0])
                    finish = read_json(Path(stage_path) / 'finish' / (digest(prompt_tokens) + '.json'))
                    truncated = finish['finish_reason'] == 'length'
                write_json(record_path(stage_path, item), dict(item=item,
                    score=float(row[BENCHMARKS[benchmark]['metric']]), truncated=truncated,
                    truncation_measure='vLLM finish_reason' if benchmark == 'ifeval' else 'not applicable',
                    execution_failure=False, raw=row))
        if any(not record_path(stage_path, i).exists() for i in batch):
            raise RuntimeError('harness omitted selected results')


def humaneval_generate(stage_path):
    import openai
    from evalplus.provider.openai import OpenAIChatDecoder
    from evalplus.gen.util import openai_request
    from evalplus.sanitize import sanitize
    frozen = read_json('/work/frozen.json')
    config = frozen['suite']
    problems = read_json('/prepared/humaneval.json')
    tokens = next(s['tokens'] for s in config['benchmarks'] if s['name'] == 'humaneval_plus')
    selected = frozen['prepared']['selection']['humaneval_plus']
    pending = [i for i in selected if not record_path(stage_path, i, 'generation').exists()]
    if not pending:
        return
    r = config['runtime']
    command = ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model',
        '--served-model-name', 'bench', '--host', '127.0.0.1', '--port', '8000',
        '--tensor-parallel-size', '2', '--max-model-len', str(r['max_model_len']),
        '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16', '--enforce-eager',
        '--no-enable-prefix-caching', '--language-model-only', '--enable-chunked-prefill',
        '--max-num-batched-tokens', str(r['max_num_batched_tokens']),
        '--max-num-seqs', '1', '--kv-cache-memory-bytes', str(r['kv_cache_memory_bytes']),
        '--seed', str(config['seed']), '--generation-config', 'vllm',
        '--default-chat-template-kwargs', '{"enable_thinking":false}']
    server = subprocess.Popen(command)
    try:
        client = openai.OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='local-unused', max_retries=0, timeout=300)
        ready = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError('owned vLLM server exited during startup')
            try:
                client.models.list()
                break
            except openai.APIConnectionError:
                if time.monotonic() >= ready:
                    raise RuntimeError('owned vLLM server startup timeout')
                time.sleep(2)
        decoder = OpenAIChatDecoder('bench', base_url='http://127.0.0.1:8000/v1', batch_size=1,
            temperature=0.0, max_new_tokens=tokens,
            instruction_prefix='Please provide a self-contained Python script that solves the following problem in a markdown code block:')
        decoder.client = client
        captured = []
        def once(*args, **kwargs):
            # Preserve upstream messages and request parameters; retries belong to supervisor.
            ret = openai_request.make_request(*args, **kwargs)
            captured[:] = [ret]
            return ret
        openai_request.make_auto_request = once
        for item in pending:
            problem = problems[item['task']]
            text = decoder.codegen(problem['prompt'].strip() + '\n', do_sample=False, num_samples=1)[0]
            write_json(record_path(stage_path, item, 'generation'), dict(item=item, raw=text,
                solution=sanitize(text, entrypoint=problem['entry_point']),
                truncated=captured[0].choices[0].finish_reason == 'length',
                usage=captured[0].usage.model_dump()))
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def grade():
    """Called only in an isolated CPU container with a single input and output directory."""
    from evalplus.evaluate import get_groundtruth, check_correctness
    value = read_json('/input/example.json')
    problem = value['problem']
    expected = get_groundtruth({problem['task_id']: problem}, digest(problem), [])
    result = check_correctness('humaneval', 0, problem, value['generation']['solution'],
        expected[problem['task_id']], fast_check=False)
    base, plus = result['base'][0], result['plus'][0]
    write_json('/output/result.json', dict(item=value['generation']['item'], score=float(base == 'pass' and plus == 'pass'),
        base_status=base, plus_status=plus, execution_failure=base != 'pass' or plus != 'pass',
        truncated=value['generation']['truncated']))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'harness', 'generate', 'grade', 'export_he'])
    parser.add_argument('--stage')
    parser.add_argument('--benchmark')
    args = parser.parse_args()
    if args.action == 'export_he':
        from evalplus.data import get_human_eval_plus
        if version('evalplus') != '0.3.1':
            raise RuntimeError('EvalPlus version mismatch')
        write_json('/work/humaneval.json', get_human_eval_plus(version='v0.1.10'))
        write_json('/work/evalplus-software.json', {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']})
    elif args.action == 'prepare':
        prepare()
    elif args.action == 'harness':
        harness(args.stage, args.benchmark)
    elif args.action == 'generate':
        humaneval_generate(args.stage)
    else:
        grade()

if __name__ == '__main__':
    main()
