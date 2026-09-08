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
        elif name in ('ifbench', 'gpqa_diamond', 'livecodebench_v6'):
            rows = load_generation_dataset(name)
            write_json('/work/' + name + '.json', rows)
            dataset_hashes[name] = file_hash('/work/' + name + '.json')
            pools['all'] = [{'id': row['id'], 'index': i} for i, row in enumerate(rows)]
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


def load_generation_dataset(name):
    """Load and normalize pinned upstream data; GPQA remains gated by design."""
    import datasets
    sources = {
        'ifbench': ('allenai/IFBench_test', 'train'),
        'gpqa_diamond': ('Idavidrein/gpqa', 'train'),
        'livecodebench_v6': ('livecodebench/code_generation_lite', 'test'),
    }
    source, split = sources[name]
    load_kwargs = {'split': split, 'revision': PINS[source]}
    if name == 'livecodebench_v6':
        load_kwargs['version_tag'] = 'release_v6'
    ds = datasets.load_dataset(source, **load_kwargs)
    rows = []
    for i, raw in enumerate(ds):
        if name == 'gpqa_diamond' and raw.get('Subset', raw.get('subset', '')).lower() != 'diamond':
            continue
        row = dict(raw)
        row['id'] = str(row.get('question_id', row.get('id', i)))
        if name == 'gpqa_diamond':
            row['options'] = [row.get(k) for k in ('choice1', 'choice2', 'choice3', 'choice4')]
            row['answer'] = str(row.get('Correct Answer', row.get('answer', ''))).strip().upper()
        rows.append(row)
    if not rows:
        raise RuntimeError(name + ' dataset selection is empty')
    return rows


def runtime_args(config, model=None):
    excluded = {'concurrency', 'batch_size', 'speculative_decoding', 'gpu_device'}
    runtime = config.get('runtimes', {}).get(model, config['runtime']) if model else config['runtime']
    return {k: v for k, v in runtime.items() if k not in excluded} | dict(
        pretrained='/model', seed=config['seed'], add_bos_token=False, batch_size=1, max_num_seqs=1)


def record_path(stage, item, kind='result'):
    return Path(stage) / kind / (digest(item) + '.json')


def harness(stage_path, benchmark, model=None, lm=None):
    if benchmark in ('ifbench', 'gpqa_diamond', 'livecodebench_v6'):
        return generation_harness(stage_path, benchmark, model, lm)
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
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    loaded = manager.load(sorted({i['task'] for i in selected}))['tasks']
    for task_name, task in loaded.items():
        if digest(list(task.eval_docs)) != frozen['prepared']['dataset_hashes'][task_name]:
            raise RuntimeError('cached dataset content differs from frozen revision: ' + task_name)
    owned_lm = lm is None
    if owned_lm:
        lm = VLLM(**runtime_args(config, model))
    lm._bench_stage_path = Path(stage_path)
    if not getattr(lm, '_bench_wrapped', False):
        original_generate = lm._model_generate
        def capture(*args, **kwargs):
            outputs = original_generate(*args, **kwargs)
            if kwargs.get('generate', True):
                for output in outputs:
                    write_json(lm._bench_stage_path / 'finish' / (digest(output.prompt_token_ids) + '.json'),
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
        lm._bench_wrapped = True
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
            gen_kwargs=({'max_gen_toks': stage_config['tokens']} | generation_kwargs(config)) if benchmark == 'ifeval' else None,
            random_seed=config['seed'], numpy_random_seed=config['seed'],
            torch_random_seed=config['seed'], fewshot_random_seed=config['seed'], bootstrap_iters=0)
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
    if owned_lm:
        lm.cache_hook.dbdict.close()


def generation_harness(stage_path, benchmark, model=None, lm=None):
    """Generation path for the three bounded comparison suites."""
    from lm_eval.api.instance import Instance
    from benchlib.newbench import grade_gpqa, grade_ifbench, parse_choice, split_response
    frozen = read_json('/work/frozen.json')
    config = frozen['suite']
    rows = read_json('/prepared/' + benchmark + '.json')
    by_id = {r['id']: r for r in rows}
    selected = frozen['prepared']['selection'][benchmark]
    pending = [i for i in selected if not record_path(stage_path, i).exists()]
    if not pending:
        return
    owned_lm = lm is None
    if owned_lm:
        from lm_eval.models.vllm_causallms import VLLM
        lm = VLLM(**runtime_args(config, model))
    generation = config.get('generation', {})
    stage = next(s for s in config['benchmarks'] if s['name'] == benchmark)
    kwargs = {'max_gen_toks': stage['tokens']} | generation_kwargs(config)
    for key in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'):
        if key in generation:
            kwargs[key] = generation[key]
    for item in pending:
        row = by_id[item['id']]
        if benchmark == 'gpqa_diamond':
            prompt = row.get('question', '') + '\n\nOptions:\n' + '\n'.join(f'{chr(65+j)}. {x}' for j, x in enumerate(row['options'])) + '\n\nReturn the final option letter.'
        else:
            prompt = row.get('prompt', row.get('question', row.get('title', '')))
        request = Instance('generate_until', (prompt, kwargs), item['index'])
        response = lm.generate_until([request])[0]
        parts = split_response(response)
        if benchmark == 'gpqa_diamond':
            graded = grade_gpqa(row, response)
        elif benchmark == 'ifbench':
            # Authentic IFBench verifier functions are serialized as upstream
            # metadata and run by the dedicated verifier when available.
            graded = {'score': None, 'invalid': not parts['has_final'], 'strict': None, 'loose': None,
                      'extraction': 'final answer after </think>', 'verifier': 'upstream IFBench verifier pending'}
        else:
            from benchlib.newbench import grade_code
            code = parts['final']
            if '```' in code:
                code = code.split('```', 2)[1]
                code = code.removeprefix('python\n')
            graded = grade_code(row, code, timeout=5) if row.get('input_output') else {
                'score': None, 'invalid': not parts['has_final'], 'extraction': 'code grader consumes final answer',
                'grader': 'LiveCodeBench grader unavailable in runtime image'}
        write_json(record_path(stage_path, item), dict(item=item, reasoning=parts['reasoning'], final=parts['final'],
            raw=response, score=graded.pop('score'), truncated=False, execution_failure=False, **graded))
    if owned_lm:
        lm.cache_hook.dbdict.close()


def generation_kwargs(config):
    generation = config.get('generation', {})
    result = {}
    if 'enable_thinking' in generation:
        result['chat_template_kwargs'] = {'enable_thinking': generation['enable_thinking']}
        if 'preserve_thinking' in generation:
            result['chat_template_kwargs']['preserve_thinking'] = generation['preserve_thinking']
    if 'reasoning_effort' in generation:
        result['reasoning_effort'] = generation['reasoning_effort']
    return result


def harness_group(stage_root, benchmarks, model=None):
    """Run native lm-eval benchmarks with one vLLM model allocation."""
    from lm_eval.models.vllm_causallms import VLLM
    config = read_json('/work/frozen.json')['suite']
    lm = VLLM(**runtime_args(config, model))
    try:
        for benchmark in benchmarks:
            started = time.monotonic()
            harness(Path(stage_root) / model / benchmark, benchmark, model, lm=lm)
            write_json(Path(stage_root) / model / benchmark / 'duration.json',
                       {'benchmark': benchmark, 'seconds': time.monotonic() - started,
                        'grouped_model_process': True})
    finally:
        lm.cache_hook.dbdict.close()


def humaneval_generate(stage_path, model=None):
    import openai
    from evalplus.provider.openai import OpenAIChatDecoder
    from evalplus.gen.util import openai_request
    from evalplus.sanitize import sanitize
    frozen = read_json('/work/frozen.json')
    config = frozen['suite']
    if file_hash('/prepared/humaneval.json') != frozen['prepared']['dataset_hashes']['humaneval_plus']:
        raise RuntimeError('HumanEval dataset differs from frozen identity')
    problems = read_json('/prepared/humaneval.json')
    tokens = next(s['tokens'] for s in config['benchmarks'] if s['name'] == 'humaneval_plus')
    selected = frozen['prepared']['selection']['humaneval_plus']
    pending = [i for i in selected if not record_path(stage_path, i, 'generation').exists()]
    if not pending:
        return
    r = config.get('runtimes', {}).get(model, config['runtime']) if model else config['runtime']
    command = ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model',
        '--served-model-name', 'bench', '--host', '127.0.0.1', '--port', '8000',
        '--tensor-parallel-size', str(r['tensor_parallel_size']), '--max-model-len', str(r['max_model_len']),
        '--dtype', r['dtype'], '--kv-cache-dtype', r['kv_cache_dtype'],
        '--language-model-only', '--enable-chunked-prefill',
        '--max-num-batched-tokens', str(r['max_num_batched_tokens']),
        '--max-num-seqs', '1', '--kv-cache-memory-bytes', str(r['kv_cache_memory_bytes']),
        '--seed', str(config['seed']), '--generation-config', 'vllm',
        '--default-chat-template-kwargs', __import__('json').dumps({'enable_thinking': config.get('generation', {}).get('enable_thinking', False),
            'preserve_thinking': config.get('generation', {}).get('preserve_thinking', True)})]
    if r['enforce_eager']:
        command.insert(command.index('--language-model-only'), '--enforce-eager')
    command.insert(command.index('--language-model-only'),
        '--enable-prefix-caching' if r['enable_prefix_caching'] else '--no-enable-prefix-caching')
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
        # The vLLM server is terminated in finally below. Its async output
        # handler can log EngineDeadError during that expected teardown race;
        # mark successful generation before sending SIGTERM so the supervisor
        # can distinguish it from an inference failure.
        print('BENCH_HUMANEVAL_GENERATION_COMPLETE', flush=True)
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
    parser.add_argument('action', choices=['prepare', 'harness', 'harness-group', 'generate', 'grade', 'export_he'])
    parser.add_argument('--stage')
    parser.add_argument('--benchmark')
    parser.add_argument('--benchmarks')
    parser.add_argument('--model')
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
        harness(args.stage, args.benchmark, args.model)
    elif args.action == 'harness-group':
        harness_group(args.stage, args.benchmarks.split(','), args.model)
    elif args.action == 'generate':
        humaneval_generate(args.stage, args.model)
    else:
        grade()

if __name__ == '__main__':
    main()
