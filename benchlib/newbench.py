"""CPU-testable contracts for the bounded generation benchmarks."""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
import json
from pathlib import Path


def nested_sample(rows, count, seed):
    import random
    if count > len(rows):
        raise ValueError('sample count exceeds dataset')
    values = sorted(rows, key=lambda r: str(r['id']))
    random.Random(seed).shuffle(values)
    return values[:count]


def split_response(text):
    text = text or ''
    match = re.search(r'<think>(.*?)</think>', text, re.I | re.S)
    if match:
        reasoning, final = match.group(1).strip(), text[match.end():].strip()
    elif re.search(r'</think>', text, re.I):
        reasoning, final = '', ''
    else:
        reasoning, final = '', text.strip()
    return {'reasoning': reasoning, 'final': final, 'has_final': bool(final)}


def parse_choice(text, option_count=4):
    final = split_response(text)['final']
    letters = re.findall(r'(?<![A-Za-z])\(?([A-Z])\)?(?![A-Za-z])', final.upper())
    valid = [x for x in letters if ord(x) - 65 < option_count]
    return valid[-1] if valid else None


def grade_gpqa(row, response):
    answer = parse_choice(response, len(row.get('options', [])) or 4)
    return {'score': float(answer == row['answer']) if answer else None,
            'answer': answer, 'invalid': answer is None,
            'extraction': 'last standalone option letter in final answer'}


def grade_ifbench(row, response):
    # The production runtime keeps IFBench's dependency in /opt/ifbench so it
    # cannot alter the pinned lm-eval environment. Use its registry as the
    # source of truth when that isolated environment is present.
    verifier = Path('/opt/ifbench/bin/python')
    if verifier.exists() and row.get('instruction_id_list'):
        script = '''import json, sys
from ifbench import instructions_registry
row, response = json.load(sys.stdin)
checks = []
for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
    cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = cls(instruction_id)
    instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=row["prompt"])
    checks.append(bool(response.strip()) and instruction.check_following(response))
print(json.dumps({"strict": all(checks), "instruction_results": checks}))
'''
        result = subprocess.run([str(verifier), '-c', script], input=json.dumps([row, response]),
            text=True, capture_output=True, timeout=30, check=True)
        value = json.loads(result.stdout)
        strict = bool(value['strict'])
        # The upstream loose evaluator tolerates boundary formatting and
        # markdown asterisks; preserve that behavior for prompt-level loose.
        candidates = {response, response.replace('*', '')}
        loose = any(_run_ifbench_checks(row, candidate, verifier) for candidate in candidates)
        return {'score': float(strict), 'strict': strict, 'loose': loose,
                'invalid': not response.strip(), 'extraction': 'final answer after </think>',
                'instruction_results': value['instruction_results'], 'verifier': 'ifbench 0.2.0 registry'}
    final = split_response(response)['final']
    checks = [bool(check(final)) for check in row.get('checks', [])]
    loose_checks = [bool(check(final)) for check in row.get('loose_checks', row.get('checks', []))]
    return {'score': float(all(checks)) if checks else None,
            'strict': all(checks) if checks else None,
            'loose': all(loose_checks) if loose_checks else None,
            'invalid': not final, 'extraction': 'final answer after </think>'}


def _run_ifbench_checks(row, response, verifier):
    script = '''import json, sys
from ifbench import instructions_registry
row, response = json.load(sys.stdin)
out = []
for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
    args = instruction.get_instruction_args()
    if args and "prompt" in args: instruction.build_description(prompt=row["prompt"])
    out.append(bool(response.strip()) and instruction.check_following(response))
print(json.dumps(all(out)))
'''
    return json.loads(subprocess.run([str(verifier), '-c', script], input=json.dumps([row, response]),
        text=True, capture_output=True, timeout=30, check=True).stdout)


def grade_code(row, code, timeout=2):
    """Small CPU fixture grader; production uses the isolated LCB container."""
    lcb_python = Path('/opt/livecodebench/bin/python')
    if lcb_python.exists() and row.get('input_output'):
        script = '''import json, sys
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
row, code, timeout = json.load(sys.stdin)
metrics = codegen_metrics([row], [[code]], k_list=[1], num_process_evaluate=1, timeout=timeout, debug=False)
print(json.dumps({"score": float(metrics[0]["pass@1"]), "metadata": metrics[2]}))
'''
        environment = dict(__import__('os').environ,
            PYTHONPATH='/opt/livecodebench-src:/usr/local/lib/python3.12/dist-packages:' + __import__('os').environ.get('PYTHONPATH', ''))
        result = subprocess.run([str(lcb_python), '-c', script], input=json.dumps([row, code, timeout]),
            text=True, capture_output=True, timeout=timeout * 4 + 10, check=True, env=environment)
        return json.loads(result.stdout.splitlines()[-1]) | {'status': 'passed' if json.loads(result.stdout.splitlines()[-1])['score'] else 'failed', 'execution_failure': False, 'grader': 'LiveCodeBench codegen_metrics'}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'submission.py'
        path.write_text(code)
        try:
            ast.parse(code)
            result = subprocess.run(['python3', str(path)], input=row.get('stdin', ''),
                capture_output=True, text=True, timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            return {'score': 0.0, 'status': 'timed_out', 'execution_failure': False}
        except (SyntaxError, OSError) as exc:
            return {'score': 0.0, 'status': 'invalid', 'error': str(exc), 'execution_failure': True}
        passed = result.returncode == 0 and result.stdout.strip() == row.get('expected_stdout', '').strip()
        return {'score': float(passed), 'status': 'passed' if passed else 'failed',
                'execution_failure': result.returncode != 0}


def fixture_rows(name):
    if name == 'ifbench':
        return [{'id': 'if-0', 'checks': [lambda x: x.endswith('blue')]},
                {'id': 'if-1', 'checks': [lambda x: len(x.split()) == 2]}]
    if name == 'gpqa_diamond':
        return [{'id': 'gpqa-0', 'options': ['a', 'b', 'c', 'd'], 'answer': 'B'},
                {'id': 'gpqa-1', 'options': ['a', 'b', 'c', 'd'], 'answer': 'D'}]
    if name == 'livecodebench_v6':
        return [{'id': 'lcb-0', 'stdin': 'x\n', 'expected_stdout': 'x'},
                {'id': 'lcb-1', 'stdin': '', 'expected_stdout': 'ok'}]
    raise ValueError(name)
