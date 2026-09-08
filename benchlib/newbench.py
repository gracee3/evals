"""CPU-testable contracts for the bounded generation benchmarks."""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
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
    final = split_response(response)['final']
    checks = [bool(check(final)) for check in row.get('checks', [])]
    loose_checks = [bool(check(final)) for check in row.get('loose_checks', row.get('checks', []))]
    return {'score': float(all(checks)) if checks else None,
            'strict': all(checks) if checks else None,
            'loose': all(loose_checks) if loose_checks else None,
            'invalid': not final, 'extraction': 'final answer after </think>'}


def grade_code(row, code, timeout=2):
    """Small CPU fixture grader; production uses the isolated LCB container."""
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
