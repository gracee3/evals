import pytest

from benchlib.core import suite
from benchlib.newbench import fixture_rows, grade_code, grade_gpqa, grade_ifbench, nested_sample, split_response


def test_nested_selection_is_reproducible_and_prefix_stable():
    rows = [{'id': f'{i:02d}'} for i in range(20)]
    small = nested_sample(rows, 5, 42)
    medium = nested_sample(rows, 10, 42)
    assert small == medium[:5]
    assert small == nested_sample(rows, 5, 42)


def test_reasoning_and_invalid_choice_are_distinct():
    parts = split_response('<think>work</think>\n\nB')
    assert parts == {'reasoning': 'work', 'final': 'B', 'has_final': True}
    assert grade_gpqa(fixture_rows('gpqa_diamond')[0], '<think>x</think>\nB')['score'] == 1.0
    assert grade_gpqa(fixture_rows('gpqa_diamond')[0], '<think>x</think>\nno choice')['invalid']


def test_ifeval_strict_and_loose_fixture_grading():
    row = fixture_rows('ifbench')[0]
    assert grade_ifbench(row, '<think>reason</think> blue')['strict'] is True
    assert grade_ifbench(row, '<think>reason</think> red')['strict'] is False
    assert grade_ifbench(row, '</think>')['invalid']


def test_code_fixture_pass_fail_and_timeout():
    row = fixture_rows('livecodebench_v6')[0]
    assert grade_code(row, "import sys; print(sys.stdin.read().strip())")['score'] == 1.0
    assert grade_code(row, "print('wrong')")['status'] == 'failed'
    assert grade_code(row, 'while True: pass', timeout=0.1)['status'] == 'timed_out'


def test_new_benchmark_generation_configuration_and_large_limit(tmp_path):
    path = tmp_path / 'suite.yaml'
    path.write_text('''
models: [int4-v1, int8-v2]
runtime: {max_model_len: 65536}
benchmarks:
  - {name: ifbench, count: 2, tokens: 20000}
  - {name: gpqa_diamond, count: 2, tokens: 30000}
  - {name: livecodebench_v6, count: 2, tokens: 30000}
generation:
  enable_thinking: true
  reasoning_effort: medium
  temperature: 1.0
  top_p: 0.95
  top_k: 20
  request_deadline_seconds: 120
''')
    config = suite(path)
    assert config['generation']['reasoning_effort'] == 'medium'
    assert config['benchmarks'][1]['tokens'] == 30000


def test_generation_validation_rejects_unknown_reasoning_effort(tmp_path):
    path = tmp_path / 'suite.yaml'
    path.write_text('generation: {reasoning_effort: huge}\n')
    with pytest.raises(ValueError):
        suite(path)
