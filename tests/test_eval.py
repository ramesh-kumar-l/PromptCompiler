"""Tests for evaluation engine: metrics, dataset loading, runner, store."""

import json
import tempfile
from pathlib import Path

import pytest

from promptlang.eval.dataset import EvalCase, EvalDataset, DatasetLoadError, load_dataset
from promptlang.eval.metrics import (
    combined_score,
    exact_match_score,
    keyword_match_score,
    length_score,
    compute_all,
)
from promptlang.eval.runner import evaluate_prompt
from promptlang.eval.store import EvalStore
from promptlang.core.compiler import compile_string


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _case(**kwargs) -> EvalCase:
    return EvalCase(input={}, **kwargs)


def test_keyword_match_all_present():
    case = _case(expected_keywords=["cat", "dog"])
    assert keyword_match_score("I love cats and dogs", case) == 1.0


def test_keyword_match_partial():
    case = _case(expected_keywords=["cat", "dog", "bird"])
    score = keyword_match_score("cats are great", case)
    assert abs(score - 1 / 3) < 1e-9


def test_keyword_match_none():
    case = _case(expected_keywords=["cat", "dog"])
    assert keyword_match_score("fish are cool", case) == 0.0


def test_keyword_match_no_keywords():
    case = _case(expected_keywords=[])
    assert keyword_match_score("anything", case) == 1.0


def test_length_score_within_bounds():
    case = _case(min_length=5, max_length=20)
    assert length_score("one two three four five six", case) == 1.0


def test_length_score_too_short():
    case = _case(min_length=10)
    score = length_score("short", case)
    assert 0.0 <= score < 1.0


def test_length_score_too_long():
    case = _case(max_length=3)
    score = length_score("one two three four five", case)
    assert 0.0 <= score < 1.0


def test_length_score_no_bounds():
    case = _case()
    assert length_score("anything goes here", case) == 1.0


def test_exact_match_hit():
    case = _case(expected_output="hello world")
    assert exact_match_score("hello world", case) == 1.0


def test_exact_match_miss():
    case = _case(expected_output="hello world")
    assert exact_match_score("hi world", case) == 0.0


def test_exact_match_no_expected():
    case = _case()
    assert exact_match_score("anything", case) == 1.0


def test_combined_score_range():
    case = _case(expected_keywords=["a"], min_length=1, max_length=50)
    score = combined_score("a quick brown fox", case)
    assert 0.0 <= score <= 1.0


def test_compute_all_keys():
    case = _case(expected_keywords=["test"])
    results = compute_all("this is a test output", case)
    assert set(results.keys()) == {"keyword_match", "length", "exact_match", "combined"}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

VALID_DATASET_YAML = """
name: test_dataset
cases:
  - input:
      text: hello
    expected_keywords: [hello, hi]
    max_length: 100
  - input:
      text: world
    expected_keywords: [world]
    tags: [geography]
"""


def test_load_dataset_from_string(tmp_path):
    f = tmp_path / "dataset.yaml"
    f.write_text(VALID_DATASET_YAML)
    ds = load_dataset(f)
    assert ds.name == "test_dataset"
    assert len(ds.cases) == 2
    assert ds.cases[0].expected_keywords == ["hello", "hi"]


def test_load_dataset_missing_file():
    with pytest.raises(DatasetLoadError):
        load_dataset("/nonexistent/path/data.yaml")


def test_load_dataset_no_cases(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("name: bad\n")
    with pytest.raises(DatasetLoadError):
        load_dataset(f)


def test_filter_by_tag():
    cases = [
        EvalCase(input={}, tags=["science"]),
        EvalCase(input={}, tags=["tech"]),
        EvalCase(input={}, tags=["science", "tech"]),
    ]
    ds = EvalDataset(name="t", cases=cases)
    filtered = ds.filter_by_tag("science")
    assert len(filtered.cases) == 2


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PROMPT_YAML = """
name: runner_test
system: You are helpful.
user: Answer about {{topic}}.
constraints:
  model: gpt-4o-mini
"""

DATASET_WITH_INPUT = EvalDataset(
    name="runner_ds",
    cases=[
        EvalCase(input={"topic": "Python"}, expected_keywords=["mock"]),
        EvalCase(input={"topic": "Java"}, expected_keywords=["mock"]),
    ],
)


def test_runner_mock_llm():
    cr = compile_string(PROMPT_YAML)
    report = evaluate_prompt(cr, DATASET_WITH_INPUT, store=None)
    assert len(report.case_results) == 2
    assert report.total_tokens > 0
    assert 0.0 <= report.avg_score <= 1.0


def test_runner_custom_llm():
    def my_llm(prompt: str, constraints: dict) -> str:
        return "mock response here"

    cr = compile_string(PROMPT_YAML)
    report = evaluate_prompt(cr, DATASET_WITH_INPUT, llm=my_llm, store=None)
    for r in report.case_results:
        assert r.output == "mock response here"


def test_runner_report_summary():
    cr = compile_string(PROMPT_YAML)
    report = evaluate_prompt(cr, DATASET_WITH_INPUT, store=None)
    summary = report.summary()
    assert "runner_test" in summary
    assert "gpt-4o-mini" in summary


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_store_save_and_retrieve(tmp_path):
    db = tmp_path / "test.db"
    with EvalStore(db) as store:
        case_results = [
            {"input": {"k": "v"}, "output": "hello", "scores": {"combined": 0.9},
             "combined_score": 0.9, "tokens": 50, "cost_usd": 0.0001},
        ]
        run_id = store.save_run("my_prompt", "1.0", "gpt-4o", "test_ds", case_results)
        assert run_id is not None

        run = store.get_run(run_id)
        assert run["prompt_name"] == "my_prompt"
        assert run["avg_score"] == pytest.approx(0.9)

        cases = store.get_cases(run_id)
        assert len(cases) == 1
        assert cases[0]["output"] == "hello"


def test_store_list_runs(tmp_path):
    db = tmp_path / "test.db"
    with EvalStore(db) as store:
        for i in range(3):
            store.save_run(f"prompt_{i}", "1.0", "gpt-4o", "ds", [
                {"input": {}, "output": "x", "scores": {}, "combined_score": 0.5,
                 "tokens": 10, "cost_usd": 0.0},
            ])
        runs = store.list_runs()
        assert len(runs) == 3
