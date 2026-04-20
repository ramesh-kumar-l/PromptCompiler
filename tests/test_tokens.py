"""Tests for token estimation and cost calculation."""

import pytest

from promptlang.tokens.estimator import estimate_tokens, estimate_prompt_tokens
from promptlang.tokens.cost import (
    cost_breakdown,
    estimate_cost,
    format_cost,
    get_price,
    MODEL_PRICING,
)


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def test_estimate_empty():
    assert estimate_tokens("") == 0


def test_estimate_non_empty():
    result = estimate_tokens("Hello, how are you?")
    assert result > 0


def test_estimate_longer_text_has_more_tokens():
    short = estimate_tokens("Hi")
    long = estimate_tokens("Hello, this is a much longer sentence with many more words.")
    assert long > short


def test_estimate_prompt_tokens_structure():
    result = estimate_prompt_tokens(system="You are helpful.", user="What is 2+2?")
    assert "system" in result
    assert "user" in result
    assert "overhead" in result
    assert "total" in result
    assert result["total"] == result["system"] + result["user"] + result["overhead"]


def test_estimate_prompt_tokens_no_system():
    result = estimate_prompt_tokens(user="Hello")
    assert result["system"] == 0
    assert result["overhead"] == 3


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def test_zero_tokens_zero_cost():
    assert estimate_cost(0, 0) == 0.0


def test_cost_positive():
    cost = estimate_cost(1000, 500, model="gpt-4o")
    assert cost > 0.0


def test_cost_output_more_expensive():
    input_only = estimate_cost(1000, 0, model="gpt-4o")
    with_output = estimate_cost(1000, 1000, model="gpt-4o")
    assert with_output > input_only


def test_cost_cheaper_model():
    expensive = estimate_cost(10000, 0, model="gpt-4-turbo")
    cheap = estimate_cost(10000, 0, model="gpt-4o-mini")
    assert cheap < expensive


def test_all_models_have_pricing():
    for model in ["gpt-4o", "gpt-4o-mini", "claude-opus-4", "claude-haiku-4-5"]:
        price = get_price(model)
        assert price.input_per_1m > 0
        assert price.output_per_1m > 0


def test_unknown_model_fallback():
    price = get_price("unknown-model-xyz")
    default_price = get_price("gpt-4o")
    assert price == default_price


def test_cost_breakdown_structure():
    result = cost_breakdown(1000, 500, model="gpt-4o")
    assert result["model"] == "gpt-4o"
    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 500
    assert "input_cost_usd" in result
    assert "output_cost_usd" in result
    assert "total_cost_usd" in result
    assert "formatted" in result
    assert result["total_cost_usd"] == pytest.approx(
        result["input_cost_usd"] + result["output_cost_usd"]
    )


def test_format_cost_zero():
    assert format_cost(0) == "$0.000000"


def test_format_cost_small():
    result = format_cost(0.0000005)
    assert result.startswith("$")


def test_format_cost_large():
    result = format_cost(1.50)
    assert result == "$1.5000"
