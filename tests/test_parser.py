"""Tests for DSL parsing and schema validation."""

import pytest

from promptlang.core.dsl_parser import DSLParseError, extract_variables, parse_string
from promptlang.core.schema import PromptDSL


MINIMAL_YAML = """
name: test
user: Hello world
"""

FULL_YAML = """
name: full_test
version: "2.0"
description: A full prompt
system: You are a helpful assistant.
user: |
  Answer this question about {{topic}}.
  Question: {{question}}
few_shot:
  - input: What is 2+2?
    output: "4"
  - input: What is the capital of France?
    output: Paris
constraints:
  max_tokens: 100
  temperature: 0.7
  model: gpt-4o-mini
metadata:
  author: test
"""


def test_parse_minimal():
    dsl = parse_string(MINIMAL_YAML)
    assert dsl.name == "test"
    assert dsl.user == "Hello world"
    assert dsl.system is None
    assert dsl.few_shot is None


def test_parse_full():
    dsl = parse_string(FULL_YAML)
    assert dsl.name == "full_test"
    assert dsl.version == "2.0"
    assert dsl.description == "A full prompt"
    assert dsl.system is not None
    assert "{{topic}}" in dsl.user
    assert len(dsl.few_shot) == 2
    assert dsl.constraints.max_tokens == 100
    assert dsl.constraints.temperature == 0.7
    assert dsl.constraints.model == "gpt-4o-mini"
    assert dsl.metadata["author"] == "test"


def test_parse_missing_user():
    with pytest.raises(DSLParseError):
        parse_string("name: test\n")


def test_parse_empty_name():
    with pytest.raises(DSLParseError):
        parse_string('name: "  "\nuser: hi')


def test_parse_invalid_yaml():
    with pytest.raises(DSLParseError):
        parse_string("name: [unclosed")


def test_parse_not_a_mapping():
    with pytest.raises(DSLParseError):
        parse_string("- item1\n- item2")


def test_extract_variables_empty():
    assert extract_variables("no variables here") == []


def test_extract_variables_single():
    assert extract_variables("Hello {{name}}") == ["name"]


def test_extract_variables_multiple():
    result = extract_variables("{{a}} and {{b}} and {{a}} again")
    assert result == ["a", "b"]  # deduplicated, order preserved


def test_extract_variables_in_template():
    dsl = parse_string(FULL_YAML)
    assert "topic" in dsl.user
    assert "question" in dsl.user
