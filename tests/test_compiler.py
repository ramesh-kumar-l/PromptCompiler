"""Tests for the compiler pipeline: DSL → IR → optimized IR → rendered output."""

import pytest

from promptlang.core.compiler import compile_string
from promptlang.core.ir import PromptIR
from promptlang.core.renderer import RenderError, render, render_messages
from promptlang.core.optimizer import (
    optimize,
    pass_trim_whitespace,
    pass_deduplicate_sentences,
    pass_remove_empty_blocks,
)


SIMPLE_YAML = """
name: greet
system: You are a friendly assistant.
user: Say hello to {{name}}.
"""

FEW_SHOT_YAML = """
name: classify
system: Classify sentiment.
few_shot:
  - input: I love this!
    output: positive
  - input: This is terrible.
    output: negative
user: |
  Classify the following text.
  Text: {{text}}
constraints:
  model: gpt-4o-mini
"""

WHITESPACE_YAML = """
name: messy
user: |
  Line one.


  Line two.

  Line three.
"""

DUPLICATE_YAML = """
name: dup
system: You are helpful. You are helpful.
user: Respond clearly.
"""


# ---------------------------------------------------------------------------
# Compiler correctness
# ---------------------------------------------------------------------------

def test_compile_basic():
    cr = compile_string(SIMPLE_YAML)
    assert cr.name == "greet"
    ir = cr.ir
    assert len(ir.system_blocks) == 1
    assert len(ir.user_blocks) == 1
    assert "{{name}}" in ir.user_blocks[0].content


def test_compile_few_shot():
    cr = compile_string(FEW_SHOT_YAML)
    assert len(cr.ir.few_shot_blocks) == 4  # 2 pairs × 2 blocks each


def test_declared_variables():
    cr = compile_string(SIMPLE_YAML)
    assert "name" in cr.ir.declared_variables()


def test_model_propagated():
    cr = compile_string(FEW_SHOT_YAML)
    assert cr.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def test_render_substitution():
    cr = compile_string(SIMPLE_YAML)
    output = render(cr.ir, {"name": "Alice"})
    assert "Alice" in output
    assert "{{name}}" not in output


def test_render_missing_variable():
    cr = compile_string(SIMPLE_YAML)
    with pytest.raises(RenderError) as exc_info:
        render(cr.ir, {})
    assert "name" in str(exc_info.value)


def test_render_messages_structure():
    cr = compile_string(FEW_SHOT_YAML)
    messages = render_messages(cr.ir, {"text": "I hate bugs"})
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles


def test_render_sections():
    cr = compile_string(SIMPLE_YAML)
    output = render(cr.ir, {"name": "Bob"})
    assert "System:" in output
    assert "User:" in output


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def test_optimizer_trims_whitespace():
    cr = compile_string(WHITESPACE_YAML)
    raw_content = cr.raw_ir.user_blocks[0].content
    opt_content = cr.ir.user_blocks[0].content
    # Trailing whitespace should be gone
    for line in opt_content.split("\n"):
        assert line == line.rstrip()


def test_optimizer_deduplicates():
    cr = compile_string(DUPLICATE_YAML)
    content = cr.ir.system_blocks[0].content
    # "You are helpful." should appear once
    count = content.lower().count("you are helpful")
    assert count == 1


def test_optimizer_removes_empty_blocks():
    from promptlang.core.ir import Block, PromptIR
    ir = PromptIR(
        name="t", version="1",
        system_blocks=[Block("system", "  "), Block("system", "real content")],
        user_blocks=[Block("user", "hi")],
    )
    optimized = optimize(ir)
    assert len(optimized.system_blocks) == 1
    assert optimized.system_blocks[0].content == "real content"


def test_raw_ir_preserved():
    cr = compile_string(DUPLICATE_YAML)
    raw = cr.raw_ir.system_blocks[0].content
    assert raw.count("You are helpful.") == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_system_allowed():
    yaml = "name: no_sys\nuser: Just a user prompt."
    cr = compile_string(yaml)
    assert cr.ir.system_blocks == []
    output = render(cr.ir, {})
    assert "System:" not in output
    assert "Just a user prompt." in output


def test_multiple_variable_types():
    yaml = "name: multi\nuser: Dear {{title}} {{last_name}}, your order {{order_id}} is ready."
    cr = compile_string(yaml)
    vars_ = cr.ir.declared_variables()
    assert set(vars_) == {"title", "last_name", "order_id"}
