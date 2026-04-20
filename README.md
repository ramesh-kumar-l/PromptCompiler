# PromptLang — Prompt Compiler System

> Treat prompts like code. Compile them, optimize them, test them.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is PromptLang?

PromptLang is an open-source **prompt compiler** that transforms structured YAML prompt definitions into optimized, testable, and cost-analyzed LLM prompts.

Most LLM applications treat prompts as raw strings buried in code. PromptLang treats them as **first-class artifacts** — versioned, structured, compiled, and evaluated. The result: prompts that are easier to maintain, cheaper to run, and measurable in quality.

---

## Why Do Prompts Need Structure?

| Without PromptLang | With PromptLang |
|---|---|
| Prompts hardcoded as Python f-strings | Prompts as versioned YAML files |
| No deduplication or whitespace cleanup | Automatic optimization passes |
| Token costs discovered in production | Cost estimated before any API call |
| Quality measured by vibes | Scored against evaluation datasets |
| No variable safety | Variables declared and validated |

---

## Architecture

```
DSL (YAML)
    │
    ▼
┌─────────┐
│  Parser │  YAML → PromptDSL (validated schema)
└────┬────┘
     │
     ▼
┌──────────────┐
│  Compiler    │  PromptDSL → PromptIR (canonical in-memory form)
└────┬─────────┘
     │
     ▼
┌──────────────┐
│  Optimizer   │  Whitespace trim · Deduplication · Few-shot injection
└────┬─────────┘
     │
     ▼
┌──────────────┐
│  Renderer    │  PromptIR + variables → final string or messages[]
└────┬─────────┘
     │
     ▼
┌──────────────────────┐
│  Evaluation Engine   │  Dataset cases → LLM → scoring → SQLite
└──────────────────────┘
```

### Key Abstractions

**PromptDSL** — The YAML schema. Defines `name`, `system`, `user`, `few_shot`, and `constraints`.

**PromptIR** — The Intermediate Representation. A structured, role-separated, variable-aware in-memory object. All optimizer passes operate on the IR.

**Optimizer** — A pipeline of pure functions `(IR) → IR`. Runs trim, dedup, and injection passes. Preserves the raw IR so you can inspect before/after.

**Renderer** — Converts IR + variables into a final prompt string or OpenAI-style `messages[]` list. Validates that all declared variables are provided.

**EvalStore** — SQLite database at `~/.promptlang/eval_results.db` storing every run with per-case scores, token costs, and timestamps.

---

## Quick Start

### Install

```bash
pip install promptlang

# With exact tokenization via tiktoken (recommended)
pip install "promptlang[tiktoken]"
```

### Write a Prompt

```yaml
# summarize.yaml
name: summarize
version: "1.0"
description: Summarize a piece of text.

system: |
  You are an expert summarizer. Your summaries are concise and accurate.

user: |
  Summarize the following in under {{max_words}} words:

  {{input_text}}

constraints:
  max_tokens: 300
  temperature: 0.3
  model: gpt-4o
```

### Compile It

```bash
$ promptlang compile summarize.yaml --vars '{"max_words": "50", "input_text": "..."}'

─── Compiled Prompt ───
System:
You are an expert summarizer. Your summaries are concise and accurate.

User:
Summarize the following in under 50 words:

...
```

### Estimate Cost

```bash
$ promptlang tokens summarize.yaml --vars '{"max_words": "50", "input_text": "Amazon rainforest..."}'

─── Token & Cost Estimate ───
  Model         : gpt-4o
  Input tokens  : 87
  Output tokens : 0
  Input cost    : $0.000218
  Total cost    : $0.000218
```

### Evaluate

```bash
$ promptlang eval summarize.yaml --dataset summarize_dataset.yaml --verbose

─── Evaluation Report ───
Prompt   : summarize v1.0
Model    : gpt-4o
Dataset  : summarize_eval
Cases    : 3
Avg Score: 0.7143
Tokens   : 312
Cost     : $0.000780
```

---

## CLI Reference

```bash
# Compile and render a prompt
promptlang compile <prompt.yaml> [--vars '{"k":"v"}'] [--vars-file vars.json] [--messages] [--show-ir]

# Dry-run with inputs — no LLM call
promptlang run <prompt.yaml> [--input '{"k":"v"}']

# Estimate token count and cost
promptlang tokens <prompt.yaml> [--output-tokens 200] [--model gpt-4o-mini]

# Evaluate against a dataset
promptlang eval <prompt.yaml> --dataset <dataset.yaml> [--verbose] [--tag <tag>] [--no-persist]

# View evaluation history
promptlang history [--prompt <name>] [--limit 20]
```

---

## Python API

```python
from promptlang import compile_prompt, evaluate, get_cost

# 1. Render a prompt with variables
rendered = compile_prompt("summarize.yaml", variables={"input_text": "...", "max_words": "50"})
print(rendered)

# 2. Get cost estimate before calling any API
cost = get_cost("summarize.yaml", variables={"input_text": "...", "max_words": "50"})
print(f"Estimated cost: {cost['formatted']}")  # $0.000218

# 3. Evaluate against a dataset (uses mock LLM by default)
report = evaluate("summarize.yaml", dataset="summarize_dataset.yaml")
print(report.summary())

# 4. Evaluate with a real LLM
import anthropic

client = anthropic.Anthropic()

def my_llm(prompt: str, constraints: dict) -> str:
    response = client.messages.create(
        model=constraints.get("model", "claude-sonnet-4-5"),
        max_tokens=constraints.get("max_tokens", 1024),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

report = evaluate("summarize.yaml", dataset="summarize_dataset.yaml", llm=my_llm)
print(f"Score: {report.avg_score:.3f} | Cost: ${report.total_cost_usd:.6f}")

# 5. Get OpenAI-style messages
from promptlang import compile_to_messages
messages = compile_to_messages("classify.yaml", variables={"text": "I love this!"})
# [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]
```

### Low-level API (IR access)

```python
from promptlang import compile_to_ir

result = compile_to_ir("summarize.yaml")
ir = result.ir

print(ir.name)                  # "summarize"
print(ir.declared_variables())  # ["input_text", "max_words"]
print(ir.constraints)           # {"max_tokens": 300, "model": "gpt-4o", ...}

# Inspect before/after optimization
print(result.raw_ir.system_blocks[0].content)  # pre-optimization
print(result.ir.system_blocks[0].content)      # post-optimization
```

---

## DSL Reference

```yaml
name: my_prompt          # required, unique identifier
version: "1.0"           # optional, default "1.0"
description: "..."       # optional, stored in metadata

system: |                # optional system message
  You are a ...

user: |                  # required user message
  Answer {{question}}.   # {{variable}} placeholders

few_shot:                # optional list of examples
  - input: "What is 2+2?"
    output: "4"
  - input: "Capital of France?"
    output: "Paris"

constraints:
  max_tokens: 500        # optional
  temperature: 0.7       # optional, 0.0–2.0
  top_p: 0.9             # optional
  model: gpt-4o          # default: gpt-4o

metadata:                # arbitrary key-value pairs
  author: team-name
  task: classification
```

### Variables

Variables use `{{double_brace}}` syntax and are resolved at render time:

- Missing variables raise `RenderError` at render time (fail fast)
- Variables are extracted from both `system` and `user` blocks
- `compile_to_ir().ir.declared_variables()` lists all required variables

---

## Dataset Format

```yaml
name: my_eval
description: "..."

cases:
  - id: case_001                      # optional identifier
    input:
      variable_name: "value"
    expected_keywords: [word1, word2] # scored by presence
    expected_output: "exact string"   # for exact match scoring
    min_length: 10                    # word count lower bound
    max_length: 100                   # word count upper bound
    tags: [tag1, tag2]                # for filtering with --tag
```

---

## Evaluation Scoring

Each case is scored by four built-in metrics:

| Metric | Formula | Weight in `combined` |
|---|---|---|
| `keyword_match` | keywords found / keywords expected | 0.70 |
| `length` | output within [min_length, max_length] | 0.30 |
| `exact_match` | output == expected_output | — |
| `combined` | 0.7 × keyword + 0.3 × length | primary score |

All scores are in `[0.0, 1.0]`.

---

## Supported Models & Pricing

| Model | Input (per 1M) | Output (per 1M) |
|---|---|---|
| `gpt-4o` | $2.50 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `claude-opus-4` | $15.00 | $75.00 |
| `claude-sonnet-4-5` | $3.00 | $15.00 |
| `claude-haiku-4-5` | $0.80 | $4.00 |
| `gemini-1.5-pro` | $1.25 | $5.00 |
| `gemini-1.5-flash` | $0.075 | $0.30 |

Token counts use `tiktoken` (exact) when installed, or a word-based heuristic otherwise.

---

## Examples

The `examples/` directory contains ready-to-run prompts and datasets:

```
examples/
├── summarize.yaml           # text summarization prompt
├── summarize_dataset.yaml   # 3 evaluation cases for summarize
├── classify.yaml            # sentiment classification with few-shot
└── classify_dataset.yaml    # 3 evaluation cases for classify
```

```bash
# Run the summarize example
promptlang compile examples/summarize.yaml \
  --vars '{"input_text": "Python is a programming language.", "max_words": "20"}'

# Evaluate it
promptlang eval examples/summarize.yaml --dataset examples/summarize_dataset.yaml

# View history
promptlang history
```

---

## Development

```bash
git clone https://github.com/promptlang/promptlang
cd promptlang
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=promptlang --cov-report=term-missing
```

---

## Benchmarks

Compilation is fast — entirely in-process with no I/O beyond file reads:

| Operation | Typical time |
|---|---|
| Parse + compile a prompt | < 2ms |
| Render with variable substitution | < 1ms |
| Token estimation (heuristic) | < 1ms |
| Token estimation (tiktoken) | 2–5ms |
| Evaluation run (mock LLM, 10 cases) | < 10ms |

---

## Limitations

- **No built-in LLM integration** — PromptLang compiles and evaluates prompts; you supply the LLM callable.
- **Token estimates are approximate** without `tiktoken` installed.
- **Scoring is structural** — `keyword_match` and `length` don't measure semantic quality.
- **SQLite store is single-process** — not suitable for concurrent evaluation runs.
- **No streaming support** — designed for batch compilation and evaluation.

---

## Roadmap

- [ ] Semantic similarity scoring (embeddings-based)
- [ ] Prompt diff viewer (compare versions)
- [ ] A/B evaluation (compare two prompt files)
- [ ] Streaming renderer
- [ ] Template inheritance (`extends: base.yaml`)
- [ ] Prompt registry (push/pull from remote)
- [ ] VS Code extension for `.promptlang.yaml` files
- [ ] GitHub Actions integration for CI eval gates

---

## License

MIT © PromptLang Contributors
