# agent-groundwork

A learning-project local agent built on Ollama for the CPU-only era. The point is to build an agent stack from first principles in service of larger personal-use agents (legal worklist assistant, family coordination bot) that come later.

## Where to start

- [`SPEC.md`](SPEC.md) — what this is, what it's made of, what it does
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — how it gets built, in order
- [`CLAUDE.md`](CLAUDE.md) — operating guide for working in the repo

## Status

**Phase 4 — CLI frontend, complete.** The full stack works end-to-end:

| Phase | What | State |
|---|---|---|
| 0 | Repo skeleton | done |
| 1 | Provider + tool framework | done |
| 2 | Bakeoff harness + first run (3 finalists picked) | done |
| 3 | Agent core (loop, compactor, tracing) | done |
| 4 | CLI frontend + `ask_user` wiring | **done** |
| 5 | Use-and-tune the system prompt + compaction against finalists | next |

The Phase 2 finalists are `gemma4:e4b` (default), `gemma4:26b`, and `gemma4:e2b`; rationale lives in [`bakeoff_results/20260407-221909/decision.md`](bakeoff_results/20260407-221909/decision.md). See [`CLAUDE.md`](CLAUDE.md) for the up-to-date phase summary.

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for environment management
- [Ollama](https://ollama.com) running locally (`ollama serve`) with the model in `config.toml` pulled

## Setup

```bash
uv sync                  # install dependencies into .venv
ollama serve &           # if it's not already running
# Make sure the model named in config.toml ([model] name) is pulled, e.g.:
ollama pull gemma4:e4b
```

## Run the agent

```bash
uv run python -m agent_groundwork
```

You should see a banner with the model, sandbox path, session id, and trace path, followed by a `you>` prompt. Type a request and the agent will respond, streaming text as it arrives and rendering tool calls inline.

A first thing to try:

```
you> Write a file called hello.md containing the word "hi", then read it back to confirm.
```

You can verify it worked by inspecting `sandbox/hello.md` after the turn finishes.

Another good first test for the `ask_user` flow (deliberately ambiguous):

```
you> Update my notes.
```

The agent should call `ask_user` to clarify; the CLI prints the question and reads your answer from stdin.

### CLI flags

| Flag | Purpose |
|---|---|
| `--config <path>` | Use a different config file (default `./config.toml`) |
| `--model <name>` | Override `config.model.name` for this session |
| `--mode {native,prompted}` | Override `config.model.tool_call_mode` |
| `--no-color` | Disable ANSI colors (auto-disabled if stdout isn't a TTY or `NO_COLOR` is set) |
| `--session-id <id>` | Override the auto-generated session id (useful for reproducible traces) |

### Slash commands inside the REPL

- `/help` — show available commands
- `/trace` — print the current trace file path
- `/quit`, `/exit` — exit the CLI
- Empty Enter — re-prompts without sending anything
- **Ctrl-C** during a turn — cancels the in-flight turn and returns to the prompt
- **Ctrl-D** at the prompt — exits

## Switching models

Two ways:

**Per session** — pass `--model`:

```bash
uv run python -m agent_groundwork --model gemma4:26b
uv run python -m agent_groundwork --model gemma4:e2b --mode prompted
```

**As the new default** — edit `[model] name` (and optionally `[model] tool_call_mode`) in `config.toml`.

Most modern Ollama models support `native` tool calling; smaller or older ones may need `prompted`. The bakeoff measured both modes per model — see [`bakeoff_results/20260407-221909/decision.md`](bakeoff_results/20260407-221909/decision.md) for which mode worked best on each finalist.

## Where things live

| What | Where |
|---|---|
| Runtime config | `config.toml` |
| Agent's working data | `sandbox/` (gitignored except `AGENT.md`) |
| Agent's system prompt + self-maintained index | `sandbox/AGENT.md` |
| Conversation summaries (after compaction) | `sandbox/conversations/<session-id>/summary.md` |
| Per-session JSONL traces | `traces/<YYYYMMDD-HHMMSS>-<session-id>.jsonl` |
| Bakeoff results | `bakeoff_results/<timestamp>/` |

## Debugging with traces

Every session writes an append-only JSONL trace. One line per event: provider calls (with timing + token counts), tool calls (with args, results, latency), compactions (with pre/post snapshots and the summary), agent events (text chunks, system-prompt edits, errors). It is the primary debugging artifact and is readable with `jq` alone.

Inspect the most recent trace:

```bash
ls -t traces/*.jsonl | head -1 | xargs -I {} jq . {} | less
```

Filter by event type:

```bash
# Show every tool call with its result
jq 'select(.event_type == "tool_call")' traces/<file>.jsonl

# Show provider call latencies and token counts
jq 'select(.event_type == "provider_call") | {model, mode: .tool_call_mode, ttft_ms, total_latency_ms, completion_tokens}' traces/<file>.jsonl

# Show only errors and parse failures
jq 'select(.event_type == "error" or .event_type == "parse_error" or .event_type == "provider_error")' traces/<file>.jsonl
```

The CLI also prints the current trace path on startup, on exit, and any time you type `/trace` at the prompt.

## Other entry points

```bash
# One-shot end-to-end agent run (Phase 3 smoketest, useful for headless verification)
uv run python scripts/agent_smoketest.py

# Re-run the bakeoff harness against the candidate model list in config.toml.
# Output goes to bakeoff_results/<timestamp>/ with a markdown report.
uv run python -m agent_groundwork.bakeoff
```

## Hardware notes

The current target is a CPU-only laptop (32 GB RAM). Inference is slow:

- 4B-class MoE: ~5–12 tok/s
- 8B dense: ~3–6 tok/s
- 12B+: single-digit tok/s

Multi-step agent loops compound this. Streaming output is essential, not optional. The bakeoff records absolute capability metrics so re-running on better hardware later gives apples-to-apples comparison.
