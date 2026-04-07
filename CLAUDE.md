# CLAUDE.md

Guidance for Claude (or any future agent / human) working in this repository.

## What this project is

`agent-groundwork` is a learning project that builds, from first principles, a minimal local agent on a CPU-only laptop using locally-hosted Ollama models. It is the foundation for later "real" agent projects (legal worklist, family coordination).

Read **SPEC.md** for the system design and **IMPLEMENTATION_PLAN.md** for the build order. This file is the daily operating guide.

## Current phase

**Phase 0 — Repository skeleton.** The next thing to build is **Phase 1** (provider and tool framework). See `IMPLEMENTATION_PLAN.md`.

(Update this section as the project moves through phases.)

## How to run things

Once Phase 1 is in:

```
uv sync
python -c "import agent_groundwork"     # smoke import check
```

Once Phase 2 is in:

```
python -m agent_groundwork.bakeoff       # run the bakeoff against config.toml's candidate list
```

Once Phase 4 is in:

```
python -m agent_groundwork               # start the CLI agent
```

## Conventions

- **Python 3.11+**, type hints throughout, Pydantic for any structured data with external boundaries.
- **Async** for the provider layer and the agent loop. Sync is fine for tool implementations (the dispatcher awaits in a thread when needed).
- **No `print()` calls inside the agent core.** All output flows through the event stream. Print is allowed in CLI frontend code and in scripts.
- **No agent frameworks.** Do not add LangChain, LlamaIndex, AutoGen, Letta, or similar. The whole point is to understand what they hide.
- **Tools always return `ToolResult`,** never raise. The dispatcher catches escapes and converts them to `ok=False` so the model can recover.
- **All file operations route through `agent_groundwork/paths.py`.** Never call `open()` directly inside a tool implementation.
- **Pydantic models for tool args.** Schemas are derived from these, both for native and prompted tool calling.
- **JSONL traces are append-only.** Don't batch, don't buffer beyond what `open(..., "a")` does.
- **Configuration goes in `config.toml`,** never hard-coded constants for model names, paths, thresholds, etc.

## Decisions worth not relitigating

These were settled during planning. Each is documented either in SPEC.md or in this list with a brief rationale.

- **Markdown files as the agent's memory substrate.** Not Letta, not a vector DB. *Why:* learning project, transparency, the user can hand-edit memory, fits the eventual real-agent use cases (notes, lists, calendars).
- **Sandbox lives inside the project at `./sandbox/`.** *Why:* easier to debug and modify everything in one place during a learning project. Contents are gitignored except `AGENT.md`.
- **CLI frontend for v1; web and Telegram are deferred but not precluded.** *Why:* faster to build, agent core decouples cleanly behind the event stream.
- **Both native and prompted tool-call modes are supported.** *Why:* small models vary; the bakeoff measures both per model.
- **Iteration cap and max-token threshold are both configurable.** *Why:* CPU latency makes runaway loops painful.
- **`ask_user` is a tool from day one.** *Why:* small models guess when ambiguous; an explicit clarification path is cheap and high-value.
- **The agent may edit its own `AGENT.md`.** *Why:* part of the learning value. Edits emit a loud `SystemPromptEdited` event in the trace and on the CLI.
- **No `delete_file` tool in v1.** *Why:* small models hallucinate; nuking the user's notes is not recoverable.
- **Compaction uses rolling summary + recent window, summary persisted as a markdown file in the sandbox.** *Why:* simple, inspectable, restartable, consistent with the markdown-as-memory philosophy.
- **Compaction defaults to single-model** (same model as the main loop). A separate, smaller summarization model is supported via config but only beneficial when both models can be co-resident in RAM. *Why:* model swap/reload would dwarf any savings; co-resident loading on 32 GB is fine for realistic v1 candidates but not guaranteed for the largest models.
- **The bakeoff is the first deliverable, not a side quest.** *Why:* it forces us to build the provider and tool layers we need anyway, and produces an empirical model choice instead of a vibes-based one.
- **CPU-only is treated as a temporary constraint.** Don't bake CPU-specific assumptions into the architecture; do tune defaults for it. Hardware upgrade is in scope and the bakeoff helps justify it.

## Things to avoid

- **Adding abstractions for hypothetical future needs.** The agent doesn't need a multi-provider routing layer until there's a second provider. The compactor doesn't need a strategy hierarchy until there's a second strategy.
- **Premature error handling.** Let exceptions surface in code that's still being shaped. Wrap them only at well-defined boundaries (the agent loop's tool dispatcher, the trace writer).
- **Refactoring code that hasn't been used yet.** Phase 5 (use-and-iterate) is where most refactoring decisions should be made, with concrete usage data.
- **Adding dependencies.** Every new dep is a small permanent tax. The current set is: `ollama` (or `httpx`), `pydantic`, `pyyaml`, stdlib. Resist additions.
- **Writing tests for code that's about to change.** Phase 5 will reshape a lot of this. The one exception is `paths.py` — sandbox escape would be a real bug.
- **Tool sprawl.** Six tools is enough for v1. Adding a seventh requires articulating which existing tool fails to cover the case.

## Key paths

| Path | What it is |
|---|---|
| `SPEC.md` | System design |
| `IMPLEMENTATION_PLAN.md` | Build order |
| `CLAUDE.md` | This file |
| `config.toml` | Runtime config |
| `agent_groundwork/` | The Python package |
| `sandbox/` | The agent's data directory (gitignored content) |
| `sandbox/AGENT.md` | The agent's system prompt + self-maintained index |
| `traces/` | JSONL session traces (gitignored content) |
| `bakeoff_results/` | Bakeoff outputs (gitignored content) |

## Hardware reality

Host machine: CPU-only laptop, 32 GB RAM. Inference is slow:
- 4B-class MoE: ~5-12 tok/s
- 8B dense: ~3-6 tok/s
- 12B+: single-digit tok/s

Multi-step agent loops compound this. Streaming output is essential, not optional.

The user plans to graduate to a desktop with real GPUs once the project demonstrates value. Bakeoff data informs that purchase — record absolute capability metrics, not just relative ones, so re-running on better hardware gives apples-to-apples comparison.
