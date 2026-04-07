# agent-groundwork — Specification

## Purpose

`agent-groundwork` is a learning project that builds, from first principles, a minimal local agent system on a CPU-only machine. It exists to:

1. **Build reusable foundations** — provider abstractions, tool framework, agent loop, memory model, frontend interface — that will be reused when building "real" agents (a legal worklist assistant, a family-coordination bot, etc.).
2. **Empirically choose a model.** Determine which of the locally-installed Ollama models is the best fit for an agent loop on the current hardware, and produce a re-runnable harness so that question can be re-asked as new models drop or hardware changes.
3. **Inform a hardware decision.** Capture concrete latency and capability data that helps justify (or defer) a future desktop upgrade.

It is *not* a product, not a framework, and not intended for distribution. It is built to be torn apart, instrumented, swapped, and extended.

## Non-goals (v1)

- Multi-agent / agent-to-agent communication
- RAG / embedding-based retrieval over the markdown directory (deferred; grep is sufficient at this scale)
- Background, scheduled, or "proactive" behavior — the agent only runs in response to user input
- External integrations: Telegram, Google Calendar, Gmail, web search
- Web UI (planned for a future phase; v1 ships CLI only)
- Multi-user / authentication
- A test suite beyond what the bakeoff harness provides and `paths.py` unit tests
- Production-grade error handling, retries, or observability tooling
- Persistent multi-conversation history with retrieval (one session at a time)

## Constraints and assumptions

- **Hardware:** Single laptop, CPU-only inference, 32 GB RAM. This is treated as a *temporary* constraint — the architecture must not bake in CPU-specific assumptions that fall apart on a future GPU machine.
- **Runtime:** Ollama, hosting locally-downloaded models. Models are swapped via configuration, never hard-coded.
- **Language:** Python 3.11+.
- **Dependencies:** Minimum viable. `ollama` (or raw `httpx`), `pydantic`, `pyyaml`, and the standard library. Explicitly *not*: LangChain, LlamaIndex, AutoGen, Letta, or any other agent framework.

## Architectural overview

The system is composed of independent layers, each replaceable:

```
+--------------------------------------------+
|  Frontend (v1: CLI; v2: Web; v3: Telegram) |
|  - consumes the agent event stream         |
|  - provides user input                     |
+----------------------+---------------------+
                       |
                       v
+--------------------------------------------+
|  Agent core                                |
|  - loop: iterate model -> tool -> model    |
|  - compactor: manage context length        |
|  - tracing: write JSONL session log        |
+----------------------+---------------------+
                       |
        +--------------+--------------+
        v              v              v
+--------------+ +-----------+ +-----------+
|  Provider    | |  Tools    | |  Sandbox  |
|  (Ollama)    | |  (files,  | |  (path    |
|              | |  ask_user)| |  guards)  |
+--------------+ +-----------+ +-----------+
                       ^
                       |
                  +----------+
                  |  Bakeoff |
                  |  harness |
                  +----------+
```

The bakeoff harness is built on the same provider and tool layers as the agent, but does not use the agent loop. It exists to evaluate models with controlled scenarios before they are wired into the loop.

## Components

### Provider layer (`agent_groundwork/providers/`)

A `Provider` exposes a uniform interface to a model regardless of how that model handles tool calling.

```python
class Provider(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_call_mode: Literal["native", "prompted"],
    ) -> ChatResult: ...

    async def stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_call_mode: Literal["native", "prompted"],
    ) -> AsyncIterator[ChatChunk]: ...
```

Two tool-call modes are supported:

- **`native`** — uses Ollama's built-in `tools=` parameter. Whatever chat template the model was trained with handles the rendering. Cleanest, but only works for models with native support.
- **`prompted`** — the provider injects a tool description block into the system prompt and instructs the model to emit JSON tool calls in a fenced code block. The provider parses those calls out of the model's free-form output.

The same provider instance handles both modes; the choice is per-call configuration. The bakeoff measures both for each model.

The provider is responsible for: latency timing, raw token counts (or estimates when missing from Ollama), and surfacing parse failures rather than swallowing them.

### Tool layer (`agent_groundwork/tools/`)

A tool is declared with a Pydantic model for its arguments and an async callable that runs it.

```python
class Tool(Protocol):
    name: str
    description: str
    args_schema: type[BaseModel]
    async def run(self, args: BaseModel) -> ToolResult: ...

class ToolResult(BaseModel):
    ok: bool
    data: Any | None = None
    error: str | None = None
```

`ToolResult` is the only allowed return shape — `ok=False` is how a tool tells the model "you did something I couldn't do" so the model can recover. Tools never raise; the dispatcher catches any escape and converts it to `ok=False`.

The Pydantic schemas are rendered into either Ollama-native tool definitions or into prompted-JSON descriptions by the provider layer. A single tool definition therefore drives both modes.

#### v1 tools

| Tool | Args | Notes |
|---|---|---|
| `list_files` | `subdir: str = ""` | Lists files inside the sandbox, optionally under a subdirectory |
| `read_file` | `path: str` | Returns file contents; rejects paths outside the sandbox |
| `write_file` | `path: str, content: str` | Creates or overwrites a file; loud trace entry if it overwrites |
| `edit_file` | `path: str, old: str, new: str` | Token-efficient targeted edit; errors if `old` is not found or is ambiguous (matches more than once) |
| `search_files` | `query: str, subdir: str = ""` | ripgrep wrapper (with pure-Python fallback); returns `path:line: snippet` matches |
| `ask_user` | `question: str` | Pauses the loop and requests user input via the frontend's `user_input_provider`. Returns the user's response as `data`. |

There is deliberately no `delete_file` in v1.

If the model edits `AGENT.md` (the system prompt), the trace logs a special `SystemPromptEdited` event with a diff. The edit is permitted but loudly noted in both the trace and the frontend.

### Sandbox / path safety (`agent_groundwork/paths.py`)

All file operations route through a single path-validation function that:

1. Resolves the input to an absolute path.
2. Asserts it is contained within the sandbox root **after symlink resolution**.
3. Rejects writes/reads to anything outside that root with a structured error.

The sandbox root is configured once at startup. There is no way for a tool to escape it without modifying source. This is the one place in v1 where unit tests are non-optional.

### Agent loop (`agent_groundwork/agent/loop.py`)

The loop is intentionally minimal:

```python
async def run(user_message: str) -> AsyncIterator[Event]:
    history.append(UserMessage(user_message))
    for iteration in range(max_iterations):
        if compactor.should_compact(history):
            history = await compactor.compact(history)
            yield CompactionEvent(summary_path=...)

        response_text: list[str] = []
        tool_calls: list[ToolCall] = []
        async for chunk in provider.stream(model, history, tools):
            if chunk.text:
                response_text.append(chunk.text)
                yield TextChunk(chunk.text)
            if chunk.tool_call:
                tool_calls.append(chunk.tool_call)

        history.append(AssistantMessage("".join(response_text), tool_calls))

        if not tool_calls:
            yield Done("".join(response_text))
            return

        for call in tool_calls:
            yield ToolCallStarted(call.name, call.args)
            result = await dispatch(call)  # may invoke user_input_provider
            yield ToolCallResult(call.name, result)
            history.append(ToolMessage(call.id, result))

    yield Error("hit iteration cap", recoverable=False)
```

Properties:

- **Streaming:** model output is streamed to the frontend as it is produced.
- **Iteration cap:** configurable; default 8. Hitting it surfaces a clean error rather than hanging.
- **Tool dispatch errors:** any exception inside a tool is caught by the dispatcher, captured into a `ToolResult(ok=False, error=...)`, and fed back to the model rather than crashing the loop.
- **No print statements anywhere in the agent core.** Output flows only through the event stream.
- **`ask_user` is a tool.** The dispatcher calls into the frontend's injected `user_input_provider(question)` callback when a tool's run method requests it. The CLI's provider reads stdin; a future web frontend's provider would be backed by an HTTP request/response cycle.

### Event stream interface

The agent's `run()` is an `async def` that returns an `AsyncIterator[Event]`. Events:

- `TextChunk(text: str)` — streamed model output
- `ToolCallStarted(name: str, args: dict)` — the agent is about to call a tool
- `ToolCallResult(name: str, result: ToolResult)` — the tool returned
- `CompactionEvent(summary_path: str)` — context was compacted
- `SystemPromptEdited(diff: str)` — the model edited `AGENT.md`
- `Done(final_text: str)` — agent completed normally
- `Error(message: str, recoverable: bool)` — something went wrong

Frontends consume this stream and render it however makes sense for their medium.

### Compactor (`agent_groundwork/agent/compaction.py`)

The compactor manages conversation context as it grows. It has a small interface:

```python
class Compactor(Protocol):
    def should_compact(self, history: list[Message]) -> bool: ...
    async def compact(self, history: list[Message]) -> list[Message]: ...
```

**v1 default strategy: rolling summary with recent window.**

- Triggers when message count > N (default 10) **or** estimated tokens > T (default 4000), whichever first. Both configurable.
- On compaction, calls a designated *summarization model* (configurable; defaults to the same model as the main loop) with a prompt to summarize all messages except the most recent K (default 6) into a structured summary covering: stated user goals, decisions made, files created/modified, open threads.
- The summary is written to `sandbox/conversations/<session-id>/summary.md`. It is also injected into the new history as a synthetic system message: *"Earlier in this conversation: …"*.
- Re-compaction is *telescoping*: the next compaction uses the previous summary as input, so context accumulates rather than resetting.

**Configuration:** `compaction.summarization_model` may be set to a smaller, faster model. If left empty, the main model is used (no model swap, no extra RAM cost). **The default is empty** — single-model compaction is the safe baseline.

**RAM trade-off note:** Using a separate summarization model is only beneficial if both models can be co-resident in RAM (Ollama's default behavior with a generous `OLLAMA_KEEP_ALIVE`). If the configuration would force a model unload/reload cycle, the cost is almost certainly worse than just using the main model: re-mmap'ing a multi-GB model from disk takes longer than the summarization itself. The bakeoff measures cold-load and warm-call latencies for each candidate so this trade-off can be made empirically per model pair.

**Pre-compaction history is always logged to the trace** alongside the resulting summary, so summarization quality can be audited after the fact.

### Tracing (`agent_groundwork/tracing.py`)

Every session writes a JSONL trace to `traces/<timestamp>-<session-id>.jsonl`. Each line is one event:

- Provider call (model, tool_call_mode, message count, prompt token estimate, latency ms, tokens generated)
- Tool call (name, args, result, latency ms)
- Compaction (pre-history snapshot, post-history snapshot, summary text)
- Agent events (TextChunk batches, Done, Error, SystemPromptEdited)

This is the primary debugging artifact. It is append-only, line-oriented, and re-readable without any project code (just `jq`).

### Configuration

A single `config.toml` at the project root drives the system. All fields are validated by a Pydantic config model at load time.

```toml
[agent]
sandbox_root = "./sandbox"
system_prompt_path = "./sandbox/AGENT.md"
max_iterations = 8

[provider]
backend = "ollama"
host = "http://localhost:11434"

[provider.ollama]
keep_alive = "30m"

[model]
name = "gemma4:e4b"
tool_call_mode = "native"  # or "prompted"

[compaction]
trigger_messages = 10
trigger_tokens = 4000
recent_window = 6
summarization_model = ""  # empty = use main model (default)
summary_dir = "./sandbox/conversations"

[tracing]
trace_dir = "./traces"

[bakeoff]
result_dir = "./bakeoff_results"
scenario_dir = "./agent_groundwork/bakeoff/scenarios"
candidate_models = [
  "gemma4:e4b",
  "gemma4:26b",
  "qwen3:8b",
  "ministral-3:14b",
  "gemma3:12b",
]
```

### Sandbox / agent data layout

The agent's view of the world lives at `./sandbox/`:

```
sandbox/
├── AGENT.md                  # system prompt + agent's self-maintained index (committed as starter)
├── .gitignore                # ignores everything except AGENT.md and itself
├── conversations/            # auto-created by compactor
│   └── <session-id>/
│       └── summary.md
├── notes/                    # the agent's organizational space (created on demand)
└── ...                       # whatever the agent creates
```

Only `AGENT.md` and the sandbox's own `.gitignore` are committed. Everything else is the agent's working data and is git-ignored.

`AGENT.md` is loaded as the system prompt at startup. It contains both the agent's instructions *and* a self-maintained index of what's in the sandbox (the agent is instructed to keep this index up to date as it creates files). This is the same pattern as Claude Code's `CLAUDE.md`.

### Frontend (v1: CLI)

The CLI frontend (`agent_groundwork/cli.py`):

- Parses CLI args (config path, model override, etc.)
- Constructs the agent
- Provides a `user_input_provider` that reads from stdin (used by `ask_user`)
- Loops: prompt user → call `agent.run()` → consume event stream → render
- Uses simple ANSI colors to distinguish: user input, agent text, tool calls, tool results, errors
- Writes the trace file via the trace writer

The CLI never calls into the agent core's internals. All communication is through the event stream and the input callback. This is the discipline that makes the future web/Telegram frontends a swap rather than a rewrite.

## Bakeoff

The bakeoff is a re-runnable model evaluation harness. See **IMPLEMENTATION_PLAN.md** for the build order and `agent_groundwork/bakeoff/` for the code.

**Inputs:** a list of models to test, a directory of scenario files.

**Outputs:**
- A JSONL file per `(model, mode, scenario)` cell with raw timing, tokens, parse results, and outputs.
- A markdown report aggregating the JSONL into per-model and per-scenario summaries.

**Scoring axes per cell:**
- Time to first token (ms)
- Total wall-clock latency (ms)
- Tokens generated
- Tokens per second
- Tool call: produced, parseable, correct tool, correct args (each as bool)
- Stopped correctly (didn't keep calling tools after task complete)
- Subjective quality 1-5 (manual review pass)

**Scenarios** (v1 set):

1. `single_tool` — "Write 'hello' to greeting.md."
2. `two_step` — "List the files in the sandbox, then read the one containing 'meeting'."
3. `search_summarize` — "Find any file mentioning Q2 budget and tell me what it says."
4. `multi_step_edit` — "Find the line in tasks.md that says 'draft brief' and mark it done."
5. `no_tool_needed` — "What is the capital of France?" (Should produce a text answer with no tool calls.)
6. `ambiguous_request` — "Update my notes." (Should call `ask_user` to clarify.)
7. `impossible_request` — "Delete all files." (No `delete_file` exists; should refuse or report inability.)
8. `error_recovery` — Pre-seeded with a tool that returns `ok=false`; agent should recover sensibly.
9. `long_context` — A large markdown file is in the sandbox; "summarize section 3."

The bakeoff also records *cold-load* latency for each model (the first call after a fresh Ollama state), separately from warm-call latency. This data feeds the compaction-model decision and the eventual hardware-upgrade decision.

## Module / file layout

```
agent-groundwork/
├── pyproject.toml
├── README.md
├── SPEC.md
├── IMPLEMENTATION_PLAN.md
├── CLAUDE.md
├── .gitignore
├── config.toml
├── agent_groundwork/
│   ├── __init__.py
│   ├── config.py            # pydantic config models + loader
│   ├── paths.py             # sandbox path validation
│   ├── tracing.py           # JSONL trace writer
│   ├── cli.py               # CLI frontend entry point
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py          # Provider protocol, message/result types
│   │   └── ollama.py        # Ollama implementation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py          # Tool protocol, ToolResult, registry
│   │   ├── files.py         # list/read/write/edit/search file tools
│   │   └── interaction.py   # ask_user
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── messages.py      # Message types
│   │   ├── events.py        # Event stream types
│   │   ├── loop.py          # the agent loop
│   │   └── compaction.py    # Compactor protocol + default strategy
│   └── bakeoff/
│       ├── __init__.py
│       ├── harness.py       # runner
│       ├── scenarios/       # scenario YAML files
│       └── report.py        # markdown report generator
├── sandbox/                  # the agent's data directory
│   ├── .gitignore           # ignores everything except AGENT.md
│   └── AGENT.md             # system prompt + index (committed as starter)
├── traces/                   # JSONL session traces (gitignored content)
└── bakeoff_results/          # bakeoff JSONL + markdown reports (gitignored content)
```

## Glossary

- **Agent loop** — the model → tool → model iteration that lets the agent take multiple steps to satisfy a request
- **Bakeoff** — the harness that evaluates multiple models against the same scenarios
- **Compactor** — the component that summarizes older conversation history to keep context size bounded
- **Compaction** — the act of summarizing older history into a single summary message
- **Event stream** — the async iterator the agent core exposes to its frontend
- **Provider** — the abstraction over a specific LLM backend (Ollama in v1)
- **Sandbox** — the directory the agent is confined to for all file operations
- **Tool** — a typed, callable capability the model can invoke to interact with the world
- **`ask_user`** — a tool that pauses the loop to request input from the human
- **`AGENT.md`** — the agent's system prompt and self-maintained index file, living inside the sandbox
- **Trace** — the per-session JSONL log of every event for post-hoc debugging
