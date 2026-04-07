# agent-groundwork — Implementation Plan

This document describes the *order* in which agent-groundwork is built. It assumes you have read **SPEC.md**.

The plan is structured in phases. Each phase has explicit deliverables, dependencies on prior phases, and "done when" criteria. Phases are sized so that each one is independently usable — the project produces value at the end of every phase, not just at the end.

## High-level shape

```
Phase 0 — Repo skeleton                              (this commit)
   |
Phase 1 — Provider + tool framework                  (foundations)
   |
Phase 2 — Bakeoff harness + first bakeoff run        (model selection)
   |        |
   |        v
   |   DECISION: pick 3-5 finalist models
   |
Phase 3 — Agent core (loop, compactor, tracing)
   |
Phase 4 — CLI frontend + ask_user wiring
   |
Phase 5 — Use it, iterate, refine system prompt
   |
   v
(future) Phase 6 — Web UI
(future) Phase 7 — Telegram frontend
(future) Phase 8 — Real agent #1 (legal worklist or family coordinator)
```

Each "build" phase ends in a usable artifact. Each "use" phase ends in lessons that feed back into earlier phases.

---

## Phase 0 — Repository skeleton

Done as part of the planning conversation that produced SPEC.md, this file, and CLAUDE.md.

**Deliverables:**
- Project directory structure
- `SPEC.md`, `IMPLEMENTATION_PLAN.md`, `CLAUDE.md`
- `pyproject.toml` with chosen dependencies
- `.gitignore`
- Empty Python package layout (`agent_groundwork/...` with `__init__.py` files)
- Starter `config.toml`
- Starter `sandbox/AGENT.md`

**Done when:** the repo can be cloned, Python deps installed (`uv sync` or equivalent), and `python -c "import agent_groundwork"` succeeds.

---

## Phase 1 — Provider and tool framework

Build the two layers that the bakeoff and the agent both depend on. This is the foundation for everything that follows; spend the time to get the abstractions right.

**Deliverables:**

1. **`agent_groundwork/providers/base.py`**
   - `Message`, `ChatResult`, `ChatChunk`, `ToolSchema`, `ToolCall` types (Pydantic models)
   - `Provider` protocol with `chat()` and `stream()` methods
   - `ParseError` and friends for surfacing tool-call failures cleanly

2. **`agent_groundwork/providers/ollama.py`**
   - Ollama HTTP client (use the official `ollama` Python package or raw `httpx` — whichever is simpler at the time)
   - `chat()` and `stream()` implementations
   - Both `tool_call_mode="native"` (uses Ollama's `tools=` parameter) and `tool_call_mode="prompted"` (injects tool descriptions into the system prompt and parses fenced JSON code blocks out of the response)
   - Latency timing per call (time-to-first-token and total)
   - Token count from Ollama's reported counts; estimated when missing
   - Robust parse failure reporting — the bakeoff needs to know *why* a call failed, not just that it did

3. **`agent_groundwork/tools/base.py`**
   - `Tool` protocol
   - `ToolResult` type (`ok`, `data`, `error`)
   - `ToolRegistry` — register tools, look them up by name, render schemas to native or prompted form
   - Schema rendering: takes a Pydantic model and produces both an Ollama-style dict and a markdown description block for prompted mode

4. **`agent_groundwork/paths.py`**
   - `validate_path(root: Path, candidate: str) -> Path` function
   - Resolves to absolute, asserts containment under `root`, rejects symlink escapes
   - Comprehensive unit tests (the one place a unit test is non-optional in this phase — sandbox escape would be a real bug)

5. **`agent_groundwork/tools/files.py`**
   - `list_files`, `read_file`, `write_file`, `edit_file`, `search_files`
   - All paths validated through `paths.py`
   - `search_files` shells out to `rg` if available, falls back to a Python implementation if not
   - `edit_file` uses literal-string match; errors clearly if `old` is missing or matches more than once
   - `write_file` produces a distinguishable trace event when overwriting an existing file

6. **`agent_groundwork/tools/interaction.py`**
   - `ask_user` tool
   - Takes a `question: str`, returns the user's response in `data`
   - Implementation calls into a `user_input_provider` callable injected at construction time. In Phase 1 this is just a stub that returns canned answers; the CLI will provide the real one in Phase 4.

7. **`agent_groundwork/config.py`**
   - Pydantic models matching the `config.toml` schema
   - `load_config(path: str) -> Config` function

**Dependencies:** Phase 0.

**Done when:**
- You can write a 30-line script that loads a tool registry and calls a model with it via either tool-call mode against any installed Ollama model.
- The same script writes a file inside the sandbox and refuses to write outside it.
- `pytest agent_groundwork/paths.py` (or equivalent) passes with sandbox-escape attempts covered.

---

## Phase 2 — Bakeoff harness and first bakeoff run

Use Phase 1's pieces to evaluate the candidate models. The bakeoff is the first artifact that produces a *decision*, not just code.

**Deliverables:**

1. **`agent_groundwork/bakeoff/scenarios/`**
   - One YAML file per scenario from the SPEC list (`single_tool`, `two_step`, `search_summarize`, `multi_step_edit`, `no_tool_needed`, `ambiguous_request`, `impossible_request`, `error_recovery`, `long_context`)
   - Each scenario specifies: name, description, available tools (real or mocked), pre-seeded sandbox state, the user prompt, and a scoring rubric (`expected_tool_calls`, `expected_no_tool_calls`, `expected_to_ask_user`, etc.)

2. **`agent_groundwork/bakeoff/harness.py`**
   - `run_bakeoff(models, scenarios) -> Results`
   - For each `(model, mode, scenario)` cell:
     - Set up a clean temp sandbox with any seeded state
     - Run the model's response (using the provider layer; the bakeoff does *not* go through the agent loop — one-shot per scenario)
     - Capture: time to first token, total latency, tokens generated, tool calls attempted, parseable y/n, correct tool y/n, correct args y/n, raw output text
     - Append to JSONL output
   - Cold-load measurement: a separate run that times the *first* call to each model after a fresh Ollama state

3. **`agent_groundwork/bakeoff/report.py`**
   - Reads the JSONL from a bakeoff run
   - Generates a markdown report with:
     - Per-model summary table (latency, tokens/sec, scenario success rate, tool-call mode that worked best)
     - Per-scenario breakdown (which models succeeded)
     - Cold-load timings
     - Notable failures: raw outputs for hand inspection
   - Output: `bakeoff_results/<timestamp>/report.md` plus the underlying JSONL

4. **Entry point: `python -m agent_groundwork.bakeoff`**
   - Reads candidate models from `config.toml`
   - Runs the harness, generates the report

**Dependencies:** Phase 1.

**Done when:**
- A full bakeoff run completes against the SPEC's proposed shortlist (`gemma4:e4b`, `gemma4:26b`, `qwen3:8b`, `ministral-3:14b`, `gemma3:12b`).
- The generated `report.md` is human-readable and ranks the models on each scoring axis.
- **Decision point:** based on the report, pick **3-5 finalist models** to carry into Phase 3+. Document the choice in `bakeoff_results/<timestamp>/decision.md` with rationale.

---

## Phase 3 — Agent core

Build the loop that the bakeoff has now informed.

**Deliverables:**

1. **`agent_groundwork/agent/messages.py`**
   - `Message` type (role, content, tool_calls, tool_call_id, etc.)
   - Conversion helpers for the provider layer

2. **`agent_groundwork/agent/events.py`**
   - All event types from the SPEC: `TextChunk`, `ToolCallStarted`, `ToolCallResult`, `CompactionEvent`, `SystemPromptEdited`, `Done`, `Error`
   - Each is a small dataclass / Pydantic model

3. **`agent_groundwork/agent/loop.py`**
   - `Agent` class constructed with: provider, tool registry, compactor, tracer, system prompt, config
   - `async def run(user_message: str) -> AsyncIterator[Event]` matching the SPEC pseudocode
   - Iteration cap (configurable, default 8)
   - Tool dispatch with per-tool exception capture → `ToolResult(ok=False, ...)`
   - Detects `edit_file` calls targeting the system prompt path and emits `SystemPromptEdited` with a diff
   - All output flows through events; *no print statements anywhere*

4. **`agent_groundwork/agent/compaction.py`**
   - `Compactor` protocol
   - `RollingSummaryCompactor` default implementation
   - Triggers on message count or estimated token count (whichever first)
   - Calls a configurable summarization model (defaults to main model)
   - Writes summary to `sandbox/conversations/<session-id>/summary.md`
   - Telescopes prior summaries on subsequent compactions
   - **Always logs pre/post snapshots to the trace** for auditing

5. **`agent_groundwork/tracing.py`**
   - `Tracer` class that opens a session JSONL file and writes events
   - Records: provider calls (with timing/tokens), tool calls (with timing), compaction (with pre/post snapshots), all agent events
   - Append-only, line-oriented, no buffering games

6. **Smoketest: `scripts/agent_smoketest.py`**
   - A throwaway script that constructs an agent, runs a single hard-coded user message, and prints events
   - Not the CLI — just enough to verify the loop end-to-end without the real frontend
   - Useful for verifying tool dispatch, compaction triggering, and trace output before Phase 4

**Dependencies:** Phase 1, Phase 2 (you need finalist models picked).

**Done when:**
- The smoketest runs an agent against a real Ollama model with real file tools, completes a multi-step task, and produces a JSONL trace.
- Forcing the iteration cap (e.g., a deliberately impossible request) produces a clean `Error` event, not a hang.
- Triggering compaction (by feeding many messages) produces a `summary.md` file in the session directory and re-uses it on the next call.

---

## Phase 4 — CLI frontend + ask_user wiring

Wrap the agent core in a usable interface.

**Deliverables:**

1. **`agent_groundwork/cli.py`**
   - `python -m agent_groundwork` entry point
   - Loads config, constructs provider/tools/compactor/tracer/agent
   - Provides a `user_input_provider` that reads from stdin (used by `ask_user`)
   - REPL loop: prompt user → run agent → render events
   - ANSI color rendering:
     - User prompts: bold
     - Agent text: default
     - Tool calls: cyan, with arg pretty-print
     - Tool results: green if ok, red if error
     - Compaction: dim gray
     - System prompt edits: yellow with the diff
     - Errors: bold red
   - Handles `Ctrl-C` cleanly (cancels in-flight, returns to prompt)
   - On exit, prints the trace file path

2. **Wire `ask_user` to the CLI's input provider**
   - When the agent calls `ask_user`, the dispatcher invokes the provider, which prints the question and reads stdin
   - The result becomes the tool's `ToolResult.data`

3. **Realistic `sandbox/AGENT.md`**
   - The starter file is a placeholder. Phase 4 writes the *real* version: who the agent is, what tools it has, when to ask vs guess, the convention that it maintains its own index, behavior on ambiguous requests, etc.
   - System prompt iteration is the main activity in Phase 5 — Phase 4 just produces the first version that's good enough to start using.

**Dependencies:** Phase 3.

**Done when:**
- `python -m agent_groundwork` starts a chat where you can ask the agent to write notes, search them, edit them, etc., and it works on at least one finalist model.
- The agent successfully calls `ask_user` mid-task and the conversation continues after you answer.
- Trace files are written and inspectable.

---

## Phase 5 — Use it, iterate

Not really a "build" phase. This is the use-and-tune phase that produces the actual learnings the project exists for.

**Activities:**
- Use the agent for a week of low-stakes tasks (note-taking, brainstorming, organizing the sandbox itself)
- Re-run the bakeoff if any new models drop, or if you change tool definitions
- Iterate on `AGENT.md`: where does the agent guess wrong? Where does it forget what it has? Where is the compactor losing important state?
- Adjust compaction triggers, recent window size, summarization model
- Try the finalist models in turn — does any one pull ahead in actual use?
- Consider new tools as concrete needs emerge (e.g., did you wish for a `move_file`? Add it.)

**Done when:** you have a clear answer to "is this approach worth investing in?" If yes, proceed to Phase 6 and beyond. If no, the bakeoff and trace logs are still standalone artifacts — the project paid for itself.

---

## Future phases (out of scope for v1)

Captured here so the v1 architecture doesn't accidentally preclude them.

### Phase 6 — Web UI

- FastAPI app exposing the agent over Server-Sent Events
- Single HTML page with a chat UI and rich tool-call rendering (collapsible blocks, syntax highlighting)
- Same agent core; new frontend that consumes the same event stream
- `user_input_provider` backed by an HTTP request/response cycle

### Phase 7 — Telegram frontend

- `python-telegram-bot` wrapper that consumes the same event stream
- Maps events to Telegram messages
- Multi-user (you and your wife) requires per-chat-id session isolation
- This is when "session lifecycle" stops being a single-process REPL and starts being persistent

### Phase 8 — Real agent #1

- Either the legal worklist agent or the family coordinator
- Built as a *configuration* of agent-groundwork, not a fork
- New tools: calendar, scheduling, whatever the use case demands
- New system prompt
- New sandbox layout convention
- This is the test of whether the v1 abstractions actually held up

---

## Things explicitly *not* in any phase (yet)

- LangChain / LlamaIndex / Letta / any agent framework
- RAG / embeddings (deferred until grep is observably insufficient)
- A test suite beyond `paths.py` unit tests and the bakeoff
- Production observability (metrics, dashboards)
- Authentication / multi-tenant
- Background jobs / scheduled execution
- Streaming voice
- Function-calling fine-tuning
