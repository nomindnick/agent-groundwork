# Findings

Empirical observations from real agent sessions. The bakeoff measures models on
scoped tasks; this directory captures what shows up in open-ended use that the
bakeoff misses. Each entry pins the variables that mattered (model, mode,
AGENT.md commit, tools, relevant config) so a future reader can tell whether a
finding still applies after the prompt or config has moved on.

Inclusion bar: *would a future-me or future-Claude need to know this before
touching the prompt, compaction, or tool layer?* If yes, write it up. If it's
just "the model was a bit verbose," skip it.

See `_template.md` for the per-finding skeleton.

## Open

- [2026-04-08 — Hallucinated tool calls in `gemma4:e4b` (both modes)](20260408-hallucinated-tool-calls.md) — Reproduced across four sessions in five shapes (post-hoc, prospective, confabulated detail, session-state amnesia), in both native and prompted mode. Mode-switch hypothesis falsified; bug is model-intrinsic.
- [2026-04-08 — Prompted-mode parser extracts fragments as tool names](20260408-prompted-mode-parser.md) — Code bug in the prompted-mode tool-call extractor: field-by-field pattern matching produces tool names like `\` and `"read_file", "args":` when the model writes bare-JSON tool calls in text. Independent of the hallucination finding.

## Mitigated

_(none yet)_

## Resolved

_(none yet)_
