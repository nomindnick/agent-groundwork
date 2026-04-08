# {{ short title }}

## Context

- Date: YYYY-MM-DD
- Model: {{ name }} ({{ native | prompted }})
- AGENT.md: {{ short hash }} — {{ 1-2 line note on what was notable about this version }}
- Tools: {{ comma-separated tool names }}
- Relevant config: {{ only the values that matter for this finding, e.g. compaction.trigger_messages=10 }}
- Session id: {{ session id, for cross-reference with local traces/ }}

## What happened

{{ 3-5 sentences of plain prose. The story, not the analysis. }}

## Evidence

{{ Excerpted JSONL lines from the trace — only the lines that matter, not the
whole file. Inline them in a fenced block; do not link to traces/ since that
directory is gitignored and session ids will mean nothing in six months. }}

```jsonl
```

## Implications

{{ What this changes about the prompt, compaction settings, tool design, or
what to watch for next time. This is the part that earns the file its keep. }}
