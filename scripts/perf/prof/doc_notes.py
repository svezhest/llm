"""The project notes the perf scripts build their long text prompts from, in a
fixed order (paths relative to the repo root). Since the CLAUDE.md split
(docs/<topic>.md) prompts built from this list differ from the ones before it."""

TOPICS = ("knobs", "prompt-cache", "scheduler", "api", "drafters", "kernels", "prefill",
          "decode", "models", "measuring", "layout", "MAP")
NOTES = ("README.md", "CLAUDE.md") + tuple(f"docs/{t}.md" for t in TOPICS) + ("BACKLOG.md",)
