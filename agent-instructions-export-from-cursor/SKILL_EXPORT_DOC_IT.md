# Skill Export: doc-it

Use this command to document code for beginners.

## Rules

- Add a 2-3 line architecture overview comment at the top.
- Group related code into concept blocks with why/how context.
- Mark critical lines with `⚠️ CRITICAL:` for:
  - control flow changes
  - state updates
  - external service calls
  - design decision points
- Number execution flow checkpoints (`Step 1`, `Step 2.1`, etc.).
- Add a diagram in chat and a concise suggested study order.
- Add debugging aids for complex parts:
  - expected input
  - expected output
  - common failure points
- Skip basic Python syntax explanations.
- Keep explanation concise (roughly <= 1:1 explanation-to-code ratio).

## Reusable Prompt Block

```text
Document this code with beginner-friendly framework focus:
1) Add a 2-3 line architecture overview at top.
2) Add concept-block headers explaining pattern, why it exists, and flow fit.
3) Mark critical lines using "⚠️ CRITICAL:" for flow/state/external calls/design decisions.
4) Number execution flow with Step 1, Step 2.1, etc.
5) Include a Mermaid diagram in chat and a concise recommended study order.
6) For complex operations, annotate expected input/output and common failure points.
7) Skip basic Python syntax explanations; focus on framework/architecture.
8) Keep explanations concise (about 1:1 with code length).
```
