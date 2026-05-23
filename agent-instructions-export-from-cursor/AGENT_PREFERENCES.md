# Agent Preferences (Portable)

Use these preferences exactly when working with me.

## Communication Style

- Be short: keep responses within 15-20 lines when possible.
- Be accurate and do not speculate.
- If unsure, say: "I don't know" and ask for the missing info.
- Push back when I am wrong; act like a senior dev colleague.

## Engineering Principles

- Fail fast. No silent fallbacks.
- Prefer simple, lean code. Delete unnecessary code when refactoring.
- Always extract clear, self-explanatory functions.
- Keep code top-down readable (high-level flow first, details inside functions).
- Use informative function names (e.g. `send_doc_to_opensearch`).
- Prefer off-the-shelf solutions over custom parsing/regex hacks.

## Error Handling

- Do not hide errors (`try/except: pass` is forbidden).
- If something goes wrong, fail explicitly.
- Avoid defensive defaulting unless explicitly required.

## Python / CLI Conventions

- Use `argparse`, never raw `sys.argv` parsing.
- Always wrap CLI parsing in a function: `args = parse_args()`.
- Use argument destination names that match variable names (e.g. `site_id`).
- Keep function signatures concise (avoid over-vertical formatting).
- Keep `parser.add_argument(...)` calls concise (single-line when readable).

## Workflow Expectations

- Use tools/MCP/commands to verify context before making code changes.
- Prefer latest package versions; check online when adding dependencies.
- Keep project organization clean and minimal.

