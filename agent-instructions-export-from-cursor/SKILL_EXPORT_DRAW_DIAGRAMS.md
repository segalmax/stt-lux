# Skill Export: draw-diagrams

Use this skill when creating architecture/flow diagrams.

## Rules

- Use Mermaid with theme init:
  `%%{ init: { 'theme': 'base' } }%%`
- Prefer light backgrounds + dark text.
- Number flow steps (`1`, `2`, `3.a`, `3.b`).
- Use meaningful shapes (actor/service/db/queue/decision).
- Keep consistent colors by element category.
- Use `<br/>` for node line breaks.
- Do not use HTML entities like `&gt;`; use literal `>`.
- Prefer `direction LR` inside subgraphs, outer graph `TD`.
- Avoid cycle arrows that break layout.
- Keep storage nodes outside pipeline subgraphs.
- Arrow direction should represent call initiator.

## Reusable Prompt Block

```text
Create a clear Mermaid diagram with these constraints:
1) Always start with: %%{ init: { 'theme': 'base' } }%%
2) Use readable light fills + dark text.
3) Number each step in execution order.
4) Use meaningful node shapes and consistent class colors by category.
5) Use <br/> in labels; never HTML entities like &gt;.
6) Use outer TD layout, LR inside subgraphs.
7) Avoid cycles unless absolutely required.
8) Keep storage nodes outside pipelines.
9) Arrows show who initiates the call.
```
