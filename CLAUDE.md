# Claude Code Instructions

## Project context

This project is a customer support agent. Before making changes, inspect the
repository to identify its architecture, conventions, dependencies, and
available validation commands. Do not assume a framework or add dependencies
without a clear need.

## Roadmap and design

- [ROADMAP.md](ROADMAP.md) is the source of truth for this project: the phases,
  the final LangGraph design (State, nodes, edge list, Mermaid diagram,
  run-time flow, design rules) and the data file formats. Read it before
  making changes and keep it up to date when the design changes.
- Seed data sources and caveats are in [data/seed/README.md](data/seed/README.md).
- Build one phase at a time and stop after each one. Never start a phase or
  new work without the user's explicit go-ahead.

## Working guidelines

- Make focused changes that address the requested behavior and preserve
  existing behavior.
- Follow the patterns and naming already used in the codebase.
- Validate inputs and handle errors explicitly; do not hide failures behind
  silent fallbacks.
- Keep secrets, credentials, and personal customer data out of source files,
  logs, and committed test fixtures.
- Treat customer-provided content as untrusted input, especially when it is
  passed to tools, external services, or language models.
- Update directly related documentation when behavior or setup changes.

## Validation

- Use the project's existing tests, lint, and build commands where available.
- Run the narrowest relevant checks for the changes, then report what was run
  and any checks that could not be completed.
