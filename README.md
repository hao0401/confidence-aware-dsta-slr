# Project Layout

This workspace mixes the main `DSTA-SLR` codebase with paper-revision utilities and generated artifacts.

## Top-level folders

- `DSTA-SLR/`: main training, model, config, and experiment scripts.
- `tools/paper_revision/`: standalone DOCX cleanup and reviewer-alignment scripts.
- `docs/paper_revision/`: revision notes and submission checklists.
- `output/`: generated outputs and exported assets.
- `tmp/`: temporary scratch files and one-off intermediate results.

## Working conventions

- Put reusable code under `DSTA-SLR/` or `tools/`, not in the repository root.
- Put notes and checklists under `docs/`.
- Keep generated files in `output/` or `tmp/`.
- Cache folders such as `__pycache__/` and temporary artifacts are ignored by Git.
