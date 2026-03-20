# Project Rules

## Package Manager
- **Always use `uv`** — never `pip`, `pip3`, or `pip install`.
- **Never manually edit `pyproject.toml` or `uv.lock`** — use `uv add <pkg>` and `uv remove <pkg>` to manage dependencies; uv handles those files automatically.
