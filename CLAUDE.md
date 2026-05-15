# butter-plugin-filesystem

Local filesystem access for [butter-agent](https://github.com/sgrech/butter-agent). The reference plugin for the **operator-config gate** pattern.

This file is loaded into every Claude Code session here. Keep it short — the user's global `~/.claude/CLAUDE.md` covers session-start, work-loop, git/PR attribution, and MCP usage. Only project-specific guidance belongs here.

## The plugin contract

butter-agent loads `manifest.toml` from this repo's root and resolves `entrypoint = "module:Class"` to a class whose instances satisfy the `Plugin` Protocol (`butter_agent.plugin_api.Plugin`):

```python
class Plugin(Protocol):
    async def execute(self, capability: str, inputs: dict[str, object], context: PluginContext) -> dict[str, object]: ...
```

Three args after `self` — `capability, inputs, context`. The Protocol is structural — `FilesystemPlugin` does not inherit from it. `butter_agent.plugin_api` is imported only under `TYPE_CHECKING` (for `PluginContext`) and at test time (`parse_manifest`, `Plugin`); runtime stays stdlib-only.

## Architecture invariants

1. **Filesystem-backed, no `requires`.** Unlike `notes`, this plugin owns its side effects directly (it *is* the writer) and never calls another plugin. `manifest.toml` declares `blast_radius = "local-write"` and no `requires`.
2. **Plugin-level blast radius.** `blast_radius` is declared once for the whole plugin and must cover its most destructive capability. Read/navigate/search ride under the same `local-write` tier as write/edit/delete — do not try to express per-capability radii (the manifest has no such field).
3. **No root jail; containment is on the destructive side.** The model may `cd` anywhere. `_cwd` (process CWD at construction) only moves via the `cd` capability; every other capability resolves a relative `path` against it through the single `_resolve` seam. Boundaries are drawn by the operator, not a path prefix.
4. **`delete` is operator-gated, not model-gated.** `delete` reads `context.config['allow_delete']` (requires butter-agent ≥ `v0.1.0`). The model cannot set this — it is operator-declared in `config.toml` and surfaced read-only via `PluginContext.config`. Recursive directory deletion additionally requires `allow_recursive_delete`. Default behaviour is move-to-`.butter-trash`, not hard delete. Protected paths (fs root, `$HOME`, the cwd, very short paths, symlink escapes) are refused.
5. **Contract breaks surface as `FilesystemPluginError`.** Input-validation failures, missing paths, wrong types, binary reads, and refused destructive ops all raise a descriptive `FilesystemPluginError`, never a bare `OSError`. The host records it as the step `failure_reason`; the loop still synthesises.
6. **Manifest is the source of truth for shape.** Capability names/inputs/outputs must round-trip through the host's `parse_manifest` (`tests/test_plugin.py::test_manifest_round_trips_through_butter_validator`).

## Search backend (build step 4)

`find_files` / `search_content` shell out to `rg` / `fd` *if present on PATH* (`shutil.which`) for `.gitignore`-aware, fast, structured (`--json` / `-0`) results, falling back to a pure-stdlib `os.walk` + `re` + `fnmatch` scan when absent. No third-party dependency — the binaries are an optimisation, never a requirement. The fallback skips `.git` and common VCS/build dirs but is **not** `.gitignore`-aware (documented limitation).

## Edit strategy (build step 3)

`edit_file` is an **exact unique-match string replace**: read the file, require `old` to occur exactly once (raise on 0 or >1 — predictable failure), replace, write atomically (temp file in the same dir + `os.replace`). `write_file` is a full atomic create/overwrite. No diff/patch format.

## Development workflow

- `uv sync` — install dev deps (editable butter-agent for the Protocol types + manifest validator).
- `just check` — ruff (lint + format check) + mypy --strict + pytest.
- `just fix` — auto-fix ruff.

CI parity = `just check` green locally before pushing. No separate CI config.

## Versioning

Semver. Three version sources must move together in one commit: `[plugin].version` in `manifest.toml`, `__version__` in `__init__.py`, and `[project].version` in `pyproject.toml`. The host pins `[[plugin]] source = "...@vX.Y.Z"` against it. This plugin requires a host providing `PluginContext.config` (butter-agent ≥ `v0.1.0`).
