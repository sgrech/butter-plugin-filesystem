# butter-plugin-filesystem

Local filesystem access for [butter-agent](https://github.com/sgrech/butter-agent). Stdlib-only at runtime — the filesystem *is* the store; no private database, no network.

A `local-write` plugin: it can navigate, inspect, and read files, and (in later build steps) write, edit, and delete them. It is the worked example for the **operator-config gate** pattern — destructive operations are gated by a `config` flag the model cannot set, not just by a planner gate.

## Capabilities

| Name | Inputs | Output | Notes |
|------|--------|--------|-------|
| `pwd` | — | `{cwd}` | The base directory relative paths resolve against. |
| `cd` | `path` | `{cwd}` | Move the working directory. Absolute, or relative to the current cwd; `~` expands. Raises on missing path / non-directory. |
| `list_dir` | optional `path` | `{path, entries: [{name, type, size}]}` | Defaults to the current cwd. `type` ∈ file/dir/symlink/other; symlink size is the link's own. |
| `stat` | `path` | `{path, type, size, mtime}` | `mtime` is tz-aware ISO-8601. |
| `read_file` | `path`, optional `offset`, `limit` | `{path, content, lines, truncated}` | Line-window read (`offset` 0-based, `limit` default 2000). Raises on a binary (non-UTF-8) file. |

> Later build steps add `write_file`, `edit_file` (exact unique-match replace), `find_files` / `search_content` (ripgrep/fd when present, stdlib fallback), and `delete`.

## Working-directory model & safety

There is **no root jail** — the model may `cd` anywhere; the operator, not a path prefix, draws the boundary. Containment lives on the destructive side (added in later steps):

- `write_file` / `edit_file` are planner-gated (`confirm`).
- `delete` is **off unless** the operator sets `config = { allow_delete = true }` on the plugin's `[[plugin]]` entry, is planner-gated `human` (the operator sees the resolved absolute target before it runs), refuses recursive directory deletion unless additionally `allow_recursive_delete = true`, refuses protected paths, and moves to a `.butter-trash/` rather than hard-deleting by default.

The `allow_delete` flag is read via the host's `PluginContext.config` — operator-declared and model-invisible, so the model cannot talk its way past it.

## Installing into a butter-agent host

External plugins are **opt-in** — declare it in your `config.toml`. Requires butter-agent ≥ `v0.1.0` (the version that introduced `PluginContext.config`). Production (pinned ref):

```toml
[[plugin]]
source = "github.com/sgrech/butter-plugin-filesystem@v0.1.0"
config = { allow_delete = false, allow_recursive_delete = false }
```

Local development (no fetch, used as-is):

```toml
[[plugin]]
path = "~/Workspace/butter-plugin-filesystem"
config = { allow_delete = true }
```

Omitting the `config` table is valid — `delete` is simply unavailable.

## Repo layout

```
butter-plugin-filesystem/
├── pyproject.toml          # uv + hatchling. butter-agent is a *dev* dep
│                           # only (typing + manifest validator).
├── manifest.toml           # Parsed by butter-agent at startup.
├── src/butter_plugin_filesystem/
│   ├── __init__.py         # Re-exports the entrypoint class.
│   └── plugin.py           # FilesystemPlugin satisfies the Plugin Protocol.
├── tests/test_plugin.py    # pytest + asyncio_mode=auto; tmp_path fixtures.
├── justfile                # Quality gates: ruff, mypy --strict, pytest.
├── CLAUDE.md               # Project instructions for AI sessions.
├── README.md               # This file.
└── .gitignore
```

## Development

```bash
uv sync                     # install dev deps (incl. butter-agent editable)
just check                  # ruff + mypy --strict + pytest
just fix                    # auto-fix ruff issues
```

## License

MIT.
