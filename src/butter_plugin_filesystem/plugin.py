"""FilesystemPlugin — local filesystem access for butter-agent.

A `local-write` plugin that lets the agent navigate, inspect, and read
files on the local disk (write / edit / delete arrive in later build
steps). Unlike `notes`, it owns no shared store and declares no
`requires`: it never calls another plugin. Persistence *is* the
filesystem.

The plugin satisfies `butter_agent.plugin_api.Plugin` structurally. It is
not typed against that Protocol explicitly and imports `PluginContext`
only under `TYPE_CHECKING`, so the package can be loaded into a
butter-agent install without importing butter at runtime — runtime stays
stdlib-only (`os`, `pathlib`, `datetime`).

Working-directory model: the plugin holds a mutable `_cwd` (the process
CWD at construction). The model moves it explicitly via the `cd`
capability; every other capability resolves a relative `path` against it.
There is deliberately no root jail — the operator, not a path prefix,
draws the boundary. Containment lives entirely on the destructive side
(write/edit/delete, added in later steps): those are planner-gated and,
for delete, additionally gated by an operator `config` flag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from butter_agent.plugin_api import PluginContext

#: A read with no explicit `limit` still caps returned lines so a
#: multi-GB file can never be pulled wholesale into a model turn. A
#: caller that needs more pages by passing an explicit larger `limit`
#: (and `offset` to page through).
_DEFAULT_READ_LINES: Final = 2000


class FilesystemPluginError(Exception):
    """Raised on any malformed or refused `filesystem.*` call.

    Propagates out of `execute`; the host's task executor catches it on
    its broad plugin-failure path and records it as the step's
    `failure_reason` (a plugin may raise for any reason and must not tear
    the loop down). Used for input-validation failures, missing paths,
    wrong path types, binary reads, and (in later build steps) refused
    destructive operations.
    """


class FilesystemPlugin:
    """`Plugin` Protocol implementation backed by the local filesystem.

    Holds a single piece of state — `_cwd`, the base directory relative
    paths resolve against. It starts at the process working directory and
    only ever changes through an explicit `cd` capability call, so path
    resolution is predictable across a multi-step plan.
    """

    def __init__(self) -> None:
        self._cwd: Path = Path.cwd()

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: PluginContext,
    ) -> dict[str, object]:
        # `context` is unused for read/navigate capabilities — they touch
        # only the local filesystem and call no other plugin. It is read
        # in later build steps (delete consults `context.config`).
        del context
        if capability == 'pwd':
            return self._pwd()
        if capability == 'cd':
            return self._cd(inputs)
        if capability == 'list_dir':
            return self._list_dir(inputs)
        if capability == 'stat':
            return self._stat(inputs)
        if capability == 'read_file':
            return self._read_file(inputs)
        raise FilesystemPluginError(
            f'unknown capability {capability!r} (expected one of: pwd, cd, list_dir, stat, read_file)',
        )

    # --- path resolution -----------------------------------------------------

    def _resolve(self, raw: object, *, key: str = 'path') -> Path:
        """Resolve a caller-supplied path against the current cwd.

        `~` is expanded; a relative path is joined onto `_cwd`; the result
        is normalised with `Path.resolve()` so `..` segments and symlinks
        collapse to a real absolute location. Resolution never touches the
        filesystem beyond what `resolve()` does and never asserts
        existence — capabilities check existence/type themselves so each
        can phrase its own error.
        """
        if not isinstance(raw, str) or not raw:
            raise FilesystemPluginError(f'input {key!r} must be a non-empty string, got {raw!r}')
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        return candidate.resolve()

    @staticmethod
    def _kind(path: Path) -> str:
        """Classify a path: symlink (checked first), dir, file, or other."""
        if path.is_symlink():
            return 'symlink'
        if path.is_dir():
            return 'dir'
        if path.is_file():
            return 'file'
        return 'other'

    # --- capabilities --------------------------------------------------------

    def _pwd(self) -> dict[str, object]:
        return {'cwd': str(self._cwd)}

    def _cd(self, inputs: dict[str, object]) -> dict[str, object]:
        target = self._resolve(inputs.get('path'))
        if not target.exists():
            raise FilesystemPluginError(f'no such path: {target}')
        if not target.is_dir():
            raise FilesystemPluginError(f'not a directory: {target}')
        self._cwd = target
        return {'cwd': str(self._cwd)}

    def _list_dir(self, inputs: dict[str, object]) -> dict[str, object]:
        raw = inputs.get('path')
        target = self._resolve(raw) if raw is not None else self._cwd
        if not target.exists():
            raise FilesystemPluginError(f'no such path: {target}')
        if not target.is_dir():
            raise FilesystemPluginError(f'not a directory: {target}')
        entries: list[dict[str, object]] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            # lstat: never follow a symlink for its size — report the
            # link itself, consistent with `_kind` flagging it 'symlink'.
            size = child.lstat().st_size
            entries.append({'name': child.name, 'type': self._kind(child), 'size': size})
        return {'path': str(target), 'entries': entries}

    def _stat(self, inputs: dict[str, object]) -> dict[str, object]:
        target = self._resolve(inputs.get('path'))
        if not target.exists():
            raise FilesystemPluginError(f'no such path: {target}')
        info = target.lstat()
        mtime = datetime.fromtimestamp(info.st_mtime, tz=UTC).isoformat()
        return {
            'path': str(target),
            'type': self._kind(target),
            'size': info.st_size,
            'mtime': mtime,
        }

    def _read_file(self, inputs: dict[str, object]) -> dict[str, object]:
        target = self._resolve(inputs.get('path'))
        if not target.exists():
            raise FilesystemPluginError(f'no such path: {target}')
        if not target.is_file():
            raise FilesystemPluginError(f'not a regular file: {target}')

        offset = _non_negative_int(inputs.get('offset'), 'offset', default=0)
        limit = _non_negative_int(inputs.get('limit'), 'limit', default=_DEFAULT_READ_LINES)

        try:
            text = target.read_text(encoding='utf-8')
        except UnicodeDecodeError as exc:
            raise FilesystemPluginError(f'not a UTF-8 text file: {target}') from exc

        all_lines = text.splitlines(keepends=True)
        window = all_lines[offset : offset + limit]
        truncated = offset > 0 or len(window) < len(all_lines)
        return {
            'path': str(target),
            'content': ''.join(window),
            'lines': len(window),
            'truncated': truncated,
        }


def _non_negative_int(value: object, key: str, *, default: int) -> int:
    """Coerce an optional non-negative-int input, rejecting bools.

    `isinstance(True, int)` is True in Python — the explicit bool guard
    stops a stray `true`/`false` masquerading as 1/0 (same stance as the
    notes plugin's limit handling).
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FilesystemPluginError(f'input {key!r} must be a non-negative integer, got {value!r}')
    return value
