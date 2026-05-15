"""FilesystemPlugin — local filesystem access for butter-agent.

A `local-write` plugin that lets the agent navigate, inspect, read,
write, edit, search, and (operator-gated) delete files on the local
disk. Unlike `notes`, it owns no shared store and declares no
`requires`: it never calls another plugin. Persistence *is* the
filesystem.

Search delegates to `fd` / `ripgrep` when they are on PATH (fast,
.gitignore-aware, structured output) and falls back to a pure-stdlib
walk otherwise — the binaries are an optional speed/accuracy upgrade,
never a packaged dependency.

`delete` is the one capability with teeth and the reference for the
operator-config gate: it refuses unless the operator set
`allow_delete = true` (read via the host's model-invisible
`PluginContext.config`), additionally requires `allow_recursive_delete`
+ an explicit `recursive` input for a non-empty directory, supports a
non-mutating `dry_run` preview, unconditionally refuses blast-radius
catastrophes (fs root, `$HOME`, the cwd or their ancestors), and moves
to a recoverable `.butter-trash/` rather than hard-deleting by default.

The plugin satisfies `butter_agent.plugin_api.Plugin` structurally. It is
not typed against that Protocol explicitly and imports `PluginContext`
only under `TYPE_CHECKING`, so the package can be loaded into a
butter-agent install without importing butter at runtime — runtime stays
stdlib-only (`os`, `pathlib`, `datetime`).

Working-directory model: the plugin holds a mutable `_cwd` (the process
CWD at construction). The model moves it explicitly via the `cd`
capability; every other capability resolves a relative `path` against it.
There is deliberately no root jail — the operator, not a path prefix,
draws the boundary. Containment lives entirely on the destructive side:
write/edit are planner-gated; delete is additionally operator-gated via
`config` and has unconditional protected-path backstops.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
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

#: Default result caps for the search capabilities — keep a model turn
#: bounded even on a huge tree; callers raise them with explicit `limit`.
_DEFAULT_FIND_LIMIT: Final = 1000
_DEFAULT_SEARCH_LIMIT: Final = 200

#: External binaries used opportunistically. Absence is normal — a
#: pure-stdlib fallback always exists; the binary is a speed/.gitignore
#: upgrade, never a dependency.
_FD_BIN: Final = 'fd'
_RG_BIN: Final = 'rg'

#: Subprocess wall-clock ceiling. A search that cannot finish in this
#: long on a local tree is pathological — fail loudly rather than hang
#: the agent turn.
_SUBPROCESS_TIMEOUT: Final = 30.0

#: Directories the stdlib fallback never descends into. ripgrep / fd get
#: this (and full .gitignore semantics) for free; the fallback is not
#: .gitignore-aware (documented limitation) so it at least skips the
#: universally-noise VCS/build/cache dirs.
_SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {
        '.git',
        '.hg',
        '.svn',
        'node_modules',
        '__pycache__',
        '.venv',
        'venv',
        '.mypy_cache',
        '.pytest_cache',
        '.ruff_cache',
        '.tox',
        '.idea',
        'dist',
        'build',
    },
)

#: Fallback content scan skips files larger than this — a multi-hundred-MB
#: blob is almost never what a text search wants and reading it stalls the
#: turn. ripgrep applies its own binary/size heuristics.
_MAX_SCAN_BYTES: Final = 5 * 1024 * 1024

#: Recoverable-delete directory. A deleted target is moved into one of
#: these at its own parent (same filesystem, so the move is a rename),
#: timestamped to avoid collisions. The operator empties it manually.
_TRASH_DIRNAME: Final = '.butter-trash'

#: A path with this few components is a filesystem root or a top-level
#: system directory (`/`, `/Users`, `/etc`, `/home`). `delete` refuses
#: anything at or below this depth regardless of other flags — the
#: blast-radius backstop against a catastrophic recursive wipe.
_MIN_DELETE_PARTS: Final = 3


class FilesystemPluginError(Exception):
    """Raised on any malformed or refused `filesystem.*` call.

    Propagates out of `execute`; the host's task executor catches it on
    its broad plugin-failure path and records it as the step's
    `failure_reason` (a plugin may raise for any reason and must not tear
    the loop down). Used for input-validation failures, missing paths,
    wrong path types, binary reads, and refused destructive operations
    (delete disabled, recursion not permitted, protected path).
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
        # `delete` is the only capability that consults the host —
        # specifically `context.config` for the operator's destructive
        # opt-in (model-invisible by design). Handle it before `context`
        # is dropped; every other capability touches only the local
        # filesystem and calls no other plugin.
        if capability == 'delete':
            return self._delete(inputs, context)
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
        if capability == 'write_file':
            return self._write_file(inputs)
        if capability == 'edit_file':
            return self._edit_file(inputs)
        if capability == 'find_files':
            return self._find_files(inputs)
        if capability == 'search_content':
            return self._search_content(inputs)
        raise FilesystemPluginError(
            f'unknown capability {capability!r} (expected one of: pwd, cd, list_dir, stat, read_file, write_file, edit_file, find_files, search_content, delete)',
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

    def _resolve_unfollowed(self, raw: object, *, key: str = 'path') -> Path:
        """Like `_resolve`, but never follows a *final* symlink.

        Delete must act on the path the caller named, not whatever it
        points at — resolving a symlink and then deleting would remove
        the link's target (a silent, dangerous escape). Intermediate
        symlinks in the parent chain are still resolved so the
        protected-path checks see a real location.
        """
        if not isinstance(raw, str) or not raw:
            raise FilesystemPluginError(f'input {key!r} must be a non-empty string, got {raw!r}')
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        return candidate.parent.resolve() / candidate.name

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

    @staticmethod
    def _read_text(target: Path) -> str:
        """Read a path as UTF-8, raising the binary-file error on failure.

        Shared by `read_file` and `edit_file` so both reject a non-UTF-8
        file identically rather than one corrupting it on write-back.
        """
        try:
            return target.read_text(encoding='utf-8')
        except UnicodeDecodeError as exc:
            raise FilesystemPluginError(f'not a UTF-8 text file: {target}') from exc

    @staticmethod
    def _atomic_write(target: Path, content: str) -> int:
        """Write `content` to `target` atomically; return bytes written.

        A sibling temp file in the *same directory* is written, flushed,
        fsync'd, then `os.replace`'d over the target — `os.replace` is
        atomic within a filesystem, so a concurrent reader sees either the
        old file or the new one, never a truncated mix. The temp file is
        cleaned up if anything fails before the rename.
        """
        data = content.encode('utf-8')
        directory = target.parent
        fd, tmp_name = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=directory)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise FilesystemPluginError(f'failed to write {target}: {exc}') from exc
        return len(data)

    def _read_file(self, inputs: dict[str, object]) -> dict[str, object]:
        target = self._resolve(inputs.get('path'))
        if not target.exists():
            raise FilesystemPluginError(f'no such path: {target}')
        if not target.is_file():
            raise FilesystemPluginError(f'not a regular file: {target}')

        offset = _non_negative_int(inputs.get('offset'), 'offset', default=0)
        limit = _non_negative_int(inputs.get('limit'), 'limit', default=_DEFAULT_READ_LINES)

        text = self._read_text(target)
        all_lines = text.splitlines(keepends=True)
        window = all_lines[offset : offset + limit]
        truncated = offset > 0 or len(window) < len(all_lines)
        return {
            'path': str(target),
            'content': ''.join(window),
            'lines': len(window),
            'truncated': truncated,
        }

    def _write_file(self, inputs: dict[str, object]) -> dict[str, object]:
        target = self._resolve(inputs.get('path'))
        content = inputs.get('content')
        if not isinstance(content, str):
            raise FilesystemPluginError(f"input 'content' must be a string, got {content!r}")
        if target.is_dir():
            raise FilesystemPluginError(f'is a directory, refusing to overwrite: {target}')
        if not target.parent.is_dir():
            # Creating intermediate directories is a separate, explicit
            # concern — keep write predictable and fail loudly rather
            # than silently materialising a tree.
            raise FilesystemPluginError(f'parent directory does not exist: {target.parent}')
        bytes_written = self._atomic_write(target, content)
        return {'path': str(target), 'bytes_written': bytes_written}

    def _edit_file(self, inputs: dict[str, object]) -> dict[str, object]:
        target = self._resolve(inputs.get('path'))
        old = inputs.get('old')
        new = inputs.get('new')
        if not isinstance(old, str) or not old:
            raise FilesystemPluginError(f"input 'old' must be a non-empty string, got {old!r}")
        if not isinstance(new, str):
            # Empty `new` is valid — it deletes the matched text.
            raise FilesystemPluginError(f"input 'new' must be a string, got {new!r}")
        if not target.exists():
            raise FilesystemPluginError(f'no such path: {target}')
        if not target.is_file():
            raise FilesystemPluginError(f'not a regular file: {target}')

        text = self._read_text(target)
        occurrences = text.count(old)
        if occurrences == 0:
            raise FilesystemPluginError(f'old string not found in {target} (nothing replaced)')
        if occurrences > 1:
            raise FilesystemPluginError(
                f'old string occurs {occurrences} times in {target}; it must match exactly once — include more surrounding context to make it unique',
            )
        self._atomic_write(target, text.replace(old, new, 1))
        return {'path': str(target), 'replaced': 1}

    # --- search --------------------------------------------------------------

    def _search_base(self, inputs: dict[str, object]) -> Path:
        """Resolve and validate the directory a search runs under."""
        raw = inputs.get('path')
        base = self._resolve(raw) if raw is not None else self._cwd
        if not base.exists():
            raise FilesystemPluginError(f'no such path: {base}')
        if not base.is_dir():
            raise FilesystemPluginError(f'not a directory: {base}')
        return base

    def _find_files(self, inputs: dict[str, object]) -> dict[str, object]:
        pattern = inputs.get('pattern')
        if not isinstance(pattern, str) or not pattern:
            raise FilesystemPluginError(f"input 'pattern' must be a non-empty string, got {pattern!r}")
        base = self._search_base(inputs)
        limit = _non_negative_int(inputs.get('limit'), 'limit', default=_DEFAULT_FIND_LIMIT)

        fd_bin = shutil.which(_FD_BIN)
        if fd_bin is not None:
            return {'paths': _fd_find(fd_bin, pattern, base, limit), 'backend': 'fd'}
        return {'paths': _walk_find(pattern, base, limit), 'backend': 'stdlib'}

    def _search_content(self, inputs: dict[str, object]) -> dict[str, object]:
        query = inputs.get('query')
        if not isinstance(query, str) or not query:
            raise FilesystemPluginError(f"input 'query' must be a non-empty string, got {query!r}")
        glob = inputs.get('glob')
        if glob is not None and (not isinstance(glob, str) or not glob):
            raise FilesystemPluginError(f"input 'glob' must be a non-empty string when given, got {glob!r}")
        # Pre-compile with Python's `re` regardless of backend: it gives a
        # single, predictable "invalid regex" error and is the matcher the
        # stdlib fallback uses. ripgrep's regex dialect is a near-superset
        # for the common cases the agent emits; exotic divergences are an
        # accepted v1 limitation (documented in the README).
        try:
            regex = re.compile(query)
        except re.error as exc:
            raise FilesystemPluginError(f'invalid regular expression {query!r}: {exc}') from exc
        base = self._search_base(inputs)
        limit = _non_negative_int(inputs.get('limit'), 'limit', default=_DEFAULT_SEARCH_LIMIT)

        rg_bin = shutil.which(_RG_BIN)
        if rg_bin is not None:
            return {'matches': _rg_search(rg_bin, query, glob, base, limit), 'backend': 'ripgrep'}
        return {'matches': _walk_search(regex, glob, base, limit), 'backend': 'stdlib'}

    # --- delete --------------------------------------------------------------

    @staticmethod
    def _flag(context: PluginContext, key: str) -> bool:
        """Read a strict boolean opt-in from the operator's plugin config.

        Only an explicit `true` counts — a missing key, a non-bool, or
        any other value is off. The model cannot influence this:
        `context.config` is operator-declared and surfaced read-only by
        the host (butter-agent >= v0.1.0). This is the gate that makes
        `delete` an operator decision, not a model one.
        """
        return context.config.get(key) is True

    def _assert_not_protected(self, target: Path) -> None:
        """Refuse blast-radius-catastrophic targets, flags notwithstanding.

        These checks are unconditional — no input or config value can
        re-enable them. They are the backstop against an `rm -rf /`-class
        mistake: the filesystem root and top-level system dirs (too few
        path components), `$HOME` or any ancestor of it, the working
        directory or any ancestor of it, and the trash dir itself.
        """
        real = target.resolve()
        if len(real.parts) < _MIN_DELETE_PARTS:
            raise FilesystemPluginError(f'refusing to delete {real}: filesystem root or top-level system path')
        home = Path.home().resolve()
        if real == home or real in home.parents:
            raise FilesystemPluginError(f'refusing to delete {real}: it is your home directory or an ancestor of it')
        cwd = self._cwd.resolve()
        if real == cwd or real in cwd.parents:
            raise FilesystemPluginError(f'refusing to delete {real}: it is the working directory or an ancestor of it')
        if real.name == _TRASH_DIRNAME:
            raise FilesystemPluginError(f'refusing to delete the trash directory itself: {real}')

    def _trash(self, target: Path) -> None:
        """Move `target` into a `.butter-trash/` beside it (recoverable).

        The trash dir lives at `target.parent`, so the move is a rename
        within one filesystem (atomic, no copy). A UTC timestamp suffix
        keeps repeated deletes of the same name from colliding.
        """
        trash_dir = target.parent / _TRASH_DIRNAME
        stamp = datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S%f')
        try:
            trash_dir.mkdir(exist_ok=True)
            shutil.move(str(target), str(trash_dir / f'{target.name}.{stamp}'))
        except OSError as exc:
            raise FilesystemPluginError(f'failed to move {target} to trash: {exc}') from exc

    @staticmethod
    def _hard_delete(target: Path, *, is_dir: bool) -> None:
        try:
            if is_dir:
                shutil.rmtree(target)
            else:
                # A symlink reaches here (is_dir is False for links):
                # unlink removes the link, never its target.
                target.unlink()
        except OSError as exc:
            raise FilesystemPluginError(f'failed to delete {target}: {exc}') from exc

    def _delete(self, inputs: dict[str, object], context: PluginContext) -> dict[str, object]:
        if not self._flag(context, 'allow_delete'):
            raise FilesystemPluginError(
                "delete is disabled: the operator has not set allow_delete = true in this plugin's config",
            )
        target = self._resolve_unfollowed(inputs.get('path'))
        recursive = _opt_bool(inputs.get('recursive'), 'recursive')
        dry_run = _opt_bool(inputs.get('dry_run'), 'dry_run')

        # A broken symlink doesn't `exist()` but is still a real thing to
        # remove — accept it if the path is a symlink either way.
        if not target.exists() and not target.is_symlink():
            raise FilesystemPluginError(f'no such path: {target}')
        self._assert_not_protected(target)

        is_symlink = target.is_symlink()
        is_dir = target.is_dir() and not is_symlink
        entry_count = sum(1 for _ in target.iterdir()) if is_dir else 0
        if is_dir and entry_count > 0:
            # The two-key lock on `rm -rf`: the model must explicitly ask
            # for recursion AND the operator must have allowed it. Either
            # missing => refuse.
            if not recursive:
                raise FilesystemPluginError(
                    f'{target} is a non-empty directory ({entry_count} entries); pass recursive=true to delete it',
                )
            if not self._flag(context, 'allow_recursive_delete'):
                raise FilesystemPluginError(
                    f'recursive delete of {target} refused: the operator has not set allow_recursive_delete = true',
                )

        # Hard delete only when the operator opted into it; otherwise the
        # target is recoverable from trash (scope decision B3.5).
        hard = self._flag(context, 'allow_recursive_delete')
        recursive_effective = bool(is_dir and entry_count > 0)

        if dry_run:
            # The verification preview: every safety check above has run,
            # nothing was mutated. The planner chains this before the
            # `human`-gated real delete so the operator sees the resolved
            # absolute target and its scale.
            return {
                'path': str(target),
                'dry_run': True,
                'trashed': not hard,
                'recursive': recursive_effective,
                'is_dir': is_dir,
                'entries': entry_count,
            }

        if hard:
            self._hard_delete(target, is_dir=is_dir)
        else:
            self._trash(target)
        return {
            'path': str(target),
            'dry_run': False,
            'trashed': not hard,
            'recursive': recursive_effective,
        }


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an external search binary, mapping failure modes to plugin errors."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FilesystemPluginError(f'{cmd[0]} timed out after {_SUBPROCESS_TIMEOUT}s') from exc
    except OSError as exc:
        raise FilesystemPluginError(f'failed to run {cmd[0]}: {exc}') from exc


def _fd_find(fd_bin: str, pattern: str, base: Path, limit: int) -> list[str]:
    """`fd` backend: glob match, files only, absolute, NUL-delimited."""
    result = _run(
        [fd_bin, '--glob', '--type', 'f', '--absolute-path', '--color', 'never', '--print0', pattern, str(base)],
    )
    if result.returncode != 0:
        raise FilesystemPluginError(f'fd failed (exit {result.returncode}): {result.stderr.strip()}')
    paths = sorted(p for p in result.stdout.split('\0') if p)
    return paths[:limit]


def _walk_find(pattern: str, base: Path, limit: int) -> list[str]:
    """Stdlib fallback for `find_files` — os.walk + fnmatch on the name."""
    out: list[str] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if fnmatch.fnmatch(name, pattern):
                out.append(str(Path(root) / name))
    return out[:limit]


def _rg_search(rg_bin: str, query: str, glob: object, base: Path, limit: int) -> list[dict[str, object]]:
    """`ripgrep` backend: parse `--json` events into {path, line, text}."""
    if limit <= 0:
        return []
    cmd = [rg_bin, '--json', '--color', 'never']
    if isinstance(glob, str):
        cmd += ['--glob', glob]
    cmd += ['--', query, str(base)]
    result = _run(cmd)
    # ripgrep: 0 = matches, 1 = no matches (not an error), 2 = real error.
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise FilesystemPluginError(f'ripgrep failed (exit {result.returncode}): {result.stderr.strip()}')

    matches: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get('type') != 'match':
            continue
        data = event.get('data', {})
        path_text = data.get('path', {}).get('text')
        line_no = data.get('line_number')
        text = data.get('lines', {}).get('text', '')
        if not isinstance(path_text, str) or not isinstance(line_no, int):
            continue
        matches.append({'path': path_text, 'line': line_no, 'text': str(text).rstrip('\n')})
        if len(matches) >= limit:
            break
    return matches


def _walk_search(regex: re.Pattern[str], glob: object, base: Path, limit: int) -> list[dict[str, object]]:
    """Stdlib fallback for `search_content` — walk + per-line regex scan.

    Not .gitignore-aware (only `_SKIP_DIRS` are pruned); skips files over
    `_MAX_SCAN_BYTES` and any that are not UTF-8 decodable.
    """
    if limit <= 0:
        return []
    matches: list[dict[str, object]] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if isinstance(glob, str) and not fnmatch.fnmatch(name, glob):
                continue
            fpath = Path(root) / name
            try:
                if fpath.stat().st_size > _MAX_SCAN_BYTES:
                    continue
                with fpath.open(encoding='utf-8') as handle:
                    for line_no, line in enumerate(handle, start=1):
                        if regex.search(line):
                            matches.append({'path': str(fpath), 'line': line_no, 'text': line.rstrip('\n')})
                            if len(matches) >= limit:
                                return matches
            except (UnicodeDecodeError, OSError):
                # Binary / unreadable file — skip, consistent with how
                # ripgrep silently ignores non-text files.
                continue
    return matches


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


def _opt_bool(value: object, key: str) -> bool:
    """Coerce an optional boolean input — absent is False, non-bool errors.

    Distinct from the operator-config flags (`_flag`): this validates a
    *model-supplied* plan input (`recursive`, `dry_run`) and rejects a
    wrong type loudly rather than silently coercing it.
    """
    if value is None:
        return False
    if not isinstance(value, bool):
        raise FilesystemPluginError(f'input {key!r} must be a boolean when given, got {value!r}')
    return value
