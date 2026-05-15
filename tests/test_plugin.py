"""Tests for the filesystem plugin + manifest round-trip.

Two layers, mirroring the butter-plugin-notes / butter-plugin-clock
convention:

- The plugin is unit-tested in isolation against real temporary
  directories (`tmp_path`) and an inline `FakePluginContext`. Filesystem
  capabilities touch only the disk and call no other plugin, so the
  context is a near-empty stand-in here; it carries a `config` mapping so
  it stays `PluginContext`-conformant as later steps consume it.
- `manifest.toml` round-trips through butter-agent's own `parse_manifest`
  — the contract check that proves the plugin loads without standing up
  a real host.

End-to-end behaviour (real host executor + gate handler, the operator
`config` flag gating `delete`) lives in the butter-agent integration
suite, not here.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
from butter_agent.plugin_api import BlastRadius, Plugin, parse_manifest

from butter_plugin_filesystem import FilesystemPlugin, FilesystemPluginError

MANIFEST_PATH = Path(__file__).resolve().parent.parent / 'manifest.toml'


def _entries(result: dict[str, object]) -> list[dict[str, object]]:
    """Narrow the `entries` field of a list_dir result for assertions."""
    entries = result['entries']
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
    return entries


@dataclass
class FakePluginContext:
    """Inline stand-in `PluginContext` for unit-testing in isolation.

    Read/navigate capabilities never touch the context, so `call` raising
    documents that. `config` mirrors the real `PluginContext.config`
    (the plugin's own operator-supplied settings); empty here until a
    config-gated capability is exercised in a later build step.
    """

    config: Mapping[str, object] = field(default_factory=dict)

    async def call(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        del inputs
        raise AssertionError(f'filesystem plugin must not call out (got {capability!r})')


# --- pwd / cd ----------------------------------------------------------------


async def test_pwd_reports_current_directory(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    await plugin.execute('cd', {'path': str(tmp_path)}, FakePluginContext())
    result = await plugin.execute('pwd', {}, FakePluginContext())
    assert result == {'cwd': str(tmp_path.resolve())}


async def test_cd_changes_cwd_and_relative_paths_follow(tmp_path: Path) -> None:
    (tmp_path / 'sub').mkdir()
    plugin = FilesystemPlugin()
    await plugin.execute('cd', {'path': str(tmp_path)}, FakePluginContext())
    # Relative target resolves against the new cwd.
    result = await plugin.execute('cd', {'path': 'sub'}, FakePluginContext())
    assert result == {'cwd': str((tmp_path / 'sub').resolve())}


async def test_cd_rejects_missing_path(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='no such path'):
        await plugin.execute('cd', {'path': str(tmp_path / 'nope')}, FakePluginContext())


async def test_cd_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='not a directory'):
        await plugin.execute('cd', {'path': str(f)}, FakePluginContext())


@pytest.mark.parametrize('bad', ['', None, 123])
async def test_cd_rejects_non_string_path(bad: object) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'path' must be a non-empty string"):
        await plugin.execute('cd', {'path': bad}, FakePluginContext())


# --- list_dir ----------------------------------------------------------------


async def test_list_dir_reports_name_type_size(tmp_path: Path) -> None:
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'a.txt').write_text('hello')
    plugin = FilesystemPlugin()
    result = await plugin.execute('list_dir', {'path': str(tmp_path)}, FakePluginContext())
    assert result['path'] == str(tmp_path.resolve())
    by_name = {e['name']: e for e in _entries(result)}
    assert by_name['sub']['type'] == 'dir'
    assert by_name['a.txt']['type'] == 'file'
    assert by_name['a.txt']['size'] == 5


async def test_list_dir_defaults_to_cwd(tmp_path: Path) -> None:
    (tmp_path / 'only.txt').write_text('1')
    plugin = FilesystemPlugin()
    await plugin.execute('cd', {'path': str(tmp_path)}, FakePluginContext())
    result = await plugin.execute('list_dir', {}, FakePluginContext())
    assert [e['name'] for e in _entries(result)] == ['only.txt']


async def test_list_dir_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='not a directory'):
        await plugin.execute('list_dir', {'path': str(f)}, FakePluginContext())


# --- stat --------------------------------------------------------------------


async def test_stat_reports_metadata(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('abcd')
    plugin = FilesystemPlugin()
    result = await plugin.execute('stat', {'path': str(f)}, FakePluginContext())
    assert result['path'] == str(f.resolve())
    assert result['type'] == 'file'
    assert result['size'] == 4
    # mtime is a valid tz-aware ISO-8601 stamp.
    parsed = datetime.fromisoformat(str(result['mtime']))
    assert parsed.tzinfo is not None


async def test_stat_rejects_missing_path(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='no such path'):
        await plugin.execute('stat', {'path': str(tmp_path / 'ghost')}, FakePluginContext())


# --- read_file ---------------------------------------------------------------


async def test_read_file_returns_full_content(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('line1\nline2\nline3\n')
    plugin = FilesystemPlugin()
    result = await plugin.execute('read_file', {'path': str(f)}, FakePluginContext())
    assert result['content'] == 'line1\nline2\nline3\n'
    assert result['lines'] == 3
    assert result['truncated'] is False


async def test_read_file_offset_and_limit_slice(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('l0\nl1\nl2\nl3\nl4\n')
    plugin = FilesystemPlugin()
    result = await plugin.execute('read_file', {'path': str(f), 'offset': 1, 'limit': 2}, FakePluginContext())
    assert result['content'] == 'l1\nl2\n'
    assert result['lines'] == 2
    assert result['truncated'] is True


async def test_read_file_rejects_binary(tmp_path: Path) -> None:
    f = tmp_path / 'b.bin'
    f.write_bytes(b'\xff\xfe\x00\x01')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='not a UTF-8 text file'):
        await plugin.execute('read_file', {'path': str(f)}, FakePluginContext())


async def test_read_file_rejects_directory(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='not a regular file'):
        await plugin.execute('read_file', {'path': str(tmp_path)}, FakePluginContext())


@pytest.mark.parametrize('bad', [-1, True, 'lots'])
async def test_read_file_rejects_invalid_limit(tmp_path: Path, bad: object) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'limit' must be a non-negative integer"):
        await plugin.execute('read_file', {'path': str(f), 'limit': bad}, FakePluginContext())


# --- write_file --------------------------------------------------------------


async def test_write_file_creates_new_file(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    target = tmp_path / 'new.txt'
    result = await plugin.execute('write_file', {'path': str(target), 'content': 'héllo'}, FakePluginContext())
    assert target.read_text(encoding='utf-8') == 'héllo'
    assert result == {'path': str(target.resolve()), 'bytes_written': len('héllo'.encode())}


async def test_write_file_overwrites_atomically(tmp_path: Path) -> None:
    target = tmp_path / 'a.txt'
    target.write_text('old contents')
    plugin = FilesystemPlugin()
    await plugin.execute('write_file', {'path': str(target), 'content': 'new'}, FakePluginContext())
    assert target.read_text() == 'new'
    # No stray temp files left behind.
    assert [p.name for p in tmp_path.iterdir()] == ['a.txt']


async def test_write_file_refuses_directory(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='is a directory'):
        await plugin.execute('write_file', {'path': str(tmp_path), 'content': 'x'}, FakePluginContext())


async def test_write_file_refuses_missing_parent(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    target = tmp_path / 'no' / 'such' / 'f.txt'
    with pytest.raises(FilesystemPluginError, match='parent directory does not exist'):
        await plugin.execute('write_file', {'path': str(target), 'content': 'x'}, FakePluginContext())


@pytest.mark.parametrize('bad', [None, 123, ['x']])
async def test_write_file_rejects_non_string_content(tmp_path: Path, bad: object) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'content' must be a string"):
        await plugin.execute('write_file', {'path': str(tmp_path / 'a.txt'), 'content': bad}, FakePluginContext())


# --- edit_file ---------------------------------------------------------------


async def test_edit_file_replaces_unique_occurrence(tmp_path: Path) -> None:
    target = tmp_path / 'a.txt'
    target.write_text('alpha beta gamma')
    plugin = FilesystemPlugin()
    result = await plugin.execute('edit_file', {'path': str(target), 'old': 'beta', 'new': 'BETA'}, FakePluginContext())
    assert target.read_text() == 'alpha BETA gamma'
    assert result == {'path': str(target.resolve()), 'replaced': 1}


async def test_edit_file_empty_new_deletes_text(tmp_path: Path) -> None:
    target = tmp_path / 'a.txt'
    target.write_text('keep DROP keep')
    plugin = FilesystemPlugin()
    await plugin.execute('edit_file', {'path': str(target), 'old': ' DROP', 'new': ''}, FakePluginContext())
    assert target.read_text() == 'keep keep'


async def test_edit_file_absent_old_raises(tmp_path: Path) -> None:
    target = tmp_path / 'a.txt'
    target.write_text('nothing here')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='old string not found'):
        await plugin.execute('edit_file', {'path': str(target), 'old': 'missing', 'new': 'x'}, FakePluginContext())


async def test_edit_file_ambiguous_old_raises_and_leaves_file_untouched(tmp_path: Path) -> None:
    target = tmp_path / 'a.txt'
    target.write_text('x x x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='occurs 3 times'):
        await plugin.execute('edit_file', {'path': str(target), 'old': 'x', 'new': 'y'}, FakePluginContext())
    assert target.read_text() == 'x x x'


async def test_edit_file_rejects_binary(tmp_path: Path) -> None:
    target = tmp_path / 'b.bin'
    target.write_bytes(b'\xff\xfe\x00')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='not a UTF-8 text file'):
        await plugin.execute('edit_file', {'path': str(target), 'old': 'a', 'new': 'b'}, FakePluginContext())


@pytest.mark.parametrize('bad', ['', None, 7])
async def test_edit_file_rejects_invalid_old(tmp_path: Path, bad: object) -> None:
    target = tmp_path / 'a.txt'
    target.write_text('content')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'old' must be a non-empty string"):
        await plugin.execute('edit_file', {'path': str(target), 'old': bad, 'new': 'x'}, FakePluginContext())


# --- find_files / search_content ---------------------------------------------


def _force_stdlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the plugin take its pure-stdlib path regardless of host PATH."""
    monkeypatch.setattr('butter_plugin_filesystem.plugin.shutil.which', lambda _name: None)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small project tree with a known NEEDLE token and noise dirs."""
    proj = tmp_path / 'proj'
    (proj / 'sub').mkdir(parents=True)
    (proj / 'a.py').write_text('import os\nNEEDLE = 1\n')
    (proj / 'sub' / 'b.txt').write_text('hello NEEDLE world\n')
    (proj / 'c.md').write_text('no token here\n')
    # Noise the stdlib fallback must prune.
    (proj / '.git').mkdir()
    (proj / '.git' / 'ignored.py').write_text('NEEDLE in vcs\n')
    (proj / 'b.bin').write_bytes(b'\xff\xfeNEEDLE')
    return proj


def _paths(result: dict[str, object]) -> list[str]:
    value = result['paths']
    assert isinstance(value, list)
    return [str(p) for p in value]


def _matches(result: dict[str, object]) -> list[dict[str, object]]:
    value = result['matches']
    assert isinstance(value, list)
    for m in value:
        assert isinstance(m, dict)
    return value


async def test_find_files_stdlib_globs_and_prunes_noise(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_stdlib(monkeypatch)
    plugin = FilesystemPlugin()
    result = await plugin.execute('find_files', {'pattern': '*.py', 'path': str(tree)}, FakePluginContext())
    assert result['backend'] == 'stdlib'
    names = {Path(p).name for p in _paths(result)}
    assert names == {'a.py'}  # .git/ignored.py pruned by _SKIP_DIRS


async def test_find_files_stdlib_limit_caps_results(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_stdlib(monkeypatch)
    plugin = FilesystemPlugin()
    result = await plugin.execute('find_files', {'pattern': '*', 'path': str(tree), 'limit': 2}, FakePluginContext())
    assert len(_paths(result)) == 2


@pytest.mark.parametrize('bad', ['', None, 5])
async def test_find_files_rejects_bad_pattern(bad: object) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'pattern' must be a non-empty string"):
        await plugin.execute('find_files', {'pattern': bad}, FakePluginContext())


async def test_search_content_stdlib_finds_token_with_location(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_stdlib(monkeypatch)
    plugin = FilesystemPlugin()
    result = await plugin.execute('search_content', {'query': 'NEEDLE', 'path': str(tree)}, FakePluginContext())
    assert result['backend'] == 'stdlib'
    hits = {(Path(str(m['path'])).name, m['line'], m['text']) for m in _matches(result)}
    # a.py:2 and sub/b.txt:1 — .git pruned, b.bin skipped (non-UTF-8).
    assert hits == {('a.py', 2, 'NEEDLE = 1'), ('b.txt', 1, 'hello NEEDLE world')}


async def test_search_content_stdlib_glob_filters_files(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_stdlib(monkeypatch)
    plugin = FilesystemPlugin()
    result = await plugin.execute('search_content', {'query': 'NEEDLE', 'path': str(tree), 'glob': '*.py'}, FakePluginContext())
    assert {Path(str(m['path'])).name for m in _matches(result)} == {'a.py'}


async def test_search_content_rejects_invalid_regex(tree: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='invalid regular expression'):
        await plugin.execute('search_content', {'query': '(', 'path': str(tree)}, FakePluginContext())


@pytest.mark.parametrize('bad', ['', None, 9])
async def test_search_content_rejects_bad_query(bad: object) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'query' must be a non-empty string"):
        await plugin.execute('search_content', {'query': bad}, FakePluginContext())


async def test_search_content_rejects_missing_directory(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='no such path'):
        await plugin.execute('search_content', {'query': 'x', 'path': str(tmp_path / 'ghost')}, FakePluginContext())


@pytest.mark.skipif(shutil.which('rg') is None, reason='ripgrep not installed')
async def test_search_content_ripgrep_backend_when_present(tree: Path) -> None:
    plugin = FilesystemPlugin()
    result = await plugin.execute('search_content', {'query': 'NEEDLE', 'path': str(tree), 'glob': '*.py'}, FakePluginContext())
    assert result['backend'] == 'ripgrep'
    assert {Path(str(m['path'])).name for m in _matches(result)} == {'a.py'}


@pytest.mark.skipif(shutil.which('fd') is None, reason='fd not installed')
async def test_find_files_fd_backend_when_present(tree: Path) -> None:
    plugin = FilesystemPlugin()
    result = await plugin.execute('find_files', {'pattern': '*.py', 'path': str(tree)}, FakePluginContext())
    assert result['backend'] == 'fd'
    assert 'a.py' in {Path(p).name for p in _paths(result)}


# --- delete ------------------------------------------------------------------


def _allow(**flags: bool) -> FakePluginContext:
    """A context carrying operator delete flags (model-invisible config)."""
    return FakePluginContext(config=dict(flags))


def _trash_contents(parent: Path) -> list[str]:
    trash = parent / '.butter-trash'
    return [p.name for p in trash.iterdir()] if trash.is_dir() else []


async def test_delete_disabled_without_operator_flag(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('x')
    plugin = FilesystemPlugin()
    # Default context: no allow_delete → refused, file untouched.
    with pytest.raises(FilesystemPluginError, match='delete is disabled'):
        await plugin.execute('delete', {'path': str(f)}, FakePluginContext())
    assert f.exists()


async def test_delete_file_moves_to_trash_by_default(tmp_path: Path) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('bye')
    plugin = FilesystemPlugin()
    result = await plugin.execute('delete', {'path': str(f)}, _allow(allow_delete=True))
    assert result == {'path': str(f.resolve()), 'dry_run': False, 'trashed': True, 'recursive': False}
    assert not f.exists()
    recovered = _trash_contents(tmp_path)
    assert len(recovered) == 1 and recovered[0].startswith('a.txt.')


async def test_delete_empty_directory_trashed(tmp_path: Path) -> None:
    d = tmp_path / 'empty'
    d.mkdir()
    plugin = FilesystemPlugin()
    result = await plugin.execute('delete', {'path': str(d)}, _allow(allow_delete=True))
    assert result['trashed'] is True
    assert not d.exists()


async def test_delete_nonempty_dir_refused_without_recursive(tmp_path: Path) -> None:
    d = tmp_path / 'full'
    (d / 'sub').mkdir(parents=True)
    (d / 'f.txt').write_text('x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='non-empty directory'):
        await plugin.execute('delete', {'path': str(d)}, _allow(allow_delete=True))
    assert d.exists()


async def test_delete_nonempty_dir_refused_without_operator_recursive_flag(tmp_path: Path) -> None:
    d = tmp_path / 'full'
    d.mkdir()
    (d / 'f.txt').write_text('x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='allow_recursive_delete'):
        await plugin.execute('delete', {'path': str(d), 'recursive': True}, _allow(allow_delete=True))
    assert d.exists()


async def test_delete_recursive_hard_deletes_with_both_flags(tmp_path: Path) -> None:
    d = tmp_path / 'full'
    (d / 'sub').mkdir(parents=True)
    (d / 'sub' / 'f.txt').write_text('x')
    plugin = FilesystemPlugin()
    ctx = _allow(allow_delete=True, allow_recursive_delete=True)
    result = await plugin.execute('delete', {'path': str(d), 'recursive': True}, ctx)
    assert result == {'path': str(d.resolve()), 'dry_run': False, 'trashed': False, 'recursive': True}
    assert not d.exists()
    # Hard delete (operator opted in) — not parked in trash.
    assert _trash_contents(tmp_path) == []


async def test_delete_dry_run_previews_without_mutating(tmp_path: Path) -> None:
    d = tmp_path / 'full'
    d.mkdir()
    (d / 'a').write_text('1')
    (d / 'b').write_text('2')
    plugin = FilesystemPlugin()
    result = await plugin.execute(
        'delete',
        {'path': str(d), 'recursive': True, 'dry_run': True},
        _allow(allow_delete=True, allow_recursive_delete=True),
    )
    assert result['dry_run'] is True
    assert result['is_dir'] is True
    assert result['entries'] == 2
    assert result['recursive'] is True
    assert d.exists()  # nothing mutated
    assert _trash_contents(tmp_path) == []


async def test_delete_refuses_working_directory(tmp_path: Path) -> None:
    d = tmp_path / 'proj'
    d.mkdir()
    plugin = FilesystemPlugin()
    await plugin.execute('cd', {'path': str(d)}, FakePluginContext())
    with pytest.raises(FilesystemPluginError, match='working directory or an ancestor'):
        await plugin.execute('delete', {'path': str(d)}, _allow(allow_delete=True, allow_recursive_delete=True))
    assert d.exists()


async def test_delete_refuses_ancestor_of_cwd(tmp_path: Path) -> None:
    d = tmp_path / 'proj'
    d.mkdir()
    plugin = FilesystemPlugin()
    await plugin.execute('cd', {'path': str(d)}, FakePluginContext())
    with pytest.raises(FilesystemPluginError, match='working directory or an ancestor'):
        await plugin.execute('delete', {'path': str(tmp_path)}, _allow(allow_delete=True, allow_recursive_delete=True))


async def test_delete_refuses_home_directory() -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='home directory or an ancestor'):
        await plugin.execute('delete', {'path': str(Path.home())}, _allow(allow_delete=True, allow_recursive_delete=True))


async def test_delete_refuses_shallow_system_path() -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='filesystem root or top-level system path'):
        await plugin.execute('delete', {'path': '/usr'}, _allow(allow_delete=True, allow_recursive_delete=True))


async def test_delete_symlink_does_not_touch_its_target(tmp_path: Path) -> None:
    real_dir = tmp_path / 'real'
    (real_dir / 'keep.txt').mkdir(parents=True)
    link = tmp_path / 'link'
    link.symlink_to(real_dir, target_is_directory=True)
    plugin = FilesystemPlugin()
    await plugin.execute('delete', {'path': str(link)}, _allow(allow_delete=True))
    assert not link.exists()  # the link is gone
    assert real_dir.is_dir()  # its target is untouched


async def test_delete_missing_path_raises(tmp_path: Path) -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='no such path'):
        await plugin.execute('delete', {'path': str(tmp_path / 'ghost')}, _allow(allow_delete=True))


@pytest.mark.parametrize('bad', [1, 'yes', []])
async def test_delete_rejects_non_bool_recursive(tmp_path: Path, bad: object) -> None:
    f = tmp_path / 'a.txt'
    f.write_text('x')
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match="'recursive' must be a boolean"):
        await plugin.execute('delete', {'path': str(f), 'recursive': bad}, _allow(allow_delete=True))


# --- dispatch ----------------------------------------------------------------


async def test_unknown_capability_raises() -> None:
    plugin = FilesystemPlugin()
    with pytest.raises(FilesystemPluginError, match='unknown capability'):
        await plugin.execute('teleport', {}, FakePluginContext())


# --- Manifest contract -------------------------------------------------------


def test_manifest_round_trips_through_butter_validator() -> None:
    manifest = parse_manifest(MANIFEST_PATH.read_text())
    assert manifest.name == 'filesystem'
    assert manifest.blast_radius is BlastRadius.LOCAL_WRITE
    assert manifest.entrypoint == 'butter_plugin_filesystem:FilesystemPlugin'
    assert {cap.name for cap in manifest.capabilities} == {'pwd', 'cd', 'list_dir', 'stat', 'read_file', 'write_file', 'edit_file', 'find_files', 'search_content', 'delete'}
    # All user-facing — they appear in the planner menu.
    assert all(not cap.internal for cap in manifest.capabilities)
    # Filesystem-backed, not database-backed: it calls no other plugin.
    assert manifest.requires == ()


def test_filesystemplugin_satisfies_protocol_structurally() -> None:
    # Structural Protocol check — having `execute` with the right shape
    # is enough; FilesystemPlugin doesn't inherit from Plugin.
    plugin: Plugin = FilesystemPlugin()
    assert plugin is not None
