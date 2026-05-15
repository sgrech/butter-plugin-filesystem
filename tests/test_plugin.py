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
    assert {cap.name for cap in manifest.capabilities} == {'pwd', 'cd', 'list_dir', 'stat', 'read_file'}
    # All user-facing — they appear in the planner menu.
    assert all(not cap.internal for cap in manifest.capabilities)
    # Filesystem-backed, not database-backed: it calls no other plugin.
    assert manifest.requires == ()


def test_filesystemplugin_satisfies_protocol_structurally() -> None:
    # Structural Protocol check — having `execute` with the right shape
    # is enough; FilesystemPlugin doesn't inherit from Plugin.
    plugin: Plugin = FilesystemPlugin()
    assert plugin is not None
