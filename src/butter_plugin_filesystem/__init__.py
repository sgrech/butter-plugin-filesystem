"""butter-plugin-filesystem — local filesystem access for butter-agent.

The `FilesystemPlugin` class is the manifest's declared entrypoint.
Importing it from the package root keeps the `module:Class` path short
(`butter_plugin_filesystem:FilesystemPlugin`). Keep `__version__` in
lock-step with `[plugin].version` in `manifest.toml`.
"""

from __future__ import annotations

from butter_plugin_filesystem.plugin import FilesystemPlugin, FilesystemPluginError

__all__ = ['FilesystemPlugin', 'FilesystemPluginError']
__version__ = '0.1.0'
