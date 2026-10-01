"""Where IRIS reads its config, for plugins.

A plugin that ships defaults in the repo's ``config/`` tree (a category list, a policy
YAML) finds them here rather than with a bare ``Path("config/...")``, which is relative
to whatever directory the process happens to run in. ``config_dir()`` resolves, in
order: ``IRIS_CONFIG_DIR``, the checkout's ``config/``, the defaults packaged into the
installed wheel (see ``iris_harness.foundation.paths``).

``iris_home()`` is where this IRIS lives (``$IRIS_HOME``, else ``~/.iris``) and
``workspace_dir()`` the owner's workspace inside it: what a plugin keeps beside the
identity files (an account's category proposals) goes there, never under a bare
``Path.home() / ".iris"``, which ignores a relocated home.

``public_base_url()`` is the address the owner reaches this IRIS at from outside
(``IRIS_PUBLIC_URL``, named by ``PUBLIC_URL_ENV``), or ``None``: what an OAuth
redirect or a link in a notification must point at.
"""

from __future__ import annotations

from iris_harness.foundation.paths import config_dir, config_path, iris_home, workspace_dir
from iris_harness.foundation.public_url import PUBLIC_URL_ENV, public_base_url

__all__ = [
    "PUBLIC_URL_ENV",
    "config_dir",
    "config_path",
    "iris_home",
    "public_base_url",
    "workspace_dir",
]
