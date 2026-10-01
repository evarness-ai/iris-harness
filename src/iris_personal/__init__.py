"""The personal-assistant domains: finance, filemanager, email, calendar, market.

A second source root, not a subpackage: the harness core never imports it (OSS plan
M6, decision 2, enforced by the ``core-does-not-import-iris-personal`` contract), and
the public tree ships without it. Each domain is a *library* -- the record and the
access to it -- plus a plugin under :mod:`iris_personal.plugins` that registers the
workflows over it through ``PluginAPI``. The plugins are discovered as
``iris_harness.plugins`` entry points, which is the same seam any third-party plugin
uses; nothing here is privileged.
"""
