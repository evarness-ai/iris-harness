"""The Gmail plugin — the mailbox provider (OSS plan M5.7, track A slice 2).

``gmail_fetch``, ``gmail_attachments`` and ``gmail_oauth`` moved here unchanged from
``iris_personal.email`` / ``iris_harness.memory.identity``; :mod:`provider` wraps them as the
``MailProvider`` the core's ``email.providers`` registry dispatches to.
"""
