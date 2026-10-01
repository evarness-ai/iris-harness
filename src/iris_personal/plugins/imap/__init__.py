"""The IMAP plugin -- the app-password mailbox provider (OSS plan R3, release 1 L1).

:mod:`provider` is the ``MailProvider`` + ``LabellingProvider`` the core's
``email.providers`` registry dispatches ``imap:`` accounts to; :mod:`account` keeps each
account's connection details and app password in the vault; :mod:`cli` is
``iris auth imap``. Stdlib ``imaplib`` only.
"""
