"""Cross-cutting data the core itself reads.

Today that is only :mod:`.dues_vocabulary`'s query matchers: the research plugin's egress
guard reads them (ADR-0102) and has to work with no domain plugin mounted.

The account registry and the categories table used to live here too. No core module read
them except ``iris system status``, so they left with the domain that owns them (core/SDK
boundary plan, PR 2); status now asks a counter that domain registers
(``services.system.status.register_account_counter``).
"""
