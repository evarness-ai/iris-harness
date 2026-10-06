# Plugin egress: the manifest declaration (issue #103)

Status: the declaration is implemented; **nothing enforces it yet**. Epic #85, exit
criterion G3. Sequenced before the MCP rung (#111-#114), which reuses the declaration.

## The problem

A plugin's own outbound HTTP passes through no governance and leaves no ledger row. The
only governance on a tool's network use is the `network_egress` `PRE_TOOL_USE` hook, which
inspects tool *names* in a built-in list; for a plugin's tool its row reads
`network_egress: not a network tool`. The manifest had no place to say which hosts a plugin
talks to (`egress:` was rejected as an extra field), and `sends_to` accepted only
`search_engine`.

## What this change adds

The first of two steps. This one is the *declaration* and its compiled policy; the second
adds the governed client that reads it. Until then the declaration changes no plugin's
behaviour: nothing in the kernel reads the compiled policy.

1. **Manifest declaration.** `egress:` lists the hosts a plugin may contact:

   ```yaml
   egress:
     hosts:
       - api.open-meteo.com                   # shorthand: https, port 443, data: internal
       - host: geocoding-api.open-meteo.com
         data: personal                       # the highest data class this host receives
       - host: "*.example.org"                # a subdomain pattern, never a bare "*"
         schemes: [https]
         ports: [443, 8443]
     open_web: false                          # true: any host (a page fetcher)
   ```

   Unknown keys are rejected. `data` is `public | internal | personal` (never `secret`);
   the default is `internal`, so a plugin that sends something personal must say so.
   `open_web: true` is the explicit, loud form of "any host", for a tool whose job is to
   fetch pages the owner or model choose (the research plugin). Absent or empty is a closed
   door, not "unrestricted".

2. **A compiled policy.** The harness compiles every mounted manifest into one
   `PluginEgressPolicy` (`runtime/egress_access.py`, `kernel/governance/plugin_egress.py`)
   and registers it after plugins mount, exactly as it does the caller policy
   (`runtime/tool_access.py`, `kernel/governance/caller_policy.py`). Config, not code: no
   plugin name is hard-coded in the kernel. `decide(plugin, scheme=, host=, port=,
   classification=)` fails closed: no policy, an unmounted plugin or an unknown host is a
   denial with a reason.

3. **Visibility.** `iris plugins show`, `iris --dump-config` and `GET /plugins` list each
   plugin's declared hosts.

4. **A generic `sends_to` destination.** `external_service` joins `search_engine`: a tool
   declaring it hands its arguments to a service outside the machine, so the owner-PII
   `egress` column (shadow mode, ADR-0125) reads its `PRE_TOOL_USE` arguments as leaving.
   A tool declaring it must belong to a plugin that declares `egress` hosts (or
   `open_web`), checked at manifest load. No shipped manifest declares it yet.

The shipped `research` manifest declares `open_web: true` and the `gmail` manifest its four
Google hosts (`data: personal`).

## Why the manifest key `egress` is stable surface

`egress` is a top-level manifest key, and the stable tier lists every manifest key
(`stable_tier.yaml`, `manifest_fields`): a plugin author writes it, and a plugin written
against 0.1.0 must keep loading. It is the only stable name this change adds. Because a
manifest with an unknown key is rejected, the key cannot be introduced later without it
being a breaking change for older harnesses; adding it first lets plugins declare before the
client that enforces it exists.

## What this does not do

Declared is not enforced. No hook point, SDK name or audit field is added here; no request a
plugin makes is checked, blocked or recorded because of the declaration.
