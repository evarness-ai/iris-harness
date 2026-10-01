# SearXNG — primary search backend for the IRIS research engine

Privacy-first, self-hosted meta-search. Once running, the research engine
(`src/iris_harness/plugins_builtin/research/`) auto-selects it as the **primary** provider, ahead of the keyless
DuckDuckGo fallback. See `docs/architecture/research-engine.md`.

## One-time setup

```bash
# 1. Set a unique secret key
sed -i '' "s/CHANGE_ME_TO_A_LONG_RANDOM_STRING/$(openssl rand -hex 32)/" deploy/searxng/settings.yml

# 2. Start it (local-only, bound to 127.0.0.1:8888)
docker compose -f deploy/searxng/docker-compose.yml up -d

# 3. Point IRIS at it and restart the IRIS API
export IRIS_SEARXNG_URL=http://localhost:8888
```

## Verify

```bash
# JSON API should return results (this is exactly what the provider calls):
curl -s 'http://localhost:8888/search?q=test&format=json' | head -c 200
```

If that returns JSON, IRIS will use SearXNG. Confirm from the engine:

```bash
IRIS_SEARXNG_URL=http://localhost:8888 poetry run python -c \
  "from iris_harness.plugins_builtin.research.providers import select_providers; \
   print([p.name for p in select_providers()])"
# -> ['searxng', 'ddg']   (searxng primary)
```

## Notes

- **`search.formats` must include `json`** (set in `settings.yml`) — SearXNG disables the
  JSON API by default; without it the provider gets nothing and IRIS falls back to DDG.
- Bound to `127.0.0.1` with no auth — fine for local single-user. Do not expose publicly
  without a reverse proxy + auth.
- Result quality depends on which upstream engines SearXNG queries; the defaults
  (Google/Bing/DuckDuckGo/etc.) are a strong start. Tune in `settings.yml` under `engines:`.
- Stop: `docker compose -f deploy/searxng/docker-compose.yml down`.
