# How CGI collects prices

This document describes exactly what the collectors do on the network. It is a
description of behaviour, not a claim about anyone's terms.

## What is fetched

Each collector reads one provider's **published pricing page or public API
endpoint** -- the same URLs a person sees in a browser without signing in. Every
request is anonymous. The collectors hold no accounts, send no credentials, no
cookies, no API keys and no tokens, and they do not use endpoints that require
an account or a key. The full source list, with the URL and the parse for each,
is in `config/raw_observatory.json` and `src/gpu_index/observatory/sources/`.

Only prices and the metadata printed alongside them are recorded. Nothing is
purchased, provisioned or benchmarked.

## How the collectors identify themselves

Every request carries the project User-Agent defined in
`src/gpu_index/common/http.py`:

```
CGI-Collector/0.1 (+https://github.com/getcomputable/gpu-index; team@getcomputable.com)
```

It names the project, links this repository and carries a working contact
address. It is the only User-Agent the collectors send. There is no browser
string, no rotation and no impersonation of any client.

## Frequency

Collection is configured to run once an hour, on the hour, in UTC
(`capture_slots_utc` in `config/raw_observatory.json`). Within a run the
sources are fetched **serially**, one at a time -- never in parallel. Most
sources are a single request per run; a few paginated or per-chip APIs (vast,
computepulse) send several, and those are spaced rather than sent back to back.
The current configuration of 30 sources works out to roughly 110 requests in an
hour across all providers combined.

## Transport limits

The shared fetch in `common/http.py` and the runner in `observatory/collect.py`
bound every request:

- HTTPS only -- a redirect to plain HTTP is refused, never followed;
- 30s socket timeout and a 90s wall-clock limit on reading a response body;
- responses are capped at 8MB and the read is abandoned past that;
- each source gets at most 180s, and a whole run at most 1500s.

## No retries

There is no retry, no backoff, no queue and no second attempt. A request that
is refused, times out or returns anything unexpected is recorded as an error
for that source and that hour, and the source simply goes dark for that hour.
Nothing re-requests it, and nothing tries a different address, header or client
to get a different answer.

## robots.txt

The collectors do not currently fetch or consult `robots.txt`. That is a known
gap, stated here plainly rather than papered over.

## Contact

A provider who does not want their prices collected here should write to
**team@getcomputable.com**. That is the route for asking to have a source
removed, and for anything else about collection. Removing a seated provider is
a panel-membership change, so it is decided and recorded the way
[GOVERNANCE.md](../GOVERNANCE.md) sets out, with the reasoning in
[CHANGELOG.md](../CHANGELOG.md).
