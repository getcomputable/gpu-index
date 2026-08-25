# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""ECB euro foreign exchange reference rates (rule R2).

The basket records non-USD prints natively (Scaleway bills EUR); the
composite calc converts them using the ECB's daily reference rate —
chosen because it is free, public, citable in a contract, published
~16:00 CET every TARGET business day, and archived (backfill-safe).

Audit + replay discipline:
  - every rate used is persisted append-only under
    ``<prefix>/fx/ecb-<YYYY-MM-DD>.json`` (fetched once, reused forever) so
    a replay years later converts with the SAME rates the original run used;
  - non-publication days (weekends/TARGET holidays) use the most recent
    published rate at or before the observation day, with the rate's actual
    date recorded as ``fx_as_of``;
  - fail-closed: no stored or fetchable rate within
    ``fx_max_staleness_days`` → raise. The caller holds non-USD sources out
    for the day, loudly. A guessed FX rate is worse than a missing source.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any, Dict, Tuple

from gpu_index.common.http import fetch
from gpu_index.common.slots import rfc3339
from gpu_index.common.store import get_object_bytes, list_object_keys, put_json_bytes

FX_SOURCE_ID = "ecb_reference_rate"
FX_BASE = "EUR"
# Documented alternate endpoint (latest day only, no backfill) — the 90d
# feed is used because one fetch covers weekends, holidays, and gap repair.
ECB_DAILY_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
ECB_90D_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
DEFAULT_FX_MAX_STALENESS_DAYS = 7
# Sanity band for EUR/USD: rates are persisted immutably first-write-wins,
# so a poisoned/anomalous value would convert every EUR print forever. A
# reference rate outside this band means the feed is broken or hostile —
# reject the whole document and let the fail-closed staleness path rule.
USD_RATE_SANE_BAND = (0.5, 2.0)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_ECB_NS = "{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}"


class FxUnavailableError(RuntimeError):
    """No reference rate within the staleness window — never guess one."""


def parse_ecb_rates(xml_text: str) -> Dict[str, Dict[str, float]]:
    """{published_date: {currency: rate}} from an ECB eurofxref document.

    xml.etree on a single trusted, TLS-verified, size-capped source; rates
    are units of CURRENCY per 1 EUR.
    """
    root = ET.fromstring(xml_text)
    out: Dict[str, Dict[str, float]] = {}
    for day_cube in root.iter(f"{_ECB_NS}Cube"):
        day = day_cube.get("time")
        if not day:
            continue
        # Feed-controlled content becomes an S3 key component (fx_key) and
        # the stored as_of — refuse anything that is not a plain ISO date.
        if not _DATE_RE.fullmatch(day):
            raise RuntimeError(f"ECB feed carries a non-date time attr {day!r}")
        date.fromisoformat(day)
        rates: Dict[str, float] = {}
        for cube in day_cube:
            ccy = cube.get("currency")
            rate = cube.get("rate")
            if not (ccy and rate):
                continue
            value = float(rate)
            if not math.isfinite(value) or value <= 0:
                if ccy == "USD":
                    # USD is what the index consumes and rates persist
                    # immutably — a bad USD value rejects the whole document.
                    raise RuntimeError(
                        f"ECB feed carries an implausible USD rate {rate!r} on {day}"
                    )
                # A glitched exotic currency must NOT blind the USD consumer:
                # the 90d feed re-serves the same window, so document-level
                # rejection would hold EUR conversion out for months. Drop
                # the one value; nothing bad is ever stored.
                continue
            rates[ccy] = value
        usd = rates.get("USD")
        if usd is not None and not (
            USD_RATE_SANE_BAND[0] <= usd <= USD_RATE_SANE_BAND[1]
        ):
            raise RuntimeError(
                f"ECB USD rate {usd} on {day} outside sane band {USD_RATE_SANE_BAND}"
            )
        if rates:
            out[day] = rates
    if not out:
        raise RuntimeError("ECB feed parsed zero dated rate cubes — shape changed")
    return out


def fx_key(prefix: str, day: str) -> str:
    return f"{prefix}/fx/ecb-{day}.json"


def load_stored_rates(
    client,
    bucket: str,
    *,
    prefix: str,
    from_day: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """All persisted ECB records keyed by published date.

    ``from_day`` (optional YYYY-MM-DD) bounds the read: keys still LIST in
    one paginated pass (names only, cheap), but records dated strictly
    before it are never fetched or parsed -- the hourly panel lanes' cost
    fence, where a years-old fx/ store must not cost one GET per record
    per firing. A key whose date segment does not parse is fetched anyway
    (fail-open to the unbounded behavior; nothing valid can be skipped).
    Default None keeps the daily lanes' load-everything behavior exactly.
    """
    out: Dict[str, Dict[str, Any]] = {}
    marker = f"{prefix}/fx/ecb-"
    for key in list_object_keys(client, bucket, marker):
        if from_day is not None:
            stem = key[len(marker):]
            if stem.endswith(".json"):
                stem = stem[: -len(".json")]
            if _DATE_RE.fullmatch(stem) and stem < str(from_day):
                continue  # dated strictly before the bound: never needed
        raw = get_object_bytes(client, bucket, key)
        if raw is None:
            continue
        record = json.loads(raw)
        if record.get("as_of"):
            out[record["as_of"]] = record
    return out


def persist_rates(
    client, bucket: str, *, prefix: str, rates_by_day: Dict[str, Dict[str, float]]
) -> int:
    """Write any not-yet-stored published dates append-only; returns count."""
    written = 0
    for day, rates in sorted(rates_by_day.items()):
        key = fx_key(prefix, day)
        if get_object_bytes(client, bucket, key) is not None:
            continue  # rates for a published date never change — first write wins
        record = {
            "source": FX_SOURCE_ID,
            "base": FX_BASE,
            "as_of": day,
            "rates": rates,
            "fetched_at": rfc3339(),
        }
        put_json_bytes(
            client,
            bucket,
            key,
            (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        written += 1
    return written


def ensure_rates(
    client,
    bucket: str,
    *,
    prefix: str,
    timeout: float = 30.0,
    persist: bool = True,
    from_day: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Stored records, topped up from the ECB 90-day feed for any new dates.

    A feed failure (network OR a shape-change/plausibility rejection from
    parse_ecb_rates) falls back to stored history — but SAYS so: a silent
    swallow here would make a sustained feed blockade look like a quiet
    offline run until sources start failing FX a week later.
    ``persist=False`` (dry runs) merges the feed in memory only.
    ``from_day`` bounds the STORED read exactly as load_stored_rates does
    (additive; default None = the daily lanes' unbounded behavior); the
    feed itself is date-complete for its 90-day window either way, and
    persisting a fresh date older than the bound is harmless append-only.
    """
    stored = load_stored_rates(client, bucket, prefix=prefix, from_day=from_day)
    try:
        feed = parse_ecb_rates(fetch(ECB_90D_URL, timeout=timeout))
    except Exception as exc:  # noqa: BLE001 — fail closed onto stored history, loudly
        print(f"WARNING: ECB feed unavailable/rejected ({exc}) — using stored FX only")
        feed = {}
    fresh = {d: r for d, r in feed.items() if d not in stored}
    if fresh:
        if persist:
            persist_rates(client, bucket, prefix=prefix, rates_by_day=fresh)
        for day, rates in fresh.items():
            stored[day] = {
                "source": FX_SOURCE_ID,
                "base": FX_BASE,
                "as_of": day,
                "rates": rates,
            }
    return stored


def rates_cover(
    stored: Dict[str, Dict[str, Any]],
    days: Any,
    *,
    currency: str = "USD",
    max_staleness_days: int = DEFAULT_FX_MAX_STALENESS_DAYS,
) -> bool:
    """True iff every day in ``days`` resolves a ``currency`` rate from
    ``stored`` alone via the standard walk-back (lookup_rate). The hourly
    panel lanes' skip-the-feed test: when every unpublished observation's
    day is already priceable from persisted records, the live ECB fetch
    is pure latency and egress and is skipped -- the fail-closed staleness
    fence is untouched (an unresolvable day triggers the fetch, and only
    a fetch failure after THAT walks into FxUnavailableError territory).
    """
    for day in days:
        try:
            lookup_rate(
                stored,
                str(day),
                currency,
                max_staleness_days=max_staleness_days,
            )
        except FxUnavailableError:
            return False
    return True


def lookup_rate(
    stored: Dict[str, Dict[str, Any]],
    day: str,
    currency: str,
    *,
    max_staleness_days: int = DEFAULT_FX_MAX_STALENESS_DAYS,
) -> Tuple[float, str]:
    """(rate, fx_as_of) for CURRENCY per EUR at <= day, walking back over
    non-publication days. Raises FxUnavailableError past the window."""
    target = date.fromisoformat(day)
    for back in range(max_staleness_days + 1):
        candidate = (target - timedelta(days=back)).isoformat()
        record = stored.get(candidate)
        if record and currency in (record.get("rates") or {}):
            return float(record["rates"][currency]), candidate
    raise FxUnavailableError(
        f"no {FX_SOURCE_ID} {currency} rate within {max_staleness_days} days "
        f"at or before {day}"
    )


def eur_to_usd(
    price_eur: float,
    stored: Dict[str, Dict[str, Any]],
    day: str,
    *,
    max_staleness_days: int = DEFAULT_FX_MAX_STALENESS_DAYS,
) -> Tuple[float, Dict[str, Any]]:
    """USD value + the audit block {fx_rate, fx_source, fx_as_of}."""
    rate, as_of = lookup_rate(
        stored, day, "USD", max_staleness_days=max_staleness_days
    )
    return price_eur * rate, {
        "fx_rate": rate,
        "fx_source": FX_SOURCE_ID,
        "fx_as_of": as_of,
    }
