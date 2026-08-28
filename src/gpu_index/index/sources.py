# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Per-source basket collectors — raw prints from each provider's own surface.

Contract per collector:
  - returns {source_id, method, url, first_party_observation, observations};
    each observation carries the RAW value as published plus our per-GPU
    normalization (never trust an aggregator's per-GPU figure) and
    an explicit currency — a non-USD list price (Scaleway bills EUR) is
    recorded natively with price_usd_gpu_hr=None, never silently mislabeled.
  - raises on failure INCLUDING parsed-nothing: a page that silently changed
    shape must surface as an error in the snapshot, never as a healthy source
    with zero prints.
  - TLS verification is never weakened. A host with a broken CA bundle fixes
    its environment (SSL_CERT_FILE); a price feed fetched unverified is worse
    than a missing one.
  - never aggregates across sources and never writes anywhere — collectors
    feed the basket snapshot only.

Every recipe below was fetched live and independently re-run before
landing.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

# The order-book population screen lives with the calc (gpu_index.index
# .composite) and is IMPORTED here so the capture's recorded population
# can never drift narrower than the statistic's claim.
from gpu_index.index.composite import us_ca_verified_host
from gpu_index.common.http import DEFAULT_TIMEOUT, fetch, norm_sku
from gpu_index.common.slots import rfc3339

BASKET_SKUS = ("B200", "B300")


def _obs(
    sku: str,
    price_per_gpu_hr: float,
    *,
    currency: str = "USD",
    raw_value: str,
    raw_unit: str = "usd_per_gpu_hr",
    gpu_count_basis: int = 1,
    tier: str = "on-demand",
    region: str = "?",
    notes: str = "",
) -> Dict[str, Any]:
    price = round(float(price_per_gpu_hr), 4)
    return {
        "sku": sku,
        "price_usd_gpu_hr": price if currency == "USD" else None,
        "price_native_per_gpu_hr": price,
        "currency": currency,
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "gpu_count_basis": gpu_count_basis,
        "tier": tier,
        "region": region,
        "notes": notes,
    }


def _result(
    source_id: str,
    *,
    method: str,
    url: str,
    observations: List[Dict[str, Any]],
    first_party: bool = True,
) -> Dict[str, Any]:
    if not observations:
        raise RuntimeError(
            f"{source_id}: parsed zero B200/B300 observations — page/API shape "
            "changed or listings pulled; refusing to record an empty print"
        )
    return {
        "source_id": source_id,
        "method": method,
        "url": url,
        "first_party_observation": first_party,
        "fetched_at": rfc3339(),
        "observations": observations,
    }


# ------------------------------------------------------------------ verda

VERDA_URL = "https://verda.com/pricing"
_VERDA_OFFER_RE = re.compile(
    r'"@type":"Offer","name":"([^"]+)"[^}]*?"price":([\d.]+)'
)
_VERDA_COUNT_RE = re.compile(r"^(\d+)x\s")


_VERDA_CURRENCY_RE = re.compile(r'"priceCurrency":"([A-Z]{3})"')


def parse_verda(html: str) -> List[Dict[str, Any]]:
    """JSON-LD offers. Instant-cluster and serverless rows are excluded (the
    cluster rows are node-total priced and the serverless-spot rows would
    pollute the spot tier); a leading 'Nx ' count normalizes node-total
    prices per-GPU (1x rows today, so usually a no-op). priceCurrency is
    read from each offer blob — a Nordic operator flipping to EUR must be
    recorded natively, never silently mislabeled USD."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for m in _VERDA_OFFER_RE.finditer(html):
        name, price = m.group(1), float(m.group(2))
        if name in seen:
            continue
        seen.add(name)
        lname = name.lower()
        sku = norm_sku(name)
        if sku not in BASKET_SKUS:
            continue
        if "serverless" in lname or "instant cluster" in lname:
            continue
        if "on-demand" in lname:
            tier = "on-demand"
        elif "spot" in lname:
            tier = "spot"
        else:
            continue
        count_m = _VERDA_COUNT_RE.match(name)
        count = int(count_m.group(1)) if count_m else 1
        # priceCurrency sits before OR after "price" depending on render;
        # search only within THIS offer's blob (bounded at its closing
        # brace, never a fixed window that could bleed into the next
        # offer). Absent currency is UNKNOWN, never assumed USD — an
        # unlabeled print must degrade visibly (price_usd_gpu_hr=null),
        # not silently enter the USD series.
        blob_end = html.find("}", m.end())
        search_end = blob_end + 1 if blob_end != -1 else m.end()
        currency_m = _VERDA_CURRENCY_RE.search(html, m.start(), search_end)
        currency = currency_m.group(1) if currency_m else "UNKNOWN"
        obs = _obs(
            sku,
            price / count,
            currency=currency,
            raw_value=str(m.group(2)),
            raw_unit="usd_per_gpu_hr" if count == 1 else "usd_per_instance_hr",
            gpu_count_basis=count,
            tier=tier,
            region="EU (Nordic DCs)",
            notes=name,
        )
        # The provider's own offer name is the STRUCTURED label the
        # product-identity token screen reads — prose notes are immune.
        obs["sku_identifier"] = name
        rows.append(obs)
    return rows


def collect_verda(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    html = fetch(VERDA_URL, timeout=timeout)
    return _result("verda", method="jsonld", url=VERDA_URL, observations=parse_verda(html))


# ------------------------------------------------------------------ nebius

NEBIUS_URL = "https://nebius.com/prices"
# The FULL header row is pinned, not a substring: a substring check would
# still pass if a new price column (e.g. "Reserved, GPU-hour") were inserted
# BEFORE these two, and the row regex would then silently record the wrong
# tier as on-demand — the exact plausible-but-wrong failure this lane must
# never feed into index history.
_NEBIUS_HEADER = (
    '["Item","vCPUs","RAM, GB","Preemptible, GPU-hour","On-demand, GPU-hour"]'
)
# (?!,"\$) refuses rows carrying a THIRD price field, same fail-closed logic.
_NEBIUS_ROW_RE = re.compile(
    r'"NVIDIA HGX (B300|B200)","(\d+)","(\d+)","\$([\d,]+(?:\.\d+)?)",'
    r'"\$([\d,]+(?:\.\d+)?)"(?!,"\$)'
)


def parse_nebius(html: str) -> List[Dict[str, Any]]:
    """Server-rendered price table embedded as an escaped-JSON blob; prices
    are already per GPU-hour (the vCPU/RAM columns are per-GPU resource
    shares, not node counts). Fail-closed header guard: ANY column change
    raises rather than guessing which price is which tier."""
    plain = html.replace("\\", "")
    if _NEBIUS_HEADER not in plain:
        raise RuntimeError(
            "nebius price-table header changed — refusing to guess column order"
        )
    rows: List[Dict[str, Any]] = []
    seen = set()
    for m in _NEBIUS_ROW_RE.finditer(plain):
        sku, vcpus, ram, preemptible, ondemand = m.groups()
        if sku in seen:
            continue
        seen.add(sku)
        note = f"NVIDIA HGX {sku} ({vcpus} vCPU / {ram} GB per-GPU share)"
        for price_str, tier in ((ondemand, "on-demand"), (preemptible, "preemptible")):
            obs = _obs(
                sku,
                float(price_str.replace(",", "")),
                raw_value=price_str,
                tier=tier,
                region="global",
                notes=note,
            )
            obs["sku_identifier"] = f"NVIDIA HGX {sku}"
            rows.append(obs)
    return rows


def collect_nebius(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    html = fetch(NEBIUS_URL, timeout=timeout)
    return _result(
        "nebius", method="html-regex", url=NEBIUS_URL, observations=parse_nebius(html)
    )


# ------------------------------------------------------------------ hyperstack

HYPERSTACK_URL = "https://www.hyperstack.cloud/gpu-pricing"
_HYPERSTACK_ROW_RE = re.compile(
    r"NVIDIA\|((?:H200\|SXM|H100\|SXM|B200|B300))\|(?:[\d.]+\|){3}\$([\d,]+(?:\.\d+)?)"
)


def parse_hyperstack(html: str) -> List[Dict[str, Any]]:
    """The exact-3-numeric-fields fence is load-bearing: it is what excludes
    the RESERVED price rows (which lack spec columns) — do not loosen it."""
    txt = re.sub(r"<[^>]+>", "|", html)
    txt = re.sub(r"(?:\||\s|&nbsp;|\xa0)+", "|", txt)
    rows: List[Dict[str, Any]] = []
    for m in _HYPERSTACK_ROW_RE.finditer(txt):
        label = m.group(1)
        sku = "B300" if "B300" in label else "B200" if "B200" in label else None
        if sku:
            obs = _obs(
                sku,
                float(m.group(2).replace(",", "")),
                raw_value=m.group(2),
                region="EU-heavy",
                notes=label.replace("|", " ")
                + (" (banner: B300 on-demand launching Aug 2026 — possibly forward-listed)" if sku == "B300" else ""),
            )
            obs["sku_identifier"] = label.replace("|", " ").strip()
            rows.append(obs)
    return rows


def collect_hyperstack(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    html = fetch(HYPERSTACK_URL, timeout=timeout)
    return _result(
        "hyperstack",
        method="html-regex",
        url=HYPERSTACK_URL,
        observations=parse_hyperstack(html),
    )


# ------------------------------------------------------------------ scaleway

SCALEWAY_URL = (
    "https://api.scaleway.com/instance/v1/zones/fr-par-2/products/servers"
)
SCALEWAY_B300_GIB = 288 * 1024**3  # gpu_memory bytes for the 288GiB standard config
SCALEWAY_PER_PAGE = 100  # the API's per_page max
# Catalog is ~136 SKUs today; 3 full pages (=300) means it tripled — fetch
# stops there but says so loudly rather than silently truncating.
SCALEWAY_MAX_PAGES = 3


def parse_scaleway(server_maps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Public unauthenticated instance-catalog API (no console auth
    involved — this is Scaleway's own api domain).
    ``server_maps``: one parsed ``servers`` dict per fetched page.
    hourly_price is per NODE in EUR; per_gpu = hourly_price / gpu. EUR is
    recorded natively (price_usd_gpu_hr=None) — FX is a calc-time
    decision, not something a collector invents."""
    rows: List[Dict[str, Any]] = []
    for servers in server_maps:
        for name, spec in sorted(servers.items()):
            gpu_info = spec.get("gpu_info") or {}
            if "B300" not in str(gpu_info.get("gpu_name", "")):
                continue
            count = spec.get("gpu") or 0
            hourly = spec.get("hourly_price")
            if not count or hourly is None:
                continue
            mem_note = ""
            if gpu_info.get("gpu_memory") != SCALEWAY_B300_GIB:
                mem_note = f" gpu_memory={gpu_info.get('gpu_memory')} (NOT 288GiB standard)"
            rows.append(
                _obs(
                    "B300",
                    float(hourly) / count,
                    currency="EUR",
                    raw_value=str(hourly),
                    raw_unit="eur_per_node_hr",
                    gpu_count_basis=int(count),
                    region="fr-par-2",
                    notes=f"{name} node {hourly} EUR/hr{mem_note}",
                )
            )
    return rows


def collect_scaleway(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    server_maps: List[Dict[str, Any]] = []
    full_pages = 0
    for page in range(1, SCALEWAY_MAX_PAGES + 1):
        body = fetch(
            f"{SCALEWAY_URL}?per_page={SCALEWAY_PER_PAGE}&page={page}",
            timeout=timeout,
        )
        servers = json.loads(body).get("servers") or {}
        if not servers:
            break
        server_maps.append(servers)
        if len(servers) < SCALEWAY_PER_PAGE:
            break  # short page == last page; don't burn a request proving it
        full_pages += 1
    result = _result(
        "scaleway",
        method="api-json",
        url=SCALEWAY_URL,
        observations=parse_scaleway(server_maps),
    )
    if full_pages == SCALEWAY_MAX_PAGES:
        result["partial_errors"] = [
            f"catalog paging stopped at {SCALEWAY_MAX_PAGES} full pages — "
            "later pages unfetched; raise SCALEWAY_MAX_PAGES"
        ]
    return result


# ------------------------------------------------------------------ runpod

RUNPOD_URL = "https://api.runpod.io/graphql"
_RUNPOD_QUERY = {
    "query": "{gpuTypes{id displayName memoryInGb securePrice}}"
}


def parse_runpod(body: str) -> List[Dict[str, Any]]:
    """SECURE CLOUD ONLY — Community Cloud is excluded from the basket
    entirely (third-party-host marketplace capacity), so communityPrice is
    deliberately never requested, recorded, or normalized here."""
    rows: List[Dict[str, Any]] = []
    for g in json.loads(body)["data"]["gpuTypes"]:
        label = g.get("displayName") or g.get("id", "")
        sku = norm_sku(label)
        if sku not in BASKET_SKUS:
            continue
        price = g.get("securePrice")
        if not price:
            continue
        mem = g.get("memoryInGb")
        obs = _obs(
            sku,
            float(price),
            raw_value=str(price),
            region="global",
            notes=f"{label.replace('NVIDIA', '').strip()} secure {mem}GB",
        )
        obs["sku_identifier"] = label
        if isinstance(mem, (int, float)) and not isinstance(mem, bool):
            obs["memory_gb_label"] = mem
        rows.append(obs)
    return rows


def collect_runpod(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    body = fetch(
        RUNPOD_URL,
        data=json.dumps(_RUNPOD_QUERY).encode(),
        headers={"content-type": "application/json"},
        timeout=timeout,
    )
    return _result(
        "runpod", method="graphql", url=RUNPOD_URL, observations=parse_runpod(body)
    )


# ------------------------------------------------------------------ vast

VAST_URL_BASE = "https://console.vast.ai/api/v0/bundles/"
VAST_FETCH_LIMIT = 50  # the whole (thin) book per sku — ranked locally
VAST_RECORD_LIMIT = 5  # cheapest-per-GPU machines recorded per sku
# The cheapest-5 truncation used to run BEFORE the
# population screens, so the recorded "verified US/CA population" was
# whichever eligible machines happened to survive a whole-book cheapest-5
# cut — one-sided low bias, invisible in artifacts. For B200 the capture
# now ALSO records every verified-US/CA machine beyond the cheapest 5
# (rows marked book_scope=verified_us_ca_population), capped only at a
# generous safety bound with an explicit overflow flag in book_stats.
VAST_POPULATION_LIMIT = 200

# Remote strings printed into job logs must never carry control chars
# (in GH Actions a newline + '::' is a workflow command).
_VAST_LOG_CLEAN_RE = re.compile(r"[\r\n\x00-\x1f]")


def _vast_log_clean(text: Any, limit: int = 40) -> str:
    return _VAST_LOG_CLEAN_RE.sub(" ", str(text))[:limit]


def _vast_query(gpu_name: str, order: str = "asc") -> str:
    # Extraction-artifact hardening (a recorded $10.94 print vs a $6.25
    # live book): no
    # num_gpus preference and no `verified` filter. The old gte-8 preferred
    # query returned exactly one expensive host (short-circuiting the anyN
    # fallback), and verified:{eq:true} has broken semantics against the
    # live API — returned offers carry verified=null (the real field is the
    # STRING `verification`) and the filter silently dropped the cheapest
    # hosts, which are "unverified". Fetch the whole thin book, rank
    # locally, record verification as metadata instead of screening on it.
    # ``order`` matters: dph_total-ASC truncates the LARGEST totals first —
    # exactly the cheap-per-GPU multi-GPU boxes — so a full-limit response
    # triggers a second DESC fetch (see collect_vast).
    q: Dict[str, Any] = {
        "gpu_name": {"eq": gpu_name},
        "rentable": {"eq": True},
        "type": "on-demand",
        "order": [["dph_total", order]],
        "limit": VAST_FETCH_LIMIT,
    }
    return VAST_URL_BASE + "?q=" + urllib.parse.quote(json.dumps(q))


def _extraction_consistent(per_gpu: float, num_gpus: int, dph_total: float) -> bool:
    """L0 integrity: the print we are about to record, times its
    basis, must reproduce the raw offer total within display-rounding
    tolerance. True by construction TODAY — this is the tripwire that fires
    if a future edit derives price and raw_value from different fields."""
    return abs(round(per_gpu, 4) * num_gpus - dph_total) <= 0.005 * num_gpus


def parse_vast_offers(body: str) -> List[Dict[str, Any]]:
    """Every valid offer in the response, identity included — no selection.

    Selection (dedup + ranking) happens in select_vast_observations; this
    stays a pure parse so the full candidate set is loggable.
    """
    offers: List[Dict[str, Any]] = []
    for o in json.loads(body).get("offers", []):
        n_raw = o.get("num_gpus")
        dph = o.get("dph_total")
        n = (
            int(n_raw)
            if isinstance(n_raw, (int, float))
            and not isinstance(n_raw, bool)
            and float(n_raw).is_integer()
            else 0
        )
        if not (1 <= n <= 16) or dph is None:
            # A missing/zero/absurd num_gpus must never default to 1 — that
            # would record a whole-instance price as a per-GPU print.
            continue
        if o.get("is_bid"):
            # Belt-and-braces on top of type=on-demand in the query: a spot
            # BID recorded as an on-demand ask would bias the print low.
            continue
        per_gpu = float(dph) / n
        if not _extraction_consistent(per_gpu, n, float(dph)):
            # Loud, never silent: an offer failing the arithmetic tripwire
            # means the parser's fields have come apart (the 08-13 class).
            print(
                f"  vast L0 ANOMALY: offer={_vast_log_clean(o.get('id'), 20)} "
                f"price*basis cannot reproduce dph_total "
                f"({per_gpu:.4f}*{n} != {float(dph):.4f}) — offer EXCLUDED"
            )
            continue
        offers.append(
            {
                "offer_id": o.get("id"),
                "machine_id": o.get("machine_id"),
                "host_id": o.get("host_id"),
                "num_gpus": n,
                "dph_total": float(dph),
                "per_gpu": per_gpu,
                "geolocation": str(o.get("geolocation") or "?"),
                "verification": str(o.get("verification") or "?"),
                "gpu_name": str(o.get("gpu_name") or "?"),
            }
        )
    return offers


def _ranked_vast_machines(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Dedup by machine (keep its cheapest per-GPU offer), rank per-GPU
    ascending. L2 identity: rank by PER-GPU price, never by
    instance total — dph_total ordering buried $6.25/GPU 8x boxes beneath
    $10.94/GPU 1-2 GPU slices of one expensive host. One row per MACHINE
    so a single host listing every slice size can't crowd the record."""
    best_per_machine: Dict[Any, Dict[str, Any]] = {}
    for idx, cand in enumerate(candidates):
        key = cand.get("machine_id")
        if key is None:
            # Never collapse offers we can't attribute to a machine.
            key = ("unidentified", cand.get("offer_id"), idx)
        cur = best_per_machine.get(key)
        if cur is None or cand["per_gpu"] < cur["per_gpu"]:
            best_per_machine[key] = cand
    return sorted(best_per_machine.values(), key=lambda c: c["per_gpu"])


def _vast_row(
    cand: Dict[str, Any], gpu_name: str, note: str = ""
) -> Dict[str, Any]:
    row = _obs(
        gpu_name,
        cand["per_gpu"],
        raw_value=str(cand["dph_total"]),
        raw_unit="usd_per_instance_hr",
        gpu_count_basis=cand["num_gpus"],
        region=cand["geolocation"],
        notes=(
            f"{cand['num_gpus']}x rentable ask (dph incl default "
            f"storage), verification={cand['verification']}, cheapest "
            f"offer of machine {cand['machine_id']}"
            + (f" [{note}]" if note else "")
        ),
    )
    # L2 identity continuity: tomorrow's print must be attributable to
    # the same machine (same-machine vs book delta in capture screens).
    row.update(
        {
            "offer_id": cand["offer_id"],
            "machine_id": cand["machine_id"],
            "host_id": cand["host_id"],
            "verification": cand["verification"],
            "sku_identifier": cand.get("gpu_name", "?"),
        }
    )
    return row


def select_vast_observations(
    candidates: List[Dict[str, Any]], gpu_name: str, note: str = ""
) -> List[Dict[str, Any]]:
    """The recorded rows for one sku's book.

    Two populations, in order:
      1. the cheapest VAST_RECORD_LIMIT machines of the WHOLE book,
         unchanged — continuity for existing pins, the
         capture screens' lowest-print delta, and the B300 lane's min rule
         (any machine beyond these prices >= the recorded minimum, so the
         daily lowest-print observation is unaffected by what follows);
      2. B200 only: every OTHER verified-US/CA machine,
         per-GPU ascending, marked book_scope=verified_us_ca_population —
         the FULL population the vast statistic runs over,
         capped at VAST_POPULATION_LIMIT (overflow is recorded in
         book_stats by collect_vast, never silent).
    """
    ranked = _ranked_vast_machines(candidates)
    rows = [_vast_row(c, gpu_name, note) for c in ranked[:VAST_RECORD_LIMIT]]
    if gpu_name == "B200":
        population = [
            c
            for c in ranked[VAST_RECORD_LIMIT:]
            if us_ca_verified_host(c["verification"], c["geolocation"])
        ]
        for cand in population[:VAST_POPULATION_LIMIT]:
            row = _vast_row(
                cand,
                gpu_name,
                note=(note + " " if note else "")
                + "§5 population row (verified US/CA beyond cheapest-5)",
            )
            row["book_scope"] = "verified_us_ca_population"
            rows.append(row)
    return rows


def vast_book_stats(
    candidates: List[Dict[str, Any]], gpu_name: str
) -> Dict[str, Any]:
    """Truncation must never again be invisible: the
    per-sku book accounting recorded alongside the rows. Computed from the
    same dedup/rank/screen helpers select_vast_observations uses."""
    ranked = _ranked_vast_machines(candidates)
    eligible = [
        c
        for c in ranked
        if us_ca_verified_host(c["verification"], c["geolocation"])
    ]
    beyond = [
        c
        for c in ranked[VAST_RECORD_LIMIT:]
        if us_ca_verified_host(c["verification"], c["geolocation"])
    ]
    recorded = min(len(ranked), VAST_RECORD_LIMIT) + (
        min(len(beyond), VAST_POPULATION_LIMIT) if gpu_name == "B200" else 0
    )
    return {
        "machines_total": len(ranked),
        "verified_us_ca_machines": len(eligible),
        "rows_recorded": recorded,
        "population_overflow": bool(
            gpu_name == "B200" and len(beyond) > VAST_POPULATION_LIMIT
        ),
    }


def parse_vast(body: str, gpu_name: str, basis_note: str = "") -> List[Dict[str, Any]]:
    return select_vast_observations(
        parse_vast_offers(body), gpu_name, note=basis_note
    )


def collect_vast(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Marketplace asks — the whole thin book per sku, one call each.

    Rewritten after a live extraction artifact: no 8x-preferred
    query, no verified filter (see _vast_query), dedup by machine, rank by
    per-GPU. The FULL candidate set is logged every capture so the next bad
    print is diagnosable from the job log alone. This lane sends 2 skus x 1
    call per capture."""
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    book_stats: Dict[str, Dict[str, Any]] = {}
    for gpu_name in BASKET_SKUS:
        try:
            body = fetch(_vast_query(gpu_name), timeout=timeout)
            raw_count = len(json.loads(body).get("offers") or [])
            candidates = parse_vast_offers(body)
            if raw_count >= VAST_FETCH_LIMIT:
                # A full-limit ascending response may be TRUNCATED, and the
                # offers it cuts first are the largest totals — exactly the
                # cheap-per-GPU multi-GPU boxes (the 08-13 burial class).
                # Fetch the descending tail once and merge; still <=2 calls
                # per sku (the old code's worst case).
                print(
                    f"  vast {gpu_name}: book at fetch limit "
                    f"({raw_count} >= {VAST_FETCH_LIMIT}) — fetching the "
                    "descending tail to cover large-total offers"
                )
                desc_body = fetch(
                    _vast_query(gpu_name, order="desc"), timeout=timeout
                )
                desc_count = len(json.loads(desc_body).get("offers") or [])
                seen_ids = {
                    c["offer_id"]
                    for c in candidates
                    if c["offer_id"] is not None
                }
                for cand in parse_vast_offers(desc_body):
                    if (
                        cand["offer_id"] is None
                        or cand["offer_id"] not in seen_ids
                    ):
                        candidates.append(cand)
                if desc_count >= VAST_FETCH_LIMIT:
                    # Book wider than both windows combined: mid-book
                    # offers may be missing — visible, never silent.
                    errors.append(
                        f"{gpu_name}: book exceeds fetch coverage "
                        f"(>= {2 * VAST_FETCH_LIMIT} offers) — mid-book "
                        "offers may be missing from the candidate set"
                    )
            for c in candidates:
                print(
                    f"  vast candidate {gpu_name}: "
                    f"offer={_vast_log_clean(c['offer_id'], 20)} "
                    f"machine={_vast_log_clean(c['machine_id'], 20)} "
                    f"host={_vast_log_clean(c['host_id'], 20)} "
                    f"n={c['num_gpus']} ${c['per_gpu']:.4f}/gpu-hr "
                    f"(total {c['dph_total']:.4f}) "
                    f"geo={_vast_log_clean(c['geolocation'])} "
                    f"verification={_vast_log_clean(c['verification'], 20)}"
                )
            rows.extend(select_vast_observations(candidates, gpu_name))
            stats = vast_book_stats(candidates, gpu_name)
            book_stats[gpu_name] = stats
            if stats["population_overflow"]:
                # Loud twice: the flag persists in the snapshot AND the job
                # log says it at capture time.
                print(
                    f"  vast {gpu_name}: §5 population exceeds "
                    f"{VAST_POPULATION_LIMIT} machines — overflow recorded "
                    "in book_stats"
                )
        except Exception as exc:  # noqa: BLE001 — one sku's feed must not hide the other
            errors.append(f"{gpu_name}: {exc}")
    if not rows and errors:
        # Zero observations AND per-sku errors: surface the real causes
        # instead of the generic parsed-nothing message.
        raise RuntimeError(f"vast: every sku fetch failed: {'; '.join(errors)}")
    result = _result(
        "vast", method="api-json", url=VAST_URL_BASE, observations=rows
    )
    if errors:
        result["partial_errors"] = errors
    if book_stats:
        result["book_stats"] = book_stats
    return result


# ------------------------------------------------------------------ massedcompute

MASSED_URL = "https://vm.massedcompute.com/pricing"
_MASSED_KEY_RE = re.compile(
    # Widened from B300-only — the same page carries the B200
    # row the B200 basket needs; norm_sku + BASKET_SKUS still gate
    # what records (GB-prefixed lookalikes never normalize to B200/B300).
    r'\\"(?P<sku>[^"\\]*B[23]00[^"\\]*?)-x\s*(?P<count>\d+)-\d+\\"'
)
_MASSED_PRICE_RE = re.compile(r"\$\$([\d,]+(?:\.\d+)?)")
# Hardening (measured live 2026-08-16): bound each price window at
# the next row key of ANY sku — not just the next B-key — so a B-row whose
# price cell is a deferred RSC ref skips cleanly (zero prices in window)
# instead of surviving only because a neighbor's strays happen to pair up.
_MASSED_ANYKEY_RE = re.compile(r'\\"[^"\\]{1,60}?-x\s*\d+-\d+\\"')
# Row key -> its own price sit ~600-1100 chars apart in the RSC flight
# stream (measured 2026-08-10); 4000 gives slack without reaching into a
# neighboring row's price.
_MASSED_PRICE_WINDOW = 4000


def parse_massedcompute(html: str) -> List[Dict[str, Any]]:
    """Next.js RSC flight strings: row keys look like '"B300 SXM6-x 8-0"' and
    a literal $ is escaped as '$$'. Each row appears twice (mobile card +
    desktop table) — dedupe on (sku, count, price). No public JSON API exists
    (/api/pricing 307s to signin)."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    key_matches = list(_MASSED_KEY_RE.finditer(html))
    for i, m in enumerate(key_matches):
        sku_label = m.group("sku").replace("-mobile", "").strip()
        count = int(m.group("count"))
        # Bound the price window at the NEXT row key of ANY sku: a
        # priceless row ("contact us" / deferred RSC price ref) must never
        # steal a neighbor's price — with differing GPU counts that yields
        # a wrong-by-integer-factor print.
        end = m.end() + _MASSED_PRICE_WINDOW
        next_any = _MASSED_ANYKEY_RE.search(html, m.end())
        if next_any:
            end = min(end, next_any.start())
        window = html[m.end() : end]
        prices = _MASSED_PRICE_RE.findall(window)
        if len(prices) != 1 or not count:
            # Zero prices: a "contact us" row. More than one: ambiguous
            # (promo/strikethrough) — skip rather than guess which is real.
            continue
        node_price = float(prices[0].replace(",", ""))
        key = (sku_label, count, node_price)
        if key in seen:
            continue
        seen.add(key)
        sku = norm_sku(sku_label)
        if sku not in BASKET_SKUS:
            continue
        obs = _obs(
            sku,
            node_price / count,
            raw_value=prices[0],
            raw_unit="usd_per_node_hr",
            gpu_count_basis=count,
            region="unspecified",
            notes=f"{sku_label} x{count} node ${node_price}/hr (on-demand VM tier)",
        )
        # A future 'B200 NVL72'-style row label must reach the product-
        # identity token screen structurally — this page already names H-series rows with
        # NVL suffixes, so the convention is live.
        obs["sku_identifier"] = sku_label
        rows.append(obs)
    return rows


def collect_massedcompute(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    html = fetch(MASSED_URL, timeout=timeout)
    return _result(
        "massedcompute",
        method="html-regex",
        url=MASSED_URL,
        observations=parse_massedcompute(html),
    )


# ------------------------------------------------------------------ latitude

LATITUDE_URL = "https://www.latitude.sh/pricing"
_LATITUDE_SLUG_RE = re.compile(
    r'\\"slug\\":\\"(g4-b300-[a-z0-9]+)\\",\\"name\\":\\"([^\\"]+)\\"'
)
_LATITUDE_GPU_RE = re.compile(
    r'"gpu":\{"count":(\d+),"type":"([^"]+)","vram_per_gpu":(\d+)'
)
_LATITUDE_REGION_RE = re.compile(
    r'\{"name":"([^"]+)","deploys_instantly".*?"stock_level":"([^"]+)",'
    r'"pricing":\{"USD":\{"hour":([0-9.]+),"month":([0-9.]+),"year":([0-9.]+)',
    re.S,
)
HOURS_PER_MONTH = 730  # latitude's own monthly/hourly convention
# Plan blob (slug -> pricing -> OS list) measured ~850 chars in flight data;
# 8000 tolerates growth, and the 'available_operating_systems' truncation
# keeps the region regex from pairing across plans.
_LATITUDE_BLOB_WINDOW = 8000


def parse_latitude(html: str) -> List[Dict[str, Any]]:
    """Bare-metal plan blob (Next.js flight data). Prices are per NODE.
    Records BOTH the on-demand hourly print AND the monthly-commit tier
    (clearly labeled): an early $8.00 scoping ballpark turned out to be
    the monthly tier — hourly on-demand is $16/GPU-hr — so observing both
    is exactly the feed-visibility this lane exists for. The calc consumes
    tier == 'on-demand' only."""
    slug_m = _LATITUDE_SLUG_RE.search(html)
    if not slug_m:
        raise RuntimeError("latitude: no g4-b300 plan blob found")
    window = html[slug_m.start() : slug_m.start() + _LATITUDE_BLOB_WINDOW]
    cut = window.find("available_operating_systems")
    if cut != -1:
        window = window[:cut]
    plain = window.replace('\\"', '"')
    gpu_m = _LATITUDE_GPU_RE.search(plain)
    if not gpu_m:
        raise RuntimeError("latitude: plan blob has no gpu spec — shape changed")
    count = int(gpu_m.group(1))
    vram = gpu_m.group(3)
    plan = slug_m.group(1)
    rows: List[Dict[str, Any]] = []
    for name, stock, hour, month, _year in _LATITUDE_REGION_RE.findall(plain):
        base_note = (
            f"{plan} {count}x {gpu_m.group(2)} {vram}GB/GPU, stock_level={stock}"
        )
        rows.append(
            _obs(
                "B300",
                float(hour) / count,
                raw_value=hour,
                raw_unit="usd_per_node_hr",
                gpu_count_basis=count,
                region=name,
                notes=base_note,
            )
        )
        rows.append(
            _obs(
                "B300",
                float(month) / HOURS_PER_MONTH / count,
                raw_value=month,
                raw_unit="usd_per_node_month",
                gpu_count_basis=count,
                tier="monthly-commit",
                region=name,
                notes=base_note + " (committed tier — excluded from the underlying rate)",
            )
        )
    return rows


def collect_latitude(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    html = fetch(LATITUDE_URL, timeout=timeout)
    return _result(
        "latitude",
        method="html-regex",
        url=LATITUDE_URL,
        observations=parse_latitude(html),
    )


# ------------------------------------------------------------------ e2e (B200 pool)

E2E_URL = "https://www.e2enetworks.com/pricing?region=us"
E2E_B200_VRAM_GB = "192"
# The stripped pricing row is ~120 chars; 3000 spans it even with markup
# churn while staying inside the B200 section.
_E2E_ROW_WINDOW = 3000


# The monthly price cell divided by the hourly one lands near 730 (E2E's own
# hours-per-month convention); a reordered/promoted column breaks the ratio.
E2E_MONTHLY_RATIO_BAND = (400.0, 1100.0)


def parse_e2e(html: str) -> List[Dict[str, Any]]:
    """USD price list on E2E's own pricing page (server-rendered table; the
    separate /gpus/nvidia-b200 page quotes INR — the two cross-check at
    ~96 INR/USD). Two pins guard the price cell: the numeric spec triple
    (192 vRAM / vCPU / RAM) must immediately precede it, and the NEXT price
    cell must be ~730x it (the monthly column) — a column reorder fails
    loudly instead of silently promoting the wrong cell."""
    idx = html.find("NVIDIA B200")
    if idx == -1:
        raise RuntimeError("e2e: no NVIDIA B200 row on the pricing page")
    window = html[idx : idx + _E2E_ROW_WINDOW].replace("<!-- -->", "")
    txt = re.sub(r"<[^>]+>", "|", window)
    cells = [c.strip() for c in txt.split("|") if c.strip()]
    dollar_idx = next(
        (i for i, c in enumerate(cells) if c.startswith("$")), None
    )
    if dollar_idx is None:
        raise RuntimeError("e2e: no $ price cell after the B200 anchor")
    spec_triple = cells[max(0, dollar_idx - 3) : dollar_idx]
    if (
        len(spec_triple) != 3
        or not all(re.fullmatch(r"\d+", c) for c in spec_triple)
        or spec_triple[0] != E2E_B200_VRAM_GB
    ):
        raise RuntimeError(
            f"e2e: B200 row identity check failed — expected the "
            f"{E2E_B200_VRAM_GB}/vCPU/RAM spec triple right before the price, "
            f"got {cells[:dollar_idx]!r}"
        )
    raw = cells[dollar_idx]
    price = float(raw.lstrip("$").replace(",", ""))
    monthly_cell = next(
        (c for c in cells[dollar_idx + 1 :] if c.startswith("$")), None
    )
    if monthly_cell is None:
        raise RuntimeError("e2e: no monthly cell to cross-check the hourly price")
    monthly = float(monthly_cell.lstrip("$").replace(",", ""))
    lo, hi = E2E_MONTHLY_RATIO_BAND
    if price <= 0 or not (lo <= monthly / price <= hi):
        raise RuntimeError(
            f"e2e: hourly/monthly cross-check failed (monthly/hourly = "
            f"{monthly / price:.0f}, expected ~730) — refusing a possible "
            "column reorder"
        )
    return [
        _obs(
            "B200",
            price,
            raw_value=raw,
            region="IN (India DCs; USD price list)",
            notes="192GB vRAM, per-minute billing; hourly cell of the USD table",
        )
    ]


def collect_e2e(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    html = fetch(E2E_URL, timeout=timeout)
    return _result(
        "e2e", method="html-regex", url=E2E_URL, observations=parse_e2e(html)
    )


# ------------------------------------------------------------------ shadeform (B200 pool only)

SHADEFORM_URLS = ("https://shadeform.com/", "https://www.shadeform.com/")
_SHADEFORM_BLOB_RE = re.compile(r'\{"cloud":".*?"deployment_type":"[a-z_]+"\}')


def parse_shadeform_b200(html: str) -> List[Dict[str, Any]]:
    """B200 rows only — Shadeform's B300 listing re-publishes Verda and is
    excluded from the basket outright (double-count); recording it here would
    smuggle a double-count into the store."""
    txt = html.replace('\\"', '"')
    rows: List[Dict[str, Any]] = []
    seen = set()
    for m in _SHADEFORM_BLOB_RE.finditer(txt):
        blob = m.group(0)
        if blob in seen:
            continue
        seen.add(blob)
        try:
            rec = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if norm_sku(rec.get("gpu_type")) != "B200":
            continue
        n = rec.get("num_gpus") or 0
        cents = rec.get("hourly_price")
        if not n or cents is None:
            continue
        regions = [a.get("display_name", "") for a in rec.get("availability", [])]
        us = any(str(r).startswith("US") for r in regions)
        rows.append(
            _obs(
                "B200",
                cents / 100.0 / n,
                raw_value=str(cents),
                raw_unit="cents_per_instance_hr",
                gpu_count_basis=int(n),
                region="US" if us else "non-US",
                notes=f"via {rec.get('cloud')}",
            )
        )
    return rows


def collect_shadeform(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    html = None
    for url in SHADEFORM_URLS:
        try:
            html = fetch(url, timeout=timeout)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if html is None:
        raise RuntimeError(f"shadeform unreachable: {last_err}")
    return _result(
        "shadeform",
        method="html-json-blobs",
        url=SHADEFORM_URLS[0],
        observations=parse_shadeform_b200(html),
        first_party=False,  # reseller re-publishing other clouds' capacity
    )


# ------------------------------------------------------------------ lambda
#
# B200 basket constituent. The on-demand INSTANCE rate, never
# the "1-Click Clusters" committed rate ($8.87-9.86, 2wk-1yr terms) that
# appears FIRST in document order. Server-rendered HubSpot page; every table
# also exists as a unicode-escaped duplicate inside a window.__islands blob,
# which the plain-quote regex structurally cannot match (recipe re-verified
# 2026-08-16 — a layout change that unescapes it would yield 2 matches and
# trip the exactly-one fence).

LAMBDA_URL = "https://lambda.ai/pricing"
_LAMBDA_INSTANCES_ANCHOR = '<h2 class="h2">Instances pricing</h2>'
_LAMBDA_CLUSTER_PLAN = "NVIDIA HGX B200"  # committed-tier plan string
# Row identity: data-plan attribute + the 8x tab's unique spec columns
# (208 vCPUs / 2900 GiB RAM / 22 TiB SSD — the 4x/2x/1x tabs price the same
# plan string at $6.79/$6.89/$6.99, so the specs are load-bearing).
# Gaps are tempered to NEVER cross a row boundary: with plain [^$]*?, an
# 8x price cell losing its '$' (em-dash outage) would let the pins bridge
# into the 4x row and silently record $6.79 — the gap may cross cells
# within the row, never </tr>.
_LAMBDA_GAP = r'(?:(?!</tr>)[^$])*?'
_LAMBDA_B200_ROW_RE = re.compile(
    r'data-plan="NVIDIA B200 SXM6"' + _LAMBDA_GAP
    + r'data-label="VRAM/GPU">180 GB</td>' + _LAMBDA_GAP
    + r'data-label="vCPUs">208</td>' + _LAMBDA_GAP
    + r'data-label="RAM">2900 GiB</td>' + _LAMBDA_GAP
    + r'data-label="STORAGE">22 TiB SSD</td>' + _LAMBDA_GAP
    + r'data-label="PRICE/GPU/HR\*">\$(\d+\.\d{2})</td>'
)


def parse_lambda(html: str) -> List[Dict[str, Any]]:
    if _LAMBDA_INSTANCES_ANCHOR not in html:
        raise RuntimeError(
            "lambda: 'Instances pricing' heading missing — page reshaped; "
            "refusing to scan (the committed 1-Click Clusters table would "
            "be first match)"
        )
    scope = html.split(_LAMBDA_INSTANCES_ANCHOR, 1)[1]
    if _LAMBDA_CLUSTER_PLAN in scope:
        raise RuntimeError(
            "lambda: committed-tier plan string leaked into the on-demand "
            "scope — layout changed; refusing to extract"
        )
    matches = _LAMBDA_B200_ROW_RE.findall(scope)
    if len(matches) != 1:
        raise RuntimeError(
            f"lambda: expected exactly one pinned B200 SXM6 8x row, found "
            f"{len(matches)} — page reshaped"
        )
    price = float(matches[0])
    obs = _obs(
        "B200",
        price,
        raw_value=f"${matches[0]}",
        raw_unit="usd_per_gpu_hr",
        gpu_count_basis=8,
        region="unspecified (region-uniform list price)",
        notes=(
            "Instances pricing 8x tab, per-GPU list rate pre-tax "
            "(footnote: plus applicable sales tax/VAT/GST)"
        ),
    )
    obs.update(
        {"sku_identifier": "NVIDIA B200 SXM6", "memory_gb_label": 180}
    )
    return [obs]


def collect_lambda(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    body = fetch(LAMBDA_URL, timeout=timeout)
    return _result(
        "lambda", method="html", url=LAMBDA_URL, observations=parse_lambda(body)
    )


# ------------------------------------------------------------------ coreweave
#
# B200 basket constituent. Per-instance pricing (8-GPU HGX
# B200, $68.80/hr NA on-demand). THE LIVE PRODUCT-IDENTITY TRAP: the same table carries
# "NVIDIA GB200 NVL72" (data-product contains the substring 'b200', GPU
# Count cell is literal '4^1', VRAM 186, $42.00 per 4 Grace-coupled GPUs)
# and GB300 rows — the segment pins below exist to keep the recipe off
# them. The collector sends the project User-Agent defined in
# gpu_index.common.http.

COREWEAVE_URL = "https://www.coreweave.com/pricing"
_COREWEAVE_ROW_SPLIT = '<div role="listitem" class="table-row-v2 w-dyn-item'
_COREWEAVE_B200_H3 = (
    '<h3 data-product="nvidia-b200" class="table-model-name">NVIDIA HGX B200</h3>'
)
_COREWEAVE_H3_RE = re.compile(
    r'<h3 data-product="[^"]+" class="table-model-name">([^<]+)</h3>'
)
_COREWEAVE_META_RE = r'<div class="table-meta-value">(\d+)</div>\s*<div>{label}</div>'
# Self-labeled on-demand span (verified live 2026-08-16): the row also
# carries an UNLABELED duplicate '<div>$68.80</div>' cell plus labeled
# spot/inference spans — only this exact labeled shape is the print.
_COREWEAVE_PRICE_RE = re.compile(
    r'<span class="instance-price">On-Demand Price:\s*'
    r'<span class="item-value">\$([\d,]+\.\d{2})</span>\s*/\s*Hour'
)


def parse_coreweave(html: str) -> List[Dict[str, Any]]:
    try:
        gpu_section = html.split("On-demand GPU instances", 1)[1].split(
            "On-demand CPU instances", 1
        )[0]
        north_america = gpu_section.split("REGION: NORTH AMERICA", 1)[1].split(
            "REGION: EUROPE", 1
        )[0]
    except IndexError as exc:
        raise RuntimeError(
            "coreweave: section anchors missing — page reshaped"
        ) from exc
    segments = [
        seg
        for seg in north_america.split(_COREWEAVE_ROW_SPLIT)
        if _COREWEAVE_B200_H3 in seg
    ]
    if len(segments) != 1:
        raise RuntimeError(
            f"coreweave: expected exactly one HGX B200 row segment, found "
            f"{len(segments)}"
        )
    seg = segments[0]
    if "nvidia-gb200-nvl72" in seg:
        raise RuntimeError(
            "coreweave: GB200 NVL72 content leaked into the B200 row "
            "segment — refusing to extract (product-identity screen)"
        )
    names = set(_COREWEAVE_H3_RE.findall(seg))
    if names != {"NVIDIA HGX B200"}:
        raise RuntimeError(
            f"coreweave: B200 row segment is impure ({sorted(names)}) — "
            "a neighbor row leaked in"
        )
    count_m = re.search(_COREWEAVE_META_RE.format(label="GPU Count"), seg)
    vram_m = re.search(_COREWEAVE_META_RE.format(label="VRAM"), seg)
    if not count_m or int(count_m.group(1)) != 8:
        # GB200/GB300 count cells are literal '4^1' text and fail the int
        # pattern — this pin is the product-identity screen made mechanical.
        raise RuntimeError("coreweave: GPU Count pin != 8 — wrong row")
    if not vram_m or int(vram_m.group(1)) != 180:
        raise RuntimeError("coreweave: VRAM pin != 180 — wrong row")
    prices = _COREWEAVE_PRICE_RE.findall(seg)
    if len(prices) != 1:
        raise RuntimeError(
            f"coreweave: expected exactly one labeled On-Demand Price in "
            f"the B200 row, found {len(prices)}"
        )
    node_price = float(prices[0].replace(",", ""))
    obs = _obs(
        "B200",
        node_price / 8,
        raw_value=prices[0],
        raw_unit="usd_per_instance_hr",
        gpu_count_basis=8,
        region="US (REGION: NORTH AMERICA table)",
        notes=(
            f"NVIDIA HGX B200 8-GPU instance ${node_price}/hr on-demand "
            "(labeled span; inference/spot tiers excluded)"
        ),
    )
    obs.update(
        {
            "sku_identifier": "NVIDIA HGX B200 (data-product=nvidia-b200)",
            "memory_gb_label": 180,
        }
    )
    return [obs]


def collect_coreweave(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    body = fetch(COREWEAVE_URL, timeout=timeout)
    return _result(
        "coreweave",
        method="html",
        url=COREWEAVE_URL,
        observations=parse_coreweave(body),
    )


# ------------------------------------------------------------------ together
#
# B200 basket constituent. GPU Clusters on-demand rate ONLY
# (256-GPU minimum, disclosed). CRITICAL: the Dedicated Inference section
# carries a BYTE-IDENTICAL 'NVIDIA HGX B200' row at $8.99 — the extraction
# regex matches it page-wide, so the section slice is load-bearing and the
# regex must never run outside it. GB200/GB300 NVL72 rows sit in the SAME
# visible table with em-dash prices; only the literal HGX B200 span +
# immediate price-cell adjacency discriminates.

TOGETHER_URL = "https://www.together.ai/pricing"
_TOGETHER_SLICE_START = 'id="gpu-clusters" class="section-anchor'
_TOGETHER_SLICE_END = 'id="sandbox" class="section-anchor'
_TOGETHER_IDENTITY_PINS = (
    "On-demand hourly rates and reserved capacity",
    ">ON-Demand<",
    "All prices are per GPU per hour",
    # Header-ORDER pin (verified exactly-one occurrence live 2026-08-16):
    # binds column 1 = ON-Demand = is-right-border, so a column reorder
    # that keeps the border class on the first cell fails loud instead of
    # silently recording a reserved tenor as on-demand.
    'caption-m">Hardware</p></div></div></th><th class="is-right-border">'
    '<div class="pricing_inline"><div class="opacity-70">'
    '<p class="caption-m">ON-Demand</p>',
)
_TOGETHER_B200_PRICE_RE = re.compile(
    r'<span>HGX B200</span></p></td><td class="is-right-border">'
    r'<div class="opacity-70"><p data-batch="" class="body-m text-weight-medium">'
    r"\s*\$(\d+\.\d{2})</p>"
)


def parse_together(html: str) -> List[Dict[str, Any]]:
    start = html.find(_TOGETHER_SLICE_START)
    end = html.find(_TOGETHER_SLICE_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "together: gpu-clusters/sandbox section anchors missing or "
            "reordered — refusing to scan (the $8.99 Dedicated Inference "
            "HGX B200 row is byte-identical to the eligible row)"
        )
    section = html[start:end]
    for pin in _TOGETHER_IDENTITY_PINS:
        if pin not in section:
            raise RuntimeError(
                f"together: identity pin {pin!r} missing from the "
                "gpu-clusters section — layout changed"
            )
    matches = _TOGETHER_B200_PRICE_RE.findall(section)
    if len(matches) != 1:
        raise RuntimeError(
            f"together: expected exactly one HGX B200 on-demand cell in "
            f"the gpu-clusters slice, found {len(matches)}"
        )
    obs = _obs(
        "B200",
        float(matches[0]),
        raw_value=f"${matches[0]}",
        raw_unit="usd_per_gpu_hr",
        gpu_count_basis=1,
        region="unspecified",
        notes=(
            "GPU Clusters on-demand rate, per GPU (256-GPU minimum — a "
            "large-quantity rate, disclosed); Dedicated "
            "Inference SKU excluded by section slice"
        ),
    )
    obs.update({"sku_identifier": "NVIDIA HGX B200", "memory_gb_label": 180})
    return [obs]


def collect_together(timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    body = fetch(TOGETHER_URL, timeout=timeout)
    return _result(
        "together",
        method="html",
        url=TOGETHER_URL,
        observations=parse_together(body),
    )


# ------------------------------------------------------------------ registry

COLLECTORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "verda": collect_verda,
    "nebius": collect_nebius,
    "hyperstack": collect_hyperstack,
    "scaleway": collect_scaleway,
    "runpod": collect_runpod,
    "vast": collect_vast,
    "massedcompute": collect_massedcompute,
    "latitude": collect_latitude,
    "e2e": collect_e2e,
    "shadeform": collect_shadeform,
    # B200 basket constituents:
    "lambda": collect_lambda,
    "coreweave": collect_coreweave,
    "together": collect_together,
}
