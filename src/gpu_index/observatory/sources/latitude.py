# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Latitude.sh -- bare-metal GPU plan blobs (Next.js flight data), all plans.

Observatory generalization of the basket lane's parse_latitude (pinned to
the single g4-b300 plan there; that recipe is untouched): this collector
iterates EVERY bare-metal plan blob in the flight data and records the ones
with a real GPU spec -- hourly (on-demand) AND monthly-commit tiers, per
region, in EVERY currency the page publishes. Prices are per NODE; the
per-GPU normalization divides by the plan's own stated gpu count
(gpu_count_basis) and, for the monthly tier, by latitude's 730 hours/month
convention (basket-proven).

Shape verified live 2026-08-22:
  - plan blobs sit in escaped flight JSON as slug/name pairs with the slug
    FIRST -- that field order is the discriminator: the vm-* virtual-machine
    entities publish name-first (and carry gpu null), and country/location
    entities never pair slug directly with name;
  - CPU-only bare-metal plans carry an EMPTY gpu object -- skipped. A
    non-empty gpu object the strict spec regex cannot read is a reshape and
    RAISES rather than silently dropping a GPU plan from the record;
  - each region row carries stock_level plus a pricing map of per-currency
    {hour, month, year} triples -- USD and BRL today, and BRL is latitude's
    own separately-set list price (5.5x USD on g3-h100-small vs 5.0x on the
    g4 plans), not an FX echo, so it is recorded natively via currency=;
  - an unavailable region publishes null/0 prices (United Kingdom on
    g4-rtx6kpro-large today): null/0 = tier not offered, skipped, never a
    $0 print. A priced region can still be stock_level "unavailable"
    (published list price; stock is metadata, recorded in extra);
  - each region row also carries, between its name anchor and its
    stock_level: deploys_instantly (OS slugs deployable instantly),
    locations.available (site codes where the plan can deploy at all) and
    locations.in_stock (site codes with inventory right now) -- recorded
    VERBATIM in extra. Site-level beats region-level: live SUBSET cases
    exist where a region is priced and stocked but only some of its sites
    hold inventory. These are availability METADATA, never a price gate:
    a missing/reshaped locations map or deploys list notes a
    partial_error and the priced observation still records (stock_level
    stays the primary region-shape tripwire). The apparent rules
    (site-in-in_stock inherits the region grade; empty deploys_instantly
    co-occurs with "unavailable") are correlation, not contract -- raw
    fields only, derive nothing.

The yearly figure in the pricing triple is deliberately NOT recorded as a
tier -- the lane config's contract for this source is hourly +
monthly-commit, and the year price is derivable from a future catalog
decision, not lost (raw flight data keeps publishing it).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "latitude"

URL = "https://www.latitude.sh/pricing"

# Escaped flight JSON: literal backslash-quote around every key/value. The
# slug-before-name order is load-bearing (see module docstring).
_PLAN_PAIR_RE = re.compile(
    r'\\"slug\\":\\"([a-z0-9-]+)\\",\\"name\\":\\"([^\\"]+)\\"'
)
# Strict field order, verified live. Matching a nonempty gpu object with
# anything looser would risk pairing a count with the wrong plan.
_GPU_SPEC_RE = re.compile(
    r'"gpu":\{"count":(\d+),"type":"([^"]+)","vram_per_gpu":(\d+)'
)
# Any gpu value at all (flat object or null) -- distinguishes "CPU plan"
# (empty/null, skip) from "GPU spec reshaped" (nonempty but unreadable,
# raise).
_GPU_ANY_RE = re.compile(r'"gpu":(\{[^{}]*\}|null)')
_INTERCONNECT_RE = re.compile(r'"interconnect":(?:null|"([^"]*)")')
_REGION_START_RE = re.compile(r'\{"name":"([^"]+)","deploys_instantly"')
_STOCK_MARKER = '"stock_level":'
_STOCK_RE = re.compile(r'"stock_level":"([^"]+)"')
# Per-site availability metadata riding between the region anchor and
# stock_level (verbatim field order live). The captured group must be a
# FLAT list of quoted strings, whole-shape: object/numeric/nested items
# fail the match entirely (a partial_error), never fabricate site codes
# or an affirmative empty list out of a reshaped payload.
_STR_LIST = r'(?:"[^"]*"(?:,"[^"]*")*)?'
_LOCATIONS_RE = re.compile(
    r'"locations":\{"available":\[(' + _STR_LIST + r')\],'
    r'"in_stock":\[(' + _STR_LIST + r')\]\}'
)
_DEPLOYS_RE = re.compile(r'"deploys_instantly":\[(' + _STR_LIST + r')\]')
_QUOTED_ITEM_RE = re.compile(r'"([^"]*)"')
_PRICING_KEY = '"pricing":{'
# One per-currency block inside pricing: {hour, month, year}, each a plain
# number or null.
_CURRENCY_TRIPLE_RE = re.compile(
    r'"([A-Z]{3})":\{'
    r'"hour":(null|[0-9]+(?:\.[0-9]+)?),'
    r'"month":(null|[0-9]+(?:\.[0-9]+)?),'
    r'"year":(null|[0-9]+(?:\.[0-9]+)?)'
)

HOURS_PER_MONTH = 730  # latitude's own monthly/hourly convention
# Plan blobs are normally bounded by the NEXT plan pair; only the last blob
# needs the cap. Measured ~850-1900 chars live; 8000 tolerates growth (the
# same figure the basket lane uses).
_BLOB_WINDOW = 8000
_OS_LIST_MARKER = "available_operating_systems"


def _quoted_items(group: str) -> List[str]:
    """Items of a flat JSON string list, verbatim (site codes, OS slugs)."""
    return _QUOTED_ITEM_RE.findall(group)


def _num(text: str) -> Optional[float]:
    """A published price, or None for null/0 (this page's two spellings of
    'not offered' -- the UK region prints null USD alongside 0 BRL)."""
    if text == "null":
        return None
    value = float(text)
    return value if value > 0 else None


def parse_latitude(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    pairs = list(_PLAN_PAIR_RE.finditer(html))
    if not pairs:
        raise RuntimeError(
            "latitude: no bare-metal plan blobs found in flight data -- "
            "page shape changed"
        )
    rows: List[Dict[str, Any]] = []
    partial_errors: List[str] = []
    # (slug, region, currency, tier) -> raw value: flight payloads can
    # duplicate blobs; identical repeats collapse, CONFLICTING repeats are
    # ambiguous and noted rather than guessed between.
    seen: Dict[Tuple[str, str, str, str], str] = {}
    for i, m in enumerate(pairs):
        slug, plan_name = m.group(1), m.group(2)
        bounded_by_next = i + 1 < len(pairs)
        end = (
            pairs[i + 1].start()
            if bounded_by_next
            else min(m.start() + _BLOB_WINDOW, len(html))
        )
        window = html[m.start() : end]
        cut = window.find(_OS_LIST_MARKER)
        if cut != -1:
            window = window[:cut]
        plain = window.replace('\\"', '"')
        if '"gpu":' not in plain:
            continue  # not a machine-plan blob shape we recognize as GPU
        gpu_any = _GPU_ANY_RE.search(plain)
        if not gpu_any:
            raise RuntimeError(
                f"latitude: plan {slug} has a gpu field the parser cannot "
                "read at all -- spec shape changed; refusing to silently "
                "drop a possible GPU plan"
            )
        if gpu_any.group(1) in ("null", "{}"):
            continue  # CPU-only plan -- out of scope for a GPU observatory
        gpu = _GPU_SPEC_RE.search(plain)
        if not gpu:
            raise RuntimeError(
                f"latitude: plan {slug} has a non-empty gpu spec the parser "
                "cannot read -- field order/shape changed; refusing to "
                "silently drop a GPU plan"
            )
        count = int(gpu.group(1))
        gpu_type = gpu.group(2)
        vram = int(gpu.group(3))
        if count < 1:
            partial_errors.append(
                f"{slug}: gpu count {count} -- per-GPU normalization "
                "impossible, plan skipped"
            )
            continue
        if not bounded_by_next and cut == -1:
            partial_errors.append(
                f"{slug}: blob window capped at {_BLOB_WINDOW} chars without "
                "an OS-list terminator -- region list may be incomplete"
            )
        inter_m = _INTERCONNECT_RE.search(plain)
        interconnect = inter_m.group(1) if inter_m else None
        region_starts = list(_REGION_START_RE.finditer(plain))
        if not region_starts:
            if '"regions":[{' in plain:
                raise RuntimeError(
                    f"latitude: GPU plan {slug} has a non-empty regions "
                    "list the parser cannot read -- region shape changed"
                )
            continue  # genuinely no regions listed for this plan
        # Every region row carries stock_level; a row the start regex cannot
        # read is otherwise invisible (absorbed into its neighbor's
        # segment), so a marker/match mismatch is the tripwire for a
        # PARTIALLY reshaped regions list.
        n_stock_markers = plain.count(_STOCK_MARKER)
        if n_stock_markers != len(region_starts):
            partial_errors.append(
                f"{slug}: {n_stock_markers} stock_level markers vs "
                f"{len(region_starts)} readable region rows -- region shape "
                "partially changed, unreadable rows skipped"
            )
        base_note = f"{slug} {count}x {gpu_type} {vram}GB/GPU bare metal"
        for j, rm in enumerate(region_starts):
            seg_end = (
                region_starts[j + 1].start()
                if j + 1 < len(region_starts)
                else len(plain)
            )
            seg = plain[rm.start() : seg_end]
            region_name = rm.group(1)
            pricing_at = seg.find(_PRICING_KEY)
            # This row's OWN stock_level is the FIRST marker in the segment
            # and must precede the row's own pricing map (published field
            # order: name, deploys_instantly, locations, stock_level,
            # pricing). A first marker sitting past pricing_at belongs to
            # an absorbed unreadable neighbor row, and a first marker whose
            # value is not a quoted string is this row's stock reshaped --
            # both are "missing stock_level". Searching the whole segment
            # instead would let an absorbed row donate its stock grade and
            # site lists to this region's name.
            marker_at = seg.find(_STOCK_MARKER)
            own_marker = 0 <= marker_at and (
                pricing_at == -1 or marker_at < pricing_at
            )
            stock_m = _STOCK_RE.match(seg, marker_at) if own_marker else None
            if not stock_m or pricing_at == -1:
                missing = "stock_level" if not stock_m else "pricing"
                partial_errors.append(
                    f"{slug}/{region_name}: region row missing {missing} -- "
                    "skipped"
                )
                continue
            # Per-site availability metadata sits between the region anchor
            # and this row's own stock_level. Bound the search there: an
            # unreadable neighbor row absorbed into this segment can never
            # donate its site lists to this region's name (same rule as the
            # brace-matched pricing map below). A miss is a partial_error,
            # never fatal -- the priced observation still records.
            head = seg[: stock_m.start()]
            loc_m = _LOCATIONS_RE.search(head)
            if not loc_m:
                partial_errors.append(
                    f"{slug}/{region_name}: region row missing a readable "
                    "locations map -- site availability not recorded"
                )
            deploys_m = _DEPLOYS_RE.search(head)
            if not deploys_m:
                partial_errors.append(
                    f"{slug}/{region_name}: region row missing a readable "
                    "deploys_instantly list -- instant-deploy flag not "
                    "recorded"
                )
            # Brace-match the pricing map so triples are read ONLY from this
            # region's own pricing -- an unreadable neighboring row absorbed
            # into this segment must never donate its prices to this
            # region's name. Currency blocks are the depth-1 -> depth-2
            # transitions, whatever their inner shape, so the
            # unreadable-block note fires on reshaped blocks too (not just
            # on blocks that still happen to start with "hour").
            map_start = pricing_at + len(_PRICING_KEY) - 1
            depth = 0
            map_end = -1
            n_blocks = 0
            for k in range(map_start, len(seg)):
                ch = seg[k]
                if ch == "{":
                    depth += 1
                    if depth == 2:
                        n_blocks += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        map_end = k + 1
                        break
            if map_end == -1:
                partial_errors.append(
                    f"{slug}/{region_name}: pricing map never closes -- "
                    "truncated blob, region skipped"
                )
                continue
            triples = _CURRENCY_TRIPLE_RE.findall(seg[map_start:map_end])
            if len(triples) != n_blocks:
                partial_errors.append(
                    f"{slug}/{region_name}: {n_blocks - len(triples)} of "
                    f"{n_blocks} currency blocks unreadable -- skipped those"
                )
            extra: Dict[str, Any] = {
                "plan": slug,
                "plan_name": plan_name,
                "stock_level": stock_m.group(1),
            }
            # Verbatim per-site lists; keys stay ABSENT on a miss (noted
            # above), never fabricated.
            if loc_m:
                extra["locations_available"] = _quoted_items(loc_m.group(1))
                extra["locations_in_stock"] = _quoted_items(loc_m.group(2))
            if deploys_m:
                extra["deploys_instantly"] = _quoted_items(deploys_m.group(1))
            if interconnect:
                extra["interconnect"] = interconnect
            for cur, hour_s, month_s, _year_s in triples:
                hour = _num(hour_s)
                month = _num(month_s)
                unit_cur = cur.lower()
                surfaces = []
                if hour is not None:
                    surfaces.append(
                        (
                            "on-demand",
                            hour / count,
                            hour_s,
                            f"{unit_cur}_per_node_hr",
                            f"{base_note}, hourly",
                        )
                    )
                if month is not None:
                    surfaces.append(
                        (
                            "monthly-commit",
                            month / HOURS_PER_MONTH / count,
                            month_s,
                            f"{unit_cur}_per_node_month",
                            f"{base_note}, monthly commit "
                            f"({HOURS_PER_MONTH}h/month convention)",
                        )
                    )
                for tier, per_gpu, raw_s, raw_unit, note in surfaces:
                    key = (slug, region_name, cur, tier)
                    prev = seen.get(key)
                    if prev == raw_s:
                        continue  # duplicated flight payload, identical print
                    if prev is not None:
                        partial_errors.append(
                            f"{slug}/{region_name}/{cur}/{tier}: conflicting "
                            f"duplicate prints ({prev} vs {raw_s}) -- kept "
                            "the first, ambiguity noted"
                        )
                        continue
                    seen[key] = raw_s
                    obs = observation(
                        sku_identifier=gpu_type,
                        price_per_gpu_hr=per_gpu,
                        currency=cur,
                        raw_value=raw_s,
                        raw_unit=raw_unit,
                        gpu_count_basis=count,
                        tier=tier,
                        region=region_name,
                        notes=note,
                        extra=dict(extra),
                    )
                    obs["memory_gb_label"] = vram
                    rows.append(obs)
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    rows, partial_errors = parse_latitude(html)
    return result(
        SOURCE_ID,
        method="html-regex",
        url=URL,
        observations=rows,
        partial_errors=partial_errors or None,
    )
