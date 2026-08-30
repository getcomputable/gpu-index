# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""CoreWeave -- pricing page, on-demand GPU instance tables, BOTH regions.

Widens the basket lane's coreweave recipe (parse_coreweave in
gpu_index.index.sources -- single HGX B200 segment, NA table only; separate module,
separate fetch, the basket recipe is untouched) to EVERY GPU row segment in
both region tables (NORTH AMERICA + EUROPE, region recorded per print).
Page shape verified live 2026-08-22; every basket trap is inherited and
generalized:

  - the CPU section BELOW the GPU section uses the SAME row-split marker
    and the SAME table-model-name h3 shape (AMD Genoa/... rows with $
    prices) -- the GPU/CPU section slice is the fence that keeps CPU prices
    out, so both section anchors are required to appear EXACTLY once in the
    body;
  - 'REGION: NORTH AMERICA' also appears in the CPU section, so region
    splitting happens INSIDE the GPU slice only, and the slice must carry
    exactly the two known region headings in order -- a third region table
    would otherwise silently record under EUROPE;
  - each row segment carries its h3 label TWICE (desktop grid + mobile
    column); identity is the distinct (data-product, h3 text) pair and a
    segment carrying two DIFFERENT labels means the row-split marker no
    longer isolates rows, so the parse raises (every boundary is suspect)
    rather than misattributing a price;
  - prices come ONLY from the row's self-labeled spans ('On-Demand Price:'
    / 'Spot Price:'); the desktop grid duplicates the same figures in
    UNLABELED cells that must never be the pin. An empty or 'N/A'
    item-value is the page's own not-offered encoding (its JS hides those
    spans) -> tier skipped; a fully unpriced row (GB300 NVL72 'Contact
    sales') is skipped and counted in partial_errors. 'Inference Single
    CPU Price' spans are a different product (footnote 2: inference
    platform customers only) and are never recorded;
  - silent-hole tripwires: every
    'table-model-name' occurrence in the GPU section must be a strict h3
    match (a reshaped h3 would otherwise drop its row without a trace);
    a price-bearing listitem with no parseable model h3 raises; and every
    'item-value' span in a row must be accounted for by the two tier pins
    plus the excluded inference span -- a NEW tier span, or a RENAMED
    label/class (which would otherwise masquerade as the page's own
    not-offered encoding), skips the row loudly instead of recording a
    lie by omission;
  - prices are per INSTANCE per hour; per-GPU normalization divides by the
    'GPU Count' meta cell. GB200/GB300 count cells are the literal
    footnote text '4^1' (footnote 1: each instance is 2 Grace-Blackwell
    superchips x 2 GPUs = 4 GPUs) -- the leading integer is the count and
    the raw cell rides in extra. GB200/GB300 NVL72 are first-class skus
    here, NOT the basket lanes' quarantine class;
  - the collector sends the project User-Agent defined in
    gpu_index.common.http.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "coreweave"

URL = "https://www.coreweave.com/pricing"

_GPU_SECTION_START = "On-demand GPU instances"
_GPU_SECTION_END = "On-demand CPU instances"
_REGION_RE = re.compile(r"REGION: [A-Z ]+")
_EXPECTED_REGIONS = ("REGION: NORTH AMERICA", "REGION: EUROPE")
_ROW_SPLIT = '<div role="listitem" class="table-row-v2 w-dyn-item'
_H3_RE = re.compile(
    r'<h3 data-product="([^"]+)" class="table-model-name">([^<]+)</h3>'
)
# Labeled meta cells (mobile right column): value div immediately followed
# by its label div. The value is captured raw ([^<]*) so the GB200/GB300
# '4^1' footnote cell is seen instead of silently failing an int pattern.
_META_VALUE_TMPL = (
    r'<div class="table-meta-value">([^<]*)</div>\s*<div>{label}</div>'
)
_COUNT_CELL_RE = re.compile(r"^(\d+)(?:\^\d+)?$")
# Self-labeled price spans (the ONLY price pins; unlabeled duplicate divs
# and the inference span are never matched).
_PRICE_SPAN_TMPL = (
    r'<span class="{cls}">{label}\s*'
    r'<span class="item-value">([^<]*)</span>\s*/\s*Hour'
)
_TIER_SPANS = (
    ("on-demand", re.compile(_PRICE_SPAN_TMPL.format(
        cls="instance-price", label="On-Demand Price:"))),
    ("spot", re.compile(_PRICE_SPAN_TMPL.format(
        cls="spot-price", label="Spot Price:"))),
)
# A price is recorded ONLY when the printed value is a plain $ figure --
# any other shape (new currency symbol, ranges, text) is a skipped pin,
# never an assumed-USD guess.
_USD_RE = re.compile(r"^\$([\d,]+(?:\.\d+)?)$")
# Accounting sentinels: every price value on the page rides in an
# item-value span; the inference span (excluded product) is the only
# labeled span besides the two tier pins. The class-name marker for h3
# rows lets a reshaped h3 (which the strict regex would silently drop)
# be counted against strict matches.
_ITEM_VALUE_MARK = 'class="item-value"'
_INFERENCE_MARK = 'class="inference-price"'
_MODEL_NAME_MARK = "table-model-name"

# (meta label, extra key) -- descriptive spec cells recorded per row.
_SPEC_META = (
    ("VRAM", "vram_gb"),
    ("vCPUs", "vcpus"),
    ("System RAM", "system_ram_gb"),
    ("Local Storage (TB)", "local_storage_tb"),
)


def _gpu_section(html: str) -> str:
    for anchor in (_GPU_SECTION_START, _GPU_SECTION_END):
        n = html.count(anchor)
        if n != 1:
            raise RuntimeError(
                f"coreweave: section anchor {anchor!r} appears {n}x (need "
                "exactly 1) -- page reshaped; refusing to guess section "
                "bounds (the CPU tables below reuse the same row markup)"
            )
    return html.split(_GPU_SECTION_START, 1)[1].split(_GPU_SECTION_END, 1)[0]


def _region_tables(section: str) -> Tuple[Tuple[str, str], ...]:
    regions = tuple(m.strip() for m in _REGION_RE.findall(section))
    if regions != _EXPECTED_REGIONS:
        raise RuntimeError(
            f"coreweave: GPU section region headings {list(regions)} != "
            f"{list(_EXPECTED_REGIONS)} -- a new/renamed/reordered region "
            "table would misattribute rows; refusing to extract"
        )
    north_america, europe = (
        section.split(_EXPECTED_REGIONS[0], 1)[1].split(
            _EXPECTED_REGIONS[1], 1
        )
    )
    return (("NORTH AMERICA", north_america), ("EUROPE", europe))


def _meta_cells(seg: str, label: str) -> List[str]:
    pattern = _META_VALUE_TMPL.format(label=re.escape(label))
    return [m.strip() for m in re.findall(pattern, seg)]


def _clean_number(text: str) -> Optional[Any]:
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _parse_segment(
    seg: str,
    region: str,
    rows: List[Dict[str, Any]],
    errors: List[str],
) -> None:
    labels = set(_H3_RE.findall(seg))
    if not labels:
        return  # table preamble before the first row -- no h3, no row
    if len(labels) > 1:
        raise RuntimeError(
            "coreweave: one row segment carries multiple model labels "
            f"({sorted(name for _, name in labels)}) -- the row-split "
            "marker no longer isolates rows, every segment boundary is "
            "suspect; refusing to extract"
        )
    data_product, name = next(iter(labels))
    where = f"{region}: {name}"

    # Span accounting BEFORE anything records: every item-value span in
    # the row must be one of the two tier pins or the excluded inference
    # span. A mismatch means a price span this recipe cannot attribute --
    # a new tier, or a renamed label/class whose absence would otherwise
    # be indistinguishable from the page's own not-offered encoding.
    tier_matches = [
        (tier, [v.strip() for v in span_re.findall(seg)])
        for tier, span_re in _TIER_SPANS
    ]
    n_item_values = seg.count(_ITEM_VALUE_MARK)
    n_accounted = sum(len(vals) for _, vals in tier_matches) + seg.count(
        _INFERENCE_MARK
    )
    if n_item_values != n_accounted:
        errors.append(
            f"{where}: {n_item_values} item-value spans but only "
            f"{n_accounted} accounted for by the labeled tier pins + "
            "excluded inference span -- an unattributable price span "
            "(new tier or renamed label?); row skipped, not guessed"
        )
        return

    count_cells = _meta_cells(seg, "GPU Count")
    if len(count_cells) != 1:
        errors.append(
            f"{where}: {len(count_cells)} labeled 'GPU Count' meta cells "
            "(need exactly 1) -- count unpinnable, row skipped"
        )
        return
    count_match = _COUNT_CELL_RE.match(count_cells[0])
    if not count_match or int(count_match.group(1)) < 1:
        errors.append(
            f"{where}: GPU Count cell {count_cells[0]!r} has no usable "
            "leading integer -- per-GPU basis unknown, row skipped"
        )
        return
    gpu_count = int(count_match.group(1))

    extra_base: Dict[str, Any] = {
        "data_product": data_product,
        # Raw cell as published -- '4^1' on GB200/GB300 rows (footnote 1:
        # 2 superchips x 2 GPUs), plain int elsewhere.
        "gpu_count_cell": count_cells[0],
    }
    memory_gb: Optional[Any] = None
    for label, key in _SPEC_META:
        cells = _meta_cells(seg, label)
        value = _clean_number(cells[0]) if len(cells) == 1 else None
        if value is not None:
            extra_base[key] = value
            if key == "vram_gb":
                memory_gb = value

    priced_any = False
    for tier, spans in tier_matches:
        if len(spans) > 1:
            errors.append(
                f"{where}: {len(spans)} labeled {tier} price spans in one "
                "row segment -- ambiguous, tier skipped"
            )
            continue
        if not spans:
            continue  # span absent entirely (regional variants drop spans)
        value = spans[0]
        if value in ("", "N/A"):
            continue  # the page's own not-offered encoding for a tier
        usd_match = _USD_RE.match(value)
        if not usd_match:
            errors.append(
                f"{where}: {tier} span value {value!r} is not a plain $ "
                "figure -- currency/shape unclear, tier skipped (never "
                "assumed USD)"
            )
            continue
        instance_price = float(usd_match.group(1).replace(",", ""))
        vram_note = f", {memory_gb:g}GB VRAM" if memory_gb is not None else ""
        obs = observation(
            sku_identifier=name,
            price_per_gpu_hr=instance_price / gpu_count,
            raw_value=value,
            raw_unit="usd_per_instance_hr",
            gpu_count_basis=gpu_count,
            tier=tier,
            region=region,
            notes=(
                f"{name} {gpu_count}-GPU instance {value}/hr {tier} "
                f"(labeled span, REGION: {region} table{vram_note}; "
                "inference tier is a separate product, excluded)"
            ),
            extra=dict(extra_base),
        )
        if memory_gb is not None:
            obs["memory_gb_label"] = memory_gb
        rows.append(obs)
        priced_any = True
    if not priced_any:
        reason = (
            "contact-sales row" if "Contact sales" in seg else "no priced tier"
        )
        errors.append(
            f"{where}: unpriced on both recorded tiers ({reason}) -- "
            "skipped, not guessed"
        )


def parse_coreweave(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse: (observations, partial_errors) from the page body."""
    section = _gpu_section(html)
    n_strict = len(_H3_RE.findall(section))
    n_marks = section.count(_MODEL_NAME_MARK)
    if n_strict != n_marks:
        raise RuntimeError(
            f"coreweave: GPU section carries {n_marks} '{_MODEL_NAME_MARK}' "
            f"h3 markers but only {n_strict} match the strict h3 pin -- a "
            "reshaped model h3 would silently drop its row; refusing to "
            "extract"
        )
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for region, table in _region_tables(section):
        row_count = len(rows)
        parts = table.split(_ROW_SPLIT)
        segments = []
        for i, seg in enumerate(parts):
            if _H3_RE.search(seg):
                segments.append(seg)
            elif i > 0 and (_ITEM_VALUE_MARK in seg or "$" in seg):
                # A listitem carrying price markup but no parseable model
                # h3 -- its identity is unpinnable and its price would
                # otherwise vanish without a trace.
                raise RuntimeError(
                    f"coreweave: REGION: {region} table has a price-bearing "
                    "row listitem with no parseable model h3 -- identity "
                    "unpinnable; refusing to record a silent hole"
                )
        if not segments:
            raise RuntimeError(
                f"coreweave: REGION: {region} heading present but its table "
                "has zero row segments -- table reshaped or listings "
                "pulled; refusing to record a silent hole"
            )
        for seg in segments:
            _parse_segment(seg, region, rows, errors)
        if len(rows) == row_count:
            # Segment-level skips are legitimate row by row, but a whole
            # region printing nothing means the pins stopped fitting the
            # table -- fail closed, the per-row reasons ride in the error.
            raise RuntimeError(
                f"coreweave: REGION: {region} table produced zero "
                f"observations across {len(segments)} row segments -- "
                f"{'; '.join(errors) or 'no row-level reasons recorded'}"
            )
    return rows, errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_coreweave(html)
    return result(
        SOURCE_ID,
        method="html-regex",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
