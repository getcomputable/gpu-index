# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Lambda -- lambda.ai/pricing, every Instances-pricing row + committed cluster tiers.

Server-rendered HubSpot page, one fetch. Two pinned surfaces (verified live
2026-08-22):

  - the "Instances pricing" island AFTER the '>Instances pricing</h2>'
    anchor: four tabs labeled '8x'/'4x'/'2x'/'1x' (the GPU count of the
    config), each a table of <tr data-plan="..."> rows with data-label
    cells VRAM/GPU, vCPUs, RAM, STORAGE, PRICE/GPU/HR*. Price is already
    per GPU/hr; the SAME plan string prices differently per tab size
    (B200 SXM6: $6.69/$6.79/$6.89/$6.99 at 8x/4x/2x/1x), so the tab label
    is identity, recorded as gpu_count_basis. tier="on-demand".
  - the 1-Click Clusters committed island BEFORE the anchor (it appears
    FIRST in document order -- the basket lane nearly recorded it as
    on-demand once): tabs labeled by plan family, rows carry DURATION /
    GPU COUNT / PRICE/GPU/HR* cells. tier="reserved"; the '1 year+' rows
    are unpriced (em-dash cell) and are skipped + counted, never guessed.

Traps this module pins against (study before editing):

  - every table ALSO exists unicode-escaped inside window.__islands script
    blobs ('\\u003Ctr ... data-plan=\\"...'), including 8 escaped committed
    rows BEFORE the anchor. All row/tab/panel regexes require literal '<'
    and plain '"' so the escaped copies structurally cannot match; the
    per-scope row-count cross-check raises if a layout change ever
    unescapes them (rows would appear outside the tab panels).
  - tab->panel pairing goes through aria-controls == tabpanel id, never
    document order; duplicate ids/controls or a tabs/panels set mismatch
    raises.
  - an instances tab label that is not '<digits>x' raises (it is the only
    honest source of gpu_count_basis); a '<digits>x' tab in the committed
    scope raises (the anchor split landed below the instances island).
  - committed-vs-instances rows are discriminated by their own data-label
    vocabulary (DURATION/GPU COUNT vs VRAM/vCPUs/RAM/STORAGE), not by
    position alone -- a DURATION cell inside the instances scope raises.
  - lookalike labels are real: committed 'NVIDIA H100' vs instance
    'NVIDIA H100 SXM'/'NVIDIA H100 PCIe', committed 'NVIDIA HGX B200' vs
    instance 'NVIDIA B200 SXM6', and two 'NVIDIA A100 SXM' rows in one tab
    differing only by VRAM (80 GB vs 40 GB). sku_identifier stays the
    provider's exact data-plan string; tier/basis/memory label carry the
    rest of the identity.
  - a price cell containing digits that fails the exact '$D.DD' pin raises
    (currency/format change must never be guessed into USD); a cell with
    no digits at all is an unpriced row -- skipped and counted in
    partial_errors.

Class names on the page are hash-suffixed (_pricingRow_3954x_36 etc.) and
deliberately never pinned.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "lambda"

URL = "https://lambda.ai/pricing"

# Deliberately class-free: survives HubSpot class-hash churn, still unique
# on the page (verified: exactly one occurrence, plain or escaped).
_INSTANCES_ANCHOR = ">Instances pricing</h2>"

_PRICE_LABEL = "PRICE/GPU/HR*"
_INSTANCE_CELLS = ("VRAM/GPU", "vCPUs", "RAM", "STORAGE", _PRICE_LABEL)
_CLUSTER_CELLS = ("DURATION", "GPU COUNT", _PRICE_LABEL)

# Literal '<' + plain '"' in every structural regex = the window.__islands
# escaped duplicates ('\\u003C...', 'data-plan=\\"') cannot match.
_TAB_RE = re.compile(
    r'<button role="tab"[^>]*aria-controls="([^"]+)"[^>]*>([^<]*)</button>'
)
_PANEL_RE = re.compile(r'<div role="tabpanel" id="([^"]+)"')
_ROW_RE = re.compile(r'<tr[^>]*\bdata-plan="([^"]+)"[^>]*>(.*?)</tr>', re.DOTALL)
_ROW_START_RE = re.compile(r'<tr[^>]*\bdata-plan="')
_CELL_RE = re.compile(r'data-label="([^"]+)">([^<]*)<')

_PRICE_RE = re.compile(r"^\$(\d+\.\d{2})$")
_TAB_SIZE_RE = re.compile(r"^(\d+)x$")
_GPU_COUNT_RE = re.compile(r"^(\d+)\+?$")
_VRAM_GB_RE = re.compile(r"^(\d+(?:\.\d+)?) GB$")
_HAS_DIGIT_RE = re.compile(r"\d")

_REGION = "unspecified (region-uniform list price)"


def _tabbed_panels(scope: str, scope_name: str) -> List[Tuple[str, str]]:
    """(tab_label, panel_body) pairs, paired via aria-controls == panel id.

    Fail-closed: duplicate controls/ids or a tabs/panels set mismatch means
    the island markup reshaped (or an escaped duplicate unescaped) --
    refuse to scan rather than mis-attribute rows to tabs.
    """
    tabs: Dict[str, str] = {}
    for controls, label in _TAB_RE.findall(scope):
        if controls in tabs:
            raise RuntimeError(
                f"lambda: duplicate tab aria-controls {controls!r} in the "
                f"{scope_name} scope -- island markup duplicated; refusing "
                "to scan"
            )
        tabs[controls] = label.strip()
    marks = list(_PANEL_RE.finditer(scope))
    ids = [m.group(1) for m in marks]
    if len(set(ids)) != len(ids):
        raise RuntimeError(
            f"lambda: duplicate tabpanel id in the {scope_name} scope -- "
            "island markup duplicated; refusing to scan"
        )
    if set(ids) != set(tabs):
        raise RuntimeError(
            f"lambda: tab/panel linkage broke in the {scope_name} scope "
            f"(tabs {sorted(tabs)}, panels {sorted(ids)}) -- page reshaped"
        )
    panels: List[Tuple[str, str]] = []
    for i, mark in enumerate(marks):
        span_end = marks[i + 1].start() if i + 1 < len(marks) else len(scope)
        span = scope[mark.start() : span_end]
        # A panel's rows live inside its single <table>; cut the body at
        # </table> so trailing text (scripts carrying the escaped
        # window.__islands copies, footer) can NEVER donate rows to the
        # last panel -- a plain data-plan row materializing there (an
        # escaped duplicate unescaping) goes unattributed and trips the
        # row-count cross-check instead of silently riding the wrong tab.
        table_end = span.find("</table>")
        if table_end < 0:
            if _ROW_START_RE.search(span):
                raise RuntimeError(
                    f"lambda: tabpanel {mark.group(1)!r} in the "
                    f"{scope_name} scope has data-plan rows but no "
                    "</table> -- island markup reshaped; refusing to scan"
                )
            body = span  # tableless panel: nothing attributable anyway
        else:
            body = span[:table_end]
        panels.append((tabs[mark.group(1)], body))
    return panels


def _rows(panel_body: str) -> List[Tuple[str, Dict[str, str]]]:
    out: List[Tuple[str, Dict[str, str]]] = []
    for plan, row_body in _ROW_RE.findall(panel_body):
        # Cells are matched strictly INSIDE this row's <tr>...</tr> span,
        # so a malformed cell can never bridge into the next row (the
        # basket lane's row-boundary-tempered-gap trap, solved
        # structurally here).
        cells: Dict[str, str] = {}
        for key, value in _CELL_RE.findall(row_body):
            if key in cells:
                # A duplicated column (a was/now promo price pair, a
                # doubled header) must never be resolved by picking one.
                raise RuntimeError(
                    f"lambda: duplicate {key!r} cell in one row (plan "
                    f"{plan.strip()!r}) -- column duplicated; refusing to "
                    "pick between conflicting cells"
                )
            cells[key] = value.strip()
        out.append((plan.strip(), cells))
    return out


def _require_cells(
    cells: Dict[str, str], required: Tuple[str, ...], where: str
) -> None:
    missing = [c for c in required if c not in cells]
    if missing:
        raise RuntimeError(
            f"lambda: {where}: row is missing expected cell(s) {missing} "
            f"(got {sorted(cells)}) -- table columns reshaped; refusing to "
            "extract"
        )


def _parse_price(cell: str, where: str) -> Optional[float]:
    """Exact '$D.DD' -> float; digit-free cell -> None (unpriced row);
    anything digit-bearing that misses the pin raises -- a reshaped or
    non-USD price must never be guessed."""
    match = _PRICE_RE.match(cell)
    if match:
        return float(match.group(1))
    if _HAS_DIGIT_RE.search(cell):
        raise RuntimeError(
            f"lambda: {where}: price cell {cell!r} looks priced but does "
            "not match the exact $D.DD pin -- currency or format changed; "
            "refusing to guess"
        )
    return None


def _check_row_attribution(
    scope: str, scope_name: str, attributed: int
) -> None:
    total = len(_ROW_START_RE.findall(scope))
    if total != attributed:
        raise RuntimeError(
            f"lambda: {total} data-plan rows in the {scope_name} scope but "
            f"{attributed} attributed to tab panels -- rows outside the "
            "tabbed islands (escaped duplicates unescaped?); refusing to "
            "extract"
        )


def _instance_observations(
    scope: str, partial_errors: List[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    attributed = 0
    for tab_label, panel_body in _tabbed_panels(scope, "instances"):
        size_match = _TAB_SIZE_RE.match(tab_label)
        if not size_match:
            raise RuntimeError(
                f"lambda: instances tab label {tab_label!r} is not "
                "'<digits>x' -- cannot derive the config GPU count; page "
                "reshaped"
            )
        gpu_count = int(size_match.group(1))
        for plan, cells in _rows(panel_body):
            attributed += 1
            where = f"instances {tab_label} tab, plan {plan!r}"
            if "DURATION" in cells:
                raise RuntimeError(
                    f"lambda: {where}: committed-tier DURATION cell inside "
                    "the instances scope -- anchor split misplaced; "
                    "refusing to extract"
                )
            _require_cells(cells, _INSTANCE_CELLS, where)
            price = _parse_price(cells[_PRICE_LABEL], where)
            if price is None:
                partial_errors.append(
                    f"instances {tab_label} tab: skipped unpriced row "
                    f"(plan {plan!r}, price cell {cells[_PRICE_LABEL]!r})"
                )
                continue
            obs = observation(
                sku_identifier=plan,
                price_per_gpu_hr=price,
                raw_value=cells[_PRICE_LABEL],
                gpu_count_basis=gpu_count,
                tier="on-demand",
                region=_REGION,
                notes=(
                    f"Instances pricing {tab_label} tab, per-GPU list "
                    "rate pre-tax (plus applicable sales tax/VAT/GST)"
                ),
                extra={
                    "tab": tab_label,
                    "vram_per_gpu": cells["VRAM/GPU"],
                    "vcpus": cells["vCPUs"],
                    "ram": cells["RAM"],
                    "storage": cells["STORAGE"],
                },
            )
            vram = _VRAM_GB_RE.match(cells["VRAM/GPU"])
            if vram:
                gb = float(vram.group(1))
                obs["memory_gb_label"] = int(gb) if gb.is_integer() else gb
            out.append(obs)
    _check_row_attribution(scope, "instances", attributed)
    if not out:
        raise RuntimeError(
            "lambda: Instances-pricing island yielded zero priced rows -- "
            "page reshaped or listings pulled"
        )
    return out


def _cluster_observations(
    scope: str, partial_errors: List[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    attributed = 0
    panels = _tabbed_panels(scope, "committed-clusters")
    for tab_label, panel_body in panels:
        if _TAB_SIZE_RE.match(tab_label):
            raise RuntimeError(
                f"lambda: config-size tab {tab_label!r} in the "
                "committed-clusters scope -- the Instances island moved "
                "above the anchor; refusing to extract"
            )
        for plan, cells in _rows(panel_body):
            attributed += 1
            where = f"committed {tab_label!r} tab, plan {plan!r}"
            _require_cells(cells, _CLUSTER_CELLS, where)
            duration = cells["DURATION"]
            count_label = cells["GPU COUNT"]
            price = _parse_price(cells[_PRICE_LABEL], where)
            if price is None:
                partial_errors.append(
                    f"committed {tab_label!r} tab: skipped unpriced row "
                    f"(DURATION {duration!r}, GPU COUNT {count_label!r}, "
                    f"price cell {cells[_PRICE_LABEL]!r})"
                )
                continue
            count_match = _GPU_COUNT_RE.match(count_label)
            if not count_match:
                partial_errors.append(
                    f"committed {tab_label!r} tab: skipped priced row with "
                    f"unpinnable GPU COUNT {count_label!r} (DURATION "
                    f"{duration!r}) -- no honest gpu_count_basis"
                )
                continue
            out.append(
                observation(
                    sku_identifier=plan,
                    price_per_gpu_hr=price,
                    raw_value=cells[_PRICE_LABEL],
                    gpu_count_basis=int(count_match.group(1)),
                    tier="reserved",
                    region=_REGION,
                    notes=(
                        "1-Click Clusters committed tier, per-GPU rate "
                        f"pre-tax; duration {duration}; GPU count "
                        f"{count_label}"
                    ),
                    extra={
                        "tab": tab_label,
                        "duration": duration,
                        "gpu_count_label": count_label,
                    },
                )
            )
    _check_row_attribution(scope, "committed-clusters", attributed)
    if not panels:
        # Committed tables vanishing is a visible note, not a dead source --
        # the Instances table is this collector's primary surface.
        partial_errors.append(
            "committed 1-Click Clusters tables not found before the "
            "Instances-pricing anchor -- committed tiers unrecorded this "
            "capture"
        )
    return out


def parse_lambda_pricing(
    html: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    if _INSTANCES_ANCHOR not in html:
        raise RuntimeError(
            "lambda: 'Instances pricing' heading missing -- page reshaped; "
            "refusing to scan (the committed 1-Click Clusters tables sit "
            "first in document order and would be misread as on-demand)"
        )
    cluster_scope, instances_scope = html.split(_INSTANCES_ANCHOR, 1)
    partial_errors: List[str] = []
    observations = _instance_observations(instances_scope, partial_errors)
    observations.extend(_cluster_observations(cluster_scope, partial_errors))
    return observations, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_lambda_pricing(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
