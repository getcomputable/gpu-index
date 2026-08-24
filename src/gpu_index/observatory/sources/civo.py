# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Civo -- civo.com/pricing, NVIDIA GPU tables, on-demand + commitment terms.

One fetch of the server-rendered pricing page. The section fence is
LOAD-BEARING: 23 tables on the page share class="pricing-table" (Kubernetes
nodes, compute instances, databases, object store), so parsing without the
<section class="pricing-product" id="nvidia-gpus"> fence would ingest CPU
nodes as GPUs -- and id="nvidia-gpus" appears TWICE (the section and a bare
<div> just inside it), so the fence anchors on the full section tag, never
the bare id. Inside the fence (verified live 2026-08-22): one <table> per
chip, its <caption>NVIDIA {label} GPU pricing</caption> carrying the
provider's structured part label (sku_identifier = "NVIDIA " + label;
currently L40S 48GB / A100 40GB / A100 80GB / H100 SXM / H200 SXM / B200).

Row identity pin: every data row states its own GPU count and model in
<div class="product-detail">N x NVIDIA {model}</div> (e.g. "8 x NVIDIA H200
- 141GB HBM3e"). Prices are per INSTANCE hour, so per-GPU = price / that
pinned count (verified exact live: L40S 8x $10.32 = 8 x $1.29). The model
is cross-checked against the caption -- chip token must match and any
caption memory size ("40GB" vs "80GB") must appear in the model -- so an
A100 lookalike row can never land under the wrong table label.

Price cells are self-labeling by td class: "pricing-data on-demand-pricing"
(tier on-demand) and "pricing-data commitment-pricing" (tier reserved, one
<div id="{6,12,24,36}_months"> block per term, commitment_months in extra
-- the together/lambda house convention for term-committed hourly rates).
Cell honesty, all fail-closed:

  - the hourly print is pinned to data-option="hourly" + the literal
    <span>per hour</span> suffix: id="price-value-hourly"/"-monthly" repeat
    dozens of times document-wide and each cell also carries a hidden
    per-month price, so only the data-option + per-hour-span pair
    discriminates; a cell whose hourly print misses the exact
    N/A-or-$D.DD pin raises (a currency/format change is never guessed
    into USD -- the page is dollar-only, zero currency-switcher markup);
  - "N/A" means genuinely unlisted -- skipped silently, never a $0 print.
    B200 is commitment-only BY DESIGN (on-demand and 6-month are N/A; the
    headline per-GPU $3.79 is the 36-MONTH term) -- dropping the term
    labels would print a fake on-demand B200 ~40% under market;
  - commitment terms default hidden by CSS (the term <select> shows 36
    months) but all four are in the bytes -- parsed by div id, visibility
    ignored; a priced print BEFORE the first term block would be
    tier-unattributable and raises;
  - a priced row that lost its product-detail pin has no honest GPU count
    -- skipped and counted in partial_errors, never guessed;
  - the row scanner tolerates <tr> attributes, and a per-table census
    requires every priced cell to land inside a scanned row -- a reshaped
    row can never make its prices vanish silently.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "civo"

URL = "https://www.civo.com/pricing"

# The fence: full section tag, never the bare id (duplicated on an inner div).
_SECTION_RE = re.compile(
    r'<section class="pricing-product" id="nvidia-gpus">(.*?)</section>',
    re.DOTALL,
)
_TABLE_RE = re.compile(
    r"<caption>NVIDIA (.+?) GPU pricing</caption>(.*?)</table>", re.DOTALL
)
# Attribute-tolerant: a <tr class="..."> data row must still be scanned
# (a bare-<tr>-only scanner would drop its prices without a trace).
_ROW_RE = re.compile(r"<tr(?:\s[^>]*)?>(.*?)</tr>", re.DOTALL)
# The row identity pin: stated GPU count + model, provider's own words.
_DETAIL_RE = re.compile(
    r'<div class="product-detail">(\d+) x NVIDIA ([^<]+?)</div>'
)
_SIZE_RE = re.compile(
    r'<td data-title="Size">\s*(.*?)\s*<div class="product-detail">',
    re.DOTALL,
)
_ONDEMAND_CELL_RE = re.compile(
    r'<td[^>]*class="pricing-data on-demand-pricing"[^>]*>(.*?)</td>',
    re.DOTALL,
)
_COMMIT_CELL_RE = re.compile(
    r'<td[^>]*class="pricing-data commitment-pricing"[^>]*>(.*?)</td>',
    re.DOTALL,
)
_TERM_SPLIT_RE = re.compile(r'<div id="(\d+)_months"')
# data-option + the per-hour span TOGETHER discriminate: ids repeat
# document-wide and every cell also holds a hidden per-month price.
_HOURLY_RE = re.compile(
    r'data-option="hourly"[^>]*>\s*(N/A|\$[\d,]+\.\d{2})<span>per hour</span>'
)
_MODEL_MEM_RE = re.compile(r"(\d+)\s*GB")
_CAPTION_MEM_RE = re.compile(r"\b(\d+GB)\b")
_WS_RE = re.compile(r"\s+")

_REGION = "unspecified"


def _gpu_section(html: str) -> str:
    sections = _SECTION_RE.findall(html)
    if len(sections) != 1:
        raise RuntimeError(
            f"civo: found {len(sections)} 'pricing-product nvidia-gpus' "
            "sections (need exactly 1) -- page reshaped; refusing to scan "
            "(23 lookalike pricing-table tables sit outside the fence)"
        )
    return sections[0]


def _hourly_print(cell_html: str, where: str) -> str:
    """The cell's single hourly print: 'N/A' or '$D.DD', fail-closed."""
    prints = _HOURLY_RE.findall(cell_html)
    if len(prints) != 1:
        raise RuntimeError(
            f"civo: {where}: found {len(prints)} hourly prints (need "
            "exactly 1) -- cell markup or currency format changed; "
            "refusing to guess"
        )
    return prints[0]


def _money(raw: str) -> float:
    return float(raw.replace("$", "").replace(",", ""))


def parse_civo_pricing(
    html: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    section = _gpu_section(html)
    tables = _TABLE_RE.findall(section)
    if not tables:
        raise RuntimeError(
            "civo: zero 'NVIDIA ... GPU pricing' captioned tables inside "
            "the nvidia-gpus section -- page reshaped or listings pulled"
        )
    rows: List[Dict[str, Any]] = []
    partial_errors: List[str] = []
    for caption, table_html in tables:
        caption = _WS_RE.sub(" ", caption).strip()
        identifier = f"NVIDIA {caption}"
        if ">On-demand<" not in table_html or ">Commitment<" not in table_html:
            raise RuntimeError(
                f"civo: table {caption!r} lost its On-demand/Commitment "
                "header -- column semantics changed; refusing to attribute "
                "prices to tiers"
            )
        chip_token = caption.split()[0]
        caption_mems = _CAPTION_MEM_RE.findall(caption)
        row_htmls = _ROW_RE.findall(table_html)
        # Priced-cell census: every price cell must sit inside a scanned
        # row. A reshaped row (or broken row markup) would otherwise drop
        # its prices with no raise and no partial_error -- silent
        # under-extraction, the one skip path the pins above can't see.
        for marker in ("on-demand-pricing", "commitment-pricing"):
            outside = table_html.count(marker) - sum(
                r.count(marker) for r in row_htmls
            )
            if outside:
                raise RuntimeError(
                    f"civo: table {caption!r}: {outside} {marker} cell(s) "
                    "outside any scanned <tr> row -- row markup reshaped; "
                    "refusing to drop prices silently"
                )
        for row_html in row_htmls:
            details = _DETAIL_RE.findall(row_html)
            if not details:
                if "pricing-data" in row_html:
                    # A priced row without the 'N x NVIDIA ...' pin has no
                    # honest GPU count -- skipped, never guessed.
                    partial_errors.append(
                        f"table {caption!r}: priced row without a "
                        "product-detail identity pin -- skipped"
                    )
                continue  # header/filler row
            if len(details) != 1:
                raise RuntimeError(
                    f"civo: table {caption!r}: row with {len(details)} "
                    "product-detail pins -- GPU count attribution ambiguous; "
                    "refusing to extract"
                )
            count_s, model = details[0]
            count = int(count_s)
            model = _WS_RE.sub(" ", model).strip()
            if count < 1:
                raise RuntimeError(
                    f"civo: table {caption!r}: row states GPU count "
                    f"{count} -- cannot normalize per GPU"
                )
            # Lookalike guard: A100 40GB vs A100 80GB (and any future twin)
            # must never land under the wrong caption.
            if model.split()[0] != chip_token or any(
                mem not in model for mem in caption_mems
            ):
                raise RuntimeError(
                    f"civo: table {caption!r}: row model {model!r} does not "
                    "match the table caption -- row/table binding broke; "
                    "refusing to extract"
                )
            size_match = _SIZE_RE.search(row_html)
            size = (
                _WS_RE.sub(" ", size_match.group(1)).strip()
                if size_match
                else ""
            )
            where = f"table {caption!r} row {size or model!r}"
            detail_text = f"{count} x NVIDIA {model}"
            mem_match = _MODEL_MEM_RE.search(model)
            extra_base = {"size_name": size, "instance_detail": detail_text}

            od_cells = _ONDEMAND_CELL_RE.findall(row_html)
            commit_cells = _COMMIT_CELL_RE.findall(row_html)
            if len(od_cells) != 1 or len(commit_cells) != 1:
                raise RuntimeError(
                    f"civo: {where}: found {len(od_cells)} on-demand and "
                    f"{len(commit_cells)} commitment price cells (need "
                    "exactly 1 of each) -- tier attribution broke; "
                    "refusing to extract"
                )

            def _record(
                raw: str, tier: str, note: str, extra: Dict[str, Any]
            ) -> None:
                obs = observation(
                    sku_identifier=identifier,
                    price_per_gpu_hr=_money(raw) / count,
                    raw_value=raw,
                    raw_unit="usd_per_instance_hr",
                    gpu_count_basis=count,
                    tier=tier,
                    region=_REGION,
                    notes=note,
                    extra=extra,
                )
                if mem_match:
                    obs["memory_gb_label"] = int(mem_match.group(1))
                rows.append(obs)

            raw = _hourly_print(od_cells[0], f"{where} on-demand cell")
            if raw != "N/A":  # N/A = genuinely unlisted, never a $0 print
                _record(
                    raw,
                    "on-demand",
                    f"{size} instance ({detail_text}), on-demand hourly "
                    "rate per instance",
                    dict(extra_base, column="On-demand"),
                )

            parts = _TERM_SPLIT_RE.split(commit_cells[0])
            if _HOURLY_RE.search(parts[0]):
                raise RuntimeError(
                    f"civo: {where}: hourly print before the first "
                    "commitment term block -- term attribution broke; "
                    "refusing to extract"
                )
            if len(parts) < 3:
                raise RuntimeError(
                    f"civo: {where}: commitment cell holds zero "
                    "N_months term blocks -- cell markup reshaped; "
                    "refusing to extract"
                )
            for i in range(1, len(parts), 2):
                months = int(parts[i])
                raw = _hourly_print(
                    parts[i + 1], f"{where} {months}-month commitment block"
                )
                if raw == "N/A":
                    continue
                _record(
                    raw,
                    "reserved",
                    f"{size} instance ({detail_text}), {months}-month "
                    "commitment hourly rate per instance",
                    dict(
                        extra_base,
                        column="Commitment",
                        commitment_months=months,
                    ),
                )
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_civo_pricing(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
