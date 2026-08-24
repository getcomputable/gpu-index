# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""GPU sku catalog: normalize provider labels to canonical chip ids.

The catalog (config/gpu_sku_catalog.json) is config, not code — adding a
chip is a config edit. Normalization is deliberately conservative and
deterministic:

  - entries match IN CATALOG ORDER, first match wins, so more-specific
    labels (GB200, RTX 6000 ADA) sit above the generic labels they contain
    (B200, RTX 6000);
  - matching is boundary-aware — a token never matches inside a longer
    alphanumeric run ('B200' does not match 'GB200', 'H20' does not match
    'H200', 'A10' does not match 'A100');
  - a label matching nothing normalizes to None, recorded honestly as
    sku_match 'unmapped' — never guessed. The snapshot's
    unmapped_identifiers rollup is the grow-the-catalog worklist.

The snapshot records BOTH the derived sku and the raw sku_identifier, so a
future catalog revision can always re-derive from raw — the derived sku is
a convenience key, the identifier is the record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = _REPO_ROOT / "config" / "gpu_sku_catalog.json"

# Fallback used only if the catalog itself omits default_plausible_usd_gpu_hr.
FALLBACK_PLAUSIBLE_USD_GPU_HR = (0.02, 100.0)

_TOKEN_CHARS_RE = re.compile(r"^[A-Z0-9 ]+$")
# Label prep: uppercase, then treat the separators providers actually use
# ('-', '_', '/') as spaces so 'RTX-4090' and 'RTX_4090' match the same
# token as 'RTX 4090'. Other punctuation is left alone — it acts as a
# boundary, which is what we want.
_SEPARATORS_RE = re.compile(r"[-_/]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Canonical digit-boundary spacing, applied to LABELS AND TOKENS alike:
# every alpha<->digit transition becomes a space, so 'RTX4090', 'RTX 4090'
# and 'RTX4090 Ada'-style partial compactions all normalize to the same
# form and match the same token. Boundary honesty survives because both
# sides are spaced identically: token 'B200' -> 'B 200' still cannot match
# inside label 'GB200' -> 'GB 200' (the lookbehind sees 'G'), and 'H20' ->
# 'H 20' still cannot match inside 'H200' -> 'H 200' (the lookahead sees
# the second '0'). Without this, a label that drops only SOME spaces
# ('RTX5000 Ada') fell through its specific entry to a generic or
# wrong-generation bucket.
_ALPHA_DIGIT_RE = re.compile(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])")


class SkuCatalogError(RuntimeError):
    """config/gpu_sku_catalog.json is missing or malformed."""


def _boundary_pattern(token: str) -> re.Pattern:
    # A token never matches inside a longer alphanumeric run: 'B200' must
    # not match 'GB200'. Spaces inside the token are literal.
    return re.compile(r"(?<![A-Z0-9])" + re.escape(token) + r"(?![A-Z0-9])")


def normalize_label(label: Optional[str]) -> str:
    text = _SEPARATORS_RE.sub(" ", str(label or "").upper())
    text = _ALPHA_DIGIT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_sku_catalog(path: Optional[Path] = None) -> Dict[str, Any]:
    cat_path = Path(path or DEFAULT_CATALOG_PATH)
    if not cat_path.exists():
        raise SkuCatalogError(f"sku catalog missing: {cat_path}")
    try:
        raw = json.loads(cat_path.read_text())
    except json.JSONDecodeError as exc:
        raise SkuCatalogError(f"sku catalog unparseable: {cat_path}: {exc}") from exc

    entries = raw.get("skus")
    if not isinstance(entries, list) or not entries:
        raise SkuCatalogError("sku catalog needs a non-empty 'skus' list")

    default_band = raw.get(
        "default_plausible_usd_gpu_hr", list(FALLBACK_PLAUSIBLE_USD_GPU_HR)
    )
    _validate_band(default_band, "default_plausible_usd_gpu_hr")

    seen = set()
    compiled: List[Dict[str, Any]] = []
    for entry in entries:
        sku = entry.get("sku")
        if not sku or not isinstance(sku, str):
            raise SkuCatalogError(f"catalog entry without sku: {entry!r}")
        if sku in seen:
            raise SkuCatalogError(f"duplicate catalog sku {sku!r}")
        seen.add(sku)
        tokens = entry.get("match_tokens")
        if not isinstance(tokens, list) or not tokens:
            raise SkuCatalogError(f"catalog sku {sku!r} needs match_tokens")
        patterns: List[re.Pattern] = []
        for token in tokens:
            token_norm = normalize_label(token)
            if not token_norm or not _TOKEN_CHARS_RE.match(token_norm):
                raise SkuCatalogError(
                    f"catalog sku {sku!r} token {token!r} must normalize to "
                    "[A-Z0-9 ]+ — punctuation belongs in the label prep, not "
                    "the token"
                )
            # Tokens pass through the SAME normalize_label as labels, so
            # 'RTX 4090' vs 'RTX4090' and every digit-adjacent partial
            # compaction land on one canonical spaced form. Alpha-alpha
            # word joins ('RTXPro6000' -> 'RTXPRO 6000') are NOT re-spaced
            # by the digit rule, so the fully-compacted token also gets a
            # pattern — run through the same normalizer so both sides stay
            # canonical.
            patterns.append(_boundary_pattern(token_norm))
            compact = normalize_label(token_norm.replace(" ", ""))
            if compact != token_norm:
                patterns.append(_boundary_pattern(compact))
        band = entry.get("plausible_usd_gpu_hr")
        if band is not None:
            _validate_band(band, f"catalog sku {sku!r} plausible_usd_gpu_hr")
        compiled.append(
            {
                "sku": sku,
                "vendor": entry.get("vendor"),
                "family": entry.get("family"),
                "patterns": patterns,
                "plausible_usd_gpu_hr": tuple(band) if band else tuple(default_band),
            }
        )
    return {
        "schema_version": raw.get("schema_version"),
        "path": str(cat_path),
        "default_plausible_usd_gpu_hr": tuple(default_band),
        "entries": compiled,
    }


def _validate_band(band: Any, label: str) -> None:
    if (
        not isinstance(band, (list, tuple))
        or len(band) != 2
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in band
        )
        or not (0 < float(band[0]) < float(band[1]))
    ):
        raise SkuCatalogError(f"{label} must be [lo, hi] with 0 < lo < hi: {band!r}")


def match_sku(
    catalog: Dict[str, Any], sku_identifier: Optional[str]
) -> Optional[Dict[str, Any]]:
    """The first catalog entry whose token matches the identifier, or None.

    First-match-wins in catalog order is the ONLY tie-break — the catalog
    file's ordering comment is load-bearing.
    """
    label = normalize_label(sku_identifier)
    if not label:
        return None
    for entry in catalog["entries"]:
        if any(p.search(label) for p in entry["patterns"]):
            return entry
    return None


def plausible_band(
    catalog: Dict[str, Any], sku: Optional[str]
) -> Tuple[float, float]:
    if sku is not None:
        for entry in catalog["entries"]:
            if entry["sku"] == sku:
                return entry["plausible_usd_gpu_hr"]
    return catalog["default_plausible_usd_gpu_hr"]
