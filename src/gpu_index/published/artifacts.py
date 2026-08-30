# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""The published record's file layout, envelope shape, and digest rule.

Mirrors the publisher contract -- key layout, envelope shape, digest
rule -- without sharing any code with the publisher. The load-bearing
duality, handled here explicitly:

  - the FILES are pretty-printed: ``JSON.stringify(sortJson(envelope),
    null, 2)`` plus a trailing newline (the publisher's file encoder);
  - the DIGEST ``artifact_sha256`` is sha256 over the COMPACT canonical
    JSON of the payload WITHOUT the digest field itself:
    ``JSON.stringify(sortJson({data, meta, license}))``
    (the publisher's digest rule).

So a verifier must parse the pretty file and re-serialize the parsed
payload compactly — never hash the file bytes. Re-serializing compactly
from Python requires reproducing TWO ECMAScript behaviors exactly:

  - key order: ``sortJson`` sorts object keys with ``localeCompare``.
    For the ASCII snake_case keys the published schema emits this equals
    codepoint order (Python ``sorted``); a non-ASCII key would be a
    divergence zone, so this module REFUSES non-ASCII keys loudly rather
    than guessing.
  - number formatting: ``JSON.stringify`` uses ECMAScript
    Number::toString, which differs from Python's repr/json formatting
    (JS writes ``0.000001`` where Python writes ``1e-06``, ``1e-7`` where
    Python writes ``1e-07``, ``2`` for the integral double Python would
    write as ``2.0``). ``_js_number`` reimplements the ECMAScript
    formatting over Python's shortest-round-trip digits; both languages
    derive the same shortest digit string for a given IEEE-754 double,
    only the surface formatting differs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any, List

# Public keys, exactly as the publisher lays the record out.
_LATEST_KEY = "latest.json"
# Series ranges the publisher writes.
SERIES_RANGES = ("24h", "7d", "30d", "90d")

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SKU_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Envelope shape: every published file carries exactly these keys.
_ENVELOPE_KEYS = frozenset({"artifact_sha256", "data", "meta", "license"})
_META_KEYS = frozenset(
    {
        "schema_version",
        "index_name",
        "generated_at",
        "from_observed_at",
        "to_observed_at",
        "observation_count",
        "disclosure_restatement_count",
    }
)
# License block: the three fields the publisher actually emits. A
# verifier must never need a licensing URL to check a price, so
# ``commercial_licensing`` is tolerated-when-present, not required --
# the publisher has never written it, and a reader that demanded it
# could not read the published record at all.
_LICENSE_KEYS = frozenset({"spdx", "url", "attribution"})
_LICENSE_OPTIONAL_KEYS = frozenset({"commercial_licensing"})
_DATA_KINDS = (
    "gpu_index_latest",
    "gpu_index_observation_day",
    "gpu_index_series",
)


class PublishedRecordError(ValueError):
    """A published artifact failed a shape or contract check."""


class ArtifactDigestError(PublishedRecordError):
    """The recomputed envelope digest does not match the embedded one."""

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            "published artifact digest FAIL: embedded artifact_sha256 "
            f"{expected} vs recomputed {actual} (recomputed over the "
            "compact canonical JSON of the parsed payload)"
        )
        self.expected = expected
        self.actual = actual


# ------------------------------------------------------------- key layout


def latest_key() -> str:
    """``latest.json``: the newest observation per lane."""
    return _LATEST_KEY


def _version_prefix(sku: str, version: int) -> str:
    if not isinstance(sku, str) or not _SKU_RE.fullmatch(sku):
        raise PublishedRecordError(
            f"published SKU must be one clean path segment, got {sku!r}"
        )
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PublishedRecordError(
            f"published version must be a positive integer, got {version!r}"
        )
    return f"{sku}/v{version}"


def day_key(
    date: str, *, sku: str | None = None, version: int | None = None
) -> str:
    """Day key for either the legacy flat or versioned public layout."""
    if not _DATE_RE.match(date):
        raise PublishedRecordError(
            f"day key needs a YYYY-MM-DD date, got {date!r}"
        )
    year, month, day = date.split("-")
    suffix = f"observations/{year}/{month}/{day}.json"
    if sku is None and version is None:
        return suffix
    if sku is None or version is None:
        raise PublishedRecordError(
            "versioned day key requires both SKU and version"
        )
    return f"{_version_prefix(sku, version)}/{suffix}"


def series_key(
    series_range: str, *, sku: str | None = None, version: int | None = None
) -> str:
    """Series key for either the legacy flat or versioned public layout."""
    if series_range not in SERIES_RANGES:
        raise PublishedRecordError(
            f"unknown series range {series_range!r}; "
            f"published ranges are {list(SERIES_RANGES)}"
        )
    suffix = f"series/{series_range}.json"
    if sku is None and version is None:
        return suffix
    if sku is None or version is None:
        raise PublishedRecordError(
            "versioned series key requires both SKU and version"
        )
    return f"{_version_prefix(sku, version)}/{suffix}"


# ------------------------------------------- compact canonical JSON + digest


def _js_number(value: Any) -> str:
    """ECMAScript Number::toString(10) over a parsed JSON number.

    Integers keep their integer spelling (a JSON integer literal has no
    fractional part, exactly what JSON.stringify emitted for it). Floats
    reuse Python's shortest-round-trip digits and reformat them per the
    ECMAScript rules (fixed notation for decimal exponents in (-6, 21],
    ``d.ddde+/-N`` outside), so ``0.000001``/``1e-7``/``1e+21`` come out
    byte-identical to JSON.stringify.
    """
    if isinstance(value, bool):  # bool is an int subclass — never a number
        raise PublishedRecordError("boolean reached the number formatter")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise PublishedRecordError(
            f"non-finite number {value!r} cannot appear in published JSON"
        )
    if value == 0.0:
        return "0"  # JSON.stringify(-0) === "0"
    sign = "-" if value < 0 else ""
    mantissa = repr(abs(value))
    if "e" in mantissa:
        digits_part, _, exponent_part = mantissa.partition("e")
        shift = int(exponent_part)
    else:
        digits_part, shift = mantissa, 0
    int_part, _, frac_part = digits_part.partition(".")
    stripped_int = int_part.lstrip("0")
    if stripped_int:
        exponent = len(stripped_int)
    else:
        exponent = -(len(frac_part) - len(frac_part.lstrip("0")))
    exponent += shift
    digits = (int_part + frac_part).strip("0")
    count = len(digits)
    if count <= exponent <= 21:
        return sign + digits + "0" * (exponent - count)
    if 0 < exponent <= 21:
        return sign + digits[:exponent] + "." + digits[exponent:]
    if -6 < exponent <= 0:
        return sign + "0." + "0" * (-exponent) + digits
    printed = exponent - 1
    suffix = f"e+{printed}" if printed >= 0 else f"e-{-printed}"
    if count == 1:
        return sign + digits + suffix
    return sign + digits[0] + "." + digits[1:] + suffix


def _canonical(value: Any, out: List[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        # json.dumps string escaping (ensure_ascii=False) matches
        # JSON.stringify for valid Unicode: both escape only ", \ and
        # C0 controls (short forms \b\t\n\f\r, lowercase \u00xx rest).
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, (int, float)):
        out.append(_js_number(value))
    elif isinstance(value, list):
        out.append("[")
        for index, entry in enumerate(value):
            if index:
                out.append(",")
            _canonical(entry, out)
        out.append("]")
    elif isinstance(value, dict):
        for key in value:
            if not isinstance(key, str) or not key.isascii():
                # localeCompare and codepoint order agree on ASCII only;
                # refuse the divergence zone rather than guess.
                raise PublishedRecordError(
                    f"non-ASCII object key {key!r}: sortJson orders keys "
                    "with localeCompare, which this mirror reproduces "
                    "only for ASCII keys"
                )
        out.append("{")
        for index, key in enumerate(sorted(value)):
            if index:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _canonical(value[key], out)
        out.append("}")
    else:
        raise PublishedRecordError(
            f"unsupported value type {type(value).__name__} in artifact"
        )


def canonical_compact_bytes(value: Any) -> bytes:
    """``JSON.stringify(sortJson(value))`` as UTF-8 bytes -- the compact
    canonical form the publisher digests."""
    out: List[str] = []
    _canonical(value, out)
    return "".join(out).encode("utf-8")


def payload_digest(payload: Any) -> str:
    """sha256 hex over the compact canonical JSON of the payload."""
    return hashlib.sha256(canonical_compact_bytes(payload)).hexdigest()


# --------------------------------------------------- envelope decode + verify


def _require_keys(
    doc: dict,
    keys: frozenset,
    where: str,
    optional: frozenset = frozenset(),
) -> None:
    """Exact key match, widened by an explicitly named ``optional`` set.

    Default behavior is unchanged: every key in ``keys`` must be present
    and nothing else may be. ``optional`` names keys that MAY appear --
    never keys that may be missing -- so a block carrying one of them
    still validates while an unnamed extra key still fails.
    """
    found = set(doc)
    missing = sorted(keys - found)
    extra = sorted(found - keys - optional)
    if missing or extra:
        raise PublishedRecordError(
            f"{where} keys diverge from the published contract: "
            f"missing {missing}, unexpected {extra}"
        )


def _validate_meta(meta: Any, observations: List[Any]) -> None:
    if not isinstance(meta, dict):
        raise PublishedRecordError("envelope meta must be an object")
    _require_keys(meta, _META_KEYS, "envelope meta")
    if meta["schema_version"] != 1:
        raise PublishedRecordError(
            f"unsupported meta.schema_version {meta['schema_version']!r} "
            "(this reader mirrors schema_version 1)"
        )
    # Meta cross-checks against the observations the envelope carries:
    # the publisher derives meta FROM the observations, so a reader can
    # re-derive and compare every meta field.
    if meta["observation_count"] != len(observations):
        raise PublishedRecordError(
            f"meta.observation_count {meta['observation_count']!r} vs "
            f"{len(observations)} observations in data"
        )
    if observations:
        stamps = sorted(
            observation["observed_at"] for observation in observations
        )
        if meta["from_observed_at"] != stamps[0]:
            raise PublishedRecordError(
                f"meta.from_observed_at {meta['from_observed_at']!r} vs "
                f"earliest observation {stamps[0]!r}"
            )
        if meta["to_observed_at"] != stamps[-1]:
            raise PublishedRecordError(
                f"meta.to_observed_at {meta['to_observed_at']!r} vs "
                f"latest observation {stamps[-1]!r}"
            )
    restatements = sum(
        len(observation.get("restatements", []))
        for observation in observations
        if isinstance(observation, dict)
    )
    if meta["disclosure_restatement_count"] != restatements:
        raise PublishedRecordError(
            "meta.disclosure_restatement_count "
            f"{meta['disclosure_restatement_count']!r} vs {restatements} "
            "restatements across the observations"
        )


def _validate_effective_from(value: Any, where: str) -> None:
    if not isinstance(value, str):
        raise PublishedRecordError(f"{where} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublishedRecordError(
            f"{where} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.utcoffset() is None:
        raise PublishedRecordError(f"{where} must carry a UTC offset")


def _validate_versions(versions: Any) -> None:
    if not isinstance(versions, list) or not versions:
        raise PublishedRecordError("latest data.versions must be a non-empty array")
    seen_skus = set()
    for index, entry in enumerate(versions):
        where = f"latest data.versions[{index}]"
        if not isinstance(entry, dict):
            raise PublishedRecordError(f"{where} must be an object")
        _require_keys(
            entry,
            frozenset(
                {
                    "sku",
                    "current_version",
                    "methodology_id",
                    "effective_from",
                    "succession",
                }
            ),
            where,
        )
        sku = entry["sku"]
        if not isinstance(sku, str) or not _SKU_RE.fullmatch(sku):
            raise PublishedRecordError(f"{where}.sku must be one clean segment")
        if sku in seen_skus:
            raise PublishedRecordError(f"latest data.versions repeats SKU {sku}")
        seen_skus.add(sku)
        current = entry["current_version"]
        if isinstance(current, bool) or not isinstance(current, int) or current < 1:
            raise PublishedRecordError(
                f"{where}.current_version must be a positive integer"
            )
        succession = entry["succession"]
        if not isinstance(succession, list) or not succession:
            raise PublishedRecordError(f"{where}.succession must be non-empty")
        seen_methods = set()
        current_entry = None
        for offset, version_entry in enumerate(succession):
            version_where = f"{where}.succession[{offset}]"
            if not isinstance(version_entry, dict):
                raise PublishedRecordError(f"{version_where} must be an object")
            _require_keys(
                version_entry,
                frozenset({"version", "methodology_id", "effective_from"}),
                version_where,
            )
            version = version_entry["version"]
            if version != offset + 1:
                raise PublishedRecordError(
                    f"{where}.succession versions must start at 1 and be contiguous"
                )
            methodology_id = version_entry["methodology_id"]
            if (
                not isinstance(methodology_id, str)
                or not _SKU_RE.fullmatch(methodology_id)
            ):
                raise PublishedRecordError(
                    f"{version_where}.methodology_id must be one clean identifier"
                )
            if methodology_id in seen_methods:
                raise PublishedRecordError(
                    f"{where}.succession repeats methodology {methodology_id}"
                )
            seen_methods.add(methodology_id)
            _validate_effective_from(
                version_entry["effective_from"], f"{version_where}.effective_from"
            )
            if version == current:
                current_entry = version_entry
        if current_entry is None:
            raise PublishedRecordError(
                f"{where}.current_version is absent from its succession"
            )
        if (
            entry["methodology_id"] != current_entry["methodology_id"]
            or entry["effective_from"] != current_entry["effective_from"]
        ):
            raise PublishedRecordError(
                f"{where} current metadata disagrees with its succession entry"
            )


def _validate_data(data: Any) -> List[Any]:
    if not isinstance(data, dict):
        raise PublishedRecordError("envelope data must be an object")
    kind = data.get("kind")
    if kind == "gpu_index_latest":
        _require_keys(
            data,
            frozenset({"kind", "observations"}),
            "latest data",
            frozenset({"versions"}),
        )
        if "versions" in data:
            _validate_versions(data["versions"])
    elif kind == "gpu_index_observation_day":
        _require_keys(
            data, frozenset({"kind", "date", "observations"}), "day data"
        )
        if not isinstance(data["date"], str) or not _DATE_RE.match(
            data["date"]
        ):
            raise PublishedRecordError(
                f"day data.date must be YYYY-MM-DD, got {data['date']!r}"
            )
    elif kind == "gpu_index_series":
        _require_keys(
            data, frozenset({"kind", "range", "observations"}), "series data"
        )
        if data["range"] not in SERIES_RANGES:
            raise PublishedRecordError(
                f"series data.range {data['range']!r} is not one of "
                f"{list(SERIES_RANGES)}"
            )
    else:
        raise PublishedRecordError(
            f"unknown artifact data.kind {kind!r}; published kinds are "
            f"{list(_DATA_KINDS)}"
        )
    observations = data["observations"]
    if not isinstance(observations, list):
        raise PublishedRecordError("data.observations must be an array")
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict) or not isinstance(
            observation.get("observed_at"), str
        ):
            raise PublishedRecordError(
                f"data.observations[{index}] must be an object with an "
                "observed_at stamp"
            )
    return observations


def decode_and_verify_artifact(raw: bytes) -> dict:
    """Parse a published file and verify its envelope digest.

    The file is pretty-printed; the digest covers the compact canonical
    form of the payload (everything but ``artifact_sha256``), so the
    check parses first and re-serializes compactly — it never hashes the
    file bytes. Raises ``ArtifactDigestError`` on a digest mismatch and
    ``PublishedRecordError`` on any shape violation.
    """
    try:
        document = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishedRecordError(
            f"published artifact must be valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise PublishedRecordError("published artifact must be an object")
    _require_keys(document, _ENVELOPE_KEYS, "envelope")
    embedded = document["artifact_sha256"]
    if not isinstance(embedded, str) or not _SHA256_RE.match(embedded):
        raise PublishedRecordError(
            f"artifact_sha256 must be 64 lowercase hex chars, "
            f"got {embedded!r}"
        )
    license_block = document["license"]
    if not isinstance(license_block, dict):
        raise PublishedRecordError("envelope license must be an object")
    _require_keys(
        license_block,
        _LICENSE_KEYS,
        "envelope license",
        optional=_LICENSE_OPTIONAL_KEYS,
    )
    if license_block["spdx"] != "CC-BY-NC-4.0":
        raise PublishedRecordError(
            f"license.spdx {license_block['spdx']!r} is not the published "
            "CC-BY-NC-4.0 pin"
        )
    observations = _validate_data(document["data"])
    _validate_meta(document["meta"], observations)
    payload = {
        key: document[key] for key in ("data", "meta", "license")
    }
    actual = payload_digest(payload)
    if actual != embedded:
        raise ArtifactDigestError(embedded, actual)
    return document
