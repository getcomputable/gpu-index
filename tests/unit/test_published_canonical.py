# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Canonical serialization + envelope digest for the PUBLISHED record.

The published files are written by the publisher pipeline:
pretty-printed with sortJson + 2-space
indent + trailing newline, while artifact_sha256 is sha256 over the
COMPACT ``JSON.stringify(sortJson(payload))`` of the payload without the
digest field. The fixtures under tests/fixtures/published/ were minted
by a byte-exact Node mirror of those behaviors (the generator's
envelope.mjs), so every digest assertion here is a CROSS-IMPLEMENTATION
check: JS wrote and digested, Python re-derives.

The number-formatting cases pin ECMAScript Number::toString against
Python's formatter; each expected string was produced by
``node -e 'console.log(JSON.stringify(v))'`` (v24), which is what wrote
the record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpu_index.published.artifacts import (
    ArtifactDigestError,
    PublishedRecordError,
    canonical_compact_bytes,
    day_key,
    decode_and_verify_artifact,
    latest_key,
    payload_digest,
    series_key,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "published"

ALL_FIXTURE_KEYS = (
    "latest.json",
    "observations/2026/08/25.json",
    "observations/2026/08/20.json",
    "observations/2026/08/23.json",
    "series/24h.json",
)


def _load(key: str) -> bytes:
    return (FIXTURES / key).read_bytes()


# ------------------------------------------------------------- key layout


def test_key_layout_matches_publisher_paths():
    # The published key layout: latest.json /
    # observations/YYYY/MM/DD.json / series/{24h,7d,30d,90d}.json.
    assert latest_key() == "latest.json"
    assert day_key("2026-08-05") == "observations/2026/08/05.json"
    assert (
        day_key("2026-08-05", sku="H100", version=2)
        == "H100/v2/observations/2026/08/05.json"
    )
    for r in ("24h", "7d", "30d", "90d"):
        assert series_key(r) == f"series/{r}.json"
        assert series_key(r, sku="H100", version=2) == f"H100/v2/series/{r}.json"
    with pytest.raises(PublishedRecordError, match="YYYY-MM-DD"):
        day_key("2026-8-5")
    with pytest.raises(PublishedRecordError, match="series range"):
        series_key("1h")
    with pytest.raises(PublishedRecordError, match="both SKU and version"):
        day_key("2026-08-05", sku="H100")
    with pytest.raises(PublishedRecordError, match="positive integer"):
        series_key("24h", sku="H100", version=0)
    with pytest.raises(PublishedRecordError, match="clean path segment"):
        day_key("2026-08-05", sku="../H100", version=1)


# ------------------------------------------------- ECMAScript number format


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # node v24: JSON.stringify(v) for each v.
        (0.000001, "0.000001"),  # Python repr says 1e-06
        (1e-7, "1e-7"),  # Python repr says 1e-07
        (1.5e-7, "1.5e-7"),
        (9.999999e-7, "9.999999e-7"),
        (1e21, "1e+21"),
        (1.23e21, "1.23e+21"),
        (1e16, "10000000000000000"),  # Python repr says 1e+16
        (1e20, "100000000000000000000"),
        (123.456, "123.456"),
        (2.0, "2"),  # integral double
        (2.5, "2.5"),
        (-0.0, "0"),  # JSON.stringify(-0) === "0"
        (0.1, "0.1"),
        (0.03125, "0.03125"),
        (5e-324, "5e-324"),  # smallest denormal
        (1234567890123456.0, "1234567890123456"),
        (0.000001999, "0.000001999"),
        (2.113437, "2.113437"),
        (7, "7"),
        (-13, "-13"),
        (0, "0"),
    ],
)
def test_number_formatting_matches_json_stringify(value, expected):
    assert canonical_compact_bytes(value).decode() == expected


def test_non_finite_numbers_refuse():
    with pytest.raises(PublishedRecordError, match="non-finite"):
        canonical_compact_bytes(float("inf"))
    with pytest.raises(PublishedRecordError, match="non-finite"):
        canonical_compact_bytes(float("nan"))


# ---------------------------------------------------- canonical compact form


def test_canonical_sorts_keys_recursively_and_stays_compact():
    value = {"b": [{"z": 1, "a": 2}], "a": {"c": None, "b": True}}
    assert (
        canonical_compact_bytes(value).decode()
        == '{"a":{"b":true,"c":null},"b":[{"a":2,"z":1}]}'
    )


def test_canonical_keeps_non_ascii_string_values_raw():
    # JSON.stringify emits non-ASCII characters unescaped; the license
    # attribution the publisher writes contains one.
    value = {"attribution": "Index by Computable — example.com"}
    assert (
        canonical_compact_bytes(value)
        == '{"attribution":"Index by Computable — example.com"}'.encode(
            "utf-8"
        )
    )


def test_canonical_refuses_non_ascii_keys():
    # sortJson orders keys with localeCompare; the mirror reproduces that
    # for ASCII keys only and must refuse the divergence zone loudly.
    with pytest.raises(PublishedRecordError, match="localeCompare"):
        canonical_compact_bytes({"précis": 1})


# ------------------------------------------------- envelope digest (fixtures)


@pytest.mark.parametrize("key", ALL_FIXTURE_KEYS)
def test_js_minted_fixture_digests_verify(key):
    envelope = decode_and_verify_artifact(_load(key))
    assert envelope["meta"]["schema_version"] == 1
    assert envelope["license"]["spdx"] == "CC-BY-NC-4.0"


@pytest.mark.parametrize("key", ALL_FIXTURE_KEYS)
def test_digest_covers_compact_payload_not_file_bytes(key):
    # The pretty-vs-compact duality, explicitly: hashing the pretty FILE
    # bytes never reproduces artifact_sha256; hashing the compact
    # canonical payload does.
    raw = _load(key)
    document = json.loads(raw)
    embedded = document["artifact_sha256"]
    assert hashlib.sha256(raw).hexdigest() != embedded
    payload = {k: document[k] for k in ("data", "meta", "license")}
    assert payload_digest(payload) == embedded


def test_tampered_digest_fails_naming_both_digests():
    raw = _load("observations/2026/08/25.json")
    document = json.loads(raw)
    document["artifact_sha256"] = "0" * 64
    tampered = json.dumps(document).encode()
    with pytest.raises(ArtifactDigestError) as excinfo:
        decode_and_verify_artifact(tampered)
    assert "digest FAIL" in str(excinfo.value)
    assert "0" * 64 in str(excinfo.value)
    assert excinfo.value.actual != excinfo.value.expected


def test_tampered_value_without_redigest_fails_the_digest():
    document = json.loads(_load("observations/2026/08/25.json"))
    document["data"]["observations"][0]["value_usd_gpu_hr"] = 9.99
    with pytest.raises(ArtifactDigestError):
        decode_and_verify_artifact(json.dumps(document).encode())


# ------------------------------------------------------ envelope shape gates


def _valid_document() -> dict:
    return json.loads(_load("observations/2026/08/20.json"))


def _encode(document: dict) -> bytes:
    return json.dumps(document).encode()


def test_extra_envelope_key_refuses():
    document = _valid_document()
    document["extra"] = 1
    with pytest.raises(PublishedRecordError, match="unexpected"):
        decode_and_verify_artifact(_encode(document))


def test_missing_license_key_refuses():
    document = _valid_document()
    del document["license"]["attribution"]
    with pytest.raises(PublishedRecordError, match="missing"):
        decode_and_verify_artifact(_encode(document))


def _redigest(document: dict) -> dict:
    """Re-stamp artifact_sha256 after editing a payload block.

    For the shape gates only. The digest ALGORITHM is cross-checked
    against the JS-minted fixtures above, so reusing it here to mint a
    variant envelope exercises the key-set gate without also re-testing
    the digest -- and leaves a key-set failure as the ONLY way these can
    fail.
    """
    payload = {key: document[key] for key in ("data", "meta", "license")}
    document["artifact_sha256"] = payload_digest(payload)
    return document


def test_license_block_without_commercial_licensing_validates():
    # The shape the publisher actually emits: spdx + url + attribution.
    # Verifying a PRICE must never depend on a licensing URL, so the
    # three-key block is valid on its own. Regression guard: the reader
    # once required a fourth key the publisher has never written, which
    # made every live artifact unreadable while these fixtures -- written
    # to the contract rather than recorded from the publisher -- passed.
    document = _valid_document()
    del document["license"]["commercial_licensing"]
    assert sorted(document["license"]) == ["attribution", "spdx", "url"]
    envelope = decode_and_verify_artifact(_encode(_redigest(document)))
    assert sorted(envelope["license"]) == ["attribution", "spdx", "url"]


def test_license_block_with_commercial_licensing_validates():
    # ...and the four-key block still validates, so the publisher adding
    # the reserved field later is not a breaking change for readers.
    document = _valid_document()
    assert document["license"]["commercial_licensing"]
    envelope = decode_and_verify_artifact(_encode(document))
    assert "commercial_licensing" in envelope["license"]


def test_unknown_license_key_refuses():
    # Tolerating one NAMED optional key is not tolerating any extra key.
    document = _valid_document()
    document["license"]["surprise"] = "x"
    with pytest.raises(PublishedRecordError, match="unexpected"):
        decode_and_verify_artifact(_encode(_redigest(document)))


def test_series_window_basis_timestamp_is_a_named_optional_meta_key():
    document = json.loads(_load("series/24h.json"))
    document["meta"]["window_basis_at"] = "2026-09-01T06:35:00.000Z"

    envelope = decode_and_verify_artifact(_encode(_redigest(document)))

    assert envelope["meta"]["window_basis_at"] == "2026-09-01T06:35:00.000Z"


def test_latest_accepts_and_validates_version_pointer_metadata():
    document = json.loads(_load("latest.json"))
    document["data"]["versions"] = [
        {
            "sku": "H100",
            "current_version": 2,
            "methodology_id": "h100_v2",
            "effective_from": "2026-08-29T00:00:00Z",
            "succession": [
                {
                    "version": 1,
                    "methodology_id": "h100_v1",
                    "effective_from": "2026-08-10T00:00:00Z",
                },
                {
                    "version": 2,
                    "methodology_id": "h100_v2",
                    "effective_from": "2026-08-29T00:00:00Z",
                },
            ],
        }
    ]
    envelope = decode_and_verify_artifact(_encode(_redigest(document)))
    assert envelope["data"]["versions"][0]["current_version"] == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row["succession"][1].update(version=3), "contiguous"),
        (
            lambda row: row["succession"][0].update(version=True),
            "positive integer",
        ),
        (
            lambda row: row["succession"][0].update(version=1.0),
            "positive integer",
        ),
        (lambda row: row.update(current_version=3), "absent"),
        (
            lambda row: row.update(methodology_id="wrong"),
            "disagrees",
        ),
        (
            lambda row: row["succession"][1].update(
                methodology_id=row["succession"][0]["methodology_id"]
            ),
            "repeats methodology",
        ),
    ],
)
def test_latest_rejects_invalid_version_pointer_metadata(mutate, message):
    document = json.loads(_load("latest.json"))
    row = {
        "sku": "H100",
        "current_version": 2,
        "methodology_id": "h100_v2",
        "effective_from": "2026-08-29T00:00:00Z",
        "succession": [
            {
                "version": 1,
                "methodology_id": "h100_v1",
                "effective_from": "2026-08-10T00:00:00Z",
            },
            {
                "version": 2,
                "methodology_id": "h100_v2",
                "effective_from": "2026-08-29T00:00:00Z",
            },
        ],
    }
    mutate(row)
    document["data"]["versions"] = [row]
    with pytest.raises(PublishedRecordError, match=message):
        decode_and_verify_artifact(_encode(_redigest(document)))


def test_wrong_spdx_refuses():
    document = _valid_document()
    document["license"]["spdx"] = "MIT"
    with pytest.raises(PublishedRecordError, match="CC-BY-NC-4.0"):
        decode_and_verify_artifact(_encode(document))


def test_meta_observation_count_cross_check():
    document = _valid_document()
    document["meta"]["observation_count"] += 1
    with pytest.raises(PublishedRecordError, match="observation_count"):
        decode_and_verify_artifact(_encode(document))


def test_meta_restatement_count_cross_check():
    document = _valid_document()
    document["meta"]["disclosure_restatement_count"] += 1
    with pytest.raises(
        PublishedRecordError, match="disclosure_restatement_count"
    ):
        decode_and_verify_artifact(_encode(document))


def test_unknown_kind_refuses():
    document = _valid_document()
    document["data"]["kind"] = "gpu_index_everything"
    with pytest.raises(PublishedRecordError, match="kind"):
        decode_and_verify_artifact(_encode(document))


def test_invalid_utf8_refuses():
    with pytest.raises(PublishedRecordError, match="UTF-8"):
        decode_and_verify_artifact(b'{"a": "\xff"}')
