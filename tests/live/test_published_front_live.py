# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""LIVE seam: the published record as the public actually reads it.

Every other test in this suite reads a fixture or a local record copy,
and that is precisely the blind spot these tests exist to close. The
fixtures under tests/fixtures/published/ were written to the CONTRACT
rather than recorded from the PUBLISHER, so a reader whose contract
disagreed with the publisher passed all of them while `./reproduce`
could not read a single live file.

So these tests refuse to use a fixture, and refuse to record one: a
self-recorded golden would be circular -- it would freeze whatever this
reader currently believes and re-assert it forever, which is the bug,
not the check. They reach the real public HTTPS front over the network
and decode exactly what it serves, so no local copy can satisfy the read.
The full re-derivation of every SKU from that record lives in the
sibling live modules; this one pins the front's envelope contract.

Discovery (which day to verify) is done with a bare HTTP GET rather than
through PublishedRecordReader, because discovery must not run the very
envelope contract under test -- otherwise a contract bug would surface
as "nothing is published" instead of as a contract failure.

Marked `live` and DESELECTED from the default `pytest` run (pyproject
addopts carries `-m "not live"`): a unit suite that fails whenever a host
is unreachable is a suite contributors learn to ignore, and it would
break offline work and forks. Run them explicitly:

    pytest -m live

CI runs them daily and on demand (the `live-record` job in ci.yml), so
drift between this reader and the publisher surfaces within a day rather
than at some stranger's first clean clone.
"""

from __future__ import annotations

import datetime
import json
import os

import httpx
import pytest

from gpu_index.published.artifacts import (
    day_key,
    decode_and_verify_artifact,
)

pytestmark = pytest.mark.live


# The official record host, overridable to point the run at another
# front. `./reproduce` applies the same default when nothing is set.
DEFAULT_PUBLIC_BASE_URL = "https://data.getcomputable.com"
PUBLIC_BASE_URL = (
    os.environ.get("GPU_INDEX_PUBLIC_BASE_URL") or DEFAULT_PUBLIC_BASE_URL
)

# How far back to look for the newest published day. Yesterday is the
# newest COMPLETE UTC day; a few days of slack absorbs a late or skipped
# publisher run without turning this into a flaky alarm, while still
# failing loudly if the record has actually gone quiet.
LOOKBACK_DAYS = 7

PUBLIC_SKUS = ("h100", "h200", "b300", "b200")



def _raw_day(date: str, sku: str, version: int | None):
    """Raw bytes of a published day file, or None when absent (404).

    Deliberately not PublishedRecordReader: no decoding, no contract.
    """
    key = day_key(date) if version is None else day_key(date, sku=sku, version=version)
    url = f"{PUBLIC_BASE_URL.rstrip('/')}/{key}"
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def _current_versions() -> dict[str, int] | None:
    """Bare-JSON discovery of current versions; decoding is tested later."""
    response = httpx.get(
        f"{PUBLIC_BASE_URL.rstrip('/')}/latest.json",
        timeout=30.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    document = json.loads(response.content)
    versions = document["data"].get("versions")
    if versions is None:
        return None
    return {entry["sku"]: entry["current_version"] for entry in versions}


@pytest.fixture(scope="module")
def live_days():
    """Newest published UTC day per SKU, as ``{sku: (date, raw)}``."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    versions = _current_versions()
    discovered = {}
    for cli_sku in PUBLIC_SKUS:
        sku = cli_sku.upper()
        tried = []
        for back in range(1, LOOKBACK_DAYS + 1):
            date = (today - datetime.timedelta(days=back)).isoformat()
            tried.append(date)
            raw = _raw_day(date, sku, None if versions is None else versions[sku])
            if raw is not None:
                discovered[cli_sku] = (date, raw)
                break
        else:
            pytest.fail(
                f"the public front {PUBLIC_BASE_URL} served no {sku} observation day "
                f"file for any of {tried}: the record front is down, or hourly "
                "publication has stopped"
            )
    return discovered


@pytest.mark.parametrize("sku", PUBLIC_SKUS)
def test_live_day_file_satisfies_the_envelope_contract(sku, live_days):
    # The contract seam, tightly. Any divergence between what this
    # reader REQUIRES and what the publisher WRITES -- an envelope key,
    # a meta key, a license key -- fails here, naming the field.
    date, raw = live_days[sku]
    envelope = decode_and_verify_artifact(raw)
    assert envelope["data"]["kind"] == "gpu_index_observation_day"
    assert envelope["data"]["date"] == date
    assert envelope["license"]["spdx"] == "CC-BY-NC-4.0"
    assert envelope["data"]["observations"], "day file carries no rows"
