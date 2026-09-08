# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""LIVE acceptance for raw-only public-history reproduction."""

from __future__ import annotations

import datetime
import os

import pytest

from gpu_index.common.bucket import BucketConfig
from gpu_index.published.full import (
    VERDICT_MATCH,
    reproduce_published_history,
)
from gpu_index.published.reader import PublishedRecordReader

pytestmark = pytest.mark.live

DEFAULT_PUBLIC_BASE_URL = "https://data.getcomputable.com"


@pytest.mark.parametrize("sku", ["H100", "H200", "B200", "B300"])
def test_current_as_published_history_reproduces_from_raw_inputs(sku):
    public_url = (
        os.environ.get("GPU_INDEX_PUBLIC_BASE_URL") or DEFAULT_PUBLIC_BASE_URL
    )
    reader = PublishedRecordReader(
        BucketConfig.from_env({"GPU_INDEX_PUBLIC_BASE_URL": public_url})
    )
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    run = reproduce_published_history(reader, sku=sku, target_date=today)

    assert run.checks, f"the public record has no {sku} observations for {today}"
    assert all(check.verdict == VERDICT_MATCH for check in run.checks)
