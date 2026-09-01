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
    read_full_history,
    reproduce_full_history,
)
from gpu_index.published.reader import PublishedRecordReader

pytestmark = pytest.mark.live

DEFAULT_PUBLIC_BASE_URL = "https://data.getcomputable.com"


def test_current_h100_reproduces_from_raw_public_history_only():
    public_url = (
        os.environ.get("GPU_INDEX_PUBLIC_BASE_URL") or DEFAULT_PUBLIC_BASE_URL
    )
    reader = PublishedRecordReader(
        BucketConfig.from_env({"GPU_INDEX_PUBLIC_BASE_URL": public_url})
    )
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    history = read_full_history(reader, sku="H100", target_date=today)
    run = reproduce_full_history(history, target_date=today)

    assert run.checks, f"the public record has no H100 observations for {today}"
    assert all(check.verdict == VERDICT_MATCH for check in run.checks)
