# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Reader + verifier for the PUBLISHED record (the public projection).

The publisher (a separate pipeline) projects the collection record
into a small public layout — the version-free ``latest.json`` pointer plus
``<sku>/v<n>/observations/YYYY/MM/DD.json`` day files and
``<sku>/v<n>/series/{24h,7d,30d,90d}.json`` — each file a self-digesting envelope of
``gpu_price_index_observation`` documents. This package mirrors that
contract in Python without sharing any code with the publisher:

  - ``artifacts``: the key layout, the envelope shape, and the
    envelope-digest rule (sha256 over the COMPACT canonical JSON of the
    payload, while the files themselves are pretty-printed);
  - ``verify``: per-observation recompute — rebuild the three sd-votes
    per contributing provider from the published receipts and re-derive
    the index value and stability band with the panel engine's own vote math
    (``gpu_index.index.panel.median_stddev_composite``), matching the
    published numbers exactly;
  - ``reader``: the store-backed reader (local directory download of the
    record, the anonymous public HTTPS front via
    ``GPU_INDEX_PUBLIC_BASE_URL``, or an S3-compatible bucket).

The public ``./reproduce`` consumes this package by default; the private
producer-record replay modes remain behind ``--producer`` for internal
ops.
"""

from gpu_index.published.artifacts import (  # noqa: F401
    ArtifactDigestError,
    PublishedRecordError,
    SERIES_RANGES,
    canonical_compact_bytes,
    day_key,
    decode_and_verify_artifact,
    latest_key,
    payload_digest,
    series_key,
)
from gpu_index.published.reader import PublishedRecordReader  # noqa: F401
from gpu_index.published.verify import (  # noqa: F401
    ObservationCheck,
    recompute_observation,
    select_observations,
)
