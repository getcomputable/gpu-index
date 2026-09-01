# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Dynamic predictive source weighting for the index baskets.

Implements the dynamic weighting methodology (METHODOLOGY.md section 8) for
annex_a_v0_2_calc_v5 / annex_a2_v0_3_calc_v4: each
constituent's index weight derives from how well its recent EXCESS price
movements predict subsequent movement in the rest of the basket —
leave-one-out excess log-return features, exponentially weighted ridge
regressions per forward horizon, in-sample R^2 scoring (R-insample), softmax
allocation with a per-source floor, per-source risk caps, and an iteratively
redistributed global cap.

Everything here is a pure function of its inputs — zero I/O, zero clock
reads — exactly like basket.composite, so the weight series is a
deterministic replay of (published artifacts, config). Design rules (the
R-* labels are pinned rule names):

  - **The estimator runs on the capture SLOT grid** (R-slots):
    timestamps are integer hours (day_ordinal * 24 +
    slot_hour_utc), the 6h capture cadence yields four sample anchors per
    day, and the horizon sets are expressed in HOURS ({6, 24, 48} both
    directions; 1h joins when capture walks to hourly snapshots). Slot
    prints are pinned in each day's artifact (weight_calc.slot_prints —
    the just-closed prior day's slots), so replay stays a pure function of
    published artifacts.
  - **Weights for day t use data through day t-1's LAST slot only.**
    Every sample endpoint — features and target alike — satisfies
    endpoint <= (t-1)'s last capture slot (R-cutoff on the slot grid), so
    nothing captured on day t can move day t's own weights: the scores are
    announceable before the day's first snapshot exists (anti-manipulation
    — a party listing on a marketplace at 15:59Z cannot move its own
    weight that day). Only the day's ELIGIBLE SET (who printed a trusted
    value today) is a same-day input.
  - **Source returns are NATIVE; the basket is USD.** r_i runs on the
    filter's recorded-currency price (the recorded-currency posture: FX must never
    masquerade as a source price move), while the leave-one-out basket,
    r_{-i}, and every target stay in the index currency. A return whose
    endpoints differ in recorded currency is undefined — dropped, never
    FX-spliced.
  - **The series holds every real TRUSTED print** — accepted AND held-out,
    the same membership rule the filter window uses; untrusted-currency
    prints never enter; manual exclusions are gaps. No carry-forward: a
    return needs a real print at both exact-date endpoints. Every log
    return entering a feature or LOO leg is winsorized at
    ``max_abs_log_return`` (R-winsor) so a single absurd trusted print the
    sigma fence held out of the INDEX cannot poison the weight lane at
    unbounded magnitude.
  - **Leave-one-out returns hold weights AND composition fixed** at the
    sample date tau: the day-tau pinned (rounded) weight vector at both
    endpoints, over sources printing at both endpoints. Weight drift and
    composition churn must never masquerade as basket movement. (A
    documented divergence from the schema-literal contemporaneous
    P_{-i,t}.) The renormalizing denominators cancel in the ratio.
  - **Undefined is not zero.** q_{i,h} exists only when >= min_train
    realized samples exist AND the weighted target variance clears the
    floor (R-insample: q scores the single EW
    ridge fit IN-SAMPLE — no out-of-sample evaluation points are
    required); Q_i exists only when every horizon's q does. The
    series starts in "fallback" mode (the config opening weights
    RESTRICTED to the day's eligible sources, deliberately unnormalized —
    every consumer is scale-invariant and renormalizes over passers, so
    fallback-mode index math is byte-identical to the prior fixed-weight
    series) and switches to "dynamic" on the first day the SWITCH QUORUM
    holds (R-quorum-v2): at least
    switch_min_eligible sources are eligible today AND have a defined Q —
    enough scored providers to carry an index; a sparse or late source no
    longer holds the switch and simply scores 0 once it is in. The switch
    is permanent for the series.
  - **The rounded-6dp vector is THE weight.** Allocation runs in full
    precision, rounds once, and that rounded vector is pinned in the
    artifact, consumed by the composite, and carried in state for future
    LOO reconstruction. Rounding leaves sum(w) approximately 1; consumers
    renormalize — never nudge a source to fix the residue.

The regression is solved by pure-Python fixed-order Gaussian elimination on
the normal equations (max 3x3 here). Deliberately NOT numpy/BLAS: the
artifact pins each day's rounded weight vector, and bit-determinism across
platforms and BLAS builds is worth more to an auditable series than
vectorized speed at n <= 90 samples.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Pinned public re-export (tests/unit/test_public_api.py): the
# minute re-base stopped using stamp_to_hour_iso here, but downstream
# consumers import it by this name, so the surface stays.
from gpu_index.index.panel_schedule import (  # noqa: F401
    stamp_to_hour_iso,
)


SCHEME_PREDICTIVE_V1 = "predictive_v1"
VALID_WEIGHT_SCHEMES = (SCHEME_PREDICTIVE_V1,)

MODE_FALLBACK = "fallback"
MODE_DYNAMIC = "dynamic"

# Float-dust guards, all orders of magnitude below the 6dp artifact
# rounding. The variance floor is in log-return^2 units: 6dp price rounding
# at ~$7.5/GPU-hr injects return quanta ~1.3e-7, variance ~1e-14 — a target
# series whose weighted variance sits below 1e-12 is rounding dust, not
# movement, and must yield an UNDEFINED score rather than a 0/0 R^2.
# _PIVOT_EPS is the solver's near-singularity gate — semantically distinct
# from _CAP_EPS despite the shared magnitude: it bounds a Gaussian pivot,
# not a weight comparison.
_CAP_EPS = 1e-12
_PIVOT_EPS = 1e-12
_STD_FLOOR = 1e-9
DEFAULT_TARGET_VARIANCE_FLOOR = 1e-12


def new_weight_state() -> Dict[str, Any]:
    """The ONE constructor for the calc_v5 weight-series state — trusted
    slot prints (hour-stamped, R-slots), each day's pinned rounded vector,
    and the fallback -> dynamic mode latch. The CLI and every test build
    state through here so a future state key cannot silently miss a call
    site.

    Also the constructor for the OBSERVATION-mode (panel) state -- same
    shape, one difference in key semantics: day mode keys `vectors` by
    DAY ORDINAL, observation mode keys `vectors` by absolute MINUTE
    stamp (day_ordinal * 1440 + minute_of_day; 15-min cadence re-base
    2026-08-27 -- hour-grid lanes occupy the :00 minutes of the same
    lattice). The two modes are NEVER mixed in one lane: a lane's state
    is advanced exclusively by advance_weight_state (day mode) or
    exclusively by advance_panel_weight_state (observation mode) for its
    whole life.

    ``_prune_threshold`` is observation-mode PRIVATE bookkeeping (the
    stamp the state was last pruned to; None = never pruned) so
    advance_panel_weight_state can defer pruning until the threshold has
    advanced >= one day of wall time. Day mode never reads or writes it;
    every reader tolerates its absence (states built by hand in tests)
    via .get."""
    return {
        "prices": {},
        "vectors": {},
        "mode": MODE_FALLBACK,
        "_prune_threshold": None,
    }


def series_print(usd: Any, observation: Tuple[float, str]) -> Dict[str, Any]:
    """The ONE constructor for a weight-series price entry: a slot's
    resolved USD print plus the trusted filter_observation value/currency
    (native terms, the recorded-currency posture). resolve_slot_prints builds every
    entry through here and the artifact pins the result verbatim
    (weight_calc.slot_prints), so the series shape is structurally — not
    just test-enforced — identical between the live path and replay."""
    native_price, native_currency = observation
    return {"usd": usd, "native": native_price, "currency": native_currency}


def _ordinal(day: str) -> int:
    return date.fromisoformat(day).toordinal()


# ------------------------------------------------------------- linear algebra


def solve_linear(
    matrix: List[List[float]], rhs: List[float]
) -> Optional[List[float]]:
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Pure Python, fixed operation order (see module docstring). Returns None
    on a (near-)singular system — the caller skips that evaluation point
    rather than publishing garbage. Sizes here are (len(D) + 1) <= 3.
    """
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < _PIVOT_EPS:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))
        x[row] = acc / a[row][row]
    return x


def fit_ridge(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    sample_weights: Sequence[float],
    *,
    ridge_lambda: float,
) -> Optional[Dict[str, Any]]:
    """Exponentially weighted ridge with an UNPENALIZED intercept over
    per-feature weighted z-scores.

    Convention pins (the lambda value is meaningless without them):
    observation weights are normalized to sum 1 over the training window,
    and each feature is z-scored by its weighted mean/std on that window —
    so ridge_lambda is a pure shrinkage dial that means the same thing for
    a 1d return feature and a 3d one. A feature whose weighted std falls
    below the floor carries no information in this window and is
    neutralized (its z-scores become 0, ridge sends its coefficient to 0)
    — the routine case here, where step-function list prices produce
    identically-zero excess returns for weeks. Predictions must
    standardize by the TRAINING window's moments, never their own.
    Returns None when the weight mass is zero/non-finite or the system is
    singular.
    """
    n = len(rows)
    if n == 0 or ridge_lambda < 0:
        return None
    k = len(rows[0])
    raw_total = float(sum(sample_weights))
    if not (raw_total > 0 and math.isfinite(raw_total)):
        return None
    norm_w = [w / raw_total for w in sample_weights]
    means = [
        sum(w * row[j] for row, w in zip(rows, norm_w)) for j in range(k)
    ]
    stds = [
        math.sqrt(
            sum(w * (row[j] - means[j]) ** 2 for row, w in zip(rows, norm_w))
        )
        for j in range(k)
    ]
    z_rows = [
        [
            (row[j] - means[j]) / stds[j] if stds[j] >= _STD_FLOOR else 0.0
            for j in range(k)
        ]
        for row in rows
    ]
    # Normal equations over [1, z] with the ridge penalty on the z block
    # only (the intercept must absorb the weighted mean of y unshrunk).
    dim = k + 1
    ata = [[0.0] * dim for _ in range(dim)]
    atb = [0.0] * dim
    for z, y, w in zip(z_rows, targets, norm_w):
        aug = [1.0] + list(z)
        for r in range(dim):
            wr = w * aug[r]
            atb[r] += wr * y
            for c in range(r, dim):
                ata[r][c] += wr * aug[c]
    for r in range(dim):
        for c in range(r):
            ata[r][c] = ata[c][r]
    for j in range(1, dim):
        ata[j][j] += ridge_lambda
    theta = solve_linear(ata, atb)
    if theta is None:
        return None
    return {
        "intercept": theta[0],
        "betas": theta[1:],
        "means": means,
        "stds": stds,
    }


def predict_ridge(model: Dict[str, Any], row: Sequence[float]) -> float:
    z = [
        (x - m) / s if s >= _STD_FLOOR else 0.0
        for x, m, s in zip(row, model["means"], model["stds"])
    ]
    return model["intercept"] + sum(b * zj for b, zj in zip(model["betas"], z))


# ----------------------------------------------------------------- returns


def _clamp_return(value: float, max_abs: Optional[float]) -> float:
    """Winsorize a log return at +/- max_abs (rule R-winsor):
    a single absurd-but-trusted print — one the sigma fence
    correctly holds OUT of the index — must not enter every OTHER source's
    features and LOO targets at unbounded magnitude and poison up to
    history_days of scoring windows. The cap is a pinned calc param,
    set generously above any genuine repricing ever captured (0.5 =~ a
    +/-65% move vs the largest real event, massedcompute's ~10.9%);
    clamping keeps the sample (a genuine reprice still registers at the
    cap) where dropping it would blind the estimator."""
    if max_abs is None:
        return value
    cap = float(max_abs)
    if value > cap:
        return cap
    if value < -cap:
        return -cap
    return value


def source_return(
    prices: Dict[int, Dict[str, Any]],
    t0: int,
    t1: int,
    *,
    max_abs_log_return: Optional[float] = None,
) -> Optional[float]:
    """The source's own log return in NATIVE terms: log(native(t1)/
    native(t0)) from real trusted prints at BOTH exact dates, in the SAME
    recorded currency — a return spanning a currency change is not a
    return (dropped, never FX-spliced). Winsorized at
    max_abs_log_return when set (R-winsor)."""
    p0 = prices.get(t0)
    p1 = prices.get(t1)
    if p0 is None or p1 is None:
        return None
    if p0.get("currency") != p1.get("currency"):
        return None
    n0 = p0.get("native")
    n1 = p1.get("native")
    if not _is_pos_number(n0) or not _is_pos_number(n1):
        return None
    return _clamp_return(math.log(n1 / n0), max_abs_log_return)


def _is_pos_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def loo_basket_return(
    price_series: Dict[str, Dict[int, Dict[str, Any]]],
    weight_vector: Dict[str, float],
    *,
    exclude: str,
    t0: int,
    t1: int,
    max_abs_log_return: Optional[float] = None,
) -> Optional[float]:
    """Leave-one-out basket log return over [t0, t1] in USD: weights AND
    composition fixed (module docstring) — one pinned weight vector at both
    endpoints, over sources (never ``exclude``) with trusted prints at
    both. The renormalizing denominators cancel in the ratio. None when no
    source spans the interval or an endpoint sum is non-positive.
    """
    num = 0.0
    den = 0.0
    common = 0
    for sid in sorted(weight_vector):
        if sid == exclude:
            continue
        weight = weight_vector[sid]
        if not _is_pos_number(weight):
            continue
        series = price_series.get(sid) or {}
        p0 = (series.get(t0) or {}).get("usd")
        p1 = (series.get(t1) or {}).get("usd")
        if not _is_pos_number(p0) or not _is_pos_number(p1):
            continue
        num += float(weight) * float(p1)
        den += float(weight) * float(p0)
        common += 1
    if common == 0 or num <= 0 or den <= 0:
        return None
    return _clamp_return(math.log(num / den), max_abs_log_return)


# ------------------------------------------------------------------ samples


def build_samples(
    price_series: Dict[str, Dict[int, Dict[str, Any]]],
    weight_vectors: Dict[int, Dict[str, float]],
    *,
    source_id: str,
    cutoff_hour: int,
    horizon_hours: int,
    lookbacks_hours: Sequence[int],
    history_hours: int,
    max_abs_log_return: Optional[float] = None,
) -> List[Tuple[int, List[float], float]]:
    """(tau, X, y) samples on the capture SLOT grid (R-slots):
    every timestamp is an integer HOUR stamp
    (day_ordinal * 24 + slot_hour_utc), so the 6h capture cadence yields
    four sample anchors per day and the horizon set {6h, 1d, 2d} is native
    — 1h joins when capture itself walks to hourly snapshots.

    Anchors are the hour stamps where the source ITSELF has a trusted
    print. A sample at tau exists only when: tau >= cutoff - history
    (window); tau + horizon <= cutoff (R-cutoff on the slot grid — every
    endpoint, feature and target alike, is realized by the prior day's
    LAST slot, so nothing captured today can move today's weights); the
    pinned weight vector of tau's DAY exists; and every lookback feature
    and the target are computable from real trusted prints at exact
    endpoints. Returns are winsorized at max_abs_log_return (R-winsor).
    """
    series = price_series.get(source_id) or {}
    samples: List[Tuple[int, List[float], float]] = []
    start = int(cutoff_hour) - int(history_hours)
    for tau in sorted(series):
        if tau < start or tau + int(horizon_hours) > int(cutoff_hour):
            continue
        vector = weight_vectors.get(tau // 24)
        if not vector:
            continue
        target = loo_basket_return(
            price_series,
            vector,
            exclude=source_id,
            t0=tau,
            t1=tau + int(horizon_hours),
            max_abs_log_return=max_abs_log_return,
        )
        if target is None:
            continue
        features: List[float] = []
        for lb in lookbacks_hours:
            own = source_return(
                series,
                tau - int(lb),
                tau,
                max_abs_log_return=max_abs_log_return,
            )
            rest = loo_basket_return(
                price_series,
                vector,
                exclude=source_id,
                t0=tau - int(lb),
                t1=tau,
                max_abs_log_return=max_abs_log_return,
            )
            if own is None or rest is None:
                features = []
                break
            features.append(own - rest)
        if not features:
            continue
        samples.append((tau, features, target))
    return samples


# ---------------------------------------------------------------- scoring


def in_sample_q(
    samples: Sequence[Tuple[int, List[float], float]],
    *,
    anchor: int,
    ridge_lambda: float,
    half_life: float,
    min_train_samples: int,
    target_variance_floor: float,
) -> Tuple[Optional[float], int]:
    """(q = max(0, weighted IN-SAMPLE R^2) or None when undefined,
    n_samples).

    R-insample (supersedes the earlier walk-forward out-of-sample
    protocol, which never published a day):
    one exponentially weighted ridge is fitted at day
    t on every realized sample — build_samples already enforces the
    anti-manipulation cutoff (tau + h <= t-1) and the history window —
    and q scores that SAME fit on its own training data. No out-of-sample
    evaluation points are required, so a score defines at
    min_train_samples realized samples (~a week earlier than the OOS
    protocol on this data). The accepted trade, recorded for the audit
    trail: in-sample R^2 is optimistic — under ridge with an unpenalized
    intercept it is mathematically >= 0 (the intercept-only fit is always
    feasible), so the max(0, .) clip guards only float dust and the
    counterweight to noise-minted scores is the ridge shrinkage itself
    (ridge_lambda 1.0 on z-scored features) plus the softmax/cap fences
    downstream.

    SSE, the variance denominator, and the target mean all use ONE
    measure — the exponential decay toward ``anchor`` with ``half_life``,
    both in the SAME units as the sample timestamps (hour stamps on the
    slot grid). Undefined (None) when fewer than min_train_samples samples
    exist, the weighted target variance sits below the floor (a frozen
    basket has no movement to explain — 0/0 must never mint a score), the
    fit is singular, or the R^2 is non-finite (a numerical pathology must
    surface as an auditable null, never launder into a plausible score —
    R-audit-null).
    """
    ordered = sorted(samples, key=lambda s: s[0])
    n = len(ordered)
    if n < int(min_train_samples):
        return None, n
    sample_weights = [
        2.0 ** (-(anchor - s[0]) / float(half_life)) for s in ordered
    ]
    targets = [s[2] for s in ordered]
    total_w = sum(sample_weights)
    y_bar = sum(w * y for w, y in zip(sample_weights, targets)) / total_w
    variance = (
        sum(w * (y - y_bar) ** 2 for w, y in zip(sample_weights, targets))
        / total_w
    )
    if variance <= 0.0 or variance < float(target_variance_floor):
        # <= 0 belt-and-braces even at floor 0: a frozen target must yield
        # UNDEFINED, never a ZeroDivisionError that darks the publish.
        return None, n
    model = fit_ridge(
        [s[1] for s in ordered],
        targets,
        sample_weights,
        ridge_lambda=ridge_lambda,
    )
    if model is None:
        return None, n
    sse = sum(
        w * (y - predict_ridge(model, s[1])) ** 2
        for w, y, s in zip(sample_weights, targets, ordered)
    )
    r2 = 1.0 - sse / (variance * total_w)
    if not math.isfinite(r2):
        return None, n
    return max(0.0, r2), n


# ------------------------------------------------------------- allocation


def allocate_weights(
    scores: Dict[str, float],
    *,
    gamma: float,
    weight_min: float,
    weight_max: float,
    source_caps: Optional[Dict[str, float]] = None,
    attendance_factors: Optional[Dict[str, float]] = None,
    attendance_eta: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Softmax allocation (METHODOLOGY.md §8.7) plus the risk-cap layer: softmax(gamma * Q)
    shares over the eligible set, everyone floored at weight_min, the
    remainder distributed by share, then caps applied by iteratively
    capping violators at cap_i = min(weight_max, source_caps[i]) and
    redistributing their excess to the uncapped proportionally to share.
    The capped set grows monotonically, so the loop terminates within N
    passes. Full precision throughout; the caller rounds ONCE at the end.

    ATTENDANCE (METHODOLOGY.md §8.7): ``attendance_factors`` ({sid: A_i
    in [0, 1], FULL precision}; named to stay clear of this module's
    attendance() ratio helper) and ``attendance_eta`` arm the
    attendance-weighted variant, dispatched to
    _allocate_weights_attendance below. The gate is STRUCTURAL: with the
    factors None (every day-mode caller -- the frozen daily series never
    passes them) or eta == 0 NOTHING new is computed -- no log, no
    per-seat floor arithmetic -- and this body runs byte-identically to
    the pre-attendance engine.

    Per-source caps are optional RISK haircuts (e.g. a thin-order-book
    marketplace cap): predictiveness allocates WITHIN risk bounds, never
    above them, or the scheme structurally walks the least reliable source
    to the global cap. Current configs set none (calc_v6 removed them);
    the mechanism remains for lanes that need one.

    Degenerate bounds are flagged LOUDLY in ``degenerate_allocation``
    rather than crashing a publish (the flags ride the artifact's
    weight_calc block so a degenerate day is visible forever):

      - total cap mass < 1 (caps cannot hold the mass; N <= 3 at the 30%
        cap is a REAL day on the B300 lane, claim floor 1) publishes
        CAP-PROPORTIONAL weights, w_i = cap_i / sum(caps) — every source
        exceeds its cap by the same minimal factor, so the negotiated
        RELATIVE haircuts survive exactly when they matter most (uniform
        1/N would hand the thin-book source 2.5x its risk cap on the thin
        days the caps exist for). Floors hold by construction (each scaled
        weight >= its cap >= weight_min).
      - N*weight_min > 1 (floors alone overflow) or a cap below the floor
        (unsatisfiable arithmetic; the validator refuses such configs at
        load — this is runtime armor) publishes exactly uniform 1/N.
    """
    if attendance_factors is not None and attendance_eta > 0:
        return _allocate_weights_attendance(
            scores,
            gamma=gamma,
            weight_min=weight_min,
            weight_max=weight_max,
            source_caps=source_caps,
            attendance_factors=attendance_factors,
            eta=attendance_eta,
        )
    ids = sorted(scores)
    n = len(ids)
    flags: Dict[str, Any] = {"degenerate_allocation": None, "capped": []}
    if n == 0:
        return {}, flags
    caps = {
        sid: min(float(weight_max), float((source_caps or {}).get(sid, weight_max)))
        for sid in ids
    }
    uniform = {sid: 1.0 / n for sid in ids}
    if n * weight_min > 1.0 + _CAP_EPS or any(
        caps[sid] < weight_min - _CAP_EPS for sid in ids
    ):
        flags["degenerate_allocation"] = "uniform"
        flags["fallback_reason"] = (
            f"bounds unsatisfiable for n={n}: n*w_min={n * weight_min:.6f}"
        )
        return uniform, flags
    cap_mass = sum(caps.values())
    if cap_mass < 1.0 - _CAP_EPS:
        flags["degenerate_allocation"] = "cap_proportional"
        flags["fallback_reason"] = (
            f"cap mass {cap_mass:.6f} < 1 for n={n}: weights scaled "
            "proportionally to the caps"
        )
        return {sid: caps[sid] / cap_mass for sid in ids}, flags
    q_max = max(scores.values())
    exp_shares = {sid: math.exp(gamma * (scores[sid] - q_max)) for sid in ids}
    share_total = sum(exp_shares[sid] for sid in ids)
    shares = {sid: exp_shares[sid] / share_total for sid in ids}
    spread = 1.0 - n * weight_min
    weights = {sid: weight_min + spread * shares[sid] for sid in ids}
    _redistribute_capped_excess(ids, weights, shares, caps, flags)
    return weights, flags


def _redistribute_capped_excess(
    ids: List[str],
    weights: Dict[str, float],
    shares: Dict[str, float],
    caps: Dict[str, float],
    flags: Dict[str, Any],
) -> None:
    """The iterative cap solver, shared VERBATIM by the legacy and the
    attendance-armed allocators (the two bodies were byte-identical
    copies): cap violators at cap_i, hand their excess to the uncapped
    proportionally to share. The capped set grows monotonically, so the
    loop terminates within len(ids) passes. Mutates ``weights`` and sets
    flags["capped"]. FLOAT-OP ORDER IS LOAD-BEARING: the legacy path's
    bytes are golden-pinned and the armed path's A_i=1 reduction is
    bit-parity-pinned against it, so any edit here must keep the exact
    operation sequence.

    Corners, each inherited with its original rationale:
      - every source at its cap: the mass invariant plus the callers'
        cap-mass gate force sum(caps) == 1 within float dust (the
        exactly-1.0 corner: three 0.30s + a 0.10) -- the honest
        allocation IS the caps; the residual excess is dust.
      - pool underflow: extreme-but-finite gamma can underflow every
        non-max share to exactly 0.0; the excess is still conserved
        deterministically by spreading it equally over the uncapped
        rather than darking the publish on a divide. (On the armed path
        an A_i = 0 seat that picks up dust here exceeds its 0 cap and is
        capped back to its explicit 0.0 row on the next pass --
        termination untouched.)
    """
    capped: List[str] = []
    for _ in range(len(ids)):
        over = [
            sid
            for sid in ids
            if sid not in capped and weights[sid] > caps[sid] + _CAP_EPS
        ]
        if not over:
            break
        excess = 0.0
        for sid in over:
            excess += weights[sid] - caps[sid]
            weights[sid] = caps[sid]
            capped.append(sid)
        uncapped = [sid for sid in ids if sid not in capped]
        if not uncapped:
            flags["capped"] = sorted(capped)
            return
        pool = sum(shares[sid] for sid in uncapped)
        if pool <= 0.0:
            for sid in uncapped:
                weights[sid] += excess / len(uncapped)
            continue
        for sid in uncapped:
            weights[sid] += excess * shares[sid] / pool
    flags["capped"] = sorted(capped)


def _allocate_weights_attendance(
    scores: Dict[str, float],
    *,
    gamma: float,
    weight_min: float,
    weight_max: float,
    source_caps: Optional[Dict[str, float]],
    attendance_factors: Dict[str, float],
    eta: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The ARMED attendance-weighted allocation (METHODOLOGY.md §8.7);
    reached only via allocate_weights' structural gate, eta > 0. The
    domain (``scores`` keys) is the caller's D4-extended voting set
    (eligible union carry-casting seats). Deviations from the legacy
    body, each ruled:

      1. EXPONENT: x_i = gamma*(Q_i - Q_max) + eta*ln(A_i), stabilized
         by subtracting max_i x_i over the COMBINED exponent (both terms
         <= 0 after the shift, so exp never overflows at any gamma/eta).
         An A_i = 0 seat is dropped from the softmax BEFORE any log (ln
         is never called on 0) -- its cap and floor are 0 and it keeps
         an explicit 0.0 weight row when in the domain (a disclosed
         invariant change while armed: the engine otherwise never
         publishes a literal-0 weight).
      2. CEILING: cap_i = min(weight_max * A_i, source_caps_i) -- the
         risk caps stay a separate upper bound.
      3. FLOOR COLLAPSES WITH THE CEILING (rule D6): per-seat floor_i =
         min(weight_min, cap_i); a fading seat rides smoothly to
         near-zero instead of hitting a second exclusion cliff at
         A_i ~ w_min/w_max. The legacy any-cap-below-floor uniform armor
         is therefore NOT here -- unreachable from attendance by
         construction (floors <= caps), it remains config armor on the
         legacy path only. The spread is written as (1 - n*w_min) +
         sum(w_min - floor_i) so at A_i = 1 for all i every intermediate
         float -- caps, floors, exponents, spread -- reduces BIT-EXACTLY
         to the legacy body (the A=1 bit-parity pin).
      4. INFEASIBLE CEILINGS (rule D7): sum(cap) < 1 publishes the
         cap-proportional expansion cap_i / sum(caps) -- the methodology's
         c_i/C formula (w_max*A_i / sum(A_j*w_max) when no risk cap
         binds); allocation pinned at the ceilings, Q/eta have no slack
         on those stamps only; flagged.
      5. CORNER GUARDS: sum(cap) == 0 (every domain seat at A_i = 0)
         publishes uniform-over-domain with a fallback_reason
         (config-armor semantics -- division guarded); n*w_min > 1 keeps
         the legacy uniform armor verbatim.
    """
    ids = sorted(scores)
    n = len(ids)
    flags: Dict[str, Any] = {"degenerate_allocation": None, "capped": []}
    if n == 0:
        return {}, flags
    factors = {sid: float(attendance_factors[sid]) for sid in ids}
    caps = {
        sid: min(
            float(weight_max) * factors[sid],
            float((source_caps or {}).get(sid, weight_max)),
        )
        for sid in ids
    }
    uniform = {sid: 1.0 / n for sid in ids}
    if n * weight_min > 1.0 + _CAP_EPS:
        flags["degenerate_allocation"] = "uniform"
        flags["fallback_reason"] = (
            f"bounds unsatisfiable for n={n}: n*w_min={n * weight_min:.6f}"
        )
        return uniform, flags
    cap_mass = sum(caps.values())
    if cap_mass <= _CAP_EPS:
        # Every domain seat at A_i = 0 (a first-print panel corner): no
        # ceiling can hold any mass -- publish uniform loudly rather
        # than divide by zero.
        flags["degenerate_allocation"] = "uniform"
        flags["fallback_reason"] = (
            f"attendance cap mass 0 for n={n}: every domain seat at "
            "A_i=0; uniform weights published"
        )
        return uniform, flags
    if cap_mass < 1.0 - _CAP_EPS:
        # Rule D7: the existing cap-proportional branch IS the
        # methodology's c_i/C expansion -- expanded ceilings sum to
        # exactly 1, the allocation is pinned at the ceilings, and the
        # stamp is audit-visible forever.
        flags["degenerate_allocation"] = "cap_proportional"
        # The reason string is the LEGACY text verbatim (the caps here
        # are the attendance-scaled ones): at A_i = 1 for all i the
        # armed body must reduce bit-exactly to the legacy body, flags
        # included -- the fallback_reason rides the artifact.
        flags["fallback_reason"] = (
            f"cap mass {cap_mass:.6f} < 1 for n={n}: weights scaled "
            "proportionally to the caps"
        )
        return {sid: caps[sid] / cap_mass for sid in ids}, flags
    floors = {sid: min(float(weight_min), caps[sid]) for sid in ids}
    q_max = max(scores.values())
    # A_i = 0 seats leave the softmax BEFORE the log (docstring item 1);
    # cap_mass >= 1 - eps above guarantees at least one A_i > 0 seat.
    exponents = {
        sid: gamma * (scores[sid] - q_max) + eta * math.log(factors[sid])
        for sid in ids
        if factors[sid] > 0.0
    }
    x_max = max(exponents.values())
    exp_shares = {sid: math.exp(x - x_max) for sid, x in exponents.items()}
    share_total = sum(exp_shares[sid] for sid in exp_shares)
    shares = {sid: exp_shares.get(sid, 0.0) / share_total for sid in ids}
    deficit_total = sum(weight_min - floors[sid] for sid in ids)
    spread = (1.0 - n * weight_min) + deficit_total
    weights = {sid: floors[sid] + spread * shares[sid] for sid in ids}
    # The SAME iterative cap solver as the legacy body (one home --
    # _redistribute_capped_excess): per-seat attendance ceilings ride in
    # through ``caps``; the A=1 bit-parity pin covers the sharing.
    _redistribute_capped_excess(ids, weights, shares, caps, flags)
    return weights, flags


# --------------------------------------------------------------- top level


def predictive_scores(
    weight_state: Dict[str, Any],
    *,
    day: str,
    source_ids: Sequence[str],
    dw_params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Per-source scoring block: q per forward horizon (None = undefined),
    sample counts, and Q (mean over horizons, defined only when EVERY
    horizon's q is — an all-or-nothing rule so Q never jumps when a longer
    horizon comes online days after a shorter one). All horizon math runs
    on the hour-stamp slot grid; the R-cutoff information boundary is the
    prior day's LAST capture slot."""
    cutoff_hour = _cutoff_hour(day, dw_params)
    prices = weight_state.get("prices") or {}
    vectors = weight_state.get("vectors") or {}
    lookbacks = [int(x) for x in dw_params["lookback_horizons_hours"]]
    horizons = [int(x) for x in dw_params["forward_horizons_hours"]]
    # DAY MODE stays on the HOUR lattice (frozen daily series -- the
    # 15-min re-base touches observation mode only; the two lattices are
    # never mixed in one lane).
    history_hours = int(dw_params["history_days"]) * 24
    half_life_hours = float(dw_params["half_life_days"]) * 24.0
    max_abs = dw_params.get("max_abs_log_return")
    out: Dict[str, Dict[str, Any]] = {}
    for sid in source_ids:
        q_by_h: Dict[str, Optional[float]] = {}
        n_by_h: Dict[str, int] = {}
        for h in horizons:
            samples = build_samples(
                prices,
                vectors,
                source_id=sid,
                cutoff_hour=cutoff_hour,
                horizon_hours=h,
                lookbacks_hours=lookbacks,
                history_hours=history_hours,
                max_abs_log_return=max_abs,
            )
            q, n_samples = in_sample_q(
                samples,
                anchor=cutoff_hour,
                ridge_lambda=float(dw_params["ridge_lambda"]),
                half_life=half_life_hours,
                min_train_samples=int(dw_params["min_train_samples"]),
                target_variance_floor=float(
                    dw_params.get(
                        "target_variance_floor", DEFAULT_TARGET_VARIANCE_FLOOR
                    )
                ),
            )
            q_by_h[str(h)] = q
            n_by_h[str(h)] = n_samples
        defined = all(q is not None for q in q_by_h.values())
        q_score = (
            sum(q for q in q_by_h.values() if q is not None) / len(q_by_h)
            if defined and q_by_h
            else None
        )
        out[sid] = {"q": q_by_h, "n_samples": n_by_h, "Q": q_score}
    return out


def _cutoff_hour(day: str, dw_params: Dict[str, Any]) -> int:
    """The R-cutoff information boundary for day t's weights, in hour
    stamps: the prior day's LAST capture slot. Everything a sample touches
    (features and target alike) must be realized by this stamp, so nothing
    captured on day t can move day t's own weights. The slot grid rides
    dw_params (slot_hours_utc, embedded in calc_params) so a capture-
    cadence change is a minted methodology change, never a silent shift of
    the cutoff."""
    return (_ordinal(day) - 1) * 24 + max(
        int(h) for h in dw_params["slot_hours_utc"]
    )


def recently_printed(
    weight_state: Dict[str, Any], *, day: str, history_days: int
) -> List[str]:
    """Sources with at least one trusted series print inside the trailing
    history window — the switch-quorum domain (rule R-quorum). Sorted
    for determinism. Stamps are hours on the slot grid."""
    end_hour = _ordinal(day) * 24
    start_hour = end_hour - int(history_days) * 24
    out = []
    for sid, series in (weight_state.get("prices") or {}).items():
        if any(start_hour <= t < end_hour for t in series):
            out.append(sid)
    return sorted(out)


def dw_vote_tail(
    prices: Optional[Dict[int, Dict[str, Any]]],
    *,
    obs_stamp: int,
    window_minutes: int,
    currency: str,
) -> List[float]:
    """The dw_history vote tail (ruling 2026-08-27) for ONE source: its
    weight-state price entries (series_print shape, owned by this module)
    with stamp in [obs_stamp - window_minutes, obs_stamp) -- closed on the
    old edge, open at the observation, minute-stamp arithmetic (the
    observation-mode lattice; the attendance() unit convention) -- whose
    recorded filter currency matches the print being voted,
    stamp-ascending, in NATIVE (filter) terms.

    Why each clause is load-bearing:
      - the weight-state prices series is naturally PRE-advance at vote
        time (advance_panel_weight_state runs after votes), so today's
        print never judges its own conviction -- the same discipline as
        the fence window;
      - CURRENCY SCOPING: the tail is EVERY same-currency trusted print
        in the span, whatever happened in between. The dw series is never
        era-reset on a currency change (deliberate -- it matches the
        regression, where a same-currency return pair straddling a
        round-trip also counts); only the FENCE's window reseeds at
        confirmation, the vote tail does not. So a currency-confirmed
        print's tail holds its trusted pending prints PLUS any older
        same-currency history in the span -- including broken-streak
        prints the legacy pending tail never saw -- and a source that
        ROUND-TRIPS back to a previously-recorded currency re-admits its
        old-era same-label prints;
      - gaps are absent stamps, never carried forward: the tail is the
        regression's own per-source price series, cadence-era sample
        counts included (4/day before the hourly cutover, 24/day after,
        96/day on a 15-minute era). One precision: this tail is
        OBS-anchored -- [obs - window_minutes, obs), the
        attendance-window convention -- where the regression's sample
        window is CUTOFF-anchored; same horizon, same series, different
        anchor;
      - fence-held-out trusted prints are IN (the fence holds a print out
        of the INDEX, never out of the weight series), manual exclusions /
        untrusted-currency / L5-quarantined prints never entered.

    Fewer than two surviving entries mean vote_stddev's sigma-0 path: the
    pct floor IS the interval, exactly the day-one rule."""
    start = int(obs_stamp) - int(window_minutes)
    end = int(obs_stamp)
    return [
        float(entry["native"])
        for stamp, entry in sorted((prices or {}).items())
        if start <= int(stamp) < end and entry.get("currency") == currency
    ]


def compute_dynamic_weights(
    weight_state: Dict[str, Any],
    *,
    day: str,
    eligible: Sequence[str],
    dw_params: Dict[str, Any],
    fallback_weights: Dict[str, float],
) -> Dict[str, Any]:
    """One day's weight_calc block: the pinned rounded-6dp weight vector
    over the ELIGIBLE sources plus every input needed to audit the softmax
    -> floor -> cap chain from the artifact alone.

    Mode: "fallback" (the config opening weights restricted to the
    eligible set — index math byte-identical to the prior fixed-weight
    series) until the switch quorum is met, then "dynamic" permanently
    (weight_state["mode"] carries the latch; the caller persists it via
    the artifact). Post-switch an undefined Q is 0.

    The switch quorum (R-quorum-v2, superseding the earlier
    all-recently-printed rule): the latch flips
    on the first day at least ``switch_min_eligible`` sources are BOTH
    eligible today AND have a defined Q — enough scored providers to
    carry an index. A sparse or late source no longer holds the switch;
    it simply
    scores 0 from the switch day, published as an auditable ``Q: null``,
    floored at weight_min once it prints. The recorded trade: the
    original rule also stopped a one-day print suppression from firing
    the permanent switch before the suppressed source was ever scored —
    that protection is deliberately given up; the eligible-count leg and
    the R-cutoff (nothing captured today moves today's weights) remain.

    NaN/inf anywhere in the result raises — a poisoned weight must kill the
    publish loudly, never price a silently-wrong index (the non-finite-vote
    precedent).
    """
    eligible = list(eligible)
    quorum_sources = recently_printed(
        weight_state, day=day, history_days=int(dw_params["history_days"])
    )
    scored = list(eligible) + [
        sid for sid in quorum_sources if sid not in eligible
    ]
    scores = predictive_scores(
        weight_state, day=day, source_ids=scored, dw_params=dw_params
    )
    prior_mode = weight_state.get("mode", MODE_FALLBACK)
    min_eligible = int(dw_params.get("switch_min_eligible", 1))
    defined_eligible = [
        sid for sid in eligible if scores[sid]["Q"] is not None
    ]
    quorum_met = (
        len(eligible) >= min_eligible
        and len(defined_eligible) >= min_eligible
    )
    mode = MODE_DYNAMIC if (prior_mode == MODE_DYNAMIC or quorum_met) else MODE_FALLBACK
    switched = mode == MODE_DYNAMIC and prior_mode != MODE_DYNAMIC

    flags: Dict[str, Any] = {"degenerate_allocation": None, "capped": []}
    if not eligible:
        weights: Dict[str, float] = {}
    elif mode == MODE_FALLBACK:
        # The opening weights RESTRICTED to the eligible set, deliberately
        # NOT renormalized here: every consumer is scale-invariant (the
        # composite renormalizes over passers exactly as the prior series
        # did; LOO returns are ratios under one vector), so pinning the raw
        # config values keeps fallback-mode index math BYTE-identical to
        # the frozen fixed-weight series — no rounding residue can leak in.
        weights = {sid: float(fallback_weights[sid]) for sid in eligible}
    else:
        q_values = {
            sid: (
                scores[sid]["Q"] if scores[sid]["Q"] is not None else 0.0
            )
            for sid in eligible
        }
        weights, flags = allocate_weights(
            q_values,
            gamma=float(dw_params["gamma"]),
            weight_min=float(dw_params["weight_min"]),
            weight_max=float(dw_params["weight_max"]),
            source_caps=dw_params.get("source_weight_caps") or {},
        )
    rounded = {sid: round(w, 6) for sid, w in weights.items()}
    for sid, w in rounded.items():
        if not (math.isfinite(w) and w >= 0):
            raise ValueError(
                f"{day}: non-finite or negative weight for {sid}: {w!r}"
            )
    block: Dict[str, Any] = {
        "scheme": dw_params["scheme"],
        "mode": mode,
        "weights": {sid: rounded[sid] for sid in sorted(rounded)},
        # q/Q/n_samples pin the audit trail. 9dp (vs the weights' 6dp): at
        # gamma=4 a 5e-7 Q rounding error can flip a 6dp weight digit, so
        # the softmax -> floor -> cap chain must be recomputable from the
        # artifact's own numbers to beyond weight precision.
        "sources": {
            sid: {
                "Q": _round_opt(scores[sid]["Q"]),
                "q": {h: _round_opt(q) for h, q in sorted(scores[sid]["q"].items())},
                "n_samples": dict(sorted(scores[sid]["n_samples"].items())),
            }
            for sid in sorted(scores)
        },
        "degenerate_allocation": flags.get("degenerate_allocation"),
        "capped": flags.get("capped", []),
    }
    if flags.get("fallback_reason"):
        block["fallback_reason"] = flags["fallback_reason"]
    if switched:
        # The day the series left fallback — a step-change in weights with
        # zero price movement; flagged forever so the artifact series
        # self-describes the kink.
        block["switched_on"] = day
    return block


def _round_opt(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 9)


def advance_weight_state(
    weight_state: Dict[str, Any],
    *,
    day: str,
    weight_block: Optional[Dict[str, Any]],
) -> None:
    """Advance the weight state with one published day's facts — R-slots:
    the day's pinned SLOT prints (weight_calc.slot_prints: the just-closed
    prior day's per-slot trusted prints, {"date": iso, "slots":
    {slot_hour_str: {sid: {usd, native, currency}}}}), the day's pinned
    rounded weight vector, and the mode latch. ONE state machine for both
    paths — compute_day (new days, which pins the block it advanced from)
    and the CLI's replay-from-published — so full replays and mid-series
    restarts rebuild identical state. Series stamps are integer HOURS
    (day_ordinal * 24 + slot_hour); the mode latch is monotonic (fallback
    -> dynamic, never back)."""
    ordinal = _ordinal(day)
    if weight_block:
        slot_prints = weight_block.get("slot_prints") or {}
        slots = slot_prints.get("slots") or {}
        if slots:
            base = _ordinal(str(slot_prints["date"])) * 24
            prices = weight_state.setdefault("prices", {})
            for hour_str, by_source in slots.items():
                stamp = base + int(hour_str)
                for sid, entry in by_source.items():
                    prices.setdefault(sid, {})[stamp] = {
                        "usd": entry["usd"],
                        "native": entry["native"],
                        "currency": entry["currency"],
                    }
        vector = weight_block.get("weights") or {}
        if vector:
            weight_state.setdefault("vectors", {})[ordinal] = dict(vector)
        if weight_block.get("mode") == MODE_DYNAMIC:
            weight_state["mode"] = MODE_DYNAMIC
        else:
            weight_state.setdefault("mode", MODE_FALLBACK)


# ------------------------------------- observation mode (hourly panel lanes)
#
# The hourly panel engine (METHODOLOGY.md) recomputes weights at EVERY scheduled observation of an
# era-aware grid (gpu_index.index.panel_schedule.PanelSchedule). Everything below is
# ADDITIVE: the day-mode functions above keep pricing the frozen daily
# B300/B200 series byte-for-byte, and the two modes never share a state
# (see new_weight_state). The pure pieces -- fit_ridge, in_sample_q,
# _clamp_return, source_return, allocate_weights, series_print -- are
# REUSED, never forked. Observation-mode divergences, each a design-doc
# section 5 rule:
#
#   - R-cutoff per observation: cutoff = the previous SCHEDULED stamp
#     (era-aware; across the 4-slot -> hourly boundary the first hourly
#     stamp's cutoff is the last 4-slot stamp). Nothing observed at t can
#     enter t's weights.
#   - Vectors per observation: state["vectors"] is keyed by hour STAMP;
#     the LOO vector for sample tau = the vector published at the LAST
#     stamp <= tau, held fixed at both endpoints (the day-mode "tau's
#     day" rule re-minted for the hourly grid).
#   - LOO arithmetic is full-sum-minus-own: one pass per (vector, t0, t1)
#     endpoint pair builds the both-endpoints contribution sums, and every
#     source's leave-one-out value is that total minus its own
#     contribution -- all N exclusions from one O(N) pass (the design's
#     perf requirement; pure Python, numpy stays forbidden). This is THE
#     definition in observation mode, cached and uncached paths
#     bit-identical by construction (test-pinned against a naive
#     recompute-everything reimplementation).
#   - A2 transition rule:
#     dynamic iff prior mode dynamic OR (a NON-EMPTY attendance-passer
#     set exists AND every ATTENDANCE-PASSER has a defined Q AND >=
#     switch_min_eligible sources eligible at this observation).
#     Attendance = trusted-print stamps / scheduled stamps over the
#     trailing history window, clipped at genesis. This deliberately
#     RESTORES (bounded by the floor) the outage protection R-quorum-v2
#     traded away: a regular provider mid warm-up holds the switch; a
#     sparse one does not; ZERO passers (post-outage) also hold it --
#     the vacuous every-passer clause must never latch on no information.
#   - No slot_prints block: every observation's own artifact records its
#     trusted prints, so replay ingests each published observation's
#     prints at that stamp AFTER computing that stamp's vector -- one
#     state machine for live and replay, same as day mode.
#   - State is PRUNED to the trailing window (+ margin) at every advance;
#     provably score-neutral (see advance_panel_weight_state).

# Pruning margin beyond history + max forward horizon, in MINUTES (the
# observation-mode stamp unit -- 15-min cadence re-base 2026-08-27). One
# week -- generously above the widest live lookback horizon (48h), which
# the prune-safety proof needs (advance_panel_weight_state enforces it).
PRUNE_MARGIN_MINUTES = 168 * 60

_UNSET = object()


def validate_attendance_floor(dw_params: Dict[str, Any]) -> float:
    """Load-time fence for the observation-mode dw_params addition
    `attendance_floor` (A2). Called by the PANEL config loader --
    deliberately NOT wired into config.py's basket validation, where the
    day-mode configs have no attendance concept. Returns the validated
    float.

    Required and in (0, 1]: a missing floor must refuse at load (the
    switch rule is undefined without it), 0 would silently resurrect the
    superseded all-recently-printed rule (every candidate passes), and
    the finiteness check is the same JSON-Infinity armor as the basket
    validator's."""
    value = dw_params.get("attendance_floor")
    if not (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 < value <= 1
    ):
        raise ValueError(
            f"dynamic_weights.attendance_floor must be a finite number in "
            f"(0, 1], got {value!r}"
        )
    return float(value)


# The attendance-weighting knob triple (METHODOLOGY.md §8.6). Names are
# CROSS-REPO FROZEN (the published records, this engine, and the panel
# configs must agree verbatim); the three ride together as one
# conditional embed -- key ABSENT means the attendance-free legacy
# engine byte-identically (the D2 dark contract).
ATTENDANCE_PARAM_KEYS = (
    "attendance_half_life_hours",
    "attendance_eta",
    "no_price_exclusion_hours",
)

# weight_state["events"] codes: state-2-with-entry ("np": read fine, no
# usable price -- the streak advances) and state-3 ("sk": OUR failure --
# dropped from the A_i sums, streak held). State-1 is implicit (stamp
# present in state prices on the grid); a no-entry stamp is implicit too
# (absent from both -- counted 0 in A_i, streak held, the ramp-in row).
EVENT_NO_PRICE = "np"
EVENT_SKIP = "sk"
VALID_EVENT_CODES = (EVENT_NO_PRICE, EVENT_SKIP)


def attendance_minted(dw_params: Optional[Dict[str, Any]]) -> bool:
    """Whether a dynamic-weights param set carries the minted attendance
    knob triple -- THE predicate for every attendance codepath, one home
    (a hand-spelled key check drifting from this one is the silent-fork
    class). Presence of ONE member decides because validation refuses
    partial sets."""
    return "attendance_half_life_hours" in (dw_params or {})


def attendance_armed(dw_params: Optional[Dict[str, Any]]) -> bool:
    """Minted AND attendance_eta > 0 -- rule R2's single arming lever
    (carry, K_A exclusion, ceiling scaling, and the tilt all gate on
    it)."""
    return (
        attendance_minted(dw_params)
        and float(dw_params["attendance_eta"]) > 0
    )


def validate_attendance_params(
    dw_params: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Load-time fence for the attendance knob triple (METHODOLOGY.md
    section 8.6), the attendance_floor pattern: called by the PANEL
    config loader, one home for the bounds. Returns None when the triple
    is wholly absent (the knob-less legacy shape), else the validated
    float triple; raises on a partial set or an out-of-range value.

    Bounds, each load-bearing: the half-life and the exclusion window are
    wall-time HOURS (rule R3 -- an observation-counted knob silently
    changes meaning at every cadence mint) and must fit inside the
    history window the events state is retained for (the determinism
    bound needs events coverage over the exclusion window); eta >= 0 with
    eta == 0 the dark posture (rule D2). The eta>0-requires-carry-knobs
    rule (D5) is validated at the CALC level in panel_config -- the carry
    pair lives one nesting level up from this block."""
    present = [k for k in ATTENDANCE_PARAM_KEYS if k in dw_params]
    if not present:
        return None
    if len(present) != len(ATTENDANCE_PARAM_KEYS):
        missing = sorted(set(ATTENDANCE_PARAM_KEYS) - set(present))
        raise ValueError(
            f"dynamic_weights attendance knobs ride together (all three of "
            f"{list(ATTENDANCE_PARAM_KEYS)} or none): missing {missing}"
        )
    history_hours = int(dw_params["history_days"]) * 24
    half_life = dw_params["attendance_half_life_hours"]
    if not (
        isinstance(half_life, (int, float))
        and not isinstance(half_life, bool)
        and math.isfinite(half_life)
        and 0 < half_life <= history_hours
    ):
        raise ValueError(
            f"dynamic_weights.attendance_half_life_hours must be a finite "
            f"number in (0, history_days*24 = {history_hours}], got "
            f"{half_life!r}"
        )
    eta = dw_params["attendance_eta"]
    if not (
        isinstance(eta, (int, float))
        and not isinstance(eta, bool)
        and math.isfinite(eta)
        and eta >= 0
    ):
        raise ValueError(
            f"dynamic_weights.attendance_eta must be a finite number >= 0, "
            f"got {eta!r}"
        )
    exclusion = dw_params["no_price_exclusion_hours"]
    if not (
        isinstance(exclusion, (int, float))
        and not isinstance(exclusion, bool)
        and math.isfinite(exclusion)
        and 0 < exclusion <= history_hours
    ):
        raise ValueError(
            f"dynamic_weights.no_price_exclusion_hours must be a finite "
            f"number in (0, history_days*24 = {history_hours}], got "
            f"{exclusion!r}"
        )
    return {
        "attendance_half_life_hours": float(half_life),
        "attendance_eta": float(eta),
        "no_price_exclusion_hours": float(exclusion),
    }


def attendance(
    state_prices: Dict[str, Dict[int, Dict[str, Any]]],
    sid: str,
    *,
    obs_stamp: int,
    schedule: Any,
    window_minutes: int,
    scheduled_window: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """A2 attendance for one source at one observation:
    {printed, scheduled, ratio}.

    printed  = the source's trusted-print stamps in [obs - history, obs)
               that sit ON a scheduled stamp (era-aware; a print at an
               unscheduled stamp -- which the engine never writes -- can
               neither help nor hurt, so ratio is structurally in [0, 1]);
    scheduled = scheduled stamps in the same half-open window, counted
               PER ERA and clipped at the lane's genesis by the schedule
               itself, so pre-genesis hours never count as missed;
    ratio    = printed / scheduled, EXCEPT 1.0 when scheduled == 0 (the
               genesis corner): nothing was scheduled, so nothing was
               missed -- and the A2 switch test must see every candidate
               as a passer there, or the vacuous every-passer-defined
               clause could fire the PERMANENT switch at the very first
               observation with zero scored history.

    `scheduled_window` lets the caller precompute one window's scheduled
    stamps for all sources (compute_panel_weights does); it MUST equal
    schedule.scheduled_stamps(obs - history, obs) when passed.
    """
    window_start = int(obs_stamp) - int(window_minutes)
    if scheduled_window is None:
        scheduled_window = schedule.scheduled_stamps(
            window_start, int(obs_stamp)
        )
    scheduled_set = {int(s) for s in scheduled_window}
    series = (state_prices or {}).get(sid) or {}
    printed = sum(1 for t in series if t in scheduled_set)
    scheduled = len(scheduled_set)
    ratio = 1.0 if scheduled == 0 else printed / scheduled
    return {"printed": printed, "scheduled": scheduled, "ratio": ratio}


def compute_attendance_view(
    weight_state: Dict[str, Any],
    ids: Sequence[str],
    *,
    obs_stamp: int,
    schedule: Any,
    dw_params: Dict[str, Any],
    scheduled_window: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """The attendance-weighting per-source view at one observation
    (METHODOLOGY.md section 8.6): {sid: {factor, streak, excluded}} for
    every id, or None when the lane's dw_params lack the minted knob
    triple (the structural skip -- a knob-less lane never computes ANY
    of this, the D2 dark contract).

    factor -- the EWMA attendance A_i, exact: over the era-aware,
    genesis-clipped scheduled stamps in the HALF-OPEN pre-advance window
    [obs - history_days*1440, obs), with the decay
    w(s) = 2**(-(obs - s) / H_A_minutes),

        A_i = sum_{s not skip} w(s)*present_i(s) / sum_{s not skip} w(s)

    where present_i(s) = 1 exactly when s sits in the source's
    weight-state PRICES series (the trusted-print presence record --
    accepted, sigma-fenced, and mismatch-pending prints alike), state-3
    stamps (events code "sk") drop from numerator AND denominator (the
    only reading under which an always-present provider scores exactly
    1.0 through our-failure stamps), and every other scheduled stamp --
    "np" events and no-entry stamps alike -- counts 0 (the ramp-in rule).
    Denominator 0 (lane genesis / all-skip window) = 1.0, the existing
    ratio's genesis convention. FULL precision: the allocator consumes
    this value raw; the published 9dp attendance_factor is
    disclosure-only rounding at publication.

    Perf: ONE decay table per observation, shared across all sources --
    one pow per stamp, not per stamp*source; fixed iteration order over
    the sorted stamp window; pure Python, no incremental accumulator
    (bounded replay rebuilds from window history). Bit-identical to the
    naive per-source recompute (test-pinned).

    streak / excluded -- the K_A hard cutoff: a BACKWARD walk over the
    SAME scheduled window the EWMA reads -- [obs - history_days*1440,
    obs), half-open, pre-advance, era-aware, genesis-clipped -- in which
    SKIP stamps ("sk" events, our failures) consume NOTHING. Each
    non-skip stamp contributes ITS OWN era's spacing (the gap to its
    previous scheduled stamp, deterministic from the schedule alone; a
    genesis stamp with no predecessor contributes 0 -- conservative,
    later exclusion):

      - first state-1 stamp (in prices)      => NOT excluded (resolve);
      - accumulated non-skip span >= K_A
        with >= 1 state-2 seen               => EXCLUDED (resolve);
      - window exhausted, no state-1: when the window is the FULL
        history span (genesis_stamp <= obs - history*1440), EXCLUDED
        iff >= 1 state-2 was seen -- a seat with zero usable prints
        across the entire history window is past any K_A <= history
        bound, and a seat with no state-2 at all has nothing to carry
        anyway; a GENESIS-CLIPPED window (lane younger than history)
        resolves by the span leg alone, so a seat quiet since genesis
        still gets its full K_A grace.

    Deterministic under prune skew (a pure function of the
    strictly-retained window every walker holds), skip-robust (our-side
    outages can never un-exclude: a 23h outage bracketed by two bad
    prints is not 24h of absence), and era-mint-robust (per-stamp
    spacing -- no flap at cadence boundaries). ``no_price_streak`` is
    the consecutive state-2 count in the same walk -- held by "sk" and
    no-entry stamps, reset by state-1 -- and the walk continues past a
    resolved verdict to state-1 or the cap for the count, so the streak
    saturates at the window's stamp count and is identical whether the
    lane is armed or dark (disclosure-only). Exclusion fires ONLY when
    armed (attendance_eta > 0, rule R2); K_A is wall-time hours (rule
    R3) snapped to whole lattice minutes exactly like
    gpu_index.index.panel.carry_window_minutes. While dark the streak
    still publishes and excluded is present-but-False.

    Hours convert to minutes HERE, once -- the one seam."""
    if not attendance_minted(dw_params):
        return None
    obs_stamp = int(obs_stamp)
    half_life_minutes = float(dw_params["attendance_half_life_hours"]) * 60.0
    exclusion_minutes = int(
        round(float(dw_params["no_price_exclusion_hours"]) * 60.0)
    )
    armed = attendance_armed(dw_params)
    history_minutes = int(dw_params["history_days"]) * 1440
    if scheduled_window is None:
        scheduled_window = schedule.scheduled_stamps(
            obs_stamp - history_minutes, obs_stamp
        )
    # ONE decay table per observation: the window is shared by all
    # sources, so w(s) is too. The exclusion walk shares the SAME window
    # (the hard determinism cap), so its per-stamp era spacings -- each
    # stamp's gap to its previous scheduled stamp -- are likewise
    # computed once here: one prev_scheduled_stamp call for the window's
    # oldest stamp (0 at genesis: no predecessor exists), plain
    # differences inside.
    decay = [
        2.0 ** (-(obs_stamp - s) / half_life_minutes)
        for s in scheduled_window
    ]
    spacings: List[int] = []
    if scheduled_window:
        before_window = schedule.prev_scheduled_stamp(scheduled_window[0])
        last = before_window
        for s in scheduled_window:
            spacings.append(0 if last is None else int(s) - int(last))
            last = s
    # The exhausted-window leg's semantics split (docstring): a FULL
    # history window ends in the dead-across-history verdict; a
    # genesis-clipped one resolves by the span leg alone.
    window_is_full = int(schedule.genesis_stamp) <= obs_stamp - history_minutes
    prices = weight_state.get("prices") or {}
    event_state = weight_state.get("events") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for sid in sorted(set(str(s) for s in ids)):
        series = prices.get(sid) or {}
        events = event_state.get(sid) or {}
        numerator = 0.0
        denominator = 0.0
        for s, w in zip(scheduled_window, decay):
            if events.get(s) == EVENT_SKIP:
                continue
            denominator += w
            if s in series:
                numerator += w
        factor = 1.0 if denominator == 0.0 else numerator / denominator
        # The backward walk (docstring): skips consume nothing; the
        # verdict LATCHES where the law resolves it, and the walk then
        # continues to state-1 / the cap purely for the streak count
        # (arm-independent disclosure).
        streak = 0
        nonskip_span = 0
        state2_seen = False
        state1_found = False
        span_verdict = False
        for i in range(len(scheduled_window) - 1, -1, -1):
            s = scheduled_window[i]
            if s in series:
                state1_found = True
                break
            code = events.get(s)
            if code == EVENT_SKIP:
                continue  # our failure: consumes nothing, holds the run
            nonskip_span += spacings[i]
            if code == EVENT_NO_PRICE:
                state2_seen = True
                streak += 1
            # no-entry: contributes span, holds the streak
            if (
                not span_verdict
                and state2_seen
                and nonskip_span >= exclusion_minutes
            ):
                span_verdict = True
        # The verdict resolves at whichever comes FIRST walking backward:
        # a latched span verdict stands even when a state-1 print sits
        # deeper (the walk only continued past it for the streak count);
        # the exhausted-window legs apply only when neither resolved.
        excluded = span_verdict or (
            not state1_found and state2_seen and window_is_full
        )
        out[sid] = {
            "factor": factor,
            "streak": streak,
            "excluded": bool(armed and excluded),
        }
    return out


def predictive_scores_obs(
    weight_state: Dict[str, Any],
    *,
    obs_stamp: int,
    source_ids: Sequence[str],
    dw_params: Dict[str, Any],
    schedule: Any,
) -> Dict[str, Dict[str, Any]]:
    """Observation-mode per-source scoring block -- the day-mode
    predictive_scores contract (q per forward horizon, sample counts, Q =
    all-or-nothing mean) re-anchored to a per-observation cutoff:

      - cutoff = schedule.prev_scheduled_stamp(obs_stamp). None (the
        genesis observation) means every score is UNDEFINED -- never zero.
      - samples exactly as build_samples: anchors are the source's own
        trusted-print stamps with tau >= cutoff - history window and
        tau + horizon <= cutoff (minute-stamp arithmetic; config horizons
        are hours, converted once at the top); the LOO vector for tau is the vector at
        the LAST stamp <= tau from the stamp-keyed state["vectors"].
      - perf (design section 5): each tau's lookback features are
        computed ONCE across all forward horizons, and all N sources'
        LOO exclusions are served from ONE full-sum-minus-own pass per
        (vector, t0, t1) endpoint pair. The both-endpoints composition
        rule is preserved: a source contributes to the cached sums only
        with positive USD prints at BOTH endpoints, so subtracting the
        scored source's own contribution (when present) is exactly the
        both-endpoint LOO sum. Results are bit-identical to computing
        every sum from scratch (test-pinned).
    """
    # Config horizons are HOURS (the wire vocabulary and the artifact's
    # q/n_samples keys); stamp arithmetic is MINUTES (the observation-mode
    # lattice). Convert once here; the artifact labels keep the hour ints
    # so hour-grid lanes' bytes never move (re-base parity rule).
    lookbacks = [int(x) for x in dw_params["lookback_horizons_hours"]]
    horizons = [int(x) for x in dw_params["forward_horizons_hours"]]
    lookback_offsets = {lb: lb * 60 for lb in lookbacks}
    horizon_offsets = {h: h * 60 for h in horizons}
    cutoff = schedule.prev_scheduled_stamp(int(obs_stamp))
    if cutoff is None:
        return {
            sid: {
                "q": {str(h): None for h in horizons},
                "n_samples": {str(h): 0 for h in horizons},
                "Q": None,
            }
            for sid in source_ids
        }
    prices = weight_state.get("prices") or {}
    vectors = weight_state.get("vectors") or {}
    history_minutes = int(dw_params["history_days"]) * 1440
    half_life_minutes = float(dw_params["half_life_days"]) * 1440.0
    max_abs = dw_params.get("max_abs_log_return")
    ridge_lambda = float(dw_params["ridge_lambda"])
    min_train = int(dw_params["min_train_samples"])
    variance_floor = float(
        dw_params.get("target_variance_floor", DEFAULT_TARGET_VARIANCE_FLOOR)
    )

    vector_stamps = sorted(int(s) for s in vectors)

    def _vector_at(tau: int) -> Optional[Tuple[int, Dict[str, float]]]:
        # The vector in effect at tau: published at the last stamp <= tau.
        idx = bisect_right(vector_stamps, tau) - 1
        if idx < 0:
            return None
        stamp = vector_stamps[idx]
        return stamp, vectors[stamp]

    # One both-endpoints pass per (vector stamp, t0, t1): contribution sums
    # over sources with positive USD prints at BOTH endpoints, plus each
    # member's own (num, den) contribution for the minus-own subtraction.
    loo_cache: Dict[
        Tuple[int, int, int],
        Tuple[float, float, int, Dict[str, Tuple[float, float]]],
    ] = {}

    def _loo(
        vector_stamp: int,
        vector: Dict[str, float],
        exclude: str,
        t0: int,
        t1: int,
    ) -> Optional[float]:
        key = (vector_stamp, t0, t1)
        entry = loo_cache.get(key)
        if entry is None:
            num_total = 0.0
            den_total = 0.0
            count = 0
            own: Dict[str, Tuple[float, float]] = {}
            for member in sorted(vector):
                weight = vector[member]
                if not _is_pos_number(weight):
                    continue
                member_series = prices.get(member) or {}
                p0 = (member_series.get(t0) or {}).get("usd")
                p1 = (member_series.get(t1) or {}).get("usd")
                if not _is_pos_number(p0) or not _is_pos_number(p1):
                    continue
                c_num = float(weight) * float(p1)
                c_den = float(weight) * float(p0)
                num_total += c_num
                den_total += c_den
                count += 1
                own[member] = (c_num, c_den)
            entry = (num_total, den_total, count, own)
            loo_cache[key] = entry
        num, den, count, own = entry
        contribution = own.get(exclude)
        if contribution is not None:
            num -= contribution[0]
            den -= contribution[1]
            count -= 1
        if count == 0 or num <= 0 or den <= 0:
            return None
        return _clamp_return(math.log(num / den), max_abs)

    start = cutoff - history_minutes
    min_horizon_offset = min(horizon_offsets.values())
    out: Dict[str, Dict[str, Any]] = {}
    for sid in source_ids:
        series = prices.get(sid) or {}
        samples_by_h: Dict[int, List[Tuple[int, List[float], float]]] = {
            h: [] for h in horizons
        }
        for tau in sorted(series):
            if tau < start or tau + min_horizon_offset > cutoff:
                continue
            resolved = _vector_at(tau)
            if resolved is None:
                continue
            vector_stamp, vector = resolved
            if not vector:
                continue
            # Features depend on tau only -- computed at most ONCE across
            # all forward horizons (lazily, on the first horizon whose
            # target realizes; None poisons every horizon at this tau).
            features: Any = _UNSET
            for h in horizons:
                offset = horizon_offsets[h]
                if tau + offset > cutoff:
                    continue
                target = _loo(vector_stamp, vector, sid, tau, tau + offset)
                if target is None:
                    continue
                if features is _UNSET:
                    feats: Optional[List[float]] = []
                    for lb in lookbacks:
                        lb_offset = lookback_offsets[lb]
                        own_r = source_return(
                            series,
                            tau - lb_offset,
                            tau,
                            max_abs_log_return=max_abs,
                        )
                        rest_r = _loo(
                            vector_stamp, vector, sid, tau - lb_offset, tau
                        )
                        if own_r is None or rest_r is None:
                            feats = None
                            break
                        feats.append(own_r - rest_r)
                    features = feats
                if features is None:
                    break
                samples_by_h[h].append((tau, features, target))
        q_by_h: Dict[str, Optional[float]] = {}
        n_by_h: Dict[str, int] = {}
        for h in horizons:
            q, n_samples = in_sample_q(
                samples_by_h[h],
                anchor=cutoff,
                ridge_lambda=ridge_lambda,
                half_life=half_life_minutes,
                min_train_samples=min_train,
                target_variance_floor=variance_floor,
            )
            q_by_h[str(h)] = q
            n_by_h[str(h)] = n_samples
        defined = all(q is not None for q in q_by_h.values())
        q_score = (
            sum(q for q in q_by_h.values() if q is not None) / len(q_by_h)
            if defined and q_by_h
            else None
        )
        out[sid] = {"q": q_by_h, "n_samples": n_by_h, "Q": q_score}
    return out


def compute_panel_weights(
    weight_state: Dict[str, Any],
    *,
    obs_stamp: int,
    eligible: Sequence[str],
    dw_params: Dict[str, Any],
    fallback_weights: Dict[str, float],
    schedule: Any,
    carrying: Sequence[str] = (),
    attendance_view: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One OBSERVATION's weight_calc block for the hourly panel lanes --
    the compute_dynamic_weights contract per observation, with the A2
    transition rule and per-source attendance audit fields.

    Fallback/dynamic modes exactly as day mode: fallback pins the config
    weights RESTRICTED to the eligible set (deliberately unnormalized --
    every consumer renormalizes, so fallback-mode index math matches the
    fixed-weight formulation byte for byte); dynamic allocates
    softmax(gamma * Q) -> floor -> cap over the eligible set with
    undefined Q scoring 0.

    TRANSITION RULE: mode = dynamic iff
    prior mode dynamic OR (the attendance-passer set is NON-EMPTY AND
    every ATTENDANCE-PASSER -- attendance ratio >=
    dw_params["attendance_floor"] over the trailing history window --
    has a defined Q AND len(eligible) >= switch_min_eligible). Permanent
    latch, carried in weight_state["mode"] by the caller's advance. A
    below-floor source stays index-ELIGIBLE and simply scores 0
    post-switch (weight floor); the floor governs the switch test ONLY.
    The non-empty leg closes the vacuous-truth corner: with zero passers
    (reachable only when EVERY candidate's attendance sits below the
    floor -- a lane-wide outage's aftermath) the every-passer clause is
    vacuously true and the eligible-count leg alone would have flipped
    the PERMANENT latch on zero attendance information, inverting the A2
    intent. Genesis is structurally unaffected: the zero-scheduled window
    makes every candidate a passer (attendance's 1.0 rule) with an
    undefined Q, so the switch holds there exactly as before.

    The SCORED set = eligible UNION attendance-passers UNION
    recently-printed (>= 1 trusted print in the trailing window). Any
    passer with a positive ratio has a print in the window, and the
    genesis corner's universal passers carry no unknown ids, so scoring
    eligible + recently-printed covers every passer by construction.

    The block pins, per scored source, attendance_printed /
    attendance_scheduled / attendance_ratio (ratio at 9dp, the q/Q
    precision rule) so the switch decision replays from artifacts alone;
    switched_on pins the lane's full observation stamp key (THH on the
    hour-keyed lanes, THHMM on minute-keyed -- schedule.stamp_key). NO
    slot_prints key: every observation's own artifact records its
    trusted prints, replay ingests them per observation.

    ATTENDANCE-WEIGHTING (METHODOLOGY.md section 8.6-8.7). On a lane
    whose dw_params carry the minted knob triple this block additionally
    computes the per-source attendance view (compute_attendance_view --
    ONE decay table per observation, the same scheduled window as the
    triple) and:

      - publishes attendance_factor (9dp disclosure rounding; the
        allocator eats FULL precision) / no_price_streak /
        no_price_excluded on every sources row, over the WIDENED
        publication domain scored UNION all members (fallback_weights'
        keyspace: a seat absent longer than the history window still
        publishes its exclusion state). Widening is
        per-source-independent (a never-printed member scores None
        everywhere), so the passer / quorum sets below, which read the
        ORIGINAL scored list, are unchanged by it.
      - when ARMED (attendance_eta > 0, rule R2) tilts the softmax by
        eta*ln(A_i) and scales each seat's ceiling and floor with its
        attendance (rules D4/D6/D7 -- see allocate_weights).

    ``carrying`` -- the ARMED state-2 carry-casting seats (rule D4, the
    caller's gate): the allocation domain extends to eligible UNION
    carrying, so a quiet seat's re-cast vote fades under its own
    (current) weight row; weights sum to 1 over the full voting set. The
    A2 switch quorum still counts len(eligible) only -- carried seats
    never help a lane latch dynamic. Non-empty carrying on a lane
    without the minted knobs raises.

    ``attendance_view`` -- an optional PRECOMPUTED compute_attendance_view
    result (the scheduled_window amortization pattern):
    compute_observation computes the one view per observation and
    threads it here; it MUST be that function's output for the same
    (pre-advance state, obs_stamp, dw_params) over ids covering every
    published seat. None on minted lanes computes it internally (direct
    callers/tests); passing one on a knob-less lane raises.
    """
    obs_stamp = int(obs_stamp)
    eligible = list(eligible)
    carrying = [str(s) for s in carrying]
    minted = attendance_minted(dw_params)
    if carrying and not minted:
        raise ValueError(
            "carrying seats require the minted attendance knob triple -- "
            "the armed D4 domain cannot extend on a knob-less lane"
        )
    if attendance_view is not None and not minted:
        raise ValueError(
            "attendance_view requires the minted attendance knob triple -- "
            "a knob-less lane has no attendance law to publish under"
        )
    history_minutes = int(dw_params["history_days"]) * 1440
    window_start = obs_stamp - history_minutes
    prices = weight_state.get("prices") or {}
    scheduled_window = schedule.scheduled_stamps(window_start, obs_stamp)
    recent = sorted(
        sid
        for sid, series in prices.items()
        if any(window_start <= t < obs_stamp for t in series)
    )
    scored = list(eligible) + [sid for sid in recent if sid not in eligible]
    # Publication domain: on minted lanes every member (the
    # fallback_weights keyspace) joins the published rows; knob-less
    # lanes keep the scored set verbatim (byte parity).
    published = scored + (
        [sid for sid in sorted(fallback_weights) if sid not in scored]
        if minted
        else []
    )
    if attendance_view is None and minted:
        attendance_view = compute_attendance_view(
            weight_state,
            published,
            obs_stamp=obs_stamp,
            schedule=schedule,
            dw_params=dw_params,
            scheduled_window=scheduled_window,
        )
    attendance_by_sid = {
        sid: attendance(
            prices,
            sid,
            obs_stamp=obs_stamp,
            schedule=schedule,
            window_minutes=history_minutes,
            scheduled_window=scheduled_window,
        )
        for sid in published
    }
    attendance_floor = float(dw_params["attendance_floor"])
    passers = [
        sid
        for sid in scored
        if attendance_by_sid[sid]["ratio"] >= attendance_floor
    ]
    scores = predictive_scores_obs(
        weight_state,
        obs_stamp=obs_stamp,
        source_ids=published,
        dw_params=dw_params,
        schedule=schedule,
    )
    prior_mode = weight_state.get("mode", MODE_FALLBACK)
    min_eligible = int(dw_params.get("switch_min_eligible", 1))
    # A2 amendment (F2): bool(passers) -- zero passers must HOLD the
    # switch (the all() clause is vacuously true there; see docstring).
    quorum_met = (
        len(eligible) >= min_eligible
        and bool(passers)
        and all(scores[sid]["Q"] is not None for sid in passers)
    )
    mode = (
        MODE_DYNAMIC
        if (prior_mode == MODE_DYNAMIC or quorum_met)
        else MODE_FALLBACK
    )
    switched = mode == MODE_DYNAMIC and prior_mode != MODE_DYNAMIC
    stamp_iso = schedule.stamp_key(obs_stamp)

    flags: Dict[str, Any] = {"degenerate_allocation": None, "capped": []}
    # Rule D4: when armed, the allocation domain is eligible UNION the
    # carry-casting state-2 seats -- a quiet seat's re-cast vote fades
    # under a REAL (current) weight row instead of a frozen booked one.
    # Empty carrying (every dark and knob-less lane) leaves the domain =
    # eligible verbatim.
    domain = eligible + [sid for sid in carrying if sid not in eligible]
    eta = float(dw_params["attendance_eta"]) if minted else 0.0
    armed = eta > 0
    if not domain:
        weights: Dict[str, float] = {}
    elif mode == MODE_FALLBACK:
        # Fallback mode ignores eta entirely (config vector, byte-parity
        # doctrine); an armed carrying seat still gets its config row --
        # its re-cast vote needs a weight whichever mode the lane is in.
        weights = {sid: float(fallback_weights[sid]) for sid in domain}
    else:
        q_values = {
            sid: (scores[sid]["Q"] if scores[sid]["Q"] is not None else 0.0)
            for sid in domain
        }
        weights, flags = allocate_weights(
            q_values,
            gamma=float(dw_params["gamma"]),
            weight_min=float(dw_params["weight_min"]),
            weight_max=float(dw_params["weight_max"]),
            source_caps=dw_params.get("source_weight_caps") or {},
            # Structural gate: the attendance inputs are NOT CONSTRUCTED
            # unless armed -- eta 0 / knobs absent run the legacy
            # allocator byte-identically (the kwargs are not even
            # passed).
            **(
                {
                    "attendance_factors": {
                        sid: attendance_view[sid]["factor"]
                        for sid in domain
                    },
                    "attendance_eta": eta,
                }
                if armed
                else {}
            ),
        )
    rounded = {sid: round(w, 6) for sid, w in weights.items()}
    for sid, w in rounded.items():
        if not (math.isfinite(w) and w >= 0):
            raise ValueError(
                f"{stamp_iso}: non-finite or negative weight for {sid}: {w!r}"
            )
    block: Dict[str, Any] = {
        "scheme": dw_params["scheme"],
        "mode": mode,
        "weights": {sid: rounded[sid] for sid in sorted(rounded)},
        "sources": {
            sid: {
                "Q": _round_opt(scores[sid]["Q"]),
                "q": {
                    h: _round_opt(q)
                    for h, q in sorted(scores[sid]["q"].items())
                },
                "n_samples": dict(sorted(scores[sid]["n_samples"].items())),
                "attendance_printed": int(
                    attendance_by_sid[sid]["printed"]
                ),
                "attendance_scheduled": int(
                    attendance_by_sid[sid]["scheduled"]
                ),
                "attendance_ratio": round(
                    float(attendance_by_sid[sid]["ratio"]), 9
                ),
                # Attendance-weighting fields, minted lanes only --
                # beside the untouched attendance triple.
                # attendance_factor is DISCLOSURE rounding (9dp, the q/Q
                # rule); the allocator consumed full precision above.
                # no_price_excluded is present-but-False while dark.
                **(
                    {
                        "attendance_factor": round(
                            float(attendance_view[sid]["factor"]), 9
                        ),
                        "no_price_streak": int(
                            attendance_view[sid]["streak"]
                        ),
                        "no_price_excluded": bool(
                            attendance_view[sid]["excluded"]
                        ),
                    }
                    if attendance_view is not None
                    else {}
                ),
            }
            for sid in sorted(scores)
        },
        "degenerate_allocation": flags.get("degenerate_allocation"),
        "capped": flags.get("capped", []),
    }
    if flags.get("fallback_reason"):
        block["fallback_reason"] = flags["fallback_reason"]
    if switched:
        block["switched_on"] = stamp_iso
    return block


def advance_panel_weight_state(
    weight_state: Dict[str, Any],
    *,
    obs_stamp: int,
    prints: Dict[str, Dict[str, Any]],
    vector: Optional[Dict[str, float]],
    mode: str,
    dw_params: Dict[str, Any],
    events: Optional[Dict[str, str]] = None,
) -> None:
    """Advance the OBSERVATION-mode weight state with one published
    observation's pinned facts, then prune. ONE state machine for live
    and replay, same as day mode: the live path advances from the block
    it just computed; replay re-applies each published observation's
    facts in order and rebuilds identical state.

      - prints: {sid: {usd, native, currency}} -- the observation's own
        trusted prints (accepted AND fenced; the fence holds a print out
        of the INDEX, never out of the weight series -- R-winsor bounds
        it), each entry shaped by series_print. Stored at obs_stamp.
      - vector: the observation's pinned rounded weight vector, stored
        under state["vectors"][obs_stamp] (minute-STAMP keyed -- never mix
        with a day-mode state, whose vectors are day-ordinal keyed).
        Empty/dark observations store NO vector, same as day mode.
      - mode: the block's pinned mode; the latch is monotonic (fallback
        -> dynamic, never back).
      - events: this observation's np/sk attendance classifications
        ({sid: code}, gpu_index.index.panel.attendance_events_for_stamp)
        -- stored under weight_state["events"][sid][stamp] ONLY when the
        lane's dw_params carry the minted attendance knob triple
        (METHODOLOGY.md section 8.6: a knob-less lane's state must never
        grow the key -- the D2 dark contract). Codes are fail-closed:
        anything outside {"np", "sk"} raises.

    PRUNING (design section 5 perf): price and vector stamps STRICTLY
    older than obs_stamp - (history + max_forward + margin), all in
    MINUTES (the observation-mode stamp unit), are
    dropped. Provably score-neutral: the engine only computes at
    scheduled stamps, so every future observation's cutoff is >=
    obs_stamp (the next stamp's previous-scheduled IS obs_stamp), the
    oldest sample anchor is cutoff - history, and the oldest
    endpoint any sample touches is anchor - max(lookback) -- inside the
    kept range because PRUNE_MARGIN_MINUTES covers the max lookback
    (enforced below, so a widened lookback can never silently break the
    proof). Attendance and recently-printed windows are narrower still.
    Vectors additionally keep the NEWEST entry below the threshold:
    vector resolution is last-at-or-below tau with unbounded lookback,
    so after a long dark spell that entry still anchors in-window taus.
    A source whose series prunes empty is dropped entirely (state size
    stays bounded; an absent series and an empty one are
    indistinguishable to every reader).

    PRUNE CADENCE (hot-loop invariant, review perf stage): the full
    prune sweep walks every source's whole series, so on a dense grid
    it must not run at every observation. It runs only when the
    threshold has advanced >= one day of WALL TIME (1440 minute-stamps,
    cadence-independent) since the last sweep (``_prune_threshold``, the
    private bookkeeping key new_weight_state seeds; a state built
    without it -- hand-made test fixtures, pre-change replays -- prunes
    on its first advance and gains the key). Pruning LESS often only
    KEEPS MORE data, and everything below the strict threshold is never
    consulted by any computation (the proof above), so deferring the
    sweep is trivially score-neutral; the extra retention is bounded at
    one day of stamps. The prices series' OTHER consumers -- the dw vote
    tail (dw_vote_tail in this module, ruling 2026-08-27) and the
    attendance A_i / streak reads (compute_attendance_view) -- read no
    older than obs_stamp - the history window in minutes (A_i; the
    bounded streak window is validated <= history at load), strictly
    inside the kept range, so the deferred sweep is attendance-neutral
    as well as vote- and score-neutral. The EVENTS series prunes in this
    same sweep at the same threshold; a source whose prices prune empty
    but still holds in-range events is deliberately NOT dropped from
    events (a silent streak reset across a long absence would fake a
    recovery).
    """
    obs_stamp = int(obs_stamp)
    history_minutes = int(dw_params["history_days"]) * 1440
    max_forward = 60 * max(
        int(x) for x in dw_params["forward_horizons_hours"]
    )
    max_lookback = 60 * max(
        int(x) for x in dw_params["lookback_horizons_hours"]
    )
    if max_lookback > max_forward + PRUNE_MARGIN_MINUTES:
        raise ValueError(
            f"lookback horizon {max_lookback}min exceeds max_forward + "
            f"PRUNE_MARGIN_MINUTES ({max_forward} + {PRUNE_MARGIN_MINUTES}); "
            f"pruning would eat live feature endpoints"
        )
    attendance_lane = attendance_minted(dw_params)
    if attendance_lane:
        # Event codes validate BEFORE any mutation below: a mid-advance
        # raise would leave the prints/vector/latch written and the
        # events not -- a torn state no replay produces. The advance is
        # all-or-nothing.
        for sid in sorted(events or {}):
            code = events[sid]
            if code not in VALID_EVENT_CODES:
                raise ValueError(
                    f"unknown attendance event code {code!r} for {sid} at "
                    f"stamp {obs_stamp}"
                )
    prices = weight_state.setdefault("prices", {})
    for sid in sorted(prints or {}):
        entry = prints[sid]
        prices.setdefault(sid, {})[obs_stamp] = {
            "usd": entry["usd"],
            "native": entry["native"],
            "currency": entry["currency"],
        }
    vectors = weight_state.setdefault("vectors", {})
    if vector:
        vectors[obs_stamp] = dict(vector)
    if mode == MODE_DYNAMIC:
        weight_state["mode"] = MODE_DYNAMIC
    else:
        weight_state.setdefault("mode", MODE_FALLBACK)
    if attendance_lane:
        # Attendance events, conditional on the MINTED knob triple: a
        # knob-less lane's state must never grow the key (the D2 dark
        # contract). np/sk only; state-1 stays implicit in prices,
        # no-entry implicit in absence from both. Codes were validated
        # fail-closed in the pre-pass above (all-or-nothing).
        event_state = weight_state.setdefault("events", {})
        for sid in sorted(events or {}):
            event_state.setdefault(sid, {})[obs_stamp] = events[sid]
    threshold = obs_stamp - (
        history_minutes + max_forward + PRUNE_MARGIN_MINUTES
    )
    last_pruned = weight_state.get("_prune_threshold")
    if last_pruned is not None and threshold - int(last_pruned) < 1440:
        return  # deferred sweep (docstring: score-neutral by the proof)
    weight_state["_prune_threshold"] = threshold
    for sid in sorted(prices):
        series = prices[sid]
        for stamp in [t for t in series if t < threshold]:
            del series[stamp]
        if not series:
            del prices[sid]
    event_series = weight_state.get("events") or {}
    for sid in sorted(event_series):
        series = event_series[sid]
        for stamp in [t for t in series if t < threshold]:
            del series[stamp]
        if not series:
            del event_series[sid]
    stale_vectors = sorted(t for t in vectors if t < threshold)
    for stamp in stale_vectors[:-1]:  # keep the newest as resolution anchor
        del vectors[stamp]
