# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Dynamic predictive source weighting for the index baskets.

Implements the dynamic weighting methodology (docs/dynamic_source_weighting.md;
METHODOLOGY.md Part III) for
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
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    site."""
    return {"prices": {}, "vectors": {}, "mode": MODE_FALLBACK}


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
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Softmax allocation (METHODOLOGY.md §12) plus the risk-cap layer: softmax(gamma * Q)
    shares over the eligible set, everyone floored at weight_min, the
    remainder distributed by share, then caps applied by iteratively
    capping violators at cap_i = min(weight_max, source_caps[i]) and
    redistributing their excess to the uncapped proportionally to share.
    The capped set grows monotonically, so the loop terminates within N
    passes. Full precision throughout; the caller rounds ONCE at the end.

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
    capped: List[str] = []
    for _ in range(n):
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
            # Every source is at its cap. The mass invariant plus the
            # cap-mass gate above force sum(caps) == 1 within float dust
            # (the exactly-1.0 corner: three 0.30s +
            # a 0.10) — the honest allocation IS the caps themselves, so
            # publish them; the residual excess is dust by construction.
            flags["capped"] = sorted(capped)
            return weights, flags
        pool = sum(shares[sid] for sid in uncapped)
        if pool <= 0.0:
            # Extreme-but-finite gamma can underflow every non-max share to
            # exactly 0.0 (exp(gamma * dQ) at dQ << 0); the excess must
            # still be conserved deterministically — spread it equally over
            # the uncapped rather than darking the publish on a divide.
            for sid in uncapped:
                weights[sid] += excess / len(uncapped)
            continue
        for sid in uncapped:
            weights[sid] += excess * shares[sid] / pool
    flags["capped"] = sorted(capped)
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
