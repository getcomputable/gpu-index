# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Job-rendered HTML dashboard for an index basket lane.

Pure renderer, ZERO I/O: ``render_report(...) -> html_str``. The caller
(scripts/compute_index_composite.py --sync) fetches the inputs, calls this,
and PUTs the result to ``<prefix>/report/index.html`` — the ONE
documented mutable key under ``index/`` (everything else is append-only; see
gpu_index/common/store.py). The report is a derived convenience view rewritten every
firing; the immutable snapshots/composites remain the record. Publishing it
is warn-only at the call site: a render bug must never fail the composite
publish.

Rendering rules (dev-tooling, not index math — this module must never
compute a price):

  - Self-contained: inline CSS, hand-rolled inline-SVG charts, a small
    vanilla-JS block for the day selector and hover tooltips. No external
    assets, no chart libraries.
  - Every remote-derived string (source notes, run ids, keys, dates read
    back from the bucket) goes through html.escape — snapshot ``notes``
    carry arbitrary scraped page text.
  - Per-day calc breakdowns are fully server-rendered (one hidden panel per
    published day); the JS only toggles visibility, so no remote string is
    ever assembled into DOM client-side.
  - Values shown are read verbatim from the stored artifacts (display
    rounding only) — if the report seems to need a calc change, stop.
"""

from __future__ import annotations

import html
import json
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["render_report", "REPORT_WINDOW_DAYS"]

# How many trailing days of composites the dashboard shows. The caller
# fetches this window (date-addressed GETs; keys are deterministic).
REPORT_WINDOW_DAYS = 30

_DASH = "—"  # em dash for empty cells


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _finite(value: Any) -> Optional[float]:
    """Stored artifacts are parsed with json.loads, which accepts bare
    NaN/Infinity — a non-finite or non-numeric value must become a gap,
    never a crashed render or a NaN SVG coordinate."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _price(value: Any) -> str:
    """Display a stored price verbatim-ish: up to 6 dp, trailing zeros
    trimmed (audit tables must not invent precision the artifact lacks)."""
    if _finite(value) is None:
        return _DASH
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _idx(value: Any) -> str:
    """Index values match the CLI's 4-dp print style ("7.5340")."""
    if _finite(value) is None:
        return _DASH
    return f"{value:.4f}"


def _attr_json(value: Any) -> str:
    """JSON for an HTML attribute or script slot: ``<`` is never emitted
    raw so remote strings can't smuggle a ``</script>`` or open a tag."""
    return json.dumps(value).replace("<", "\\u003c")


def _chip(kind: str, label: str) -> str:
    """Status chip: colored dot + text label (the label carries the
    meaning; color never stands alone)."""
    return (
        f'<span class="chip"><span class="dot dot-{_esc(kind)}"></span>'
        f"{_esc(label)}</span>"
    )


# ------------------------------------------------------------------ charts


def _nice_ticks(lo: float, hi: float, target: int = 4) -> List[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    tick = math.ceil(lo / step) * step
    ticks = []
    while tick <= hi + step * 1e-6:
        ticks.append(round(tick, 10))
        tick += step
    return ticks


def _line_chart(
    days: Sequence[str],
    values: Sequence[Optional[float]],
    *,
    held_days: Sequence[str] = (),
    dark_days: Sequence[str] = (),
    null_label: str = "not published",
    width: int = 860,
    height: int = 240,
    margin_left: int = 56,
    x_ticks: bool = True,
    label_last: bool = True,
    tick_count: int = 4,
) -> str:
    """One single-series SVG line chart. ``None`` values are gaps (a dark
    or unpublished day must read as absence, never as zero). Held-out days
    get a warning marker. Hover data rides data-* attributes; the shared JS
    attaches the crosshair + tooltip."""
    points = [v for v in values if v is not None]
    if not points:
        return '<div class="empty">no published values in this window</div>'

    m_top, m_right = 14, 18
    m_bottom = 30 if x_ticks else 12
    lo, hi = min(points), max(points)
    span = (hi - lo) or max(abs(hi) * 0.02, 0.05)
    lo_p, hi_p = lo - span * 0.10, hi + span * 0.10
    plot_w = width - margin_left - m_right
    plot_h = height - m_top - m_bottom
    n = len(days)

    def x_at(i: int) -> float:
        frac = 0.5 if n == 1 else i / (n - 1)
        return round(margin_left + frac * plot_w, 2)

    def y_at(v: float) -> float:
        return round(m_top + (1 - (v - lo_p) / (hi_p - lo_p)) * plot_h, 2)

    parts: List[str] = []
    for tick in _nice_ticks(lo_p, hi_p, tick_count):
        ty = y_at(tick)
        parts.append(
            f'<line class="grid" x1="{margin_left}" y1="{ty}" '
            f'x2="{width - m_right}" y2="{ty}"/>'
            f'<text class="tick" x="{margin_left - 8}" y="{ty + 3.5}" '
            f'text-anchor="end">{_price(tick)}</text>'
        )
    if x_ticks and n > 1:
        for i in sorted({0, n // 2, n - 1}):
            parts.append(
                f'<text class="tick" x="{x_at(i)}" y="{height - 8}" '
                f'text-anchor="middle">{_esc(days[i][5:])}</text>'
            )

    segment: List[str] = []
    segments: List[str] = []
    for i, v in enumerate(values):
        if v is None:
            if len(segment) > 1:
                segments.append("M" + " L".join(segment))
            segment = []
            continue
        segment.append(f"{x_at(i)} {y_at(v)}")
    if len(segment) > 1:
        segments.append("M" + " L".join(segment))
    for d_attr in segments:
        parts.append(f'<path class="series" d="{d_attr}"/>')
    # Single-point segments (a value between two gaps) still need a mark —
    # but a held-out point keeps its amber marker (drawn below); a blue dot
    # painted on top would relabel it as accepted.
    held_set = set(held_days)
    for i, v in enumerate(values):
        if v is None or days[i] in held_set:
            continue
        prev_gap = i == 0 or values[i - 1] is None
        next_gap = i == n - 1 or values[i + 1] is None
        if prev_gap and next_gap:
            parts.append(
                f'<circle class="pt" cx="{x_at(i)}" cy="{y_at(v)}" r="4"/>'
            )

    for i, v in enumerate(values):
        if v is not None and days[i] in held_set:
            parts.append(
                f'<circle class="pt-held" cx="{x_at(i)}" cy="{y_at(v)}" '
                'r="4"/>'
            )

    last_i = max(i for i, v in enumerate(values) if v is not None)
    last_v = values[last_i]
    if days[last_i] not in held_set:
        parts.append(
            f'<circle class="pt" cx="{x_at(last_i)}" cy="{y_at(last_v)}" '
            'r="4"/>'
        )
    if label_last:
        anchor = "end" if x_at(last_i) > width - 90 else "start"
        lx = x_at(last_i) - 8 if anchor == "end" else x_at(last_i) + 8
        ly = max(y_at(last_v) - 9, 12)
        parts.append(
            f'<text class="lastval" x="{lx}" y="{ly}" '
            f'text-anchor="{anchor}">{_idx(last_v)}</text>'
        )

    parts.append(
        f'<line class="crosshair" x1="0" x2="0" y1="{m_top}" '
        f'y2="{height - m_bottom}" style="display:none"/>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" data-chart="line" '
        f"data-days=\"{_esc(_attr_json(list(days)))}\" "
        f"data-values=\"{_esc(_attr_json(list(values)))}\" "
        f"data-held=\"{_esc(_attr_json(sorted(held_set)))}\" "
        f"data-dark=\"{_esc(_attr_json(sorted(set(dark_days))))}\" "
        f"data-nulllabel=\"{_esc(null_label)}\" "
        f'data-x0="{margin_left}" data-x1="{margin_left + plot_w}">'
        + "".join(parts)
        + "</svg>"
    )


# ------------------------------------------------------------ series prep


def _window_days(composites_by_date: Dict[str, Dict[str, Any]]) -> List[str]:
    """Continuous day axis from first to last published day — gaps stay on
    the axis so time reads true."""
    if not composites_by_date:
        return []
    first = date.fromisoformat(min(composites_by_date))
    last = date.fromisoformat(max(composites_by_date))
    return [
        (first + timedelta(days=i)).isoformat()
        for i in range((last - first).days + 1)
    ]


def _index_series(
    days: Sequence[str], composites: Dict[str, Dict[str, Any]]
) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for day in days:
        index = (composites.get(day) or {}).get("index") or {}
        out.append(_finite(index.get("value_usd_gpu_hr")))
    return out


def _dark_days(
    days: Sequence[str], composites: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Days that ARE published but carry no index value (basket_dark /
    day_missed) — the tooltip must not call them 'not published'."""
    return [
        day
        for day in days
        if day in composites
        and _finite(
            ((composites[day].get("index")) or {}).get("value_usd_gpu_hr")
        )
        is None
    ]


def _source_entry(
    composite: Optional[Dict[str, Any]], source_id: str
) -> Optional[Dict[str, Any]]:
    for entry in (composite or {}).get("sources", []):
        if entry.get("source_id") == source_id:
            return entry
    return None


def _source_series(
    days: Sequence[str],
    composites: Dict[str, Dict[str, Any]],
    source_id: str,
) -> Tuple[List[Optional[float]], List[str]]:
    values: List[Optional[float]] = []
    held: List[str] = []
    for day in days:
        entry = _source_entry(composites.get(day), source_id)
        chosen = (entry or {}).get("chosen") or {}
        value = _finite(chosen.get("usd_per_gpu_hr"))
        values.append(value)
        verdict = (entry or {}).get("filter") or {}
        if value is not None and verdict and not verdict.get("accepted"):
            held.append(day)
    return values, held


def _verdict_chip(entry: Optional[Dict[str, Any]]) -> str:
    if entry is None:
        return _chip("serious", "missing")
    status = entry.get("status")
    if status != "ok":
        return _chip("serious", str(status or "missing"))
    if not entry.get("chosen"):
        # Source collected fine but nothing was eligible that day (e.g.
        # every print interruptible/implausible/quarantined) — that is not
        # a filter hold-out and must not read as one.
        return _chip("warning", "no eligible print")
    verdict = entry.get("filter") or {}
    if not verdict.get("accepted"):
        # Rule D1: currency anomalies are HELD OUT (fail-closed), window
        # preserved — the chip must say which kind, not impersonate an
        # ordinary sigma hold-out.
        if verdict.get("untrusted_currency"):
            return _chip("warning", "held out (untrusted currency)")
        if verdict.get("currency_mismatch"):
            return _chip(
                "warning",
                f"held out (currency change "
                f"{verdict.get('pending_count', '?')}/"
                f"{verdict.get('confirm_after', '?')})",
            )
        label = "held out"
        if verdict.get("sigma_zero"):
            label += " (sigma=0)"
        return _chip("warning", label)
    if verdict.get("unfiltered"):
        if verdict.get("currency_confirmed"):
            # Third consecutive same-new-currency print: change confirmed,
            # window reseeded (still warm-up). A confirmed verdict is
            # unfiltered, so compute_day's R3 loop can still flag it
            # manual_verify — the mark must not vanish here.
            label = "currency confirmed (window reseeded)"
            if verdict.get("manual_verify"):
                label += " manual-verify"
            return _chip("warning", label)
        label = f"unfiltered (n={verdict.get('n_history', '?')})"
        if verdict.get("manual_verify"):
            return _chip("warning", label + " manual-verify")
        return _chip("neutral", label)
    return _chip("good", "accepted")


# ------------------------------------------------------------- sections


def _kpi_tiles(
    latest_day: Optional[str],
    composites: Dict[str, Dict[str, Any]],
    pointer: Optional[Dict[str, Any]],
) -> str:
    tiles: List[str] = []
    latest = composites.get(latest_day) if latest_day else None
    index = (latest or {}).get("index") or {}
    value = _finite(index.get("value_usd_gpu_hr"))
    hero_value = f"${_idx(value)}" if value is not None else "DARK"

    delta_txt = ""
    published = [
        d
        for d in sorted(composites)
        if _finite((composites[d].get("index") or {}).get("value_usd_gpu_hr"))
        is not None
    ]
    if value is not None and len(published) >= 2 and published[-1] == latest_day:
        prev_day = published[-2]
        prev = composites[prev_day]["index"]["value_usd_gpu_hr"]
        sign = "+" if value >= prev else "-"
        delta_txt = f"{sign}{_idx(abs(value - prev))} vs {_esc(prev_day)}"
    # calc_v4 artifacts carry an aggregate dispersion (legacy
    # days don't, and their tiles must render byte-identical to before.
    conf = _finite(index.get("confidence_usd_gpu_hr"))
    conf_txt = f" &#183; &#177;{_idx(conf)} CI" if conf is not None else ""
    tiles.append(
        '<div class="tile hero"><div class="label">Index '
        f"(latest publish{', ' + _esc(latest_day) if latest_day else ''})"
        f'</div><div class="value">{hero_value}</div>'
        f'<div class="delta">USD per GPU-hr{conf_txt}'
        f"{' &#183; ' + delta_txt if delta_txt else ''}</div></div>"
    )

    n_constituents = len((latest or {}).get("sources", [])) or None
    used = index.get("sources_used_count")
    # Lazy import, config.py's rule: composite is the single home of the
    # statistic id. Dark days publish index null, so the artifact's own
    # calc_params is the fallback — a v4-era dark day must not be labeled
    # with the retired statistic.
    from gpu_index.index.composite import MEDIAN_STDDEV_VOTES

    statistic = index.get("statistic") or (
        ((latest or {}).get("calc_params") or {}).get("composite_statistic")
    )
    statistic_label = (
        "median of stddev votes, weighted"
        if statistic == MEDIAN_STDDEV_VOTES
        else "weighted mean, renormalized"
    )
    tiles.append(
        '<div class="tile"><div class="label">Sources used</div>'
        f'<div class="value">{_esc(used) if used is not None else _DASH}'
        f"{'/' + _esc(n_constituents) if n_constituents else ''}</div>"
        f'<div class="delta">{statistic_label}</div></div>'
    )
    tiles.append(
        '<div class="tile"><div class="label">Unweighted mean</div>'
        f'<div class="value">{"$" + _idx(index.get("unweighted_mean_usd_gpu_hr")) if index.get("unweighted_mean_usd_gpu_hr") is not None else _DASH}</div>'
        '<div class="delta">same passing set</div></div>'
    )
    # calc_v5 (dynamic weighting): the day's weighting posture — present
    # only when the artifact carries a weight_calc block, so legacy series
    # render byte-identically. The switch day, a degenerate allocation, and
    # quorum blockers must be a dashboard read, not a bucket dig.
    weight_calc = (latest or {}).get("weight_calc") or {}
    if weight_calc:
        weighting_notes = []
        if weight_calc.get("switched_on"):
            weighting_notes.append(
                "switched " + str(weight_calc["switched_on"])
            )
        if weight_calc.get("degenerate_allocation"):
            weighting_notes.append(
                str(weight_calc["degenerate_allocation"]) + " (degenerate)"
            )
        if weight_calc.get("capped"):
            weighting_notes.append(
                "capped: " + ", ".join(weight_calc["capped"])
            )
        pending = sorted(
            sid
            for sid, entry in (weight_calc.get("sources") or {}).items()
            if entry.get("Q") is None
        )
        if weight_calc.get("mode") == "fallback" and pending:
            weighting_notes.append("switch pending: " + ", ".join(pending))
        tiles.append(
            '<div class="tile"><div class="label">Weighting</div>'
            f'<div class="value">{_esc(weight_calc.get("mode") or _DASH)}</div>'
            f'<div class="delta">{" &middot; ".join(_esc(n) for n in weighting_notes) if weighting_notes else "dynamic predictive weighting"}</div></div>'
        )
    pool = (latest or {}).get("fallback_pool") or {}
    pool_mean = pool.get("mean_usd_gpu_hr")
    pool_n = sum(1 for s in pool.get("sources", []) if s.get("chosen"))
    tiles.append(
        '<div class="tile"><div class="label">B200 fallback pool</div>'
        f'<div class="value">{"$" + _idx(pool_mean) if pool_mean is not None else _DASH}</div>'
        f'<div class="delta">simple mean &#183; {pool_n} sources</div></div>'
    )
    if pointer:
        ok = len(pointer.get("basket_sources_ok") or [])
        slot = pointer.get("slot_hour_utc")
        slot_txt = f"slot {int(slot):02d}Z" if slot is not None else _DASH
        sub = f"{_esc(pointer.get('capture_date'))} &#183; {ok}/8 basket ok"
        if pointer.get("late_fill"):
            sub += " &#183; late fill"
        tiles.append(
            '<div class="tile"><div class="label">Last capture</div>'
            f'<div class="value">{_esc(slot_txt)}</div>'
            f'<div class="delta">{sub}</div></div>'
        )
    return '<section class="kpis">' + "".join(tiles) + "</section>"


def _header_chips(
    latest: Optional[Dict[str, Any]],
    pointer: Optional[Dict[str, Any]],
    now: datetime,
) -> str:
    chips: List[str] = []
    if latest is None:
        chips.append(_chip("serious", "no composite published yet"))
    elif latest.get("basket_dark"):
        chips.append(
            _chip(
                "critical",
                "basket dark"
                + (" (day missed)" if latest.get("day_missed") else ""),
            )
        )
    else:
        chips.append(_chip("good", "publishing"))
    if latest is not None and latest.get("snapshot_late_fill"):
        chips.append(_chip("warning", "latest day was a late fill"))
    if latest is not None and latest.get("substituted_from_slot") is not None:
        chips.append(
            _chip(
                "warning",
                f"substituted from slot "
                f"{int(latest['substituted_from_slot']):02d}Z",
            )
        )
    if pointer and pointer.get("published_at"):
        try:
            beat = datetime.fromisoformat(
                str(pointer["published_at"]).replace("Z", "+00:00")
            )
            age_h = (now - beat).total_seconds() / 3600.0
            kind = "warning" if age_h > 26 else "neutral"
            chips.append(_chip(kind, f"capture heartbeat {age_h:.1f}h ago"))
        except (ValueError, TypeError):
            # TypeError: a hand-healed pointer can carry a NAIVE timestamp,
            # which fromisoformat accepts but aware-minus-naive rejects.
            chips.append(_chip("warning", "capture heartbeat unparseable"))
    return '<div class="chips">' + "".join(chips) + "</div>"


def _index_table(
    days: Sequence[str], composites: Dict[str, Dict[str, Any]]
) -> str:
    rows = []
    for day in reversed(list(days)):
        comp = composites.get(day)
        if comp is None:
            rows.append(
                f"<tr><td>{_esc(day)}</td><td class='num'>{_DASH}</td>"
                f"<td class='num'>{_DASH}</td>"
                f"<td class='num'>{_DASH}</td><td class='num'>{_DASH}</td>"
                "<td>not published</td></tr>"
            )
            continue
        index = comp.get("index") or {}
        flags = []
        if comp.get("basket_dark"):
            flags.append("basket_dark")
        if comp.get("day_missed"):
            flags.append("day_missed")
        if comp.get("snapshot_late_fill"):
            flags.append("late_fill")
        if comp.get("substituted_from_slot") is not None:
            flags.append(f"slot{int(comp['substituted_from_slot']):02d}")
        rows.append(
            f"<tr><td>{_esc(day)}</td>"
            f"<td class='num'>{_idx(index.get('value_usd_gpu_hr'))}</td>"
            f"<td class='num'>{_idx(index.get('confidence_usd_gpu_hr'))}</td>"
            f"<td class='num'>{_idx(index.get('unweighted_mean_usd_gpu_hr'))}</td>"
            f"<td class='num'>{_esc(index.get('sources_used_count', _DASH))}</td>"
            f"<td>{_esc(', '.join(flags)) if flags else ''}</td></tr>"
        )
    return (
        "<table><thead><tr><th>date</th><th class='num'>index $/GPU-hr</th>"
        "<th class='num'>&#177;CI</th>"
        "<th class='num'>unweighted</th><th class='num'>sources</th>"
        "<th>flags</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _source_panels(
    days: Sequence[str],
    composites: Dict[str, Dict[str, Any]],
    latest_day: Optional[str],
) -> str:
    latest = composites.get(latest_day) if latest_day else None
    if latest is None:
        return '<div class="empty">no published composites yet</div>'
    panels = []
    for entry in latest.get("sources", []):
        source_id = entry.get("source_id", "?")
        values, held = _source_series(days, composites, source_id)
        latest_value = next(
            (v for v in reversed(values) if v is not None), None
        )
        chart = _line_chart(
            days,
            values,
            held_days=held,
            null_label="no print recorded",
            width=270,
            height=96,
            margin_left=46,
            x_ticks=False,
            label_last=False,
            tick_count=2,
        )
        panels.append(
            '<div class="panel"><div class="panel-head">'
            f'<span class="pname">{_esc(source_id)}</span>'
            f'<span class="pval">'
            f"{'$' + _price(latest_value) if latest_value is not None else _DASH}"
            f"</span>{_verdict_chip(entry)}</div>{chart}</div>"
        )
    return '<div class="grid">' + "".join(panels) + "</div>"


def _fx_cell(chosen: Dict[str, Any]) -> str:
    if chosen.get("fx_rate") is None:
        return _DASH
    native = _price(chosen.get("native_per_gpu_hr"))
    cur = _esc(chosen.get("currency"))
    return (
        f"{native} {cur} @ {_price(chosen.get('fx_rate'))} "
        f"(as of {_esc(chosen.get('fx_as_of'))})"
    )


def _filter_cell(verdict: Optional[Dict[str, Any]]) -> str:
    if not verdict:
        return _DASH
    # Rule D1 currency verdicts carry no mu/sigma (the test never ran) —
    # the cell states what happened and that the window survived.
    if verdict.get("untrusted_currency"):
        return (
            f"untrusted currency label {_esc(verdict.get('currency_label'))} "
            "&#183; held out fail-closed &#183; window preserved"
        )
    if verdict.get("currency_mismatch"):
        return (
            f"currency changed {_esc(verdict.get('window_currency'))} "
            f"&#8594; {_esc(verdict.get('currency'))} &#183; held out "
            f"{_esc(verdict.get('pending_count', '?'))}/"
            f"{_esc(verdict.get('confirm_after', '?'))} &#183; "
            "window preserved"
        )
    if verdict.get("unfiltered"):
        if verdict.get("currency_confirmed"):
            cell = (
                f"currency confirmed {_esc(verdict.get('window_currency'))} "
                f"&#8594; {_esc(verdict.get('currency'))} &#183; window "
                f"reseeded, n={_esc(verdict.get('n_history', '?'))}"
            )
            # Confirmed verdicts are unfiltered, so the R3 manual_verify
            # flag can land on them too — it must not vanish from the cell.
            if verdict.get("manual_verify"):
                cell += " &#183; manual-verify"
            return cell
        return f"warm-up, n={_esc(verdict.get('n_history', '?'))}"
    bits = (
        f"mu {_price(verdict.get('mu'))} &#183; "
        f"sigma {_price(verdict.get('sigma'))} &#183; "
        f"dev {_price(verdict.get('deviation'))} &#183; "
        f"n {_esc(verdict.get('n_history', '?'))}"
    )
    # calc_v3 verdicts also record the explicit fence the print
    # was judged against and the filter's operating currency — shown
    # verbatim; currency labeled only when
    # non-USD, the exception worth flagging.
    if verdict.get("lo") is not None and verdict.get("hi") is not None:
        bits += (
            f" &#183; fence [{_price(verdict.get('lo'))}, "
            f"{_price(verdict.get('hi'))}]"
        )
    currency = verdict.get("currency")
    if currency and currency != "USD":
        bits += f" &#183; in {_esc(currency)}"
    return bits


def _vote_suffix(vote: Optional[Dict[str, Any]]) -> str:
    """calc_v4 vote CI for a passing source — appended to the
    filter cell so legacy days (no ``vote`` block) render byte-identical."""
    if not vote:
        return ""
    suffix = (
        f" &#183; vote &#177;{_price(vote.get('conf_usd_gpu_hr'))}"
        f" (sigma {_price(vote.get('sigma'))}"
    )
    if vote.get("sigma_floored"):
        suffix += ", floored"
    return suffix + ")"


def _day_breakdown(day: str, comp: Dict[str, Any], *, visible: bool) -> str:
    index = comp.get("index") or {}
    weights = index.get("renormalized_weights") or {}
    flags = []
    if comp.get("basket_dark"):
        flags.append(_chip("critical", "basket dark"))
    if comp.get("day_missed"):
        flags.append(_chip("critical", "day missed (no snapshot)"))
    if comp.get("snapshot_late_fill"):
        flags.append(_chip("warning", "late fill"))
    if comp.get("substituted_from_slot") is not None:
        flags.append(
            _chip(
                "warning",
                f"substituted from slot {int(comp['substituted_from_slot']):02d}Z",
            )
        )
    if not flags:
        flags.append(_chip("good", "canonical publish"))
    # calc_v4 days carry an aggregate dispersion + vote band;
    # legacy days render byte-identical to before (segments only-if-present).
    # _finite, not is-not-None: the hero tile's guard, so the two can never
    # disagree on a poisoned stored value (json.loads accepts bare NaN).
    conf_seg = ""
    day_conf = _finite(index.get("confidence_usd_gpu_hr"))
    if day_conf is not None:
        conf_seg = (
            f" &#177; {_idx(day_conf)} "
            f"(votes [{_idx(index.get('vote_p25_usd_gpu_hr'))}, "
            f"{_idx(index.get('vote_p75_usd_gpu_hr'))}])"
        )
    summary = (
        '<div class="day-summary">'
        f"<strong>{_esc(day)}</strong> &#183; index "
        f"<strong>{'$' + _idx(index.get('value_usd_gpu_hr')) if index.get('value_usd_gpu_hr') is not None else 'DARK'}</strong>"
        f"{conf_seg}"
        f" &#183; unweighted {_idx(index.get('unweighted_mean_usd_gpu_hr'))}"
        f" &#183; {_esc(index.get('sources_used_count', 0))} sources"
        f" &#183; snapshot run {_esc(comp.get('snapshot_run_id') or _DASH)}"
        f'<span class="chips inline">{"".join(flags)}</span></div>'
    )

    rows = []
    for entry in comp.get("sources", []):
        source_id = entry.get("source_id", "?")
        chosen = entry.get("chosen") or {}
        verdict = entry.get("filter")
        renorm = weights.get(source_id)
        rows.append(
            f"<tr><td>{_esc(source_id)}</td>"
            f"<td class='num'>{_price(entry.get('weight'))}</td>"
            f"<td class='num'>{_price(renorm) if renorm is not None else _DASH}</td>"
            f"<td class='num'>{_price(chosen.get('usd_per_gpu_hr'))}</td>"
            f"<td>{_fx_cell(chosen) if chosen else _DASH}</td>"
            f"<td>{_esc(chosen.get('tier', _DASH))}</td>"
            f"<td class='num'>{_esc(chosen.get('gpu_count_basis', _DASH))}</td>"
            f"<td class='num'>{_esc(chosen.get('n_eligible_prints', _DASH))}</td>"
            f"<td>{_verdict_chip(entry)}</td>"
            f"<td>{_filter_cell(verdict)}{_vote_suffix(entry.get('vote'))}</td></tr>"
        )
    constituents = (
        "<table><thead><tr><th>source</th><th class='num'>weight</th>"
        "<th class='num'>renorm</th><th class='num'>$/GPU-hr</th>"
        "<th>fx</th><th>tier</th><th class='num'>basis</th>"
        "<th class='num'>prints</th><th>verdict</th><th>filter</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    pool = comp.get("fallback_pool") or {}
    pool_rows = []
    for entry in pool.get("sources", []):
        chosen = entry.get("chosen")
        pool_rows.append(
            f"<tr><td>{_esc(entry.get('source_id', '?'))}</td>"
            f"<td class='num'>{_price((chosen or {}).get('usd_per_gpu_hr'))}</td>"
            f"<td>{_esc(entry.get('status', 'ok' if chosen else _DASH))}</td>"
            "</tr>"
        )
    pool_table = (
        "<table class='pool'><thead><tr><th>pool source (B200)</th>"
        "<th class='num'>$/GPU-hr</th><th>status</th></tr></thead><tbody>"
        + "".join(pool_rows)
        + f"</tbody></table><div class='sub'>pool mean: "
        f"{'$' + _price(pool.get('mean_usd_gpu_hr')) if pool.get('mean_usd_gpu_hr') is not None else _DASH}</div>"
    )

    params = comp.get("calc_params") or {}
    params_line = (
        f"methodology {_esc(comp.get('methodology_id'))} &#183; "
        f"window {_esc(params.get('filter_window'))} &#183; "
        f"sigma {_esc(params.get('filter_sigma'))} &#183; "
        f"warmup {_esc(params.get('filter_warmup'))} &#183; "
        f"manual-verify {_esc(params.get('manual_verify_pct'))}% &#183; "
        f"fx staleness {_esc(params.get('fx_max_staleness_days'))}d &#183; "
        f"tie-break {_esc(params.get('promote_tie_break'))}"
    )
    # calc_v3 knobs are conditional in calc_params (frozen legacy artifacts
    # never carry them) — segments appear only when present so a legacy
    # day's footer renders byte-identical to before.
    if "filter_terms" in params:
        params_line += f" &#183; terms {_esc(params.get('filter_terms'))}"
    if "filter_sigma_floor" in params:
        params_line += (
            f" &#183; sigma-floor {_esc(params.get('filter_sigma_floor'))}"
        )
    if "composite_statistic" in params:
        params_line += (
            f" &#183; statistic {_esc(params.get('composite_statistic'))}"
        )

    # Non-selected panels are hidden server-side so the pre-JS (and no-JS)
    # document shows exactly one breakdown; the selector JS only toggles.
    style = "" if visible else ' style="display:none"'
    return (
        f'<div class="card day-panel" data-day-panel="{_esc(day)}"{style}>'
        + summary
        + constituents
        + pool_table
        + f'<div class="sub params">{params_line}</div></div>'
    )


def _day_selector(
    composites: Dict[str, Dict[str, Any]], latest_day: Optional[str]
) -> str:
    if not composites:
        return '<div class="empty">no published composites yet</div>'
    options = []
    for day in sorted(composites, reverse=True):
        index = (composites[day].get("index")) or {}
        value = _finite(index.get("value_usd_gpu_hr"))
        label = (
            f"{day} — ${_idx(value)} "
            f"({index.get('sources_used_count', 0)} sources)"
            if value is not None
            else f"{day} — dark"
        )
        selected = " selected" if day == latest_day else ""
        options.append(
            f'<option value="{_esc(day)}"{selected}>{_esc(label)}</option>'
        )
    panels = "".join(
        _day_breakdown(day, composites[day], visible=day == latest_day)
        for day in sorted(composites, reverse=True)
    )
    return (
        '<div class="selector-row"><label for="day-select">Published day'
        '</label> <select id="day-select">' + "".join(options) + "</select>"
        "</div>" + panels
    )


def _snapshot_section(
    latest_snapshot: Optional[Dict[str, Any]],
    pointer: Optional[Dict[str, Any]],
) -> str:
    if latest_snapshot is None:
        return '<div class="empty">latest snapshot unavailable</div>'
    meta_bits = []
    if pointer:
        meta_bits.append(f"key {_esc(pointer.get('snapshot_key'))}")
    meta_bits.append(f"run {_esc(latest_snapshot.get('run_id'))}")
    if latest_snapshot.get("captured_at"):
        meta_bits.append(f"captured {_esc(latest_snapshot['captured_at'])}")
    if latest_snapshot.get("late_fill"):
        meta_bits.append("late fill")
    rows = []
    for entry in latest_snapshot.get("sources", []):
        source_id = entry.get("source_id", "?")
        observations = entry.get("observations") or []
        if not observations:
            rows.append(
                f"<tr><td>{_esc(source_id)}</td>"
                f"<td colspan='7'>status: {_esc(entry.get('status'))}</td></tr>"
            )
            continue
        for obs in observations:
            usd = obs.get("price_usd_gpu_hr")
            native = obs.get("price_native_per_gpu_hr")
            currency = obs.get("currency", "USD")
            if usd is not None:
                price = f"${_price(usd)}"
            elif native is not None:
                # EUR shown unconverted — conversion is the calc's job.
                price = f"{_price(native)} {_esc(currency)} (unconverted)"
            else:
                price = _DASH
            notes = obs.get("notes") or ""
            flags = "implausible" if obs.get("implausible") else ""
            rows.append(
                f"<tr><td>{_esc(source_id)}</td><td>{_esc(obs.get('sku'))}</td>"
                f"<td>{_esc(obs.get('tier'))}</td><td class='num'>{price}</td>"
                f"<td class='num'>{_esc(obs.get('gpu_count_basis', _DASH))}</td>"
                f"<td>{_esc(obs.get('region') or _DASH)}</td>"
                f"<td>{_esc(flags)}</td><td class='notes'>{_esc(notes)}</td></tr>"
            )
    table = (
        "<table><thead><tr><th>source</th><th>sku</th><th>tier</th>"
        "<th class='num'>price</th><th class='num'>basis</th><th>region</th>"
        "<th>flags</th><th>notes</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return (
        f'<div class="sub">{" &#183; ".join(meta_bits)}</div>' + table
    )


# ---------------------------------------------------------------- assembly

_CSS = """
:root{
  color-scheme:light dark;
  --page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;
  --text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;
  --baseline:#c3c2b7;--border:rgba(11,11,11,0.10);--series-1:#2a78d6;
  --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
  --neutral:#898781;
}
@media (prefers-color-scheme: dark){
  :root{
    --page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#ffffff;
    --text-secondary:#c3c2b7;--grid:#2c2c2a;--baseline:#383835;
    --border:rgba(255,255,255,0.10);--series-1:#3987e5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{max-width:940px;margin:0 auto;padding:24px 20px 48px}
h1{font-size:20px;margin:0}
h2{font-size:15px;margin:30px 0 8px}
.sub{color:var(--text-secondary);font-size:12.5px;margin:2px 0 6px}
.chips{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.chips.inline{display:inline-flex;margin:0 0 0 10px;vertical-align:middle}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;
  color:var(--text-secondary);border:1px solid var(--border);
  border-radius:999px;padding:1px 9px;background:var(--surface-1);
  white-space:nowrap}
.chip .dot{width:9px;height:9px;border-radius:50%;flex:none}
.dot-good{background:var(--good)} .dot-warning{background:var(--warning)}
.dot-serious{background:var(--serious)} .dot-critical{background:var(--critical)}
.dot-neutral{background:var(--neutral)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:10px;margin-top:18px}
.tile{background:var(--surface-1);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px}
.tile .label{font-size:12px;color:var(--text-secondary)}
.tile .value{font-size:22px;font-weight:600;margin-top:2px}
.tile.hero .value{font-size:46px;font-weight:650;line-height:1.15}
.tile .delta{font-size:12px;color:var(--text-secondary);margin-top:2px}
.card{background:var(--surface-1);border:1px solid var(--border);
  border-radius:10px;padding:14px;margin-top:10px}
svg{max-width:100%;height:auto;display:block}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .tick{fill:var(--muted);font-size:11px}
svg .series{fill:none;stroke:var(--series-1);stroke-width:2;
  stroke-linejoin:round;stroke-linecap:round}
svg .pt{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}
svg .pt-held{fill:var(--warning);stroke:var(--surface-1);stroke-width:2}
svg .lastval{fill:var(--text-primary);font-size:12px;font-weight:600}
svg .crosshair{stroke:var(--baseline);stroke-width:1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
  gap:10px}
.panel{background:var(--surface-1);border:1px solid var(--border);
  border-radius:10px;padding:10px 12px}
.panel-head{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.pname{font-weight:600;font-size:13px}
.pval{color:var(--text-secondary);font-size:12.5px;
  font-variant-numeric:tabular-nums;margin-left:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px}
th{text-align:left;color:var(--text-secondary);font-weight:500}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);
  vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.notes{color:var(--text-secondary);font-size:11.5px;max-width:260px;
  overflow-wrap:anywhere}
table.pool{max-width:420px}
.selector-row{margin:8px 0}
.selector-row label{color:var(--text-secondary);font-size:12.5px;
  margin-right:6px}
select{font:inherit;background:var(--surface-1);color:var(--text-primary);
  border:1px solid var(--border);border-radius:6px;padding:4px 8px}
.day-summary{font-size:13px}
.params{margin-top:8px}
.empty{color:var(--muted);font-size:13px;padding:18px 0}
details summary{cursor:pointer;color:var(--text-secondary);
  font-size:12.5px;margin:8px 0 4px}
footer{margin-top:36px;color:var(--muted);font-size:11.5px;
  line-height:1.6}
#tip{position:fixed;display:none;background:var(--surface-1);
  border:1px solid var(--border);border-radius:8px;padding:6px 10px;
  font-size:12px;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.15);
  z-index:10;max-width:260px}
#tip .tip-val{font-weight:600;color:var(--text-primary)}
#tip .tip-day{color:var(--text-secondary)}
"""

_JS = """
(function () {
  var tip = document.getElementById('tip');
  var tipVal = tip.querySelector('.tip-val');
  var tipDay = tip.querySelector('.tip-day');
  var charts = document.querySelectorAll('svg[data-chart]');
  charts.forEach(function (svg) {
    var days, values, held, dark;
    try {
      days = JSON.parse(svg.dataset.days);
      values = JSON.parse(svg.dataset.values);
      held = JSON.parse(svg.dataset.held || '[]');
      dark = JSON.parse(svg.dataset.dark || '[]');
    } catch (e) { return; }
    var nullLabel = svg.dataset.nulllabel || 'not published';
    var x0 = +svg.dataset.x0, x1 = +svg.dataset.x1;
    var cross = svg.querySelector('.crosshair');
    if (!days.length || !cross) return;
    svg.addEventListener('pointermove', function (evt) {
      var rect = svg.getBoundingClientRect();
      var scale = rect.width / svg.viewBox.baseVal.width;
      var x = (evt.clientX - rect.left) / scale;
      var i = days.length === 1 ? 0
        : Math.round(((x - x0) / (x1 - x0)) * (days.length - 1));
      i = Math.max(0, Math.min(days.length - 1, i));
      var px = days.length === 1 ? (x0 + x1) / 2
        : x0 + (i / (days.length - 1)) * (x1 - x0);
      cross.setAttribute('x1', px);
      cross.setAttribute('x2', px);
      cross.style.display = '';
      var v = values[i];
      tipVal.textContent = v == null
        ? (dark.indexOf(days[i]) >= 0 ? 'published dark (no index)'
          : nullLabel)
        : '$' + v.toFixed(4) + ' / GPU-hr';
      tipDay.textContent = days[i]
        + (held.indexOf(days[i]) >= 0 ? ' \\u2014 held out by filter' : '');
      tip.style.display = 'block';
      tip.style.left = Math.min(evt.clientX + 14,
        window.innerWidth - tip.offsetWidth - 8) + 'px';
      tip.style.top = (evt.clientY + 14) + 'px';
    });
    svg.addEventListener('pointerleave', function () {
      cross.style.display = 'none';
      tip.style.display = 'none';
    });
  });
  var sel = document.getElementById('day-select');
  if (sel) {
    var apply = function () {
      document.querySelectorAll('[data-day-panel]').forEach(function (el) {
        el.style.display =
          el.getAttribute('data-day-panel') === sel.value ? '' : 'none';
      });
    };
    sel.addEventListener('change', apply);
    apply();
  }
})();
"""


def render_report(
    *,
    pointer: Optional[Dict[str, Any]],
    latest_snapshot: Optional[Dict[str, Any]],
    composites_by_date: Dict[str, Dict[str, Any]],
    now: datetime,
    basket_label: str = "B300 index basket",
) -> str:
    """Assemble the full self-contained dashboard HTML.

    ``basket_label`` titles the page per lane; the default keeps
    the B300 lane's rendered bytes exactly as before.

    ``composites_by_date`` maps ISO date -> stored composite payload for the
    report window (missing days = not yet published). ``pointer`` /
    ``latest_snapshot`` may be None early in the lane's life — the report
    renders whatever exists.
    """
    days = _window_days(composites_by_date)
    latest_day = max(composites_by_date) if composites_by_date else None
    latest = composites_by_date.get(latest_day) if latest_day else None
    methodology = (latest or {}).get("methodology_id", "—")

    index_values = _index_series(days, composites_by_date)
    if days and any(v is not None for v in index_values):
        index_chart = _line_chart(
            days, index_values, dark_days=_dark_days(days, composites_by_date)
        )
        index_section = (
            index_chart
            + "<details><summary>Table view</summary>"
            + _index_table(days, composites_by_date)
            + "</details>"
        )
    elif days:
        index_section = (
            '<div class="empty">every day in the window is dark</div>'
            + _index_table(days, composites_by_date)
        )
    else:
        index_section = '<div class="empty">no published composites yet</div>'

    window_txt = (
        f"{_esc(days[0])} to {_esc(days[-1])}" if days else "empty window"
    )
    generated = now.strftime("%Y-%m-%d %H:%M:%SZ")

    doc = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="robots" content="noindex">',
        f"<title>{_esc(basket_label)} — {_esc(latest_day or 'no data')}</title>",
        f"<style>{_CSS}</style></head><body>",
        '<div class="viz-root">',
        f"<header><h1>{_esc(basket_label)}</h1>",
        f'<div class="sub">ops dashboard &#183; methodology '
        f"{_esc(methodology)} &#183; window {window_txt}</div>",
        _header_chips(latest, pointer, now),
        "</header>",
        _kpi_tiles(latest_day, composites_by_date, pointer),
        "<section><h2>Index history</h2>",
        index_section,
        "</section>",
        "<section><h2>Per-source daily prints</h2>",
        '<div class="sub">chosen (lowest eligible) USD per GPU-hr per '
        "published day — same window as above, y-scale per panel; "
        "amber points were held out by the filter</div>",
        _source_panels(days, composites_by_date, latest_day),
        "</section>",
        "<section><h2>Calc breakdown by day</h2>",
        _day_selector(composites_by_date, latest_day),
        "</section>",
        "<section><h2>Latest capture snapshot</h2>",
        _snapshot_section(latest_snapshot, pointer),
        "</section>",
        f"<footer>generated {_esc(generated)} UTC &#183; snapshot run "
        f"{_esc((latest_snapshot or {}).get('run_id') or _DASH)} &#183; "
        "report is rewritten in place every firing (the one mutable key "
        "under index/)</footer>",
        "</div>",
        '<div id="tip"><div class="tip-val"></div>'
        '<div class="tip-day"></div></div>',
        f"<script>{_JS}</script>",
        "</body></html>",
    ]
    return "".join(doc)
