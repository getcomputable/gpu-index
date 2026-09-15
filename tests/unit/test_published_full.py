# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Raw-only full reproduction of published observations."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from gpu_index.index.weights import EVENT_NO_PRICE, EVENT_SKIP
from gpu_index.published.full import (
    FullReproductionRefusal,
    VERDICT_MATCH,
    public_attendance_events,
    public_weight_print,
    read_full_history,
    reproduce_full_history,
    reproduce_published_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _params():
    return {
        "aggregation": "median_ci_votes",
        "iqm_alpha": 0.16666,
        "min_sources_to_publish": 5,
        "manual_exclusions": [],
        "members": [
            {"source_id": f"s{i}", "opening_weight": 0.2}
            for i in range(5)
        ],
        "liveness": {
            "scheme": "predictive_v1",
            "lookback_horizons_hours": [1],
            "forward_horizons_hours": [1],
            "history_days": 1,
            "half_life_days": 1,
            "ridge_lambda": 1.0,
            "gamma": 4.0,
            "weight_min": 0.025,
            "weight_max": 0.3,
            "min_train_samples": 10,
            "target_variance_floor": 1e-12,
            "switch_min_eligible": 5,
            "max_abs_log_return": 0.5,
            "attendance_floor": 0.5,
            "attendance_half_life_hours": 6,
            "attendance_eta": 0.5,
            "no_price_exclusion_hours": 24,
        },
    }


def _observation():
    return {
        "kind": "gpu_price_index_observation",
        "sku": "H100",
        "methodology_id": "h100_sxm_v1_calc_v8",
        "observed_at": "2026-09-01T00:00:00.000Z",
        "status": "ok",
        "reason": None,
        "calc_params": _params(),
        "value_usd_gpu_hr": 3.0,
        "stability_band_usd_gpu_hr": 1.1,
        "receipts": [
            {
                "source_id": f"s{i}",
                "upstream_status": "ok",
                "status": "ok",
                "carry_basis": None,
                "filter_verdict": "accepted",
                "price_disclosure": "published",
                "price": float(i + 1),
                "sd": 0.1,
                "currency": "USD",
                "fx_rate": None,
                # Forbidden as derivation inputs. Deliberately nonsense.
                "weight": 0.2,
                "liveness_score": None,
                "attendance_factor": 1.0,
            }
            for i in range(5)
        ],
    }


def test_full_reproduction_never_consumes_published_derived_intermediates():
    observation = _observation()
    first = reproduce_full_history([observation], target_date="2026-09-01")

    tampered = copy.deepcopy(observation)
    for index, receipt in enumerate(tampered["receipts"]):
        receipt["weight"] = 10_000 + index
        receipt["liveness_score"] = -10_000 - index
        receipt["attendance_factor"] = index / 10
    second = reproduce_full_history([tampered], target_date="2026-09-01")

    assert len(first.checks) == 1
    check = first.checks[0]
    assert check.verdict == VERDICT_MATCH
    assert check.derived_value == 3.0
    assert check.derived_band == 1.1
    assert check.derived_weights == {f"s{i}": 0.2 for i in range(5)}
    tampered_check = second.checks[0]
    assert tampered_check.derived_value == check.derived_value
    assert tampered_check.derived_band == check.derived_band
    assert tampered_check.derived_weights == check.derived_weights
    assert tampered_check.first_divergence.quantity == "attendance"
    assert tampered_check.first_divergence.source_id == "s0"


@pytest.mark.parametrize("carry_basis", ["no_price", None])
def test_full_reproduction_rebuilds_carried_votes_from_prior_raw_bytes(
    carry_basis,
):
    first = _observation()
    second = copy.deepcopy(first)
    second["observed_at"] = "2026-09-01T00:15:00.000Z"
    carried = second["receipts"][2]
    carried.update(
        {
            "upstream_status": "carried",
            "carry_basis": carry_basis,
            # Published carry outputs are derived intermediates. Corrupting
            # them must not change a raw-only reconstruction.
            "price": 999.0,
            "sd": 999.0,
            "weight": 0.2,
        }
    )

    result = reproduce_full_history(
        [first, second], target_date="2026-09-01"
    )

    assert [check.verdict for check in result.checks] == [
        VERDICT_MATCH,
        VERDICT_MATCH,
    ]
    assert result.checks[1].derived_value == 3.0
    assert result.checks[1].derived_band == 1.1


def test_public_attendance_classifier_preserves_the_engine_event_table():
    observation = _observation()
    observation["receipts"] = [
        {
            "source_id": "present_but_filtered",
            "upstream_status": "ok",
            "price": 1.0,
            "filter_verdict": "rejected",
        },
        {
            "source_id": "no_usable_price",
            "upstream_status": "ok",
            "price": None,
            "filter_verdict": "not_evaluated",
        },
        {
            "source_id": "provider_carry",
            "upstream_status": "carried",
            "carry_basis": "no_price",
        },
        {
            "source_id": "collector_carry",
            "upstream_status": "carried",
            "carry_basis": None,
        },
        {
            "source_id": "collector_error",
            "upstream_status": "error",
        },
        {
            "source_id": "ramp_in",
            "upstream_status": "missing",
        },
    ]

    assert public_attendance_events(observation) == {
        "no_usable_price": EVENT_NO_PRICE,
        "provider_carry": EVENT_NO_PRICE,
        "collector_carry": EVENT_SKIP,
        "collector_error": EVENT_SKIP,
    }


def test_public_top_level_outage_flag_skips_every_seat():
    observation = _observation()
    observation["reason"] = "observation_missed"

    assert public_attendance_events(observation) == {
        f"s{i}": EVENT_SKIP for i in range(5)
    }


def test_public_weight_print_restores_recorded_currency_terms():
    assert public_weight_print(
        {
            "source_id": "eu-cloud",
            "price": 2.3286,
            "currency": "EUR",
            "fx_rate": 1.1643,
        },
        observed_at="2026-09-01T00:00:00.000Z",
    ) == {"usd": 2.3286, "native": 2.0, "currency": "EUR"}
    assert public_weight_print(
        {
            "source_id": "rounded-eu-cloud",
            "price": 3.686523,
            "currency": "EUR",
            "fx_rate": 1.1643,
        },
        observed_at="2026-09-01T00:00:00.000Z",
    )["native"] == 3.1663


def test_full_reproduction_typed_refusal_names_missing_status_history():
    observation = _observation()
    del observation["receipts"][2]["upstream_status"]

    with pytest.raises(FullReproductionRefusal) as caught:
        reproduce_full_history([observation], target_date="2026-09-01")

    assert caught.value.code == "missing_upstream_status"
    assert "s2" in str(caught.value)


@pytest.mark.parametrize("version", [None, 5, 6])
def test_history_loader_uses_the_public_series_origin_and_verified_days(version):
    observation = _observation()

    class Reader:
        def read_series(self, _range, *, sku, **kwargs):
            assert kwargs == ({"version": version} if version is not None else {})
            assert (_range, sku) == ("90d", "H100")
            return {
                "meta": {"from_observed_at": observation["observed_at"]},
                "data": {
                    "observations": [
                        {"observed_at": observation["observed_at"]}
                    ]
                },
            }

        def read_day(self, date, *, sku, **kwargs):
            assert kwargs == ({"version": version} if version is not None else {})
            assert sku == "H100"
            if date == "2026-08-31":
                return None
            if date == "2026-09-01":
                return {"data": {"observations": [observation]}}
            raise AssertionError(f"unexpected day read {date}")

    history = read_full_history(
        Reader(), sku="H100", target_date="2026-09-01", version=version
    )

    assert history == [observation]


def test_history_loader_typed_refusal_names_the_missing_bound():
    class Reader:
        def read_series(self, _range, *, sku):
            return {"meta": {"from_observed_at": "2026-08-31T00:00:00.000Z"}}

        def read_day(self, date, *, sku):
            return None

    with pytest.raises(FullReproductionRefusal) as caught:
        read_full_history(Reader(), sku="H100", target_date="2026-09-01")

    assert caught.value.code == "insufficient_observable_history"
    assert "public corpus origin 2026-08-31" in str(caught.value)
    assert "published day 2026-08-31 is unavailable" in str(caught.value)


def test_history_loader_refuses_when_series_and_day_lattices_disagree():
    observation = _observation()

    class Reader:
        def read_series(self, _range, *, sku):
            return {
                "meta": {"from_observed_at": observation["observed_at"]},
                "data": {
                    "observations": [
                        {"observed_at": observation["observed_at"]},
                        {"observed_at": "2026-09-01T00:15:00.000Z"},
                    ]
                },
            }

        def read_day(self, date, *, sku):
            if date == "2026-08-31":
                return None
            return {"data": {"observations": [observation]}}

    with pytest.raises(FullReproductionRefusal) as caught:
        read_full_history(Reader(), sku="H100", target_date="2026-09-01")

    assert caught.value.code == "insufficient_observable_history"
    assert "series/day observation lattice differs" in str(caught.value)


@pytest.mark.parametrize("version", [None, 5])
def test_full_cli_prints_derived_vector_and_value_match(monkeypatch, capsys, version):
    observation = _observation()

    class Reader:
        def version_pointer(self, sku):
            if version is None:
                return None
            return {"history_path": "H100/published", "succession": [{
                "version": version, "methodology_id": observation["methodology_id"],
                "effective_from": "2026-09-01T00:13:39Z",
            }]}

        def describe(self):
            return "test public record"

        def read_series(self, _range, *, sku, **kwargs):
            assert kwargs == ({"version": version} if version is not None else {})
            return {
                "meta": {"from_observed_at": observation["observed_at"]},
                "data": {
                    "observations": [
                        {"observed_at": observation["observed_at"]}
                    ]
                },
            }

        def read_day(self, date, *, sku, **kwargs):
            assert kwargs == ({"version": version} if version is not None else {})
            if date == "2026-08-31":
                return None
            return {"data": {"observations": [observation]}}

    spec = importlib.util.spec_from_file_location(
        "verify_published_record_full_test",
        REPO_ROOT / "scripts" / "verify_published_record.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "PublishedRecordReader", Reader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_published_record.py",
            "--sku",
            "H100",
            "--date",
            "2026-09-01",
            "--full",
            *(["--version", str(version)] if version is not None else []),
        ],
    )

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "raw-only full reproduction" in output
    assert "derived 3.0" in output
    assert "published 3.0" in output
    assert "MATCH" in output
    assert "weights: s0=0.2" in output
    assert "1 MATCH, 0 MISMATCH" in output
    if version is not None:
        assert "version 5 methodology_id h100_sxm_v1_calc_v8 back-calculated" in output

    observation["receipts"][1]["weight"] = 999.0
    assert cli.main() == 1
    output = capsys.readouterr().out
    assert (
        "FIRST DIVERGENCE: 2026-09-01T00:00:00.000Z s1 weight "
        "derived 0.2 published 999.0"
    ) in output


@pytest.mark.parametrize("age", [91, 99, 100, 110])
@pytest.mark.parametrize("history_days", [90, 120])
def test_history_loader_reaches_available_origin_or_full_bound(age, history_days):
    from gpu_index.published.verify import _history_bound_days

    bound = _history_bound_days(history_days=history_days, forward_horizons_hours=[48])

    target = date(2026, 9, 1)
    origin = target - timedelta(days=age - 1)
    series_start = target - timedelta(days=89)
    days = {
        (origin + timedelta(days=i)).isoformat(): {
            "observed_at": f"{origin + timedelta(days=i)}T00:00:00.000Z",
            "calc_params": {"liveness": {"history_days": history_days,
                                          "forward_horizons_hours": [48]}},
        }
        for i in range(age)
    }
    reads = []

    class Reader:
        def read_series(self, _range, *, sku, version):
            assert (sku, version) == ("H100", 5)
            return {
                "meta": {"from_observed_at": days[series_start.isoformat()]["observed_at"]},
                "data": {"observations": [row for day, row in days.items()
                                          if day >= series_start.isoformat()]},
            }

        def read_day(self, day, *, sku, version):
            assert (sku, version) == ("H100", 5)
            reads.append(day)
            return {"data": {"observations": [days[day]]}} if day in days else None

    history = read_full_history(Reader(), sku="H100", target_date=str(target), version=5)
    assert len(history) == min(age, bound)
    assert history[0]["observed_at"] == (
        f"{max(origin, target - timedelta(days=bound - 1))}T00:00:00.000Z"
    )
    assert len(reads) == len(set(reads))


def _two_version_reader():
    # Version 2's own earlier raw price is needed for its carried target.
    # Mixing the as-published lookback would produce 3.0 instead of 6.0.
    early = _observation()
    early["observed_at"] = "2026-09-03T18:45:00.000Z"
    later = copy.deepcopy(early)
    later["methodology_id"] = "h100_sxm_v1_calc_v10"
    later["value_usd_gpu_hr"] = 6.0
    later["stability_band_usd_gpu_hr"] = 2.2
    for receipt in later["receipts"]:
        receipt["price"] *= 2
        receipt["sd"] *= 2
    carried = copy.deepcopy(later)
    carried["observed_at"] = "2026-09-03T19:00:00.000Z"
    for receipt in carried["receipts"]:
        receipt["upstream_status"] = "carried"
    earlier_end = copy.deepcopy(early)
    earlier_end["observed_at"] = carried["observed_at"]

    class Reader:
        def __init__(self):
            self.published = [early, carried]
            self.histories = {1: [early, earlier_end], 2: [later, carried]}
            self.read_versions = []
            self.pointer = {
                "current_version": 2, "history_path": "H100/published",
                "succession": [
                    {"version": 1, "methodology_id": early["methodology_id"],
                     "effective_from": "2026-09-01T00:13:39Z"},
                    {"version": 2, "methodology_id": later["methodology_id"],
                     "effective_from": "2026-09-03T18:59:44Z"},
                ],
            }

        def describe(self):
            return "synthetic public record"

        def version_pointer(self, sku):
            return self.pointer

        def read_series(self, _range, *, sku, version):
            self.read_versions.append(version)
            rows = self.histories[version]
            return {"meta": {"from_observed_at": rows[0]["observed_at"]},
                    "data": {"observations": rows}}

        def read_day(self, day, *, sku, version=None):
            rows = self.published if version is None else self.histories[version]
            matching = [row for row in rows if row["observed_at"][:10] == day]
            return {"data": {"observations": matching}} if matching else None

    return Reader()


def test_as_published_full_uses_each_versions_own_raw_history():
    reader = _two_version_reader()
    result = reproduce_published_history(reader, sku="H100", target_date="2026-09-03")
    assert [(check.verdict, check.derived_value, check.version, check.methodology_id)
            for check in result.checks] == [
        (VERDICT_MATCH, 3.0, 1, "h100_sxm_v1_calc_v8"),
        (VERDICT_MATCH, 6.0, 2, "h100_sxm_v1_calc_v10"),
    ]
    assert reader.read_versions == [1, 2]


def test_as_published_full_cli_reports_both_versions(monkeypatch, capsys):
    reader = _two_version_reader()
    spec = importlib.util.spec_from_file_location(
        "as_published_cli", REPO_ROOT / "scripts" / "verify_published_record.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "PublishedRecordReader", lambda: reader)
    assert cli.main(["--sku", "H100", "--date", "2026-09-03", "--full"]) == 0
    output = capsys.readouterr().out
    assert "version 1 methodology_id h100_sxm_v1_calc_v8" in output
    assert "version 2 methodology_id h100_sxm_v1_calc_v10" in output
    assert "2 MATCH, 0 MISMATCH, 0 degraded" in output
    assert "NOTICE" not in output


def test_pre_launch_history_selects_launch_version_and_labels_rows(monkeypatch, capsys):
    reader = _two_version_reader()
    # Advertise an earlier version as well: pre-launch means launch version,
    # not the oldest advertised version or the version effective on that date.
    reader.histories = {version + 1: rows for version, rows in reader.histories.items()}
    reader.pointer["current_version"] += 1
    for entry in reader.pointer["succession"]:
        entry["version"] += 1
    reader.pointer["succession"].insert(0, {
        "version": 1, "methodology_id": "earlier_method",
        "effective_from": "2026-08-01T00:00:00Z",
    })
    before = copy.deepcopy(reader.histories[2][0])
    before["observed_at"] = "2026-08-25T00:00:00.000Z"
    reader.published = [before]
    reader.histories[2] = [before]
    spec = importlib.util.spec_from_file_location(
        "pre_launch_cli", REPO_ROOT / "scripts" / "verify_published_record.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "PublishedRecordReader", lambda: reader)
    assert cli.main(["--sku", "H100", "--date", "2026-08-25", "--full"]) == 0
    assert reader.read_versions == [2]
    assert ("version 2 methodology_id h100_sxm_v1_calc_v8 back-calculated"
            in capsys.readouterr().out)


@pytest.mark.parametrize("field", ["value_usd_gpu_hr", "stability_band_usd_gpu_hr",
                                   "weight", "liveness_score", "attendance_factor"])
def test_full_compares_against_as_published_outputs(field):
    reader = _two_version_reader()
    reader.published = copy.deepcopy(reader.published)
    if field in reader.published[1]:
        reader.published[1][field] = 999.0
    else:
        reader.published[1]["receipts"][0][field] = 999.0
    result = reproduce_published_history(reader, sku="H100", target_date="2026-09-03")
    assert result.checks[0].verdict == VERDICT_MATCH
    assert result.checks[1].verdict == "mismatch"
    assert result.checks[1].derived_value == 6.0


def test_as_published_methodology_must_be_effective_at_stamp():
    reader = _two_version_reader()
    reader.published = copy.deepcopy(reader.published)
    reader.published[1]["methodology_id"] = reader.published[0]["methodology_id"]
    with pytest.raises(ValueError, match="disagrees with effective version 2"):
        reproduce_published_history(reader, sku="H100", target_date="2026-09-03")


def test_as_published_target_must_exist_in_version_history():
    reader = _two_version_reader()
    reader.histories[2] = reader.histories[2][:1]
    with pytest.raises(FullReproductionRefusal, match="every as-published stamp"):
        reproduce_published_history(reader, sku="H100", target_date="2026-09-03")


@pytest.mark.parametrize("stamp,version", [
    ("2026-09-03T18:59:43.999Z", 1),
    ("2026-09-03T18:59:44.000Z", 2),
    ("2026-09-03T11:59:44-07:00", 2),
])
def test_effective_time_selection_is_inclusive_and_timezone_aware(stamp, version):
    reader = _two_version_reader()
    target = copy.deepcopy(reader.histories[version][0])
    target["observed_at"] = stamp
    reader.published = [target]
    reader.histories[version] = [target]
    result = reproduce_published_history(reader, sku="H100", target_date="2026-09-03")
    assert result.checks[0].version == version
    assert result.checks[0].verdict == VERDICT_MATCH


@pytest.mark.parametrize("pointer", [None, {"current_version": 2}])
def test_as_published_full_requires_advertised_history(pointer):
    reader = _two_version_reader()
    reader.pointer = pointer
    with pytest.raises(FullReproductionRefusal, match="does not advertise as-published history"):
        reproduce_published_history(reader, sku="H100", target_date="2026-09-03")


def test_disclosure_bound_uses_history_and_longest_forward_horizon():
    from gpu_index.published.full import FULL_HISTORY_BOUND_DAYS
    from gpu_index.published.verify import MIN_DISCLOSURE_WINDOW_DAYS, _history_bound_days

    assert FULL_HISTORY_BOUND_DAYS == MIN_DISCLOSURE_WINDOW_DAYS == 100
    assert _history_bound_days(history_days=90, forward_horizons_hours=[6, 49]) == 101
    assert _history_bound_days(history_days=120, forward_horizons_hours=[72]) == 131


@pytest.mark.parametrize("pre_launch", [False, True])
def test_default_command_reproduces_a_digest_verified_two_version_corpus(tmp_path, pre_launch):
    from gpu_index.published.artifacts import payload_digest

    corpus = _two_version_reader()
    target_date = "2026-09-03"
    if pre_launch:
        target_date = "2026-08-25"
        row = copy.deepcopy(corpus.histories[1][0])
        row["observed_at"] = f"{target_date}T00:00:00.000Z"
        corpus.published = [row]
        corpus.histories[1] = [row]
    template = json.loads((REPO_ROOT / "tests/fixtures/published/latest.json").read_text())
    root = tmp_path / "record"

    def write(key, data):
        doc = copy.deepcopy(template)
        doc["data"] = data
        rows = data["observations"]
        doc["meta"].update(
            observation_count=len(rows),
            from_observed_at=min(row["observed_at"] for row in rows),
            to_observed_at=max(row["observed_at"] for row in rows),
        )
        doc["artifact_sha256"] = payload_digest(
            {key: doc[key] for key in ("data", "meta", "license")}
        )
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))

    pointer = corpus.pointer
    pointer.update(sku="H100", **{key: pointer["succession"][-1][key]
                                  for key in ("methodology_id", "effective_from")})
    write("latest.json", {"kind": "gpu_index_latest", "observations": corpus.published,
                          "versions": [pointer]})
    day_key = f"observations/{target_date.replace('-', '/')}.json"
    write(f"H100/published/{day_key}", {
        "kind": "gpu_index_observation_day", "date": target_date,
        "observations": corpus.published,
    })
    for version, rows in corpus.histories.items():
        if pre_launch and version == 2:
            continue
        write(f"H100/v{version}/{day_key}", {
            "kind": "gpu_index_observation_day", "date": target_date, "observations": rows,
        })
        write(f"H100/v{version}/series/90d.json", {
            "kind": "gpu_index_series", "range": "90d", "observations": rows,
        })
    env = {key: value for key, value in os.environ.items() if not key.startswith("GPU_INDEX_")}
    env.update(PYTHON=sys.executable, GPU_INDEX_DATA_DIR=str(root))

    def run(*flags):
        return subprocess.run(
            [str(REPO_ROOT / "reproduce"), *flags, "h100", target_date],
            env=env, capture_output=True, text=True, timeout=30,
        )

    full = run()
    assert full.returncode == 0, full.stderr
    assert "raw-only full reproduction" in full.stdout
    assert "NOTICE" not in full.stdout
    count = 1 if pre_launch else 2
    assert f"{count} MATCH, 0 MISMATCH, 0 degraded" in full.stdout
    assert "version 1 methodology_id h100_sxm_v1_calc_v8" in full.stdout
    assert ("back-calculated" in full.stdout) == pre_launch
    if not pre_launch:
        assert "version 2 methodology_id h100_sxm_v1_calc_v10" in full.stdout
        assert "derived 6.0" in full.stdout
    receipts = run("--receipts")
    assert receipts.returncode == 0, receipts.stderr
    assert "raw-only full reproduction" not in receipts.stdout
    assert f"{count} MATCH, 0 MISMATCH, 0 degraded" in receipts.stdout
    # The fast check remains independent of version-history availability.
    (root / f"H100/v1/{day_key}").unlink()
    assert run().returncode == 2
    assert run("--receipts").stdout == receipts.stdout


# ---------------------------------------- smoothing-armed generations
# EWMA vote pre-smoothing + fence_reject_carry (the
# 2026-09-14 calc_v17/calc_v16 mints). Raw-only reproduction does NOT
# rerun the engine's EWMA: on an armed lane every voting receipt
# disclosed its exact cast price (smoothed_vote_usd), which prices the
# vote; the raw print stays evidence (it still feeds the weight-series
# history, except on fence-reject carries, whose print the engine
# deliberately keeps out of the presence record).


def _armed_observation(*, cast_shift=0.11):
    observation = _observation()
    observation["calc_params"]["pre_smoothing_half_life_hours"] = 1
    observation["calc_params"]["liveness"]["fence_reject_carry"] = True
    for receipt in observation["receipts"]:
        receipt["smoothed_vote_usd"] = receipt["price"] + cast_shift
    return observation


def test_full_armed_votes_disclosed_cast_prices_never_raw():
    """Fresh prints stay 1..5 but every seat cast price+0.11: the
    derivation prices the casts (3.11/1.1), and a published value equal
    to the raw ballot's 3.0 MISMATCHES -- the contrapositive pins that
    the vote centers moved off the raw prints."""
    observation = _armed_observation()
    observation["value_usd_gpu_hr"] = 3.11
    result = reproduce_full_history([observation], target_date="2026-09-01")
    assert result.checks[0].verdict == VERDICT_MATCH
    assert result.checks[0].derived_value == 3.11
    assert result.checks[0].derived_band == 1.1

    raw_published = _armed_observation()
    result = reproduce_full_history(
        [raw_published], target_date="2026-09-01"
    )
    assert result.checks[0].verdict == "mismatch"
    assert result.checks[0].derived_value == 3.11
    assert result.checks[0].published_value == 3.0


def _fence_reject_pair(*, min_publish_second=4):
    """Two armed stamps; on the second, s2's fresh 9.0 print is
    fence-rejected and its booked smoothed vote (3.0, from the first
    stamp) is cast instead. The published carry outputs on the row
    (sd) and its weight-print terms (currency) are corrupted on purpose:
    a raw-only reconstruction must resolve the vote dispersion from the
    prior stamp's bytes and must NOT feed the rejected print into the
    weight history (the engine keeps the seat out of the presence
    record)."""
    first = _armed_observation(cast_shift=0.0)
    second = copy.deepcopy(first)
    second["observed_at"] = "2026-09-01T00:15:00.000Z"
    second["calc_params"]["min_sources_to_publish"] = min_publish_second
    rejected = second["receipts"][2]
    rejected.update(
        {
            "filter_verdict": "rejected",
            "price": 9.0,
            "smoothed_vote_usd": 3.0,
            # The corpus flattens the engine's carried_vote block (the
            # publisher's projection; the cross-repo pin suite holds the
            # shape).
            "carried_vote_from": "2026-09-01T00:00:00.000Z",
            "carry_basis": "no_price",
            "sd": 999.0,
            "currency": None,
            "fx_rate": None,
        }
    )
    return first, second


def test_full_armed_fence_reject_carried_vote_recasts_the_booked_price():
    first, second = _fence_reject_pair()
    result = reproduce_full_history(
        [first, second], target_date="2026-09-01"
    )
    assert [check.verdict for check in result.checks] == [
        VERDICT_MATCH,
        VERDICT_MATCH,
    ]
    # s2 voted its booked 3.0 (never the rejected 9.0, which would move
    # the IQM), with the BOOKED dispersion (the row's 999.0 is ignored)
    # and its CURRENT weight; the corrupted currency proves the rejected
    # print never reached public_weight_print (not a weight-series row).
    assert result.checks[1].derived_value == 3.0
    assert result.checks[1].derived_band == 1.1


def test_full_armed_carried_votes_never_satisfy_the_observed_floor():
    """Five seats vote on the second stamp but s2's is a carried vote:
    4 observed < min_sources_to_publish 5 derives NO composite."""
    first, second = _fence_reject_pair(min_publish_second=5)
    result = reproduce_full_history(
        [first, second], target_date="2026-09-01"
    )
    assert result.checks[0].verdict == VERDICT_MATCH
    assert result.checks[1].verdict == "mismatch"
    assert result.checks[1].derived_value is None


def test_full_armed_status_carried_recasts_the_frozen_smoothed_state():
    """A status-carried row disclosed its frozen smoothed state (4.5,
    deliberately != its booked raw 3.0): the vote prices the DISCLOSED
    cast, with the booked dispersion and the current fading weight --
    corrupting the row's published price/sd changes nothing else."""
    first = _armed_observation(cast_shift=0.0)
    second = copy.deepcopy(first)
    second["observed_at"] = "2026-09-01T00:15:00.000Z"
    # The carried voter never satisfies the observed floor (4 observed).
    second["calc_params"]["min_sources_to_publish"] = 4
    carried = second["receipts"][2]
    carried.update(
        {
            "upstream_status": "carried",
            "carry_basis": "no_price",
            "price": 999.0,
            "sd": 999.0,
            "smoothed_vote_usd": 4.5,
        }
    )
    second["value_usd_gpu_hr"] = 3.700018
    second["stability_band_usd_gpu_hr"] = 1.800018
    result = reproduce_full_history(
        [first, second], target_date="2026-09-01"
    )
    assert [check.verdict for check in result.checks] == [
        VERDICT_MATCH,
        VERDICT_MATCH,
    ]
    assert result.checks[1].derived_value == 3.700018
    assert result.checks[1].derived_band == 1.800018


def test_full_armed_missing_cast_price_refuses_naming_stamp_and_seat():
    observation = _armed_observation()
    del observation["receipts"][2]["smoothed_vote_usd"]
    with pytest.raises(FullReproductionRefusal) as caught:
        reproduce_full_history([observation], target_date="2026-09-01")
    assert caught.value.code == "missing_cast_price"
    assert "2026-09-01T00:00:00.000Z" in str(caught.value)
    assert "s2" in str(caught.value)


@pytest.mark.parametrize("unusable", ["3.11", 0, -1, float("inf")])
def test_full_armed_unusable_cast_price_refuses(unusable):
    observation = _armed_observation()
    observation["receipts"][2]["smoothed_vote_usd"] = unusable
    with pytest.raises(FullReproductionRefusal) as caught:
        reproduce_full_history([observation], target_date="2026-09-01")
    assert caught.value.code == "unusable_cast_price"
    assert "s2" in str(caught.value)


@pytest.mark.parametrize("invalid", ["1h", 0, -1, None, True])
def test_full_invalid_pre_smoothing_half_life_refuses(invalid):
    observation = _armed_observation()
    observation["calc_params"]["pre_smoothing_half_life_hours"] = invalid
    with pytest.raises(FullReproductionRefusal) as caught:
        reproduce_full_history([observation], target_date="2026-09-01")
    assert caught.value.code == "invalid_smoothing_params"


def test_armed_classifier_reads_a_fence_reject_carried_vote_as_absent():
    """The engine's carried_vote arm: the row's status/print stay the
    untouched real print, so the disclosure block is the ONLY absence
    signal -- and it only exists on armed lanes (the same receipt on a
    pre-smoothing observation classifies present, byte-identically to
    today)."""
    observation = _armed_observation(cast_shift=0.0)
    observation["receipts"][2].update(
        {
            "filter_verdict": "rejected",
            "carried_vote_from": "2026-09-01T00:00:00.000Z",
            "carry_basis": "no_price",
        }
    )
    assert public_attendance_events(observation) == {"s2": EVENT_NO_PRICE}

    unarmed = copy.deepcopy(observation)
    del unarmed["calc_params"]["pre_smoothing_half_life_hours"]
    assert public_attendance_events(unarmed) == {}
