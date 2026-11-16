"""Tests of the normative setpoints, the activity windows and the gap."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import comfort_gap as cg


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day_type": ["weekday", "weekend", "weekday", "weekend"],
            "persons": [1, 1, 2, 2],
            "wake_min": [420, 480, 360, 420],
            "bed_min": [1320, 1320, 1380, 1380],
        }
    )


def test_sleep_times_finds_wake_and_bed():
    acts = pd.DataFrame(
        {
            "TUCASEID": [1, 1, 2, 2],
            "TUSTARTTIM": ["04:00:00", "22:30:00", "04:00:00", "23:00:00"],
            "TUSTOPTIME": ["07:00:00", "04:00:00", "06:30:00", "04:00:00"],
            "TRCODEP": [10101, 10101, 10101, 10101],
        }
    )
    out = cg.sleep_times(acts).set_index("TUCASEID")
    assert out.loc[1, "wake_min"] == 420
    assert out.loc[1, "bed_min"] == 22 * 60 + 30
    assert out.loc[2, "wake_min"] == 390
    assert out.loc[2, "bed_min"] == 23 * 60


def test_sleep_times_merges_short_pauses_and_ignores_other_activities():
    acts = pd.DataFrame(
        {
            "TUCASEID": [1, 1, 1, 1],
            "TUSTARTTIM": ["04:00:00", "05:30:00", "10:00:00", "23:00:00"],
            "TUSTOPTIME": ["05:00:00", "07:00:00", "11:00:00", "04:00:00"],
            "TRCODEP": [10101, 10101, 20101, 10101],
        }
    )
    out = cg.sleep_times(acts)
    assert out["wake_min"].iloc[0] == 420


def test_sleep_times_needs_columns():
    with pytest.raises(ValueError, match="needs the columns"):
        cg.sleep_times(pd.DataFrame({"TUCASEID": [1]}))


def _respondents(wake, bed, n=60):
    return pd.DataFrame(
        {
            "wake_min": [wake] * n,
            "bed_min": [bed] * n,
            "TUFNWGTP": 1.0,
            "TUDIARYDAY": [2, 3, 4, 5, 6, 1] * (n // 6),
            "PRTAGE": 40,
        }
    )


def test_household_window_table_constant_times():
    table = cg.household_window_table(
        _respondents(420, 1320), max_persons=3, n_draws=500
    )
    assert len(table) == 6
    assert set(table["wake_min"]) == {420}
    assert set(table["bed_min"]) == {1320}
    assert set(table["day_type"]) == {"weekday", "weekend"}


def test_household_window_table_more_persons_widen_the_window():
    rng = np.random.default_rng(1)
    resp = _respondents(420, 1320, n=600)
    resp["wake_min"] = rng.choice([360, 420, 480, 540], len(resp))
    resp["bed_min"] = rng.choice([1290, 1350, 1410], len(resp))
    table = cg.household_window_table(resp, max_persons=4, n_draws=5000, seed=3)
    day = table[table["day_type"] == "weekday"].set_index("persons")
    assert day.loc[4, "wake_min"] <= day.loc[1, "wake_min"]
    assert day.loc[4, "bed_min"] >= day.loc[1, "bed_min"]


def test_household_window_table_caps_bed_at_midnight_and_checks_input():
    late = _respondents(420, 1500)
    table = cg.household_window_table(late, max_persons=1, n_draws=100)
    assert set(table["bed_min"]) == {1440}
    with pytest.raises(ValueError, match="needs the columns"):
        cg.household_window_table(pd.DataFrame({"wake_min": [1]}))
    only_weekdays = _respondents(420, 1320)
    only_weekdays = only_weekdays[only_weekdays["TUDIARYDAY"].between(2, 6)]
    with pytest.raises(ValueError, match="no respondent"):
        cg.household_window_table(only_weekdays, max_persons=1, n_draws=10)


def test_daylight_saving_mask_2018():
    mask = cg.daylight_saving_mask(2018, step_minutes=60)
    span = pd.Timestamp("2018-11-04 01:00") - pd.Timestamp("2018-03-11 02:00")
    assert int(mask.sum()) == int(span / pd.Timedelta(hours=1))
    assert len(mask) == 8760
    assert not cg.daylight_saving_mask(2018, observes=False).any()
    assert len(cg.daylight_saving_mask(2018)) == 35040


def test_required_masks_follow_the_window_and_the_day_type():
    required, pre = cg.required_masks(np.ones(96 * 7), 1, _windows())
    assert int(required[:96].sum()) == 60
    assert int(required[96 * 5 : 96 * 6].sum()) == 56  # Saturday, wakes at 8:00
    assert int(pre[:96].sum()) == 12
    assert not (required & pre).any()
    assert required[28] and not required[27] and pre[27]


def test_required_masks_need_someone_at_home_and_cap_persons():
    presence = np.zeros(96 * 7)
    presence[40:50] = 1.0
    required, _ = cg.required_masks(presence, 5, _windows())
    assert int(required.sum()) == 10  # persons above the table use the largest row
    with pytest.raises(ValueError, match="no row"):
        cg.required_masks(np.ones(96), 1, _windows().iloc[[0, 2, 3]])


def test_normative_setpoints_raise_heating_and_lower_cooling():
    heating, cooling = cg.normative_setpoints(
        np.array([15.0, 20.0, 15.0]),
        np.array([30.0, 22.0, 30.0]),
        np.array([True, True, False]),
        np.array([False, False, False]),
    )
    assert heating.tolist() == [18.0, 20.0, 15.0]
    assert cooling.tolist() == [26.0, 22.0, 30.0]


def test_normative_setpoints_without_cooling_equipment():
    heating, cooling = cg.normative_setpoints(
        np.array([15.0, 15.0, 15.0]),
        np.array([99.0, 99.0, 99.0]),
        np.array([True, False, False]),
        np.array([False, True, False]),
        no_cooling=True,
    )
    assert cooling.tolist() == [26.0, 26.0, 99.0]
    assert (cooling >= heating).all()


def test_to_standard_time_shifts_only_during_daylight_saving():
    mask = np.arange(12) == 6
    daylight = np.array([True] * 6 + [False] * 6)
    out = cg.to_standard_time(mask, daylight, step_minutes=15)
    assert out[2]  # clock step 6 is standard step 2 while the clock is ahead
    assert out[6]  # no shift once daylight saving time has ended
    assert int(out.sum()) == 2
    with pytest.raises(ValueError, match="same length"):
        cg.to_standard_time(mask, daylight[:5])


def test_out_of_range_counts_hours_and_degree_hours_with_tolerance():
    t = np.array([17.0, 17.9, 20.0, 26.1, 27.0])
    result = cg.out_of_range(t, np.ones(5, dtype=bool), step_minutes=60)
    assert result["hours_low"] == 1.0  # 17.9 is inside the 0.2 tolerance
    assert result["hours_high"] == 1.0
    assert result["degree_hours_low"] == pytest.approx(1.1)
    assert result["degree_hours_high"] == pytest.approx(1.1)
    none = cg.out_of_range(t, np.zeros(5, dtype=bool), step_minutes=60)
    assert none["hours_low"] == 0.0 and none["required_hours"] == 0.0


def test_dwelling_gap_removes_the_unserved_part_and_keeps_signs():
    gap = cg.dwelling_gap(
        np.array([0.0, 1.0]),
        np.array([3.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([3.0, 0.5]),
        cooling_unserved_base=np.array([2.0, 0.0]),
    )
    assert gap["cooling_current"].tolist() == [1.0, 1.0]
    assert gap["cooling_gap"].tolist() == [2.0, -0.5]
    assert gap["heating_gap"].tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="same length"):
        cg.dwelling_gap(np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(2))


def test_tract_profiles_add_the_series_with_their_coefficients():
    coef = pd.DataFrame(
        {
            "GEOID": ["a", "a", "b"],
            "building_id": [1, 2, 1],
            "coef": [0.5, 0.5, 1.0],
            "coef_heating": [1.0, 0.0, 2.0],
        }
    )
    series = {
        1: pd.DataFrame({"x": [1.0, 2.0, 3.0], "h": [1.0, 1.0, 1.0]}),
        2: pd.DataFrame({"x": [3.0, 2.0, 1.0], "h": [4.0, 4.0, 4.0]}),
    }
    tracts, acc = cg.accumulate_tract_profiles(
        coef,
        lambda b: series[b],
        ["x", "h"],
        {"x": "coef", "h": "coef_heating"},
        hours=3,
    )
    assert tracts.tolist() == ["a", "b"]
    assert acc[0, 0].tolist() == [2.0, 2.0, 2.0]  # 0.5 * (1+3), 0.5 * (2+2), ...
    assert acc[0, 1].tolist() == [1.0, 2.0, 3.0]
    assert acc[1, 0].tolist() == [
        1.0,
        1.0,
        1.0,
    ]  # the second dwelling has coefficient 0
    assert acc[1, 1].tolist() == [2.0, 2.0, 2.0]


def test_tract_profiles_check_rows_and_coefficients():
    coef = pd.DataFrame({"GEOID": ["a"], "building_id": [1], "coef": [1.0]})
    with pytest.raises(ValueError, match="No coefficient"):
        cg.accumulate_tract_profiles(coef, lambda b: None, ["x"], {}, hours=2)
    with pytest.raises(ValueError, match="rows"):
        cg.accumulate_tract_profiles(
            coef, lambda b: pd.DataFrame({"x": [1.0]}), ["x"], {"x": "coef"}, hours=2
        )


def test_tract_profiles_accept_a_series_that_differs_by_tract():
    coef = pd.DataFrame(
        {"GEOID": ["a", "b"], "building_id": [1, 1], "coef": [1.0, 0.5]}
    )

    def by_tract(building, geoids):
        out = np.zeros((1, len(geoids), 2), dtype=np.float32)
        for i, geoid in enumerate(geoids):
            out[0, i] = [1.0, 1.0] if geoid == "a" else [4.0, 2.0]
        return out

    tracts, acc = cg.accumulate_tract_profiles(
        coef, None, ["x"], {"x": "coef"}, hours=2, tract_series=by_tract
    )
    assert acc[0, 0].tolist() == [1.0, 1.0]
    assert acc[0, 1].tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="are not"):
        cg.accumulate_tract_profiles(
            coef, None, ["x"], {"x": "coef"}, hours=3, tract_series=by_tract
        )
