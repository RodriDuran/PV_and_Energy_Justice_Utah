"""Tests of the climate correction of the annual loads."""

import doctest

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import energy_requirement as er
from utah_ejpi.energy_requirement import (
    DegreeDaySensitivity,
    build_tract_type_energy,
    climate_factor,
    fit_degree_day_sensitivity,
    leave_one_county_out,
    tract_energy_summary,
)
from utah_ejpi.utils import validate_table

LINE = DegreeDaySensitivity("heating", 0.0, 1.0, np.eye(2), 10, 3, 1.0, 3.0)


def test_docstring_examples_run():
    assert doctest.testmod(er).failed == 0


def _stations(hdd, prefix="C"):
    return pd.DataFrame(
        {
            "id": [f"{prefix}{i}" for i in range(len(hdd))],
            "kind": "station",
            "hdd": hdd,
            "cdd": [0.0] * len(hdd),
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )


def _synthetic(
    intercept,
    slope,
    n_per_county=40,
    seed=0,
    hdd=(1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0),
    prefix="C",
):
    """Dwellings whose heating load is a scale times a line of the degree days."""
    rng = np.random.default_rng(seed)
    hdd = np.array(hdd)
    rows = []
    for c, h in enumerate(hdd):
        for _ in range(n_per_county):
            bedrooms = int(rng.integers(1, 5))
            scale = 10000.0 * bedrooms * (1.5 if rng.random() < 0.5 else 1.0)
            rows.append(
                {
                    "county": f"{prefix}{c}",
                    "occupied": True,
                    "tenure": "Owner",
                    "vintage": "1980 to 1999",
                    "building_type": "1" if scale > 20000 else "2 to 4",
                    "bedrooms": bedrooms,
                    "floor_area": "1500-1999",
                    "heating_load_kbtu": scale * (intercept + slope * h / 1000.0),
                    "cooling_load_kbtu": 0.0,
                }
            )
    return pd.DataFrame(rows), _stations(hdd, prefix)


def test_fit_recovers_the_shape_of_the_line_and_the_range_of_the_stations():
    dwellings, stations = _synthetic(0.2, 1.0)
    s = fit_degree_day_sensitivity(dwellings, stations, "heating")
    assert s.n_stations == 6
    assert (s.dd_min, s.dd_max) == (1.5, 4.0)
    assert s.covariance.shape == (2, 2)
    # the ratio of two values of the line is what matters
    true = (0.2 + 1.0 * 3.5) / (0.2 + 1.0 * 2.0)
    est = (s.intercept + s.slope * 3.5) / (s.intercept + s.slope * 2.0)
    assert est == pytest.approx(true, rel=0.1)


def test_counties_of_two_states_extend_the_range_of_the_fit():
    warm, warm_stations = _synthetic(0.2, 1.0)
    cold, cold_stations = _synthetic(
        0.2, 1.0, seed=1, hdd=(4500.0, 5000.0, 5500.0), prefix="D"
    )
    alone = fit_degree_day_sensitivity(warm, warm_stations, "heating")
    pooled = fit_degree_day_sensitivity(
        pd.concat([warm, cold], ignore_index=True),
        pd.concat([warm_stations, cold_stations], ignore_index=True),
        "heating",
    )
    assert alone.dd_max == 4.0 and pooled.dd_max == 5.5 and pooled.n_stations == 9
    # a tract of 5,200 degree days is outside the range of one state and inside the
    # pooled range
    tract, station = [5200.0], [3000.0]
    assert climate_factor(alone, tract, station)[1].tolist() == [True]
    assert climate_factor(pooled, tract, station)[1].tolist() == [False]
    true = (0.2 + 5.2) / (0.2 + 3.0)
    assert climate_factor(pooled, tract, station)[0][0] == pytest.approx(true, rel=0.05)


def test_dwellings_without_a_system_are_left_out_of_the_fit():
    dwellings, stations = _synthetic(0.2, 1.0)
    extra = dwellings.iloc[:5].copy()
    extra["heating_load_kbtu"] = 0.0
    both = pd.concat([dwellings, extra], ignore_index=True)
    a = fit_degree_day_sensitivity(dwellings, stations, "heating")
    b = fit_degree_day_sensitivity(both, stations, "heating")
    assert b.n_dwellings == a.n_dwellings
    assert b.slope == pytest.approx(a.slope)


def test_county_without_station_and_unknown_end_use_fail():
    dwellings, stations = _synthetic(0.2, 1.0)
    with pytest.raises(ValueError, match="without a weather station"):
        fit_degree_day_sensitivity(dwellings, stations.iloc[:3], "heating")
    with pytest.raises(KeyError):
        fit_degree_day_sensitivity(dwellings, stations, "lighting")


def test_factor_is_one_at_the_station_and_zero_for_a_dwelling_without_system():
    factor, out = climate_factor(LINE, [2500.0], [2500.0])
    assert factor.tolist() == [1.0]
    assert out.tolist() == [False]
    # a load of zero stays zero whatever the factor is
    assert (np.array([0.0]) * climate_factor(LINE, [4000.0], [2000.0])[0]).tolist() == [
        0.0
    ]


def test_factor_extrapolates_or_caps_and_is_never_negative():
    free = DegreeDaySensitivity("heating", -2.0, 1.0, np.eye(2), 10, 3, 2.5, 3.5)
    factor, _ = climate_factor(free, [1000.0, 6000.0], [3000.0, 3000.0])
    assert factor[0] == 0.0
    assert factor[1] == pytest.approx((6.0 - 2.0) / (3.0 - 2.0))
    capped, out = climate_factor(free, [6000.0], [3000.0], "cap")
    assert capped[0] == pytest.approx((3.5 - 2.0) / (3.0 - 2.0))
    assert out.tolist() == [True]
    with pytest.raises(ValueError, match="'linear' or 'cap'"):
        climate_factor(free, [1.0], [1.0], "none")
    with pytest.raises(ValueError, match="not positive"):
        climate_factor(free, [3000.0], [1000.0])


def test_leave_one_county_out_returns_one_row_per_county():
    dwellings, stations = _synthetic(0.2, 1.0)
    out = leave_one_county_out(dwellings, stations, "heating")
    assert len(out) == 6
    assert out["n"].sum() == len(dwellings)
    err_with = (out.predicted_with_climate - out.observed).abs().mean()
    err_without = (out.predicted_without_climate - out.observed).abs().mean()
    assert err_with < err_without


def _energy_inputs():
    geoid = "49001000100"
    weights = pd.DataFrame(
        {
            "GEOID": [geoid, geoid],
            "tenure": ["Owner", "Renter"],
            "vintage": ["1980 to 1999", "1960 to 1979"],
            "building_type": ["1", "5 to 19"],
            "weight": [100.0, 50.0],
        }
    )
    donors = pd.DataFrame(
        {
            "GEOID": [geoid] * 3,
            "tenure": ["Owner", "Owner", "Renter"],
            "vintage": ["1980 to 1999", "1980 to 1999", "1960 to 1979"],
            "building_type": ["1", "1", "5 to 19"],
            "building_id": [1, 2, 3],
            "donor_weight": [0.5, 0.5, 1.0],
            "source_level": "puma",
        }
    )
    dwellings = pd.DataFrame(
        {
            "building_id": [1, 2, 3],
            "upgrade": 0,
            "county": ["C1", "C1", "C2"],
            "occupants": [2.0, 4.0, 1.0],
            "heating_load_kbtu": [100.0, 200.0, 60.0],
            "cooling_load_kbtu": [10.0, 0.0, 20.0],
        }
    )
    tract = pd.DataFrame(
        {
            "id": [geoid],
            "kind": "tract",
            "hdd": [4000.0],
            "cdd": [200.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    stations = pd.DataFrame(
        {
            "id": ["C1", "C2"],
            "kind": "station",
            "hdd": [2000.0, 4000.0],
            "cdd": [100.0, 200.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    return weights, donors, dwellings, tract, stations


def test_tract_energy_applies_the_factor_of_each_donor():
    weights, donors, dwellings, tract, stations = _energy_inputs()
    cool = DegreeDaySensitivity("cooling", 0.0, 1.0, np.eye(2), 10, 2, 0.1, 0.3)
    out = build_tract_type_energy(
        weights, donors, dwellings, tract, stations, LINE, cool
    )
    validate_table(out, "tract_type_energy")
    owner = out[out.tenure == "Owner"].iloc[0]
    # heating factors: tract 4000 over station 2000 = 2 for both owner donors
    assert owner.heating_kbtu == pytest.approx(0.5 * 100 * 2 + 0.5 * 200 * 2)
    assert owner.heating_kbtu_uncorrected == pytest.approx(150.0)
    # cooling factor 200 over 100 = 2 for the donors of county C1
    assert owner.cooling_kbtu == pytest.approx(0.5 * 10 * 2)
    renter = out[out.tenure == "Renter"].iloc[0]
    # the renter donor lives at a station with the same degree days as the tract
    assert renter.heating_kbtu == pytest.approx(60.0)
    assert renter.dwellings == 50.0
    assert owner.occupants == pytest.approx(3.0)
    # the tract (4000) is beyond the fitted range of the heating line (1000 to 3000)
    assert owner.extrapolated_heating == pytest.approx(1.0)
    assert owner.upgrade == 0


def test_cap_gives_a_smaller_correction_and_summary_adds_the_types():
    weights, donors, dwellings, tract, stations = _energy_inputs()
    cool = DegreeDaySensitivity("cooling", 0.0, 1.0, np.eye(2), 10, 2, 0.1, 0.3)
    free = build_tract_type_energy(
        weights, donors, dwellings, tract, stations, LINE, cool
    )
    capped = build_tract_type_energy(
        weights, donors, dwellings, tract, stations, LINE, cool, "cap"
    )
    assert capped.heating_kbtu.sum() < free.heating_kbtu.sum()
    summary = tract_energy_summary(free)
    row = summary.iloc[0]
    assert row.dwellings == 150.0
    total = 100 * 300.0 + 50 * 60.0
    assert row.heating_kbtu_total == pytest.approx(total)
    assert row.heating_kbtu == pytest.approx(total / 150.0)
    assert row.persons == pytest.approx(100 * 3.0 + 50 * 1.0)
    assert row.heating_kbtu_per_person == pytest.approx(total / 350.0)
    # the whole tract lies outside the fitted range, so every dwelling is extrapolated
    assert row.extrapolated_heating == pytest.approx(1.0)


def test_upgrades_cannot_be_mixed():
    weights, donors, dwellings, tract, stations = _energy_inputs()
    dwellings.loc[2, "upgrade"] = 4
    cool = DegreeDaySensitivity("cooling", 0.0, 1.0, np.eye(2), 10, 2, 0.1, 0.3)
    with pytest.raises(ValueError, match="one upgrade"):
        build_tract_type_energy(weights, donors, dwellings, tract, stations, LINE, cool)
