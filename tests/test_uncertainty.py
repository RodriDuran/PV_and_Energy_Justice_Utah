"""Tests of the simulation of the uncertainty of the energy of each tract."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import uncertainty as un
from utah_ejpi.energy_requirement import DegreeDaySensitivity

T1, T2 = "49001000100", "49001000200"
A = ("Owner", "1980 to 1999", "1")
B = ("Renter", "1960 to 1979", "5 to 19")


def _inputs(moe=0.0, total_moe=0.0):
    keys = ["tenure", "vintage", "building_type"]
    rows = [(T1, A, 100.0), (T1, B, 50.0), (T2, A, 80.0), (T2, B, 20.0)]
    weights = pd.DataFrame(
        [
            {"GEOID": g, **dict(zip(keys, k, strict=True)), "weight": n, "moe": moe}
            for g, k, n in rows
        ]
    )
    # dwelling 1 is the donor of type A in both tracts; B has a different dwelling
    donor_rows = [(T1, A, 1), (T1, B, 2), (T2, A, 1), (T2, B, 3)]
    donors = pd.DataFrame(
        [
            {
                "GEOID": g,
                **dict(zip(keys, k, strict=True)),
                "building_id": i,
                "donor_weight": 1.0,
                "source_level": "puma",
            }
            for g, k, i in donor_rows
        ]
    )
    dwellings = pd.DataFrame(
        {
            "building_id": [1, 2, 3],
            "upgrade": 0,
            "county": ["C1", "C1", "C1"],
            "occupants": [2.0, 3.0, 1.0],
            "heating_load_kbtu": [100.0, 200.0, 300.0],
            "cooling_load_kbtu": [10.0, 0.0, 30.0],
            "heating_kwh_natural_gas": [30.0, 60.0, 90.0],
            "cooling_kwh_electricity": [3.0, 0.0, 9.0],
        }
    )
    totals = pd.DataFrame(
        {
            "GEOID": [T1, T2],
            "total": [150.0, 100.0],
            "total_moe": [total_moe, total_moe],
        }
    )
    tract = pd.DataFrame(
        {
            "id": [T1, T2],
            "kind": "tract",
            "hdd": [4000.0, 2000.0],
            "cdd": [100.0, 100.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    stations = pd.DataFrame(
        {
            "id": ["C1"],
            "kind": "station",
            "hdd": [2000.0],
            "cdd": [100.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    zero = np.zeros((2, 2))
    heating = DegreeDaySensitivity("heating", 0.0, 1.0, zero, 10, 1, 1.0, 5.0)
    cooling = DegreeDaySensitivity("cooling", 1.0, 0.0, zero, 10, 1, 0.0, 1.0)
    return weights, donors, dwellings, totals, tract, stations, heating, cooling


def _run(dispersion, n_draws=3, seed=1, moe=0.0, total_moe=0.0, **kwargs):
    weights, donors, dwellings, totals, tract, stations, heating, cooling = _inputs(
        moe, total_moe
    )
    return un.simulate_tract_energy(
        weights,
        donors,
        dwellings,
        totals,
        tract,
        stations,
        heating,
        cooling,
        dispersion,
        n_draws=n_draws,
        seed=seed,
        **kwargs,
    )


NO_NOISE = un.Dispersion(0.0, 0.0, 0.0)


def test_without_uncertainty_every_simulation_gives_the_deterministic_value():
    out = _run(NO_NOISE)
    first = out[(out.GEOID == T1) & (out.group == "all")]
    assert len(first) == 3
    # heating factor of tract 1: 4000 / 2000 = 2; types A (100) and B (50)
    expected = (100 * 100 * 2 + 50 * 200 * 2) / 150
    assert first["heating_kbtu"].tolist() == pytest.approx([expected] * 3)
    assert first["dwellings"].tolist() == pytest.approx([150.0] * 3)
    assert first["persons"].iloc[0] == pytest.approx(100 * 2 + 50 * 3)
    owner = out[(out.GEOID == T2) & (out.group == "Owner")].iloc[0]
    assert owner["heating_kbtu"] == pytest.approx(100.0)
    assert owner["cooling_kbtu"] == pytest.approx(10.0)
    assert owner["heating_kwh"] == pytest.approx(30.0)
    renter = out[(out.GEOID == T1) & (out.group == "Renter")].iloc[0]
    assert renter["cooling_kbtu"] == 0.0


def test_the_same_seed_gives_the_same_simulations():
    a = _run(un.Dispersion(0.5, 0.5, 0.3), seed=7, moe=10.0, total_moe=10.0)
    b = _run(un.Dispersion(0.5, 0.5, 0.3), seed=7, moe=10.0, total_moe=10.0)
    c = _run(un.Dispersion(0.5, 0.5, 0.3), seed=8, moe=10.0, total_moe=10.0)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["heating_kbtu"], c["heating_kbtu"])


def test_dwellings_are_rescaled_to_the_drawn_total_and_zero_cells_stay_zero():
    out = _run(NO_NOISE, n_draws=500, moe=40.0, total_moe=0.0)
    sums = out[(out.group == "all")].groupby(["GEOID", "draw"])["dwellings"].first()
    # the total of the tract is fixed (margin of error zero) in every simulation
    assert sums.loc[T1].tolist() == pytest.approx([150.0] * 500)
    # but the split between owners and renters varies
    owners = out[(out.group == "Owner") & (out.GEOID == T1)]["dwellings"]
    assert owners.std() > 1.0
    assert owners.mean() == pytest.approx(100.0, abs=3.0)


def test_total_of_the_tract_varies_with_its_margin_of_error():
    out = _run(NO_NOISE, n_draws=2000, moe=0.0, total_moe=16.45)
    total = out[(out.group == "all") & (out.GEOID == T1)]["dwellings"]
    assert total.mean() == pytest.approx(150.0, abs=1.0)
    assert total.std() == pytest.approx(10.0, rel=0.1)


def test_dispersion_of_a_dwelling_is_lognormal_with_mean_one_and_the_given_cv():
    out = _run(un.Dispersion(0.6, 0.0, 0.0), n_draws=4000)
    x = out[(out.GEOID == T2) & (out.group == "Owner")]["heating_kbtu"] / 100.0
    assert x.mean() == pytest.approx(1.0, abs=0.04)
    assert x.std() / x.mean() == pytest.approx(0.6, rel=0.1)


def test_tracts_that_share_a_dwelling_share_its_error():
    out = _run(un.Dispersion(0.6, 0.0, 0.0), n_draws=300)
    a = out[(out.GEOID == T1) & (out.group == "Owner")].set_index("draw")[
        "heating_kbtu"
    ]
    b = out[(out.GEOID == T2) & (out.group == "Owner")].set_index("draw")[
        "heating_kbtu"
    ]
    # same dwelling, climate factors 2 and 1: the two series are proportional
    assert (a / b).tolist() == pytest.approx([2.0] * 300)
    c = out[(out.GEOID == T1) & (out.group == "Renter")].set_index("draw")[
        "heating_kbtu"
    ]
    d = out[(out.GEOID == T2) & (out.group == "Renter")].set_index("draw")[
        "heating_kbtu"
    ]
    # different dwellings: the errors are independent
    assert abs(np.corrcoef(c, d)[0, 1]) < 0.2


def test_cap_gives_less_heating_than_the_extension_in_a_colder_tract():
    free = _run(NO_NOISE, extrapolation="linear")
    capped = _run(NO_NOISE, extrapolation="cap")
    a = free[(free.GEOID == T1) & (free.group == "all")]["heating_kbtu"].iloc[0]
    b = capped[(capped.GEOID == T1) & (capped.group == "all")]["heating_kbtu"].iloc[0]
    # tract 1 has 4 thousand degree days, beyond the stations used (1 to 5 here), so
    # the cap changes nothing; make the range smaller to see an effect
    assert a == pytest.approx(b)
    weights, donors, dwellings, totals, tract, stations, heating, cooling = _inputs()
    narrow = DegreeDaySensitivity(
        "heating", 0.0, 1.0, np.zeros((2, 2)), 10, 1, 1.0, 3.0
    )
    capped = un.simulate_tract_energy(
        weights,
        donors,
        dwellings,
        totals,
        tract,
        stations,
        narrow,
        cooling,
        NO_NOISE,
        2,
        1,
        "cap",
    )
    value = capped[(capped.GEOID == T1) & (capped.group == "all")]["heating_kbtu"].iloc[
        0
    ]
    assert value == pytest.approx((100 * 100 * 1.5 + 50 * 200 * 1.5) / 150)


def test_invalid_inputs_fail():
    with pytest.raises(ValueError, match="'linear' or 'cap'"):
        _run(NO_NOISE, extrapolation="none")
    weights, donors, dwellings, totals, tract, stations, heating, cooling = _inputs()
    with pytest.raises(ValueError, match="no total"):
        un.simulate_tract_energy(
            weights,
            donors,
            dwellings,
            totals.iloc[:1],
            tract,
            stations,
            heating,
            cooling,
            NO_NOISE,
        )
    mixed = dwellings.assign(upgrade=[0, 4, 0])
    with pytest.raises(ValueError, match="one upgrade"):
        un.simulate_tract_energy(
            weights, donors, mixed, totals, tract, stations, heating, cooling, NO_NOISE
        )


def test_level_factors_scale_only_the_energy_columns():
    out = _run(NO_NOISE)
    scaled = un.apply_level_factors(out, 0.8, 0.5)
    assert scaled["heating_kbtu"].iloc[0] == pytest.approx(
        0.8 * out["heating_kbtu"].iloc[0]
    )
    assert scaled["cooling_kwh"].iloc[0] == pytest.approx(
        0.5 * out["cooling_kwh"].iloc[0]
    )
    assert scaled["dwellings"].tolist() == out["dwellings"].tolist()


def test_summary_gives_mean_interval_and_quality_class():
    draws = pd.DataFrame(
        {
            "GEOID": [T1] * 100 + [T2] * 100,
            "group": "all",
            "draw": list(range(100)) * 2,
            "heating_kbtu": list(np.linspace(90, 110, 100))
            + list(np.linspace(10, 190, 100)),
        }
    )
    s = un.summarize_draws(draws, ["heating_kbtu"]).set_index("GEOID")
    assert s.loc[T1, "heating_kbtu"] == pytest.approx(100.0)
    assert s.loc[T1, "heating_kbtu_lo"] < 92.5 < s.loc[T1, "heating_kbtu_hi"] - 10
    assert s.loc[T1, "heating_kbtu_quality"] == "high"
    assert s.loc[T2, "heating_kbtu_quality"] == "low"


def test_rank_stability_reports_position_and_class_of_each_tract():
    rng = np.random.default_rng(0)
    rows = []
    for draw in range(200):
        rows += [
            ("A", draw, 10.0 + rng.normal(0, 0.1)),
            ("B", draw, 50.0 + rng.normal(0, 30.0)),
            ("C", draw, 100.0 + rng.normal(0, 0.1)),
        ]
    draws = pd.DataFrame(rows, columns=["GEOID", "draw", "heating_kbtu"])
    draws["group"] = "all"
    out = un.rank_stability(draws, "heating_kbtu", n_classes=3).set_index("GEOID")
    assert out.loc["A", "modal_class"] == 1
    assert out.loc["C", "modal_class"] == 3
    # tract B overtakes tract A in about 5 % of the simulations
    assert out.loc["A", "class_share"] > 0.9
    assert (
        out.loc["B", "rank_hi"] - out.loc["B", "rank_lo"]
        > out.loc["A", "rank_hi"] - out.loc["A", "rank_lo"]
    )


def test_within_type_dispersion_pools_the_types_with_several_dwellings():
    rows = []
    for k, (a, b) in enumerate([(90.0, 110.0), (180.0, 220.0)]):
        for load in (a, b):
            rows.append(
                {
                    "occupied": True,
                    "tenure": "Owner",
                    "vintage": "1980 to 1999",
                    "building_type": str(k),
                    "heating_load_kbtu": load,
                    "cooling_load_kbtu": load / 2,
                }
            )
    out = un.within_type_dispersion(pd.DataFrame(rows))
    # each type has two dwellings 20 % apart from the mean in relative terms
    assert out.cv_heating == pytest.approx(np.sqrt(0.5 * (200 + 800)) / 150.0, rel=1e-6)
    assert out.correlation == pytest.approx(1.0)
