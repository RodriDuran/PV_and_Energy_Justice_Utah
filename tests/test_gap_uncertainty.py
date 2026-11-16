"""Tests of the simulation of the energy gap with the donors resampled."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import gap_uncertainty as gu
from utah_ejpi.energy_requirement import DegreeDaySensitivity

T1, T2 = "49001000100", "49001000200"
A = ("Owner", "1980 to 1999", "1")
B = ("Renter", "1960 to 1979", "5 to 19")
KEYS = ["tenure", "vintage", "building_type"]
USES = {"cur_cooling": "cooling", "gap_cooling": "cooling", "norm_cooling": "cooling"}


def _inputs(donors_of_a=(1, 2, 3, 4)):
    weights = pd.DataFrame(
        [
            {"GEOID": g, **dict(zip(KEYS, k, strict=True)), "weight": n, "moe": 0.0}
            for g, k, n in [(T1, A, 100.0), (T1, B, 50.0), (T2, A, 80.0), (T2, B, 20.0)]
        ]
    )
    rows = []
    for tract in (T1, T2):
        rows += [(tract, A, i) for i in donors_of_a]
        rows += [(tract, B, 5)]
    donors = pd.DataFrame(
        [
            {
                "GEOID": g,
                **dict(zip(KEYS, k, strict=True)),
                "building_id": i,
                "donor_weight": 1.0 / (len(donors_of_a) if k == A else 1),
                "source_level": "puma",
            }
            for g, k, i in rows
        ]
    )
    dwellings = pd.DataFrame(
        {
            "building_id": [1, 2, 3, 4, 5],
            "county": "C1",
            "occupants": [2.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    cur = np.array([10.0, 10.0, 10.0, 10.0, 20.0])
    gap = np.array([0.0, 10.0, 0.0, 10.0, 5.0])
    quantities = pd.DataFrame(
        {"cur_cooling": cur, "gap_cooling": gap, "norm_cooling": cur + gap},
        index=pd.Index([1, 2, 3, 4, 5], name="building_id"),
    )
    totals = pd.DataFrame(
        {"GEOID": [T1, T2], "total": [150.0, 100.0], "total_moe": [0.0, 0.0]}
    )
    tract = pd.DataFrame(
        {
            "id": [T1, T2],
            "kind": "tract",
            "hdd": [2000.0, 2000.0],
            "cdd": [200.0, 100.0],
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
    cooling = DegreeDaySensitivity("cooling", 1.0, 1.0, zero, 10, 1, 0.0, 1.0)
    return dict(
        weights=weights,
        donors=donors,
        dwellings=dwellings,
        quantities=quantities,
        uses=USES,
        tract_totals=totals,
        tract_degree_days=tract,
        station_degree_days=stations,
        heating=heating,
        cooling=cooling,
    )


def _run(inputs=None, **kwargs):
    args = inputs or _inputs()
    kwargs.setdefault("n_draws", 4)
    return gu.simulate_tract_quantities(**args, **kwargs)


def test_without_resampling_every_simulation_is_the_weighted_mean():
    out = _run(resample=False)
    cell_a = out[(out.GEOID == T2) & (out.group == "Owner")]
    assert cell_a["cur_cooling"].tolist() == [10.0] * 4
    assert cell_a["gap_cooling"].tolist() == [5.0] * 4  # mean of 0, 10, 0 and 10
    renter = out[(out.GEOID == T2) & (out.group == "Renter")]
    assert renter["gap_cooling"].tolist() == [5.0] * 4
    both = out[(out.GEOID == T2) & (out.group == "all")]
    assert both["gap_cooling"].iloc[0] == pytest.approx((80 * 5.0 + 20 * 5.0) / 100)


def test_a_cell_with_one_donor_has_no_sampling_error():
    out = _run(n_draws=20)
    renter = out[(out.GEOID == T1) & (out.group == "Renter")]
    assert renter["cur_cooling"].std() == pytest.approx(0.0, abs=1e-9)


def test_resampling_the_donors_gives_the_variance_of_the_bayesian_bootstrap():
    out = _run(n_draws=3000, seed=5)
    owner = out[(out.GEOID == T2) & (out.group == "Owner")]["gap_cooling"]
    # Dirichlet(1, 1, 1, 1) shares: the mean of four values 0, 10, 0, 10 has a
    # variance of 100 * 0.5 * 0.5 / 5 = 5.
    assert owner.mean() == pytest.approx(5.0, abs=0.15)
    assert owner.std() == pytest.approx(np.sqrt(5.0), rel=0.1)
    assert out[(out.GEOID == T2) & (out.group == "Owner")]["cur_cooling"].std() < 1e-9


def test_the_gap_travels_with_the_donor_and_the_sum_is_kept():
    out = _run(n_draws=50, seed=2)
    assert np.allclose(
        out["norm_cooling"], out["cur_cooling"] + out["gap_cooling"], atol=1e-9
    )


def test_the_same_seed_gives_the_same_simulations():
    first = _run(n_draws=6, seed=9)
    second = _run(n_draws=6, seed=9)
    pd.testing.assert_frame_equal(first, second)


def test_climate_correction_applies_only_to_the_chosen_quantities():
    out = _run(resample=False, climate_corrected=["cur_cooling"])
    # Cooling line: a=1, b=1 per 1,000 degree days; tract T1 has 200, station 100.
    # Ratio (1 + 0.2) / (1 + 0.1).
    cell = out[(out.GEOID == T1) & (out.group == "Owner")].iloc[0]
    assert cell["cur_cooling"] == pytest.approx(10.0 * 1.2 / 1.1)
    assert cell["gap_cooling"] == pytest.approx(5.0)


def test_a_missing_value_drops_the_donor_and_an_empty_cell_drops_the_cell():
    inputs = _inputs()
    inputs["quantities"].loc[1, "gap_cooling"] = np.nan
    out = _run(inputs, resample=False)
    owner = out[(out.GEOID == T2) & (out.group == "Owner")]
    assert owner["gap_cooling"].tolist() == [pytest.approx(20.0 / 3.0)] * 4
    inputs = _inputs()
    inputs["quantities"].loc[5, "gap_cooling"] = np.nan  # the only donor of type B
    out = _run(inputs, resample=False)
    assert out[out.group == "Renter"].empty
    both = out[(out.GEOID == T1) & (out.group == "all")]
    assert both["dwellings"].tolist() == [150.0] * 4  # rescaled to the total


def test_transfer_error_needs_groups_and_only_hits_borrowed_cells():
    with pytest.raises(ValueError, match="transfer_groups"):
        _run(transfer_tau={"heating": 0.1, "cooling": 0.1})
    inputs = _inputs()
    inputs["weights"]["source_level"] = ["puma", "county", "puma", "puma"]
    out = _run(
        inputs,
        n_draws=30,
        resample=False,
        transfer_tau={"heating": 0.0, "cooling": 0.2},
        transfer_groups=pd.Series({T1: "g1", T2: "g1"}),
    )
    borrowed = out[(out.GEOID == T1) & (out.group == "Renter")]
    own = out[(out.GEOID == T2) & (out.group == "Owner")]
    assert borrowed["cur_cooling"].std() > 0.1
    assert own["cur_cooling"].std() == pytest.approx(0.0, abs=1e-9)


def test_invalid_inputs_fail():
    inputs = _inputs()
    with pytest.raises(ValueError, match="one entry"):
        _run({**inputs, "uses": {"cur_cooling": "cooling"}})
    with pytest.raises(ValueError, match="'heating' or 'cooling'"):
        _run({**inputs, "uses": {**USES, "gap_cooling": "lighting"}})
    inputs["quantities"].loc[:, :] = np.nan
    with pytest.raises(ValueError, match="No cell"):
        _run(inputs)
    inputs = _inputs()
    inputs["tract_totals"] = inputs["tract_totals"].iloc[:1]
    with pytest.raises(ValueError, match="no total"):
        _run(inputs)


def _population():
    """Six counties in two PUMAs; the gap does not follow the climate line."""
    from test_known_population import _population as population

    dwellings, stations = population()
    cdd = dict(zip(stations["id"], stations["cdd"], strict=True))
    index = pd.Index(dwellings["building_id"], name="building_id")
    cur = dwellings["cooling_load_kbtu"].to_numpy()
    gap = 0.5 * dwellings["county"].map(cdd).to_numpy() / 100.0
    quantities = pd.DataFrame({"cur_cooling": cur, "gap_cooling": gap}, index=index)
    return dwellings, stations, quantities


def test_known_population_recovers_the_energy_that_follows_the_climate():
    dwellings, stations, quantities = _population()
    uses = {"cur_cooling": "cooling", "gap_cooling": "cooling"}
    table = gu.known_population_quantities(
        dwellings,
        quantities,
        uses,
        stations,
        climate_corrected=["cur_cooling", "gap_cooling"],
        unit="county",
        n_draws=40,
    )
    assert len(table) == 6
    for q in uses:
        assert f"truth_{q}" in table and f"no_climate_{q}" in table
        assert np.isfinite(table[f"estimate_{q}"]).all()
    # the current energy follows the climate line, so the correction recovers it
    rel = (table["estimate_cur_cooling"] - table["truth_cur_cooling"]) / table[
        "truth_cur_cooling"
    ]
    assert rel.abs().max() < 0.2
    assert (table["lo_gap_cooling"] <= table["hi_gap_cooling"]).all()


def test_transfer_deviation_is_the_excess_of_the_error_over_the_simulations():
    table = pd.DataFrame(
        {
            "dwellings": [100, 100, 10],
            "truth_a": [10.0, 10.0, 10.0],
            "estimate_a": [11.0, 9.0, 20.0],
            "lo_a": [10.0, 8.0, 15.0],
            "hi_a": [12.0, 10.0, 25.0],
        }
    )
    tau = gu.estimate_transfer_tau_quantities(table, ["a"], min_dwellings=30)
    rel2 = np.mean([0.1**2, 0.1**2])
    sd2 = np.mean([(2 / (2 * 1.645) / 11) ** 2, (2 / (2 * 1.645) / 9) ** 2])
    assert tau["a"] == pytest.approx(np.sqrt(rel2 - sd2))
    with pytest.raises(ValueError, match="enough"):
        gu.estimate_transfer_tau_quantities(table, ["a"], min_dwellings=1000)


def test_known_population_checks_the_unit_and_the_upgrade():
    dwellings, stations, quantities = _population()
    uses = {"cur_cooling": "cooling", "gap_cooling": "cooling"}
    with pytest.raises(ValueError, match="unit"):
        gu.known_population_quantities(dwellings, quantities, uses, stations, unit="x")
    mixed = dwellings.assign(upgrade=[0, 1] * (len(dwellings) // 2))
    with pytest.raises(ValueError, match="one upgrade"):
        gu.known_population_quantities(mixed, quantities, uses, stations)
