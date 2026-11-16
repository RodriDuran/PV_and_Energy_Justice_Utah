"""Tests of the leave-one-unit-out test with a known population."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import uncertainty as un

TYPES = [
    ("Owner", "1980 to 1999", "1", 100.0),
    ("Renter", "1960 to 1979", "5 to 19", 50.0),
]
HDD = [1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
CDD = [400.0, 300.0, 250.0, 150.0, 100.0, 50.0]


def _population():
    """Six counties in two PUMAs whose loads follow a line of the degree days."""
    rows, k = [], 0
    for c in range(6):
        for tenure, vintage, kind, scale in TYPES:
            for _ in range(6):
                k += 1
                rows.append(
                    {
                        "building_id": k,
                        "upgrade": 0,
                        "occupied": True,
                        "county": f"C{c}",
                        "puma": "P0" if c < 3 else "P1",
                        "tenure": tenure,
                        "vintage": vintage,
                        "building_type": kind,
                        "bedrooms": 3,
                        "floor_area": "1500-1999",
                        "occupants": 2.0,
                        "heating_load_kbtu": scale * (0.2 + HDD[c] / 1000.0),
                        "cooling_load_kbtu": scale * (1.0 + 0.5 * CDD[c] / 1000.0),
                    }
                )
    stations = pd.DataFrame(
        {
            "id": [f"C{c}" for c in range(6)],
            "kind": "station",
            "hdd": HDD,
            "cdd": CDD,
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    return pd.DataFrame(rows), stations


def test_climate_correction_recovers_the_truth_of_a_unit_left_out():
    dwellings, stations = _population()
    table = un.known_population_test(dwellings, stations, unit="county", n_draws=50)
    assert len(table) == 6
    assert (table["dwellings"] == 12).all()
    for use in ("heating", "cooling"):
        rel = (table[f"estimate_{use}"] - table[f"truth_{use}"]) / table[f"truth_{use}"]
        assert rel.abs().max() < 0.2
        # the correction explains most of what the type alone cannot
        raw = (table[f"no_climate_{use}"] - table[f"truth_{use}"]).abs().mean()
        corrected = (table[f"estimate_{use}"] - table[f"truth_{use}"]).abs().mean()
        assert corrected < raw
    shares = table[[c for c in table.columns if c.startswith("share_")]].sum(axis=1)
    assert shares.tolist() == pytest.approx([1.0] * 6)


def test_a_unit_can_be_a_puma_and_the_other_dwellings_are_borrowed():
    dwellings, stations = _population()
    table = un.known_population_test(dwellings, stations, unit="puma", n_draws=30)
    assert table["unit"].tolist() == ["P0", "P1"]
    # the dwellings of the PUMA left out cannot be used, so the PUMA level is empty
    assert table["share_puma"].tolist() == pytest.approx([0.0, 0.0])
    assert (table["lo_heating"] <= table["hi_heating"]).all()


def test_summary_gives_error_measures_and_coverage_of_the_study_estimate():
    dwellings, stations = _population()
    table = un.known_population_test(dwellings, stations, unit="county", n_draws=50)
    summary = un.known_population_summary(table)
    assert len(summary) == 6
    estimate = summary[
        (summary.use == "heating") & (summary.estimator == "estimate")
    ].iloc[0]
    assert 0.0 <= estimate["coverage"] <= 1.0
    assert estimate["rrmse"] >= abs(estimate["bias"])
    benchmark = summary[summary.estimator == "state"]
    assert benchmark["coverage"].isna().all()
    manual = (table["truth_heating"] - table["estimate_heating"]).abs().median()
    assert estimate["median_abs_error"] == pytest.approx(
        (manual / table["truth_heating"]).median(), rel=0.5
    )


def test_invalid_unit_or_mixed_upgrades_fail():
    dwellings, stations = _population()
    with pytest.raises(ValueError, match="unit must be"):
        un.known_population_test(dwellings, stations, unit="tract")
    mixed = dwellings.assign(upgrade=np.r_[0, np.full(len(dwellings) - 1, 4)])
    with pytest.raises(ValueError, match="one upgrade"):
        un.known_population_test(mixed, stations)


def test_transfer_tau_is_the_error_not_explained_by_the_simulations():
    table = pd.DataFrame(
        {
            "dwellings": [40, 40, 5],
            "truth_heating": [100.0, 100.0, 100.0],
            "estimate_heating": [110.0, 90.0, 300.0],
            "lo_heating": [105.0, 95.0, 0.0],
            "hi_heating": [115.0, 105.0, 0.0],
            "truth_cooling": [100.0, 100.0, 100.0],
            "estimate_cooling": [100.0, 100.0, 100.0],
            "lo_cooling": [80.0, 80.0, 0.0],
            "hi_cooling": [120.0, 120.0, 0.0],
        }
    )
    tau = un.estimate_transfer_tau(table)
    # relative error 0.1 in both units; simulated sd = 10 / (2 * 1.645) / estimate
    sd = np.array([10 / 3.29 / 110, 10 / 3.29 / 90])
    assert tau["heating"] == pytest.approx(np.sqrt(0.01 - (sd**2).mean()))
    # the small unit is left out; an exact estimate leaves nothing to explain
    assert tau["cooling"] == 0.0
    with pytest.raises(ValueError, match="enough dwellings"):
        un.estimate_transfer_tau(table, min_dwellings=100)


def test_transfer_error_only_widens_the_simulations_of_borrowed_cells():
    dwellings, stations = _population()
    tau = {"heating": 0.3, "cooling": 0.3}
    plain = un.known_population_test(dwellings, stations, unit="puma", n_draws=100)
    wide = un.known_population_test(
        dwellings, stations, unit="puma", n_draws=100, transfer_tau=tau
    )
    # every dwelling of a PUMA left out is borrowed, so the interval widens
    for use in ("heating", "cooling"):
        w_plain = plain[f"hi_{use}"] - plain[f"lo_{use}"]
        w_wide = wide[f"hi_{use}"] - wide[f"lo_{use}"]
        assert (w_wide > w_plain).all()
        # the point estimate does not depend on the simulations
        assert wide[f"estimate_{use}"].tolist() == pytest.approx(
            plain[f"estimate_{use}"].tolist()
        )


def test_aggregate_draws_weights_the_tracts_by_their_dwellings():
    draws = pd.DataFrame(
        {
            "GEOID": ["a", "b", "c"] * 2,
            "group": "all",
            "draw": [0, 0, 0, 1, 1, 1],
            "dwellings": [100.0, 300.0, 50.0] * 2,
            "heating_kbtu": [10.0, 20.0, 99.0, 12.0, 24.0, 99.0],
        }
    )
    units = pd.Series({"a": "u", "b": "u"})  # tract c has no unit
    out = un.aggregate_draws(draws, units, ["heating_kbtu"])
    assert out["GEOID"].tolist() == ["u", "u"]
    assert out["heating_kbtu"].tolist() == pytest.approx([17.5, 21.0])
    assert out["dwellings"].tolist() == pytest.approx([400.0, 400.0])
    summary = un.summarize_draws(out, ["heating_kbtu"])
    assert summary["heating_kbtu"].iloc[0] == pytest.approx(19.25)


def test_dwellings_of_other_states_enter_the_fit_of_the_line_but_are_not_donors():
    dwellings, stations = _population()
    other = dwellings.iloc[:24].copy()
    other["county"] = "D0"
    other["puma"] = "Q0"
    other["building_id"] = other["building_id"] + 1000
    other["heating_load_kbtu"] = other["heating_load_kbtu"] * 2.0
    extra_stations = pd.concat(
        [
            stations,
            pd.DataFrame(
                {
                    "id": ["D0"],
                    "kind": "station",
                    "hdd": [4500.0],
                    "cdd": [0.0],
                    "hdd_base_c": 18.0,
                    "cdd_base_c": 26.0,
                }
            ),
        ],
        ignore_index=True,
    )
    plain = un.known_population_test(dwellings, stations, unit="county", n_draws=20)
    pooled = un.known_population_test(
        dwellings,
        stations,
        unit="county",
        n_draws=20,
        extra_fit_dwellings={"heating": other},
        extra_fit_stations=extra_stations,
    )
    assert pooled["unit"].tolist() == plain["unit"].tolist()
    assert (pooled["dwellings"] == plain["dwellings"]).all()
    assert not np.allclose(pooled["estimate_heating"], plain["estimate_heating"])
    assert np.allclose(pooled["estimate_cooling"], plain["estimate_cooling"])
    with pytest.raises(ValueError, match="extra_fit_stations"):
        un.known_population_test(
            dwellings,
            stations,
            unit="county",
            n_draws=20,
            extra_fit_dwellings={"heating": other},
        )
