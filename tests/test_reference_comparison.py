"""Tests of the comparison of the modeled energy with RECS and LEAD."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import energy_requirement as er
from utah_ejpi import validation as val


def _recs_file(tmp_path, replicate_shift=0.0):
    """Write a small RECS file: 3 Utah households and one of another state."""
    rows = {
        "state_postal": ["UT", "UT", "UT", "CO"],
        "NWEIGHT": [100.0, 100.0, 200.0, 50.0],
        "KOWNRENT": [1, 2, 3, 1],
        "TYPEHUQ": [2, 5, 1, 2],
        "FUELHEAT": [1, 5, 2, 1],
        "TOTALBTUSPH": [34121.4, 6824.28, 17060.7, 1.0],
        "KWHCOL": [1000.0, 200.0, 400.0, 1.0],
        "ACEQUIPM_PUB": [1, 6, 3, 1],
    }
    frame = pd.DataFrame(rows)
    for i in range(1, 61):
        frame[f"NWEIGHT{i}"] = frame["NWEIGHT"]
    frame.loc[0, "NWEIGHT1"] = frame.loc[0, "NWEIGHT"] + replicate_shift
    path = tmp_path / "recs.csv"
    frame.to_csv(path, index=False)
    return path


def test_read_recs_keeps_the_state_and_the_replicate_weights(tmp_path):
    recs = val.read_recs(_recs_file(tmp_path), "UT")
    assert len(recs) == 3
    assert "NWEIGHT60" in recs.columns


def test_recs_means_use_the_jackknife_formula(tmp_path):
    recs = val.read_recs(_recs_file(tmp_path, replicate_shift=100.0), "UT")
    table = val.recs_means(recs, "heating_kwh").set_index(["group_type", "group"])
    # heating in kWh: 10000, 2000, 5000 with weights 100, 100, 200
    mean = (10000 * 100 + 2000 * 100 + 5000 * 200) / 400
    assert table.loc[("all", "all"), "mean"] == pytest.approx(mean)
    rep1 = (10000 * 200 + 2000 * 100 + 5000 * 200) / 500
    expected_se = np.sqrt(59 / 60 * (rep1 - mean) ** 2)
    assert table.loc[("all", "all"), "se"] == pytest.approx(expected_se)
    # a household that occupies without paying rent counts as a renter
    assert table.loc[("tenure", "Renter"), "n"] == 2
    assert table.loc[("structure", "mobile_home"), "mean"] == pytest.approx(5000.0)
    assert table.loc[("heating_fuel", "propane"), "n"] == 1


def _donor_frame():
    """Donor-level energy of two types with known consumption."""
    return pd.DataFrame(
        {
            "tenure": ["Owner", "Renter"],
            "building_type": ["1", "5 to 19"],
            "heating_fuel": ["Natural Gas", "Electricity"],
            "weight": [300.0, 100.0],
            "heating_kwh": [9000.0, 3000.0],
            "heating_kwh_uncorrected": [8000.0, 3000.0],
            "cooling_kwh": [1200.0, 400.0],
            "cooling_kwh_uncorrected": [1200.0, 400.0],
        }
    )


def _native_dwellings():
    return pd.DataFrame(
        {
            "occupied": [True, True, False],
            "tenure": ["Owner", "Renter", None],
            "building_type": ["1", "5 to 19", "1"],
            "heating_fuel": ["Natural Gas", "Electricity", "None"],
            "weight": [253.9, 253.9, 253.9],
            "heating_kwh_natural_gas": [7000.0, 0.0, 0.0],
            "heating_kwh_electricity": [500.0, 2500.0, 0.0],
            "heating_kwh_propane": [0.0, 0.0, 0.0],
            "heating_kwh_fuel_oil": [0.0, 0.0, 0.0],
            "cooling_kwh_electricity": [1000.0, 500.0, 0.0],
        }
    )


def test_comparison_with_recs_gives_ratio_interval_and_consistency(tmp_path):
    recs = val.read_recs(_recs_file(tmp_path, replicate_shift=100.0), "UT")
    out = val.compare_with_recs(
        recs, _native_dwellings(), _donor_frame(), "heating_kwh"
    )
    row = out[(out.group_type == "all")].iloc[0]
    assert row["tracts_corrected"] == pytest.approx((9000 * 300 + 3000 * 100) / 400)
    assert row["tracts_uncorrected"] == pytest.approx((8000 * 300 + 3000 * 100) / 400)
    assert row["resstock"] == pytest.approx((7500 + 2500) / 2)
    assert row["ratio"] == pytest.approx(row["tracts_corrected"] / row["mean"])
    assert row["low"] == pytest.approx(row["mean"] - 2.001 * row["se"])
    owner = out[(out.group_type == "tenure") & (out.group == "Owner")].iloc[0]
    assert owner["tracts_corrected"] == 9000.0
    gas = out[(out.group_type == "heating_fuel") & (out.group == "utility_gas")].iloc[0]
    assert gas["resstock"] == pytest.approx(7500.0)
    cooling = val.compare_with_recs(
        recs, _native_dwellings(), _donor_frame(), "cooling_kwh"
    )
    assert cooling[cooling.group_type == "all"].iloc[0]["resstock"] == pytest.approx(
        750.0
    )


LEAD_MEMBER = "UT AMI Census Tracts 2022.csv"


def _lead_zip(tmp_path):
    rows = []
    for geoid, elec, gas, fuel in (
        ("49001000100", 1000.0, 400.0, 10.0),
        ("49001000200", 1200.0, 600.0, 20.0),
        ("49001000300", 1500.0, 900.0, 5.0),
    ):
        for units in (1.0, 3.0):
            rows.append(
                {
                    "FIP": geoid,
                    "UNITS": units,
                    "ELEP*UNITS": elec * units,
                    "ELEP UNITS": units,
                    "GASP*UNITS": gas * units,
                    "GASP UNITS": units,
                    "FULP*UNITS": fuel * units,
                    "FULP UNITS": units,
                }
            )
    path = tmp_path / "lead.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(LEAD_MEMBER, pd.DataFrame(rows).to_csv(index=False))
    return path


def test_lead_tract_means_divide_the_sums_over_the_rows(tmp_path):
    out = val.read_lead_tract_means(_lead_zip(tmp_path), LEAD_MEMBER).set_index("GEOID")
    assert out.loc["49001000200", "electricity_usd"] == pytest.approx(1200.0)
    assert out.loc["49001000300", "households"] == pytest.approx(4.0)
    assert out.loc["49001000100", "other_fuel_usd"] == pytest.approx(10.0)


def test_lead_convergence_is_a_rank_correlation_among_tracts(tmp_path):
    lead = val.read_lead_tract_means(_lead_zip(tmp_path), LEAD_MEMBER)
    tracts = ["49001000100", "49001000200", "49001000300"]
    energy = pd.DataFrame(
        {
            "GEOID": tracts,
            "dwellings": [10.0, 20.0, 30.0],
            "heating_kwh_natural_gas": [5000.0, 7000.0, 9000.0],
            "heating_kwh_electricity": [100.0, 200.0, 300.0],
            "cooling_kwh_electricity": [50.0, 40.0, 30.0],
            "heating_kwh_propane": [30.0, 20.0, 10.0],
            "heating_kwh_fuel_oil": [0.0, 0.0, 0.0],
        }
    )
    out = val.lead_convergence(energy, lead).set_index("modeled")
    assert out.loc["heating_kwh_natural_gas", "spearman"] == pytest.approx(1.0)
    assert out.loc["electricity_kwh", "spearman"] == pytest.approx(1.0)
    # propane ranks 30, 20, 10 against the other-fuel spending 10, 20, 5
    assert out.loc["propane_fuel_oil_kwh", "spearman"] == pytest.approx(0.5)
    assert (out["tracts"] == 3).all()


def _energy_inputs():
    geoid = "49001000100"
    keys = {"tenure": "Owner", "vintage": "1980 to 1999", "building_type": "1"}
    weights = pd.DataFrame(
        {"GEOID": [geoid], **{k: [v] for k, v in keys.items()}, "weight": [80.0]}
    )
    donors = pd.DataFrame(
        {
            "GEOID": [geoid] * 2,
            **{k: [v] * 2 for k, v in keys.items()},
            "building_id": [1, 2],
            "donor_weight": [0.25, 0.75],
            "source_level": "puma",
        }
    )
    dwellings = pd.DataFrame(
        {
            "building_id": [1, 2],
            "upgrade": 0,
            "county": ["C1", "C1"],
            "occupants": [2.0, 4.0],
            "heating_fuel": ["Natural Gas", "Electricity"],
            "heating_load_kbtu": [100.0, 200.0],
            "cooling_load_kbtu": [10.0, 0.0],
            "heating_kwh_electricity": [0.0, 60.0],
            "heating_kwh_natural_gas": [30.0, 0.0],
            "heating_kwh_propane": [0.0, 0.0],
            "heating_kwh_fuel_oil": [0.0, 0.0],
            "cooling_kwh_electricity": [3.0, 0.0],
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
            "id": ["C1"],
            "kind": "station",
            "hdd": [2000.0],
            "cdd": [100.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    return weights, donors, dwellings, tract, stations


def test_donor_level_energy_corrects_load_and_consumption_with_the_same_factor():
    weights, donors, dwellings, tract, stations = _energy_inputs()
    heating = er.DegreeDaySensitivity("heating", 0.0, 1.0, np.eye(2), 10, 1, 1.0, 5.0)
    cooling = er.DegreeDaySensitivity("cooling", 0.0, 1.0, np.eye(2), 10, 1, 0.0, 1.0)
    donor = er.donor_level_energy(
        weights, donors, dwellings, tract, stations, heating, cooling
    )
    assert donor["weight"].tolist() == [20.0, 60.0]
    assert donor["factor_heating"].tolist() == [2.0, 2.0]
    assert donor["heating_kwh_natural_gas"].tolist() == [60.0, 0.0]
    assert donor["heating_kwh"].tolist() == [60.0, 120.0]
    assert donor["heating_kwh_uncorrected"].tolist() == [30.0, 60.0]
    assert donor["cooling_kwh"].tolist() == [6.0, 0.0]
    tract_type = er.build_tract_type_energy(
        weights, donors, dwellings, tract, stations, heating, cooling
    )
    row = tract_type.iloc[0]
    assert row["heating_kwh"] == pytest.approx(0.25 * 60 + 0.75 * 120)
    assert row["heating_kwh_electricity"] == pytest.approx(0.75 * 120)
    assert row["heating_kwh_uncorrected"] == pytest.approx(0.25 * 30 + 0.75 * 60)
    assert row["heating_kbtu"] == pytest.approx(0.25 * 200 + 0.75 * 400)


def test_recs_share_uses_the_weights_and_the_jackknife_formula(tmp_path):
    recs = val.read_recs(_recs_file(tmp_path, replicate_shift=100.0), "UT")
    share, se = val.recs_share(recs, "ACEQUIPM_PUB", 6)
    assert share == pytest.approx(100.0 / 400.0)
    rep1 = 100.0 / 500.0
    assert se == pytest.approx(np.sqrt(59 / 60 * (rep1 - share) ** 2))
    both, _ = val.recs_share(recs, "ACEQUIPM_PUB", (3, 6))
    assert both == pytest.approx(300.0 / 400.0)
    with pytest.raises(KeyError):
        val.recs_share(recs, "NOT_A_COLUMN", 1)
