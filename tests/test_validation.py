"""Tests of the comparison of the chosen dwellings with Census compositions."""

import pandas as pd
import pytest

from utah_ejpi import validation as val

T = "49001000100"
A = ("Owner", "1980 to 1999", "1")
B = ("Owner", "2000 to 2019", "1")


def _weights():
    return pd.DataFrame(
        [
            {
                "GEOID": T,
                "tenure": a[0],
                "vintage": a[1],
                "building_type": a[2],
                "weight": w,
            }
            for a, w in ((A, 100.0), (B, 50.0))
        ]
    )


def _donors():
    rows = [(A, 1, 0.5), (A, 2, 0.5), (B, 3, 1.0)]
    return pd.DataFrame(
        [
            {
                "GEOID": T,
                "tenure": k[0],
                "vintage": k[1],
                "building_type": k[2],
                "building_id": i,
                "donor_weight": w,
                "source_level": "puma",
            }
            for k, i, w in rows
        ]
    )


def _dwellings():
    return pd.DataFrame(
        {
            "building_id": [1, 2, 3],
            "heating_fuel": ["Natural Gas", "Electricity", "Fuel Oil"],
            "bedrooms": [3, 1, 5],
            "occupants": [2.0, 4.0, 7.0],
            "cooling_type": ["Central AC", "None", "Room AC"],
        }
    )


def test_every_class_of_the_census_tables_is_listed():
    cells = val.composition_classes()
    assert len(cells) == 44
    assert cells.groupby("table").size().to_dict() == {
        "bedrooms": 12,
        "heating_fuel": 18,
        "household_size": 14,
    }
    bedrooms = cells[cells.table == "bedrooms"]
    assert set(bedrooms["class"]) == {"0 or 1", "2", "3", "4", "5 or more"}
    assert set(cells[cells.table == "heating_fuel"]["class"]) == {
        "utility_gas",
        "propane",
        "electricity",
        "wood",
        "other",
    }


def test_resstock_classes_follow_the_census_classes():
    d = _dwellings()
    assert val.resstock_class(d, "heating_fuel").tolist() == [
        "utility_gas",
        "electricity",
        "other",
    ]
    assert val.resstock_class(d, "bedrooms").tolist() == ["3", "0 or 1", "5 or more"]
    assert val.resstock_class(d, "household_size").tolist() == ["2", "4", "7 or more"]
    with pytest.raises(KeyError):
        val.resstock_class(d, "rooms")


def test_implied_composition_distributes_each_type_by_donor_weight():
    out = val.implied_composition(_weights(), _donors(), _dwellings(), "heating_fuel")
    got = out.set_index("class")["implied"].to_dict()
    assert got == pytest.approx(
        {"utility_gas": 50.0, "electricity": 50.0, "other": 50.0}
    )
    assert out["implied"].sum() == pytest.approx(150.0)


def test_observed_composition_adds_census_categories_into_classes():
    acs = pd.DataFrame(
        {
            "GEOID": [T] * 5 + ["08001000100"],
            # owner: utility gas, electricity, fuel oil, no fuel used; another state
            "variable": [
                "B25117_003",
                "B25117_005",
                "B25117_006",
                "B25117_011",
                "B25117_002",
                "B25117_003",
            ],
            "estimate": [90.0, 60.0, 3.0, 2.0, 999.0, 7.0],
        }
    )
    observed = val.observed_composition(acs, "heating_fuel", "49")
    out = observed.set_index("class")["observed"]
    # the subtotal B25117_002 and the other state are ignored; two categories
    # form the class "other"
    assert out.to_dict() == {"utility_gas": 90.0, "electricity": 60.0, "other": 5.0}


def test_errors_compare_the_mix_after_scaling_to_the_census_total():
    implied = val.implied_composition(
        _weights(), _donors(), _dwellings(), "heating_fuel"
    )
    observed = pd.DataFrame(
        {
            "GEOID": [T] * 3,
            "tenure": ["Owner"] * 3,
            "class": ["utility_gas", "electricity", "other"],
            "observed": [60.0, 60.0, 60.0],
        }
    )
    by_tract, by_class = val.composition_errors(implied, observed)
    # implied 50 / 50 / 50 (150) scaled to 60 / 60 / 60 (180): no difference
    assert by_tract.loc[0, "tae"] == pytest.approx(0.0)
    observed.loc[0, "observed"] = 90.0
    observed.loc[2, "observed"] = 30.0
    by_tract, by_class = val.composition_errors(implied, observed)
    # scaled implied: 60 each; errors 30 + 0 + 30 over 180 dwellings
    assert by_tract.loc[0, "tae"] == pytest.approx(60.0)
    assert by_tract.loc[0, "tae_share"] == pytest.approx(60.0 / 180.0)
    gas = by_class.set_index("class").loc["utility_gas"]
    assert gas["difference"] == pytest.approx(-30.0)
    assert gas["apd"] == pytest.approx(30.0 / 90.0)


def test_air_conditioning_share_is_compared_with_the_census_estimate():
    lace = pd.DataFrame(
        {
            "STATE": [49],
            "COUNTY": [1],
            "TRACT": [100],
            "HSE_OCC_E": [150.0],
            "AC_E": [120.0],
        }
    )
    out = val.air_conditioning_comparison(_weights(), _donors(), _dwellings(), lace)
    # type A: half of its donors are cooled, type B all: (50 + 50) / 150
    assert out.loc[0, "implied_share"] == pytest.approx(100.0 / 150.0)
    assert out.loc[0, "observed_share"] == pytest.approx(0.8)
    assert out.loc[0, "difference"] == pytest.approx(100.0 / 150.0 - 0.8)
