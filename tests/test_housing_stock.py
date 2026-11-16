"""Tests of the choice of modeled dwellings that stand for each type in a tract."""

import pandas as pd
import pytest

from utah_ejpi import housing_stock as hs
from utah_ejpi.utils import validate_table

T1, T2, T3 = "49001000100", "49001000200", "49001000300"
A = ("Owner", "1980 to 1999", "1")
B = ("Owner", "2000 to 2019", "1")
C = ("Renter", "1960 to 1979", "5 to 19")
MOBILE_OLD = ("Renter", "1940 to 1959", "mobile")
NEWEST = ("Owner", "2020 or later", "1")


def _dwellings():
    """Modeled dwellings: type A in two PUMAs, B and C in one PUMA each."""
    rows = [
        (1, "P1", "C1", A),
        (2, "P1", "C1", A),
        (3, "P3", "C2", A),
        (4, "P2", "C1", B),
        (5, "P3", "C2", C),
        (6, "P3", "C2", ("Renter", "1960 to 1979", "mobile")),
        (7, "P3", "C2", ("Renter", "1980 to 1999", "mobile")),
    ]
    return pd.DataFrame(
        [
            {
                "building_id": i,
                "puma": p,
                "county": c,
                "occupied": True,
                "tenure": t[0],
                "vintage": t[1],
                "building_type": t[2],
            }
            for i, p, c, t in rows
        ]
    )


def _counts(geoid, types):
    return pd.DataFrame(
        [
            {
                "GEOID": geoid,
                "tenure": t[0],
                "vintage": t[1],
                "building_type": t[2],
                "estimate": 10.0 * (k + 1),
                "moe": 3.0,
            }
            for k, t in enumerate(types)
        ]
    )


SHARES = pd.DataFrame(
    {
        "GEOID": [T1, T2, T2, T3, T3],
        "resstock_puma": ["P1", "P1", "P3", "P2", "P1"],
        "share": [1.0, 0.7, 0.3, 0.6, 0.4],
        "occupied_units": [100.0, 70.0, 30.0, 60.0, 40.0],
    }
)
MASTER = pd.DataFrame({"GEOID": [T1, T2, T3], "resstock_county": ["C1", "C1", "C1"]})
RANK = pd.DataFrame(
    {
        "GEOID": [T1] * 3 + [T2] * 3 + [T3] * 3,
        "resstock_puma": ["P1", "P2", "P3"] * 3,
        "rank": [1, 2, 3, 1, 3, 2, 2, 1, 3],
        "distance": [0.0, 1.0, 2.0] * 3,
    }
)


def _run(types, geoid=T1, min_donors=1):
    return hs.build_type_donors(
        _counts(geoid, types), _dwellings(), SHARES, MASTER, RANK, min_donors
    )


def _donors_of(donors, key):
    sel = donors[
        (donors.tenure == key[0])
        & (donors.vintage == key[1])
        & (donors.building_type == key[2])
    ].set_index("building_id")
    return sel.donor_weight.to_dict()


def test_each_level_of_the_fallback_is_used_in_order():
    weights, donors = _run([A, B, C, MOBILE_OLD])
    levels = weights.set_index(["tenure", "vintage", "building_type"]).source_level
    assert levels[A] == "puma"
    assert levels[B] == "county"
    assert levels[C] == "similar_climate_puma"
    assert levels[MOBILE_OLD] == "nearest_vintage"
    assert _donors_of(donors, A) == {1: 0.5, 2: 0.5}
    assert _donors_of(donors, B) == {4: 1.0}
    assert _donors_of(donors, C) == {5: 1.0}
    # nearest class to 1940 to 1959 with this tenure and type: 1960 to 1979
    assert _donors_of(donors, MOBILE_OLD) == {6: 1.0}
    validate_table(weights, "tract_type_weights")
    validate_table(donors, "tract_type_donors")


def test_tract_in_two_pumas_pools_donors_by_share():
    weights, donors = _run([A], geoid=T2)
    got = _donors_of(donors, A)
    assert got == pytest.approx({1: 0.35, 2: 0.35, 3: 0.3})
    assert weights.source_level.tolist() == ["puma"]


def test_main_puma_without_donors_sends_the_type_to_the_county():
    weights, donors = _run([A], geoid=T3)
    assert weights.source_level.tolist() == ["county"]
    assert _donors_of(donors, A) == {1: 0.5, 2: 0.5}


def test_newest_class_is_represented_by_the_class_2000_to_2019():
    weights, donors = _run([NEWEST])
    row = weights.iloc[0]
    assert bool(row.built_2020_or_later) is True
    assert row.vintage == "2020 or later"
    assert _donors_of(donors, NEWEST) == {4: 1.0}


def test_minimum_number_of_donors_moves_types_to_wider_levels():
    weights, donors = _run([A, B], min_donors=2)
    levels = weights.set_index(["tenure", "vintage", "building_type"]).source_level
    assert levels[A] == "puma"
    # one donor in the county and in the climate-ordered PUMAs: the nearest
    # class (1980 to 1999) supplies the three donors of type A
    assert levels[B] == "nearest_vintage"
    assert _donors_of(donors, B) == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})


def test_a_type_that_exists_nowhere_fails():
    with pytest.raises(ValueError, match="No modeled dwelling"):
        _run([("Renter", "1960 to 1979", "50 or more")])


def test_repeated_buildings_are_refused():
    twice = pd.concat([_dwellings(), _dwellings()])
    with pytest.raises(ValueError, match="repeated buildings"):
        hs.build_type_donors(_counts(T1, [A]), twice, SHARES, MASTER, RANK)


def test_fallback_shares_report_dwellings_by_level():
    weights, _ = _run([A, B, C, NEWEST])
    out = hs.fallback_shares(weights).iloc[0]
    # counts 10, 20, 30, 40: A at puma, B and NEWEST at county, C at climate
    assert out["puma"] == pytest.approx(0.1)
    assert out["county"] == pytest.approx(0.6)
    assert out["similar_climate_puma"] == pytest.approx(0.3)
    assert out["nearest_vintage"] == 0.0
    assert out["built_2020_or_later"] == pytest.approx(0.4)


def test_vintage_candidates_start_from_the_nearest_classes():
    assert hs._vintage_candidates("1960 to 1979")[0] == [
        "1940 to 1959",
        "1980 to 1999",
    ]
    assert hs._vintage_candidates("2020 or later")[0] == ["1980 to 1999"]
    assert hs._vintage_candidates("1939 or earlier")[0] == ["1940 to 1959"]


def test_the_72_cells_of_the_census_table_are_complete():
    cells = hs.b25127_cells()
    assert len(cells) == 72
    assert cells.groupby(["tenure", "vintage", "building_type"]).size().eq(1).all()
    assert cells.tenure.value_counts().to_dict() == {"Owner": 36, "Renter": 36}
    assert set(cells.vintage) == set(hs.VINTAGE_ORDER) | {"2020 or later"}


def test_acs_counts_are_built_from_the_long_table_and_checked():
    cells = hs.b25127_cells()
    rows = []
    for geoid in ["49001000100", "49001000200", "08001000100"]:
        for k, var in enumerate(cells.variable):
            rows.append((geoid, "x", var, float(k), 2.0))
    table = pd.DataFrame(rows, columns=["GEOID", "NAME", "variable", "estimate", "moe"])
    # subtotal rows of the table are ignored
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [("49001000100", "x", "B25127_001", 999.0, 5.0)], columns=table.columns
            ),
        ]
    )
    out = hs.acs_type_counts(table, "49")
    validate_table(out, "acs_type_counts")
    assert out.GEOID.nunique() == 2
    assert len(out) == 144
    with pytest.raises(ValueError, match="do not have the 72 cells"):
        hs.acs_type_counts(table.iloc[10:], "49")
    broken = table.copy()
    broken.loc[broken.index[0], "estimate"] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        hs.acs_type_counts(broken.reset_index(drop=True), "49")
