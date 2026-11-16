"""Tests of the reweighting of the modeled dwellings to several Census tables."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import energy_requirement as er
from utah_ejpi import housing_stock as hs
from utah_ejpi.utils import validate_table

T = "49001000100"
OWNER_1 = ("Owner", "1980 to 1999", "1")
RENTER_APT = ("Renter", "1960 to 1979", "5 to 19")


def test_raking_reproduces_the_classic_two_way_solution():
    # four cells of a 2 x 2 table with a uniform seed: the fitted table is the
    # product of the margins divided by the total
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    w, iterations, converged, error, bad, worst = hs._rake(
        np.ones(4),
        [rows, cols],
        [np.array([60.0, 40.0]), np.array([70.0, 30.0])],
        tolerance=1e-9,
        max_iterations=50,
    )
    assert converged
    assert w.tolist() == pytest.approx([42.0, 18.0, 28.0, 12.0])
    assert not any(b.any() for b in bad)


def test_raking_reports_categories_that_cannot_be_matched_and_ignored_targets():
    cat = np.array([0, 0, 1])
    target = np.array([10.0, 5.0, np.nan])
    # category 2 has a positive target and no dwelling; category 3 is not imposed
    cat2 = np.array([0, 1, 1])
    target2 = np.array([4.0, 6.0, 3.0])
    w, _, _, _, bad, _ = hs._rake(np.ones(3), [cat, cat2], [target, target2], 1e-6, 100)
    assert bad[1].tolist() == [False, False, True]
    assert bad[0].tolist() == [False, False, False]


def _stock(extra=()):
    rows = [
        (1, "P1", "C1", OWNER_1, "Natural Gas", 3, 2.0),
        (2, "P1", "C1", OWNER_1, "Electricity", 3, 2.0),
        (3, "P1", "C1", RENTER_APT, "Natural Gas", 2, 2.0),
        (4, "P2", "C2", OWNER_1, "Natural Gas", 3, 2.0),
        *extra,
    ]
    return pd.DataFrame(
        [
            {
                "building_id": i,
                "upgrade": 0,
                "occupied": True,
                "puma": p,
                "county": c,
                "tenure": k[0],
                "vintage": k[1],
                "building_type": k[2],
                "heating_fuel": fuel,
                "bedrooms": bed,
                "occupants": occ,
            }
            for i, p, c, k, fuel, bed, occ in rows
        ]
    )


def _tables(type_counts, fuel_counts):
    counts = pd.DataFrame(
        [
            {
                "GEOID": T,
                "tenure": k[0],
                "vintage": k[1],
                "building_type": k[2],
                "estimate": n,
                "moe": 1.0,
            }
            for k, n in type_counts.items()
        ]
    )
    cells = hs.b25127_cells()
    full = cells[["tenure", "vintage", "building_type"]].merge(
        counts, how="left", on=["tenure", "vintage", "building_type"]
    )
    full["GEOID"] = T
    full["estimate"] = full["estimate"].fillna(0.0)
    full["moe"] = full["moe"].fillna(0.0)
    fuel = pd.DataFrame(
        [
            {"GEOID": T, "tenure": t, "class": c, "observed": n}
            for (t, c), n in fuel_counts.items()
        ]
    )
    shares = pd.DataFrame(
        {
            "GEOID": [T],
            "resstock_puma": ["P1"],
            "share": [1.0],
            "occupied_units": [150.0],
        }
    )
    master = pd.DataFrame({"GEOID": [T], "resstock_county": ["C1"]})
    rank = pd.DataFrame(
        {
            "GEOID": [T, T],
            "resstock_puma": ["P2", "P1"],
            "rank": [1, 2],
            "distance": [0.0, 1.0],
        }
    )
    return full, {"heating_fuel": fuel}, shares, master, rank


def test_weights_match_every_table_and_favour_the_dwellings_near_the_tract():
    counts, comp, shares, master, rank = _tables(
        {OWNER_1: 100.0, RENTER_APT: 50.0},
        {
            ("Owner", "utility_gas"): 60.0,
            ("Owner", "electricity"): 40.0,
            ("Renter", "utility_gas"): 50.0,
        },
    )
    w, diag = hs.reweight_tracts(
        counts,
        comp,
        _stock(),
        shares,
        master,
        rank,
        sets=("type", "heating_fuel"),
        min_weight_share=0.0,
    )
    validate_table(w, "tract_dwelling_weights")
    validate_table(diag, "reweighting_diagnostics")
    got = w[~w.copy_2020].set_index("building_id").weight
    assert got.sum() == pytest.approx(150.0, rel=0.01)
    assert got[2] == pytest.approx(40.0, rel=0.01)
    assert got[3] == pytest.approx(50.0, rel=0.01)
    # the gas owners are split between the dwelling of the PUMA (weight 1) and the
    # one of the PUMA of closest climate in another county (weight 0.01)
    assert got[1] == pytest.approx(60.0 / 1.01, rel=0.01)
    assert got[4] == pytest.approx(60.0 * 0.01 / 1.01, rel=0.01)
    row = diag.iloc[0]
    assert bool(row.converged)
    assert row.tier_1_share > 0.99
    assert row.infeasible_categories == 0
    assert 1.0 < row.effective_n < 4.0


def test_a_category_without_dwellings_is_reported_and_small_weights_are_pruned():
    counts, comp, shares, master, rank = _tables(
        {OWNER_1: 100.0, RENTER_APT: 50.0},
        {("Owner", "utility_gas"): 100.0, ("Renter", "propane"): 50.0},
    )
    w, diag = hs.reweight_tracts(
        counts,
        comp,
        _stock(),
        shares,
        master,
        rank,
        sets=("type", "heating_fuel"),
        min_weight_share=0.01,
    )
    assert diag.iloc[0].infeasible_categories >= 1
    assert diag.iloc[0].infeasible_share > 0
    # the dwelling of another PUMA gets less than one dwelling and is dropped
    assert 4 not in w.building_id.tolist()
    assert diag.iloc[0].pruned_share > 0


def test_dwellings_of_2000_to_2019_also_stand_for_the_newest_class():
    newest = ("Owner", "2020 or later", "1")
    older = ("Owner", "2000 to 2019", "1")
    counts, comp, shares, master, rank = _tables({OWNER_1: 80.0, newest: 20.0}, {})
    stock = _stock(extra=[(5, "P1", "C1", older, "Natural Gas", 3, 2.0)])
    w, diag = hs.reweight_tracts(
        counts, comp, stock, shares, master, rank, sets=("type",), min_weight_share=0.0
    )
    copy = w[w.copy_2020].set_index("building_id").weight
    assert copy.sum() == pytest.approx(20.0)
    assert copy.index.tolist() == [5]
    original = w[~w.copy_2020].set_index("building_id").weight
    assert original.get(5, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert bool(diag.iloc[0].converged)


def test_composition_tables_are_scaled_to_the_total_of_each_tenure():
    counts, comp, shares, master, rank = _tables(
        {OWNER_1: 100.0, RENTER_APT: 50.0},
        # the fuel table sums to 200 owners and 25 renters: it is rescaled
        {
            ("Owner", "utility_gas"): 120.0,
            ("Owner", "electricity"): 80.0,
            ("Renter", "utility_gas"): 25.0,
        },
    )
    w, _ = hs.reweight_tracts(
        counts,
        comp,
        _stock(),
        shares,
        master,
        rank,
        sets=("type", "heating_fuel"),
        min_weight_share=0.0,
    )
    owners = w[w.building_id.isin([1, 2, 4])].weight.sum()
    assert owners == pytest.approx(100.0, rel=0.01)
    assert w[w.building_id == 2].weight.iloc[0] == pytest.approx(40.0, rel=0.01)


def test_invalid_sets_and_repeated_buildings_fail():
    counts, comp, shares, master, rank = _tables({OWNER_1: 10.0}, {})
    with pytest.raises(ValueError, match="prefix"):
        hs.reweight_tracts(
            counts, comp, _stock(), shares, master, rank, sets=("heating_fuel",)
        )
    with pytest.raises(ValueError, match="repeated"):
        hs.reweight_tracts(
            counts,
            comp,
            pd.concat([_stock(), _stock()]),
            shares,
            master,
            rank,
            sets=("type",),
        )


def test_energy_of_reweighted_dwellings_uses_the_weights_and_the_climate_factor():
    weights = pd.DataFrame(
        {
            "GEOID": [T, T],
            "building_id": [1, 2],
            "copy_2020": [False, False],
            "weight": [30.0, 10.0],
            "tier": [1, 1],
        }
    )
    dwellings = (
        _stock()
        .iloc[:2]
        .assign(
            heating_load_kbtu=[100.0, 300.0],
            cooling_load_kbtu=[10.0, 0.0],
        )
    )
    tract = pd.DataFrame(
        {
            "id": [T],
            "kind": "tract",
            "hdd": [4000.0],
            "cdd": [100.0],
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
    heating = er.DegreeDaySensitivity("heating", 0.0, 1.0, np.eye(2), 10, 1, 1.0, 5.0)
    cooling = er.DegreeDaySensitivity("cooling", 0.0, 1.0, np.eye(2), 10, 1, 0.0, 1.0)
    frame = er.weighted_dwelling_energy(
        weights, dwellings, tract, stations, heating, cooling
    )
    means = er.weighted_tract_means(frame).iloc[0]
    assert means["dwellings"] == 40.0
    assert means["heating_kbtu"] == pytest.approx((30 * 200 + 10 * 600) / 40)
    assert means["heating_kbtu_uncorrected"] == pytest.approx(
        (30 * 100 + 10 * 300) / 40
    )
    assert means["cooling_kbtu"] == pytest.approx(30 * 10 / 40)
    assert means["persons"] == pytest.approx(80.0)
    assert means["extrapolated_heating"] == 0.0


def test_a_type_without_dwellings_does_not_make_the_tables_contradict_each_other():
    missing = ("Owner", "1960 to 1979", "1")
    counts, comp, shares, master, rank = _tables(
        {OWNER_1: 100.0, missing: 20.0, RENTER_APT: 50.0},
        {
            ("Owner", "utility_gas"): 70.0,
            ("Owner", "electricity"): 50.0,
            ("Renter", "utility_gas"): 50.0,
        },
    )
    w, diag = hs.reweight_tracts(
        counts,
        comp,
        _stock(),
        shares,
        master,
        rank,
        sets=("type", "heating_fuel"),
        min_weight_share=0.0,
    )
    row = diag.iloc[0]
    assert row.infeasible_categories >= 1
    # the 20 owners of the type that no modeled dwelling has cannot be represented,
    # so the fuel table is scaled to the 100 owners that can
    assert bool(row.converged)
    owners = w[w.building_id.isin([1, 2, 4])].weight.sum()
    assert owners == pytest.approx(100.0, rel=0.01)
    assert w[w.building_id == 2].weight.iloc[0] == pytest.approx(
        100.0 * 50 / 120, rel=0.02
    )


def test_composition_shares_scale_each_table_to_the_tenure_totals_of_the_types():
    counts, comp, _, _, _ = _tables(
        {OWNER_1: 100.0, RENTER_APT: 50.0},
        {
            ("Owner", "utility_gas"): 120.0,
            ("Owner", "electricity"): 80.0,
            ("Renter", "utility_gas"): 25.0,
        },
    )
    shares = hs.composition_shares(counts, comp, ("heating_fuel",))
    row = shares.loc[T]
    assert row["heating_fuel:Owner|utility_gas"] == pytest.approx(60.0 / 150.0)
    assert row["heating_fuel:Owner|electricity"] == pytest.approx(40.0 / 150.0)
    assert row["heating_fuel:Renter|utility_gas"] == pytest.approx(50.0 / 150.0)
    assert row.sum() == pytest.approx(1.0)
    empty = counts.assign(estimate=0.0)
    assert hs.composition_shares(empty, comp, ("heating_fuel",)).empty


def test_class_indicators_give_each_dwelling_one_class_per_table():
    ind = hs.class_indicators(_stock(), ("heating_fuel", "bedrooms"))
    assert ind.index.tolist() == [1, 2, 3, 4]
    assert (ind.sum(axis=1) == 2).all()
    assert ind.loc[2, "heating_fuel:Owner|electricity"] == 1.0
    assert ind.loc[3, "bedrooms:Renter|2"] == 1.0
