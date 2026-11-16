"""Tests of the simulation of the energy gap with the donors resampled."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import gap_uncertainty as gu
from utah_ejpi.energy_requirement import (
    AdditiveSlope,
    DegreeDaySensitivity,
    fit_additive_slope,
)

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


def test_known_population_with_an_additive_slope_recovers_a_gap_linear_in_the_climate():
    dwellings, stations, quantities = _population()
    uses = {"cur_cooling": "cooling", "gap_cooling": "cooling"}
    table = gu.known_population_quantities(
        dwellings,
        quantities,
        uses,
        stations,
        unit="county",
        n_draws=40,
        additive_candidates=["gap_cooling"],
    )
    assert "additive_cur_cooling" not in table
    error_additive = (table["additive_gap_cooling"] - table["truth_gap_cooling"]).abs()
    error_none = (table["no_climate_gap_cooling"] - table["truth_gap_cooling"]).abs()
    assert error_additive.mean() < 0.5 * error_none.mean()
    assert (table["additive_lo_gap_cooling"] <= table["additive_hi_gap_cooling"]).all()


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


def test_zone_effect_is_shared_by_the_tracts_of_a_zone_and_keeps_the_mean():
    zone_of = pd.Series({T1: "z1", T2: "z1"})
    base = _run(resample=False, n_draws=4000, seed=3)
    out = _run(
        resample=False,
        n_draws=4000,
        seed=3,
        zone_effect_sd={"cur_cooling": 0.3},
        zone_of=zone_of,
    )
    a = out[(out.GEOID == T1) & (out.group == "all")]["cur_cooling"].to_numpy()
    b = out[(out.GEOID == T2) & (out.group == "all")]["cur_cooling"].to_numpy()
    a0 = base[(base.GEOID == T1) & (base.group == "all")]["cur_cooling"].to_numpy()
    b0 = base[(base.GEOID == T2) & (base.group == "all")]["cur_cooling"].to_numpy()
    assert np.allclose(a / a0, b / b0)
    assert (a / a0).mean() == pytest.approx(1.0, abs=0.02)
    assert (a / a0).std() == pytest.approx(0.3, rel=0.1)
    # a quantity without an entry does not move
    other = out[(out.GEOID == T1) & (out.group == "all")]["gap_cooling"].to_numpy()
    assert np.allclose(
        other, base[(base.GEOID == T1) & (base.group == "all")]["gap_cooling"]
    )
    separate = _run(
        resample=False,
        n_draws=4000,
        seed=3,
        zone_effect_sd={"cur_cooling": 0.3},
        zone_of=pd.Series({T1: "z1", T2: "z2"}),
    )
    s1 = separate[(separate.GEOID == T1) & (separate.group == "all")]["cur_cooling"]
    s2 = separate[(separate.GEOID == T2) & (separate.group == "all")]["cur_cooling"]
    assert abs(np.corrcoef(s1, s2)[0, 1]) < 0.1


def test_composition_shift_moves_quantities_together_through_one_number():
    shift = pd.DataFrame(
        {
            "cur_cooling": [2.0, 0.0],
            "gap_cooling": [-1.0, 0.0],
            "norm_cooling": [1.0, 0.0],
        },
        index=pd.Index([T1, T2], name="GEOID"),
    )
    base = _run(resample=False, n_draws=3000, seed=4)
    out = _run(resample=False, n_draws=3000, seed=4, composition_shift=shift)
    one = out[(out.GEOID == T1) & (out.group == "all")].reset_index(drop=True)
    zero = base[(base.GEOID == T1) & (base.group == "all")].reset_index(drop=True)
    u = (one["cur_cooling"] - zero["cur_cooling"]) / 2.0
    assert u.mean() == pytest.approx(0.0, abs=0.06)
    assert u.std() == pytest.approx(1.0, rel=0.06)
    assert np.allclose((one["gap_cooling"] - zero["gap_cooling"]), -1.0 * u)
    assert np.allclose(one["norm_cooling"], one["cur_cooling"] + one["gap_cooling"])
    # the tract without change is not moved
    two = out[(out.GEOID == T2) & (out.group == "all")]["cur_cooling"].to_numpy()
    assert np.allclose(
        two, base[(base.GEOID == T2) & (base.group == "all")]["cur_cooling"]
    )


def test_effects_need_their_inputs():
    with pytest.raises(ValueError, match="zone_of"):
        _run(zone_effect_sd={"cur_cooling": 0.1})
    with pytest.raises(ValueError, match="no zone"):
        _run(zone_effect_sd={"cur_cooling": 0.1}, zone_of=pd.Series({T1: "z"}))


def test_composition_shift_is_the_difference_of_the_weighted_means():
    q = pd.DataFrame(
        {"x": [1.0, 3.0, 5.0]}, index=pd.Index([1, 2, 3], name="building_id")
    )
    cols = {"GEOID": ["a", "a", "b"], "building_id": [1, 2, 3]}
    base = pd.DataFrame({**cols, "weight": [1.0, 1.0, 2.0]})
    alt = pd.DataFrame({**cols, "weight": [3.0, 1.0, 5.0]})
    out = gu.composition_shift(base, alt, q)
    assert out.loc["a", "x"] == pytest.approx(1.5 - 2.0)
    assert out.loc["b", "x"] == pytest.approx(0.0)


def test_zone_effect_sd_recovers_the_effect_and_finds_none_when_there_is_none():
    rng = np.random.default_rng(7)
    idx = pd.Index(range(6000), name="building_id")
    zone = pd.Series(np.repeat(np.arange(60), 100), index=idx)
    kind = pd.DataFrame({"t": rng.choice(["a", "b", "c"], 6000)}, index=idx)
    level = {"a": 8.0, "b": 10.0, "c": 12.0}
    base = kind["t"].map(level) + rng.normal(0, 3, 6000)
    effect = pd.Series(np.repeat(rng.normal(0, 1.5, 60), 100), index=idx)
    q = pd.DataFrame({"with": base + effect, "without": base})
    sd = gu.zone_effect_sd(q, zone, kind)
    assert sd["with"] == pytest.approx(1.5 / 10.0, abs=0.04)
    assert sd["without"] < 0.03


def test_additive_correction_moves_the_quantity_by_the_slope_and_keeps_the_rest():
    slope = AdditiveSlope("cooling", 4.0, 0.0, 10, 3, 0.0, 1.0)
    out = _run(resample=False, climate_additive={"gap_cooling": slope})
    # Tract T1: 0.2 thousand degree days against 0.1 at the station.
    cell = out[(out.GEOID == T1) & (out.group == "Owner")].iloc[0]
    assert cell["gap_cooling"] == pytest.approx(5.0 + 4.0 * 0.1)
    assert cell["cur_cooling"] == pytest.approx(10.0)


def test_additive_correction_stays_inside_the_range_of_the_fit():
    slope = AdditiveSlope("cooling", 4.0, 0.0, 10, 3, 0.0, 0.15)
    out = _run(resample=False, climate_additive={"gap_cooling": slope})
    cell = out[(out.GEOID == T1) & (out.group == "Owner")].iloc[0]
    assert cell["gap_cooling"] == pytest.approx(5.0 + 4.0 * (0.15 - 0.1))


def test_additive_slope_is_drawn_with_its_variance_and_conflicts_are_refused():
    slope = AdditiveSlope("cooling", 4.0, 4.0, 10, 3, 0.0, 1.0)
    out = _run(
        n_draws=2000, resample=False, climate_additive={"gap_cooling": slope}, seed=3
    )
    cell = out[(out.GEOID == T1) & (out.group == "Owner")]["gap_cooling"]
    assert cell.mean() == pytest.approx(5.4, abs=0.02)
    assert cell.std() == pytest.approx(0.2, rel=0.1)  # sd 2 times 0.1
    with pytest.raises(ValueError, match="both climate"):
        _run(climate_corrected=["gap_cooling"], climate_additive={"gap_cooling": slope})
    with pytest.raises(ValueError, match="not in quantities"):
        _run(climate_additive={"other": slope})


def test_fit_additive_slope_recovers_the_slope_and_its_range():
    rng = np.random.default_rng(4)
    counties = [f"c{i}" for i in range(8)]
    cdd = dict(zip(counties, np.linspace(0.0, 700.0, 8), strict=True))
    frame = pd.DataFrame(
        {
            "building_id": range(400),
            "occupied": True,
            "county": rng.choice(counties, 400),
            "tenure": rng.choice(["Owner", "Renter"], 400),
        }
    )
    stations = pd.DataFrame(
        {"id": counties, "kind": "station", "cdd": [cdd[c] for c in counties]}
    )
    dd = frame["county"].map(cdd) / 1000.0
    y = pd.Series(
        100.0
        + 30.0 * (frame["tenure"] == "Owner")
        + 500.0 * dd
        + rng.normal(0.0, 5.0, 400),
        index=frame["building_id"],
    )
    fit = fit_additive_slope(y, frame, stations, "cooling", ("tenure",))
    assert fit.slope == pytest.approx(500.0, rel=0.05)
    assert fit.variance > 0
    assert (fit.dd_min, fit.dd_max) == pytest.approx((0.0, 0.7))
    with pytest.raises(ValueError, match="two counties"):
        fit_additive_slope(
            y[frame["county"] == "c1"], frame, stations, "cooling", ("tenure",)
        )
    with pytest.raises(KeyError):
        fit_additive_slope(y, frame, stations, "lighting", ("tenure",))


def test_a_derived_quantity_is_the_sum_of_its_parts_after_every_correction():
    inputs = _inputs()
    inputs["weights"]["source_level"] = ["puma", "county", "puma", "puma"]
    out = _run(
        inputs,
        n_draws=30,
        seed=7,
        climate_corrected=["cur_cooling", "norm_cooling"],
        transfer_tau={"cur_cooling": 0.3, "gap_cooling": 0.5, "norm_cooling": 0.1},
        transfer_groups=pd.Series({T1: "z", T2: "z"}),
        zone_effect_sd={"cur_cooling": 0.2, "gap_cooling": 0.4},
        zone_of=pd.Series({T1: "z", T2: "z"}),
        derived={"norm_cooling": ("cur_cooling", "gap_cooling")},
    )
    assert np.allclose(
        out["norm_cooling"], out["cur_cooling"] + out["gap_cooling"], atol=1e-9
    )
    # the correction of the current energy reaches the derived quantity; the gap's
    # does not
    plain = _run(
        inputs, resample=False, climate_corrected=["cur_cooling"],
        derived={"norm_cooling": ("cur_cooling", "gap_cooling")},
    )  # fmt: skip
    cell = plain[(plain.GEOID == T1) & (plain.group == "Owner")].iloc[0]
    assert cell["norm_cooling"] == pytest.approx(10.0 * 1.2 / 1.1 + 5.0)


def test_a_derived_quantity_needs_existing_parts_of_the_same_use():
    inputs = _inputs()
    with pytest.raises(ValueError, match="two quantities"):
        _run(inputs, derived={"norm_cooling": ("cur_cooling", "other")})
    inputs["uses"] = {**USES, "gap_cooling": "heating"}
    with pytest.raises(ValueError, match="share the use"):
        _run(inputs, derived={"norm_cooling": ("cur_cooling", "gap_cooling")})


def test_composite_weight_recovers_the_mix_and_its_limits():
    anchor = np.full(40, 10.0)
    estimate = 10.0 + np.linspace(-5.0, 5.0, 40)
    truth = anchor + 0.4 * (estimate - anchor)
    weight, sd = gu.composite_weight(estimate, anchor, truth, seed=1)
    assert weight == pytest.approx(0.4)
    assert sd == pytest.approx(0.0, abs=1e-9)
    noisy = (
        anchor
        + 0.4 * (estimate - anchor)
        + np.random.default_rng(2).normal(0.0, 2.0, 40)
    )
    weight, sd = gu.composite_weight(
        estimate, anchor, noisy, groups=np.repeat(np.arange(8), 5), seed=1
    )
    assert 0.0 < weight < 1.0 and sd > 0.0
    assert (
        gu.composite_weight(estimate, anchor, anchor + 3 * (estimate - anchor))[0]
        == 1.0
    )
    assert (
        gu.composite_weight(estimate, anchor, anchor - 3 * (estimate - anchor))[0]
        == 0.0
    )
    with pytest.raises(ValueError, match="same length"):
        gu.composite_weight([1.0], [1.0], [1.0])
    with pytest.raises(ValueError, match="anchor everywhere"):
        gu.composite_weight([1.0, 1.0], [1.0, 1.0], [2.0, 3.0])


def test_shrinking_moves_the_center_and_keeps_the_spread_and_the_sums():
    out = _run(n_draws=50, seed=3)
    derived = {"norm_cooling": ("cur_cooling", "gap_cooling")}
    rule = gu.Shrinkage(0.5, 0.0, 100.0)
    moved = gu.shrink_draws(out, {"gap_cooling": rule}, derived)
    for _, part in moved.groupby(["GEOID", "group"]):
        same = out.loc[part.index, "gap_cooling"]
        assert part["gap_cooling"].std() == pytest.approx(same.std())
        assert part["gap_cooling"].mean() == pytest.approx(0.5 * same.mean() + 50.0)
    assert np.allclose(
        moved["norm_cooling"], moved["cur_cooling"] + moved["gap_cooling"], atol=1e-9
    )
    assert np.allclose(moved["cur_cooling"], out["cur_cooling"])
    flat = gu.shrink_draws(out, {"gap_cooling": gu.Shrinkage(0.0, 0.0, 7.0)})
    assert np.allclose(
        flat["gap_cooling"]
        - flat.groupby(["GEOID", "group"])["gap_cooling"].transform("mean"),
        out["gap_cooling"]
        - out.groupby(["GEOID", "group"])["gap_cooling"].transform("mean"),
    )
    assert np.allclose(flat.groupby(["GEOID", "group"])["gap_cooling"].mean(), 7.0)
    same = gu.shrink_draws(out, {"gap_cooling": gu.Shrinkage(1.0, 0.0, 7.0)})
    assert np.allclose(same["gap_cooling"], out["gap_cooling"])
    with pytest.raises(ValueError, match="every quantity"):
        gu.shrink_draws(out, {"other": rule})


def test_the_weight_is_drawn_once_per_simulation_for_every_tract():
    out = _run(n_draws=300, resample=False, seed=4)
    moved = gu.shrink_draws(out, {"gap_cooling": gu.Shrinkage(0.5, 0.2, 100.0)})
    wide = moved[moved.group == "all"].pivot(
        index="draw", columns="GEOID", values="gap_cooling"
    )
    assert wide.std().min() > 5.0
    assert np.corrcoef(wide[T1], wide[T2])[0, 1] > 0.99


def test_known_population_accepts_any_unit_and_shrinks_toward_the_mean_of_the_rest():
    dwellings, stations, quantities = _population()
    uses = {"cur_cooling": "cooling", "gap_cooling": "cooling"}
    dwellings = dwellings.assign(part=dwellings["county"].astype(str) + "_x")
    table = gu.known_population_quantities(
        dwellings,
        quantities,
        uses,
        stations,
        unit="part",
        n_draws=20,
        shrink_weights={"gap_cooling": (0.0, 0.0)},
    )
    assert len(table) == 6
    assert np.allclose(table["estimate_gap_cooling"], table["state_gap_cooling"])
    assert not np.allclose(table["estimate_cur_cooling"], table["state_cur_cooling"])


def _regression_inputs():
    idx = pd.Index(range(1, 9), name="building_id")
    gas = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    size = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    q = pd.DataFrame({"x": 100.0 - 40.0 * (1 - gas) + 5.0 * size}, index=idx)
    ind = pd.DataFrame(
        {"fuel:gas": gas, "fuel:elec": 1 - gas, "size:2": 1 - size, "size:3": size},
        index=idx,
    )
    base = pd.DataFrame({"GEOID": "t", "building_id": list(idx), "weight": 1.0})
    return q, ind, base


def test_the_regression_shift_is_the_effect_times_the_change_of_the_shares():
    q, ind, base = _regression_inputs()
    # the base weights give 50 % gas and 50 % size 3; the Census says 20 % gas
    shares = pd.DataFrame(
        {"fuel:gas": [0.2], "fuel:elec": [0.8], "size:2": [0.5], "size:3": [0.5]},
        index=pd.Index(["t"], name="GEOID"),
    )
    shift = gu.composition_shift_regression(base, q, ind, shares)
    assert shift.loc["t", "x"] == pytest.approx(-0.3 * 40.0)
    same = shares.assign(**{"fuel:gas": 0.5, "fuel:elec": 0.5})
    assert gu.composition_shift_regression(base, q, ind, same).loc["t", "x"] == (
        pytest.approx(0.0, abs=1e-9)
    )


def test_the_regression_shift_takes_the_type_out_and_checks_its_columns():
    q, ind, base = _regression_inputs()
    # a type that varies independently of the fuel and adds 10 to the quantity
    owner = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    q = q.assign(x=q["x"] + 10.0 * owner)
    controls = pd.DataFrame({"owner": owner, "renter": 1 - owner}, index=q.index)
    shares = pd.DataFrame(
        {"fuel:gas": [0.2], "fuel:elec": [0.8]}, index=pd.Index(["t"], name="GEOID")
    )
    shift = gu.composition_shift_regression(base, q, ind, shares, controls)
    assert shift.loc["t", "x"] == pytest.approx(-0.3 * 40.0)
    with pytest.raises(ValueError, match="classes"):
        gu.composition_shift_regression(
            base, q, ind, shares.assign(other=0.0), controls
        )


def test_the_composition_band_adds_one_number_per_tract_and_simulation():
    out = _run(n_draws=100, resample=False, seed=2)
    shift = pd.DataFrame(
        {"gap_cooling": [4.0, 0.0]}, index=pd.Index([T1, T2], name="GEOID")
    )
    derived = {"norm_cooling": ("cur_cooling", "gap_cooling")}
    wide = gu.add_composition_draws(out, shift, derived, seed=1)
    first = wide[(wide.GEOID == T1) & (wide.group == "all")]
    second = wide[(wide.GEOID == T2) & (wide.group == "all")]
    assert first["gap_cooling"].std() == pytest.approx(4.0, rel=0.2)
    assert second["gap_cooling"].std() == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(wide["norm_cooling"], wide["cur_cooling"] + wide["gap_cooling"])
    # the simulations of the validated interval do not change
    assert np.allclose(out["cur_cooling"], wide["cur_cooling"])
    # the same number is added to every group of a tract in a simulation
    both = wide[wide.GEOID == T1].pivot(
        index="draw", columns="group", values="gap_cooling"
    ) - out[out.GEOID == T1].pivot(index="draw", columns="group", values="gap_cooling")
    assert np.allclose(both["all"], both["Owner"])
    with pytest.raises(ValueError, match="every quantity"):
        gu.add_composition_draws(out, shift.rename(columns={"gap_cooling": "x"}))
