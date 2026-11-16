"""Simulations of the energy gap of each tract, with the donors resampled.

The energy of a tract (see :mod:`utah_ejpi.uncertainty`) is a weighted mean of the
modeled dwellings that stand for each type of dwelling of the tract. For the
energy that the dwellings use today, the spread among the dwellings of a type is
about proportional to their energy, and the module ``uncertainty`` represents it
with a random factor of mean one on every dwelling. The energy gap (what a
dwelling lacks to stay inside the comfort band) does not behave like that: it
depends mostly on whether the dwelling has cooling equipment at all, a yes-or-no
property, so two dwellings of a type can have a gap of zero and of ten thousand
kilobritish thermal units. Multiplying a gap by a factor of mean one would
understate the spread.

Here the sampling error of the mean of a type is simulated by resampling its
donors: in every simulation the weights of the donors of a cell are drawn from a
Dirichlet distribution whose mean is the donor weights and whose concentration is
the number of donors (the Bayesian bootstrap of Rubin 1981). A cell with a single
donor has no sampling error from this source. The same weights apply to every
quantity of the dwelling, so the energy a dwelling uses today, the gap and their
sum stay linked: the gap travels with the donor.

The counts of dwellings by type, the coefficients of the climate line and the
error of transferring dwellings between areas are simulated as in
:func:`utah_ejpi.uncertainty.simulate_tract_energy`.

Two more sources describe how a tract can differ from the mean of its type in its
zone. The composition of the tract in variables that the type does not include
(heating fuel, bedrooms, household size) moves its mean by the difference between
the means with and without the Census counts of those variables
(:func:`composition_shift`). The zone itself can differ from what the type predicts
(a common level of thermostat habits, for example): its standard deviation is
estimated from the modeled dwellings with :func:`zone_effect_sd` and one random
factor per zone is shared by all its tracts. The location effect that the
covariates do not capture is the main risk of the methods that transfer a model to
small areas (Elbers et al. 2002; Demombynes et al. 2007; Naidoo 2010).

Whether the climate line corrects each quantity is an argument, because it has to
be decided for the gap with the known-population test, not assumed.

A quantity whose estimate by tract is no better than the mean of the state (the gap
of heating, in the known-population tests) can be moved toward that mean: the center
of a tract is a weighted mean of its own estimate and the mean of all the modeled
dwellings, the form of the composite estimators of small-area estimation (Rao and
Molina 2015; Pfeffermann 2013). The weight that minimizes the squared error is
estimated with units whose truth is known (:func:`composite_weight`) and applied to
the simulations with :func:`shrink_draws`, which moves the center and keeps the spread
of the simulations: the error of the mean of a tract with respect to its true value is
not reduced by the mix, so the interval keeps the width that the known-population test
validated.

References
----------
Rubin (1981), The Bayesian bootstrap (``rubin_1981_bayesian_bootstrap``, NO
VERIFICADO); Elbers et al. (2002, ``elbers_2002_micro_level_welfare``); Rao and
Molina (2015, ``rao_molina_2015_sae``); Pfeffermann (2013, ``pfeffermann_2013_sae``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utah_ejpi.climate import rank_pumas_by_climate
from utah_ejpi.energy_requirement import (
    END_USES,
    UNIT,
    AdditiveSlope,
    DegreeDaySensitivity,
    fit_additive_slope,
)
from utah_ejpi.housing_stock import build_type_donors
from utah_ejpi.uncertainty import (
    Z90,
    _aggregate,
    _fit_without_unit,
    _valid_coefficients,
)

__all__ = [
    "Shrinkage",
    "add_composition_draws",
    "composite_weight",
    "composition_shift",
    "composition_shift_regression",
    "shrink_draws",
    "estimate_transfer_tau_quantities",
    "known_population_quantities",
    "simulate_tract_quantities",
    "zone_effect_sd",
]


@dataclass(frozen=True)
class Shrinkage:
    """Weight of a quantity toward the mean of the state.

    The center of a tract is ``weight * own + (1 - weight) * anchor``.

    Attributes
    ----------
    weight : float
        Weight of the own estimate of the tract, between 0 and 1.
    weight_sd : float
        Standard deviation of the weight (it is drawn in every simulation, the same
        for every tract, in :func:`shrink_draws`).
    anchor : float
        Value toward which the estimate is shrunk, the mean of the modeled dwellings.
    """

    weight: float
    weight_sd: float
    anchor: float


def composite_weight(
    estimate: pd.Series | np.ndarray,
    anchor: pd.Series | np.ndarray,
    truth: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 2026,
) -> tuple[float, float]:
    """Weight of the own estimate that minimizes the squared error of the mix.

    For units whose true mean is known, the weight ``w`` of
    ``w * estimate + (1 - w) * anchor`` that minimizes the sum of the squared
    errors is ``sum((e - a) * (t - a)) / sum((e - a)**2)``, kept between 0 and 1.
    Its standard deviation comes from resampling the groups (the zones, if the
    units are parts of them), which are the independent pieces of the test.

    Parameters
    ----------
    estimate, anchor, truth : array-like
        For every unit, the estimate with its own donors, the mean of the other
        dwellings (the mean of the state) and the true mean.
    groups : array-like, optional
        Group of every unit; units of a group are resampled together. Without it
        the units are resampled one by one.
    n_boot : int
        Number of resamples.
    seed : int
        Seed of the random generator.

    Returns
    -------
    tuple of float
        ``(weight, weight_sd)``.

    Raises
    ------
    ValueError
        If the inputs differ in length, have fewer than two units or the estimates
        equal the anchor everywhere.

    Examples
    --------
    >>> w, sd = composite_weight([2.0, 4.0, 6.0], [4.0, 4.0, 4.0], [3.0, 4.0, 5.0])
    >>> round(w, 3)
    0.5
    """
    e = np.asarray(estimate, dtype=float)
    a = np.asarray(anchor, dtype=float)
    t = np.asarray(truth, dtype=float)
    if not (len(e) == len(a) == len(t)) or len(e) < 2:
        raise ValueError(
            "estimate, anchor and truth need the same length (two or more)."
        )
    ok = np.isfinite(e) & np.isfinite(a) & np.isfinite(t)
    e, a, t = e[ok], a[ok], t[ok]
    g = np.arange(len(e)) if groups is None else pd.factorize(np.asarray(groups)[ok])[0]

    def weight(idx: np.ndarray) -> float:
        d = e[idx] - a[idx]
        denominator = float((d**2).sum())
        if denominator == 0.0:
            raise ValueError("The estimates equal the anchor everywhere.")
        return float(np.clip((d * (t[idx] - a[idx])).sum() / denominator, 0.0, 1.0))

    value = weight(np.arange(len(e)))
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        if float(((e[idx] - a[idx]) ** 2).sum()) > 0.0:
            boot.append(weight(idx))
    return value, float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0


def add_composition_draws(
    draws: pd.DataFrame,
    shift: pd.DataFrame,
    derived: Mapping[str, tuple[str, str]] | None = None,
    seed: int = 2026,
) -> pd.DataFrame:
    """Add the composition of the tract to the simulations as a separate source.

    In every simulation each tract draws one standard normal number and adds that
    number times its change by composition (:func:`composition_shift_regression`)
    to its quantities, the same number for every quantity and group, as
    :func:`simulate_tract_quantities` does with ``composition_shift``. Applying it
    to simulations made without that source gives the wider band that includes it,
    and the simulations without it remain the validated interval.

    Parameters
    ----------
    draws : pandas.DataFrame
        Output of :func:`simulate_tract_quantities` (with ``GEOID`` and ``draw``).
    shift : pandas.DataFrame
        Signed change by tract (index ``GEOID``) and quantity (columns).
    derived : mapping of str to tuple of str, optional
        Quantities that are the sum of two others, recomputed in the order given
        after the shift.
    seed : int
        Seed of the random generator.

    Returns
    -------
    pandas.DataFrame
        A copy of ``draws`` with the shifted and recomputed quantities.

    Raises
    ------
    ValueError
        If a quantity of ``shift`` or ``derived`` is not in the simulations.

    Examples
    --------
    >>> d = pd.DataFrame({"GEOID": ["a"] * 4, "group": "all", "draw": range(4),
    ...                   "q": 1.0})
    >>> s = pd.DataFrame({"q": [2.0]}, index=pd.Index(["a"], name="GEOID"))
    >>> out = add_composition_draws(d, s)
    >>> bool(out["q"].std() > 0)
    True
    """
    needed = {*shift.columns, *(derived or {})}
    for parts in (derived or {}).values():
        needed.update(parts)
    if not needed <= set(draws.columns):
        raise ValueError("The simulations do not have every quantity named.")
    out = draws.copy()
    rng = np.random.default_rng(seed)
    tracts = np.sort(out["GEOID"].unique())
    draw_ids = np.sort(out["draw"].unique())
    u = pd.DataFrame(
        rng.standard_normal((len(tracts), len(draw_ids))),
        index=tracts,
        columns=draw_ids,
    ).stack()
    key = pd.MultiIndex.from_arrays([out["GEOID"], out["draw"]])
    factor = u.reindex(key).to_numpy()
    for name in shift.columns:
        delta = out["GEOID"].map(shift[name]).fillna(0.0).to_numpy()
        out[name] = out[name] + delta * factor
    for name, (first_part, second_part) in (derived or {}).items():
        out[name] = out[first_part] + out[second_part]
    return out


def shrink_draws(
    draws: pd.DataFrame,
    shrink: Mapping[str, Shrinkage],
    derived: Mapping[str, tuple[str, str]] | None = None,
    seed: int = 2026,
) -> pd.DataFrame:
    """Move the center of the simulations of some quantities toward their anchor.

    For every tract and group, the simulations of a quantity are shifted by
    ``(1 - w) * (anchor - mean)``, with ``mean`` the mean of its simulations and
    ``w`` the weight of the simulation. The weight is drawn in every simulation
    (normal with the standard deviation of the rule, kept between 0 and 1), with the
    same standard normal number for every quantity and tract. The spread of the
    simulations around their mean does not change, so the interval keeps the width
    that the known-population test validated. Quantities that are the sum of others
    are then recomputed.

    Parameters
    ----------
    draws : pandas.DataFrame
        Output of :func:`simulate_tract_quantities`, with ``GEOID``, ``group`` and
        ``draw``.
    shrink : mapping of str to Shrinkage
        Rule of every quantity moved.
    derived : mapping of str to tuple of str, optional
        Quantities that are the sum of two others, recomputed in the order given
        after the shift (as in :func:`simulate_tract_quantities`).
    seed : int
        Seed of the random generator.

    Returns
    -------
    pandas.DataFrame
        A copy of ``draws`` with the shifted and recomputed quantities.

    Raises
    ------
    ValueError
        If a quantity of ``shrink`` or ``derived`` is not in the simulations.

    Examples
    --------
    >>> draws = pd.DataFrame({"GEOID": ["a"] * 2, "group": "all", "draw": [0, 1],
    ...                       "q": [1.0, 3.0]})
    >>> out = shrink_draws(draws, {"q": Shrinkage(0.5, 0.0, 10.0)})
    >>> out["q"].tolist()
    [5.0, 7.0]
    """
    needed = {*shrink, *(derived or {})}
    for parts in (derived or {}).values():
        needed.update(parts)
    if not needed <= set(draws.columns):
        raise ValueError("The simulations do not have every quantity named.")
    out = draws.copy()
    rng = np.random.default_rng(seed)
    draw_ids = np.sort(out["draw"].unique())
    z = pd.Series(rng.standard_normal(len(draw_ids)), index=draw_ids)
    z = out["draw"].map(z).to_numpy()
    for name, rule in shrink.items():
        mean = out.groupby(["GEOID", "group"])[name].transform("mean")
        weight = np.clip(rule.weight + rule.weight_sd * z, 0.0, 1.0)
        out[name] = out[name] + (1.0 - weight) * (rule.anchor - mean)
    for name, (first_part, second_part) in (derived or {}).items():
        out[name] = out[first_part] + out[second_part]
    return out


def simulate_tract_quantities(
    weights: pd.DataFrame,
    donors: pd.DataFrame,
    dwellings: pd.DataFrame,
    quantities: pd.DataFrame,
    uses: Mapping[str, str],
    tract_totals: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    climate_corrected: Collection[str] = (),
    n_draws: int = 200,
    seed: int = 2026,
    chunk: int = 25,
    transfer_tau: dict[str, float] | None = None,
    transfer_groups: pd.Series | None = None,
    resample: bool = True,
    zone_effect_sd: Mapping[str, float] | None = None,
    zone_of: pd.Series | None = None,
    composition_shift: pd.DataFrame | None = None,
    climate_additive: Mapping[str, AdditiveSlope] | None = None,
    derived: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Simulate the mean per dwelling of dwelling-level quantities in every tract.

    Parameters
    ----------
    weights, donors : pandas.DataFrame
        Contracts ``tract_type_weights`` (with ``weight`` and ``moe``, and
        ``source_level`` when ``transfer_tau`` is used) and ``tract_type_donors``.
    dwellings : pandas.DataFrame
        Modeled dwellings with ``building_id``, ``county`` (the weather station)
        and ``occupants``.
    quantities : pandas.DataFrame
        One row per modeled dwelling, indexed by ``building_id``, with the
        quantities to average (annual energy, for example the energy used today,
        the gap and their sum, by use). Dwellings with a missing value are
        dropped from the donors of the cells; a cell left without donors is
        dropped and the tract is rescaled to its total, which assumes that the
        dwellings of that type resemble the rest of the tract.
    uses : mapping of str to str
        For every column of ``quantities``, ``"heating"`` or ``"cooling"``: the
        use whose climate line and transfer error apply to it.
    tract_totals : pandas.DataFrame
        ``GEOID``, ``total`` and ``total_moe`` of occupied dwellings.
    tract_degree_days, station_degree_days : pandas.DataFrame
        Contract ``climate_degree_days`` with the kinds ``"tract"`` and
        ``"station"``.
    heating, cooling : DegreeDaySensitivity
        Climate lines with their covariance.
    climate_corrected : collection of str
        Columns of ``quantities`` that are multiplied by the ratio of the climate
        line at the tract to the line at the station. The others are used as they
        are.
    n_draws : int
        Number of simulations.
    seed : int
        Seed of the random generator.
    chunk : int
        Simulations computed at a time (limits memory use).
    transfer_tau : dict, optional
        Relative standard deviation of the transfer error, by quantity (a column
        of ``quantities``) or, for the quantities without an entry of their own,
        by use (``"heating"`` and ``"cooling"``). It is applied to the cells with
        donors from outside the own PUMA. The quantities of a use share the
        random number of each group, so they stay in step. Needs
        ``transfer_groups``.
    transfer_groups : pandas.Series, optional
        Group of every tract (index ``GEOID``) that shares one transfer factor
        per simulation.
    resample : bool
        Resample the donors of each cell (Bayesian bootstrap). With ``False`` the
        weights are the donor weights in every simulation, which isolates the
        other sources of error.
    zone_effect_sd : mapping of str to float, optional
        Relative standard deviation of the zone effect, by quantity (see
        :func:`zone_effect_sd`). In every simulation each zone draws one lognormal
        factor of mean one that multiplies the quantity in all its tracts; the
        quantities of a use share the random number of the zone. Needs
        ``zone_of``.
    zone_of : pandas.Series, optional
        Zone of every tract (index ``GEOID``).
    composition_shift : pandas.DataFrame, optional
        Output of :func:`composition_shift`: for each tract (index ``GEOID``) and
        quantity (columns), the signed change of the mean per dwelling when the
        Census counts of the variables outside the type are used. In every
        simulation each tract draws one standard normal number ``u`` and adds
        ``u`` times the change to its quantities, the same ``u`` for all of them,
        so the sum of two quantities moves as the sum of their changes.
    climate_additive : mapping of str to AdditiveSlope, optional
        For the quantities named, the value of every dwelling moves by the slope
        times the difference of degree days between the tract and the station of the
        dwelling (instead of being multiplied by the ratio of the climate line). The
        degree days of the tract are kept inside the range of the fit. The slope is
        drawn in every simulation from its variance. A quantity cannot be in this
        mapping and in ``climate_corrected``.
    derived : mapping of str to tuple of str, optional
        Quantities that are the sum of two others, as ``{"norm": ("current",
        "gap")}``. In every simulation they are set to the sum of the two others after
        all the corrections and errors, so that the sum holds in every tract and
        simulation even when the two parts are corrected differently. Their own
        correction and errors are not used. They are computed in the order given, so
        a quantity can be the sum of a derived one.

    Returns
    -------
    pandas.DataFrame
        One row per tract, group (``"all"``, ``"Owner"``, ``"Renter"``) and
        simulation with ``GEOID``, ``group``, ``draw``, ``dwellings``,
        ``persons`` and one column per quantity (the mean per dwelling).

    Raises
    ------
    ValueError
        If the quantities and the uses do not match, a use is unknown, a tract has
        no total or no transfer group, a zone effect has no zone for a tract, a
        quantity has both climate corrections, a derived quantity is not the sum of
        two quantities of the same use, or no cell has a valid donor.

    Notes
    -----
    The ratio of the climate line is the one of the energy of a dwelling, whose
    unit is energy per year; it is applied as a factor to any quantity chosen in
    ``climate_corrected``, also to a quantity that can be negative.

    Examples
    --------
    >>> draws = simulate_tract_quantities(weights, donors, dwellings, quantities,
    ...     uses,  # doctest: +SKIP
    >>>                                   totals, tract_dd, station_dd, heating,
    ...     cooling)  # doctest: +SKIP
    """
    if set(uses) != set(quantities.columns):
        raise ValueError("uses must have one entry for each column of quantities.")
    if not set(uses.values()) <= {"heating", "cooling"}:
        raise ValueError("Every use must be 'heating' or 'cooling'.")
    if transfer_tau is not None and transfer_groups is None:
        raise ValueError("transfer_tau needs transfer_groups.")
    if zone_effect_sd is not None and zone_of is None:
        raise ValueError("zone_effect_sd needs zone_of.")
    additive = dict(climate_additive or {})
    if set(additive) & set(climate_corrected):
        raise ValueError("A quantity cannot have both climate corrections.")
    if not set(additive) <= set(quantities.columns):
        raise ValueError("climate_additive names a quantity that is not in quantities.")
    derived = dict(derived or {})
    for name, parts_of in derived.items():
        if len(parts_of) != 2 or not {name, *parts_of} <= set(quantities.columns):
            raise ValueError("A derived quantity needs two quantities that exist.")
        if len({uses[name], *(uses[c] for c in parts_of)}) != 1:
            raise ValueError("A derived quantity and its parts must share the use.")
    rng = np.random.default_rng(seed)
    names = list(quantities.columns)
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    extra = ["source_level"] if transfer_tau is not None else []
    cells = weights[[*keys, "weight", "moe", *extra]].sort_values(keys)
    cells = cells.reset_index(drop=True)
    cells["cell"] = np.arange(len(cells))
    totals = tract_totals.set_index("GEOID")
    if not set(cells["GEOID"]) <= set(totals.index):
        raise ValueError("Some tracts have no total of dwellings.")

    valid = quantities.dropna().index
    rows = donors[[*keys, "building_id", "donor_weight"]].merge(
        cells[[*keys, "cell"]], on=keys
    )
    rows = rows[rows["building_id"].isin(valid)]
    if rows.empty:
        raise ValueError("No cell has a donor with all the quantities.")
    kept = np.sort(rows["cell"].unique())
    cells = cells[cells["cell"].isin(kept)].reset_index(drop=True)
    remap = pd.Series(np.arange(len(cells)), index=kept)
    rows = rows.assign(cell=rows["cell"].map(remap)).sort_values("cell")
    rows = rows.reset_index(drop=True)
    dw = rows.merge(
        dwellings[["building_id", "county", "occupants"]],
        on="building_id",
        how="left",
        validate="many_to_one",
    )
    values = {n: quantities.loc[dw["building_id"], n].to_numpy(float) for n in names}
    starts = np.flatnonzero(np.r_[True, np.diff(dw["cell"].to_numpy()) != 0])
    n_donors = np.diff(np.r_[starts, len(dw)])
    row_donors = np.repeat(n_donors, n_donors).astype(float)
    donor_weight = dw["donor_weight"].to_numpy(float)
    donor_weight = donor_weight / np.repeat(
        np.add.reduceat(donor_weight, starts), n_donors
    )
    occupants_cell = np.add.reduceat(
        dw["occupants"].to_numpy(float) * donor_weight, starts
    )

    tract_dd = tract_degree_days[tract_degree_days["kind"] == "tract"].set_index("id")
    station_dd = station_degree_days[
        station_degree_days["kind"] == "station"
    ].set_index("id")
    dd_t = {
        use: dw["GEOID"].map(tract_dd[col]).to_numpy() / UNIT
        for use, (_, col) in END_USES.items()
    }
    dd_s = {
        use: dw["county"].map(station_dd[col]).to_numpy() / UNIT
        for use, (_, col) in END_USES.items()
    }
    sens = {"heating": heating, "cooling": cooling}

    cell_tract = cells["GEOID"].to_numpy()
    cell_tenure = cells["tenure"].to_numpy()
    tract_start = np.flatnonzero(np.r_[True, cell_tract[1:] != cell_tract[:-1]])
    tract_ids = cell_tract[tract_start]
    estimate = cells["weight"].to_numpy(dtype=float)
    se_cell = np.nan_to_num(cells["moe"].to_numpy(dtype=float)) / Z90
    positive = estimate > 0
    total_est = totals.loc[tract_ids, "total"].to_numpy(dtype=float)
    total_se = (
        np.nan_to_num(totals.loc[tract_ids, "total_moe"].to_numpy(dtype=float)) / Z90
    )
    if transfer_tau is not None:
        borrowed = (cells["source_level"] != "puma").to_numpy()
        group_codes, group_names = pd.factorize(cells["GEOID"].map(transfer_groups))
        if (group_codes < 0).any():
            raise ValueError("Some tracts have no transfer group.")

    tract_codes = np.repeat(
        np.arange(len(tract_ids)), np.diff(np.r_[tract_start, len(cells)])
    )
    if zone_effect_sd is not None:
        zone_codes, zone_names = pd.factorize(pd.Series(cell_tract).map(zone_of))
        if (zone_codes < 0).any():
            raise ValueError("Some tracts have no zone.")
    if composition_shift is not None:
        shift_cell = {
            name: composition_shift[name]
            .reindex(tract_ids)
            .fillna(0.0)
            .to_numpy(float)[tract_codes]
            for name in names
            if name in composition_shift
        }

    parts = []
    for first in range(0, n_draws, chunk):
        size = min(chunk, n_draws - first)
        draw_ids = np.arange(first, first + size)
        if resample:
            shape = np.maximum(donor_weight * row_donors, 1e-9)[:, None]
            gamma = rng.gamma(shape, size=(len(dw), size))
            share = gamma / np.repeat(
                np.add.reduceat(gamma, starts, axis=0), n_donors, axis=0
            )
        else:
            share = np.repeat(donor_weight[:, None], size, axis=1)
        coef = {use: _valid_coefficients(rng, s, size) for use, s in sens.items()}
        energy = {}
        for name in names:
            use = uses[name]
            if name in climate_corrected:
                a, b = coef[use][:, 0][None, :], coef[use][:, 1][None, :]
                line_t = np.clip(a + b * dd_t[use][:, None], 0.0, None)
                line_s = a + b * dd_s[use][:, None]
                row_value = values[name][:, None] * (line_t / line_s)
            elif name in additive:
                fit = additive[name]
                slope = fit.slope + np.sqrt(fit.variance) * rng.standard_normal(size)
                dd_tract = np.clip(dd_t[use], fit.dd_min, fit.dd_max)
                row_value = (
                    values[name][:, None]
                    + slope[None, :] * (dd_tract - dd_s[use])[:, None]
                )
            else:
                row_value = np.repeat(values[name][:, None], size, axis=1)
            energy[name] = np.add.reduceat(row_value * share, starts, axis=0)
        if transfer_tau is not None:
            for use in ("heating", "cooling"):
                z = rng.standard_normal((len(group_names), size))[group_codes]
                for name in names:
                    tau = transfer_tau.get(name, transfer_tau.get(use))
                    if uses[name] != use or tau is None:
                        continue
                    factor = np.exp(tau * z - tau**2 / 2)
                    energy[name] = energy[name] * np.where(
                        borrowed[:, None], factor, 1.0
                    )

        if zone_effect_sd is not None:
            for use in ("heating", "cooling"):
                z = rng.standard_normal((len(zone_names), size))[zone_codes]
                for name in names:
                    sd = zone_effect_sd.get(name)
                    if uses[name] != use or not sd:
                        continue
                    energy[name] = energy[name] * np.exp(sd * z - sd**2 / 2)
        if composition_shift is not None:
            u = rng.standard_normal((len(tract_ids), size))[tract_codes]
            for name, delta in shift_cell.items():
                energy[name] = energy[name] + delta[:, None] * u
        for name, (first_part, second_part) in derived.items():
            energy[name] = energy[first_part] + energy[second_part]

        count = np.where(
            positive[:, None],
            estimate[:, None]
            + se_cell[:, None] * rng.standard_normal((len(cells), size)),
            0.0,
        )
        count = np.clip(count, 0.0, None)
        total_draw = np.clip(
            total_est[:, None]
            + total_se[:, None] * rng.standard_normal((len(tract_ids), size)),
            0.0,
            None,
        )
        tract_sum = np.add.reduceat(count, tract_start, axis=0)
        scale = np.divide(
            total_draw, tract_sum, out=np.zeros_like(total_draw), where=tract_sum > 0
        )
        count = count * np.repeat(
            scale, np.diff(np.r_[tract_start, len(cells)]), axis=0
        )
        for group in ("all", "Owner", "Renter"):
            mask = (
                np.ones(len(cells), dtype=bool)
                if group == "all"
                else cell_tenure == group
            )
            part = _aggregate(
                group,
                tract_ids,
                cell_tract,
                mask,
                count,
                energy,
                occupants_cell,
                draw_ids,
            )
            if part is not None:
                parts.append(part)
    return pd.concat(parts, ignore_index=True)


def known_population_quantities(
    dwellings: pd.DataFrame,
    quantities: pd.DataFrame,
    uses: Mapping[str, str],
    station_degree_days: pd.DataFrame,
    climate_corrected: Collection[str] = (),
    unit: str = "puma",
    n_draws: int = 200,
    seed: int = 2026,
    transfer_tau: dict[str, float] | None = None,
    extra_fit_dwellings: Mapping[str, pd.DataFrame] | None = None,
    extra_fit_stations: pd.DataFrame | None = None,
    resample: bool = True,
    zone_effect_sd: Mapping[str, float] | None = None,
    additive_candidates: Collection[str] = (),
    derived: Mapping[str, tuple[str, str]] | None = None,
    shrink_weights: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Estimate each PUMA or county of ResStock without its own dwellings.

    Same design as :func:`utah_ejpi.uncertainty.known_population_test` for any
    dwelling-level quantity: the true mean of a unit is the mean of its modeled
    dwellings, the unit is treated as a tract with exact counts of dwellings by
    type, and its mean is estimated with the dwellings of the other units, the
    climate line fitted without the unit and the simulations of
    :func:`simulate_tract_quantities`. The estimate is run twice, with the climate
    correction on the quantities of ``climate_corrected`` and without any, so that
    the correction can be judged for each quantity.

    Parameters
    ----------
    dwellings : pandas.DataFrame
        Occupied modeled dwellings of one upgrade (contract
        ``resstock_dwellings``), with the loads used to fit the climate line.
    quantities : pandas.DataFrame
        Dwelling-level quantities indexed by ``building_id`` (see
        :func:`simulate_tract_quantities`).
    uses : mapping of str to str
        Use (``"heating"`` or ``"cooling"``) of every quantity.
    station_degree_days : pandas.DataFrame
        Contract ``climate_degree_days`` with kind ``"station"``.
    climate_corrected : collection of str
        Quantities corrected with the climate line in the first estimate.
    unit : str
        Column of ``dwellings`` that defines the units to leave out one at a time:
        ``"county"``, ``"puma"`` or any other grouping of the dwellings (parts of a
        zone, for example, to test with units as small as a tract).
    n_draws, seed, transfer_tau, extra_fit_dwellings, extra_fit_stations
        As in :func:`utah_ejpi.uncertainty.known_population_test`.
    resample : bool
        Resample the donors in the simulations.
    zone_effect_sd : mapping of str to float, optional
        Zone effect of :func:`simulate_tract_quantities`; the unit left out is its
        own zone, so it draws one factor per simulation.
    derived : mapping of str to tuple of str, optional
        Quantities that are the sum of two others, as in
        :func:`simulate_tract_quantities`; applied in the estimate with the climate
        correction and in the one with the own slope.
    shrink_weights : mapping of str to tuple of float, optional
        For each quantity named, the weight and its standard deviation of
        :class:`Shrinkage`; the anchor of each unit is the mean of the dwellings
        outside it. Applied in every estimate.
    additive_candidates : collection of str
        Quantities for which the correction with their own slope of the degree days
        (see :func:`utah_ejpi.energy_requirement.fit_additive_slope`, fitted without
        the unit left out) is also estimated.

    Returns
    -------
    pandas.DataFrame
        One row per unit with ``unit``, ``dwellings`` and, for every quantity
        ``q``: ``truth_q``, ``estimate_q`` (mean of the simulations, with the
        climate correction of ``climate_corrected``), ``lo_q`` and ``hi_q`` (90 %
        interval), ``no_climate_q`` (mean without any correction) and ``state_q``
        (mean of all the other dwellings, ignoring types). For every quantity of
        ``additive_candidates``, also ``additive_q`` (mean of the simulations with
        the correction of its own slope, without the correction of the line of the
        use) and ``additive_lo_q`` and ``additive_hi_q`` (its 90 % interval).

    Raises
    ------
    ValueError
        If the unit is not a column of the dwellings or the dwellings span more than
        one upgrade.

    Examples
    --------
    >>> known_population_quantities(dwellings, quantities, uses, station_dd,
    ...     unit="puma")  # doctest: +SKIP
    """
    if unit not in dwellings.columns:
        raise ValueError("unit must be a column of dwellings.")
    if dwellings["upgrade"].nunique() != 1:
        raise ValueError("Pass the dwellings of one upgrade.")
    names = list(quantities.columns)
    stock = dwellings[dwellings["occupied"]].reset_index(drop=True)
    stock = stock[stock["building_id"].isin(quantities.dropna().index)]
    stock = stock.reset_index(drop=True)
    stations = station_degree_days[station_degree_days["kind"] == "station"].set_index(
        "id"
    )
    stock = stock.assign(
        station_hdd=stock["county"].map(stations["hdd"]),
        station_cdd=stock["county"].map(stations["cdd"]),
    )
    type_keys = ["tenure", "vintage", "building_type"]
    rows = []
    for name in sorted(stock[unit].unique()):
        inside = stock[unit] == name
        truth_stock, rest = stock[inside], stock[~inside]
        if rest.empty:
            continue
        unit_dd = pd.DataFrame(
            {
                "id": [name],
                "kind": "tract",
                "hdd": [truth_stock["station_hdd"].mean()],
                "cdd": [truth_stock["station_cdd"].mean()],
                "hdd_base_c": 18.0,
                "cdd_base_c": 26.0,
            }
        )
        by_puma = rest.groupby("puma")[["station_hdd", "station_cdd"]].mean()
        puma_dd = pd.DataFrame(
            {
                "id": by_puma.index.astype(str),
                "kind": "puma",
                "hdd": by_puma["station_hdd"].to_numpy(),
                "cdd": by_puma["station_cdd"].to_numpy(),
                "hdd_base_c": 18.0,
                "cdd_base_c": 26.0,
            }
        )
        rank = rank_pumas_by_climate(unit_dd, puma_dd)
        counts = (
            truth_stock.groupby(type_keys)
            .size()
            .rename("estimate")
            .reset_index()
            .assign(GEOID=name, moe=0.0)
        )
        share = truth_stock["puma"].value_counts(normalize=True)
        shares = pd.DataFrame(
            {
                "GEOID": name,
                "resstock_puma": share.index,
                "share": share.to_numpy(),
                "occupied_units": share.to_numpy(),
            }
        )
        master = pd.DataFrame(
            {"GEOID": [name], "resstock_county": [truth_stock["county"].mode().iloc[0]]}
        )
        weights, donors = build_type_donors(counts, rest, shares, master, rank)
        fits = {
            use: _fit_without_unit(
                rest, station_degree_days, use, extra_fit_dwellings, extra_fit_stations
            )
            for use in ("heating", "cooling")
        }
        totals = pd.DataFrame(
            {"GEOID": [name], "total": [float(len(truth_stock))], "total_moe": [0.0]}
        )
        runs = {}
        slopes = {
            q: fit_additive_slope(quantities[q], rest, station_degree_days, uses[q])
            for q in additive_candidates
        }
        rules = (
            {
                q: Shrinkage(
                    w, sd, float(quantities.loc[rest["building_id"], q].mean())
                )
                for q, (w, sd) in shrink_weights.items()
            }
            if shrink_weights
            else None
        )
        variants = [("with", climate_corrected, None), ("without", (), None)]
        if slopes:
            variants.append(("additive", (), slopes))
        for label, corrected, additive in variants:
            runs[label] = simulate_tract_quantities(
                weights,
                donors,
                rest,
                quantities,
                uses,
                totals,
                unit_dd,
                station_degree_days,
                fits["heating"],
                fits["cooling"],
                climate_corrected=corrected,
                n_draws=n_draws,
                seed=seed,
                transfer_tau=transfer_tau,
                transfer_groups=pd.Series({name: name}) if transfer_tau else None,
                resample=resample,
                zone_effect_sd=zone_effect_sd,
                zone_of=pd.Series({name: name}) if zone_effect_sd else None,
                climate_additive=additive,
                derived=derived if label != "without" else None,
            )
            runs[label] = runs[label][runs[label]["group"] == "all"]
            if rules:
                runs[label] = shrink_draws(
                    runs[label], rules, derived if label != "without" else None, seed
                )
        row = {"unit": name, "dwellings": len(truth_stock)}
        for q in names:
            ids_in, ids_rest = truth_stock["building_id"], rest["building_id"]
            row[f"truth_{q}"] = float(quantities.loc[ids_in, q].mean())
            sim = runs["with"][q].dropna()
            row[f"estimate_{q}"] = float(sim.mean())
            row[f"lo_{q}"] = float(sim.quantile(0.05))
            row[f"hi_{q}"] = float(sim.quantile(0.95))
            row[f"no_climate_{q}"] = float(runs["without"][q].dropna().mean())
            row[f"state_{q}"] = float(quantities.loc[ids_rest, q].mean())
            if q in slopes:
                added = runs["additive"][q].dropna()
                row[f"additive_{q}"] = float(added.mean())
                row[f"additive_lo_{q}"] = float(added.quantile(0.05))
                row[f"additive_hi_{q}"] = float(added.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_transfer_tau_quantities(
    table: pd.DataFrame,
    names: Collection[str],
    min_dwellings: int = 30,
    min_abs: float | Mapping[str, float] = 1.0,
) -> dict[str, float]:
    """Estimate the relative deviation of the transfer error of each quantity.

    As :func:`utah_ejpi.uncertainty.estimate_transfer_tau`: the excess of the
    mean squared relative error of the estimates of the units left out over the
    mean squared relative deviation of their simulations, in square root. The
    relative error is measured against the truth of each unit, so units whose
    truth is close to zero dominate; those whose truth is not above ``min_abs`` in
    absolute value are left out.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`known_population_quantities` run without transfer error.
    names : collection of str
        Quantities to estimate.
    min_dwellings : int
        Units with fewer modeled dwellings are not used.
    min_abs : float or mapping of str to float
        Smallest absolute truth that is used, one value for all the quantities or
        one by quantity (a share between 0 and 1 needs a much smaller value than
        an energy).

    Returns
    -------
    dict
        Deviation of the transfer error by quantity.

    Raises
    ------
    ValueError
        If no unit has ``min_dwellings`` dwellings.

    Examples
    --------
    >>> table = pd.DataFrame({"dwellings": [100, 100], "truth_a": [10.0, 10.0],
    ...     "estimate_a": [11.0, 9.0], "lo_a": [10.0, 8.0], "hi_a": [12.0, 10.0]})
    >>> round(estimate_transfer_tau_quantities(table, ["a"])["a"], 3)
    0.079
    """
    big = table[table["dwellings"] >= min_dwellings]
    if big.empty:
        raise ValueError("No unit has enough dwellings.")
    tau = {}
    for q in names:
        truth = big[f"truth_{q}"]
        floor = min_abs if np.isscalar(min_abs) else min_abs.get(q, 1.0)
        usable = truth.abs() > floor
        rel = ((big[f"estimate_{q}"] - truth) / truth)[usable]
        sd = ((big[f"hi_{q}"] - big[f"lo_{q}"]) / (2 * Z90) / big[f"estimate_{q}"])[
            usable
        ]
        tau[q] = float(np.sqrt(max((rel**2).mean() - (sd**2).mean(), 0.0)))
    return tau


def composition_shift_regression(
    weights_base: pd.DataFrame,
    quantities: pd.DataFrame,
    indicators: pd.DataFrame,
    shares: pd.DataFrame,
    controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Change of the mean of each tract with the Census composition, by regression.

    The type of dwelling leaves out the heating fuel, the bedrooms and the size of
    the household. Each quantity of the modeled dwellings is regressed on the type
    and on the classes of those three variables; the change of the mean of a tract
    is then the sum, over the classes, of the coefficient times the difference
    between the share of the tract that the Census reports and the share that the
    weights of the base method give. It needs only the margins of the Census, so
    it exists in every tract, unlike the reweighting of the dwellings to the
    tables, which has no solution when the modeled dwellings lack some combination
    that the Census reports. The model is additive: it assumes no interaction
    among the variables.

    Parameters
    ----------
    weights_base : pandas.DataFrame
        Weights of the base method (``GEOID``, ``building_id``, ``weight``).
    quantities : pandas.DataFrame
        Dwelling-level quantities indexed by ``building_id``; dwellings with a
        missing value are left out of the regression.
    indicators : pandas.DataFrame
        Zero-one indicators of the classes of the dwellings
        (:func:`utah_ejpi.housing_stock.class_indicators`), indexed by
        ``building_id``.
    shares : pandas.DataFrame
        Share of each tract in every class
        (:func:`utah_ejpi.housing_stock.composition_shares`), indexed by ``GEOID``;
        its columns are those of ``indicators``.
    controls : pandas.DataFrame, optional
        Indicators of the type of the dwellings (tenure, period and building
        type), indexed by ``building_id``, used only to take the type out of the
        coefficients.

    Returns
    -------
    pandas.DataFrame
        Index ``GEOID``, one column per quantity: the signed change of the mean per
        dwelling. The tracts that have weights and shares.

    Raises
    ------
    ValueError
        If the shares have columns that the indicators do not have.

    Examples
    --------
    >>> idx = pd.Index([1, 2, 3, 4], name="building_id")
    >>> q = pd.DataFrame({"x": [10.0, 10.0, 30.0, 30.0]}, index=idx)
    >>> ind = pd.DataFrame({"a": [1.0, 1.0, 0.0, 0.0]}, index=idx)
    >>> ind["b"] = 1.0 - ind["a"]
    >>> base = pd.DataFrame({"GEOID": "t", "building_id": [1, 2, 3, 4], "weight": 1.0})
    >>> shares = pd.DataFrame({"a": [0.25], "b": [0.75]}, index=["t"])
    >>> shift = composition_shift_regression(base, q, ind, shares)
    >>> round(float(shift.loc["t", "x"]), 6)
    5.0

    References
    ----------
    Elbers et al. (2002, ``elbers_2002_micro_level_welfare``): census means of the
    variables outside the model capture part of the location effect; the form is a
    synthetic estimator by regression (Rao and Molina 2015,
    ``rao_molina_2015_sae``).
    """
    if not set(shares.columns) <= set(indicators.columns):
        raise ValueError("The shares have classes that the indicators do not have.")
    y = quantities.dropna()
    parts = [indicators.reindex(y.index)]
    if controls is not None:
        parts.insert(0, controls.reindex(y.index))
    x = pd.concat(parts, axis=1).fillna(0.0)
    design = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    coefficient, *_ = np.linalg.lstsq(design, y.to_numpy(float), rcond=None)
    beta = pd.DataFrame(coefficient[1:], index=x.columns, columns=y.columns).reindex(
        shares.columns
    )
    columns = list(shares.columns)
    base = weights_base.merge(
        indicators[columns].reset_index(), on="building_id", how="inner"
    )
    weighted = base[columns].mul(base["weight"], axis=0)
    weighted["GEOID"] = base["GEOID"]
    base_share = (
        weighted.groupby("GEOID")
        .sum()
        .div(base.groupby("GEOID")["weight"].sum(), axis=0)
    )
    common = base_share.index.intersection(shares.index)
    difference = shares.loc[common, columns] - base_share.loc[common, columns]
    return difference.fillna(0.0) @ beta.fillna(0.0)


def composition_shift(
    weights_base: pd.DataFrame,
    weights_alt: pd.DataFrame,
    quantities: pd.DataFrame,
) -> pd.DataFrame:
    """Change of the mean of each tract when the Census composition is added.

    The type of dwelling does not include the heating fuel, the bedrooms or the
    size of the household, which the Census counts by tract. Weighting the modeled
    dwellings so that they also match those counts gives another estimate of the
    mean of each tract; the difference with the estimate that uses only the type
    is the size of what the type leaves out.

    Parameters
    ----------
    weights_base, weights_alt : pandas.DataFrame
        Weights of the modeled dwellings in every tract, with ``GEOID``,
        ``building_id`` and ``weight``: the one that uses the type only and the one
        that also matches the Census counts of the other variables.
    quantities : pandas.DataFrame
        Dwelling-level quantities indexed by ``building_id``.

    Returns
    -------
    pandas.DataFrame
        Index ``GEOID``, one column per quantity: mean per dwelling with the
        alternative weights minus mean with the base weights, signed.

    Examples
    --------
    >>> q = pd.DataFrame({"x": [1.0, 3.0]}, index=pd.Index([1, 2], name="building_id"))
    >>> cols = {"GEOID": ["a", "a"], "building_id": [1, 2]}
    >>> base = pd.DataFrame({**cols, "weight": [1, 1]})
    >>> alt = pd.DataFrame({**cols, "weight": [1, 3]})
    >>> composition_shift(base, alt, q).loc["a", "x"]
    0.5

    References
    ----------
    Elbers et al. (2002, ``elbers_2002_micro_level_welfare``): census means of the
    variables outside the model capture part of the location effect.
    """

    def tract_means(weights: pd.DataFrame) -> pd.DataFrame:
        merged = weights[["GEOID", "building_id", "weight"]].merge(
            quantities, left_on="building_id", right_index=True, how="inner"
        )
        w = merged["weight"].to_numpy(float)
        weighted = merged[list(quantities.columns)].mul(w, axis=0)
        weighted["GEOID"] = merged["GEOID"].to_numpy()
        total = weighted.groupby("GEOID").sum()
        return total.div(merged.groupby("GEOID")["weight"].sum(), axis=0)

    base, alt = tract_means(weights_base), tract_means(weights_alt)
    common = base.index.intersection(alt.index)
    return (alt.loc[common] - base.loc[common]).rename_axis("GEOID")


def zone_effect_sd(
    quantities: pd.DataFrame,
    zone: pd.Series,
    categorical: pd.DataFrame,
    numeric: pd.DataFrame | None = None,
    with_numeric: Collection[str] = (),
) -> dict[str, float]:
    """Relative standard deviation of the zone effect of each quantity.

    The quantity of every modeled dwelling is regressed on the variables that the
    Census counts (the categorical ones, and the numeric ones for the quantities
    listed in ``with_numeric``). The mean residual of each zone should be zero if
    the zones only differ in those variables; its spread, after removing what
    having few dwellings per zone produces by chance, is the zone effect.

    Parameters
    ----------
    quantities : pandas.DataFrame
        Dwelling-level quantities indexed by ``building_id``.
    zone : pandas.Series
        Zone of every dwelling (index ``building_id``).
    categorical : pandas.DataFrame
        Categorical variables of every dwelling (index ``building_id``): type of
        dwelling, heating fuel, bedrooms and size, for example.
    numeric : pandas.DataFrame, optional
        Numeric covariates, for example the degree days of the weather station.
    with_numeric : collection of str
        Quantities whose regression includes ``numeric`` (those that the model
        corrects for the climate).

    Returns
    -------
    dict
        Standard deviation of the zone effect over the mean of the quantity.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> idx = pd.Index(range(4000), name="building_id")
    >>> zone = pd.Series(np.repeat(np.arange(80), 50), index=idx)
    >>> effect = pd.Series(np.repeat(rng.normal(0, 2, 80), 50), index=idx)
    >>> q = pd.DataFrame({"y": 10 + effect + rng.normal(0, 1, 4000)}, index=idx)
    >>> cat = pd.DataFrame({"t": rng.choice(["a", "b"], 4000)}, index=idx)
    >>> bool(0.15 < zone_effect_sd(q, zone, cat)["y"] < 0.25)
    True

    References
    ----------
    Elbers et al. (2002, ``elbers_2002_micro_level_welfare``) on the location
    effect; the moment estimator removes the variance of a mean of ``n`` dwellings.
    """
    zone = zone.reindex(quantities.index)
    base = pd.get_dummies(
        categorical.reindex(quantities.index).astype(str), drop_first=True
    ).astype(float)
    out = {}
    for name in quantities.columns:
        design = base.copy()
        if numeric is not None and name in with_numeric:
            design = design.join(numeric.reindex(quantities.index))
        design.insert(0, "const", 1.0)
        y = quantities[name].to_numpy(float)
        x = design.to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        df = max(len(y) - np.linalg.matrix_rank(x), 1)
        sigma2 = float((resid**2).sum() / df)
        by_zone = pd.Series(resid, index=quantities.index).groupby(zone.to_numpy())
        means, counts = by_zone.mean(), by_zone.size()
        between = float(means.var(ddof=1))
        expected = float((sigma2 / counts).mean())
        mean_y = abs(float(np.mean(y)))
        out[name] = float(np.sqrt(max(between - expected, 0.0)) / mean_y)
    return out
