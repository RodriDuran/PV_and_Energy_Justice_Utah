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
:func:`utah_ejpi.uncertainty.simulate_tract_energy`. Whether the climate line
corrects each quantity is an argument, because it has to be decided for the gap
with the known-population test, not assumed.

References
----------
Rubin (1981), The Bayesian bootstrap (``rubin_1981_bayesian_bootstrap``, NO
VERIFICADO); Elbers et al. (2002, ``elbers_2002_micro_level_welfare``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd

from utah_ejpi.climate import rank_pumas_by_climate
from utah_ejpi.energy_requirement import END_USES, UNIT, DegreeDaySensitivity
from utah_ejpi.housing_stock import build_type_donors
from utah_ejpi.uncertainty import (
    Z90,
    _aggregate,
    _fit_without_unit,
    _valid_coefficients,
)

__all__ = [
    "estimate_transfer_tau_quantities",
    "known_population_quantities",
    "simulate_tract_quantities",
]


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
        no total or no transfer group, or no cell has a valid donor.

    Notes
    -----
    The ratio of the climate line is the one of the energy of a dwelling, whose
    unit is energy per year; it is applied as a factor to any quantity chosen in
    ``climate_corrected``, also to a quantity that can be negative.
    """
    if set(uses) != set(quantities.columns):
        raise ValueError("uses must have one entry for each column of quantities.")
    if not set(uses.values()) <= {"heating", "cooling"}:
        raise ValueError("Every use must be 'heating' or 'cooling'.")
    if transfer_tau is not None and transfer_groups is None:
        raise ValueError("transfer_tau needs transfer_groups.")
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
    unit : {"county", "puma"}
        Units to leave out one at a time.
    n_draws, seed, transfer_tau, extra_fit_dwellings, extra_fit_stations
        As in :func:`utah_ejpi.uncertainty.known_population_test`.
    resample : bool
        Resample the donors in the simulations.

    Returns
    -------
    pandas.DataFrame
        One row per unit with ``unit``, ``dwellings`` and, for every quantity
        ``q``: ``truth_q``, ``estimate_q`` (mean of the simulations, with the
        climate correction of ``climate_corrected``), ``lo_q`` and ``hi_q`` (90 %
        interval), ``no_climate_q`` (mean without any correction) and ``state_q``
        (mean of all the other dwellings, ignoring types).

    Raises
    ------
    ValueError
        If the unit is unknown or the dwellings span more than one upgrade.
    """
    if unit not in ("county", "puma"):
        raise ValueError("unit must be 'county' or 'puma'.")
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
        for label, corrected in (("with", climate_corrected), ("without", ())):
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
            )
            runs[label] = runs[label][runs[label]["group"] == "all"]
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
