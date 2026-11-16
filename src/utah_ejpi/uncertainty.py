"""Uncertainty of the energy of each tract, as a set of simulations.

The energy of a tract (see :mod:`utah_ejpi.energy_requirement`) is the result of
counting dwellings with a survey, representing each type of dwelling with a few
modeled dwellings and correcting them with a climate line estimated from 29
weather stations. This module repeats the calculation many times, each time
drawing the uncertain inputs, so that every tract has a distribution of values
and not a single one. The same set of simulations can be used by the later steps
of the study (energy poverty, rooftop solar generation, the index), which then
carry the uncertainty of the energy to their own results.

Three statistical sources are simulated, all in the same simulation so that the
errors that tracts share stay correlated:

* **Counts of dwellings by type.** The cells of the Census table B25127 with a
  positive estimate are drawn from a normal distribution with the standard error
  of the survey (the margin of error divided by 1.645), censored at zero, and
  rescaled to a total of dwellings of the tract that is drawn with the margin of
  error of its own table. Cells with an estimate of zero stay at zero, since
  drawing them would create dwellings that the survey did not find (U.S. Census
  Bureau 2020, ``census_2020_acs_handbook_ch8``).
* **Coefficients of the climate line.** One draw per simulation, shared by all the
  tracts, from the multivariate normal distribution with the covariance robust
  to the clustering by county (Fels 1986, ``fels_1986_prism``).
* **Modeled dwellings.** Each modeled dwelling is multiplied by a random factor of
  mean one whose relative dispersion is the one observed among the dwellings of
  a same type in ResStock. The factor is the same wherever the dwelling is used,
  so tracts that share dwellings share their error. This is the idiosyncratic
  error of Elbers et al. (2002, ``elbers_2002_micro_level_welfare``): the mean of
  a type built from a few dwellings is uncertain in proportion to the dispersion
  among dwellings divided by the square root of their number.

Systematic uncertainties (the level of ResStock compared with the national
survey, the extrapolation beyond the climate of the stations) are not simulated:
they move all the tracts in the same direction and are shown as scenarios.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utah_ejpi.climate import rank_pumas_by_climate
from utah_ejpi.energy_requirement import (
    END_USES,
    UNIT,
    DegreeDaySensitivity,
    build_tract_type_energy,
    fit_degree_day_sensitivity,
)
from utah_ejpi.housing_stock import FALLBACK_LEVELS, build_type_donors
from utah_ejpi.resstock import FUEL_COLUMNS
from utah_ejpi.utils import cv_quality

__all__ = [
    "Dispersion",
    "aggregate_draws",
    "apply_level_factors",
    "estimate_transfer_tau",
    "known_population_summary",
    "known_population_test",
    "rank_stability",
    "simulate_tract_energy",
    "summarize_draws",
    "within_type_dispersion",
]

#: Standard errors are the margins of error of the survey divided by this value.
Z90 = 1.645

#: Columns of the annual loads in the dwellings table.
END_USES_LOADS = [load for load, _ in END_USES.values()]


@dataclass(frozen=True)
class Dispersion:
    """Dispersion of the annual load among the dwellings of a same type.

    Attributes
    ----------
    cv_heating, cv_cooling : float
        Pooled coefficient of variation (standard deviation over mean) of the
        heating and cooling loads among the dwellings with a system.
    correlation : float
        Pooled correlation, within types, between the logarithms of both loads.
    """

    cv_heating: float
    cv_cooling: float
    correlation: float


def within_type_dispersion(dwellings: pd.DataFrame) -> Dispersion:
    """Measure the dispersion of the loads among dwellings of the same type.

    Parameters
    ----------
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings``; occupied dwellings are used.

    Returns
    -------
    Dispersion
        Pooled over the types with two or more dwellings with a system, weighting
        each type by its degrees of freedom.

    Examples
    --------
    >>> within_type_dispersion(dwellings)  # doctest: +SKIP
    """
    stock = dwellings[dwellings["occupied"]].copy()
    stock["cell"] = (
        stock["tenure"] + "|" + stock["vintage"] + "|" + stock["building_type"]
    )
    cvs = []
    for load in ("heating_load_kbtu", "cooling_load_kbtu"):
        x = stock[stock[load] > 0]
        grouped = x.groupby("cell")[load]
        n, mean, var = grouped.size(), grouped.mean(), grouped.var(ddof=1)
        ok = n >= 2
        pooled = np.sqrt(((n[ok] - 1) * var[ok]).sum() / (n[ok] - 1).sum())
        cvs.append(float(pooled / ((n * mean).sum() / n.sum())))
    both = stock[
        (stock["heating_load_kbtu"] > 0) & (stock["cooling_load_kbtu"] > 0)
    ].copy()
    logs = np.log(both[["heating_load_kbtu", "cooling_load_kbtu"]])
    centered = logs - logs.groupby(both["cell"]).transform("mean")
    size = both.groupby("cell")["cell"].transform("size")
    centered = centered[size >= 2]
    rho = float(np.corrcoef(centered.iloc[:, 0], centered.iloc[:, 1])[0, 1])
    return Dispersion(cvs[0], cvs[1], rho)


def _valid_coefficients(
    rng: np.random.Generator, sens: DegreeDaySensitivity, n: int
) -> np.ndarray:
    """Draw coefficients whose line is positive over the range of the stations."""
    mean = np.array([sens.intercept, sens.slope])
    out = rng.multivariate_normal(mean, sens.covariance, size=n)
    for _ in range(100):
        at_min = out[:, 0] + out[:, 1] * sens.dd_min
        at_max = out[:, 0] + out[:, 1] * sens.dd_max
        bad = (at_min <= 0) | (at_max <= 0)
        if not bad.any():
            return out
        out[bad] = rng.multivariate_normal(mean, sens.covariance, size=int(bad.sum()))
    raise ValueError("Could not draw coefficients with a positive climate line.")


def simulate_tract_energy(
    weights: pd.DataFrame,
    donors: pd.DataFrame,
    dwellings: pd.DataFrame,
    tract_totals: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    dispersion: Dispersion,
    n_draws: int = 200,
    seed: int = 2026,
    extrapolation: str = "linear",
    chunk: int = 25,
    transfer_tau: dict[str, float] | None = None,
    transfer_groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate the energy of every tract many times.

    Parameters
    ----------
    weights, donors : pandas.DataFrame
        Contracts ``tract_type_weights`` (with ``weight`` and ``moe``) and
        ``tract_type_donors``.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` of one upgrade, with the annual consumption
        columns.
    tract_totals : pandas.DataFrame
        ``GEOID``, ``total`` and ``total_moe``: occupied dwellings of each tract
        and the margin of error of that table.
    tract_degree_days, station_degree_days : pandas.DataFrame
        Contract ``climate_degree_days`` with the kinds ``"tract"`` and
        ``"station"``.
    heating, cooling : DegreeDaySensitivity
        Climate lines with their covariance.
    dispersion : Dispersion
        Relative dispersion of the loads among dwellings of a type.
    n_draws : int
        Number of simulations.
    seed : int
        Seed of the random generator; the same seed gives the same simulations.
    extrapolation : {"linear", "cap"}
        Treatment of tracts beyond the climate of the stations (see
        :func:`utah_ejpi.energy_requirement.climate_factor`). It is a scenario,
        not a source of random variation.
    chunk : int
        Number of simulations computed at a time (limits memory use).
    transfer_tau : dict, optional
        Relative standard deviation of the transfer error, with keys ``"heating"``
        and ``"cooling"`` (see :func:`estimate_transfer_tau`). When given, the
        energy of the cells whose donors come from outside the own PUMA of the
        tract is multiplied by a lognormal factor of mean one and this deviation.
    transfer_groups : pandas.Series, optional
        Group of every tract (index ``GEOID``) that shares one transfer factor per
        simulation; required with ``transfer_tau``. Tracts that borrow from the
        same dwellings should share it, for example their main PUMA.

    Returns
    -------
    pandas.DataFrame
        One row per tract, group and simulation. ``group`` is ``"all"``,
        ``"Owner"`` or ``"Renter"``. Columns: ``GEOID``, ``group``, ``draw``,
        ``dwellings`` (simulated occupied dwellings), ``persons``, and, per
        dwelling, ``heating_kbtu``, ``cooling_kbtu`` (loads delivered), the annual
        consumption by use and fuel (``heating_kwh_electricity``, ... ,
        ``cooling_kwh_electricity``) and the totals ``heating_kwh`` and
        ``cooling_kwh``. Groups without dwellings in a simulation have missing
        energy values.

    Raises
    ------
    ValueError
        If the dwellings table spans more than one upgrade, a tract lacks totals,
        or the rule of extrapolation is unknown.

    Notes
    -----
    The occupants of each type are the plain mean of its donors. The relative
    factor of each dwelling is lognormal with mean one; the factors of heating and
    cooling of a dwelling are correlated (``dispersion.correlation``) and a
    dwelling without a system keeps zero.

    Examples
    --------
    >>> simulate_tract_energy(weights, donors, dwellings, tract_totals, tract_dd,
    ...     # doctest: +SKIP
    >>>                       station_dd, heating, cooling)  # doctest: +SKIP
    """
    if dwellings["upgrade"].nunique() != 1:
        raise ValueError("Pass the dwellings of one upgrade.")
    if extrapolation not in ("linear", "cap"):
        raise ValueError("extrapolation must be 'linear' or 'cap'.")
    if transfer_tau is not None and transfer_groups is None:
        raise ValueError("transfer_tau needs transfer_groups.")
    rng = np.random.default_rng(seed)
    fuels = [c for c in FUEL_COLUMNS if c in dwellings.columns]
    keys = ["GEOID", "tenure", "vintage", "building_type"]

    extra = ["source_level"] if transfer_tau is not None else []
    cells = weights[[*keys, "weight", "moe", *extra]].sort_values(
        ["GEOID", "tenure", "vintage", "building_type"]
    )
    cells = cells.reset_index(drop=True)
    cells["cell"] = np.arange(len(cells))
    totals = tract_totals.set_index("GEOID")
    if not set(cells["GEOID"]) <= set(totals.index):
        raise ValueError("Some tracts have no total of dwellings.")

    # Donor rows, ordered by cell so that the cells are contiguous segments.
    attr = ["building_id", "county", "occupants", *END_USES_LOADS, *fuels]
    rows = donors[[*keys, "building_id", "donor_weight"]].merge(
        cells[[*keys, "cell"]], on=keys
    )
    rows = rows.sort_values("cell").reset_index(drop=True)
    dw = rows.merge(
        dwellings[attr], on="building_id", how="left", validate="many_to_one"
    )
    dwelling_ids = dwellings["building_id"].to_numpy()
    dw_index = pd.Series(np.arange(len(dwelling_ids)), index=dwelling_ids)
    row_dwelling = dw_index.loc[dw["building_id"]].to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(dw["cell"].to_numpy()) != 0])
    donor_weight = dw["donor_weight"].to_numpy()[:, None]

    tract_dd = tract_degree_days[tract_degree_days["kind"] == "tract"].set_index("id")
    station_dd = station_degree_days[
        station_degree_days["kind"] == "station"
    ].set_index("id")
    row_tract = dw["GEOID"]
    dd_t = {
        use: row_tract.map(tract_dd[col]).to_numpy() / UNIT
        for use, (_, col) in END_USES.items()
    }
    dd_s = {
        use: dw["county"].map(station_dd[col]).to_numpy() / UNIT
        for use, (_, col) in END_USES.items()
    }
    sens = {"heating": heating, "cooling": cooling}
    for use, s in sens.items():
        if extrapolation == "cap":
            dd_t[use] = np.clip(dd_t[use], s.dd_min, s.dd_max)

    value_columns = {
        "heating_kbtu": ("heating", "heating_load_kbtu"),
        "cooling_kbtu": ("cooling", "cooling_load_kbtu"),
        **{c: ("heating" if c.startswith("heating") else "cooling", c) for c in fuels},
    }
    values = {
        name: dw[col].to_numpy(dtype=float) for name, (_, col) in value_columns.items()
    }
    occupants_cell = np.add.reduceat(
        dw["occupants"].to_numpy(dtype=float) * dw["donor_weight"].to_numpy(), starts
    )

    # Cell structure: contiguous segments by tract and by tract and tenure.
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

    sigma = {
        "heating": np.sqrt(np.log1p(dispersion.cv_heating**2)),
        "cooling": np.sqrt(np.log1p(dispersion.cv_cooling**2)),
    }
    rho = dispersion.correlation
    if transfer_tau is not None:
        # Only cells served by dwellings of other PUMAs carry the transfer error.
        borrowed = (cells["source_level"] != "puma").to_numpy()
        group_codes, group_names = pd.factorize(cells["GEOID"].map(transfer_groups))
        if (group_codes < 0).any():
            raise ValueError("Some tracts have no transfer group.")
    parts = []
    for first in range(0, n_draws, chunk):
        size = min(chunk, n_draws - first)
        draw_ids = np.arange(first, first + size)

        # Factors of the dwellings (shared by every tract that uses them).
        z1 = rng.standard_normal((len(dwelling_ids), size))
        z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * rng.standard_normal(
            (len(dwelling_ids), size)
        )
        xi = {
            "heating": np.exp(sigma["heating"] * z1 - sigma["heating"] ** 2 / 2),
            "cooling": np.exp(sigma["cooling"] * z2 - sigma["cooling"] ** 2 / 2),
        }

        # Coefficients of the lines (shared by every tract).
        coef = {use: _valid_coefficients(rng, s, size) for use, s in sens.items()}

        # Energy of each cell: donors perturbed, corrected to the climate of the tract.
        energy = {}
        for name, (use, _) in value_columns.items():
            a, b = coef[use][:, 0][None, :], coef[use][:, 1][None, :]
            line_t = np.clip(a + b * dd_t[use][:, None], 0.0, None)
            line_s = a + b * dd_s[use][:, None]
            row_value = (
                values[name][:, None] * xi[use][row_dwelling] * (line_t / line_s)
            )
            energy[name] = np.add.reduceat(row_value * donor_weight, starts, axis=0)

        # Transfer error: one factor per group, use and simulation, applied to the
        # cells with borrowed donors.
        if transfer_tau is not None:
            for use in ("heating", "cooling"):
                tau = transfer_tau[use]
                factor = np.exp(
                    tau * rng.standard_normal((len(group_names), size)) - tau**2 / 2
                )[group_codes]
                factor = np.where(borrowed[:, None], factor, 1.0)
                for name, (u, _) in value_columns.items():
                    if u == use:
                        energy[name] = energy[name] * factor

        # Dwellings of each cell: draw the cells with positive estimate and rescale
        # to a total of the tract that is drawn with its own margin of error.
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


def _aggregate(
    group, tract_ids, cell_tract, mask, count, energy, occupants_cell, draw_ids
) -> pd.DataFrame:
    """Add the cells of each tract that belong to a group, for every simulation."""
    keep = np.flatnonzero(mask)
    if len(keep) == 0:
        return None
    tracts = cell_tract[keep]
    first = np.flatnonzero(np.r_[True, tracts[1:] != tracts[:-1]])
    present = tracts[first]
    n = count[keep]
    dwellings = np.add.reduceat(n, first, axis=0)
    persons = np.add.reduceat(n * occupants_cell[keep][:, None], first, axis=0)
    out = {
        "GEOID": np.repeat(present, len(draw_ids)),
        "group": group,
        "draw": np.tile(draw_ids, len(present)),
        "dwellings": dwellings.ravel(),
        "persons": persons.ravel(),
    }
    safe = np.where(dwellings > 0, dwellings, np.nan)
    for name, e in energy.items():
        out[name] = (np.add.reduceat(n * e[keep], first, axis=0) / safe).ravel()
    frame = pd.DataFrame(out)
    for use in ("heating", "cooling"):
        fuel_cols = [c for c in energy if c.startswith(f"{use}_kwh")]
        if fuel_cols:
            frame[f"{use}_kwh"] = frame[fuel_cols].sum(axis=1, min_count=1)
    return frame


def apply_level_factors(
    draws: pd.DataFrame, heating_factor: float, cooling_factor: float
) -> pd.DataFrame:
    """Scale the simulated energy to another level of ResStock.

    The level of ResStock compared with the national survey is a systematic
    uncertainty, so it is shown as scenarios and not simulated: the energy of
    heating and of cooling is multiplied by a factor, the same in every tract.

    Parameters
    ----------
    draws : pandas.DataFrame
        Output of :func:`simulate_tract_energy`.
    heating_factor, cooling_factor : float
        Factors for the columns that start with ``heating`` and ``cooling``.

    Returns
    -------
    pandas.DataFrame
        A copy of the simulations with the energy columns scaled.

    Examples
    --------
    >>> draws = pd.DataFrame({"heating_kbtu": [10.0, 20.0], "cooling_kbtu": [5.0, 5.0]})
    >>> apply_level_factors(draws, 0.8, 1.1).round(2).values.tolist()
    [[8.0, 5.5], [16.0, 5.5]]
    """
    out = draws.copy()
    for col in out.columns:
        if col.startswith("heating"):
            out[col] = out[col] * heating_factor
        elif col.startswith("cooling"):
            out[col] = out[col] * cooling_factor
    return out


def aggregate_draws(
    draws: pd.DataFrame, units: pd.Series, columns: list[str]
) -> pd.DataFrame:
    """Aggregate the simulations of the tracts into larger geographic units.

    In each simulation, the value of a unit is the mean of its tracts weighted by
    their simulated dwellings, so the errors that tracts share (the same modeled
    dwellings, the same climate line) add up instead of canceling, and only the
    independent part of the error shrinks with the size of the unit.

    Parameters
    ----------
    draws : pandas.DataFrame
        Output of :func:`simulate_tract_energy`; the group ``"all"`` is used.
    units : pandas.Series
        Unit of every tract, indexed by ``GEOID``. Tracts without a unit are left
        out.
    columns : list of str
        Per-dwelling quantities to average.

    Returns
    -------
    pandas.DataFrame
        One row per unit and simulation with ``GEOID`` (the unit), ``group``
        (``"all"``), ``draw``, ``dwellings`` (sum) and the averaged columns, so
        that :func:`summarize_draws` and :func:`rank_stability` apply to it.

    Examples
    --------
    >>> draws = pd.DataFrame({"GEOID": ["a", "b"] * 2, "group": "all", "draw": [0,
    ...     0, 1, 1], "dwellings": [1.0, 3.0, 1.0, 3.0], "x": [2.0, 4.0, 2.0, 4.0]})
    >>> units = pd.Series({"a": "u", "b": "u"})
    >>> out = aggregate_draws(draws, units, ["x"])
    >>> out["x"].round(2).tolist()
    [3.5, 3.5]
    """
    sub = draws[draws["group"] == "all"].copy()
    sub["unit"] = sub["GEOID"].map(units)
    sub = sub.dropna(subset=["unit", *columns])
    weighted = pd.DataFrame(
        {col: sub[col] * sub["dwellings"] for col in columns}
    ).assign(unit=sub["unit"], draw=sub["draw"], dwellings=sub["dwellings"])
    total = weighted.groupby(["unit", "draw"]).sum().reset_index()
    for col in columns:
        total[col] = total[col] / total["dwellings"]
    return total.rename(columns={"unit": "GEOID"}).assign(group="all")


def summarize_draws(
    draws: pd.DataFrame, columns: list[str], level: float = 0.90
) -> pd.DataFrame:
    """Summarize the simulations of each tract and group.

    Parameters
    ----------
    draws : pandas.DataFrame
        Output of :func:`simulate_tract_energy` (possibly scaled).
    columns : list of str
        Columns to summarize.
    level : float
        Level of the interval; 0.90 matches the confidence of the survey margins.

    Returns
    -------
    pandas.DataFrame
        One row per tract and group with, for each column, the mean over the
        simulations (``<column>``), its standard deviation (``<column>_sd``), the
        limits of the interval (``<column>_lo`` and ``<column>_hi``), the
        coefficient of variation in percent (``<column>_cv``) and its quality
        class (``<column>_quality``: high below 12 %, medium up to 40 %, low
        above), following the convention of the project for the survey.

    Examples
    --------
    >>> draws = pd.DataFrame({"GEOID": ["a"] * 4, "group": "all", "draw": range(4),
    ...     "x": [1.0, 2.0, 3.0, 4.0]})
    >>> table = summarize_draws(draws, ["x"])
    >>> table.columns.tolist()
    ['GEOID', 'group', 'x', 'x_sd', 'x_lo', 'x_hi', 'x_cv', 'x_quality']
    """
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    grouped = draws.groupby(["GEOID", "group"])
    parts = []
    for col in columns:
        g = grouped[col]
        mean, sd = g.mean(), g.std(ddof=1)
        cv = 100 * sd / mean.where(mean > 0)
        parts.append(
            pd.DataFrame(
                {
                    col: mean,
                    f"{col}_sd": sd,
                    f"{col}_lo": g.quantile(low),
                    f"{col}_hi": g.quantile(high),
                    f"{col}_cv": cv,
                    f"{col}_quality": cv_quality(cv.to_numpy()).to_numpy(),
                }
            )
        )
    out = pd.concat(parts, axis=1)
    return out.reset_index()


def rank_stability(
    draws: pd.DataFrame,
    column: str,
    group: str = "all",
    n_classes: int = 5,
    level: float = 0.90,
) -> pd.DataFrame:
    """Measure how stable the position of each tract is among the tracts.

    In each simulation the tracts are ranked by ``column`` (percentile rank from 0
    to 1) and assigned to ``n_classes`` equal classes; the result tells how much
    the position and the class of a tract change with the uncertainty.

    Parameters
    ----------
    draws : pandas.DataFrame
        Output of :func:`simulate_tract_energy`.
    column : str
        Quantity to rank, for example ``"heating_kbtu"``.
    group : str
        Group of the simulations (``"all"``, ``"Owner"`` or ``"Renter"``).
    n_classes : int
        Number of classes (5 gives quintiles).
    level : float
        Level of the interval of the percentile rank.

    Returns
    -------
    pandas.DataFrame
        One row per tract with ``rank_mean``, ``rank_lo`` and ``rank_hi``
        (percentile rank), ``modal_class`` (class most often assigned, 1 is the
        lowest) and ``class_share`` (share of the simulations in that class).

    Examples
    --------
    >>> rank_stability(draws, "heating_kbtu", n_classes=5)  # doctest: +SKIP
    """
    sub = draws[draws["group"] == group].dropna(subset=[column])
    rank = sub.groupby("draw")[column].rank(pct=True)
    cls = np.minimum(np.ceil(rank * n_classes).astype(int), n_classes)
    work = pd.DataFrame({"GEOID": sub["GEOID"], "rank": rank, "cls": cls})
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    g = work.groupby("GEOID")
    out = pd.DataFrame(
        {
            "rank_mean": g["rank"].mean(),
            "rank_lo": g["rank"].quantile(low),
            "rank_hi": g["rank"].quantile(high),
        }
    )
    counts = work.groupby(["GEOID", "cls"]).size().unstack(fill_value=0)
    out["modal_class"] = counts.idxmax(axis=1)
    out["class_share"] = counts.max(axis=1) / counts.sum(axis=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Test with ResStock as a population whose truth is known
# ---------------------------------------------------------------------------


def _fit_without_unit(
    rest: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    end_use: str,
    extra_dwellings: Mapping[str, pd.DataFrame] | None,
    extra_stations: pd.DataFrame | None,
):
    """Fit the climate line with the dwellings outside the unit and, if any, others."""
    extra = None if extra_dwellings is None else extra_dwellings.get(end_use)
    if extra is None:
        return fit_degree_day_sensitivity(rest, station_degree_days, end_use)
    if extra_stations is None:
        raise ValueError("extra_fit_dwellings needs extra_fit_stations.")
    return fit_degree_day_sensitivity(
        pd.concat([rest, extra], ignore_index=True), extra_stations, end_use
    )


def known_population_test(
    dwellings: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    unit: str = "county",
    n_draws: int = 200,
    seed: int = 2026,
    extrapolation: str = "linear",
    transfer_tau: dict[str, float] | None = None,
    extra_fit_dwellings: Mapping[str, pd.DataFrame] | None = None,
    extra_fit_stations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate each county or PUMA of ResStock without using its own dwellings.

    The occupied modeled dwellings of ResStock are taken as a population whose
    truth is known: the mean annual load of the dwellings of a county or of a PUMA.
    For each unit, the unit is treated as a tract whose counts of dwellings by type
    are known exactly, and its energy is estimated with the method of the study
    (:func:`utah_ejpi.housing_stock.build_type_donors`, the climate line of
    :func:`utah_ejpi.energy_requirement.fit_degree_day_sensitivity` and the
    simulations of :func:`simulate_tract_energy`) using only the dwellings of the
    other units, so that the estimate and the truth do not share any dwelling. The
    climate of the unit is that of the weather stations that ResStock used for its
    dwellings (the mean of their stations, weighted by dwellings). The test measures
    the error of borrowing dwellings from other units and of the climate
    correction, and whether the simulated intervals cover it; it does not include
    the error of the Census counts, which are exact here. It follows the design of
    evaluating small-area estimators against a known population (Demombynes et al.
    2007, ``demombynes_2007_how_good_a_map``; White et al. 2025,
    ``white_2025_artificial_populations_sae``).

    Parameters
    ----------
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` of one upgrade, with the consumption
        columns; occupied dwellings are used.
    station_degree_days : pandas.DataFrame
        Contract ``climate_degree_days`` with kind ``"station"`` and ``id`` equal to
        the county code of ResStock.
    unit : {"county", "puma"}
        Units to leave out one at a time.
    n_draws : int
        Simulations of the interval of each unit.
    seed : int
        Seed of the random generator.
    extrapolation : {"linear", "cap"}
        See :func:`utah_ejpi.energy_requirement.climate_factor`.
    transfer_tau : dict, optional
        Transfer error of the simulations (see :func:`simulate_tract_energy`); the
        unit is its own transfer group.
    extra_fit_dwellings : mapping of str to pandas.DataFrame, optional
        For the uses of energy (``"heating"``, ``"cooling"``) whose climate line is
        fitted with counties of other states, the dwellings of those states. They
        are added to the dwellings outside the unit in the fit of the line only;
        they are never donors. Needs ``extra_fit_stations``.
    extra_fit_stations : pandas.DataFrame, optional
        Contract ``climate_degree_days`` with the stations of the state and of the
        other states that enter the fits.

    Returns
    -------
    pandas.DataFrame
        One row per unit with ``unit``, ``dwellings`` (modeled dwellings of the
        unit), and for heating and cooling (suffixes ``_heating`` and ``_cooling``):
        ``truth``, ``estimate``, ``lo`` and ``hi`` (interval of the simulations at the
        90 % level), ``no_climate`` (the estimate without the climate correction) and
        ``state`` (mean of all the other dwellings, ignoring the types). It also has
        the share of the dwellings of the unit served by each level of the fallback
        rule (``share_puma``, ``share_county``, ``share_similar_climate_puma`` and
        ``share_nearest_vintage``).

    Raises
    ------
    ValueError
        If the unit is unknown or the dwellings span more than one upgrade.

    Examples
    --------
    >>> known_population_test(dwellings, station_dd, unit="puma")  # doctest: +SKIP
    """
    if unit not in ("county", "puma"):
        raise ValueError("unit must be 'county' or 'puma'.")
    if dwellings["upgrade"].nunique() != 1:
        raise ValueError("Pass the dwellings of one upgrade.")
    stock = dwellings[dwellings["occupied"]].reset_index(drop=True)
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
        # Climate of the unit: mean of the stations of its dwellings.
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
        # Climate of the PUMAs of the rest of the state, to order them by closeness.
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
        main_county = truth_stock["county"].mode().iloc[0]
        master = pd.DataFrame({"GEOID": [name], "resstock_county": [main_county]})
        weights, donors = build_type_donors(counts, rest, shares, master, rank)

        fits = {
            use: _fit_without_unit(
                rest, station_degree_days, use, extra_fit_dwellings, extra_fit_stations
            )
            for use in ("heating", "cooling")
        }
        heating, cooling = fits["heating"], fits["cooling"]
        energy = build_tract_type_energy(
            weights,
            donors,
            rest,
            unit_dd,
            station_degree_days,
            heating,
            cooling,
            extrapolation,
        )
        total = energy["dwellings"].sum()
        point = {
            "heating": (energy["heating_kbtu"] * energy["dwellings"]).sum() / total,
            "cooling": (energy["cooling_kbtu"] * energy["dwellings"]).sum() / total,
        }
        raw = {
            "heating": (energy["heating_kbtu_uncorrected"] * energy["dwellings"]).sum()
            / total,
            "cooling": (energy["cooling_kbtu_uncorrected"] * energy["dwellings"]).sum()
            / total,
        }
        totals = pd.DataFrame(
            {"GEOID": [name], "total": [float(total)], "total_moe": [0.0]}
        )
        draws = simulate_tract_energy(
            weights,
            donors,
            rest,
            totals,
            unit_dd,
            station_degree_days,
            heating,
            cooling,
            within_type_dispersion(rest),
            n_draws=n_draws,
            seed=seed,
            extrapolation=extrapolation,
            transfer_tau=transfer_tau,
            transfer_groups=pd.Series({name: name}) if transfer_tau else None,
        )
        draws = draws[draws["group"] == "all"]
        row = {"unit": name, "dwellings": len(truth_stock)}
        for use, load in (
            ("heating", "heating_load_kbtu"),
            ("cooling", "cooling_load_kbtu"),
        ):
            sim = draws[f"{use}_kbtu"].dropna()
            row[f"truth_{use}"] = float(truth_stock[load].mean())
            row[f"estimate_{use}"] = float(point[use])
            row[f"lo_{use}"] = float(sim.quantile(0.05))
            row[f"hi_{use}"] = float(sim.quantile(0.95))
            row[f"no_climate_{use}"] = float(raw[use])
            row[f"state_{use}"] = float(rest[load].mean())
        by_level = (
            weights.groupby("source_level")["weight"].sum() / weights["weight"].sum()
        )
        for level in FALLBACK_LEVELS:
            row[f"share_{level}"] = float(by_level.get(level, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_transfer_tau(
    table: pd.DataFrame, min_dwellings: int = 30
) -> dict[str, float]:
    """Estimate the relative deviation of the error of borrowing dwellings.

    The squared relative error of the estimate of a unit left out has two parts:
    the statistical variance that the simulations already represent and the
    transfer error that they do not. The second is the excess of the first over
    the second, averaged over the units, and its square root is the deviation
    (``tau``). It is zero when the simulations already explain the error. The
    deviation of the simulations is taken from their 90 % interval as
    ``(hi - lo) / (2 * 1.645)`` (normal approximation).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`known_population_test` run without transfer error.
    min_dwellings : int
        Units with fewer modeled dwellings are not used: their true mean is too
        noisy to measure the error of the estimate.

    Returns
    -------
    dict
        ``{"heating": tau, "cooling": tau}``.

    Raises
    ------
    ValueError
        If no unit has ``min_dwellings`` dwellings.

    Examples
    --------
    >>> estimate_transfer_tau(known_population_table)  # doctest: +SKIP
    """
    big = table[table["dwellings"] >= min_dwellings]
    if big.empty:
        raise ValueError("No unit has enough dwellings.")
    tau = {}
    for use in ("heating", "cooling"):
        rel = (big[f"estimate_{use}"] - big[f"truth_{use}"]) / big[f"truth_{use}"]
        sd = (big[f"hi_{use}"] - big[f"lo_{use}"]) / (2 * Z90) / big[f"estimate_{use}"]
        tau[use] = float(np.sqrt(max((rel**2).mean() - (sd**2).mean(), 0.0)))
    return tau


def known_population_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize the error of the estimates against the known truth.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`known_population_test`.

    Returns
    -------
    pandas.DataFrame
        For heating and cooling and for each estimator (``estimate`` with the
        method of the study, ``no_climate`` without the climate correction and
        ``state`` the mean of the rest of the state): ``bias`` (mean relative
        error), ``rrmse`` (root mean squared relative error), ``median_abs_error``
        (median absolute relative error) and ``spearman`` (rank correlation with the
        truth among the units). For ``estimate`` it also gives ``coverage`` (share
        of units whose truth is inside the interval) and ``half_width`` (median
        half-width of the interval relative to the estimate). The number of units
        is in ``units``.

    Examples
    --------
    >>> known_population_summary(known_population_table)  # doctest: +SKIP
    """
    rows = []
    for use in ("heating", "cooling"):
        truth = table[f"truth_{use}"]
        for estimator in ("estimate", "no_climate", "state"):
            value = table[f"{estimator}_{use}"]
            rel = (value - truth) / truth.where(truth > 0)
            row = {
                "use": use,
                "estimator": estimator,
                "units": len(table),
                "bias": rel.mean(),
                "rrmse": float(np.sqrt((rel**2).mean())),
                "median_abs_error": rel.abs().median(),
                "spearman": value.corr(truth, method="spearman"),
            }
            if estimator == "estimate":
                inside = (truth >= table[f"lo_{use}"]) & (truth <= table[f"hi_{use}"])
                row["coverage"] = float(inside.mean())
                row["half_width"] = float(
                    ((table[f"hi_{use}"] - table[f"lo_{use}"]) / 2 / value).median()
                )
            rows.append(row)
    return pd.DataFrame(rows)
