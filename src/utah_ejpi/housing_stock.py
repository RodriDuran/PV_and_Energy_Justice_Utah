"""Housing stock of each tract: which modeled dwellings stand for each type.

A *type* of dwelling is a combination of tenure, year built and units in
structure, as in the American Community Survey (ACS) table B25127 (tenure by
year structure built by units in structure). The ACS gives, for every tract, how
many occupied dwellings there are of each type; ResStock gives modeled dwellings
with their energy use but only by county and PUMA (a statistical area of at
least 100,000 people). This module links the two: for every tract and type it
chooses the modeled dwellings that stand for the type, and later steps average
their energy with the dwelling counts of the tract as weights.

The base method is the mean of the modeled dwellings of the same type (the
"cell mean" estimator that the small area estimation literature describes as
consistent when dwellings of the same type behave alike in every area; Naidoo
2010, ``naidoo_2010_poverty_mapping``; Das and Lahiri 2025,
``das_2025_hb_poststratification``). Its known weakness is the cells with few or
no modeled dwellings (Gelman and Little 1997,
``gelman_little_1997_poststratification``), which are frequent here, so a fixed
order of fallback levels decides where to look for dwellings of a type that is
missing in the PUMA of the tract, following the practice of the U.S. Department
of Energy community program, which widens the search to similar dwellings when
an area has too few of them (Liu et al. 2023, ``liu_2023_leap``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import resources

import numpy as np
import pandas as pd

from utah_ejpi.validation import composition_classes, resstock_class

__all__ = [
    "FALLBACK_LEVELS",
    "REWEIGHT_SETS",
    "TIER_WEIGHTS",
    "VINTAGE_ORDER",
    "acs_type_counts",
    "b25127_cells",
    "build_type_donors",
    "class_indicators",
    "composition_shares",
    "fallback_shares",
    "read_acs_table",
    "reweight_tracts",
]

#: Year-built classes of the ACS table from the oldest to the newest that
#: ResStock has. The class "2020 or later" exists in the table but not in
#: ResStock, which models the stock of 2018.
VINTAGE_ORDER = [
    "1939 or earlier",
    "1940 to 1959",
    "1960 to 1979",
    "1980 to 1999",
    "2000 to 2019",
]
NEWEST_CLASS = "2020 or later"

#: Levels of the fallback rule, in the order they are tried.
FALLBACK_LEVELS = ("puma", "county", "similar_climate_puma", "nearest_vintage")


def read_acs_table(path: str | os.PathLike) -> pd.DataFrame:
    """Read an ACS table written by ``download_acs``.

    Parameters
    ----------
    path : path-like
        CSV file in long format with the columns ``GEOID``, ``NAME``,
        ``variable``, ``estimate`` and ``moe``.

    Returns
    -------
    pandas.DataFrame
        The table, with ``GEOID`` as a string.
    """
    return pd.read_csv(path, dtype={"GEOID": str})


def b25127_cells() -> pd.DataFrame:
    """Return the 72 cells of the ACS table B25127 as types of dwelling.

    Returns
    -------
    pandas.DataFrame
        Columns ``variable`` (for example ``B25127_004``), ``tenure``
        (``Owner`` or ``Renter``), ``vintage`` (year-built class) and
        ``building_type`` (units-in-structure class: ``1``, ``2 to 4``,
        ``5 to 19``, ``20 to 49``, ``50 or more`` or ``mobile``).

    Notes
    -----
    The table was built from the variable labels of the Census Bureau API
    (https://api.census.gov/data/2024/acs/acs5/groups/B25127.json): the 72
    variables whose label has four levels (total, tenure, year built, units).
    The subtotals are not included.
    """
    path = resources.files("utah_ejpi") / "data" / "b25127_cells.csv"
    with resources.as_file(path) as file:
        return pd.read_csv(file)


def acs_type_counts(table: pd.DataFrame, state_fips: str) -> pd.DataFrame:
    """Count the occupied dwellings of each tract by type.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`read_acs_table` for table B25127.
    state_fips : str
        Two-digit state FIPS code; tracts of other states are dropped.

    Returns
    -------
    pandas.DataFrame
        Contract ``acs_type_counts``: one row per tract and type with the
        ``estimate`` and its margin of error ``moe`` (90 % level).

    Raises
    ------
    ValueError
        If a tract lacks some of the 72 cells or has a missing estimate.
    """
    cells = b25127_cells()
    rows = table[table["GEOID"].str.startswith(state_fips)]
    out = rows.merge(cells, on="variable", how="inner")
    per_tract = out.groupby("GEOID").size()
    if (per_tract != len(cells)).any():
        raise ValueError(
            f"{int((per_tract != len(cells)).sum())} tracts do not have the "
            f"{len(cells)} cells of table B25127."
        )
    if out["estimate"].isna().any():
        raise ValueError("Some estimates of table B25127 are missing.")
    keep = ["GEOID", "tenure", "vintage", "building_type", "estimate", "moe"]
    return out[keep].sort_values(keep[:4]).reset_index(drop=True)


def _vintage_candidates(vintage: str) -> list[list[str]]:
    """Return groups of year-built classes from the nearest to the farthest.

    The class "2020 or later" is represented by the newest class that ResStock
    has, so it starts the search from there.
    """
    start = VINTAGE_ORDER.index("2000 to 2019" if vintage == NEWEST_CLASS else vintage)
    groups = []
    for distance in range(1, len(VINTAGE_ORDER)):
        group = [
            VINTAGE_ORDER[i]
            for i in (start - distance, start + distance)
            if 0 <= i < len(VINTAGE_ORDER)
        ]
        if group:
            groups.append(group)
    return groups


def build_type_donors(
    counts: pd.DataFrame,
    dwellings: pd.DataFrame,
    tract_puma_share: pd.DataFrame,
    tract_master: pd.DataFrame,
    climate_rank: pd.DataFrame,
    min_donors: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the modeled dwellings that stand for each type in each tract.

    For every tract and type the levels below are tried in order and the first
    that supplies at least ``min_donors`` modeled dwellings is used. A type
    built in 2020 or later has no modeled dwelling in ResStock and is
    represented by the dwellings of the class 2000 to 2019 with the same tenure
    and units in structure.

    1. ``puma``: dwellings of the type in the ResStock PUMAs that hold part of
       the tract. The PUMA with most of the tract's dwellings must have enough
       of them. Each PUMA counts in proportion to the share of the tract's
       occupied dwellings that lies in it, and the dwellings of a PUMA count
       equally.
    2. ``county``: dwellings of the type in the ResStock county of the tract.
    3. ``similar_climate_puma``: dwellings of the type in the PUMAs of the state,
       taking the PUMAs in order of climatic closeness to the tract (degree
       days) until there are enough.
    4. ``nearest_vintage``: dwellings with the same tenure and units in structure
       in the nearest year-built class of the state, taking classes in order of
       distance until there are enough.

    Parameters
    ----------
    counts : pandas.DataFrame
        Contract ``acs_type_counts``.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` of one upgrade (the baseline); only
        occupied dwellings are used.
    tract_puma_share : pandas.DataFrame
        Contract ``tract_puma_share``.
    tract_master : pandas.DataFrame
        Contract ``tract_master`` (gives the ResStock county of each tract).
    climate_rank : pandas.DataFrame
        Contract ``puma_climate_rank``.
    min_donors : int
        Smallest number of modeled dwellings that a level must supply. With 1
        a type is looked for elsewhere only when it is empty; a larger value
        trades locality for stability and is meant for sensitivity runs.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(weights, donors)``. ``weights`` is the contract ``tract_type_weights``
        (one row per tract and type: ``weight`` is the number of occupied
        dwellings, ``moe`` its margin of error, ``source_level`` the level used,
        ``n_donors`` the number of modeled dwellings and
        ``built_2020_or_later`` whether the type is represented by an older
        class). ``donors`` is the contract ``tract_type_donors`` (one row per
        tract, type and modeled dwelling with ``donor_weight``, which adds to 1
        within a tract and type).

    Raises
    ------
    ValueError
        If the dwellings table repeats a building or a type has no modeled
        dwelling in the whole state in any year-built class.

    Notes
    -----
    Modeled dwellings of ResStock all carry the same weight, so the mean of a
    type is the plain mean of its donors. The share of the dwellings of a tract
    served by each level is available from :func:`fallback_shares` and is
    reported with the results.
    """
    stock = dwellings[dwellings["occupied"]]
    if stock["building_id"].duplicated().any():
        raise ValueError(
            "The dwellings table has repeated buildings; pass one upgrade."
        )
    type_cols = ["tenure", "vintage", "building_type"]

    # Modeled dwellings of every type, by PUMA, by county and in the state.
    def ids_by(keys: list[str]) -> dict:
        return {
            k: np.asarray(g, dtype=np.int64)
            for k, g in stock.groupby(keys)["building_id"].apply(list).items()
        }

    by_puma = ids_by(["puma", *type_cols])
    by_county = ids_by(["county", *type_cols])
    by_state = ids_by(type_cols)

    county_of = tract_master.set_index("GEOID")["resstock_county"].to_dict()
    shares = {
        geoid: list(zip(g["resstock_puma"], g["share"], strict=True))
        for geoid, g in tract_puma_share.sort_values(
            ["GEOID", "share"], ascending=[True, False]
        ).groupby("GEOID")
    }
    ranked = {
        geoid: g.sort_values("rank")["resstock_puma"].tolist()
        for geoid, g in climate_rank.groupby("GEOID")
    }
    empty = np.empty(0, dtype=np.int64)

    def level_puma(geoid, key):
        # The PUMA with most of the tract's dwellings must have enough of them.
        parts = [(p, s) for p, s in shares[geoid] if (p, *key) in by_puma]
        if len(by_puma.get((shares[geoid][0][0], *key), empty)) < min_donors:
            return empty, empty.astype(float)
        total = sum(s for _, s in parts)
        ids, wts = [], []
        for puma, share in parts:
            members = by_puma[(puma, *key)]
            ids.append(members)
            wts.append(np.full(len(members), share / total / len(members)))
        return np.concatenate(ids), np.concatenate(wts)

    def equal(ids):
        return ids, np.full(len(ids), 1.0 / len(ids)) if len(ids) else ids.astype(float)

    def level_county(geoid, key):
        ids = by_county.get((county_of[geoid], *key), empty)
        return equal(ids) if len(ids) >= min_donors else (empty, empty.astype(float))

    def level_climate(geoid, key):
        pooled = []
        for puma in ranked[geoid]:
            pooled.append(by_puma.get((puma, *key), empty))
            if sum(len(p) for p in pooled) >= min_donors:
                return equal(np.concatenate(pooled))
        return empty, empty.astype(float)

    def level_vintage(key):
        tenure, vintage, building_type = key
        pooled = []
        for group in _vintage_candidates(vintage):
            pooled.extend(
                by_state.get((tenure, v, building_type), empty) for v in group
            )
            if sum(len(p) for p in pooled) >= min_donors:
                break
        ids = np.concatenate(pooled) if pooled else empty
        if not len(ids):
            raise ValueError(
                f"No modeled dwelling of tenure {tenure} and type "
                f"{building_type} exists in the state in any year-built class."
            )
        return equal(ids)

    weight_rows, donor_parts = [], []
    for geoid, tract in counts[counts["GEOID"].isin(shares)].groupby("GEOID"):
        for row in tract.itertuples(index=False):
            newest = row.vintage == NEWEST_CLASS
            # The class 2020 or later is represented by the class 2000 to 2019.
            key = (
                row.tenure,
                "2000 to 2019" if newest else row.vintage,
                row.building_type,
            )
            for level in FALLBACK_LEVELS:
                if level == "puma":
                    ids, wts = level_puma(geoid, key)
                elif level == "county":
                    ids, wts = level_county(geoid, key)
                elif level == "similar_climate_puma":
                    ids, wts = level_climate(geoid, key)
                else:
                    ids, wts = level_vintage(key)
                if len(ids) >= min_donors or level == FALLBACK_LEVELS[-1]:
                    break
            weight_rows.append(
                (
                    geoid,
                    row.tenure,
                    row.vintage,
                    row.building_type,
                    float(row.estimate),
                    float(row.moe),
                    newest,
                    level,
                    len(ids),
                )
            )
            donor_parts.append(
                pd.DataFrame(
                    {
                        "GEOID": geoid,
                        "tenure": row.tenure,
                        "vintage": row.vintage,
                        "building_type": row.building_type,
                        "building_id": ids,
                        "donor_weight": wts,
                        "source_level": level,
                    }
                )
            )
    weights = pd.DataFrame(
        weight_rows,
        columns=[
            "GEOID",
            "tenure",
            "vintage",
            "building_type",
            "weight",
            "moe",
            "built_2020_or_later",
            "source_level",
            "n_donors",
        ],
    )
    donors = pd.concat(donor_parts, ignore_index=True)
    return weights, donors


def fallback_shares(weights: pd.DataFrame) -> pd.DataFrame:
    """Report which share of the dwellings of each tract uses each level.

    Parameters
    ----------
    weights : pandas.DataFrame
        Contract ``tract_type_weights`` (first output of
        :func:`build_type_donors`).

    Returns
    -------
    pandas.DataFrame
        One row per tract with the share of its occupied dwellings served by
        each level (columns named after the levels) and
        ``built_2020_or_later``, the share of dwellings built in 2020 or later,
        which are represented by older ones whatever the level. Tracts without
        dwellings are left out.
    """
    total = weights.groupby("GEOID")["weight"].sum()
    out = (
        weights.pivot_table(
            index="GEOID",
            columns="source_level",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=list(FALLBACK_LEVELS), fill_value=0.0)
        .div(total, axis=0)
    )
    newest = weights[weights["built_2020_or_later"]].groupby("GEOID")["weight"].sum()
    out["built_2020_or_later"] = newest.reindex(out.index).fillna(0.0) / total
    return out[total > 0].reset_index()


# ---------------------------------------------------------------------------
# Reweighting of the modeled dwellings to several Census tables
# ---------------------------------------------------------------------------

#: Tables that can be matched, in the order they are applied. The first one, the
#: type of dwelling, is always used; the others add the heating fuel, the number
#: of bedrooms and the household size, each by tenure.
REWEIGHT_SETS = ("type", "heating_fuel", "bedrooms", "household_size")

#: Initial weight of a modeled dwelling in the PUMA of the tract, in the county,
#: in the PUMAs of closest climate, and elsewhere in the state.
TIER_WEIGHTS = (1.0, 0.1, 0.01, 0.001)


def class_indicators(
    dwellings: pd.DataFrame, sets: Sequence[str] = REWEIGHT_SETS[1:]
) -> pd.DataFrame:
    """Return the indicator of the class of every modeled dwelling in each table.

    Parameters
    ----------
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` with the columns ``tenure``,
        ``heating_fuel``, ``bedrooms`` and ``occupants`` (occupied dwellings are
        used).
    sets : sequence of str
        Tables, a subset of ``"heating_fuel"``, ``"bedrooms"`` and
        ``"household_size"``.

    Returns
    -------
    pandas.DataFrame
        Index ``building_id`` and one column of zeros and ones per class, named
        ``"table:tenure|class"`` (the same names as in
        :func:`composition_shares`). A class that no dwelling belongs to has no
        column.

    Examples
    --------
    >>> d = pd.DataFrame({"building_id": [1], "occupied": [True], "tenure": ["Owner"],
    ...                   "heating_fuel": ["Natural Gas"], "bedrooms": [3],
    ...                   "occupants": [2.0]})
    >>> class_indicators(d, ("bedrooms",)).columns.tolist()
    ['bedrooms:Owner|3']
    """
    stock = dwellings[dwellings["occupied"]].reset_index(drop=True)
    parts = []
    for name in sets:
        label = name + ":" + stock["tenure"] + "|" + resstock_class(stock, name)
        part = pd.get_dummies(label).astype(float)
        part.index = stock["building_id"]
        parts.append(part)
    return pd.concat(parts, axis=1)


def composition_shares(
    counts: pd.DataFrame,
    composition: dict[str, pd.DataFrame],
    sets: Sequence[str] = REWEIGHT_SETS[1:],
) -> pd.DataFrame:
    """Share of the dwellings of each tract in every class of the Census tables.

    Within each tenure the counts of a table are scaled to the total of that tenure
    in the table of types (the Census tables do not add to the same total), and
    every count is divided by the total of dwellings of the tract.

    Parameters
    ----------
    counts : pandas.DataFrame
        Contract ``acs_type_counts``.
    composition : dict of pandas.DataFrame
        For each table of ``sets``, the output of
        :func:`utah_ejpi.validation.observed_composition`.
    sets : sequence of str
        Tables to use.

    Returns
    -------
    pandas.DataFrame
        Index ``GEOID`` and one column per class named ``"table:tenure|class"``
        with the share of the dwellings of the tract. The classes of a tenure that
        a table does not report for a tract are zero. Tracts without dwellings in
        the table of types are left out.

    Examples
    --------
    >>> c = pd.DataFrame({"GEOID": ["a", "a"], "tenure": ["Owner", "Owner"],
    ...                   "vintage": ["1939 or earlier", "1940 to 1959"],
    ...                   "building_type": ["1", "1"], "estimate": [30.0, 70.0]})
    >>> obs = pd.DataFrame({"GEOID": ["a", "a"], "tenure": ["Owner", "Owner"],
    ...                     "class": ["utility_gas", "electricity"],
    ...                     "observed": [40.0, 10.0]})
    >>> shares = composition_shares(c, {"heating_fuel": obs}, ("heating_fuel",))
    >>> shares.round(2).to_dict("records")
    [{'heating_fuel:Owner|electricity': 0.2, 'heating_fuel:Owner|utility_gas': 0.8}]
    """
    tenure_total = counts.groupby(["GEOID", "tenure"])["estimate"].sum().unstack()
    total = tenure_total.sum(axis=1)
    pieces = []
    for name in sets:
        table = composition[name].copy()
        table["column"] = name + ":" + table["tenure"] + "|" + table["class"]
        have = table.groupby(["GEOID", "tenure"])["observed"].transform("sum")
        want = [
            tenure_total.at[g, t] if g in tenure_total.index else 0.0
            for g, t in zip(table["GEOID"], table["tenure"], strict=True)
        ]
        scaled = np.where(
            have > 0, table["observed"] * want / have.where(have > 0, 1), 0.0
        )
        denom = table["GEOID"].map(total)
        table["share"] = scaled / denom
        pieces.append(
            table.pivot_table(
                index="GEOID", columns="column", values="share", aggfunc="sum"
            )
        )
    out = pd.concat(pieces, axis=1).fillna(0.0)
    return out[out.index.isin(total[total > 0].index)]


def _rake(
    weights: np.ndarray,
    categories: list[np.ndarray],
    targets: list[np.ndarray],
    tolerance: float,
    max_iterations: int,
):
    """Apply iterative proportional fitting to a vector of weights.

    Each table is a partition of the dwellings into categories (an index per
    dwelling) with a target for the sum of the weights of every category; a
    target that is ``NaN`` is not imposed. The tables are applied one after the
    other, multiplying the weights of each category by the ratio between its
    target and its current sum, and the cycle is repeated.

    Returns
    -------
    tuple
        ``(weights, iterations, converged, max_error, infeasible, worst)`` where
        the error is the largest of ``abs(sum - target) / max(target, 1)`` over
        the categories that can be matched, ``infeasible`` lists, for each table,
        the categories with a positive target and no weight to scale and
        ``worst`` is the table and category with the largest error.
    """
    w = weights.astype(float).copy()
    infeasible = [np.zeros(len(t), dtype=bool) for t in targets]
    error = np.inf
    worst = (0, 0)
    for iteration in range(1, max_iterations + 1):
        for k, (cat, target) in enumerate(zip(categories, targets, strict=True)):
            sums = np.bincount(cat, weights=w, minlength=len(target))
            active = ~np.isnan(target)
            factor = np.ones(len(target))
            scalable = active & (sums > 0)
            factor[scalable] = target[scalable] / sums[scalable]
            infeasible[k] = active & (target > 0) & (sums <= 0)
            w *= factor[cat]
        error = 0.0
        for k, (cat, target, bad) in enumerate(
            zip(categories, targets, infeasible, strict=True)
        ):
            sums = np.bincount(cat, weights=w, minlength=len(target))
            ok = ~np.isnan(target) & ~bad
            if ok.any():
                gap = np.abs(sums - target) / np.maximum(target, 1.0)
                gap[~ok] = 0.0
                if gap.max() > error:
                    error, worst = float(gap.max()), (k, int(gap.argmax()))
        if error < tolerance:
            return w, iteration, True, error, infeasible, worst
    return w, max_iterations, False, error, infeasible, worst


def reweight_tracts(
    counts: pd.DataFrame,
    composition: dict[str, pd.DataFrame],
    dwellings: pd.DataFrame,
    tract_puma_share: pd.DataFrame,
    tract_master: pd.DataFrame,
    climate_rank: pd.DataFrame,
    sets: tuple[str, ...] = REWEIGHT_SETS,
    tier_weights: tuple[float, float, float, float] = TIER_WEIGHTS,
    n_similar_pumas: int = 5,
    tolerance: float = 0.01,
    max_iterations: int = 200,
    min_weight_share: float = 1e-4,
    tracts: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give each modeled dwelling a weight in each tract that matches Census tables.

    The occupied modeled dwellings of the whole state are the seed. Their
    initial weights favour those close to the tract (the PUMAs of the tract, then
    its county, then the PUMAs of closest climate, then the rest of the state;
    see ``tier_weights``), and iterative proportional fitting multiplies the
    weights until the weighted dwellings reproduce, in every category, the Census
    counts of the tract in the chosen tables. The fitted weights are the closest
    to the initial ones, in the sense of relative entropy, among those that match
    the tables, so a category that has dwellings near the tract is represented by
    them and a category that has none borrows from farther ones. This is the
    reweighting of ResStock to areas smaller than a PUMA done by Liu et al.
    (2023, ``liu_2023_leap``), with the sequential use of tables of LEAD (Ma et
    al. 2024, ``doe_2024_lead_methodology``); the sensitivity of the method to
    categories without seed dwellings is the one documented by Lovelace et al.
    (2015, ``lovelace_2015_ipf``).

    Parameters
    ----------
    counts : pandas.DataFrame
        Contract ``acs_type_counts`` (the 72 types of the Census table B25127).
    composition : dict of pandas.DataFrame
        For each of ``"heating_fuel"``, ``"bedrooms"`` and ``"household_size"``
        used in ``sets``, the output of
        :func:`utah_ejpi.validation.observed_composition` (``GEOID``,
        ``tenure``, ``class``, ``observed``). Within each tenure, the counts of the
        categories that some modeled dwelling can match are scaled to the total of
        that tenure that the types of ``counts`` with a modeled dwelling can reach,
        since the Census tables do not add to the same total and the dwellings of a
        type without any modeled dwelling cannot be represented.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` of one upgrade, with the columns
        ``heating_fuel``, ``bedrooms`` and ``occupants``.
    tract_puma_share, tract_master, climate_rank : pandas.DataFrame
        Contracts ``tract_puma_share``, ``tract_master`` and ``puma_climate_rank``.
    sets : tuple of str
        Tables to match, a prefix of :data:`REWEIGHT_SETS`.
    tier_weights : tuple of float
        Initial weights for the four tiers of closeness.
    n_similar_pumas : int
        Number of PUMAs of closest climate (outside the tract's PUMAs) that form
        the third tier.
    tolerance : float
        The fitting stops when no category differs from its target by more than
        this relative error (relative to the larger of the target and 1 dwelling).
    max_iterations : int
        Maximum number of cycles through the tables.
    min_weight_share : float
        Dwellings whose weight is below this share of the tract total are not kept
        in the output; the share of the weight left out is reported.
    tracts : sequence of str, optional
        Restrict the calculation to these tracts.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(weights, diagnostics)``. ``weights`` is the contract
        ``tract_dwelling_weights``: one row per tract and modeled dwelling with
        its ``weight`` (number of dwellings of the tract that it stands for, not
        an integer), ``tier`` (1 to 4) and ``copy_2020``, which marks the copy of
        a dwelling built in 2000 to 2019 that stands for the dwellings built in
        2020 or later. ``diagnostics`` is the contract ``reweighting_diagnostics``:
        for each tract, whether the fitting converged, the cycles used, the
        largest relative error, the effective number of dwellings (Kish: the
        squared sum of the weights over the sum of their squares), the share of
        the weight in each tier, the number of categories with a target and no
        dwelling to match it (and the share of the dwellings of the tract in
        them) and the share of weight left out for being small.

    Raises
    ------
    ValueError
        If ``sets`` is not a prefix of :data:`REWEIGHT_SETS`, or the dwellings
        table repeats a building.

    Notes
    -----
    The dwellings built in 2000 to 2019 enter twice, once as themselves and once
    as a stand-in for the class 2020 or later, which ResStock does not have; the
    two copies are fitted separately. Tracts with no dwellings in the Census table
    are left out.
    """
    sets = tuple(sets)
    if sets != REWEIGHT_SETS[: len(sets)] or not sets:
        raise ValueError(f"sets must be a prefix of {REWEIGHT_SETS}.")
    stock = dwellings[dwellings["occupied"]].reset_index(drop=True)
    if stock["building_id"].duplicated().any():
        raise ValueError(
            "The dwellings table has repeated buildings; pass one upgrade."
        )

    # Rows to be weighted: every dwelling, plus a copy of the ones built in 2000
    # to 2019 that stands for the class 2020 or later.
    base = stock.assign(copy_2020=False)
    newest = stock[stock["vintage"] == "2000 to 2019"].assign(
        copy_2020=True, vintage=NEWEST_CLASS
    )
    rows = pd.concat([base, newest], ignore_index=True)
    rows_type = rows["tenure"] + "|" + rows["vintage"] + "|" + rows["building_type"]
    cells = b25127_cells()
    type_vocab = (
        cells["tenure"] + "|" + cells["vintage"] + "|" + cells["building_type"]
    ).tolist()
    type_index = {name: i for i, name in enumerate(type_vocab)}
    categories = [rows_type.map(type_index).to_numpy()]
    vocabs = {"type": type_vocab}
    classes = composition_classes()
    for name in sets[1:]:
        table = classes[classes["table"] == name]
        vocab = sorted((table["tenure"] + "|" + table["class"]).unique())
        index = {v: i for i, v in enumerate(vocab)}
        row_class = rows["tenure"] + "|" + resstock_class(rows, name)
        categories.append(row_class.map(index).to_numpy())
        vocabs[name] = vocab
    if any(np.isnan(c.astype(float)).any() for c in categories):
        raise ValueError(
            "Some modeled dwellings are outside the categories of the tables."
        )
    categories = [c.astype(int) for c in categories]

    puma_codes = rows["puma"].to_numpy()
    county_codes = rows["county"].to_numpy()
    county_of = tract_master.set_index("GEOID")["resstock_county"].to_dict()
    shares = {
        g: dict(zip(x["resstock_puma"], x["share"], strict=True))
        for g, x in tract_puma_share.groupby("GEOID")
    }
    ranked = {
        g: x.sort_values("rank")["resstock_puma"].tolist()
        for g, x in climate_rank.groupby("GEOID")
    }
    type_counts = counts.pivot_table(
        index="GEOID",
        columns=["tenure", "vintage", "building_type"],
        values="estimate",
        aggfunc="sum",
    )
    comp = {}
    for name in sets[1:]:
        t = composition[name].copy()
        t["key"] = t["tenure"] + "|" + t["class"]
        comp[name] = t.pivot_table(
            index="GEOID", columns="key", values="observed", aggfunc="sum"
        )

    wanted = list(tracts) if tracts is not None else sorted(shares)
    weight_parts, diagnostics = [], []
    tenure_of_type = np.array([v.split("|")[0] for v in type_vocab])
    # A category can be matched only if some modeled dwelling of the state belongs
    # to it; the seed gives every dwelling of the state a positive weight, so this
    # does not depend on the tract.
    can_match = [
        np.bincount(c, minlength=len(vocabs[name])) > 0
        for c, name in zip(categories, sets, strict=True)
    ]
    for geoid in wanted:
        if geoid not in type_counts.index or geoid not in shares:
            continue
        type_target = np.array(
            [
                type_counts.loc[geoid].get((c.tenure, c.vintage, c.building_type), 0.0)
                for c in cells.itertuples()
            ],
            dtype=float,
        )
        total = type_target.sum()
        if total <= 0:
            continue
        targets = [type_target]
        asked = [type_target]
        # The dwellings of a type that no modeled dwelling has cannot be
        # represented, so the tables that follow are scaled to what the table of
        # types can reach in each tenure; otherwise they would ask for a total that
        # the types cannot give and the fitting would not converge.
        reachable = {
            tenure: type_target[(tenure_of_type == tenure) & can_match[0]].sum()
            for tenure in ("Owner", "Renter")
        }
        for k, name in enumerate(sets[1:], start=1):
            vocab = vocabs[name]
            observed = (
                comp[name].loc[geoid].reindex(vocab).fillna(0.0).to_numpy()
                if geoid in comp[name].index
                else np.zeros(len(vocab))
            )
            target = observed.astype(float)
            requested = target.copy()
            for tenure in ("Owner", "Renter"):
                in_tenure = np.array([v.startswith(tenure + "|") for v in vocab])
                want = reachable[tenure]
                have = target[in_tenure & can_match[k]].sum()
                # The tables of the Census do not add up to the same total, so each
                # tenure is scaled to the total that the table of types can reach.
                if have > 0 and want > 0:
                    target[in_tenure] *= want / have
                    requested[in_tenure] *= want / have
                else:
                    target[in_tenure] = np.nan
            targets.append(target)
            asked.append(requested)

        own = shares[geoid]
        in_own = np.isin(puma_codes, list(own))
        similar = [p for p in ranked.get(geoid, []) if p not in own][:n_similar_pumas]
        tier = np.full(len(rows), 4)
        tier[np.isin(puma_codes, similar)] = 3
        tier[county_codes == county_of[geoid]] = 2
        tier[in_own] = 1
        initial = np.array(tier_weights)[tier - 1].astype(float)
        for puma, share in own.items():
            initial[puma_codes == puma] = tier_weights[0] * share

        w, iterations, converged, error, infeasible, worst = _rake(
            initial, categories, targets, tolerance, max_iterations
        )
        # Categories that the Census asks for and that no modeled dwelling can fill.
        unmatched = [(t > 0) & ~m for t, m in zip(asked, can_match, strict=True)]
        n_bad = int(sum(b.sum() for b in unmatched))
        bad_share = max(
            (float(t[b].sum() / t.sum()) if t.sum() > 0 else 0.0)
            for t, b in zip(asked, unmatched, strict=True)
        )
        mass = w.sum()
        keep = w >= min_weight_share * mass
        diagnostics.append(
            {
                "GEOID": geoid,
                "converged": converged,
                "iterations": iterations,
                "max_error": error,
                "worst_table": sets[worst[0]],
                "worst_category": vocabs[sets[worst[0]]][worst[1]],
                "effective_n": float(mass**2 / (w**2).sum()) if mass > 0 else 0.0,
                "tier_1_share": float(w[tier == 1].sum() / mass) if mass > 0 else 0.0,
                "tier_2_share": float(w[tier == 2].sum() / mass) if mass > 0 else 0.0,
                "tier_3_share": float(w[tier == 3].sum() / mass) if mass > 0 else 0.0,
                "tier_4_share": float(w[tier == 4].sum() / mass) if mass > 0 else 0.0,
                "infeasible_categories": n_bad,
                "infeasible_share": bad_share,
                "pruned_share": float(w[~keep].sum() / mass) if mass > 0 else 0.0,
            }
        )
        weight_parts.append(
            pd.DataFrame(
                {
                    "GEOID": geoid,
                    "building_id": rows["building_id"].to_numpy()[keep],
                    "copy_2020": rows["copy_2020"].to_numpy()[keep],
                    "weight": w[keep],
                    "tier": tier[keep],
                }
            )
        )
    return pd.concat(weight_parts, ignore_index=True), pd.DataFrame(diagnostics)
