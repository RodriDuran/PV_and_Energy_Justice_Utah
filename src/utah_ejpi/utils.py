"""Shared helpers: Census margins of error, validation metrics and table contracts.

The table contracts describe the tables that pass between the modules of the
package (column names, kinds and key). ``validate_table`` checks a table
against its contract and fails with a message that lists every problem found,
so an error in one module is caught where the table is built and not where it
is used. The Census margin-of-error formulas follow the American Community
Survey handbook (``census_2020_acs_handbook_ch8``); the validation metrics are
the ones used to evaluate small area estimators with a known truth (White et
al. 2025, ``white_2025_artificial_populations_sae``; Demombynes et al. 2007,
``demombynes_2007_how_good_a_map``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt

__all__ = [
    "CONTRACTS",
    "TableContract",
    "coefficient_of_variation",
    "cv_quality",
    "interval_coverage",
    "moe_proportion",
    "moe_sum",
    "moe_to_se",
    "relative_bias",
    "relative_rmse",
    "validate_table",
]

# z value of the 90 % confidence level used by the American Community Survey
# to publish margins of error.
ACS_Z = 1.645


# ---------------------------------------------------------------------------
# Census margins of error
# ---------------------------------------------------------------------------


def moe_to_se(moe: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """Convert an ACS margin of error to a standard error.

    Parameters
    ----------
    moe : array-like or float
        Margin of error at the 90 % confidence level.

    Returns
    -------
    array-like or float
        Standard error, ``moe / 1.645``.

    Examples
    --------
    >>> round(moe_to_se(3860), 0)
    2347.0

    References
    ----------
    U.S. Census Bureau (2020), Understanding and Using American Community
    Survey Data, chapter 8.
    """
    return moe / ACS_Z


def coefficient_of_variation(
    estimate: np.ndarray | pd.Series | float,
    moe: np.ndarray | pd.Series | float,
) -> np.ndarray | pd.Series | float:
    """Compute the coefficient of variation of an ACS estimate, in percent.

    Parameters
    ----------
    estimate : array-like or float
        Estimates.
    moe : array-like or float
        Margins of error of the estimates.

    Returns
    -------
    array-like or float
        ``100 * moe / (1.645 * estimate)``; ``NaN`` where the estimate is zero,
        because the coefficient is undefined there.

    Examples
    --------
    >>> round(coefficient_of_variation(1000.0, 329.0), 1)
    20.0
    """
    est = np.asarray(estimate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(est == 0, np.nan, 100.0 * np.asarray(moe) / (ACS_Z * est))
    if isinstance(estimate, pd.Series):
        return pd.Series(cv, index=estimate.index, name=estimate.name)
    return cv if cv.ndim else float(cv)


def cv_quality(cv: pd.Series | np.ndarray) -> pd.Series:
    """Classify the quality of estimates by their coefficient of variation.

    Parameters
    ----------
    cv : array-like
        Coefficients of variation in percent.

    Returns
    -------
    pandas.Series
        ``"high"`` below 12 %, ``"medium"`` from 12 % to 40 %, ``"low"`` above
        40 %, and ``"undefined"`` where the coefficient is missing.

    Examples
    --------
    >>> cv_quality([5.0, 20.0, 55.0, float("nan")]).tolist()
    ['high', 'medium', 'low', 'undefined']
    """
    s = pd.Series(np.asarray(cv, dtype=float))
    out = pd.Series("undefined", index=s.index, dtype=object)
    out[s < 12] = "high"
    out[(s >= 12) & (s <= 40)] = "medium"
    out[s > 40] = "low"
    return out


def moe_sum(moes: np.ndarray | pd.DataFrame, axis: int = -1) -> np.ndarray | pd.Series:
    """Margin of error of a sum of independent ACS estimates.

    Parameters
    ----------
    moes : array-like or pandas.DataFrame
        Margins of error of the components.
    axis : int
        Axis along which the components are summed (for a DataFrame, columns
        are summed when ``axis`` is ``-1`` or ``1``).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Square root of the sum of the squared margins of error.

    Examples
    --------
    >>> round(float(moe_sum([391, 572, 459])), 0)
    831.0

    References
    ----------
    U.S. Census Bureau (2020), Understanding and Using American Community
    Survey Data, chapter 8, formula (1).
    """
    if isinstance(moes, pd.DataFrame):
        return np.sqrt((moes**2).sum(axis=1 if axis in (-1, 1) else 0))
    return np.sqrt((np.asarray(moes, dtype=float) ** 2).sum(axis=axis))


def moe_proportion(
    numerator: np.ndarray | pd.Series | float,
    denominator: np.ndarray | pd.Series | float,
    moe_numerator: np.ndarray | pd.Series | float,
    moe_denominator: np.ndarray | pd.Series | float,
) -> np.ndarray | pd.Series | float:
    """Margin of error of a proportion whose numerator is part of its denominator.

    Parameters
    ----------
    numerator, denominator : array-like or float
        Estimates; the numerator is a subset of the denominator.
    moe_numerator, moe_denominator : array-like or float
        Margins of error of the two estimates.

    Returns
    -------
    array-like or float
        ``sqrt(moe_num**2 - p**2 * moe_den**2) / den`` with ``p = num / den``.
        Where the value under the square root is negative the sign inside is
        changed to plus, as the Census Bureau prescribes (this is the formula
        of a ratio). ``NaN`` where the denominator is zero.

    Examples
    --------
    >>> round(float(moe_proportion(203119, 630498, 5070, 831)), 4)
    0.008

    References
    ----------
    U.S. Census Bureau (2020), Understanding and Using American Community
    Survey Data, chapter 8, formula (6).
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    m_num = np.asarray(moe_numerator, dtype=float)
    m_den = np.asarray(moe_denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = num / den
        radicand = m_num**2 - p**2 * m_den**2
        radicand = np.where(radicand < 0, m_num**2 + p**2 * m_den**2, radicand)
        out = np.where(den == 0, np.nan, np.sqrt(radicand) / den)
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------


def _as_replicates(estimates: np.ndarray | pd.DataFrame) -> np.ndarray:
    arr = np.asarray(estimates, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def relative_bias(
    estimates: np.ndarray | pd.DataFrame,
    truth: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Relative bias of an estimator over repeated samples, per area.

    Parameters
    ----------
    estimates : array-like, shape (n_replicates, n_areas)
        Estimate of each area in each replicated sample. A one-dimensional
        input is read as one area.
    truth : array-like or float
        True value of each area.

    Returns
    -------
    numpy.ndarray
        ``(mean over replicates of the estimate - truth) / truth`` for each
        area; ``NaN`` where the truth is zero.

    Examples
    --------
    >>> relative_bias([[9.0, 20.0], [11.0, 24.0]], [10.0, 20.0]).tolist()
    [0.0, 0.1]

    References
    ----------
    White et al. (2025), arXiv:2306.15607, evaluation metrics.
    """
    est = _as_replicates(estimates)
    tru = np.asarray(truth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tru == 0, np.nan, (est.mean(axis=0) - tru) / tru)


def relative_rmse(
    estimates: np.ndarray | pd.DataFrame,
    truth: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Relative root mean squared error of an estimator over samples, per area.

    Parameters
    ----------
    estimates : array-like, shape (n_replicates, n_areas)
        Estimate of each area in each replicated sample.
    truth : array-like or float
        True value of each area.

    Returns
    -------
    numpy.ndarray
        ``sqrt(mean((estimate - truth)**2)) / truth`` for each area; ``NaN``
        where the truth is zero.

    Examples
    --------
    >>> relative_rmse([[8.0], [12.0]], [10.0]).tolist()
    [0.2]
    """
    est = _as_replicates(estimates)
    tru = np.asarray(truth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rmse = np.sqrt(((est - tru) ** 2).mean(axis=0))
        return np.where(tru == 0, np.nan, rmse / tru)


def interval_coverage(
    lower: np.ndarray | pd.DataFrame,
    upper: np.ndarray | pd.DataFrame,
    truth: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Share of replicated intervals that contain the true value, per area.

    Parameters
    ----------
    lower, upper : array-like, shape (n_replicates, n_areas)
        Interval limits of each area in each replicated sample.
    truth : array-like or float
        True value of each area.

    Returns
    -------
    numpy.ndarray
        Proportion of replicates with ``lower <= truth <= upper``.

    Examples
    --------
    >>> interval_coverage([[1.0], [1.0]], [[3.0], [1.5]], [2.0]).tolist()
    [0.5]
    """
    lo = _as_replicates(lower)
    up = _as_replicates(upper)
    tru = np.asarray(truth, dtype=float)
    return ((lo <= tru) & (tru <= up)).mean(axis=0)


# ---------------------------------------------------------------------------
# Table contracts
# ---------------------------------------------------------------------------

# Kinds a column can have. ``float`` accepts any numeric column, since counts
# read from a file may arrive as integers. Strings can be stored as ``object``
# or as the pandas string type.
_KINDS: dict[str, Callable[[pd.Series], bool]] = {
    "str": lambda s: pdt.is_string_dtype(s) or pdt.is_object_dtype(s),
    "int": pdt.is_integer_dtype,
    "float": pdt.is_numeric_dtype,
    "bool": pdt.is_bool_dtype,
}

_Check = Callable[[pd.DataFrame], list[str]]


@dataclass(frozen=True)
class TableContract:
    """Columns, key and extra checks of a table that passes between modules.

    Parameters
    ----------
    name : str
        Name of the table.
    key : tuple of str
        Columns that identify a row; they must be unique and have no missing
        values.
    columns : dict
        Required columns and their kind: ``"str"``, ``"int"``, ``"float"``
        (any numeric) or ``"bool"``.
    checks : tuple of callables
        Functions that receive the table and return a list of problems.
    """

    name: str
    key: tuple[str, ...]
    columns: dict[str, str]
    checks: tuple[_Check, ...] = field(default_factory=tuple)


def _check_share_sums(df: pd.DataFrame) -> list[str]:
    bad = df.groupby("GEOID")["share"].sum().sub(1.0).abs() > 1e-6
    out = []
    if bad.any():
        out.append(f"share does not add to 1 in {int(bad.sum())} tracts")
    if ((df["share"] < 0) | (df["share"] > 1)).any():
        out.append("share outside [0, 1]")
    return out


def _check_donor_weights(df: pd.DataFrame) -> list[str]:
    sums = df.groupby(["GEOID", "tenure", "vintage", "building_type"])[
        "donor_weight"
    ].sum()
    bad = (sums - 1.0).abs() > 1e-9
    return [f"donor_weight does not add to 1 in {int(bad.sum())} types"] * bool(
        bad.any()
    )


def _check_nonnegative(*cols: str) -> _Check:
    def check(df: pd.DataFrame) -> list[str]:
        return [f"negative values in {c}" for c in cols if (df[c].dropna() < 0).any()]

    return check


def _check_levels(col: str, allowed: tuple[str, ...]) -> _Check:
    def check(df: pd.DataFrame) -> list[str]:
        unknown = sorted(set(df[col].dropna()) - set(allowed))
        return [f"{col} has values outside {list(allowed)}: {unknown}"] * bool(unknown)

    return check


def _check_interval_order(pairs: tuple[tuple[str, str, str], ...]) -> _Check:
    def check(df: pd.DataFrame) -> list[str]:
        out = []
        for lo, mid, hi in pairs:
            # The interval columns are optional: they are added by the step that
            # propagates the uncertainty.
            if not {lo, mid, hi} <= set(df.columns):
                continue
            if ((df[lo] > df[mid]) | (df[mid] > df[hi])).any():
                out.append(f"{lo} <= {mid} <= {hi} fails")
        return out

    return check


# A "type" of dwelling is the combination of tenure, year built and units in
# structure of the Census table B25127; the same three columns identify a type
# in every table that carries one.
_TYPE_KEY = ("tenure", "vintage", "building_type")
_TYPE_COLS = {"tenure": "str", "vintage": "str", "building_type": "str"}

# One entry per table that passes between modules.
# Tables of the hourly profile library and of the validation tests are added
# when the modules that produce them are written.
CONTRACTS: dict[str, TableContract] = {
    "tract_master": TableContract(
        "tract_master",
        ("GEOID",),
        {
            "GEOID": "str",
            "county_fips": "str",
            "resstock_puma": "str",
            "resstock_county": "str",
            "climate_zone_ba": "str",
            "climate_zone_iecc": "str",
        },
    ),
    # Share of the occupied dwellings of each tract in each ResStock PUMA; the
    # shares of a tract add to one (checked below).
    "tract_puma_share": TableContract(
        "tract_puma_share",
        ("GEOID", "resstock_puma"),
        {
            "GEOID": "str",
            "resstock_puma": "str",
            "occupied_units": "float",
            "share": "float",
        },
        (_check_share_sums, _check_nonnegative("occupied_units")),
    ),
    # Modeled dwellings of ResStock, one row per dwelling and upgrade. Loads
    # are energy delivered by the existing system (kBtu per year).
    "resstock_dwellings": TableContract(
        "resstock_dwellings",
        ("building_id", "upgrade"),
        {
            "building_id": "int",
            "upgrade": "int",
            "applicable": "bool",
            "county": "str",
            "puma": "str",
            "occupied": "bool",
            **_TYPE_COLS,
            "weight": "float",
            "heating_load_kbtu": "float",
            "cooling_load_kbtu": "float",
            "unmet_heating_hours": "float",
            "unmet_cooling_hours": "float",
            "occupants": "float",
            "weather_lat": "float",
            "weather_lon": "float",
        },
        (
            _check_nonnegative(
                "weight",
                "heating_load_kbtu",
                "cooling_load_kbtu",
                "unmet_heating_hours",
                "unmet_cooling_hours",
            ),
        ),
    ),
    # Occupied dwellings per tract and type from the Census table B25127, with
    # the margin of error (90 % level) that the survey publishes.
    "acs_type_counts": TableContract(
        "acs_type_counts",
        ("GEOID", *_TYPE_KEY),
        {"GEOID": "str", **_TYPE_COLS, "estimate": "float", "moe": "float"},
        (_check_nonnegative("estimate", "moe"),),
    ),
    # Dwellings per tract and type, and the level of the fallback rule from which
    # the modeled dwellings that stand for the type were taken: the PUMA(s) of
    # the tract, the county, the PUMAs of similar climate, or the nearest
    # year-built class (order fixed in the plan).
    "tract_type_weights": TableContract(
        "tract_type_weights",
        ("GEOID", *_TYPE_KEY),
        {"GEOID": "str", **_TYPE_COLS, "weight": "float", "source_level": "str"},
        (
            _check_nonnegative("weight"),
            _check_levels(
                "source_level",
                ("puma", "county", "similar_climate_puma", "nearest_vintage"),
            ),
        ),
    ),
    # Heating and cooling degree days for the tracts, the weather stations of
    # ResStock and the PUMAs, computed with the same temperature source and the
    # same base temperatures so that the two can be compared.
    "climate_degree_days": TableContract(
        "climate_degree_days",
        ("id", "kind"),
        {
            "id": "str",
            "kind": "str",
            "hdd": "float",
            "cdd": "float",
            "hdd_base_c": "float",
            "cdd_base_c": "float",
        },
        (
            _check_levels("kind", ("tract", "station", "puma")),
            _check_nonnegative("hdd", "cdd"),
        ),
    ),
    # Modeled dwellings that stand for each type in each tract; the weights of
    # the dwellings add to one within a tract and type (checked below).
    "tract_type_donors": TableContract(
        "tract_type_donors",
        ("GEOID", *_TYPE_KEY, "building_id"),
        {
            "GEOID": "str",
            **_TYPE_COLS,
            "building_id": "int",
            "donor_weight": "float",
            "source_level": "str",
        },
        (_check_nonnegative("donor_weight"), _check_donor_weights),
    ),
    # Weights of the modeled dwellings in each tract after matching several Census
    # tables; ``copy_2020`` marks the copy of a dwelling of 2000 to 2019 that stands
    # for the class 2020 or later.
    "tract_dwelling_weights": TableContract(
        "tract_dwelling_weights",
        ("GEOID", "building_id", "copy_2020"),
        {
            "GEOID": "str",
            "building_id": "int",
            "copy_2020": "bool",
            "weight": "float",
            "tier": "int",
        },
        (_check_nonnegative("weight"),),
    ),
    # How each tract was fitted: convergence, concentration of the weights and
    # where the weight comes from.
    "reweighting_diagnostics": TableContract(
        "reweighting_diagnostics",
        ("GEOID",),
        {
            "GEOID": "str",
            "converged": "bool",
            "iterations": "int",
            "max_error": "float",
            "worst_table": "str",
            "worst_category": "str",
            "effective_n": "float",
            "infeasible_categories": "int",
            "infeasible_share": "float",
            "pruned_share": "float",
        },
        (_check_nonnegative("max_error", "effective_n", "pruned_share"),),
    ),
    # Climatic closeness of every PUMA to every tract (1 is the closest); used
    # for the third level of the fallback rule.
    "puma_climate_rank": TableContract(
        "puma_climate_rank",
        ("GEOID", "resstock_puma"),
        {
            "GEOID": "str",
            "resstock_puma": "str",
            "distance": "float",
            "rank": "int",
        },
        (_check_nonnegative("distance"),),
    ),
    # Result of the energy block: annual heating and cooling energy per type
    # and tract; ``dwellings`` is the weight of the type. Interval columns
    # (``*_lo``, ``*_hi``) are optional and come with the uncertainty step.
    "tract_type_energy": TableContract(
        "tract_type_energy",
        ("GEOID", *_TYPE_KEY, "upgrade"),
        {
            "GEOID": "str",
            **_TYPE_COLS,
            "upgrade": "int",
            "dwellings": "float",
            "heating_kbtu": "float",
            "cooling_kbtu": "float",
        },
        (
            _check_nonnegative("dwellings", "heating_kbtu", "cooling_kbtu"),
            _check_interval_order(
                (
                    ("heating_kbtu_lo", "heating_kbtu", "heating_kbtu_hi"),
                    ("cooling_kbtu_lo", "cooling_kbtu", "cooling_kbtu_hi"),
                )
            ),
        ),
    ),
}


def validate_table(
    df: pd.DataFrame,
    contract: str | TableContract,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Check a table against its contract.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to check.
    contract : str or TableContract
        Name of a contract in ``CONTRACTS`` or a contract object.
    allow_extra : bool
        Accept columns that the contract does not list.

    Returns
    -------
    pandas.DataFrame
        The same table, so the call can be chained.

    Raises
    ------
    KeyError
        If the contract name is unknown.
    ValueError
        If any problem is found; the message lists all of them.

    Examples
    --------
    >>> t = pd.DataFrame({"GEOID": ["49001100100"], "id": [1]})
    >>> c = TableContract("demo", ("GEOID",), {"GEOID": "str"})
    >>> validate_table(t, c).shape
    (1, 2)
    """
    spec = CONTRACTS[contract] if isinstance(contract, str) else contract
    problems: list[str] = []
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    for col, kind in spec.columns.items():
        if col in df.columns and not _KINDS[kind](df[col]):
            problems.append(f"column {col} is not of kind {kind} ({df[col].dtype})")
    if not allow_extra:
        extra = [c for c in df.columns if c not in spec.columns]
        if extra:
            problems.append(f"unexpected columns: {extra}")
    key_present = all(k in df.columns for k in spec.key)
    if key_present:
        if df[list(spec.key)].isna().any().any():
            problems.append(f"missing values in the key {list(spec.key)}")
        if df.duplicated(list(spec.key)).any():
            problems.append(f"duplicated rows for the key {list(spec.key)}")
    if (
        "GEOID" in df.columns
        and not df["GEOID"].dropna().astype(str).str.len().eq(11).all()
    ):
        problems.append("GEOID values that are not 11 characters long")
    if not missing:
        for check in spec.checks:
            problems.extend(check(df))
    if problems:
        listed = "; ".join(problems)
        raise ValueError(f"table {spec.name} breaks its contract: {listed}")
    return df
