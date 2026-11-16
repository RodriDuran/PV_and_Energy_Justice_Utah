"""Annual heating and cooling energy of the dwellings of each tract.

ResStock simulates every modeled dwelling with the 2018 weather of one station
per county (National Renewable Energy Laboratory, ``resstock_trg_2025``, sections
on weather data), placed near where the population of the county lives. A tract
can be colder or hotter than that station, mostly because of its elevation. This
module corrects the annual load of each modeled dwelling to the climate of the
tract where it is used, and averages the corrected loads of the dwellings that
stand for each type of dwelling (see :mod:`utah_ejpi.housing_stock`).

The correction is a multiplicative factor built on degree days (see
:mod:`utah_ejpi.climate`). The load of a dwelling is taken to be a linear
function of the degree days with an intercept, as in the degree-day regressions
used to normalize energy use for weather (Fels 1986, ``fels_1986_prism``), and
the intercept absorbs the internal and solar gains that make the balance
temperature of a dwelling lower than its thermostat setting (Dubin 2008,
``dubin_2008_balance_point``). The sensitivity is estimated across the weather
stations of the state with the dwellings of ResStock themselves, so it needs no
assumption of proportionality. A factor keeps a dwelling without cooling at zero.

Degree days at the tracts and at the stations come from the same source (PRISM,
``prism_2018_daily_tmean``; Daly et al. 2008, ``daly_2008_prism``), which adjusts
temperature for elevation, so the factor depends only on the difference of
climate between the two places.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utah_ejpi.resstock import FUEL_COLUMNS

__all__ = [
    "AdditiveSlope",
    "DegreeDaySensitivity",
    "build_tract_type_energy",
    "climate_factor",
    "donor_level_energy",
    "fit_additive_slope",
    "fit_degree_day_sensitivity",
    "leave_one_county_out",
    "tract_energy_summary",
    "weighted_dwelling_energy",
    "weighted_tract_means",
]

#: Degree days are expressed in thousands in the fitted coefficients.
UNIT = 1000.0

#: Use of energy -> (column of the load, column of the degree days).
END_USES = {
    "heating": ("heating_load_kbtu", "hdd"),
    "cooling": ("cooling_load_kbtu", "cdd"),
}

DEFAULT_CONTROLS = ("tenure", "vintage", "building_type", "bedrooms", "floor_area")


@dataclass(frozen=True)
class DegreeDaySensitivity:
    """Relation between the annual load and the degree days of the station.

    The relative load of a dwelling is ``intercept + slope * dd`` with ``dd`` in
    thousands of degree days. Only the ratio of two values of this line is used,
    so its scale is arbitrary.

    Attributes
    ----------
    end_use : str
        ``"heating"`` or ``"cooling"``.
    intercept, slope : float
        Coefficients of the line (slope per thousand degree days).
    covariance : numpy.ndarray
        2 x 2 covariance matrix of the coefficients, robust to the correlation
        of the dwellings of a same county.
    n_dwellings, n_stations : int
        Dwellings and stations used in the fit.
    dd_min, dd_max : float
        Range of degree days of the stations used in the fit, in thousands.
    """

    end_use: str
    intercept: float
    slope: float
    covariance: np.ndarray
    n_dwellings: int
    n_stations: int
    dd_min: float
    dd_max: float


def _design(frame: pd.DataFrame, controls: tuple[str, ...]) -> np.ndarray:
    """Build the matrix of dummy variables of the control columns."""
    parts = [np.ones((len(frame), 1))]
    for col in controls:
        dummies = pd.get_dummies(frame[col].astype(str), drop_first=True, dtype=float)
        parts.append(dummies.to_numpy())
    return np.hstack(parts)


def _cluster_covariance(
    x: np.ndarray, resid: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Return the covariance of OLS coefficients robust to clusters.

    Uses the usual sandwich estimator with the small-sample factor
    ``G / (G - 1) * (N - 1) / (N - K)`` for ``G`` clusters.
    """
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((k, k))
    clusters = np.unique(groups)
    for g in clusters:
        idx = groups == g
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    g_count = len(clusters)
    factor = g_count / max(g_count - 1, 1) * (n - 1) / max(n - k, 1)
    return factor * bread @ meat @ bread


def fit_degree_day_sensitivity(
    dwellings: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    end_use: str,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> DegreeDaySensitivity:
    """Estimate how the annual load changes with the degree days of the station.

    The estimate is made in two steps. First, the logarithm of the load of the
    dwellings that have a system is regressed on the controls and on the degree
    days, only to remove the effect of the controls (type, bedrooms, floor
    area): their fitted effect gives the expected load of each dwelling at the
    reference climate. Second, the load divided by that expected load is
    regressed on the degree days with a line, because the correction is a ratio
    of two values of a linear function of the degree days.

    Parameters
    ----------
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` (occupied dwellings are used) with the
        control columns.
    station_degree_days : pandas.DataFrame
        Degree days of the weather stations, contract ``climate_degree_days``
        with kind ``"station"`` and ``id`` equal to the county code of ResStock.
    end_use : {"heating", "cooling"}
        Use of energy.
    controls : tuple of str
        Columns of ``dwellings`` that describe the dwelling and are not
        climate. They are used only to estimate the slope; they need not be
        known for the tracts.

    Returns
    -------
    DegreeDaySensitivity
        Coefficients, their covariance and the range of the degree days.

    Raises
    ------
    KeyError
        If the use of energy is unknown.
    ValueError
        If a county of the dwellings has no station.

    Notes
    -----
    Dwellings without a system (load of zero) are left out of the fit and keep
    their zero when the correction is applied. The correction is estimated
    from differences between counties, so it assumes that, once the controls are
    taken into account, the counties differ in the load of their dwellings only
    through climate.
    """
    load_col, dd_col = END_USES[end_use]
    stations = station_degree_days[station_degree_days["kind"] == "station"]
    dd_by_county = stations.set_index("id")[dd_col]
    sample = dwellings[dwellings["occupied"] & (dwellings[load_col] > 0)].copy()
    missing = set(sample["county"]) - set(dd_by_county.index)
    if missing:
        raise ValueError(f"Counties without a weather station: {sorted(missing)[:3]}")
    sample["dd"] = sample["county"].map(dd_by_county).astype(float) / UNIT

    # Step 1: remove the effect of the controls in logarithms.
    x1 = np.hstack([_design(sample, controls), sample[["dd"]].to_numpy()])
    y1 = np.log(sample[load_col].to_numpy(dtype=float))
    coef1, *_ = np.linalg.lstsq(x1, y1, rcond=None)
    expected = np.exp(x1[:, :-1] @ coef1[:-1])

    # Step 2: the relative load as a line of the degree days.
    ratio = sample[load_col].to_numpy(dtype=float) / expected
    x2 = np.column_stack([np.ones(len(sample)), sample["dd"].to_numpy()])
    coef2, *_ = np.linalg.lstsq(x2, ratio, rcond=None)
    groups = sample["county"].to_numpy()
    cov = _cluster_covariance(x2, ratio - x2 @ coef2, groups)
    used = stations[stations["id"].isin(sample["county"])][dd_col] / UNIT
    return DegreeDaySensitivity(
        end_use=end_use,
        intercept=float(coef2[0]),
        slope=float(coef2[1]),
        covariance=cov,
        n_dwellings=len(sample),
        n_stations=int(sample["county"].nunique()),
        dd_min=float(used.min()),
        dd_max=float(used.max()),
    )


@dataclass(frozen=True)
class AdditiveSlope:
    """Change of a per-dwelling quantity with the degree days of the station.

    Unlike :class:`DegreeDaySensitivity`, the quantity moves by a fixed amount per
    thousand degree days instead of by a ratio, so it can be zero or negative (a
    gap between two energy uses, for example).

    Attributes
    ----------
    end_use : str
        ``"heating"`` or ``"cooling"``: the degree days that apply.
    slope : float
        Change of the quantity per thousand degree days.
    variance : float
        Variance of the slope, robust to the correlation of the dwellings of a same
        county.
    n_dwellings, n_stations : int
        Dwellings and stations used in the fit.
    dd_min, dd_max : float
        Range of degree days of the stations used in the fit, in thousands.
    """

    end_use: str
    slope: float
    variance: float
    n_dwellings: int
    n_stations: int
    dd_min: float
    dd_max: float


def fit_additive_slope(
    values: pd.Series,
    dwellings: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    end_use: str,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> AdditiveSlope:
    """Estimate how a quantity per dwelling changes with the degree days.

    The quantity is regressed on the controls (dummy variables) and on the degree
    days of the weather station of the county; the coefficient of the degree days
    is the slope. The controls are used only to estimate it.

    Parameters
    ----------
    values : pandas.Series
        Quantity per modeled dwelling, indexed by ``building_id``. Dwellings with a
        missing value are left out.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` (occupied dwellings are used) with the
        control columns.
    station_degree_days : pandas.DataFrame
        Degree days of the weather stations, contract ``climate_degree_days``
        with kind ``"station"`` and ``id`` equal to the county code of ResStock.
    end_use : {"heating", "cooling"}
        Use of energy, which sets the degree days.
    controls : tuple of str
        Columns of ``dwellings`` that describe the dwelling and are not climate.

    Returns
    -------
    AdditiveSlope
        Slope, its variance and the range of the degree days.

    Raises
    ------
    KeyError
        If the use of energy is unknown.
    ValueError
        If a county of the dwellings has no station or fewer than two counties
        remain.

    Notes
    -----
    The slope comes from differences between counties, so it assumes that, once the
    controls are taken into account, the counties differ in the quantity only
    through climate. With few counties of high degree days it rests on them; the
    range of the fit is returned so that the caller does not extrapolate.

    Examples
    --------
    >>> frame = pd.DataFrame({
    ...     "building_id": range(6), "occupied": True,
    ...     "county": ["a", "a", "b", "b", "c", "c"], "tenure": "Owner",
    ... })
    >>> stations = pd.DataFrame({
    ...     "id": ["a", "b", "c"], "kind": "station", "cdd": [0.0, 1000.0, 2000.0],
    ... })
    >>> y = pd.Series([1.0, 1.0, 3.0, 3.0, 5.0, 5.0], index=range(6))
    >>> fit = fit_additive_slope(y, frame, stations, "cooling", ("tenure",))
    >>> round(fit.slope, 6)
    2.0
    """
    _, dd_col = END_USES[end_use]
    stations = station_degree_days[station_degree_days["kind"] == "station"]
    dd_by_county = stations.set_index("id")[dd_col]
    sample = dwellings[dwellings["occupied"]].set_index("building_id")
    sample = sample.loc[sample.index.intersection(values.dropna().index)].copy()
    missing = set(sample["county"]) - set(dd_by_county.index)
    if missing:
        raise ValueError(f"Counties without a weather station: {sorted(missing)[:3]}")
    if sample["county"].nunique() < 2:
        raise ValueError("The slope needs dwellings of at least two counties.")
    sample["dd"] = sample["county"].map(dd_by_county).astype(float) / UNIT
    x = np.hstack([_design(sample, controls), sample[["dd"]].to_numpy()])
    y = values.loc[sample.index].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    cov = _cluster_covariance(x, y - x @ coef, sample["county"].to_numpy())
    used = stations[stations["id"].isin(sample["county"])][dd_col] / UNIT
    return AdditiveSlope(
        end_use=end_use,
        slope=float(coef[-1]),
        variance=float(cov[-1, -1]),
        n_dwellings=len(sample),
        n_stations=int(sample["county"].nunique()),
        dd_min=float(used.min()),
        dd_max=float(used.max()),
    )


def climate_factor(
    sensitivity: DegreeDaySensitivity,
    degree_days_tract: np.ndarray | pd.Series,
    degree_days_station: np.ndarray | pd.Series,
    extrapolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the factor that takes a load from the station climate to the tract.

    Parameters
    ----------
    sensitivity : DegreeDaySensitivity
        Output of :func:`fit_degree_day_sensitivity`.
    degree_days_tract, degree_days_station : array-like
        Degree days (not in thousands) of the tract and of the station of the
        dwelling, one pair per dwelling.
    extrapolation : {"linear", "cap"}
        What to do when the tract is outside the range of degree days of the
        stations used in the fit. ``"linear"`` extends the line; ``"cap"`` keeps
        the tract at the edge of the range, which gives the smallest correction
        that the data support.

    Returns
    -------
    tuple of numpy.ndarray
        ``(factor, extrapolated)``: the multiplicative factor (at least zero)
        and whether the tract is outside the fitted range.

    Raises
    ------
    ValueError
        If the extrapolation rule is unknown or the line is not positive at the
        station (which would make the ratio meaningless).

    Examples
    --------
    >>> s = DegreeDaySensitivity("heating", 0.0, 1.0, np.eye(2), 10, 3, 1.0, 3.0)
    >>> factor, out = climate_factor(s, [2000.0, 6000.0], [2000.0, 3000.0])
    >>> factor.tolist(), out.tolist()
    ([1.0, 2.0], [False, True])
    >>> climate_factor(s, [6000.0], [3000.0], "cap")[0].tolist()
    [1.0]
    """
    if extrapolation not in ("linear", "cap"):
        raise ValueError("extrapolation must be 'linear' or 'cap'.")
    dd_t = np.asarray(degree_days_tract, dtype=float) / UNIT
    dd_s = np.asarray(degree_days_station, dtype=float) / UNIT
    outside = (dd_t < sensitivity.dd_min) | (dd_t > sensitivity.dd_max)
    if extrapolation == "cap":
        dd_t = np.clip(dd_t, sensitivity.dd_min, sensitivity.dd_max)
    line_t = sensitivity.intercept + sensitivity.slope * dd_t
    line_s = sensitivity.intercept + sensitivity.slope * dd_s
    if (line_s <= 0).any():
        raise ValueError("The fitted line is not positive at a weather station.")
    return np.clip(line_t, 0.0, None) / line_s, outside


def leave_one_county_out(
    dwellings: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    end_use: str,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> pd.DataFrame:
    """Check how well the degree days predict the load of a county left out.

    For each county, the sensitivity is estimated without its dwellings and used
    to predict the mean relative load of that county from the mean of the other
    counties; the prediction is compared with a prediction that ignores the
    climate (the relative load of the other counties).

    Parameters
    ----------
    dwellings, station_degree_days, end_use, controls
        As in :func:`fit_degree_day_sensitivity`.

    Returns
    -------
    pandas.DataFrame
        One row per county with ``county``, ``n``, ``observed`` (mean load of
        the county relative to the expected load given the controls),
        ``predicted_with_climate`` and ``predicted_without_climate``.

    Notes
    -----
    The expected load given the controls is computed with all the counties, so
    this is a check of the climate line and not of the whole procedure.
    """
    load_col, dd_col = END_USES[end_use]
    stations = station_degree_days[station_degree_days["kind"] == "station"]
    dd_by_county = stations.set_index("id")[dd_col]
    sample = dwellings[dwellings["occupied"] & (dwellings[load_col] > 0)].copy()
    sample["dd"] = sample["county"].map(dd_by_county).astype(float) / UNIT
    x1 = np.hstack([_design(sample, controls), sample[["dd"]].to_numpy()])
    coef1, *_ = np.linalg.lstsq(
        x1, np.log(sample[load_col].to_numpy(dtype=float)), rcond=None
    )
    sample["ratio"] = sample[load_col] / np.exp(x1[:, :-1] @ coef1[:-1])
    rows = []
    for county, test in sample.groupby("county"):
        train = sample[sample["county"] != county]
        x2 = np.column_stack([np.ones(len(train)), train["dd"]])
        coef, *_ = np.linalg.lstsq(x2, train["ratio"].to_numpy(), rcond=None)
        dd_test = float(test["dd"].iloc[0])
        rows.append(
            (
                county,
                len(test),
                float(test["ratio"].mean()),
                float(coef[0] + coef[1] * dd_test),
                float(train["ratio"].mean()),
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "county",
            "n",
            "observed",
            "predicted_with_climate",
            "predicted_without_climate",
        ],
    )


def _correct_to_tract_climate(
    data: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    extrapolation: str,
    fuels: list[str],
) -> pd.DataFrame:
    """Multiply the loads and consumption of each row by its climate factor.

    The rows need ``GEOID`` (tract), ``county`` (station of the dwelling), the
    load columns and the consumption columns listed in ``fuels``.
    """
    tract_dd = tract_degree_days[tract_degree_days["kind"] == "tract"].set_index("id")
    station_dd = station_degree_days[
        station_degree_days["kind"] == "station"
    ].set_index("id")
    for sens, load, dd_col in (
        (heating, "heating_load_kbtu", "hdd"),
        (cooling, "cooling_load_kbtu", "cdd"),
    ):
        use = sens.end_use
        factor, outside = climate_factor(
            sens,
            data["GEOID"].map(tract_dd[dd_col]).to_numpy(),
            data["county"].map(station_dd[dd_col]).to_numpy(),
            extrapolation,
        )
        data[f"factor_{use}"] = factor
        data[f"extrapolated_{use}"] = outside
        data[f"{use}_kbtu_uncorrected"] = data[load]
        data[f"{use}_kbtu"] = data[load] * factor
        for col in (c for c in fuels if c.startswith(use)):
            data[f"{col}_uncorrected"] = data[col]
            data[col] = data[col] * factor
    for use in ("heating", "cooling"):
        parts = [
            c for c in fuels if c.startswith(use) and not c.endswith("_uncorrected")
        ]
        if parts:
            data[f"{use}_kwh"] = data[parts].sum(axis=1)
            data[f"{use}_kwh_uncorrected"] = data[
                [f"{c}_uncorrected" for c in parts]
            ].sum(axis=1)
    return data.drop(columns=["heating_load_kbtu", "cooling_load_kbtu"])


def donor_level_energy(
    weights: pd.DataFrame,
    donors: pd.DataFrame,
    dwellings: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    extrapolation: str = "linear",
) -> pd.DataFrame:
    """Correct the energy of every donor dwelling to the climate of its tract.

    Each modeled dwelling that stands for a type of dwelling in a tract has its
    annual load multiplied by the climate factor of the tract relative to its
    weather station (see :func:`climate_factor`). The annual consumption of each
    fuel is multiplied by the factor of its use, which assumes that the
    efficiency of the equipment does not change with the climate; this is not
    true for heat pumps, whose efficiency falls in the cold.

    Parameters
    ----------
    weights, donors : pandas.DataFrame
        Contracts ``tract_type_weights`` and ``tract_type_donors``.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` of one upgrade. If it has the annual
        consumption columns of :data:`utah_ejpi.resstock.FUEL_COLUMNS`, they are
        corrected too.
    tract_degree_days, station_degree_days : pandas.DataFrame
        Contract ``climate_degree_days`` with the kinds ``"tract"`` and
        ``"station"``.
    heating, cooling : DegreeDaySensitivity
        Sensitivities of the heating and cooling loads.
    extrapolation : {"linear", "cap"}
        See :func:`climate_factor`.

    Returns
    -------
    pandas.DataFrame
        One row per tract, type and donor, with the type (``tenure``,
        ``vintage``, ``building_type``), ``building_id``, ``donor_weight`` (weight
        within the type), ``dwellings`` (occupied dwellings of the type),
        ``weight`` (dwellings that this donor stands for), the donor's
        ``heating_fuel``, ``county`` and ``occupants``, the loads and consumption
        before (``*_uncorrected``) and after the correction, and the factors
        ``factor_heating`` and ``factor_cooling`` with the marks
        ``extrapolated_heating`` and ``extrapolated_cooling``.

    Raises
    ------
    ValueError
        If the dwellings table spans more than one upgrade.
    """
    if dwellings["upgrade"].nunique() != 1:
        raise ValueError("Pass the dwellings of one upgrade.")
    fuels = [c for c in FUEL_COLUMNS if c in dwellings.columns]
    attrs = ["building_id", "county", "occupants", "upgrade"]
    attrs += [c for c in ("heating_fuel",) if c in dwellings.columns]
    cols = attrs + [c for _, (load, _) in END_USES.items() for c in (load,)] + fuels
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    data = donors[[*keys, "building_id", "donor_weight"]].merge(
        dwellings[cols], on="building_id", how="left", validate="many_to_one"
    )
    data = data.merge(
        weights[[*keys, "weight"]].rename(columns={"weight": "dwellings"}), on=keys
    )
    data["weight"] = data["dwellings"] * data["donor_weight"]
    return _correct_to_tract_climate(
        data,
        tract_degree_days,
        station_degree_days,
        heating,
        cooling,
        extrapolation,
        fuels,
    )


def build_tract_type_energy(
    weights: pd.DataFrame,
    donors: pd.DataFrame,
    dwellings: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    extrapolation: str = "linear",
) -> pd.DataFrame:
    """Compute the annual heating and cooling energy of each type in each tract.

    The corrected energy of the donor dwellings of a type (see
    :func:`donor_level_energy`) is averaged with the weights of the donors, which
    gives the mean energy of a dwelling of the type in the tract.

    Parameters
    ----------
    weights, donors, dwellings, tract_degree_days, station_degree_days, heating, cooling
        As in :func:`donor_level_energy`.
    extrapolation : {"linear", "cap"}
        See :func:`climate_factor`.

    Returns
    -------
    pandas.DataFrame
        Contract ``tract_type_energy``: ``GEOID``, type, ``upgrade``,
        ``dwellings`` (occupied dwellings of the type), ``heating_kbtu`` and
        ``cooling_kbtu`` (annual energy delivered per dwelling, corrected),
        plus ``heating_kbtu_uncorrected`` and ``cooling_kbtu_uncorrected`` (the
        same mean without the climate correction), ``occupants`` (mean
        occupants per dwelling of the donors), the shares ``extrapolated_heating``
        and ``extrapolated_cooling`` of the donors whose tract lies outside the
        range of the stations and, when the dwellings carry their annual
        consumption, ``heating_kwh``, ``cooling_kwh`` and the consumption by fuel
        (corrected and ``*_uncorrected``), all per dwelling.
    """
    data = donor_level_energy(
        weights,
        donors,
        dwellings,
        tract_degree_days,
        station_degree_days,
        heating,
        cooling,
        extrapolation,
    )
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    skip = set(keys) | {
        "building_id",
        "donor_weight",
        "dwellings",
        "weight",
        "county",
        "heating_fuel",
        "upgrade",
        "occupants",
        "factor_heating",
        "factor_cooling",
    }
    values = [c for c in data.columns if c not in skip]
    flags = [c for c in values if c.startswith("extrapolated_")]
    data[flags] = data[flags].astype(float)
    data["occupants"] = data["occupants"].astype(float)
    weighted = data[[*values, "occupants"]].mul(data["donor_weight"], axis=0)
    agg = pd.concat([data[keys], weighted], axis=1).groupby(keys, as_index=False).sum()
    out = weights[[*keys, "weight"]].merge(
        agg, on=keys, how="left", validate="one_to_one"
    )
    out = out.rename(columns={"weight": "dwellings"})
    out.insert(4, "upgrade", int(dwellings["upgrade"].iloc[0]))
    return out


def tract_energy_summary(tract_type_energy: pd.DataFrame) -> pd.DataFrame:
    """Summarize the energy of each tract over its types of dwelling.

    Parameters
    ----------
    tract_type_energy : pandas.DataFrame
        Output of :func:`build_tract_type_energy`.

    Returns
    -------
    pandas.DataFrame
        One row per tract with ``dwellings``, the mean annual ``heating_kbtu`` and
        ``cooling_kbtu`` per dwelling (types weighted by their dwellings), the
        totals ``heating_kbtu_total`` and ``cooling_kbtu_total``, ``persons``,
        the energy per person and the share of dwellings whose correction
        extrapolated beyond the stations (``extrapolated_heating`` and
        ``extrapolated_cooling``). Tracts without dwellings are left out.
    """
    d = tract_type_energy
    w = d["dwellings"]
    frame = pd.DataFrame(
        {
            "GEOID": d["GEOID"],
            "dwellings": w,
            "heating_total": d["heating_kbtu"] * w,
            "cooling_total": d["cooling_kbtu"] * w,
            "persons": d["occupants"] * w,
            "extrapolated_heating": d["extrapolated_heating"] * w,
            "extrapolated_cooling": d["extrapolated_cooling"] * w,
        }
    )
    g = frame.groupby("GEOID").sum()
    g = g[g["dwellings"] > 0]
    out = pd.DataFrame(
        {
            "dwellings": g["dwellings"],
            "heating_kbtu": g["heating_total"] / g["dwellings"],
            "cooling_kbtu": g["cooling_total"] / g["dwellings"],
            "heating_kbtu_total": g["heating_total"],
            "cooling_kbtu_total": g["cooling_total"],
            "persons": g["persons"],
            "heating_kbtu_per_person": g["heating_total"]
            / g["persons"].where(g["persons"] > 0),
            "cooling_kbtu_per_person": g["cooling_total"]
            / g["persons"].where(g["persons"] > 0),
            "extrapolated_heating": g["extrapolated_heating"] / g["dwellings"],
            "extrapolated_cooling": g["extrapolated_cooling"] / g["dwellings"],
        }
    )
    return out.reset_index()


def weighted_dwelling_energy(
    dwelling_weights: pd.DataFrame,
    dwellings: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    extrapolation: str = "linear",
) -> pd.DataFrame:
    """Correct to the climate of each tract the energy of its reweighted dwellings.

    Parameters
    ----------
    dwelling_weights : pandas.DataFrame
        Weights of the modeled dwellings in each tract, as returned by
        :func:`utah_ejpi.housing_stock.reweight_tracts`: ``GEOID``,
        ``building_id``, ``copy_2020`` and ``weight``.
    dwellings, tract_degree_days, station_degree_days, heating, cooling
        As in :func:`donor_level_energy`.
    extrapolation : {"linear", "cap"}
        See :func:`climate_factor`.

    Returns
    -------
    pandas.DataFrame
        One row per tract and weighted dwelling with its weight, attributes
        (``tenure``, ``building_type``, ``heating_fuel``, ``county``,
        ``occupants``) and the loads and consumption before (``*_uncorrected``)
        and after the correction, as in :func:`donor_level_energy`.

    Raises
    ------
    ValueError
        If the dwellings table spans more than one upgrade.
    """
    if dwellings["upgrade"].nunique() != 1:
        raise ValueError("Pass the dwellings of one upgrade.")
    fuels = [c for c in FUEL_COLUMNS if c in dwellings.columns]
    attrs = ["building_id", "county", "occupants", "upgrade", "tenure", "building_type"]
    attrs += [c for c in ("heating_fuel", "cooling_type") if c in dwellings.columns]
    cols = attrs + [load for load, _ in END_USES.values()] + fuels
    data = dwelling_weights.merge(
        dwellings[cols], on="building_id", how="left", validate="many_to_one"
    )
    return _correct_to_tract_climate(
        data,
        tract_degree_days,
        station_degree_days,
        heating,
        cooling,
        extrapolation,
        fuels,
    )


def weighted_tract_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Average the energy of the weighted dwellings of each tract.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`weighted_dwelling_energy`.

    Returns
    -------
    pandas.DataFrame
        One row per tract with ``dwellings`` (sum of the weights), the mean per
        dwelling of every energy column of the input (loads and consumption,
        before and after the correction), ``occupants`` (mean occupants), the
        ``persons`` of the tract and the shares ``extrapolated_heating`` and
        ``extrapolated_cooling`` of the dwellings whose tract lies outside the
        range of the stations.
    """
    weight = frame["weight"]
    skip = {"GEOID", "building_id", "copy_2020", "weight", "county", "upgrade"}
    skip |= {
        "tenure",
        "building_type",
        "heating_fuel",
        "cooling_type",
        "factor_heating",
        "factor_cooling",
    }
    values = [c for c in frame.columns if c not in skip]
    flags = [c for c in values if c.startswith("extrapolated_")]
    work = frame[values].astype(float).mul(weight, axis=0)
    work.insert(0, "GEOID", frame["GEOID"])
    work["dwellings"] = weight
    sums = work.groupby("GEOID").sum()
    sums = sums[sums["dwellings"] > 0]
    out = sums[values].div(sums["dwellings"], axis=0)
    out["dwellings"] = sums["dwellings"]
    out["persons"] = sums["occupants"]
    out[flags] = sums[flags].div(sums["dwellings"], axis=0)
    return out.reset_index()
