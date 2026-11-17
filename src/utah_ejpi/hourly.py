"""Hourly profiles of the energy of each tract.

ResStock publishes, for every modeled dwelling, a series with a value every 15
minutes of the whole year (35,040 values) for each end use and fuel, the heating
and cooling loads delivered, and the indoor and outdoor temperatures. The annual
energy of a tract (see :mod:`utah_ejpi.energy_requirement`) is a weighted mean of
the modeled dwellings that stand for the types of dwelling of the tract, each one
multiplied by a factor of climate in heating and in cooling. The hourly profile of
a tract is the same weighted mean applied to the hourly series, so that its sum
over the year is the annual energy of the tract and the two cannot disagree:

``profile(tract, hour) = sum over dwellings of coefficient(tract, dwelling) *
factor(tract, dwelling, use) * series(dwelling, hour)``

where the coefficient is the share of the dwellings of the tract that the modeled
dwelling stands for. Only the heating and cooling columns carry the climate
factor, which rescales the whole year. The other uses are taken as modeled.

Shape of heating and cooling. The hourly shape of a series is that of the weather
station of the county of the dwelling, which has too many hours of peak load in a
tract colder than the station. When daily temperatures of the tracts and the
stations are given to :func:`build_tract_profiles`, the heating and cooling columns
of each dwelling take, in each tract, the shape of the response of the load of the
dwelling to the hourly outdoor temperature (:func:`fit_temperature_response`):
the series of the dwelling plus the difference between that response evaluated at
the hourly temperature of the tract (:func:`tract_hourly_temperature`) and at the
temperature of the station (:func:`reshape_load`), with a floor of zero, rescaled
so that the annual sum is unchanged. The annual energy is then the same with or
without the shape. Keeping the series of the dwelling keeps its own pattern of
occupancy and thermostat, which the response alone loses.

The profiles are written in one Parquet file per tract inside a folder
(``GEOID=<tract>/profile.parquet``), which :func:`read_tract_profile` and
:func:`read_profiles` read back.

Time. The series of ResStock have a value every 15 minutes stamped at the end of
the interval (the first one is 00:15 of January 1, 2018 and the last one 00:00 of
January 1, 2019). The four values of an hour are added (energy) or averaged
(temperature), and the hour is labeled with its start. The time zone of the
stamps is not stated in the data dictionary (NO VERIFICADO); comparing the
outdoor temperature of the series with the weather file of the same county, the
series run 2 hours ahead of the local standard time of the weather file all year
(no daylight saving), which is consistent with United States Eastern Standard Time
(UTC-5) for Utah (UTC-7). The shift is a property of the time zone of the county,
so it is estimated for every county by :func:`county_clock_shifts` (0 to 3 hours in
the contiguous states if the series are in Eastern Standard Time everywhere, which
was checked only in Utah, NO VERIFICADO elsewhere) and passed to the functions that
read the series. The year is cyclic (the first hours of the files are the last of
the local year), so each series is rotated by its shift and the profiles are in
local standard time of the state (without daylight saving), without a zone in the
timestamps.

The columns of indoor and outdoor temperature are not described in the data
dictionary of ResStock (NO VERIFICADO); their meaning is taken from their names.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utah_ejpi.energy_requirement import (
    DegreeDaySensitivity,
    donor_level_energy,
)
from utah_ejpi.resstock import FUEL_COLUMNS

__all__ = [
    "COMFORT_LIMITS_C",
    "INDICATORS",
    "MAX_CLOCK_SHIFT_HOURS",
    "TemperatureResponse",
    "build_tract_profiles",
    "county_clock_shifts",
    "donor_coefficients",
    "estimate_clock_shift_hours",
    "fit_temperature_response",
    "fuel_use_profiles",
    "hourly_components",
    "monthly_sums",
    "read_dwelling_hourly",
    "reshape_load",
    "read_profiles",
    "state_annual_means",
    "state_hourly_means",
    "state_monthly_means",
    "read_tract_profile",
    "short_name",
    "temperature_response_shape",
    "tract_hourly_temperature",
]

logger = logging.getLogger(__name__)

#: Hours of a (non-leap) year; 2018 is the weather year of ResStock.
HOURS = 8760
#: Values of 15 minutes of the year.
STEPS_PER_HOUR = 4
#: Largest shift (hours) that :func:`estimate_clock_shift_hours` tries by default:
#: Eastern against Pacific standard time is 3 hours, Alaska 4, Hawaii 5.
MAX_CLOCK_SHIFT_HOURS = 6
#: Base temperatures (°C) tried when fitting the response of the load to the outdoor
#: temperature, one degree apart.
BASE_GRID_C = {
    "heating": np.arange(8.0, 24.0),
    "cooling": np.arange(8.0, 28.0),
}
HOURS_PER_DAY = 24

#: Indoor temperature limits of comfort (WHO 2018 for winter; the guidance of
#: Tham et al. 2020 for summer), in degrees Celsius.
COMFORT_LIMITS_C = (18.0, 26.0)

_INDOOR = "out.indoor_temperature.conditioned_space..c"
_OUTDOOR = "out.outdoor_air_drybulb_temp..c"
_OCCUPANTS = "out.schedules.occupants"
_FUELS = ("electricity", "natural_gas", "propane", "fuel_oil")
_ENERGY = re.compile(
    r"^out\.(?:" + "|".join(_FUELS) + r")\.(?!total\.|net\.)[a-z_]+"
    r"\.energy_consumption\.\.kwh$"
)
_HEATING = re.compile(r"\.heating")
_COOLING = re.compile(r"\.cooling")

#: Names added to the profile of a tract: share of its dwellings below / above the
#: comfort limits in each hour, share with someone at home, and share below / above
#: the limits while someone is at home (the hours that count as a comfort deficit).
SHARE_BELOW = "share_below_comfort"
SHARE_ABOVE = "share_above_comfort"
SHARE_OCCUPIED = "share_occupied"
SHARE_BELOW_OCCUPIED = "share_below_comfort_occupied"
SHARE_ABOVE_OCCUPIED = "share_above_comfort_occupied"
INDICATORS = (
    SHARE_BELOW,
    SHARE_ABOVE,
    SHARE_OCCUPIED,
    SHARE_BELOW_OCCUPIED,
    SHARE_ABOVE_OCCUPIED,
)


def short_name(column: str) -> str:
    """Return the short name used in the profiles for a ResStock column.

    Parameters
    ----------
    column : str
        Name of a ResStock column, for example
        ``"out.electricity.heating.energy_consumption..kwh"``.

    Returns
    -------
    str
        The name without the ``out.`` prefix and the ``energy_consumption`` or
        ``energy_delivered`` words, with the unit as a suffix and underscores,
        for example ``"electricity_heating_kwh"``.

    Examples
    --------
    >>> short_name("out.electricity.heating.energy_consumption..kwh")
    'electricity_heating_kwh'
    >>> short_name("out.load.cooling.energy_delivered..kbtu")
    'load_cooling_kbtu'
    """
    name = column.removeprefix("out.")
    name = name.replace(".energy_consumption", "").replace(".energy_delivered", "")
    name = re.sub(r"\.\.(kwh|kbtu|c)$", r"_\1", name)
    return name.replace(".", "_")


def hourly_components(names: Sequence[str]) -> list[str]:
    """Select the columns of a ResStock series that make up the profiles.

    Parameters
    ----------
    names : sequence of str
        Column names of one time series file.

    Returns
    -------
    list of str
        Consumption of every end use and fuel in kWh (electricity, natural gas,
        propane and fuel oil, including the photovoltaic generation that some
        dwellings already have, and excluding the totals, which are the sum of
        the end uses and are derived when needed), the loads in kBtu, and the
        indoor and outdoor temperatures, and the schedule of occupants (fraction of
        the occupants at home; NO VERIFICADO, not in the data dictionary).
        Intensities, emissions and the other schedules are left out.

    Raises
    ------
    KeyError
        If a temperature column is missing.

    Examples
    --------
    >>> hourly_components(series_columns)  # doctest: +SKIP
    """
    energy = [n for n in names if _ENERGY.match(n)]
    loads = [n for n in names if n.startswith("out.load.") and n.endswith("..kbtu")]
    for needed in (_INDOOR, _OUTDOOR, _OCCUPANTS):
        if needed not in names:
            raise KeyError(f"The series lack the column {needed}.")
    return [*energy, *loads, _INDOOR, _OUTDOOR, _OCCUPANTS]


def _is_mean(column: str) -> bool:
    """Temperatures and the occupancy fraction are averaged, energy is added."""
    return column.endswith("..c") or column == _OCCUPANTS


def read_dwelling_hourly(
    path: str | os.PathLike,
    components: Sequence[str],
    clock_shift_hours: int,
    comfort_limits: tuple[float, float] = COMFORT_LIMITS_C,
) -> pd.DataFrame:
    """Read the series of one modeled dwelling and reduce them to hours.

    Parameters
    ----------
    path : path-like
        Parquet file of one dwelling (``<building id>-<upgrade>.parquet``).
    components : sequence of str
        Columns to read, as returned by :func:`hourly_components`.
    clock_shift_hours : int
        Hours by which the clock of the files runs ahead of the local standard time
        of the county of the dwelling (see :func:`county_clock_shifts`); the year is
        rotated so that row 0 is the hour starting at 00:00 local standard time on
        January 1.
    comfort_limits : tuple of float
        Lower and upper limits of the indoor temperature of comfort (°C).

    Returns
    -------
    pandas.DataFrame
        8,760 rows, one per hour of the year in local standard time, with the short
        names of the columns (a load or end use that the series lack for the
        dwelling, which ResStock writes as a column of missing values, is zero):
        energy and loads added over the four values of the hour, temperatures
        averaged, and the indicators of :data:`INDICATORS` (1 or 0): the hourly mean
        indoor temperature below / above the limits, someone at home (the hourly
        mean of the schedule of occupants above zero), and both conditions at
        once.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the series do not have 35,040 values in order, one every 15 minutes.

    Examples
    --------
    >>> read_dwelling_hourly("100-0.parquet", components, clock_shift_hours=2)
    ...     # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Time series not found: {path}")
    raw = pd.read_parquet(path, columns=["timestamp", *components])
    steps = HOURS * STEPS_PER_HOUR
    if len(raw) != steps:
        raise ValueError(f"{path.name} has {len(raw)} rows, expected {steps}.")
    gaps = raw["timestamp"].diff().dropna()
    if not (gaps == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{path.name} is not a regular series of 15 minutes.")
    out = {}
    for column in components:
        values = raw[column].to_numpy(dtype=np.float64).reshape(HOURS, STEPS_PER_HOUR)
        reduced = values.mean(axis=1) if _is_mean(column) else values.sum(axis=1)
        if not _is_mean(column):
            # A load or end use the dwelling lacks comes as a column of missing values
            reduced = np.nan_to_num(reduced, nan=0.0)
        out[short_name(column)] = np.roll(reduced, -clock_shift_hours)
    frame = pd.DataFrame(out)
    indoor = frame[short_name(_INDOOR)]
    low, high = comfort_limits
    home = frame[short_name(_OCCUPANTS)] > 0
    frame[SHARE_BELOW] = (indoor < low).astype(float)
    frame[SHARE_ABOVE] = (indoor > high).astype(float)
    frame[SHARE_OCCUPIED] = home.astype(float)
    frame[SHARE_BELOW_OCCUPIED] = ((indoor < low) & home).astype(float)
    frame[SHARE_ABOVE_OCCUPIED] = ((indoor > high) & home).astype(float)
    return frame


@dataclass(frozen=True)
class TemperatureResponse:
    """Response of the hourly load of one dwelling to the outdoor temperature.

    Attributes
    ----------
    use : {"heating", "cooling"}
        End use of the load.
    base_c : float
        Base temperature (°C): the load is zero above it in heating and below it
        in cooling.
    coefficients : numpy.ndarray, shape (24,)
        Load per degree (kBtu per hour per °C) in each hour of the day.
    r2 : float
        Share of the variance of the hourly load that the response explains.
    """

    use: str
    base_c: float
    coefficients: np.ndarray
    r2: float


def _degrees(temperature: np.ndarray, base_c: float, use: str) -> np.ndarray:
    """Degrees of heating (below the base) or cooling (above it), by day and hour."""
    excess = base_c - temperature if use == "heating" else temperature - base_c
    return np.maximum(excess, 0.0).reshape(-1, HOURS_PER_DAY)


def fit_temperature_response(
    load: Sequence[float], temperature: Sequence[float], use: str
) -> TemperatureResponse | None:
    """Fit the response of an hourly load to the hourly outdoor temperature.

    The load of hour ``h`` of the day is ``coefficient[h] * degrees``, where the
    degrees are ``max(base - temperature, 0)`` in heating and
    ``max(temperature - base, 0)`` in cooling. The coefficient of each hour of the
    day absorbs the schedule of the setpoint and of the occupants. For each base
    temperature of :data:`BASE_GRID_C` the coefficients are the least squares
    solution (one regressor per hour of the day, so it has a closed form, set to
    zero when negative), and the base with the smallest squared error is kept.

    Parameters
    ----------
    load : sequence of float
        Hourly load of the dwelling over a year of 8,760 hours (kBtu).
    temperature : sequence of float
        Hourly outdoor temperature in the same hours (°C).
    use : {"heating", "cooling"}
        End use of the load.

    Returns
    -------
    TemperatureResponse or None
        ``None`` if the load is zero in the whole year.

    Raises
    ------
    ValueError
        If the series are not 8,760 hours or ``use`` is unknown.

    Examples
    --------
    >>> hours = np.arange(8760)
    >>> temperature = 10 + 15 * np.sin(2 * np.pi * hours / 8760)
    >>> load = 2.0 * np.maximum(15.0 - temperature, 0)
    >>> fit = fit_temperature_response(load, temperature, "heating")
    >>> fit.base_c, round(float(fit.coefficients[0]), 3)
    (15.0, 2.0)
    """
    if use not in BASE_GRID_C:
        raise ValueError(f"use must be 'heating' or 'cooling', not {use!r}.")
    load = np.asarray(load, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if load.shape != (HOURS,) or temperature.shape != (HOURS,):
        raise ValueError(f"The series must have {HOURS} hours.")
    if not (load > 0).any():
        return None
    x = load.reshape(-1, HOURS_PER_DAY)
    total_ss = float(((load - load.mean()) ** 2).sum())
    best = None
    for base in BASE_GRID_C[use]:
        d = _degrees(temperature, float(base), use)
        denominator = (d * d).sum(axis=0)
        safe = np.where(denominator > 0, denominator, 1.0)
        beta = np.where(denominator > 0, (d * x).sum(axis=0) / safe, 0.0)
        beta = np.maximum(beta, 0.0)
        sse = float(((x - d * beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(base), beta)
    sse, base, beta = best
    r2 = 1.0 - sse / total_ss if total_ss > 0 else 0.0
    return TemperatureResponse(use, base, beta, r2)


def temperature_response_shape(
    response: TemperatureResponse, temperature: np.ndarray
) -> np.ndarray:
    """Hourly load that a response gives for one or several temperature series.

    Parameters
    ----------
    response : TemperatureResponse
        Output of :func:`fit_temperature_response`.
    temperature : numpy.ndarray, shape (8760,) or (n, 8760)
        Hourly outdoor temperature (°C), one series per row.

    Returns
    -------
    numpy.ndarray
        Load (kBtu per hour) with the shape of ``temperature``.

    Examples
    --------
    >>> temperature_response_shape(response, outdoor_temperature)  # doctest: +SKIP
    """
    temperature = np.asarray(temperature, dtype=float)
    flat = np.atleast_2d(temperature)
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        d = _degrees(row, response.base_c, response.use)
        out[i] = (d * response.coefficients).reshape(-1)
    return out.reshape(temperature.shape)


def reshape_load(
    load: np.ndarray,
    response: TemperatureResponse,
    original_temperature: np.ndarray,
    new_temperature: np.ndarray,
) -> np.ndarray:
    """Hourly load of a dwelling moved to another outdoor temperature.

    The series of the dwelling is kept and the difference between the response
    at the new temperature and at the original one is added to it, with a floor of
    zero. Unlike the response alone, the pattern of occupancy and thermostat of
    the dwelling is not lost. In 315 ResStock dwellings simulated again with the
    temperature of the weather file moved by -4, -2, +2 and +4 degrees Celsius,
    this form had an error 20 % to 45 % lower than leaving the shape unchanged and
    lower than the response alone in all 16 combinations of use and shift (own
    check, not published).

    Parameters
    ----------
    load : numpy.ndarray, shape (8760,)
        Hourly load of the dwelling at ``original_temperature``.
    response : TemperatureResponse
        Output of :func:`fit_temperature_response` for that load.
    original_temperature : numpy.ndarray, shape (8760,)
        Hourly outdoor temperature of the series (°C).
    new_temperature : numpy.ndarray, shape (8760,) or (n, 8760)
        Hourly outdoor temperature to move the load to (°C), one series per row.

    Returns
    -------
    numpy.ndarray
        Load with the shape of ``new_temperature``. The annual sum is not
        preserved; the caller rescales it.

    Examples
    --------
    >>> t = np.tile(np.linspace(-5.0, 5.0, 24), 365)
    >>> load = 2.0 * np.maximum(10.0 - t, 0.0)
    >>> fit = fit_temperature_response(load, t, "heating")
    >>> bool(reshape_load(load, fit, t, t - 2.0).sum() > load.sum())
    True
    """
    original = temperature_response_shape(response, original_temperature)
    new = temperature_response_shape(response, new_temperature)
    return np.clip(np.asarray(load, dtype=float) + new - original, 0.0, None)


def tract_hourly_temperature(
    station_hourly: Sequence[float],
    station_daily: Sequence[float],
    tract_daily: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Hourly outdoor temperature of a tract from the series of a weather station.

    The tract has the hourly series of the station plus, in every hour of each day,
    the difference between the daily mean temperatures of the tract and of the
    station taken from a gridded product of one same source, so that the daily
    cycle is that of the station and the daily mean is that of the tract.

    Parameters
    ----------
    station_hourly : sequence of float
        Hourly temperature of the station over 8,760 hours (°C).
    station_daily : sequence of float
        Daily mean temperature of the station in the gridded product (365 values).
    tract_daily : array-like, shape (365,) or (n, 365)
        Daily mean temperature of one or several tracts in the same product.

    Returns
    -------
    numpy.ndarray, shape (8760,) or (n, 8760)
        Hourly temperature of each tract.

    Raises
    ------
    ValueError
        If the lengths are not 8,760 hours and 365 days.

    Examples
    --------
    >>> t = tract_hourly_temperature(np.zeros(8760), np.zeros(365), np.full(365, -2.0))
    >>> float(t[100])
    -2.0
    """
    station_hourly = np.asarray(station_hourly, dtype=float)
    station_daily = np.asarray(station_daily, dtype=float)
    tract_daily = np.asarray(tract_daily, dtype=float)
    days = HOURS // HOURS_PER_DAY
    if station_hourly.shape != (HOURS,) or station_daily.shape != (days,):
        raise ValueError(f"Expected {HOURS} hours and {days} days.")
    if tract_daily.shape[-1] != days:
        raise ValueError(f"The daily temperatures of the tracts need {days} days.")
    shift = np.repeat(tract_daily - station_daily, HOURS_PER_DAY, axis=-1)
    return station_hourly + shift


def estimate_clock_shift_hours(
    series_temperature: Sequence[float],
    weather_temperature: Sequence[float],
    max_shift: int = MAX_CLOCK_SHIFT_HOURS,
    min_correlation: float = 0.95,
) -> int:
    """Find how many hours the clock of a series runs ahead of the weather file.

    The outdoor temperature of the series of a dwelling comes from the weather file
    of its county, so the two hourly curves are the same up to a rotation of the
    year. The shift is the rotation, between 0 and ``max_shift`` hours, that gives
    the smallest mean absolute difference (the correlation separates neighboring
    hours worse, because the daily cycle is smooth).

    Parameters
    ----------
    series_temperature : sequence of float
        Hourly outdoor temperature (8,760 values) of the series, as read without
        any shift.
    weather_temperature : sequence of float
        Hourly dry-bulb temperature (8,760 values) of the weather file of the
        county, in local standard time.
    max_shift : int
        Largest shift tried, in hours.
    min_correlation : float
        Smallest correlation accepted at the best shift.

    Returns
    -------
    int
        Hours by which the clock of the series runs ahead of the weather file.

    Raises
    ------
    ValueError
        If the curves do not have 8,760 values or the best correlation is below
        ``min_correlation`` (the series and the file do not describe the same
        weather).

    Examples
    --------
    >>> t = np.sin(np.arange(HOURS) * 2 * np.pi / 24)
    >>> estimate_clock_shift_hours(np.roll(t, 3), t)
    3
    """
    series = np.asarray(series_temperature, dtype=float)
    weather = np.asarray(weather_temperature, dtype=float)
    if len(series) != HOURS or len(weather) != HOURS:
        raise ValueError(f"Both curves need {HOURS} hourly values.")
    errors = [
        np.abs(np.roll(series, -shift) - weather).mean()
        for shift in range(max_shift + 1)
    ]
    best = int(np.argmin(errors))
    correlation = np.corrcoef(np.roll(series, -best), weather)[0, 1]
    if correlation < min_correlation:
        raise ValueError(
            f"The correlation between the series and the weather file at the best "
            f"shift is {correlation:.3f}, below {min_correlation}."
        )
    return best


def county_clock_shifts(
    building_county: Mapping[int, str] | pd.Series,
    series_dir: str | os.PathLike,
    weather_dir: str | os.PathLike,
    upgrade: int = 0,
    weather_name: str = "{county}_2018.csv",
    max_shift: int = MAX_CLOCK_SHIFT_HOURS,
) -> dict[str, int]:
    """Estimate the clock shift of the series of every county.

    For each county, the outdoor temperature of the series of one of its dwellings
    is compared with the hourly temperature of the weather file of the county
    (:func:`estimate_clock_shift_hours`). The result does not depend on a table of
    time zones, so it holds in any state and in counties split by a time-zone
    boundary.

    Parameters
    ----------
    building_county : mapping or pandas.Series
        ResStock county code of each ``building_id``.
    series_dir : path-like
        Folder with ``<building id>-<upgrade>.parquet`` series.
    weather_dir : path-like
        Folder with the weather file of each county.
    upgrade : int
        ResStock upgrade of the series files.
    weather_name : str
        Name of the weather file, with ``{county}`` for the county code.
    max_shift : int
        Largest shift tried, in hours.

    Returns
    -------
    dict
        Hours of shift by county code.

    Raises
    ------
    FileNotFoundError
        If a series or a weather file is missing.
    ValueError
        If a weather file has no dry-bulb column or 8,760 rows, or a county's
        curves do not match.

    Examples
    --------
    >>> county_clock_shifts(building_county, series_dir, weather_dir)  # doctest: +SKIP
    """
    series_dir, weather_dir = Path(series_dir), Path(weather_dir)
    county_of = pd.Series(dict(building_county))
    shifts: dict[str, int] = {}
    for county, group in county_of.groupby(county_of):
        building = int(group.index.min())
        path = series_dir / f"{building}-{upgrade}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Time series not found: {path}")
        raw = pd.read_parquet(path, columns=[_OUTDOOR])[_OUTDOOR]
        series = raw.to_numpy(dtype=float).reshape(HOURS, STEPS_PER_HOUR).mean(axis=1)
        weather_path = weather_dir / weather_name.format(county=county)
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")
        weather = pd.read_csv(weather_path, encoding="latin-1")
        columns = [c for c in weather.columns if c.startswith("Dry Bulb Temperature")]
        if not columns or len(weather) != HOURS:
            raise ValueError(f"{weather_path} is not an hourly weather file of a year.")
        shifts[str(county)] = estimate_clock_shift_hours(
            series, weather[columns[0]].to_numpy(dtype=float), max_shift
        )
    return shifts


def donor_coefficients(
    weights: pd.DataFrame,
    donors: pd.DataFrame,
    dwellings: pd.DataFrame,
    tract_degree_days: pd.DataFrame,
    station_degree_days: pd.DataFrame,
    heating: DegreeDaySensitivity,
    cooling: DegreeDaySensitivity,
    extrapolation: str = "linear",
) -> pd.DataFrame:
    """Coefficients that turn the series of the modeled dwellings into a tract profile.

    The coefficient of a modeled dwelling in a tract is the number of dwellings of
    the tract that it stands for (the Census count of each type times its weight
    among the donors of the type, added over the types in which it is a donor)
    divided by the dwellings of the tract. Those of heating and cooling are
    multiplied by the climate factor of the dwelling in the tract.

    Parameters
    ----------
    weights, donors, dwellings, tract_degree_days, station_degree_days, heating, cooling
        As in :func:`utah_ejpi.energy_requirement.donor_level_energy`.
    extrapolation : {"linear", "cap"}
        See :func:`utah_ejpi.energy_requirement.climate_factor`.

    Returns
    -------
    pandas.DataFrame
        One row per tract and dwelling with ``GEOID``, ``building_id``, ``coef``
        (without climate factor; adds up to one in each tract),
        ``coef_heating`` and ``coef_cooling`` (with the factor of that use).

    Examples
    --------
    >>> donor_coefficients(weights, donors, dwellings, tract_dd, station_dd,
    ...     heating, cooling)  # doctest: +SKIP
    """
    needed = [
        "building_id",
        "county",
        "occupants",
        "upgrade",
        "heating_load_kbtu",
        "cooling_load_kbtu",
    ]
    data = donor_level_energy(
        weights,
        donors,
        dwellings[needed],
        tract_degree_days,
        station_degree_days,
        heating,
        cooling,
        extrapolation,
    )
    data["w_heating"] = data["weight"] * data["factor_heating"]
    data["w_cooling"] = data["weight"] * data["factor_cooling"]
    total = data.groupby("GEOID")["weight"].transform("sum")
    sums = (
        data.assign(total=total)
        .groupby(["GEOID", "building_id"], as_index=False)[
            ["weight", "w_heating", "w_cooling", "total"]
        ]
        .agg(
            {"weight": "sum", "w_heating": "sum", "w_cooling": "sum", "total": "first"}
        )
    )
    out = pd.DataFrame(
        {
            "GEOID": sums["GEOID"],
            "building_id": sums["building_id"],
            "coef": sums["weight"] / sums["total"],
            "coef_heating": sums["w_heating"] / sums["total"],
            "coef_cooling": sums["w_cooling"] / sums["total"],
        }
    )
    return out[out["coef"] > 0].reset_index(drop=True)


def _group_of(column: str) -> str:
    """Climate group of a column of the profile: heating, cooling or none."""
    if _HEATING.search(column):
        return "heating"
    if _COOLING.search(column):
        return "cooling"
    return "none"


def build_tract_profiles(
    coefficients: pd.DataFrame,
    series_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    clock_shift_hours: int | Mapping[int, int] | pd.Series,
    upgrade: int = 0,
    comfort_limits: tuple[float, float] = COMFORT_LIMITS_C,
    tract_daily_temperature: pd.DataFrame | None = None,
    station_daily_temperature: pd.DataFrame | None = None,
    dwelling_station: Mapping[int, str] | pd.Series | None = None,
) -> pd.DataFrame:
    """Build and write the hourly profile of every tract.

    The series of the dwellings are read once and added into the profiles of the
    tracts that use them, so that memory is the size of the profiles
    (about 60 columns x tracts x 8,760 hours x 4 bytes). Reading the files of a
    state takes minutes to an hour, so the analysis layer runs it behind a flag.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        Output of :func:`donor_coefficients`.
    series_dir : path-like
        Folder with ``<building id>-<upgrade>.parquet`` for every dwelling used.
    out_dir : path-like
        Folder where the profiles are written, one file per tract.
    clock_shift_hours : int or mapping or pandas.Series
        Hours by which the clock of the series runs ahead of the local standard
        time: one value for all the dwellings, or one per ``building_id`` (a
        county's shift from :func:`county_clock_shifts`, repeated for its
        dwellings).
    upgrade : int
        ResStock upgrade of the files (0 is the baseline).
    comfort_limits : tuple of float
        Limits of the indoor temperature of comfort (°C).
    tract_daily_temperature : pandas.DataFrame, optional
        Daily mean temperature of every tract (rows indexed by ``GEOID``, 365
        columns) from a gridded product. With the next two arguments, the heating
        and cooling columns take the shape of the response of each dwelling to the
        hourly temperature of the tract (see the module description). Without
        them, the shape is that of the weather station of the dwelling.
    station_daily_temperature : pandas.DataFrame, optional
        Daily mean temperature of the weather stations in the same product (rows
        indexed by the station code, 365 columns).
    dwelling_station : mapping or pandas.Series, optional
        Station code of each modeled dwelling, by ``building_id``.

    Returns
    -------
    pandas.DataFrame
        One row per tract with ``GEOID``, ``path``, ``dwellings_used`` (modeled
        dwellings in its profile) and ``coef_sum`` (adds to one).

    Raises
    ------
    FileNotFoundError
        If a series is missing.
    ValueError
        If only some of the three arguments of the shape are given, a tract or
        station lacks daily temperatures, or a dwelling lacks a clock shift.

    Examples
    --------
    >>> build_tract_profiles(coefficients, series_dir, out_dir, clock_shift_hours=2)
    ...     # doctest: +SKIP
    """
    shaping = [
        tract_daily_temperature is not None,
        station_daily_temperature is not None,
        dwelling_station is not None,
    ]
    if any(shaping) and not all(shaping):
        raise ValueError(
            "The shape needs the daily temperatures of the tracts and of the "
            "stations and the station of each dwelling."
        )
    series_dir, out_dir = Path(series_dir), Path(out_dir)
    tracts = np.sort(coefficients["GEOID"].unique())
    position = pd.Series(np.arange(len(tracts)), index=tracts)
    ids = np.sort(coefficients["building_id"].unique())

    def file_of(building_id: int) -> Path:
        return series_dir / f"{int(building_id)}-{upgrade}.parquet"

    if isinstance(clock_shift_hours, (int, np.integer)):
        fixed = int(clock_shift_hours)

        def shift_of(building_id: int) -> int:
            return fixed

    else:
        shifts = dict(clock_shift_hours)
        missing = set(ids) - set(shifts)
        if missing:
            raise ValueError(f"No clock shift for {len(missing)} dwellings.")

        def shift_of(building_id: int) -> int:
            return int(shifts[building_id])

    first = file_of(ids[0])
    if not first.exists():
        raise FileNotFoundError(f"Time series not found: {first}")
    names = pq.read_schema(first).names
    components = hourly_components(names)
    columns = [short_name(c) for c in components] + list(INDICATORS)
    group = np.array([_group_of(c) for c in components] + ["none"] * len(INDICATORS))
    coef_name = {"none": "coef", "heating": "coef_heating", "cooling": "coef_cooling"}
    outdoor = columns.index(short_name(_OUTDOOR))
    load_rows = {
        use: [
            i
            for i, c in enumerate(components)
            if group[i] == use and c.startswith("out.load.")
        ]
        for use in ("heating", "cooling")
    }
    if all(shaping):
        missing = set(tracts) - set(tract_daily_temperature.index)
        if missing:
            raise ValueError(f"No daily temperatures for {len(missing)} tracts.")
        dwelling_station = dict(dwelling_station)
        missing = set(dwelling_station.values()) - set(station_daily_temperature.index)
        if missing:
            raise ValueError(f"No daily temperatures for stations {sorted(missing)}.")
    acc = np.zeros((len(columns), len(tracts), HOURS), dtype=np.float32)
    by_building = {b: g for b, g in coefficients.groupby("building_id")}
    for n, building_id in enumerate(ids, start=1):
        use = by_building[building_id]
        series = read_dwelling_hourly(
            file_of(building_id), components, shift_of(building_id), comfort_limits
        )
        values = series[columns].to_numpy(dtype=np.float32).T
        rows = position.loc[use["GEOID"]].to_numpy()
        shapes = {}
        if all(shaping):
            station = dwelling_station[building_id]
            station_daily = station_daily_temperature.loc[station].to_numpy()
            tract_t = tract_hourly_temperature(
                values[outdoor],
                station_daily,
                tract_daily_temperature.loc[use["GEOID"]].to_numpy(),
            )
            for kind in ("heating", "cooling"):
                kind_load = values[load_rows[kind]].sum(axis=0)
                response = fit_temperature_response(kind_load, values[outdoor], kind)
                if response is not None:
                    shapes[kind] = reshape_load(
                        kind_load, response, values[outdoor], tract_t
                    )
        for kind, name in coef_name.items():
            mask = group == kind
            weight = use[name].to_numpy(dtype=np.float32)
            annual = values[mask].sum(axis=1)
            for k, (row, w) in enumerate(zip(rows, weight, strict=True)):
                shape = shapes.get(kind)
                total = shape[k].sum() if shape is not None else 0.0
                if total > 0:
                    acc[mask, row, :] += w * np.outer(
                        annual, (shape[k] / total).astype(np.float32)
                    )
                else:
                    acc[mask, row, :] += w * values[mask]
        if n % 500 == 0:
            logger.info("dwellings read: %d of %d", n, len(ids))

    hours = pd.date_range("2018-01-01", periods=HOURS, freq="h")
    index = []
    for t, geoid in enumerate(tracts):
        profile = pd.DataFrame(acc[:, t, :].T, columns=columns)
        profile.insert(0, "timestamp", hours)
        folder = out_dir / f"GEOID={geoid}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "profile.parquet"
        profile.to_parquet(path, index=False)
        sub = coefficients[coefficients["GEOID"] == geoid]
        index.append(
            {
                "GEOID": geoid,
                "path": str(path),
                "dwellings_used": len(sub),
                "coef_sum": float(sub["coef"].sum()),
            }
        )
    return pd.DataFrame(index)


def read_tract_profile(profile_dir: str | os.PathLike, geoid: str) -> pd.DataFrame:
    """Read the hourly profile of one tract.

    Parameters
    ----------
    profile_dir : path-like
        Folder written by :func:`build_tract_profiles`.
    geoid : str
        Tract code (11 characters).

    Returns
    -------
    pandas.DataFrame
        8,760 rows with ``timestamp`` and the columns of the profile, per dwelling
        of the tract (energy in kWh or kBtu per hour, temperatures in °C, shares
        between 0 and 1).

    Raises
    ------
    FileNotFoundError
        If the tract has no profile.

    Examples
    --------
    >>> read_tract_profile("profiles", "49035100100")  # doctest: +SKIP
    """
    path = Path(profile_dir) / f"GEOID={geoid}" / "profile.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No hourly profile for tract {geoid}: {path}")
    return pd.read_parquet(path)


#: Groups of annual energy summed by :func:`state_annual_means`: name -> test on
#: the short name of a profile column in kWh.
_MEAN_GROUPS = {
    "electricity": lambda c: c.startswith("electricity_") and c != "electricity_pv_kwh",
    "electricity_pv": lambda c: c == "electricity_pv_kwh",
    "electricity_heating": lambda c: c.startswith("electricity_") and "_heating" in c,
    "electricity_cooling": lambda c: c.startswith("electricity_") and "_cooling" in c,
    "natural_gas": lambda c: c.startswith("natural_gas_"),
    "natural_gas_heating": lambda c: c.startswith("natural_gas_") and "_heating" in c,
    "propane": lambda c: c.startswith("propane_"),
    "fuel_oil": lambda c: c.startswith("fuel_oil_"),
}


def state_annual_means(
    profile_dir: str | os.PathLike, tract_dwellings: pd.Series
) -> dict[str, float]:
    """Mean annual energy per dwelling of all the tracts, by fuel and use.

    Sums the 8,760 hours of the profile of every tract and averages the tracts
    weighted by their dwellings. It is the modeled consumption that is compared with
    the totals of the state sold by the utilities.

    Parameters
    ----------
    profile_dir : path-like
        Folder written by :func:`build_tract_profiles`.
    tract_dwellings : pandas.Series
        Occupied dwellings of every tract, indexed by ``GEOID``; the tracts are the
        ones averaged.

    Returns
    -------
    dict
        kWh per dwelling and year of ``electricity`` (without the generation of
        rooftop solar panels already installed), ``electricity_pv`` (that
        generation, negative), ``electricity_heating``, ``electricity_cooling``,
        ``natural_gas``, ``natural_gas_heating``, ``propane`` and ``fuel_oil``.
        Heating and cooling include fans and pumps, as in the energy of the tracts.

    Raises
    ------
    FileNotFoundError
        If a tract has no profile.

    Examples
    --------
    >>> state_annual_means(profile_dir, tract_dwellings)  # doctest: +SKIP
    """
    totals = dict.fromkeys(_MEAN_GROUPS, 0.0)
    for geoid, n in tract_dwellings.items():
        profile = read_tract_profile(profile_dir, geoid)
        sums = profile.drop(columns="timestamp").sum()
        kwh = [c for c in sums.index if c.endswith("_kwh")]
        for name, test in _MEAN_GROUPS.items():
            totals[name] += float(n) * float(sums[[c for c in kwh if test(c)]].sum())
    n_total = float(tract_dwellings.sum())
    return {name: value / n_total for name, value in totals.items()}


def monthly_sums(profile: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add up the hours of each calendar month of a profile.

    Parameters
    ----------
    profile : pandas.DataFrame
        A profile with a ``timestamp`` column in local standard time (see
        :func:`read_tract_profile`), one row per hour of one year.
    columns : sequence of str
        Columns to add up.

    Returns
    -------
    pandas.DataFrame
        Twelve rows indexed by month (1 to 12) with the sum of each column.

    Examples
    --------
    >>> hours = pd.date_range("2018-01-01", periods=8760, freq="h")
    >>> frame = pd.DataFrame({"timestamp": hours, "x": 1.0})
    >>> monthly_sums(frame, ["x"]).loc[[1, 2], "x"].tolist()
    [744.0, 672.0]
    """
    month = pd.DatetimeIndex(profile["timestamp"]).month
    sums = profile[list(columns)].groupby(month).sum()
    sums.index.name = "month"
    return sums


def state_monthly_means(
    profile_dir: str | os.PathLike, tract_dwellings: pd.Series
) -> pd.DataFrame:
    """Mean monthly energy per dwelling of all the tracts, by fuel and use.

    Like :func:`state_annual_means` but by calendar month: the hours of each month
    of every tract are added up and the tracts are averaged weighted by their
    dwellings. It is the modeled consumption that is compared with the monthly
    totals of the state sold by the utilities.

    Parameters
    ----------
    profile_dir : path-like
        Folder written by :func:`build_tract_profiles`.
    tract_dwellings : pandas.Series
        Occupied dwellings of every tract, indexed by ``GEOID``; the tracts are the
        ones averaged.

    Returns
    -------
    pandas.DataFrame
        Twelve rows indexed by month; kWh per dwelling and month of the same groups
        as :func:`state_annual_means`, with its column names.

    Raises
    ------
    FileNotFoundError
        If a tract has no profile.

    Examples
    --------
    >>> state_monthly_means(profile_dir, tract_dwellings)  # doctest: +SKIP
    """
    totals = pd.DataFrame(
        0.0, index=pd.RangeIndex(1, 13, name="month"), columns=list(_MEAN_GROUPS)
    )
    for geoid, n in tract_dwellings.items():
        profile = read_tract_profile(profile_dir, geoid)
        kwh = [c for c in profile.columns if c.endswith("_kwh")]
        monthly = monthly_sums(profile, kwh)
        for name, test in _MEAN_GROUPS.items():
            members = [c for c in kwh if test(c)]
            totals[name] += float(n) * monthly[members].sum(axis=1)
    return totals / float(tract_dwellings.sum())


def state_hourly_means(
    profile_dir: str | os.PathLike, tract_dwellings: pd.Series
) -> pd.DataFrame:
    """Mean hourly energy per dwelling of all the tracts, column by column.

    Like :func:`state_annual_means` but keeps the 8,760 hours and every energy
    column in kWh of the profiles (fuel and use), so the state series can be grouped
    by use, by month or shifted in time.

    Parameters
    ----------
    profile_dir : path-like
        Folder written by :func:`build_tract_profiles`.
    tract_dwellings : pandas.Series
        Occupied dwellings of every tract, indexed by ``GEOID``; the tracts are the
        ones averaged.

    Returns
    -------
    pandas.DataFrame
        8,760 rows indexed by the hour (local standard time), one column per
        profile column that ends in ``_kwh``, in kWh per dwelling and hour.

    Raises
    ------
    FileNotFoundError
        If a tract has no profile.

    Examples
    --------
    >>> state_hourly_means(profile_dir, tract_dwellings)  # doctest: +SKIP
    """
    total, hours, names = None, None, None
    for geoid, n in tract_dwellings.items():
        profile = read_tract_profile(profile_dir, geoid)
        names = [c for c in profile.columns if c.endswith("_kwh")]
        values = profile[names].to_numpy(dtype="float64") * float(n)
        total = values if total is None else total + values
        hours = profile["timestamp"]
    if total is None:
        raise ValueError("tract_dwellings is empty.")
    out = pd.DataFrame(total / float(tract_dwellings.sum()), columns=names)
    out.index = pd.DatetimeIndex(hours, name="hour")
    return out


def read_profiles(
    profile_dir: str | os.PathLike,
    columns: Sequence[str],
    geoids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read some columns of the profiles of many tracts as one table.

    Parameters
    ----------
    profile_dir : path-like
        Folder written by :func:`build_tract_profiles`.
    columns : sequence of str
        Columns to read besides ``timestamp``.
    geoids : sequence of str, optional
        Tracts to read; all by default.

    Returns
    -------
    pandas.DataFrame
        Long table with ``GEOID``, ``timestamp`` and the columns.

    Examples
    --------
    >>> read_profiles("profiles", ["heating_kbtu"], geoids=["49035100100"])
    ...     # doctest: +SKIP
    """
    folders = sorted(Path(profile_dir).glob("GEOID=*"))
    wanted = None if geoids is None else set(geoids)
    parts = []
    for folder in folders:
        geoid = folder.name.removeprefix("GEOID=")
        if wanted is not None and geoid not in wanted:
            continue
        part = pd.read_parquet(
            folder / "profile.parquet", columns=["timestamp", *columns]
        )
        parts.append(part.assign(GEOID=geoid))
    if not parts:
        return pd.DataFrame(columns=["GEOID", "timestamp", *columns])
    return pd.concat(parts, ignore_index=True)[["GEOID", "timestamp", *columns]]


def fuel_use_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    """Add to a profile the consumption by use and fuel and the totals by fuel.

    Parameters
    ----------
    profile : pandas.DataFrame
        Output of :func:`read_tract_profile`.

    Returns
    -------
    pandas.DataFrame
        The profile with the columns ``heating_kwh_electricity``,
        ``heating_kwh_natural_gas``, ``heating_kwh_propane``,
        ``heating_kwh_fuel_oil`` and ``cooling_kwh_electricity`` (the same sums as
        the annual columns of :data:`utah_ejpi.resstock.FUEL_COLUMNS`; the series
        of ResStock have no column for the propane and fuel oil heat pump backup,
        which the annual results report as zero in Utah),
        ``heating_kwh`` and ``cooling_kwh``, and ``electricity_total_kwh``,
        ``natural_gas_total_kwh``, ``propane_total_kwh``, ``fuel_oil_total_kwh``
        (sum of the end uses of each fuel, without the photovoltaic generation)
        and ``electricity_net_kwh`` (total plus the generation of the dwellings
        that already have photovoltaic panels, which ResStock reports as negative).

    Examples
    --------
    >>> fuel_use_profiles(profile)  # doctest: +SKIP
    """
    out = profile.copy()
    for name, sources in FUEL_COLUMNS.items():
        present = [short_name(c) for c in sources if short_name(c) in profile.columns]
        out[name] = profile[present].sum(axis=1)
    for use in ("heating", "cooling"):
        out[f"{use}_kwh"] = out[[c for c in FUEL_COLUMNS if c.startswith(use)]].sum(
            axis=1
        )
    for fuel in _FUELS:
        parts = [
            c
            for c in profile.columns
            if c.startswith(f"{fuel}_")
            and c.endswith("_kwh")
            and "_total_" not in c
            and not c.startswith(f"{fuel}_pv")
        ]
        out[f"{fuel}_total_kwh"] = profile[parts].sum(axis=1)
    pv = "electricity_pv_kwh"
    out["electricity_net_kwh"] = out["electricity_total_kwh"] + (
        profile[pv] if pv in profile.columns else 0.0
    )
    return out
