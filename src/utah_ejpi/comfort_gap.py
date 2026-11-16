"""Normative setpoints, activity windows and the comfort gap of a dwelling.

The energy that a dwelling needs to be inside the comfort band (at least
``floor_c`` degrees Celsius of heating and at most ``ceiling_c`` of cooling)
while someone is at home and awake is not an output of ResStock: ResStock
simulates every dwelling with its own thermostat. The comfort gap is obtained
by simulating the same dwelling a second time with a normative thermostat and
subtracting the useful energy of the first simulation. This module builds the
normative thermostat and reads the two simulations; the simulations
themselves are run by the analysis layer (see ``utah_ejpi.resimulation``).

Four decisions shape the normative thermostat. First, the household is active
from the time the first person gets up until the last person goes to bed; the
times come from the American Time Use Survey diaries, by number of persons and
by weekday or weekend, and the people of a household are assumed to get up and
go to bed independently of each other, which makes the window longer than the
real one when the members keep similar hours. Second, the band is required
only when someone is at home, according to the occupancy schedule of the
dwelling. Third, the thermostat is also normative during the hours before each
block of required hours (pre-conditioning) so that the dwelling is already
inside the band when the person arrives or gets up; the equipment only works
when the dwelling is outside the band. Fourth, the schedules of EnergyPlus run
on the clock (with daylight saving time) while its outputs are written in
standard time; masks are therefore converted before they are compared with
simulated temperatures.

All time series are on a regular grid of ``step_minutes`` minutes that covers
one non-leap year. The weekday of the first step is an argument; the
defaults are those of 2018 (a Monday) at 15 minutes.

References
----------
Bureau of Labor Statistics, American Time Use Survey microdata
(``bls_atus_microdata``); National Renewable Energy Laboratory (2025),
ResStock 2025 release 1 (``resstock_2025_release1_readme``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "accumulate_tract_profiles",
    "DEFAULT_CEILING_C",
    "DEFAULT_FLOOR_C",
    "SLEEP_ACTIVITY_CODE",
    "daylight_saving_mask",
    "dwelling_gap",
    "household_window_table",
    "normative_setpoints",
    "out_of_range",
    "required_masks",
    "sleep_times",
    "to_standard_time",
]

#: Lower limit of the comfort band, in degrees Celsius (convention of the study).
DEFAULT_FLOOR_C = 18.0
#: Upper limit of the comfort band, in degrees Celsius (convention of the study).
DEFAULT_CEILING_C = 26.0
#: Code of the activity "sleeping" in the lexicon of the time use survey.
#: NO VERIFICADO against the lexicon document; checked only through the
#: plausibility of the resulting times (median wake time 07:00).
SLEEP_ACTIVITY_CODE = 10101

_DAY_MINUTES = 1440


def _minutes(clock: pd.Series) -> pd.Series:
    """Minutes since midnight of a column of ``HH:MM:SS`` strings."""
    hours = clock.str.slice(0, 2).astype(int)
    minutes = clock.str.slice(3, 5).astype(int)
    return hours * 60 + minutes


def sleep_times(
    activities: pd.DataFrame,
    sleep_code: int = SLEEP_ACTIVITY_CODE,
    merge_gap_minutes: int = 60,
    min_night_minutes: int = 120,
    evening_start_minute: int = 17 * 60,
) -> pd.DataFrame:
    """Wake-up and bed times of every respondent from the activity diary.

    The diary of the time use survey runs from 4:00 of the diary day to 4:00 of
    the next day, so the sleep that ends the previous night is at the start of
    the diary and the sleep of the coming night is at the end. Episodes of
    sleep separated by less than ``merge_gap_minutes`` are merged. The wake-up
    time is the end of the first block when the respondent is asleep at the
    start of the diary, and otherwise the end of the first block of at least
    three hours that starts and ends before 14:00. The bed time is the start
    of the longest block that starts after ``evening_start_minute`` and lasts
    between ``min_night_minutes`` and 16 hours.

    Parameters
    ----------
    activities : pandas.DataFrame
        Activity file of the survey with ``TUCASEID``, ``TUSTARTTIM`` and
        ``TUSTOPTIME`` (``HH:MM:SS`` strings) and ``TRCODEP`` (activity code).
    sleep_code : int
        Activity code of sleeping.
    merge_gap_minutes : int
        Largest pause between two episodes that are treated as one sleep.
    min_night_minutes : int
        Shortest block that counts as the night sleep when looking for the bed
        time.
    evening_start_minute : int
        Earliest start, in minutes since midnight, of the night sleep.

    Returns
    -------
    pandas.DataFrame
        One row per respondent with sleep records: ``TUCASEID``, ``wake_min`` and
        ``bed_min`` (minutes since midnight; missing when not identified).

    Raises
    ------
    ValueError
        If a required column is missing.

    Examples
    --------
    >>> acts = pd.DataFrame({
    ...     "TUCASEID": [1, 1],
    ...     "TUSTARTTIM": ["04:00:00", "22:30:00"],
    ...     "TUSTOPTIME": ["07:00:00", "04:00:00"],
    ...     "TRCODEP": [10101, 10101],
    ... })
    >>> sleep_times(acts).iloc[0].to_dict()
    {'TUCASEID': 1, 'wake_min': 420, 'bed_min': 1350}
    """
    needed = {"TUCASEID", "TUSTARTTIM", "TUSTOPTIME", "TRCODEP"}
    if not needed <= set(activities.columns):
        raise ValueError(f"activities needs the columns {sorted(needed)}.")
    sleep = activities.loc[activities["TRCODEP"] == sleep_code].copy()
    start = _minutes(sleep["TUSTARTTIM"])
    stop = _minutes(sleep["TUSTOPTIME"])
    stop = stop.where(stop > start, stop + _DAY_MINUTES)
    late = start < 240
    sleep["start"] = start.where(~late, start + _DAY_MINUTES)
    sleep["stop"] = stop.where(~late, stop + _DAY_MINUTES)
    sleep = sleep.sort_values(["TUCASEID", "start"])
    rows = []
    for case, block in sleep.groupby("TUCASEID", sort=False):
        merged: list[tuple[int, int]] = []
        for s, e in zip(
            block["start"].to_numpy(), block["stop"].to_numpy(), strict=True
        ):
            if merged and s - merged[-1][1] < merge_gap_minutes:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((int(s), int(e)))
        wake = np.nan
        first = merged[0]
        if first[0] <= 240 + 30:
            wake = first[1] % _DAY_MINUTES
        else:
            for s, e in merged:
                limit = 14 * 60
                starts_early = s % _DAY_MINUTES < limit
                ends_early = e % _DAY_MINUTES < limit
                if starts_early and ends_early and e - s >= 180:
                    wake = e % _DAY_MINUTES
                    break
        night = [
            (s, e)
            for s, e in merged
            if s >= evening_start_minute and min_night_minutes <= e - s < 16 * 60
        ]
        bed = np.nan
        if night:
            s, _ = max(night, key=lambda x: x[1] - x[0])
            bed = s % _DAY_MINUTES
        rows.append((case, wake, bed))
    return pd.DataFrame(rows, columns=["TUCASEID", "wake_min", "bed_min"])


def household_window_table(
    respondents: pd.DataFrame,
    max_persons: int = 4,
    n_draws: int = 200_000,
    seed: int = 2026,
    min_age: int = 15,
    weekend_days: tuple[int, ...] = (1, 7),
    step_minutes: int = 15,
) -> pd.DataFrame:
    """Median wake-up and bed times of a household, by persons and day type.

    For a household of ``n`` persons the first wake-up time is the earliest of
    ``n`` independent draws from the weighted distribution of the respondents of
    the same day type, and the last bed time is the latest of ``n`` draws. The
    median of each over many draws is the window of the household. The
    independence of the members is an assumption that overstates the window when
    the members keep similar hours; the window of one person is the lower bound.

    Parameters
    ----------
    respondents : pandas.DataFrame
        One row per respondent with ``wake_min``, ``bed_min``, ``TUFNWGTP``
        (final weight), ``TUDIARYDAY`` (1 is Sunday and 7 Saturday) and
        ``PRTAGE`` (age).
    max_persons : int
        Households of ``max_persons`` or more persons share the last row.
    n_draws : int
        Number of random households per row of the table.
    seed : int
        Seed of the random generator.
    min_age : int
        Youngest respondent that is used.
    weekend_days : tuple of int
        Codes of ``TUDIARYDAY`` that are weekend days.
    step_minutes : int
        The medians are rounded to this many minutes.

    Returns
    -------
    pandas.DataFrame
        One row per day type (``"weekday"``, ``"weekend"``) and number of persons
        with ``wake_min``, ``bed_min`` (medians, minutes since midnight; a bed
        time after midnight is written as the minutes of the same evening and is
        capped at 1440) and the quartiles ``wake_p25``, ``wake_p75``,
        ``bed_p25`` and ``bed_p75``.

    Raises
    ------
    ValueError
        If a required column is missing or a day type has no respondent.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> resp = pd.DataFrame({
    ...     "wake_min": rng.choice([360, 420, 480], 300),
    ...     "bed_min": rng.choice([1320, 1380], 300),
    ...     "TUFNWGTP": 1.0, "TUDIARYDAY": rng.integers(1, 8, 300), "PRTAGE": 40,
    ... })
    >>> household_window_table(resp, max_persons=2, n_draws=1000).shape
    (4, 8)
    """
    needed = {"wake_min", "bed_min", "TUFNWGTP", "TUDIARYDAY", "PRTAGE"}
    if not needed <= set(respondents.columns):
        raise ValueError(f"respondents needs the columns {sorted(needed)}.")
    data = respondents[
        (respondents["PRTAGE"] >= min_age)
        & respondents["wake_min"].notna()
        & respondents["bed_min"].notna()
    ].copy()
    # A bed time after midnight belongs to the evening of the same day.
    data["bed"] = np.where(
        data["bed_min"] < 720, data["bed_min"] + _DAY_MINUTES, data["bed_min"]
    )
    data["weekend"] = data["TUDIARYDAY"].isin(weekend_days)
    rng = np.random.default_rng(seed)

    def rounded(value: float) -> int:
        return int(round(value / step_minutes) * step_minutes)

    rows = []
    for weekend in (False, True):
        group = data[data["weekend"] == weekend]
        if group.empty:
            raise ValueError("A day type has no respondent.")
        weights = group["TUFNWGTP"].to_numpy(float)
        prob = weights / weights.sum()
        wake = group["wake_min"].to_numpy(float)
        bed = group["bed"].to_numpy(float)
        for persons in range(1, max_persons + 1):
            first = np.min(
                np.stack([rng.choice(wake, n_draws, p=prob) for _ in range(persons)]),
                axis=0,
            )
            last = np.max(
                np.stack([rng.choice(bed, n_draws, p=prob) for _ in range(persons)]),
                axis=0,
            )
            rows.append(
                {
                    "day_type": "weekend" if weekend else "weekday",
                    "persons": persons,
                    "wake_min": rounded(np.median(first)),
                    "bed_min": min(rounded(np.median(last)), _DAY_MINUTES),
                    "wake_p25": rounded(np.percentile(first, 25)),
                    "wake_p75": rounded(np.percentile(first, 75)),
                    "bed_p25": rounded(np.percentile(last, 25)),
                    "bed_p75": rounded(np.percentile(last, 75)),
                }
            )
    return pd.DataFrame(rows)


def daylight_saving_mask(
    year: int = 2018,
    observes: bool = True,
    step_minutes: int = 15,
) -> np.ndarray:
    """Flag the steps of the year in which the clock is ahead of standard time.

    The rule is that of the United States since 2007: from 02:00 of the second
    Sunday of March to 02:00 (daylight time) of the first Sunday of November.

    Parameters
    ----------
    year : int
        Calendar year of the simulation.
    observes : bool
        ``False`` for the places that keep standard time all year (Arizona
        outside the Navajo Nation, Hawaii).
    step_minutes : int
        Length of a step.

    Returns
    -------
    numpy.ndarray
        Boolean array with one value per step of the year, indexed in standard
        time.

    Notes
    -----
    Valid for the United States; other countries need their own rule.

    Examples
    --------
    >>> m = daylight_saving_mask(2018, step_minutes=60)
    >>> int(m.sum())
    5711
    """
    index = pd.date_range(
        f"{year}-01-01",
        f"{year + 1}-01-01",
        freq=f"{step_minutes}min",
        inclusive="left",
    )
    if not observes:
        return np.zeros(len(index), dtype=bool)
    march = pd.date_range(f"{year}-03-01", f"{year}-03-31", freq="D")
    november = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    start = march[march.dayofweek == 6][1] + pd.Timedelta(hours=2)
    end = november[november.dayofweek == 6][0] + pd.Timedelta(hours=1)
    return np.asarray((index >= start) & (index < end))


def required_masks(
    presence: np.ndarray,
    persons: int,
    windows: pd.DataFrame,
    lead_hours: float = 3.0,
    step_minutes: int = 15,
    first_weekday: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the steps in which the comfort band is required and those before them.

    The band is required when someone is at home (``presence`` above zero) and
    the household is active (between its wake-up and bed times of that day
    type). The pre-conditioning steps are the ``lead_hours`` before each start of
    a block of required steps that are not themselves required. Both masks are on
    the clock of the schedules of the dwelling, not in standard time.

    Parameters
    ----------
    presence : numpy.ndarray
        Occupancy schedule of the dwelling, one value per step.
    persons : int
        Persons of the household; values above the largest in ``windows`` use
        the largest.
    windows : pandas.DataFrame
        Output of :func:`household_window_table`.
    lead_hours : float
        Hours of pre-conditioning.
    step_minutes : int
        Length of a step.
    first_weekday : int
        Weekday of the first step, with Monday equal to 0 (2018 starts on a
        Monday).

    Returns
    -------
    tuple of numpy.ndarray
        ``required`` and ``preconditioning`` (boolean, one value per step).

    Raises
    ------
    ValueError
        If ``windows`` has no row for the number of persons.

    Examples
    --------
    >>> w = pd.DataFrame({"day_type": ["weekday", "weekend"], "persons": [1, 1],
    ...                   "wake_min": [420, 480], "bed_min": [1320, 1320]})
    >>> req, pre = required_masks(np.ones(96 * 7), 1, w, step_minutes=15)
    >>> int(req[:96].sum()), int(pre[:96].sum())
    (60, 12)
    """
    presence = np.asarray(presence, float)
    n = len(presence)
    per_day = _DAY_MINUTES // step_minutes
    persons = int(min(persons, windows["persons"].max()))
    minute = (np.arange(n) % per_day) * step_minutes
    weekend = ((np.arange(n) // per_day + first_weekday) % 7) >= 5
    wake = np.empty(n)
    bed = np.empty(n)
    for is_weekend in (False, True):
        row = windows[
            (windows["day_type"] == ("weekend" if is_weekend else "weekday"))
            & (windows["persons"] == persons)
        ]
        if row.empty:
            raise ValueError(f"windows has no row for {persons} persons.")
        wake[weekend == is_weekend] = float(row["wake_min"].iloc[0])
        bed[weekend == is_weekend] = float(row["bed_min"].iloc[0])
    required = (presence > 0) & (minute >= wake) & (minute < bed)
    starts = required & ~np.roll(required, 1)
    steps = int(round(lead_hours * 60 / step_minutes))
    pre = np.zeros(n, dtype=bool)
    for lag in range(1, steps + 1):
        pre |= np.roll(starts, -lag)
    return required, pre & ~required


def normative_setpoints(
    own_heating: np.ndarray,
    own_cooling: np.ndarray,
    required: np.ndarray,
    preconditioning: np.ndarray,
    floor_c: float = DEFAULT_FLOOR_C,
    ceiling_c: float = DEFAULT_CEILING_C,
    no_cooling: bool = False,
    off_c: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Heating and cooling setpoints of the normative simulation.

    While the band is required or being prepared, heating is at least
    ``floor_c`` and cooling at most ``ceiling_c`` (a dwelling whose own setpoint
    is already stricter keeps it). Outside those steps the own setpoints are
    kept. A dwelling without cooling equipment has no own cooling setpoint, so
    cooling is off (``off_c``) except in the required and preparation steps.

    Parameters
    ----------
    own_heating, own_cooling : numpy.ndarray
        Own setpoints of the dwelling, in degrees Celsius, one value per step.
    required, preconditioning : numpy.ndarray
        Output of :func:`required_masks`.
    floor_c, ceiling_c : float
        Limits of the comfort band.
    no_cooling : bool
        ``True`` for a dwelling without cooling equipment.
    off_c : float
        Cooling setpoint that keeps the cooling off.

    Returns
    -------
    tuple of numpy.ndarray
        Heating and cooling setpoints. The cooling setpoint is never below the
        heating one.

    Examples
    --------
    >>> h, c = normative_setpoints(np.array([15.0, 15.0]), np.array([30.0, 30.0]),
    ...                            np.array([True, False]), np.array([False, False]))
    >>> h.tolist(), c.tolist()
    ([18.0, 15.0], [26.0, 30.0])
    """
    active = np.asarray(required) | np.asarray(preconditioning)
    heating = np.where(active, np.maximum(own_heating, floor_c), own_heating)
    if no_cooling:
        cooling = np.where(active, ceiling_c, off_c)
    else:
        cooling = np.where(active, np.minimum(own_cooling, ceiling_c), own_cooling)
    cooling = np.maximum(cooling, heating)
    return heating.astype(float), cooling.astype(float)


def to_standard_time(
    mask: np.ndarray, daylight: np.ndarray, step_minutes: int = 15
) -> np.ndarray:
    """Convert a mask on the clock of the schedules to standard time.

    During daylight saving time the clock is one hour ahead of standard time, so
    the value that the schedule has at clock step ``i + k`` applies at standard
    step ``i``, where ``k`` is the number of steps in one hour.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean or numeric array on the clock, one value per step.
    daylight : numpy.ndarray
        Output of :func:`daylight_saving_mask`, indexed in standard time.
    step_minutes : int
        Length of a step.

    Returns
    -------
    numpy.ndarray
        The mask in standard time.

    Raises
    ------
    ValueError
        If the two arrays differ in length.

    Examples
    --------
    >>> m = np.arange(8) % 4 == 0
    >>> to_standard_time(m, np.ones(8, dtype=bool), step_minutes=15).tolist()
    [True, False, False, False, True, False, False, False]
    """
    mask = np.asarray(mask)
    daylight = np.asarray(daylight, bool)
    if len(mask) != len(daylight):
        raise ValueError("mask and daylight must have the same length.")
    shifted = np.roll(mask, -(60 // step_minutes))
    return np.where(daylight, shifted, mask)


def out_of_range(
    temperature: np.ndarray,
    required: np.ndarray,
    low_c: float = DEFAULT_FLOOR_C,
    high_c: float = DEFAULT_CEILING_C,
    tolerance_c: float = 0.2,
    step_minutes: int = 15,
) -> dict[str, float]:
    """Hours and degree-hours outside the band in the required steps.

    Parameters
    ----------
    temperature : numpy.ndarray
        Indoor temperature, one value per step, in degrees Celsius.
    required : numpy.ndarray
        Steps in which the band is required, in the same time as the
        temperature (standard time).
    low_c, high_c : float
        Limits of the band.
    tolerance_c : float
        Margin that is not counted: a step is outside the band when the
        temperature is more than ``tolerance_c`` below ``low_c`` or above
        ``high_c``. It absorbs the convergence tolerance of the simulation.
    step_minutes : int
        Length of a step.

    Returns
    -------
    dict
        ``required_hours``, ``hours_low``, ``hours_high`` and the degree-hours
        ``degree_hours_low`` and ``degree_hours_high`` (measured from the limit
        of the band, without the tolerance subtracted).

    Examples
    --------
    >>> out_of_range(np.array([17.0, 20.0, 27.0, 26.1]), np.ones(4, dtype=bool),
    ...              step_minutes=60)["hours_high"]
    1.0
    """
    t = np.asarray(temperature, float)
    req = np.asarray(required, bool)
    hours = step_minutes / 60.0
    below = req & (t < low_c - tolerance_c)
    above = req & (t > high_c + tolerance_c)
    return {
        "required_hours": float(req.sum() * hours),
        "hours_low": float(below.sum() * hours),
        "hours_high": float(above.sum() * hours),
        "degree_hours_low": float(np.clip(low_c - t, 0, None)[req].sum() * hours),
        "degree_hours_high": float(np.clip(t - high_c, 0, None)[req].sum() * hours),
    }


def dwelling_gap(
    heating_base: np.ndarray,
    cooling_base: np.ndarray,
    heating_normative: np.ndarray,
    cooling_normative: np.ndarray,
    cooling_unserved_base: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute the useful energy that a dwelling lacks to stay in the band.

    The gap is the useful energy of the normative simulation minus that of the
    baseline simulation, by use. ResStock models the part of the dwelling that a
    partial cooling system (a room air conditioner) does not serve with an ideal
    system of its own whose energy does not enter the delivered load that
    ResStock publishes; ``cooling_unserved_base`` is removed from the baseline so
    that the baseline equals the delivered load and that part belongs to the
    gap.

    Parameters
    ----------
    heating_base, cooling_base : numpy.ndarray
        Useful energy of the baseline simulation (any consistent energy unit).
    heating_normative, cooling_normative : numpy.ndarray
        Useful energy of the normative simulation, same unit and length.
    cooling_unserved_base : numpy.ndarray, optional
        Energy of the ideal system of ResStock in the baseline simulation.

    Returns
    -------
    pandas.DataFrame
        One row per step with ``heating_current``, ``cooling_current``,
        ``heating_normative``, ``cooling_normative``, ``heating_gap`` and
        ``cooling_gap``. The gap can be negative in a step (a dwelling that cools
        below the ceiling) and is not clipped.

    Raises
    ------
    ValueError
        If the arrays differ in length.

    Examples
    --------
    >>> g = dwelling_gap(np.zeros(2), np.array([1.0, 2.0]), np.ones(2),
    ...                  np.array([2.0, 2.0]))
    >>> g["cooling_gap"].tolist(), g["heating_gap"].tolist()
    ([1.0, 0.0], [1.0, 1.0])
    """
    arrays = [heating_base, cooling_base, heating_normative, cooling_normative]
    if cooling_unserved_base is not None:
        arrays.append(cooling_unserved_base)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("All arrays must have the same length.")
    unserved = (
        np.zeros(len(cooling_base))
        if cooling_unserved_base is None
        else np.asarray(cooling_unserved_base, float)
    )
    out = pd.DataFrame(
        {
            "heating_current": np.asarray(heating_base, float),
            "cooling_current": np.asarray(cooling_base, float) - unserved,
            "heating_normative": np.asarray(heating_normative, float),
            "cooling_normative": np.asarray(cooling_normative, float),
        }
    )
    out["heating_gap"] = out["heating_normative"] - out["heating_current"]
    out["cooling_gap"] = out["cooling_normative"] - out["cooling_current"]
    return out


def accumulate_tract_profiles(
    coefficients: pd.DataFrame,
    read_series: Callable[[int], pd.DataFrame],
    columns: Sequence[str],
    coefficient_of: Mapping[str, str],
    hours: int = 8760,
    tract_series: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the hourly series of the donor dwellings into the profile of every tract.

    Each dwelling is read once and added, multiplied by its coefficient, into
    the profile of every tract that uses it, so the memory is that of the
    profiles (columns x tracts x hours x 4 bytes). The coefficient of a dwelling
    in a tract is the share of the dwellings of the tract that it stands for
    (see :func:`utah_ejpi.hourly.donor_coefficients`), with the climate factor
    of the use when the series is corrected to the climate of the tract. The sum
    of the hours of a profile is then the annual mean per dwelling of the tract
    that the simulations of :func:`utah_ejpi.gap_uncertainty.
    simulate_tract_quantities` give for the same coefficient.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        One row per tract and dwelling with ``GEOID``, ``building_id`` and the
        columns of coefficients named in ``coefficient_of``.
    read_series : callable
        Function that returns, for a ``building_id``, a table of ``hours`` rows
        with at least ``columns``, in the same time for every dwelling.
    columns : sequence of str
        Series to accumulate.
    coefficient_of : mapping of str to str
        Column of ``coefficients`` that weights each series (for example
        ``"coef"``, ``"coef_heating"`` or ``"coef_cooling"``).
    hours : int
        Rows of every series.
    tract_series : callable, optional
        Function of a ``building_id`` and the array of tracts that use it that
        returns an array of shape ``(len(columns), n_tracts, hours)``: the series
        of the dwelling as it applies to each of those tracts (for example with
        the hourly shape of the climate of the tract). It replaces
        ``read_series``, which can then be ``None``.

    Returns
    -------
    tuple of numpy.ndarray
        The sorted tract codes and an array of shape
        ``(len(columns), n_tracts, hours)`` with the profiles.

    Raises
    ------
    ValueError
        If a series has not ``hours`` rows or a column has no coefficient.

    Examples
    --------
    >>> coef = pd.DataFrame({"GEOID": ["a", "b"], "building_id": [1, 1],
    ...                      "coef": [0.5, 1.0]})
    >>> series = lambda b: pd.DataFrame({"x": [1.0, 2.0]})
    >>> tracts, acc = accumulate_tract_profiles(coef, series, ["x"], {"x": "coef"}, 2)
    >>> acc[0].tolist()
    [[0.5, 1.0], [1.0, 2.0]]
    """
    missing = set(columns) - set(coefficient_of)
    if missing:
        raise ValueError(f"No coefficient for {sorted(missing)}.")
    tracts = np.sort(coefficients["GEOID"].unique())
    position = pd.Series(np.arange(len(tracts)), index=tracts)
    acc = np.zeros((len(columns), len(tracts), hours), dtype=np.float32)
    for building_id, use in coefficients.groupby("building_id"):
        rows = position.loc[use["GEOID"]].to_numpy()
        if tract_series is not None:
            values = np.asarray(
                tract_series(int(building_id), use["GEOID"].to_numpy()),
                dtype=np.float32,
            )
            expected = (len(columns), len(rows), hours)
            if values.shape != expected:
                raise ValueError(
                    f"The series of dwelling {building_id} are not {expected}."
                )
        else:
            series = read_series(int(building_id))
            if len(series) != hours:
                raise ValueError(
                    f"The series of dwelling {building_id} has no {hours} rows."
                )
            values = np.broadcast_to(
                series[list(columns)].to_numpy(dtype=np.float32).T[:, None, :],
                (len(columns), len(rows), hours),
            )
        for k, name in enumerate(columns):
            weight = use[coefficient_of[name]].to_numpy(dtype=np.float32)
            acc[k, rows, :] += weight[:, None] * values[k]
    return tracts, acc
