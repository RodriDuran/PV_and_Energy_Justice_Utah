"""Electricity of a heat pump from the useful energy it delivers.

The useful energy of a dwelling (heat or cold that reaches the air of the home)
becomes electricity through the efficiency of the equipment, which for a heat
pump depends on the outdoor temperature. Rather than assuming a curve, the
efficiency is measured on simulations of ResStock's heat pump package, an
air-source heat pump with ducts and electric resistance backup that NREL
validates: for every hour the useful energy delivered and the electricity used
are summed by bin of outdoor temperature and the ratio is the coefficient of
performance (COP) of the bin. The curve is then applied to the useful energy of
any dwelling, also to those to which the package does not apply.

The curve mixes the part-load behavior, the backup resistance and the losses of
the ducts and the fan of the dwellings in which it was measured. It is therefore
a curve of the system in the dwellings of the package (ducted), not of the
compressor alone; applied to dwellings without ducts it is an extrapolation.

References
----------
National Renewable Energy Laboratory (2025), ResStock 2025 release 1 upgrade
list (``resstock_2025_release1_readme``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cop_curve", "electricity_from_useful", "fan_electricity_by_mode"]


def fan_electricity_by_mode(
    heating: np.ndarray, cooling: np.ndarray, fan: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign the electricity of the fans to heating or cooling in each step.

    The fans run for both uses and are metered together. In every step they are
    assigned to the use that delivers more energy in that step.

    Parameters
    ----------
    heating, cooling : numpy.ndarray
        Useful energy delivered in each step.
    fan : numpy.ndarray
        Electricity of the fans in each step.

    Returns
    -------
    tuple of numpy.ndarray
        Fan electricity of heating and of cooling.

    Examples
    --------
    >>> h, c = fan_electricity_by_mode(np.array([1.0, 0.0]), np.array([0.0, 2.0]),
    ...                               np.array([3.0, 4.0]))
    >>> h.tolist(), c.tolist()
    ([3.0, 0.0], [0.0, 4.0])
    """
    heating_mode = np.asarray(heating, float) >= np.asarray(cooling, float)
    fan = np.asarray(fan, float)
    return np.where(heating_mode, fan, 0.0), np.where(heating_mode, 0.0, fan)


def cop_curve(
    useful: np.ndarray,
    electricity: np.ndarray,
    outdoor_temperature: np.ndarray,
    bin_width: float = 2.5,
    t_min: float = -25.0,
    t_max: float = 45.0,
    min_useful: float = 0.0,
) -> pd.DataFrame:
    """Coefficient of performance by bin of outdoor temperature.

    Parameters
    ----------
    useful : numpy.ndarray
        Useful energy delivered, one value per hour and dwelling (any unit).
    electricity : numpy.ndarray
        Electricity used, same unit and shape.
    outdoor_temperature : numpy.ndarray
        Outdoor temperature of each value, in degrees Celsius.
    bin_width : float
        Width of the bins, in degrees Celsius.
    t_min, t_max : float
        Range of temperatures that are binned; values outside it are ignored.
    min_useful : float
        Bins whose total useful energy is not above this value are dropped
        (they have too little energy for a stable ratio).

    Returns
    -------
    pandas.DataFrame
        One row per bin with ``t_out`` (center of the bin), ``cop`` (useful
        energy over electricity) and ``useful`` (total useful energy of the bin).

    Raises
    ------
    ValueError
        If the arrays differ in length.

    Examples
    --------
    >>> t = np.array([0.0, 0.5, 10.0])
    >>> cop_curve(np.array([2.0, 4.0, 6.0]), np.array([1.0, 2.0, 1.5]), t,
    ...           bin_width=5.0, t_min=-5.0, t_max=15.0)["cop"].tolist()
    [2.0, 4.0]
    """
    useful = np.asarray(useful, float).ravel()
    electricity = np.asarray(electricity, float).ravel()
    t = np.asarray(outdoor_temperature, float).ravel()
    if not len(useful) == len(electricity) == len(t):
        raise ValueError("useful, electricity and outdoor_temperature must match.")
    edges = np.arange(t_min, t_max + bin_width, bin_width)
    index = np.digitize(t, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        in_bin = index == k
        energy = useful[in_bin].sum()
        power = electricity[in_bin].sum()
        if in_bin.any() and energy > min_useful and power > 0:
            rows.append((edges[k] + bin_width / 2, energy / power, energy))
    return pd.DataFrame(rows, columns=["t_out", "cop", "useful"])


def electricity_from_useful(
    useful: np.ndarray, outdoor_temperature: np.ndarray, curve: pd.DataFrame
) -> np.ndarray:
    """Electricity needed to deliver a useful energy at a given temperature.

    The COP is interpolated linearly between the centers of the bins and is
    constant beyond the first and the last bin.

    Parameters
    ----------
    useful : numpy.ndarray
        Useful energy to deliver, one value per hour.
    outdoor_temperature : numpy.ndarray
        Outdoor temperature of each hour, in degrees Celsius.
    curve : pandas.DataFrame
        Output of :func:`cop_curve` with at least one row.

    Returns
    -------
    numpy.ndarray
        Electricity of each hour, in the unit of ``useful``. A negative useful
        energy gives a negative electricity, so that signed gaps can be summed.

    Raises
    ------
    ValueError
        If the curve is empty or the arrays differ in length.

    Examples
    --------
    >>> curve = pd.DataFrame({"t_out": [0.0, 10.0], "cop": [2.0, 4.0]})
    >>> electricity_from_useful(np.array([4.0, 4.0]), np.array([0.0, 10.0]),
    ...                         curve).tolist()
    [2.0, 1.0]
    """
    useful = np.asarray(useful, float)
    t = np.asarray(outdoor_temperature, float)
    if curve.empty:
        raise ValueError("The curve is empty.")
    if len(useful) != len(t):
        raise ValueError("useful and outdoor_temperature must have the same length.")
    ordered = curve.sort_values("t_out")
    cop = np.interp(t, ordered["t_out"].to_numpy(), ordered["cop"].to_numpy())
    return useful / cop
