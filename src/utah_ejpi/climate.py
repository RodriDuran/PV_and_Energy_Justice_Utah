"""Degree days of the tracts, of the ResStock weather stations and of the PUMAs.

Heating and cooling degree days summarize how cold and how hot a year is: the
sum over the days of the amount by which the daily mean temperature is below a
heating base temperature or above a cooling base temperature. They are used in
two places of the study:

* the energy of each modeled dwelling, simulated by ResStock with the 2018
  weather file of a station of its county, is corrected to the climate of the
  tract where it is used (by the degree days of both places);
* when a type of dwelling has no modeled dwelling in the PUMA of a tract, the
  PUMAs of the state are ordered by how close their degree days are to those of
  the tract, and the dwellings are taken from the closest ones.

The temperature of the tracts and of the stations comes from the same source
(the PRISM daily mean temperature grid) so that the two can be compared; the
hourly weather files that ResStock used are read only to check that PRISM
agrees with them at the stations.

The base temperatures are arguments without default value: they are part of the
definition of thermal comfort, which the study has not fixed yet.

References
----------
PRISM Climate Group, Oregon State University (``prism_2018_daily_tmean``): daily
mean temperature on a 4 km grid. Smith et al. (2025, ``smith_2025_gridded``)
compare gridded climate products with weather stations and find PRISM the best
for cold extremes in the western United States. Degree days from daily means
follow the usual definition; Batlle et al. (2024, ``batlle_2024_energy_poverty_
allocations``) and the allocation formula of the U.S. low-income energy
assistance program use heating and cooling degree days as the measure of need.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "build_degree_day_table",
    "compare_degree_days",
    "degree_days",
    "prism_daily_at_points",
    "puma_degree_days",
    "rank_pumas_by_climate",
    "resstock_weather_degree_days",
    "tract_points",
]

# Value that PRISM writes in cells without data (outside the United States).
PRISM_NODATA = -9999.0

# ResStock weather files hold the 8760 hours of a non-leap year (2018).
HOURS_PER_YEAR = 8760


def degree_days(
    daily_temperature_c: np.ndarray | pd.DataFrame,
    heating_base_c: float,
    cooling_base_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute heating and cooling degree days from daily mean temperatures.

    Parameters
    ----------
    daily_temperature_c : array-like, shape (n_days, n_places)
        Daily mean temperature in degrees Celsius; a one-dimensional input is
        read as one place.
    heating_base_c : float
        Temperature below which a day counts as a heating day.
    cooling_base_c : float
        Temperature above which a day counts as a cooling day.

    Returns
    -------
    tuple of numpy.ndarray
        ``(hdd, cdd)``: for each place, the sum over the days of
        ``max(heating_base_c - T, 0)`` and of ``max(T - cooling_base_c, 0)``,
        in degrees Celsius times days.

    Raises
    ------
    ValueError
        If the heating base is above the cooling base, which would make a day
        both a heating and a cooling day.

    Examples
    --------
    >>> hdd, cdd = degree_days([[10.0], [20.0], [30.0]], 18.0, 24.0)
    >>> float(hdd[0]), float(cdd[0])
    (8.0, 6.0)
    """
    if heating_base_c > cooling_base_c:
        raise ValueError("The heating base temperature is above the cooling base.")
    temp = np.asarray(daily_temperature_c, dtype=float)
    if temp.ndim == 1:
        temp = temp.reshape(-1, 1)
    hdd = np.clip(heating_base_c - temp, 0.0, None).sum(axis=0)
    cdd = np.clip(temp - cooling_base_c, 0.0, None).sum(axis=0)
    return hdd, cdd


def _grid_of(page) -> tuple[float, float, float, float]:
    """Return (west edge, north edge, cell width, cell height) of a GeoTIFF page.

    The tie point of a GeoTIFF whose raster type is "pixel is area" is the outer
    corner of the upper-left cell. PRISM files are written that way (GeoTIFF key
    1025 equal to 1); any other layout is refused rather than guessed.
    """
    tags = {tag.name: tag.value for tag in page.tags.values()}
    keys = tags.get("GeoKeyDirectoryTag")
    raster_type = None
    if keys is not None:
        # The directory is a header of four numbers followed by entries of four
        # numbers (key id, location, count, value).
        for i in range(4, len(keys), 4):
            if keys[i] == 1025:
                raster_type = keys[i + 3]
    if raster_type != 1:
        raise ValueError("The raster is not of the 'pixel is area' type.")
    dx, dy, _ = tags["ModelPixelScaleTag"]
    _, _, _, west, north, _ = tags["ModelTiepointTag"]
    return west, north, dx, dy


def prism_daily_at_points(
    files: Sequence[str | os.PathLike],
    lon: Sequence[float],
    lat: Sequence[float],
) -> np.ndarray:
    """Read the daily mean temperature of PRISM files at a set of points.

    Parameters
    ----------
    files : sequence of path-like
        One GeoTIFF per day, in date order, all on the same grid.
    lon, lat : sequence of float
        Longitude and latitude of the points in degrees. PRISM is on the NAD83
        datum; the difference with WGS84 is a few meters, far below the 4 km cell,
        and is ignored.

    Returns
    -------
    numpy.ndarray, shape (n_days, n_points)
        Temperature of the grid cell that contains each point, in degrees
        Celsius.

    Raises
    ------
    ImportError
        If the optional package ``tifffile`` is not installed.
    ValueError
        If the files are not on the same grid, a point lies outside the grid or
        in a cell without data.

    Examples
    --------
    >>> files = sorted(prism_dir.glob("*.tif"))  # doctest: +SKIP
    >>> prism_daily_at_points(files, [-111.9], [40.8]).shape  # doctest: +SKIP
    """
    try:
        import tifffile  # optional dependency, imported when needed
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading PRISM files needs the optional package 'tifffile' "
            "(pip install utah-ejpi[climate])."
        ) from err
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out = np.empty((len(files), len(lon)))
    grid = None
    for i, path in enumerate(files):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            this = (page.shape, _grid_of(page))
            if grid is None:
                grid = this
                west, north, dx, dy = this[1]
                # Cell that contains each point (row counts down from the north).
                col = np.floor((lon - west) / dx).astype(int)
                row = np.floor((north - lat) / dy).astype(int)
                n_rows, n_cols = page.shape
                outside = (row < 0) | (row >= n_rows) | (col < 0) | (col >= n_cols)
                if outside.any():
                    raise ValueError(
                        f"{int(outside.sum())} points lie outside the PRISM grid"
                    )
            elif this != grid:
                raise ValueError(f"{path} is not on the same grid as the first file.")
            out[i] = page.asarray()[row, col]
    nodata = np.isclose(out, PRISM_NODATA).any(axis=0)
    if nodata.any():
        raise ValueError(
            f"{int(nodata.sum())} points fall in PRISM cells without data "
            f"(positions {np.flatnonzero(nodata)[:5].tolist()} ...)"
        )
    return out


def tract_points(tiger_path: str | os.PathLike) -> pd.DataFrame:
    """Return a point inside each tract, where its climate is read.

    Parameters
    ----------
    tiger_path : path-like
        TIGER/Line 2020 tract file (shapefile or zip).

    Returns
    -------
    pandas.DataFrame
        ``id`` (the 11-character ``GEOID``), ``lon`` and ``lat`` of a point
        guaranteed to lie inside the tract (its representative point, which
        unlike the centroid cannot fall outside an irregular tract). A tract is
        summarized by one 4 km cell, so large tracts whose climate changes
        within them are only represented at that point.

    Notes
    -----
    Requires the optional package ``geopandas``. TIGER/Line files are on the
    NAD83 datum, the same as PRISM, so no transformation is needed.

    Examples
    --------
    >>> tract_points("tl_2020_49_tract.dbf").head()  # doctest: +SKIP
    """
    import geopandas  # optional dependency, imported when needed

    tracts = geopandas.read_file(tiger_path)
    points = tracts.geometry.representative_point()
    return pd.DataFrame({"id": tracts["GEOID"], "lon": points.x, "lat": points.y})


def build_degree_day_table(
    files: Sequence[str | os.PathLike],
    points: pd.DataFrame,
    kind: str,
    heating_base_c: float,
    cooling_base_c: float,
) -> pd.DataFrame:
    """Compute the degree days of a set of points with the PRISM grid.

    Parameters
    ----------
    files : sequence of path-like
        Daily PRISM files of one year (see :func:`prism_daily_at_points`).
    points : pandas.DataFrame
        Columns ``id``, ``lon`` and ``lat``.
    kind : {"tract", "station"}
        Kind of place, written in the result.
    heating_base_c, cooling_base_c : float
        Base temperatures in degrees Celsius.

    Returns
    -------
    pandas.DataFrame
        Contract ``climate_degree_days``: ``id``, ``kind``, ``hdd``, ``cdd`` and
        the two base temperatures.

    Examples
    --------
    >>> files = sorted(prism_dir.glob("*.tif"))  # doctest: +SKIP
    >>> points = tract_points("tl_2020_49_tract.dbf")  # doctest: +SKIP
    >>> build_degree_day_table(files, points, "tract", 18.0, 26.0)  # doctest: +SKIP
    """
    daily = prism_daily_at_points(files, points["lon"], points["lat"])
    hdd, cdd = degree_days(daily, heating_base_c, cooling_base_c)
    return pd.DataFrame(
        {
            "id": points["id"].astype(str).to_numpy(),
            "kind": kind,
            "hdd": hdd,
            "cdd": cdd,
            "hdd_base_c": float(heating_base_c),
            "cdd_base_c": float(cooling_base_c),
        }
    )


def resstock_weather_degree_days(
    path: str | os.PathLike, heating_base_c: float, cooling_base_c: float
) -> tuple[float, float]:
    """Compute the degree days of one ResStock hourly weather file.

    Parameters
    ----------
    path : path-like
        ResStock weather file (CSV of 8760 hourly rows with a dry-bulb
        temperature column in degrees Celsius).
    heating_base_c, cooling_base_c : float
        Base temperatures in degrees Celsius.

    Returns
    -------
    tuple of float
        ``(hdd, cdd)`` from the daily means of the hourly temperature. This mean
        of 24 hours differs a little from the PRISM daily mean, which is the
        average of the daily maximum and minimum.

    Raises
    ------
    ValueError
        If the file has no dry-bulb temperature column or not 8760 rows.

    Examples
    --------
    >>> resstock_weather_degree_days("G4900350.epw.parquet", 18.0, 26.0)
    ...     # doctest: +SKIP
    """
    weather = pd.read_csv(path, encoding="latin-1")
    columns = [c for c in weather.columns if c.startswith("Dry Bulb Temperature")]
    if not columns or len(weather) != HOURS_PER_YEAR:
        raise ValueError(f"{path} is not a ResStock hourly weather file of one year.")
    hourly = weather[columns[0]].to_numpy(dtype=float)
    daily = hourly.reshape(-1, 24).mean(axis=1)
    hdd, cdd = degree_days(daily, heating_base_c, cooling_base_c)
    return float(hdd[0]), float(cdd[0])


def compare_degree_days(prism: pd.DataFrame, weather_files: pd.DataFrame) -> dict:
    """Compare the degree days of PRISM with those of the ResStock weather files.

    Parameters
    ----------
    prism : pandas.DataFrame
        Degree days of the stations from PRISM (contract ``climate_degree_days``).
    weather_files : pandas.DataFrame
        Columns ``id``, ``hdd`` and ``cdd`` computed from the ResStock files at
        the same stations with the same base temperatures.

    Returns
    -------
    dict
        For heating and for cooling: the correlation, the mean difference
        (PRISM minus file) and the largest absolute difference, in degree days.

    Examples
    --------
    >>> compare_degree_days(prism_table, weather_table)  # doctest: +SKIP
    """
    merged = prism.merge(weather_files, on="id", suffixes=("_prism", "_file"))
    if len(merged) != len(prism):
        raise ValueError("The two tables do not cover the same stations.")
    out = {}
    for name in ("hdd", "cdd"):
        diff = merged[f"{name}_prism"] - merged[f"{name}_file"]
        out[name] = {
            "correlation": float(merged[f"{name}_prism"].corr(merged[f"{name}_file"])),
            "mean_difference": float(diff.mean()),
            "max_abs_difference": float(diff.abs().max()),
        }
    return out


def puma_degree_days(
    tract_degree_days: pd.DataFrame, tract_puma_share: pd.DataFrame
) -> pd.DataFrame:
    """Summarize the climate of each ResStock PUMA from its tracts.

    Parameters
    ----------
    tract_degree_days : pandas.DataFrame
        Degree days of the tracts (contract ``climate_degree_days``, kind
        ``"tract"``; ``id`` is the tract ``GEOID``).
    tract_puma_share : pandas.DataFrame
        Contract ``tract_puma_share``.

    Returns
    -------
    pandas.DataFrame
        Contract ``climate_degree_days`` with ``kind`` equal to ``"puma"``: the
        degree days of a PUMA are the average of those of the tracts that lie in
        it, weighted by the occupied dwellings of each tract in the PUMA, so that
        the climate of a PUMA is the one its households live in.

    Examples
    --------
    >>> puma_degree_days(tract_degree_days, tract_puma_share)  # doctest: +SKIP
    """
    tracts = tract_degree_days[tract_degree_days["kind"] == "tract"]
    joined = tract_puma_share.merge(
        tracts, left_on="GEOID", right_on="id", how="inner", validate="many_to_one"
    )
    if len(joined) != len(tract_puma_share):
        raise ValueError("Some tracts of the share table have no degree days.")
    weights = joined["occupied_units"]
    base_h = float(tracts["hdd_base_c"].iloc[0])
    base_c = float(tracts["cdd_base_c"].iloc[0])
    grouped = joined.assign(
        hdd_w=joined["hdd"] * weights, cdd_w=joined["cdd"] * weights
    ).groupby("resstock_puma")
    total = grouped["occupied_units"].sum()
    return pd.DataFrame(
        {
            "id": total.index.astype(str),
            "kind": "puma",
            "hdd": (grouped["hdd_w"].sum() / total).to_numpy(),
            "cdd": (grouped["cdd_w"].sum() / total).to_numpy(),
            "hdd_base_c": base_h,
            "cdd_base_c": base_c,
        }
    )


def rank_pumas_by_climate(
    tract_degree_days: pd.DataFrame, puma_degree_days_table: pd.DataFrame
) -> pd.DataFrame:
    """Order the PUMAs of the state by climatic closeness to each tract.

    Parameters
    ----------
    tract_degree_days : pandas.DataFrame
        Degree days of the tracts (kind ``"tract"``).
    puma_degree_days_table : pandas.DataFrame
        Output of :func:`puma_degree_days`.

    Returns
    -------
    pandas.DataFrame
        One row per tract and PUMA with ``GEOID``, ``resstock_puma``,
        ``distance`` and ``rank`` (1 is the closest PUMA). The distance is
        Euclidean in the space of heating and cooling degree days, each divided
        by its standard deviation among the PUMAs so that heating degree days
        (thousands) do not outweigh cooling degree days (hundreds); with one
        PUMA, or no variation, the scale is 1.

    Notes
    -----
    This is the third level of the fallback rule for types of dwelling that have
    no modeled dwelling in the PUMA or the county of a tract.

    Examples
    --------
    >>> rank_pumas_by_climate(tract_degree_days, puma_table)  # doctest: +SKIP
    """
    tracts = tract_degree_days[tract_degree_days["kind"] == "tract"]
    pumas = puma_degree_days_table
    scale_h = float(pumas["hdd"].std(ddof=0)) or 1.0
    scale_c = float(pumas["cdd"].std(ddof=0)) or 1.0
    dh = (
        tracts["hdd"].to_numpy()[:, None] - pumas["hdd"].to_numpy()[None, :]
    ) / scale_h
    dc = (
        tracts["cdd"].to_numpy()[:, None] - pumas["cdd"].to_numpy()[None, :]
    ) / scale_c
    distance = np.sqrt(dh**2 + dc**2)
    out = pd.DataFrame(
        {
            "GEOID": np.repeat(tracts["id"].to_numpy(), len(pumas)),
            "resstock_puma": np.tile(pumas["id"].to_numpy(), len(tracts)),
            "distance": distance.ravel(),
        }
    )
    # Ties (identical distance) are broken by the PUMA code so that the order is
    # reproducible.
    out = out.sort_values(["GEOID", "distance", "resstock_puma"], kind="stable")
    out["rank"] = out.groupby("GEOID").cumcount() + 1
    return out.reset_index(drop=True)
