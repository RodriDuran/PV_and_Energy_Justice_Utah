"""Dwellings and stations that enter the fit of the climate line of each use.

The line that takes the load of a dwelling from the climate of its weather station
to the climate of a tract is fitted with the counties of the states listed in
``config.CLIMATE_FIT_STATES`` for that use. Besides the state of the run, the
listed states bring counties of other climates, so that the tracts of the state
fall inside the range of the fit instead of beyond it.
"""

import pandas as pd

import config
from utah_ejpi import climate, resstock
from utah_ejpi import energy_requirement as er

FIT_STATIONS = f"climate_fit_stations_{config.STATE.lower()}.csv"


def fit_states(end_use: str) -> list[str]:
    """States whose counties are used to fit the line of one use of energy."""
    return list(config.CLIMATE_FIT_STATES.get(end_use, [config.STATE]))


def all_fit_states() -> list[str]:
    """Every state that enters a fit, the state of the run first."""
    others = {s for states in config.CLIMATE_FIT_STATES.values() for s in states}
    return [config.STATE, *sorted(others - {config.STATE})]


def read_dwellings(state: str) -> pd.DataFrame:
    """Modeled dwellings of the baseline of one state (contract resstock_dwellings)."""
    path = (
        config.RAW
        / "resstock"
        / "2025_amy2018_release_1"
        / f"metadata_annual_{state.lower()}"
        / f"{state}_upgrade0.parquet"
    )
    return resstock.build_resstock_dwellings(resstock.read_resstock_annual(path))


def fit_inputs(
    end_use: str, dwellings: pd.DataFrame, degree_days: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dwellings and station degree days for the fit of one use of energy.

    Parameters
    ----------
    end_use : {"heating", "cooling"}
        Use of energy.
    dwellings, degree_days : pandas.DataFrame
        Modeled dwellings and degree days of the state of the run; returned as they
        are when the use is fitted with that state alone.

    Returns
    -------
    tuple of pandas.DataFrame
        Dwellings of the fit states and the table of degree days with their
        stations.
    """
    states = fit_states(end_use)
    if states == [config.STATE]:
        return dwellings, degree_days
    path = config.INTERIM / FIT_STATIONS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run section B of 02_housing_stock.py first."
        )
    stations = pd.read_csv(path, dtype={"id": str})
    frames = [dwellings if s == config.STATE else read_dwellings(s) for s in states]
    return pd.concat(frames, ignore_index=True), stations


def extra_fit_data(
    dwellings: pd.DataFrame, degree_days: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Dwellings of the other fit states by use, and the stations of every state.

    For the known-population test, which leaves a unit of the state out and fits
    the line again: the dwellings of the other states are always in the fit.
    """
    extra = {}
    fit_stations = None
    for end_use in config.CLIMATE_FIT_STATES:
        fit_dwellings, stations = fit_inputs(end_use, dwellings, degree_days)
        others = fit_dwellings[~fit_dwellings["county"].isin(dwellings["county"])]
        if len(others):
            extra[end_use] = others
            fit_stations = stations
    return extra, fit_stations


def fit_sensitivity(
    end_use: str,
    dwellings: pd.DataFrame,
    degree_days: pd.DataFrame,
    controls: tuple[str, ...] = er.DEFAULT_CONTROLS,
) -> er.DegreeDaySensitivity:
    """Fit the climate line of one use with the states chosen in the configuration."""
    fit_dwellings, fit_stations = fit_inputs(end_use, dwellings, degree_days)
    return er.fit_degree_day_sensitivity(fit_dwellings, fit_stations, end_use, controls)


def daily_temperatures(dwellings: pd.DataFrame) -> tuple:
    """Daily mean temperature of PRISM at the tracts and at the stations.

    Returns the table of the tracts and the table of the stations (rows indexed by
    code, 365 columns) and the station of each modeled dwelling.
    """
    raw = config.RAW
    files = sorted((raw / "climate" / "prism_4km_tmean_daily_2018").glob("*.tif"))
    if len(files) != 365:
        raise FileNotFoundError(
            f"Expected 365 daily PRISM files in data/raw/climate, found {len(files)}."
        )
    tiger = (
        raw
        / "geography"
        / f"tiger_2020_tract_{config.STATE.lower()}"
        / f"tl_2020_{config.STATE_FIPS}_tract.shp"
    )
    points = climate.tract_points(tiger)
    tracts = pd.DataFrame(
        climate.prism_daily_at_points(files, points["lon"], points["lat"]).T,
        index=points["id"].astype(str),
    )
    stations = (
        dwellings.drop_duplicates("county")
        .rename(columns={"county": "id", "weather_lon": "lon", "weather_lat": "lat"})[
            ["id", "lon", "lat"]
        ]
        .reset_index(drop=True)
    )
    station_daily = pd.DataFrame(
        climate.prism_daily_at_points(files, stations["lon"], stations["lat"]).T,
        index=stations["id"],
    )
    return tracts, station_daily, dwellings.set_index("building_id")["county"]
