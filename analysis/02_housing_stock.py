"""Housing stock of each tract: link between the tracts and the ResStock dwellings.

Run from the project root: ``python analysis/02_housing_stock.py``. It needs the
raw data of ``01_download_data.py`` (blocks, block relationship, ResStock tract
lookup) and writes its results to ``data/interim/``.

Section A. Share of the occupied dwellings of each 2020 tract that lies in each
ResStock PUMA (``tract_puma_share_ut.csv``).

Section B. Degree days of the tracts, of the ResStock weather stations and of the
PUMAs, and the order of the PUMAs by climatic closeness to each tract
(``climate_degree_days_ut.csv``, ``puma_climate_rank_ut.csv``). Needs the PRISM
daily files and the ResStock weather files.

Section C. Modeled dwellings that stand for each type of dwelling in each tract,
and the dwellings of each tract by type (``tract_type_weights_ut.csv``,
``tract_type_donors_ut.csv``, ``fallback_shares_ut.csv``).
"""

import logging

import pandas as pd

import climate_fit
import config
from utah_ejpi import climate, geography, housing_stock, resstock
from utah_ejpi.utils import validate_table

STATE_FIPS = config.STATE_FIPS
STATE = config.STATE.lower()


def section_a_tract_puma_share() -> None:
    """Compute and save the share of each tract's dwellings by ResStock PUMA."""
    raw = config.RAW
    shares = geography.build_tract_puma_share(
        geography.read_block_housing(
            raw / "census_2020" / f"H3_dhc2020_block_{STATE}.csv"
        ),
        geography.read_block_relationship(
            raw
            / "geography"
            / "relationship_blocks_2010_2020"
            / f"tab2010_tab2020_st{STATE_FIPS}_{STATE}.txt"
        ),
        geography.read_resstock_tract_lookup(
            raw
            / "resstock"
            / "2025_amy2018_release_1"
            / "geo"
            / "spatial_tract_lookup_table.csv",
            STATE_FIPS,
        ),
        STATE_FIPS,
    )
    validate_table(shares, "tract_puma_share")
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    shares.to_csv(config.INTERIM / f"tract_puma_share_{STATE}.csv", index=False)
    split = shares.groupby("GEOID").size().gt(1).sum()
    logging.info("%d tracts, %d split between PUMAs", shares["GEOID"].nunique(), split)


def section_b_climate() -> None:
    """Compute degree days and order the PUMAs by climatic closeness."""
    raw = config.RAW
    heat, cool = config.HEATING_BASE_C, config.COOLING_BASE_C
    files = sorted((raw / "climate" / "prism_4km_tmean_daily_2018").glob("*.tif"))
    if len(files) != 365:
        raise FileNotFoundError(
            f"Expected 365 daily PRISM files in data/raw/climate, found {len(files)}."
        )
    tiger = (
        raw
        / "geography"
        / f"tiger_2020_tract_{STATE}"
        / f"tl_2020_{STATE_FIPS}_tract.shp"
    )
    tracts = climate.build_degree_day_table(
        files, climate.tract_points(tiger), "tract", heat, cool
    )

    # One weather station per ResStock county: the point of the file that
    # ResStock used to simulate the dwellings of that county.
    root = raw / "resstock" / "2025_amy2018_release_1"
    dwellings = resstock.build_resstock_dwellings(
        resstock.read_resstock_annual(
            root / f"metadata_annual_{STATE}" / f"{config.STATE}_upgrade0.parquet"
        )
    )
    stations = (
        dwellings.drop_duplicates("county")
        .rename(columns={"county": "id", "weather_lon": "lon", "weather_lat": "lat"})[
            ["id", "lon", "lat"]
        ]
        .reset_index(drop=True)
    )
    station_dd = climate.build_degree_day_table(files, stations, "station", heat, cool)
    fit_stations = [station_dd]
    for other in climate_fit.all_fit_states()[1:]:
        counties = climate_fit.read_dwellings(other).drop_duplicates("county")
        points = counties.rename(
            columns={"county": "id", "weather_lon": "lon", "weather_lat": "lat"}
        )[["id", "lon", "lat"]]
        fit_stations.append(
            climate.build_degree_day_table(files, points, "station", heat, cool)
        )
    fit_table = pd.concat(fit_stations, ignore_index=True)
    validate_table(fit_table, "climate_degree_days")
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    fit_table.to_csv(config.INTERIM / climate_fit.FIT_STATIONS, index=False)
    from_files = pd.DataFrame(
        [
            (
                sid,
                *climate.resstock_weather_degree_days(
                    root / f"weather_{STATE}" / f"{sid}_2018.csv", heat, cool
                ),
            )
            for sid in stations["id"]
        ],
        columns=["id", "hdd", "cdd"],
    )
    logging.info(
        "PRISM against ResStock weather files: %s",
        climate.compare_degree_days(station_dd, from_files),
    )

    shares = pd.read_csv(
        config.INTERIM / f"tract_puma_share_{STATE}.csv", dtype={"GEOID": str}
    )
    pumas = climate.puma_degree_days(tracts, shares)
    ranks = climate.rank_pumas_by_climate(tracts, pumas)
    validate_table(pd.concat([tracts, station_dd, pumas]), "climate_degree_days")
    validate_table(ranks, "puma_climate_rank")
    pd.concat([tracts, station_dd, pumas]).to_csv(
        config.INTERIM / f"climate_degree_days_{STATE}.csv", index=False
    )
    ranks.to_csv(config.INTERIM / f"puma_climate_rank_{STATE}.csv", index=False)


def section_c_type_weights() -> None:
    """Choose the modeled dwellings of each type and tract (base method)."""
    raw, out = config.RAW, config.INTERIM
    counts = housing_stock.acs_type_counts(
        housing_stock.read_acs_table(raw / "acs" / "acs5_2024" / "B25127_tract_us.csv"),
        STATE_FIPS,
    )
    dwellings = resstock.build_resstock_dwellings(
        resstock.read_resstock_annual(
            raw
            / "resstock"
            / "2025_amy2018_release_1"
            / f"metadata_annual_{STATE}"
            / f"{config.STATE}_upgrade0.parquet"
        )
    )
    shares = pd.read_csv(out / f"tract_puma_share_{STATE}.csv", dtype={"GEOID": str})
    master = pd.read_csv(out / f"tract_master_{STATE}.csv", dtype=str)
    rank = pd.read_csv(
        out / f"puma_climate_rank_{STATE}.csv",
        dtype={"GEOID": str, "resstock_puma": str},
    )
    weights, donors = housing_stock.build_type_donors(
        counts, dwellings, shares, master, rank, config.MIN_DONORS
    )
    validate_table(weights, "tract_type_weights")
    validate_table(donors, "tract_type_donors")
    weights.to_csv(out / f"tract_type_weights_{STATE}.csv", index=False)
    donors.to_csv(out / f"tract_type_donors_{STATE}.csv", index=False)
    summary = housing_stock.fallback_shares(weights)
    summary.to_csv(out / f"fallback_shares_{STATE}.csv", index=False)
    total = weights["weight"].sum()
    by_level = weights.groupby("source_level")["weight"].sum() / total
    logging.info(
        "Share of dwellings by fallback level: %s", by_level.round(4).to_dict()
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    section_a_tract_puma_share()
    section_b_climate()
    section_c_type_weights()
