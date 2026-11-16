"""Master table of census tracts and their link to ResStock geography.

ResStock describes its dwellings by county and Public Use Microdata Area
(PUMA, a statistical area of at least 100,000 people that is a union of
tracts) defined with 2010 census geography. The study works with 2020 tracts.
The official relationship file between 2010 and 2020 tracts is used to assign
each 2020 tract to a ResStock PUMA, with the share of its land area that falls
in that PUMA as a measure of how clean the assignment is.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

__all__ = [
    "build_tract_master",
    "build_tract_puma_share",
    "read_block_housing",
    "read_block_relationship",
    "read_resstock_tract_lookup",
    "resstock_gisjoin_to_geoid",
]

logger = logging.getLogger(__name__)

LOOKUP_COLUMNS = {
    "nhgis_2010_tract_gisjoin": "gisjoin_tract_2010",
    "nhgis_2010_county_gisjoin": "resstock_county",
    "nhgis_2010_puma_gisjoin": "resstock_puma",
    "puma_name": "resstock_puma_name",
    "building_america_climate_zone": "climate_zone_ba",
    "iecc_2012_climate_zone": "climate_zone_iecc",
}


def resstock_gisjoin_to_geoid(gisjoin: pd.Series) -> pd.Series:
    """Convert tract identifiers of the form ``G0100010020100`` to GEOID.

    The NHGIS identifier inserts a zero after the state and after the county
    codes: ``G`` + state (2) + ``0`` + county (3) + ``0`` + tract (6).

    Parameters
    ----------
    gisjoin : pandas.Series
        NHGIS tract identifiers (14 characters).

    Returns
    -------
    pandas.Series
        Eleven-character GEOID (state + county + tract).
    """
    return gisjoin.str[1:3] + gisjoin.str[4:7] + gisjoin.str[8:14]


def build_tract_master(
    tiger_dbf: str | os.PathLike,
    tract_puma_2020: str | os.PathLike,
    relationship_2010_2020: str | os.PathLike,
    resstock_lookup: str | os.PathLike,
    state_fips: str,
) -> pd.DataFrame:
    """Build the master table of 2020 census tracts of one state.

    Each 2020 tract is assigned to the ResStock county and PUMA that hold the
    largest share of its land area, using the 2010-2020 relationship file and
    the ResStock tract lookup table.

    Parameters
    ----------
    tiger_dbf : path-like
        Attribute table (``.dbf``) of the TIGER/Line 2020 tract file.
    tract_puma_2020 : path-like
        Census Bureau file assigning 2020 tracts to 2020 PUMAs.
    relationship_2010_2020 : path-like
        Census Bureau national relationship file between 2020 and 2010 tracts.
    resstock_lookup : path-like
        ResStock ``spatial_tract_lookup_table.csv``.
    state_fips : str
        Two-digit state FIPS code.

    Returns
    -------
    pandas.DataFrame
        One row per 2020 tract with ``GEOID``, ``county_fips``, ``puma_2020``,
        land and water area, the ResStock county, PUMA and climate zones,
        ``share_in_resstock_puma`` (share of the tract's land area, among the
        2010 tracts it overlaps, that lies in the assigned PUMA),
        ``n_resstock_pumas`` (PUMAs overlapped) and ``clean_puma_assignment``
        (the whole land area falls in one ResStock PUMA).
    """
    import geopandas  # optional dependency, imported when needed

    tracts = geopandas.read_file(tiger_dbf).drop(columns="geometry", errors="ignore")
    tracts = pd.DataFrame(tracts)[["GEOID", "NAMELSAD", "ALAND", "AWATER"]]
    tracts["county_fips"] = tracts.GEOID.str[:5]

    puma = pd.read_csv(tract_puma_2020, dtype=str, encoding="utf-8-sig")
    puma = puma[puma.STATEFP == state_fips]
    puma["GEOID"] = puma.STATEFP + puma.COUNTYFP + puma.TRACTCE
    tracts = tracts.merge(
        puma[["GEOID", "PUMA5CE"]].rename(columns={"PUMA5CE": "puma_2020"}),
        on="GEOID",
        how="left",
    )

    lookup = pd.read_csv(resstock_lookup, dtype=str)
    lookup = lookup[lookup.nhgis_2010_tract_gisjoin.str[1:3] == state_fips]
    lookup = lookup[list(LOOKUP_COLUMNS)].rename(columns=LOOKUP_COLUMNS)
    lookup["GEOID_TRACT_10"] = resstock_gisjoin_to_geoid(lookup.gisjoin_tract_2010)

    rel = pd.read_csv(relationship_2010_2020, sep="|", dtype=str, encoding="utf-8-sig")
    rel = rel[rel.GEOID_TRACT_20.str.startswith(state_fips, na=False)]
    rel = rel[rel.GEOID_TRACT_10.notna()].copy()
    rel["part"] = rel.AREALAND_PART.astype(float)
    rel = rel.merge(lookup, on="GEOID_TRACT_10", how="left")

    by_puma = (
        rel.groupby(["GEOID_TRACT_20", "resstock_puma"], dropna=False)["part"]
        .sum()
        .reset_index()
    )
    total = by_puma.groupby("GEOID_TRACT_20")["part"].transform("sum")
    by_puma["share"] = by_puma.part / total.where(total > 0)
    by_puma = by_puma.sort_values(["GEOID_TRACT_20", "share"], ascending=[True, False])
    n_pumas = by_puma.groupby("GEOID_TRACT_20")["resstock_puma"].nunique()
    best = by_puma.drop_duplicates("GEOID_TRACT_20").rename(
        columns={"GEOID_TRACT_20": "GEOID", "share": "share_in_resstock_puma"}
    )[["GEOID", "resstock_puma", "share_in_resstock_puma"]]

    attrs = (
        rel.merge(
            best[["GEOID", "resstock_puma"]],
            left_on=["GEOID_TRACT_20", "resstock_puma"],
            right_on=["GEOID", "resstock_puma"],
        )
        .sort_values("part", ascending=False)
        .drop_duplicates("GEOID")[
            [
                "GEOID",
                "resstock_county",
                "resstock_puma_name",
                "climate_zone_ba",
                "climate_zone_iecc",
            ]
        ]
    )
    out = tracts.merge(best, on="GEOID", how="left").merge(
        attrs, on="GEOID", how="left"
    )
    out["n_resstock_pumas"] = out.GEOID.map(n_pumas)
    out["clean_puma_assignment"] = out.share_in_resstock_puma.round(6) == 1
    return out.sort_values("GEOID").reset_index(drop=True)


def read_block_housing(path: str | os.PathLike) -> pd.DataFrame:
    """Read the occupancy counts by block written by ``download_decennial_blocks``.

    Parameters
    ----------
    path : path-like
        CSV file with the columns ``GEOID_BLOCK``, ``H3_001N``, ``H3_002N`` and
        ``H3_003N``.

    Returns
    -------
    pandas.DataFrame
        One row per 2020 block; ``GEOID_BLOCK`` is a 15-character string and the
        counts are integers (``H3_002N`` is the number of occupied housing units).
    """
    return pd.read_csv(path, dtype={"GEOID_BLOCK": str})


def read_block_relationship(path: str | os.PathLike) -> pd.DataFrame:
    """Read the 2010-2020 census block relationship file of one state.

    Parameters
    ----------
    path : path-like
        Pipe-separated text file ``tab2010_tab2020_st<fips>_<state>.txt`` of the
        Census Bureau.

    Returns
    -------
    pandas.DataFrame
        One row per intersection of a 2020 block with a 2010 block, with the
        identifiers of both (``STATE_*``, ``COUNTY_*``, ``TRACT_*``, ``BLK_*``)
        as strings and the land area of the intersection ``AREALAND_INT`` in
        square meters as float.
    """
    rel = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    rel["AREALAND_INT"] = rel["AREALAND_INT"].astype(float)
    return rel


def read_resstock_tract_lookup(
    path: str | os.PathLike, state_fips: str
) -> pd.DataFrame:
    """Read the ResStock table that assigns each 2010 tract to a county and PUMA.

    Parameters
    ----------
    path : path-like
        ResStock ``spatial_tract_lookup_table.csv``.
    state_fips : str
        Two-digit state FIPS code.

    Returns
    -------
    pandas.DataFrame
        One row per 2010 tract of the state with ``GEOID_TRACT_10`` (11
        characters), ``resstock_county`` and ``resstock_puma``.
    """
    lookup = pd.read_csv(path, dtype=str)
    lookup = lookup[lookup["nhgis_2010_tract_gisjoin"].str[1:3] == state_fips]
    out = lookup[
        [
            "nhgis_2010_tract_gisjoin",
            "nhgis_2010_county_gisjoin",
            "nhgis_2010_puma_gisjoin",
        ]
    ].rename(columns=LOOKUP_COLUMNS)
    out["GEOID_TRACT_10"] = resstock_gisjoin_to_geoid(out["gisjoin_tract_2010"])
    return out[["GEOID_TRACT_10", "resstock_county", "resstock_puma"]].reset_index(
        drop=True
    )


def build_tract_puma_share(
    block_housing: pd.DataFrame,
    block_relationship: pd.DataFrame,
    resstock_lookup: pd.DataFrame,
    state_fips: str,
) -> pd.DataFrame:
    """Split each 2020 tract among ResStock PUMAs by its occupied dwellings.

    ResStock assigns whole 2010 tracts to PUMAs, while the study uses 2020
    tracts, which can straddle two or more of those PUMAs. The share of a
    tract's occupied dwellings in each PUMA is later used to pool the modeled
    dwellings of all the PUMAs it overlaps, instead of forcing the tract into
    one PUMA. Dwellings, not land area, are used because the land of a tract can
    be mostly uninhabited; the U.S. Department of Energy LEAD tool also weights
    tracts that fall in two PUMAs by their occupied dwellings (Ma et al. 2024).

    Parameters
    ----------
    block_housing : pandas.DataFrame
        Output of :func:`read_block_housing` (2020 occupied dwellings by block).
    block_relationship : pandas.DataFrame
        Output of :func:`read_block_relationship`.
    resstock_lookup : pandas.DataFrame
        Output of :func:`read_resstock_tract_lookup`.
    state_fips : str
        Two-digit state FIPS code of the 2020 tracts.

    Returns
    -------
    pandas.DataFrame
        Contract ``tract_puma_share``: one row per 2020 tract and ResStock PUMA
        with ``occupied_units`` (occupied dwellings of the tract allocated to
        that PUMA; not an integer) and ``share`` (``occupied_units`` over the
        tract total; the shares of a tract add to one). Tracts without occupied
        dwellings are left out and reported at INFO level, because their share
        is undefined.

    Raises
    ------
    ValueError
        If occupied dwellings lie in a 2010 tract that is not in the ResStock
        lookup, or if a block with occupied dwellings is missing from the
        relationship file. Dropping them silently would lose dwellings.

    Notes
    -----
    The method assumes that the dwellings of a 2020 block are spread evenly
    over its land area, so the part that lies in each 2010 block is proportional
    to the land area of the intersection (areal interpolation). Blocks with no
    land in any intersection (water only) are split equally among their
    intersections. Intersections with 2010 blocks of another state are dropped,
    since the ResStock results of this state do not cover them; they are a few
    rows of border blocks. Each 2010 tract belongs to one PUMA in ResStock.

    References
    ----------
    Ma et al. (2024), LEAD Tool methodology, U.S. Department of Energy
    (``doe_2024_lead_methodology``). U.S. Census Bureau, 2020 block
    relationship files.

    Examples
    --------
    One 2020 block with 10 occupied dwellings, three quarters of its land in a
    2010 block of the first PUMA and one quarter in a block of the second:

    >>> housing = pd.DataFrame({"GEOID_BLOCK": ["490010001000001"], "H3_002N": [10]})
    >>> rel = pd.DataFrame(
    ...     {
    ...         "STATE_2010": ["49", "49"],
    ...         "COUNTY_2010": ["001", "001"],
    ...         "TRACT_2010": ["000100", "000200"],
    ...         "STATE_2020": ["49", "49"],
    ...         "COUNTY_2020": ["001", "001"],
    ...         "TRACT_2020": ["000100", "000100"],
    ...         "BLK_2020": ["0001", "0001"],
    ...         "AREALAND_INT": [30.0, 10.0],
    ...     }
    ... )
    >>> lookup = pd.DataFrame(
    ...     {
    ...         "GEOID_TRACT_10": ["49001000100", "49001000200"],
    ...         "resstock_county": ["G4900010", "G4900010"],
    ...         "resstock_puma": ["G49000001", "G49000002"],
    ...     }
    ... )
    >>> out = build_tract_puma_share(housing, rel, lookup, "49")
    >>> out[["resstock_puma", "share"]].values.tolist()
    [['G49000001', 0.75], ['G49000002', 0.25]]
    """
    # Keep the intersections between blocks of this state; the 2010 side may
    # hold a few blocks of neighbouring states along the border.
    rel = block_relationship[
        (block_relationship["STATE_2020"] == state_fips)
        & (block_relationship["STATE_2010"] == state_fips)
    ].copy()
    rel["GEOID_BLOCK"] = (
        rel["STATE_2020"] + rel["COUNTY_2020"] + rel["TRACT_2020"] + rel["BLK_2020"]
    )
    rel["GEOID_TRACT_20"] = rel["GEOID_BLOCK"].str[:11]
    rel["GEOID_TRACT_10"] = rel["STATE_2010"] + rel["COUNTY_2010"] + rel["TRACT_2010"]

    # Every block that has dwellings must be in the relationship file.
    housing = block_housing[["GEOID_BLOCK", "H3_002N"]]
    missing = set(housing.loc[housing["H3_002N"] > 0, "GEOID_BLOCK"]) - set(
        rel["GEOID_BLOCK"]
    )
    if missing:
        raise ValueError(
            f"{len(missing)} blocks with occupied dwellings are not in the "
            f"relationship file, for example {sorted(missing)[:3]}"
        )

    # Fraction of each 2020 block that lies in each intersection: proportional
    # to land area, or equal parts for blocks with no land (water only).
    land = rel.groupby("GEOID_BLOCK")["AREALAND_INT"].transform("sum")
    n_parts = rel.groupby("GEOID_BLOCK")["AREALAND_INT"].transform("size")
    rel["part"] = np.where(
        land > 0, rel["AREALAND_INT"] / land.where(land > 0, 1.0), 1.0 / n_parts
    )

    # Allocate the occupied dwellings of each block to the 2010 tracts and take
    # them to the ResStock PUMA of each 2010 tract.
    parts = rel.merge(housing, on="GEOID_BLOCK", how="inner")
    parts["occupied_units"] = parts["H3_002N"] * parts["part"]
    parts = parts.merge(
        resstock_lookup[["GEOID_TRACT_10", "resstock_puma"]],
        on="GEOID_TRACT_10",
        how="left",
    )
    lost = parts.loc[parts["resstock_puma"].isna(), "occupied_units"].sum()
    if lost > 0:
        raise ValueError(
            f"{lost:.1f} occupied dwellings lie in 2010 tracts that are not in "
            "the ResStock lookup"
        )

    out = parts.groupby(["GEOID_TRACT_20", "resstock_puma"], as_index=False)[
        "occupied_units"
    ].sum()
    out = out.rename(columns={"GEOID_TRACT_20": "GEOID"})
    total = out.groupby("GEOID")["occupied_units"].transform("sum")
    empty = out.loc[total <= 0, "GEOID"].unique()
    if len(empty):
        logger.info(
            "%d tracts without occupied dwellings left out: %s",
            len(empty),
            ", ".join(sorted(empty)),
        )
    # A PUMA with no occupied dwellings of the tract contributes nothing, so the
    # table lists only the PUMAs that hold part of the tract's dwellings.
    keep = (total > 0) & (out["occupied_units"] > 0)
    out = out[keep].copy()
    out["share"] = out["occupied_units"] / total[keep]
    return out.sort_values(["GEOID", "resstock_puma"]).reset_index(drop=True)
