"""Download of the raw data sources used by the study.

Every function writes under a destination directory given explicitly by the
caller, skips files that already exist (unless ``overwrite=True``) and returns
the list of local paths. Sources have national coverage; the ``state``
or ``states`` argument is always given by the caller, so the same call serves any
state.
"""

from __future__ import annotations

import csv
import io
import os
import re
import time
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

__all__ = [
    "ACS_YEAR",
    "acs_to_long",
    "download_ac_prevalence",
    "download_acs",
    "download_block_relationship",
    "download_decennial",
    "download_decennial_blocks",
    "download_eia930",
    "download_eia_state_totals",
    "download_file",
    "download_geography",
    "download_lead",
    "download_prism",
    "download_recs",
    "download_resstock",
    "download_resstock_timeseries",
]

ACS_YEAR = 2024
USER_AGENT = "utah-ejpi (research; python-requests)"

STATE_LIST = "https://www2.census.gov/geo/docs/reference/state.txt"
TIGER_TRACT = (
    "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip"
)
CARTO_TRACT = (
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_{fips}_tract_500k.zip"
)
RELATIONSHIP_2010_2020 = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/"
    "tab20_tract20_tract10_natl.txt"
)
REL2020 = "https://www2.census.gov/geo/docs/maps-data/data/rel2020"
AIANNH_TRACT = f"{REL2020}/aiannh/tab20_aiannh20_tract20_natl.txt"
BLOCK_RELATIONSHIP = f"{REL2020}/t10t20/TAB2010_TAB2020_ST{{fips}}.zip"
TRACT_PUMA_2020 = f"{REL2020}/2020_Census_Tract_to_2020_PUMA.txt"
ACS_API = "https://api.census.gov/data/{year}/acs/acs5"
DECENNIAL_API = "https://api.census.gov/data/2020/dec/dhc"

RESSTOCK_BASE = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1"
)
RESSTOCK_DOCS = (
    "README_resstock_2025_1.pdf",
    "ResStockTechnicalReferenceGuide_2025_1.pdf",
    "data_dictionary.tsv",
    "enumeration_dictionary.tsv",
    "upgrades_lookup.json",
    "Column Name Crosswalk.csv",
)

RECS_BASE = "https://www.eia.gov/consumption/residential/data/2020"
EIA930_BASE = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
EIA930_HALVES = ("Jan_Jun", "Jul_Dec")
EIA_STATE_BASE = "https://www.eia.gov"
RECS_FILES = {
    "recs2020_public_v7.csv": f"{RECS_BASE}/csv/recs2020_public_v7.csv",
    "microdata-guide.pdf": f"{RECS_BASE}/pdf/microdata-guide.pdf",
    "RECS_2020_Codebook_for_Public_File_-_v7.xlsx": (
        f"{RECS_BASE}/xls/RECS%202020%20Codebook%20for%20Public%20File%20-%20v7.xlsx"
    ),
}

PRISM_DAILY = "https://services.nacse.org/prism/data/get/us/{res}/{element}/{date}"

LEAD_BASE = "https://data.openei.org/files/6219"
LEAD_GENERAL = (
    "Data%20Dictionary%202022.xlsx",
    "LEAD%20Tool%20Census%20Tracts%20List%202022.xlsx",
)
LEAD_METHODOLOGY = (
    "https://www.energy.gov/sites/default/files/2024-12/lead-methodology_122024.pdf"
)

LACE_FILES = {
    "LACE_23_Tract.csv": (
        "https://www2.census.gov/programs-surveys/demo/datasets/lace/2023/"
        "LACE_23_Tract.csv"
    ),
    "2023-LACE-Quick-Guide.pdf": (
        "https://www2.census.gov/programs-surveys/demo/technical-documentation/"
        "lace/2023-LACE-Quick-Guide.pdf"
    ),
}
# Harvard Dataverse, doi:10.7910/DVN/7GLPD7 (Ahn and Uejio 2025, tract summary).
AHN_FILES = {"README_summary_tract.txt": 11905137, "summary_tract.zip": 13916327}
DATAVERSE_ACCESS = "https://dataverse.harvard.edu/api/access/datafile/{id}"


@lru_cache(maxsize=1)
def _state_table() -> dict[str, str]:
    """Map USPS codes to two-digit FIPS codes using the Census Bureau list."""
    resp = requests.get(STATE_LIST, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    rows = csv.reader(io.StringIO(resp.text), delimiter="|")
    next(rows)
    return {r[1]: r[0] for r in rows if len(r) >= 2}


def _fips(state: str) -> str:
    """Return the two-digit FIPS code of a state given as USPS code or FIPS."""
    if state.isdigit():
        return state.zfill(2)
    table = _state_table()
    try:
        return table[state.upper()]
    except KeyError as err:
        raise ValueError(f"Unknown state {state!r}.") from err


def download_file(
    url: str,
    dest: str | os.PathLike,
    overwrite: bool = False,
    retries: int = 3,
    timeout: int = 120,
) -> Path:
    """Download one file to ``dest`` (streamed, written atomically).

    Parameters
    ----------
    url : str
        Source address.
    dest : path-like
        Local file path; parent directories are created.
    overwrite : bool
        Download again even if ``dest`` exists.
    retries : int
        Attempts before raising.
    timeout : int
        Seconds without data before an attempt fails.

    Returns
    -------
    pathlib.Path
        The local file path.

    Raises
    ------
    RuntimeError
        If every attempt fails (the source may be unavailable or have moved).

    Examples
    --------
    >>> download_file("https://example.org/table.csv", "data/raw/table.csv")
    ...     # doctest: +SKIP
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            ) as resp:
                resp.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            part.replace(dest)
            return dest
        except requests.RequestException as err:
            last = err
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def _download_many(items: Iterable[tuple[str, Path]], overwrite: bool) -> list[Path]:
    return [download_file(url, path, overwrite) for url, path in items]


def download_geography(
    dest: str | os.PathLike,
    states: Sequence[str],
    national: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """Download 2020 census-tract boundaries and the 2010-2020 relationship file.

    Parameters
    ----------
    dest : path-like
        Directory for the downloads.
    states : sequence of str
        States whose full (TIGER/Line) and cartographic boundary files are
        downloaded, as USPS code or FIPS code.
    national : bool
        Also download the national cartographic file, the official
        relationship file between 2010 and 2020 tracts, the file relating
        American Indian, Alaska Native and Native Hawaiian areas to tracts and
        the 2020 tract-to-PUMA file.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths of the zip and text files.

    Examples
    --------
    >>> download_geography("data/raw/geography", ["49"])  # doctest: +SKIP
    """
    dest = Path(dest)
    items = []
    for st in states:
        fips = _fips(st)
        items.append(
            (TIGER_TRACT.format(fips=fips), dest / f"tiger_2020_tract_{fips}.zip")
        )
        items.append(
            (
                CARTO_TRACT.format(fips=fips),
                dest / f"cartographic_2020_tract_{fips}.zip",
            )
        )
    if national:
        items.append(
            (CARTO_TRACT.format(fips="us"), dest / "cartographic_2020_tract_us.zip")
        )
        items.append(
            (
                RELATIONSHIP_2010_2020,
                dest / "relationship_2010_2020" / "tab20_tract20_tract10_natl.txt",
            )
        )
        items.append(
            (
                AIANNH_TRACT,
                dest
                / "relationship_aiannh_tract_2020"
                / "tab20_aiannh20_tract20_natl.txt",
            )
        )
        items.append(
            (
                TRACT_PUMA_2020,
                dest
                / "relationship_tract_puma_2020"
                / "2020_Census_Tract_to_2020_PUMA.txt",
            )
        )
    return _download_many(items, overwrite)


def _acs_group(table: str, state_fips: str, year: int, key: str | None) -> list:
    params = {
        "get": f"NAME,group({table})",
        "for": "tract:*",
        "in": f"state:{state_fips}",
    }
    if key:
        params["key"] = key
    resp = requests.get(
        ACS_API.format(year=year),
        params=params,
        timeout=300,
        headers={"User-Agent": USER_AGENT},
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Census API returned {resp.status_code} for {table}, state "
            f"{state_fips}: {resp.text[:200]}"
        )
    return resp.json()


def _acs_clean(value: str | None) -> str:
    """Blank out missing values and the API's large negative sentinels."""
    if value is None or value == "":
        return ""
    try:
        return "" if float(value) <= -100000000 else value
    except ValueError:
        return ""


def acs_to_long(rows: list, table: str) -> list[tuple[str, str, str, str, str]]:
    """Reshape an API response to (GEOID, NAME, variable, estimate, moe) rows.

    Parameters
    ----------
    rows : list
        Response of the Census API for tracts: the first row holds the names of the
        columns (``NAME``, the estimates ``<table>_<n>E``, their margins
        ``<table>_<n>M``, ``state``, ``county`` and ``tract``).
    table : str
        Table whose estimates are kept (for example ``"B25127"``).

    Returns
    -------
    list of tuple
        One row per tract and variable: ``(GEOID, NAME, variable, estimate,
        moe)``; the API's large negative codes for missing values become empty
        strings.

    Examples
    --------
    >>> rows = [
    ...     ["NAME", "B25127_002E", "B25127_002M", "state", "county", "tract"],
    ...     ["Tract 1", "10", "3", "49", "035", "100100"],
    ... ]
    >>> acs_to_long(rows, "B25127")
    [('49035100100', 'Tract 1', 'B25127_002', '10', '3')]
    """
    head = rows[0]
    idx = {name: i for i, name in enumerate(head)}
    pattern = re.compile(rf"^{re.escape(table)}_\d+E$")
    est_vars = sorted(v for v in head if pattern.match(v))
    out = []
    for r in rows[1:]:
        geoid = r[idx["state"]] + r[idx["county"]] + r[idx["tract"]]
        for ev in est_vars:
            mv = ev[:-1] + "M"
            out.append(
                (
                    geoid,
                    r[idx["NAME"]],
                    ev[:-1],
                    _acs_clean(r[idx[ev]]),
                    _acs_clean(r[idx[mv]]),
                )
            )
    return out


def download_acs(
    dest: str | os.PathLike,
    tables: Sequence[str],
    states: Sequence[str],
    year: int = ACS_YEAR,
    api_key: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download American Community Survey 5-year tables by census tract.

    One CSV per table is written in long format with columns ``GEOID``,
    ``NAME``, ``variable``, ``estimate`` and ``moe`` (margin of error).

    Parameters
    ----------
    dest : path-like
        Directory for the CSV files.
    tables : sequence of str
        Table identifiers, for example ``"B25127"``.
    states : sequence of str
        States to request, as USPS code or FIPS code.
    year : int
        Last year of the 5-year period.
    api_key : str, optional
        Census API key; defaults to the ``CENSUS_API_KEY`` environment variable.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local CSV paths, named ``<table>_tract_<state>.csv`` (``us`` when
        several states are requested).

    Examples
    --------
    >>> download_acs("data/raw/acs", ["B25127"], ["49"], year=2024)  # doctest: +SKIP
    """
    key = api_key or os.environ.get("CENSUS_API_KEY")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = "us" if len(states) > 1 else states[0].lower()
    paths = []
    for table in tables:
        path = dest / f"{table}_tract_{suffix}.csv"
        paths.append(path)
        if path.exists() and not overwrite:
            continue
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["GEOID", "NAME", "variable", "estimate", "moe"])
        for st in states:
            rows = _acs_group(table, _fips(st), year, key)
            writer.writerows(acs_to_long(rows, table))
        path.write_text(buf.getvalue(), encoding="utf-8")
    return paths


def download_decennial(
    dest: str | os.PathLike,
    states: Sequence[str],
    tables: Sequence[str] = ("P2", "H2"),
    api_key: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download 2020 Census (DHC file) tables by tract.

    Tables ``P2`` and ``H2`` give population and housing units in urban and
    rural areas. Counts are exact tabulations, so there is no margin of error.
    One CSV per table is written in long format (``GEOID``, ``NAME``,
    ``variable``, ``estimate``).

    Parameters
    ----------
    dest : path-like
        Directory for the CSV files.
    tables : sequence of str
        Table identifiers of the 2020 DHC file.
    states : sequence of str
        States to request, as USPS code or FIPS code.
    api_key : str, optional
        Census API key; defaults to the ``CENSUS_API_KEY`` environment variable.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local CSV paths, named ``<table>_dhc2020_tract_<state>.csv``.

    Examples
    --------
    >>> download_decennial("data/raw/census_2020", ["49"])  # doctest: +SKIP
    """
    key = api_key or os.environ.get("CENSUS_API_KEY")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = "us" if len(states) > 1 else states[0].lower()
    paths = []
    for table in tables:
        path = dest / f"{table}_dhc2020_tract_{suffix}.csv"
        paths.append(path)
        if path.exists() and not overwrite:
            continue
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["GEOID", "NAME", "variable", "estimate"])
        for st in states:
            params = {
                "get": f"NAME,group({table})",
                "for": "tract:*",
                "in": f"state:{_fips(st)}",
            }
            if key:
                params["key"] = key
            resp = requests.get(
                DECENNIAL_API,
                params=params,
                timeout=300,
                headers={"User-Agent": USER_AGENT},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Census API returned {resp.status_code} for {table}: "
                    f"{resp.text[:200]}"
                )
            rows = resp.json()
            idx = {name: i for i, name in enumerate(rows[0])}
            pattern = re.compile(rf"^{re.escape(table)}_\d+N$")
            cols = sorted(v for v in idx if pattern.match(v))
            for r in rows[1:]:
                geoid = r[idx["state"]] + r[idx["county"]] + r[idx["tract"]]
                for c in cols:
                    writer.writerow([geoid, r[idx["NAME"]], c, r[idx[c]]])
        path.write_text(buf.getvalue(), encoding="utf-8")
    return paths


def _block_rows_to_table(rows: list[list[str]]) -> pd.DataFrame:
    """Turn one county response of the block API into a table of housing counts."""
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    records = [
        {
            # A block code is state (2) + county (3) + tract (6) + block (4).
            "GEOID_BLOCK": r[idx["state"]]
            + r[idx["county"]]
            + r[idx["tract"]]
            + r[idx["block"]],
            "H3_001N": int(r[idx["H3_001N"]]),
            "H3_002N": int(r[idx["H3_002N"]]),
            "H3_003N": int(r[idx["H3_003N"]]),
        }
        for r in rows[1:]
    ]
    return pd.DataFrame(
        records, columns=["GEOID_BLOCK", "H3_001N", "H3_002N", "H3_003N"]
    )


def download_decennial_blocks(
    dest: str | os.PathLike,
    states: Sequence[str],
    api_key: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download 2020 Census housing-unit occupancy (table H3) by block.

    The counts are used to split each 2020 tract between the ResStock areas
    that overlap it, in proportion to its occupied dwellings (see
    :func:`utah_ejpi.geography.build_tract_puma_share`). The block API answers
    one county at a time, so the counties of each state are requested in turn.

    Parameters
    ----------
    dest : path-like
        Directory for the CSV files.
    states : sequence of str
        States to request, as USPS code or FIPS code.
    api_key : str, optional
        Census API key; defaults to the ``CENSUS_API_KEY`` environment variable.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        One CSV per state, named ``H3_dhc2020_block_<state>.csv``, with the
        columns ``GEOID_BLOCK`` (15 characters), ``H3_001N`` (housing units),
        ``H3_002N`` (occupied) and ``H3_003N`` (vacant).

    Raises
    ------
    RuntimeError
        If the Census API does not answer with status 200.


    Examples
    --------
    >>> download_decennial_blocks("data/raw/census_2020", ["49"])  # doctest: +SKIP

    References
    ----------
    U.S. Census Bureau, 2020 Census Demographic and Housing Characteristics
    File, table H3 (occupancy status), https://api.census.gov/data/2020/dec/dhc.
    """
    key = api_key or os.environ.get("CENSUS_API_KEY")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for st in states:
        fips = _fips(st)
        path = dest / f"H3_dhc2020_block_{st.lower() if not st.isdigit() else fips}.csv"
        paths.append(path)
        if path.exists() and not overwrite:
            continue
        auth = {"key": key} if key else {}
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(
            DECENNIAL_API,
            params={"get": "NAME", "for": "county:*", "in": f"state:{fips}", **auth},
            timeout=300,
            headers=headers,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Census API returned {resp.status_code} for the counties of "
                f"state {fips}: {resp.text[:200]}"
            )
        counties = [row[-1] for row in resp.json()[1:]]
        parts = []
        for county in counties:
            resp = requests.get(
                DECENNIAL_API,
                params={
                    "get": "H3_001N,H3_002N,H3_003N",
                    "for": "block:*",
                    "in": f"state:{fips} county:{county} tract:*",
                    **auth,
                },
                timeout=300,
                headers=headers,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Census API returned {resp.status_code} for blocks of "
                    f"county {fips}{county}: {resp.text[:200]}"
                )
            parts.append(_block_rows_to_table(resp.json()))
        pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return paths


def download_block_relationship(
    dest: str | os.PathLike,
    states: Sequence[str],
    overwrite: bool = False,
) -> list[Path]:
    """Download the 2010-2020 census block relationship file of each state.

    The file lists every intersection between a 2020 block and a 2010 block
    with the land and water area of the intersection. It is keyed by the 2020
    state, so it also holds the few 2010 blocks of neighbouring states that
    overlap its border blocks.

    Parameters
    ----------
    dest : path-like
        Directory for the downloads; the files are stored in the subdirectory
        ``relationship_blocks_2010_2020``.
    states : sequence of str
        States to request, as USPS code or FIPS code.
    overwrite : bool
        Download again even if the files exist.

    Returns
    -------
    list of pathlib.Path
        Paths of the extracted text files (pipe separated).


    Examples
    --------
    >>> download_block_relationship("data/raw/census_2020", ["49"])  # doctest: +SKIP

    References
    ----------
    U.S. Census Bureau, 2020 Census block relationship files,
    https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.2020.html.
    """
    folder = Path(dest) / "relationship_blocks_2010_2020"
    out = []
    for st in states:
        fips = _fips(st)
        archive = download_file(
            BLOCK_RELATIONSHIP.format(fips=fips),
            folder / f"TAB2010_TAB2020_ST{fips}.zip",
            overwrite,
        )
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                target = folder / name
                if overwrite or not target.exists():
                    zf.extract(name, folder)
                out.append(target)
    return out


def download_resstock(
    dest: str | os.PathLike,
    state: str,
    upgrades: Sequence[int] = (0,),
    weather: bool = True,
    timeseries_buildings: Sequence[str] = (),
    overwrite: bool = False,
) -> list[Path]:
    """Download ResStock 2025 release 1 (actual 2018 weather) files for a state.

    Always downloads the documentation, the tract-to-county/PUMA lookup table
    and the annual results of the requested upgrades. Series per dwelling are
    large (6-7 MB each) and are only downloaded for the file names given.

    Parameters
    ----------
    dest : path-like
        Directory for the release.
    state : str
        USPS state code.
    upgrades : sequence of int
        Upgrade packages whose annual results are downloaded (0 is baseline).
    weather : bool
        Download the weather file of every county of the state.
    timeseries_buildings : sequence of str
        File names such as ``"10007-0.parquet"`` of the baseline (upgrade 0)
        15-minute series to fetch.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths.

    Examples
    --------
    >>> download_resstock("data/raw/resstock", "UT")  # doctest: +SKIP
    """
    dest = Path(dest)
    st = state.upper()
    items = [
        (
            f"{RESSTOCK_BASE}/{name.replace(' ', '%20')}",
            dest / "docs" / name.replace(" ", "_"),
        )
        for name in RESSTOCK_DOCS
    ]
    items.append(
        (
            f"{RESSTOCK_BASE}/geographic_information/spatial_tract_lookup_table.csv",
            dest / "geo" / "spatial_tract_lookup_table.csv",
        )
    )
    annual = f"{RESSTOCK_BASE}/metadata_and_annual_results/by_state/full/parquet"
    for up in upgrades:
        items.append(
            (
                f"{annual}/state={st}/{st}_upgrade{up}.parquet",
                dest / f"metadata_annual_{st.lower()}" / f"{st}_upgrade{up}.parquet",
            )
        )
    for name in timeseries_buildings:
        items.append(
            (
                f"{RESSTOCK_BASE}/timeseries_individual_buildings/by_state/"
                f"upgrade=0/state={st}/{name}",
                dest / f"timeseries_{st.lower()}" / name,
            )
        )
    if weather:
        prefix = RESSTOCK_BASE.split(".amazonaws.com/", 1)[1]
        listing = requests.get(
            "https://oedi-data-lake.s3.amazonaws.com/",
            params={"list-type": "2", "prefix": f"{prefix}/weather/state={st}/"},
            timeout=60,
        )
        listing.raise_for_status()
        for key in re.findall(r"<Key>([^<]*_2018\.csv)</Key>", listing.text):
            fname = key.rsplit("/", 1)[1]
            items.append(
                (
                    f"{RESSTOCK_BASE}/weather/state={st}/{fname}",
                    dest / f"weather_{st.lower()}" / fname,
                )
            )
    return _download_many(items, overwrite)


def download_resstock_timeseries(
    dest: str | os.PathLike,
    building_ids: Iterable[int],
    state: str,
    upgrade: int = 0,
    workers: int = 8,
) -> list[Path]:
    """Download the 15-minute ResStock series of many dwellings in parallel.

    Files that already exist are skipped, so an interrupted run can be resumed.

    Parameters
    ----------
    dest : path-like
        Directory for the series.
    building_ids : iterable of int
        Dwelling identifiers (``bldg_id`` in the annual results).
    state : str
        USPS state code.
    upgrade : int
        Upgrade package (0 is the baseline).
    workers : int
        Simultaneous downloads.

    Returns
    -------
    list of pathlib.Path
        Local paths, named ``<bldg_id>-<upgrade>.parquet``.

    Examples
    --------
    >>> download_resstock_timeseries("data/raw/resstock_ts", [100, 200], "UT")
    ...     # doctest: +SKIP
    """
    dest = Path(dest)
    st = state.upper()

    def fetch(bid: int) -> Path:
        name = f"{bid}-{upgrade}.parquet"
        url = (
            f"{RESSTOCK_BASE}/timeseries_individual_buildings/by_state/"
            f"upgrade={upgrade}/state={st}/{name}"
        )
        return download_file(url, dest / name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, building_ids))


def download_recs(dest: str | os.PathLike, overwrite: bool = False) -> list[Path]:
    """Download the 2020 Residential Energy Consumption Survey public microdata.

    Parameters
    ----------
    dest : path-like
        Directory for the files (microdata CSV, microdata guide, codebook).
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths.

    Examples
    --------
    >>> download_recs("data/raw/recs_2020")  # doctest: +SKIP
    """
    dest = Path(dest)
    return _download_many(((u, dest / n) for n, u in RECS_FILES.items()), overwrite)


def download_eia930(
    dest: str | os.PathLike, year: int = 2018, overwrite: bool = False
) -> list[Path]:
    """Download the hourly demand of the balancing authorities for one year.

    The U.S. Energy Information Administration publishes, for every electric
    balancing authority, the hourly demand, generation and interchange it receives
    from the operators (form EIA-930), in two files of six months for each year.

    Parameters
    ----------
    dest : path-like
        Directory for the files.
    year : int
        Calendar year.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths of the two files (about 75 MB together for the whole country).

    Examples
    --------
    >>> download_eia930("data/raw/eia930_2018", year=2018)  # doctest: +SKIP
    """
    dest = Path(dest)
    items = (
        (
            f"{EIA930_BASE}/EIA930_BALANCE_{year}_{half}.csv",
            dest / f"EIA930_BALANCE_{year}_{half}.csv",
        )
        for half in EIA930_HALVES
    )
    return _download_many(items, overwrite)


def download_eia_state_totals(
    dest: str | os.PathLike, state: str, overwrite: bool = False
) -> list[Path]:
    """Download the measured residential consumption of a state.

    Three official files of the U.S. Energy Information Administration: the
    residential energy consumption of the state by fuel (State Energy Data
    System, table of the residential sector), the annual residential natural gas
    consumption of the state, and the electricity sales to ultimate customers by
    state and sector. They are totals sold by the utilities for the whole state,
    so they check the level of modeled consumption, not its split by use or tract.

    Parameters
    ----------
    dest : path-like
        Directory for the files.
    state : str
        Two-letter postal code of the state.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths of the three files (under 1 MB together).


    Examples
    --------
    >>> download_eia_state_totals("data/raw/eia_state_totals", "UT")  # doctest: +SKIP

    References
    ----------
    eia_2025_seds_consumption, eia_ng_residential_consumption_ut,
    eia_state_electricity_sales.
    """
    dest = Path(dest)
    code = state.upper()
    items = (
        (
            f"{EIA_STATE_BASE}/state/seds/sep_use/res/pdf/use_res_{code}.pdf",
            dest / f"use_res_{code}.pdf",
        ),
        (
            f"{EIA_STATE_BASE}/dnav/ng/hist/n3010{code.lower()}2A.htm",
            dest / f"n3010{code.lower()}2A.htm",
        ),
        (
            f"{EIA_STATE_BASE}/electricity/data/state/sales_annual.xlsx",
            dest / "sales_annual.xlsx",
        ),
    )
    return _download_many(items, overwrite)


def download_prism(
    dest: str | os.PathLike,
    year: int = 2018,
    element: str = "tmean",
    resolution: str = "4km",
    overwrite: bool = False,
) -> list[Path]:
    """Download PRISM daily gridded climate rasters for one year.

    PRISM (Oregon State University) interpolates station data to a national
    grid; each daily download is a zip with a GeoTIFF, unzipped in place.

    Parameters
    ----------
    dest : path-like
        Directory for the rasters.
    year : int
        Calendar year.
    element : str
        Climate element (``"tmean"`` is mean temperature).
    resolution : str
        ``"4km"`` or ``"800m"``.
    overwrite : bool
        Download again even if the raster exists.

    Returns
    -------
    list of pathlib.Path
        Paths of the GeoTIFF rasters.

    Examples
    --------
    >>> download_prism("data/raw/climate", year=2018)  # doctest: +SKIP
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31").strftime("%Y%m%d")
    tifs = []
    for d in days:
        tif = next(iter(dest.glob(f"*_{d}.tif")), None)
        if tif is not None and not overwrite:
            tifs.append(tif)
            continue
        zpath = download_file(
            PRISM_DAILY.format(res=resolution, element=element, date=d),
            dest / f"{d}.zip",
            overwrite=True,
        )
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest)
        zpath.unlink()
        tifs.extend(dest.glob(f"*_{d}.tif"))
    return tifs


def download_lead(
    dest: str | os.PathLike,
    states: Sequence[str],
    overwrite: bool = False,
) -> list[Path]:
    """Download the 2022 Low-Income Energy Affordability Data (LEAD).

    LEAD is a product of the U.S. Department of Energy with household energy
    spending by census tract.

    Parameters
    ----------
    dest : path-like
        Directory for the files.
    states : sequence of str
        USPS state codes (one zip per state).
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths (state zips, data dictionary, tract list, methodology).

    Examples
    --------
    >>> download_lead("data/raw/lead_2022", ["49"])  # doctest: +SKIP
    """
    dest = Path(dest)
    items = [(f"{LEAD_BASE}/{n}", dest / n.replace("%20", "_")) for n in LEAD_GENERAL]
    items.append((LEAD_METHODOLOGY, dest / "lead-methodology_122024.pdf"))
    for s in states:
        name = f"{s.upper()}-2022-LEAD-data.zip"
        items.append((f"{LEAD_BASE}/{name}", dest / name))
    return _download_many(items, overwrite)


def download_ac_prevalence(
    dest: str | os.PathLike, overwrite: bool = False
) -> list[Path]:
    """Download the two tract-level air-conditioning prevalence sources.

    The Census Bureau's Local Air Conditioning Estimates (2023, experimental)
    and the Ahn and Uejio (2025) modeled prevalence by type (Harvard Dataverse,
    doi:10.7910/DVN/7GLPD7), whose zip is unpacked.

    Parameters
    ----------
    dest : path-like
        Directory for the files.
    overwrite : bool
        Download again even if the file exists.

    Returns
    -------
    list of pathlib.Path
        Local paths.

    Examples
    --------
    >>> download_ac_prevalence("data/raw/ac_prevalence")  # doctest: +SKIP
    """
    dest = Path(dest)
    items = [(u, dest / "lace_2023" / n) for n, u in LACE_FILES.items()]
    items += [
        (DATAVERSE_ACCESS.format(id=i), dest / "ahn_uejio_2025_tract" / n)
        for n, i in AHN_FILES.items()
    ]
    paths = _download_many(items, overwrite)
    zpath = dest / "ahn_uejio_2025_tract" / "summary_tract.zip"
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            target = zpath.parent / name
            if name.endswith(".csv") and "__MACOSX" not in name and not target.exists():
                zf.extract(name, zpath.parent)
    return paths
