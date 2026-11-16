"""Read ResStock annual results and express each dwelling in Census categories.

ResStock (National Renewable Energy Laboratory) simulates a sample of dwellings
and publishes, for each one, its characteristics and its annual results. This
module reads the annual file of a state and returns the table of modeled
dwellings (contract ``resstock_dwellings``) with tenure, year built and type of
building written with the labels of the American Community Survey table of
tenure by year built by units in structure (table B25127), so that the dwellings
can be matched with the counts of dwellings of each tract.

Two properties of the ResStock results shape how the table must be used. First,
the heating and cooling loads are the energy delivered by the system that each
dwelling has (they include distribution losses), so a dwelling without cooling
has a cooling load of zero. Second, each modeled dwelling stands for
``weight`` real dwellings (the same value for every dwelling of a state).

References
----------
White et al. (2024), ResStock 2024.2 technical documentation
(``white_2024_resstock``); National Renewable Energy Laboratory (2025), ResStock
2025 release 1 technical reference and data dictionary (``resstock_trg_2025``).
"""

from __future__ import annotations

import os

import pandas as pd

__all__ = [
    "BUILDING_TYPE_MAP",
    "FUEL_COLUMNS",
    "SOURCE_COLUMNS",
    "VINTAGE_MAP",
    "build_resstock_dwellings",
    "harmonize_resstock_types",
    "read_resstock_annual",
]

# Year-built classes of the Census table B25127 are: 2020 or later, 2000 to 2019,
# 1980 to 1999, 1960 to 1979, 1940 to 1959 and 1939 or earlier. Each ResStock
# decade falls wholly inside one class, so the translation needs no assumption.
# ResStock has no dwelling built in 2020 or later (its newest decade is the
# 2010s), so the Census cells of that class have no modeled dwelling. In the
# Utah baseline file the grouping coincides with ResStock's own field
# ``in.vintage_acs``, which is defined with the Census bins.
#: ResStock decade of construction -> year-built class of the Census table.
VINTAGE_MAP: dict[str, str] = {
    "<1940": "1939 or earlier",
    "1940s": "1940 to 1959",
    "1950s": "1940 to 1959",
    "1960s": "1960 to 1979",
    "1970s": "1960 to 1979",
    "1980s": "1980 to 1999",
    "1990s": "1980 to 1999",
    "2000s": "2000 to 2019",
    "2010s": "2000 to 2019",
}

# The units-in-structure classes of B25127 are: 1 (detached or attached), 2 to 4,
# 5 to 19, 20 to 49, 50 or more, and mobile home, boat, RV or van. ResStock's
# field ``in.geometry_building_type_acs`` is its building type "according to the
# U.S. Census American Community Survey" (data dictionary), in nine classes
# that group exactly into those six: detached and attached single-family
# dwellings are both "1", as in the Census table; boats, RVs and vans are not
# modeled, so "mobile" holds only mobile homes.
#: ResStock building type (Census classification) -> units-in-structure class.
BUILDING_TYPE_MAP: dict[str, str] = {
    "Single-Family Detached": "1",
    "Single-Family Attached": "1",
    "2 Unit": "2 to 4",
    "3 or 4 Unit": "2 to 4",
    "5 to 9 Unit": "5 to 19",
    "10 to 19 Unit": "5 to 19",
    "20 to 49 Unit": "20 to 49",
    "50 or more Unit": "50 or more",
    "Mobile Home": "mobile",
}

_KWH = "..kwh"

# ResStock reports the consumption of each end use apart from the consumption
# of the fans and pumps that distribute the heating or cooling, and apart from
# the heat pump backup (data dictionary: "excludes usage by fans/pumps" and
# "excludes heat pump backup"). The energy a household pays for heating or
# cooling includes all of them, so they are added by fuel here. Fans and pumps
# run on electricity whatever the heating fuel is.
#: Annual consumption (kWh) summed into one column per use and fuel. Heating
#: includes the heat pump backup and fans and pumps; cooling includes fans and
#: pumps. Wood and other fuels are not reported by ResStock.
FUEL_COLUMNS: dict[str, list[str]] = {
    "heating_kwh_electricity": [
        "out.electricity.heating.energy_consumption" + _KWH,
        "out.electricity.heating_hp_bkup.energy_consumption" + _KWH,
        "out.electricity.heating_hp_bkup_fa.energy_consumption" + _KWH,
        "out.electricity.heating_fans_pumps.energy_consumption" + _KWH,
    ],
    "heating_kwh_natural_gas": [
        "out.natural_gas.heating.energy_consumption" + _KWH,
        "out.natural_gas.heating_hp_bkup.energy_consumption" + _KWH,
    ],
    "heating_kwh_propane": [
        "out.propane.heating.energy_consumption" + _KWH,
        "out.propane.heating_hp_bkup.energy_consumption" + _KWH,
    ],
    "heating_kwh_fuel_oil": [
        "out.fuel_oil.heating.energy_consumption" + _KWH,
        "out.fuel_oil.heating_hp_bkup.energy_consumption" + _KWH,
    ],
    "cooling_kwh_electricity": [
        "out.electricity.cooling.energy_consumption" + _KWH,
        "out.electricity.cooling_fans_pumps.energy_consumption" + _KWH,
    ],
}

#: ResStock columns the package reads besides the consumption columns.
SOURCE_COLUMNS = [
    "bldg_id",
    "upgrade",
    "applicability",
    "in.county",
    "in.puma",
    "in.vacancy_status",
    "in.tenure",
    "in.vintage",
    "in.geometry_building_type_acs",
    "in.heating_fuel",
    "in.bedrooms",
    "in.geometry_floor_area",
    "in.hvac_cooling_type",
    "in.occupants",
    "in.heating_setpoint",
    "in.cooling_setpoint",
    "in.weather_file_latitude",
    "in.weather_file_longitude",
    "weight",
    "out.load.heating.energy_delivered..kbtu",
    "out.load.cooling.energy_delivered..kbtu",
    "out.unmet_hours.heating..hr",
    "out.unmet_hours.cooling..hr",
]


def read_resstock_annual(path: str | os.PathLike) -> pd.DataFrame:
    """Read the columns this package uses from a ResStock annual results file.

    Parameters
    ----------
    path : path-like
        Parquet file of annual results of one state and one upgrade, for
        example ``UT_upgrade0.parquet`` (the baseline without upgrade).

    Returns
    -------
    pandas.DataFrame
        One row per modeled dwelling with the ResStock column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the file lacks a column the package needs, which happens when the
        ResStock release has a different data dictionary.
    """
    columns = [*SOURCE_COLUMNS, *(c for cols in FUEL_COLUMNS.values() for c in cols)]
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as err:
        raise KeyError(
            f"{path} lacks a column the package needs ({err}); check the "
            "data dictionary of the ResStock release"
        ) from err


def harmonize_resstock_types(raw: pd.DataFrame) -> pd.DataFrame:
    """Translate tenure, year built and building type to Census classes.

    Parameters
    ----------
    raw : pandas.DataFrame
        Table with the ResStock columns ``in.vacancy_status``, ``in.tenure``,
        ``in.vintage`` and ``in.geometry_building_type_acs`` (or the renamed
        ``vacancy_status``, ``tenure_source``, ``vintage_source`` and
        ``building_type_source``).

    Returns
    -------
    pandas.DataFrame
        Same index with columns ``tenure`` (``"Owner"`` or ``"Renter"``; missing
        for vacant dwellings, which have no tenure), ``vintage`` and
        ``building_type``.

    Raises
    ------
    ValueError
        If an occupied dwelling has a category without Census equivalent.

    Examples
    --------
    >>> raw = pd.DataFrame(
    ...     {
    ...         "vacancy_status": ["Occupied"],
    ...         "tenure_source": ["Renter"],
    ...         "vintage_source": ["1950s"],
    ...         "building_type_source": ["3 or 4 Unit"],
    ...     }
    ... )
    >>> harmonize_resstock_types(raw).iloc[0].tolist()
    ['Renter', '1940 to 1959', '2 to 4']
    """
    # Accept the original ResStock column names or the short names used inside
    # the package, so the function works on raw and on prepared tables.
    names = {
        "vacancy_status": "in.vacancy_status",
        "tenure_source": "in.tenure",
        "vintage_source": "in.vintage",
        "building_type_source": "in.geometry_building_type_acs",
    }
    src = pd.DataFrame(
        {
            short: raw[short] if short in raw.columns else raw[long]
            for short, long in names.items()
        },
        index=raw.index,
    )
    # Vacant dwellings have no household and ResStock labels their tenure "Not
    # Available"; they stay without tenure instead of receiving an invented one.
    out = pd.DataFrame(
        {
            "tenure": src["tenure_source"].where(
                src["tenure_source"].isin(["Owner", "Renter"])
            ),
            "vintage": src["vintage_source"].map(VINTAGE_MAP),
            "building_type": src["building_type_source"].map(BUILDING_TYPE_MAP),
        },
        index=raw.index,
    )
    # Every occupied dwelling must translate: a silent gap would drop dwellings
    # from the cells that are later matched with the Census counts.
    occupied = src["vacancy_status"] == "Occupied"
    for col, source in (
        ("tenure", "tenure_source"),
        ("vintage", "vintage_source"),
        ("building_type", "building_type_source"),
    ):
        bad = occupied & out[col].isna()
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} occupied dwellings have {source} values "
                f"without Census equivalent: {sorted(set(src.loc[bad, source]))}"
            )
    return out


def _to_float_setpoint(series: pd.Series) -> pd.Series:
    """Convert ResStock setpoint labels such as ``"68F"`` to degrees Fahrenheit."""
    return pd.to_numeric(series.astype(str).str.replace("F", "", regex=False))


def build_resstock_dwellings(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the table of modeled dwellings from an annual results file.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`read_resstock_annual`.

    Returns
    -------
    pandas.DataFrame
        One row per modeled dwelling and upgrade, as the contract
        ``resstock_dwellings`` requires, plus: ``applicable`` (``False`` where
        the upgrade does not apply to the dwelling; ResStock then repeats the
        baseline results), ``heating_fuel`` (primary heating fuel),
        ``heating_setpoint_f`` and ``cooling_setpoint_f`` (baseline setpoints,
        degrees Fahrenheit, without the setback offsets) and the annual
        consumption in kWh by use and fuel (``heating_kwh_electricity``,
        ``heating_kwh_natural_gas``, ``heating_kwh_propane``,
        ``heating_kwh_fuel_oil`` and ``cooling_kwh_electricity``). Heating and
        cooling loads are the energy delivered by the existing system, in kBtu.
        ``occupants`` counts ``"10+"`` as 10.

    Raises
    ------
    ValueError
        If an occupied dwelling has a category without Census equivalent.
    """
    types = harmonize_resstock_types(raw)
    out = pd.DataFrame(index=raw.index)
    # Identification. A dwelling appears once per upgrade file, so the pair
    # (building_id, upgrade) is the key.
    out["building_id"] = raw["bldg_id"].astype("int64")
    out["upgrade"] = raw["upgrade"].astype("int64")
    out["applicable"] = raw["applicability"].astype(bool)
    out["county"] = raw["in.county"].astype(str)
    out["puma"] = raw["in.puma"].astype(str)
    # ``in.puma`` is the 2010 PUMA (data dictionary), not the 2020 one.
    out["occupied"] = raw["in.vacancy_status"] == "Occupied"
    out[["tenure", "vintage", "building_type"]] = types
    out["heating_fuel"] = raw["in.heating_fuel"].astype(str)
    # Number of real dwellings that each modeled dwelling stands for (data
    # dictionary); it is constant within a state and year of the stock.
    # Characteristics that are not part of the type of dwelling but are published
    # by the Census for each tract, used to check the composition of the dwellings
    # chosen for a tract. ResStock codes bedrooms 1 to 5 (data dictionary) and
    # names the cooling system; "None" means no cooling.
    out["bedrooms"] = pd.to_numeric(raw["in.bedrooms"].astype(str)).astype("int64")
    out["cooling_type"] = raw["in.hvac_cooling_type"].astype(str)
    # Finished floor area in the bins of the American Housing Survey (data
    # dictionary), for example "1500-1999" square feet; used as a control when
    # the sensitivity of the load to the climate is estimated.
    out["floor_area"] = raw["in.geometry_floor_area"].astype(str)
    out["weight"] = raw["weight"].astype(float)
    # Loads: energy delivered by the existing heating or cooling system, with
    # distribution losses (data dictionary). Unmet hours: hours in which the
    # setpoint is not kept, which feed the thermal-comfort deficit later.
    for target, source in (
        ("heating_load_kbtu", "out.load.heating.energy_delivered..kbtu"),
        ("cooling_load_kbtu", "out.load.cooling.energy_delivered..kbtu"),
        ("unmet_heating_hours", "out.unmet_hours.heating..hr"),
        ("unmet_cooling_hours", "out.unmet_hours.cooling..hr"),
    ):
        out[target] = raw[source].astype(float)
    # The top class of ResStock is "10+"; it counts as 10, a lower bound.
    out["occupants"] = pd.to_numeric(
        raw["in.occupants"].astype(str).str.replace("+", "", regex=False)
    ).astype(float)
    # Baseline setpoints, without the setback offsets that ResStock applies at
    # some hours (fields ``in.*_setpoint_offset_*``); the offsets are not used.
    out["heating_setpoint_f"] = _to_float_setpoint(raw["in.heating_setpoint"])
    out["cooling_setpoint_f"] = _to_float_setpoint(raw["in.cooling_setpoint"])
    # Location of the weather station whose 2018 file drove the simulation; the
    # tract climate is later compared with the climate of this point.
    out["weather_lat"] = raw["in.weather_file_latitude"].astype(float)
    out["weather_lon"] = raw["in.weather_file_longitude"].astype(float)
    # Consumption by use and fuel; a missing component means no consumption.
    for target, sources in FUEL_COLUMNS.items():
        out[target] = raw[sources].fillna(0.0).sum(axis=1).astype(float)
    return out.reset_index(drop=True)
