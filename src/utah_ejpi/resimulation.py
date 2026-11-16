"""Edit the EnergyPlus models of ResStock and its weather to simulate them again.

ResStock publishes, for every modeled dwelling, the input file of EnergyPlus
(``in.idf``) that produced its results, but not the weather file. To find the
energy that a dwelling needs under another thermostat, the model is simulated
again with the normative thermostat of :mod:`utah_ejpi.comfort_gap`. This module
holds the pure text edits that this takes: a weather file in the EnergyPlus
format rebuilt from the hourly weather table that ResStock publishes, the
requests of output variables, the normative thermostat, the ideal cooling system
for dwellings that have none, and two repairs that some models need because no
sizing run is made.

The edits are written against the models of ResStock 2025 release 1, which are
EnergyPlus 24.2 input files. They depend on the names that ResStock gives to its
objects (for example ``conditioned space``); :func:`check_energyplus_version`
and the assertions of each function fail instead of editing a model that does not
have the expected structure.

The weather file is a reconstruction. ResStock builds its weather files from the
hourly observations of the Integrated Surface Database for temperature, humidity,
wind and pressure and from satellite radiation of the National Solar Radiation
Database, and does not publish them. The table that it does publish has
temperature, humidity, wind and the three radiation components; the rest of the
fields are written as missing, and EnergyPlus computes the sky temperature from
the temperature and the dew point. Tested against the published loads, the
simulations reproduce the heating load within a few percent and the sensible
cooling at the cooling coil within a few percent; completing the sky cover from
the station observations did not improve the match.

References
----------
National Renewable Energy Laboratory (2025), ResStock 2025 release 1 technical
reference guide, pp. 51-52 (``resstock_trg_2025``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "ENERGYPLUS_VERSION",
    "OUTPUT_REQUESTS",
    "add_ideal_cooling",
    "build_weather_epw",
    "check_energyplus_version",
    "fix_foundation_custom_blocks",
    "fix_hydronic_flows",
    "normative_model",
    "prepare_model",
]

#: Version of EnergyPlus of the models of ResStock 2025 release 1.
ENERGYPLUS_VERSION = "24.2"

#: Output variables that the analysis reads, requested at every time step.
OUTPUT_REQUESTS = """
Output:Variable,*,Zone Mean Air Temperature,Timestep;
Output:Variable,*,Zone Air System Sensible Heating Energy,Timestep;
Output:Variable,*,Zone Air System Sensible Cooling Energy,Timestep;
Output:Variable,*,Cooling Coil Sensible Cooling Energy,Timestep;
Output:Variable,ideal air system,Zone Ideal Loads Zone Sensible Cooling Energy,Timestep;
Output:Variable,*,Baseboard Total Heating Energy,Timestep;
Output:Variable,Environment,Site Outdoor Air Drybulb Temperature,Timestep;
Output:Variable,heating setpoint,Schedule Value,Timestep;
Output:Variable,cooling setpoint,Schedule Value,Timestep;
Output:Meter,Heating:Electricity,Timestep;
Output:Meter,Cooling:Electricity,Timestep;
Output:Meter,Fans:Electricity,Timestep;
Output:Meter,Heating:NaturalGas,Timestep;
Output:Meter,Heating:Propane,Timestep;
Output:Meter,Heating:FuelOilNo2,Timestep;
"""

_WEATHER_COLUMNS = {
    "dry_bulb": "Dry Bulb",
    "humidity": "Relative Humidity",
    "wind_speed": "Wind Speed",
    "wind_direction": "Wind Direction",
    "global": "Global Horizontal",
    "direct": "Direct Normal",
    "diffuse": "Diffuse Horizontal",
}

# Fields of an EnergyPlus weather line that this module cannot fill; the
# values are the codes of "missing" of the format.
_QUALITY_FLAGS = "?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9"


def check_energyplus_version(idf_text: str, expected: str = ENERGYPLUS_VERSION) -> None:
    r"""Fail if a model is not an input file of the expected EnergyPlus version.

    Parameters
    ----------
    idf_text : str
        Text of the input file.
    expected : str
        Expected ``major.minor`` version.

    Raises
    ------
    ValueError
        If the file has no version object or the version differs.

    Examples
    --------
    >>> check_energyplus_version("Version,\n  24.2;\n")
    """
    match = re.search(r"^\s*Version,\s*([\d.]+)", idf_text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("The model has no Version object.")
    found = ".".join(match.group(1).split(".")[:2])
    if found != expected:
        raise ValueError(f"The model is for EnergyPlus {found}, not {expected}.")


def _column(table: pd.DataFrame, key: str) -> pd.Series:
    """Column of the weather table whose name starts with the key."""
    hits = [c for c in table.columns if str(c).startswith(key)]
    if len(hits) != 1:
        raise ValueError(f"The weather table needs exactly one column '{key}...'.")
    return table[hits[0]]


def build_weather_epw(
    weather: pd.DataFrame,
    name: str,
    latitude: float,
    longitude: float,
    time_zone: float,
    elevation: float,
    state_code: str = "UT",
    year: int = 2018,
) -> str:
    """Text of an EnergyPlus weather file from the hourly weather of ResStock.

    Parameters
    ----------
    weather : pandas.DataFrame
        Hourly table of ResStock for one county, 8,760 rows in local standard
        time starting at 01:00 of 1 January, with columns that start with
        ``Dry Bulb``, ``Relative Humidity``, ``Wind Speed``, ``Wind Direction``,
        ``Global Horizontal``, ``Direct Normal`` and ``Diffuse Horizontal``.
    name, latitude, longitude, time_zone, elevation : str or float
        Location, as written in the ``Site:Location`` object of the model.
        ``elevation`` is in meters and is used for the atmospheric pressure.
    state_code : str
        Two-letter code written in the header.
    year : int
        Calendar year of the data; it must start on a Monday (2018 does).

    Returns
    -------
    str
        Content of the weather file. The dew point is computed from the
        temperature and the humidity (Magnus formula) and the pressure from the
        elevation (standard atmosphere); radiation components and wind are copied;
        the other fields are written as missing.

    Raises
    ------
    ValueError
        If the table does not have 8,760 rows or lacks a column.

    Examples
    --------
    >>> import numpy as np
    >>> table = pd.DataFrame({
    ...     "Dry Bulb Temperature [C]": np.full(8760, 10.0),
    ...     "Relative Humidity [%]": 50.0, "Wind Speed [m/s]": 2.0,
    ...     "Wind Direction [Deg]": 90.0, "Global Horizontal Radiation [W/m2]": 0.0,
    ...     "Direct Normal Radiation [W/m2]": 0.0,
    ...     "Diffuse Horizontal Radiation [W/m2]": 0.0})
    >>> text = build_weather_epw(table, "Test", 40.0, -111.0, -7.0, 1300.0)
    >>> len(text.splitlines())
    8768
    """
    if len(weather) != 8760:
        raise ValueError("The weather table must have 8,760 rows.")
    cols = {k: _column(weather, v).to_numpy(float) for k, v in _WEATHER_COLUMNS.items()}
    temp, rh = cols["dry_bulb"], cols["humidity"]
    a, b = 17.62, 243.12
    gamma = np.log(np.clip(rh, 1, 100) / 100) + a * temp / (b + temp)
    dew = b * gamma / (a - gamma)
    pressure = 101325 * (1 - 2.25577e-5 * elevation) ** 5.25588
    lines = [
        f"LOCATION,{name},{state_code},USA,AMY{year},000000,{latitude},{longitude},"
        f"{time_zone},{elevation}",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,built from the hourly weather table of ResStock",
        "COMMENTS 2,",
        "DATA PERIODS,1,1,Data,Monday,1/1,12/31",
    ]
    stamps = pd.date_range(f"{year}-01-01 01:00", periods=8760, freq="h")
    for i, stamp in enumerate(stamps):
        hour = stamp.hour if stamp.hour != 0 else 24
        fields = [
            year, stamp.month, stamp.day, hour, 0, _QUALITY_FLAGS,
            f"{temp[i]:.1f}", f"{dew[i]:.1f}", f"{rh[i]:.0f}", f"{pressure:.0f}",
            9999, 9999, 9999,
            f"{cols['global'][i]:.0f}", f"{cols['direct'][i]:.0f}",
            f"{cols['diffuse'][i]:.0f}", 999999, 999999, 999999, 9999,
            f"{cols['wind_direction'][i]:.0f}", f"{cols['wind_speed'][i]:.1f}",
            99, 99, 9999, 99999, 9, 999999999, 999, 0.999, 999, 99, 999, 999, 99,
        ]  # fmt: skip
        lines.append(",".join(str(x) for x in fields))
    return "\n".join(lines) + "\n"


def fix_hydronic_flows(idf_text: str) -> str:
    r"""Set the autosized water flows of hydronic baseboard systems.

    Without a sizing run the autosized flow of the plant loop is zero and the
    baseboards deliver no heat. The loop flow is set to the sum of the maximum
    flows that the models already give to the baseboards.

    Parameters
    ----------
    idf_text : str
        Text of the model.

    Returns
    -------
    str
        The model with the plant flows written; unchanged when it has no
        baseboard flows.

    Examples
    --------
    >>> text = ("  0.001, !- Maximum Water Flow Rate {m3/s}\\n"
    ...         "  Autosize, !- Maximum Loop Flow Rate")
    >>> "0.001000000" in fix_hydronic_flows(text)
    True
    """
    flows = [
        float(x)
        for x in re.findall(
            r"([-\d.eE+]+),\s*!- Maximum Water Flow Rate \{m3/s\}", idf_text
        )
    ]
    if not flows:
        return idf_text
    total = f"{sum(flows):.9f}"
    text = re.sub(r"Autosize(,\s*!- Maximum Loop Flow Rate)", total + r"\1", idf_text)
    return re.sub(r"Autosize(,\s*!- Design Water Flow Rate)", total + r"\1", text)


def fix_foundation_custom_blocks(idf_text: str) -> str:
    """Move a custom insulation block of a foundation to the exterior fields.

    Some models of ResStock describe the insulation of a foundation with a
    ``Custom Block`` that EnergyPlus 24.2 rejects. The material and depth of
    the block are written in the fields of exterior vertical insulation, which
    is an approximation (the insulation is moved to the outside of the wall).

    Parameters
    ----------
    idf_text : str
        Text of the model.

    Returns
    -------
    str
        The model without custom blocks; unchanged when it has none.

    Examples
    --------
    >>> idf = open("run.idf", encoding="utf-8").read()  # doctest: +SKIP
    >>> fix_foundation_custom_blocks(idf)  # doctest: +SKIP
    """

    def one(match: re.Match[str]) -> str:
        obj = match.group(0)
        if "Custom Block Material Name" not in obj:
            return obj
        material = re.search(r"\n\s*([^,\n]+),\s*!- Custom Block Material Name 1", obj)
        depth = re.search(r"\n\s*([^,\n]+),\s*!- Custom Block Depth 1", obj)
        out = []
        for line in obj.rstrip(";").split("\n"):
            if "Custom Block" in line:
                continue
            if "Exterior Vertical Insulation Material Name" in line and material:
                line = (
                    f"  {material.group(1).strip()},"
                    "   !- Exterior Vertical Insulation Material Name"
                )
            if "Exterior Vertical Insulation Depth" in line and depth:
                line = (
                    f"  {depth.group(1).strip()},"
                    "   !- Exterior Vertical Insulation Depth {m}"
                )
            out.append(line)
        return "\n".join(out).rstrip().rstrip(",") + ";"

    return re.sub(r"Foundation:Kiva,\s*\n[^;]*;", one, idf_text)


def prepare_model(idf_text: str, fix_hydronic: bool = True) -> str:
    """Prepare a model of ResStock for a simulation that writes the outputs needed.

    Writes the file of time-step results, repairs the water flows of hydronic
    systems and appends the output requests of :data:`OUTPUT_REQUESTS`.

    Parameters
    ----------
    idf_text : str
        Text of the model.
    fix_hydronic : bool
        Apply :func:`fix_hydronic_flows`.

    Returns
    -------
    str
        The prepared model.

    Raises
    ------
    ValueError
        If the model is not for the expected version of EnergyPlus.

    Examples
    --------
    >>> idf = open("run.idf", encoding="utf-8").read()  # doctest: +SKIP
    >>> prepare_model(idf)  # doctest: +SKIP
    """
    check_energyplus_version(idf_text)
    text = fix_hydronic_flows(idf_text) if fix_hydronic else idf_text
    text = re.sub(
        r"(OutputControl:Files,\s*\n\s*No,[^\n]*\n\s*No,[^\n]*\n\s*)"
        r"No(,\s*!- Output ESO)",
        r"\1Yes\2",
        text,
    )
    return text + "\n" + OUTPUT_REQUESTS


def _field(chunk: str) -> str:
    """Value of a field of an input file without its comment."""
    return re.sub(r"!.*", "", chunk).strip()


def add_ideal_cooling(idf_text: str) -> str:
    """Add a cooling-only ideal system as the last equipment of the zone.

    The system has no capacity limit, no heating and no humidity control, and it
    is added as the last equipment of the conditioned zone so that the equipment
    of the dwelling serves the load first. The energy it delivers is the useful
    energy that the existing equipment does not provide. It takes the load of the
    dwellings with no cooling equipment and the remainder of partial systems.

    Parameters
    ----------
    idf_text : str
        Text of a model of ResStock with a zone called ``conditioned space``.

    Returns
    -------
    str
        The model with the ideal system.

    Raises
    ------
    AssertionError
        If the model does not have the structure that ResStock gives to the
        conditioned zone.

    Examples
    --------
    >>> idf = open("run.idf", encoding="utf-8").read()  # doctest: +SKIP
    >>> add_ideal_cooling(idf)  # doctest: +SKIP
    """
    connections = re.search(
        r"(ZoneHVAC:EquipmentConnections,\s*\n\s*conditioned space,[^\n]*\n\s*"
        r"conditioned space Equipment List,[^\n]*\n\s*)([^,\n]*),([^\n]*\n)",
        idf_text,
    )
    assert connections, "zone equipment connections not found"
    inlet = connections.group(2).strip()
    new = idf_text
    if inlet == "":
        new = new.replace(
            connections.group(0),
            connections.group(1) + "ideal supply node," + connections.group(3),
            1,
        )
    else:
        node_list = re.search(
            r"NodeList,\s*\n\s*" + re.escape(inlet) + r",[^\n]*\n\s*([^;]*?);", new
        )
        if node_list:
            new = new.replace(
                node_list.group(0),
                node_list.group(0)[:-1].rstrip() + ",\n  ideal supply node;",
                1,
            )
        else:
            new = new.replace(
                connections.group(0),
                connections.group(1) + "ideal inlet list," + connections.group(3),
                1,
            )
            new += (
                f"\nNodeList,\n  ideal inlet list,\n  {inlet},\n  ideal supply node;\n"
            )
    full = re.search(
        r"ZoneHVAC:EquipmentConnections,\s*\n\s*conditioned space,[^;]*;", new
    )
    assert full, "zone equipment connections not found"
    if len(full.group(0).rstrip(";").split(",")) <= 6:
        new = new.replace(
            full.group(0), full.group(0)[:-1] + ",\n  ideal return node;", 1
        )
    listing = re.search(
        r"ZoneHVAC:EquipmentList,\s*\n\s*conditioned space Equipment List,[^;]*;", new
    )
    assert listing, "equipment list not found"
    fields = [_field(x) for x in listing.group(0).rstrip(";").split(",")][1:]
    n_equipment = (len(fields) - 2) // 6
    # The fractions are written explicitly: without them EnergyPlus stops when
    # the other equipment of the list has fraction schedules.
    block = (
        listing.group(0).rstrip(";")
        + ",\n  ZoneHVAC:IdealLoadsAirSystem,\n  ideal cooling,\n"
        + f"  {n_equipment + 1},\n  {n_equipment + 1},\n"
        + "  Always On Continuous,\n  Always On Continuous;"
    )
    new = new.replace(listing.group(0), block)
    return new + _IDEAL_COOLING


_IDEAL_COOLING = """
ZoneHVAC:IdealLoadsAirSystem,
  ideal cooling,            !- Name
  ,                         !- Availability Schedule Name
  ideal supply node,        !- Zone Supply Air Node Name
  ,                         !- Zone Exhaust Air Node Name
  ,                         !- System Inlet Air Node Name
  50,                       !- Maximum Heating Supply Air Temperature
  13,                       !- Minimum Cooling Supply Air Temperature
  0.0156,                   !- Maximum Heating Supply Air Humidity Ratio
  0.0077,                   !- Minimum Cooling Supply Air Humidity Ratio
  LimitCapacity,            !- Heating Limit
  ,                         !- Maximum Heating Air Flow Rate
  0,                        !- Maximum Sensible Heating Capacity
  NoLimit,                  !- Cooling Limit
  ,                         !- Maximum Cooling Air Flow Rate
  ,                         !- Maximum Total Cooling Capacity
  ,                         !- Heating Availability Schedule Name
  ,                         !- Cooling Availability Schedule Name
  None,                     !- Dehumidification Control Type
  0.7,                      !- Cooling Sensible Heat Ratio
  None;                     !- Humidification Control Type
"""


def normative_model(
    idf_text: str, setpoint_file: str, ideal_cooling: bool = True
) -> str:
    """Replace the thermostat schedules of a model by those of a setpoint file.

    The thermostat of ResStock (``ThermostatSetpoint:DualSetpoint`` of the zone
    ``conditioned space``) points at the schedules ``heating setpoint`` and
    ``cooling setpoint``. They are replaced by two schedules read from a file
    that sits next to the model: column 1 is the heating setpoint and column 2
    the cooling setpoint, one row per 15 minutes of the year, with a header row.

    Parameters
    ----------
    idf_text : str
        Text of a prepared model (see :func:`prepare_model`).
    setpoint_file : str
        Name of the file of setpoints, with columns ``heating setpoint norm`` and
        ``cooling setpoint norm`` as written by the analysis layer.
    ideal_cooling : bool
        Add the ideal cooling system (see :func:`add_ideal_cooling`).

    Returns
    -------
    str
        The normative model.

    Raises
    ------
    ValueError
        If the model has no thermostat of the expected structure.

    Examples
    --------
    >>> idf = open("run.idf", encoding="utf-8").read()  # doctest: +SKIP
    >>> normative_model(idf, "setpoints.csv")  # doctest: +SKIP
    """
    pattern = (
        r"(ThermostatSetpoint:DualSetpoint,\s*\n\s*conditioned space temperature "
        r"setpoint,[^\n]*\n\s*)heating setpoint,(\s*!- Heating[^\n]*\n\s*)"
        r"cooling setpoint;"
    )
    text, count = re.subn(
        pattern, r"\1heating setpoint norm,\2cooling setpoint norm;", idf_text
    )
    if count != 1:
        raise ValueError("The model has no thermostat of the expected structure.")
    for column, name in ((1, "heating setpoint norm"), (2, "cooling setpoint norm")):
        text += (
            f"\nSchedule:File,\n  {name},\n  Temperature,\n  {setpoint_file},\n"
            f"  {column},\n  1,\n  8760,\n  Comma,\n  No,\n  15,\n  Yes;\n"
        )
    return add_ideal_cooling(text) if ideal_cooling else text
