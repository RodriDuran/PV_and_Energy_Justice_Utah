"""Tests of the edits to the EnergyPlus models and the weather file."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import resimulation as rs

MODEL = """Version,
  24.2;

OutputControl:Files,
  No,                                     !- Output CSV
  No,                                     !- Output MTR
  No,                                     !- Output ESO
  No;                                     !- Output EIO

ZoneHVAC:EquipmentConnections,
  conditioned space,                      !- Zone Name
  conditioned space Equipment List,       !- Zone Conditioning Equipment List Name
  conditioned space Inlet Node List,      !- Zone Air Inlet Node or NodeList Name
  ,                                       !- Zone Air Exhaust Node or NodeList Name
  Node 1,                                 !- Zone Air Node Name
  conditioned space Return Node List;     !- Zone Return Air Node or NodeList Name

NodeList,
  conditioned space Inlet Node List,      !- Name
  Node 6;                                 !- Node Name 1

NodeList,
  conditioned space Return Node List,     !- Name
  Node 7;                                 !- Node Name 1

ZoneHVAC:EquipmentList,
  conditioned space Equipment List,       !- Name
  SequentialLoad,                         !- Load Distribution Scheme
  ZoneHVAC:AirDistributionUnit,           !- Zone Equipment Object Type 1
  ADU furnace terminal,                   !- Zone Equipment Name 1
  1,                                      !- Zone Equipment Cooling Sequence 1
  1,                                      !- Zone Equipment Heating Sequence 1
  Sequential Fraction Schedule 1,         !- Cooling Fraction Schedule Name 1
  Sequential Fraction Schedule;           !- Heating Fraction Schedule Name 1

ThermostatSetpoint:DualSetpoint,
  conditioned space temperature setpoint, !- Name
  heating setpoint,                       !- Heating Setpoint Temperature Schedule Name
  cooling setpoint;                       !- Cooling Setpoint Temperature Schedule Name
"""


def _weather(n: int = 8760) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_time": pd.date_range("2018-01-01 01:00", periods=n, freq="h"),
            "Dry Bulb Temperature [C]": np.full(n, 10.0),
            "Relative Humidity [%]": np.full(n, 50.0),
            "Wind Speed [m/s]": np.full(n, 2.0),
            "Wind Direction [Deg]": np.full(n, 90.0),
            "Global Horizontal Radiation [W/m2]": np.full(n, 100.0),
            "Direct Normal Radiation [W/m2]": np.full(n, 80.0),
            "Diffuse Horizontal Radiation [W/m2]": np.full(n, 20.0),
        }
    )


def test_version_check():
    rs.check_energyplus_version(MODEL)
    with pytest.raises(ValueError, match="not 24.2"):
        rs.check_energyplus_version("Version,\n  23.1;\n")
    with pytest.raises(ValueError, match="no Version"):
        rs.check_energyplus_version("Zone,\n  a;\n")


def test_weather_file_has_the_header_and_one_line_per_hour():
    text = rs.build_weather_epw(_weather(), "Test_UT", 40.7, -112.0, -7.0, 1288.0)
    lines = text.splitlines()
    assert len(lines) == 8 + 8760
    assert lines[0].startswith("LOCATION,Test_UT,UT,USA,AMY2018")
    first = lines[8].split(",")
    assert len(first) == 35
    assert first[:5] == ["2018", "1", "1", "1", "0"]
    assert first[6] == "10.0" and first[8] == "50"
    assert float(first[7]) == pytest.approx(0.1, abs=0.5)  # dew point at 10 C, 50 %
    assert int(first[9]) == pytest.approx(86710, abs=200)  # pressure at 1288 m
    assert lines[-1].split(",")[3] == "24"  # last hour of 31 December


def test_weather_file_checks_the_table():
    with pytest.raises(ValueError, match="8,760"):
        rs.build_weather_epw(_weather(10), "x", 0, 0, 0, 0)
    without_wind = _weather().drop(columns="Wind Speed [m/s]")
    with pytest.raises(ValueError, match="exactly one column"):
        rs.build_weather_epw(without_wind, "x", 0, 0, 0, 0)


def test_hydronic_flows_are_summed_and_written():
    text = (
        "  0.001, !- Maximum Water Flow Rate {m3/s}\n"
        "  0.002, !- Maximum Water Flow Rate {m3/s}\n"
        "  Autosize, !- Maximum Loop Flow Rate {m3/s}\n"
        "  Autosize, !- Design Water Flow Rate {m3/s}\n"
    )
    fixed = rs.fix_hydronic_flows(text)
    assert fixed.count("0.003000000") == 2
    assert "Autosize" not in fixed
    assert rs.fix_hydronic_flows("nothing") == "nothing"


def test_custom_block_is_moved_to_the_exterior_fields():
    text = (
        "Foundation:Kiva,\n"
        "  Foundation Kiva 1,   !- Name\n"
        "  ,   !- Exterior Vertical Insulation Material Name\n"
        "  ,   !- Exterior Vertical Insulation Depth {m}\n"
        "  Rigid foam,   !- Custom Block Material Name 1\n"
        "  0.6,   !- Custom Block Depth 1\n"
        "  1;   !- Last field\n"
    )
    out = rs.fix_foundation_custom_blocks(text)
    assert "Custom Block" not in out
    assert "Rigid foam,   !- Exterior Vertical Insulation Material Name" in out
    assert "0.6,   !- Exterior Vertical Insulation Depth" in out
    plain = "Foundation:Kiva,\n  a;\n"
    assert rs.fix_foundation_custom_blocks(plain) == plain


def test_prepare_model_requests_outputs_and_turns_on_the_eso_file():
    out = rs.prepare_model(MODEL)
    assert "Yes,                                     !- Output ESO" in out
    assert "Zone Air System Sensible Cooling Energy" in out
    assert "Output:Meter,Heating:NaturalGas,Timestep;" in out
    with pytest.raises(ValueError, match="EnergyPlus"):
        rs.prepare_model(MODEL.replace("24.2", "22.1"))


def test_ideal_cooling_is_added_as_the_last_equipment():
    out = rs.add_ideal_cooling(MODEL)
    assert "ideal supply node;" in out
    assert "ZoneHVAC:IdealLoadsAirSystem,\n  ideal cooling,\n  2,\n  2," in out
    assert out.count("ZoneHVAC:IdealLoadsAirSystem,") == 2  # in the list and the object
    assert "NoLimit" in out


def test_ideal_cooling_fails_on_an_unexpected_model():
    with pytest.raises(AssertionError):
        rs.add_ideal_cooling("Version,\n  24.2;\n")


def test_normative_model_swaps_the_thermostat_schedules():
    out = rs.normative_model(MODEL, "setpoints_main.csv")
    assert "heating setpoint norm," in out and "cooling setpoint norm;" in out
    assert out.count("Schedule:File,") == 2
    assert "setpoints_main.csv" in out
    plain = rs.normative_model(MODEL, "s.csv", ideal_cooling=False)
    assert "ideal cooling" not in plain
    with pytest.raises(ValueError, match="thermostat"):
        rs.normative_model("Version,\n  24.2;\n", "s.csv")
