"""Tests of the comparison of the modeled energy with the totals sold in a state."""

import doctest
import sys
import types

import pandas as pd
import pytest

from utah_ejpi import hourly as hr
from utah_ejpi import validation as val

SEDS_TEXT = """Table CT4.  Residential sector energy consumption estimates
2017 06 72 3 648 (s) 671 9,511
2018 06 72 6 656 (s) 682 9,715
Trillion Btu
2017 0.0 69.6 0.1 2.5 (s) 2.6 3.2 0.1 0.8 32.5 108.7 57.0 165.7
2018 0.0 70.3 0.2 2.5 (s) 2.7 4.2 0.1 1.1 33.1 111.5 57.5 169.0
"""


def test_docstring_examples_run():
    assert doctest.testmod(val).failed == 0


def test_gas_is_taken_from_the_table_in_trillion_btu():
    assert val._seds_gas_trillion_btu(SEDS_TEXT, 2018) == 70.3
    assert val._seds_gas_trillion_btu(SEDS_TEXT, 2017) == 69.6
    with pytest.raises(ValueError, match="2019"):
        val._seds_gas_trillion_btu(SEDS_TEXT, 2019)


def _eia_folder(tmp_path, monkeypatch):
    class Page:
        def extract_text(self):
            return SEDS_TEXT

    class Reader:
        def __init__(self, path):
            self.pages = [Page()]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=Reader))
    (tmp_path / "use_res_UT.pdf").write_bytes(b"%PDF")
    sales = pd.DataFrame(
        {
            "Year": [2018, 2018],
            "State": ["UT", "AZ"],
            "Industry Sector Category": ["Total Electric Industry"] * 2,
            "Residential": [9714507, 31000000],
        }
    )
    with pd.ExcelWriter(tmp_path / "sales_annual.xlsx") as writer:
        sales.to_excel(writer, index=False, startrow=1)
    return tmp_path


def test_residential_totals_of_a_state_come_in_kwh(tmp_path, monkeypatch):
    folder = _eia_folder(tmp_path, monkeypatch)
    out = val.read_eia_residential_totals(folder, "ut", 2018)
    assert out["natural_gas_kwh"] == pytest.approx(70.3e9 / 3.41214)
    assert out["electricity_kwh"] == pytest.approx(9714507e3)
    with pytest.raises(ValueError, match="No sales"):
        val.read_eia_residential_totals(folder, "UT", 2017)
    with pytest.raises(FileNotFoundError):
        val.read_eia_residential_totals(tmp_path / "none", "UT", 2018)


def test_factors_follow_where_the_difference_is_assumed_to_be():
    measured = {"natural_gas_kwh": 240.0, "electricity_kwh": 180.0}
    modeled = {
        "natural_gas": 100.0,
        "natural_gas_heating": 80.0,
        "electricity": 100.0,
        "electricity_cooling": 40.0,
    }
    use = val.measured_level_factors(measured, modeled, 2.0)
    assert use == pytest.approx({"heating": 1.25, "cooling": 0.75})
    even = val.measured_level_factors(measured, modeled, 2.0, "proportional")
    assert even == pytest.approx({"heating": 1.2, "cooling": 0.9})
    with pytest.raises(ValueError, match="allocation"):
        val.measured_level_factors(measured, modeled, 2.0, "other")
    low = {"natural_gas_kwh": 240.0, "electricity_kwh": 20.0}
    with pytest.raises(ValueError, match="non-positive"):
        val.measured_level_factors(low, modeled, 2.0)


def test_state_means_weight_the_tracts_by_their_dwellings(tmp_path):
    columns = {
        "electricity_heating_kwh": 1.0,
        "electricity_cooling_kwh": 2.0,
        "electricity_cooling_fans_pumps_kwh": 1.0,
        "electricity_lighting_interior_kwh": 3.0,
        "electricity_pv_kwh": -1.0,
        "natural_gas_heating_kwh": 4.0,
        "natural_gas_hot_water_kwh": 2.0,
        "load_heating_kbtu": 99.0,
    }
    for geoid, scale in (("49001000100", 1.0), ("49001000200", 3.0)):
        folder = tmp_path / f"GEOID={geoid}"
        folder.mkdir()
        frame = pd.DataFrame(
            {name: v * scale / 8760 for name, v in columns.items()}, index=range(8760)
        )
        frame.insert(
            0, "timestamp", pd.date_range("2018-01-01", periods=8760, freq="h")
        )
        frame.to_parquet(folder / "profile.parquet")
    dwellings = pd.Series({"49001000100": 100.0, "49001000200": 300.0})
    means = hr.state_annual_means(tmp_path, dwellings)
    factor = (100 * 1.0 + 300 * 3.0) / 400
    assert means["electricity"] == pytest.approx(7.0 * factor)
    assert means["electricity_pv"] == pytest.approx(-1.0 * factor)
    assert means["electricity_heating"] == pytest.approx(1.0 * factor)
    assert means["electricity_cooling"] == pytest.approx(3.0 * factor)
    assert means["natural_gas"] == pytest.approx(6.0 * factor)
    assert means["natural_gas_heating"] == pytest.approx(4.0 * factor)
    assert means["propane"] == 0.0
    with pytest.raises(FileNotFoundError):
        hr.state_annual_means(tmp_path, pd.Series({"49999999999": 1.0}))
