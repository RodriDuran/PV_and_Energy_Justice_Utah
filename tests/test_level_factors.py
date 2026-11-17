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


def _monthly_folder(tmp_path, monkeypatch, gas_rows=None):
    folder = _eia_folder(tmp_path, monkeypatch)
    cells = "".join(f"<td>{v:,}</td>" for v in range(1000, 13000, 1000))
    rows = gas_rows or (
        f"<tr><td>&nbsp;&nbsp;2017</td>{cells}</tr><tr><td>&nbsp;&nbsp;2018</td>{cells}</tr>"
    )
    (folder / "n3010ut2m.htm").write_text(
        f"<table><tr><td>Year</td><td>Jan</td></tr>{rows}</table>", encoding="utf-8"
    )
    sales = pd.DataFrame(
        {
            "Year": [2018] * 12 + [2018, 2017],
            "Month": list(range(12, 0, -1)) + [1, 1],
            "State": ["UT"] * 12 + ["AZ", "UT"],
            "Data Status": ["Final"] * 14,
            "Revenue": [1.0] * 14,
            "Sales": [float(m) * 100 for m in range(12, 0, -1)] + [5.0, 7.0],
            "Customers": [1000.0] * 14,
        }
    )
    with pd.ExcelWriter(folder / "sales_revenue.xlsx") as writer:
        sales.to_excel(writer, sheet_name="Monthly-States", index=False, startrow=2)
    return folder


def test_monthly_totals_add_up_to_the_annual_total(tmp_path, monkeypatch):
    folder = _monthly_folder(tmp_path, monkeypatch)
    out = val.read_eia_monthly_residential(folder, "ut", 2018)
    assert out["month"].tolist() == list(range(1, 13))
    assert out["natural_gas_mmcf"].tolist() == list(range(1000, 13000, 1000))
    assert out["natural_gas_kwh"].sum() == pytest.approx(70.3e9 / 3.41214)
    assert out["electricity_mwh"].tolist() == [m * 100.0 for m in range(1, 13)]
    assert out["electricity_kwh"].iloc[0] == pytest.approx(100.0 * 1000.0)


def test_monthly_totals_need_a_complete_year(tmp_path, monkeypatch):
    folder = _monthly_folder(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="2017"):
        val.read_eia_monthly_residential(folder, "UT", 2017)
    with pytest.raises(ValueError, match="2016"):
        val.read_eia_monthly_residential(folder, "UT", 2016)
    with pytest.raises(FileNotFoundError):
        val.read_eia_monthly_residential(tmp_path / "none", "UT", 2018)


def test_hours_are_added_by_calendar_month(tmp_path):
    hours = pd.date_range("2018-01-01", periods=8760, freq="h")
    frame = pd.DataFrame({"timestamp": hours, "a": 1.0, "b": 2.0})
    months = hr.monthly_sums(frame, ["a"])
    assert months.index.tolist() == list(range(1, 13))
    assert months["a"].sum() == 8760.0
    assert months.loc[2, "a"] == 28 * 24


def test_state_monthly_means_agree_with_the_annual_means(tmp_path):
    for geoid, scale in (("49001000100", 1.0), ("49001000200", 3.0)):
        folder = tmp_path / f"GEOID={geoid}"
        folder.mkdir()
        frame = pd.DataFrame(
            {
                "electricity_heating_kwh": scale / 8760,
                "natural_gas_hot_water_kwh": 2.0 * scale / 8760,
                "electricity_pv_kwh": -scale / 8760,
            },
            index=range(8760),
        )
        frame.insert(
            0, "timestamp", pd.date_range("2018-01-01", periods=8760, freq="h")
        )
        frame.to_parquet(folder / "profile.parquet")
    dwellings = pd.Series({"49001000100": 100.0, "49001000200": 300.0})
    monthly = hr.state_monthly_means(tmp_path, dwellings)
    annual = hr.state_annual_means(tmp_path, dwellings)
    assert monthly.index.tolist() == list(range(1, 13))
    for name in ("electricity", "electricity_pv", "natural_gas"):
        assert monthly[name].sum() == pytest.approx(annual[name])
    assert monthly.loc[1, "natural_gas"] == pytest.approx(
        annual["natural_gas"] * 31 / 365
    )


def test_recs_end_use_means_use_the_state_and_the_weights(tmp_path):
    replicates = {f"NWEIGHT{i}": [1.0, 3.0, 5.0] for i in range(1, 61)}
    frame = pd.DataFrame(
        {
            "state_postal": ["UT", "UT", "AZ"],
            "NWEIGHT": [1.0, 3.0, 5.0],
            "BTUNGWTH": [3412.14, 0.0, 9999.0],
            "BTUNGNEC": [0.0, 3412.14, 0.0],
            **replicates,
        }
    )
    path = tmp_path / "recs.csv"
    frame.to_csv(path, index=False)
    out = val.recs_end_use_means(
        path, "ut", {"water": ("BTUNGWTH",), "both": ("BTUNGWTH", "BTUNGNEC")}
    ).set_index("use")
    assert out.loc["water", "mean_kwh"] == pytest.approx(1000.0 * 1 / 4)
    assert out.loc["both", "mean_kwh"] == pytest.approx(1000.0)
    assert out.loc["water", "share_with_use"] == pytest.approx(0.25)
    assert out.loc["water", "n_with_use"] == 1
    assert out.loc["water", "se_kwh"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        val.recs_end_use_means(path, "ut", {"x": ("MISSING",)})


def test_gas_customers_are_read_from_the_decade_table(tmp_path):
    page = (
        "<td>Decade</td><td>1990's</td><td>453,023</td><td>455,649</td>"
        "<td>467,664</td><td>484,438</td><td>503,583</td><td>523,622</td>"
        "<td>562,343</td><td>567,786</td><td>588,364</td><td>609,603</td>"
        "<td>2010's</td><td>821,525</td><td>830,219</td><td>840,687</td>"
        "<td>854,389</td><td>869,052</td><td>891,917</td><td>905,849</td>"
        "<td>926,623</td><td>950,644</td><td>976,042</td>"
        "<td>2020's</td><td>1,002,463</td><td>1,029,304</td>"
    )
    (tmp_path / "na1501_sut_8a.htm").write_text(page, encoding="utf-8")
    assert val.read_eia_gas_customers(tmp_path, "UT", 2018) == 950644.0
    assert val.read_eia_gas_customers(tmp_path, "ut", 2021) == 1029304.0
    with pytest.raises(ValueError, match="2023"):
        val.read_eia_gas_customers(tmp_path, "UT", 2023)
    with pytest.raises(ValueError, match="1985"):
        val.read_eia_gas_customers(tmp_path, "UT", 1985)
    with pytest.raises(FileNotFoundError):
        val.read_eia_gas_customers(tmp_path, "AZ", 2018)


def test_state_hourly_means_keep_the_hours_and_the_weights(tmp_path):
    for geoid, scale in (("49001000100", 1.0), ("49001000200", 3.0)):
        folder = tmp_path / f"GEOID={geoid}"
        folder.mkdir()
        frame = pd.DataFrame(
            {"natural_gas_hot_water_kwh": scale, "load_heating_kbtu": 9.0},
            index=range(8760),
        )
        frame.insert(
            0, "timestamp", pd.date_range("2018-01-01", periods=8760, freq="h")
        )
        frame.to_parquet(folder / "profile.parquet")
    dwellings = pd.Series({"49001000100": 100.0, "49001000200": 300.0})
    out = hr.state_hourly_means(tmp_path, dwellings)
    assert out.shape == (8760, 1)
    assert out["natural_gas_hot_water_kwh"].iloc[0] == pytest.approx(2.5)
    assert out.index[0] == pd.Timestamp("2018-01-01")
    with pytest.raises(ValueError, match="empty"):
        hr.state_hourly_means(tmp_path, pd.Series(dtype=float))


def test_base_load_factor_finds_the_other_uses_in_the_summer():
    heating = [10.0, 8, 5, 2, 0.5, 0.2, 0, 0, 0.2, 2, 5, 10]
    other = [1.0, 1, 1, 1, 1, 0.8, 0.7, 0.7, 0.8, 1, 1, 1]
    measured = [0.9 * h + 2.0 * o for h, o in zip(heating, other, strict=True)]
    out = val.base_load_heating_factor(measured, heating, other)
    assert out["other_uses_factor"] == pytest.approx(2.0, rel=0.02)
    assert out["heating_factor"] == pytest.approx(0.9, rel=0.02)
    assert abs(out["out_of_sample_error"]) < 0.02


def test_base_load_factor_needs_twelve_months_and_some_heating():
    with pytest.raises(ValueError, match="twelve"):
        val.base_load_heating_factor([1.0] * 11, [1.0] * 12, [1.0] * 12)
    heating = [1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    measured = [2.0, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2]
    with pytest.raises(ValueError, match="no gas for heating"):
        val.base_load_heating_factor(measured, heating, [1.0] * 12)


def test_resstock_end_use_means_use_occupied_dwellings_and_weights(tmp_path):
    frame = pd.DataFrame(
        {
            "weight": [1.0, 3.0, 5.0],
            "in.vacancy_status": ["Occupied", "Occupied", "Vacant"],
            "a": [4.0, 0.0, 99.0],
            "b": [0.0, 8.0, 99.0],
        }
    )
    path = tmp_path / "state.parquet"
    frame.to_parquet(path)
    means, households = val.resstock_end_use_means(
        path, {"a": ("a",), "both": ("a", "b")}
    )
    assert households == 4.0
    assert means["a"] == pytest.approx(1.0)
    assert means["both"] == pytest.approx(7.0)
