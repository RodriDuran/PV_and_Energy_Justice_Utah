"""Offline tests of the download helpers."""

import pytest

from utah_ejpi import download as dl


def test_fips_accepts_numeric_codes():
    assert dl._fips("49") == "49"
    assert dl._fips("6") == "06"


def test_fips_rejects_unknown_state(monkeypatch):
    monkeypatch.setattr(dl, "_state_table", lambda: {"UT": "49"})
    with pytest.raises(ValueError):
        dl._fips("ZZ")


def test_acs_to_long_reshapes_and_blanks_sentinels():
    rows = [
        [
            "NAME",
            "B25001_001E",
            "B25001_001M",
            "B25001_001EA",
            "B25001_001MA",
            "state",
            "county",
            "tract",
        ],
        ["Tract 1", "100", "20", "", "", "49", "001", "000100"],
        ["Tract 2", "-666666666", "-555555555", "", "", "49", "001", "000200"],
    ]
    out = dl.acs_to_long(rows, "B25001")
    assert out[0] == ("49001000100", "Tract 1", "B25001_001", "100", "20")
    assert out[1][3:] == ("", "")


def test_download_file_skips_existing(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x")
    assert dl.download_file("http://invalid.invalid/a", target) == target


def test_eia_state_totals_download_the_three_files_of_the_state(monkeypatch, tmp_path):
    seen = []

    def fake(url, path, overwrite=False):
        seen.append(url)
        return path

    monkeypatch.setattr(dl, "download_file", fake)
    paths = dl.download_eia_state_totals(tmp_path, "ut")
    assert [p.name for p in paths] == [
        "use_res_UT.pdf",
        "n3010ut2A.htm",
        "sales_annual.xlsx",
    ]
    assert seen[0].endswith("/state/seds/sep_use/res/pdf/use_res_UT.pdf")
    assert seen[1].endswith("/dnav/ng/hist/n3010ut2A.htm")


def test_eia930_downloads_the_two_halves_of_the_year(monkeypatch, tmp_path):
    seen = []

    def fake(url, path, overwrite=False):
        seen.append((url, path))
        return path

    monkeypatch.setattr(dl, "download_file", fake)
    paths = dl.download_eia930(tmp_path, 2018)
    assert [p.name for p in paths] == [
        "EIA930_BALANCE_2018_Jan_Jun.csv",
        "EIA930_BALANCE_2018_Jul_Dec.csv",
    ]
    assert seen[0][0].endswith("sixMonthFiles/EIA930_BALANCE_2018_Jan_Jun.csv")


def test_eia_monthly_download_the_three_files_of_the_state(monkeypatch, tmp_path):
    seen = []

    def fake(url, path, overwrite=False):
        seen.append(url)
        return path

    monkeypatch.setattr(dl, "download_file", fake)
    paths = dl.download_eia_monthly(tmp_path, "ut")
    assert [p.name for p in paths] == [
        "n3010ut2m.htm",
        "sales_revenue.xlsx",
        "na1501_sut_8a.htm",
    ]
    assert seen[0].endswith("/dnav/ng/hist/n3010ut2m.htm")
    assert seen[1].endswith("/electricity/data/eia861m/xls/sales_revenue.xlsx")
    assert seen[2].endswith("/dnav/ng/hist/na1501_sut_8a.htm")
