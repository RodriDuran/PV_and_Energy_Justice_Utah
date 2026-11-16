"""Offline tests of the downloads by census block."""

import zipfile

import pandas as pd
import pytest

from utah_ejpi import download as dl

HEADER = ["H3_001N", "H3_002N", "H3_003N", "state", "county", "tract", "block"]


class Reply:
    """Minimal stand-in for a successful ``requests`` response."""

    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        """Return the payload as the API would."""
        return self._payload


def test_block_rows_to_table_builds_the_block_code():
    rows = [HEADER, ["12", "10", "2", "49", "001", "000100", "1000"]]
    out = dl._block_rows_to_table(rows)
    assert out.GEOID_BLOCK.tolist() == ["490010001001000"]
    assert out.H3_002N.tolist() == [10]


def test_download_decennial_blocks_requests_each_county_once(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, timeout, headers):
        calls.append(params)
        if params["for"] == "county:*":
            names = [
                ["NAME", "state", "county"],
                ["A", "49", "001"],
                ["B", "49", "003"],
            ]
            return Reply(names)
        county = params["in"].split("county:")[1].split()[0]
        return Reply([HEADER, ["5", "4", "1", "49", county, "000100", "1000"]])

    monkeypatch.setattr(dl.requests, "get", fake_get)
    (path,) = dl.download_decennial_blocks(tmp_path, ["49"], api_key="k")
    table = pd.read_csv(path, dtype={"GEOID_BLOCK": str})
    assert table.GEOID_BLOCK.tolist() == ["490010001001000", "490030001001000"]
    assert len(calls) == 3
    # a second call finds the file and does not request again
    dl.download_decennial_blocks(tmp_path, ["49"], api_key="k")
    assert len(calls) == 3


def test_download_decennial_blocks_reports_api_errors(tmp_path, monkeypatch):
    class Bad:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: Bad())
    with pytest.raises(RuntimeError, match="500"):
        dl.download_decennial_blocks(tmp_path, ["49"], api_key="k")


def test_download_block_relationship_extracts_the_text_file(tmp_path, monkeypatch):
    def fake_download(url, dest, overwrite=False, **kwargs):
        assert url.endswith("TAB2010_TAB2020_ST49.zip")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("tab2010_tab2020_st49_ut.txt", "STATE_2010|X\n49|1\n")
        return dest

    monkeypatch.setattr(dl, "download_file", fake_download)
    (path,) = dl.download_block_relationship(tmp_path, ["49"])
    assert path.name == "tab2010_tab2020_st49_ut.txt"
    assert path.read_text().startswith("STATE_2010")
