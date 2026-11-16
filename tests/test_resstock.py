"""Tests of the reading and translation of ResStock results."""

import pandas as pd
import pytest

from utah_ejpi.resstock import (
    BUILDING_TYPE_MAP,
    VINTAGE_MAP,
    build_resstock_dwellings,
    harmonize_resstock_types,
    read_resstock_annual,
)
from utah_ejpi.utils import validate_table


def test_every_resstock_category_maps_to_a_census_class():
    assert len(set(VINTAGE_MAP.values())) == 5
    assert set(BUILDING_TYPE_MAP.values()) == {
        "1",
        "2 to 4",
        "5 to 19",
        "20 to 49",
        "50 or more",
        "mobile",
    }


def test_harmonize_translates_and_leaves_vacant_tenure_missing(resstock_raw):
    out = harmonize_resstock_types(resstock_raw)
    assert out.loc[0].tolist() == ["Owner", "1980 to 1999", "1"]
    assert out.loc[1].tolist() == ["Renter", "1940 to 1959", "2 to 4"]
    assert out.loc[2].tolist() == ["Owner", "2000 to 2019", "mobile"]
    assert pd.isna(out.loc[3, "tenure"])
    assert out.loc[3, "vintage"] == "1939 or earlier"


def test_harmonize_fails_on_unknown_category_of_an_occupied_dwelling(resstock_raw):
    resstock_raw.loc[0, "in.vintage"] = "2020s"
    with pytest.raises(ValueError, match="2020s"):
        harmonize_resstock_types(resstock_raw)
    resstock_raw.loc[0, "in.vacancy_status"] = "Vacant"
    harmonize_resstock_types(resstock_raw)


def test_build_follows_the_contract_and_computes_columns(resstock_raw):
    out = build_resstock_dwellings(resstock_raw)
    validate_table(out, "resstock_dwellings")
    assert out["occupied"].tolist() == [True, True, True, False]
    assert out["occupants"].tolist() == [3.0, 10.0, 1.0, 0.0]
    assert out["bedrooms"].tolist() == [3, 2, 4, 1]
    assert out["floor_area"].tolist()[0] == "1500-1999"
    assert out["cooling_type"].tolist()[:2] == ["Central AC", "None"]
    assert out["heating_setpoint_f"].tolist() == [68.0, 70.0, 65.0, 55.0]
    assert out.loc[0, "heating_kwh_natural_gas"] == pytest.approx(9001.0)
    assert out.loc[1, "heating_kwh_electricity"] == pytest.approx(4500.0)
    assert out.loc[0, "heating_load_kbtu"] == 50000.0
    assert out["county"].tolist()[0] == "G4900010"


def test_read_annual_selects_columns_and_reports_missing_ones(resstock_raw, tmp_path):
    path = tmp_path / "annual.parquet"
    resstock_raw.assign(extra=1).to_parquet(path)
    assert "extra" not in read_resstock_annual(path).columns
    resstock_raw.drop(columns="in.county").to_parquet(path)
    with pytest.raises(KeyError, match="data dictionary"):
        read_resstock_annual(path)
    with pytest.raises(FileNotFoundError):
        read_resstock_annual(tmp_path / "none.parquet")
