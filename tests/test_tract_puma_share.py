"""Tests of the split of 2020 tracts among ResStock PUMAs by occupied dwellings."""

import pandas as pd
import pytest

from utah_ejpi.geography import (
    build_tract_puma_share,
    read_block_housing,
    read_block_relationship,
    read_resstock_tract_lookup,
)
from utah_ejpi.utils import validate_table


def _rel(rows):
    """Build a block relationship table; rows are (tract10, tract20, blk20, area)."""
    return pd.DataFrame(
        {
            "STATE_2010": ["49"] * len(rows),
            "COUNTY_2010": ["001"] * len(rows),
            "TRACT_2010": [r[0] for r in rows],
            "STATE_2020": ["49"] * len(rows),
            "COUNTY_2020": ["001"] * len(rows),
            "TRACT_2020": [r[1] for r in rows],
            "BLK_2020": [r[2] for r in rows],
            "AREALAND_INT": [float(r[3]) for r in rows],
        }
    )


# 2010 tracts 100 and 200 belong to the first PUMA, tract 300 to the second.
LOOKUP = pd.DataFrame(
    {
        "GEOID_TRACT_10": ["49001000100", "49001000200", "49001000300"],
        "resstock_county": ["G4900010"] * 3,
        "resstock_puma": ["G49000001", "G49000001", "G49000002"],
    }
)


def test_share_follows_land_area_inside_a_block_and_conserves_dwellings():
    housing = pd.DataFrame(
        {
            "GEOID_BLOCK": ["490010001000001", "490010001000002", "490010002000001"],
            "H3_002N": [10, 20, 40],
        }
    )
    rel = _rel(
        [
            # block 1 of tract 100: 30 % in 2010 tract 100, 70 % in 2010 tract 300
            ("000100", "000100", "0001", 30),
            ("000300", "000100", "0001", 70),
            # block 2 of tract 100: all in 2010 tract 200 (same PUMA as 100)
            ("000200", "000100", "0002", 55),
            # the other 2020 tract: all in the second PUMA
            ("000300", "000200", "0001", 12),
        ]
    )
    out = build_tract_puma_share(housing, rel, LOOKUP, "49")
    validate_table(out, "tract_puma_share")
    first = out[out.GEOID == "49001000100"].set_index("resstock_puma")
    # PUMA 1: 10 * 0.3 + 20 = 23 dwellings; PUMA 2: 10 * 0.7 = 7 dwellings
    assert first.loc["G49000001", "occupied_units"] == pytest.approx(23.0)
    assert first.loc["G49000002", "occupied_units"] == pytest.approx(7.0)
    assert first.loc["G49000001", "share"] == pytest.approx(23 / 30)
    assert out.occupied_units.sum() == pytest.approx(housing.H3_002N.sum())
    assert out[out.GEOID == "49001000200"].share.tolist() == [1.0]


def test_blocks_without_land_are_split_equally():
    housing = pd.DataFrame({"GEOID_BLOCK": ["490010001000001"], "H3_002N": [8]})
    rel = _rel([("000100", "000100", "0001", 0), ("000300", "000100", "0001", 0)])
    out = build_tract_puma_share(housing, rel, LOOKUP, "49")
    assert out.share.tolist() == [0.5, 0.5]


def test_tracts_without_occupied_dwellings_are_left_out(caplog):
    housing = pd.DataFrame(
        {"GEOID_BLOCK": ["490010001000001", "490010002000001"], "H3_002N": [5, 0]}
    )
    rel = _rel([("000100", "000100", "0001", 9), ("000200", "000200", "0001", 9)])
    with caplog.at_level("INFO"):
        out = build_tract_puma_share(housing, rel, LOOKUP, "49")
    assert out.GEOID.tolist() == ["49001000100"]
    assert "49001000200" in caplog.text


def test_intersections_with_blocks_of_other_states_are_dropped():
    housing = pd.DataFrame({"GEOID_BLOCK": ["490010001000001"], "H3_002N": [10]})
    rel = _rel([("000100", "000100", "0001", 50), ("000100", "000100", "0001", 50)])
    rel.loc[1, "STATE_2010"] = "08"
    out = build_tract_puma_share(housing, rel, LOOKUP, "49")
    assert out.occupied_units.sum() == pytest.approx(10.0)


def test_missing_block_or_lookup_fails_instead_of_losing_dwellings():
    housing = pd.DataFrame({"GEOID_BLOCK": ["490010001000009"], "H3_002N": [3]})
    rel = _rel([("000100", "000100", "0001", 9)])
    with pytest.raises(ValueError, match="not in the relationship file"):
        build_tract_puma_share(housing, rel, LOOKUP, "49")
    housing = pd.DataFrame({"GEOID_BLOCK": ["490010001000001"], "H3_002N": [3]})
    rel = _rel([("000900", "000100", "0001", 9)])
    with pytest.raises(ValueError, match="ResStock lookup"):
        build_tract_puma_share(housing, rel, LOOKUP, "49")


def test_readers_parse_the_census_and_resstock_files(tmp_path):
    blocks = tmp_path / "h3.csv"
    blocks.write_text("GEOID_BLOCK,H3_001N,H3_002N,H3_003N\n490010001000001,12,10,2\n")
    assert read_block_housing(blocks).GEOID_BLOCK.tolist() == ["490010001000001"]

    rel = tmp_path / "rel.txt"
    rel.write_text(
        "﻿STATE_2010|COUNTY_2010|TRACT_2010|BLK_2010|AREALAND_INT\n"
        "49|001|000100|1000|55\n",
        encoding="utf-8",
    )
    got = read_block_relationship(rel)
    assert got.AREALAND_INT.tolist() == [55.0]
    assert "STATE_2010" in got.columns

    lookup = tmp_path / "lookup.csv"
    lookup.write_text(
        "nhgis_2010_tract_gisjoin,nhgis_2010_county_gisjoin,nhgis_2010_puma_gisjoin\n"
        "G4900010000100,G4900010,G49000001\n"
        "G0100010020100,G0100010,G01002100\n"
    )
    assert read_resstock_tract_lookup(lookup, "49").to_dict("records") == [
        {
            "GEOID_TRACT_10": "49001000100",
            "resstock_county": "G4900010",
            "resstock_puma": "G49000001",
        }
    ]
