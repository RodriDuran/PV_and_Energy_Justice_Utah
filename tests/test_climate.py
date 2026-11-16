"""Tests of the degree-day calculations with small synthetic grids."""

import doctest

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import climate
from utah_ejpi.climate import (
    build_degree_day_table,
    compare_degree_days,
    degree_days,
    prism_daily_at_points,
    puma_degree_days,
    rank_pumas_by_climate,
    resstock_weather_degree_days,
)
from utah_ejpi.utils import validate_table

tifffile = pytest.importorskip("tifffile")

# Grid of 4 rows by 5 columns, cells of 0.5 degrees, upper-left corner at
# longitude -112 and latitude 41.
WEST, NORTH, STEP = -112.0, 41.0, 0.5


def _write_grid(path, values, pixel_is_area=True):
    """Write a GeoTIFF with the tags that PRISM files carry."""
    keys = [1, 1, 0, 2, 1025, 0, 1, 1 if pixel_is_area else 2, 2048, 0, 1, 4269]
    tifffile.imwrite(
        path,
        np.asarray(values, dtype="float32"),
        extratags=[
            (33550, "d", 3, (STEP, STEP, 0.0), True),
            (33922, "d", 6, (0.0, 0.0, 0.0, WEST, NORTH, 0.0), True),
            (34735, "H", len(keys), keys, True),
        ],
    )


@pytest.fixture
def grid_files(tmp_path):
    """Three daily grids; the value of cell (row, col) on day d is 10 * d + row."""
    paths = []
    for day in range(3):
        values = np.add.outer(np.arange(4.0), np.zeros(5)) + 10.0 * day
        path = tmp_path / f"day{day}.tif"
        _write_grid(path, values)
        paths.append(path)
    return paths


def test_docstring_examples_run():
    assert doctest.testmod(climate).failed == 0


def test_degree_days_known_answer_and_check_of_bases():
    hdd, cdd = degree_days(np.array([[10.0, 30.0], [20.0, 30.0]]), 18.0, 24.0)
    assert hdd.tolist() == [8.0, 0.0]
    assert cdd.tolist() == [0.0, 12.0]
    with pytest.raises(ValueError, match="above the cooling base"):
        degree_days([1.0], 25.0, 20.0)


def test_prism_values_are_read_from_the_cell_that_holds_each_point(grid_files):
    # first point: row 0 (latitude 40.9), second: row 3 (latitude 39.2)
    out = prism_daily_at_points(grid_files, [-111.8, -109.7], [40.9, 39.2])
    assert out.shape == (3, 2)
    assert out[:, 0].tolist() == [0.0, 10.0, 20.0]
    assert out[:, 1].tolist() == [3.0, 13.0, 23.0]


def test_points_outside_the_grid_or_without_data_fail(grid_files, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prism_daily_at_points(grid_files, [-120.0], [40.0])
    values = np.zeros((4, 5))
    values[0, 0] = -9999.0
    holes = [tmp_path / "holes.tif"]
    _write_grid(holes[0], values)
    with pytest.raises(ValueError, match="without data"):
        prism_daily_at_points(holes, [-111.9], [40.9])


def test_files_must_share_the_grid_and_be_pixel_is_area(tmp_path):
    first, second, third = (tmp_path / f"{n}.tif" for n in "abc")
    _write_grid(first, np.zeros((4, 5)))
    _write_grid(second, np.zeros((3, 5)))
    with pytest.raises(ValueError, match="same grid"):
        prism_daily_at_points([first, second], [-111.9], [40.9])
    _write_grid(third, np.zeros((4, 5)), pixel_is_area=False)
    with pytest.raises(ValueError, match="pixel is area"):
        prism_daily_at_points([third], [-111.9], [40.9])


def test_degree_day_table_follows_the_contract(grid_files):
    points = pd.DataFrame(
        {
            "id": ["49001000100", "49001000200"],
            "lon": [-111.8, -109.7],
            "lat": [40.9, 39.2],
        }
    )
    table = build_degree_day_table(grid_files, points, "tract", 18.0, 24.0)
    validate_table(table, "climate_degree_days")
    # first point: daily 0, 10, 20 -> heating 18 + 8 + 0, cooling 0
    assert table.loc[0, "hdd"] == pytest.approx(26.0)
    assert table.loc[0, "cdd"] == pytest.approx(0.0)
    # second point: daily 3, 13, 23 -> heating 15 + 5 + 0, no cooling day
    assert table.loc[1, "hdd"] == pytest.approx(20.0)
    assert table.loc[1, "cdd"] == pytest.approx(0.0)
    assert set(table["hdd_base_c"]) == {18.0}


def test_resstock_weather_file_degree_days(tmp_path):
    path = tmp_path / "G4900010_2018.csv"
    pd.DataFrame({"Dry Bulb Temperature [°C]": np.full(8760, 10.0)}).to_csv(
        path, index=False, encoding="latin-1"
    )
    hdd, cdd = resstock_weather_degree_days(path, 18.0, 24.0)
    assert hdd == pytest.approx(8.0 * 365)
    assert cdd == 0.0
    pd.DataFrame({"Dry Bulb Temperature [°C]": [1.0] * 10}).to_csv(
        path, index=False, encoding="latin-1"
    )
    with pytest.raises(ValueError, match="hourly weather file"):
        resstock_weather_degree_days(path, 18.0, 24.0)


def test_compare_degree_days_summarizes_agreement():
    prism = pd.DataFrame(
        {"id": ["a", "b", "c"], "hdd": [100.0, 200.0, 300.0], "cdd": [5.0, 6.0, 7.0]}
    )
    files = pd.DataFrame(
        {"id": ["a", "b", "c"], "hdd": [90.0, 190.0, 290.0], "cdd": [5.0, 6.0, 8.0]}
    )
    out = compare_degree_days(prism, files)
    assert out["hdd"]["mean_difference"] == pytest.approx(10.0)
    assert out["hdd"]["correlation"] == pytest.approx(1.0)
    assert out["cdd"]["max_abs_difference"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same stations"):
        compare_degree_days(prism, files.iloc[:2])


def _tract_dd():
    return pd.DataFrame(
        {
            "id": ["49001000100", "49001000200", "49003000100"],
            "kind": "tract",
            "hdd": [3000.0, 2000.0, 1000.0],
            "cdd": [100.0, 200.0, 400.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 24.0,
        }
    )


def test_puma_climate_is_the_dwelling_weighted_mean_of_its_tracts(tract_puma_share):
    # PUMA 1 holds tracts 100 and 200 entirely (1000 and 1500 dwellings) and
    # 100 dwellings of the third tract; PUMA 2 holds 900 dwellings of the third
    share = tract_puma_share
    out = puma_degree_days(_tract_dd(), share).set_index("id")
    validate_table(out.reset_index(), "climate_degree_days")
    expected = (3000 * 1000 + 2000 * 1500 + 1000 * 100) / 2600
    assert out.loc["G49000001", "hdd"] == pytest.approx(expected)
    assert out.loc["G49000002", "hdd"] == pytest.approx(1000.0)
    assert (out["kind"] == "puma").all()


def test_ranking_orders_pumas_by_scaled_distance():
    pumas = pd.DataFrame(
        {
            "id": ["P1", "P2", "P3"],
            "kind": "puma",
            "hdd": [3000.0, 2000.0, 1000.0],
            "cdd": [100.0, 150.0, 400.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 24.0,
        }
    )
    ranks = rank_pumas_by_climate(_tract_dd(), pumas)
    validate_table(ranks, "puma_climate_rank")
    first = ranks[ranks.GEOID == "49001000100"].set_index("resstock_puma")
    assert first.loc["P1", "rank"] == 1
    assert first.loc["P3", "rank"] == 3
    assert first.loc["P1", "distance"] == pytest.approx(0.0)
    third = ranks[ranks.GEOID == "49003000100"].sort_values("rank")
    assert third.resstock_puma.tolist() == ["P3", "P2", "P1"]
    # the scale is the standard deviation among the PUMAs, so a heating gap of
    # one standard deviation counts as much as a cooling gap of one
    sd_h = np.std([3000.0, 2000.0, 1000.0])
    sd_c = np.std([100.0, 150.0, 400.0])
    d_p2 = np.hypot((1000.0 - 2000.0) / sd_h, (400.0 - 150.0) / sd_c)
    assert third.set_index("resstock_puma").loc["P2", "distance"] == pytest.approx(d_p2)


def test_tract_points_fall_inside_irregular_tracts(tmp_path):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    # an L-shaped tract whose centroid lies outside it
    shape = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])
    frame = geopandas.GeoDataFrame(
        {"GEOID": ["49001000100"]}, geometry=[shape], crs="EPSG:4269"
    )
    assert not shape.contains(shape.centroid)
    path = tmp_path / "tract.gpkg"
    frame.to_file(path, driver="GPKG")
    out = climate.tract_points(path)
    point = geopandas.points_from_xy(out["lon"], out["lat"])[0]
    assert shape.contains(point)
    assert out["id"].tolist() == ["49001000100"]
