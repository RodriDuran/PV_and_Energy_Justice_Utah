"""Tests of the hourly profiles of the tracts."""

import doctest

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import hourly as hr
from utah_ejpi.energy_requirement import DegreeDaySensitivity
from utah_ejpi.resstock import FUEL_COLUMNS

STEPS = 35040
SHIFT = 2  # hours the clock of the test series runs ahead of local standard time
HEAT_KWH = "out.electricity.heating.energy_consumption..kwh"
COOL_KWH = "out.electricity.cooling.energy_consumption..kwh"
LIGHT_KWH = "out.electricity.lighting_interior.energy_consumption..kwh"
GAS_KWH = "out.natural_gas.heating.energy_consumption..kwh"
PV_KWH = "out.electricity.pv.energy_consumption..kwh"
HEAT_LOAD = "out.load.heating.energy_delivered..kbtu"
INDOOR = "out.indoor_temperature.conditioned_space..c"
OUTDOOR = "out.outdoor_air_drybulb_temp..c"
OCCUPANTS = "out.schedules.occupants"

LINE = DegreeDaySensitivity("heating", 0.0, 1.0, np.eye(2), 10, 3, 1.0, 3.0)
COOL_LINE = DegreeDaySensitivity("cooling", 0.0, 1.0, np.eye(2), 10, 2, 0.1, 0.3)


def test_docstring_examples_run():
    assert doctest.testmod(hr).failed == 0


def _write_series(path, heat, cool, light, gas, indoor, pv=0.0, away_hours=0):
    """One year of 15 minutes with constant hourly values (per hour, not per step)."""
    stamps = pd.date_range("2018-01-01 00:15", periods=STEPS, freq="15min")
    columns = {"timestamp": stamps}
    for names in FUEL_COLUMNS.values():
        for name in names:
            # like the real series, which lack the backup of propane and fuel oil
            lacking = "hp_bkup" in name and ("propane" in name or "fuel_oil" in name)
            if not lacking:
                columns[name] = np.zeros(STEPS)
    columns[HEAT_KWH] = np.full(STEPS, heat / 4)
    columns[COOL_KWH] = np.full(STEPS, cool / 4)
    columns[LIGHT_KWH] = np.full(STEPS, light / 4)
    columns[GAS_KWH] = np.full(STEPS, gas / 4)
    columns[PV_KWH] = np.full(STEPS, pv / 4)
    columns[HEAT_LOAD] = np.full(STEPS, heat * 3.0 / 4)
    columns[INDOOR] = np.full(STEPS, indoor)
    columns[OUTDOOR] = np.full(STEPS, 5.0)
    # nobody at home during the first `away_hours` hours of the year
    at_home = np.ones(STEPS)
    at_home[: away_hours * 4] = 0.0
    columns[OCCUPANTS] = at_home
    columns["out.electricity.total.energy_consumption..kwh"] = np.full(STEPS, 99.0)
    columns["out.emissions.total.aer_midcase_avg..co2e_kg"] = np.full(STEPS, 1.0)
    pd.DataFrame(columns).to_parquet(path, index=False)


def _inputs():
    geoid = "49001000100"
    weights = pd.DataFrame(
        {
            "GEOID": [geoid, geoid],
            "tenure": ["Owner", "Renter"],
            "vintage": ["1980 to 1999", "1960 to 1979"],
            "building_type": ["1", "5 to 19"],
            "weight": [100.0, 50.0],
        }
    )
    donors = pd.DataFrame(
        {
            "GEOID": [geoid] * 3,
            "tenure": ["Owner", "Owner", "Renter"],
            "vintage": ["1980 to 1999", "1980 to 1999", "1960 to 1979"],
            "building_type": ["1", "1", "5 to 19"],
            "building_id": [1, 2, 3],
            "donor_weight": [0.5, 0.5, 1.0],
            "source_level": "puma",
        }
    )
    dwellings = pd.DataFrame(
        {
            "building_id": [1, 2, 3],
            "upgrade": 0,
            "county": ["C1", "C1", "C2"],
            "occupants": [2.0, 4.0, 1.0],
            "heating_load_kbtu": [100.0, 200.0, 60.0],
            "cooling_load_kbtu": [10.0, 0.0, 20.0],
        }
    )
    tract = pd.DataFrame(
        {
            "id": [geoid],
            "kind": "tract",
            "hdd": [4000.0],
            "cdd": [200.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    stations = pd.DataFrame(
        {
            "id": ["C1", "C2"],
            "kind": "station",
            "hdd": [2000.0, 4000.0],
            "cdd": [100.0, 200.0],
            "hdd_base_c": 18.0,
            "cdd_base_c": 26.0,
        }
    )
    return geoid, weights, donors, dwellings, tract, stations


def _coefficients():
    geoid, weights, donors, dwellings, tract, stations = _inputs()
    return hr.donor_coefficients(
        weights, donors, dwellings, tract, stations, LINE, COOL_LINE
    )


def test_coefficients_are_the_share_of_the_tract_that_each_dwelling_stands_for():
    coef = _coefficients().set_index("building_id")
    assert coef["coef"].tolist() == pytest.approx([1 / 3] * 3)
    assert coef["coef"].sum() == pytest.approx(1.0)
    # heating factors: tract 4000 over station 2000 is 2 (county C1) and 1 (county C2)
    assert coef["coef_heating"].tolist() == pytest.approx(
        [50 * 2 / 150, 50 * 2 / 150, 50 * 1 / 150]
    )
    # cooling factors: 200 over 100 is 2 for C1 and 1 for C2
    assert coef["coef_cooling"].tolist() == pytest.approx(
        [50 * 2 / 150, 50 * 2 / 150, 50 * 1 / 150]
    )


def test_a_dwelling_in_two_types_adds_its_coefficients():
    geoid, weights, donors, dwellings, tract, stations = _inputs()
    extra = donors[donors["building_id"] == 3].assign(
        tenure="Owner", vintage="1980 to 1999", building_type="1", donor_weight=0.0
    )
    both = pd.concat([donors, extra], ignore_index=True)
    owner = both["tenure"].eq("Owner")
    both.loc[owner & both["building_id"].eq(3), "donor_weight"] = 0.5
    both.loc[owner & both["building_id"].isin([1, 2]), "donor_weight"] = 0.25
    coef = hr.donor_coefficients(
        weights, both, dwellings, tract, stations, LINE, COOL_LINE
    ).set_index("building_id")
    assert coef.loc[3, "coef"] == pytest.approx((100 * 0.5 + 50 * 1.0) / 150)
    assert coef["coef"].sum() == pytest.approx(1.0)


def test_hourly_series_adds_the_energy_and_averages_the_temperature(tmp_path):
    path = tmp_path / "1-0.parquet"
    _write_series(
        path, heat=2.0, cool=0.0, light=1.0, gas=0.0, indoor=17.0, away_hours=10
    )
    names = pd.read_parquet(path).columns.tolist()
    components = hr.hourly_components(names)
    assert HEAT_KWH in components
    assert "out.electricity.total.energy_consumption..kwh" not in components
    assert not any("emissions" in c for c in components)
    series = hr.read_dwelling_hourly(path, components, SHIFT)
    assert len(series) == 8760
    assert series["electricity_heating_kwh"].tolist() == pytest.approx([2.0] * 8760)
    assert series["indoor_temperature_conditioned_space_c"].iloc[0] == 17.0
    assert series["share_below_comfort"].eq(1.0).all()
    assert series["share_above_comfort"].eq(0.0).all()
    # nobody is at home in the first ten hours, so those hours are not a deficit
    # the clock of the files runs 2 hours ahead of the local standard time, so the
    # first two hours of the files are the last two of the local year
    assert series["share_occupied"].tolist() == [0.0] * 8 + [1.0] * 8750 + [0.0] * 2
    assert series["share_below_comfort_occupied"].sum() == 8750
    assert series["share_above_comfort_occupied"].sum() == 0
    assert series["schedules_occupants"].iloc[0] == 0.0
    unshifted = hr.read_dwelling_hourly(path, components, 0)
    assert unshifted["share_occupied"].tolist() == [0.0] * 10 + [1.0] * 8750


def test_irregular_or_short_series_fail(tmp_path):
    path = tmp_path / "1-0.parquet"
    _write_series(path, heat=1.0, cool=0.0, light=0.0, gas=0.0, indoor=20.0)
    frame = pd.read_parquet(path)
    components = hr.hourly_components(frame.columns.tolist())
    frame.iloc[:-4].to_parquet(tmp_path / "short-0.parquet", index=False)
    with pytest.raises(ValueError, match="expected 35040"):
        hr.read_dwelling_hourly(tmp_path / "short-0.parquet", components, SHIFT)
    frame.loc[100, "timestamp"] += pd.Timedelta(minutes=5)
    frame.to_parquet(tmp_path / "gap-0.parquet", index=False)
    with pytest.raises(ValueError, match="regular"):
        hr.read_dwelling_hourly(tmp_path / "gap-0.parquet", components, SHIFT)
    with pytest.raises(FileNotFoundError):
        hr.read_dwelling_hourly(tmp_path / "missing.parquet", components, SHIFT)


def test_tract_profile_is_the_weighted_sum_and_only_heating_gets_the_factor(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    heat = {1: 2.0, 2: 4.0, 3: 6.0}
    indoor = {1: 17.0, 2: 20.0, 3: 27.0}
    for b in (1, 2, 3):
        _write_series(
            series / f"{b}-0.parquet",
            heat=heat[b],
            cool=1.0,
            light=3.0,
            gas=heat[b] * 10,
            indoor=indoor[b],
            pv=-1.0 if b == 1 else 0.0,
            away_hours=5 if b == 1 else 0,
        )
    coef = _coefficients()
    out = tmp_path / "profiles"
    index = hr.build_tract_profiles(coef, series, out, SHIFT)
    assert index["GEOID"].tolist() == ["49001000100"]
    assert index["dwellings_used"].iloc[0] == 3
    assert index["coef_sum"].iloc[0] == pytest.approx(1.0)

    profile = hr.read_tract_profile(out, "49001000100")
    assert len(profile) == 8760
    assert profile["timestamp"].iloc[0] == pd.Timestamp("2018-01-01 00:00")
    assert profile["timestamp"].iloc[-1] == pd.Timestamp("2018-12-31 23:00")
    # heating carries the climate factor, lighting does not
    heating = (50 * 2 * 2.0 + 50 * 2 * 4.0 + 50 * 1 * 6.0) / 150
    assert profile["electricity_heating_kwh"].iloc[0] == pytest.approx(heating)
    assert profile["natural_gas_heating_kwh"].iloc[0] == pytest.approx(heating * 10)
    assert profile["electricity_lighting_interior_kwh"].iloc[0] == pytest.approx(3.0)
    # cooling is 1 kWh in every dwelling, with factors 2, 2 and 1
    assert profile["electricity_cooling_kwh"].iloc[0] == pytest.approx(
        (50 * 2 + 50 * 2 + 50 * 1) / 150
    )
    # the share of dwellings out of the comfort limits uses the plain coefficients
    assert profile["share_below_comfort"].iloc[0] == pytest.approx(1 / 3)
    assert profile["share_above_comfort"].iloc[0] == pytest.approx(1 / 3)
    # dwelling 1 (17 degrees) is away in the first five hours: no deficit there
    assert profile["share_occupied"].iloc[0] == pytest.approx(2 / 3)
    assert profile["share_below_comfort_occupied"].iloc[0] == 0.0
    assert profile["share_below_comfort_occupied"].iloc[5] == pytest.approx(1 / 3)
    assert profile["share_above_comfort_occupied"].iloc[0] == pytest.approx(1 / 3)
    assert profile["indoor_temperature_conditioned_space_c"].iloc[0] == pytest.approx(
        (17 + 20 + 27) / 3
    )

    full = hr.fuel_use_profiles(profile)
    # heating adds all the fuels (electricity and natural gas here)
    assert full["heating_kwh"].iloc[0] == pytest.approx(heating * 11)
    assert full["heating_kwh_natural_gas"].iloc[0] == pytest.approx(heating * 10)
    assert full["electricity_total_kwh"].iloc[0] == pytest.approx(
        heating + full["electricity_cooling_kwh"].iloc[0] + 3.0
    )
    assert full["electricity_net_kwh"].iloc[0] == pytest.approx(
        full["electricity_total_kwh"].iloc[0] - 1 / 3
    )
    assert full["natural_gas_total_kwh"].iloc[0] == pytest.approx(heating * 10)

    # the sum over the year is the annual energy of the tract
    assert profile["electricity_heating_kwh"].sum() == pytest.approx(heating * 8760)
    long = hr.read_profiles(out, ["electricity_heating_kwh"], ["49001000100"])
    assert long.columns.tolist() == [
        "GEOID",
        "timestamp",
        "electricity_heating_kwh",
    ]
    assert len(long) == 8760
    assert hr.read_profiles(out, ["electricity_heating_kwh"], ["none"]).empty
    with pytest.raises(FileNotFoundError, match="No hourly profile"):
        hr.read_tract_profile(out, "49999999999")


def test_missing_series_file_fails(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    with pytest.raises(FileNotFoundError):
        hr.build_tract_profiles(_coefficients(), series, tmp_path / "out", SHIFT)


def _seasonal_temperature():
    """Hourly outdoor temperature with a yearly and a daily cycle (local time)."""
    hours = np.arange(8760)
    day, hod = hours // 24, hours % 24
    return (
        8
        + 12 * np.sin(2 * np.pi * (day - 110) / 365)
        + 5 * np.sin(2 * np.pi * (hod - 9) / 24)
    )


def _write_seasonal_series(path, temperature):
    """Series whose heating load follows 2 * max(15 - T, 0) kBtu per hour."""
    load = 2.0 * np.maximum(15.0 - temperature, 0)

    def quarter(hourly, divide=True):
        in_file_clock = np.roll(hourly, SHIFT)
        return np.repeat(in_file_clock, 4) / (4 if divide else 1)

    stamps = pd.date_range("2018-01-01 00:15", periods=STEPS, freq="15min")
    columns = {"timestamp": stamps}
    for names in FUEL_COLUMNS.values():
        for name in names:
            lacking = "hp_bkup" in name and ("propane" in name or "fuel_oil" in name)
            if not lacking:
                columns[name] = np.zeros(STEPS)
    columns[HEAT_KWH] = quarter(load / 3.0)
    columns[HEAT_LOAD] = quarter(load)
    columns[INDOOR] = quarter(np.full(8760, 20.0), divide=False)
    columns[OUTDOOR] = quarter(temperature, divide=False)
    columns[OCCUPANTS] = np.ones(STEPS)
    pd.DataFrame(columns).to_parquet(path, index=False)


def test_reshape_load_keeps_the_pattern_of_the_dwelling_and_moves_with_temperature():
    temperature = _seasonal_temperature()
    schedule = np.where(np.arange(8760) % 24 < 12, 1.0, 0.0)
    load = 2.0 * schedule * np.maximum(15.0 - temperature, 0)
    fit = hr.fit_temperature_response(load, temperature, "heating")
    same = hr.reshape_load(load, fit, temperature, temperature)
    assert same == pytest.approx(load)
    colder = hr.reshape_load(load, fit, temperature, temperature - 3.0)
    assert colder.sum() > load.sum()
    assert colder[schedule == 0] == pytest.approx(0.0, abs=1e-6)
    two = hr.reshape_load(load, fit, temperature, np.vstack([temperature - 3.0] * 2))
    assert two.shape == (2, 8760)
    assert (two >= 0).all()


def test_temperature_response_is_recovered_and_applied_to_other_temperatures():
    temperature = _seasonal_temperature()
    load = 2.0 * np.maximum(15.0 - temperature, 0)
    fit = hr.fit_temperature_response(load, temperature, "heating")
    assert fit.base_c == 15.0
    assert fit.r2 == pytest.approx(1.0)
    assert fit.coefficients == pytest.approx(np.full(24, 2.0))
    colder = hr.temperature_response_shape(fit, temperature - 5.0)
    assert colder.sum() > load.sum()
    assert hr.fit_temperature_response(np.zeros(8760), temperature, "heating") is None
    with pytest.raises(ValueError, match="use must be"):
        hr.fit_temperature_response(load, temperature, "lighting")
    with pytest.raises(ValueError, match="8760 hours"):
        hr.fit_temperature_response(load[:10], temperature[:10], "heating")
    with pytest.raises(ValueError, match="Expected 8760 hours"):
        hr.tract_hourly_temperature(np.zeros(5), np.zeros(365), np.zeros(365))
    with pytest.raises(ValueError, match="need 365 days"):
        hr.tract_hourly_temperature(np.zeros(8760), np.zeros(365), np.zeros(10))


def test_shape_follows_the_temperature_of_the_tract_and_keeps_the_annual_sum(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    temperature = _seasonal_temperature()
    _write_seasonal_series(series / "1-0.parquet", temperature)
    daily = temperature.reshape(365, 24).mean(axis=1)
    tracts = pd.DataFrame([daily, daily - 8.0], index=["49001000100", "49001000200"])
    stations = pd.DataFrame([daily], index=["G4900010"])
    coef = pd.DataFrame(
        {
            "GEOID": ["49001000100", "49001000200"],
            "building_id": [1, 1],
            "coef": 1.0,
            "coef_heating": 1.0,
            "coef_cooling": 1.0,
        }
    )
    plain = hr.build_tract_profiles(coef, series, tmp_path / "plain", SHIFT)
    shaped = hr.build_tract_profiles(
        coef,
        series,
        tmp_path / "shaped",
        SHIFT,
        tract_daily_temperature=tracts,
        station_daily_temperature=stations,
        dwelling_station={1: "G4900010"},
    )
    assert shaped["GEOID"].tolist() == plain["GEOID"].tolist()
    name = "electricity_heating_kwh"
    original = hr.read_tract_profile(tmp_path / "plain", "49001000100")[name]
    same = hr.read_tract_profile(tmp_path / "shaped", "49001000100")[name]
    colder = hr.read_tract_profile(tmp_path / "shaped", "49001000200")[name]
    # the tract with the climate of the station keeps its profile; the annual sum
    # is unchanged in both
    assert same.to_numpy() == pytest.approx(original.to_numpy(), abs=1e-4)
    assert colder.sum() == pytest.approx(original.sum(), rel=1e-4)
    # the colder tract heats in more hours, so the peak is lower relative to the mean
    assert (colder > 0).sum() > (original > 0).sum()
    assert colder.max() / colder.mean() < original.max() / original.mean()
    # uses outside heating and cooling keep the temperature of the station
    outdoor = "outdoor_air_drybulb_temp_c"
    colder_t = hr.read_tract_profile(tmp_path / "shaped", "49001000200")[outdoor]
    assert colder_t.to_numpy() == pytest.approx(temperature, abs=1e-4)


def test_shape_needs_all_its_arguments_and_the_temperatures(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    temperature = _seasonal_temperature()
    _write_seasonal_series(series / "1-0.parquet", temperature)
    daily = temperature.reshape(365, 24).mean(axis=1)
    coef = pd.DataFrame(
        {
            "GEOID": ["49001000100"],
            "building_id": [1],
            "coef": 1.0,
            "coef_heating": 1.0,
            "coef_cooling": 1.0,
        }
    )
    tracts = pd.DataFrame([daily], index=["49001000100"])
    stations = pd.DataFrame([daily], index=["G4900010"])
    with pytest.raises(ValueError, match="The shape needs"):
        hr.build_tract_profiles(
            coef, series, tmp_path / "a", SHIFT, tract_daily_temperature=tracts
        )
    kwargs = dict(
        tract_daily_temperature=tracts,
        station_daily_temperature=stations,
        dwelling_station={1: "G4900010"},
    )
    with pytest.raises(ValueError, match="No daily temperatures for 1 tracts"):
        hr.build_tract_profiles(
            coef.assign(GEOID="49001000300"), series, tmp_path / "b", SHIFT, **kwargs
        )
    with pytest.raises(ValueError, match="No daily temperatures for stations"):
        hr.build_tract_profiles(
            coef,
            series,
            tmp_path / "c",
            SHIFT,
            **{**kwargs, "dwelling_station": {1: "G9999999"}},
        )


def test_dwelling_station_can_be_a_series(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    temperature = _seasonal_temperature()
    _write_seasonal_series(series / "1-0.parquet", temperature)
    daily = temperature.reshape(365, 24).mean(axis=1)
    coef = pd.DataFrame(
        {
            "GEOID": ["49001000100"],
            "building_id": [1],
            "coef": 1.0,
            "coef_heating": 1.0,
            "coef_cooling": 1.0,
        }
    )
    index = hr.build_tract_profiles(
        coef,
        series,
        tmp_path / "out",
        SHIFT,
        tract_daily_temperature=pd.DataFrame([daily], index=["49001000100"]),
        station_daily_temperature=pd.DataFrame([daily], index=["G4900010"]),
        dwelling_station=pd.Series({1: "G4900010"}),
    )
    assert index["dwellings_used"].iloc[0] == 1


def test_columns_of_missing_values_are_zero_loads(tmp_path):
    path = tmp_path / "1-0.parquet"
    _write_series(path, heat=1.0, cool=0.0, light=0.0, gas=0.0, indoor=20.0)
    frame = pd.read_parquet(path)
    frame["out.load.heating_heat_pump_backup..kbtu"] = np.nan
    frame.to_parquet(path, index=False)
    components = hr.hourly_components(frame.columns.tolist())
    series = hr.read_dwelling_hourly(path, components, SHIFT)
    assert series["load_heating_heat_pump_backup_kbtu"].eq(0.0).all()
    assert not series.isna().any().any()


def test_clock_shift_is_the_rotation_that_matches_the_weather_file():
    temperature = _seasonal_temperature()
    for shift in (0, 1, 3):
        found = hr.estimate_clock_shift_hours(np.roll(temperature, shift), temperature)
        assert found == shift
    noise = np.random.default_rng(0).normal(size=8760)
    with pytest.raises(ValueError, match="correlation"):
        hr.estimate_clock_shift_hours(noise, temperature)
    with pytest.raises(ValueError, match="8760"):
        hr.estimate_clock_shift_hours(temperature[:100], temperature[:100])


def test_county_clock_shifts_read_one_dwelling_per_county(tmp_path):
    series, weather = tmp_path / "series", tmp_path / "weather"
    series.mkdir()
    weather.mkdir()
    temperature = _seasonal_temperature()
    counties = {1: "G4900010", 2: "G4900010", 3: "G0600010"}
    shifts = {"G4900010": 2, "G0600010": 3}
    for building, county in counties.items():
        # the file clock runs shifts[county] hours ahead of the weather file
        _write_seasonal_series(
            series / f"{building}-0.parquet",
            np.roll(temperature, shifts[county] - SHIFT),
        )
    for county in shifts:
        pd.DataFrame({"Dry Bulb Temperature [°C]": temperature}).to_csv(
            weather / f"{county}_2018.csv", index=False
        )
    assert hr.county_clock_shifts(counties, series, weather) == shifts
    with pytest.raises(FileNotFoundError, match="Weather file"):
        hr.county_clock_shifts({1: "G9999999"}, series, weather)


def test_each_dwelling_can_have_its_own_clock_shift(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    for b in (1, 2, 3):
        _write_series(
            series / f"{b}-0.parquet",
            heat=1.0,
            cool=0.0,
            light=1.0,
            gas=0.0,
            indoor=20.0,
            away_hours=10,
        )
    coef = _coefficients()
    hr.build_tract_profiles(coef, series, tmp_path / "int", 2)
    hr.build_tract_profiles(coef, series, tmp_path / "map", {1: 2, 2: 2, 3: 2})
    one = hr.read_tract_profile(tmp_path / "int", "49001000100")
    two = hr.read_tract_profile(tmp_path / "map", "49001000100")
    pd.testing.assert_frame_equal(one, two)
    with pytest.raises(ValueError, match="No clock shift"):
        hr.build_tract_profiles(coef, series, tmp_path / "bad", {1: 2})
