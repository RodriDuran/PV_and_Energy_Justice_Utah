"""Small synthetic tables that satisfy the table contracts of the package.

They are invented (not real tracts) and have known answers: the dwellings of
each tract add up to the figures written here.
"""

import numpy as np
import pandas as pd
import pytest

TYPES = [
    ("Owner", "1980 to 1999", "1"),
    ("Owner", "2000 to 2019", "1"),
    ("Renter", "1960 to 1979", "5 to 19"),
]


@pytest.fixture
def tract_master() -> pd.DataFrame:
    """Three tracts: two in one PUMA, one split in the data of the shares."""
    return pd.DataFrame(
        {
            "GEOID": ["49001000100", "49001000200", "49003000100"],
            "county_fips": ["49001", "49001", "49003"],
            "resstock_puma": ["G49000001", "G49000001", "G49000002"],
            "resstock_county": ["G4900010", "G4900010", "G4900030"],
            "climate_zone_ba": ["Cold", "Cold", "Mixed-Dry"],
            "climate_zone_iecc": ["5B", "5B", "4B"],
        }
    )


@pytest.fixture
def tract_puma_share() -> pd.DataFrame:
    """Shares of occupied dwellings by PUMA; the third tract is split."""
    return pd.DataFrame(
        {
            "GEOID": ["49001000100", "49001000200", "49003000100", "49003000100"],
            "resstock_puma": ["G49000001", "G49000001", "G49000002", "G49000001"],
            "occupied_units": [1000.0, 1500.0, 900.0, 100.0],
            "share": [1.0, 1.0, 0.9, 0.1],
        }
    )


@pytest.fixture
def resstock_dwellings() -> pd.DataFrame:
    """Twelve modeled dwellings, four per type, in two PUMAs and two counties."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(12):
        ten, vin, typ = TYPES[i % 3]
        puma = "G49000001" if i < 8 else "G49000002"
        rows.append(
            {
                "building_id": i + 1,
                "upgrade": 0,
                "applicable": True,
                "county": "G4900010" if i < 8 else "G4900030",
                "puma": puma,
                "occupied": True,
                "tenure": ten,
                "vintage": vin,
                "building_type": typ,
                "weight": 253.9,
                "heating_load_kbtu": float(rng.uniform(20000, 90000)),
                "cooling_load_kbtu": float(rng.uniform(0, 30000)),
                "unmet_heating_hours": float(rng.integers(0, 50)),
                "unmet_cooling_hours": float(rng.integers(0, 50)),
                "occupants": float(rng.integers(1, 6)),
                "weather_lat": 40.8,
                "weather_lon": -111.9,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def acs_type_counts(tract_master: pd.DataFrame) -> pd.DataFrame:
    """Build counts of dwellings by type for each tract, with margins of error."""
    rows = []
    for k, geoid in enumerate(tract_master["GEOID"]):
        for j, (ten, vin, typ) in enumerate(TYPES):
            est = 100.0 * (k + 1) + 20.0 * j
            rows.append(
                {
                    "GEOID": geoid,
                    "tenure": ten,
                    "vintage": vin,
                    "building_type": typ,
                    "estimate": est,
                    "moe": 0.3 * est,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def resstock_raw() -> pd.DataFrame:
    """Build a table with the ResStock annual columns the package reads."""
    from utah_ejpi.resstock import FUEL_COLUMNS, SOURCE_COLUMNS

    n = 4
    raw = pd.DataFrame(
        {
            "bldg_id": [101, 102, 103, 104],
            "upgrade": [0] * n,
            "applicability": [True] * n,
            "in.county": ["G4900010", "G4900010", "G4900030", "G4900030"],
            "in.puma": ["G49000001", "G49000001", "G49000002", "G49000002"],
            "in.vacancy_status": ["Occupied", "Occupied", "Occupied", "Vacant"],
            "in.tenure": ["Owner", "Renter", "Owner", "Not Available"],
            "in.vintage": ["1980s", "1950s", "2010s", "<1940"],
            "in.geometry_building_type_acs": [
                "Single-Family Detached",
                "3 or 4 Unit",
                "Mobile Home",
                "50 or more Unit",
            ],
            "in.heating_fuel": ["Natural Gas", "Electricity", "Propane", "None"],
            "in.bedrooms": ["3", "2", "4", "1"],
            "in.geometry_floor_area": ["1500-1999", "750-999", "2500-2999", "500-749"],
            "in.hvac_cooling_type": ["Central AC", "None", "Room AC", "None"],
            "in.occupants": ["3", "10+", "1", "0"],
            "in.heating_setpoint": ["68F", "70F", "65F", "55F"],
            "in.cooling_setpoint": ["78F", "75F", "80F", "70F"],
            "in.weather_file_latitude": [40.8, 40.8, 37.1, 37.1],
            "in.weather_file_longitude": [-111.9, -111.9, -113.5, -113.5],
            "weight": [253.9] * n,
            "out.load.heating.energy_delivered..kbtu": [50000.0, 30000.0, 0.0, 9.0],
            "out.load.cooling.energy_delivered..kbtu": [1000.0, 0.0, 20000.0, 5.0],
            "out.unmet_hours.heating..hr": [0.0, 12.0, 0.0, 0.0],
            "out.unmet_hours.cooling..hr": [3.0, 0.0, 0.0, 0.0],
        }
    )
    for cols in FUEL_COLUMNS.values():
        for c in cols:
            raw[c] = 0.0
    raw[FUEL_COLUMNS["heating_kwh_natural_gas"][0]] = [9000.0, 0.0, 0.0, 0.0]
    raw[FUEL_COLUMNS["heating_kwh_natural_gas"][1]] = [1.0, 0.0, 0.0, 0.0]
    raw[FUEL_COLUMNS["heating_kwh_electricity"][0]] = [0.0, 4000.0, 0.0, 0.0]
    raw[FUEL_COLUMNS["heating_kwh_electricity"][3]] = [0.0, 500.0, 0.0, 0.0]
    assert set(SOURCE_COLUMNS) <= set(raw.columns)
    return raw
