"""Settings of the analysis layer.

Each ``DOWNLOAD_*`` flag turns on one download. All are ``False`` by default so
the evaluator decides what to run. Approximate size and time are given for
Utah only; the data already stored in ``data/raw`` are never overwritten.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# State of the run: USPS code and the two-digit Census (FIPS) code of the same state.
# Every analysis script reads them from here, so the pipeline runs for another state
# by changing these two lines (and the files that the downloads bring for it).
STATE = "UT"
STATE_FIPS = "49"
STATES = [STATE]

# States whose ResStock counties are used to fit the line that corrects the load of
# a dwelling from the climate of its weather station to the climate of a tract, for
# each use of energy. The state of the run is always one of them or the only one.
# Heating uses Utah and the cold states next to it, so that the coldest tracts of
# Utah fall inside the range of the fit; cooling uses the state alone because the
# cold states do not extend its range (see the method report, section 3.5). For
# another state, list states of the same climate region. Their baseline annual
# tables (about 5-12 MB each) are downloaded with DOWNLOAD_RESSTOCK.
# Year of the measured residential totals of the state (U.S. Energy Information
# Administration) that give the second reference level of the energy, and the
# households of the state in that year: American Community Survey 1-year, table
# B25002 occupied units, Utah 2018 (998,891; queried 2026-11-10). For another state
# or year, look up both. The sales workbook of the Energy Information Administration
# must include the year.
MEASURED_YEAR = 2018
# Year of the household survey (RECS) used for the comparison of end uses.
RECS_YEAR = 2020
MEASURED_YEAR_HOUSEHOLDS = 998891

CLIMATE_FIT_STATES = {
    "heating": [STATE, "CO", "WY", "ID", "MT"],
    "cooling": [STATE],
}

# Tract boundaries 2020 (TIGER/Line and cartographic) and the official 2010-2020
# relationship file. About 100 MB for Utah plus the national files; minutes.
DOWNLOAD_GEOGRAPHY = False

# American Community Survey 5-year tables by tract (needs CENSUS_API_KEY in the
# environment). Utah only: about 100 MB in total; a few minutes.
DOWNLOAD_ACS = False
ACS_YEAR = 2024
ACS_HOUSING_TABLES = [
    "B25001",
    "B25002",
    "B25004",
    "B25017",
    "B25024",
    "B25034",
    "B25040",
    "B25041",
    "B25117",
    "B25127",
    "B25042",
    "B25009",
]
ACS_JUSTICE_TABLES = [
    "B19001",
    "B17001",
    "C17002",
    "B25132",
    "B25133",
    "B25088",
    "B25091",
    "B25070",
    "B23025",
    "B15003",
    "B18101",
    "B16002",
    "B01001",
    "B01002",
    "B25014",
    "B21001",
    "B27001",
    "B11010",
    "B11003",
    "B02001",
    "B03002",
]

# 2020 Census urban/rural population and housing units by tract (Utah: a few MB).
DOWNLOAD_DECENNIAL = False

# Base temperatures (degrees Celsius) of the degree days: the lower and upper
# limits of the thermal-comfort band of the study (World Health Organization
# 2018 for the cold limit; Tham et al. 2020 for the heat limit).
HEATING_BASE_C = 18.0
COOLING_BASE_C = 26.0

# Smallest number of modeled ResStock dwellings that a level of the fallback rule
# must supply for a type of dwelling in a tract. With 1 a type is looked for in a
# wider level only when it is empty (rule of the plan); larger values are for
# sensitivity runs.
MIN_DONORS = 1

# Uncertainty of the energy of each tract: number of simulations and the seed of
# the random generator (the same seed gives the same simulations).
N_DRAWS = 200
# Census tables that give, by tenure, the heating fuel, the bedrooms and the household
# size of the dwellings of every tract, with the scope of the file that holds each
# one ("us": the table of the whole country; "state": the table of the state of the
# run). They are used by scripts 04_reweighting.py and 07_gap_by_tract.py.
COMPOSITION_TABLES = {
    "heating_fuel": ("B25117", "us"),
    "bedrooms": ("B25042", "state"),
    "household_size": ("B25009", "state"),
}
# Weights of the dwellings that also match the Census counts of the heating fuel, the
# bedrooms and the household size (script 04_reweighting.py). The composition error of
# script 07_gap_by_tract.py is the change of the mean of each tract with the Census
# shares of those variables, estimated by regression for every tract; these weights are
# kept as a control where they exist. Valid for any state with the same Census tables.
COMPOSITION_VARIANT = "type_fuel_bedrooms_size"
# Quantities (gaps between the normative and the current energy) for which a slope of
# their own against the degree days is tried as the climate correction of the tract
# (script 07_gap_by_tract.py); it is applied only where it lowers the median error of
# the known-population test. The slope is fitted with the counties of the state alone.
GAP_ADDITIVE_CANDIDATES = ("gap_heating", "gap_cooling")
# Quantities that are the sum of two others in every tract (script 07_gap_by_tract.py):
# the gap is the sum of its parts and the normative energy is the energy used today
# plus the gap, each with its own climate correction, so that they add up.
DERIVED_QUANTITIES = {
    "gap_heating": ("gap_over_heating", "gap_under_heating"),
    "gap_cooling": ("gap_over_cooling", "gap_under_cooling"),
    "norm_heating": ("cur_heating", "gap_heating"),
    "norm_cooling": ("cur_cooling", "gap_cooling"),
}
# The gap of every dwelling is split in whether it has a gap (more than this many kWh
# a year; with 1 or 50 the conclusions are the same), the gap of those that have one
# and the rest. The three columns are named with these prefixes plus the use, are not
# corrected for climate and are mixed with the mean of the state (script
# 07_gap_by_tract.py).
GAP_THRESHOLD_KWH = 10.0
SPLIT_PREFIXES = ("gap_share_", "gap_over_", "gap_under_")
# Test of units as small as a tract: every zone is cut at random in parts of about
# this many dwellings, repeated with this many seeds.
MINI_TRACT_SIZE = 40
MINI_TRACT_SEEDS = 3
SEED = 2026

# 2020 Census occupied housing units by block (table H3) and the 2010-2020 block
# relationship file, used to split each tract among the ResStock PUMAs it
# overlaps. Needs CENSUS_API_KEY. Utah: about 71,000 blocks (a few MB, 29 county
# requests, a few minutes) and a 2 MB zip.
DOWNLOAD_BLOCKS = False

# ResStock 2025 release 1 (actual 2018 weather): documentation, tract lookup,
# annual results of the listed upgrades and county weather. About 500 MB.
DOWNLOAD_RESSTOCK = False

# States used to check whether the differences found in the state of the study repeat
# elsewhere (section I of 13_validation.py): the annual results of ResStock (about
# 5-30 MB each) and the residential totals of the Energy Information Administration of
# each one. The neighbors of Utah plus states of a very different climate. A few
# minutes.
COMPARISON_STATES = [
    "CO",
    "ID",
    "WY",
    "NV",
    "AZ",
    "NM",
    "MT",
    "MN",
    "TX",
    "WA",
    "OH",
    "NY",
]
DOWNLOAD_COMPARISON_STATES = False
RESSTOCK_UPGRADES = [0, 1, 4, 11, 12, 13, 14, 15, 16, 17]
# Hourly (15-minute) series per dwelling, 6-7 MB each. Empty by default.
RESSTOCK_TIMESERIES = []

# Hourly profile of every tract (8,760 hours; all end uses by fuel, loads, indoor
# and outdoor temperatures, share of dwellings outside the comfort limits), built
# from the 15-minute series of the modeled dwellings used by the tracts. Needs the
# series in data/raw/resstock/.../timeseries_ut_upgrade0 (about 26 GB for the 4,203
# dwellings of Utah). Takes tens of minutes, about 2 GB of memory, and writes one
# Parquet file of about 1 MB per tract in data/processed/hourly_profiles_ut.
BUILD_HOURLY_PROFILES = False

# Shape of the heating and cooling of each tract in the hourly profile: with True,
# each dwelling follows the response of its load to the hourly temperature of the
# tract (station series plus the daily difference of PRISM between the tract and
# the station), rescaled to the same annual energy; with False, the shape is that
# of the weather station of the county. Needs the 365 daily PRISM files of 2018.
# It adds the daily PRISM reading of the tracts (a few minutes) and a response fit
# per dwelling to the profile build.
HOURLY_SHAPE_BY_TEMPERATURE = True

# Residential Energy Consumption Survey 2020 microdata and codebook. About 60 MB.
DOWNLOAD_RECS = False

# PRISM daily mean temperature, 4 km grid, one year: 365 files, about 1 GB; an
# hour or more because the service serves one day per request.
DOWNLOAD_PRISM = False
PRISM_YEAR = 2018

# Hourly demand of the electric balancing authorities (form EIA-930) for 2018, used
# to check the hour and the pattern of the hourly profiles. Two files, about 75 MB.
DOWNLOAD_EIA930 = False

# Measured residential consumption of the state (U.S. Energy Information
# Administration): natural gas and electricity totals, used to check the level of
# the modeled energy. Three small files, a few seconds.
DOWNLOAD_EIA_STATE_TOTALS = False

# Measured residential consumption of the state by month (U.S. Energy Information
# Administration): natural gas and electricity, used to check the seasonal pattern
# of the modeled energy. Two small files (about 2.3 MB), a few seconds.
DOWNLOAD_EIA_MONTHLY = False

# Low-Income Energy Affordability Data 2022 (U.S. Department of Energy). About
# 60 MB for Utah.
DOWNLOAD_LEAD = False

# Air-conditioning prevalence by tract: Census Bureau LACE 2023 and Ahn and
# Uejio 2025 (national files). About 15 MB.
DOWNLOAD_AC_PREVALENCE = False

# Re-simulation of the donor dwellings of ResStock with the normative thermostat
# (script 06_resimulation.py). Section A of the script (windows from the time use
# survey) always runs when its table is missing; the two flags below turn on the
# long computations. They need the program EnergyPlus 24.2 (about 180 MB, unpacked
# in ENERGYPLUS_DIR) and the files of the American Time Use Survey in data/raw/atus.
# RUN_RESIMULATION: every donor dwelling twice (own thermostat and normative) and,
# where the heat pump package applies, the package once. Utah: about 3,770 dwellings,
# 250 KB downloaded per model, about 1.5 GB written, 7 to 9 hours with 14 processes.
# RUN_RESIMULATION_VARIANTS: one dwelling in RESIMULATION_SAMPLE_EVERY with six
# variants of the thermostat; about 1.5 hours with 14 processes.
RUN_RESIMULATION = False
RUN_RESIMULATION_VARIANTS = False
RESIMULATION_SAMPLE_EVERY = 12
RESIMULATION_WORKERS = 14
ENERGYPLUS_DIR = (
    ROOT
    / "data"
    / "raw"
    / "software"
    / "energyplus_24.2.0"
    / "EnergyPlus-24.2.0-e7ecb2d53b-Windows-x86_64"
)
# Hours before each block of required hours in which the normative thermostat is
# already on (the equipment works only if the dwelling is outside the band).
PRECONDITIONING_HOURS = 3.0
# False for a state or area that keeps standard time all year (Arizona except the
# Navajo Nation, Hawaii). The rule of daylight saving time is that of the United
# States and the year of the weather is 2018.
OBSERVES_DAYLIGHT_SAVING = True

# Shares of the dwellings without cooling equipment that are taken to have an
# evaporative cooler in the sensitivity of script 08_sensitivities.py. ResStock does
# not model evaporative coolers; the 2020 national residential energy survey would
# put them in about 7 % of the dwellings of the northern Mountain states (own
# division of a table, NO VERIFICADO), and the dwellings without cooling are 13 % of
# those of Utah.
EVAPORATIVE_SHARES = (0.0, 0.25, 0.5)

# Physical check of the correction for the climate of the tract (script
# 10_climate_shift.py): one dwelling in CLIMATE_SHIFT_SAMPLE_EVERY is simulated again,
# with the baseline and the normative thermostat, with the temperature of its weather
# file moved by each shift (degrees Celsius). Two simulations per dwelling and shift;
# Utah, 4 shifts and 1 dwelling in 12: about 80 minutes with 14 processes.
RUN_CLIMATE_SHIFT = False
CLIMATE_SHIFTS = (-4.0, -2.0, 2.0, 4.0)
# Larger drops of temperature, for the tracts that are much colder than the weather
# station of their county (in Utah 8 of 716 are more than 4 degrees colder in annual
# mean, by the heating degree days, and 2 more than 6; the largest is 6.4). They are
# simulated for the same dwellings and saved apart, so the shifts above are not
# repeated: 2 shifts, about 40 minutes with 14 processes.
CLIMATE_SHIFTS_COLD = (-8.0, -6.0)
CLIMATE_SHIFT_SAMPLE_EVERY = 12
