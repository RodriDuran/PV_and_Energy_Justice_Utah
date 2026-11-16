"""Annual heating and cooling energy of the dwellings of each tract.

Run from the project root: ``python analysis/03_energy_requirement.py``. It needs
the results of ``02_housing_stock.py`` (weights, donors and degree days).

Section A. Sensitivity of the load to the degree days of the weather station,
estimated with and without the dwelling controls, and a check that leaves one
county out (``data/results/tables/``).

Section B. Energy per tract and type of dwelling and per tract, with the
correction extended linearly beyond the range of the stations and with the
correction capped at the edge of the range (``data/interim/``).

Section C. Hourly profile of every tract, one Parquet file per tract in
``data/processed/hourly_profiles_ut`` (behind ``config.BUILD_HOURLY_PROFILES``),
and a check that the sum of the hours is the annual energy of section B. With
``config.HOURLY_SHAPE_BY_TEMPERATURE`` the shape of heating and cooling follows
the hourly temperature of the tract (needs the daily PRISM files of 2018).
"""

import logging

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import energy_requirement as er
from utah_ejpi import hourly, resstock
from utah_ejpi.utils import validate_table

STATE = config.STATE.lower()
STATE_FIPS = config.STATE_FIPS
SPECS = {
    "type_only": ("tenure", "vintage", "building_type"),
    "with_controls": er.DEFAULT_CONTROLS,
}


def _inputs():
    raw, interim = config.RAW, config.INTERIM
    dwellings = resstock.build_resstock_dwellings(
        resstock.read_resstock_annual(
            raw
            / "resstock"
            / "2025_amy2018_release_1"
            / f"metadata_annual_{STATE}"
            / f"{config.STATE}_upgrade0.parquet"
        )
    )
    degree_days = pd.read_csv(
        interim / f"climate_degree_days_{STATE}.csv", dtype={"id": str}
    )
    return dwellings, degree_days


def section_a_sensitivity() -> dict:
    """Estimate the sensitivities and check them leaving one county out."""
    dwellings, degree_days = _inputs()
    results = config.ROOT / "data" / "results" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    rows, loco = [], []
    fitted = {}
    for spec, controls in SPECS.items():
        for end_use in er.END_USES:
            s = climate_fit.fit_sensitivity(end_use, dwellings, degree_days, controls)
            fitted[(spec, end_use)] = s
            se = np.sqrt(np.diag(s.covariance))
            rows.append(
                {
                    "specification": spec,
                    "end_use": end_use,
                    "intercept": s.intercept,
                    "intercept_se": se[0],
                    "slope_per_1000": s.slope,
                    "slope_se": se[1],
                    "dwellings": s.n_dwellings,
                    "stations": s.n_stations,
                    "dd_min": s.dd_min * er.UNIT,
                    "dd_max": s.dd_max * er.UNIT,
                }
            )
            fit_dwellings, fit_stations = climate_fit.fit_inputs(
                end_use, dwellings, degree_days
            )
            check = er.leave_one_county_out(
                fit_dwellings, fit_stations, end_use, controls
            )
            check["error_with_climate"] = (
                check["predicted_with_climate"] - check["observed"]
            )
            check["error_without_climate"] = (
                check["predicted_without_climate"] - check["observed"]
            )
            check.insert(0, "end_use", end_use)
            check.insert(0, "specification", spec)
            loco.append(check)
            weights = check["n"] / check["n"].sum()
            logging.info(
                "%s %s: line %.3f + %.3f * (degree days / 1000); leave-one-county-out "
                "root mean squared error %.3f with climate, %.3f without",
                spec,
                end_use,
                s.intercept,
                s.slope,
                np.sqrt((weights * check["error_with_climate"] ** 2).sum()),
                np.sqrt((weights * check["error_without_climate"] ** 2).sum()),
            )
    pd.DataFrame(rows).to_csv(
        results / f"degree_day_sensitivity_{STATE}.csv", index=False
    )
    pd.concat(loco).to_csv(
        results / f"degree_day_leave_one_county_out_{STATE}.csv", index=False
    )
    return fitted


def section_b_tract_energy(fitted: dict) -> None:
    """Compute the energy per tract and type for both rules of extrapolation."""
    dwellings, degree_days = _inputs()
    interim = config.INTERIM
    weights = pd.read_csv(
        interim / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    donors = pd.read_csv(
        interim / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    )
    weights = weights[weights["weight"] > 0]
    donors = donors.merge(weights[["GEOID", "tenure", "vintage", "building_type"]])
    heating, cooling = (
        fitted[("with_controls", "heating")],
        fitted[("with_controls", "cooling")],
    )
    for rule in ("linear", "cap"):
        energy = er.build_tract_type_energy(
            weights, donors, dwellings, degree_days, degree_days, heating, cooling, rule
        )
        validate_table(energy, "tract_type_energy")
        energy.to_csv(interim / f"tract_type_energy_{rule}_{STATE}.csv", index=False)
        summary = er.tract_energy_summary(energy)
        summary.to_csv(
            interim / f"tract_energy_summary_{rule}_{STATE}.csv", index=False
        )
        total = summary["dwellings"].sum()
        logging.info(
            "%s: mean per dwelling heating %.0f kBtu, cooling %.0f kBtu",
            rule,
            (summary["heating_kbtu"] * summary["dwellings"]).sum() / total,
            (summary["cooling_kbtu"] * summary["dwellings"]).sum() / total,
        )


def section_c_hourly_profiles(fitted: dict) -> None:
    """Build the hourly profile of every tract and check it against section B."""
    dwellings, degree_days = _inputs()
    interim = config.INTERIM
    weights = pd.read_csv(
        interim / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    donors = pd.read_csv(
        interim / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    )
    weights = weights[weights["weight"] > 0]
    donors = donors.merge(weights[["GEOID", "tenure", "vintage", "building_type"]])
    heating, cooling = (
        fitted[("with_controls", "heating")],
        fitted[("with_controls", "cooling")],
    )
    coefficients = hourly.donor_coefficients(
        weights, donors, dwellings, degree_days, degree_days, heating, cooling
    )
    release = config.RAW / "resstock" / "2025_amy2018_release_1"
    series = release / f"timeseries_{STATE}_upgrade0"
    used = dwellings[dwellings["building_id"].isin(coefficients["building_id"])]
    county_shift = hourly.county_clock_shifts(
        used.set_index("building_id")["county"], series, release / f"weather_{STATE}"
    )
    logging.info("clock shift of the series by county (hours): %s", county_shift)
    clock_shift = used.set_index("building_id")["county"].map(county_shift)
    out_dir = config.ROOT / "data" / "processed" / f"hourly_profiles_{STATE}"
    shape = {}
    if config.HOURLY_SHAPE_BY_TEMPERATURE:
        tract_daily, station_daily, station_of = climate_fit.daily_temperatures(
            dwellings
        )
        shape = {
            "tract_daily_temperature": tract_daily,
            "station_daily_temperature": station_daily,
            "dwelling_station": station_of,
        }
    index = hourly.build_tract_profiles(
        coefficients, series, out_dir, clock_shift, **shape
    )
    index.to_csv(interim / f"hourly_profiles_index_{STATE}.csv", index=False)
    logging.info("profiles written: %d tracts in %s", len(index), out_dir)

    # The sum of the hours must be the annual energy of the tract (section B).
    energy = er.build_tract_type_energy(
        weights, donors, dwellings, degree_days, degree_days, heating, cooling
    )
    w = energy["dwellings"]
    totals = (
        pd.DataFrame(
            {
                "GEOID": energy["GEOID"],
                "dwellings": w,
                "heating_kwh": energy["heating_kwh"] * w,
                "cooling_kwh": energy["cooling_kwh"] * w,
            }
        )
        .groupby("GEOID")
        .sum()
    )
    annual = totals[["heating_kwh", "cooling_kwh"]].div(totals["dwellings"], axis=0)
    rows = []
    for geoid in annual.index:
        profile = hourly.fuel_use_profiles(hourly.read_tract_profile(out_dir, geoid))
        rows.append(
            {
                "GEOID": geoid,
                "heating_kwh_hours": profile["heating_kwh"].sum(),
                "cooling_kwh_hours": profile["cooling_kwh"].sum(),
            }
        )
    check = pd.DataFrame(rows).set_index("GEOID").join(annual)
    for use in ("heating", "cooling"):
        gap = (check[f"{use}_kwh_hours"] - check[f"{use}_kwh"]).abs()
        rel = gap / check[f"{use}_kwh"].where(check[f"{use}_kwh"] > 0)
        logging.info(
            "%s: largest relative gap between the sum of the hours and the annual "
            "energy %.2e (kWh per dwelling)",
            use,
            rel.max(),
        )
    check.to_csv(
        config.ROOT / "data" / "results" / "tables" / f"hourly_annual_check_{STATE}.csv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fitted_lines = section_a_sensitivity()
    section_b_tract_energy(fitted_lines)
    if config.BUILD_HOURLY_PROFILES:
        section_c_hourly_profiles(fitted_lines)
