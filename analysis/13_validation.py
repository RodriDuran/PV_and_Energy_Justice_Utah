"""Validation of the link between the tracts and the modeled ResStock dwellings.

Run from the project root: ``python analysis/13_validation.py``. It needs the
results of ``02_housing_stock.py`` and the Census tables of ``01_download_data.py``.

Section A. Composition: the dwellings chosen for each tract are compared with the
Census counts of heating fuel, bedrooms, household size and air conditioning
(``data/results/tables/``).

Section B. Energy: the modeled consumption of heating and cooling is compared with
the 2020 Residential Energy Consumption Survey (RECS) households of Utah, by group.

Section C. Energy: the ordering of the tracts is compared with the LEAD tool
expenditure on electricity, gas and other fuels (rank correlation).

Section D. Known population: each county and each PUMA of ResStock is estimated
without using its own dwellings and compared with its true mean (leave-one-unit-out).
"""

import logging

import pandas as pd

import climate_fit
import config
from utah_ejpi import energy_requirement as er
from utah_ejpi import housing_stock, resstock, uncertainty, validation

STATE_FIPS = config.STATE_FIPS
STATE = config.STATE.lower()
TABLES = {"heating_fuel": "B25117", "bedrooms": "B25042", "household_size": "B25009"}
TABLE_SUFFIX = {"B25117": "us", "B25042": STATE, "B25009": STATE}


def section_a_composition() -> None:
    """Compare the composition implied by the chosen dwellings with the Census."""
    raw, interim = config.RAW, config.INTERIM
    results = config.ROOT / "data" / "results" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    dwellings = resstock.build_resstock_dwellings(
        resstock.read_resstock_annual(
            raw
            / "resstock"
            / "2025_amy2018_release_1"
            / f"metadata_annual_{STATE}"
            / f"{config.STATE}_upgrade0.parquet"
        )
    )
    weights = pd.read_csv(
        interim / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    donors = pd.read_csv(
        interim / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    )
    # Tracts with no dwellings in the survey have nothing to compare.
    weights = weights[weights.groupby("GEOID")["weight"].transform("sum") > 0]
    donors = donors[donors["GEOID"].isin(weights["GEOID"])]

    for name, table in TABLES.items():
        acs = housing_stock.read_acs_table(
            raw / "acs" / "acs5_2024" / f"{table}_tract_{TABLE_SUFFIX[table]}.csv"
        )
        observed = validation.observed_composition(acs, name, STATE_FIPS)
        implied = validation.implied_composition(weights, donors, dwellings, name)
        by_tract, by_class = validation.composition_errors(implied, observed)
        by_tract.to_csv(
            results / f"composition_{name}_by_tract_{STATE}.csv", index=False
        )
        by_class.to_csv(
            results / f"composition_{name}_by_class_{STATE}.csv", index=False
        )
        logging.info(
            "%s: median share of dwellings to reclassify %.3f, 90th percentile %.3f",
            name,
            by_tract["tae_share"].median() / 2,
            by_tract["tae_share"].quantile(0.9) / 2,
        )

    lace = pd.read_csv(
        raw / "ac_prevalence" / "lace_2023" / "LACE_23_Tract.csv",
        dtype={"STATE": str, "COUNTY": str, "TRACT": str},
    )
    ac = validation.air_conditioning_comparison(weights, donors, dwellings, lace)
    ac.to_csv(
        results / f"composition_air_conditioning_by_tract_{STATE}.csv", index=False
    )
    logging.info(
        "air conditioning: mean difference %.3f, median absolute difference %.3f",
        ac["difference"].mean(),
        ac["difference"].abs().median(),
    )


def _energy_inputs():
    """Read the dwellings, weights, donors and degree days of Utah."""
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
    weights = pd.read_csv(
        interim / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    weights = weights[weights["weight"] > 0]
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    donors = pd.read_csv(
        interim / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    )
    donors = donors.merge(weights[keys])
    return dwellings, degree_days, weights, donors


def section_b_recs() -> None:
    """Compare the modeled consumption with the RECS households of Utah."""
    dwellings, degree_days, weights, donors = _energy_inputs()
    results = config.ROOT / "data" / "results" / "tables"
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    donor_energy = er.donor_level_energy(
        weights, donors, dwellings, degree_days, degree_days, heating, cooling, "linear"
    )
    recs = validation.read_recs(
        config.RAW / "recs_2020" / "recs2020_public_v7.csv", config.STATE
    )
    tables = [
        validation.compare_with_recs(recs, dwellings, donor_energy, variable)
        for variable in ("heating_kwh", "cooling_kwh")
    ]
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(results / f"recs_comparison_{STATE}.csv", index=False)
    for _, row in out[out["group_type"] == "all"].iterrows():
        logging.info(
            "%s: RECS %.0f (se %.0f) kWh, ResStock %.0f, tracts %.0f, "
            "tracts corrected %.0f",
            row["variable"],
            row["mean"],
            row["se"],
            row["resstock"],
            row["tracts_uncorrected"],
            row["tracts_corrected"],
        )


def section_c_lead() -> None:
    """Correlate the ordering of the tracts with the LEAD expenditure."""
    dwellings, degree_days, weights, donors = _energy_inputs()
    results = config.ROOT / "data" / "results" / "tables"
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    energy = er.build_tract_type_energy(
        weights, donors, dwellings, degree_days, degree_days, heating, cooling, "linear"
    )
    lead = validation.read_lead_tract_means(
        config.RAW / "lead_2022" / f"{config.STATE}-2022-LEAD-data.zip",
        f"{config.STATE} AMI Census Tracts 2022.csv",
    )
    out = validation.lead_convergence(energy, lead)
    out.to_csv(results / f"lead_convergence_{STATE}.csv", index=False)
    logging.info("rank correlation with LEAD: %s", out.to_dict("records"))


def section_d_known_population() -> None:
    """Leave one county or PUMA out and compare the estimate with its true mean."""
    dwellings, degree_days, _, _ = _energy_inputs()
    results = config.ROOT / "data" / "results" / "tables"
    extra, extra_stations = climate_fit.extra_fit_data(dwellings, degree_days)
    for unit in ("county", "puma"):
        table = uncertainty.known_population_test(
            dwellings,
            degree_days,
            unit=unit,
            n_draws=config.N_DRAWS,
            seed=config.SEED,
            extra_fit_dwellings=extra,
            extra_fit_stations=extra_stations,
        )
        table.to_csv(results / f"known_population_{unit}_{STATE}.csv", index=False)
        summary = uncertainty.known_population_summary(table)
        summary.to_csv(
            results / f"known_population_summary_{unit}_{STATE}.csv", index=False
        )
        big = table[table["dwellings"] >= 30]
        summary_big = uncertainty.known_population_summary(big)
        summary_big.to_csv(
            results / f"known_population_summary_{unit}_30dwellings_{STATE}.csv",
            index=False,
        )
        logging.info(
            "%s (%d units, %d with 30 or more dwellings):\n%s",
            unit,
            len(table),
            len(big),
            summary.round(3).to_string(index=False),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    section_a_composition()
    section_b_recs()
    section_c_lead()
    section_d_known_population()
