"""Reweight the modeled dwellings to several Census tables, tract by tract.

Run from the project root: ``python analysis/04_reweighting.py``. It needs the
results of ``02_housing_stock.py`` and the Census tables of ``01_download_data.py``.

Four nested variants are fitted: the types of dwelling alone, then adding the heating
fuel, the bedrooms and the household size. For each one the dwelling weights of every
tract and the diagnostics of the fitting are saved in ``data/interim/``, and the energy
per tract and the comparisons with the other sources in ``data/results/tables/``.
"""

import logging

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import energy_requirement as er
from utah_ejpi import housing_stock, resstock, validation
from utah_ejpi.utils import validate_table

STATE_FIPS = config.STATE_FIPS
STATE = config.STATE.lower()
VARIANTS = {
    "type": housing_stock.REWEIGHT_SETS[:1],
    "type_fuel": housing_stock.REWEIGHT_SETS[:2],
    "type_fuel_bedrooms": housing_stock.REWEIGHT_SETS[:3],
    "type_fuel_bedrooms_size": housing_stock.REWEIGHT_SETS,
}


def _read_inputs():
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
    counts = housing_stock.acs_type_counts(
        housing_stock.read_acs_table(raw / "acs" / "acs5_2024" / "B25127_tract_us.csv"),
        STATE_FIPS,
    )
    composition = {
        name: validation.observed_composition(
            housing_stock.read_acs_table(
                raw
                / "acs"
                / "acs5_2024"
                / f"{table}_tract_{STATE if scope == 'state' else scope}.csv"
            ),
            name,
            STATE_FIPS,
        )
        for name, (table, scope) in config.COMPOSITION_TABLES.items()
    }
    shares = pd.read_csv(
        interim / f"tract_puma_share_{STATE}.csv", dtype={"GEOID": str}
    )
    master = pd.read_csv(interim / f"tract_master_{STATE}.csv", dtype=str)
    rank = pd.read_csv(
        interim / f"puma_climate_rank_{STATE}.csv",
        dtype={"GEOID": str, "resstock_puma": str},
    )
    degree_days = pd.read_csv(
        interim / f"climate_degree_days_{STATE}.csv", dtype={"id": str}
    )
    return dwellings, counts, composition, shares, master, rank, degree_days


def run() -> None:
    """Fit the four variants and compare them with the base method and the others."""
    dwellings, counts, composition, shares, master, rank, degree_days = _read_inputs()
    results = config.ROOT / "data" / "results" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    recs = validation.read_recs(
        config.RAW / "recs_2020" / "recs2020_public_v7.csv", config.STATE
    )
    lead = validation.read_lead_tract_means(
        config.RAW / "lead_2022" / f"{config.STATE}-2022-LEAD-data.zip",
        f"{config.STATE} AMI Census Tracts 2022.csv",
    )
    lace = pd.read_csv(
        config.RAW / "ac_prevalence" / "lace_2023" / "LACE_23_Tract.csv",
        dtype={"STATE": str, "COUNTY": str, "TRACT": str},
    )
    base = pd.read_csv(
        config.INTERIM / f"tract_energy_summary_linear_{STATE}.csv",
        dtype={"GEOID": str},
    )

    summary_rows, recs_rows, lead_rows = [], [], []
    for name, sets in VARIANTS.items():
        weights, diagnostics = housing_stock.reweight_tracts(
            counts, composition, dwellings, shares, master, rank, sets=sets
        )
        validate_table(weights, "tract_dwelling_weights")
        validate_table(diagnostics, "reweighting_diagnostics")
        weights.to_csv(
            config.INTERIM / f"tract_dwelling_weights_{name}_{STATE}.csv", index=False
        )
        diagnostics.to_csv(
            config.INTERIM / f"reweighting_diagnostics_{name}_{STATE}.csv", index=False
        )
        frame = er.weighted_dwelling_energy(
            weights, dwellings, degree_days, degree_days, heating, cooling, "linear"
        )
        means = er.weighted_tract_means(frame)
        means.to_csv(
            config.INTERIM / f"tract_energy_summary_reweighted_{name}_{STATE}.csv",
            index=False,
        )

        total = means["dwellings"].sum()
        merged = means.merge(base, on="GEOID", suffixes=("", "_base"))
        summary_rows.append(
            {
                "variant": name,
                "tracts": len(diagnostics),
                "converged": int(diagnostics["converged"].sum()),
                "max_error_median": diagnostics["max_error"].median(),
                "max_error_p90": diagnostics["max_error"].quantile(0.9),
                "effective_n_median": diagnostics["effective_n"].median(),
                "effective_n_p10": diagnostics["effective_n"].quantile(0.1),
                "tier_1_share_median": diagnostics["tier_1_share"].median(),
                "infeasible_tracts": int(
                    (diagnostics["infeasible_categories"] > 0).sum()
                ),
                "pruned_share_max": diagnostics["pruned_share"].max(),
                "heating_kbtu_mean": (means["heating_kbtu"] * means["dwellings"]).sum()
                / total,
                "cooling_kbtu_mean": (means["cooling_kbtu"] * means["dwellings"]).sum()
                / total,
                "heating_rank_corr_with_base": merged["heating_kbtu"].corr(
                    merged["heating_kbtu_base"], method="spearman"
                ),
                "cooling_rank_corr_with_base": merged["cooling_kbtu"].corr(
                    merged["cooling_kbtu_base"], method="spearman"
                ),
                "heating_median_abs_change": (
                    (merged["heating_kbtu"] - merged["heating_kbtu_base"]).abs()
                    / merged["heating_kbtu_base"]
                ).median(),
            }
        )

        # Reference comparisons use the weighted dwellings as if they were donors.
        for variable in ("heating_kwh", "cooling_kwh"):
            table = validation.compare_with_recs(recs, dwellings, frame, variable)
            table.insert(0, "variant", name)
            recs_rows.append(table)

        tract_fuel = pd.DataFrame(
            {"GEOID": means["GEOID"], "dwellings": means["dwellings"]}
        )
        for col in (
            "heating_kwh_natural_gas",
            "heating_kwh_electricity",
            "heating_kwh_propane",
            "heating_kwh_fuel_oil",
            "cooling_kwh_electricity",
        ):
            tract_fuel[col] = means[col]
        conv = validation.lead_convergence(tract_fuel, lead)
        conv.insert(0, "variant", name)
        lead_rows.append(conv)

        ac = _air_conditioning(frame, lace)
        summary_rows[-1]["ac_mean_difference"] = ac["difference"].mean()
        summary_rows[-1]["ac_median_abs_difference"] = ac["difference"].abs().median()

    pd.DataFrame(summary_rows).to_csv(
        results / f"reweighting_summary_{STATE}.csv", index=False
    )
    pd.concat(recs_rows).to_csv(
        results / f"reweighting_recs_comparison_{STATE}.csv", index=False
    )
    pd.concat(lead_rows).to_csv(
        results / f"reweighting_lead_convergence_{STATE}.csv", index=False
    )
    logging.info("\n%s", pd.DataFrame(summary_rows).round(3).T.to_string())


def _air_conditioning(frame: pd.DataFrame, lace: pd.DataFrame) -> pd.DataFrame:
    """Compare the share of reweighted dwellings with cooling with LACE."""
    cooled = (
        (frame["cooling_type"] != "None") * frame["weight"]
        if "cooling_type" in frame
        else None
    )
    if cooled is None:
        raise KeyError("The weighted dwellings need the cooling type.")
    by_tract = (
        pd.DataFrame({"GEOID": frame["GEOID"], "cooled": cooled, "w": frame["weight"]})
        .groupby("GEOID")
        .sum()
    )
    geoid = (
        lace["STATE"].astype(str).str.zfill(2)
        + lace["COUNTY"].astype(str).str.zfill(3)
        + lace["TRACT"].astype(str).str.zfill(6)
    )
    observed = pd.DataFrame(
        {
            "GEOID": geoid,
            "observed": lace["AC_E"] / lace["HSE_OCC_E"].where(lace["HSE_OCC_E"] > 0),
        }
    )
    out = observed.merge(
        (by_tract["cooled"] / by_tract["w"]).rename("implied").reset_index()
    )
    out["difference"] = out["implied"] - out["observed"]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.seterr(all="ignore")
    run()
