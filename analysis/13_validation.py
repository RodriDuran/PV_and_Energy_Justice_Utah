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

Section E. Wake-up and bed times of the time use survey: the rule of
``comfort_gap.sleep_times`` is compared with two published rules (Knutson and
Lauderdale; Basner and Dinges) on the same diaries, with the medians and quartiles
by day type and the window of a household of one to four persons that each rule gives.

Section F. Composition: the energy of each tract by fuel with and without the change
by composition (regression on the Census shares) is compared with the LEAD
expenditure (rank correlation), to see whether adding the change improves the
agreement.

Section G. Seasonal pattern: the modeled natural gas and electricity of each month
are compared with the residential gas and electricity sold in the state that month
(Energy Information Administration, 2018), by the share of the year that falls in each
month and by the error of each month, with the modeled heating and cooling next to the
totals.

Section H. Origin of the seasonal differences: the modeled end uses against the RECS
households, what the monthly pattern would be with some uses at the RECS level, the
billing lag of the monthly series and the gas customers that are not in the model.

Section I. Other states: the end uses of ResStock against RECS and the total gas and
electricity against the totals sold, for the neighbors of the study state and states of
a different climate, to see whether the differences found in section H repeat.
"""

import logging
import zipfile

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import comfort_gap as cg
from utah_ejpi import energy_requirement as er
from utah_ejpi import gap_uncertainty as gu
from utah_ejpi import hourly, housing_stock, resstock, uncertainty, validation

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


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    """Weighted quantile of the values that are not missing."""
    ok = values.notna()
    v, w = values[ok].to_numpy(float), weights[ok].to_numpy(float)
    order = np.argsort(v)
    cumulative = np.cumsum(w[order]) / w.sum()
    return float(v[order][np.searchsorted(cumulative, q)])


def section_e_sleep_rules() -> None:
    """Compare the sleep rule of the study with two published rules."""
    atus = config.RAW / "atus"

    def read(zip_name, member, columns):
        with zipfile.ZipFile(atus / zip_name) as z, z.open(member) as f:
            return pd.read_csv(f, usecols=columns)

    resp = read(
        "atusresp-0324.zip",
        "atusresp_0324.dat",
        ["TUCASEID", "TULINENO", "TUFNWGTP", "TUDIARYDAY", "TUYEAR"],
    )
    cps = read(
        "atuscps-0324.zip",
        "atuscps_0324.dat",
        ["TUCASEID", "TULINENO", "PRTAGE", "GESTFIPS"],
    )
    acts = read(
        "atusact-0324.zip",
        "atusact_0324.dat",
        ["TUCASEID", "TUSTARTTIM", "TUSTOPTIME", "TRCODEP"],
    )
    resp = resp.merge(cps, on=["TUCASEID", "TULINENO"], how="left")
    resp = resp[(resp["TUYEAR"] >= 2014) & (resp["TUYEAR"] != 2020)]
    resp = resp[resp["PRTAGE"] >= 15]
    acts = acts[acts["TUCASEID"].isin(resp["TUCASEID"])]
    rules = {
        "study": cg.sleep_times(acts),
        "knutson_lauderdale": cg.sleep_times_published(acts, "knutson_lauderdale"),
        "basner_dinges": cg.sleep_times_published(acts, "basner_dinges"),
    }
    weekend = resp["TUDIARYDAY"].isin([1, 7])
    groups = {
        "all": resp.index == resp.index,
        "weekday": ~weekend,
        "weekend": weekend,
        "state": resp["GESTFIPS"] == int(config.STATE_FIPS),
    }
    rows, windows = [], []
    for rule, times in rules.items():
        data = resp.merge(times, on="TUCASEID", how="left")
        data["bed"] = data["bed_min"].where(
            data["bed_min"] >= 720, data["bed_min"] + 1440
        )
        for group, mask in groups.items():
            d = data[np.asarray(mask)]
            row = {
                "rule": rule,
                "group": group,
                "respondents": len(d),
                "wake_identified": float(d["wake_min"].notna().mean()),
                "bed_identified": float(d["bed_min"].notna().mean()),
            }
            for name, column in (("wake", "wake_min"), ("bed", "bed")):
                for q in (0.25, 0.5, 0.75):
                    row[f"{name}_p{int(q * 100)}"] = (
                        _weighted_quantile(d[column], d["TUFNWGTP"], q) % 1440
                    )
            rows.append(row)
        table = cg.household_window_table(data, seed=config.SEED)
        windows.append(table.assign(rule=rule))
    results = config.ROOT / "data" / "results" / "tables"
    pd.DataFrame(rows).to_csv(results / f"atus_sleep_rules_{STATE}.csv", index=False)
    pd.concat(windows).to_csv(
        results / f"atus_sleep_rules_windows_{STATE}.csv", index=False
    )
    logging.info("sleep rules compared in %d respondents", len(resp))


def section_f_composition_vs_lead() -> None:
    """Compare the tract energy with and without the change by composition."""
    raw = config.RAW
    results = config.ROOT / "data" / "results" / "tables"
    dwellings = climate_fit.read_dwellings(config.STATE)
    degree_days = pd.read_csv(
        config.INTERIM / f"climate_degree_days_{STATE}.csv", dtype={"id": str}
    )
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    base = pd.read_csv(
        config.INTERIM / f"tract_dwelling_weights_type_{STATE}.csv",
        dtype={"GEOID": str},
    )
    frame = er.weighted_dwelling_energy(
        base, dwellings, degree_days, degree_days, heating, cooling, "linear"
    )
    means = er.weighted_tract_means(frame)
    columns = [
        "heating_kwh_natural_gas",
        "heating_kwh_electricity",
        "heating_kwh_propane",
        "heating_kwh_fuel_oil",
        "cooling_kwh_electricity",
    ]
    stock = dwellings[dwellings["occupied"]]
    counts = housing_stock.acs_type_counts(
        housing_stock.read_acs_table(raw / "acs" / "acs5_2024" / "B25127_tract_us.csv"),
        config.STATE_FIPS,
    )
    observed = {
        name: validation.observed_composition(
            housing_stock.read_acs_table(
                raw
                / "acs"
                / "acs5_2024"
                / f"{table}_tract_{STATE if scope == 'state' else scope}.csv"
            ),
            name,
            config.STATE_FIPS,
        )
        for name, (table, scope) in config.COMPOSITION_TABLES.items()
    }
    shift = gu.composition_shift_regression(
        base,
        stock.set_index("building_id")[columns],
        housing_stock.class_indicators(stock),
        housing_stock.composition_shares(counts, observed),
        pd.get_dummies(
            stock.set_index("building_id")[
                ["tenure", "vintage", "building_type"]
            ].astype(str)
        ).astype(float),
    )
    lead = validation.read_lead_tract_means(
        raw / "lead_2022" / f"{config.STATE}-2022-LEAD-data.zip",
        f"{config.STATE} AMI Census Tracts 2022.csv",
    )
    tract = means[["GEOID", "dwellings", *columns]].copy()
    shifted = tract.copy()
    change = shift.reindex(shifted["GEOID"]).fillna(0.0)
    for column in columns:
        shifted[column] = (shifted[column].to_numpy() + change[column].to_numpy()).clip(
            min=0
        )
    rows = []
    for label, energy in (("type only", tract), ("type and composition", shifted)):
        table = validation.lead_convergence(energy, lead)
        table.insert(0, "energy", label)
        rows.append(table)
    pd.concat(rows).to_csv(results / f"composition_lead_check_{STATE}.csv", index=False)


def section_g_monthly_vs_eia() -> None:
    """Compare the modeled energy by month with the monthly totals of the state.

    The modeled natural gas and electricity (every use, electricity net of the
    generation of the panels already installed, as the sales to customers are) of
    each month are scaled to the households of the state of ``MEASURED_YEAR`` and
    compared with the residential gas and electricity sold in that month. The level
    was checked before for the whole year, so the pattern of the year is judged on
    the share of the annual total that falls in each month. The modeled heating,
    cooling and the rest of the uses (water heating, cooking, lighting, appliances)
    are written next to the totals to read where a seasonal difference comes from.
    """
    results = config.ROOT / "data" / "results" / "tables"
    directory = config.RAW / f"eia_state_totals_{STATE}"
    measured = validation.read_eia_monthly_residential(
        directory, config.STATE, config.MEASURED_YEAR
    )
    summary = pd.read_csv(
        config.INTERIM / f"tract_energy_summary_linear_{STATE}.csv",
        dtype={"GEOID": str},
    ).set_index("GEOID")["dwellings"]
    modeled = hourly.state_monthly_means(
        config.ROOT / "data" / "processed" / f"hourly_profiles_{STATE}", summary
    )
    households = float(config.MEASURED_YEAR_HOUSEHOLDS)
    parts = {
        "natural_gas": {
            "total": modeled["natural_gas"],
            "heating": modeled["natural_gas_heating"],
            "cooling": 0.0 * modeled["natural_gas"],
        },
        "electricity": {
            "total": modeled["electricity"] + modeled["electricity_pv"],
            "heating": modeled["electricity_heating"],
            "cooling": modeled["electricity_cooling"],
        },
    }
    rows = []
    for fuel, part in parts.items():
        observed = measured[f"{fuel}_kwh"].to_numpy()
        model = part["total"].to_numpy() * households
        share_model, share_observed = model / model.sum(), observed / observed.sum()
        for m in range(12):
            rows.append(
                {
                    "fuel": fuel,
                    "month": m + 1,
                    "measured_kwh": observed[m],
                    "modeled_kwh": model[m],
                    "error_percent": 100.0 * (model[m] / observed[m] - 1.0),
                    "share_measured_percent": 100.0 * share_observed[m],
                    "share_modeled_percent": 100.0 * share_model[m],
                    "share_error_points": 100.0 * (share_model[m] - share_observed[m]),
                    "modeled_heating_kwh": part["heating"].to_numpy()[m] * households,
                    "modeled_cooling_kwh": part["cooling"].to_numpy()[m] * households,
                }
            )
    table = pd.DataFrame(rows)
    table["modeled_other_kwh"] = (
        table["modeled_kwh"]
        - table["modeled_heating_kwh"]
        - table["modeled_cooling_kwh"]
    )
    table.to_csv(results / f"monthly_state_check_{STATE}.csv", index=False)
    for fuel, sub in table.groupby("fuel"):
        logging.info(
            "%s: year %.1f %%, correlation of the shares %.3f, largest share "
            "error %.2f points, peak month modeled %d measured %d",
            fuel,
            100.0 * (sub["modeled_kwh"].sum() / sub["measured_kwh"].sum() - 1.0),
            np.corrcoef(sub["share_modeled_percent"], sub["share_measured_percent"])[
                0, 1
            ],
            sub["share_error_points"].abs().max(),
            int(sub.loc[sub["modeled_kwh"].idxmax(), "month"]),
            int(sub.loc[sub["measured_kwh"].idxmax(), "month"]),
        )


GAS_USES = {
    "heating": (
        ("natural_gas_heating", "natural_gas_heating_hp_bkup"),
        ("BTUNGSPH",),
    ),
    "water heating": (("natural_gas_hot_water",), ("BTUNGWTH",)),
    "cooking": (("natural_gas_range_oven",), ("BTUNGCOK",)),
    "clothes dryer": (("natural_gas_clothes_dryer",), ("BTUNGCDR",)),
    "pool and spa heaters": (
        ("natural_gas_pool_heater", "natural_gas_permanent_spa_heat"),
        ("BTUNGPLHEAT", "BTUNGHTBHEAT"),
    ),
    "fireplace, grill and lighting": (
        ("natural_gas_fireplace", "natural_gas_grill", "natural_gas_lighting"),
        ("BTUNGNEC",),
    ),
}
ELECTRICITY_USES = {
    "heating": (
        (
            "electricity_heating",
            "electricity_heating_hp_bkup",
            "electricity_heating_hp_bkup_fa",
            "electricity_heating_fans_pumps",
        ),
        ("BTUELSPH", "BTUELAHUHEAT"),
    ),
    "cooling": (
        ("electricity_cooling", "electricity_cooling_fans_pumps"),
        ("BTUELCOL", "BTUELAHUCOL"),
    ),
    "water heating": (("electricity_hot_water",), ("BTUELWTH",)),
    "refrigerator and freezer": (
        ("electricity_refrigerator", "electricity_freezer"),
        ("BTUELRFG", "BTUELFRZ"),
    ),
    "cooking": (("electricity_range_oven",), ("BTUELCOK", "BTUELMICRO")),
    "clothes dryer": (("electricity_clothes_dryer",), ("BTUELCDR",)),
    "washer and dishwasher": (
        ("electricity_clothes_washer", "electricity_dishwasher"),
        ("BTUELCW", "BTUELDWH"),
    ),
    "lighting": (
        (
            "electricity_lighting_interior",
            "electricity_lighting_exterior",
            "electricity_lighting_garage",
        ),
        ("BTUELLGT",),
    ),
    "television": (("electricity_television",), ("BTUELTVREL",)),
    "electric vehicle": (("electricity_ev_charging",), ("BTUELEVCHRG",)),
    "pool and spa": (
        (
            "electricity_pool_pump",
            "electricity_pool_heater",
            "electricity_permanent_spa_pump",
            "electricity_permanent_spa_heat",
        ),
        ("BTUELPLPMP", "BTUELHTBPMP", "BTUELHTBHEAT"),
    ),
}
SEASONS = {"winter": [10, 11, 0, 1, 2], "shoulder": [3, 4, 8, 9], "summer": [5, 6, 7]}
REST = "rest of the uses"


def _uses_by_hour(state, uses, fuel):
    """Hourly series of each group of uses and of the rest of the fuel (kWh)."""
    series = {
        name: state[[c + "_kwh" for c in cols]].sum(axis=1).to_numpy()
        for name, (cols, _) in uses.items()
    }
    used = {c + "_kwh" for cols, _ in uses.values() for c in cols}
    rest = [
        c
        for c in state.columns
        if c.startswith(fuel + "_") and c not in used and c != "electricity_pv_kwh"
    ]
    series[REST] = state[rest].sum(axis=1).to_numpy()
    return series


def _month_sums(x, months, lag_days=0):
    """Monthly sums of an hourly series whose use is billed ``lag_days`` later."""
    x = np.roll(x, lag_days * 24)
    return np.array([x[months == k].sum() for k in range(1, 13)])


def _scenario_row(name, fuel, modeled, observed, months, lag_days=0):
    """Error of a monthly series by season and the error of its shares."""
    monthly = _month_sums(modeled, months, lag_days)
    share, share_obs = monthly / monthly.sum(), observed / observed.sum()
    row = {
        "fuel": fuel,
        "scenario": name,
        "lag_days": lag_days,
        "year_error_percent": 100.0 * (monthly.sum() / observed.sum() - 1.0),
    }
    for season, idx in SEASONS.items():
        row[f"{season}_error_percent"] = 100.0 * (
            monthly[idx].sum() / observed[idx].sum() - 1.0
        )
    rmse = float(np.sqrt(((share - share_obs) ** 2).mean()))
    row["share_rmse_points"] = 100.0 * rmse
    row["share_correlation"] = float(np.corrcoef(share, share_obs)[0, 1])
    return row


def _end_use_table(state, fuel, uses, recs_path):
    """Modeled annual energy of each end use against the RECS households (kWh)."""
    hours = _uses_by_hour(state, uses, fuel)
    recs = validation.recs_end_use_means(
        recs_path, config.STATE, {k: v[1] for k, v in uses.items()}
    ).set_index("use")
    total_column = "BTUNG" if fuel == "natural_gas" else "BTUEL"
    total = validation.recs_end_use_means(
        recs_path, config.STATE, {"total": (total_column,)}
    ).loc[0]
    rows = []
    for use, values in hours.items():
        row = {"fuel": fuel, "use": use, "modeled_kwh": float(values.sum())}
        if use in recs.index:
            row["recs_kwh"] = recs.loc[use, "mean_kwh"]
            row["recs_se_kwh"] = recs.loc[use, "se_kwh"]
            row["recs_share_with_use"] = recs.loc[use, "share_with_use"]
            row["recs_households_with_use"] = int(recs.loc[use, "n_with_use"])
        rows.append(row)
    table = pd.DataFrame(rows)
    rest = table["use"] == REST
    if float(table.loc[rest, "modeled_kwh"].iloc[0]) == 0.0:
        table = table[~rest].reset_index(drop=True)
        rest = table["use"] == REST
    table.loc[rest, "recs_kwh"] = total["mean_kwh"] - table.loc[~rest, "recs_kwh"].sum()
    table.loc[len(table)] = {
        "fuel": fuel,
        "use": "total",
        "modeled_kwh": float(sum(hours.values()).sum()),
        "recs_kwh": total["mean_kwh"],
        "recs_se_kwh": total["se_kwh"],
        "recs_share_with_use": total["share_with_use"],
        "recs_households_with_use": int(total["n_with_use"]),
    }
    table["ratio_modeled_over_recs"] = table["modeled_kwh"] / table["recs_kwh"]
    return table, hours


def _modeled_gas_households_share() -> float:
    """Share of the dwellings of the model whose donor uses natural gas."""
    meta = pd.read_parquet(
        config.RAW
        / "resstock"
        / "2025_amy2018_release_1"
        / f"metadata_annual_{STATE}"
        / f"{config.STATE}_upgrade0.parquet"
    ).set_index("bldg_id")
    weights = pd.read_csv(
        config.INTERIM / f"tract_dwelling_weights_type_{STATE}.csv",
        dtype={"GEOID": str},
    )
    summary = pd.read_csv(
        config.INTERIM / f"tract_energy_summary_linear_{STATE}.csv",
        dtype={"GEOID": str},
    ).set_index("GEOID")["dwellings"]
    share = weights["weight"] / weights.groupby("GEOID")["weight"].transform("sum")
    count = (share * weights["GEOID"].map(summary)).groupby(weights["building_id"])
    dwellings = count.sum().reindex(meta.index).fillna(0.0)
    uses_gas = meta["out.natural_gas.total.energy_consumption..kwh"] > 0
    return float((dwellings * uses_gas).sum() / dwellings.sum())


def section_h_origin_of_the_difference() -> None:
    """Look for the origin of the seasonal differences with the monthly totals.

    Four checks, each written to ``data/results/tables/``:

    * ``monthly_end_use_check``: the modeled annual energy of each end use against
      the 2020 RECS households of the state (jackknife error), for natural gas and
      electricity. RECS is not metered and its Utah sample is below the totals
      sold (section B), so only large differences count.
    * ``monthly_scenarios``: the monthly pattern if some modeled uses were at the
      RECS level (and, in the last ones, with the level of the year matched). They
      are diagnoses of which change moves the pattern in the right direction, not
      corrections: none is adopted.
    * ``monthly_lag_sensitivity``: the error of the pattern when the use of each
      month is billed up to 28 days later (the monthly series of the Energy
      Information Administration are by billing month, not calendar month).
    * ``monthly_gas_coverage``: the residential gas customers against the
      dwellings of the model that use gas, and the gas that the difference would
      represent if those customers used a quarter, half or all of the mean.
    """
    results = config.ROOT / "data" / "results" / "tables"
    directory = config.RAW / f"eia_state_totals_{STATE}"
    households = float(config.MEASURED_YEAR_HOUSEHOLDS)
    measured = validation.read_eia_monthly_residential(
        directory, config.STATE, config.MEASURED_YEAR
    )
    summary = pd.read_csv(
        config.INTERIM / f"tract_energy_summary_linear_{STATE}.csv",
        dtype={"GEOID": str},
    ).set_index("GEOID")["dwellings"]
    state = hourly.state_hourly_means(
        config.ROOT / "data" / "processed" / f"hourly_profiles_{STATE}", summary
    )
    months = state.index.month.to_numpy()
    recs_path = config.RAW / "recs_2020" / "recs2020_public_v7.csv"
    gas_table, gas_hours = _end_use_table(state, "natural_gas", GAS_USES, recs_path)
    el_table, el_hours = _end_use_table(
        state, "electricity", ELECTRICITY_USES, recs_path
    )
    both = pd.concat([gas_table, el_table], ignore_index=True)
    both["difference_gwh_state"] = (
        (both["recs_kwh"] - both["modeled_kwh"]) * households / 1e6
    )
    both.to_csv(results / f"monthly_end_use_check_{STATE}.csv", index=False)

    gas = {k: v * households for k, v in gas_hours.items()}
    gas_obs = measured["natural_gas_kwh"].to_numpy()
    base = sum(gas.values())
    gas_ratio = gas_table.set_index("use")["ratio_modeled_over_recs"].rdiv(1.0)
    others = [k for k in GAS_USES if k != "heating"]
    elec = {k: v * households for k, v in el_hours.items()}
    pv = state["electricity_pv_kwh"].to_numpy() * households
    e_obs = measured["electricity_kwh"].to_numpy()
    e_base = sum(elec.values()) + pv
    e_ratio = el_table.set_index("use")["ratio_modeled_over_recs"].rdiv(1.0)

    def at_recs(parts, ratio, chosen, extra=0.0):
        return (
            sum(ratio[k] * parts[k] if k in chosen else parts[k] for k in parts) + extra
        )

    rows = []
    for lag in (0, 7):
        hot = at_recs(gas, gas_ratio, ["water heating"])
        every = at_recs(gas, gas_ratio, others)
        full = at_recs(gas, gas_ratio, list(GAS_USES))
        matched = full * gas_obs.sum() / _month_sums(full, months).sum()
        for name, x in (
            ("model as it is", base),
            ("water heating at the RECS level", hot),
            ("every use except heating at the RECS level", every),
            ("every use at the RECS level, year matched", matched),
        ):
            rows.append(_scenario_row(name, "natural_gas", x, gas_obs, months, lag))
    for lag in (0, 3):
        part = at_recs(elec, e_ratio, ["heating", "lighting"], pv)
        part = part * e_obs.sum() / _month_sums(part, months).sum()
        full = at_recs(elec, e_ratio, list(ELECTRICITY_USES), pv)
        full = full * e_obs.sum() / _month_sums(full, months).sum()
        for name, x in (
            ("model as it is", e_base),
            ("heating and lighting at the RECS level, year matched", part),
            ("every use at the RECS level, year matched", full),
        ):
            rows.append(_scenario_row(name, "electricity", x, e_obs, months, lag))
    pd.DataFrame(rows).to_csv(results / f"monthly_scenarios_{STATE}.csv", index=False)

    lags = []
    for lag in (0, 3, 7, 10, 14, 21, 28):
        lags.append(
            _scenario_row("model as it is", "natural_gas", base, gas_obs, months, lag)
        )
        lags.append(
            _scenario_row("model as it is", "electricity", e_base, e_obs, months, lag)
        )
    pd.DataFrame(lags).to_csv(
        results / f"monthly_lag_sensitivity_{STATE}.csv", index=False
    )

    customers = validation.read_eia_gas_customers(
        directory, config.STATE, config.MEASURED_YEAR
    )
    share_gas = _modeled_gas_households_share()
    extra = customers - share_gas * households
    per_customer = float(gas_obs.sum()) / customers
    coverage = {
        "gas_customers": customers,
        "households_of_the_state": households,
        "modeled_share_with_gas": share_gas,
        "customers_not_in_the_model": extra,
        "measured_kwh_per_customer": per_customer,
        "gap_gwh_year": float(gas_obs.sum() - base.sum()) / 1e6,
    }
    for fraction in (0.25, 0.5, 1.0):
        key = f"gwh_if_extra_customers_use_{int(fraction * 100)}_percent"
        coverage[key] = extra * per_customer * fraction / 1e6
    pd.DataFrame([coverage]).to_csv(
        results / f"monthly_gas_coverage_{STATE}.csv", index=False
    )
    logging.info("origin of the difference: %s", coverage)


_KWH_PER_KBTU = 1.0 / 3.41214

STATE_USES = {
    "natural_gas": {
        "heating": (
            ("out.natural_gas.heating.energy_consumption..kwh",),
            ("BTUNGSPH",),
        ),
        "water heating": (
            ("out.natural_gas.hot_water.energy_consumption..kwh",),
            ("BTUNGWTH",),
        ),
        "cooking": (
            ("out.natural_gas.range_oven.energy_consumption..kwh",),
            ("BTUNGCOK",),
        ),
        "clothes dryer": (
            ("out.natural_gas.clothes_dryer.energy_consumption..kwh",),
            ("BTUNGCDR",),
        ),
        "pool and spa heaters": (
            (
                "out.natural_gas.pool_heater.energy_consumption..kwh",
                "out.natural_gas.permanent_spa_heat.energy_consumption..kwh",
            ),
            ("BTUNGPLHEAT", "BTUNGHTBHEAT"),
        ),
        "fireplace, grill and lighting": (
            (
                "out.natural_gas.fireplace.energy_consumption..kwh",
                "out.natural_gas.grill.energy_consumption..kwh",
                "out.natural_gas.lighting.energy_consumption..kwh",
            ),
            ("BTUNGNEC",),
        ),
        "total": (("out.natural_gas.total.energy_consumption..kwh",), ("BTUNG",)),
    },
    "electricity": {
        "heating": (
            (
                "out.electricity.heating.energy_consumption..kwh",
                "out.electricity.heating_hp_bkup.energy_consumption..kwh",
                "out.electricity.heating_fans_pumps.energy_consumption..kwh",
            ),
            ("BTUELSPH", "BTUELAHUHEAT"),
        ),
        "cooling": (
            (
                "out.electricity.cooling.energy_consumption..kwh",
                "out.electricity.cooling_fans_pumps.energy_consumption..kwh",
            ),
            ("BTUELCOL", "BTUELAHUCOL"),
        ),
        "lighting": (
            (
                "out.electricity.lighting_interior.energy_consumption..kwh",
                "out.electricity.lighting_exterior.energy_consumption..kwh",
                "out.electricity.lighting_garage.energy_consumption..kwh",
            ),
            ("BTUELLGT",),
        ),
        "water heating": (
            ("out.electricity.hot_water.energy_consumption..kwh",),
            ("BTUELWTH",),
        ),
        "total": (
            ("out.electricity.total.energy_consumption..kwh",),
            ("BTUEL",),
        ),
    },
}


def section_i_other_states() -> None:
    """Check in other states whether the differences found in the study state repeat.

    For each state of ``config.COMPARISON_STATES`` (and for the study state) the
    annual energy per occupied dwelling of each end use of the ResStock release is
    compared with the households of the same state in the 2020 RECS (jackknife
    error), and the total gas and electricity per household with the residential
    totals sold in the state in ``MEASURED_YEAR`` (Energy Information
    Administration). The households of a state are the occupied dwellings that the
    ResStock weights represent. If the shortfall of water heating gas, the excess of
    electric heating and lighting, or the shortfall of the total gas appear in
    most states, they are a property of ResStock and not of the Utah sample.
    The RECS totals of each state are also compared with the totals sold in the year
    of the survey. Tables: ``states_end_use_check``, ``states_total_check`` and
    ``states_recs_vs_totals``.
    """
    results = config.ROOT / "data" / "results" / "tables"
    release = config.RAW / "resstock" / "2025_amy2018_release_1"
    recs_path = config.RAW / "recs_2020" / "recs2020_public_v7.csv"
    rows, totals = [], []
    for state in [config.STATE, *config.COMPARISON_STATES]:
        path = (
            release / f"metadata_annual_{state.lower()}" / f"{state}_upgrade0.parquet"
        )
        for fuel, uses in STATE_USES.items():
            means, households = validation.resstock_end_use_means(
                path, {k: v[0] for k, v in uses.items()}
            )
            recs = validation.recs_end_use_means(
                recs_path, state, {k: v[1] for k, v in uses.items()}
            ).set_index("use")
            for use in uses:
                rows.append(
                    {
                        "state": state,
                        "fuel": fuel,
                        "use": use,
                        "resstock_kwh": means[use],
                        "recs_kwh": recs.loc[use, "mean_kwh"],
                        "recs_se_kwh": recs.loc[use, "se_kwh"],
                        "recs_households_with_use": int(recs.loc[use, "n_with_use"]),
                        "ratio_resstock_over_recs": means[use]
                        / recs.loc[use, "mean_kwh"],
                        "difference_in_errors": (means[use] - recs.loc[use, "mean_kwh"])
                        / recs.loc[use, "se_kwh"],
                    }
                )
            measured = validation.read_eia_residential_totals(
                config.RAW / f"eia_state_totals_{state.lower()}",
                state,
                config.MEASURED_YEAR,
            )
            key = "natural_gas_kwh" if fuel == "natural_gas" else "electricity_kwh"
            net = means["total"]
            if fuel == "electricity":
                pv = validation.resstock_end_use_means(
                    path, {"pv": ("out.electricity.pv.energy_consumption..kwh",)}
                )[0]["pv"]
                net = means["total"] + pv
            totals.append(
                {
                    "state": state,
                    "fuel": fuel,
                    "households_resstock": households,
                    "resstock_total_gwh": net * households / 1e6,
                    "measured_total_gwh": measured[key] / 1e6,
                    "error_percent": 100.0 * (net * households / measured[key] - 1.0),
                }
            )
    table = pd.DataFrame(rows)
    recs_totals = []
    recs_all = pd.read_csv(
        recs_path, usecols=["state_postal", "NWEIGHT", "BTUNG", "BTUEL"]
    )
    for state in [config.STATE, *config.COMPARISON_STATES]:
        sub = recs_all[recs_all["state_postal"] == state]
        sold = validation.read_eia_residential_totals(
            config.RAW / f"eia_state_totals_{state.lower()}", state, config.RECS_YEAR
        )
        recs_totals.append(
            {
                "state": state,
                "gas_error_percent": 100.0
                * (
                    (sub["BTUNG"] * sub["NWEIGHT"]).sum()
                    * _KWH_PER_KBTU
                    / sold["natural_gas_kwh"]
                    - 1.0
                ),
                "electricity_error_percent": 100.0
                * (
                    (sub["BTUEL"] * sub["NWEIGHT"]).sum()
                    * _KWH_PER_KBTU
                    / sold["electricity_kwh"]
                    - 1.0
                ),
            }
        )
    pd.DataFrame(recs_totals).to_csv(
        results / f"states_recs_vs_totals_{STATE}.csv", index=False
    )
    table.to_csv(results / f"states_end_use_check_{STATE}.csv", index=False)
    pd.DataFrame(totals).to_csv(
        results / f"states_total_check_{STATE}.csv", index=False
    )
    summary = table.groupby(["fuel", "use"])[
        ["ratio_resstock_over_recs", "difference_in_errors"]
    ].median()
    logging.info("median over states:\n%s", summary.round(2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    section_a_composition()
    section_b_recs()
    section_c_lead()
    section_d_known_population()
    section_e_sleep_rules()
    section_f_composition_vs_lead()
    section_g_monthly_vs_eia()
    section_h_origin_of_the_difference()
    section_i_other_states()
