"""Uncertainty of the energy of each tract, as a set of simulations.

Run from the project root: ``python analysis/05_uncertainty.py``. It needs the results
of ``02_housing_stock.py`` and the Census tables of ``01_download_data.py``.

The simulations of the energy of every tract (``config.N_DRAWS``) are drawn for the
two rules of extrapolation of the climate (linear and capped); they are saved in
``data/interim/``. The summaries by tract (mean, interval, coefficient of variation and
quality class) and the stability of the position of the tracts are saved in
``data/results/tables/`` for three levels of the energy: ResStock's own (the high
reference), the heating level that the gas sold in the months without heating leaves
(``config.MEASURED_YEAR``) and a low reference implied by the national survey of Utah
households (RECS). It needs the hourly profiles of the tracts (to take the modeled
consumption of all the uses) and the files of ``download_eia_state_totals`` and
``download_eia_monthly``. With ``--levels-only`` it applies the levels again to the
simulations already saved, without simulating.

The simulations include the transfer error of the dwellings borrowed from other PUMAs
(a PUMA is a Census statistical area of at least 100,000 people). Its deviation is
estimated by leaving each PUMA of ResStock out in turn
(``uncertainty.known_population_test``) and saved in
``data/results/tables/transfer_tau_ut.csv``.
"""

import logging

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import hourly, housing_stock, resstock, uncertainty, validation

STATE = config.STATE.lower()
STATE_FIPS = config.STATE_FIPS
QUANTITIES = [
    "heating_kbtu",
    "cooling_kbtu",
    "heating_kwh",
    "cooling_kwh",
    "persons",
    "dwellings",
]


def _tract_totals() -> pd.DataFrame:
    """Occupied dwellings of each tract and the margin of error of that table."""
    table = housing_stock.read_acs_table(
        config.RAW / "acs" / "acs5_2024" / "B25002_tract_us.csv"
    )
    rows = table[
        table["GEOID"].str.startswith(STATE_FIPS) & (table["variable"] == "B25002_002")
    ]
    return rows.rename(columns={"estimate": "total", "moe": "total_moe"})[
        ["GEOID", "total", "total_moe"]
    ]


def _level_factors() -> dict[str, tuple[float, float]]:
    """Factors of the level of the energy: ResStock, the gas base load and RECS.

    ``resstock`` is the energy as ResStock models it (the high reference);
    ``base_load_implied`` takes the gas heating to the level that the gas sold in
    the months without heating leaves (``validation.base_load_heating_factor``;
    there is no equivalent for cooling, so its factor is 1); ``recs_implied`` takes
    ResStock to the means of RECS (low reference). The factor that would assign the
    whole difference with the totals sold to the heating and to the cooling is
    written to ``level_factors_<state>.csv`` as ``rejected_eia_implied``: the
    monthly pattern does not support it (section H of ``13_validation.py``).
    """
    table = pd.read_csv(
        config.ROOT / "data" / "results" / "tables" / f"recs_comparison_{STATE}.csv"
    )
    row = table[table["group_type"] == "all"].set_index("variable")
    directory = config.RAW / f"eia_state_totals_{STATE}"
    measured = validation.read_eia_residential_totals(
        directory, config.STATE, config.MEASURED_YEAR
    )
    summary = pd.read_csv(
        config.INTERIM / f"tract_energy_summary_linear_{STATE}.csv",
        dtype={"GEOID": str},
    ).set_index("GEOID")["dwellings"]
    profiles = config.ROOT / "data" / "processed" / f"hourly_profiles_{STATE}"
    modeled = hourly.state_annual_means(profiles, summary)
    # The sales to customers are net of the generation of the panels already installed.
    net = {**modeled, "electricity": modeled["electricity"] + modeled["electricity_pv"]}
    eia = validation.measured_level_factors(
        measured, net, config.MEASURED_YEAR_HOUSEHOLDS
    )
    monthly = validation.read_eia_monthly_residential(
        directory, config.STATE, config.MEASURED_YEAR
    )
    state = hourly.state_hourly_means(profiles, summary)
    gas = [c for c in state.columns if c.startswith("natural_gas_")]
    heat = [c for c in gas if "_heating" in c]
    months = state.index.month
    households = float(config.MEASURED_YEAR_HOUSEHOLDS)
    by_month = lambda cols: (  # noqa: E731
        state[cols].sum(axis=1).groupby(months).sum().to_numpy() * households
    )
    base = validation.base_load_heating_factor(
        monthly["natural_gas_kwh"].to_numpy(),
        by_month(heat),
        by_month([c for c in gas if c not in heat]),
    )
    logging.info("measured %s; modeled %s; factors %s %s", measured, modeled, eia, base)
    recs = (
        float(row.loc["heating_kwh", "mean"] / row.loc["heating_kwh", "resstock"]),
        float(row.loc["cooling_kwh", "mean"] / row.loc["cooling_kwh", "resstock"]),
    )
    pd.DataFrame(
        [
            {"reference": "resstock", "heating": 1.0, "cooling": 1.0},
            {
                "reference": "base_load_implied",
                "heating": base["heating_factor"],
                "cooling": 1.0,
            },
            {"reference": "recs_implied", "heating": recs[0], "cooling": recs[1]},
            {
                "reference": "rejected_eia_implied",
                "heating": eia["heating"],
                "cooling": eia["cooling"],
            },
        ]
    ).assign(
        other_gas_uses_factor=[1.0, base["other_uses_factor"], np.nan, np.nan],
        out_of_sample_error=[np.nan, base["out_of_sample_error"], np.nan, np.nan],
    ).to_csv(
        config.ROOT / "data" / "results" / "tables" / f"level_factors_{STATE}.csv",
        index=False,
    )
    return {
        "resstock": (1.0, 1.0),
        "base_load_implied": (base["heating_factor"], 1.0),
        "recs_implied": recs,
    }


def _random_groups(county: pd.Series, size: int, seed: int) -> pd.Series:
    """Split the tracts of each county at random into groups of ``size`` tracts."""
    rng = np.random.default_rng(seed)
    labels = {}
    for code, tracts in county.groupby(county):
        order = rng.permutation(tracts.index.to_numpy())
        for k in range(len(order) // size):
            for geoid in order[k * size : (k + 1) * size]:
                labels[geoid] = f"{code}-{size}-{k}"
    return pd.Series(labels)


def _scale_table(
    draws: pd.DataFrame, main_puma: pd.Series, column: str
) -> pd.DataFrame:
    """Uncertainty of one quantity by geographic unit, from the same simulations.

    Reports, for tracts, PUMAs, counties and random groups of tracts of a county,
    the median over units of the half-width of the 90 % interval and of the
    coefficient of variation, and the share of units that stay in the same tercile
    (quintile for tracts) in at least 80 % of the simulations.
    """
    tracts = pd.Series(draws["GEOID"].unique())
    county = pd.Series(tracts.str[:5].to_numpy(), index=tracts.to_numpy())
    layers = {
        "tract": pd.Series(tracts.to_numpy(), index=tracts.to_numpy()),
        "puma": main_puma.reindex(tracts),
        "county": county,
        **{
            f"{size}_tracts_of_a_county": _random_groups(county, size, config.SEED)
            for size in (5, 10, 25, 50)
        },
    }
    rows = []
    for name, units in layers.items():
        agg = uncertainty.aggregate_draws(draws, units, [column])
        summary = uncertainty.summarize_draws(agg, [column])
        half = (summary[f"{column}_hi"] - summary[f"{column}_lo"]) / 2 / summary[column]
        row = {
            "unit": name,
            "units": len(summary),
            "median_half_width": half.median(),
            "median_cv_percent": summary[f"{column}_cv"].median(),
        }
        if name in ("tract", "puma", "county"):
            classes = 5 if name == "tract" else 3
            stab = uncertainty.rank_stability(agg, column, n_classes=classes)
            row["same_class_share"] = (stab["class_share"] >= 0.8).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _write_levels(rule, draws, factors, main_puma, results) -> None:
    """Scale the simulations to each level reference and write the summaries."""
    for name, (heating_factor, cooling_factor) in factors.items():
        scaled = uncertainty.apply_level_factors(draws, heating_factor, cooling_factor)
        summary = uncertainty.summarize_draws(scaled, QUANTITIES)
        summary.to_csv(
            results / f"tract_energy_uncertainty_{rule}_{name}_{STATE}.csv",
            index=False,
        )
        for quantity in ("heating_kbtu", "cooling_kbtu"):
            stability = uncertainty.rank_stability(scaled, quantity)
            stability.to_csv(
                results / f"rank_stability_{quantity}_{rule}_{name}_{STATE}.csv",
                index=False,
            )
        if rule == "linear" and name == "resstock":
            for quantity in ("heating_kbtu", "cooling_kbtu"):
                scale = _scale_table(scaled, main_puma, quantity)
                scale.to_csv(
                    results / f"uncertainty_by_scale_{quantity}_{STATE}.csv",
                    index=False,
                )
                logging.info("by scale, %s:\n%s", quantity, scale.round(3))
        all_tracts = summary[summary["group"] == "all"]
        logging.info(
            "%s / %s: median coefficient of variation heating %.1f %%, "
            "cooling %.1f %%; quality of heating: %s",
            rule,
            name,
            all_tracts["heating_kbtu_cv"].median(),
            all_tracts["cooling_kbtu_cv"].median(),
            all_tracts["heating_kbtu_quality"].value_counts().to_dict(),
        )


def _main_puma() -> pd.Series:
    """PUMA that holds the largest share of each tract."""
    share = pd.read_csv(
        config.INTERIM / f"tract_puma_share_{STATE}.csv", dtype={"GEOID": str}
    )
    return share.loc[share.groupby("GEOID")["share"].idxmax()].set_index("GEOID")[
        "resstock_puma"
    ]


def levels_only() -> None:
    """Apply the level references again to the simulations already saved."""
    results = config.ROOT / "data" / "results" / "tables"
    factors = _level_factors()
    logging.info("level factors: %s", factors)
    for rule in ("linear", "cap"):
        draws = pd.read_parquet(
            config.INTERIM / f"tract_energy_draws_{rule}_{STATE}.parquet"
        )
        _write_levels(rule, draws, factors, _main_puma(), results)


def run() -> None:
    """Simulate, save the simulations and the summaries."""
    interim = config.INTERIM
    results = config.ROOT / "data" / "results" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    dwellings = resstock.build_resstock_dwellings(
        resstock.read_resstock_annual(
            config.RAW
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
    totals = _tract_totals()
    totals = totals[totals["GEOID"].isin(weights["GEOID"])]
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    dispersion = uncertainty.within_type_dispersion(dwellings)
    logging.info("within-type dispersion: %s", dispersion)
    extra, extra_stations = climate_fit.extra_fit_data(dwellings, degree_days)
    base_test = uncertainty.known_population_test(
        dwellings,
        degree_days,
        unit="puma",
        n_draws=config.N_DRAWS,
        seed=config.SEED,
        extra_fit_dwellings=extra,
        extra_fit_stations=extra_stations,
    )
    transfer_tau = uncertainty.estimate_transfer_tau(base_test)
    pd.DataFrame(
        {"use": list(transfer_tau), "tau": list(transfer_tau.values())}
    ).to_csv(results / f"transfer_tau_{STATE}.csv", index=False)
    logging.info("transfer error deviation: %s", transfer_tau)
    main_puma = _main_puma()
    factors = _level_factors()
    logging.info("level factors: %s", factors)

    for rule in ("linear", "cap"):
        draws = uncertainty.simulate_tract_energy(
            weights,
            donors,
            dwellings,
            totals,
            degree_days,
            degree_days,
            heating,
            cooling,
            dispersion,
            n_draws=config.N_DRAWS,
            seed=config.SEED,
            extrapolation=rule,
            transfer_tau=transfer_tau,
            transfer_groups=main_puma,
        )
        draws.to_parquet(
            interim / f"tract_energy_draws_{rule}_{STATE}.parquet", index=False
        )
        _write_levels(rule, draws, factors, main_puma, results)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if "--levels-only" in sys.argv:
        levels_only()
    else:
        run()
