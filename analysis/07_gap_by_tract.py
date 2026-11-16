"""Energy gap and normative energy of each tract, with their uncertainty.

Run from the project root: ``python analysis/07_gap_by_tract.py``. It needs the
results of ``02_housing_stock.py`` and ``05_uncertainty.py`` and the simulations of
``06_resimulation.py`` (``data/interim/comfort_gap_resim_<state>``).

Section A. One row per donor dwelling with the annual useful energy used today, the
normative energy and the gap, the fuels, the electricity of a heat pump (from a
curve of efficiency by outdoor temperature measured on the simulations of the heat
pump package) and the share of the required hours outside the comfort band. The
curve and the table are saved in ``data/results/tables``.

Section B. Test with a known population of the quantities of section A (each PUMA
left out in turn, ``utah_ejpi.gap_uncertainty.known_population_quantities``). It
decides, for each quantity, whether the climate correction of the tract is applied
(only if it lowers the median error), and estimates the transfer error of each
quantity.

Section C. Simulations of the quantities by tract with the donors resampled and the
transfer error, and their summary (mean, interval, coefficient of variation and
quality class) in ``data/results/tables``.

Section D. Checks of the tract results: agreement of the energy used today with the
estimate of ``03_energy_requirement.py``, stability of the position of the tracts
under the uncertainty, and a leave-one-dwelling-out check of the efficiency curve of
the heat pump against the electricity that the package simulation gives directly.
There are no observations of the gap at tract level, so these checks test the
consistency and the precision of the estimate, not its accuracy.
"""

import logging

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import gap_uncertainty as gu
from utah_ejpi import heat_pump as hp
from utah_ejpi import housing_stock, uncertainty

STATE = config.STATE.lower()
RUNS = config.INTERIM / f"comfort_gap_resim_{STATE}"
TABLES = config.ROOT / "data" / "results" / "tables"
KWH = 1 / 3.412
#: Tolerances of the share of the required hours outside the band, declared as a
#: convention (category II of EN 16798-2: 3 % of the hours of the year and 12 % of
#: those of a month); they are not validated for homes (see the method report).
ANNUAL_TOLERANCE = 0.03
MONTHLY_TOLERANCE = 0.12
MONTH = pd.date_range("2018-01-01", periods=8760, freq="h").month.to_numpy()
FUELS_HEATING = ("g_heat", "p_heat", "o_heat")


def _read(folder: str, building: int) -> pd.DataFrame | None:
    path = RUNS / folder / f"{building}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def cop_curves(buildings: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curves of efficiency by outdoor temperature from the package simulations."""
    heat = {"useful": [], "electricity": [], "temperature": []}
    cool = {"useful": [], "electricity": [], "temperature": []}
    for b in buildings:
        package, main = _read("u4", b), _read("main", b)
        if package is None or main is None:
            continue
        fan_heat, fan_cool = hp.fan_electricity_by_mode(
            package["heat_norm4"], package["cool_norm4"], package["e_fans_norm4"]
        )
        heat["useful"].append(package["heat_norm4"].to_numpy())
        heat["electricity"].append((package["e_heat_norm4"] + fan_heat).to_numpy())
        cool["useful"].append(package["cool_norm4"].to_numpy())
        cool["electricity"].append((package["e_cool_norm4"] + fan_cool).to_numpy())
        for d in (heat, cool):
            d["temperature"].append(main["t_out"].to_numpy())
    return tuple(
        hp.cop_curve(
            np.concatenate(d["useful"]),
            np.concatenate(d["electricity"]),
            np.concatenate(d["temperature"]),
            min_useful=1.0,
        )
        for d in (heat, cool)
    )


def dwelling_table() -> pd.DataFrame:
    """Section A: annual quantities of every donor dwelling."""
    donors = pd.read_csv(config.INTERIM / f"tract_type_donors_{config.STATE}.csv")
    buildings = sorted(donors["building_id"].unique())
    curve_heat, curve_cool = cop_curves(buildings)
    TABLES.mkdir(parents=True, exist_ok=True)
    curve_heat.to_csv(TABLES / f"heat_pump_cop_heating_{STATE}.csv", index=False)
    curve_cool.to_csv(TABLES / f"heat_pump_cop_cooling_{STATE}.csv", index=False)
    rows = []
    for b in buildings:
        s = _read("main", b)
        if s is None:
            continue
        cur_c = s["cool_base"] - s["cool_unserved_base"]
        t = s["t_out"].to_numpy()
        required = s["required_std"].sum()
        row = {
            "building_id": b,
            "cur_heating": s["heat_base"].sum(),
            "cur_cooling": cur_c.sum(),
            "norm_heating": s["heat_norm"].sum(),
            "norm_cooling": s["cool_norm"].sum(),
            "required_hours": required,
            "preconditioning_hours": s["preheat_std"].sum(),
            "low_share_current": s["out_low_base"].sum() / required,
            "high_share_current": s["out_high_base"].sum() / required,
            "low_share_normative": s["out_low_norm"].sum() / required,
            "high_share_normative": s["out_high_norm"].sum() / required,
        }
        for tag, name in (("base", "current"), ("norm", "normative")):
            outside = (s[f"out_low_{tag}"] + s[f"out_high_{tag}"]).to_numpy()
            by_month = np.bincount(MONTH, outside, 13)[1:]
            required_month = np.bincount(MONTH, s["required_std"].to_numpy(), 13)[1:]
            ratio = np.divide(
                by_month, required_month, out=np.zeros(12), where=required_month > 0
            )
            row[f"months_over_tolerance_{name}"] = float(
                (ratio > MONTHLY_TOLERANCE).sum()
            )
            row[f"over_annual_tolerance_{name}"] = float(
                outside.sum() / required > ANNUAL_TOLERANCE
            )
            row[f"degree_hours_{name}"] = (
                s[f"dh_low_{tag}"].sum() + s[f"dh_high_{tag}"].sum()
            )
        row["gap_heating"] = row["norm_heating"] - row["cur_heating"]
        row["gap_cooling"] = row["norm_cooling"] - row["cur_cooling"]
        fuel_base = sum(s[f"{k}_base"].sum() for k in FUELS_HEATING)
        fuel_norm = sum(s[f"{k}_norm"].sum() for k in FUELS_HEATING)
        row["gap_fuel_heating_kwh"] = (fuel_norm - fuel_base) * KWH
        row["gap_electricity_heating_kwh"] = (
            s["e_heat_norm"].sum() - s["e_heat_base"].sum()
        ) * KWH
        row["gap_electricity_cooling_kwh"] = (
            s["e_cool_norm"].sum() + s["e_fans_norm"].sum()
            - s["e_cool_base"].sum() - s["e_fans_base"].sum()
        ) * KWH  # fmt: skip
        gap_h = (s["heat_norm"] - s["heat_base"]).to_numpy()
        gap_c = (s["cool_norm"] - cur_c).to_numpy()
        for tag, series_h, series_c in (
            ("heat_pump_auxiliary", gap_h, gap_c),
            (
                "heat_pump_replacement",
                s["heat_norm"].to_numpy(),
                s["cool_norm"].to_numpy(),
            ),
            ("heat_pump_current", s["heat_base"].to_numpy(), cur_c.to_numpy()),
        ):
            row[f"{tag}_heating_kwh"] = (
                hp.electricity_from_useful(series_h, t, curve_heat).sum() * KWH
            )
            row[f"{tag}_cooling_kwh"] = (
                hp.electricity_from_useful(series_c, t, curve_cool).sum() * KWH
            )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("building_id")
    table.to_csv(TABLES / f"comfort_gap_by_dwelling_{STATE}.csv")
    return table


def uses_of(columns) -> dict[str, str]:
    """Use of every quantity: heating or cooling (lower tail of the band is heating)."""
    return {
        c: "cooling" if ("cooling" in c or c.startswith("high_share")) else "heating"
        for c in columns
    }


def simulate(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sections B and C: known-population test, transfer error and simulations."""
    quantities = table.drop(
        columns=["required_hours", "preconditioning_hours"], errors="ignore"
    )
    uses = uses_of(quantities.columns)
    dwellings = climate_fit.read_dwellings(config.STATE)
    degree_days = pd.read_csv(
        config.INTERIM / f"climate_degree_days_{config.STATE}.csv", dtype={"id": str}
    )
    extra, extra_stations = climate_fit.extra_fit_data(dwellings, degree_days)
    test = gu.known_population_quantities(
        dwellings,
        quantities,
        uses,
        degree_days,
        climate_corrected=list(quantities.columns),
        unit="puma",
        n_draws=config.N_DRAWS,
        seed=config.SEED,
        extra_fit_dwellings=extra,
        extra_fit_stations=extra_stations,
    )
    floor = {q: 0.001 if "_share_" in q else 1.0 for q in quantities.columns}
    rows = []
    corrected = []
    for q in quantities.columns:
        truth = test[f"truth_{q}"]
        scale = truth.abs().where(truth.abs() > floor[q])
        with_ = ((test[f"estimate_{q}"] - truth) / scale).abs().median()
        without = ((test[f"no_climate_{q}"] - truth) / scale).abs().median()
        state = ((test[f"state_{q}"] - truth) / scale).abs().median()
        use_it = bool(with_ < without)
        if use_it:
            corrected.append(q)
        rows.append(
            {"quantity": q, "median_error_with_climate": with_,
             "median_error_without_climate": without, "median_error_state_mean": state,
             "climate_correction_used": use_it}
        )  # fmt: skip
    pd.DataFrame(rows).to_csv(TABLES / f"known_population_gap_{STATE}.csv", index=False)
    test.to_csv(TABLES / f"known_population_gap_units_{STATE}.csv", index=False)
    first = gu.known_population_quantities(
        dwellings,
        quantities,
        uses,
        degree_days,
        climate_corrected=corrected,
        unit="puma",
        n_draws=config.N_DRAWS,
        seed=config.SEED,
        extra_fit_dwellings=extra,
        extra_fit_stations=extra_stations,
    )
    tau = gu.estimate_transfer_tau_quantities(first, quantities.columns, min_abs=floor)
    pd.DataFrame({"quantity": list(tau), "tau": list(tau.values())}).to_csv(
        TABLES / f"transfer_tau_gap_{STATE}.csv", index=False
    )
    logging.info("transfer error: %s", tau)

    weights = pd.read_csv(
        config.INTERIM / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    weights = weights[weights["weight"] > 0]
    donors = pd.read_csv(
        config.INTERIM / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    )
    donors = donors.merge(weights[["GEOID", "tenure", "vintage", "building_type"]])
    census = housing_stock.read_acs_table(
        config.RAW / "acs" / "acs5_2024" / "B25002_tract_us.csv"
    )
    census = census[
        census["GEOID"].str.startswith(config.STATE_FIPS)
        & (census["variable"] == "B25002_002")
    ]
    totals = census.rename(columns={"estimate": "total", "moe": "total_moe"})[
        ["GEOID", "total", "total_moe"]
    ]
    totals = totals[totals["GEOID"].isin(weights["GEOID"])]
    share = pd.read_csv(
        config.INTERIM / f"tract_puma_share_{STATE}.csv", dtype={"GEOID": str}
    )
    main_puma = share.loc[share.groupby("GEOID")["share"].idxmax()].set_index("GEOID")[
        "resstock_puma"
    ]
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    draws = gu.simulate_tract_quantities(
        weights,
        donors,
        dwellings,
        quantities,
        uses,
        totals,
        degree_days,
        degree_days,
        heating,
        cooling,
        climate_corrected=corrected,
        n_draws=config.N_DRAWS,
        seed=config.SEED,
        transfer_tau=tau,
        transfer_groups=main_puma,
    )
    draws.to_parquet(config.INTERIM / f"tract_gap_draws_{STATE}.parquet", index=False)
    summary = uncertainty.summarize_draws(draws, list(quantities.columns))
    summary.to_csv(TABLES / f"tract_gap_uncertainty_{STATE}.csv", index=False)
    return summary, draws


def curve_check(buildings: list[int]) -> pd.DataFrame:
    """Leave-one-dwelling-out check of the heat pump curve against the package run.

    For every dwelling with a package simulation the curve is rebuilt from the bin
    sums of all the other dwellings, applied to its useful energy and compared with
    the electricity that its own simulation used (heating, cooling and fans).
    """
    edges = np.arange(-25.0, 45.0 + 2.5, 2.5)
    sums = {}
    for b in buildings:
        package, main = _read("u4", b), _read("main", b)
        if package is None or main is None:
            continue
        fan_heat, fan_cool = hp.fan_electricity_by_mode(
            package["heat_norm4"], package["cool_norm4"], package["e_fans_norm4"]
        )
        k = np.digitize(main["t_out"].to_numpy(), edges) - 1
        inside = (k >= 0) & (k < len(edges) - 1)
        entry = {"t": main["t_out"].to_numpy()}
        for use, useful, electricity in (
            ("heat", package["heat_norm4"], package["e_heat_norm4"] + fan_heat),
            ("cool", package["cool_norm4"], package["e_cool_norm4"] + fan_cool),
        ):
            n = len(edges) - 1
            entry[use] = {
                "useful": np.bincount(k[inside], useful.to_numpy()[inside], n),
                "electricity": np.bincount(
                    k[inside], electricity.to_numpy()[inside], n
                ),
                "series": useful.to_numpy(),
                "direct": float(electricity.sum()),
            }
        sums[b] = entry
    total = {
        use: {
            q: sum(e[use][q] for e in sums.values()) for q in ("useful", "electricity")
        }
        for use in ("heat", "cool")
    }
    centers = edges[:-1] + 1.25
    rows = []
    for b, entry in sums.items():
        curve_kwh = direct_kwh = 0.0
        for use in ("heat", "cool"):
            useful = total[use]["useful"] - entry[use]["useful"]
            electricity = total[use]["electricity"] - entry[use]["electricity"]
            keep = (useful > 1.0) & (electricity > 0)
            curve = pd.DataFrame(
                {"t_out": centers[keep], "cop": useful[keep] / electricity[keep]}
            )
            curve_kwh += hp.electricity_from_useful(
                entry[use]["series"], entry["t"], curve
            ).sum()
            direct_kwh += entry[use]["direct"]
        rows.append(
            {
                "building_id": b,
                "direct_kwh": direct_kwh * KWH,
                "curve_kwh": curve_kwh * KWH,
            }
        )
    out = pd.DataFrame(rows)
    out["error"] = out["curve_kwh"] / out["direct_kwh"] - 1
    return out


def validate(table: pd.DataFrame, summary: pd.DataFrame, draws: pd.DataFrame) -> None:
    """Section D: consistency, stability of the position and the curve check."""
    previous = pd.read_csv(
        config.INTERIM / f"tract_energy_summary_linear_{STATE}.csv",
        dtype={"GEOID": str},
    ).set_index("GEOID")
    mean = summary[summary["group"] == "all"].set_index("GEOID")
    rows = []
    for new, old in (("cur_heating", "heating_kbtu"), ("cur_cooling", "cooling_kbtu")):
        ratio = mean[new] / previous[old]
        rows.append(
            {
                "quantity": new,
                "median_ratio": ratio.median(),
                "p10_ratio": ratio.quantile(0.1),
                "p90_ratio": ratio.quantile(0.9),
                "spearman": mean[new].corr(previous[old], method="spearman"),
            }
        )
    pd.DataFrame(rows).to_csv(
        TABLES / f"tract_gap_consistency_{STATE}.csv", index=False
    )
    for quantity in ("gap_heating", "gap_cooling", "norm_heating", "norm_cooling"):
        uncertainty.rank_stability(draws, quantity).to_csv(
            TABLES / f"rank_stability_{quantity}_{STATE}.csv", index=False
        )
    check = curve_check(sorted(table.index))
    check.to_csv(TABLES / f"heat_pump_curve_check_{STATE}.csv", index=False)
    if len(check):
        logging.info(
            "curve check: %d dwellings, total error %.1f%%, median abs. error %.1f%%",
            len(check),
            100 * (check["curve_kwh"].sum() / check["direct_kwh"].sum() - 1),
            100 * check["error"].abs().median(),
        )


def run() -> None:
    """Build the table of the dwellings, simulate the tracts and check them."""
    table = dwelling_table()
    logging.info("%d dwellings in the table", len(table))
    summary, draws = simulate(table)
    validate(table, summary, draws)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
