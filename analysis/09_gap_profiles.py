"""Hourly profile of the energy gap and of the normative energy of every tract.

Run from the project root: ``python analysis/09_gap_profiles.py``. It needs the
simulations of ``06_resimulation.py`` and the tables of ``07_gap_by_tract.py`` (the
efficiency curves of the heat pump and the decision about the climate correction of
each quantity).

For every tract it writes ``data/processed/gap_profiles_<state>/GEOID=<id>/
profile.parquet`` with 8,760 hourly rows in local standard time and, per dwelling of
the tract (a mean over its dwellings): the useful energy used today, the normative
energy and the gap, by use (kBtu); the electricity of a heat pump that delivers the
gap, the normative energy or the energy used today (kWh); the electricity and gas of
the equipment of the dwelling today and under the normative thermostat (kWh); the
share of the hour in which the comfort band is required, the share of the required
time outside it today and under the normative thermostat, and the outdoor
temperature of the weather station.

The hourly shape follows the climate of the tract, with the method of
``utah_ejpi.hourly`` (the series of the dwelling plus the difference of its response
to the hourly temperature between the temperature of the tract and that of the
weather station of the donor; the temperature of the tract is the hourly series of
the station plus the daily difference of PRISM between the tract and the station). The
useful energy used today and the normative energy are reshaped separately, each with
its own response, and the gap is their difference; the heat pump electricity is
computed from the reshaped series with the efficiency at the temperature of the
tract; the fuel and the electricity of heating and cooling follow the shape of the
useful energy with the efficiency of the dwelling constant in the day. Every
reshaped series keeps the annual sum of the simulation. The shares of the required
hours and of the hours outside the band keep the shape of the station. The annual
total is the one of the tract (with the climate factor when the quantity is
corrected, see section B of ``07_gap_by_tract.py``). The simulated series are
already in local standard time, so they take no clock shift (the series that
ResStock publishes run two hours ahead of it in every county of Utah).
"""

import logging

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import comfort_gap as cg
from utah_ejpi import heat_pump as hp
from utah_ejpi import hourly

STATE = config.STATE.lower()
RUNS = config.INTERIM / f"comfort_gap_resim_{STATE}"
TABLES = config.ROOT / "data" / "results" / "tables"
OUT = config.ROOT / "data" / "processed" / f"gap_profiles_{STATE}"
KWH = 1 / 3.412
FUEL_HEATING = ("g_heat", "p_heat", "o_heat")

#: column of the profile -> (quantity of script 07 whose climate decision it follows,
#: use). A column without quantity takes the plain coefficient.
COLUMNS = {
    "heating_current_kbtu": ("cur_heating", "heating"),
    "cooling_current_kbtu": ("cur_cooling", "cooling"),
    "heating_normative_kbtu": ("norm_heating", "heating"),
    "cooling_normative_kbtu": ("norm_cooling", "cooling"),
    "heating_gap_kbtu": ("gap_heating", "heating"),
    "cooling_gap_kbtu": ("gap_cooling", "cooling"),
    "heat_pump_auxiliary_heating_kwh": ("heat_pump_auxiliary_heating_kwh", "heating"),
    "heat_pump_auxiliary_cooling_kwh": ("heat_pump_auxiliary_cooling_kwh", "cooling"),
    "heat_pump_replacement_heating_kwh": (
        "heat_pump_replacement_heating_kwh",
        "heating",
    ),
    "heat_pump_replacement_cooling_kwh": (
        "heat_pump_replacement_cooling_kwh",
        "cooling",
    ),
    "heat_pump_current_heating_kwh": ("heat_pump_current_heating_kwh", "heating"),
    "heat_pump_current_cooling_kwh": ("heat_pump_current_cooling_kwh", "cooling"),
    "fuel_heating_current_kwh": (None, "heating"),
    "fuel_heating_normative_kwh": (None, "heating"),
    "electricity_current_kwh": (None, "none"),
    "electricity_normative_kwh": (None, "none"),
    "required_share": (None, "none"),
    "preconditioning_share": (None, "none"),
    "outside_current_share": (None, "none"),
    "outside_normative_share": (None, "none"),
    "outdoor_temperature_c": (None, "none"),
}


def dwelling_inputs(building: int) -> dict[str, np.ndarray]:
    """Hourly series of one dwelling, at the weather station (kBtu unless noted)."""
    s = pd.read_parquet(RUNS / "main" / f"{building}.parquet")
    return {
        "heat_b": s["heat_base"].to_numpy(float),
        "heat_n": s["heat_norm"].to_numpy(float),
        "cool_b": (s["cool_base"] - s["cool_unserved_base"]).to_numpy(float),
        "cool_n": s["cool_norm"].to_numpy(float),
        "fuel_b": sum(s[f"{k}_base"].to_numpy(float) for k in FUEL_HEATING),
        "fuel_n": sum(s[f"{k}_norm"].to_numpy(float) for k in FUEL_HEATING),
        "e_heat_b": s["e_heat_base"].to_numpy(float),
        "e_heat_n": s["e_heat_norm"].to_numpy(float),
        "e_cool_b": s["e_cool_base"].to_numpy(float),
        "e_cool_n": s["e_cool_norm"].to_numpy(float),
        "e_fans_b": s["e_fans_base"].to_numpy(float),
        "e_fans_n": s["e_fans_norm"].to_numpy(float),
        "required": s["required_std"].to_numpy(float),
        "pre": s["preheat_std"].to_numpy(float),
        "out_b": (s["out_low_base"] + s["out_high_base"]).to_numpy(float),
        "out_n": (s["out_low_norm"] + s["out_high_norm"]).to_numpy(float),
        "t": s["t_out"].to_numpy(float),
    }


def _reshaped(series, use, station_t, tract_t) -> np.ndarray:
    """Series of one dwelling with the shape of the climate of each tract."""
    response = hourly.fit_temperature_response(series, station_t, use)
    n = len(tract_t)
    if response is None:
        return np.tile(series, (n, 1))
    shape = hourly.reshape_load(series, response, station_t, tract_t)
    total = shape.sum(axis=1, keepdims=True)
    scaled = shape * (series.sum() / np.where(total > 0, total, 1.0))
    return np.where(total > 0, scaled, series[None, :])


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Annual ratio of two series, zero when the denominator is zero."""
    d = denominator.sum()
    return float(numerator.sum() / d) if d > 0 else 0.0


def tract_series(
    building: int, geoids: np.ndarray, county, station_daily, tract_daily, curves
) -> np.ndarray:
    """Columns of the profile for one dwelling in each tract that uses it."""
    x = dwelling_inputs(building)
    n = len(geoids)
    tract_t = hourly.tract_hourly_temperature(
        x["t"], station_daily.loc[county].to_numpy(), tract_daily.loc[geoids].to_numpy()
    )
    hc = _reshaped(x["heat_b"], "heating", x["t"], tract_t)
    hn = _reshaped(x["heat_n"], "heating", x["t"], tract_t)
    cc = _reshaped(x["cool_b"], "cooling", x["t"], tract_t)
    cn = _reshaped(x["cool_n"], "cooling", x["t"], tract_t)
    gap_h, gap_c = hn - hc, cn - cc
    curve_heat, curve_cool = curves
    columns = {
        "heating_current_kbtu": hc,
        "cooling_current_kbtu": cc,
        "heating_normative_kbtu": hn,
        "cooling_normative_kbtu": cn,
        "heating_gap_kbtu": gap_h,
        "cooling_gap_kbtu": gap_c,
    }
    for tag, h, c in (
        ("auxiliary", gap_h, gap_c),
        ("replacement", hn, cn),
        ("current", hc, cc),
    ):
        columns[f"heat_pump_{tag}_heating_kwh"] = (
            hp.electricity_from_useful(h, tract_t, curve_heat) * KWH
        )
        columns[f"heat_pump_{tag}_cooling_kwh"] = (
            hp.electricity_from_useful(c, tract_t, curve_cool) * KWH
        )
    columns["fuel_heating_current_kwh"] = hc * _ratio(x["fuel_b"], x["heat_b"]) * KWH
    columns["fuel_heating_normative_kwh"] = hn * _ratio(x["fuel_n"], x["heat_n"]) * KWH
    columns["electricity_current_kwh"] = (
        hc * _ratio(x["e_heat_b"], x["heat_b"])
        + cc * _ratio(x["e_cool_b"], x["cool_b"])
        + x["e_fans_b"][None, :]
    ) * KWH
    columns["electricity_normative_kwh"] = (
        hn * _ratio(x["e_heat_n"], x["heat_n"])
        + cn * _ratio(x["e_cool_n"], x["cool_n"])
        + x["e_fans_n"][None, :]
    ) * KWH
    for name, key in (
        ("required_share", "required"),
        ("preconditioning_share", "pre"),
        ("outside_current_share", "out_b"),
        ("outside_normative_share", "out_n"),
    ):
        columns[name] = np.tile(x[key], (n, 1))
    columns["outdoor_temperature_c"] = tract_t
    return np.stack([columns[c] for c in COLUMNS])


def coefficient_names() -> dict[str, str]:
    """Coefficient that weights each column: with the climate factor if corrected."""
    decision = pd.read_csv(TABLES / f"known_population_gap_{STATE}.csv").set_index(
        "quantity"
    )["climate_correction_used"]
    names = {}
    for column, (quantity, use) in COLUMNS.items():
        corrected = bool(quantity is not None and decision.get(quantity, False))
        names[column] = f"coef_{use}" if corrected and use != "none" else "coef"
    return names


def build() -> pd.DataFrame:
    """Accumulate the profiles and write one file per tract."""
    weights = pd.read_csv(
        config.INTERIM / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    weights = weights[weights["weight"] > 0]
    donors = pd.read_csv(
        config.INTERIM / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    ).merge(weights[["GEOID", "tenure", "vintage", "building_type"]])
    dwellings = climate_fit.read_dwellings(config.STATE)
    degree_days = pd.read_csv(
        config.INTERIM / f"climate_degree_days_{config.STATE}.csv", dtype={"id": str}
    )
    heating = climate_fit.fit_sensitivity("heating", dwellings, degree_days)
    cooling = climate_fit.fit_sensitivity("cooling", dwellings, degree_days)
    coefficients = hourly.donor_coefficients(
        weights, donors, dwellings, degree_days, degree_days, heating, cooling
    )
    available = {int(p.stem) for p in (RUNS / "main").glob("*.parquet")}
    coefficients = coefficients[coefficients["building_id"].isin(available)]
    curve_heat = pd.read_csv(TABLES / f"heat_pump_cop_heating_{STATE}.csv")
    curve_cool = pd.read_csv(TABLES / f"heat_pump_cop_cooling_{STATE}.csv")
    columns = list(COLUMNS)
    tract_daily, station_daily, station_of = climate_fit.daily_temperatures(dwellings)
    tracts, acc = cg.accumulate_tract_profiles(
        coefficients,
        None,
        columns,
        coefficient_names(),
        tract_series=lambda b, geoids: tract_series(
            b,
            geoids,
            station_of[b],
            station_daily,
            tract_daily,
            (curve_heat, curve_cool),
        ),
    )
    # The coefficients of a tract add to one over the dwellings with a simulation
    # only when all of them have one; rescale the others so that it holds.
    kept = coefficients.groupby("GEOID")["coef"].sum().reindex(tracts)
    acc = acc / kept.to_numpy(dtype=np.float32)[None, :, None]
    hours = pd.date_range("2018-01-01", periods=acc.shape[2], freq="h")
    rows = []
    for t, geoid in enumerate(tracts):
        profile = pd.DataFrame(acc[:, t, :].T, columns=columns)
        profile.insert(0, "timestamp", hours)
        folder = OUT / f"GEOID={geoid}"
        folder.mkdir(parents=True, exist_ok=True)
        profile.to_parquet(folder / "profile.parquet", index=False)
        rows.append(
            {
                "GEOID": geoid,
                **{
                    f"annual_{c}": float(acc[k, t].sum())
                    for k, c in enumerate(columns)
                    if c.endswith(("_kbtu", "_kwh"))
                },
            }
        )
    annual = pd.DataFrame(rows)
    annual.to_csv(TABLES / f"gap_profiles_annual_{STATE}.csv", index=False)
    logging.info("profiles written for %d tracts", len(tracts))
    return annual


def check(annual: pd.DataFrame) -> None:
    """Compare the sum of the hours of a profile with the annual mean of the tract."""
    summary = pd.read_csv(
        TABLES / f"tract_gap_uncertainty_{STATE}.csv", dtype={"GEOID": str}
    )
    mean = summary[summary["group"] == "all"].set_index("GEOID")
    annual = annual.set_index("GEOID")
    pairs = (
        ("heating_current_kbtu", "cur_heating"),
        ("cooling_current_kbtu", "cur_cooling"),
        ("heating_gap_kbtu", "gap_heating"),
        ("cooling_gap_kbtu", "gap_cooling"),
    )
    for column, quantity in pairs:
        ratio = annual[f"annual_{column}"] / mean[quantity].reindex(annual.index)
        logging.info(
            "%s: profile over simulation mean, median %.3f (p10 %.3f, p90 %.3f)",
            column,
            ratio.median(),
            ratio.quantile(0.1),
            ratio.quantile(0.9),
        )


def run() -> None:
    """Build the profiles and compare their totals with the simulations."""
    annual = build()
    check(annual)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
