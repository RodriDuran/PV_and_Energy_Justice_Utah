"""Sensitivity of the energy gap to the parameters of the normative thermostat.

Run from the project root: ``python analysis/08_sensitivities.py``. It needs the
simulations of ``06_resimulation.py`` (sections B and C) and the table of the
dwellings of ``07_gap_by_tract.py``.

Section A. Effect of each variant of the thermostat (window of one person, one and
five hours of pre-conditioning, a minimum heating of 20 degrees, a maximum cooling
of 25 and 27 degrees) on the mean gap per dwelling of the state, with the dwellings
weighted by the number of dwellings of the state that each one stands for. The
variants were simulated on a sample, so the intervals (90 %, bootstrap over the
sampled dwellings) are those of a sample of the dwellings. The share of the required
hours outside the band is computed from the hourly mean temperatures and is meant to
compare the variants, not to be compared with the share of the table of section A of
``07_gap_by_tract.py`` (computed by time step).

Section B. Sensitivity to the dwellings that have an evaporative cooler: ResStock
does not model evaporative coolers, so a dwelling that has one is simulated with no
cooling and receives the gap of a standard air conditioner. The mean gap of cooling
is recomputed when a share of the dwellings without cooling equipment is taken to
have a cooler that already serves them (``config.EVAPORATIVE_SHARES``).

Section C. Fidelity of the baseline simulation of the heat pump package against the
loads and electricity that ResStock publishes for the package.
"""

import logging

import numpy as np
import pandas as pd

import config
from utah_ejpi import resstock

STATE = config.STATE.lower()
RUNS = config.INTERIM / f"comfort_gap_resim_{STATE}"
TABLES = config.ROOT / "data" / "results" / "tables"
RELEASE = config.RAW / "resstock" / "2025_amy2018_release_1"
VARIANTS = ["single_window", "lead1", "lead5", "floor20", "ceil25", "ceil27"]
KWH = 1 / 3.412


def dwelling_weights() -> pd.Series:
    """Dwellings of the state that each donor dwelling stands for."""
    weights = pd.read_csv(
        config.INTERIM / f"tract_type_weights_{STATE}.csv", dtype={"GEOID": str}
    )
    donors = pd.read_csv(
        config.INTERIM / f"tract_type_donors_{STATE}.csv", dtype={"GEOID": str}
    )
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    merged = donors.merge(weights[[*keys, "weight"]], on=keys)
    merged["dwellings"] = merged["weight"] * merged["donor_weight"]
    return merged.groupby("building_id")["dwellings"].sum()


#: Band of every variant (floor, ceiling) in degrees Celsius; the others use 18 and 26.
BANDS = {"floor20": (20.0, 26.0), "ceil25": (18.0, 25.0), "ceil27": (18.0, 27.0)}


def _out_share(
    temperature: pd.Series,
    required: pd.Series,
    band: tuple[float, float] = (18.0, 26.0),
) -> float:
    """Share of the required hours outside the band, from hourly mean temperatures."""
    outside = (temperature < band[0] - 0.2) | (temperature > band[1] + 0.2)
    return float((outside * required).sum() / required.sum())


def variant_table() -> pd.DataFrame:
    """Section A: per-dwelling annual gap and share outside the band by variant."""
    rows = []
    for path in sorted((RUNS / "variants").glob("*.parquet")):
        building = int(path.stem)
        main = pd.read_parquet(RUNS / "main" / f"{building}.parquet")
        variants = pd.read_parquet(path)
        current_c = (main["cool_base"] - main["cool_unserved_base"]).sum()
        row = {"building_id": building}
        for name in ["main", *VARIANTS]:
            if name == "main":
                heat, cool = main["heat_norm"].sum(), main["cool_norm"].sum()
                temp, req = main["t_norm"], main["required_std"]
            elif f"heat_{name}" in variants:
                heat, cool = (
                    variants[f"heat_{name}"].sum(),
                    variants[f"cool_{name}"].sum(),
                )
                temp, req = variants[f"t_{name}"], variants[f"req_{name}"]
            else:
                continue
            row[f"gap_heating_{name}"] = heat - main["heat_base"].sum()
            row[f"gap_cooling_{name}"] = cool - current_c
            row[f"out_share_{name}"] = _out_share(
                temp, req, BANDS.get(name, (18.0, 26.0))
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("building_id")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Mean of the values with the weights of the same index."""
    w = weights.reindex(values.index).fillna(0.0)
    return float((values * w).sum() / w.sum())


def summarize_variants(table: pd.DataFrame, weights: pd.Series, n_boot: int = 500):
    """Section A: effect of every variant on the weighted mean gap of the state."""
    rng = np.random.default_rng(config.SEED)
    rows = []
    for name in VARIANTS:
        if f"gap_heating_{name}" not in table:
            continue
        row = {
            "variant": name,
            "dwellings": int(table[f"gap_heating_{name}"].notna().sum()),
        }
        for use in ("heating", "cooling"):
            both = table[[f"gap_{use}_{name}", f"gap_{use}_main"]].dropna()
            for label, col in (
                ("variant", f"gap_{use}_{name}"),
                ("main", f"gap_{use}_main"),
            ):
                row[f"mean_gap_{use}_{label}"] = weighted_mean(both[col], weights)
            ratio = []
            for _ in range(n_boot):
                pick = rng.choice(both.index.to_numpy(), len(both))
                sub = both.loc[pick]
                w = weights.reindex(pick).fillna(0.0).to_numpy()
                a = (sub[f"gap_{use}_{name}"].to_numpy() * w).sum()
                b = (sub[f"gap_{use}_main"].to_numpy() * w).sum()
                ratio.append(a / b if b else np.nan)
            row[f"ratio_{use}"] = (
                row[f"mean_gap_{use}_variant"] / row[f"mean_gap_{use}_main"]
            )
            row[f"ratio_{use}_lo"], row[f"ratio_{use}_hi"] = np.nanpercentile(
                ratio, [5, 95]
            )
        row["out_share_variant"] = weighted_mean(
            table[f"out_share_{name}"].dropna(), weights
        )
        row["out_share_main"] = weighted_mean(table["out_share_main"].dropna(), weights)
        rows.append(row)
    return pd.DataFrame(rows)


def evaporative(weights: pd.Series) -> pd.DataFrame:
    """Section B: mean cooling gap if some dwellings without cooling have a cooler."""
    gaps = pd.read_csv(TABLES / f"comfort_gap_by_dwelling_{STATE}.csv").set_index(
        "building_id"
    )
    raw = resstock.read_resstock_annual(
        RELEASE / f"metadata_annual_{STATE}" / f"{config.STATE}_upgrade0.parquet"
    ).set_index("bldg_id")
    none = raw["in.hvac_cooling_type"].reindex(gaps.index) == "None"
    w = weights.reindex(gaps.index).fillna(0.0)
    total = (gaps["gap_cooling"] * w).sum() / w.sum()
    in_none = (gaps["gap_cooling"] * w * none).sum() / w.sum()
    rows = []
    for share in config.EVAPORATIVE_SHARES:
        rows.append(
            {
                "share_of_dwellings_without_cooling_with_cooler": share,
                "mean_gap_cooling": total - share * in_none,
                "change": -share * in_none / total,
                "gap_from_dwellings_without_cooling": in_none / total,
            }
        )
    return pd.DataFrame(rows)


def package_fidelity() -> pd.DataFrame:
    """Section C: baseline of the package against the published package results."""
    published = pd.read_parquet(
        RELEASE / f"metadata_annual_{STATE}" / f"{config.STATE}_upgrade4.parquet"
    ).set_index("bldg_id")
    rows = []
    for path in sorted((RUNS / "variants").glob("*.parquet")):
        building = int(path.stem)
        variants = pd.read_parquet(path)
        if "heat_base4" not in variants or building not in published.index:
            continue
        p = published.loc[building]
        electricity_heat = (
            p["out.electricity.heating.energy_consumption..kwh"]
            + p["out.electricity.heating_hp_bkup.energy_consumption..kwh"]
            + p["out.electricity.heating_hp_bkup_fa.energy_consumption..kwh"]
        )
        rows.append(
            {
                "building_id": building,
                "published_heating_kbtu": p["out.load.heating.energy_delivered..kbtu"],
                "simulated_heating_kbtu": variants["heat_base4"].sum(),
                "published_cooling_kbtu": p["out.load.cooling.energy_delivered..kbtu"],
                "simulated_coil_cooling_kbtu": variants["coilc_base4"].sum(),
                "published_electricity_heating_kwh": electricity_heat,
                "simulated_electricity_heating_kwh": variants["e_heat_base4"].sum()
                * KWH,
                "published_electricity_cooling_kwh": p[
                    "out.electricity.cooling.energy_consumption..kwh"
                ],
                "simulated_electricity_cooling_kwh": variants["e_cool_base4"].sum()
                * KWH,
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    """Run the three sections and write the tables."""
    logging.basicConfig(level=logging.INFO)
    TABLES.mkdir(parents=True, exist_ok=True)
    weights = dwelling_weights()
    table = variant_table()
    logging.info("%d sampled dwellings with variants", len(table))
    if len(table):
        table.to_csv(TABLES / f"sensitivity_by_dwelling_{STATE}.csv")
        summarize_variants(table, weights).to_csv(
            TABLES / f"sensitivity_thermostat_{STATE}.csv", index=False
        )
    evaporative(weights).to_csv(
        TABLES / f"sensitivity_evaporative_{STATE}.csv", index=False
    )
    fidelity = package_fidelity()
    fidelity.to_csv(TABLES / f"heat_pump_package_fidelity_{STATE}.csv", index=False)


if __name__ == "__main__":
    run()
