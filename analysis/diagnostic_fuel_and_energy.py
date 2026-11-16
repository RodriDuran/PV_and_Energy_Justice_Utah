"""Diagnostic: dwelling energy in ResStock and what a linking method must cope with.

Part 1 computes annual heating and cooling consumption per modeled dwelling
and summarizes it by heating fuel. Part 2 estimates how many dwellings fall in
combinations (tenure x year built x structure type x heating fuel) with no or
few modeled dwellings in their PUMA. Part 3 compares, by cross-validation,
how well simple and richer predictors reproduce the dwelling energy.

Outputs in data/interim/: resstock_dwelling_energy_<state>.csv,
fuel_cells_by_puma_<state>.csv, link_method_cv_<state>.csv.
"""

import re

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import config

STATE = config.STATE.lower()

RAW = "data/raw/"
RS = (
    RAW
    + "resstock/2025_amy2018_release_1/"
    + f"metadata_annual_{STATE}/{config.STATE}_upgrade0.parquet"
)
OUT = "data/interim/"

KWH = "..kwh"
COLS = {
    "heat_elec": [
        "out.electricity.heating.energy_consumption" + KWH,
        "out.electricity.heating_hp_bkup.energy_consumption" + KWH,
        "out.electricity.heating_hp_bkup_fa.energy_consumption" + KWH,
        "out.electricity.heating_fans_pumps.energy_consumption" + KWH,
    ],
    "heat_gas": [
        "out.natural_gas.heating.energy_consumption" + KWH,
        "out.natural_gas.heating_hp_bkup.energy_consumption" + KWH,
    ],
    "heat_propane": [
        "out.propane.heating.energy_consumption" + KWH,
        "out.propane.heating_hp_bkup.energy_consumption" + KWH,
    ],
    "heat_oil": [
        "out.fuel_oil.heating.energy_consumption" + KWH,
        "out.fuel_oil.heating_hp_bkup.energy_consumption" + KWH,
    ],
    "cool_elec": [
        "out.electricity.cooling.energy_consumption" + KWH,
        "out.electricity.cooling_fans_pumps.energy_consumption" + KWH,
    ],
}
LOADS = {
    "load_heat_kbtu": "out.load.heating.energy_delivered..kbtu",
    "load_cool_kbtu": "out.load.cooling.energy_delivered..kbtu",
}
VINTAGE = {
    "<1940": "1939 or earlier",
    "1940s": "1940-1959",
    "1950s": "1940-1959",
    "1960s": "1960-1979",
    "1970s": "1960-1979",
    "1980s": "1980-1999",
    "1990s": "1980-1999",
    "2000s": "2000-2019",
    "2010s": "2000-2019",
}
TYPE = {
    "Single-Family Detached": "1 unit",
    "Single-Family Attached": "1 unit",
    "2 Unit": "2-4",
    "3 or 4 Unit": "2-4",
    "5 to 9 Unit": "5-19",
    "10 to 19 Unit": "5-19",
    "20 to 49 Unit": "20-49",
    "50 or more Unit": "50+",
    "Mobile Home": "mobile",
}
FUEL = {
    "Natural Gas": "utility gas",
    "Propane": "bottled gas",
    "Electricity": "electricity",
    "Wood": "wood",
    "Other Fuel": "other",
    "None": "none",
}
ACS_FUEL = {
    "Utility gas": "utility gas",
    "Bottled or tank gas (propane, butane, etc.)": "bottled gas",
    "Electricity": "electricity",
    "Fuel oil, kerosene, etc.": "other",
    "Coal or coke": "other",
    "Wood": "wood",
    "Solar energy": "other",
    "Other fuel": "other",
    "No fuel used": "none",
}


def dwelling_table() -> pd.DataFrame:
    """Return ResStock dwellings with grouped characteristics and energy."""
    want = [
        "bldg_id",
        "weight",
        "in.vacancy_status",
        "in.puma",
        "in.tenure",
        "in.vintage",
        "in.geometry_building_type_acs",
        "in.heating_fuel",
        "in.heating_setpoint",
        "in.hvac_cooling_type",
    ]
    cols = want + sum(COLS.values(), []) + list(LOADS.values())
    d = pd.read_parquet(RS, columns=cols)
    out = d[want].rename(columns=lambda c: c.replace("in.", ""))
    for name, parts in COLS.items():
        out[name] = d[parts].sum(axis=1)
    for name, col in LOADS.items():
        out[name] = d[col]
    out["heat_total"] = out[["heat_elec", "heat_gas", "heat_propane", "heat_oil"]].sum(
        axis=1
    )
    out["occupied"] = out["vacancy_status"] == "Occupied"
    out["ten"] = out["tenure"]
    out["vin"] = out["vintage"].map(VINTAGE)
    out["typ"] = out["geometry_building_type_acs"].map(TYPE)
    out["fuel"] = out["heating_fuel"].map(FUEL)
    return out


def acs_cells(master: pd.DataFrame):
    """ACS dwellings per PUMA in cells of tenure x vintage x type, and fuel shares."""
    g = requests.get(
        "https://api.census.gov/data/2024/acs/acs5/groups/B25127.json", timeout=60
    ).json()["variables"]
    cells = {}
    for k, v in g.items():
        if re.fullmatch(r"B25127_\d+E", k):
            p = v["label"].split("!!")[1:]
            if len(p) == 4:
                typ = {
                    "1, detached or attached": "1 unit",
                    "2 to 4": "2-4",
                    "5 to 19": "5-19",
                    "20 to 49": "20-49",
                    "50 or more": "50+",
                }.get(p[3], "mobile")
                vin = p[2].replace("Built ", "").rstrip(":").replace(" to ", "-")
                cells[k[:-1]] = (p[1].split()[0], vin, typ)
    a = pd.read_csv(RAW + "acs/acs5_2024/B25127_tract_us.csv", dtype={"GEOID": str})
    a = a[a.GEOID.str.startswith(config.STATE_FIPS) & a.variable.isin(cells)].copy()
    a[["ten", "vin", "typ"]] = pd.DataFrame(
        a.variable.map(cells).tolist(), index=a.index
    )
    a = a.merge(master[["GEOID", "resstock_puma"]], on="GEOID")
    h = requests.get(
        "https://api.census.gov/data/2024/acs/acs5/groups/B25117.json", timeout=60
    ).json()["variables"]
    fuels = {}
    for k, v in h.items():
        if re.fullmatch(r"B25117_\d+E", k):
            p = v["label"].split("!!")[1:]
            if len(p) == 3:
                fuels[k[:-1]] = (p[1].split()[0], ACS_FUEL[p[2]])
    f = pd.read_csv(RAW + "acs/acs5_2024/B25117_tract_us.csv", dtype={"GEOID": str})
    f = f[f.GEOID.str.startswith(config.STATE_FIPS) & f.variable.isin(fuels)].copy()
    f[["ten", "fuel"]] = pd.DataFrame(f.variable.map(fuels).tolist(), index=f.index)
    f = f.merge(master[["GEOID", "resstock_puma"]], on="GEOID")
    return a, f


def empty_fuel_cells(d: pd.DataFrame, a: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Compute the share of dwellings in cells with 0 or <=2 modeled dwellings.

    The joint distribution with fuel is built assuming that, within tenure,
    heating fuel is independent of year built and structure type (the Census
    tables do not cross them), so the shares are an approximation.
    """
    a = a[a.vin != "2020 or later"]
    rows = []
    for p, g in d[d.occupied].groupby("puma"):
        cnt3 = g.groupby(["ten", "vin", "typ"]).size()
        cnt4 = g.groupby(["ten", "vin", "typ", "fuel"]).size()
        ap = a[a.resstock_puma == p].groupby(["ten", "vin", "typ"]).estimate.sum()
        fp = f[f.resstock_puma == p].groupby(["ten", "fuel"]).estimate.sum()
        pf = fp / fp.groupby(level="ten").transform("sum")
        e0 = e2 = e0f = e2f = tot = 0.0
        for (ten, vin, typ), n in ap.items():
            c3 = cnt3.get((ten, vin, typ), 0)
            tot += n
            e0 += n * (c3 == 0)
            e2 += n * (c3 <= 2)
            for fu, share in pf.loc[ten].items():
                c4 = cnt4.get((ten, vin, typ, fu), 0)
                e0f += n * share * (c4 == 0)
                e2f += n * share * (c4 <= 2)
        rows.append(
            dict(
                puma=p,
                modeled_occupied=len(g),
                pct_empty_3=100 * e0 / tot,
                pct_le2_3=100 * e2 / tot,
                pct_empty_4=100 * e0f / tot,
                pct_le2_4=100 * e2f / tot,
            )
        )
    return pd.DataFrame(rows)


def cv_methods(d: pd.DataFrame, target: str, seed: int = 1) -> dict:
    """Cross-validated R2 of predictors of one energy variable (occupied dwellings)."""
    x = d[d.occupied].reset_index(drop=True)
    y = x[target].to_numpy()
    sse = {m: 0.0 for m in ("puma", "cell", "cell_fuel", "additive", "boosting")}
    enc = OneHotEncoder(handle_unknown="ignore")
    feats = ["puma", "ten", "vin", "typ", "fuel"]
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        a, b = x.iloc[tr], x.iloc[te]
        gm = a[target].mean()
        pred = {}
        pred["puma"] = b.puma.map(a.groupby("puma")[target].mean()).fillna(gm)
        tt = a.groupby(["ten", "typ"])[target].mean()
        c3 = a.groupby(["ten", "vin", "typ"])[target].mean()
        base = pd.Series(
            [tt.get((t, ty), gm) for t, ty in zip(b.ten, b.typ, strict=False)],
            index=b.index,
        )
        pred["cell"] = pd.Series(
            [c3.get(k, np.nan) for k in zip(b.ten, b.vin, b.typ, strict=False)],
            index=b.index,
        ).fillna(base)
        c4 = a.groupby(["ten", "vin", "typ", "fuel"])[target].mean()
        pred["cell_fuel"] = pd.Series(
            [c4.get(k, np.nan) for k in zip(b.ten, b.vin, b.typ, b.fuel, strict=False)],
            index=b.index,
        ).fillna(pred["cell"])
        xa = enc.fit_transform(a[feats])
        pred["additive"] = (
            Ridge(alpha=3.0).fit(xa, a[target]).predict(enc.transform(b[feats]))
        )
        xo = pd.get_dummies(x[feats]).astype(float)
        pred["boosting"] = (
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, random_state=0
            )
            .fit(xo.iloc[tr], a[target])
            .predict(xo.iloc[te])
        )
        for m, p in pred.items():
            sse[m] += float(((b[target].to_numpy() - np.asarray(p)) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {m: 1 - s / sst for m, s in sse.items()}


def main() -> None:
    """Run the three parts and write the outputs."""
    d = dwelling_table()
    d.to_csv(OUT + f"resstock_dwelling_energy_{STATE}.csv", index=False)
    occ = d[d.occupied]
    print("Occupied dwellings:", len(occ), "of", len(d))
    summ = (
        occ.groupby("fuel")
        .agg(
            n=("bldg_id", "size"),
            heat_total=("heat_total", "mean"),
            heat_elec=("heat_elec", "mean"),
            heat_gas=("heat_gas", "mean"),
            heat_propane=("heat_propane", "mean"),
            cool_elec=("cool_elec", "mean"),
            load_heat_kbtu=("load_heat_kbtu", "mean"),
            load_cool_kbtu=("load_cool_kbtu", "mean"),
        )
        .round(0)
    )
    print(
        "\nMean annual kWh per occupied dwelling, by heating fuel:\n", summ.to_string()
    )
    master = pd.read_csv(OUT + f"tract_master_{STATE}.csv", dtype=str)
    a, f = acs_cells(master)
    r = empty_fuel_cells(d, a, f)
    r.round(1).to_csv(OUT + f"fuel_cells_by_puma_{STATE}.csv", index=False)
    print("\nShare of dwellings (percent) in cells with no / <=2 modeled dwellings")
    print(
        "  without fuel: "
        f"empty {r.pct_empty_3.mean():.1f}, <=2 {r.pct_le2_3.mean():.1f}"
    )
    print(
        "  with fuel   : "
        f"empty {r.pct_empty_4.mean():.1f}, <=2 {r.pct_le2_4.mean():.1f}"
    )
    print(
        "  with fuel by PUMA, empty: "
        f"min {r.pct_empty_4.min():.1f} max {r.pct_empty_4.max():.1f}"
    )
    res = {
        t: cv_methods(d, t)
        for t in ("heat_total", "cool_elec", "load_heat_kbtu", "load_cool_kbtu")
    }
    cv = pd.DataFrame(res).T.round(3)
    cv.to_csv(OUT + f"link_method_cv_{STATE}.csv")
    print("\nCross-validated R2 (5 folds, occupied dwellings):\n", cv.to_string())


if __name__ == "__main__":
    main()
