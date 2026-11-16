"""Diagnostic: ResStock cells (tenure x vintage x type) left empty in each PUMA.

Compares the 72 cells of the Census table of tenure by year built by units in
structure with the occupied ResStock dwellings of each PUMA of the state.
Output: data/interim/empty_cells_by_puma_<state>.csv.
"""

import re

import pandas as pd
import requests

import config

STATE = config.STATE.lower()

R = (
    "data/raw/resstock/2025_amy2018_release_1/"
    f"metadata_annual_{STATE}/{config.STATE}_upgrade0.parquet"
)
d = pd.read_parquet(
    R,
    columns=[
        "in.vacancy_status",
        "in.tenure",
        "in.vintage",
        "in.geometry_building_type_acs",
        "in.heating_fuel",
        "in.puma",
        "weight",
    ],
)
d = d[d["in.vacancy_status"] == "Occupied"].copy()
V = {
    "<1940": "1939 or earlier",
    "1940s": "1940 to 1959",
    "1950s": "1940 to 1959",
    "1960s": "1960 to 1979",
    "1970s": "1960 to 1979",
    "1980s": "1980 to 1999",
    "1990s": "1980 to 1999",
    "2000s": "2000 to 2019",
    "2010s": "2000 to 2019",
}
T = {
    "Single-Family Detached": "1",
    "Single-Family Attached": "1",
    "2 Unit": "2 to 4",
    "3 or 4 Unit": "2 to 4",
    "5 to 9 Unit": "5 to 19",
    "10 to 19 Unit": "5 to 19",
    "20 to 49 Unit": "20 to 49",
    "50 or more Unit": "50 or more",
    "Mobile Home": "mobile",
}
d["ten"] = d["in.tenure"]
d["vin"] = d["in.vintage"].map(V)
d["typ"] = d["in.geometry_building_type_acs"].map(T)
assert d[["ten", "vin", "typ"]].notna().all().all()
d["cell"] = d.ten + "|" + d.vin + "|" + d.typ
# ACS cells
lab = requests.get(
    "https://api.census.gov/data/2024/acs/acs5/groups/B25127.json"
).json()["variables"]
cells = {}
for k, v in lab.items():
    m = re.fullmatch(r"B25127_(\d+)E", k)
    if not m:
        continue
    parts = [p for p in v["label"].split("!!")][1:]
    if len(parts) == 4:
        ten = parts[1].split()[0]
        vin = parts[2].replace("Built ", "").rstrip(":")
        typ = parts[3]
        typ = {
            "1, detached or attached": "1",
            "Mobile home, boat, RV, van, etc.": "mobile",
        }.get(typ, typ)
        cells[k[:-1]] = f"{ten}|{vin}|{typ}"
a = pd.read_csv("data/raw/acs/acs5_2024/B25127_tract_us.csv", dtype={"GEOID": str})
a = a[a.GEOID.str.startswith(config.STATE_FIPS) & a.variable.isin(cells)].copy()
a["cell"] = a.variable.map(cells)
m = pd.read_csv(f"data/interim/tract_master_{STATE}.csv", dtype=str)
a = a.merge(m[["GEOID", "resstock_puma"]], on="GEOID")
print("ACS occupied (leaf sum):", int(a.estimate.sum()))
all_cells = sorted(set(a.cell))
print("ACS cells:", len(all_cells), "ResStock cells present:", d.cell.nunique())


rows = []
pumas = sorted(d["in.puma"].unique())
a["pum"] = a.resstock_puma
tot = a.estimate.sum()
rows = []
for p in pumas:
    g = d[d["in.puma"] == p]
    present = set(g.cell)
    ap = a[a.pum == p]
    t_ = ap.estimate.sum()
    miss_dw = ap[~ap.cell.isin(present)].estimate.sum()
    cnt = g.cell.value_counts()
    low = ap[ap.cell.map(cnt).fillna(0) <= 2].estimate.sum()
    rows.append(
        dict(
            puma=p,
            modeled=len(g),
            present=len(present),
            empty=len(set(all_cells) - present),
            acs_dw=int(t_),
            pct_acs_dw_in_empty=round(100 * miss_dw / t_, 1),
            pct_acs_dw_in_le2=round(100 * low / t_, 1),
            ntr=ap.GEOID.nunique(),
        )
    )
r = pd.DataFrame(rows)
print(r.to_string())
print(
    "ACS cells with 2020+ :",
    sum(1 for c in all_cells if "2020 or later" in c),
    len(all_cells),
)
b = a[a.cell.str.contains("2020 or later")].estimate.sum()
print("share built 2020+: %.1f%%" % (100 * b / tot))
a2 = a[~a.cell.str.contains("2020 or later")]
miss = 0
le2 = 0
for p in pumas:
    present = set(d[d["in.puma"] == p].cell)
    cnt = d[d["in.puma"] == p].cell.value_counts()
    ap = a2[a2.pum == p]
    miss += ap[~ap.cell.isin(present)].estimate.sum()
    le2 += ap[ap.cell.map(cnt).fillna(0) <= 2].estimate.sum()
print(
    "excluding 2020+: dwellings in empty cells "
    f"{100 * miss / a2.estimate.sum():.1f}%, in cells with <=2 modeled "
    f"{100 * le2 / a2.estimate.sum():.1f}%"
)
print("cells per PUMA present/empty: median", r.present.median(), r["empty"].median())
r.to_csv(f"data/interim/empty_cells_by_puma_{STATE}.csv", index=False)
