"""Tools shared by the scripts that simulate the ResStock dwellings again.

Download of the published EnergyPlus models, preparation of a model and its weather
file, runs of EnergyPlus at low priority, reading of the outputs and the normative
thermostat files. They are used by ``06_resimulation.py`` and by
``10_climate_shift.py``.
"""

import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
from utah_ejpi import comfort_gap as cg
from utah_ejpi import resimulation as rs

STATE = config.STATE.lower()
RELEASE = config.RAW / "resstock" / "2025_amy2018_release_1"
MODELS_URL = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/"
    "building_energy_models"
)
J_TO_KBTU = 1 / 1.055056e6
STEPS = 35040
BELOW_NORMAL = 0x00004000
METERS = {
    "e_heat": "Heating:Electricity",
    "e_cool": "Cooling:Electricity",
    "e_fans": "Fans:Electricity",
    "g_heat": "Heating:NaturalGas",
    "p_heat": "Heating:Propane",
    "o_heat": "Heating:FuelOilNo2",
}


def run_energyplus(idf: Path, epw: Path, outdir: Path, retries: int = 3) -> int:
    """Run EnergyPlus at low priority; retry, because runs fail at random under load."""
    outdir = Path(outdir).resolve()
    idf, epw = Path(idf).resolve(), Path(epw).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    exe = Path(config.ENERGYPLUS_DIR) / "energyplus.exe"
    code = 1
    for wait in range(1, retries + 1):
        proc = subprocess.Popen(
            [str(exe), "-w", str(epw), "-d", str(outdir), "-r", str(idf)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=BELOW_NORMAL if os.name == "nt" else 0,
            cwd=str(outdir),
        )
        code = proc.wait(timeout=1800)
        if code == 0:
            return 0
        time.sleep(20 * wait)
    return code


def hourly_sum(values, how="sum") -> np.ndarray:
    """Add (or average) the four 15-minute values of each hour of a year."""
    block = np.asarray(values, float).reshape(8760, 4)
    return block.sum(1) if how == "sum" else block.mean(1)


def fetch_model(upgrade: int, building: int, work: Path) -> Path:
    """Download and unpack the model of a dwelling for an upgrade."""
    url = f"{MODELS_URL}/upgrade={upgrade}/bldg{building:07d}-up{upgrade:02d}.zip"
    target = work / f"model{upgrade}"
    target.mkdir(parents=True, exist_ok=True)
    path = work / f"model{upgrade}.zip"
    for _ in range(3):
        try:
            reply = requests.get(url, timeout=120)
            reply.raise_for_status()
            path.write_bytes(reply.content)
            break
        except requests.RequestException:
            time.sleep(5)
    with zipfile.ZipFile(path) as z:
        z.extractall(target)
    return target


def prepare_model_dir(model: Path, county: str, work: Path) -> Path:
    """Write the prepared model next to the schedules; build the weather file."""
    idf = (model / "in.idf").read_text(encoding="utf-8", errors="replace")
    place = re.search(
        r"Site:Location,\s*([^,]*),[^\n]*\n\s*([-\d.]+),[^\n]*\n\s*([-\d.]+),"
        r"[^\n]*\n\s*([-\d.]+),[^\n]*\n\s*([-\d.]+),",
        idf,
    )
    assert place, "Site:Location not found"
    epw = work / "weather.epw"
    if not epw.exists():
        table = pd.read_csv(RELEASE / f"weather_{STATE}" / f"{county}_2018.csv")
        epw.write_text(
            rs.build_weather_epw(
                table,
                place.group(1),
                float(place.group(2)),
                float(place.group(3)),
                float(place.group(4)),
                float(place.group(5)),
                state_code=config.STATE,
            ),
            encoding="utf-8",
        )
    (model / "run.idf").write_text(rs.prepare_model(idf), encoding="utf-8")
    return epw


def summarize_run(frame: pd.DataFrame, tag: str) -> dict[str, np.ndarray]:
    """Hourly useful energy, temperature and meters (kBtu) of one simulation."""
    zone = "CONDITIONED SPACE:Zone Air System Sensible {} Energy [J](TimeStep)"
    out = {
        f"heat_{tag}": hourly_sum(frame[zone.format("Heating")] * J_TO_KBTU),
        f"cool_{tag}": hourly_sum(frame[zone.format("Cooling")] * J_TO_KBTU),
        f"t_{tag}": hourly_sum(
            frame["CONDITIONED SPACE:Zone Mean Air Temperature [C](TimeStep)"], "mean"
        ),
    }
    for key, meter in METERS.items():
        column = f"{meter} [J](TimeStep)"
        values = frame[column] if column in frame else np.zeros(len(frame))
        out[f"{key}_{tag}"] = hourly_sum(np.asarray(values) * J_TO_KBTU)
    coils = [
        c
        for c in frame.columns
        if "Cooling Coil Sensible Cooling Energy" in c and "(TimeStep)" in c
    ]
    out[f"coilc_{tag}"] = (
        hourly_sum(frame[coils].sum(axis=1).to_numpy() * J_TO_KBTU)
        if coils
        else np.zeros(8760)
    )
    return out


def read_outputs(outdir: Path) -> pd.DataFrame:
    """Read the table of time-step results of an EnergyPlus run."""
    frame = pd.read_csv(outdir / "eplusout.csv")
    frame.columns = frame.columns.str.strip()
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a table in single precision, replacing the file only when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.astype("float32").to_parquet(tmp, index=False)
    os.replace(tmp, path)


def normative_files(
    model, tag, own_h, own_c, presence, persons, lead, floor, ceiling, no_cool, windows
):
    """Write the normative model next to the schedules and return its path."""
    required, pre = cg.required_masks(presence, persons, windows, lead_hours=lead)
    heating, cooling = cg.normative_setpoints(
        own_h,
        own_c,
        required,
        pre,
        floor_c=floor,
        ceiling_c=ceiling,
        no_cooling=no_cool,
    )
    name = f"setpoints_{tag}.csv"
    pd.DataFrame(
        {"heating setpoint norm": heating, "cooling setpoint norm": cooling}
    ).to_csv(model / name, index=False)
    text = rs.normative_model((model / "run.idf").read_text(encoding="utf-8"), name)
    path = model / f"norm_{tag}.idf"
    path.write_text(text, encoding="utf-8")
    return path, required, pre


def donor_jobs() -> list[tuple]:
    """Jobs of the donor dwellings: id, county, persons, no cooling, package applies."""
    from utah_ejpi import resstock

    raw = resstock.read_resstock_annual(
        RELEASE / f"metadata_annual_{STATE}" / f"{config.STATE}_upgrade0.parquet"
    ).set_index("bldg_id")
    upgrade = pd.read_parquet(
        RELEASE / f"metadata_annual_{STATE}" / f"{config.STATE}_upgrade4.parquet"
    ).set_index("bldg_id")["applicability"]
    donors = pd.read_csv(config.INTERIM / f"tract_type_donors_{config.STATE}.csv")
    ids = sorted(donors["building_id"].unique())
    persons = (
        raw["in.occupants"].astype(str).str.replace("+", "", regex=False).astype(int)
    )
    return [
        (
            int(b),
            raw.loc[b, "in.county"],
            int(np.clip(persons.loc[b], 1, 4)),
            raw.loc[b, "in.hvac_cooling_type"] == "None",
            bool(upgrade.get(b, False)),
        )
        for b in ids
    ]


def load_setpoints(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Own setpoints and occupancy schedule saved with a dwelling.

    The occupancy is stored as ``presence``; the files of the first overnight run
    call it ``sch``.
    """
    saved = np.load(path)
    key = "presence" if "presence" in saved.files else "sch"
    return (
        saved["own_h"].astype(float),
        saved["own_c"].astype(float),
        saved[key].astype(float),
    )
