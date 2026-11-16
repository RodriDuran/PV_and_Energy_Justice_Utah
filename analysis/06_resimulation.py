"""Useful energy needed to stay inside the comfort band: windows and re-simulation.

Run from the project root: ``python analysis/06_resimulation.py``. It needs the
donor table of ``02_housing_stock.py``, the baseline annual file and weather
tables of ResStock (``config.DOWNLOAD_RESSTOCK``) and the files of the American
Time Use Survey in ``data/raw/atus`` (activity, respondent and linked population
survey files of the Bureau of Labor Statistics).

Section A. Wake-up and bed times of the respondents of the survey and the window
of a household by number of persons and day type
(``data/results/tables/atus_household_window_ut.csv``). Takes a few minutes.

Section B (behind ``config.RUN_RESIMULATION``). Every donor dwelling of ResStock
is simulated again with EnergyPlus: once with its own thermostat and once with the
normative thermostat of :mod:`utah_ejpi.comfort_gap`; where the heat pump package
applies, the package is also simulated with the normative thermostat. One Parquet
file per dwelling and stage is written to
``data/interim/comfort_gap_resim_<state>``; the run resumes where it stopped. It
needs the program EnergyPlus 24.2 (``config.ENERGYPLUS_DIR``), downloads about
250 KB per dwelling (and again for the package) and, in Utah with 14 processes,
takes about 7 to 9 hours.

Section C (behind ``config.RUN_RESIMULATION_VARIANTS``). A sample of one dwelling
in ``config.RESIMULATION_SAMPLE_EVERY`` is simulated again with variants of the
parameters of the normative thermostat (one-person window, other pre-conditioning
times, other limits of the band).
"""

import logging
import shutil
import time
import traceback
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import resimulation_tools as tools
from utah_ejpi import comfort_gap as cg
from utah_ejpi import resimulation as rs

STATE = config.STATE.lower()
RELEASE = config.RAW / "resstock" / "2025_amy2018_release_1"
OUT = config.INTERIM / f"comfort_gap_resim_{STATE}"
WORK = config.INTERIM / f"comfort_gap_resim_work_{STATE}"
WINDOWS = (
    config.ROOT / "data" / "results" / "tables" / f"atus_household_window_{STATE}.csv"
)
#: name -> (persons override, lead hours, floor, ceiling) of the variants of section C
VARIANTS = {
    "single_window": (1, 3.0, 18.0, 26.0),
    "lead1": (None, 1.0, 18.0, 26.0),
    "lead5": (None, 5.0, 18.0, 26.0),
    "floor20": (None, 3.0, 20.0, 26.0),
    "ceil25": (None, 3.0, 18.0, 25.0),
    "ceil27": (None, 3.0, 18.0, 27.0),
}


def build_windows() -> pd.DataFrame:
    """Section A: window of a household by number of persons and day type."""
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
        "atuscps-0324.zip", "atuscps_0324.dat", ["TUCASEID", "TULINENO", "PRTAGE"]
    )
    acts = read(
        "atusact-0324.zip",
        "atusact_0324.dat",
        ["TUCASEID", "TUSTARTTIM", "TUSTOPTIME", "TRCODEP"],
    )
    resp = resp.merge(cps, on=["TUCASEID", "TULINENO"], how="left")
    resp = resp.merge(cg.sleep_times(acts), on="TUCASEID", how="left")
    resp = resp[(resp["TUYEAR"] >= 2014) & (resp["TUYEAR"] != 2020)]
    table = cg.household_window_table(resp, seed=config.SEED)
    WINDOWS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(WINDOWS, index=False)
    return table


def simulate_dwelling(job: tuple) -> tuple[int, str, float]:
    """Section B for one dwelling: baseline, normative and package runs."""
    building, county, persons, no_cool, applicable = job
    dest = OUT / "main" / f"{building}.parquet"
    if dest.exists():
        return building, "skip", 0.0
    start = time.time()
    work = WORK / f"b{building}"
    try:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        windows = pd.read_csv(WINDOWS)
        model = tools.fetch_model(0, building, work)
        epw = tools.prepare_model_dir(model, county, work)
        run_idf = model / "run.idf"
        foundation = False
        if tools.run_energyplus(run_idf, epw, work / "out_base") != 0:
            log = work / "out_base" / "eplusout.err"
            text = log.read_text(errors="replace") if log.exists() else ""
            if "Custom Block Material Name" not in text:
                raise RuntimeError("baseline run failed")
            run_idf.write_text(
                rs.fix_foundation_custom_blocks(run_idf.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            if tools.run_energyplus(run_idf, epw, work / "out_base") != 0:
                raise RuntimeError("baseline run failed after the foundation repair")
            foundation = True
        base = tools.read_outputs(work / "out_base")
        own_h = base["HEATING SETPOINT:Schedule Value [](TimeStep)"].to_numpy(float)
        own_c = base["COOLING SETPOINT:Schedule Value [](TimeStep)"].to_numpy(float)
        presence = pd.read_csv(model / "in.schedules.csv", usecols=["occupants"])[
            "occupants"
        ].to_numpy(float)
        assert len(own_h) == tools.STEPS == len(presence)
        (OUT / "setpoints").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT / "setpoints" / f"{building}.npz",
            own_h=own_h.astype("float32"),
            own_c=own_c.astype("float32"),
            presence=presence.astype("float32"),
        )
        norm_idf, required, pre = tools.normative_files(
            model,
            "main",
            own_h,
            own_c,
            presence,
            persons,
            config.PRECONDITIONING_HOURS,
            cg.DEFAULT_FLOOR_C,
            cg.DEFAULT_CEILING_C,
            no_cool,
            windows,
        )
        if tools.run_energyplus(norm_idf, epw, work / "out_norm") != 0:
            raise RuntimeError("normative run failed")
        norm = tools.read_outputs(work / "out_norm")
        daylight = cg.daylight_saving_mask(
            2018, observes=config.OBSERVES_DAYLIGHT_SAVING
        )
        required_std = cg.to_standard_time(required, daylight)
        unserved = (
            "IDEAL AIR SYSTEM:Zone Ideal Loads Zone Sensible Cooling Energy "
            "[J](TimeStep)"
        )
        cols = {
            **tools.summarize_run(base, "base"),
            **tools.summarize_run(norm, "norm"),
        }
        cols["cool_unserved_base"] = (
            tools.hourly_sum(base[unserved] * tools.J_TO_KBTU)
            if unserved in base
            else np.zeros(8760)
        )
        cols["required"] = tools.hourly_sum(required.astype(float), "mean")
        cols["required_std"] = tools.hourly_sum(required_std.astype(float), "mean")
        cols["preheat_std"] = tools.hourly_sum(
            cg.to_standard_time(pre, daylight).astype(float), "mean"
        )
        step_req = required_std.astype(float)
        for tag, series in (("norm", norm), ("base", base)):
            t = series["CONDITIONED SPACE:Zone Mean Air Temperature [C](TimeStep)"]
            t = t.to_numpy(float)
            lo = np.clip(17.8 - t, 0, None) * step_req / 4.0
            hi = np.clip(t - 26.2, 0, None) * step_req / 4.0
            cols[f"out_low_{tag}"] = tools.hourly_sum((t < 17.8) * step_req, "mean")
            cols[f"out_high_{tag}"] = tools.hourly_sum((t > 26.2) * step_req, "mean")
            cols[f"dh_low_{tag}"] = tools.hourly_sum(lo)
            cols[f"dh_high_{tag}"] = tools.hourly_sum(hi)
        cols["t_out"] = tools.hourly_sum(
            base["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"],
            "mean",
        )
        frame = pd.DataFrame(cols)
        frame["kiva_fix"] = float(foundation)
        tools.write_parquet(frame, dest)
        if applicable:
            try:
                pkg = tools.fetch_model(4, building, work)
                tools.prepare_model_dir(pkg, county, work)
                pkg_run = pkg / "run.idf"
                pkg_run.write_text(
                    rs.fix_foundation_custom_blocks(
                        pkg_run.read_text(encoding="utf-8")
                    ),
                    encoding="utf-8",
                )
                pkg_idf, _, _ = tools.normative_files(
                    pkg,
                    "main",
                    own_h,
                    own_c,
                    presence,
                    persons,
                    config.PRECONDITIONING_HOURS,
                    cg.DEFAULT_FLOOR_C,
                    cg.DEFAULT_CEILING_C,
                    no_cool,
                    windows,
                )
                if tools.run_energyplus(pkg_idf, epw, work / "out_norm4") != 0:
                    raise RuntimeError("package normative run failed")
                tools.write_parquet(
                    pd.DataFrame(
                        tools.summarize_run(
                            tools.read_outputs(work / "out_norm4"), "norm4"
                        )
                    ),
                    OUT / "u4" / f"{building}.parquet",
                )
            except Exception:
                (OUT / "errors").mkdir(parents=True, exist_ok=True)
                (OUT / "errors" / f"{building}_u4.txt").write_text(
                    traceback.format_exc()
                )
        shutil.rmtree(work, ignore_errors=True)
        return building, "ok", time.time() - start
    except Exception as error:
        (OUT / "errors").mkdir(parents=True, exist_ok=True)
        (OUT / "errors" / f"{building}.txt").write_text(traceback.format_exc())
        shutil.rmtree(work, ignore_errors=True)
        return building, f"error: {error}", time.time() - start


def simulate_variants(job: tuple) -> tuple[int, str, float]:
    """Section C for one dwelling: variants of the normative thermostat."""
    building, county, persons, no_cool, applicable = job
    dest = OUT / "variants" / f"{building}.parquet"
    if dest.exists():
        return building, "skip", 0.0
    start = time.time()
    work = WORK / f"x{building}"
    try:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        windows = pd.read_csv(WINDOWS)
        model = tools.fetch_model(0, building, work)
        epw = tools.prepare_model_dir(model, county, work)
        own_h, own_c, presence = tools.load_setpoints(
            OUT / "setpoints" / f"{building}.npz"
        )
        daylight = cg.daylight_saving_mask(
            2018, observes=config.OBSERVES_DAYLIGHT_SAVING
        )
        cols: dict[str, np.ndarray] = {}
        for name, (override, lead, floor, ceiling) in VARIANTS.items():
            idf, required, _ = tools.normative_files(
                model,
                name,
                own_h,
                own_c,
                presence,
                override or persons,
                lead,
                floor,
                ceiling,
                no_cool,
                windows,
            )
            if tools.run_energyplus(idf, epw, work / f"out_{name}") != 0:
                continue
            summary = tools.summarize_run(tools.read_outputs(work / f"out_{name}"), "v")
            for key in ("heat", "cool", "t"):
                cols[f"{key}_{name}"] = summary[f"{key}_v"]
            cols[f"req_{name}"] = tools.hourly_sum(
                cg.to_standard_time(required, daylight).astype(float), "mean"
            )
        if applicable:
            package = tools.fetch_model(4, building, work)
            tools.prepare_model_dir(package, county, work)
            if tools.run_energyplus(package / "run.idf", epw, work / "out_base4") == 0:
                cols.update(
                    tools.summarize_run(tools.read_outputs(work / "out_base4"), "base4")
                )
        tools.write_parquet(pd.DataFrame(cols), dest)
        shutil.rmtree(work, ignore_errors=True)
        return building, "ok", time.time() - start
    except Exception as error:
        (OUT / "errors").mkdir(parents=True, exist_ok=True)
        (OUT / "errors" / f"{building}_x.txt").write_text(traceback.format_exc())
        shutil.rmtree(work, ignore_errors=True)
        return building, f"error: {error}", time.time() - start


def run() -> None:
    """Run the sections that the flags of ``config`` turn on."""
    logging.basicConfig(level=logging.INFO)
    if not WINDOWS.exists():
        logging.info("section A: building the windows from the time use survey")
        build_windows()
    if not config.RUN_RESIMULATION and not config.RUN_RESIMULATION_VARIANTS:
        logging.info("section B and C are off (config.RUN_RESIMULATION*)")
        return
    jobs = tools.donor_jobs()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stages = []
    if config.RUN_RESIMULATION:
        stages.append((simulate_dwelling, jobs))
    if config.RUN_RESIMULATION_VARIANTS:
        stages.append((simulate_variants, jobs[:: config.RESIMULATION_SAMPLE_EVERY]))
    for function, selected in stages:
        started = time.time()
        with Pool(config.RESIMULATION_WORKERS) as pool:
            for k, (building, status, seconds) in enumerate(
                pool.imap_unordered(function, selected, chunksize=1), 1
            ):
                logging.info(
                    "%s %d %s %.0fs (%d/%d, %.0f min)",
                    function.__name__,
                    building,
                    status,
                    seconds,
                    k,
                    len(selected),
                    (time.time() - started) / 60,
                )


if __name__ == "__main__":
    run()
