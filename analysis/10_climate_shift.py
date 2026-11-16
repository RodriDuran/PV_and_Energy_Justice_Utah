"""Physical check of the correction for the climate of the tract.

Run from the project root: ``python analysis/10_climate_shift.py``. It needs the
results of ``06_resimulation.py`` (the setpoints saved with every dwelling and the
simulations of the baseline) and the climate lines of ``03_energy_requirement.py``.

The correction that takes the energy of a dwelling from the climate of its weather
station to that of a tract has two parts: a factor for the annual total (the ratio of
the climate line at the tract to the line at the station, see
``utah_ejpi.energy_requirement``) and a change of the hourly shape (the response of
the dwelling to the outdoor temperature evaluated at the hourly temperature of the
tract, see ``utah_ejpi.hourly``). Both were checked by comparing weather stations,
which mixes the climate with every other difference between the dwellings. Here the
check is physical: a sample of dwellings is simulated again with the temperature of
its weather file shifted by a few degrees (the humidity, wind and radiation do not
change) and the simulated change is compared with what the correction predicts.

Section A (behind ``config.RUN_CLIMATE_SHIFT``). For one dwelling in
``config.CLIMATE_SHIFT_SAMPLE_EVERY`` and every shift of ``config.CLIMATE_SHIFTS`` and
``config.CLIMATE_SHIFTS_COLD`` (the second group is saved apart),
the baseline and the normative simulations. Two simulations per shift and dwelling;
in Utah about 80 minutes with 14 processes for 4 shifts.

Section B. Comparison with the prediction, for the annual energy used today, the
normative energy and the gap, and for the hourly shape.
"""

import logging
import shutil
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

import climate_fit
import config
import resimulation_tools as tools
from utah_ejpi import hourly
from utah_ejpi import resimulation as rs

STATE = config.STATE.lower()
RUNS = config.INTERIM / f"comfort_gap_resim_{STATE}"
OUT = config.INTERIM / f"climate_shift_{STATE}"
OUT_COLD = config.INTERIM / f"climate_shift_cold_{STATE}"
WORK = config.INTERIM / f"climate_shift_work_{STATE}"
TABLES = config.ROOT / "data" / "results" / "tables"
WINDOWS = TABLES / f"atus_household_window_{STATE}.csv"
FIRST_DATA_LINE = 8


def _shifted_epw(epw_text: str, shift: float) -> str:
    """Weather file with the dry bulb and the dew point moved by ``shift`` degrees."""
    lines = epw_text.split("\n")
    for i in range(FIRST_DATA_LINE, len(lines)):
        if not lines[i]:
            continue
        fields = lines[i].split(",")
        fields[6] = f"{float(fields[6]) + shift:.1f}"
        fields[7] = f"{float(fields[7]) + shift:.1f}"
        lines[i] = ",".join(fields)
    return "\n".join(lines)


def simulate_shifted(job: tuple) -> tuple[int, str, float]:
    """Section A for one dwelling and the shifts of ``config.CLIMATE_SHIFTS``."""
    return _simulate(job, OUT, config.CLIMATE_SHIFTS)


def simulate_cold(job: tuple) -> tuple[int, str, float]:
    """Section A for one dwelling and the shifts of ``config.CLIMATE_SHIFTS_COLD``."""
    return _simulate(job, OUT_COLD, config.CLIMATE_SHIFTS_COLD)


def _shift_files():
    """List the saved runs with the shifts that each one holds."""
    for out, shifts in (
        (OUT, config.CLIMATE_SHIFTS),
        (OUT_COLD, config.CLIMATE_SHIFTS_COLD),
    ):
        for path in sorted(out.glob("*.parquet")):
            yield path, shifts


def _simulate(job: tuple, out, shifts) -> tuple[int, str, float]:
    """Baseline and normative runs of one dwelling at every shift, saved in ``out``."""
    building, county, persons, no_cool, _ = job
    dest = out / f"{building}.parquet"
    if dest.exists():
        return building, "skip", 0.0
    start = time.time()
    work = WORK / f"s{building}"
    try:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        windows = pd.read_csv(WINDOWS)
        model = tools.fetch_model(0, building, work)
        epw = tools.prepare_model_dir(model, county, work)
        run_idf = model / "run.idf"
        idf_text = run_idf.read_text(encoding="utf-8")
        if "Custom Block" in idf_text:
            run_idf.write_text(
                rs.fix_foundation_custom_blocks(idf_text), encoding="utf-8"
            )
        own_h, own_c, presence = tools.load_setpoints(
            RUNS / "setpoints" / f"{building}.npz"
        )
        norm_idf, _, _ = tools.normative_files(
            model,
            "shift",
            own_h,
            own_c,
            presence,
            persons,
            config.PRECONDITIONING_HOURS,
            18.0,
            26.0,
            no_cool,
            windows,
        )
        base_idf = model / "run.idf"
        text = epw.read_text(encoding="utf-8")
        cols = {}
        for k, shift in enumerate(shifts):
            shifted = work / f"weather_{k}.epw"
            shifted.write_text(_shifted_epw(text, shift), encoding="utf-8")
            for tag, idf in (("base", base_idf), ("norm", norm_idf)):
                outdir = work / f"out_{tag}_{k}"
                if tools.run_energyplus(idf, shifted, outdir) != 0:
                    raise RuntimeError(f"{tag} run failed at shift {shift}")
                frame = tools.read_outputs(outdir)
                summary = tools.summarize_run(frame, tag)
                cols[f"heat_{tag}_{k}"] = summary[f"heat_{tag}"]
                cols[f"cool_{tag}_{k}"] = summary[f"cool_{tag}"]
                if tag == "base":
                    zone = (
                        "IDEAL AIR SYSTEM:Zone Ideal Loads Zone Sensible "
                        "Cooling Energy [J](TimeStep)"
                    )
                    cols[f"unserved_{k}"] = (
                        tools.hourly_sum(frame[zone] * tools.J_TO_KBTU)
                        if zone in frame
                        else np.zeros(8760)
                    )
                    column = (
                        "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"
                    )
                    cols[f"t_out_{k}"] = tools.hourly_sum(frame[column], "mean")
        tools.write_parquet(pd.DataFrame(cols), dest)
        shutil.rmtree(work, ignore_errors=True)
        return building, "ok", time.time() - start
    except Exception as error:
        (out / "errors").mkdir(parents=True, exist_ok=True)
        (out / "errors" / f"{building}.txt").write_text(traceback.format_exc())
        shutil.rmtree(work, ignore_errors=True)
        return building, f"error: {error}", time.time() - start


def _degree_days(temperature: np.ndarray, use: str) -> float:
    """Thousands of degree days of an hourly series, with the bases of the study."""
    daily = np.asarray(temperature, float).reshape(365, 24).mean(axis=1)
    if use == "heating":
        return float(np.clip(18.0 - daily, 0, None).sum() / 1000.0)
    return float(np.clip(daily - 26.0, 0, None).sum() / 1000.0)


def annual_check(weights: pd.Series, lines: dict) -> pd.DataFrame:
    """Section B, annual totals: simulated change against the climate factor."""
    rows = []
    for path, shifts in _shift_files():
        b = int(path.stem)
        shift = pd.read_parquet(path)
        main = pd.read_parquet(RUNS / "main" / f"{b}.parquet")
        base = {
            "heating_current": main["heat_base"].sum(),
            "cooling_current": (main["cool_base"] - main["cool_unserved_base"]).sum(),
            "heating_normative": main["heat_norm"].sum(),
            "cooling_normative": main["cool_norm"].sum(),
        }
        base["heating_gap"] = base["heating_normative"] - base["heating_current"]
        base["cooling_gap"] = base["cooling_normative"] - base["cooling_current"]
        for k, d in enumerate(shifts):
            sim = {
                "heating_current": shift[f"heat_base_{k}"].sum(),
                "cooling_current": (
                    shift[f"cool_base_{k}"] - shift[f"unserved_{k}"]
                ).sum(),
                "heating_normative": shift[f"heat_norm_{k}"].sum(),
                "cooling_normative": shift[f"cool_norm_{k}"].sum(),
            }
            sim["heating_gap"] = sim["heating_normative"] - sim["heating_current"]
            sim["cooling_gap"] = sim["cooling_normative"] - sim["cooling_current"]
            for use in ("heating", "cooling"):
                a, s = lines[use]
                dd0 = _degree_days(main["t_out"].to_numpy(), use)
                dd1 = _degree_days(shift[f"t_out_{k}"].to_numpy(), use)
                factor = (a + s * dd1) / (a + s * dd0)
                for kind in ("current", "normative", "gap"):
                    name = f"{use}_{kind}"
                    rows.append(
                        {
                            "building_id": b,
                            "shift": d,
                            "quantity": name,
                            "base": base[name],
                            "simulated": sim[name],
                            "predicted_factor": factor,
                            "weight": float(weights.get(b, 0.0)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_annual(table: pd.DataFrame) -> pd.DataFrame:
    """Compare the simulated and the predicted change of the weighted total."""
    out = []
    for (quantity, shift), g in table.groupby(["quantity", "shift"]):
        w = g["weight"].to_numpy()
        base = (g["base"] * w).sum()
        simulated = (g["simulated"] * w).sum()
        predicted = (g["base"] * g["predicted_factor"] * w).sum()
        out.append(
            {
                "quantity": quantity,
                "shift": shift,
                "dwellings": len(g),
                "simulated_change": simulated / base - 1 if base else np.nan,
                "predicted_change": predicted / base - 1 if base else np.nan,
                "error_of_the_factor": (predicted - simulated) / base
                if base
                else np.nan,
            }
        )
    return pd.DataFrame(out)


def _relative_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Root mean squared error over the mean of the truth."""
    return float(np.sqrt(((estimate - truth) ** 2).mean()) / truth.mean())


def shape_check(weights: pd.Series) -> pd.DataFrame:
    """Section B, hourly shape: fitted response against the simulated series."""
    rows = []
    for path, shifts in _shift_files():
        b = int(path.stem)
        shift = pd.read_parquet(path)
        main = pd.read_parquet(RUNS / "main" / f"{b}.parquet")
        t0 = main["t_out"].to_numpy()
        current_cooling = main["cool_base"] - main["cool_unserved_base"]
        series = {
            "heating_current": ("heating", main["heat_base"], "heat_base"),
            "cooling_current": ("cooling", current_cooling, "cool_base"),
            "heating_normative": ("heating", main["heat_norm"], "heat_norm"),
            "cooling_normative": ("cooling", main["cool_norm"], "cool_norm"),
        }
        for name, (use, values, column) in series.items():
            original = values.to_numpy()
            response = hourly.fit_temperature_response(original, t0, use)
            if response is None:
                continue
            for k, d in enumerate(shifts):
                truth = shift[f"{column}_{k}"].to_numpy()
                if name == "cooling_current":
                    truth = truth - shift[f"unserved_{k}"].to_numpy()
                if truth.sum() <= 0:
                    continue
                t1 = shift[f"t_out_{k}"].to_numpy()
                predicted = hourly.temperature_response_shape(response, t1)
                if predicted.sum() <= 0:
                    continue
                hybrid = hourly.reshape_load(original, response, t0, t1)
                if hybrid.sum() <= 0:
                    continue
                hybrid = hybrid * truth.sum() / hybrid.sum()
                predicted = predicted * truth.sum() / predicted.sum()
                unchanged = original * truth.sum() / original.sum()
                rows.append(
                    {
                        "building_id": b,
                        "shift": d,
                        "series": name,
                        "weight": float(weights.get(b, 0.0)),
                        "peak_to_mean_truth": truth.max() / truth.mean(),
                        "peak_to_mean_response": predicted.max() / predicted.mean(),
                        "peak_to_mean_unchanged": unchanged.max() / unchanged.mean(),
                        "rmse_response": _relative_rmse(predicted, truth),
                        "rmse_unchanged": _relative_rmse(unchanged, truth),
                        "rmse_hybrid": _relative_rmse(hybrid, truth),
                        "peak_to_mean_hybrid": hybrid.max() / hybrid.mean(),
                        "corr_response": float(np.corrcoef(predicted, truth)[0, 1]),
                        "corr_unchanged": float(np.corrcoef(unchanged, truth)[0, 1]),
                        "corr_hybrid": float(np.corrcoef(hybrid, truth)[0, 1]),
                    }
                )
    return pd.DataFrame(rows)


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


def run() -> None:
    """Run the simulations if the flag is on and compare them with the correction."""
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_COLD.mkdir(parents=True, exist_ok=True)
    if config.RUN_CLIMATE_SHIFT:
        WORK.mkdir(parents=True, exist_ok=True)
        jobs = tools.donor_jobs()[:: config.CLIMATE_SHIFT_SAMPLE_EVERY]
        started = time.time()
        for worker in (simulate_shifted, simulate_cold):
            with Pool(config.RESIMULATION_WORKERS) as pool:
                for k, (building, status, seconds) in enumerate(
                    pool.imap_unordered(worker, jobs, chunksize=1), 1
                ):
                    logging.info(
                        "%s %d %s %.0fs (%d/%d, %.0f min)",
                        worker.__name__,
                        building,
                        status,
                        seconds,
                        k,
                        len(jobs),
                        (time.time() - started) / 60,
                    )
    dwellings = climate_fit.read_dwellings(config.STATE)
    degree_days = pd.read_csv(
        config.INTERIM / f"climate_degree_days_{config.STATE}.csv", dtype={"id": str}
    )
    lines = {}
    for use in ("heating", "cooling"):
        fitted = climate_fit.fit_sensitivity(use, dwellings, degree_days)
        lines[use] = (fitted.intercept, fitted.slope)
    weights = dwelling_weights()
    table = annual_check(weights, lines)
    table.to_csv(TABLES / f"climate_shift_by_dwelling_{STATE}.csv", index=False)
    summarize_annual(table).to_csv(
        TABLES / f"climate_shift_annual_{STATE}.csv", index=False
    )
    shape = shape_check(weights)
    shape.to_csv(TABLES / f"climate_shift_shape_by_dwelling_{STATE}.csv", index=False)
    if len(shape):
        keep = ["peak_to_mean_truth", "peak_to_mean_response", "peak_to_mean_unchanged"]
        keep += [
            "rmse_response",
            "rmse_unchanged",
            "rmse_hybrid",
            "peak_to_mean_hybrid",
        ]
        keep += ["corr_response", "corr_unchanged", "corr_hybrid"]
        shape.groupby(["series", "shift"])[keep].median().to_csv(
            TABLES / f"climate_shift_shape_{STATE}.csv"
        )


if __name__ == "__main__":
    run()
