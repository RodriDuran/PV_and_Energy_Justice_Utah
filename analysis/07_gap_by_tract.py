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
quantity. For the gaps (``config.GAP_ADDITIVE_CANDIDATES``) a third correction is also
tried: a slope of their own against the degree days, added to the value of each dwelling
by the difference of degree days between the tract and the station (the gap is not
proportional to the energy, so the ratio of the line of the energy does not describe
it). It is used only if it lowers the median error more than the other two.
The normative energy of each tract is the energy used today (with its climate
correction) plus the gap (with its own), so that the three add up in every tract and
simulation (``config.DERIVED_QUANTITIES``); the error of that sum is compared with the
direct estimate of the normative energy in ``derived_quantities_check_<state>.csv``.

Section C. Simulations of the quantities by tract with the donors resampled, the
transfer error, the zone effect and the composition of the tract, and their summary
(mean, interval, coefficient of variation and quality class) in
``data/results/tables``. The zone effect is the spread, between the ResStock zones,
of the mean of a quantity once the type, fuel, bedrooms, size and occupants of the
dwellings are taken out; the composition is the change of the mean of each tract when
the Census shares of fuel, bedrooms and household size are used as well, estimated by
regression in every tract. The coverage
of the 90 % interval in the test of section B is saved with and without those two
sources (``known_population_coverage_gap_<state>.csv``). The share of the variance of
each quantity that every source brings is saved in
``uncertainty_sources_gap_<state>.csv``: each source is simulated alone with the
others switched off, and its variance is divided by the sum of the variances of the
six sources.

The gap of each dwelling is also split in three columns that add up to it: whether it
has a gap (more than ``config.GAP_THRESHOLD_KWH`` a year), the gap of the dwellings
that have one and the rest. The share of the dwellings with a gap, the gap of those
that have it and the rest are simulated like the other quantities and then mixed with
the mean of the state, with a weight that minimizes the squared error in the test of
units as small as a tract (parts of about ``config.MINI_TRACT_SIZE`` dwellings of a
zone, estimated with the donors of the rest of the zone). The weights and their
standard deviation are saved in ``composite_weights_<state>.csv`` and the errors of
the mix, with the weight estimated without the zone of each unit, in
``composite_check_<state>.csv``. The mix moves the center of the simulations of each
tract and keeps their spread (``utah_ejpi.gap_uncertainty.shrink_draws``), so the
interval keeps the width validated by the test of the 90 % interval. The gap of a
tract is the sum of its parts, and the gap of those that have one is its part over
the share (``gap_magnitude_*``).

The interval of each tract (the one whose coverage section B validates) does not
include the composition of the tract. The composition is a separate band: the same
simulations plus the change by composition, summarized in the columns ending in
``_wide`` of the table by tract and in ``composition_band_<state>.csv``, with the
number of tracts whose quality class changes. The cooling gap of the dwellings that
have no cooling (ResStock does not model evaporative coolers, which RECS reports for
some of them) is reduced in a sensitivity: a share of those dwellings, the total of
RECS divided by the dwellings without cooling of ResStock, is taken to have a cooler
that removes their gap (``gap_cooling_evap_*`` and ``norm_cooling_evap_*``;
parameters in ``evaporative_cooler_sensitivity_<state>.csv``). The central value is
not changed.

Section D. Checks of the tract results: agreement of the energy used today with the
estimate of ``03_energy_requirement.py``, stability of the position of the tracts
under the uncertainty, and a leave-one-dwelling-out check of the efficiency curve of
the heat pump against the electricity that the package simulation gives directly.
There are no observations of the gap at tract level, so these checks test the
consistency and the precision of the estimate, not its accuracy.
"""

import logging
from dataclasses import replace

import numpy as np
import pandas as pd

import climate_fit
import config
from utah_ejpi import gap_uncertainty as gu
from utah_ejpi import heat_pump as hp
from utah_ejpi import housing_stock, uncertainty, validation
from utah_ejpi.energy_requirement import fit_additive_slope

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
    threshold = config.GAP_THRESHOLD_KWH / KWH
    for use in ("heating", "cooling"):
        gap = table[f"gap_{use}"]
        has = gap > threshold
        table[f"gap_share_{use}"] = has.astype(float)
        table[f"gap_over_{use}"] = gap.where(has, 0.0)
        table[f"gap_under_{use}"] = gap.where(~has, 0.0)
    table.to_csv(TABLES / f"comfort_gap_by_dwelling_{STATE}.csv")
    return table


def uses_of(columns) -> dict[str, str]:
    """Use of every quantity: heating or cooling (lower tail of the band is heating)."""
    return {
        c: "cooling" if ("cooling" in c or c.startswith("high_share")) else "heating"
        for c in columns
    }


ENERGY_PREFIXES = ("cur_", "norm_", "gap_", "heat_pump_")
CATEGORICAL = (
    "tenure",
    "vintage",
    "building_type",
    "heating_fuel",
    "bedrooms",
    "floor_area_bin",
    "occupants",
)


def energy_columns(columns) -> list[str]:
    """Quantities that are energy per dwelling (the others are shares and counts)."""
    return [c for c in columns if c.startswith(ENERGY_PREFIXES)]


def zone_effects(
    quantities: pd.DataFrame,
    dwellings: pd.DataFrame,
    degree_days: pd.DataFrame,
    corrected: list[str],
) -> dict[str, float]:
    """Relative standard deviation of the zone effect of each energy quantity."""
    raw = pd.read_parquet(
        config.RAW
        / "resstock"
        / "2025_amy2018_release_1"
        / f"metadata_annual_{STATE}"
        / f"{config.STATE}_upgrade0.parquet",
        columns=[
            "bldg_id",
            "in.heating_fuel",
            "in.bedrooms",
            "in.geometry_floor_area_bin",
            "in.occupants",
        ],
    ).set_index("bldg_id")
    stock = dwellings[dwellings["occupied"]].set_index("building_id")
    stock = stock.loc[stock.index.intersection(quantities.dropna().index)]
    categorical = pd.DataFrame(
        {
            "tenure": stock["tenure"],
            "vintage": stock["vintage"],
            "building_type": stock["building_type"],
            "heating_fuel": raw["in.heating_fuel"].reindex(stock.index),
            "bedrooms": raw["in.bedrooms"].reindex(stock.index),
            "floor_area_bin": raw["in.geometry_floor_area_bin"].reindex(stock.index),
            "occupants": raw["in.occupants"].reindex(stock.index),
        }
    )
    stations = degree_days[degree_days["kind"] == "station"].set_index("id")
    numeric = pd.DataFrame(
        {
            "hdd": stock["county"].map(stations["hdd"]),
            "cdd": stock["county"].map(stations["cdd"]),
        }
    )
    columns = energy_columns(quantities.columns)
    sd = gu.zone_effect_sd(
        quantities.loc[stock.index, columns],
        stock["puma"],
        categorical,
        numeric,
        with_numeric=[c for c in corrected if c in columns],
    )
    pd.DataFrame({"quantity": list(sd), "zone_effect_sd": list(sd.values())}).to_csv(
        TABLES / f"zone_effect_gap_{STATE}.csv", index=False
    )
    return sd


def composition(quantities: pd.DataFrame, dwellings: pd.DataFrame) -> pd.DataFrame:
    """Change of the mean of each tract with the Census composition (all dwellings).

    It is estimated by regression on the Census shares of the heating fuel, the
    bedrooms and the household size (``utah_ejpi.gap_uncertainty``), which exists in
    every tract. Where the weights that match the Census tables converge, their
    change is the control: the correlation and the error of the regression with
    respect to it are saved in ``composition_regression_check_<state>.csv``.
    """
    stock = dwellings[dwellings["occupied"]]
    counts = housing_stock.acs_type_counts(
        housing_stock.read_acs_table(
            config.RAW / "acs" / "acs5_2024" / "B25127_tract_us.csv"
        ),
        config.STATE_FIPS,
    )
    observed = {
        name: validation.observed_composition(
            housing_stock.read_acs_table(
                config.RAW
                / "acs"
                / "acs5_2024"
                / f"{table}_tract_{STATE if scope == 'state' else scope}.csv"
            ),
            name,
            config.STATE_FIPS,
        )
        for name, (table, scope) in config.COMPOSITION_TABLES.items()
    }
    shares = housing_stock.composition_shares(counts, observed)
    indicators = housing_stock.class_indicators(stock)
    controls = pd.get_dummies(
        stock.set_index("building_id")[["tenure", "vintage", "building_type"]].astype(
            str
        )
    ).astype(float)
    base = pd.read_csv(
        config.INTERIM / f"tract_dwelling_weights_type_{STATE}.csv",
        dtype={"GEOID": str},
    )
    columns = energy_columns(quantities.columns)
    shift = gu.composition_shift_regression(
        base, quantities[columns], indicators, shares, controls
    )

    variant = config.COMPOSITION_VARIANT
    alt = pd.read_csv(
        config.INTERIM / f"tract_dwelling_weights_{variant}_{STATE}.csv",
        dtype={"GEOID": str},
    )
    diag = pd.read_csv(
        config.INTERIM / f"reweighting_diagnostics_{variant}_{STATE}.csv",
        dtype={"GEOID": str},
    ).set_index("GEOID")
    control = gu.composition_shift(base, alt, quantities[columns].dropna())
    ok = diag["converged"].reindex(shift.index).fillna(False).astype(bool)
    rows = []
    for name in columns:
        truth = control.loc[ok[ok].index.intersection(control.index), name]
        est = shift.loc[truth.index, name]
        rows.append(
            {
                "quantity": name,
                "tracts": len(truth),
                "correlation": float(np.corrcoef(est, truth)[0, 1]),
                "rmse": float(np.sqrt(((est - truth) ** 2).mean())),
                "rmse_of_no_change": float(np.sqrt((truth**2).mean())),
            }
        )
    pd.DataFrame(rows).to_csv(
        TABLES / f"composition_regression_check_{STATE}.csv", index=False
    )
    shift.assign(weights_converged=ok.reindex(shift.index).to_numpy()).to_csv(
        TABLES / f"composition_shift_gap_{STATE}.csv"
    )
    logging.info(
        "composition by regression in %d tracts; median absolute change of the "
        "heating energy %.1f%% of the mean; the weights converge in %d",
        len(shift),
        100 * shift["cur_heating"].abs().median() / quantities["cur_heating"].mean(),
        int(ok.sum()),
    )
    return shift


BAND_COLUMNS = ("lo", "hi", "cv", "quality")


def composition_band(
    summary: pd.DataFrame, wide: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Add the columns of the band with composition and report its effect.

    The wide band is the summary of the simulations plus the composition; it is
    joined to the summary of the validated interval with the suffix ``_wide``.
    """
    keep = ["GEOID", "group"]
    for q in columns:
        keep += [f"{q}_{c}" for c in BAND_COLUMNS]
    renamed = wide[keep].rename(
        columns={c: f"{c}_wide" for c in keep if c not in ("GEOID", "group")}
    )
    out = summary.merge(renamed, on=["GEOID", "group"], how="left")
    whole = out[out["group"] == "all"]
    rows = []
    for q in columns:
        rows.append(
            {
                "quantity": q,
                "cv_median_validated": float(whole[f"{q}_cv"].median()),
                "cv_median_with_composition": float(whole[f"{q}_cv_wide"].median()),
                "tracts": len(whole),
                "quality_changes": int(
                    (whole[f"{q}_quality"] != whole[f"{q}_quality_wide"]).sum()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / f"composition_band_{STATE}.csv", index=False)
    return out


def evaporative_sensitivity(
    summary: pd.DataFrame, dwellings: pd.DataFrame, quantities: pd.DataFrame
) -> pd.DataFrame:
    """Reduce the cooling gap of the dwellings without cooling by a share of them.

    RECS gives the share of the households of the state that have an evaporative
    cooler as main equipment (with its jackknife error); ResStock has no such
    equipment and models those dwellings with no cooling. If all of them are among
    the dwellings of ResStock without cooling, the share of those that have a cooler
    is the RECS share over the share of ResStock's dwellings without cooling. Who has
    a cooler is not known, so the same share applies to every tract, and the cooler
    is taken to remove the gap of the dwelling (an upper bound of the reduction).
    The central value of the table is not changed; the reduced values are columns.
    """
    recs = validation.read_recs(
        config.RAW / "recs_2020" / "recs2020_public_v7.csv", config.STATE
    )
    share, se = validation.recs_share(recs, "ACEQUIPM_PUB", 6)
    stock = dwellings[dwellings["occupied"]].set_index("building_id")
    without = (stock["cooling_type"] == "None").astype(float)
    weight = stock["weight"]
    none_share = float((without * weight).sum() / weight.sum())
    high = share + 1.645 * se
    f_central, f_high = min(1.0, share / none_share), min(1.0, high / none_share)
    base = pd.read_csv(
        config.INTERIM / f"tract_dwelling_weights_type_{STATE}.csv",
        dtype={"GEOID": str},
    )
    base = base[base["building_id"].isin(quantities.index)]
    gap_no_cooling = quantities["gap_cooling"] * without.reindex(quantities.index)
    base = base.assign(
        gap=base["building_id"].map(gap_no_cooling).fillna(0.0) * base["weight"]
    )
    by_tract = (
        base.groupby("GEOID")["gap"].sum() / base.groupby("GEOID")["weight"].sum()
    )
    whole = summary["group"] == "all"
    noac = summary["GEOID"].map(by_tract).where(whole)
    out = summary.assign(gap_cooling_noac=noac)
    for tag, f in (("central", f_central), ("max", f_high)):
        out[f"gap_cooling_evap_{tag}"] = out["gap_cooling"] - f * noac
        out[f"norm_cooling_evap_{tag}"] = out["norm_cooling"] - f * noac
    mean_gap = float(out.loc[whole, "gap_cooling"].mean())
    pd.DataFrame(
        [
            {
                "recs_evaporative_share": share,
                "recs_evaporative_share_se": se,
                "recs_households": len(recs),
                "recs_evaporative_households": int((recs["ACEQUIPM_PUB"] == 6).sum()),
                "resstock_share_without_cooling": none_share,
                "share_of_those_with_cooler": f_central,
                "share_of_those_with_cooler_upper_90": f_high,
                "gap_cooling_reduction_central": float(
                    f_central * out.loc[whole, "gap_cooling_noac"].mean() / mean_gap
                ),
                "gap_cooling_reduction_max": float(
                    f_high * out.loc[whole, "gap_cooling_noac"].mean() / mean_gap
                ),
            }
        ]
    ).to_csv(TABLES / f"evaporative_cooler_sensitivity_{STATE}.csv", index=False)
    return out


SPLIT = tuple(f"{p}{u}" for p in config.SPLIT_PREFIXES for u in ("heating", "cooling"))


def mini_tract_split(
    dwellings: pd.DataFrame, buildings: pd.Index, seed: int
) -> pd.DataFrame:
    """Dwellings with a column ``mini``: random parts of every zone of about a tract."""
    rng = np.random.default_rng(seed)
    stock = dwellings[dwellings["occupied"] & dwellings["building_id"].isin(buildings)]
    stock = stock.copy()
    part = pd.Series(0, index=stock.index)
    for _, index in stock.groupby("puma").groups.items():
        index = rng.permutation(np.array(list(index)))
        k = max(2, round(len(index) / config.MINI_TRACT_SIZE))
        part.loc[index] = np.arange(len(index)) % k
    stock["mini"] = stock["puma"].astype(str) + "_" + part.astype(str)
    return stock


def composite_weights(
    dwellings, quantities, uses, degree_days, extra, stations
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Weights of the mix with the mean of the state and the test that supports them.

    Returns the table of weights and the table of the units of the test of parts of a
    zone (``zone`` is the zone of each unit), with the estimate, the mean of the other
    dwellings and the true mean of the quantities of the split and of the gap.
    """
    names = [*SPLIT, "gap_heating", "gap_cooling"]
    part_uses = {k: uses[k] for k in names}
    tables = []
    for seed in range(config.MINI_TRACT_SEEDS):
        stock = mini_tract_split(dwellings, quantities.index, seed)
        table = gu.known_population_quantities(
            stock,
            quantities[names],
            part_uses,
            degree_days,
            climate_corrected=[],
            unit="mini",
            n_draws=1,
            seed=seed,
            resample=False,
            extra_fit_dwellings=extra,
            extra_fit_stations=stations,
        )
        tables.append(table.assign(seed=seed))
    mini = pd.concat(tables, ignore_index=True)
    mini["zone"] = mini["unit"].str.rsplit("_", n=1).str[0]
    rows = []
    for q in SPLIT:
        w, sd = gu.composite_weight(
            mini[f"estimate_{q}"], mini[f"state_{q}"], mini[f"truth_{q}"],
            groups=mini["zone"], seed=config.SEED,
        )  # fmt: skip
        rows.append({"quantity": q, "weight": w, "weight_sd": sd, "units": len(mini)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"composite_weights_{STATE}.csv", index=False)
    return out, mini


def composite_check(mini: pd.DataFrame) -> pd.DataFrame:
    """Error of the mix with the weight estimated without the zone of each unit.

    For every quantity of the split and for the gap of each use (the sum of the
    mixed parts), the median relative error and the root mean square error of the
    estimate with the donors of the zone, of the mean of the state and of the mix.
    """
    zones = mini["zone"].unique()

    def mix(q: str) -> tuple[pd.Series, pd.Series]:
        own = pd.Series(np.nan, index=mini.index)
        full = pd.Series(np.nan, index=mini.index)
        w_all = gu.composite_weight(
            mini[f"estimate_{q}"], mini[f"state_{q}"], mini[f"truth_{q}"],
            groups=mini["zone"], n_boot=0,
        )[0]  # fmt: skip
        full[:] = w_all * mini[f"estimate_{q}"] + (1 - w_all) * mini[f"state_{q}"]
        for z in zones:
            keep = mini["zone"] != z
            w = gu.composite_weight(
                mini.loc[keep, f"estimate_{q}"], mini.loc[keep, f"state_{q}"],
                mini.loc[keep, f"truth_{q}"], groups=mini.loc[keep, "zone"], n_boot=0,
            )[0]  # fmt: skip
            hit = ~keep
            estimate = mini.loc[hit, f"estimate_{q}"]
            state = mini.loc[hit, f"state_{q}"]
            own[hit] = w * estimate + (1 - w) * state
        return own, full

    mixes = {q: mix(q) for q in SPLIT}
    rows = []

    def add(label, truth, direct, state, nested, in_sample, floor):
        scale = truth.abs().where(truth.abs() > floor)
        entry = {"quantity": label}
        variants = {
            "direct": direct,
            "state": state,
            "mix_without_zone": nested,
            "mix_all_zones": in_sample,
        }
        for key, value in variants.items():
            error = value - truth
            entry[f"median_error_{key}"] = float((error / scale).abs().median())
            entry[f"rmse_{key}"] = float(np.sqrt((error**2).mean()))
        rows.append(entry)

    for q in SPLIT:
        floor = 0.001 if "_share_" in q else 1.0
        add(q, mini[f"truth_{q}"], mini[f"estimate_{q}"], mini[f"state_{q}"],
            mixes[q][0], mixes[q][1], floor)  # fmt: skip
    for use in ("heating", "cooling"):
        parts = [f"gap_over_{use}", f"gap_under_{use}"]
        add(f"gap_{use}", mini[f"truth_gap_{use}"], mini[f"estimate_gap_{use}"],
            mini[f"state_gap_{use}"], sum(mixes[q][0] for q in parts),
            sum(mixes[q][1] for q in parts), 1.0)  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"composite_check_{STATE}.csv", index=False)
    return out


def with_additive(table: pd.DataFrame, chosen: list[str]) -> pd.DataFrame:
    """Known-population table whose estimates of ``chosen`` use their own slope."""
    out = table.copy()
    for q in chosen:
        for column, source in (
            ("estimate", "additive"),
            ("lo", "additive_lo"),
            ("hi", "additive_hi"),
        ):
            out[f"{column}_{q}"] = out[f"{source}_{q}"]
    return out


def coverage_check(
    dwellings, quantities, uses, degree_days, corrected, tau, zone_sd, extra, stations,
    additive, weights,
) -> pd.DataFrame:  # fmt: skip
    """Share of the units left out whose true mean is inside the 90 % interval."""
    rows = []
    energy = energy_columns(quantities.columns)
    for label, zone in (("transfer_only", None), ("transfer_and_zone", zone_sd)):
        table = gu.known_population_quantities(
            dwellings,
            quantities,
            uses,
            degree_days,
            climate_corrected=corrected,
            unit="puma",
            n_draws=config.N_DRAWS,
            seed=config.SEED,
            transfer_tau=tau,
            extra_fit_dwellings=extra,
            extra_fit_stations=stations,
            zone_effect_sd=zone,
            additive_candidates=additive,
            derived=config.DERIVED_QUANTITIES,
            shrink_weights=weights,
        )
        table = with_additive(table, additive)
        for q in energy:
            inside = (table[f"truth_{q}"] >= table[f"lo_{q}"]) & (
                table[f"truth_{q}"] <= table[f"hi_{q}"]
            )
            rows.append(
                {
                    "quantity": q,
                    "interval": label,
                    "coverage": float(inside.mean()),
                    "units": len(table),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"known_population_coverage_gap_{STATE}.csv", index=False)
    return out


SOURCES = ("donors", "census", "climate", "transfer", "composition", "zone")


def source_decomposition(
    weights, donors, dwellings, quantities, uses, totals, degree_days,
    heating, cooling, corrected, tau, main_puma, zone_sd, shift, additive,
) -> pd.DataFrame:  # fmt: skip
    """Variance that each source of error brings to each quantity of a tract.

    Every source is simulated alone (the others off) with the same number of draws
    and seed; the sources are independent, so their variances add up. The share of a
    source is its variance over the sum of the six, tract by tract, summarized by
    the median over tracts.
    """
    names = energy_columns(quantities.columns)
    q = quantities[names]
    u = {k: v for k, v in uses.items() if k in names}
    zero_w = weights.assign(moe=0.0)
    zero_t = totals.assign(total_moe=0.0)
    flat_h = replace(heating, covariance=np.zeros((2, 2)))
    flat_c = replace(cooling, covariance=np.zeros((2, 2)))
    flat_a = {k: replace(v, variance=0.0) for k, v in additive.items() if k in names}
    base = dict(
        climate_additive=flat_a,
        derived={k: v for k, v in config.DERIVED_QUANTITIES.items() if k in names},
        donors=donors, dwellings=dwellings, quantities=q, uses=u,
        tract_degree_days=degree_days, station_degree_days=degree_days,
        climate_corrected=[c for c in corrected if c in names],
        n_draws=100, seed=config.SEED,
    )  # fmt: skip
    runs = {
        "donors": dict(weights=zero_w, tract_totals=zero_t, heating=flat_h,
                       cooling=flat_c, resample=True),
        "census": dict(weights=weights, tract_totals=totals, heating=flat_h,
                       cooling=flat_c, resample=False),
        "climate": dict(weights=zero_w, tract_totals=zero_t, heating=heating,
                        cooling=cooling, resample=False,
                        climate_additive={k: v for k, v in additive.items()
                                          if k in names}),
        "transfer": dict(weights=zero_w, tract_totals=zero_t, heating=flat_h,
                         cooling=flat_c, resample=False,
                         transfer_tau={k: v for k, v in tau.items() if k in names},
                         transfer_groups=main_puma),
        "composition": dict(weights=zero_w, tract_totals=zero_t, heating=flat_h,
                            cooling=flat_c, resample=False,
                            composition_shift=shift[names]),
        "zone": dict(weights=zero_w, tract_totals=zero_t, heating=flat_h,
                     cooling=flat_c, resample=False,
                     zone_effect_sd={k: v for k, v in zone_sd.items() if k in names},
                     zone_of=main_puma),
    }  # fmt: skip
    sd = {}
    mean = None
    for source, extra in runs.items():
        draws = gu.simulate_tract_quantities(**{**base, **extra})
        draws = draws[draws["group"] == "all"]
        grouped = draws.groupby("GEOID")[names]
        sd[source] = grouped.std(ddof=1)
        mean = grouped.mean() if mean is None else mean
    variance = {k: v**2 for k, v in sd.items()}
    total = sum(variance.values())
    rows = []
    for name in names:
        row = {"quantity": name}
        for source in SOURCES:
            row[f"share_{source}"] = float(
                (variance[source][name] / total[name]).median()
            )
            row[f"relative_sd_{source}"] = float(
                (sd[source][name] / mean[name].abs()).median()
            )
        row["relative_sd_sum"] = float(
            (np.sqrt(total[name]) / mean[name].abs()).median()
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"uncertainty_sources_gap_{STATE}.csv", index=False)
    return out


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
        additive_candidates=list(config.GAP_ADDITIVE_CANDIDATES),
    )
    floor = {q: 0.001 if "_share_" in q else 1.0 for q in quantities.columns}
    rows = []
    corrected = []
    additive = []
    for q in quantities.columns:
        truth = test[f"truth_{q}"]
        scale = truth.abs().where(truth.abs() > floor[q])
        with_ = ((test[f"estimate_{q}"] - truth) / scale).abs().median()
        without = ((test[f"no_climate_{q}"] - truth) / scale).abs().median()
        state = ((test[f"state_{q}"] - truth) / scale).abs().median()
        own = np.nan
        if q in config.GAP_ADDITIVE_CANDIDATES:
            own = ((test[f"additive_{q}"] - truth) / scale).abs().median()
        use_own = bool(own < min(with_, without))
        use_it = bool(with_ < without) and not use_own
        if q in SPLIT:
            use_own = use_it = False
        if use_own:
            additive.append(q)
        if use_it:
            corrected.append(q)
        rows.append(
            {"quantity": q, "median_error_with_climate": with_,
             "median_error_without_climate": without, "median_error_state_mean": state,
             "median_error_own_slope": own,
             "climate_correction_used": use_it, "own_slope_used": use_own}
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
        additive_candidates=additive,
        derived=config.DERIVED_QUANTITIES,
    )
    first = with_additive(first, additive)
    check = []
    for name in config.DERIVED_QUANTITIES:
        truth = first[f"truth_{name}"]
        scale = truth.abs().where(truth.abs() > floor[name])
        direct = (
            test[f"estimate_{name}"]
            if name in corrected
            else test[f"no_climate_{name}"]
        )
        check.append(
            {
                "quantity": name,
                "median_error_derived": ((first[f"estimate_{name}"] - truth) / scale)
                .abs()
                .median(),
                "median_error_direct": ((direct - truth) / scale).abs().median(),
            }
        )
    pd.DataFrame(check).to_csv(
        TABLES / f"derived_quantities_check_{STATE}.csv", index=False
    )
    weights_table, mini = composite_weights(
        dwellings, quantities, uses, degree_days, extra, extra_stations
    )
    logging.info("weights of the mix:\n%s", weights_table.round(3).to_string())
    logging.info("check of the mix:\n%s", composite_check(mini).round(3).to_string())
    mix_weights = {
        r.quantity: (float(r.weight), float(r.weight_sd))
        for r in weights_table.itertuples()
    }
    shrink = {
        q: gu.Shrinkage(w, sd, float(quantities[q].mean()))
        for q, (w, sd) in mix_weights.items()
    }
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
    own_slopes = {
        q: fit_additive_slope(quantities[q], dwellings, degree_days, uses[q])
        for q in additive
    }
    for q, fit in own_slopes.items():
        logging.info(
            "own slope of %s: %.1f per thousand degree days (sd %.1f), %d counties, "
            "range %.2f-%.2f",
            q, fit.slope, fit.variance**0.5, fit.n_stations, fit.dd_min, fit.dd_max,
        )  # fmt: skip
    pd.DataFrame(
        [
            {
                "quantity": q,
                "slope_per_thousand_degree_days": f.slope,
                "slope_sd": f.variance**0.5,
                "counties": f.n_stations,
                "dd_min": f.dd_min,
                "dd_max": f.dd_max,
            }
            for q, f in own_slopes.items()
        ],
        columns=[
            "quantity",
            "slope_per_thousand_degree_days",
            "slope_sd",
            "counties",
            "dd_min",
            "dd_max",
        ],
    ).to_csv(TABLES / f"own_slope_gap_{STATE}.csv", index=False)
    zone_sd = zone_effects(quantities, dwellings, degree_days, corrected + additive)
    logging.info("zone effect: %s", {k: round(v, 3) for k, v in zone_sd.items()})
    shift = composition(quantities, dwellings)
    coverage = coverage_check(
        dwellings, quantities, uses, degree_days, corrected, tau, zone_sd,
        extra, extra_stations, additive, mix_weights,
    )  # fmt: skip
    logging.info(
        "coverage of the 90 %% interval:\n%s",
        coverage.pivot(index="quantity", columns="interval", values="coverage"),
    )
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
        zone_effect_sd=zone_sd,
        zone_of=main_puma,
        climate_additive=own_slopes,
        derived=config.DERIVED_QUANTITIES,
    )
    draws = gu.shrink_draws(draws, shrink, config.DERIVED_QUANTITIES, seed=config.SEED)
    wide = gu.add_composition_draws(
        draws, shift, config.DERIVED_QUANTITIES, seed=config.SEED
    )
    for frame in (draws, wide):
        for use in ("heating", "cooling"):
            share = frame[f"gap_share_{use}"]
            frame[f"gap_magnitude_{use}"] = frame[f"gap_over_{use}"] / share.where(
                share > 0
            )
    draws.to_parquet(config.INTERIM / f"tract_gap_draws_{STATE}.parquet", index=False)
    sources = source_decomposition(
        weights, donors, dwellings, quantities, uses, totals, degree_days,
        heating, cooling, corrected, tau, main_puma, zone_sd, shift, own_slopes,
    )  # fmt: skip
    logging.info("variance share of each source:\n%s", sources.round(2).to_string())
    columns = [*quantities.columns, "gap_magnitude_heating", "gap_magnitude_cooling"]
    summary = uncertainty.summarize_draws(draws, columns)
    summary = composition_band(
        summary, uncertainty.summarize_draws(wide, columns), columns
    )
    summary = evaporative_sensitivity(summary, dwellings, quantities)
    summary.to_csv(TABLES / f"tract_gap_uncertainty_{STATE}.csv", index=False)
    whole = summary[summary["group"] == "all"]
    for name, (first_part, second_part) in config.DERIVED_QUANTITIES.items():
        gap = (whole[name] - whole[first_part] - whole[second_part]).abs().max()
        logging.info("largest difference of %s with its two parts: %.6f", name, gap)
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
