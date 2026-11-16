"""Validation of the link between the tracts and the modeled ResStock dwellings.

The base method builds the types of dwelling of a tract from the Census table of
tenure by year built by units in structure, so it reproduces that table by
construction. What can fail is everything else: the modeled dwellings chosen for
each type may not look like the dwellings of the tract in characteristics that
the Census also publishes for each tract and that matter for the energy used.
This module compares, tract by tract, the composition implied by the chosen
dwellings with the Census counts for

* the main heating fuel by tenure (ACS table B25117),
* the number of bedrooms by tenure (B25042),
* the household size by tenure (B25009), and
* the presence of air conditioning (Census Bureau local estimates, LACE).

The measures follow the quality checks of synthetic populations: the total
absolute error, which adds the absolute differences between the synthetic and the
Census counts, and the absolute percentage difference of each category (Jiang et
al. 2024, ``jiang_2024_synthetic``). The comparison is made on the mix, not on the
totals: the implied counts of a tenure are scaled to the Census total of that
tenure in the compared table, so that the small differences between two ACS
tables do not count as errors.

References
----------
Jiang et al. (2024), synthetic population of the United States
(``jiang_2024_synthetic``); Kuh et al. (2022, ``kuh_2022_loo_mrp``) warn that
errors measured on single dwellings say little about the error of an area, which
is why the comparison is made for each tract.
"""

from __future__ import annotations

import os
import re
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "COMPARISONS",
    "GROUP_TYPES",
    "compare_with_recs",
    "lead_convergence",
    "measured_level_factors",
    "read_eia_residential_totals",
    "read_lead_tract_means",
    "read_recs",
    "recs_means",
    "composition_classes",
    "air_conditioning_comparison",
    "composition_errors",
    "implied_composition",
    "observed_composition",
    "resstock_class",
]

#: Comparisons by tenure: name -> (ACS table, ResStock column that gives the class).
COMPARISONS = {
    "heating_fuel": ("B25117", "heating_fuel"),
    "bedrooms": ("B25042", "bedrooms"),
    "household_size": ("B25009", "occupants"),
}

# ResStock heating fuels -> comparison classes. Fuel oil, other fuel and no
# heating fuel are grouped with the rare Census fuels (fuel oil, coal, solar,
# other fuel, no fuel used) in the class "other", because the two sources do not
# distinguish them in the same way.
_FUEL_CLASS = {
    "Natural Gas": "utility_gas",
    "Propane": "propane",
    "Electricity": "electricity",
    "Wood": "wood",
}


def composition_classes() -> pd.DataFrame:
    """Return the classes in which the Census tables are compared.

    Returns
    -------
    pandas.DataFrame
        Columns ``table`` (``heating_fuel``, ``bedrooms`` or ``household_size``),
        ``variable`` (variable of the ACS table), ``tenure`` and ``class``.
        Built from the variable labels of the Census Bureau API for the tables
        B25117, B25042 and B25009; the totals and subtotals are not included.
        Bedrooms "none" and "1" form one class because ResStock has no
        dwelling with no bedroom; household sizes of 7 or more form one class.
    """
    path = resources.files("utah_ejpi") / "data" / "composition_classes.csv"
    with resources.as_file(path) as file:
        return pd.read_csv(file, dtype=str)


def resstock_class(dwellings: pd.DataFrame, table: str) -> pd.Series:
    """Give each modeled dwelling its class for one comparison.

    Parameters
    ----------
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` with the columns ``heating_fuel``,
        ``bedrooms`` and ``occupants``.
    table : {"heating_fuel", "bedrooms", "household_size"}
        Comparison.

    Returns
    -------
    pandas.Series
        Class of each dwelling, with the labels of :func:`composition_classes`.

    Raises
    ------
    KeyError
        If the comparison is unknown.

    Notes
    -----
    ResStock codes bedrooms from 1 to 5 and the enumeration does not say that 5
    means "5 or more"; its technical reference guide (2025 release 1, section on
    the Bedrooms characteristic) states that more than 5 bedrooms are labeled 5 and
    0 bedrooms are labeled 1, so it is compared with the Census class "5 or more"
    (and the Census class "0 or 1" holds the 1).
    """
    if table == "heating_fuel":
        return dwellings["heating_fuel"].map(_FUEL_CLASS).fillna("other")
    if table == "bedrooms":
        bed = dwellings["bedrooms"]
        return bed.map(
            lambda n: "0 or 1" if n <= 1 else ("5 or more" if n >= 5 else str(n))
        )
    if table == "household_size":
        size = dwellings["occupants"].astype(int)
        return size.map(lambda n: "7 or more" if n >= 7 else str(n))
    raise KeyError(f"Unknown comparison {table!r}.")


def observed_composition(
    acs_table: pd.DataFrame, table: str, state_fips: str
) -> pd.DataFrame:
    """Count the dwellings of each tract by tenure and class from a Census table.

    Parameters
    ----------
    acs_table : pandas.DataFrame
        ACS table in long format (``GEOID``, ``variable``, ``estimate``).
    table : {"heating_fuel", "bedrooms", "household_size"}
        Comparison; selects the table and the classes.
    state_fips : str
        Two-digit state FIPS code.

    Returns
    -------
    pandas.DataFrame
        ``GEOID``, ``tenure``, ``class`` and ``observed`` (dwellings), adding the
        Census categories that form each class.
    """
    classes = composition_classes()
    classes = classes[classes["table"] == table]
    rows = acs_table[acs_table["GEOID"].str.startswith(state_fips)]
    joined = rows.merge(classes, on="variable", how="inner")
    return (
        joined.groupby(["GEOID", "tenure", "class"], as_index=False)["estimate"]
        .sum()
        .rename(columns={"estimate": "observed"})
    )


def implied_composition(
    weights: pd.DataFrame, donors: pd.DataFrame, dwellings: pd.DataFrame, table: str
) -> pd.DataFrame:
    """Count the dwellings of each tract by tenure and class implied by the donors.

    Parameters
    ----------
    weights : pandas.DataFrame
        Contract ``tract_type_weights`` (dwellings of each type in each tract).
    donors : pandas.DataFrame
        Contract ``tract_type_donors``.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` with the columns of the comparison.
    table : {"heating_fuel", "bedrooms", "household_size"}
        Comparison.

    Returns
    -------
    pandas.DataFrame
        ``GEOID``, ``tenure``, ``class`` and ``implied``: for every type of
        dwelling, its Census count is distributed among the classes in proportion
        to the weights of the modeled dwellings that stand for it, and the
        results are added over the types of the same tenure. The tenure is the
        one of the Census type: tenure is part of the type, so donors always
        have the tenure of the type they represent, except when the nearest
        year-built class is used, which keeps tenure too.
    """
    attrs = dwellings[["building_id"]].assign(
        **{"class": resstock_class(dwellings, table)}
    )
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    joined = donors.merge(attrs, on="building_id", how="left", validate="many_to_one")
    if joined["class"].isna().any():
        raise ValueError("Some donors are not in the dwellings table.")
    share = joined.groupby([*keys, "class"], as_index=False)["donor_weight"].sum()
    share = share.merge(weights[[*keys, "weight"]], on=keys, validate="many_to_one")
    share["implied"] = share["donor_weight"] * share["weight"]
    return share.groupby(["GEOID", "tenure", "class"], as_index=False)["implied"].sum()


def composition_errors(
    implied: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the implied composition with the Census counts.

    Parameters
    ----------
    implied, observed : pandas.DataFrame
        Outputs of :func:`implied_composition` and :func:`observed_composition`.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(by_tract, by_class)``. ``by_tract`` has ``GEOID``, ``dwellings`` (Census
        total of the compared table), ``tae`` (total absolute error: the sum over
        tenure and class of the absolute difference between the implied and the
        Census counts, after scaling the implied counts of each tenure to the
        Census total of that tenure) and ``tae_share`` (``tae`` over
        ``dwellings``; it is twice the share of dwellings that would have to
        change class to match the Census). ``by_class`` has, for the state,
        ``tenure``, ``class``, ``observed``, ``implied``, ``difference`` and
        ``apd`` (absolute percentage difference of the class).

    Notes
    -----
    A tract with no Census dwellings of a tenure contributes nothing for that
    tenure. The tracts of the Census table that lack donors are left out.
    """
    both = observed.merge(implied, on=["GEOID", "tenure", "class"], how="outer").fillna(
        {"observed": 0.0, "implied": 0.0}
    )
    obs_total = both.groupby(["GEOID", "tenure"])["observed"].transform("sum")
    imp_total = both.groupby(["GEOID", "tenure"])["implied"].transform("sum")
    # Scale the implied counts to the Census total of the tenure so that only
    # the mix is compared.
    scale = np.where(
        imp_total > 0, obs_total / imp_total.where(imp_total > 0, 1.0), 0.0
    )
    both["implied"] = both["implied"] * scale
    both["abs_error"] = (both["implied"] - both["observed"]).abs()
    by_tract = both.groupby("GEOID").agg(
        dwellings=("observed", "sum"), tae=("abs_error", "sum")
    )
    by_tract["tae_share"] = by_tract["tae"] / by_tract["dwellings"].where(
        by_tract["dwellings"] > 0
    )
    by_tract = by_tract.reset_index()
    by_class = both.groupby(["tenure", "class"], as_index=False)[
        ["observed", "implied"]
    ].sum()
    by_class["difference"] = by_class["implied"] - by_class["observed"]
    by_class["apd"] = by_class["difference"].abs() / by_class["observed"].where(
        by_class["observed"] > 0
    )
    return by_tract, by_class


def air_conditioning_comparison(
    weights: pd.DataFrame,
    donors: pd.DataFrame,
    dwellings: pd.DataFrame,
    lace: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the share of dwellings with cooling with the Census estimates.

    Parameters
    ----------
    weights : pandas.DataFrame
        Contract ``tract_type_weights``.
    donors : pandas.DataFrame
        Contract ``tract_type_donors``.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` with the column ``cooling_type``.
    lace : pandas.DataFrame
        Census Bureau local estimates of air conditioning (LACE) by tract, with
        the columns ``STATE``, ``COUNTY``, ``TRACT``, ``HSE_OCC_E`` (occupied
        dwellings) and ``AC_E`` (occupied dwellings with air conditioning).

    Returns
    -------
    pandas.DataFrame
        ``GEOID``, ``observed_share`` (``AC_E / HSE_OCC_E``), ``implied_share``
        (share of the tract's dwellings whose modeled donors have a cooling
        system, weighting donors and types as in :func:`implied_composition`) and
        ``difference`` (implied minus observed).

    Notes
    -----
    LACE is a model-based estimate of the Census Bureau, not a count, and it is
    not stated whether evaporative coolers count as air conditioning (NO
    VERIFICADO). ResStock does not model evaporative coolers, so dwellings that
    have one are modeled with no cooling or with another system.
    """
    cooled = dwellings[["building_id"]].assign(
        cooled=(dwellings["cooling_type"] != "None").astype(float)
    )
    keys = ["GEOID", "tenure", "vintage", "building_type"]
    joined = donors.merge(cooled, on="building_id", how="left", validate="many_to_one")
    joined["cooled"] = joined["cooled"] * joined["donor_weight"]
    share = joined.groupby(keys, as_index=False)["cooled"].sum()
    share = share.merge(weights[[*keys, "weight"]], on=keys, validate="many_to_one")
    share["with_ac"] = share["cooled"] * share["weight"]
    by_tract = share.groupby("GEOID").agg(
        with_ac=("with_ac", "sum"), total=("weight", "sum")
    )
    by_tract["implied_share"] = by_tract["with_ac"] / by_tract["total"].where(
        by_tract["total"] > 0
    )
    geoid = lace["STATE"].astype(str).str.zfill(2) + lace["COUNTY"].astype(
        str
    ).str.zfill(3)
    geoid = geoid + lace["TRACT"].astype(str).str.zfill(6)
    obs = pd.DataFrame(
        {
            "GEOID": geoid,
            "observed_share": lace["AC_E"]
            / lace["HSE_OCC_E"].where(lace["HSE_OCC_E"] > 0),
        }
    )
    out = obs.merge(by_tract[["implied_share"]].reset_index(), on="GEOID", how="inner")
    out["difference"] = out["implied_share"] - out["observed_share"]
    return out


# ---------------------------------------------------------------------------
# Comparison of the energy with reference data
# ---------------------------------------------------------------------------

#: Thousand Btu in a kilowatt-hour (1 kWh = 3,412.14 Btu).
KBTU_PER_KWH = 3.41214

#: Number of jackknife replicate weights of the 2020 RECS and the Student t
#: quantile of the 97.5 percentile for its 59 nominal degrees of freedom.
RECS_REPLICATES = 60
T_975_59_DF = 2.001

_STRUCTURE_RECS = {
    1: "mobile_home",
    2: "single_family",
    3: "single_family",
    4: "multifamily_2_to_4",
    5: "multifamily_5_or_more",
}
_STRUCTURE_RESSTOCK = {
    "1": "single_family",
    "2 to 4": "multifamily_2_to_4",
    "5 to 19": "multifamily_5_or_more",
    "20 to 49": "multifamily_5_or_more",
    "50 or more": "multifamily_5_or_more",
    "mobile": "mobile_home",
}
_FUEL_RECS = {1: "utility_gas", 5: "electricity", 2: "propane"}
_FUEL_RESSTOCK = {
    "Natural Gas": "utility_gas",
    "Electricity": "electricity",
    "Propane": "propane",
}
GROUP_TYPES = ("all", "tenure", "structure", "heating_fuel")


def read_recs(path: str | os.PathLike, state: str) -> pd.DataFrame:
    """Read the households of one state from the 2020 RECS public file.

    Parameters
    ----------
    path : path-like
        ``recs2020_public_v7.csv`` of the U.S. Energy Information Administration.
    state : str
        Two-letter postal code of the state.

    Returns
    -------
    pandas.DataFrame
        The households of the state with their final weight ``NWEIGHT``, the 60
        replicate weights ``NWEIGHT1`` to ``NWEIGHT60``, and the columns used in
        the comparison: ``KOWNRENT``, ``TYPEHUQ``, ``FUELHEAT``, ``TOTALBTUSPH``
        (space heating, all fuels, thousand Btu) and ``KWHCOL`` (electricity for
        cooling, kWh).

    Notes
    -----
    The end uses of the RECS are not metered: the codebook calls them
    "calibrated" usage, estimated from bills and adjusted to totals. The cooling
    figure includes evaporative coolers, which ResStock does not model, and the
    heating figure includes main and secondary equipment.
    """
    columns = [
        "state_postal",
        "NWEIGHT",
        *[f"NWEIGHT{i}" for i in range(1, RECS_REPLICATES + 1)],
        "KOWNRENT",
        "TYPEHUQ",
        "FUELHEAT",
        "TOTALBTUSPH",
        "KWHCOL",
    ]
    recs = pd.read_csv(path, usecols=columns)
    return recs[recs["state_postal"] == state].reset_index(drop=True)


def _recs_labels(recs: pd.DataFrame) -> pd.DataFrame:
    """Add the group labels and the values of the comparison to RECS households."""
    out = recs.copy()
    out["all"] = "all"
    # Households that occupy without paying rent count as renters, as in the
    # Census tables.
    out["tenure"] = np.where(out["KOWNRENT"] == 1, "Owner", "Renter")
    out["structure"] = out["TYPEHUQ"].map(_STRUCTURE_RECS)
    out["heating_fuel"] = out["FUELHEAT"].map(_FUEL_RECS).fillna("other")
    out["heating_kwh"] = out["TOTALBTUSPH"] / KBTU_PER_KWH
    out["cooling_kwh"] = out["KWHCOL"]
    return out


def recs_means(recs: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Compute the weighted mean of a RECS variable by group, with its error.

    Parameters
    ----------
    recs : pandas.DataFrame
        Output of :func:`read_recs`.
    variable : {"heating_kwh", "cooling_kwh"}
        Annual space heating (all fuels) or cooling electricity, in kWh.

    Returns
    -------
    pandas.DataFrame
        ``group_type`` (``all``, ``tenure``, ``structure`` or ``heating_fuel``),
        ``group``, ``mean``, ``se`` and ``n`` (households in the sample). The
        standard error follows the jackknife formula of the RECS guide:
        ``sqrt((R - 1) / R * sum((estimate_r - estimate) ** 2))`` over the ``R``
        = 60 replicate weights.

    References
    ----------
    U.S. Energy Information Administration (2023), 2020 RECS microdata guide
    (``eia_2023_recs2020``), section on the jackknife method.
    """
    data = _recs_labels(recs)
    replicate_cols = [f"NWEIGHT{i}" for i in range(1, RECS_REPLICATES + 1)]
    rows = []
    for group_type in GROUP_TYPES:
        for group, sub in data.groupby(group_type):
            value = sub[variable]
            mean = (value * sub["NWEIGHT"]).sum() / sub["NWEIGHT"].sum()
            reps = (sub[replicate_cols].mul(value, axis=0)).sum() / sub[
                replicate_cols
            ].sum()
            se = np.sqrt(
                (RECS_REPLICATES - 1) / RECS_REPLICATES * ((reps - mean) ** 2).sum()
            )
            rows.append((group_type, group, mean, se, len(sub)))
    return pd.DataFrame(rows, columns=["group_type", "group", "mean", "se", "n"])


def _modeled_groups(
    frame: pd.DataFrame, tenure: str, building_type: str, fuel: str
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["all"] = "all"
    out["tenure"] = frame[tenure]
    out["structure"] = frame[building_type].map(_STRUCTURE_RESSTOCK)
    out["heating_fuel"] = frame[fuel].map(_FUEL_RESSTOCK).fillna("other")
    return out


def _weighted_means(
    groups: pd.DataFrame, values: pd.Series, weights: pd.Series, name: str
) -> pd.DataFrame:
    rows = []
    for group_type in GROUP_TYPES:
        frame = pd.DataFrame(
            {"g": groups[group_type], "v": values * weights, "w": weights}
        )
        sums = frame.groupby("g")[["v", "w"]].sum()
        for group, row in sums.iterrows():
            rows.append(
                (group_type, group, row["v"] / row["w"] if row["w"] > 0 else np.nan)
            )
    return pd.DataFrame(rows, columns=["group_type", "group", name])


def compare_with_recs(
    recs: pd.DataFrame,
    dwellings: pd.DataFrame,
    donor_energy: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """Compare the modeled consumption with the RECS means, by group.

    Three modeled versions are compared with the RECS to separate causes: the
    occupied dwellings of ResStock with their own weights (``resstock``), the
    dwellings of the tracts built with the Census counts without the climate
    correction (``tracts_uncorrected``) and with it (``tracts_corrected``).

    Parameters
    ----------
    recs : pandas.DataFrame
        Output of :func:`read_recs`.
    dwellings : pandas.DataFrame
        Contract ``resstock_dwellings`` with the consumption columns.
    donor_energy : pandas.DataFrame
        Output of :func:`utah_ejpi.energy_requirement.donor_level_energy`.
    variable : {"heating_kwh", "cooling_kwh"}
        Annual space heating (all fuels) or cooling electricity, in kWh.

    Returns
    -------
    pandas.DataFrame
        One row per group with the RECS ``mean``, ``se`` and ``n``, the three
        modeled means, ``ratio`` (corrected tract mean over the RECS mean),
        ``low`` and ``high`` (95 % interval of the RECS mean, with the t quantile
        of 59 degrees of freedom) and ``consistent`` (the corrected mean lies in
        that interval). The variable is written in ``variable``.

    Notes
    -----
    The RECS interval covers only the sampling error of the survey, not the
    error of the model that produced its end uses.
    """
    reference = recs_means(recs, variable)
    use = variable.split("_")[0]
    native = dwellings[dwellings["occupied"]]
    cols = [c for c in native.columns if c.startswith(f"{use}_kwh_")]
    native_values = native[cols].sum(axis=1)
    native_groups = _modeled_groups(native, "tenure", "building_type", "heating_fuel")
    modeled = _weighted_means(
        native_groups, native_values, native["weight"], "resstock"
    )
    groups = _modeled_groups(donor_energy, "tenure", "building_type", "heating_fuel")
    for name, column in (
        ("tracts_uncorrected", f"{variable}_uncorrected"),
        ("tracts_corrected", variable),
    ):
        part = _weighted_means(
            groups, donor_energy[column], donor_energy["weight"], name
        )
        modeled = modeled.merge(part, on=["group_type", "group"], how="outer")
    out = reference.merge(modeled, on=["group_type", "group"], how="left")
    out["ratio"] = out["tracts_corrected"] / out["mean"]
    out["low"] = out["mean"] - T_975_59_DF * out["se"]
    out["high"] = out["mean"] + T_975_59_DF * out["se"]
    out["consistent"] = (out["tracts_corrected"] >= out["low"]) & (
        out["tracts_corrected"] <= out["high"]
    )
    out.insert(0, "variable", variable)
    return out


_KWH_PER_KBTU = 1.0 / 3.41214  # kWh in one thousand Btu


def _seds_gas_trillion_btu(text: str, year: int) -> float:
    """Natural gas of the residential table of the State Energy Data System.

    The table in trillion Btu has one row per year with the fuels in order: coal,
    natural gas, distillate fuel oil, ...; the table in physical units above it has
    no decimals in its rows and is skipped.
    """
    for line in text.splitlines():
        fields = line.split()
        if (
            len(fields) >= 12
            and fields[0] == str(year)
            and re.fullmatch(r"\d+\.\d", fields[2])
        ):
            return float(fields[2])
    raise ValueError(f"No row of {year} in trillion Btu in the table of the state.")


def read_eia_residential_totals(
    directory: str | os.PathLike, state: str, year: int
) -> dict[str, float]:
    """Read the residential natural gas and electricity sold in a state in a year.

    Uses the files of :func:`utah_ejpi.download.download_eia_state_totals`: the
    residential table of the State Energy Data System (natural gas in trillion Btu,
    which avoids a heat content) and the sales of electricity by state and sector.
    They are totals sold by the utilities for the whole state, so they check the
    level of the modeled energy and not its split by use or tract.

    Parameters
    ----------
    directory : path-like
        Folder with ``use_res_<ST>.pdf`` and ``sales_annual.xlsx``.
    state : str
        Two-letter postal code.
    year : int
        Calendar year; the sales workbook must include it.

    Returns
    -------
    dict
        ``natural_gas_kwh``, ``electricity_kwh`` and the values they come from,
        ``natural_gas_trillion_btu`` and ``electricity_mwh``.

    Raises
    ------
    FileNotFoundError
        If a file is missing.
    ValueError
        If the year or the state is not in the files.
    ImportError
        If ``pypdf`` or ``openpyxl`` (optional dependencies) are not installed.

    References
    ----------
    eia_2025_seds_consumption, eia_state_electricity_sales.
    """
    from pypdf import PdfReader

    directory = Path(directory)
    code = state.upper()
    pdf, xlsx = directory / f"use_res_{code}.pdf", directory / "sales_annual.xlsx"
    for path in (pdf, xlsx):
        if not path.exists():
            raise FileNotFoundError(f"EIA file not found: {path}")
    text = "\n".join(page.extract_text() for page in PdfReader(str(pdf)).pages)
    gas_tbtu = _seds_gas_trillion_btu(text, year)
    sales = pd.read_excel(xlsx, header=1)
    row = sales[
        (sales["State"] == code)
        & (sales["Year"] == year)
        & (sales["Industry Sector Category"] == "Total Electric Industry")
    ]
    if row.empty:
        raise ValueError(f"No sales of {code} in {year} in {xlsx.name}.")
    mwh = float(row["Residential"].iloc[0])
    return {
        "natural_gas_trillion_btu": gas_tbtu,
        "natural_gas_kwh": gas_tbtu * 1e9 * _KWH_PER_KBTU,
        "electricity_mwh": mwh,
        "electricity_kwh": mwh * 1000.0,
    }


def measured_level_factors(
    measured: dict[str, float],
    modeled: dict[str, float],
    households_measured: float,
    allocation: str = "use",
) -> dict[str, float]:
    """Factors that take the modeled heating and cooling to the measured totals.

    The totals sold include every use of the fuel, so the factor of a use depends on
    where the difference between the modeled and the measured total is assumed to
    be. With ``allocation="use"`` all of it is in the use (the widest range that
    the data allow: if the rest of the fuel is right, the use carries the whole
    difference); with ``"proportional"`` every use of the fuel is off by the same
    factor.

    Parameters
    ----------
    measured : dict
        ``natural_gas_kwh`` and ``electricity_kwh`` sold to the households of the
        state in the year (:func:`read_eia_residential_totals`).
    modeled : dict
        Mean kWh per dwelling and year (:func:`utah_ejpi.hourly.state_annual_means`):
        ``natural_gas``, ``natural_gas_heating``, ``electricity`` and
        ``electricity_cooling``.
    households_measured : float
        Households of the state in the year of the measured totals; the totals are
        divided by them to compare with the means per dwelling.
    allocation : {"use", "proportional"}
        Where the difference is assumed to be.

    Returns
    -------
    dict
        ``heating`` (factor of the heating with natural gas) and ``cooling`` (factor
        of the electric cooling); the factor of the other fuels of heating is one.

    Raises
    ------
    ValueError
        If the allocation is unknown or a factor is not positive.

    Examples
    --------
    >>> measured = {"natural_gas_kwh": 120.0, "electricity_kwh": 90.0}
    >>> modeled = {"natural_gas": 100.0, "natural_gas_heating": 80.0,
    ...            "electricity": 100.0, "electricity_cooling": 40.0}
    >>> measured_level_factors(measured, modeled, 1.0)
    {'heating': 1.25, 'cooling': 0.75}
    >>> measured_level_factors(measured, modeled, 1.0, "proportional")
    {'heating': 1.2, 'cooling': 0.9}
    """
    if allocation not in ("use", "proportional"):
        raise ValueError("allocation must be 'use' or 'proportional'.")
    gas = measured["natural_gas_kwh"] / households_measured
    electricity = measured["electricity_kwh"] / households_measured
    heat, cool = modeled["natural_gas_heating"], modeled["electricity_cooling"]
    if allocation == "use":
        heating = (gas - (modeled["natural_gas"] - heat)) / heat
        cooling = (electricity - (modeled["electricity"] - cool)) / cool
    else:
        heating = gas / modeled["natural_gas"]
        cooling = electricity / modeled["electricity"]
    if heating <= 0 or cooling <= 0:
        raise ValueError("The measured totals leave a non-positive factor.")
    return {"heating": float(heating), "cooling": float(cooling)}


def read_lead_tract_means(zip_path: str | os.PathLike, member: str) -> pd.DataFrame:
    """Read the mean annual energy expenditure by tract from the LEAD data.

    Parameters
    ----------
    zip_path : path-like
        Zip file of the LEAD tool data of a state (U.S. Department of Energy).
    member : str
        File of the zip with the tract tables by income; every tract appears
        split in rows by income bracket, tenure, year built, building type and
        heating fuel.

    Returns
    -------
    pandas.DataFrame
        ``GEOID``, ``households`` and the mean annual household expenditure in
        dollars on electricity (``electricity_usd``), gas (``gas_usd``) and other
        fuels (``other_fuel_usd``). The mean of a tract is the sum of
        ``<fuel>*UNITS`` over the rows divided by the sum of ``<fuel> UNITS``.

    Notes
    -----
    LEAD is a model of the spending reported to the Census survey, calibrated to
    sales of the utilities (Ma et al. 2024, ``doe_2024_lead_methodology``); the
    expenditure covers every use of the dwelling, not only heating and cooling.
    """
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        table = pd.read_csv(fh, dtype={"FIP": str})
    fuels = {"ELEP": "electricity_usd", "GASP": "gas_usd", "FULP": "other_fuel_usd"}
    cols = ["FIP", "UNITS"]
    for key in fuels:
        cols += [f"{key}*UNITS", f"{key} UNITS"]
    sums = table[cols].groupby("FIP").sum()
    out = pd.DataFrame({"households": sums["UNITS"]})
    for key, name in fuels.items():
        out[name] = sums[f"{key}*UNITS"] / sums[f"{key} UNITS"].where(
            sums[f"{key} UNITS"] > 0
        )
    out.index.name = "GEOID"
    return out.reset_index()


def lead_convergence(tract_energy: pd.DataFrame, lead: pd.DataFrame) -> pd.DataFrame:
    """Correlate the modeled consumption of the tracts with the LEAD expenditure.

    Parameters
    ----------
    tract_energy : pandas.DataFrame
        Output of :func:`utah_ejpi.energy_requirement.build_tract_type_energy`
        with the consumption by fuel.
    lead : pandas.DataFrame
        Output of :func:`read_lead_tract_means`.

    Returns
    -------
    pandas.DataFrame
        For each pair of modeled consumption and LEAD expenditure: ``modeled``,
        ``lead``, ``spearman`` (rank correlation among tracts) and ``tracts``.

    Notes
    -----
    The pairs are heating with natural gas against gas expenditure, electricity
    for heating and cooling against electricity expenditure, and heating with
    propane and fuel oil against other-fuel expenditure. The rank correlation
    says whether the tracts are ordered alike by two independent models; it says
    nothing about the level.
    """
    w = tract_energy["dwellings"]
    frame = pd.DataFrame({"GEOID": tract_energy["GEOID"], "w": w})
    modeled = {
        "heating_kwh_natural_gas": tract_energy["heating_kwh_natural_gas"],
        "electricity_kwh": tract_energy["heating_kwh_electricity"]
        + tract_energy["cooling_kwh_electricity"],
        "propane_fuel_oil_kwh": tract_energy["heating_kwh_propane"]
        + tract_energy["heating_kwh_fuel_oil"],
    }
    for name, values in modeled.items():
        frame[name] = values * w
    sums = frame.groupby("GEOID").sum()
    sums = sums[sums["w"] > 0]
    means = sums[list(modeled)].div(sums["w"], axis=0).reset_index()
    joined = means.merge(lead, on="GEOID", how="inner")
    pairs = [
        ("heating_kwh_natural_gas", "gas_usd"),
        ("electricity_kwh", "electricity_usd"),
        ("propane_fuel_oil_kwh", "other_fuel_usd"),
    ]
    rows = []
    for mod, ref in pairs:
        both = joined[[mod, ref]].dropna()
        rows.append((mod, ref, both[mod].corr(both[ref], method="spearman"), len(both)))
    return pd.DataFrame(rows, columns=["modeled", "lead", "spearman", "tracts"])
