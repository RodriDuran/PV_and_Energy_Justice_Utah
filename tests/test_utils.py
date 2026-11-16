"""Tests of the shared helpers and the table contracts."""

import doctest

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import utils
from utah_ejpi.utils import (
    CONTRACTS,
    TableContract,
    coefficient_of_variation,
    cv_quality,
    interval_coverage,
    moe_proportion,
    moe_sum,
    moe_to_se,
    relative_bias,
    relative_rmse,
    validate_table,
)


def test_docstring_examples_run():
    result = doctest.testmod(utils)
    assert result.failed == 0
    assert result.attempted > 0


def test_moe_to_se_and_cv_follow_the_census_convention():
    assert moe_to_se(1.645) == pytest.approx(1.0)
    assert coefficient_of_variation(1000.0, 164.5) == pytest.approx(10.0)


def test_cv_is_undefined_for_a_zero_estimate():
    cv = coefficient_of_variation(pd.Series([0.0, 100.0]), pd.Series([5.0, 5.0]))
    assert np.isnan(cv.iloc[0])
    assert cv.iloc[1] == pytest.approx(100 * 5 / (1.645 * 100))


def test_cv_quality_classes_and_limits():
    out = cv_quality([11.9, 12.0, 40.0, 40.1, np.nan])
    assert out.tolist() == ["high", "medium", "medium", "low", "undefined"]


def test_moe_sum_matches_the_handbook_example():
    assert float(moe_sum([391, 572, 459])) == pytest.approx(831, abs=0.5)
    frame = pd.DataFrame({"a": [3.0], "b": [4.0]})
    assert moe_sum(frame).iloc[0] == pytest.approx(5.0)


def test_moe_proportion_matches_the_handbook_example():
    moe = moe_proportion(203119, 630498, 5070, 831)
    assert 100 * moe == pytest.approx(0.8, abs=0.05)


def test_moe_proportion_switches_to_the_ratio_formula_when_negative():
    # numerator equal to denominator and a small numerator margin: the
    # radicand is negative with the minus sign
    moe = moe_proportion(100.0, 100.0, 1.0, 50.0)
    assert moe == pytest.approx(np.sqrt(1.0 + 1.0 * 50.0**2) / 100.0)
    assert np.isnan(moe_proportion(1.0, 0.0, 1.0, 1.0))


def test_relative_bias_rmse_and_coverage():
    est = np.array([[9.0, 20.0], [11.0, 24.0]])
    truth = np.array([10.0, 20.0])
    assert relative_bias(est, truth).tolist() == pytest.approx([0.0, 0.1])
    assert relative_rmse(est, truth).tolist() == pytest.approx(
        [0.1, np.sqrt(8.0) / 20.0]
    )
    lo, hi = est - 1.5, est + 1.5
    assert interval_coverage(lo, hi, truth).tolist() == [1.0, 0.5]
    assert np.isnan(relative_bias([[1.0], [2.0]], [0.0])).all()


def test_fixtures_satisfy_their_contracts(
    tract_master, tract_puma_share, resstock_dwellings, acs_type_counts
):
    validate_table(tract_master, "tract_master")
    validate_table(tract_puma_share, "tract_puma_share")
    validate_table(resstock_dwellings, "resstock_dwellings")
    validate_table(acs_type_counts, "acs_type_counts")


def test_every_contract_key_is_a_required_column():
    for spec in CONTRACTS.values():
        assert set(spec.key) <= set(spec.columns), spec.name


def test_validate_table_lists_every_problem(acs_type_counts):
    bad = acs_type_counts.drop(columns="moe")
    bad.loc[0, "estimate"] = -1.0
    bad.loc[1, "GEOID"] = "123"
    bad = pd.concat([bad, bad.iloc[[2]]])
    with pytest.raises(ValueError) as err:
        validate_table(bad, "acs_type_counts")
    assert "missing columns" in str(err.value)
    assert "11 characters" in str(err.value)
    bad = bad.assign(moe=1.0)
    with pytest.raises(ValueError) as err:
        validate_table(bad, "acs_type_counts")
    message = str(err.value)
    assert "duplicated rows" in message
    assert "negative values in estimate" in message


def test_validate_table_checks_kinds_and_extra_columns(tract_master):
    wrong = tract_master.assign(county_fips=[1, 2, 3])
    with pytest.raises(ValueError, match="county_fips is not of kind str"):
        validate_table(wrong, "tract_master")
    extra = tract_master.assign(other=1)
    validate_table(extra, "tract_master")
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_table(extra, "tract_master", allow_extra=False)


def test_shares_must_add_to_one(tract_puma_share):
    bad = tract_puma_share.copy()
    bad.loc[3, "share"] = 0.2
    with pytest.raises(ValueError, match="does not add to 1"):
        validate_table(bad, "tract_puma_share")


def test_unknown_contract_and_custom_contract():
    with pytest.raises(KeyError):
        validate_table(pd.DataFrame(), "no_such_table")
    spec = TableContract("demo", ("a",), {"a": "int"})
    assert validate_table(pd.DataFrame({"a": [1, 2]}), spec).shape == (2, 1)
    with pytest.raises(ValueError, match="duplicated"):
        validate_table(pd.DataFrame({"a": [1, 1]}), spec)


def test_levels_and_interval_order_checks():
    weights = pd.DataFrame(
        {
            "GEOID": ["49001000100"],
            "tenure": ["Owner"],
            "vintage": ["1980 to 1999"],
            "building_type": ["1"],
            "weight": [10.0],
            "source_level": ["region"],
        }
    )
    with pytest.raises(ValueError, match="source_level"):
        validate_table(weights, "tract_type_weights")
    weights["source_level"] = "county"
    validate_table(weights, "tract_type_weights")
    energy = weights.drop(columns=["weight", "source_level"]).assign(
        upgrade=0,
        dwellings=10.0,
        heating_kbtu=5.0,
        heating_kbtu_lo=6.0,
        heating_kbtu_hi=7.0,
        cooling_kbtu=1.0,
        cooling_kbtu_lo=0.5,
        cooling_kbtu_hi=2.0,
    )
    with pytest.raises(ValueError, match="heating_kbtu_lo <= heating_kbtu"):
        validate_table(energy, "tract_type_energy")
