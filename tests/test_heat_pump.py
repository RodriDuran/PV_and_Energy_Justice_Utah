"""Tests of the efficiency curve of the heat pump."""

import numpy as np
import pandas as pd
import pytest

from utah_ejpi import heat_pump as hp


def test_cop_curve_is_energy_over_electricity_by_bin():
    t = np.array([0.0, 0.5, 10.0, 30.0])
    curve = hp.cop_curve(
        np.array([2.0, 4.0, 6.0, 1.0]),
        np.array([1.0, 2.0, 1.5, 0.5]),
        t,
        bin_width=5.0,
        t_min=-5.0,
        t_max=15.0,
    )
    assert curve["cop"].tolist() == [2.0, 4.0]
    assert curve["useful"].tolist() == [6.0, 6.0]  # the 30 degree hour is out


def test_cop_curve_drops_bins_with_little_energy():
    curve = hp.cop_curve(
        np.array([0.5, 5.0]),
        np.array([0.25, 1.0]),
        np.array([0.0, 10.0]),
        bin_width=5.0,
        t_min=-5.0,
        t_max=15.0,
        min_useful=1.0,
    )
    assert len(curve) == 1
    assert curve["cop"].iloc[0] == 5.0


def test_cop_curve_checks_lengths():
    with pytest.raises(ValueError, match="must match"):
        hp.cop_curve(np.zeros(2), np.zeros(3), np.zeros(2))


def test_electricity_interpolates_and_keeps_the_sign():
    curve = pd.DataFrame({"t_out": [0.0, 10.0], "cop": [2.0, 4.0]})
    out = hp.electricity_from_useful(
        np.array([4.0, 4.0, 4.0, -4.0]), np.array([0.0, 10.0, 5.0, 20.0]), curve
    )
    assert out.tolist() == [2.0, 1.0, pytest.approx(4.0 / 3.0), -1.0]


def test_electricity_checks_input():
    empty = pd.DataFrame(columns=["t_out", "cop"])
    with pytest.raises(ValueError, match="empty"):
        hp.electricity_from_useful(np.zeros(1), np.zeros(1), empty)
    curve = pd.DataFrame({"t_out": [0.0], "cop": [2.0]})
    with pytest.raises(ValueError, match="same length"):
        hp.electricity_from_useful(np.zeros(2), np.zeros(3), curve)


def test_a_curve_measured_on_runs_reproduces_their_total():
    rng = np.random.default_rng(0)
    t = rng.uniform(-10, 10, 2000)
    cop = 3.0 + 0.1 * t
    useful = rng.uniform(1, 5, 2000)
    curve = hp.cop_curve(
        useful, useful / cop, t, bin_width=1.0, t_min=-11.0, t_max=11.0
    )
    back = hp.electricity_from_useful(useful, t, curve).sum()
    assert back == pytest.approx((useful / cop).sum(), rel=0.02)


def test_fan_electricity_goes_to_the_dominant_use():
    h, c = hp.fan_electricity_by_mode(
        np.array([1.0, 0.0, 1.0]), np.array([0.0, 2.0, 1.0]), np.array([3.0, 4.0, 5.0])
    )
    assert h.tolist() == [3.0, 0.0, 5.0]
    assert c.tolist() == [0.0, 4.0, 0.0]
