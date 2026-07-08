from inferlab.estimation import ml_estimate, moment_estimate, bayes_binomial, bayes_normal
from inferlab.datasets import load_emergencyroom, load_iq, load_reactiontime, load_rutherford

import numpy as np

from pytest import approx
import pytest

import scipy


@pytest.fixture
def normal_data():
    rng = np.random.default_rng(42)
    return rng.normal(100, 15, 1000)


@pytest.fixture
def exp_data():
    rng = np.random.default_rng(43)
    return rng.exponential(scale=1 / 0.05, size=2000)


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(44)
    return rng.poisson(lam=4.2, size=5000)


@pytest.fixture
def geom_data():
    rng = np.random.default_rng(45)
    return rng.geometric(p=0.3, size=5000) - 1


def test_ml_estimate(normal_data, exp_data, poisson_data, geom_data):
    # reference values
    assert ml_estimate(load_rutherford("raw"), "poisson") == approx(3.87, rel=1e-3)
    assert ml_estimate(load_emergencyroom(), "exponential") == approx(0.0304, rel=1e-3)

    # scipy normal
    normal_ml = ml_estimate(normal_data, "normal")
    normal_ref = scipy.stats.norm.fit(normal_data)
    assert normal_ml == approx(normal_ref, rel=1e-2)

    # scipy exponential
    exp_ml = ml_estimate(exp_data, "exponential")
    exp_loc, exp_scale = scipy.stats.expon.fit(exp_data, floc=0)
    exp_ref = 1 / exp_scale
    assert exp_ml == approx(exp_ref, rel=1e-2)

    # poisson comparison with calculated result
    poisson_ml = ml_estimate(poisson_data, "poisson")
    poisson_ref = np.mean(poisson_data)
    assert poisson_ml == approx(poisson_ref, rel=1e-3)

    # geometric comparison with calculated result
    geom_ml = ml_estimate(geom_data, "geometric")
    geom_ref = 1 / (1 + np.mean(geom_data))
    assert geom_ml == approx(geom_ref, rel=1e-2)


def test_moment_estimate(normal_data, exp_data, poisson_data):
    # reference values
    assert moment_estimate(load_rutherford("raw"), "poisson") == approx(3.87, rel=1e-3)
    assert moment_estimate(load_emergencyroom(), "exponential") == approx(0.0304, rel=1e-3)

    # comparison to ml_estimates
    assert moment_estimate(normal_data, "normal") == approx(ml_estimate(normal_data, "normal"), rel=1e-3)
    assert moment_estimate(exp_data, "exponential") == approx(ml_estimate(exp_data, "exponential"), rel=1e-3)
    assert moment_estimate(poisson_data, "poisson") == approx(ml_estimate(poisson_data, "poisson"), rel=1e-3)

    # scipy normal moment method
    normal_mm = moment_estimate(normal_data, "normal")
    normal_mm_ref = scipy.stats.norm.fit(normal_data, method="MM")
    assert normal_mm == approx(normal_mm_ref, rel=1e-2)

    # scipy exponential moment method
    exp_mm = moment_estimate(exp_data, "exponential")
    exp_mm_loc, exp_mm_scale = scipy.stats.expon.fit(exp_data, method="MM", floc=0)
    exp_mm_ref = 1 / exp_mm_scale
    assert exp_mm == approx(exp_mm_ref, rel=1e-2)


def test_bayes_binomial():
    # reference value
    assert bayes_binomial(x=8, n=10, a=1, b=1) == 0.75

    # large n
    assert bayes_binomial(x=800, n=1000, a=1, b=1) == approx(0.8, rel=1e-2)

    # no data (n=0)
    assert bayes_binomial(x=0, n=0, a=2, b=3) == approx(0.4)

    # convex combination: estimate lies between prior mean and empirical rate
    prior_mean = 2 / (2 + 3)
    empirical_rate = 6 / 10
    estimate = bayes_binomial(x=6, n=10, a=2, b=3)
    assert min(prior_mean, empirical_rate) <= estimate <= max(prior_mean, empirical_rate)


def test_bayes_normal(normal_data):
    # uninformative prior (tau huge): estimate approaches sample mean
    uninformative = bayes_normal(normal_data, sigma=15, nu=0, tau=1e6)
    assert uninformative == approx(np.mean(normal_data), rel=1e-3)

    # informative prior (tau tiny): estimate approaches prior mean nu
    informative = bayes_normal(normal_data, sigma=15, nu=50, tau=1e-6)
    assert informative == approx(50, rel=1e-3)

    # convex combination: estimate lies between prior mean and sample mean
    nu = 80
    x_bar = np.mean(normal_data)
    estimate = bayes_normal(normal_data, sigma=15, nu=nu, tau=3.0)
    assert min(nu, x_bar) <= estimate <= max(nu, x_bar)

    # prior mean equals sample mean: estimate equals that common value
    estimate = bayes_normal(normal_data, sigma=15, nu=x_bar, tau=3.0)
    assert estimate == approx(x_bar)

def test_ml_estimate_invalid_distribution(normal_data):
    # unknown distribution key should raise a clear error
    with pytest.raises(ValueError):
        ml_estimate(normal_data, "not_a_distribution")


def test_moment_estimate_invalid_distribution(normal_data):
    # unknown distribution key should raise a clear error
    with pytest.raises(ValueError):
        moment_estimate(normal_data, "not_a_distribution")