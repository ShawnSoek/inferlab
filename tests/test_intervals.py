from inferlab.intervals import (
    ci_mean_known_sigma,
    ci_mean_unknown_sigma,
    ci_variance,
    bootstrap_percentile_ci,
    bootstrap_t_ci_mean,
)

import numpy as np
from pytest import approx
import scipy



# Comparision against scipy

def test_ci_mean_known_sigma_matches_scipy():
    mean, sigma, n, confidence = 100, 15, 25, 0.95
    ci = ci_mean_known_sigma(mean=mean, sigma=sigma, n=n, confidence_level=confidence)
    reference = scipy.stats.norm.interval(confidence, loc=mean, scale=sigma / np.sqrt(n))
    assert ci == approx(reference, rel=1e-9)


def test_ci_mean_unknown_sigma_matches_scipy():
    mean, s, n, confidence = 100, 15, 25, 0.95
    ci = ci_mean_unknown_sigma(mean=mean, s=s, n=n, confidence_level=confidence)
    reference = scipy.stats.t.interval(confidence, df=n - 1, loc=mean, scale=s / np.sqrt(n))
    assert ci == approx(reference, rel=1e-9)


# Test structural propersites

def test_bounds_are_ordered():
    assert (lambda low, high: low < high)(*ci_mean_known_sigma(100, 15, 25, 0.95)) # * unpacks tuple, lambda function is applied to the values
    assert (lambda low, high: low < high)(*ci_mean_unknown_sigma(100, 15, 25, 0.95))
    assert (lambda low, high: low < high)(*ci_variance(15, 25, 0.95))


def test_mean_intervals_are_symmetric():
    # sample mean is the midpoint of the intervals
    mean, sigma, n, confidence = 100, 15, 25, 0.95
    low, high = ci_mean_known_sigma(mean, sigma, n, confidence)
    assert (low + high) / 2 == approx(mean)

    s = 15
    low, high = ci_mean_unknown_sigma(mean, s, n, confidence)
    assert (low + high) / 2 == approx(mean)


def test_variance_interval_is_asymmetric():
    # Since chigh-squared is skewed the variance interval should not be symmetric
    s, n, confidence = 15, 25, 0.95
    low, high = ci_variance(s=s, n=n, confidence_level=confidence)
    midpoint = (low + high) / 2
    assert midpoint != approx(s**2, rel=1e-3)


def test_highgher_confidence_widens_interval():
    # a 99% interval is wider than a 95% interval on the same data
    def width(interval):
        low, high = interval
        return high - low

    args = {"mean": 100, "sigma": 15, "n": 25}
    w95 = width(ci_mean_known_sigma(**args, confidence_level=0.95))
    w99 = width(ci_mean_known_sigma(**args, confidence_level=0.99))
    assert w99 > w95

    args = {"mean": 100, "s": 15, "n": 25}
    w95 = width(ci_mean_unknown_sigma(**args, confidence_level=0.95))
    w99 = width(ci_mean_unknown_sigma(**args, confidence_level=0.99))
    assert w99 > w95

    args = {"s": 15, "n": 25}
    w95 = width(ci_variance(**args, confidence_level=0.95))
    w99 = width(ci_variance(**args, confidence_level=0.99))
    assert w99 > w95


# Tests for Bootstrap CIs

def test_variance_interval_contains_s_squared():
    # s**2 must lie inside its own confidence interval
    s, n, confidence = 15, 25, 0.95
    low, high = ci_variance(s=s, n=n, confidence_level=confidence)
    assert low < s**2 < high

def test_bootstrap_intervals_not_random_with_same_rng():
    data = np.random.default_rng(42).normal(100, 15, 10000)

    percentile_ci1 = bootstrap_percentile_ci(data=data, statistic=np.var, B=1000, confidence_level=0.95, rng=np.random.default_rng(42))
    percentile_ci2 = bootstrap_percentile_ci(data=data, statistic=np.var, B=1000, confidence_level=0.95, rng=np.random.default_rng(42))
    assert percentile_ci1 == percentile_ci2

    t_ci_mean1 = bootstrap_t_ci_mean(data=data, B=1000, confidence_level=0.95, rng=np.random.default_rng(42))
    t_ci_mean2 = bootstrap_t_ci_mean(data=data, B=1000, confidence_level=0.95, rng=np.random.default_rng(42))
    assert t_ci_mean1 == t_ci_mean2


def test_bootstrap_percentile_contains_point_estimate():
    data = np.random.default_rng(7).normal(10000, 15, 500)
    low, high = bootstrap_percentile_ci(data=data, statistic=np.mean, B=5000, confidence_level=0.95, rng=np.random.default_rng(42))
    assert low < np.mean(data) < high


def test_bootstrap_percentile_approaches_classical_ci():
    # for large n and normal data, the bootstrap percentile interval converges to the classical t-interval
    data = np.random.default_rng(7).normal(10000, 15, 500)

    boot = bootstrap_percentile_ci(data=data, statistic=np.mean, B=5000, confidence_level=0.95, rng=np.random.default_rng(42))
    classical = ci_mean_unknown_sigma(mean=np.mean(data), s=np.std(data, ddof=1), n=len(data),confidence_level=0.95)
    assert boot == approx(classical, rel=0.02)