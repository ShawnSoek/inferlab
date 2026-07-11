from inferlab.hypothesis_tests import (
    p_value,
    gauss_test,
    t_test,
    variance_test,
    log_likelihood,
    likelihood_ratio_test,
)
from inferlab.intervals import ci_mean_unknown_sigma
from inferlab.estimation import ml_estimate
from inferlab.datasets import load_rutherford

import numpy as np
from pytest import approx
import pytest
import scipy


@pytest.fixture
def normal_sample():
    return np.random.default_rng(5).normal(52, 8, 40)



# Comparison against scipy

def test_t_test_matches_scipy(normal_sample):
    mean = np.mean(normal_sample)
    s = np.std(normal_sample, ddof=1)
    n = len(normal_sample)

    statistic, p = t_test(mean=mean, mu0=50, s=s, n=n, alternative="two_sided")
    ref = scipy.stats.ttest_1samp(normal_sample, popmean=50)
    assert statistic == approx(ref.statistic, rel=1e-9)
    assert p == approx(ref.pvalue, rel=1e-9)

    _, p_greater = t_test(mean=mean, mu0=50, s=s, n=n, alternative="greater")
    ref_greater = scipy.stats.ttest_1samp(normal_sample, 50, alternative="greater")
    assert p_greater == approx(ref_greater.pvalue, rel=1e-9)

    _, p_less = t_test(mean=mean, mu0=50, s=s, n=n, alternative="less")
    ref_less = scipy.stats.ttest_1samp(normal_sample, 50, alternative="less")
    assert p_less == approx(ref_less.pvalue, rel=1e-9)



# The CI-test duality


def test_t_test_duality_with_confidence_interval(normal_sample):

    mean = np.mean(normal_sample)
    s = np.std(normal_sample, ddof=1)
    n = len(normal_sample)
    alpha = 0.05

    lower, upper = ci_mean_unknown_sigma(mean=mean, s=s, n=n, confidence_level=1 - alpha)

    for mu0 in [48, 50, 52, 54, 56, 58]:
        _, p = t_test(mean=mean, mu0=mu0, s=s, n=n, alternative="two_sided")
        rejects = p < alpha
        outside_ci = not (lower < mu0 < upper)
        assert rejects == outside_ci



# Properties of p-value


def test_two_sided_is_twice_the_smaller_tail(normal_sample):
    mean = np.mean(normal_sample)
    s = np.std(normal_sample, ddof=1)
    n = len(normal_sample)

    _, p_two_sided = t_test(mean, 50, s, n, alternative="two_sided")
    _, p_less = t_test(mean, 50, s, n, alternative="less")
    _, p_greater = t_test(mean, 50, s, n, alternative="greater")

    assert p_two_sided == approx(2 * min(p_less, p_greater))


def test_p_values_lie_in_unit_interval(normal_sample):
    mean = np.mean(normal_sample)
    s = np.std(normal_sample, ddof=1)
    n = len(normal_sample)

    for mu0 in (30, 50, 52, 70):
        for alt in ("two_sided", "less", "greater"):
            _, p = t_test(mean, mu0, s, n, alternative=alt)
            assert 0.0 <= p <= 1.0


def test_p_value_maximal_when_h0_equals_estimate(normal_sample):
    mean = np.mean(normal_sample)
    s = np.std(normal_sample, ddof=1)
    n = len(normal_sample)

    _, p_at_estimate = t_test(mean, mean, s, n, alternative="two_sided")

    assert p_at_estimate == approx(1.0)


def test_p_value_rejects_unknown_alternative():
    with pytest.raises(ValueError):
        p_value(1.5, scipy.stats.norm, alternative="not_an_alternative")



# Gauss test and variance test


def test_gauss_test_statistic_and_p_value():
    mean, mu0, sigma, n = 105, 100, 15, 25
    z, p = gauss_test(mean=mean, mu0=mu0, sigma=sigma, n=n, alternative="two_sided")

    expected_z = (mean - mu0) / (sigma / np.sqrt(n))
    expected_p = 2 * scipy.stats.norm.sf(abs(expected_z))

    assert z == approx(expected_z)
    assert p == approx(expected_p)


def test_variance_test_statistic_and_p_value():

    s, sigma0, n = 14, 10, 25
    statistic, p = variance_test(s=s, sigma0=sigma0, n=n, alternative="two_sided")

    expected_stat = (n - 1) * s**2 / sigma0**2
    null = scipy.stats.chi2(df=n - 1)
    expected_p = min(2 * min(null.sf(expected_stat), null.cdf(expected_stat)), 1.0)

    assert statistic == approx(expected_stat)
    assert p == approx(expected_p)


def test_variance_test_p_value_large_when_h0_true():
    s, n = 12, 30
    _, p = variance_test(s=s, sigma0=s, n=n, alternative="two_sided")
    assert p > 0.5



# Likelihood ratio test


def test_log_likelihood_maximal_at_ml_estimate():
    data = load_rutherford("raw")
    lambda_hat = ml_estimate(data, "poisson")

    ll_at_ml = log_likelihood(data, "poisson", lambda_hat)
    for other in [2.0, 3.0, 4.5, 6.0]:
        assert log_likelihood(data, "poisson", other) <= ll_at_ml


def test_lrt_does_not_reject_true_null():
    data = load_rutherford("raw")
    lambda_hat = ml_estimate(data, "poisson")[0]

    w, p = likelihood_ratio_test(data, "poisson", lambda_hat)
    assert w == approx(0.0, abs=1e-6)
    assert p > 0.99


def test_lrt_rejects_false_null():
    data = load_rutherford("raw")
    w, p = likelihood_ratio_test(data, "poisson", 5.0)

    assert w > 100
    assert p < 0.001


def test_lrt_statistic_is_non_negative():
    # the unrestricted model can never fit worse than the restricted one
    data = load_rutherford("raw")
    for lambda0 in [1.0, 3.0, 3.87, 4.0, 8.0]:
        w, _ = likelihood_ratio_test(data, "poisson", lambda0)
        assert w >= 0