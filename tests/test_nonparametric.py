from inferlab.nonparametric import (
    index_of_dispersion_test,
    sign_test,
    chi_squared_goodness_of_fit,
    wilcoxon_rank_sum_test,
)
from inferlab.datasets import load_rutherford, load_emergencyroom

import numpy as np
from pytest import approx
import pytest
import scipy
from scipy import stats


@pytest.fixture
def rutherford_stats():
    """Sample size, mean and standard deviation of the Rutherford data."""
    data = load_rutherford("raw")
    return len(data), np.mean(data), np.std(data, ddof=1)


@pytest.fixture
def rutherford_grouped():
    """Rutherford counts grouped into 12 classes: 0..10 and '11+'."""
    table = load_rutherford("table")
    data = load_rutherford("raw")

    observed = np.array(
        [table.get(k, 0) for k in range(11)]
        + [sum(count for k, count in table.items() if k >= 11)]
    )
    lambda_hat = np.mean(data)
    probabilities = np.array(
        [stats.poisson.pmf(k, lambda_hat) for k in range(11)]
        + [stats.poisson.sf(10, lambda_hat)]  # sf(10) = P(X >= 11)
    )
    return observed, probabilities


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(9)
    return rng.normal(100, 15, 25), rng.normal(92, 15, 30)



# 1. Index of dispersion test


def test_dispersion_test_reproduces_lecture_value(rutherford_stats):
    n, mean, s = rutherford_stats
    d, p = index_of_dispersion_test(mean=mean, s=s, n=n)

    assert d == approx(2489, abs=1)
    assert p == approx(0.1, abs=0.01)
    assert p > 0.05


def test_dispersion_test_accepts_genuine_poisson_data():
    # data actually drawn from a Poisson distribution must not be rejected
    rng = np.random.default_rng(31)
    data = rng.poisson(lam=4.0, size=2000)

    _, p = index_of_dispersion_test(
        mean=np.mean(data), s=np.std(data, ddof=1), n=len(data)
    )
    assert p > 0.05


def test_dispersion_test_rejects_overdispersed_data():
    # negative binomial data are overdispersed (variance > mean) and must be
    # rejected as Poisson
    rng = np.random.default_rng(32)
    data = rng.negative_binomial(n=3, p=0.4, size=2000)

    _, p = index_of_dispersion_test(
        mean=np.mean(data), s=np.std(data, ddof=1), n=len(data)
    )
    assert p < 0.01


# 2. Sign test

def test_sign_test_reproduces_arbuthnot_example():
    # Arbuthnot observed more male than female births in all 82 years;
    data = np.ones(82)
    count, p = sign_test(data, m0=0, alternative="greater")

    assert count == 82
    assert p == approx(0.5**82, rel=1e-12)


def test_sign_test_matches_scipy_binomtest(two_groups):
    data = load_emergencyroom()

    for m0 in [20, 30, 45]:
        n_above = int(np.sum(data > m0))
        n_effective = n_above + int(np.sum(data < m0))

        count, p = sign_test(data, m0=m0, alternative="two_sided")
        reference = scipy.stats.binomtest(
            n_above, n=n_effective, p=0.5, alternative="two-sided"
        )

        assert count == n_above
        assert p == approx(reference.pvalue, rel=1e-12)


def test_sign_test_excludes_ties_from_sample_size():
    # values exactly equal to m0 carry no sign and must be dropped entirely
    data = np.array([1.0, 2.0, 5.0, 5.0, 5.0, 8.0, 9.0])
    count, p = sign_test(data, m0=5.0, alternative="two_sided")

    assert count == 2  # only 8 and 9 lie above
    # effective n is 4 not 7
    expected = scipy.stats.binomtest(2, n=4, p=0.5, alternative="two-sided").pvalue
    assert p == approx(expected, rel=1e-12)


def test_sign_test_at_sample_median_gives_large_p_value():
    data = load_emergencyroom()
    _, p = sign_test(data, m0=np.median(data), alternative="two_sided")
    assert p > 0.5


def test_sign_test_rejects_unknown_alternative():
    with pytest.raises(ValueError):
        sign_test(load_emergencyroom(), m0=30, alternative="not_an_alternative")


# 3. Chi-squared goodness of fit

def test_goodness_of_fit_reproduces_lecture_value(rutherford_grouped):
    observed, probabilities = rutherford_grouped

    statistic, p = chi_squared_goodness_of_fit(
        observed, probabilities, n_estimated_params=1
    )

    assert statistic == approx(13.0, abs=0.1)
    assert p > 0.1  # Poisson hypothesis not rejected


def test_goodness_of_fit_degrees_of_freedom_matter(rutherford_grouped):
    # estimating lambda from the data costs one degree of freedom:
    # df = 12 - 1 - 1 = 10 instead of df = 12 - 1 = 11
    observed, probabilities = rutherford_grouped

    statistic_estimated, p_estimated = chi_squared_goodness_of_fit(
        observed, probabilities, n_estimated_params=1
    )
    statistic_known, p_known = chi_squared_goodness_of_fit(
        observed, probabilities, n_estimated_params=0
    )

    # same statistic, different null distribution
    assert statistic_estimated == approx(statistic_known)
    # fewer degrees of freedom -> smaller p-value (stricter test)
    assert p_estimated < p_known
    assert p_estimated == approx(stats.chi2.sf(statistic_estimated, df=10), rel=1e-12)
    assert p_known == approx(stats.chi2.sf(statistic_known, df=11), rel=1e-12)


def test_goodness_of_fit_perfect_match_gives_zero_statistic():
    # if observed frequencies match expectations exactly, X^2 = 0 and p = 1
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    observed = np.array([100, 100, 100, 100])

    statistic, p = chi_squared_goodness_of_fit(
        observed, probabilities, n_estimated_params=0
    )

    assert statistic == approx(0.0)
    assert p == approx(1.0)


def test_goodness_of_fit_matches_scipy_for_simple_hypothesis():
    # with fully specified probabilities, scipy.stats.chisquare is equivalent
    observed = np.array([89, 37, 30, 28, 2])
    probabilities = np.array([0.40, 0.20, 0.20, 0.15, 0.05])

    statistic, p = chi_squared_goodness_of_fit(
        observed, probabilities, n_estimated_params=0
    )
    reference = scipy.stats.chisquare(observed, f_exp=observed.sum() * probabilities)

    assert statistic == approx(reference.statistic, rel=1e-12)
    assert p == approx(reference.pvalue, rel=1e-12)


def test_goodness_of_fit_detects_bad_fit():
    # observed frequencies far from a uniform expectation must be rejected
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    observed = np.array([200, 100, 60, 40])

    _, p = chi_squared_goodness_of_fit(observed, probabilities, n_estimated_params=0)
    assert p < 0.001


# 4. Wilcoxon rank-sum test

def test_wilcoxon_matches_scipy_all_alternatives(two_groups):
    x, y = two_groups

    for alternative, scipy_alternative in [
        ("two_sided", "two-sided"),
        ("greater", "greater"),
        ("less", "less"),
    ]:
        _, p = wilcoxon_rank_sum_test(x, y, alternative=alternative)
        reference = scipy.stats.mannwhitneyu(
            x, y, alternative=scipy_alternative, method="asymptotic"
        )
        assert p == approx(reference.pvalue, rel=1e-12)


def test_wilcoxon_matches_scipy_with_ties():
    # the tie correction shrinks the variance; without it the p-value is wrong
    x = np.array([1, 2, 2, 3, 4])
    y = np.array([2, 3, 3, 5])

    for alternative, scipy_alternative in [
        ("two_sided", "two-sided"),
        ("greater", "greater"),
        ("less", "less"),
    ]:
        _, p = wilcoxon_rank_sum_test(x, y, alternative=alternative)
        reference = scipy.stats.mannwhitneyu(
            x, y, alternative=scipy_alternative, method="asymptotic"
        )
        assert p == approx(reference.pvalue, rel=1e-12)


def test_wilcoxon_matches_scipy_when_first_group_is_smaller():
    # guards against a sign error: here W lies below its expectation
    rng = np.random.default_rng(21)
    x = rng.normal(90, 10, 20)
    y = rng.normal(100, 10, 22)

    for alternative, scipy_alternative in [
        ("two_sided", "two-sided"),
        ("greater", "greater"),
        ("less", "less"),
    ]:
        _, p = wilcoxon_rank_sum_test(x, y, alternative=alternative)
        reference = scipy.stats.mannwhitneyu(
            x, y, alternative=scipy_alternative, method="asymptotic"
        )
        assert p == approx(reference.pvalue, rel=1e-12)


def test_wilcoxon_swapping_groups_flips_statistic(two_groups):
    x, y = two_groups

    z_forward, p_forward = wilcoxon_rank_sum_test(x, y, alternative="two_sided")
    z_reverse, p_reverse = wilcoxon_rank_sum_test(y, x, alternative="two_sided")

    assert z_forward == approx(-z_reverse)
    assert p_forward == approx(p_reverse)


def test_wilcoxon_one_sided_p_values_are_complementary(two_groups):
    # with the continuity correction switched off, the two one-sided p-values
    # are based on the same z and therefore sum to one
    x, y = two_groups

    _, p_greater = wilcoxon_rank_sum_test(
        x, y, alternative="greater", continuity_correction=False
    )
    _, p_less = wilcoxon_rank_sum_test(
        x, y, alternative="less", continuity_correction=False
    )

    assert p_greater + p_less == approx(1.0)


def test_wilcoxon_detects_shifted_distributions():
    rng = np.random.default_rng(41)
    x = rng.normal(100, 5, 40)
    y = rng.normal(90, 5, 40)

    _, p = wilcoxon_rank_sum_test(x, y, alternative="two_sided")
    assert p < 0.001


def test_wilcoxon_accepts_identical_distributions():
    rng = np.random.default_rng(42)
    x = rng.normal(50, 8, 60)
    y = rng.normal(50, 8, 60)

    _, p = wilcoxon_rank_sum_test(x, y, alternative="two_sided")
    assert p > 0.05


def test_wilcoxon_rejects_unknown_alternative(two_groups):
    x, y = two_groups
    with pytest.raises(ValueError):
        wilcoxon_rank_sum_test(x, y, alternative="not_an_alternative")