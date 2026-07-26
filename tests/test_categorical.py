from inferlab.categorical import (
    chi_squared_independence_test,
    fisher_exact_test,
)
from inferlab.datasets import load_ptca

import numpy as np
from pytest import approx
import pytest
import scipy
from scipy import stats


@pytest.fixture
def ptca_table():
    return load_ptca()


@pytest.fixture
def homogeneity_table():
    """k x 2 table: three groups with a binary outcome."""
    group_sizes = np.array([48, 83, 60])
    successes = np.array([35, 76, 40])
    return np.column_stack([successes, group_sizes - successes])


# 1. PTCA example

def test_chi_squared_reproduces_example(ptca_table):
    statistic, p = chi_squared_independence_test(ptca_table)

    assert statistic == approx(8.1767, abs=0.001)
    assert p == approx(0.004243, abs=1e-5)
    assert p < 0.01 


def test_chi_squared_expected_frequencies_match_example(ptca_table):
    n = ptca_table.sum()
    row_sums = ptca_table.sum(axis=1)
    col_sums = ptca_table.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    assert expected == approx(
        np.array([[40.67, 7.33], [70.33, 12.67]]), abs=0.01
    )
    # one expected count is well below 10, which is why Fisher's exact test
    # is the more trustworthy choice on this table
    assert expected.min() < 10


def test_fisher_reproduces_example_table(ptca_table):
    p = fisher_exact_test(ptca_table, alternative="two_sided")

    assert p == approx(0.005824, abs=1e-5)
    assert p < 0.01


# 2. Wald / chi-squared duality

def test_chi_squared_equals_squared_wald_statistic(ptca_table):
    # in the 2x2 case the chi-squared statistic is exactly the square of the
    # Wald statistic on the difference of proportions
    row_sums = ptca_table.sum(axis=1)
    col_sums = ptca_table.sum(axis=0)
    n = ptca_table.sum()

    p1 = ptca_table[0, 0] / row_sums[0]
    p2 = ptca_table[1, 0] / row_sums[1]
    p_pooled = col_sums[0] / n

    wald = (p1 - p2) / np.sqrt(
        p_pooled * (1 - p_pooled) * (1 / row_sums[0] + 1 / row_sums[1])
    )
    wald_p = 2 * scipy.stats.norm.sf(abs(wald))

    statistic, p = chi_squared_independence_test(ptca_table)

    assert statistic == approx(wald**2, rel=1e-12)
    assert p == approx(wald_p, rel=1e-12)


def test_chi_squared_equals_homogeneity_statistic(homogeneity_table):
    group_sizes = homogeneity_table.sum(axis=1)
    successes = homogeneity_table[:, 0]

    p_hat_j = successes / group_sizes
    p_hat = successes.sum() / group_sizes.sum()
    D = np.sum(group_sizes * (p_hat_j - p_hat) ** 2 / (p_hat * (1 - p_hat)))

    statistic, _ = chi_squared_independence_test(homogeneity_table)

    assert statistic == approx(D, rel=1e-12)


# 3. Comparison against scipy

def test_chi_squared_matches_scipy_2x2(ptca_table):
    # correction=False: scipy applies Yates' continuity correction to 2x2
    # tables by default, which this implementation deliberately does not
    statistic, p = chi_squared_independence_test(ptca_table)
    ref_statistic, ref_p, _, _ = scipy.stats.chi2_contingency(
        ptca_table, correction=False
    )

    assert statistic == approx(ref_statistic, rel=1e-12)
    assert p == approx(ref_p, rel=1e-12)


def test_chi_squared_matches_scipy_3x3():
    table = np.array([[20, 15, 10], [12, 25, 18], [8, 11, 30]])

    statistic, p = chi_squared_independence_test(table)
    ref_statistic, ref_p, ref_df, _ = scipy.stats.chi2_contingency(
        table, correction=False
    )

    assert statistic == approx(ref_statistic, rel=1e-12)
    assert p == approx(ref_p, rel=1e-12)
    assert ref_df == 4  # (3-1)*(3-1)


def test_chi_squared_matches_scipy_for_rectangular_table():
    table = np.array([[10, 20, 30, 15], [25, 15, 20, 40]])

    statistic, p = chi_squared_independence_test(table)
    ref_statistic, ref_p, ref_df, _ = scipy.stats.chi2_contingency(
        table, correction=False
    )

    assert statistic == approx(ref_statistic, rel=1e-12)
    assert p == approx(ref_p, rel=1e-12)
    assert ref_df == 3  # (2-1)*(4-1)


def test_fisher_matches_scipy_all_alternatives(ptca_table):
    for alternative, scipy_alternative in [
        ("two_sided", "two-sided"),
        ("greater", "greater"),
        ("less", "less"),
    ]:
        p = fisher_exact_test(ptca_table, alternative=alternative)
        _, ref_p = scipy.stats.fisher_exact(
            ptca_table, alternative=scipy_alternative
        )
        assert p == approx(ref_p, rel=1e-12)


@pytest.mark.parametrize(
    "table",
    [
        np.array([[8, 2], [1, 5]]),      # small counts
        np.array([[1, 9], [11, 3]]),     # strong negative association
        np.array([[3, 3], [3, 3]]),      # perfectly balanced
        np.array([[0, 5], [5, 0]]),      # zero cells
        np.array([[10, 20], [30, 40]]),  # moderate, no association
    ],
)
def test_fisher_matches_scipy_across_tables(table):
    for alternative, scipy_alternative in [
        ("two_sided", "two-sided"),
        ("greater", "greater"),
        ("less", "less"),
    ]:
        p = fisher_exact_test(table, alternative=alternative)
        _, ref_p = scipy.stats.fisher_exact(
            table, alternative=scipy_alternative
        )
        assert p == approx(ref_p, rel=1e-10)


# 4. Structural properties

def test_chi_squared_invariant_under_transposition(ptca_table):
    # independence is symmetric in rows and columns
    statistic, p = chi_squared_independence_test(ptca_table)
    statistic_t, p_t = chi_squared_independence_test(ptca_table.T)

    assert statistic == approx(statistic_t, rel=1e-12)
    assert p == approx(p_t, rel=1e-12)


def test_fisher_invariant_under_transposition(ptca_table):
    p = fisher_exact_test(ptca_table, alternative="two_sided")
    p_t = fisher_exact_test(ptca_table.T, alternative="two_sided")

    assert p == approx(p_t, rel=1e-12)


def test_chi_squared_zero_for_perfectly_independent_table():
    # rows are exact multiples of each other: observed equals expected
    table = np.array([[10, 20], [20, 40]])
    statistic, p = chi_squared_independence_test(table)

    assert statistic == approx(0.0, abs=1e-12)
    assert p == approx(1.0)


def test_chi_squared_statistic_is_non_negative():
    rng = np.random.default_rng(77)
    for _ in range(20):
        table = rng.integers(1, 50, size=(3, 4))
        statistic, _ = chi_squared_independence_test(table)
        assert statistic >= 0


def test_chi_squared_detects_strong_association():
    table = np.array([[90, 10], [10, 90]])
    _, p = chi_squared_independence_test(table)
    assert p < 1e-10


def test_fisher_is_more_conservative_than_chi_squared(ptca_table):
    # the exact test does not rely on the asymptotic approximation and
    # yields a slightly larger p-value on this table
    _, chi_squared_p = chi_squared_independence_test(ptca_table)
    fisher_p = fisher_exact_test(ptca_table, alternative="two_sided")

    assert fisher_p > chi_squared_p


def test_fisher_and_chi_squared_agree_for_large_samples():
    # with large expected counts the asymptotic approximation is good and
    # both tests lead to the same conclusion
    table = np.array([[300, 200], [250, 250]])

    _, chi_squared_p = chi_squared_independence_test(table)
    fisher_p = fisher_exact_test(table, alternative="two_sided")

    assert chi_squared_p < 0.01
    assert fisher_p < 0.01
    assert fisher_p == approx(chi_squared_p, abs=0.001)


def test_fisher_one_sided_p_values_bracket_the_two_sided():
    # each one-sided p-value is at most the two-sided one on the relevant side
    table = np.array([[8, 2], [1, 5]])

    p_two = fisher_exact_test(table, alternative="two_sided")
    p_greater = fisher_exact_test(table, alternative="greater")
    p_less = fisher_exact_test(table, alternative="less")

    assert p_greater <= p_two
    assert 0.0 <= p_greater <= 1.0
    assert 0.0 <= p_less <= 1.0
    assert 0.0 <= p_two <= 1.0


def test_fisher_symmetric_table_gives_p_value_one():
    # a perfectly balanced table gives no evidence against independence
    table = np.array([[3, 3], [3, 3]])
    p = fisher_exact_test(table, alternative="two_sided")
    assert p == approx(1.0)

