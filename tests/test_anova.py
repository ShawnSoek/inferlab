from inferlab.anova import one_way_anova, bonferroni_pairwise, tukey_hsd
from inferlab.hypothesis_tests import two_sample_t_test

import numpy as np
from pytest import approx
import pytest
import scipy
from scipy import stats


@pytest.fixture
def three_groups():
    rng = np.random.default_rng(7)
    return [
        rng.normal(50, 5, 15),
        rng.normal(55, 5, 20),
        rng.normal(52, 5, 18),
    ]


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(3)
    return [rng.normal(50, 5, 20), rng.normal(54, 5, 25)]


@pytest.fixture
def identical_groups():
    """Four groups drawn from the same distribution: H0 is true."""
    rng = np.random.default_rng(101)
    return [rng.normal(50, 5, 30) for _ in range(4)]


# 1. Comparison against scipy

def test_one_way_anova_matches_scipy(three_groups):
    F, p = one_way_anova(three_groups)
    reference = scipy.stats.f_oneway(*three_groups)

    assert F == approx(reference.statistic, rel=1e-10)
    assert p == approx(reference.pvalue, rel=1e-10)


def test_one_way_anova_matches_scipy_for_equal_group_sizes():
    rng = np.random.default_rng(55)
    groups = [rng.normal(m, 4, 25) for m in (10, 12, 11, 14)]

    F, p = one_way_anova(groups)
    reference = scipy.stats.f_oneway(*groups)

    assert F == approx(reference.statistic, rel=1e-10)
    assert p == approx(reference.pvalue, rel=1e-10)


def test_tukey_matches_scipy(three_groups):
    comparisons = tukey_hsd(three_groups)
    reference = scipy.stats.tukey_hsd(*three_groups)

    for comparison in comparisons:
        expected_p = reference.pvalue[comparison.group1, comparison.group2]
        assert comparison.p_value == approx(expected_p, rel=1e-8)


# 2. The variance decomposition

def test_sum_of_squares_decomposition_holds(three_groups):
    # SS_total = SS_between + SS_within is an algebraic identity, not an
    # approximation. It must hold exactly
    all_data = np.concatenate(three_groups)
    grand_mean = np.mean(all_data)

    ss_total = np.sum((all_data - grand_mean) ** 2)
    ss_between = np.sum(
        [len(g) * (np.mean(g) - grand_mean) ** 2 for g in three_groups]
    )
    ss_within = np.sum([np.sum((g - np.mean(g)) ** 2) for g in three_groups])

    assert ss_total == approx(ss_between + ss_within, rel=1e-12)


# 3. The t-test / ANOVA duality

def test_anova_equals_squared_t_statistic_for_two_groups(two_groups):
    # for k = 2 the one-way ANOVA is exactly the pooled two-sample t-test,
    # with F = t^2 and identical p-values
    x, y = two_groups

    t, t_p = two_sample_t_test(
        mean1=np.mean(x), s1=np.std(x, ddof=1), n1=len(x),
        mean2=np.mean(y), s2=np.std(y, ddof=1), n2=len(y),
        alternative="two_sided",
    )
    F, F_p = one_way_anova(two_groups)

    assert F == approx(t**2, rel=1e-10)
    assert F_p == approx(t_p, rel=1e-10)


# 4. Invariance properties

def test_anova_invariant_under_shift(three_groups):
    # adding the same constant to every observation changes no deviation
    shifted = [g + 100 for g in three_groups]

    F, p = one_way_anova(three_groups)
    F_shifted, p_shifted = one_way_anova(shifted)

    assert F == approx(F_shifted, rel=1e-10)
    assert p == approx(p_shifted, rel=1e-10)


def test_anova_invariant_under_scaling(three_groups):
    # F is a ratio of sums of squares, so a common scale factor cancels
    scaled = [g * 3 for g in three_groups]

    F, p = one_way_anova(three_groups)
    F_scaled, p_scaled = one_way_anova(scaled)

    assert F == approx(F_scaled, rel=1e-10)
    assert p == approx(p_scaled, rel=1e-10)


def test_anova_invariant_under_group_reordering(three_groups):
    F, p = one_way_anova(three_groups)
    F_reversed, p_reversed = one_way_anova(three_groups[::-1])

    assert F == approx(F_reversed, rel=1e-12)
    assert p == approx(p_reversed, rel=1e-12)


# 5. Detection behaviour

def test_anova_does_not_reject_identical_distributions(identical_groups):
    _, p = one_way_anova(identical_groups)
    assert p > 0.05


def test_anova_detects_clearly_different_means():
    rng = np.random.default_rng(102)
    groups = [rng.normal(m, 2, 30) for m in (50, 70, 90)]

    F, p = one_way_anova(groups)
    assert F > 100
    assert p < 1e-20


def test_anova_statistic_is_non_negative():
    rng = np.random.default_rng(103)
    for _ in range(20):
        groups = [rng.normal(50, 5, rng.integers(5, 20)) for _ in range(3)]
        F, _ = one_way_anova(groups)
        assert F >= 0


# 6. Multiple comparison procedures

@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_number_of_pairwise_comparisons(k):
    rng = np.random.default_rng(200 + k)
    groups = [rng.normal(50, 5, 12) for _ in range(k)]

    expected = k * (k - 1) // 2
    assert len(bonferroni_pairwise(groups)) == expected
    assert len(tukey_hsd(groups)) == expected


def test_bonferroni_multiplies_raw_p_values(three_groups):
    comparisons = bonferroni_pairwise(three_groups)
    n_comparisons = len(comparisons)

    for comparison in comparisons:
        gi = three_groups[comparison.group1]
        gj = three_groups[comparison.group2]
        _, raw_p = two_sample_t_test(
            mean1=np.mean(gi), s1=np.std(gi, ddof=1), n1=len(gi),
            mean2=np.mean(gj), s2=np.std(gj, ddof=1), n2=len(gj),
            alternative="two_sided",
        )
        expected = min(raw_p * n_comparisons, 1.0)
        assert comparison.p_value == approx(expected, rel=1e-12)


def test_bonferroni_is_more_conservative_than_tukey(three_groups):
    # Tukey is tailored to all-pairwise comparisons and therefore loses less
    # power than the general-purpose Bonferroni bound
    bonferroni = bonferroni_pairwise(three_groups)
    tukey = tukey_hsd(three_groups)

    for b, t in zip(bonferroni, tukey):
        assert b.group1 == t.group1
        assert b.group2 == t.group2
        assert b.p_value >= t.p_value


def test_multiple_comparisons_detect_the_separated_group():
    # groups 0 and 1 are far apart, group 2 sits with group 1
    rng = np.random.default_rng(205)
    groups = [rng.normal(50, 3, 25), rng.normal(70, 3, 25), rng.normal(70, 3, 25)]

    results = {(c.group1, c.group2): c.p_value for c in tukey_hsd(groups)}

    assert results[(0, 1)] < 0.001
    assert results[(0, 2)] < 0.001
    assert results[(1, 2)] > 0.05


def test_tukey_uses_pooled_within_groups_variance(three_groups):
    # the standard error must use MS_within from all groups, not just the pair
    k = len(three_groups)
    n = sum(len(g) for g in three_groups)
    df_within = n - k
    ss_within = np.sum([np.sum((g - np.mean(g)) ** 2) for g in three_groups])
    mean_square_within = ss_within / df_within

    comparisons = tukey_hsd(three_groups)
    first = comparisons[0]
    gi, gj = three_groups[first.group1], three_groups[first.group2]

    standard_error = np.sqrt(
        mean_square_within / 2 * (1 / len(gi) + 1 / len(gj))
    )
    expected_q = abs(np.mean(gi) - np.mean(gj)) / standard_error

    assert first.statistic == approx(expected_q, rel=1e-12)


# 7. Error paths

def test_one_way_anova_rejects_single_group():
    with pytest.raises(ValueError):
        one_way_anova([np.array([1.0, 2.0, 3.0])])


def test_one_way_anova_rejects_empty_group():
    with pytest.raises(ValueError):
        one_way_anova([np.array([1.0, 2.0]), np.array([])])


def test_pairwise_procedures_reject_single_group():
    single = [np.array([1.0, 2.0, 3.0])]
    with pytest.raises(ValueError):
        bonferroni_pairwise(single)
    with pytest.raises(ValueError):
        tukey_hsd(single)