import numpy as np
from scipy import stats
from collections import namedtuple
from itertools import combinations

from inferlab.hypothesis_tests import p_value
from inferlab.hypothesis_tests import two_sample_t_test


PairwiseComparison = namedtuple(
    "PairwiseComparison", ["group1", "group2", "statistic", "p_value"]
)

def one_way_anova(groups):
    """
    One-way analysis of variance for k independent groups.

    Decomposes the total sum of squares into a between-groups and a
    within-groups component (SS_total = SS_between + SS_within) and compares
    their mean squares. Under H0 (all group means equal), the ratio follows
    an F distribution with (k-1, n-k) degrees of freedom.

    The test is inherently one-sided: only large F values indicate that
    between-group variation exceeds what within-group noise would produce.

    Parameters
    ----------
    groups : sequence of array_like
        Observations of each group. Groups may have different sizes.

    Returns
    -------
    tuple of float
        (F statistic, p-value).

    Raises
    ------
    ValueError
        If fewer than two groups are given, or if any group is empty.
    """
    groups = [np.asarray(g) for g in groups]

    if len(groups) < 2:
        raise ValueError(f"Need at least two groups, got {len(groups)}.")
    if any(len(g) == 0 for g in groups):
        raise ValueError("All groups must contain at least one observation.")

    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    group_means = np.array([np.mean(g) for g in groups])
    group_sizes = np.array([len(g) for g in groups])

    ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
    ss_within = np.sum([np.sum((g - np.mean(g)) ** 2) for g in groups])

    k = len(groups)
    n = len(all_data)
    df_between = k - 1
    df_within = n - k

    mean_square_between = ss_between / df_between
    mean_square_within = ss_within / df_within

    F = mean_square_between / mean_square_within

    return F, p_value(
        F, stats.f(dfn=df_between, dfd=df_within), alternative="greater"
    )




def bonferroni_pairwise(groups):
    """
    All pairwise two-sample t-tests with Bonferroni-adjusted p-values.

    Running k(k-1)/2 pairwise tests at level alpha inflates the probability of
    at least one false rejection well beyond alpha. Bonferroni counteracts
    this by multiplying each p-value by the number of comparisons (capped at
    1), which is equivalent to testing each pair at level alpha/m.

    The correction is universally valid but conservative: it loses power,
    increasingly so as the number of groups grows. Tukey's HSD is tailored to
    exactly this situation and is less conservative.

    Parameters
    ----------
    groups : sequence of array_like
        Observations of each group. Groups may have different sizes.

    Returns
    -------
    list of PairwiseComparison
        One entry per pair, with the t statistic and the adjusted p-value.

    Raises
    ------
    ValueError
        If fewer than two groups are given.
    """
    groups = [np.asarray(g) for g in groups]

    if len(groups) < 2:
        raise ValueError(f"Need at least two groups, got {len(groups)}.")

    comparisons = []
    for i, j in combinations(range(len(groups)), 2):
        gi, gj = groups[i], groups[j]
        t, p = two_sample_t_test(
            mean1=np.mean(gi), s1=np.std(gi, ddof=1), n1=len(gi),
            mean2=np.mean(gj), s2=np.std(gj, ddof=1), n2=len(gj),
            alternative="two_sided",
        )
        comparisons.append(PairwiseComparison(i, j, t, p))

    n_comparisons = len(comparisons)
    return [
        c._replace(p_value=min(c.p_value * n_comparisons, 1.0))
        for c in comparisons
    ]


def tukey_hsd(groups):
    """
    Tukey's honestly significant difference test for all pairwise comparisons.

    Uses the studentized range distribution, which accounts for the fact that
    the largest of k group means is being compared against the smallest. This
    makes it less conservative than Bonferroni while still controlling the
    family-wise error rate, but it is valid only for this specific setting
    (all pairwise comparisons after a one-way ANOVA).

    Uses the pooled within-groups mean square from the ANOVA as the common
    variance estimate, and the Tukey-Kramer form of the standard error, which
    also handles unequal group sizes.

    Parameters
    ----------
    groups : sequence of array_like
        Observations of each group. Groups may have different sizes.

    Returns
    -------
    list of PairwiseComparison
        One entry per pair, with the studentized range statistic q and its
        p-value.

    Raises
    ------
    ValueError
        If fewer than two groups are given.
    """
    groups = [np.asarray(g) for g in groups]

    k = len(groups)
    if k < 2:
        raise ValueError(f"Need at least two groups, got {k}.")

    n = sum(len(g) for g in groups)
    df_within = n - k

    # pooled within-groups variance estimate, identical to the ANOVA's MS_within
    ss_within = np.sum([np.sum((g - np.mean(g)) ** 2) for g in groups])
    mean_square_within = ss_within / df_within

    comparisons = []
    for i, j in combinations(range(k), 2):
        gi, gj = groups[i], groups[j]

        difference = np.mean(gi) - np.mean(gj)
        standard_error = np.sqrt(
            mean_square_within / 2 * (1 / len(gi) + 1 / len(gj))
        )
        q = abs(difference) / standard_error

        p = stats.studentized_range.sf(q, k, df_within)
        comparisons.append(PairwiseComparison(i, j, q, p))

    return comparisons
    