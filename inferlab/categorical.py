import numpy as np
from scipy import stats

from inferlab.hypothesis_tests import p_value


def chi_squared_independence_test(table):
    """
    Chi-squared test of independence for a k x m contingency table.

    Under H0 (row and column variables are independent), expected cell
    frequencies follow from the marginal totals. The Pearson statistic
    compares observed against expected, asymptotically chi-squared
    distributed with (k-1)(m-1) degrees of freedom.

    Parameters
    ----------
    table : array_like
        k x m matrix of observed frequencies.

    Returns
    -------
    tuple of float
        (test statistic X^2, p-value).
    """
    table = np.asarray(table)
    n = table.sum()
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    expected = np.outer(row_sums, col_sums) / n

    statistic = np.sum((table - expected) ** 2 / expected)
    df = (table.shape[0] - 1) * (table.shape[1] - 1)

    return statistic, p_value(statistic, stats.chi2(df=df), alternative="greater")


def fisher_exact_test(table, alternative="two_sided"):
    """
    Fisher's exact test for a 2x2 contingency table.

    Conditions on both sets of marginal totals. Under H0 (independence),
    the top-left cell then follows a hypergeometric distribution, so the
    p-value can be computed exactly instead of relying on the chi-squared
    approximation. Preferable when expected cell frequencies are small.

    Parameters
    ----------
    table : array_like
        2x2 matrix of observed frequencies.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    float
        The exact p-value.

    Raises
    ------
    ValueError
        If alternative is not one of the three accepted values.
    """
    table = np.asarray(table)
    observed = table[0, 0]
    n = table.sum()
    row1_total = table[0, :].sum()
    col1_total = table[:, 0].sum()

    # with both margins fixed, the top-left cell is hypergeometric under H0
    null_distribution = stats.hypergeom(M=n, n=col1_total, N=row1_total)

    if alternative == "greater":
        # P(X >= observed): sf(k) is P(X > k), so shift by one
        return null_distribution.sf(observed - 1)

    elif alternative == "less":
        return null_distribution.cdf(observed)

    elif alternative == "two_sided":
        # every table whose probability is no greater than the observed
        # table's counts as "at least as extreme"
        support = np.arange(max(0, row1_total + col1_total - n), min(row1_total, col1_total) + 1)
        probabilities = null_distribution.pmf(support)
        observed_probability = null_distribution.pmf(observed)

        tolerance = 1 + 1e-7  # guards against floating-point ties
        extreme = probabilities <= observed_probability * tolerance
        return min(probabilities[extreme].sum(), 1.0)

    else:
        raise ValueError(
            f"Unknown alternative '{alternative}'. "
            "Expected 'two_sided', 'greater' or 'less'."
        )