import numpy as np
from scipy import stats

from inferlab.hypothesis_tests import p_value
from inferlab.datasets import load_rutherford


def index_of_dispersion_test(mean, s, n):
    """
    Dispersion index test for a Poisson assumption.

    Under H0, the data follow a Poisson distribution, for which mean and
    variance are equal (index of dispersion = 1). The statistic
    D = (n-1)*s^2/mean is then (asymptotically) chi-squared distributed with n-1 degrees
    of freedom. Both very small and very large values of D speak against
    the Poisson assumption.

    Parameters
    ----------
    mean : float
        Sample mean.
    s : float
        Sample standard deviation (ddof=1).
    n : int
        Sample size.

    Returns
    -------
    tuple of float
        (test statistic D, p-value).
    """
    d = ((n - 1) * s**2) / mean

    return d, p_value(d, stats.chi2(df=n - 1), "two_sided")


def sign_test(data, m0, alternative="two_sided"):
    """
    Sign test for the median.

    Counts observations above vs. below m0 and tests them as a binomial
    proportion under H0: P(X > m0) = 0.5. Observations exactly equal to m0
    carry no sign and are excluded from both the count and the effective
    sample size.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    m0 : float
        Hypothesized median.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    tuple of (int, float)
        (number of observations above m0, p-value).

    Raises
    ------
    ValueError
        If alternative is not one of the three accepted values.
    """
    data = np.asarray(data)
    n_above = int(np.sum(data > m0))
    n_below = int(np.sum(data < m0))
    n_effective = n_above + n_below

    null_distribution = stats.binom(n=n_effective, p=0.5)

    if alternative == "greater":
        p = null_distribution.sf(n_above - 1)
    elif alternative == "less":
        p = null_distribution.cdf(n_above)
    elif alternative == "two_sided":
        p_greater = null_distribution.sf(n_above - 1)
        p_less = null_distribution.cdf(n_above)
        p = min(2 * min(p_greater, p_less), 1.0)
    else:
        raise ValueError(
            f"Unknown alternative '{alternative}'. "
            "Expected 'two_sided', 'greater' or 'less'."
        )

    return n_above, p

def chi_squared_goodness_of_fit(observed, expected_probabilities, n_estimated_params):
    """
    Pearson's chi-squared goodness-of-fit test.

    Compares observed class frequencies against those expected under H0.
    Covers both the simple case (H0 fully specifies the probabilities,
    n_estimated_params=0, df = s-1) and the composite case (probabilities
    depend on parameters estimated from the same data, df = s-k-1, where
    k = n_estimated_params).


    Parameters
    ----------
    observed : array_like
        Observed frequency in each class. Must sum to the total sample size.
    expected_probabilities : array_like
        Probability of each class under H0, same length and order as
        observed. Must sum to 1.
    n_estimated_params : int
        Number of parameters estimated from the data to compute
        expected_probabilities (0 if all probabilities were fixed in advance).

    Returns
    -------
    tuple of float
        (test statistic X^2, p-value).
    """
    observed = np.asarray(observed)
    expected_probabilities = np.asarray(expected_probabilities)

    n = observed.sum()
    expected = n * expected_probabilities

    statistic = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - n_estimated_params - 1

    return statistic, p_value(statistic, stats.chi2(df=df), alternative="greater")

def wilcoxon_rank_sum_test(x, y, alternative="two_sided", continuity_correction=True):
    """
    Wilcoxon rank-sum test (equivalent to Mann-Whitney U) for two independent
    samples, using the normal approximation.

    Pools both samples, ranks them jointly (ties get the average rank), and
    compares the rank sum of the first sample against its expectation under
    H0 (both samples drawn from the same distribution). Requires no
    distributional assumption on the raw data.

    Parameters
    ----------
    x, y : array_like
        Observations of the two independent groups.
    alternative : str
        One of "two_sided", "greater", "less".
    continuity_correction : bool, optional
        Whether to apply the continuity correction when approximating the
        discrete rank sum by a normal distribution. Default True, matching
        scipy.stats.mannwhitneyu's asymptotic method.

    Returns
    -------
    tuple of float
        (z statistic, p-value).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m, n = len(x), len(y)

    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    W = np.sum(ranks[:m])

    expected_W = m * (m + n + 1) / 2

    # tie correction: variance shrinks when observations are tied
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_correction = np.sum(tie_counts**3 - tie_counts) / ((m + n) * (m + n - 1))
    variance_W = (m * n / 12) * ((m + n + 1) - tie_correction)

    diff = W - expected_W
    cc = 0.5 if continuity_correction else 0.0

    if alternative == "greater":
        z = (diff - cc) / np.sqrt(variance_W)
        p = stats.norm.sf(z)
    elif alternative == "less":
        z = (diff + cc) / np.sqrt(variance_W)
        p = stats.norm.cdf(z)
    elif alternative == "two_sided":
        z = (abs(diff) - cc) / np.sqrt(variance_W)
        p = min(2 * stats.norm.sf(z), 1.0)
        z = np.sign(diff) * z
    else:
        raise ValueError(
            f"Unknown alternative '{alternative}'. "
            "Expected 'two_sided', 'greater' or 'less'."
        )

    return z, p