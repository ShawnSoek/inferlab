import numpy as np
from scipy import stats
from inferlab.estimation import ml_estimate


def p_value(statistic, null_distribution, alternative):
    """
    Compute the p-value of an observed test statistic under a null distribution.

    Parameters
    ----------
    statistic : float
        Observed value of the test statistic.
    null_distribution : scipy.stats frozen distribution
        Distribution of the statistic under H0, e.g. stats.norm or
        stats.t(df=n-1). Must provide .cdf() and .sf().
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    float
        The p-value.

    Raises
    ------
    ValueError
        If alternative is not one of the three accepted values.
    """
    if alternative == "two_sided":
        # 2 * smaller tail
        p = 2 * min(null_distribution.sf(statistic), null_distribution.cdf(statistic))
        return min(p, 1.0)
    elif alternative == "greater":
        return null_distribution.sf(statistic)
    elif alternative == "less":
        return null_distribution.cdf(statistic)
    else:
        raise ValueError(
            f"Unknown alternative '{alternative}'. "
            "Expected 'two_sided', 'greater' or 'less'."
        )


def gauss_test(mean, mu0, sigma, n, alternative="two_sided"):
    """
    One-sample Gauss test for the mean of a normal distribution with known sigma.

    Returns
    -------
    tuple of float
        (test statistic, p-value).
    """
    z = (mean - mu0) / (sigma / np.sqrt(n))
    return z, p_value(z, stats.norm, alternative)


def t_test(mean, mu0, s, n, alternative="two_sided"):
    """
    One-sample t-test for the mean of a normal distribution with unknown sigma.

    Returns
    -------
    tuple of float
        (test statistic, p-value).
    """
    t = (mean - mu0) / (s / np.sqrt(n))
    return t, p_value(t, stats.t(df=n - 1), alternative)


def variance_test(s, sigma0, n, alternative="two_sided"):
    """
    Chi-squared test for the variance of a normal distribution.

    Note the null distribution is skewed, so the two-sided p-value is computed
    as twice the smaller tail rather than by doubling a symmetric tail.

    Returns
    -------
    tuple of float
        (test statistic, p-value).
    """
    chi2_statistic = (n - 1) * s**2 / sigma0**2
    return chi2_statistic, p_value(chi2_statistic, stats.chi2(df=n - 1), alternative)

def log_likelihood(data, distribution_assumption, params):
    """
    Evaluate the log-likelihood of a sample at a given parameter value.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    distribution_assumption : str
        Key into the distribution lookup table, e.g. "poisson" or "normal".
    params : float or array_like
        Parameter value(s) at which to evaluate the log-likelihood, in the
        order the distribution expects them.

    Returns
    -------
    float
        The log-likelihood at the given parameter value.
    """
    lookup_table = {
        "poisson":     (stats.poisson.logpmf, [1], [(1e-9, None)]),
        "normal":      (stats.norm.logpdf,    [0, 1], [(None, None), (1e-9, None)]),
        "exponential": (lambda data, rate: stats.expon.logpdf(data, scale=1/rate), [1], [(1e-9, None)]),
        "geometric":   (lambda data, p: stats.geom.logpmf(data, p, loc=-1),    [0.5], [(1e-9, 1 - 1e-9)]),
        "binomial":    (lambda data, p, n: stats.binom.logpmf(data, n, p), [0.5], [(1e-9, 1 - 1e-9)]),
    }
    logdensity_fn, _, _ = lookup_table[distribution_assumption]
    return np.sum(logdensity_fn(data, *np.atleast_1d(params)))


def likelihood_ratio_test(data, distribution_assumption, params_h0):
    """
    Likelihood ratio test (Wilks) for a fully specified simple null hypothesis.

    Compares the maximum log-likelihood achievable under H0 (where all
    parameters are fixed) against the maximum over the unrestricted parameter
    space. The test statistic W = 2 * (l_full - l_h0) is asymptotically
    chi-squared distributed with degrees of freedom equal to the number of
    parameters constrained by H0 (Wilks' theorem).

    The test is inherently one-sided: only large values of W speak against H0,
    since the unrestricted model can never fit worse than the restricted one.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    distribution_assumption : str
        Key into the distribution lookup table.
    params_h0 : float or array_like
        Parameter value(s) fixed by the null hypothesis. Must specify every
        parameter of the distribution.

    Returns
    -------
    tuple of float
        (test statistic W, p-value).
    """
    params_h0 = np.atleast_1d(params_h0)

    ll_h0 = log_likelihood(data, distribution_assumption, params_h0)

    params_full = ml_estimate(data, distribution_assumption)
    ll_full = log_likelihood(data, distribution_assumption, params_full)

    w = 2 * (ll_full - ll_h0)
    df = len(params_h0)

    return w, p_value(w, stats.chi2(df=df), alternative="greater")

def two_sample_t_test(mean1, s1, n1, mean2, s2, n2, alternative="two_sided"):
    """
    Two-sample t-test for equal means, assuming equal (unknown) variances.

    Pools the two sample variances into a single estimate, weighted by their
    degrees of freedom. Requires the equal-variance assumption.

    Parameters
    ----------
    mean1, mean2 : float
        Sample means of the two groups.
    s1, s2 : float
        Sample standard deviations (ddof=1) of the two groups.
    n1, n2 : int
        Sample sizes of the two groups.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    tuple of float
        (test statistic, p-value).
    """
    pooled_variance = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    standard_error = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))

    t = (mean1 - mean2) / standard_error
    df = n1 + n2 - 2

    return t, p_value(t, stats.t(df=df), alternative)


def welch_test(mean1, s1, n1, mean2, s2, n2, alternative="two_sided"):
    """
    Welch's t-test for equal means without assuming equal variances.

    Does not pool the variances. The degrees of freedom are approximated by the
    Welch-Satterthwaite equation and are generally not an integer.

    Parameters
    ----------
    mean1, mean2 : float
        Sample means of the two groups.
    s1, s2 : float
        Sample standard deviations (ddof=1) of the two groups.
    n1, n2 : int
        Sample sizes of the two groups.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    tuple of float
        (test statistic, p-value).
    """
    variance1 = s1**2 / n1
    variance2 = s2**2 / n2
    standard_error = np.sqrt(variance1 + variance2)

    t = (mean1 - mean2) / standard_error

    # Welch-Satterthwaite approximation
    numerator = (variance1 + variance2) ** 2
    denominator = variance1**2 / (n1 - 1) + variance2**2 / (n2 - 1)
    df = numerator / denominator

    return t, p_value(t, stats.t(df=df), alternative)


def f_test(s1, n1, s2, n2, alternative="two_sided"):
    """
    F-test for the equality of two variances.

    The test statistic is the ratio of the two sample variances. 
    Can be used to check whether the equal-variance assumption
    is justified.

    Parameters
    ----------
    s1, s2 : float
        Sample standard deviations (ddof=1) of the two groups.
    n1, n2 : int
        Sample sizes of the two groups.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    tuple of float
        (test statistic, p-value).
    """
    f = s1**2 / s2**2
    return f, p_value(f, stats.f(dfn=n1 - 1, dfd=n2 - 1), alternative)


def paired_t_test(x, y, alternative="two_sided"):
    """
    Paired t-test for dependent samples.

    Reduces to a one-sample t-test on the pairwise differences against mu0 = 0.

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations. Must have the same length.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    tuple of float
        (test statistic, p-value).

    Raises
    ------
    ValueError
        If x and y have different lengths.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(
            f"Paired samples must have equal length, got {len(x)} and {len(y)}."
        )

    differences = x - y

    return t_test(mean=np.mean(differences), mu0=0,
        s=np.std(differences, ddof=1),
        n=len(differences),
        alternative=alternative,
    )