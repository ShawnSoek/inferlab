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