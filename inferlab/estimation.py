from scipy import stats
from scipy.optimize import minimize

import numpy as np



def ml_estimate(data, distribution_assumption):
    """
    Generic maximum-likelihood estimator via numerical optimization.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    distribution_assumption : str
        Key into the distribution lookup table, e.g. "poisson" or "normal".

    Returns
    -------
    np.ndarray
        Estimated parameter(s), in the order the distribution expects them.
    """
    lookup_table = {
        "poisson":     (stats.poisson.logpmf, [1], [(1e-9, None)]),
        "normal":      (stats.norm.logpdf,    [0, 1], [(None, None), (1e-9, None)]),
        "exponential": (lambda data, rate: stats.expon.logpdf(data, scale=1/rate), [1], [(1e-9, None)]),
        "geometric":   (lambda data, p: stats.geom.logpmf(data, p, loc=-1),    [0.5], [(1e-9, 1 - 1e-9)]),
        "binomial":    (lambda data, p, n: stats.binom.logpmf(data, n, p), [0.5], [(1e-9, 1 - 1e-9)]),
    }

    if distribution_assumption not in lookup_table:
        raise ValueError(
            f"Unknown distribution '{distribution_assumption}'. "
            f"Available: {list(lookup_table.keys())}"
        )
    logdensity_fn, x0, bounds = lookup_table[distribution_assumption]

    def neg_log_likelihood(params):
        return -np.sum(logdensity_fn(data, *params))

    result = minimize(fun=neg_log_likelihood, x0=x0, bounds=bounds, method="L-BFGS-B")
    return result.x


def moment_estimate(data, distribution_assumption):
    """
    Method-of-moments estimator.

    Equates theoretical moments with empirical moments and solves for the
    parameter(s) in closed form. For Poisson, exponential and normal
    distributions this yields the same result as maximum likelihood.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    distribution_assumption : str
        Key into the moment-estimator lookup table.

    Returns
    -------
    np.ndarray
        Estimated parameter(s), in the order the distribution expects them.
    """
    moment_estimators = {
        "poisson":     lambda d: np.array([np.mean(d)]),
        "exponential": lambda d: np.array([1 / np.mean(d)]),
        "geometric":   lambda d: np.array([1 / (1 + np.mean(d))]),
        "normal":      lambda d: np.array([np.mean(d), np.sqrt(np.var(d, ddof=0))]),
    }
     
    if distribution_assumption not in moment_estimators:
        raise ValueError(
            f"Unknown distribution '{distribution_assumption}'. "
            f"Available: {list(moment_estimators.keys())}"
        )

    return moment_estimators[distribution_assumption](data)


def bayes_binomial(x, n, a, b):
    """
    Bayes estimator for the success probability of a binomial model
    under a conjugate Beta prior.

    With prior theta ~ Beta(a, b) and x successes in n trials, the
    posterior is Beta(x + a, n - x + b). The estimator is its mean.

    Parameters
    ----------
    x : int
        Number of observed successes.
    n : int
        Number of trials.
    a, b : float
        Shape parameters of the Beta prior (a, b > 0).

    Returns
    -------
    float
        Posterior mean estimate of the success probability.
    """
    return (x + a) / (n + a + b)

def bayes_normal(data, sigma, nu, tau):
    """
    Bayes estimator for the mean of a normal model with known variance
    under a conjugate normal prior.

    With prior mu ~ N(nu, tau**2) and data ~ N(mu, sigma**2), the
    posterior mean is a precision-weighted average of the prior mean nu
    and the sample mean.

    Parameters
    ----------
    data : np.ndarray
        Observed sample from N(mu, sigma**2).
    sigma : float
        Known standard deviation of the data.
    nu : float
        Mean of the normal prior.
    tau : float
        Standard deviation of the normal prior.

    Returns
    -------
    float
        Posterior mean estimate of mu.
    """
    n = len(data)
    x_bar = np.mean(data)

    data_weight = tau**2 / (tau**2 + sigma**2 / n)
    prior_weight = (sigma**2 / n) / (tau**2 + sigma**2 / n)

    return data_weight * x_bar + prior_weight * nu



