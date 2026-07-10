import scipy
import numpy as np

def ci_mean_known_sigma(mean, sigma, n, confidence_level):
    """
    Confidence interval for the mean of a normal distribution when the
    standard deviation is known.

    Parameters
    ----------
    mean : float
        Sample mean.
    sigma : float
        Known standard deviation.
    n : int
        Sample size.
    confidence_level : float
        Desired confidence level, e.g. 0.95.

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the confidence interval.
    """
    alpha = 1-confidence_level
    z = scipy.stats.norm.ppf(1-alpha/2)
    lower = mean - (sigma * z / np.sqrt(n))
    upper = mean + (sigma * z / np.sqrt(n))

    return (lower, upper)


def ci_mean_unknown_sigma(mean, s, n, confidence_level):
    """
    Confidence interval for the mean of a normal distribution when the
    standard deviation is unknown and estimated from the data.

    Uses the t-distribution with n-1 degrees of freedom (Student's theorem).

    Parameters
    ----------
    mean : float
        Sample mean.
    s : float
        Sample standard deviation (with ddof=1).
    n : int
        Sample size.
    confidence_level : float
        Desired confidence level, e.g. 0.95.

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the confidence interval.
    """
    alpha = 1 - confidence_level
    t = scipy.stats.t.ppf(1 - alpha / 2, df=n - 1)
    lower = mean - (s * t / np.sqrt(n))
    upper = mean + (s * t / np.sqrt(n))

    return (lower, upper)


def ci_variance(s, n, confidence_level):
    """
    Confidence interval for the variance of a normal distribution.

    Uses the chi-squared distribution with n-1 degrees of freedom. The
    interval is asymmetric, unlike the mean intervals.

    Parameters
    ----------
    s : float
        Sample standard deviation (with ddof=1).
    n : int
        Sample size.
    confidence_level : float
        Desired confidence level, e.g. 0.95.

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the confidence interval for the variance.
    """
    alpha = 1 - confidence_level
    chi2_lower = scipy.stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_upper = scipy.stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    lower = (n - 1) * s**2 / chi2_upper
    upper = (n - 1) * s**2 / chi2_lower

    return (lower, upper)

def bootstrap_percentile_ci(data, statistic, B, confidence_level, rng):
    """
    Bootstrap percentile confidence interval for an arbitrary statistic.

    Resamples the data with replacement B times, evaluates the statistic on
    each resample to approximate its sampling distribution, and reads off the
    empirical quantiles as interval bounds. Makes no distributional
    assumption about the data.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    statistic : callable
        Function mapping a sample to a scalar, e.g. np.mean or np.median.
    B : int
        Number of bootstrap resamples.
    confidence_level : float
        Desired confidence level, e.g. 0.95.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the confidence interval.
    """
    simulated_distribution = []
    for _ in range(B):
        sample = rng.choice(data, size=len(data), replace=True)
        simulated_distribution.append(statistic(sample))
    
    alpha = 1-confidence_level
    lower = np.percentile(simulated_distribution, 100*alpha/2)
    upper = np.percentile(simulated_distribution, 100*(1-alpha/2))

    return (lower, upper)

def bootstrap_t_ci_mean(data, B, confidence_level, rng):
    """
    Bootstrap-t confidence interval for the mean.

    Studentizes each bootstrap replicate using the closed-form standard
    error of the mean (S / sqrt(n)). This matches the lecture's worked
    example and is valid specifically for the mean, not a general statistic.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    B : int
        Number of bootstrap resamples.
    confidence_level : float
        Desired confidence level, e.g. 0.95.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the confidence interval.
    """
    n = len(data)
    theta_hat = np.mean(data)

    t_stars = []
    for _ in range(B):
        sample = rng.choice(data, size=n, replace=True)
        theta_star = np.mean(sample)
        se_star = np.std(sample, ddof=1) / np.sqrt(n)
        t_stars.append((theta_star - theta_hat) / se_star)

    alpha = 1 - confidence_level
    t_lower = np.percentile(t_stars, 100 * (alpha / 2))
    t_upper = np.percentile(t_stars, 100 * (1 - alpha / 2))

    se_hat = np.std(data, ddof=1) / np.sqrt(n)
    lower = theta_hat - t_upper * se_hat
    upper = theta_hat - t_lower * se_hat

    return (lower, upper)        

    
    