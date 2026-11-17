"""Monte Carlo simulation lab.

Where the other modules implement statistical procedures, this module puts
them to the test empirically: each function repeats an experiment many times
and reports what actually happens, so that theoretical claims (consistency,
efficiency, coverage, power, robustness) can be seen rather than assumed.

Every function returns a named tuple carrying the raw per-replication results
alongside summary numbers, so that plots can show the full distribution rather
than only its summary. Where a closed-form answer exists, it is returned too,
ready to be overlaid on the simulated curve.
"""

from collections import namedtuple

import numpy as np
from scipy import stats


# Result containers

SamplingDistributionResult = namedtuple(
    "SamplingDistributionResult",
    ["estimates", "sample_sizes", "true_value", "means", "standard_deviations"],
)

CoverageResult = namedtuple(
    "CoverageResult",
    ["coverage", "covers", "lower", "upper", "true_value", "nominal_level"],
)

PowerCurveResult = namedtuple(
    "PowerCurveResult",
    ["effect_sizes", "power", "theoretical_power", "alpha", "sample_size"],
)

ErrorRateResult = namedtuple(
    "ErrorRateResult",
    ["rejection_rate", "p_values", "nominal_alpha", "label"],
)

ConvergenceResult = namedtuple(
    "ConvergenceResult",
    ["sample_sizes", "paths", "true_value"],
)


# 1. Sampling distribution of an estimator

def sampling_distribution(sampler, estimator, true_value, sample_sizes,
                          n_simulations, rng):
    """
    Simulate the sampling distribution of an estimator across sample sizes.

    Draws n_simulations samples of each requested size, applies the estimator
    to every one of them, and collects all resulting estimates. Shows
    consistency (the spread shrinks as n grows) and asymptotic normality (the
    shape approaches a bell curve) directly.

    Parameters
    ----------
    sampler : callable
        Function (rng, size) -> np.ndarray drawing one sample of the given size.
    estimator : callable
        Function mapping a sample to a scalar estimate, e.g. np.mean.
    true_value : float
        The parameter value the sampler was built with.
    sample_sizes : sequence of int
        Sample sizes to simulate.
    n_simulations : int
        Number of replications per sample size.
    rng : np.random.Generator
        Random generator, for reproducibility.

    Returns
    -------
    SamplingDistributionResult
        estimates is a dict mapping each sample size to its array of estimates;
        means and standard_deviations summarise them per sample size.
    """
    estimates = {}
    means = np.empty(len(sample_sizes))
    standard_deviations = np.empty(len(sample_sizes))

    for i, n in enumerate(sample_sizes):
        values = np.array(
            [estimator(sampler(rng, n)) for _ in range(n_simulations)]
        )
        estimates[n] = values
        means[i] = values.mean()
        standard_deviations[i] = values.std(ddof=1)

    return SamplingDistributionResult(
        estimates=estimates,
        sample_sizes=np.asarray(sample_sizes),
        true_value=true_value,
        means=means,
        standard_deviations=standard_deviations,
    )


def cramer_rao_comparison(sample_sizes, true_lambda, n_simulations, rng):
    """
    Compare the variance of the Poisson ML estimator to the Cramer-Rao bound.

    For the Poisson distribution the Fisher information is 1/lambda, so no
    unbiased estimator can have variance below lambda/n. The ML estimator is
    the sample mean, which attains this bound exactly - the simulation makes
    that visible.

    Parameters
    ----------
    sample_sizes : sequence of int
        Sample sizes to simulate.
    true_lambda : float
        True Poisson parameter.
    n_simulations : int
        Number of replications per sample size.
    rng : np.random.Generator
        Random generator, for reproducibility.

    Returns
    -------
    dict
        Keys "sample_sizes", "empirical_variance" and "cramer_rao_bound",
        each an array aligned with sample_sizes.
    """
    sample_sizes = np.asarray(sample_sizes)
    empirical_variance = np.empty(len(sample_sizes), dtype=float)

    for i, n in enumerate(sample_sizes):
        samples = rng.poisson(true_lambda, size=(n_simulations, n))
        empirical_variance[i] = samples.mean(axis=1).var(ddof=1)

    return {
        "sample_sizes": sample_sizes,
        "empirical_variance": empirical_variance,
        "cramer_rao_bound": true_lambda / sample_sizes,
    }


# 2. Coverage of confidence intervals

def coverage_probability(sampler, interval_fn, true_value, n, n_simulations,
                         rng, confidence_level=0.95):
    """
    Estimate how often a confidence interval actually covers the true value.

    A 95% interval claims to contain the parameter in 95% of repeated
    experiments. This function runs those repeated experiments and counts.

    Parameters
    ----------
    sampler : callable
        Function (rng, size) -> np.ndarray drawing one sample.
    interval_fn : callable
        Function (sample, confidence_level) -> (lower, upper).
    true_value : float
        The parameter the interval is meant to cover.
    n : int
        Sample size per replication.
    n_simulations : int
        Number of replications.
    rng : np.random.Generator
        Random generator, for reproducibility.
    confidence_level : float, optional
        Nominal confidence level. Default 0.95.

    Returns
    -------
    CoverageResult
        Empirical coverage plus the individual intervals, so that a plot can
        show which replications missed.
    """
    lower = np.empty(n_simulations)
    upper = np.empty(n_simulations)

    for i in range(n_simulations):
        sample = sampler(rng, n)
        lower[i], upper[i] = interval_fn(sample, confidence_level)

    covers = (lower <= true_value) & (true_value <= upper)

    return CoverageResult(
        coverage=float(covers.mean()),
        covers=covers,
        lower=lower,
        upper=upper,
        true_value=true_value,
        nominal_level=confidence_level,
    )


def coverage_across_sample_sizes(sampler, interval_fn, true_value,
                                 sample_sizes, n_simulations, rng,
                                 confidence_level=0.95):
    """
    Run coverage_probability for several sample sizes.

    Useful for showing that an asymptotic interval only reaches its nominal
    level once n is large enough.

    Returns
    -------
    dict
        Keys "sample_sizes" and "coverage", aligned arrays ready to plot.
    """
    coverages = np.array([
        coverage_probability(
            sampler, interval_fn, true_value, n, n_simulations, rng,
            confidence_level,
        ).coverage
        for n in sample_sizes
    ])

    return {"sample_sizes": np.asarray(sample_sizes), "coverage": coverages}


# 3. Power of a test

def power_curve(effect_sizes, n, n_simulations, rng, alpha=0.05,
                include_theory=True):
    """
    Simulate the power of the one-sample t-test against H0: mu = 0.

    For each effect size (the true mean, in units of the standard deviation),
    counts how often the test rejects. At effect size zero this is the type I
    error rate, which should sit at alpha; as the effect grows, power rises
    towards one.

    Parameters
    ----------
    effect_sizes : sequence of float
        True means to simulate, with unit standard deviation.
    n : int
        Sample size per replication.
    n_simulations : int
        Number of replications per effect size.
    rng : np.random.Generator
        Random generator, for reproducibility.
    alpha : float, optional
        Significance level. Default 0.05.
    include_theory : bool, optional
        Also return the exact power from the non-central t distribution,
        for overlaying on the simulated curve. Default True.

    Returns
    -------
    PowerCurveResult
        Simulated power per effect size, and the theoretical curve.
    """
    effect_sizes = np.asarray(effect_sizes, dtype=float)
    power = np.empty(len(effect_sizes))

    null_distribution = stats.t(df=n - 1)
    critical_value = null_distribution.ppf(1 - alpha / 2)

    for i, effect in enumerate(effect_sizes):
        # one row per replication - vectorised over all simulations at once
        samples = rng.normal(effect, 1.0, size=(n_simulations, n))
        t_statistics = samples.mean(axis=1) / (
            samples.std(axis=1, ddof=1) / np.sqrt(n)
        )
        power[i] = np.mean(np.abs(t_statistics) > critical_value)

    theoretical_power = None
    if include_theory:
        noncentrality = effect_sizes * np.sqrt(n)
        theoretical_power = (
            stats.nct.sf(critical_value, n - 1, noncentrality)
            + stats.nct.cdf(-critical_value, n - 1, noncentrality)
        )

    return PowerCurveResult(
        effect_sizes=effect_sizes,
        power=power,
        theoretical_power=theoretical_power,
        alpha=alpha,
        sample_size=n,
    )


def power_across_sample_sizes(effect_size, sample_sizes, n_simulations, rng,
                              alpha=0.05):
    """
    Power of the one-sample t-test at a fixed effect size, as n grows.

    Answers the practical question "how many observations do I need?".

    Returns
    -------
    dict
        Keys "sample_sizes" and "power", aligned arrays ready to plot.
    """
    power = np.empty(len(sample_sizes))

    for i, n in enumerate(sample_sizes):
        samples = rng.normal(effect_size, 1.0, size=(n_simulations, n))
        t_statistics = samples.mean(axis=1) / (
            samples.std(axis=1, ddof=1) / np.sqrt(n)
        )
        critical_value = stats.t(df=n - 1).ppf(1 - alpha / 2)
        power[i] = np.mean(np.abs(t_statistics) > critical_value)

    return {"sample_sizes": np.asarray(sample_sizes), "power": power}


# 4. Robustness: does a test hold its nominal level?

def type_one_error_rate(sampler, p_value_fn, n, n_simulations, rng,
                        alpha=0.05, label=""):
    """
    Estimate a test's actual type I error rate when H0 is true.

    Draws samples for which the null hypothesis genuinely holds and counts how
    often the test rejects anyway. A well-behaved test rejects at rate alpha;
    a test whose assumptions are violated may reject far more often.

    Parameters
    ----------
    sampler : callable
        Function (rng, size) -> np.ndarray drawing a sample under H0.
    p_value_fn : callable
        Function mapping a sample to a p-value.
    n : int
        Sample size per replication.
    n_simulations : int
        Number of replications.
    rng : np.random.Generator
        Random generator, for reproducibility.
    alpha : float, optional
        Nominal significance level. Default 0.05.
    label : str, optional
        Description carried through to the result, useful for plot legends.

    Returns
    -------
    ErrorRateResult
        The observed rejection rate and every individual p-value, so that a
        plot can show the whole p-value distribution (uniform if the test is
        well calibrated).
    """
    p_values = np.array(
        [p_value_fn(sampler(rng, n)) for _ in range(n_simulations)]
    )

    return ErrorRateResult(
        rejection_rate=float(np.mean(p_values < alpha)),
        p_values=p_values,
        nominal_alpha=alpha,
        label=label,
    )


def robustness_comparison(scenarios, n, n_simulations, rng, alpha=0.05):
    """
    Compare the type I error rates of several test/distribution combinations.

    Built for the classic demonstration that the t-test keeps its level under
    non-normality (thanks to the central limit theorem) while the chi-squared
    variance test does not.

    Parameters
    ----------
    scenarios : dict
        Maps a label to a (sampler, p_value_fn) pair.
    n : int
        Sample size per replication.
    n_simulations : int
        Number of replications per scenario.
    rng : np.random.Generator
        Random generator, for reproducibility.
    alpha : float, optional
        Nominal significance level. Default 0.05.

    Returns
    -------
    list of ErrorRateResult
        One entry per scenario, in the order given.
    """
    return [
        type_one_error_rate(
            sampler, p_value_fn, n, n_simulations, rng, alpha, label
        )
        for label, (sampler, p_value_fn) in scenarios.items()
    ]


# 5. Limit theorems

def central_limit_demo(sampler, sample_sizes, n_simulations, rng):
    """
    Standardised sample means for growing n, from an arbitrary distribution.

    Whatever the shape of the underlying distribution, the standardised mean
    approaches a standard normal as n grows. Plotting a histogram per sample
    size against the standard normal density shows this happening.

    Parameters
    ----------
    sampler : callable
        Function (rng, size) -> np.ndarray drawing one sample.
    sample_sizes : sequence of int
        Sample sizes to simulate.
    n_simulations : int
        Number of replications per sample size.
    rng : np.random.Generator
        Random generator, for reproducibility.

    Returns
    -------
    dict
        Maps each sample size to its array of standardised sample means.
    """
    standardised = {}

    for n in sample_sizes:
        means = np.array(
            [sampler(rng, n).mean() for _ in range(n_simulations)]
        )
        standardised[n] = (means - means.mean()) / means.std(ddof=1)

    return standardised


def law_of_large_numbers_paths(sampler, true_value, max_n, n_paths, rng):
    """
    Running averages of several independent sequences.

    Each path is the cumulative mean of one long sample. Plotted together they
    show the paths funnelling in on the true value - the law of large numbers
    as a picture rather than a statement.

    Parameters
    ----------
    sampler : callable
        Function (rng, size) -> np.ndarray drawing one sample.
    true_value : float
        The expectation the averages converge to.
    max_n : int
        Length of each path.
    n_paths : int
        Number of independent paths.
    rng : np.random.Generator
        Random generator, for reproducibility.

    Returns
    -------
    ConvergenceResult
        paths has shape (n_paths, max_n); entry [i, k] is the mean of the
        first k+1 observations of path i.
    """
    paths = np.empty((n_paths, max_n))

    for i in range(n_paths):
        sample = sampler(rng, max_n)
        paths[i] = np.cumsum(sample) / np.arange(1, max_n + 1)

    return ConvergenceResult(
        sample_sizes=np.arange(1, max_n + 1),
        paths=paths,
        true_value=true_value,
    )


# 6. Bootstrap versus classical intervals

def bootstrap_vs_classical_coverage(sampler, true_value, sample_sizes,
                                    n_simulations, rng, bootstrap_ci_fn,
                                    classical_ci_fn, confidence_level=0.95):
    """
    Compare the coverage of a bootstrap and a classical interval side by side.

    On skewed data with small samples the classical interval, which assumes
    normality, can fall short of its nominal level while the bootstrap holds
    up better. On large samples the two converge.

    Parameters
    ----------
    sampler : callable
        Function (rng, size) -> np.ndarray drawing one sample.
    true_value : float
        The parameter both intervals target.
    sample_sizes : sequence of int
        Sample sizes to compare.
    n_simulations : int
        Number of replications per sample size.
    rng : np.random.Generator
        Random generator, for reproducibility.
    bootstrap_ci_fn, classical_ci_fn : callable
        Functions (sample, confidence_level) -> (lower, upper).
    confidence_level : float, optional
        Nominal confidence level. Default 0.95.

    Returns
    -------
    dict
        Keys "sample_sizes", "bootstrap_coverage" and "classical_coverage".
    """
    bootstrap_coverage = np.empty(len(sample_sizes))
    classical_coverage = np.empty(len(sample_sizes))

    for i, n in enumerate(sample_sizes):
        bootstrap_hits = 0
        classical_hits = 0

        for _ in range(n_simulations):
            sample = sampler(rng, n)

            lower, upper = bootstrap_ci_fn(sample, confidence_level)
            bootstrap_hits += lower <= true_value <= upper

            lower, upper = classical_ci_fn(sample, confidence_level)
            classical_hits += lower <= true_value <= upper

        bootstrap_coverage[i] = bootstrap_hits / n_simulations
        classical_coverage[i] = classical_hits / n_simulations

    return {
        "sample_sizes": np.asarray(sample_sizes),
        "bootstrap_coverage": bootstrap_coverage,
        "classical_coverage": classical_coverage,
    }