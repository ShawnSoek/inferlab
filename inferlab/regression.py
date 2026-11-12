from collections import namedtuple

import numpy as np
from scipy import stats

from inferlab.hypothesis_tests import p_value


RegressionResult = namedtuple(
    "RegressionResult",
    ["coefficients", "fitted", "residuals", "sigma_squared",
     "design_matrix", "n_observations", "n_parameters"],
)


def linear_regression(X, y, add_intercept=True):
    """
    Ordinary least squares estimation for the linear model y = X*beta + eps.

    Solves the normal equations (X'X)beta = X'y. Works for simple regression
    (one predictor) and multiple regression alike.
    Parameters
    ----------
    X : array_like
        Predictor(s). Either a 1-D array for a single predictor, or an
        n x q matrix with one column per predictor.
    y : array_like
        Response vector of length n.
    add_intercept : bool, optional
        Whether to prepend a column of ones to the design matrix, so that the
        first coefficient is the intercept. Default True.

    Returns
    -------
    RegressionResult
        Named tuple with the estimated coefficients, fitted values, residuals,
        the residual variance estimate sigma^2, the design matrix actually
        used, and the dimensions n and p.

    Raises
    ------
    ValueError
        If X and y have incompatible lengths, or if there are not more
        observations than parameters.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)  # a single predictor becomes one column

    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    if X.shape[0] != len(y):
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {len(y)} entries."
        )

    n, p = X.shape
    if n <= p:
        raise ValueError(
            f"Need more observations than parameters, got n={n} and p={p}."
        )

    # solve (X'X)beta = X'y directly rather than inverting X'X
    coefficients = np.linalg.solve(X.T @ X, X.T @ y)

    fitted = X @ coefficients
    residuals = y - fitted
    sigma_squared = np.sum(residuals**2) / (n - p)

    return RegressionResult(
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        sigma_squared=sigma_squared,
        design_matrix=X,
        n_observations=n,
        n_parameters=p,
    )


def coefficient_standard_errors(result):
    """
    Standard errors of the estimated regression coefficients.

    The covariance matrix of beta_hat is sigma^2 * (X'X)^-1; the standard
    errors are the square roots of its diagonal entries.

    Parameters
    ----------
    result : RegressionResult
        Output of linear_regression.

    Returns
    -------
    np.ndarray
        Standard error of each coefficient, in the same order.
    """
    X = result.design_matrix
    covariance = result.sigma_squared * np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(covariance))


def coefficient_tests(result, alternative="two_sided"):
    """
    t-tests for the individual regression coefficients against H0: beta_j = 0.

    Each coefficient is divided by its standard error; under H0 the ratio
    follows a t distribution with n - p degrees of freedom.

    Parameters
    ----------
    result : RegressionResult
        Output of linear_regression.
    alternative : str
        One of "two_sided", "greater", "less".

    Returns
    -------
    tuple of np.ndarray
        (t statistics, p-values), one entry per coefficient.
    """
    standard_errors = coefficient_standard_errors(result)
    t_statistics = result.coefficients / standard_errors

    df = result.n_observations - result.n_parameters
    null_distribution = stats.t(df=df)

    p_values = np.array(
        [p_value(t, null_distribution, alternative) for t in t_statistics]
    )
    return t_statistics, p_values


def r_squared(result, y):
    """
    Coefficient of determination and its adjusted version.

    R^2 is the share of total variation explained by the model. The adjusted
    version penalises additional predictors, since R^2 can only increase when
    columns are added to the design matrix.

    Parameters
    ----------
    result : RegressionResult
        Output of linear_regression.
    y : array_like
        The original response vector.

    Returns
    -------
    tuple of float
        (R^2, adjusted R^2).
    """
    y = np.asarray(y, dtype=float)

    ss_residual = np.sum(result.residuals**2)
    ss_total = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_residual / ss_total

    n, p = result.n_observations, result.n_parameters
    adjusted = 1 - (1 - r2) * (n - 1) / (n - p)

    return r2, adjusted


def overall_f_test(result, y):
    """
    Global F-test of the regression: H0 says all slopes are zero.

    Rests on the same variance decomposition as the ANOVA, only named
    differently: SS_total = SS_regression + SS_residual. For simple linear
    regression this reduces to the t-test of the slope, with F = t^2.

    The test is inherently one-sided: only large F values indicate that the
    model explains more than residual noise would.

    Parameters
    ----------
    result : RegressionResult
        Output of linear_regression.
    y : array_like
        The original response vector.

    Returns
    -------
    tuple of float
        (F statistic, p-value).
    """
    y = np.asarray(y, dtype=float)

    ss_residual = np.sum(result.residuals**2)
    ss_regression = np.sum((result.fitted - np.mean(y)) ** 2)

    n, p = result.n_observations, result.n_parameters
    df_regression = p - 1
    df_residual = n - p

    F = (ss_regression / df_regression) / (ss_residual / df_residual)

    return F, p_value(
        F, stats.f(dfn=df_regression, dfd=df_residual), alternative="greater"
    )


def ci_mean_response(result, x0, confidence_level=0.95, add_intercept=True):
    """
    Confidence interval for the expected response at a given predictor value.

    Estimates where the regression line itself lies at x0, i.e. the mean of y
    over all observations with that predictor value. Narrower than a
    prediction interval, which additionally carries the noise of a single new
    observation.

    Parameters
    ----------
    result : RegressionResult
        Output of linear_regression.
    x0 : array_like
        Predictor value(s) at which to evaluate, without the intercept entry.
    confidence_level : float, optional
        Desired confidence level. Default 0.95.
    add_intercept : bool, optional
        Must match the setting used when fitting. Default True.

    Returns
    -------
    tuple of float
        (lower, upper) bounds.
    """
    x0 = _prepare_x0(x0, add_intercept)
    X = result.design_matrix
    XtX_inverse = np.linalg.inv(X.T @ X)

    prediction = x0 @ result.coefficients
    standard_error = np.sqrt(result.sigma_squared * (x0 @ XtX_inverse @ x0))

    df = result.n_observations - result.n_parameters
    t = stats.t.ppf(1 - (1 - confidence_level) / 2, df=df)

    return (prediction - t * standard_error, prediction + t * standard_error)


def prediction_interval(result, x0, confidence_level=0.95, add_intercept=True):
    """
    Prediction interval for a single new observation at a given predictor value.

    Differs from the confidence interval by one extra term: predicting an
    individual observation must account for the error term's own variance,
    not just the uncertainty of the fitted line. The interval is therefore
    always wider.

    Parameters
    ----------
    result : RegressionResult
        Output of linear_regression.
    x0 : array_like
        Predictor value(s) at which to predict, without the intercept entry.
    confidence_level : float, optional
        Desired confidence level. Default 0.95.
    add_intercept : bool, optional
        Must match the setting used when fitting. Default True.

    Returns
    -------
    tuple of float
        (lower, upper) bounds.
    """
    x0 = _prepare_x0(x0, add_intercept)
    X = result.design_matrix
    XtX_inverse = np.linalg.inv(X.T @ X)

    prediction = x0 @ result.coefficients
    # the extra "1 +" is the variance of the new observation's error term
    standard_error = np.sqrt(
        result.sigma_squared * (1 + x0 @ XtX_inverse @ x0)
    )

    df = result.n_observations - result.n_parameters
    t = stats.t.ppf(1 - (1 - confidence_level) / 2, df=df)

    return (prediction - t * standard_error, prediction + t * standard_error)


def _prepare_x0(x0, add_intercept):
    """Turn a predictor value into a design-matrix row, adding the intercept."""
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    if add_intercept:
        x0 = np.concatenate([[1.0], x0])
    return x0