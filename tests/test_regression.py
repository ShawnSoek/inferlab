from inferlab.regression import (
    linear_regression,
    coefficient_standard_errors,
    coefficient_tests,
    r_squared,
    overall_f_test,
    ci_mean_response,
    prediction_interval,
)

import numpy as np
from pytest import approx
import pytest
import scipy
from scipy import stats
import statsmodels.api as sm


TRUE_INTERCEPT = 3.0
TRUE_SLOPE = 2.5


@pytest.fixture
def simple_data():
    """Single predictor with known true coefficients (3.0, 2.5)."""
    rng = np.random.default_rng(17)
    x = rng.uniform(0, 10, 40)
    y = TRUE_INTERCEPT + TRUE_SLOPE * x + rng.normal(0, 2, 40)
    return x, y


@pytest.fixture
def multiple_data():
    """Two predictors with known true coefficients (1.0, 2.0, -1.5)."""
    rng = np.random.default_rng(23)
    x1 = rng.uniform(0, 10, 60)
    x2 = rng.uniform(-5, 5, 60)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 1, 60)
    return np.column_stack([x1, x2]), y


# ---------------------------------------------------------------------------
# 1. Comparison against scipy (simple regression)
# ---------------------------------------------------------------------------

def test_coefficients_match_scipy(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)
    reference = scipy.stats.linregress(x, y)

    assert result.coefficients[0] == approx(reference.intercept, rel=1e-10)
    assert result.coefficients[1] == approx(reference.slope, rel=1e-10)


def test_standard_errors_match_scipy(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)
    errors = coefficient_standard_errors(result)
    reference = scipy.stats.linregress(x, y)

    assert errors[0] == approx(reference.intercept_stderr, rel=1e-10)
    assert errors[1] == approx(reference.stderr, rel=1e-10)


def test_coefficient_p_value_matches_scipy(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)
    _, p_values = coefficient_tests(result)
    reference = scipy.stats.linregress(x, y)

    assert p_values[1] == approx(reference.pvalue, rel=1e-10)


def test_r_squared_matches_scipy(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)
    r2, _ = r_squared(result, y)
    reference = scipy.stats.linregress(x, y)

    assert r2 == approx(reference.rvalue**2, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. Comparison against statsmodels (multiple regression)
# ---------------------------------------------------------------------------

def test_multiple_regression_matches_statsmodels(multiple_data):
    X, y = multiple_data
    result = linear_regression(X, y)
    reference = sm.OLS(y, sm.add_constant(X)).fit()

    assert result.coefficients == approx(reference.params, rel=1e-9)
    assert coefficient_standard_errors(result) == approx(reference.bse, rel=1e-9)

    t_statistics, p_values = coefficient_tests(result)
    assert t_statistics == approx(reference.tvalues, rel=1e-9)
    assert p_values == approx(reference.pvalues, rel=1e-9)


def test_multiple_r_squared_matches_statsmodels(multiple_data):
    X, y = multiple_data
    result = linear_regression(X, y)
    r2, adjusted = r_squared(result, y)
    reference = sm.OLS(y, sm.add_constant(X)).fit()

    assert r2 == approx(reference.rsquared, rel=1e-9)
    assert adjusted == approx(reference.rsquared_adj, rel=1e-9)


def test_multiple_f_test_matches_statsmodels(multiple_data):
    X, y = multiple_data
    result = linear_regression(X, y)
    F, p = overall_f_test(result, y)
    reference = sm.OLS(y, sm.add_constant(X)).fit()

    assert F == approx(reference.fvalue, rel=1e-9)
    assert p == approx(reference.f_pvalue, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Recovering known truth
# ---------------------------------------------------------------------------

def test_estimates_recover_true_coefficients(simple_data):
    # the data were generated with a known intercept and slope
    x, y = simple_data
    result = linear_regression(x, y)

    assert result.coefficients[0] == approx(TRUE_INTERCEPT, abs=1.5)
    assert result.coefficients[1] == approx(TRUE_SLOPE, abs=0.2)


def test_true_coefficients_lie_in_their_confidence_intervals(simple_data):
    # each estimate plus/minus t * se must cover the true value
    x, y = simple_data
    result = linear_regression(x, y)
    errors = coefficient_standard_errors(result)

    df = result.n_observations - result.n_parameters
    t = scipy.stats.t.ppf(0.975, df=df)

    for estimate, error, truth in zip(
        result.coefficients, errors, [TRUE_INTERCEPT, TRUE_SLOPE]
    ):
        assert estimate - t * error < truth < estimate + t * error


def test_multiple_regression_recovers_true_coefficients(multiple_data):
    X, y = multiple_data
    result = linear_regression(X, y)

    assert result.coefficients == approx(np.array([1.0, 2.0, -1.5]), abs=0.5)


# ---------------------------------------------------------------------------
# 4. Least squares properties
# ---------------------------------------------------------------------------

def test_residuals_are_orthogonal_to_design_matrix(simple_data):
    # X'e = 0 is the defining property of the least squares solution
    x, y = simple_data
    result = linear_regression(x, y)

    assert result.design_matrix.T @ result.residuals == approx(
        np.zeros(result.n_parameters), abs=1e-9
    )


def test_residuals_sum_to_zero_with_intercept(simple_data):
    # follows from orthogonality against the column of ones
    x, y = simple_data
    result = linear_regression(x, y)

    assert np.sum(result.residuals) == approx(0.0, abs=1e-9)


def test_fitted_values_have_the_same_mean_as_the_response(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)

    assert np.mean(result.fitted) == approx(np.mean(y), rel=1e-12)


def test_variance_decomposition_holds(simple_data):
    # SS_total = SS_regression + SS_residual, the same identity as in ANOVA
    x, y = simple_data
    result = linear_regression(x, y)

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_regression = np.sum((result.fitted - np.mean(y)) ** 2)
    ss_residual = np.sum(result.residuals**2)

    assert ss_total == approx(ss_regression + ss_residual, rel=1e-10)


def test_perfect_fit_gives_zero_residuals_and_r_squared_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 + 3.0 * x  # exactly on a line, no noise

    result = linear_regression(x, y)
    r2, _ = r_squared(result, y)

    assert result.coefficients == approx(np.array([2.0, 3.0]), abs=1e-10)
    assert result.residuals == approx(np.zeros(5), abs=1e-10)
    assert r2 == approx(1.0)


# ---------------------------------------------------------------------------
# 5. The F = t^2 duality
# ---------------------------------------------------------------------------

def test_overall_f_equals_squared_slope_t_for_simple_regression(simple_data):
    # with a single predictor the global F-test is the slope's t-test
    x, y = simple_data
    result = linear_regression(x, y)

    t_statistics, p_values = coefficient_tests(result)
    F, F_p = overall_f_test(result, y)

    assert F == approx(t_statistics[1] ** 2, rel=1e-9)
    assert F_p == approx(p_values[1], rel=1e-9)


# ---------------------------------------------------------------------------
# 6. Confidence and prediction intervals
# ---------------------------------------------------------------------------

def test_prediction_interval_is_wider_than_confidence_interval(simple_data):
    # predicting one new observation carries the error term's variance on top
    x, y = simple_data
    result = linear_regression(x, y)

    ci_lower, ci_upper = ci_mean_response(result, 5.0)
    pi_lower, pi_upper = prediction_interval(result, 5.0)

    assert pi_upper - pi_lower > ci_upper - ci_lower
    assert pi_lower < ci_lower
    assert pi_upper > ci_upper


def test_intervals_are_centered_on_the_fitted_value(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)
    x0 = 5.0
    expected = result.coefficients[0] + result.coefficients[1] * x0

    ci_lower, ci_upper = ci_mean_response(result, x0)
    pi_lower, pi_upper = prediction_interval(result, x0)

    assert (ci_lower + ci_upper) / 2 == approx(expected)
    assert (pi_lower + pi_upper) / 2 == approx(expected)


def test_confidence_interval_is_narrowest_at_the_mean_predictor(simple_data):
    # the further x0 lies from x-bar, the more uncertain the fitted line
    x, y = simple_data
    result = linear_regression(x, y)

    def width(interval):
        lower, upper = interval
        return upper - lower

    at_mean = width(ci_mean_response(result, np.mean(x)))
    far_away = width(ci_mean_response(result, np.mean(x) + 4))

    assert at_mean < far_away


def test_higher_confidence_widens_both_intervals(simple_data):
    x, y = simple_data
    result = linear_regression(x, y)

    def width(interval):
        lower, upper = interval
        return upper - lower

    assert width(ci_mean_response(result, 5.0, confidence_level=0.99)) > width(
        ci_mean_response(result, 5.0, confidence_level=0.95)
    )
    assert width(prediction_interval(result, 5.0, confidence_level=0.99)) > width(
        prediction_interval(result, 5.0, confidence_level=0.95)
    )


def test_intervals_work_for_multiple_predictors(multiple_data):
    X, y = multiple_data
    result = linear_regression(X, y)

    ci_lower, ci_upper = ci_mean_response(result, [5.0, 0.0])
    pi_lower, pi_upper = prediction_interval(result, [5.0, 0.0])

    expected = result.coefficients @ np.array([1.0, 5.0, 0.0])
    assert ci_lower < expected < ci_upper
    assert pi_upper - pi_lower > ci_upper - ci_lower


# ---------------------------------------------------------------------------
# 7. Adjusted R-squared and model size
# ---------------------------------------------------------------------------

def test_adjusted_r_squared_is_at_most_r_squared(multiple_data):
    X, y = multiple_data
    result = linear_regression(X, y)
    r2, adjusted = r_squared(result, y)

    assert adjusted <= r2


def test_r_squared_never_decreases_when_adding_a_predictor(multiple_data):
    # even pure noise cannot lower R^2 -- which is why the adjusted version
    # exists
    X, y = multiple_data
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(y)).reshape(-1, 1)

    r2_small, _ = r_squared(linear_regression(X, y), y)
    r2_large, _ = r_squared(linear_regression(np.column_stack([X, noise]), y), y)

    assert r2_large >= r2_small - 1e-12


def test_adjusted_r_squared_drops_when_adding_a_useless_predictor(multiple_data):
    # adjusted R^2 only falls if the new predictor has |t| < 1; with seed 0
    # the noise column is clearly insignificant (checked via its p-value)
    X, y = multiple_data
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(y)).reshape(-1, 1)

    X_extended = np.column_stack([X, noise])
    extended = linear_regression(X_extended, y)
    _, p_values = coefficient_tests(extended)
    assert p_values[-1] > 0.5

    _, adjusted_small = r_squared(linear_regression(X, y), y)
    _, adjusted_large = r_squared(extended, y)

    assert adjusted_large < adjusted_small


# ---------------------------------------------------------------------------
# 8. Options and error paths
# ---------------------------------------------------------------------------

def test_regression_without_intercept_omits_the_ones_column(simple_data):
    x, y = simple_data
    result = linear_regression(x, y, add_intercept=False)

    assert result.n_parameters == 1
    assert result.design_matrix.shape == (len(x), 1)


def test_regression_accepts_plain_python_lists():
    x = [1, 2, 3, 4, 5]
    y = [2.1, 4.2, 5.9, 8.1, 9.9]

    result = linear_regression(x, y)
    assert result.coefficients.shape == (2,)


def test_regression_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        linear_regression(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_regression_rejects_too_few_observations():
    # n must exceed p, otherwise sigma^2 is undefined
    with pytest.raises(ValueError):
        linear_regression(np.array([1.0, 2.0]), np.array([3.0, 4.0]))