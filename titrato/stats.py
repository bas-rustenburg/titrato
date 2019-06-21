import numpy as np
from scipy import stats
from typing import Tuple, Union
from warnings import warn
import numexpr as ne


def pearson_with_confidence(
    x: np.ndarray, y: np.ndarray, α: float
) -> Tuple[float, float, float]:
    """
    Calculate the Pearson r coefficient and confidence intervals.
    
    Parameters
    ----------
    x - array of the first dataset
    y - array of the second dataset
    α - 1 - α is the confidence level.
        E.g. for 95% confidence, alpha = 0.05
    
    Returns
    -------
    Pearson r, r-95%, r+95%

    """
    r, p = stats.pearsonr(x, y)
    n = x.size
    # two-tailed significance
    t_statistic = stats.t.ppf(1.0 - α / 2.0, n - 1)
    # Fisher transform
    F = np.arctanh(r)
    F_min = F - (t_statistic / np.sqrt(n - 3))
    F_plus = F + (t_statistic / np.sqrt(n - 3))
    # Inverse Fisher transform
    r_min = np.tanh(F_min)
    r_plus = np.tanh(F_plus)

    return r, r_min, r_plus


def array_rmse(x1: np.ndarray, x2: np.ndarray) -> float:
    """Returns the root mean squared error between two arrays."""
    return np.sqrt(((x1 - x2) ** 2).mean())


def array_mae(x1: np.ndarray, x2: np.ndarray) -> float:
    """Returns the mean_absolute error between two arrays."""
    return np.asscalar((np.abs(x1 - x2)).mean())


def wrap_pearsonr(x1: np.ndarray, x2: np.ndarray) -> float:
    """Return only pearson r, not the probability."""
    return stats.pearsonr(x1, x2)[0]


def msd(x1: float, x2: float) -> float:
    """Returns the mean squared deviation between two numbers."""
    return np.mean((x1 - x2) ** 2)


def array_median_error(x1: np.ndarray, x2: np.ndarray) -> float:
    """Return the median deviation between numbers in the array."""
    return np.asscalar((np.abs(x1 - x2)).median())


def absolute_loss(x1: float, x2: float):
    """Returns the absolute difference between two numbers."""
    return np.abs(x1 - x2)


def squared_loss(x1: float, x2: float):
    """Returns the squared difference between two numbers"""
    return (x1 - x2) ** 2


def rmsd_curve(curve1: np.ndarray, curve2: np.ndarray, *args) -> float:
    """Calculate the RMSD between two curves."""
    return np.linalg.norm(curve2 - curve1) / np.sqrt(curve2.size)


def msd_curve(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Calculate the MSD between two curves."""
    return np.power(array_rmse(curve2, curve1), 2)


def area_between_curves(curve1: np.ndarray, curve2: np.ndarray, dx=0.1):
    """Calculate area between curves using chunks of size dx."""
    diff = np.abs(curve1 - curve2)
    area = diff * dx

    return np.sum(area)


def area_between_curves_ne(curve1: np.ndarray, curve2: np.ndarray, dx=0.1, pHdim=1):
    """Calculate area between curves using chunks of size dx."""
    areas = ne.evaluate("(curve1 - curve2) * dx")
    return ne.evaluate("sum(areas, axis=1)")


def area_curve_vectorized(curve1, curve2, dx, pHdim=1):
    diff = np.abs(curve1 - curve2)
    if len(curve1.shape) < 2:
        return np.trapz(diff, dx=dx, axis=0)
    else:
        return np.trapz(diff, dx=dx, axis=pHdim)


def rmsd_curve_vectorized(
    curve1: np.ndarray, curve2: np.ndarray, *args, pHdim=1
) -> float:
    """Calculate the RMSD between two sets curves."""
    if len(curve1.shape) < 2:
        return np.linalg.norm(curve2 - curve1, axis=0) / np.sqrt(curve2.shape[0])
    else:
        return np.linalg.norm(curve2 - curve1, axis=pHdim) / np.sqrt(
            curve2.shape[pHdim]
        )


def sum_of_squares(curve1: np.ndarray, curve2: np.ndarray, *args):
    """Sum of the squared differences between two curves. Ignore extra arguments"""
    diff = np.square(curve1 - curve2)
    return np.sum(diff)


def maximum_deviation(curve1: np.ndarray, curve2: np.ndarray, *args):
    """Calculate the largest deviation for any two curves. Ignores extra arguments"""
    diff = curve1 - curve2
    # Return the largest deviation in absolute terms (without losing sign)
    return max(diff.min(), diff.max(), key=abs)


def median_absolute_deviation(curve1: np.ndarray, curve2: np.ndarray, *args):
    """Calculate the median deviation."""
    diff = np.abs(curve1 - curve2)
    return np.median(diff)


def mean_absolute_deviation(curve1: np.ndarray, curve2: np.ndarray, *args):
    """Calculate the mean deviation."""
    diff = np.abs(curve1 - curve2)
    return np.mean(diff)


def gaussian_bootstrap_sample(
    mean: Union[np.ndarray, float], sd: float, bias: float
) -> Union[np.ndarray, float]:
    """Return a sample from a Gaussian bootstrap distribution.
    
    Parameters
    ----------
    mean - average value of the bootstrap distribution
    sd - standard deviation of the boostrap distribution
    bias - bias of the bootstrap distribution.    
    """
    return np.random.normal(mean, sd)


class BootstrapDistribution:
    """Represents a bootstrap distribution, and allows calculation of useful statistics."""

    def __init__(self, sample_estimate: float, bootstrap_estimates: np.array) -> None:
        """
        Parameters
        ----------
        sample_estimate - estimated value from original sample
        bootstrap_estimates - estimated values from bootstrap samples
        """
        self._sample = sample_estimate
        self._bootstrap = bootstrap_estimates
        # approximation of δ = X - μ
        # denoted as δ* = X* - X
        self._delta = bootstrap_estimates - sample_estimate
        self._significance: int = 16
        return

    def __float__(self):
        return float(self._sample)

    def empirical_bootstrap_confidence_intervals(
        self, percent: float
    ) -> Tuple[float, float, float]:
        """Return % confidence intervals.
        
        Uses the approximation that the variation around the center of the 
        bootstrap distribution is the same as the variation around the 
        center of the true distribution. E.g. not sensitive to a bootstrap
        distribution is biased in the median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            warn("Received a fraction but expected a percentile.", UserWarning)

        a = (100.0 - percent) / 2
        upper = np.percentile(self._delta, a)
        lower = np.percentile(self._delta, 100 - a)
        return self._sig_figures(
            self._sample, self._sample - lower, self._sample - upper
        )

    def bootstrap_percentiles(self, percent: float) -> Tuple[float, float, float]:
        """Return percentiles of the bootstrap distribution.
        
        This assumes that the bootstrap distribution is similar to the real 
        distribution. This breaks down when the median of the bootstrap 
        distribution is very different from the sample/true variable median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            raise UserWarning("Received a fraction but expected a percentage.")

        a = (100.0 - percent) / 2
        lower = np.percentile(self._bootstrap, a)
        upper = np.percentile(self._bootstrap, 100 - a)
        return self._sig_figures(self._sample, lower, upper)

    def standard_error(self) -> float:
        """Return the standard error for the bootstrap estimate."""
        # Is calculated by taking the standard deviation of the bootstrap distribution
        return self._bootstrap.std()

    def _sig_figures(
        self, mean: float, lower: float, upper: float, max_sig: int = 16
    ) -> Tuple[float, float, float]:
        """Find the lowest number of significant figures that distinguishes
        the mean from the lower and upper bound.
        """
        i = 16
        for i in range(1, max_sig):
            if (round(mean, i) != round(lower, i)) and (
                round(mean, i) != round(upper, i)
            ):
                break
        self._significance = i
        return round(mean, i), round(lower, i), round(upper, i)

    def __repr__(self) -> str:
        try:
            return "{0:.{3}f} [{1:.{3}f}, {2:.{3}f}]".format(
                *self.bootstrap_percentiles(95), self._significance
            )
        except:
            return str(self.__dict__)

    def _round_float(self, flt):
        return (flt * pow(10, self._significance)) / pow(10, self._significance)

    def to_tuple(self, percent=95):
        """Return the mean, lower and upper percentiles rounded to significant digits"""
        mean, lower, upper = self.bootstrap_percentiles(percent)
        mean = self._round_float(mean)
        lower = self._round_float(lower)
        upper = self._round_float(upper)
        return mean, lower, upper


class PearsonRBootstrapDistribution:
    """Represents a bootstrap distribution, and allows calculation of useful statistics."""

    def __init__(self, sample_estimate: float, bootstrap_estimates: np.array) -> None:
        """
        Parameters
        ----------
        sample_estimate - estimated value from original sample
        bootstrap_estimates - estimated values from bootstrap samples
        """
        self._sample = sample_estimate
        self._bootstrap = bootstrap_estimates
        # approximation of δ = X - μ
        # denoted as δ* = X* - X
        self._delta = bootstrap_estimates - sample_estimate
        self._significance: int = 16
        return

    def empirical_bootstrap_confidence_intervals(
        self, percent: float
    ) -> Tuple[float, float, float]:
        """Return % confidence intervals.
        
        Uses the approximation that the variation around the center of the 
        bootstrap distribution is the same as the variation around the 
        center of the true distribution. E.g. not sensitive to a bootstrap
        distribution is biased in the median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            warn("Received a fraction but expected a percentile.", UserWarning)

        a = (100.0 - percent) / 2
        upper = np.percentile(self._delta, a)
        lower = np.percentile(self._delta, 100 - a)
        new_upper = min(1.00, self._sample - upper)
        return self._sig_figures(self._sample, self._sample - lower, new_upper)

    def bootstrap_percentiles(self, percent: float) -> Tuple[float, float, float]:
        """Return percentiles of the bootstrap distribution.
        
        This assumes that the bootstrap distribution is similar to the real 
        distribution. This breaks down when the median of the bootstrap 
        distribution is very different from the sample/true variable median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            raise UserWarning("Received a fraction but expected a percentage.")

        a = (100.0 - percent) / 2
        lower = np.percentile(self._bootstrap, a)
        upper = np.percentile(self._bootstrap, 100 - a)
        return self._sig_figures(self._sample, lower, min(1.00, upper))

    def standard_error(self) -> float:
        """Return the standard error for the bootstrap estimate."""
        # Is calculated by taking the standard deviation of the bootstrap distribution
        return self._bootstrap.std()

    def _sig_figures(
        self, mean: float, lower: float, upper: float, max_sig: int = 16
    ) -> Tuple[float, float, float]:
        """Find the lowest number of significant figures that distinguishes
        the mean from the lower and upper bound.
        """
        i: int = 1
        for i in range(1, max_sig):
            if (round(mean, i) != round(lower, i)) and (
                round(mean, i) != round(upper, i)
            ):
                break
        self._significance = i
        return round(mean, i), round(lower, i), round(upper, i)

    def __repr__(self) -> str:
        return "{0:.{3}f}; [{1:.{3}f}, {2:.{3}f}]".format(
            *self.bootstrap_percentiles(95), self._significance
        )


def bootstrapped_func(values: np.ndarray, nbootstrap_samples=10000, func=np.mean):
    """Perform a non-parametric bootstrap over values to estimate a function such as the mean."""
    # Draw random array indices to resample data points with replacement
    samples = np.random.randint(0, values.size, values.size * nbootstrap_samples)
    # Get the right output shape
    samples = samples.reshape(values.size, nbootstrap_samples)
    # Allow 2D indexing by adding empty axis
    value_2d = values[:, np.newaxis]
    return BootstrapDistribution(func(values), func(value_2d[samples], axis=0))
