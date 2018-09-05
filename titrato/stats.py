import numpy as np
from scipy import stats
from typing import Tuple, Union

def pearson_with_confidence(x:np.ndarray,y:np.ndarray, α: float)-> Tuple[float,float,float]: 
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
    r,p = stats.pearsonr(x,y)
    n = x.size
    # two-tailed significance
    t_statistic = stats.t.ppf(1.0-α/2.0, n-1)
    # Fisher transform
    F = np.arctanh(r)    
    F_min = F - (t_statistic / np.sqrt(n-3))    
    F_plus = F + (t_statistic / np.sqrt(n-3))
    # Inverse Fisher transform
    r_min = np.tanh(F_min)
    r_plus = np.tanh(F_plus)

    return r, r_min, r_plus

def array_rmse(x1: np.ndarray, x2:np.ndarray) -> float:
    """Returns the root mean squared error between two arrays."""
    return np.sqrt(((x1-x2)**2).mean())

def msd(x1:float, x2:float) -> float:
    """Returns the mean squared deviation between two numbers."""
    return np.mean((x1-x2)**2)

def absolute_difference(x1: float, x2: float):
    """Returns the absolute difference between two numbers."""
    return np.abs(x1-x2)

def rmsd_curve(curve1:np.ndarray, curve2:np.ndarray)-> float:
    """Calculate the RMSD between two curves."""
    return np.linalg.norm(curve2 - curve1) / np.sqrt(curve2.size)

def msd_curve(curve1:np.ndarray, curve2:np.ndarray)-> float:
    """Calculate the MSD between two curves."""
    return np.power(array_rmse(curve2, curve1), 2)

def area_between_curves(curve1:np.ndarray, curve2:np.ndarray, dx):
    diff = np.abs(curve1-curve2)
    area = diff * dx
    return np.sum(area)

def gaussian_bootstrap_sample(mean:Union[np.ndarray,float], sd: float, bias:float) -> Union[np.ndarray, float]:
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

    def empirical_bootstrap_confidence_intervals(self, percent:float) -> Tuple[float,float,float]:
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
            raise UserWarning("Received a fraction but expected a percentile.")
        
        a = (100.0 - percent)/2        
        upper = np.percentile(self._delta, a)
        lower = np.percentile(self._delta, 100-a)        
        return self._sig_figures(self._sample, self._sample - lower, self._sample - upper)

    def bootstrap_percentiles(self, percent:float) -> Tuple[float,float,float]:
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
        
        a = (100.0 - percent) /2        
        lower = np.percentile(self._bootstrap, a)
        upper = np.percentile(self._bootstrap, 100-a)        
        return self._sig_figures(self._sample, lower, upper)

    def standard_error(self)-> float:
        """Return the standard error for the bootstrap estimate."""
        # Is calculated by taking the standard deviation of the bootstrap distribution
        return self._bootstrap.std()
    
    def _sig_figures(self, mean:float, lower:float, upper:float, max_sig:int=16) -> Tuple[float,float,float]:
        """Find the lowest number of significant figures that distinguishes
        the mean from the lower and upper bound.
        """
        
        for i in range(1,max_sig):
            if (round(mean,i) != round(lower,i)) and (round(mean,i) != round(upper,i)):
                break
        self._significance = i
        return round(mean,i), round(lower,i), round(upper,i)
    
    def __repr__(self) -> str:        
        return "{0:.{3}f}; [{1:.{3}f}, {2:.{3}f}]".format(*self.empirical_bootstrap_confidence_intervals(95), self._significance)   
 

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

    def empirical_bootstrap_confidence_intervals(self, percent:float) -> Tuple[float,float,float]:
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
            raise UserWarning("Received a fraction but expected a percentile.")
        
        a = (100.0 - percent)/2        
        upper = np.percentile(self._delta, a)
        lower = np.percentile(self._delta, 100-a)        
        new_upper = min(1.00, self._sample - upper)
        return self._sig_figures(self._sample, self._sample - lower, new_upper)

    def bootstrap_percentiles(self, percent:float) -> Tuple[float,float,float]:
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
        
        a = (100.0 - percent) /2        
        lower = np.percentile(self._bootstrap, a)
        upper = np.percentile(self._bootstrap, 100-a)        
        return self._sig_figures(self._sample, lower, min(1.00, upper))

    def standard_error(self)-> float:
        """Return the standard error for the bootstrap estimate."""
        # Is calculated by taking the standard deviation of the bootstrap distribution
        return self._bootstrap.std()
    
    def _sig_figures(self, mean:float, lower:float, upper:float, max_sig:int=16) -> Tuple[float,float,float]:
        """Find the lowest number of significant figures that distinguishes
        the mean from the lower and upper bound.
        """
        
        for i in range(1,max_sig):
            if (round(mean,i) != round(lower,i)) and (round(mean,i) != round(upper,i)):
                break
        self._significance = i
        return round(mean,i), round(lower,i), round(upper,i)
    
    def __repr__(self) -> str:        
        return "{0:.{3}f}; [{1:.{3}f}, {2:.{3}f}]".format(*self.empirical_bootstrap_confidence_intervals(95), self._significance)     
    


