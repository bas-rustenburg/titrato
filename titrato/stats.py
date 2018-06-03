import numpy as np
from scipy import stats
from typing import Tuple

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

def rmsd(x1: float, x2:float) -> float:    
    """Returns the root mean squared deviation between two numbers"""
    return np.sqrt((x1 -x2)**2)

def msd(x1:float, x2:float) -> float:
    """Returns the mean squared deviation between two numbers."""
    return (x1 -x2)**2

def absolute_difference(x1: float, x2: float):
    """Returns the absolute difference between two numbers."""
    return np.abs(x1-x2)

def rmsd_curve(curve1:np.ndarray, curve2:np.ndarray)-> float:
    """Calculate the RMSD between two curves."""
    return np.linalg.norm(curve2 - curve1) / np.sqrt(curve2.size)

def msd_curve(curve1:np.ndarray, curve2:np.ndarray)-> float:
    """Calculate the MSD between two curves."""
    return np.power(rmsd(curve2, curve1), 2)

def area_between_curves(curve1:np.ndarray, curve2:np.ndarray, dx):
    diff = np.abs(curve1-curve2)
    area = diff * dx
    return np.sum(area)

