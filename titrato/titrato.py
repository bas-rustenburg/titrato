"""
titrato.py
Utilities for calculating titration curves.
"""

import numpy as np
import pandas as pd
import holoviews as hv
import logging
from itertools import chain, combinations

# Make sure to install typing_extensions
from typing import (
    Dict,
    List,
    Iterator,
    Union,
    Iterable,
    Tuple,
    Any,
    Optional,
    Callable,
    TypeVar,
)
from typing_extensions import Protocol
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import warnings
from . import data_dir
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
import math
from copy import deepcopy

logger = logging.getLogger()

# Constant factor for transforming a number that is log base 10 to log base e
ln10 = np.log(10)

# Keep track of papers that we take equations from
references = [
    "RI Allen et al J Pharm Biomed Anal 17 (1998) 699-712",
    "Ullmann, J Phys Chem B vol 107, No 5, 2003. 1263-1271",
    "Mehtap Isik et al, to be submitted to JCAMD 2018",
]


def quicksave(
    plot_object: Any,
    outfile_basename: str,
    renderer: str = "matplotlib",
    fmt: Optional[str] = None,
) -> None:
    """Quicksave function for a holoviews plot to a standalone html file.

    Parameters
    ----------
    plot_object - a holoviews layout object    
    outfile_basename - The base name of the output file.
    renderer - name of a holoviews renderer, e.g 'bokeh', 'matplotlib', or 'plotly'
    """
    hv.renderer(renderer).save(plot_object, outfile_basename, fmt=fmt)

    return


def graph_to_axes(
    graph: Union[nx.DiGraph, nx.Graph], ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a graph with labels on edges, and nodes.

    Parameters
    ----------
    graph - the networkx Graph/DiGraph to visualize
    ax - the axes object, if none will be plotted in current axes.

    Returns
    -------
    The Axes with the plot of the graph

    """
    colors = [
        "#007fff",
        "#e52b50",
        "#a4c639",
        "#fdee00",
        "#ed872d",
        "#966fd6",
        "#eeeeee",
        "#f4bbff",
        "#465945",
        "#a50b5e",
    ]
    pos = nx.spring_layout(graph, iterations=5000)
    if ax is None:
        ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, ax=ax)
    # nx.draw_networkx_edge_labels(
    #    graph, pos, font_size=10, font_family='sans-serif', ax=ax)

    for i, nodes in enumerate(nx.strongly_connected_components(graph)):
        nx.draw_networkx_nodes(
            graph.subgraph(nodes),
            pos,
            node_size=5000,
            node_color=colors[i % len(colors)],
            alpha=0.3,
            node_shape="s",
            ax=ax,
        )

    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif", ax=ax)

    return ax


def free_energy_from_pka(
    bound_protons: int, Σ_pKa: float, pH: np.ndarray,
) -> np.ndarray:
    """This function calculates the energy of a state based on unbound protons, pH and the pKa

    bound_protons - Number of protons bound to this state
    pH - 1D-array of pH values (float)
    Σ_pKa - Sum of the microscopic acid dissociation constants for occupied sites. 

    Returns
    -------
    The free energy of the given state.

    Notes
    -----
    Defined as ((n_bound * pH) - Σ(micro_pKas)) * ln 10

    See also Ullmann, J Phys Chem B vol 107, No 5, 2003.
    """
    nμ = bound_protons * pH
    G = -Σ_pKa
    return (nμ + G) * ln10 


def populations_from_free_energies(free_energies: np.ndarray) -> np.ndarray:
    """Calculate the populations of states from their relative free energies.

    Parameters
    ----------
    free_energies - 2D-array of free energies with axes [state,pH]

    Returns
    -------
    2D-array of populations with axes [state,pH]
    """
    probability = np.exp(-free_energies)
    Z = np.sum(probability, axis=0)
    return probability / Z


def micro_pKas_to_equilibrium_graph(csvfile: str) -> nx.DiGraph:
    """
    Takes a microstate pairs csv file and returns directed Graph

    Parameters
    ----------
    csvfile - The path of a csvfile.

    The csv file is expected to have the format:
    Protonated state, Deprotonated state, pKa
    on each line. Header is optional.

    Returns
    -------
    Directed graph of the micropKas

    """
    mol_frame = pd.read_csv(csvfile)
    # Override the column names, in case csv file has header
    # respect extra columns
    new_names = list(mol_frame.columns)
    new_names[:3] = ["Protonated", "Deprotonated", "pKa"]
    mol_frame.columns = new_names

    # Direction of edges of the graph is deprotonated -> protonated state
    from_list = list(mol_frame["Deprotonated"])
    to_list = list(mol_frame["Protonated"])
    # Add properties
    # TODO, allow reading additional properties from csv (such as images)?
    properties = [dict(pKa=p) for p in mol_frame["pKa"]]

    graph = nx.DiGraph()
    graph.add_edges_from(zip(from_list, to_list, properties))

    return graph


def macro_pkas_to_species_concentration(
    pkas: np.ndarray, ph: float, total_concentration: float
) -> np.ndarray:
    """
    Convert macroscopic pKas to concentrations using a system of equations.

    Parameters
    ----------
    pkas - 1D array of macroscopic pKa values
    ph - The pH value.
        Note: This function is not compatible with arrays of pH values
    total_concentration - the total concentration of all species.
        Set this to 1 for fractional concentrations.

    Returns
    -------
    Concentration of macroscopic species in order of deprotonated->most protonated


    Notes
    -----

    According to RI Allen et al J Pharm Biomed Anal 17 (1998) 699-712, the
    concentrations of the macroscates, C(n), where n is the index of species
    can be calculated according to a system of equations from a series of m = n-1
    pKas, Kn, given initial concentration y.

    If we treat the Kn as known, the concentrations can be determined as follows.

    |y|  | 1.   1.   1.   1. |^-1    |C(1)|
    |0|  | K1  -H.   0.   0. |       |C(2)|
    |0|  | 0.   K2  -H.   0. |    =  |C(3)|
    |0|  | 0.   0.   Km  -H. |       |C(n)|

    Shorthand:
    C = Y M

    Rearranging gives
    C = M-1 Y

    """
    # TODO see if we can vectorize the procedure for an entire set of pH values

    # Retrieve actual equilibrium constants
    kas = np.power(10, -pkas)
    # Concentration of hydrogen atoms
    H = np.power(10, -ph)
    # One more species than macropKa
    num_species = pkas.shape[0] + 1
    initial_concentrations = np.zeros(num_species)
    initial_concentrations[0] = total_concentration
    rows = list()
    # First row is all ones
    rows.append(np.ones(num_species))

    # Each row has the Ka, and -H along the diagonal
    for row_num, ka in enumerate(kas):
        row = np.zeros(num_species)
        row[row_num] = ka
        row[row_num + 1] = -H
        rows.append(row)

    system_of_equations = np.array(rows)
    # Need to transpose the matrix for the inversion to work correctly
    species = np.linalg.inv(system_of_equations) @ initial_concentrations
    species = np.asarray(species)
    # Remove 0-size dimensions
    species = np.squeeze(species)
    # Return in order of deprotonated->most-protonated
    # for consistency with micropKa state orders
    return species[::-1]


def populations_from_macro_pka(pkas: np.ndarray, phvalues: np.ndarray) -> np.ndarray:
    """Convert a set of macro pKa values to a set of populations.

    Parameters
    ----------
    pkas - 1D array of macroscopic pKa values
    ph - The pH value.

    """
    # create array in memory
    populations = np.empty([pkas.size + 1, phvalues.size])

    # Set concentration for each state for each pH
    for col, ph in enumerate(phvalues):
        # The initial concentrations is the is set to 1, so that all concentrations are fractional
        populations[:, col] = macro_pkas_to_species_concentration(pkas, ph, 1)

    return populations


def free_energy_from_population(
    populations: np.ndarray, nan_r_tolerance=0.95
) -> np.ndarray:
    """Calculate relative free energies from their populations.

    Parameters    
    ----------
    populations - 2D-array of populations with axes [state,pH]

    nan_r_tolerance - If linear regression for missing values has absolute R value below this number, show warning.
        Due to numeric instability, free energy curves start incorporating errors at small populations.
        Small amounts of missing data are replaced with a linear regression.


    Returns
    -------
    free_energies - 2D-array of free energies with axes [state,pH]

    Notes
    -----
    The relative free energy is defined as:
    relative dG[1..n] = - ln (Pop[1..n]) - ln (Pop)[n]
    """
    old_settings = np.geterr()
    np.seterr(invalid="ignore", divide="ignore")

    # Set to 0.0 value to invalidate numbers that are too small
    # only for cases where 2/3 majority of curve is still good.
    small_values = populations < sys.float_info.epsilon
    if np.count_nonzero(small_values) < populations.shape[1] // 2:
        populations = deepcopy(populations)
        populations[small_values] = 0.0
    free_energies = -np.log(populations)
    np.seterr(**old_settings)

    # Due to floating point precision limitations, small populations are troublesome
    # However, free energy curves should be linear so we can rescue values by linear regression
    if not np.all(np.isfinite(free_energies)):

        # Normalize if possible by the first state without nans, to obtain mostly linear curves

        for state in range(free_energies.shape[0]):
            if np.all(np.isfinite(free_energies[state])):
                free_energies -= free_energies[state]
                break

        x_val = np.arange(free_energies.shape[1])
        for state in range(free_energies.shape[0]):
            if not np.all(np.isfinite(free_energies[state])):
                nans, index_helper = invalid_value_helper(free_energies[state])
                num_nans = np.count_nonzero(nans)
                if num_nans > populations.shape[1] // 2:
                    warnings.warn(
                        "More than 1/2 ({}) NaN free energies found ({}).".format(
                            populations.shape[1] // 2, num_nans
                        )
                    )

                slope, intercept, r_value, p_value, std_err = linregress(
                    x_val[~nans], free_energies[state, ~nans]
                )
                if np.abs(r_value) < nan_r_tolerance:
                    warnings.warn(
                        "Absolute R-value (|{}|) < {}, free energy slope may not be linear.".format(
                            r_value, nan_r_tolerance
                        )
                    )

                for nan_index in index_helper(nans):
                    free_energies[state][nan_index] = slope * nan_index + intercept

    return free_energies - free_energies[0]


def powerset(iterable: Iterable[float]) -> Iterator[Tuple[float, ...]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def invalid_value_helper(y):
    """Helper to handle indices and logical indices of NaNs and infinite values.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= invalid_value_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    Original source:
        https://stackoverflow.com/a/652069
    """

    return ~np.isfinite(y), lambda z: z.nonzero()[0]


def dynamic_range_solve(
    x1: float, x2: float, lower: float, upper: float
) -> Tuple[float, float]:
    """Fill any missing values with the lower, or upper bound of a dynamic range, based on the position of the value to be compared against."""
    if np.isnan(x1):
        if x2 >= 7.0:
            return upper, x2
        elif x2 < 7.0:
            return lower, x2
    elif np.isnan(x2):
        if x1 >= 7.0:
            return x1, upper
        elif x1 < 7.0:
            return x1, lower
    else:
        return x1, x2


def fixed_cost_solve(x1: float, x2: float, cost: float = 10.0) -> Tuple[float, float]:
    """Fill any missing value with the other value, plus an offset"""
    if np.isnan(x1):
        return x2 + cost, x2
    elif np.isnan(x2):
        return x1, x1 + cost
    else:
        return x1, x2


def fixed_value_solve(x1: float, x2: float, value: float = 0.0) -> Tuple[float, float]:
    """Fill any missing value with a fixed value."""
    if np.isnan(x1):
        return value, x2
    elif np.isnan(x2):
        return x1, value
    else:
        return x1, x2


def fit_titration_curve(
    curve1, curve2, fitness_function, *fitness_args
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align two curves by integer values using fitness function.
    Returns
    -------
    curve 1 (unmodified), curve 2 (aligned), fittness score

    """
    # Grab charge curves, and ensure lowest value is 0.
    q1 = curve1 - int(round(min(curve1)))
    q2 = curve2 - int(round(min(curve2)))
    max1 = int(round(max(q1)))
    max2 = int(round(max(q2)))

    # Maximum range of the alignment is from -max, max
    m = max([max1, max2])
    offsets = (np.arange(-m, m + 1) * np.ones([q2.size, 2 * m + 1])).T
    # charges in q2 + all possible offsets
    q_array = (np.ones([2 * m + 1, q2.size]) * q2) + offsets
    scores = np.apply_along_axis(fitness_function, 1, q_array, curve1, *fitness_args)
    # Return the best solution
    argmin = np.argmin(scores)
    return curve1, q_array[argmin, :], scores[argmin]


def fit_titration_curves_3d(curves1, curves2, fitness_function, *fitness_args):
    """Vectorized version of fit_titration_curve,

    Parameters
    ----------
    curves1 - [,pH]
    curves2
    fitness_function
    fitness_args

    Returns
    -------

    """
    q1 = curves1 - np.round(np.min(curves1, axis=1))[:, np.newaxis]
    q2 = curves2 - np.round(np.min(curves2, axis=1))[:, np.newaxis]
    max1 = np.round(np.max(q1, axis=1)).astype(np.int)
    max2 = np.round(np.max(q2, axis=1)).astype(np.int)
    m = np.max([max1, max2])

    curve1_multi = np.repeat(curves1[:, :, np.newaxis], 2 * m + 1, axis=2)

    # Maximum range of the alignment is from -max, max
    offsets = np.ones([*q2.shape, 2 * m + 1]) * np.arange(-m, m + 1)
    q_array = (np.ones([*q2.shape, 2 * m + 1]) * q2[:, :, np.newaxis]) + offsets
    scores = fitness_function(q_array, curve1_multi, *fitness_args)
    argmin = np.argmin(scores, axis=1)
    # return curve1, curve2
    return curves1, q_array[np.arange(q_array.shape[0]), :, argmin], scores.min(axis=1)


def hungarian_pka(
    experimental_pkas: np.ndarray,
    predicted_pkas: np.ndarray,
    cost_function: Callable[[float, float], float],
) -> pd.DataFrame:
    """Using the Hungarian algorithm (a.ka. linear sum assignment), return a mapping between experiment, and predicted pKas,
    and the error.    
   
    Parameters
    ----------
    experimental_pkas - 1D array of experimental pKa values
    predicted_pkas - 1D array of predicted pKa values
    cost_function - function to calculate the cost of any invidual mapping    
    """

    n_experimental = experimental_pkas["pKa"].values.size
    n_predicted = predicted_pkas["pKa"].values.size
    size = max([n_experimental, n_predicted])
    # not matched up has 0 cost
    cost_matrix = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if i < n_experimental and j < n_predicted:
                experimental_pka = experimental_pkas["pKa"].values[i]
                predicted_pka = predicted_pkas["pKa"].values[j]
                cost_matrix[i, j] = cost_function(experimental_pka, predicted_pka)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    df = pd.DataFrame(columns=["Experimental", "Predicted", "Cost"])
    for i, row_id in enumerate(row_indices):
        col_id = col_indices[i]
        # Skip unmatched
        if row_id >= n_experimental:
            df = df.append(
                {
                    "Experimental": np.nan,
                    "Experimental SEM": np.nan,
                    "Predicted": predicted_pkas["pKa"].values[col_id],
                    "Predicted SEM": predicted_pkas["SEM"].values[col_id],
                    "Cost": cost_matrix[row_id, col_id],
                },
                ignore_index=True,
            )
        elif col_id >= n_predicted:
            df = df.append(
                {
                    "Experimental": experimental_pkas["pKa"].values[row_id],
                    "Experimental SEM": experimental_pkas["SEM"].values[row_id],
                    "Predicted": np.nan,
                    "Predicted SEM": np.nan,
                    "Cost": cost_matrix[row_id, col_id],
                },
                ignore_index=True,
            )
        else:
            df = df.append(
                {
                    "Experimental": experimental_pkas["pKa"].values[row_id],
                    "Experimental SEM": experimental_pkas["SEM"].values[row_id],
                    "Predicted": predicted_pkas["pKa"].values[col_id],
                    "Predicted SEM": predicted_pkas["SEM"].values[col_id],
                    "Cost": cost_matrix[row_id, col_id],
                },
                ignore_index=True,
            )

    return df


def align_pka(
    experimental_pkas: np.ndarray,
    predicted_pkas: np.ndarray,
    cost_function: Callable[[float, float], float],
):
    """Align pKas sequentialy and find the alignment that minimizes the cost.
    
    Parameters
    ----------
    experimental_pkas - 1D array of experimental pKa values
    predicted_pkas - 1D array of predicted pKa values
    cost_function - function to calculate the cost of any invidual mapping.    
    """
    n_experimental = experimental_pkas["pKa"].values.size
    n_predicted = predicted_pkas["pKa"].values.size
    # biggest size, and additional zero
    num_ka = max([n_experimental, n_predicted])
    exp = np.empty(num_ka)
    pred = np.empty(num_ka)
    sem_exp = np.empty(num_ka)
    sem_pred = np.empty(num_ka)
    pred[:] = np.nan
    exp[:] = np.nan
    sem_pred[:] = np.nan
    sem_exp[:] = np.nan

    experimental_pkas["pKa"].values.sort()
    predicted_pkas["pKa"].values.sort()
    exp[:n_experimental] = experimental_pkas["pKa"].values
    pred[:n_predicted] = predicted_pkas["pKa"].values

    sem_pred[:n_predicted] = predicted_pkas["SEM"].values
    sem_exp[:n_experimental] = experimental_pkas["SEM"].values

    min_cost = 1.0e14
    solution = deepcopy(pred)
    sol_sem = deepcopy(sem_pred)
    solution_cost = np.empty(num_ka)
    for _ in range(num_ka):
        pred = np.roll(pred, 1)
        sem_pred = np.roll(sem_pred, 1)
        cost = []
        for e1, p1 in np.array([deepcopy(exp), deepcopy(pred)]).T:
            e1, p1 = fixed_cost_solve(e1, p1, 14.00)
            cost.append(cost_function(e1, p1))
        total_cost = np.sum(cost)
        if total_cost < min_cost:
            solution = deepcopy(pred)
            min_cost = total_cost
            solution_cost = cost
            sol_sem = deepcopy(sem_pred)

    return pd.DataFrame.from_dict(
        {
            "Experimental": exp,
            "Experimental SEM": sem_exp,
            "Predicted": solution,
            "Predicted SEM": sol_sem,
            "Cost": solution_cost,
        }
    )


def closest_pka(
    experimental_pkas: np.ndarray,
    predicted_pkas: np.ndarray,
    cost_function: Callable[[float, float], float],
) -> pd.DataFrame:
    """Find the closest match-ups between experiment and prediction.    

    Parameters
    ----------
    experimental_pkas - 1D array of experimental pKa values
    predicted_pkas - 1D array of predicted pKa values
    cost_function - function to calculate the cost of any invidual mapping.

    Notes
    -----

    The algorithm
    1. ) construct a cost matrix, matching up each experiment (rows0) and each prediction (columns) and calculating the cost.
    2. ) Find the minimum value in the matrix, add match to matches
    3. ) Remove row, column
    4. ) Repeat 2-3 until matrix has size 0
    5. ) Any remaing pKa values are mapped to NaN

    """

    experiment_copy = deepcopy(experimental_pkas["pKa"].values)
    predicted_copy = deepcopy(predicted_pkas["pKa"].values)
    sem_experiment_copy = deepcopy(experimental_pkas["SEM"].values)
    sem_predicted_copy = deepcopy(predicted_pkas["SEM"].values)

    n_experimental = experiment_copy.size
    n_predicted = predicted_copy.size
    cost_matrix = np.empty([n_experimental, n_predicted])
    for i in range(n_experimental):
        experimental_pka = experiment_copy[i]
        for j in range(n_predicted):
            predicted_pka = predicted_copy[j]
            cost_matrix[i, j] = cost_function(experimental_pka, predicted_pka)

    df = pd.DataFrame(
        columns=[
            "Experimental",
            "Experimental SEM",
            "Predicted",
            "Predicted SEM",
            "Cost",
        ]
    )

    # Continue
    while cost_matrix.size > 0:
        # Find the best match
        match = np.nanargmin(cost_matrix)
        # divide gives row, mod gives col
        row, col = divmod(match, cost_matrix.shape[1])
        experimental_pka = experiment_copy[row]
        predicted_pka = predicted_copy[col]
        experimental_sem = sem_experiment_copy[row]
        predicted_sem = sem_predicted_copy[col]
        cost = cost_matrix[row, col]
        df = df.append(
            {
                "Experimental": experimental_pka,
                "Experimental SEM": experimental_sem,
                "Predicted": predicted_pka,
                "Predicted SEM": predicted_sem,
                "Cost": cost,
            },
            ignore_index=True,
        )
        experiment_copy = np.delete(experiment_copy, row)
        predicted_copy = np.delete(predicted_copy, col)
        sem_experiment_copy = np.delete(sem_experiment_copy, row)
        sem_predicted_copy = np.delete(sem_predicted_copy, col)
        cost_matrix = np.delete(cost_matrix, row, 0)
        cost_matrix = np.delete(cost_matrix, col, 1)

    # return any left over as nan match and calculate cost based on distance to 0.0, or 14.0
    for leftover_pka, leftover_sem in zip(experiment_copy, sem_experiment_copy):
        df = df.append(
            {
                "Experimental": leftover_pka,
                "Experimental SEM": leftover_sem,
                "Predicted": np.nan,
                "Predicted SEM": np.nan,
                "Cost": cost_function(
                    *dynamic_range_solve(leftover_pka, np.nan, 0.0, 14.0)
                ),
            },
            ignore_index=True,
        )
    for leftover_pka, leftover_sem in zip(predicted_copy, sem_predicted_copy):
        df = df.append(
            {
                "Experimental": np.nan,
                "Experimental SEM": np.nan,
                "Predicted": leftover_pka,
                "Predicted SEM": leftover_sem,
                "Cost": cost_function(
                    *dynamic_range_solve(leftover_pka, np.nan, 0.0, 14.0)
                ),
            },
            ignore_index=True,
        )

    return df


def remove_unmatched_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    """Remove predicted pKas that have no experimental match."""
    return matches[matches.Experimental != np.nan]


def add_reverse_equilibrium_arrows(graph: nx.DiGraph):
    """Adds the pKa for the reverse direction of the equilibrium to the graph.
    
    Parameters
    ----------
    graph - a pKa equilibrium represented as a networkx directional graph (DiGraph)

    Notes
    -----
    Does not modify the graph in place.
    You won't be able to call a topological sort on the resulting graph.

    Returns
    -------
    Equilibrium graph with two directions
    """
    graph_copy = deepcopy(graph)

    for from_state, to_state in graph.edges:
        # Retrieve the edge properties
        props = deepcopy(graph.edges[from_state, to_state])
        # invert the Ka
        props["pKa"] = -props["pKa"]

        # Add edge in reverse order
        graph_copy.add_edge(to_state, from_state, **props)

    return graph_copy


def add_Ka_equil_graph(graph: nx.DiGraph, inplace=True):
    """Transform pKa into Ka on graph.


    Parameters
    ----------
    graph - a pKa equilibrium represented as a networkx directional graph (DiGraph)
    inplace - modifies graph in place

    Notes
    -----
    Does not modify the graph in place.

    Returns
    -------
    Equilibrium graph with Ka values
    """
    if not inplace:
        graph = deepcopy(graph)

    for from_state, to_state in graph.edges:
        # Retrieve the edge properties

        props = graph.edges[from_state, to_state]
        # invert the Ka
        props["Ka"] = math.pow(10, -props["pKa"])
        props["Kainv"] = 1.0 / math.pow(10, -props["pKa"])
        props["pKa7"] = abs(props["pKa"] - 7.0)  # closest to 7

    return graph


class TitrationCurve:
    """The representation of a titration curve of multiple protonation states over a range of pH values.

    Attributes
    ----------
    free_energies - 2D array, indexed by [state, pH] of relative free energies of titration states at a given pH
    populations - 2D array, indexed by [state, pH] of relative free energies of titration states at a given pH
    ph_values - 1D array of pH values corresponding to the pH of the other arrays
    state_ids - 1D array of identifiers for each state
    mean_charge - 2D array indexed by [state, pH] of the mean molecular charge at each pH value (relative to state 0)
    charge - 1D array of the charge assigned to each state
    graph - nx.DiGraph for micropKa derived titration curves with the one-directional network of titration states.
    augumented_graph - nx.DiGraph for micropKa derived titration curves with bidirectional network of titration states.
    pka_paths - List[List[str]] For each state, the pKa values used to derive the free energy. 
    """

    def __init__(self):
        """Instantiate a bare titration curve."""

        self.free_energies = None
        self.populations = None
        self.ph_values = None
        self.state_ids = None
        self.mean_charge = None
        self.charges = None
        self.pka_paths = None
        self.augmented_graph = None
        self.graph = None

    def _update_charges_from_file(self, charge_file, charge_header=0):
        """"""
        # Charge data from standard file
        charges = pd.read_csv(charge_file, header=charge_header)
        for idx, state in enumerate(self.state_ids):
            charge = charges[charges["Microstate ID"] == state]["Charge"].iloc[0]
            self.charges[idx] = charge

        self.mean_charge = self.charges @ self.populations
        return

    def _update_charges_from_dict(self, charge_dict: Dict[str, int]):
        """Update charge data from a dict of state: charge."""
        for idx, state in enumerate(self.state_ids):
            self.charges[idx] = charge_dict[state]
        self.mean_charge = self.charges @ self.populations

    def _override_charges(self, charges: np.ndarray):
        """Set the charges from an array, and update the mean charge"""
        self.charges = charges
        self.mean_charge = self.charges @ self.populations

        return

    def _set_free_energy_reference_state(self, state_index: int):
        """Correct the relative free energy with respect to one state."""
        self.free_energies -= np.squeeze(self.free_energies[state_index])

    def _pick_zero_charge_ref_state(self, fallback=True):
        """Find the lowest free energy state with charge equals 0, and set that as reference."""
        zeros = np.atleast_1d(np.squeeze(np.where(self.charges == 0)))

        if zeros.size < 1:
            return
        min_free = 1.0e16  # large number
        final_zero = None
        updated = False
        for zero in zeros:
            # If state isn't present/populated, all around cant use it as a reference
            if (
                np.isnan(self.free_energies[zero]).any()
                or np.isinf(self.free_energies[zero]).any()
            ):
                continue
            # Just check the first free energy for each state, since supposedly all parallel
            free_ene = deepcopy(self.free_energies[zero, 0])
            if free_ene < min_free:
                min_free = free_ene
                final_zero = zero
                updated = True

        if updated:
            self._set_free_energy_reference_state(final_zero)
        # attempt using a charged state as reference.
        elif fallback:
            warnings.warn(
                "No contiguous neutral state present. Using a charged state as reference.",
                RuntimeWarning,
            )
            min_free = 1.0e16  # large number
            final_zero = None
            updated = False
            for idx in range(self.free_energies.shape[0]):
                # If state isn't present/populated, all around cant use it as a reference
                if (
                    np.isnan(self.free_energies[idx]).any()
                    or np.isinf(self.free_energies[idx]).any()
                ):
                    continue
                # Just check the first free energy for each state, since supposedly all parallel
                free_ene = deepcopy(self.free_energies[idx, 0])
                if free_ene < min_free:
                    min_free = free_ene
                    final_zero = idx
                    updated = True
            if updated:
                self._set_free_energy_reference_state(final_zero)
            else:
                warnings.warn("Could not find any contiguous state.", RuntimeWarning)
        else:
            warnings.warn(
                "Could not find an appropriate neutral reference state.", RuntimeWarning
            )

    @classmethod
    def from_micro_pkas(cls, micropkas: np.ndarray, ph_values: np.ndarray):
        """Instantiate a titration curve specified using independent site micro pKas

        Parameters
        ----------        
        micropkas - 1D-array of float pKa values        
        ph_values - 1D-array of pH values that correspond to the curve
        """
        # Sort for convenience
        micropkas.sort()
        instance = cls()
        energies: List[np.ndarray] = list()
        state_ids: List[str] = list()
        nbound: List[int] = list()

        # If we assume every pKa is an independent proton,
        # then the number of possible states is the powerset
        # of each equilibrium.
        for included_pks in powerset(micropkas):
            bound_protons = len(included_pks)
            # Free energy according to Ullmann (2003)
            energies.append(
                free_energy_from_pka(bound_protons, np.sum(included_pks), ph_values)
            )
            # Identifier for each state
            state_ids.append("+".join(["{:.2f}".format(pk) for pk in included_pks]))
            nbound.append(bound_protons)
        instance.free_energies = np.asarray(energies)
        instance.populations = populations_from_free_energies(instance.free_energies)
        instance.ph_values = ph_values
        instance.state_ids = state_ids
        instance.charges = np.asarray(nbound)
        instance.mean_charge = instance.charges @ instance.populations
        # Set lowest value to 0
        instance.mean_charge -= int(round(min(instance.mean_charge)))

        return instance

    @classmethod
    def from_equilibrium_graph(cls, graph: nx.DiGraph, ph_values: np.ndarray):
        """Instantiate a titration curve specified using microequilibrium pKas

        Parameters
        ----------        
        graph - a directed graph of microequilibria.            
        ph_values - 1D-array of pH values that correspond to the curve

        Notes
        -----
        In the equilibrium graph every node is a state.
        Edges between states have a pKa associated with them. 
        Every edge has direction from Deprotonated -> Protonated.

        Can be generated using `micro_pKas_to_equilibrium_graph`
        """

        # First state has least protons bound, set to 0 to be reference to other states
        reference = list(nx.algorithms.dag.dag_longest_path(graph))[0]
        all_nodes = list(deepcopy(graph.nodes))
        all_nodes.remove(reference)
        all_nodes.insert(0, reference)
        augmented_graph = add_reverse_equilibrium_arrows(graph)
        augmented_graph = add_Ka_equil_graph(augmented_graph)
        pka_paths: List[List[str]] = [[reference]]
        instance = cls()
        instance.augmented_graph = augmented_graph
        energies: List[np.ndarray] = list()
    
        nbound: List[int] = [0]

        energies.append(free_energy_from_pka(0, 0.0, ph_values))

        # Every node is a state
        for s, state in enumerate(all_nodes[1:], start=1):
            # Least number of equilibria to pass through to reach a state
            # If there are more than one path, the shortest one is the one that uses pKas closer to 7
            # Which should be the most relevant range, and likely the applicable range of most techniques
            path = nx.shortest_path(
                augmented_graph, reference, all_nodes[s], weight="pKa7"
            )
            # The number of protons is equal to the number of equilibria traversed
            bound_protons = len(path) - 1
            sumpKa = 0
            # Add pKa along edges of the path
            for edge in range(bound_protons):
                sumpKa += augmented_graph[path[edge]][path[edge + 1]]["pKa"]
                # For reverse paths, deduct one proton
                if not graph.has_edge(path[edge], path[edge + 1]):
                    bound_protons -= 1

            # Free energy calculated according to Ullmann (2003).
            energies.append(free_energy_from_pka(bound_protons, sumpKa, ph_values))
            nbound.append(bound_protons)
            pka_paths.append(path)

        instance.free_energies = np.asarray(energies)
        instance.populations = populations_from_free_energies(instance.free_energies)
        instance.ph_values = ph_values
        instance.state_ids = all_nodes
        instance.charges = np.asarray(nbound)
        instance.mean_charge = instance.charges @ instance.populations
        # Set lowest value to 0
        instance.mean_charge -= int(round(min(instance.mean_charge)))
        instance.pka_paths = pka_paths

        return instance

    @classmethod
    def from_macro_pkas(cls, macropkas: np.ndarray, ph_values: np.ndarray):
        """Instantiate a titration curve specified using pKas

        Parameters
        ----------

        macropkas - 1D-array of float pKa values        
        ph_values - 1D-array of pH values that correspond to the curve
        """
        macropkas.sort()
        instance = cls()
        instance.pkas = macropkas
        instance.sems = np.zeros_like(macropkas)
        instance.populations = populations_from_macro_pka(macropkas, ph_values)
        instance.free_energies = free_energy_from_population(instance.populations)

        instance.ph_values = ph_values
        state_ids: List[str] = ["Deprotonated"]
        nbound: List[int] = [0]
        for n, pKa in enumerate(macropkas, start=1):
            state_ids.append(f"+{n:d} protons (pKa={pKa:.2f})")
            nbound.append(n)
        instance.state_ids = state_ids
        instance.charges = np.asarray(nbound)
        instance.mean_charge = instance.charges @ instance.populations
        # Set lowest value to 0
        instance.mean_charge -= int(round(min(instance.mean_charge)))

        return instance

    @classmethod
    def from_populations(
        cls,
        populations: np.ndarray,
        ph_values: np.ndarray,
        nbound: np.ndarray,
        state_ids: Optional[List[str]] = None,
    ):
        """Instantiate a TitrationCurve from individual state populations.

        Parameters
        ---------

        populations - 2D-array [state,pH] of populations of each state, at a given pH value
        pH values - 1D array of pH values that correspond to the population curves provided.
        nbound - 1D array of int values that include the relative number of protons bound to each state
        state_ids - 1D list of str values
        """
        if not populations.shape[1] == ph_values.size:
            raise ValueError(
                "The second dim of populations and the size of ph_values need to match."
            )

        if state_ids is not None:
            num_ids = len(state_ids)
            num_states = populations.shape[0]
            if num_ids != num_states:
                raise ValueError(
                    f"Number of state identifiers ({num_ids}) and number of states ({num_states}) don't match."
                )

        instance = cls()
        instance.state_ids = state_ids
        instance.populations = populations
        instance.free_energies = -np.log(populations)  # unnormalized
        instance.ph_values = ph_values
        instance.charges = np.asarray(nbound)
        instance.mean_charge = instance.charges @ instance.populations
        # Set lowest value to 0
        instance.mean_charge -= int(round(min(instance.mean_charge)))

        return instance

    def align_mean_charge(
        self,
        other_curve,
        distance_function: Callable[[np.ndarray, np.ndarray, Optional[Any]], float],
        *args,
    ) -> None:
        """Find the offset between the mean charge of this curve and the other that produces the closest match and shifts this mean charge curve by that offset.
        
        Parameters
        ----------
        self - this titration curve object.
        other_curve - the second titration curve object
        distance_functions - function that takes two curves and returns the distance.
        args - additional positional arguments to pass into the distance function.
        Notes
        -----
        Modifies this curve in place.

        Returns
        -------
        None    
        """

        # Grab charge curves, and ensure lowest value is 0.
        q1 = deepcopy(other_curve.mean_charge)
        q1 -= int(round(min(q1)))
        q2 = deepcopy(self.mean_charge)
        q2 -= int(round(min(q2)))

        max1 = int(round(max(q1)))
        max2 = int(round(max(q2)))

        # Maximum range of the alignment is from -max, max
        m = max([max1, max2])
        distance = 1.0e16
        aligned_q2 = deepcopy(q2)
        offset = 0
        for i in range(-m, m + 1):
            new_q2 = q2 + i
            new_distance = distance_function(other_curve.mean_charge, new_q2, *args)
            if new_distance < distance:
                distance = new_distance
                aligned_q2 = new_q2
                offset = i

        self.mean_charge = aligned_q2
