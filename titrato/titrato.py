"""
titrato.py
Utilities for calculating titration curves.
"""

import numpy as np
import pandas as pd
import holoviews as hv
import logging
from itertools import chain, combinations
from typing import Dict, List, Iterator, Union, Iterable, Tuple, Any, Optional, Callable
import networkx as nx
import matplotlib.pyplot as plt
import os
import warnings
from . import data_dir
from scipy.optimize import linear_sum_assignment
import math
from copy import deepcopy

logger = logging.getLogger()

# Constant factor for transforming a number that is log base 10 to log base e
ln10 = np.log(10)

# Keep track of papers that we take equations from
references = [
    "RI Allen et al J Pharm Biomed Anal 17 (1998) 699-712", "Ullmann, J Phys Chem B vol 107, No 5, 2003. 1263-1271", "Mehtap Isik et al, to be submitted to JCAMD 2018"]


def quicksave(plot_object: Any, outfile_basename: str, renderer: str='matplotlib', fmt: Optional[str]=None) -> None:
    """Quicksave function for a holoviews plot to a standalone html file.

    Parameters
    ----------
    plot_object - a holoviews layout object    
    outfile_basename - The base name of the output file.
    renderer - name of a holoviews renderer, e.g 'bokeh', 'matplotlib', or 'plotly'
    """
    hv.renderer(renderer).save(plot_object, outfile_basename, fmt=fmt)

    return


def graph_to_axes(graph: Union[nx.DiGraph, nx.Graph], ax: Optional[plt.Axes]=None) -> plt.Axes:
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
    pos = nx.spring_layout(graph)
    if ax is None:
        ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, ax=ax)
    nx.draw_networkx_edge_labels(
        graph, pos, font_size=10, font_family='sans-serif', ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=5000,
                           node_color='#E5e5e5', alpha=0.3, node_shape='s', ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10,
                            font_family='sans-serif', ax=ax)

    return ax


def free_energy_from_pka(bound_protons: int, Σ_pKa: float, pH: np.ndarray) -> np.ndarray:
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
    return (nμ+G) * ln10


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
    return probability/Z


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


def macro_pkas_to_species_concentration(pkas: np.ndarray, ph: float, total_concentration: float) -> np.ndarray:
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
        row[row_num+1] = -H
        rows.append(row)

    system_of_equations = np.matrix(rows)
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
        Note: This function is not compatible with arrays of pH values    

    """
    # create array in memory
    populations = np.empty([pkas.size+1, phvalues.size])

    # Set concentration for each state for each pH
    for col, ph in enumerate(phvalues):
        # The initial concentrations is the is set to 1, so that all concentrations are fractional
        populations[:, col] = macro_pkas_to_species_concentration(pkas, ph, 1)

    return populations


def free_energy_from_population(populations: np.ndarray) -> np.ndarray:
    """Calculate relative free energies from their populations.

    Parameters    
    ----------
    populations - 2D-array of populations with axes [state,pH]

    Returns
    -------
    free_energies - 2D-array of free energies with axes [state,pH]

    Notes
    -----
    The relative free energy is defined as:
    relative dG[1..n] = - ln (Pop[1..n]) - ln (Pop)[n]
    """
    free_energies = - np.log(populations)

    return free_energies - free_energies[0]


def powerset(iterable: Iterable[float]) -> Iterator[Tuple[float, ...]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def dynamic_range_solve(x1: float,x2: float, lower:float, upper:float)-> Tuple[float, float]:
    """Backfill nan values with the lower, or upper bound of a dynamic range, based on the position of the value to be compared against."""
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

def fixed_cost_solve(x1: float,x2: float, cost:float=0.0)-> Tuple[float, float]:
    """Build in a fixed cost if one of two numbers is not defined."""
    if np.isnan(x1):        
        return x2+cost, x2        
    elif np.isnan(x2):        
        return x1,x1+cost
    else: 
        return x1,x2

def fixed_value_solve(x1: float,x2: float, value:float=0.0)-> Tuple[float, float]:

    if np.isnan(x1):        
        return value, x2        
    elif np.isnan(x2):        
        return x1, value
    else: 
        return x1,x2



def hungarian_pka(experimental_pkas: np.ndarray, predicted_pkas: np.ndarray, cost_function: Callable[[float,float], float]) -> pd.DataFrame:
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
    cost_matrix = np.zeros([size,size])
    for i in range(size):                
        for j in range(size):            
            if i < n_experimental and j < n_predicted:
                experimental_pka = experimental_pkas["pKa"].values[i]
                predicted_pka = predicted_pkas["pKa"].values[j]
                cost_matrix[i,j] = cost_function(experimental_pka, predicted_pka)           
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
    df = pd.DataFrame(columns=["Experimental", "Predicted", "Cost"])
    for i, row_id in enumerate(row_indices):
        col_id = col_indices[i]
        # Skip unmatched
        if row_id >= n_experimental:
            df = df.append({"Experimental": np.nan, "Experimental SEM" : np.nan, "Predicted": predicted_pkas["pKa"].values[col_id], "Predicted SEM": predicted_pkas["SEM"].values[col_id], "Cost": cost_matrix[row_id, col_id]}, ignore_index=True)        
        elif col_id >= n_predicted:
            df = df.append({"Experimental": experimental_pkas["pKa"].values[row_id], "Experimental SEM" : experimental_pkas["SEM"].values[row_id], "Predicted": np.nan, "Predicted SEM": np.nan, "Cost": cost_matrix[row_id, col_id]}, ignore_index=True)         
        else:
            df = df.append({"Experimental": experimental_pkas["pKa"].values[row_id], "Experimental SEM" : experimental_pkas["SEM"].values[row_id], "Predicted": predicted_pkas["pKa"].values[col_id], "Predicted SEM": predicted_pkas["SEM"].values[col_id], "Cost": cost_matrix[row_id, col_id]}, ignore_index=True)

    return df

def align_pka(experimental_pkas: np.ndarray, predicted_pkas: np.ndarray, cost_function: Callable[[float,float], float]):
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
        for e1,p1 in np.array([deepcopy(exp), deepcopy(pred)]).T:
            # if a pKa is dropped at the low end, match to 0
            # If a pKa is dropped at the high end, match to 14            
            e1,p1 = dynamic_range_solve(e1,p1, 0.0, 14.0)                            
            cost.append(cost_function(e1,p1))
        total_cost = np.sum(cost)
        if total_cost < min_cost:
            solution = deepcopy(pred)
            min_cost = total_cost
            solution_cost = cost
            sol_sem = deepcopy(sem_pred)

    return pd.DataFrame.from_dict({"Experimental":  exp, "Experimental SEM": sem_exp,  "Predicted":  solution, "Predicted SEM": sol_sem,  "Cost": solution_cost})

def closest_pka(experimental_pkas:np.ndarray, predicted_pkas:np.ndarray, cost_function:Callable[[float,float], float])-> pd.DataFrame:
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
    sem_predicted_copy =  deepcopy(predicted_pkas["SEM"].values)
    
    n_experimental = experiment_copy.size
    n_predicted = predicted_copy.size
    cost_matrix = np.empty([n_experimental,n_predicted])
    for i in range(n_experimental):        
        experimental_pka = experiment_copy[i]
        for j in range(n_predicted):
            predicted_pka = predicted_copy[j]
            cost_matrix[i,j] = cost_function(experimental_pka, predicted_pka)
    
    df = pd.DataFrame(columns=["Experimental", "Experimental SEM", "Predicted", "Predicted SEM", "Cost"])
    
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
        cost = cost_matrix[row,col]
        df = df.append({"Experimental": experimental_pka, "Experimental SEM": experimental_sem, "Predicted": predicted_pka, "Predicted SEM": predicted_sem, "Cost": cost}, ignore_index=True)
        experiment_copy = np.delete(experiment_copy, row)
        predicted_copy = np.delete(predicted_copy, col)
        sem_experiment_copy = np.delete(sem_experiment_copy, row)
        sem_predicted_copy = np.delete(sem_predicted_copy, col)
        cost_matrix = np.delete(cost_matrix,row,0)
        cost_matrix = np.delete(cost_matrix,col,1)        
        
    
    # return any left over as nan match and calculate cost based on distance to 0.0, or 14.0
    for leftover_pka, leftover_sem in zip(experiment_copy, sem_experiment_copy):
        df = df.append({"Experimental": leftover_pka, "Experimental SEM": leftover_sem, "Predicted": np.nan, "Predicted SEM": np.nan, "Cost": cost_function(*dynamic_range_solve(leftover_pka, np.nan,0.0, 14.0))}, ignore_index=True)
    for leftover_pka, leftover_sem in zip(predicted_copy, sem_predicted_copy):
        df = df.append({"Experimental": np.nan, "Experimental SEM": np.nan, "Predicted": leftover_pka, "Predicted SEM": leftover_sem, "Cost": cost_function(*dynamic_range_solve(leftover_pka, np.nan,0.0, 14.0))}, ignore_index=True)
        
    return df


def remove_unmatched_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    """Remove predicted pKas that have no experimental match."""
    return matches[matches.Experimental != np.nan]



class TitrationCurve:
    """The representation of a titration curve of multiple protonation states over a range of pH values.

    Attributes
    ----------
    free_energies - 2D array, indexed by [state, pH] of relative free energies of titration states at a given pH
    populations - 2D array, indexed by [state, pH] of relative free energies of titration states at a given pH
    ph_values - 1D array of pH values corresponding to the pH of the other arrays
    state_ids - 1D array of identifiers for each state
    mean_charge - 2D array indexed by [state, pH] of the mean molecular charge at each pH value (relative to state 0)
    nbound - 1D array of number of protons bound to each state
    """

    def __init__(self):
        """Instantiate a bare titration curve."""

        self.free_energies = None
        self.populations = None
        self.ph_values = None
        self.state_ids = None
        self.mean_charge = None
        self.nbound = None

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
            energies.append(free_energy_from_pka(
                bound_protons, np.sum(included_pks), ph_values))
            # Identifier for each state
            state_ids.append(
                "+".join(["{:.2f}".format(pk) for pk in included_pks]))
            nbound.append(bound_protons)
        instance.free_energies = np.asarray(energies)
        instance.populations = populations_from_free_energies(
            instance.free_energies)
        instance.ph_values = ph_values
        instance.state_ids = state_ids
        instance.nbound = np.asarray(nbound)
        instance.mean_charge = instance.nbound @ instance.populations

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
        # List of nodes (states). Order least->most protonated
        node_topology = list(nx.algorithms.dag.topological_sort(graph))

        instance = cls()
        energies: List[np.ndarray] = list()
        nbound: List[int] = [0]


        # First state has least protons bound, set to 0 to be reference to other states
        reference = node_topology[0]
        energies.append(free_energy_from_pka(0, 0.0, ph_values))


        # Every node is a state
        for s, state in enumerate(node_topology[1:], start=1):
            # Least number of equilibria to pass through to reach a state
            path = nx.shortest_path(graph, reference, node_topology[s])
            # The number of protons is equal to the number of equilibria traversed
            bound_protons = len(path)-1
            sumpKa = 0
            # Add pKa along edges of the path
            for edge in range(bound_protons):
                sumpKa += graph[path[edge]][path[edge+1]]['pKa']

            # Free energy calculated according to Ullmann (2003).
            energies.append(free_energy_from_pka(
                bound_protons, sumpKa, ph_values))
            nbound.append(bound_protons)

        instance.free_energies = np.asarray(energies)
        instance.populations = populations_from_free_energies(
            instance.free_energies)
        instance.ph_values = ph_values
        instance.state_ids = node_topology
        instance.nbound = np.asarray(nbound)
        instance.mean_charge = instance.nbound @ instance.populations

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
        instance.populations = populations_from_macro_pka(macropkas, ph_values)
        instance.free_energies = free_energy_from_population(
            instance.populations)
        instance.ph_values = ph_values
        state_ids: List[str] = ["Deprotonated"]
        nbound: List[int] = [0]
        for n, pKa in enumerate(macropkas, start=1):
            state_ids.append(f"+{n:d} protons (pKa={pKa:.2f})")
            nbound.append(n)
        instance.state_ids = state_ids
        instance.nbound = np.asarray(nbound)
        instance.mean_charge = instance.nbound @ instance.populations
        

        return instance

    @classmethod
    def from_populations(cls, populations: np.ndarray, ph_values: np.ndarray, nbound: np.ndarray, state_ids: Optional[List[str]]=None):
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
                "The second dim of populations and the size of ph_values need to match.")

        if state_ids is not None:
            num_ids = len(state_ids)
            num_states = populations.shape[0]
            if num_ids != num_states:
                raise ValueError(
                    f"Number of state identifiers ({num_ids}) and number of states ({num_states}) don't match.")

        instance = cls()
        instance.state_ids = state_ids
        instance.populations = populations
        instance.free_energies = free_energy_from_population(populations)
        instance.ph_values = ph_values
        instance.nbound = np.asarray(nbound)
        instance.mean_charge = instance.nbound @ instance.populations

        return instance

    def plot(self, category_name: str) -> hv.Layout:
        """Plot the titration curve as a function of pH.

        Parameters
        ----------
        category 
        """
        category_data: Optional[np.ndarray] = None
        if category_name == 'population':
            category_data = self.populations
        elif category_name == 'free_energy':
            category_data = self.free_energies
        elif category_name == 'charge':
            category_data = self.mean_charge
        else:
            raise ValueError(
                "Pick population or free_energy, or charge as category names.")

        if category_name in ["population", "free_energy"]:
            data_series = dict(pH=self.ph_values)
            state_vars = []
            for s, state in enumerate(category_data):
                if self.state_ids is None:
                    state_var = "{}".format(s+1)
                else:
                    state_var = self.state_ids[s]
                data_series[state_var] = state
                state_vars.append(state_var)

            df = pd.DataFrame.from_dict(data_series)
            df_cat = df.melt(id_vars=["pH"], value_vars=state_vars,
                            var_name="State", value_name=category_name)
            ds_cat = hv.Dataset(df_cat, vdims=category_name)
            return ds_cat.to(hv.Curve, 'pH', groupby='State').overlay()
        elif category_name in ["charge"]:
            xy = np.asarray([self.ph_values, self.mean_charge]).T
            return hv.Curve(xy, "pH", "Mean molecular charge")



