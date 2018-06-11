"""This module contains classes and functions specific to SAMPL6 data files"""
import pandas as pd
import numpy as np
from .titrato import TitrationCurve, free_energy_from_population
from .titrato import data_dir
from .stats import area_between_curves
from networkx import DiGraph
import warnings
import os
import matplotlib
from matplotlib import pyplot as plt 
from typing import Tuple, Optional, List


def get_typei_pka_data(
    molecule_name: str, datafile: str, header: Optional[int] = 0
) -> Tuple[DiGraph, pd.DataFrame]:
    """Retrieve type I pka data for a single molecule from the datafile.
    
    Parameters
    ----------
    molecule_name - SAMPL6 identifier of the molecule.
    datafile - location of csv file in type I format (micropKa)
    header - optional, which lines are header lines, set to None for file without headers

    Returns
    -------
    graph of states connected by pKa, dataframe of all pKa values.    
    """
    df = pd.read_csv(datafile, header=header)
    # Override column names
    df.columns = ["Protonated", "Deprotonated", "pKa", "SEM"]
    df["Molecule"] = df["Protonated"].apply(lambda string: string.split("_")[0])
    mol_frame = df[df["Molecule"] == molecule_name]

    return mol_frame


def create_graph_from_typei_df(mol_frame):
    """Create a graph from a typei dataframe for a single molecule."""
    # Direction of edges of the graph is deprotonated -> protonated state
    from_list = list(mol_frame["Deprotonated"])
    to_list = list(mol_frame["Protonated"])
    # Add properties
    properties = [
        dict(pKa=row["pKa"], SEM=row["SEM"]) for i, row in mol_frame.iterrows()
    ]
    graph = DiGraph()
    graph.add_edges_from(zip(from_list, to_list, properties))
    return graph


def get_typeii_logp_data(
    molecule_name: str,
    datafile,
    charge_file: str,
    header: Optional[int] = 0,
    charge_header: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve type II log population data for a single molecule from the datafile.
    
    Parameters
    ----------
    molecule_name - SAMPL6 identifier of the molecule.
    datafile - location of csv file in type II format (microstate log populations)
    charge_file - location of csv file with charges for each state. Example format:
        Molecule,Microstate ID,Charge
        SM01,SM01_micro001,1
        SM01,SM01_micro002,-2
        SM01,SM01_micro004,-1
        .. .. .. et cetera
    header - optional, which lines are header lines, set to None for file without headers
    charge_header - optional, which lines are header lines in the charge file, set to None for file without headers

    Returns
    -------
    Dataframe with populations, dataframe with charges
    
    """
    df = pd.read_csv(datafile, header=header)
    charges = pd.read_csv(charge_file, header=charge_header)
    colnames = list(df.columns)
    colnames[0] = "Microstate ID"
    df.columns = colnames
    df["Molecule"] = df["Microstate ID"].apply(lambda id: id.split("_")[0])
    return (
        df[df["Molecule"] == molecule_name],
        charges[charges["Molecule"] == molecule_name],
    )


def get_typeiii_pka_data(molecule_name: str, datafile: str, header: Optional[int] = 0):
    """Retrieve type III macroscopic pKa data for a single molecule from the data file
    
    Parameters
    ----------
    molecule_name - SAMPL6 identifier of the molecule.
    datafile - location of csv file in type III format (macropKa)
    header - optional, which lines are header lines, set to None for file without headers

    Returns
    -------
    graph of states connected by pKa, dataframe of all pKa values.  """
    df = pd.read_csv(datafile, header=header)
    # Override column names
    df.columns = ["Molecule", "pKa", "SEM"]
    return df[df["Molecule"] == molecule_name]


def get_experimental_pKa_data(
    molecule_name: str,
    datafile: str = os.path.join(data_dir, "SAMPL6_experimental_pkas.csv"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve experimental pKa values, and errors from the experimental csv file."""
    df = pd.read_csv(datafile)

    pKas = list()
    sems = list()
    # Should match only one row, but have to grab the first entry
    mol_match = df[df["Molecule ID"] == molecule_name].iloc[0]
    for x in range(1, 4):
        pKas.append(mol_match[f"pKa{x} mean"])
        sems.append(mol_match[f"pKa{x} SEM"])

    pKas = np.asarray(pKas)
    sems = np.asarray(sems)
    mask = np.isnan(pKas)
    pKas = pKas[~mask]
    sems = sems[~mask]
    new_df = pd.DataFrame.from_records(dict(pKa=pKas, SEM=sems))
    new_df["Molecule"] = molecule_name

    return new_df[["Molecule", "pKa", "SEM"]]


class TypeIPrediction(TitrationCurve):
    """Representation of a Type I (micropKa) prediction for SAMPL6"""

    ph_range = np.linspace(2, 12, num=101)

    def __init__(self):
        super(TypeIPrediction, self).__init__()
        self.pkas = None
        self.sems = None

        return

    @classmethod
    def from_id(
        cls,
        mol_id: str,
        datafile: str,
        header: int = 0,
        drop_nodes: Optional[List[str]] = None,
    ):
        """Retrieve the titration curve for one molecule from typeI predicted micropKas.
        
        Parameters
        ----------
        mol_id - the SAMPL6 identifier for this molecule
        datafile - source of the type I pKa values as a csv file
        header - integer index for the header, set to None if no header
        drop_nodes - drop these states from generating the graph.        
        """
        data = get_typei_pka_data(mol_id, datafile, header)
        graph = create_graph_from_typei_df(data)

        # Drop any requested nodes.
        if drop_nodes is not None:
            for node in drop_nodes:
                graph.remove_node(node)
        micropKas = np.asarray(data["pKa"])
        sems = np.asarray(data["SEM"])

        instance = cls.from_equilibrium_graph(graph, cls.ph_range)
        # Store data for reference
        instance.pkas = micropKas
        instance.sems = sems
        return instance

    @classmethod
    def bootstrap_from_id(
        cls,
        mol_id: str,
        datafile: str,
        n_samples: int,
        n_bootstrap: int,
        header: int = 0,
        drop_nodes: Optional[List[str]] = None,
    ) -> Tuple:
        """Retrieve the titration curve for one molecule from typeI predicted micropKas.
        
        Parameters
        ----------
        mol_id - the SAMPL6 identifier for this molecule
        datafile - source of the type I pKa values as a csv file
        header - integer index for the header, set to None if no header
        drop_nodes - drop these states from generating the graph.        
        n_samples - the number of samples over which the SEM was determined
        n_bootstrap - number of curves to return.

        Returns
        -------
        original curve, list of bootstrap curves
        """
        data = get_typei_pka_data(mol_id, datafile, header)

        graph = create_graph_from_typei_df(data)
        # Drop any requested nodes.
        if drop_nodes is not None:
            for node in drop_nodes:
                graph.remove_node(node)

        instances = list()
        for bootstrap_sample in range(n_bootstrap):
            bootstrap_copy = data.copy()
            bootstrap_copy["pKa"] = bootstrap_copy.apply(lambda row: np.random.normal(row["pKa"], row["SEM"] * np.sqrt(n_samples)), axis=1,)
            bootstrap_graph = create_graph_from_typei_df(bootstrap_copy)
            # Drop any requested nodes.
            if drop_nodes is not None:
                for node in drop_nodes:
                    bootstrap_graph.remove_node(node)
            instances.append(cls.from_equilibrium_graph(bootstrap_graph, cls.ph_range))

        micropKas = np.asarray(data["pKa"])
        sems = np.asarray(data["SEM"])

        instance = cls.from_equilibrium_graph(graph, cls.ph_range)
        # Store data for reference
        instance.pkas = micropKas
        instance.sems = sems
        return instance, instances


class TypeIIPrediction(TitrationCurve):
    """Representation of a Type II (microstate log population) prediction for SAMPL6"""

    ph_range = np.linspace(2, 12, num=101)

    def __init__(self):
        super(TypeIIPrediction, self).__init__()
        return

    @classmethod
    def from_id(
        cls,
        molecule_name: str,
        datafile: str,
        charge_file: str,
        header=0,
        charge_header=0,
    ):
        """Instantiate a titration curve for one molecule from Type II predicted log populations."""
        data, charges = get_typeii_logp_data(
            molecule_name,
            datafile,
            charge_file,
            header=header,
            charge_header=charge_header,
        )
        state_ids = data["Microstate ID"]
        nbound = [
            int(charges.loc[charges["Microstate ID"] == id, "Charge"])
            for id in state_ids
        ]
        log_pop = data.iloc[:, 1:-1].values
        pop = np.exp(np.asarray(log_pop))
        # normalize
        pop /= np.sum(pop, axis=0)[None, :]
        instance = cls.from_populations(pop, cls.ph_range, nbound, state_ids)
        return instance


class TypeIIIPrediction(TitrationCurve):
    """Representation of a Type III (macropKa) prediction for SAMPL6."""

    ph_range = np.linspace(2, 12, num=101)

    def __init__(self):

        super(TypeIIIPrediction, self).__init__()
        self.pkas = None
        self.sems = None

        return

    @classmethod
    def from_id(cls, mol_id: str, datafile: str, header: int = 0):
        """Retrieve the titration curve for one molecule from typeIII predicted macropKas.

        Parameters
        ----------
        mol_id - the identifier for the molecule, e.g. "SM01".
        datafile - location to take type III data from.            
        header - index of the header line in the csv file.

        Notes
        -----
        Titration curves are defined over a pH range of 2-12 with intervals of 0.1 pH unit.

        """
        data = get_typeiii_pka_data(mol_id, datafile, header)
        macropKas = np.asarray(data["pKa"])
        sems = np.asarray(data["SEM"])
        instance = cls.from_macro_pkas(macropKas, cls.ph_range)
        # Store data for reference
        instance.pkas = macropKas
        instance.sems = sems
        return instance

    @classmethod
    def bootstrap_from_id(cls, mol_id: str, datafile, n_bootstrap, n_samples,header: int = 0):
        """
        Retrieve the titration curve for one molecule from typeIII predicted macropKas.

        Parameters
        ----------
        mol_id - the identifier for the molecule, e.g. "SM01".
        datafile - location to take type III data from.
        n_bootstrap - number of curves to return.
        n_samples - the number of samples over which the SEM was determined
        header - index of the header line in the csv file.    

        """
        data = get_typeiii_pka_data(mol_id, datafile, header)

        instances = list()
        for bootstrap_sample in range(n_bootstrap):
            bootstrap_copy = data.copy()
            bootstrap_copy["pKa"] = bootstrap_copy.apply(lambda row: np.random.normal(row["pKa"], row["SEM"] * np.sqrt(n_samples)), axis=1,)
            instances.append(cls.from_macro_pkas(np.asarray(bootstrap_copy["pKa"]), cls.ph_range))
        # Store data for reference

        macropKas = np.asarray(data["pKa"])
        sems = np.asarray(data["SEM"])
        instance = cls.from_macro_pkas(macropKas, cls.ph_range)
        # Store data for reference
        instance.pkas = macropKas
        instance.sems = sems
        return instance, instances


class SAMPL6Experiment(TitrationCurve):
    """Class to represent a Sirius T3 experimental titration curve from the SAMPL6 dataset."""

    # Experiments by Mehtap Isik, 2018
    experimental_data_file = os.path.join(data_dir, "SAMPL6_experimental_pkas.csv")
    ph_range = np.linspace(2, 12, num=101)

    def __init__(self):

        super(SAMPL6Experiment, self).__init__()
        self.pkas = None
        self.sems = None

    @classmethod
    def from_id(cls, mol_id: str, datafile: Optional[str] = None):
        """Retrieve the titration curve for one molecule from the experiment.

        Parameters
        ----------
        mol_id - the identifier for the molecule, e.g. "SM01".
        datafile - optional, location to take experimental data from. 
            Uses the file "experimental_pkas.csv" by default.

        Notes
        -----
        The experiments are defined over a pH range of 2-12.

        """
        # Use built in file for convenience
        if datafile is None:
            datafile = cls.experimental_data_file
        data = get_experimental_pKa_data(mol_id, datafile)
        macropKas = np.asarray(data["pKa"])
        sems = np.asarray(data["SEM"])
        instance = cls.from_macro_pkas(macropKas, cls.ph_range)
        # Store data for reference
        instance.pkas = macropKas
        instance.sems = sems
        return instance

    @classmethod
    def bootstrap_from_id(cls, mol_id: str, n_bootstrap, n_samples=3, datafile: Optional[str] = None):
        """Retrieve the titration curve for one molecule from the experiment.

        Parameters
        ----------
        mol_id - the identifier for the molecule, e.g. "SM01".
        datafile - optional, location to take experimental data from. 
            Uses the file "experimental_pkas.csv" by default.
        n_bootstrap - number of bootstrap samples to generate
        n_samples - number of samples used to determine SEM (was three for the data set)
        
        Notes
        -----
        The experiments are defined over a pH range of 2-12.

        """
        # Use built in file for convenience
        if datafile is None:
            datafile = cls.experimental_data_file
        data = get_experimental_pKa_data(mol_id, datafile)
        
        instances = list()
        for bootstrap_sample in range(n_bootstrap):
            bootstrap_copy = data.copy()
            bootstrap_copy["pKa"] = bootstrap_copy.apply(lambda row: np.random.normal(row["pKa"], row["SEM"] * np.sqrt(n_samples)), axis=1,)
            instances.append(cls.from_macro_pkas(np.asarray(bootstrap_copy["pKa"]), cls.ph_range))
        # Store data for reference
        
        macropKas = np.asarray(data["pKa"])
        sems = np.asarray(data["SEM"])
        instance = cls.from_macro_pkas(macropKas, cls.ph_range)
        instance.pkas = macropKas
        instance.sems = sems
        return instance, instances

    def add_unobserved_state(self):
        """Adds a new, unvisited state to the system.         

        Note
        ----

        This hypothetical state can be useful for modeling purposes, as it provides a point to match any unmatched prediction to this state.
        """
        # Assumed all states are at 0 population.
        new_state_population = np.zeros(self.populations.shape[1], dtype=float)
        self.populations = np.vstack((self.populations, new_state_population))
        # Recalculate free energies for consistency.
        # note these will be taking the log of 0, so it will give a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.free_energies = free_energy_from_population(self.populations)
        self.state_ids.append("Unobserved")

def get_percentiles(array, percentiles):
    nums = list()
    for q in percentiles:
        nums.append(np.percentile(array, q, axis=0))
    return nums

def plot_quantiles(curves: np.ndarray, ph_range:np.ndarray, color:str , perc: float=5, fill=True):
    """Plot the median, and outer percentiles.
    
    Parameters
    ----------
    curves - 2D array of bootstrap titration curves, with the 0 axis being the different curves, anx the 1 axis the pH values.
    ph_range - the ph values that each point corresponds to.
    color - a matplotlib color for the elements in the plot
    perc - percentile, and 100-percentile to plot 
        default 5, so 5th and 95th are plotted.
    fill - fill the area between percentiles with color.
    """
    quantiles = get_percentiles(curves, [50., perc, 100.-perc])               
    plt.plot(ph_range, quantiles[0], '-', color=color,alpha=1.0, label="median")
    plt.plot(ph_range, quantiles[1], ':', color=color,alpha=1.0, label="{:.0f}th/{:.0f}th percentile".format(perc, 100-perc))
    plt.plot(ph_range, quantiles[2], ':', color=color,alpha=1.0)
    if fill:
        plt.fill_between(ph_range, quantiles[2],quantiles[1], facecolor=color, alpha=0.1)

def plot_mean_twosigma(curves:np.ndarray, ph_range:np.ndarray, color: str, fill=True):
    """Plot the mean, plus/minus 2 sigma.
    
    Parameters
    ----------
    curves - 2D array of bootstrap titration curves, with the 0 axis being the different curves, anx the 1 axis the pH values.
    ph_range - the ph values that each point corresponds to.
    color - a matplotlib color for the elements in the plot    
    fill - fill the area between +/- 2 sigma with color.
    """
    mean = np.mean(curves, axis=0)
    std = np.std(curves,axis=0)
    plt.plot(ph_range, mean, '-', color=color, label="mean")
    plt.plot(ph_range, mean+2*std,':', alpha=1.0, color=color, label=r"$\pm$2$\sigma$")
    plt.plot(ph_range, mean-2*std, ':', alpha=1.0, color=color)
    if fill:
        plt.fill_between(ph_range, mean+2*std, mean-2*std, facecolor=color, alpha=0.1)

def plot_subset(curves, ph_range, n_choices: int, color='gray', alpha=0.1):
    """Plot a subset of bootstrap samples.    
    
    Parameters
    ----------
    curves - 2D array of bootstrap titration curves, with the 0 axis being the different curves, anx the 1 axis the pH values.
    ph_range - the ph values that each point corresponds to.
    n_choices - number of samples to plot
    color - a matplotlib color for the elements in the plot
    alpha - transparency of the curves.

    """
    choices = np.random.choice(curves.shape[0] ,n_choices, replace=False)
    for i in choices:
        plt.plot(ph_range, curves[i], '-', color=color, zorder=0, alpha=alpha)

def bootstrap_comparison(molecule:str, prediction_file:str, datatype:str, n_samples=1, n_bootstrap=1000, **kwargs):
    """Perform a bootstrap analysis on the experimental and the computed titration curve.
    
    Parameters
    ----------
    molecule - SAMPL6 identifier of the molecule.
    prediction_file - file name containing the computed pKa values.
    datatype - typeI or typeIII, (type II doesnt have error bars so we cant bootstrap)
    n_samples - number of samples used to determine the standard error.
    n_bootstrap - number of bootstrap samples to draw.
    """
    
    if datatype == "typeI":
        predicted_curve, strapped_curves = TypeIPrediction.bootstrap_from_id(molecule, prediction_file, n_samples, n_bootstrap, **kwargs)
    elif datatype == "typeIII":
        predicted_curve, strapped_curves = TypeIIIPrediction.bootstrap_from_id(molecule, prediction_file, n_samples, n_bootstrap, **kwargs)
    
    experimental_curve, exp_strapped_curves = SAMPL6Experiment.bootstrap_from_id(molecule, n_bootstrap)
    df = pd.DataFrame(columns=["Molecule", "Δ"])
    predicted_curve.align_mean_charge(experimental_curve, area_between_curves,0.1)    
    for i, (curve, exp_curve) in enumerate(zip(strapped_curves, exp_strapped_curves)):
        curve.align_mean_charge(exp_curve, area_between_curves, 0.1)
        Δ = area_between_curves(curve.mean_charge, exp_curve.mean_charge, 0.1)
        df.loc[i] = [molecule, Δ]

    return df
