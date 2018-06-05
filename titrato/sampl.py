"""This module contains classes and functions specific to SAMPL6 data files"""
import pandas as pd
import numpy as np
from .titrato import TitrationCurve, free_energy_from_population
from .titrato import data_dir
from networkx import DiGraph
import warnings
import os
from typing import Tuple, Optional, List


def get_typei_pka_data(molecule_name: str, datafile:str, header:Optional[int]=0)-> Tuple[nx.DiGraph, pd.DataFrame]:
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
    df["Molecule"] = df["Protonated"].apply(lambda string: string.split('_')[0])    
    mol_frame = df[df['Molecule'] == molecule_name]
    
    # Direction of edges of the graph is deprotonated -> protonated state
    from_list = list(mol_frame["Deprotonated"])
    to_list = list(mol_frame["Protonated"])
    # Add properties    
    properties = [dict(pKa=row["pKa"], SEM=row["SEM"]) for i,row in mol_frame.iterrows()]
    graph = DiGraph()
    graph.add_edges_from(zip(from_list, to_list, properties))

    return graph, mol_frame

def get_typeii_logp_data(molecule_name: str, datafile, charge_file:str, header: Optional[int]=0, charge_header: Optional[int]=0)-> Tuple[pd.DataFrame, pd.DataFrame]:
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
    return df[df["Molecule"] == molecule_name], charges[charges["Molecule"] == molecule_name]
    
def get_typeiii_pka_data(molecule_name: str, datafile:str, header: Optional[int]=0):
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
    return df[df['Molecule'] == molecule_name]

def get_experimental_pKa_data(molecule_name: str, datafile: str=os.path.join(data_dir, "SAMPL6_experimental_pkas.csv")) -> Tuple[np.ndarray, np.ndarray]:
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
    sems= sems[~mask]
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
    def from_id(cls, mol_id:str, datafile:str, header:int=0, drop_nodes:Optional[List[str]]=None):
        """Retrieve the titration curve for one molecule from typeI predicted micropKas.
        
        Parameters
        ----------
        mol_id - the SAMPL6 identifier for this molecule
        datafile - source of the type I pKa values as a csv file
        header - integer index for the header, set to None if no header
        drop_nodes - drop these states from generating the graph.        
        """
        graph, data = get_typei_pka_data(mol_id, datafile, header)

        # Drop any requested nodes.
        if drop_nodes is not None:
            for node in drop_nodes:
                graph.remove_node(node)
        micropKas = np.asarray(data["pKa"])
        sems =  np.asarray(data["SEM"])
        
        instance = cls.from_equilibrium_graph(graph, cls.ph_range)
        # Store data for reference
        instance.pkas = micropKas
        instance.sems = sems
        return instance

class TypeIIPrediction(TitrationCurve):
    """Representation of a Type II (microstate log population) prediction for SAMPL6"""

    ph_range = np.linspace(2, 12, num=101)

    def __init__(self):
        super(TypeIIPrediction, self).__init__()
        return

    @classmethod
    def from_id(cls, molecule_name: str, datafile: str, charge_file: str, header=0, charge_header=0):
        """Instantiate a titration curve for one molecule from Type II predicted log populations."""
        data, charges = get_typeii_logp_data(molecule_name, datafile, charge_file, header=header, charge_header=charge_header)
        state_ids = data["Microstate ID"]     
        nbound = [int(charges.loc[charges["Microstate ID"] == id, 'Charge'])for id in state_ids]
        log_pop = data.iloc[:,1:-1].values
        pop = np.exp(np.asarray(log_pop))
        # normalize
        pop /= np.sum(pop,axis=0)[None,:]
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
    def from_id(cls, mol_id: str, datafile: str, header: int=0):
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
        sems =  np.asarray(data["SEM"])
        instance = cls.from_macro_pkas(macropKas, cls.ph_range)
        # Store data for reference
        instance.pkas = macropKas
        instance.sems = sems
        return instance
    

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
    def from_id(cls, mol_id: str, datafile: Optional[str]=None):
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
        data = get_experimental_pKa_data(
            mol_id, datafile)
        macropKas = np.asarray(data["pKa"])
        sems =  np.asarray(data["SEM"])
        instance = cls.from_macro_pkas(macropKas, cls.ph_range)
        # Store data for reference
        instance.pkas = macropKas
        instance.sems = sems
        return instance

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


