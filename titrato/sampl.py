"""This module contains classes and functions specific to SAMPL6 data files"""
import pandas as pd
import numpy as np
from .titrato import TitrationCurve, free_energy_from_population
from .titrato import data_dir
import warnings
import os
from typing import Tuple, Optional
# SAMPL6_ID,pKa,uncertainty

def get_typeiii_pka_data(molecule_name: str, datafile, header=0):
    """Retrieve predicted pKa values and errors from a SAMPL6 type III data file."""
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

class TypeIIIPrediction(TitrationCurve):
    """Representation of a Type III (macropKa) prediction for SAMPL6."""

    ph_range = np.linspace(2, 12, num=201)
    def __init__(self):

        super(TypeIIIPrediction, self).__init__()
        self.pkas = None
        self.sems = None

        return
    
    @classmethod
    def from_id(cls, mol_id: str, datafile: str, header: int=0):
        """Retrieve the titration curve for one molecule from the experiment.

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
    ph_range = np.linspace(2, 12, num=201)

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


