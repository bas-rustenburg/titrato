"""
Unit and regression test for the titrato package.
"""

# Import package, test suite, and other packages as needed
import titrato
import titrato.sampl
import titrato.reports
import pytest
import sys
import os
import numpy

# numpy.set_printoptions(threshold=numpy.nan)


def test_titrato_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "titrato" in sys.modules


def test_macroscopic_provider():
    """Test performing bootstrap on macroscopic titration curves."""
    jaguar_file = os.path.join(titrato.data_dir, "jaguar-typeIII-raw.csv")
    charge_file = os.path.join(titrato.data_dir, "jaguar-typeIII-charges.csv")
    exp_file = os.path.join(titrato.data_dir, "SAMPL6_experimental_pkas.csv")

    exp = titrato.sampl.SAMPL6DataProvider(
        exp_file, "exp", "Experiment", bootstrap_options={"n_samples": 3}
    )
    jaguarmacro = titrato.sampl.SAMPL6DataProvider(
        jaguar_file,
        "typeiii",
        "Jaguar-macro",
        bootstrap_options={"n_samples": 1},
        typeiii_charge_file=charge_file,
    )
    rep = titrato.reports.TitrationComparison(
        exp,
        [jaguarmacro],
        included_molecules=["SM08"],
        # n_bootstrap_correlation=1,
        n_bootstrap_titration=50,
    )

    rep.analyze_all()

    print(rep.table_curve_area())
    return
