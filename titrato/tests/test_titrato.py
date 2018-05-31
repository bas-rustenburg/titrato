"""
Unit and regression test for the titrato package.
"""

# Import package, test suite, and other packages as needed
import titrato
import pytest
import sys

def test_titrato_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "titrato" in sys.modules
