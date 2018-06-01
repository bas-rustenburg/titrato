"""
titrato
Utilities for calculating titration curves.
"""
import os
pkg_dir =  os.path.split(__file__)[0]
data_dir = os.path.join(pkg_dir, "data")

from .titrato import *


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
