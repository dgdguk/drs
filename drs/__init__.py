"""
drs.__init__
************

:author: David Griffin <david.griffin@york.ac.uk>
:coauthors: Rob Davis <rob.davis@york.ac.uk>, Iain Bate <iain.bate@york.ac.uk>
:copyright: 2020, David Griffin / University of York
:license: MIT - A copy of this license should have been provided with this file
:publication: Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests

Implementation of the Dirichlet Rescale Algorithm

Requirements: Python 3.7

This exposes the drs function, which for almost all use cases is all that is needed.
If you need something else, it can be imported from drs.drs, but this is only
really needed if you're doing things such as evaluating the internals of the
DRS algorithm.

Note: For best performance, you must disable NumPy's default multithreading. drs
attempts to do so on import, unless the DRS_USE_NUMPY_MP environment variable is
set. However, for this to work you must import drs *before* importing NumPy,
as the vector libraries that provide NumPy's multithreading are configured by
environment variables, and only check these variables at import time (i.e. if you
import numpy before DRS, it's too late and NumPy will be using multithreading)
"""

from . import drs as drs_module
from .drs import drs

def set_epsilon(epsilon: float):
    drs_module.EPSILON = epsilon
