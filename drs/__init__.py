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

See drs.py for more detailed documentation.
"""

from . import drs as drs_module
from .drs import drs as drs

def set_epsilon(epsilon: float):
    """Set the EPSILON parameter. Results are accurate to within EPSILON"""
    drs_module.EPSILON = epsilon

def set_retries(retries: int):
    """Sets the number of retries for DRS"""
    drs_module.DRS_RETRIES = retries
