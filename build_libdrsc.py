
"""
build_libdrsc.py
****************

:author: David Griffin <david.griffin@york.ac.uk>
:coauthors: Rob Davis <rob.davis@york.ac.uk>, Iain Bate <iain.bate@york.ac.uk>
:copyright: 2020, David Griffin / University of York
:license: MIT - A copy of this license should have been provided with this file
:publication: Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests

Requirements: cffi (version > 1.15)

Uses cffi (version > 1.15) to embed DRS into a shared library with a C ABI. This allows
DRS to be used directly from any programming language that supports calling C functions,
which is basically all of them. However, this shared library is dependent on the
Python installation used to compile it.

If using C or C++, the header file drsc.h can be used.

This file is adapted from the CFFI documentation.

libdrsc exposes the following functions:

void set_epsilon(float epsilon)
  adjusts the epsilon parameter of DRS
void set_seed(int seed)
  sets the seed used in DRS
void drs(int n, float U, float* out)
  runs drs without upper or lower bounds
void drs_ub(int n, float U, float* upper_bounds, float* out)
  runs drs with upper bounds
void drs_lb(int n, float U, float* lower_bounds, float* out)
  runs drs with lower bounds
void drs_ub_lb(int n, float U, float* upper_bounds, float* lower_bounds,
               float* out)
  runs drs with upper and lower bounds

The drs* functions use the following parameters

  n - number of values to return
  U - sum of values to return
  upper_bounds - length n array of upper_bounds
  lower_bounds - length n array of lower_bounds
  out - length n array to store output in

Common problems:

1) On Windows, some Python distributions may not set the environment variables
   needed to find the Python installation. If this is the case, try adding
   the Python installation to your PATH, or setting the PYTHONPATH environment
   variable to point at the Python installation.
2) On Windows, only MSVC is supported, and only versions compatible with
   the Python build. It is theoretically possible to use MingW, but this
   has not been tested.
3) On Windows, there may be some issues using VirtualEnvs.
4) (On Linux and Mac OS, everything seems to work perfectly)
"""


import cffi
import re
ffibuilder = cffi.FFI()

CPLUSPLUS_DEF_REMOVER = re.compile("#ifdef __cplusplus\n.*?\n#endif")
DEFINE_REMOVER = re.compile("#.*\n")

with open('drsc.h') as f:
    # read plugin.h and pass it to embedding_api(), manually
    # removing the '#' directives and the CFFI_DLLEXPORT
    data = f.read()
    data = CPLUSPLUS_DEF_REMOVER.sub('', data)
    data = DEFINE_REMOVER.sub('', data)
    data = data.replace('CFFI_DLLEXPORT', '')
    ffibuilder.embedding_api(data)

ffibuilder.set_source("libdrsc", """
    #include "drsc.h"
""")

drsc_code = """
import numpy as np
import random

from drs import drs_module
from drs import drs as drs_py

from libdrsc import ffi

@ffi.def_extern()
def set_epsilon(epsilon):
    global EPSILON
    EPSILON = drs_module.epsilon

@ffi.def_extern()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

@ffi.def_extern()
def drs(n, U, out):
    drs_output = drs_py(n, U)
    for index, val in enumerate(drs_output):
        out[index] = val

@ffi.def_extern()
def drs_ub(n, U, upper_constraints, out):
    upper_constraints = [upper_constraints[x] for x in range(n)]
    drs_output = drs_py(n, U, upper_constraints)
    for index, val in enumerate(drs_output):
        out[index] = val

@ffi.def_extern()
def drs_lb(n, U, lower_constraints, out):
    lower_constraints = [lower_constraints[x] for x in range(n)]
    drs_output = drs_py(n, U, lower_constraints=lower_constraints)
    for index, val in enumerate(drs_output):
        out[index] = val

@ffi.def_extern()
def drs_ub_lb(n, U, upper_constraints, lower_constraints, out):
    upper_constraints = [upper_constraints[x] for x in range(n)]
    lower_constraints = [lower_constraints[x] for x in range(n)]
    drs_output = drs_py(n, U, upper_constraints, lower_constraints)
    for index, val in enumerate(drs_output):
        out[index] = val
"""

ffibuilder.embedding_init_code(drsc_code)

ffibuilder.compile(target="libdrsc.*", verbose=True)
