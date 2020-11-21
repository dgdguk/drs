/*
drsc.h
******

:author: David Griffin <david.griffin@york.ac.uk>
:coauthors: Rob Davis <rob.davis@york.ac.uk>, Iain Bate <iain.bate@york.ac.uk>
:copyright: 2020, David Griffin / University of York
:license: MIT - A copy of this license should have been provided with this file
:publication: Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests

Header file to use when calling DRS from C. Modified from the CFFI documentation.
*/

#ifdef __cplusplus
  extern "C" {
#endif

// Magic to make MSVC work

#ifndef CFFI_DLLEXPORT
#  if defined(_MSC_VER)
#    define CFFI_DLLEXPORT  extern __declspec(dllimport)
#  else
#    define CFFI_DLLEXPORT  extern
#  endif
#endif

// set_epsilon: sets the epsilon parameter of DRS
CFFI_DLLEXPORT void set_epsilon(float epsilon);
// set_seed: sets the seed used in DRS
CFFI_DLLEXPORT void set_seed(int n);
// drs function parameters
// parameter: n - number of values to return
// parameter: U - sum of values to return
// parameter: upper_bounds - length n array of upper_bounds
// parameter: lower_bounds - length n array of lower_bounds
// parameter: out - length n array to store output in
// drs: runs drs without upper or lower bounds
CFFI_DLLEXPORT void drs(int n, float U, float* out);
// drs_ub: runs drs with upper bounds
CFFI_DLLEXPORT void drs_ub(int n, float U, float* upper_bounds, float* out);
// drs_lb: runs drs with lower bounds
CFFI_DLLEXPORT void drs_lb(int n, float U, float* lower_bounds, float* out);
// drs_ub_lb: runs drs with upper and lower bounds
CFFI_DLLEXPORT void drs_ub_lb(int n, float U, float* upper_bounds,
                              float* lower_bounds, float* out);

#ifdef __cplusplus
  }
#endif