"""
drs.py
******

:author: David Griffin <david.griffin@york.ac.uk>
:coauthors: Rob Davis <rob.davis@york.ac.uk>, Iain Bate <iain.bate@york.ac.uk>
:copyright: 2020, David Griffin / University of York
:license: MIT - A copy of this license should have been provided with this file
:publication: Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests

Implementation of the Dirichlet Rescale Algorithm

Requirements: Python 3.7

Limits: Tested Max number of tasks: 100
        Untested Max number of tasks: 1015
        Cayley Megner Deterimant of standard 1016D simplex overflows
        64-bit floating point, and Numpy's linalg module refuses to
        use 128-bit floating points.

Note: For best performance, you must disable NumPy's default multithreading. drs
attempts to do so on import, unless the DRS_USE_NUMPY_MP environment variable is
set. However, for this to work you must import drs *before* importing NumPy,
as the vector libraries that provide NumPy's multithreading are configured by
environment variables, and only check these variables at import time (i.e. if you
import numpy before DRS, it's too late and NumPy will be using multithreading)
"""

from __future__ import annotations

import os
from typing import Iterable, Dict, List, Optional, Tuple, Union, Sequence, NamedTuple
import warnings
from functools import lru_cache
import math
import random
    
# Disable Numpy multithreading. In development, with OpenBLAS, it was observed
# that there was a significant penalty for using multithreading in this manner.
# Unfortunately the authors do not have the ability to test the other
# implementations.

if 'DRS_USE_NUMPY_MP' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np  # type: ignore

if 'DEBUG' in os.environ:
    np.seterr('raise')  # Raise NumPy errors if debugging

try:
    DTYPE = np.float32
    DTYPE = np.float64
    if 'DRS_USE_FLOAT128' in os.environ:
        DTYPE = np.float128
except AttributeError:
    warnings.warn(f'Numpy does not have high precision floating point support available, using DTYPE={DTYPE}')

USE_MPMATH = 'DRS_USE_MPMATH' in os.environ

if USE_MPMATH:
    from mpmath import mpf
    warnings.warn('Using mpmath everywhere is slow and probably breaks uniformity of outputs.')
    DTYPE = mpf  # This is a bad idea
    

from scipy.stats import dirichlet  # type: ignore
from scipy.spatial import distance  # type: ignore
from math import inf, factorial


def scaled_dirichlet(n: int, u: float) -> List[float]:
    """A Python implementation of the Standard Dirichlet. This is
    faster than using scipy.dirichlet for single values"""
    assert n > 0
    if n == 1: return [u]
    intermediate = [-math.log(1 - random.random()) for _ in range(n)]
    divisor = sum(intermediate)
    return [x*u/divisor for x in intermediate]


def cm_matrix_det_ns(vertices: np.ndarray) -> float:
    """This computes the Cayley-Megner Matrix determinant, and then
    normalises its sign based on what the normal simplex volume
    calculation would do. It does not normalise the the values as
    this calculation tends to break floating points for n > 102.
    The resulting value can be used to determine which of two
    n-dimensional simplicies is bigger, but little else."""
    vertices = np.asarray(vertices, dtype=DTYPE)
    square_dists = distance.pdist(vertices, metric='sqeuclidean')
    number_of_vertices = distance.num_obs_y(square_dists)
    bordered_values = np.concatenate((np.ones(number_of_vertices), square_dists))
    distance_matrix = distance.squareform(bordered_values)
    det = np.linalg.det(distance_matrix)
    if vertices.size % 2 == 1:
        det = -det
    if det <= 0:
        raise ValueError('Degenerate or invalid simplex')
    return det

simplex_volume = cm_matrix_det_ns


def embed_mult_debed(matrix: np.array, vec: np.array) -> np.array:
    """Takes an N-length vector, embeds it into the N+1 transformation
    vector, multiplies by the N+1xN+1 transformation matrix (which represents
    an N dimensional transformation), and returns the unembedded N-length
    vector."""
    return (matrix @ np.concatenate([vec, np.ones(1)]))[0:-1]


def __cts(limits: np.array) -> np.array:
    """Converts constraints into the coodinates of a constraints simplex"""
    simplex_coords = np.zeros([limits.size, limits.size], dtype=DTYPE)
    for index in range(len(limits)):
        simplex_coords[index][:] = limits
        simplex_coords[index][index] = 0.0
        simplex_coords[index][index] = 1 - simplex_coords[index].sum()
    return simplex_coords

def cts(limits: Iterable[float]) -> np.array:
    """Converts constraints into the coodinates of a constraints simplex"""
    return __cts(np.asarray(limits, dtype=DTYPE))


def rmss(coords: np.array) -> np.array:
    """Given the coordinates of a simplex, constructs a matrix which
    rescales the given simplex to the standard simplex through via
    translate-scale-translate"""
    sz = len(coords)
    coords_2 = np.ones((sz+1, sz+1), dtype=DTYPE)
    coords_2[0:-1, 0:-1] = coords
    translate_matrix = np.identity(sz+1, dtype=DTYPE)
    translate_matrix[0:-1, -1] = -coords[0]
    scale_matrix = np.identity(sz+1, dtype=DTYPE)
    for n in range(sz):
        if n == 0:
            scale_matrix[n,n] = -1/((translate_matrix @ coords_2[1])[0])
        else:
            scale_matrix[n,n] = 1/((translate_matrix @ coords_2[n])[n])
    translate_matrix2 = np.identity(sz+1, dtype=DTYPE)
    translate_matrix2[0][-1] = 1
    return translate_matrix2 @ scale_matrix @ translate_matrix


def power_condition(matrix: np.array, vec: np.array) -> bool:
    """Condition used in power scaling"""
    vec_t = embed_mult_debed(matrix, vec)
    return (vec_t > 0).all()


def power_scale(limits: np.array, vec: np.array) -> np.ndarray:
    """Given a set of limits for a simplex, works out the maximum power
    to which the corresponding rescale matrix can be raised such that
    M^P.V lies within the standard simplex, then returns this point."""
    matrix = rmss(cts(limits))
    cache = {1: matrix}
    power = 1
    while power_condition(matrix, vec):
        cache[power] = matrix
        power *= 2
        matrix = matrix @ matrix
    # We have the least power of 2 such that we rescale outside of allowed area
    power //= 2
    matrix = cache[power]  # Greatest power of 2 inside allowed area
    power //= 2
    while power >= 1:  # Get remaining powers of 2
        matrix2 = matrix @ cache[power]
        if power_condition(matrix2, vec):
            matrix = matrix2
        power //= 2
    return embed_mult_debed(matrix, vec)


@lru_cache(maxsize=1024)
def standard_simplex_vol(sz: int):
    """Returns the volume of the sz-dimensional standard simplex"""
    result = simplex_volume(np.identity(sz, dtype=DTYPE))
    if result == inf:
        raise ValueError(f'Cannot compute volume of standard {sz}-simplex')
    return result


def __rescale(limits: np.array, coord: np.array, max_rescales: int=1000,
              epsilon=1e-4) -> Tuple[Union[float, int], Optional[np.array]]:
    """Performs the rescale operation given limits and an initial starting coordinate.
    The variable max_rescales determines the maximum number of rescale operations
    allowed before declaring failure; This should be at least the number of
    dimensions. epsilon is a parameter used to detect if floating point errors
    have caused the algorithm to diverge from its target"""
    count = 0
    sz = len(limits)
    base_limits = np.zeros(sz, dtype=DTYPE)
    for count in range(max_rescales):
        overlimits = coord > limits
        if np.nonzero(overlimits)[0].size == 0:
            return count, coord
        bounding_limits = np.choose(overlimits, (base_limits, limits))
        coord = power_scale(bounding_limits, coord)
        divergence = 1 - coord.sum()
        if np.abs(divergence) > epsilon:
            return inf, None  # Diverged
    return count, None


MAX_RESCALES = 1000
EPSILON = 1e-4


def rescale(limits: Iterable[float], coords: Iterable[float], max_rescales: Optional[int]=None,
            epsilon: Optional[float]=None) -> Tuple[Union[float, int], Optional[np.array]]:
    """Performs the rescale operation given limits and an initial starting coordinate.
    The variable max_rescales determines the maximum number of rescale operations
    allowed before declaring failure; This should be at least the number of
    dimensions. epsilon is a parameter used to detect if floating point errors
    have caused the algorithm to diverge from its target. See __rescale for the
    true implementation - this merely wraps __rescale so standard Python types
    can be used to specify inputs"""
    if max_rescales is None: max_rescales = MAX_RESCALES
    if epsilon is None: epsilon = EPSILON
    limits_array = np.asarray(limits, dtype=DTYPE)
    coords_array = np.asarray(coords, dtype=DTYPE)
    ret = __rescale(limits_array, coords_array, max_rescales, epsilon)
    return ret


def __ssr(limits: np.array, coord: np.array) -> Tuple[Union[float, int], Optional[np.array]]:
    """Implementation of the smallest simplex first rescale"""
    limits_simplex = cts(limits)
    try:
        limits_simplex_vol = simplex_volume(limits_simplex)
    except ValueError:
        limits_simplex_vol = 0
    except FloatingPointError:
        limits_simplex_vol = 0
    print(limits)
    print(limits_simplex_vol, standard_simplex_vol(limits.size))
    if limits_simplex_vol < standard_simplex_vol(limits.size):
        matrix = rmss(limits_simplex)
        new_limits = embed_mult_debed(matrix, np.zeros(limits.size))
        inv_matrix = rmss(cts(new_limits))
        niterations, result = rescale(new_limits, coord)
        if result is None: return niterations, result
        true_result = embed_mult_debed(inv_matrix, result)
        return niterations, true_result
    else:
        return rescale(limits, coord)


def ssr(limits: Iterable[float], coord: Iterable[float]) -> Tuple[Union[float, int], Optional[np.array]]:
    """Wrapper around the smallest simplex first rescale"""
    return __ssr(np.asarray(limits, dtype=DTYPE), np.asarray(coord, dtype=DTYPE))


class DRSError(Exception):
    """Error class for when DRS either diverges or fails to converge.
    Note that in current testing, this has never occurred, but it is
    theoretically possible."""


DRS_RETRIES = 1000


class DRSInstrumentedResult(NamedTuple):
    output_point: np.array
    initial_point: Sequence[float]
    rescales_required: Union[int, float]
    retries: int


def drs_i(ntasks: int, max_utilisation: float, upper_constraints: Optional[Sequence[float]]=None,
          lower_constraints: Optional[Sequence[float]]=None) -> DRSInstrumentedResult:
    """Interface of the DRS algorithm with instrumentation.
       Returns number of iterations required and point"""
    if upper_constraints is None:
        if lower_constraints is None:
            result = scaled_dirichlet(ntasks, max_utilisation)
        else:
            transformed_result = scaled_dirichlet(ntasks, max_utilisation - sum(lower_constraints))
            result = [o + l for o, l in zip(transformed_result, lower_constraints)]
        return DRSInstrumentedResult(result, result, 0, 0)
    if lower_constraints is not None:
        if sum(lower_constraints) >= max_utilisation:
            raise ValueError('Lower constraints must sum to less than max utilisation')
        transformed_upper_constraints = [u - l for u, l in zip(upper_constraints, lower_constraints)]
        if any(x <= 0 for x in transformed_upper_constraints):
            raise ValueError('Upper constraint specified which was lower than a lower constraint')
        transformed_problem_result = drs_i(ntasks, max_utilisation - sum(lower_constraints),
                                           transformed_upper_constraints)
        return DRSInstrumentedResult(
            [o + l for o, l in zip(transformed_problem_result.output_point, lower_constraints)],
            [i + l for i, l in zip(transformed_problem_result.initial_point, lower_constraints)],
            transformed_problem_result.rescales_required,
            transformed_problem_result.retries
        )
    if ntasks != len(upper_constraints):
        raise ValueError(f'ntasks={ntasks}, but utilisation constraints has length {len(upper_constraints)}')
    if sum(upper_constraints) <= max_utilisation:
        raise ValueError(f'Upper constraints must sum to more than max_utilisation')
    limits = [min(1, u/max_utilisation) for u in upper_constraints]
    for count in range(DRS_RETRIES):
        initial_point = scaled_dirichlet(ntasks, 1)
        iterations_required, final_unit_point = ssr(limits, initial_point)
        if final_unit_point is not None:
            break
    else:
        raise DRSError(f'In {DRS_RETRIES} attempts, DRS failed to find a point that converged before floating point error crept in.')
    final_point = final_unit_point * max_utilisation
    return DRSInstrumentedResult(
        final_point, initial_point, iterations_required, count)


def drs(ntasks: int, max_utilisation: float, upper_constraints: Optional[Sequence[float]]=None,
        lower_constraints: Optional[Sequence[float]]=None) -> np.array:
    """Interface of the DRS algorithm with no instrumentation.
    Only returns the final point."""
    return drs_i(ntasks, max_utilisation, upper_constraints, lower_constraints).output_point
