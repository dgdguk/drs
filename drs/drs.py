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

Config environment variables:

DRS_USE_NUMPY_MP: If set, allows Numpy's default multithreading behaviour.
                  In testing, Numpy's default multithreading results in
                  a slowdown of about 20x above 10 partitions over single
                  threaded mode, so instead it is recommended to use
                  standard Python multiprocessing.
DRS_USE_FLOAT128: If set, uses FLOAT128 if available. Can avoid some
                  floating point errors, but as Numpy cannot generate
                  random 128-bit floats easily, does not prevent finite
                  entropy problems causing retries.
DRS_USE_MPMATH:   If set, uses mpmath mixed precision. This is slow, but
                  starts to be necessary above 140 paritions (in the worst
                  case)
DRS_MPMATH_PREC:  Controls the precision of mpmath mode. Can be set to any
                  integer bit length. Defaults to 256.

Limits: Tested Max number of partitions: 100
        Untested Max number of partitions: 1015 (some testing up to 200)
        Cayley Megner Deterimant of standard 1016D simplex overflows
        64-bit floating point, and Numpy's linalg module refuses to
        use 128-bit floating points.

Note: For best performance, it is necessary to disable
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple, Union, Sequence, NamedTuple
import warnings
from functools import lru_cache
import math
import random
from scipy.spatial import distance  # type: ignore

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

if 'DRS_DEBUG' in os.environ:
    np.seterr('raise')  # Raise NumPy errors if debugging

USE_FLOAT128 = False

try:
    DTYPE = np.float32
    DTYPE = np.float64
    if 'DRS_USE_FLOAT128' in os.environ:
        DTYPE = np.float128
        USE_FLOAT128 = True
        warnings.warn('Numpy does not have fp128 support for random number generation'
                      ' which means that using fp128 can only help with floating point'
                      ' errors, and will not help with limited entropy. If you need more'
                      ' entropy, try using DRS_USE_MPMATH instead.')
except AttributeError:
    warnings.warn(f'Numpy does not have high precision floating point support available, using DTYPE={DTYPE}')

USE_MPMATH = 'DRS_USE_MPMATH' in os.environ
MPMATH_PREC = int(os.environ['DRS_MPMATH_PREC']) if 'DRS_MPMATH_PREC' in os.environ else 256

if USE_MPMATH:
    import mpmath  # type: ignore
    warnings.warn('DRS MPMATH mode is very slow and is not tested as well.')
    DTYPE = mpmath.mpf
    mpmath.prec = MPMATH_PREC
    def scaled_dirichlet(n: int, u: float) -> List[float]:
        assert n > 0
        if n == 1: return np.asarray([u], dtype=DTYPE)
        random_vals = [mpmath.rand() for _ in range(n)]
        intermediate = [-mpmath.log(1 - x) for x in random_vals]
        divisor = sum(intermediate)
        result = [x*u/divisor for x in intermediate]
        return result
else:
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
    result = cm_matrix_det_ns(np.identity(sz, dtype=DTYPE))
    if result == math.inf:
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
            return math.inf, None  # Diverged
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
        limits_simplex_vol = cm_matrix_det_ns(limits_simplex)
    except ValueError:
        limits_simplex_vol = 0
    except FloatingPointError:
        limits_simplex_vol = 0
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
FLOAT_TOLERANCE = 1e-10

class DRSInstrumentedResult(NamedTuple):
    output_point: List[float]
    initial_point: List[float]
    rescales_required: Union[int, float]
    retries: int


def drs_i(n: int, sumu: float, upper_bounds: Optional[List[float]]=None,
          lower_bounds: Optional[List[float]]=None) -> DRSInstrumentedResult:
    """Interface of the DRS algorithm with instrumentation.
       Returns number of iterations required and point"""
    if n == 0:
        if upper_bounds is not None and len(upper_bounds) != 0:
            raise ValueError("len(upper_bounds) must be equal to n")
        if lower_bounds is not None and len(lower_bounds) != 0:
            raise ValueError("len(lower_bounds) must be equal to n")
        return DRSInstrumentedResult([], [], 0, 0)
    if n == 1:
        if upper_bounds is not None:
            if len(upper_bounds) != 1:
                raise ValueError("len(upper_bounds) must be equal to n")
            elif sumu - upper_bounds[0] > FLOAT_TOLERANCE:
                raise ValueError("Upper bounds must sum to more than sumu")
        if lower_bounds is not None:
            if len(lower_bounds) != 1:
                raise ValueError("len(upper_bounds) must be equal to n")
            elif lower_bounds[0] - sumu > FLOAT_TOLERANCE:
                raise ValueError("Lower bounds must sum to less than max utilisation")
            if upper_bounds is not None and upper_bounds[0] == lower_bounds[0]:
                # sumu is prone to floating point error (if multiple bounds removed)
                # but if upper_bounds == lower_bounds, we know the true value
                sumu = upper_bounds[0]
        return DRSInstrumentedResult([sumu], [sumu], 0, 0)
    if upper_bounds is None:
        if lower_bounds is None:
            result = scaled_dirichlet(n, sumu)
        elif abs(sum(lower_bounds) - sumu) < FLOAT_TOLERANCE:
            return DRSInstrumentedResult(lower_bounds, lower_bounds, 0, 0)
        else:
            transformed_result = scaled_dirichlet(n, sumu - sum(lower_bounds))
            result = [o + l for o, l in zip(transformed_result, lower_bounds)]
        return DRSInstrumentedResult(result, result, 0, 0)
    if lower_bounds is not None:
        if lower_bounds[0] - sumu > 1e-10:
            raise ValueError('Lower bounds must sum to less than max utilisation')
        for index, bounds in enumerate(zip(upper_bounds, lower_bounds)):
            upper_bound, lower_bound = bounds
            if lower_bound > upper_bound:
                raise ValueError(f'Lower bound > Upper bound ({lower_bound} > {upper_bound})')
            elif lower_bound == upper_bound:
                fixed_point = upper_bounds.pop(index)
                lower_bounds.pop(index)
                drs_result = drs_i(
                    n - 1,
                    sumu - lower_bound,
                    upper_bounds,
                    lower_bounds,
                    )
                drs_result.initial_point.insert(index, fixed_point)
                drs_result.output_point.insert(index, fixed_point)
                return drs_result

        transformed_upper_bounds = [u - l for u, l in zip(upper_bounds, lower_bounds)]
        assert all(x > 0 for x in transformed_upper_bounds), 'Bug detected with input, please report'
        transformed_problem_result = drs_i(n, sumu - sum(lower_bounds),
                                           transformed_upper_bounds)
        return DRSInstrumentedResult(
            [o + l for o, l in zip(transformed_problem_result.output_point, lower_bounds)],
            [i + l for i, l in zip(transformed_problem_result.initial_point, lower_bounds)],
            transformed_problem_result.rescales_required,
            transformed_problem_result.retries
        )
    if n != len(upper_bounds):
        raise ValueError(f'n={n}, but utilisation constraints has length {len(upper_bounds)}')
    if abs(sum(upper_bounds) - sumu) < FLOAT_TOLERANCE:
        return DRSInstrumentedResult(upper_bounds, upper_bounds, 0, 0)
    if sum(upper_bounds) < sumu:
        raise ValueError(f'Upper bounds must sum to more than sumu')
    limits = [min(1, u/sumu) for u in upper_bounds]
    for count in range(DRS_RETRIES):
        initial_point = scaled_dirichlet(n, 1)
        iterations_required, final_unit_point = ssr(limits, initial_point)
        if final_unit_point is not None:
            break
    else:
        raise DRSError(f'In {DRS_RETRIES} attempts, DRS failed to find a point that converged before floating point error crept in.')
    final_point = final_unit_point * sumu
    return DRSInstrumentedResult(
        list(final_point), list(initial_point), iterations_required, count)


def drs(n: int, sumu: float, upper_bounds: Optional[Sequence[float]]=None,
        lower_bounds: Optional[Sequence[float]]=None) -> List[float]:
    """Interface of the DRS algorithm with no instrumentation.
    Only returns the final point. Arguments:
    
    - n: Number of partitions
    - sumu: Value which output must sum to
    - upper_bounds: A sequence of length n which describes the upper bounds
                    of the output vector
    - lower_bounds: A squence of length n which describes the lower bounds
                    of the output vector
    """
    if upper_bounds is not None: upper_bounds = list(upper_bounds).copy()
    if lower_bounds is not None: lower_bounds = list(lower_bounds).copy()
    return drs_i(n, sumu, upper_bounds, lower_bounds).output_point
