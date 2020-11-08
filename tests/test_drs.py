"""
tests.py
********

:author: David Griffin <david.griffin@york.ac.uk>
:coauthors: Rob Davis <rob.davis@york.ac.uk>, Iain Bate <iain.bate@york.ac.uk>
:copyright: 2020, David Griffin / University of York
:license: MIT - A copy of this license should have been provided with this file
:publication: Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests

Provides various functional tests for the DRS algorithm.

Note that these tests only check *functional* properties. There are
many important *non-functional* properties, such as the distribution
of returned values, and these properties are left to be evaluated
in the paper. The best way to think of these tests are for
as some basic tests that indicate there is nothing fundamentally
wrong with the DRS algorithm and that the interfaces provided
work as advertised with the constraints of a single function call.
However, these tests alone will not guarantee the uniformity
of distributions, and so more in-depth analyses are needed. The
authors refer to the publication for the details of these analyses.

These tests are designed to be run with the PyTest unit testing
framework i.e. run 'pytest tests.py'

As these tests are designed to run quickly, for the purposes
of detecting errors while changing DRS, only 5, 10, 15 and 20
tasks are considered. Test constraints are generated via DRS
without upper or lower constraints.

These tests give coverage of all of DRS.py apart from certain
trivial error conditions (e.g. invalid problem specifications)
"""

from drs import drs_module as drs
import pytest
from scipy.stats import dirichlet
import random

NTASKS_LIST = [5, 10, 15, 20]
REPEATS = 100
UPPER_CONSTRAINT_TOTAL = 2
LOWER_CONSTRAINT_TOTAL = .5

def test_simplex_volume():
    """This is more of a test of the underlying system. When Python
    does not have access to sufficiently precise floating points, this
    will break. Computing the standard simplex volume is sufficient,
    as errors encountered during the constraints simplex will indicate
    that the constraints simplex is smaller."""
    for x in range(2, 103):
        drs.standard_simplex_vol(x)


def generate_constraints_random(ntasks, total):
    """Generate some random constraints that sum to a given total.
    These constraints are *not* uniformly distributed."""
    vals = [random.random() for _ in range(ntasks)]
    sumv = sum(vals)
    ret = [x * (total / sumv) for x in vals]
    assert abs(sum(ret) - total) < 1e-5  # Allows for some floating point error
    return list(ret)


def generate_constraints_dirichlet(ntasks, total):
    """Generate some random constraints that sum to a given total.
    These constraints are uniformly distributed via the Dirichlet
    Distribution"""
    ret = dirichlet.rvs([1]*ntasks)[0] * total
    assert abs(sum(ret) - total) < 1e-5  # Allows for some floating point error
    return list(ret)

    
def test_drs_no_constraints():
    """Test DRS with no constraints. Checks to see that the returned
    values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        for _ in range(REPEATS):
            result = drs.drs(n, 1)
            assert abs(1 - sum(result)) < drs.EPSILON
            assert all(x > 0 for x in result)
            assert all(x < 1 for x in result)


@pytest.mark.parametrize('constraint_gen', [generate_constraints_dirichlet, generate_constraints_random])
def test_drs_lower_bounds(constraint_gen):
    """Test DRS with lower bounds. Checks no lower bound is
    broken, and that the returned values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        for _ in range(REPEATS):
            lower_bounds = constraint_gen(n, LOWER_CONSTRAINT_TOTAL)
            util = 1.0
            result = drs.drs(n, util, lower_bounds=lower_bounds)
            assert all(x > y for x, y in zip(result, lower_bounds))
            assert abs(1 - sum(result)) < drs.EPSILON
            assert all(x < 1 for x in result)


@pytest.mark.parametrize('constraint_gen', [generate_constraints_dirichlet, generate_constraints_random])
def test_drs_upper_constraints(constraint_gen):
    """Test DRS with upper constraints. Checks no upper constraint is
    broken, and that the returned values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        for _ in range(REPEATS):
            upper_bounds = constraint_gen(n, UPPER_CONSTRAINT_TOTAL)
            result = drs.drs(n, 1, upper_bounds)
            assert all(x < y for x, y in zip(result, upper_bounds))
            assert abs(1 - sum(result)) < drs.EPSILON
            assert all(x > 0 for x in result)


@pytest.mark.parametrize('constraint_gen,insert_fixed_point', [
    [generate_constraints_dirichlet, True], 
    [generate_constraints_dirichlet, False],
    [generate_constraints_random, True], 
    [generate_constraints_random, False]
])
def test_drs_upper_and_lower_constraints(constraint_gen, insert_fixed_point):
    """Test DRS with both upper and lower constraints. Checks no constraint is
    broken, and that the returned values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        x = 0
        while x < REPEATS:
            upper_bounds = constraint_gen(n, UPPER_CONSTRAINT_TOTAL)
            lower_bounds = constraint_gen(n, LOWER_CONSTRAINT_TOTAL)
            if all(l < u for l, u in zip(lower_bounds, upper_bounds)):
                util = 1
                i = 0
                if insert_fixed_point:
                    position = random.randint(0, n-1)
                    fixed_point = random.random()
                    assert isinstance(lower_bounds, list)
                    assert isinstance(upper_bounds, list)
                    lower_bounds.insert(position, fixed_point)
                    upper_bounds.insert(position, fixed_point)
                    util += fixed_point
                    i = 1
                assert sum(lower_bounds) <= util
                result = drs.drs(n+i, util, upper_bounds, lower_bounds)
                if insert_fixed_point:
                    assert isinstance(result, list)
                    fixed_point_returned = result.pop(position)
                    assert fixed_point_returned == fixed_point
                    upper_bounds.pop(position)
                    lower_bounds.pop(position)
                    assert len(upper_bounds) == len(result)
                assert all([x < y for x, y in list(zip(result, upper_bounds))])
                assert all(x > y for x, y in zip(result, lower_bounds))
                assert abs(1 - sum(result)) < drs.EPSILON
                x += 1


@pytest.mark.parametrize('constraint_gen', [generate_constraints_dirichlet, generate_constraints_random])
def test_drs_fixed_point(constraint_gen):
    """Test to see that DRS works with constraints where only a single
    point is valid"""
    for n in NTASKS_LIST:
        for _ in range(REPEATS):
            constraints = constraint_gen(n, 1)
            result = drs.drs(n, sum(constraints), constraints, constraints)
            assert result == constraints
