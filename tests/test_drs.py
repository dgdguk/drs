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
of distributions, and so more in-depth analyses (such as those
carried out in make_figure_13_data.py) are needed.

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
import numpy as np
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
    return ret


def generate_constraints_dirichlet(ntasks, total):
    """Generate some random constraints that sum to a given total.
    These constraints are uniformly distributed via the Dirichlet
    Distribution"""
    ret = dirichlet.rvs([1]*ntasks)[0] * total
    assert abs(sum(ret) - total) < 1e-5  # Allows for some floating point error
    return ret

    
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
def test_drs_lower_constraints(constraint_gen):
    """Test DRS with lower constraints. Checks no lower constraint is
    broken, and that the returned values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        for _ in range(REPEATS):
            lower_constraints = constraint_gen(n, LOWER_CONSTRAINT_TOTAL)
            result = drs.drs(n, 1, lower_constraints=lower_constraints)
            assert all(x > y for x, y in zip(result, lower_constraints))
            assert abs(1 - sum(result)) < drs.EPSILON
            assert all(x < 1 for x in result)


@pytest.mark.parametrize('constraint_gen', [generate_constraints_dirichlet, generate_constraints_random])
def test_drs_upper_constraints(constraint_gen):
    """Test DRS with upper constraints. Checks no upper constraint is
    broken, and that the returned values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        for _ in range(REPEATS):
            upper_constraints = constraint_gen(n, UPPER_CONSTRAINT_TOTAL)
            result = drs.drs(n, 1, upper_constraints)
            assert all(x < y for x, y in zip(result, upper_constraints))
            assert abs(1 - sum(result)) < drs.EPSILON
            assert all(x > 0 for x in result)


@pytest.mark.parametrize('constraint_gen', [generate_constraints_dirichlet, generate_constraints_random])
def test_drs_upper_and_lower_constraints(constraint_gen):
    """Test DRS with both upper and lower constraints. Checks no constraint is
    broken, and that the returned values sum to within drs.EPSILON of 1"""
    for n in NTASKS_LIST:
        x = 0
        while x < REPEATS:
            upper_constraints = constraint_gen(n, UPPER_CONSTRAINT_TOTAL)
            lower_constraints = constraint_gen(n, LOWER_CONSTRAINT_TOTAL)
            if all(l < u for l, u in zip(lower_constraints, upper_constraints)):
                result = drs.drs(n, 1, upper_constraints, lower_constraints)
                assert all(x < y for x, y in zip(result, upper_constraints))
                assert all(x > y for x, y in zip(result, lower_constraints))
                assert abs(1 - sum(result)) < drs.EPSILON
                x += 1

