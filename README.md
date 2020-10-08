# drs

The Dirichlet-Rescale (DRS) algorithm is a method for generating
vectors of random numbers such that:

1. The values of the vector sum to a given total U
2. Given a vector of upper constraints, each element of the returned vector is less than its corresponding upper constraint
3. Given a vector of lower constraints, each element of the returned vector is greater than its corresponding lower constraint
4. The distribution of the vectors in the space defined by the constraints is uniform.

DRS accomplishes this by drawing an initial point from the flat Dirichlet
Distribution and performing rescaling operations until the point
lies within the accepted region. The way in which the rescaling
operations are performed preseves the uniformity of the distribution;
the remainder of the algorithm is all about efficiently performing
these operations and minimising the effects of the rescale operations
(floating point error, running out of the finite amount of entropy
encoded in the initial point).

DRS can be thought of as a more generalised version of the UUnifast
or RandFixedSum algorithms. In general it can be used as a replacement
for UUnifast. It may not always be appropriate to use as a replacement
for RandFixedSum as there are cases where RandFixedSum is faster (when
generating a large number of vectors with the same constraints).

If you wish to cite this work, please use the following reference:

```bibtex
@article{Griffin_2020,
 author = {Griffin, D. and Bate, I and Davis, R. I.},
 year = {2020},
 journal = {41st IEEE Real-Time Systems Symposium (RTSS 2020)},
 title = {Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests}
}
```

DRS is licensed under the MIT license.

# Usage

For general usage, there is only one function to consider

```python
def drs(
  n: int, 
  sumu: float, 
  upper_constraints: Optional[Sequence[Union[int, float]]],
  lower_constraints: Optional[Sequence[Union[int, float]]]
) -> Sequence[float]: ...
```

The parameters are as follows

* `n`: The number of values to generate
* `sumu`: The target sum for the generated values
* `upper_constraints`: An optional sequence of length `n` which gives the upper constraints on each returned value. If given, then `all(x < y for x, y in zip(output, upper_constraints))`
* `upper_constraints`: An optional sequence of length `n` which gives the lower constraints on each returned value. If given, then `all(x > y for x, y in zip(output, lower_constraints))`

# Examples

```python
from drs import drs
result = drs(2, 2)
```

Will produce vectors of length two such that `sum(result) == 2`

```python
from drs import drs
result = drs(2, 3, [1.5, 3])
```

Will produce vectors of length two such that `result[0] < 1.5`, `result[1] < 3`, and `sum(result) == 3`.

```python
from drs import drs
result = drs(2, 4, [2, 3], [1, 2])
```

Will produce vectors of length two such that `1 < result[0] < 2`, `2 < result[1] < 3`, and `sum(result) == 4`.

# Other functions

Due to the amount of entropy in a floating point being finite, and DRS's nature as a *rescaling* algorithm, it is possible for DRS to exhaust it's source of entropy. This behaviour is controlled by the *epsilon* parameter, which defaults to `10**-4`. DRS only guarantees that the values returned sum to within *epsilon* of the target utilisation, and that only the part of the result greater than *epsilon* is uniformly distributed. If more precision is required, the `set_epsilon(epsilon: float)` function can be used to adjust the epsilon parameter.

# Limits

The maximum size of output vector DRS can produce is theoretically capped at 1015 for versions of Python that use 64-bit floats. In practice it's expected that this will be too computationally expensive for practical use. DRS has been tested to produce output vectors of up to size 200.
