# drs

> [!CAUTION]
> Subsequent research has found that the Dirichlet-Rescale algorithm does not always return uniform values. Unless studying the effect of this, it is not recommended to use DRS. For an algorithm which provides an actual uniform sampler for the same problem, please use the [ConvolutionalFixedSum](https://github.com/dgdguk/convolutionalfixedsum/) algorithm. The associated research also includes information on how the non-uniformity present in DRS can be triggered.

The Dirichlet-Rescale (DRS) algorithm is a method for generating
vectors of random numbers such that:

1. The values of the vector sum to a given total U
2. Given a vector of upper bounds, each element of the returned vector is less than or equal to its corresponding upper bound
3. Given a vector of lower bounds, each element of the returned vector is greater or equal to than its corresponding lower bound
4. The distribution of the vectors in the space defined by the constraints is uniform.

DRS accomplishes this by drawing an initial point from the flat Dirichlet
Distribution and performing rescaling operations until the point
lies within the accepted region. The way in which the rescaling
operations are performed preseves the uniformity of the distribution;
the remainder of the algorithm is all about efficiently performing
these operations and minimising the effects of the rescale operations
(floating point error, running out of the finite amount of entropy
encoded in the initial point).

DRS can be thought of as a generalised version of the UUnifast and
RandFixedSum algorithms, and can be used as a replacement for both.
Note that while RandFixedSum only supports symmetrical bounds (the
same for each component of the vector), it may be faster than
DRS when generating a large number of vectors with the same
symmetric constraints.

The algorithm is described in more detail in the paper "Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests", published at RTSS 2020. The authors version can be found here: http://eprints.whiterose.ac.uk/167646/ and a narrated presentation here: https://www.youtube.com/watch?v=mwkmXYXc28k

If you wish to cite this work, please use the following references:

```bibtex
@inproceedings{GriffinRTSS2020,
  author = {David Griffin and Iain Bate and Robert I. Davis},
  title = {Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests},
  booktitle = {{IEEE} Real-Time Systems Symposium, {RTSS} 2020, Houston, Texas, USA},
  December 1-4, 2020},
  publisher = {{IEEE}},
  year = {2020},
  url = {http://eprints.whiterose.ac.uk/167646/}
}

@software{david_griffin_2020_4118059,
  author = {David Griffin and Iain Bate and Robert I. Davis},
  title = {dgdguk/drs},
  publisher = {Zenodo},
  version = {latest}
  doi = {10.5281/zenodo.4118058},
  url = {https://doi.org/10.5281/zenodo.4118058}
}
```

If citing the software itself, please cite the correct version (the DOI
of the above reference always resolves to the most recent version; the DOIs
of specific versions can be found there).

DRS is licensed under the MIT license.

# Usage

For general use, there is only one function to consider

```python
def drs(
  n: int, 
  sumu: float, 
  upper_bounds: Optional[Sequence[Union[int, float]]]=None,
  lower_bounds: Optional[Sequence[Union[int, float]]]=None
) -> Sequence[float]: ...
```

The parameters are as follows

* `n`: The number of elements to generate
* `sumu`: The target sum for the generated elements
* `upper_bounds`: An optional sequence of length `n` which gives the upper bounds on each returned value. If given, then `all(x <= y for x, y in zip(output, upper_bounds))`. If not provided, all upper bounds are set to `sumu`.
* `lower_bounds`: An optional sequence of length `n` which gives the lower bounds on each returned value. If given, then `all(x >= y for x, y in zip(output, lower_bounds))` If not provided, all lower bounds are set to `0`.

Invalid inputs are checked for and will result in a `ValueError` (e.g. if `sumu > sum(upper_bounds)`, or `upper_bounds[n] < lower_bounds[n]`).

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

Will produce vectors of length two such that `result[0] <= 1.5`, `result[1] <= 3`, and `sum(result) == 3`.

```python
from drs import drs
result = drs(2, 4, [2, 3], [1, 2])
```

Will produce vectors of length two such that `1 <= result[0] <= 2`, `2 <= result[1] <= 3`, and `sum(result) == 4`.

# Other functions

Due to the amount of entropy in a floating point being finite, and DRS's nature as a *rescaling* algorithm, it is possible for DRS to exhaust it's source of entropy. This behaviour is controlled by the *epsilon* parameter, which defaults to `10**-4`. DRS only guarantees that the values returned sum to within `sumu*epsilon` of the target, and that only the part of the result greater than `sumu*epsilon` is uniformly distributed. If more precision is required, the `set_epsilon(epsilon: float)` function can be used to adjust the epsilon parameter.

# Limits

The maximum size of output vector DRS can produce is theoretically capped at 1015 for versions of Python that use 64-bit floats. In practice it's expected that this will be too computationally expensive for practical use. DRS has been tested to produce output vectors of up to size 200, however above 140 it may be necessary to use the optional mpmath support. Consult drs.py for more information.

# libdrsc

This repository also includes `build_libdrsc.py` which uses CFFI to produce a shared
library to embded the DRS algorithm into any programming language that supports the C
ABI. This is currently in testing and will be finalised in the next release.
