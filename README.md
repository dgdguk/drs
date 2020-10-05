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
for UUnifast. It may not always be appropriate to use it as a replacement
for RandFixedSum as there are cases where RandFixedSum is faster (when
generating a large number of vectors with the same constraints).

If you wish to cite this work, please use the following reference:

```bibtex
@article{Griffin_2020,
 author = {Griffin, D. and Bate, I and Davis, R. I.},
 year = {2020},
 journal = {Accepted to 41st IEEE Real-Time Systems Symposium (RTSS 2020)},
 title = {Generating Utilization Vectors for the Systematic Evaluation of Schedulability Tests}
}
```

DRS is licensed under the MIT license.
