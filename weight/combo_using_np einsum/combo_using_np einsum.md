# Combo Using np.einsum

# Easy approach

We have a variable, weights, which is a 2 dimensional array and alphas which is a 3 dimensional array. A simple way to get weighted average of the alphas is:

```python
combo = np.sum(weights[..., None] * alphas, axis=0)
```

A problem with this approach is that the multiplication with the broad casted weights results in another 3 dimensional array of the size of alphas. This is memory inefficient and slower than np.einsum.

# np.einsum

If we use np.einsum:

```python
combo = np.einsum('at,ati->ti', weights, alphas)
```

With this line of code, we do not introduce another large array of the size of signals.

# References

[https://numpy.org/doc/stable/reference/generated/numpy.einsum.html](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)

[https://en.wikipedia.org/wiki/Einstein_notation](https://en.wikipedia.org/wiki/Einstein_notation)

[https://towardsdatascience.com/einstein-index-notation-d62d48795378](https://towardsdatascience.com/einstein-index-notation-d62d48795378)