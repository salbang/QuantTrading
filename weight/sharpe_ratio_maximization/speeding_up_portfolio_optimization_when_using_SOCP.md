# Speeding up Portfolio Optimization when using SOCP Solver

### When you use SOCP (second order conic problem) solver, you better not form the covariance matrix. Let’s think about the case where we have a decomposed form of a covariance matrix $\Sigma$:

1. $\Sigma = F^TF$
2. $\Sigma = F^TF + diag(g)^2$

### The first case mentioned above is a special case of 2. So just let’s investigate the second case.

If you have the portfolio variance in the objective function, for instance,

$$
\min_{\mathbf{w}} ...+\frac{c}{2}\mathbf{w}^\top\Sigma \mathbf{w} + ...
$$

and you just pass that objective term for the portfolio variance to the optimizer as is, then the SOCP solver will do some decomposition inside and may use iterative method or use SMW (Sherman-Morrison-Woodbury) formula inside, because the objective must be linear, and it will be broken down to some conic constraints with linear objective. If you use python and cvxpy as the modeling library, it would look like:

```python
import cvxpy as cp
...
w = cp.Variable(n)
objective = ... + 0.5 * c * cp.quad_form(w, Sigma) + ...
prob = cp.Problem(objective, ...)
...
```

This objective looks simple but, indeed, it is not internally. In addition, it makes the solver terribly slow.

A better way is to use the decomposed form if available. So, for the decomposed form of the case 2, note the variance term becomes:

$$
\mathbf{w}^\top \Sigma \mathbf{w}=\mathbf{w}^\top(\mathbf{F}^\top \mathbf{F} + diag(\mathbf{g})^2)w\newline
=(\mathbf{Fw})^\top\mathbf{Fw}+(diag(\mathbf{g})\mathbf{w})^\top diag(\mathbf{g})\mathbf{w}
$$

In python, the code would look like:

```python
import cvxpy as cp
...
f = F @ w
h = cp.multiply(g, w) # element wise vector multiplication
objective = ... + 0.5 * c * cvxpy.sum_squares(f) + 0.5 * c * cvxpy.sum_squares(h) + ...
prob = cp.Problem(objective)
```

Internally the objective 0.5 * c * cvxpy.sum_squares(f) translates to

$$
\min_{\mathbf{w}, t, \mathbf{f}} ... + 0.5 * c * t + ... \newline
s.t. \newline
\|\mathbf{f}\|\le\sqrt{t}
$$

It is always better to avoid using cvxpy.quad_form if possible. This way is more than x10 times faster.

If you do not have a decomposed form of the covariance, you may do Cholesky factorization of it:

$$
\mathbf{\Sigma}:=\mathbf{U}^\top\mathbf{U}
$$

, where $\mathbf{U}$ is an upper triangular matrix. If $\mathbf{\Sigma}$ is rank deficient, you may use incomplete Cholesky factorization with pivoting, QR factorization with pivoting, or SVD, although it is not recommended to use such a rank deficient covariance. Refer to 

[Dealing Covariance Rank Deficiency: Shrinkage or Truncated PCA](https://www.notion.so/Dealing-Covariance-Rank-Deficiency-Shrinkage-or-Truncated-PCA-2dffa623b9524f4c822316213cc68956)
