# Trade Optimizer

(author: jjung)

# Basic Objective

## Making target close to the ideal alpha/signal

The basic objective is to make the target as close as possible to the ideal alpha/signal. For this end, we can either use the maximization of similarity or the minimization of a distance measure.

An example of similarity maximization is:

$$
\min_{\boldsymbol{\alpha}}-\boldsymbol{\alpha}^{\mathbf{o}\top} \boldsymbol{\alpha}
$$

, given that the one norms of $\boldsymbol{\alpha}_0$ and $\boldsymbol{\alpha}$ are fixed, it can be close to correlation, thus similarity.

```python
n = len(alpha_0)
alpha = cp.Variable(n)
objective = -alpha_0 @ alpha
```

Given the ideal target alpha/signal denoted by $\boldsymbol{\alpha^o}$ is, indeed, a representation of our best return prediction of assets, this objective is similar to the return maximization part of Markowitz Portfolio Optimization problem, i.e., $\min_{\bf{w}} -\boldsymbol{\mu}^\top\bf{w}$.

An example of distance minimization is:

$$
\min_{\boldsymbol{\alpha}}\frac{1}{2}\|\boldsymbol{\alpha}-\boldsymbol{\alpha^o}\|_2^2
$$

Expanding these yields $\min_{\boldsymbol{\alpha}}\frac{1}{2}\boldsymbol{\alpha}\top\boldsymbol{\alpha}-\boldsymbol{\alpha}^{\mathbf{o}\top}\boldsymbol{\alpha}$, which is a combination of the similarity maximization and the regularization term $\frac{1}{2}\boldsymbol{\alpha}\top\boldsymbol{\alpha}=\frac{1}{2}\|\boldsymbol{\alpha}\|_2^2$.

Alternatively, you may target bigger dot product, then you may use a parameter for regularization $\lambda \in [0, 1]$,

$$
\min_{\boldsymbol{\alpha}}\frac{\lambda}{2}\boldsymbol{\alpha}\top\boldsymbol{\alpha}-\boldsymbol{\alpha^o}^\top\boldsymbol{\alpha}
$$

```python
objective = 0.5 * lambda * cp.sum_squares(alpha) - alpha_0 @ alpha
```