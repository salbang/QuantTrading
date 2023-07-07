# Trade Optimizer

(author: jjung)

# Basic Objective

## Making target close to the ideal alpha/signal

The basic objective is to make the target as close as possible to the ideal alpha/signal. For this end, we can either use the maximization of similarity or the minimization of a distance measure.

An example of similarity maximization is:

$$
\min_{\alpha}-{\bm{\alpha^o}^\top \bm{\alpha}}
$$

, given that the one norms of $\bm{\alpha}_0$ and $\bm{\alpha}$ are fixed, it can be close to correlation, thus similarity.

```python
n = len(alpha_0)
alpha = cp.Variable(n)
objective = -alpha_0 @ alpha
```

Given the ideal target alpha/signal denoted by $\bm{\alpha^o}$ is, indeed, a representation of our best return prediction of assets, this objective is similar to the return maximization part of Markowitz Portfolio Optimization problem, i.e., $\min_{\bf{w}} -\bm{\mu}^\top\bf{w}$.

An example of distance minimization is:

$$
\min_{\bm{\alpha}}\frac{1}{2}\|\bm{\alpha}-\bm{\alpha^o}\|_2^2
$$

Expanding these yields $\min_{\bm{\alpha}}\frac{1}{2}\bm{\alpha}\top\bm{\alpha}-\bm{\alpha^o}^\top\bm{\alpha}$, which is a combination of the similarity maximization and the regularization term $\frac{1}{2}\bm{\alpha}\top\bm{\alpha}=\frac{1}{2}\|\bm{\alpha}\|_2^2$.

Alternatively, you may target bigger dot product, then you may use a parameter for regularization $\lambda \in [0, 1]$,

$$
\min_{\bm{\alpha}}\frac{\lambda}{2}\bm{\alpha}\top\bm{\alpha}-\bm{\alpha^o}^\top\bm{\alpha}
$$

```python
objective = 0.5 * lambda * cp.sum_squares(alpha) - alpha_0 @ alpha
```