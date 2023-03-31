# Adding Constraints and More Objective Terms for SOCP Solver

For $m$ time intervals and $n$ signals/alphas,

$$
\max_\mathbf{w}\frac{\mathbf{\mu}^\top\mathbf{w}}{\sqrt{\mathbf{w}^\top\mathbf{\Sigma w}}}
$$

To the above basic formula, we may add constraints such as size limit, upper/lower bounds, long/short balance and etc. Exploiting the scale invariance which is used in the non-constrained case is still useful.

$$
\min_{\mathbf{y}, \kappa} \frac{1}{2}\mathbf{y}^T \mathbf{\Sigma} \mathbf{y} $$
$$ s.t.\space \mathbf{\mu}^\top \mathbf{y}=1 \space\space\space\space\space\space\space\space\space\space(1) $$
$$
\kappa \ge 0
$$

Remember $\mathbf{w}:=\frac{1}{\kappa}\mathbf{y}$.

# Size Limit

If we want to allow negative weight, then we would want to keep the sum of the absolute value of the weights would be 1, i.e.:

$$
\|\mathbf{w}\|_1=1
$$

, which is equivalent to:

$$
\|\mathbf{y}\|_1=\kappa \space\space\space\space\space\space\space\space \text{ (2)}
$$

However, (2) is not convex and cannot be fed into SOCP solver. An alternative would be:

$$
\|\mathbf{y}\|_1 \le \kappa
$$

With this constraints the final weight vector $\mathbf{w}$ may not satisfy $\|\mathbf{w}\|_1 = 1$. That means the portfolio is not capital efficient. The remainder, $1 - \|\mathbf{w}\|_1$, should be allocated to cash or risk free assets. If you would still want to keep the absolute sum to be 1, then you may scale the final result.

An example implementation for the problem with the absolute sum of weight being 1 could be:

```python
import cvxpy as cp
import numpy as np
...
def max_sharpe(R, turnover, w_L, w_U, diversity_coefficient):
	m, n = R.shape
	mu = np.mean(R, axis=0)
	average_turnover = np.mean(turnover, axis=0)
	# Initialization
	constraints = []
	objective = 0
	
	y = cp.Variable(n)
	kappa = cp.Varianble(1)

	f = R @ y # Note that 1 / (m - 1) * f.T @ f is y.T @ Sigma @ y
	
	constraints += [
		0 <= kappa,
		cp.norm(y, 1) <= kappa,
		mu.T @ y == 1,
	]
	
	objective += 0.5 / (m - 1) * f.T @ f

	... # More constraints and objective terms

	prob = cp.Problem(cp.Minimize(objective), constraints)
```

If you want to have long bias, then you may instead make the sum of the value of the weights 1, i.e.:

$$
\mathbf{e}^\top\mathbf{y} = \kappa$$
$$
\kappa \ge 0
$$

# Upper/Lower Bounds

For $\mathbf{w}_U$, upper bounds, and $\mathbf{w}_L$, lower bounds, simply add:

$$
\kappa \mathbf{w}_L \le \mathbf{y} $$
$$
\mathbf{y} \le \kappa \mathbf{w}_U
$$

This would be implemented as:

```python
	constraitns += [
		kappa * w_L <= y,
		y <= kappa * w_U,
	]
```

# Diversity

For this end, there are multiple approaches, but I would introduce two. You may come up with a better idea.

## 1. As Objective

Using Tikhonov regularization or Ridge regression to avoid over-fitting in $L_2$ regression indeed promotes non-sparsity of the solution. As the coefficient of regularization grows, the solution to regression approaches a vector of the same values. This can be used in portfolio optimization too. Indeed this is actually equal to Shrinkage approach.

$$
\min_\mathbf{y, \kappa} \frac{1}{2}(\mathbf{y}^T \mathbf{\Sigma} \mathbf{y} + \lambda \|\mathbf{y}\|_2^2)
$$

or,

$$
\min_\mathbf{y, \kappa} \frac{1}{2}((1-\lambda)\mathbf{y}^T \mathbf{\Sigma} \mathbf{y} + \lambda \|\mathbf{y}\|_2^2)
$$

, where this would be the same as using the shrunken covariance of $\mathbf{\Sigma}+\lambda\mathbf{I}$  or $(1-\lambda)\mathbf{\Sigma}+\lambda\mathbf{I}$, respectively. Do not form the shrunken covariance explicitly as explained in:

[Speeding up Portfolio Optimization when using SOCP Solver](https://www.notion.so/Speeding-up-Portfolio-Optimization-when-using-SOCP-Solver-072c5c7d380f47d1bf34ddd0f3a1117d)

If you would rather want a sparse solution, you may use 1-norm regularization, just like the lasso regression.

## 2. As a Constraint

We can use the difference of the norm values of $\mathbf{w}$, equivalently $\mathbf{y}$, in a different order. For instance, we may consider using the ratio of 4-norm to 2-norm.

A fully diversified portfolio would be allocating equal weight to the alphas. In such case, the norm values would be:

$$
\|\mathbf{w}\|_2=\bigg(\frac{1}{m}\bigg)^\frac{1}{2} $$
$$
\|\mathbf{w}\|_4=\bigg(\frac{1}{m}\bigg)^\frac{1}{4}
$$

The ratio is then

$$
\frac{\|\mathbf{w}\|_4}{\|\mathbf{w}\|_2}=\frac{1}{\sqrt{m}}
$$

A fully non-diversified portfolio would be allocating weight to only one alpha. In this case, the norm values are 1, so is the ratio. Thus, the ratio of 4-norm to 2-norm lies within $[\frac{1}{\sqrt{m}}, 1]$. So for a scalar parameter, $d \in [\frac{1}{\sqrt{m}}, 1]$, within the range, we may add a constraint:

$$
\|\mathbf{y}\|_4 \le d\|\mathbf{y}\|_2
$$

This would be implemented as:

```python
	constraints += [ cp.norm(y, 4) <= diversity_coefficient * cp.norm(y, 2) ]
```

# Turnover Control

If you have a turnover matrix $\mathbf{T} \in \mathbb{R}^{m \times n}$, you may form an average turnover vector over the time window as $\mathbf{t}:=\frac{1}{m}\mathbf{T}^\top\mathbf{e}$.

1. You can penalize the average portfolio turnover in the objective function with a scalar parameter $\tau$ and Hadamard (a.k.a. element-wise) product $\odot$:
$$\min_{\mathbf{y}, \kappa} ...+\tau \|\mathbf{t}\odot\mathbf{y}\|_1$$
If you are concerned that the minimization of the variance part is quadratic and the turnover penalization is linear, then you may make the turnover penalization quadratic
    
$$
    \min_{\mathbf{y},\kappa,t}...+\tau t $$
$$
    s.t. \space \|\mathbf{t}\odot\mathbf{y}\| <= \sqrt{t}
    $$
    
2. You may limit the maximum portfolio turnover, $\tau_U$, as constraints:
$$
\|\mathbf{t}\odot\mathbf{y}\|_1 \le \kappa \tau_U
$$
    

# Transaction Cost Penalization

If you have a slippage matrix $\mathbf{S} \in \mathbb{R}^{m \times n}$, you may form an average slippage vector over the time window as $\mathbf{s}:=\frac{1}{m}\mathbf{S}^\top \mathbf{e}$. With a scalar parameter, $\xi$, you may add the penalization term to the objective function as:

$$
\min_{\mathbf{y}, \kappa}...+ \xi\|\mathbf{s}\odot \mathbf{y}\|_1
$$

# Category Limit

You may have alphas in distinct categories, such as reversion, momentum, group momentum, earning surprise and etc., and may have much more alphas in some specific categories. In such case, you may limit some categories from getting too much weight allocation.

$$
\sum_{i \in C_k}y_i \le \kappa w_{C_k} \text{ for } k \in \{ \text{ \\{indices of the category you want to limit\\}} \}
$$

# Minimum Return Required

For minimum targeting return, $\mu_o$ > 0, noting (1):

$$
\mu_o \le \mathbf{\mu}^\top\mathbf{w}  $$
$$
\mu_o \le \frac{1}{\kappa}\mathbf{\mu}^\top\mathbf{y}  $$
$$
\kappa \le \frac{1}{\mu_o}
$$