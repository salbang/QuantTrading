# MDD Minimization
Other than maximizing Sharpe ratio of a portfolio, minimizing the portfolio’s maximum drawdown (MDD) would be an interesting objective. While both Sharpe ratio maximization and Markowitz mean-variance portfolio optimization considers variance as risk, MDD minimization considers downside risk only. Considering variance as risk also penalizes upside rewards, which is the reason why the Markowitz portfolio optimization problem is often criticized. 

$$
\min_{\mathbf{w}} \mathbf{\mu}^\top\mathbf{w} + \lambda\cdot\text{MDD}(\mathbf{R}, \mathbf{w})
$$

Modeling this is a bit tricky, but not quite complex. First, we need to form a time series of portfolio return:

$$
\mathbf{r}_p:=\mathbf{R}\mathbf{w}
$$

Then we form a cumulative return of it:

$$
\mathbf{c}:=\text{cumsum}(\mathbf{r}_p)
$$

, which is an affine function available in cvxpy as cvxpy.cumsum.

With the cumsum vector, we define the record high vector, $\mathbf{h}$, as:

$$ \displaylines{
\max(c_0, 0) \le h_0 \\
\max(c_t, \space h_{t-1}) \le h_t\space\space\space\space \forall t \in \{1, 2, ..., m-1\}
} $$

Don’t mind the inequality because we will minimize it in the end. Thus, it doesn’t matter whether it is equality or inequality, but if we use equality constraints, the constraints are not convex.

Note that $\max(\cdot, \cdot)$ is a convex function, indeed, a piece wise linear function. More specifically, it can be broken down into two inequality constraints as:

$$ \displaylines{
c_i\le h_i \\
h_{i-1} \le h_i
} $$

With the record high, we define the drawdown vector, $\mathbf{d}$, as:

$$
\mathbf{d}:=\mathbf{h} - \mathbf{c}
$$

Now we can minimize the maximum drawdown:

$$
\min_{\mathbf{w, d, h, c}} \mathbf{\mu}^\top\mathbf{w} + \lambda||\mathbf{d}||_\infty
$$

$$ \displaylines{
s.t. \space\space \mathbf{c} = \text{cumsum}(\mathbf{Rw}) \\
\max(c_0, 0) \le h_0 \\
\max(c_t, \space h_{t-1}) \le h_i, \space\space\space\space \forall t \in \{1, 2, ..., m-1\} \\
\mathbf{d} = \mathbf{h} - \mathbf{c} \\
||\mathbf{w}||_1 \le 1
} $$

It is also possible to use this objective together with Sharpe ratio maximization or mean-variance optimization.

# Maximization of Average Return over MDD

$$
\max_\mathbf{w} \frac{\mathbf{\mu}^\top\mathbf{w}}{\text{MDD}(\mathbf{R},\mathbf{w})}
$$

Note the above maximization problem is also scale invariant with respect to $\mathbf{w}$. Thus, we can use the same technique used in Sharpe ratio maximization also to this problem.

$$ \displaylines{
\min_{\mathbf{y}, \kappa} \text{MDD}(\mathbf{R}, \mathbf{y}) \\
s.t.\space\space \mathbf{\mu}^\top \mathbf{y} = 1
} $$

For the size constraint, add either $||\mathbf{y}||_1 \le \kappa$ or $\mathbf{e}^\top\mathbf{y}=\kappa$.

# Caution!

**When we have more alphas/signals than the time intervals, i.e., $m \ll n$, then there exists more than one weight that make MDD zero**. In such case, it is better to use this optimization with any regularization method such as Tikhonov regularization, i.e., adding $\lambda||\mathbf{w}||_2^2$.
