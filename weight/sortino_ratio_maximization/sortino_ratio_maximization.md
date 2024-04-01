# Sortino Ratio Maximization

Unlike Sharpe ratio, Sortino ratio only considers the downside variance as risk. Maximization of the ratio can be formulated as:

$$
\max_\mathbf{w}\frac{\boldsymbol{\mu}^\top\mathbf{w}}{\text{DR}(\mathbf{Rw})}
$$

, where $\mathbf{R}\in\mathbb{R}^{m \times n}$ and $\mathbf{w} \in \mathbb{R}^n$. $\boldsymbol{\mu}$ is the vector of average return, $m$ is the number of time intervals and $n$ is the number of assets/signals/alphas. $\text{DR}(\cdot)$ is the function that outputs the downside deviation:

$$
\text{DR}(\mathbf{r}):=\sqrt{\sum_{t=0}^{m-1} \max(-r_t, \space 0)^2}
$$

Note that the objective function is scale invariant. Thus, we can use the same technique we exploited in Sharpe ratio maximization.

$$ \displaylines{
\min_\mathbf{y, \kappa} \text{DR}(\mathbf{Ry}) \\ s.t.\space \boldsymbol{\mu}^\top \mathbf{y}=1 \space\space\space\space\space\space\space\space\space\space(1)
} $$

, where $\mathbf{y}:=\kappa \mathbf{w}$.

We define the downside return vector as:

$$
\mathbf{d}:= \text{maximum}(-\mathbf{Ry}, \mathbf{0})
$$

, where $\text{maximum}$ is an element-wise max function that works on vectors or matrices.

Then the constraints for the downside risk will be:

$$
\frac{1}{\sqrt{m - 1}}\|\mathbf{d}\|_2 <= q
$$

Now the problem can be formulated as:

$$ \displaylines{
\min_{\mathbf{y}, \kappa, \mathbf{d}, q} q \\ 
s.t.\space \boldsymbol{\mu}^\top \mathbf{y}=1 \\
\mathbf{e}^\top\mathbf{y}=\kappa \text{ or } \|\mathbf{y}\|_1 \le \kappa \\
0 \le \kappa\\
\text{maximum}(-\mathbf{Ry}, \space \mathbf{0}) \le \mathbf{d} \\
\|\mathbf{d}\|_2 <= q
} $$

# Caution!

As always, if $m \ll n$, there is a linear combination of the columns of $\mathbf{R}$ that makes the variance of the portfolio 0 and, thus, downside risk 0, too. Therefore, it is recommended to use regularization in such a case.
