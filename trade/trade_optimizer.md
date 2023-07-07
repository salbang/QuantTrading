# Trade Optimizer

# Basic Objective

## Making target close to the ideal alpha/signal

The basic objective is to make the target as close as possible to the ideal alpha/signal. For this end, we can either use the maximization of similarity or the minimization of a distance measure.

An example of similarity maximization is:

$$
\min_{\boldsymbol{\alpha}}-\boldsymbol{\alpha}^{o\top} \boldsymbol{\alpha}
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
\min_{\boldsymbol{\alpha}}\frac{1}{2}||\boldsymbol{\alpha}-\boldsymbol{\alpha^o}||_2^2
$$

Expanding these yields $\min_{\boldsymbol{\alpha}}\frac{1}{2}\boldsymbol{\alpha}\top\boldsymbol{\alpha}-\boldsymbol{\alpha}^{o\top}\boldsymbol{\alpha}$, which is a combination of the similarity maximization and the regularization term $\frac{1}{2}\boldsymbol{\alpha}\top\boldsymbol{\alpha}=\frac{1}{2}||\boldsymbol{\alpha}||_2^2$.

Alternatively, you may target bigger dot product, then you may use a parameter for regularization $\lambda \in [0, 1]$,

$$
\min_{\boldsymbol{\alpha}}\frac{\lambda}{2}\boldsymbol{\alpha}^\top\boldsymbol{\alpha}-\boldsymbol{\alpha}^{o\top}\boldsymbol{\alpha}
$$

```python
objective = 0.5 * lambda * cp.sum_squares(alpha) - alpha_0 @ alpha
```

# Size Constraints

Basic size constraints would look like with the dollar neutrality:

$$
\min_{\boldsymbol{\alpha}}\frac{1}{2}||\boldsymbol{\alpha}-\boldsymbol{\alpha}^o||_2^2
$$

$$
s.t.\space\space ||\boldsymbol{\alpha} ||_1 \le ||\boldsymbol{\alpha}^o||_1
$$

$$
\mathbf{1}^\top\boldsymbol{\alpha}=0
$$

```python
constraitns = []
constraints += [
	cp.norm(alpha, 1) <= np.linalg.norm(alpha_0, ord=1),
	cp.sum(alpha) == 0
]
```

If you want to have a little long or short exposure relative to the book size you may change $\mathbf{1}^\top\boldsymbol{\alpha}=0$ to 

$$
-\epsilon||\boldsymbol{\alpha}^o||_1\le\mathbf{1}^\top\boldsymbol{\alpha}\le\epsilon||\boldsymbol{\alpha}^o||_1
$$

```python
constraints += [ 
	-epsilon * np.linalg.norm(alpha_0, ord=1) <= cp.sum(alpha) <= epsilon * np.linalg.norm(alpha_0, ord=1)
]
```

# Risk Objective and Constraints

## Reducing risk

Given a risk model at time $t$ (or $t-1$ if we need a delay because of the data availability) in the form:

$\mathbf{r} = \mathbf{f}^\top\mathbf{B} +\epsilon$ 

, where $\bf{f}$  is the factor return and $\bf{B} \in \mathbb{R}^{k \times n}$ is the matrix of factor exposure beta of assets.

The asset covariance at time $t$ (or $t-1$) derived from the factor model is:

$\mathbf{\Sigma}:=\mathsf{var}(\mathbf{r})
=\mathsf{var}(\mathbf{f}^\top\mathbf{B}) + \mathsf{var}(\mathbf{\epsilon}) 
=\mathbf{B}^\top\mathbf{\Omega}\mathbf{B} + \mathbf{S}^2$

If the number of assets, $n$, is much larger than the number of factors, $k$, then it is better to factorize the factor covariance, $\bf{\Omega},$ using Cholesky decomposition:

$\bf{\Omega}:= \bf{L}^\top\bf{L}$

Then we can add the minimization of the risk to the optimization objective:

$$
\begin{aligned}
&\min_{\boldsymbol{\alpha}}\frac{1}{2}||\boldsymbol{\alpha}-\boldsymbol{\alpha^o}||_2^2 + \frac{\gamma}{2} \left( \bf{g}^\top\bf{g} + \bf{h}^\top \bf{h} \right) \\
s.t. \space&\\
&\bf{g} = \bf{LB}\boldsymbol{\alpha} \\
&\bf{h} = \bf{s}\odot\boldsymbol{\alpha}
\end{aligned}
$$

, where $\gamma$ is a **risk aversion** parameter and $\mathbf{s}:=\text{diag}(\mathbf{S})$.

```python
objective += 0.5 * risk_coefficient * (g @ g + h @ h)
constraints += [
	g == L @ ( B @ alpha ),
	h == specific * alpha
]
```

## Limiting exposure to selected factors

We may want to limit exposure of our portfolio on selected factors. For instance, most popular factor exposures that are often neutralized or limited are Momentum, Value, Growth, Size, Volatility.

Let $Q$ be the set of indices for the selected factors, with the upper and lower bounds vectors, $\bf{e}_L$ and $\bf{e}_U$ in $\mathbb{R}^{|Q|}$:

$$
\begin{aligned}
&\min_{\boldsymbol{\alpha}}\frac{1}{2}||\boldsymbol{\alpha}-\boldsymbol{\alpha^o}||_2^2 + \frac{\gamma}{2} \left( \mathbf{g}^\top\mathbf{g} + \mathbf{h}^\top \mathbf{h} \right) \\
s.t. \space&\\
&\mathbf{g} = \mathbf{LB}\boldsymbol{\alpha} \\
&\mathbf{h} = \mathbf{s}\odot\boldsymbol{\alpha} \\
&\mathbf{e}_L \le \mathbf{B}_{Q\cdot}\boldsymbol{\alpha} \le \mathbf{e}_U
\end{aligned}
$$

, where $\mathbf{B}_{Q\cdot}$ is the sub-matrix of $\bf{B}$ where only the rows are indexed by $Q$. In numpy notation: $\mathbf{B}_{Q\cdot}$ is B[Q, :]. 

## Limiting systematic risk

This is to make more gain from specific risk.

$$
\mathbf{g}^\top\mathbf{g} \le \xi
$$

A strategy with less systematic risk performs better during market crash (i.e., less MDD), but in usual time, this leads to less return.