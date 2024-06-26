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

$$
\mathbf{\Sigma}:=\mathsf{var}(\mathbf{r})
=\mathsf{var}(\mathbf{f}^\top\mathbf{B}) + \mathsf{var}(\mathbf{\epsilon}) 
=\mathbf{B}^\top\mathbf{\Omega}\mathbf{B} + \mathbf{S}^2
$$

, where $\bf{S}$ is the specific risk matrix which is diagonal.

If the number of assets, $n$, is much larger than the number of factors, $k$, then it is better to factorize the factor covariance, $\bf{\Omega},$ using Cholesky decomposition:

$\bf{\Omega}:= \bf{L}^\top\bf{L}$

Then we can add the minimization of the risk to the optimization objective:

$$
\begin{aligned}
\min_{\boldsymbol{\alpha}}\frac{1}{2}||\boldsymbol{\alpha}&-\boldsymbol{\alpha^o}||_2^2 + \frac{\gamma}{2} \left( \bf{g}^\top\bf{g} + \bf{h}^\top \bf{h} \right) \\
s.t. \space\space & \bf{g} = \bf{LB}\boldsymbol{\alpha} \\
&\bf{h} = \bf{s}\odot\boldsymbol{\alpha}
\end{aligned}
$$

, where $\gamma$ is a **risk aversion** parameter and $\mathbf{s}$ is the vector version of the specific risk matrix, $\mathbf{s}:=\text{diag}(\mathbf{S})$.

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
\min_{\boldsymbol{\alpha}}\frac{1}{2}||\boldsymbol{\alpha}&-\boldsymbol{\alpha^o}||_2^2 + \frac{\gamma}{2} \left( \mathbf{g}^\top\mathbf{g} + \mathbf{h}^\top \mathbf{h} \right) \\
s.t. \space\space& \mathbf{g} = \mathbf{LB}\boldsymbol{\alpha} \\
&\mathbf{h} = \mathbf{s}\odot\boldsymbol{\alpha} \\
&\mathbf{e}_L \le \mathbf{B}_Q.\boldsymbol{\alpha} \le \mathbf{e}_U
\end{aligned}
$$

, where $\mathbf{B}_Q.$ is the sub-matrix of $\mathbf{B}$ where only the rows are indexed by $Q$. In numpy notation: $\mathbf{B}_Q.$ is B[Q, :].

## Limiting systematic risk

This is to make more gain from specific risk.

$$
\mathbf{g}^\top\mathbf{g} \le \xi
$$

If we want to limit the systematic risk relative to the total risk, we need a estimation of the total risk, which might be estimated from the total risk of the target of the previous interval, $\boldsymbol{\alpha}_{prev}$, which leads to:

$$
\mathbf{g}^\top\mathbf{g} \le \xi \boldsymbol{\alpha}^\top_{prev}\boldsymbol{\Sigma}\boldsymbol{\alpha}_{prev}
$$

or

$$
\mathbf{g}^\top\mathbf{g} \le \xi {\boldsymbol{\alpha}^\top_{prev}} {\boldsymbol{\Sigma_{\textit{prev}}}} \boldsymbol{\alpha}_{prev}
$$

A strategy with less systematic risk performs better during market crash (i.e., less MDD), but in usual time, this leads to less return.

# Position Constraints

## Limiting concentration

We may limit the maximum weight allocated on each asset:

$$
-\psi \|\boldsymbol{\alpha}^o\|_1\le \boldsymbol{\alpha} \le \psi \|\boldsymbol{\alpha}^o\|_1
$$

Each asset size is limited by 1% of the book size for instance.

## Limiting holding volume with respect to average daily volume

This limit is related to how much cost the portfolio would require when it would need to get liquidated.

$$
-h_v\boldsymbol{\nu}_h\le \boldsymbol{\alpha} \le h_v\boldsymbol{\nu}_h
$$

, where $\boldsymbol{\nu}_h$ is the average daily trading volume for N days used for holding limit.

# Trade Constraints and Objectives

First, we define the trade:

$$
\boldsymbol{\tau}:= \boldsymbol{\alpha}-\boldsymbol{\alpha}_{prev}
$$

, where $\boldsymbol{\alpha}_{prev}$ is the position vector of the previous time tick.

If we would like to work on the number of shares term instead of dollar term, we need to consider the price changes of $\boldsymbol{\alpha}_{prev}$. In such case:

$$
{\boldsymbol{\tau}} := {\boldsymbol{\alpha}} - {\boldsymbol{\alpha}}_{prev} {\odot} ( {\mathbf{1}} + {\mathbf{r}}_p )
$$

, where $\mathbf{r}_p$ is the return of the previous time tick.

## Limiting turnover

$$
-\tau_{max} \le \boldsymbol{\tau} \le \tau_{max}
$$

## Limiting transaction volume with respect to average daily volume

$$
-\tau_v\boldsymbol{\nu}_t\le \boldsymbol{\tau} \le \tau_v\boldsymbol{\nu}_t
$$

, where $\boldsymbol{\nu}_t$ is the average daily trading volume for N’ days used for trading limit. It is better to use shorter period of average daily trading volume for trading limit than that used for holding volume limit. Short period of average daily trading volume has better prediction power for the next day than longer period. If you use too short period of average daily volume for holding volume limit, it fluctuates too much and induces unnecessary noise to the final alpha value.

If we can have a better prediction for expected trading volume of next day, we may use it.

## Reducing transaction cost (expected slippage)

Let $\mathbf{c}$ be the slippage vector (could be past N’’ days average of spread-slippage, for instance),

$$
\min_{\boldsymbol{\alpha}}\frac{1}{2}\|\boldsymbol{\alpha}-\boldsymbol{\alpha^o}\|_2^2 + \frac{\gamma}{2} \left( \mathbf{g}^\top\mathbf{g} + \mathbf{h}^\top \mathbf{h} \right) + \xi \|\mathbf{c}\odot\boldsymbol{\tau}\|_1
$$
