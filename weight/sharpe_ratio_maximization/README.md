# Sharpe Ratio Maximization with No Constraint

# Solve:

$$
\max_\mathbf{w}\frac{\mathbf{\mu}^\top\mathbf{w}}{\sqrt{\mathbf{w}^\top\mathbf{\Sigma w}}}
$$

, where $\mathbf{\mu}$ is the vector of average return of signals/alphas, $\mathbf{w}$ is the vector of weight allocation and $\Sigma$ is the covariance matrix of signals/alphas.

Since the objective function is scale invariant with respect to $w$, it is not important to impose the restriction on $w$ so that it sums to 1. Now let $\mathbf{y}=\kappa \mathbf{w}$ such that $\mathbf{\mu}^\top \mathbf{y} = 1$ for $\kappa > 0$.

Then the problem becomes

$$
\max_{\mathbf{y}, \kappa}\frac{\mathbf{\mu}^\top \mathbf{y} / \kappa}{\sqrt{\mathbf{y}^\top\mathbf{\Sigma y}} / \kappa} 
$$

$$
s.t.\space \mathbf{\mu}^\top \mathbf{y} = 1
$$

This is equivalent to:

$$
\min_{\mathbf{y}} \frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma y} 
$$
$$
s.t.\space \mathbf{\mu}^T \mathbf{y}=1
$$

This is a quite simple convex optimization problem. To solve this, we form the Lagrangian of the problem and find the stationary point of it.

The Lagrangian is

$$
\mathcal{L}(\mathbf{y}, \lambda)=\frac{1}{2}\mathbf{y}^\top\mathbf{\Sigma y} - \lambda (\mathbf{\mu}^\top\mathbf{y} - 1)
$$

To find the stationary point of $\mathcal{L}$, we get the gradient of it and find the root:

$$\nabla_\mathbf{y} \mathcal{L} = \mathbf{\Sigma y} - \lambda \mathbf{\mu} = 0$$
$$\nabla_\lambda \mathcal{L} = -\mathbf{\mu}^\top\mathbf{y} + 1 = 0$$

Simple solution acquired from the first line of the two equations is:

$$\mathbf{y}=\lambda\mathbf{\Sigma}^{-1}\mathbf{\mu}$$

Since the original problem is scale invariant with respect to $\mathbf{w}$, it is not important to get the exact $\lambda$. You can just assume $\lambda$ be 1 or any other positive scalar value then scale $\mathbf{y}$ so that $\mathbf{w}$ becomes either $\mathbf{e}^\top\mathbf{w}=1$, where $\mathbf{e}$ is the vector of all 1 as its elements, or  $\|\mathbf{w}\|_1 = 1$. **(Never, ever use the matrix inverse to get the solution!!!)**

The optimal solution is, indeed, the same as the typical unconstrained Markowitz mean-variance portfolio optimization problem:

$$
\max_{\mathbf{w}} \mathbf{\mu}^\top\mathbf{w} - \frac{\gamma}{2}\mathbf{w}^\top\mathbf{\Sigma w}
$$

with only scale difference.

With constraints, however, two problems result in different solutions.

You may add more terms to the objective function and more constraints to the problem. Refer to:

[Adding Constraints and More Objective Terms for SOCP Solver](Sharpe%20Ratio%20Maximization%20with%20No%20Constraint%2030a3a4e7fb404c6ca9ce19250a6dfa31/Adding%20Constraints%20and%20More%20Objective%20Terms%20for%20SO%20f552e23a0d4143558011a4e0daeff8ec.md)
