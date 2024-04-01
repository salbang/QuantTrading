# Dealing Covariance Rank Deficiency: Shrinkage or Truncated PCA

### If you have more signals/alphas than number of time intervals when creating a covariance matrix, the matrix becomes singular and positive semidefinite. This makes the Markowitz portfolio optimization or Sharpe ratio maximization problem degenerate and the solutions are not unique, and the Sharpe ratio can become infinite. To deal with this there are two basic approaches, Shrinkage or Truncated PCA.

## Definitions

$R$ : The matrix of size $m\times n$, where $m$ is the number of time intervals and $n$ is the number of signals/alphas. We assume $m < n$.

$\bar{R} := R - e_m \mu^T$, where $e_m \in \mathbb{R}^m$, $e_m=(1, 1, ..., 1)^T$ and  $\mu = \frac{1}{m} R^Te_m$.

$\Sigma$ is the empirical covariance derived from simple math, i.e., $\Sigma:=\frac{1}{n - 1}\bar{R}^T\bar{R}$

## Shrinkage

[https://repositori.upf.edu/bitstream/handle/10230/560/691.pdf?sequence=1](https://repositori.upf.edu/bitstream/handle/10230/560/691.pdf?sequence=1)

This approach uses the linear combination of the empirical covariance $\Sigma$ with some chosen target $F$ as the estimated covariance:

$$
\tilde \Sigma := (1 - \alpha) \Sigma + \alpha F
$$

Frequently used targets include $I$, the identity matrix, or a diagonal of the empirical covariance, $diag(diag(\Sigma))$, where the diagonal elements are the variances of each signal/alpha and the off-diagonals are simply zero. $diag$ extracts diagonal of a matrix or forms a square matrix from a vector as the vector values in the diagonal of the matrix.

## Truncated PCA

This approach targets the best rank $k$ spectral approximation to the empirical covariance and preserving the original variance of of each signal/alpha.

First, do compact SVD (where the singular value matrix is square, i.e., full_matrices=False when using np.linalg.svd) on $\frac{1}{\sqrt{n-1}}\bar{R}$ so that we get:

$$
\frac{1}{\sqrt{n-1}}\bar{R} = USV^T,
$$

where $U$ is an $m \times m$ orthogonal matrix, S is an $m \times m$ matrix with singular values on the diagonal in descending order (i.e., $s_i \ge s_j\space for\space i > j$), and $V$ is an $n\times m$ matrix where each columns are in unit length and are mutually orthogonal, i.e.,  $V^TV = I_m$, where $I_m$ is an identity matrix of size $m\times m$.

The rank you would like to best approximate the covariance is up to your discretion. You may decide the rank based on the variance that the approximation can explain. For instance, if you want to make the approximation explain at least 95% of the empirical covariance, you may find the smallest $k$ which satisfies ${\sum}_i^k s_i^2 \ge 0.95 * {\sum_i}^m s_i^2$.

Assuming $k$ is decided, the approximation to the covariance is

$$
\tilde{\Sigma}:=V[:, :k]S[:k, :k]^2V[:, :k]^T + diag(diag(V[:, k:]S[k:, k:]^2V[:, k:]^T))
$$

(Used python and numpy style notation to express sub-matrix assuming the index starts from 0, i.e., $[0:k]$ means elements from 0 to k-1.) 

The first term is the best rank $k$ spectral approximation and the second term is the residual variance of each signal/alpha.

This can be simplified (and with less calculations)

$$
\tilde{\Sigma}:=V[:, :k]S[:k, :k]^2V[:, :k]^T + diag(\text{np.sum(np.square}(V[:, k:]s[\text{None}, k:]), \text{axis=1,  keepdims=True}))
$$

, where $s=diag(S)$.

It is noteworthy that np.linalg.svd returns the singular values as an array instead of a matrix. I would recommend coding like:

```python
U, s, VT = np.svd(1/np.sqrt(n-1) * R_bar, full_matrices=False, compute_uv=True)
F = VT[:, :k].T * s[None, :k].T
g = np.sqrt(np.sum(np.square(V[:, k] * s[None, k:]), axis=1, keepdims=True)))
# We don't need to form Sigma_tilde
```

Indeed, you don’t need to form the full covariance matrix if you are going to use it for conic optimization. Refer to 

[Speeding up Portfolio Optimization when using SOCP Solver](https://www.notion.so/Speeding-up-Portfolio-Optimization-when-using-SOCP-Solver-658d2e331a8f4c1399e0291e17e01311?pvs=21)

[When do improved covariance matrix estimators enhance portfolio optimization? An empirical comparative study of nine estimators [](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1596865)](https://www.notion.so/When-do-improved-covariance-matrix-estimators-enhance-portfolio-optimization-An-empirical-comparati-c6eb8c23cd454c539af8865751402e22?pvs=21)