#Linear Manifold Cluster Generator

[LinearManifoldCluster]() package allows to build multiple linear manifold clusters given a set of parameters which characterize manifold construction.

Model of a $M$-dimensional linear manifold cluster in $\mathbb{R}^N$ can be defined by following random process <a name="lmcm">\[1\]</a>:
$$
\mathbf{x} = \mathbf{\mu}^{N \times 1} + \mathbf{B}^{N \times M} \mathbf{\phi}^{M \times 1} + \overline{\mathbf{B}}^{N \times N-M} \mathbf{\varepsilon}^{N-M \times 1}
$$
where

- $\mathbf{\mu} \in \mathbb{R}^N$, a manifold mean (or a translation vector),
- $\mathbf{B}$, a linear manifold basis which is a matrix whose columns are orthonormal vectors that span $M$-dimensional LM,
- $\overline{\mathbf{B}}$, an orthogonal complement of a linear manifold basis which is a matrix whose columns span subspace orthogonal to spanned by columns of $\mathbf{B}$,
- $\mathbf{\phi}$, a zero-mean random vector whose entries are i.i.d. from a support of LM,
- $\mathbf{\varepsilon}$, a zero mean random vector with small variance independent of $\mathbf{\phi}$.

## Generation procedure

All linear manifold (LM) clusters are generated in a unit hypercube, $[0,1]^N$, where $N$ is a dimension of the full space. In order to generate linear manifold cluster of the dimension $M$ which contains $m$ points, following steps are performed:

1. Construct a translation vector $\mu$ of the linear manifold cluster.
    a) Generate $N$ Gaussian random variables $x_1, \dots, x_{N}$ and form a point $o$.
    b) Transform point $o$ to a point on a $N$-dimensional unit ball by dividing coordinates of a vector on a radius of the hypersphere $r=\sqrt{x_1^2+ \dots + x_{N}^2}$, $o = [x_1 \dots  x_{N}]^T/r$. [\[Hypersphere\]][1]
    c) Point $o$ now has coordinates on a $N$-dimensional unit ball or radius 1. In order for point $o$ be an origin of LM, that is bounded in a unit hypercube, we define bounds on how far our origin should extend from the center of the unit hypercube using bounds $[o_{min}, o_{max}]$.
    d) By picking extent value $\tau$ from a uniform distribution over $[o_{min}, o_{max}]$, we generate LM translation vector as follows $\mathbf{\mu} = \frac{\tau o + \mathbf{1}^N}{2}$, where $\mathbf{\mu}$ is a vector of coordinates of the LM translation vector from the origin point. $\mathbf{\mu}$ must be located inside a unit hypercube.

2. Construct a basis $\mathbf{B}$ for a $M$-dimensional linear manifold and its orthogonal complement $\overline{\mathbf{B}}$.
    a) Generate $N+1$ points $X = \[x_1, \dots, x_{N+1}\]$, $x_i \in \mathbb{R}^N$, from a normal Gaussian distribution.
    b) Perform a principle component analysis of generated points using SVD factorization as follows [\[PCA\]][2]:
        - Subtract off the mean for each dimension: $X' = X - \frac{\sum X}{N+1}$
        - Form matrix $Y = \frac{1}{\sqrt{N}}X^T$
        - Perform SVD factorization of matrix $Y = U \Sigma V^T$

    c) Form the basis matrix $\mathbf{B}$ of the linear manifold from first $M$ principal components (columns of matrix $V$).
    d) Form the manifolds basis orthogonal complement matrix $\overline{\mathbf{B}}$ last $N-M$ principal components (columns of matrix $V$).

3. Create a *bounded* linear manifold cluster $C$ of size $n$ using formula [\[1\]](#lmcm) given the translation vector $\mu$, the linear manifold basis $\mathbf{B}$, its orthogonal complement $\overline{\mathbf{B}}$, and bounds $\Phi = \[\Phi_1, \dots, \Phi_M \]$ and $E  = \[E_1, \dots, E_{N-M} \]$. Bounds $\Phi$ and $E$ are vectors that contain intervals  variances (per dimensionlimit the LM cluster by provi characterize  can be scalar values.
    a) Generate $M$-dimensional vector $\mathbf{\phi}$ where $\phi_i \in \sqcup [0, \Phi_i]$, $i=1,\dots,M$.
    b) Generate ($N-M$)-dimensional vector $\mathbf{\varepsilon}$ where $\varepsilon_i \in \sqcup [0,E_i]$, $i=1,\dots,N-M$.
    c) Calculate coordinates of a LM cluster point $x \in \mathbb{R}^N$ using formula [\[1\]](#lmcm).
    d) Check if the point is inside a unit hypercube then add it to the LM cluster, otherwise reject point and continue with the sampling process.
    e) When required number of points in the cluster is generated, return the LM cluster, the manifold basis $\mathbf{B}$ and the translation vector $\mu$.

4. Repeat above steps multiple times until required number of linear manifold clusters are generated. The number of generated clusters can be identified as a vector with specification of manifolds dimensions or as an integer value which will trigger automatic generation of manifold dimensions (see below).

## Generator parameters
- $N$, a dimensionality of a full space where manifolds reside
- $n$, a number of generated LM clusters
- $M_1, \dots, M_n$, a dimensions of generated LM clusters. By default, if dimensions are not specified, first seven clusters will be created with increasing dimensions up to 7 and for the rest of clusters the dimension will not increase.
- $m_1, \dots, m_n$, a number of points per generated LM clusters. By default, all clusters have 1000 points.
- $\tau = \[o_{min}, o_{max}\]$, bounds for a LM cluster origin (translation vector) point generation, default values [0.25, 0.75]
- $\beta$, This parameter can be defined as array of vectors which contains bounds per LM cluster or just vector with bounds used to generate all LM clusters.
- $\Phi$, a variance of a manifold point distribution (if scalar then same variance is used to generate all LM clusters)
- *E*, a variance of distribution for an orthogonal compliment extent of manifold points (if scalar than same variance is used to generate all LM clusters)

There are two modes for generating manifolds:

1. User provides explicitly vector of manifold dimensions, and corresponding bounds for each manifold w.r.t. manifold dimensionality.
2. User provides number of manifolds to generate. In this case, dimensions of manifold calculated incrementally until a manifold dimension is less threshold value which is $N-1$ or *dim_cut* parameter (whichever is less). After dimension threshold is met manifolds will be generated with a dimension equal to the threshold, min($N-1$, *dim_cut*).
- *dim_cut*, a manifold dimensionality bound. Default value is 7.

## Example
```julia
n = 1000               # number of points in generated manifold
N = 10                 # space dimensionality
M = [1,3,4]            # generated manifolds dimensions
τ = [.25, .75]         # translation vector bounds
E = .01                # bound on an extend from manifold  for all manifolds)
Φ = Vector{Float64}[   # bounds on manifold points (per manifold wrt dims)
    [.3],
    [.2,  .3, .5],
    [.1, .25, .4, .2]
]
lmcs = generate(n,N,M,τ,Φ,E)  # generated points as Vector{Vector{Float64}}
```

## References
[Hypersphere Point Picking][1]
[A Tutorial on Principal Component Analysis][2]

[1]: http://mathworld.wolfram.com/HyperspherePointPicking.html "Hypersphere"
[2]: http://arxiv.org/abs/1404.1100 "PCA"