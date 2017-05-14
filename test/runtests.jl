using LinearManifoldCluster
using Distributions
using Distances
using LMCLUS
using Base.Test

# random direction
n = 1000
N = 10
pinf, psup = extrema(LinearManifoldCluster.on_ball(N, n))
@test pinf > -1.0
@test psup <  1.0
pinf, psup = extrema(LinearManifoldCluster.in_ball(N, n))
@test pinf > -1.0
@test psup <  1.0
for d in colwise(Euclidean(), LinearManifoldCluster.on_ball(N,n), zeros(N))
    @test_approx_eq d 1.
end

#translation vectors
m = 1000
μ_bounds = [0.25, 0.75]
@test_throws AssertionError LinearManifoldCluster.translations(N, m, [μ_bounds; 0.])
μ = LinearManifoldCluster.translations(N, m, μ_bounds)
# distance from center of unit hypercube to
# any translation vector should be within bounds
for d in colwise(Euclidean(), μ, ones(N)/2.)
    @test μ_bounds[1]/2. <= d <= μ_bounds[2]/2.
end

# basis formation
M = 3
B, B′ = LinearManifoldCluster.lm_basis(N, M)
@test size(B) == (N,M)
@test size(B′) == (N,N-M)
P = B*B'
@test_approx_eq P*P P
@test_approx_eq P' P
P′ = B′*(B′)'
@test_approx_eq P′*P′ P′
@test_approx_eq (P′)' P′


# manifold cluster generator
N = 3
M = 2
t = LinearManifoldCluster.translations(N, 1, μ_bounds)[:]
D = diagm(ones(N))
B = D[:,1:2]
B′ = D[:, 3:3]
Φ = 0.5
E = 0.05

srand(230898573857)
mfld_s = LinearManifoldCluster.generate_cluster(n, t, B, B′, Φ, E)

srand(230898573857)
Φ = fill(0.5, M)
E = fill(0.05, N-M)
mfld = LinearManifoldCluster.generate_cluster(n, t, B, B′, Φ, E)
# one scalar bounds and array bounds return values
@test mfld_s == mfld

# manifold size & space dimension
@test size(mfld) == (N,n)
# if Φ is scalar, then manifold forms a hypercube
# with side equal to Φ (± E)?
mbound_max = norm(fill(Φ[1]+E[1],M),2)
for d in colwise(Euclidean(), mfld.-t, zeros(N))
    @test 0.0 <= d <= mbound_max
end
# if E is scalar, then a distance from any point to
# the manifold should be within ||[ɛ, ...,ɛ]||
mdist_max = norm(E,2)
for d in distance_to_manifold(mfld, t, B)
    @test 0.0 <= d <= mdist_max
end

# manifold dimension generator
@test LinearManifoldCluster.mdims(10,13,7) == [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7]

# multi-manifold generator
n = 1000
N = 3
M = [1,2,2]
τ = [0.25, 0.75]
Φ = Vector{Float64}[   # bounds on manifold points (per manifold wrt dims)
    [.6],
    [.2,  .73],
    [.4, .65]
]
E = Vector{Float64}[   # bound of a point extent (per manifold wrt dims)
    [.05,  .03],
    [.01],
    [.04]
]
lmcs, Bs, μs = generate(n,N,M,τ,Φ,E)
@test length(lmcs) == length(M) # number of manifold
@test size(lmcs[1]) == (N,n)      # number of manifold points
for i in length(Φ)
    @test size(Bs[i]) == (N,length(Φ[i]))  # bases dimensions
end
@test size(μs) == (N,length(M))


# multi-manifold generator (scalar)
n = 1000
N = 3
m = 12
τ = [0.15, 0.75]
κ = 1.5
lmcs, Bs, μs = generate(n,N,m,τ,κ)
@test length(lmcs) == m
