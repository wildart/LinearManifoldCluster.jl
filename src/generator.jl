"""Generates *n* points at random on a *N*-dimensional ball."""
function on_ball(N::Int, n::Int)
    p = rand(Normal(), N, n)
    r = sqrt.(sum(abs2, p, dims=1))
    return p./r
end

"""Generates *n* points at random in a *N*-dimensional ball."""
function in_ball(N::Int, n::Int)
    X = reshape(rand(Normal(), n*N), N, n)
    Y = rand(Poisson(), n)
    r = sqrt.(Y' .+ sum(abs2, X, dims=1))
    return X./r
end

"""Generates *m* random *N*-dimensional translation vectors with *μ_bounds* bounds"""
function translations(N::Int, n::Int, μ_bounds::Tuple{Float64,Float64})
    o_min, o_max = μ_bounds
    Dτ = Uniform(o_min, o_max)
    o = on_ball(N, n)
    τ = rand(Dτ, n)
    return (o.*τ' .+ fill(1.0,N,n))./2.
end

"""Generates manifold basis and its complement"""
function lm_basis(N::Int, M::Int)
    @assert N > M
    X = rand(Normal(),N,N+1)
    Y = (X.-mean(X, dims=2))'./sqrt(N)
    F = svd(Y)
    F.V[:,1:M], F.V[:,(M+1):end]
end

"""Generates *n* points of a *M*-dimensional linear manifold cluster
situated in a unit hypercube."""
function generate_cluster(
                  n::Int,                   # number of points in generated manifold
                  μ::Vector{Float64},       # manifold translation vector
                  B::Matrix{Float64},       # manifold basis
                  B′::Matrix{Float64},      # orthogonal complement to manifold basis
                  DΦ::Vector{T},  # bound of manifold points
                  DE::Vector{T}   # bound of a point extent from a manifold
                ) where T<:Distribution
    N, M = size(B)
    @assert size(B′) == (N, N-M) "NULL space dimention must be $(N-M)"
    @assert length(DΦ) == M
    @assert length(DE) == N-M

    mfld = zeros(N,n)
    c = 1
    while c <= n
        ϕ = map(d->rand(d), DΦ)
        ɛ = map(d->rand(d), DE)
        x = μ + B*ϕ + B′*ɛ
        # check if manifold point outside of unit hypercube
        b = extrema(x)
        if b[1] < 0 || b[2] > 1.
            continue
        end
        mfld[:,c] = x
        c+=1
    end
    return mfld
end

"""Generates *n* points of a *M*-dimensional linear manifold cluster
situated in a unit hypercube."""
function generate_cluster(
                  n::Int,                   # number of points in generated manifold
                  μ::Vector{Float64},       # manifold translation vector
                  B::Matrix{Float64},       # manifold basis
                  B′::Matrix{Float64},      # orthogonal complement to manifold basis
                  Φ::Vector{Float64},       # bound of manifold points
                  E::Vector{Float64}        # bound of a point extent from a manifold
                )
    DΦ = map(ϕ->Uniform(0., ϕ), Φ)
    DE = map(ɛ->Uniform(0., ɛ), E)
    return generate_cluster(n, μ, B, B′, DΦ, DE)
end

"""Generates *n* points of a *M*-dimensional linear manifold cluster
situated in a unit hypercube."""
function generate_cluster(
                  n::Int,                   # number of points in generated manifold
                  μ::Vector{Float64},       # manifold translation vector
                  B::Matrix{Float64},       # manifold basis
                  B′::Matrix{Float64},      # orthogonal complement to manifold basis
                  Φ::Float64,               # bound of manifold points
                  E::Float64                # bound of a point extent from a manifold
                )
    N, M = size(B)
    DΦ = fill(Uniform(0., Φ), M)
    DE = fill(Uniform(0., E), N-M)
    return generate_cluster(n, μ, B, B′, DΦ, DE)
end

""" Generates *m* linear manifold clusters
where *m* is size of parameter *M* """
function generate(n::Int,                     # number of points in generated manifold
                  N::Int,                     # space dimensionality
                  M::Vector{Int},             # generated manifolds dimensions
                  τ::Tuple{Float64,Float64},   # translation vector bounds
                  Φ::Vector{Vector{Float64}}, # bound of manifold points
                  E::Vector{Vector{Float64}}  # bound of a point extent from a manifold
                )
    m = length(M) # number of manifolds to generate

    @assert ( length(E) == m && length(Φ) == m ) "Bounds should correspond to number of generated manifolds"

    μ = translations(N, m, τ)
    bases = Matrix{Float64}[]
    manifolds = Matrix{Float64}[]
    for i in 1:m
        B, B′ = lm_basis(N, M[i])
        mfld = generate_cluster(n, μ[:,i], B, B′, Φ[i], E[i])
        push!(bases, B)
        push!(manifolds, mfld)
    end
    return manifolds, bases, μ
end

"""Generate manifold dimensions."""
function mdims(N, m, dim_cut::Int = 7)
    M = cumsum(ones(Int,m))
    if m >= N
        n = N >= dim_cut ? dim_cut : N
        M[n:end] .= n
    end
    return M
end

""" Generates *m* linear manifold clusters
where *m* is size of parameter *M* """
function generate(n::Int,                     # number of points in generated manifold
                  N::Int,                     # space dimensionality
                  M::Vector{Int},             # number of manifolds
                  τ::Tuple{Float64,Float64},   # translation vector bounds
                  κ::Float64
                )
    m = length(M)


    # generate bounds
    D = Uniform(0.2, 0.99); rand(D, 1000);
    ΔΦ = [sort(rand(D, 2))::Vector{Float64} for i in 1:m]
    ΔE = [sort(rand(D, 2)*diff(mean(ΔΦ))[1]/κ)::Vector{Float64} for i in 1:m]

    Φ = Vector{Float64}[]
    E = Vector{Float64}[]
    for i in M
        DΦ = Uniform(ΔΦ[i][1], ΔΦ[i][2]); rand(DΦ, 1000);
        DE = Uniform(ΔE[i][1], ΔE[i][2]); rand(DE, 1000);
        push!(Φ, rand(DΦ, M[i]))
        push!(E, rand(DE, N-M[i]))
    end
    return generate(n, N, M, τ, Φ, E)
end

""" Generates *m* linear manifold clusters"""
function generate(n::Int,                     # number of points in generated manifold
                  N::Int,                     # space dimensionality
                  m::Int,                     # number of manifolds
                  τ::Pair{Float64,Float64},   # translation vector bounds
                  κ::Float64;
                  dim_cut::Int = 7
                )
    # generate manifold dimensions
    M = mdims(N-1,m,dim_cut)
    return generate(n, N, M, τ, κ)
end

function noise(N::Int, n::Int)
    return rand(N, n)
end
