# Defines constructions for Boolean circuits using different random feature maps
using LinearAlgebra
using Random
using Statistics
using DataFrames

export ReLULayer, BoolCirc, BoolFunc, And, Construction, BernoulliUNAnd, SymmTerUNAnd, GaussianUNAnd, RademacherUNAnd, sampleNetwork, readoff, error_var_estimate, error_exp_estimate, construct_superposition_UAND, readoff_vector, cD, sampleSparseVec, errorVarEstimate, errorExEstimate, estimate_error_stats, batch_readoff

struct ReLULayer
    W::Matrix{Float64}
    b::Vector{Float64}
end

relu(x) = max(x, 0)

(layer::ReLULayer)(x::Vector) = relu.(layer.W * x .+ layer.b)

abstract type BoolCirc end
abstract type BoolFunc <: BoolCirc end

struct And{n} <: BoolFunc where {n}
    inds::NTuple{n,Int}
end

(and::And)(x::Vector) = prod(x[i] for i in and.inds)

abstract type Construction end

@kwdef struct BernoulliUNAnd <: Construction
    p::Float64
    n::Integer
    m::Integer
    d::Integer
end

function sampleNetwork((; p, n, m, d)::BernoulliUNAnd)
    bias = zeros(d) .- (n - 1)
    W = rand(d, m) .< p
    return ReLULayer(W, bias)
end

"""
    readoff(::BernoulliUNAnd, layer::ReLULayer, nand::And{n}, s::Integer) where {n}
Read-off function for the UAND constructed via Bernoulli random features.
"""
function readoff(c::BernoulliUNAnd, layer::ReLULayer, nand::And{n}, s::Integer) where {n}
    return readoff(c, layer, nand)
end

function readoff(::BernoulliUNAnd, layer::ReLULayer, nand::And{n}) where {n}
    r = trues(size(layer.W, 1))
    for k in nand.inds
        r .= r .& (layer.W[:, k] .== 1)
    end
    
    return r / sum(r)
end


"""
Symmetric Ternary distribution based UAND construction, which uses weights in {-1, 0, 1} with probabilities (p/2, 1 - p, p/2).
"""
@kwdef struct SymmTerUNAnd <: Construction
    p::Float64
    n::Integer
    m::Integer
    d::Integer
end

function sampleNetwork((; p, n, m, d)::SymmTerUNAnd)
    bias = zeros(d)
    Wber = rand(d, m) .< p
    Wrad = 2 * Wber .- 1
    return ReLULayer(Wrad, bias)
end

@kwdef struct GaussianUNAnd <: Construction
    σ::Float64
    n::Integer
    m::Integer
    d::Integer
end

function sampleNetwork((; σ, n, m, d)::GaussianUNAnd)
    bias = zeros(d)
    W = σ * randn(d, m)
    return ReLULayer(W, bias)
end

using Distributions
using QuadGK


function cD(::GaussianUNAnd, s::Integer)
    if s < 2
        return 0.0
    end
    σ_z = sqrt(s - 2)
    dist = Normal(0, 1)
    
    # Analytical expectation of relu(a + Z) where Z ~ N(0, σ_z^2)
    function h(a)
        if σ_z == 0
            return relu(a)
        end
        u = a / σ_z
        # a * Φ(u) + σ_z * ϕ(u)
        return a * cdf(dist, u) + σ_z * pdf(dist, u)
    end
    
    # Integrand: sgn(w1*w2) * h(w1+w2) * pdf(w1) * pdf(w2)
    # We can exploit symmetry: 
    # Quadrants 1 (++) and 3 (--) give positive sign.
    # Quadrants 2 (-+) and 4 (+-) give negative sign.
    # Also w1, w2 are symmetric.
    
    # Let's just integrate 2D using nested QuadGK for simplicity and robustness
    # Inner integral over w2
    inner(w1) = quadgk(w2 -> sign(w1*w2) * h(w1+w2) * pdf(dist, w2), -Inf, Inf)[1]
    
    # Outer integral over w1
    result = quadgk(w1 -> inner(w1) * pdf(dist, w1), -Inf, Inf)[1]
    
    return result
end

"""
    readoff(::GaussianUNAnd, layer::ReLULayer, nand::And{n}, s::Integer)
Read-off vector for the Gaussian UAND construction.
"""
function readoff(c::GaussianUNAnd, layer::ReLULayer, nand::And{n}, s::Integer) where {n}
    c_val = cD(c, s)
    d = size(layer.W, 1)
    
    w_cols = layer.W[:, collect(nand.inds)]
    prod_w = prod(w_cols, dims=2)[:]
    
    return sign.(prod_w) ./ (d * c_val)
end

"""
Rademacher distribution based UAND construction, which uses weights in {-1, 1} with equal probabilities.
"""
@kwdef struct RademacherUNAnd <: Construction
    n::Integer
    m::Integer
    d::Integer
end
function sampleNetwork((; n, m, d)::RademacherUNAnd)
    bias = zeros(d)
    W = 2 * (rand(d, m) .< 0.5) .- 1
    return ReLULayer(W, bias)
end

function cD(::RademacherUNAnd, s::Integer)
    if s < 2
        return 0.0
    end
    n = s - 2
    # Z = sum of n Rademacher vars. Z = 2k - n where k ~ Bin(n, 0.5)
    dist = Binomial(n, 0.5)
    
    function h(a)
        val = 0.0
        for k in 0:n
            z = 2*k - n
            prob = pdf(dist, k)
            val += relu(a + z) * prob
        end
        return val
    end
    
    # cD = 1/4 * (h(2) - 2h(0) + h(-2))
    return 0.25 * (h(2.0) - 2*h(0.0) + h(-2.0))
end

"""
    readoff(::RademacherUNAnd, layer::ReLULayer, nand::And{n}, s::Integer)
Read-off vector for the Rademacher UAND construction.
"""
function readoff(c::RademacherUNAnd, layer::ReLULayer, nand::And{n}, s::Integer) where {n}
    c_val = cD(c, s)
    d = size(layer.W, 1)
    
    w_cols = layer.W[:, collect(nand.inds)]
    prod_w = prod(w_cols, dims=2)[:]
    
    return prod_w ./ (d * c_val)
end

const readoff_vector = readoff

"""
    sampleSparseVec(m :: Integer, s :: Integer)
Sample a sparse Boolean vector of length m with exactly s non-zero entries.
"""
function sampleSparseVec(m :: Integer, s :: Integer)
    x = zeros(Bool, m)
    inds = randperm(m)[1:s]
    x[inds] .= 1
    return x
end

"""
    errorVarEstimate(uand :: BernoulliUNAnd, sr :: Integer)
Estimate of the variance of the error when reading off an AND from the UAND constructed via Bernoulli random features.
Here sr is the number of non-zero features that are not part of the AND. I.e. let x ∈ {0, 1}^m be the input vector, and let the AND be over indices k₁, k₂. Then sr = Σ_{i ≠ k₁, k₂} x[i].
"""
errorVarEstimate((; p, n, m, d) :: BernoulliUNAnd, sr::Integer) = sr^2 * d^(-1) * (1 - p^2) + sr * (d * p)^(-1) * (1 - p)
# d^(-1) s'^2 (1 - p^2) + s' (d p)^(-1) (1 - p)

"""
    errorVarEstimate(uand :: RademacherUNAnd, sr :: Integer)
Estimate of the variance of the error when reading off an AND from the UAND constructed via Rademacher random features.
"""
function errorVarEstimate(uand::RademacherUNAnd, sr::Integer)
    s = sr + 2
    cD_val = cD(uand, s)
    σD = 1.0 # Variance of Rademacher is 1
    # Var(ε) = s (σD / cD(s))^2 d^(-1)
    return s * (σD / cD_val)^2 * uand.d^(-1)
end

"""
    errorVarEstimate(uand :: GaussianUNAnd, sr :: Integer)
Estimate of the variance of the error when reading off an AND from the UAND constructed via Gaussian random features.
"""
function errorVarEstimate(uand::GaussianUNAnd, sr::Integer)
    (;d ) = uand
    s = sr + 2
    cD_val = cD(uand, s)
    σD = 1.0 # Variance of Standard Normal is 1 (assuming σ=1 in construction for now or normalized)
    # If the construction uses σ != 1, the weights are scaled, but the ratio σD/cD should be invariant if cD scales linearly with σD.
    # Let's assume standard normal for the analytical estimate as per the derivation.
    
    # Var(ε) = s (σD / cD(s))^2 d^(-1)
    return s * (σD / cD_val)^2 * d^(-1)
end

"Estimate of the expectation of the error when reading off a Boolean function from a construction."
errorExEstimate((; p, n, m, d)::BernoulliUNAnd, sr::Integer) = sr * p
errorExEstimate(::RademacherUNAnd, sr::Integer) = 0.0 # Unbiased
errorExEstimate(::GaussianUNAnd, sr::Integer) = 0.0 # Unbiased

"""
    estimate_error_stats(constr::Construction, N_nets::Int, N_vecs::Int, s::Int, b1::Bool, b2::Bool)

Estimates the mean and variance of the error for a given construction and a fixed input bit combination (b1, b2).
Samples random AND gates (random indices) and random input vectors (with fixed b1, b2).
For each vector, a new pair of indices is chosen.
Returns (mean, var, errors).
"""
function estimate_error_stats(constr::Construction, N_nets::Int, N_vecs::Int, s::Int, b1::Bool, b2::Bool)
    m = constr.m
    d = constr.d
    errors = Float64[]
    
    for i_net in 1:N_nets
        layer = sampleNetwork(constr)
        
        # Pre-allocate arrays
        X = zeros(Float64, m, N_vecs)
        targets = zeros(Float64, N_vecs)
        I1 = Vector{Int}(undef, N_vecs)
        I2 = Vector{Int}(undef, N_vecs)
        
        for i_vec in 1:N_vecs
            # Pick random indices for the AND gate
            inds = randperm(m)
            i1, i2 = inds[1], inds[2]
            I1[i_vec] = i1
            I2[i_vec] = i2
            
            # Use fixed AND bits
            # b1, b2 passed as arguments
            
            X[i1, i_vec] = b1
            X[i2, i_vec] = b2
            targets[i_vec] = (b1 && b2) ? 1.0 : 0.0
            
            # Sample remaining m-2 bits with s-2 ones
            other_inds = inds[3:end]
            X[other_inds, i_vec] .= sampleSparseVec(m-2, s - 2)
        end
        
        # Batch forward pass
        activations = relu.(layer.W * X .+ layer.b)
        
        # Batch readoff calculation
        R = batch_readoff(constr, layer, I1, I2, s)
        
        cnn_outputs = vec(sum(R .* activations, dims=1))
        
        current_errors = cnn_outputs .- targets
        append!(errors, current_errors)
    end
    
    return mean(errors), var(errors), errors
end

"""
    batch_readoff(constr::Construction, layer::ReLULayer, I1::Vector{Int}, I2::Vector{Int}, s::Int)

Computes the read-off vectors for a batch of AND gates defined by indices I1 and I2.
Returns a matrix of size (d, N_vecs).
"""
function batch_readoff(constr::BernoulliUNAnd, layer::ReLULayer, I1::Vector{Int}, I2::Vector{Int}, s::Int)
    mask = (layer.W[:, I1] .== 1) .& (layer.W[:, I2] .== 1)
    sums = sum(mask, dims=1)
    safe_sums = replace(sums, 0.0 => 1.0)
    return mask ./ safe_sums
end

function batch_readoff(constr::GaussianUNAnd, layer::ReLULayer, I1::Vector{Int}, I2::Vector{Int}, s::Int)
    d = size(layer.W, 1)
    c_val = cD(constr, s)
    prod_w = layer.W[:, I1] .* layer.W[:, I2]
    return sign.(prod_w) ./ (d * c_val)
end

function batch_readoff(constr::RademacherUNAnd, layer::ReLULayer, I1::Vector{Int}, I2::Vector{Int}, s::Int)
    d = size(layer.W, 1)
    c_val = cD(constr, s)
    prod_w = layer.W[:, I1] .* layer.W[:, I2]
    return prod_w ./ (d * c_val)
end

