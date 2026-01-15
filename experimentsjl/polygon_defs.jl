# fundamental definitions and functions for the general non-zero bias polygon analysis
using JlCode
using ColorSchemes
using Plots
import Flux
using LinearAlgebra
using Optim
using ForwardDiff
using FastGaussQuadrature
using Polynomials

"B(ρ, γ) = 1/3 γ'^3 + 1/3 (1-ρ)^2 (1 - γ'^3) + (1 - ρ) ρ γ (1 - γ'^2) + (ρ γ)^2 (1 - γ')"
function benefit_term(ρ, γ)
    γ_ = clamp(γ, 0, 1)

    (1 / 3 * γ_^3 + 1 / 3 * (1 - ρ)^2 * (1 - γ_^3) + (1 - ρ) * ρ * γ * (1 - γ_^2) + (ρ * γ)^2 * (1 - γ_))
end

"I_Δi (γ, β) = ρ^2 / (3 c_Δi) (c_Δi - γ)^3 - ρ^2 / (3 c_Δi) (c_Δi γ_Δi' - γ)^3"
function interference_term(ρ, γ, c_Δi) # note - invalid for positive biases ⇔ negative γ
    γ_Δi_ = clamp(γ / c_Δi, 0, 1)
    ρ^2 / (3 * c_Δi) * (c_Δi - γ)^3 - ρ^2 / (3 * c_Δi) * (c_Δi * γ_Δi_ - γ)^3
end

function l1_loss_derived(; r, β, k, nf)
    ρ = r^2
    γ = -β / r^2
    Δφ = 2π / nf
    return nf * (benefit_term(ρ, γ) +
                 2 * sum(interference_term(ρ, γ, cos(Δi * Δφ)) for Δi in 1:k; init=0.))
end

"
Calculate the unidirectional number of neighbors (k) for a given γ and nf.
An interference term contributes if γ / c_Δi ≤ 1, i.e., if c_Δi ≥ γ; I.e. k is the smallest integer such that cos(Δi * 2π / nf) ≤  γ
"
k_of_nf(γ, nf) = findfirst(Δi -> cos(Δi * 2π / nf) ≤ γ, 1:ceil(Int64, nf / 2)) - 1


relu(x) = max(0, x)

"""
  L  = Ex_Psparse [sum_i I_i (hat(X)_i - X_i)^2] = sum_k^(nf) binom(nf, k) (1-S)^k S^(nf - k) L_k, 
  
  L_1 = sum_i underbrace(I_i Ex[(Y - relu(n2(w_i)² Y + b_i))^2], "Feature Benefit") + sum_(i ≠ j) underbrace(I_j Ex[relu(w_j ⋅ w_i Y + b_j)^2], "Interference").
"""

function integrate_trapezoid(f, a, b, n)
    h = (b - a) / n
    sum = 0.5 * (f(a) + f(b))
    for i in 1:(n-1)
        sum += f(a + i * h)
    end
    return sum * h
end

function integrate_gauss_legendre(f, a, b, n)
    nodes, weights = gausslegendre(n)
    # Transform from [-1,1] to [a,b]
    transformed_nodes = 0.5 * (b - a) * nodes .+ 0.5 * (b + a)
    return 0.5 * (b - a) * sum(weights .* f.(transformed_nodes))
end

function integrate(f, a, b, n; method=:gauss_legendre)
    return if method == :gauss_legendre
        integrate_gauss_legendre(f, a, b, n)
    elseif method == :trapezoid
        integrate_trapezoid(f, a, b, n)
    else
        error("Unknown integration method: $method")
    end
end

function polygon_matrix(nf, radius=1.0)
    Δφ = 2 * π / nf
    W = cat(([cos(Δφ * i), sin(Δφ * i)] for i in 0:(nf-1))..., dims=2) * radius
    return W
end

Ex(f; n_integration=50, integration_method=:gauss_legendre) = integrate(f, 0, 1, n_integration; method=integration_method)

function l1_loss_extended(W, b; n_integration=50, integration_method=:gauss_legendre)
    _Ex(f) = Ex(f; n_integration=n_integration, integration_method=integration_method)
    nf = size(W, 2)
    feature_benefit = _Ex(y -> sum((y - relu(y * norm(W[:, i])^2 + b[i]))^2 for i in 1:nf))
    interference(y, i, j) = relu(dot(W[:, j], W[:, i]) * y + b[j])^2
    interference = _Ex(y -> sum(interference(y, i, j) for j in 1:nf for i in 1:nf if j != i; init=0.))
    return feature_benefit + interference, (
        feature_benefit=feature_benefit,
        interference=interference
    )
end

l1_loss(W, b; n_integration=50, integration_method=:gauss_legendre) = l1_loss_extended(W, b; n_integration=n_integration, integration_method=integration_method)[1]

"""
    find_polygon_solution_γ_ρ(nf; initial_radius = 1.0, initial_bias = -0.)

Find a polygon solution for the given number of features `nf` by optimizing the parameters (ρ, γ).
This function optimizes the polygon solution by minimizing the L₁ loss over the parameters ρ and γ.
"""
function find_polygon_solution(nf; initial_radius=1.0, initial_bias=-0., loss_type=:general_derived, k=nothing, n_integration = 50)
    # Objective function: minimize L₁ loss over (radius, β)
    function general_derived_objective((r, β))
        W = polygon_matrix(nf, r)
        b = fill(β, nf)
        gen_l1_loss_der(W, b)
    end
    function numerical_objective((r, β))
        W = polygon_matrix(nf, r)
        b = fill(β, nf)
        l1_loss(W, b; n_integration=n_integration)
    end
    function derived_objective((r, β))
        ρ = r^2
        γ = -β / ρ
        k = k === nothing ? k_of_nf(γ, nf) : k
        l1_loss_derived(; r=r, β=β, k=k, nf=nf)
    end

    # Optimize with custom tolerances and iteration limits
    options = Optim.Options(
        g_abstol=1e-8,           # Gradient tolerance (stopping when |∇f| < g_tol)
        f_abstol=1e-8,          # Function tolerance (stopping when |f_k - f_{k-1}| < f_tol) 
        x_abstol=1e-8,          # Parameter tolerance (stopping when |x_k - x_{k-1}| < x_tol)
        iterations=1000,      # Maximum number of iterations
        show_trace=false,     # Set to true to see optimization progress
        store_trace=false     # Set to true to store optimization history
    )

    objective =
        if loss_type == :general_derived
            general_derived_objective
        elseif loss_type == :numeric
            numerical_objective
        elseif loss_type == :derived
            derived_objective
        else
            error("Unknown loss type: $loss_type. Use :numeric or :derived or :general_derived.")
        end

    result = optimize(objective, [initial_radius, initial_bias], BFGS(), options)

    # Extract solution
    r_opt, β_opt = result.minimizer
    # Compute Hessian at optimum
    _hessian = ForwardDiff.hessian(objective, [r_opt, β_opt])

    if !isinteger(nf)
        return (
            radius=r_opt,
            bias=β_opt,
            loss=result.minimum,
            hessian=_hessian,
            is_minimum=isposdef(_hessian),
        )
    else # Ensure nf is an integer
        nf = round(Int, nf)
    end

    W_opt = polygon_matrix(nf, r_opt)
    b_opt = fill(β_opt, nf)

    return (
        radius=r_opt,
        bias=β_opt,
        loss=result.minimum,
        W=W_opt,
        b=b_opt,
        hessian=_hessian,
        is_minimum=isposdef(_hessian)
    )
end

function train_full_model(W_init, b_init; n_steps=100, return_trajectory::Bool=false)
    """
        Train the full model using BFGS optimization on all parameters (W, b).
        If return_trajectory=true, also return the full optimization trajectory (W, b at each step).
        Returns: (model, loss, hessian, is_minimum, [trajectory])
    """
    nf = size(W_init, 2)
    params0 = vcat(vec(W_init), b_init)
    function objective(params)
        W = reshape(params[1:2*nf], 2, nf)
        b = params[2*nf+1:end]
        gen_l1_loss_der(W, b)
    end
    options = Optim.Options(
        g_abstol=1e-8,
        f_abstol=1e-8,
        x_abstol=1e-8,
        iterations=n_steps,
        show_trace=false,
        store_trace=return_trajectory
    )
    initial_hessian = ForwardDiff.hessian(objective, params0)
    initial_model = CoupledToyModel(W_init, b_init)
    result = optimize(objective, params0, BFGS(), options)
    params_opt = result.minimizer
    W_opt = reshape(params_opt[1:2*nf], 2, nf)
    b_opt = params_opt[2*nf+1:end]
    final_hessian = ForwardDiff.hessian(objective, params_opt)
    final_model = CoupledToyModel(W_opt, b_opt)
    final_loss = result.minimum
    is_minimum = isposdef(final_hessian)
    return_tpl = (; initial_hessian, initial_model, final_model, final_loss, final_hessian, is_minimum)
    if return_trajectory && hasproperty(result, :trace) && !isempty(result.trace)
        xs = [t.metadata.x for t in result.trace]
        Ws = [reshape(x[1:2*nf], 2, nf) for x in xs]
        bs = [x[2*nf+1:end] for x in xs]
        return_tpl = (
            ; return_tpl...,
            trajectory=(Ws, bs)
        )
    end
    return return_tpl
end

# we'd like to now more about the solution; e.g.:
# - whether it is a minimum of the L₁ loss (i.e., there is no descent direction)
# - whether there is a lower order polygon solution with a lower loss
# - number of active neighbors, i.e. neighbors that contribute to interference
# - and whether the Hessian is positive semidefinite (we cannot expect it to be positive definite, as we can always rotate the polygon, which keeps the loss constant)

struct CoupledToyModel
    W::Matrix{Float64}
    b::Vector{Float64}
end

loss(model::CoupledToyModel) = l1_loss(model.W, model.b)

function num_interfering_neighbors(model::CoupledToyModel, i_weight; ε=1e-5)
    # Count the number of neighbors that contribute to interference for the i-th weight
    count = 0
    for j in 1:size(model.W, 2)
        if j != i_weight && dot(model.W[:, j], model.W[:, i_weight]) + model.b[j] > ε
            count += 1
        end
    end
    return count
end

"""
The more general derived L₁ loss for arbitrary W, b
"""
gen_l1_loss_der(W, b) = gen_l1_loss_extended(W, b)[1]

function benefit_term(W, b, i)
    ρi = dot(W[:, i], W[:, i])
    ρi = ρi == 0 ? 1e-10 : ρi
    γi = clamp(-b[i] / ρi, 0, 1)
    return γi^3 / 3 + 1 / 3 * (1 - ρi)^2 * (1 - γi^3) - b[i] * (1 - ρi) * (1 - γi^2) + b[i]^2 * (1 - γi)
end
function interference_term(W, b, i, j)
    large_number = 10^10
    ρij = dot(W[:, i], W[:, j])
    valid_y_range = clamp(interval(-b[j], large_number) / ρij, 0, 1)
    return eval_on_boundary((y -> 1 / 3 * ρij^2 * y^3 + ρij * b[j] * y^2 + b[j]^2 * y), valid_y_range)
end

function gen_l1_loss_extended(W, b)
    nf = size(W, 2)
    benefit = sum(benefit_term(W, b, i) for i in 1:nf)
    interference = sum(interference_term(W, b, i, j) for i in 1:nf for j in 1:nf if i != j)
    return benefit + interference, (benefit, interference)
end