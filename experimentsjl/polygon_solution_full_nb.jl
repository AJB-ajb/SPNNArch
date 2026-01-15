# %%
using JlCode
using ColorSchemes
using Plots
import Flux
using LinearAlgebra
using Optim
# using ForwardDiff
using FastGaussQuadrature
using Polynomials
using StatsPlots, DataFrames, CSV
using Statistics
using ProgressMeter: @showprogress
using CategoricalArrays
using PrettyTables, Crayons
using Measures
using Test
import ReverseDiff
import ForwardDiff

include("polygon_defs.jl")

begin # start investigation
    logger = ExpLogger("polygon_solution"; prefix="analysis_v2-0", tmp = true)

    # we'd like to find all nontrivial solutions for polygons in nfs
    # a solution is nontrivial if some relu is not zero; i.e. r² + b > 0
    
    starting_biases = range(-0.99, 2.0, length = 8)
    starting_radii = [0.9, 1.0, 1.5, 2.0, 4.]
    println("Starting biases: ", collect(starting_biases))

    # We'd like to group the solutions (grouping the same solutions together)

    # now store them in a dataframe
    
    # Store solutions for later use
    nfs = 3:20
    n_integration = 256
    function compute_solutions(nfs, starting_biases, starting_radii)
        solutions = []
        df_solutions = DataFrame(
            nf = Int[],
            radius = Float64[],
            bias = Float64[],
            ρ = Float64[],
            γ = Float64[],
            loss = Float64[],
            is_rb_minimum = Bool[],
            num_neighbors = Int[],
            is_trivial = Bool[]
        )
        @showprogress for nf in nfs
            println("\nFinding solutions for nf = $nf... ")
            for bias in starting_biases
                for initial_radius in starting_radii
                sol = find_polygon_solution(nf; initial_radius = initial_radius, initial_bias = bias, n_integration)
                push!(solutions, sol)
                ρ, γ = sol.radius^2, -sol.bias / sol.radius^2
                push!(df_solutions, (nf, sol.radius, sol.bias, ρ, γ, sol.loss, isposdef(sol.hessian), num_interfering_neighbors(CoupledToyModel(sol.W, sol.b), 1), sol.radius^2 + sol.bias ≤ 0))
                end
            end
        end
        return solutions, df_solutions
    end

    solutions, df_solutions = compute_solutions(nfs, starting_biases, starting_radii)
    
    display_df_colorbg(df_solutions)

end
begin
# remove the trivial and negative radius solutions
    function filter_valid_solutions!(df; eps=1e-4)
        # Keep positive radius, non-trivial solutions (r^2 + b > -eps),
        # and γ within [-eps, 1+eps]. Mutates df in place.
        filter!(row -> (row.radius > 0) &&
                       ((row.radius^2 + row.bias) > -eps) &&
                       (row.γ >= -eps) && (row.γ <= 1 + eps),
                df)
        return df
    end
    filter_valid_solutions!(df_solutions)
end

begin 
    # find solutions of the analytical loss, setting nf to continuous values for fixed k
    nfs_cont = 3:0.1:maximum(nfs)
    ks = [0, 1, 2, 3] # number of one-sided neighbors to consider

    function compute_derived_solutions(nfs, ks; starting_biases, starting_radii)
        df_solutions_cont = DataFrame(
            nf=Float64[],
            radius=Float64[],
            bias=Float64[],
            ρ=Float64[],
            γ=Float64[],
            k_defined=Int[],
            loss=Float64[],
            is_rb_minimum=Bool[], # whether the solution is a minimum of the derived loss L₁(r,b) — this is not sufficient to be a minimum of the original loss, but necessary
            min_rb_eigenval=Float64[] # minimum eigenvalue of the Hessian of the derived loss
        )
        
        @showprogress for nf in nfs
            for k in ks, initial_bias in starting_biases, initial_radius in starting_radii
                # Calculate the solution using the derived objective
                sol = find_polygon_solution(nf; initial_radius = initial_radius, initial_bias = initial_bias, loss_type = :derived, k = k)
                ρ, γ = sol.radius^2, -sol.bias / sol.radius^2
                min_rb_eigenval = minimum(real.(eigvals(Symmetric(sol.hessian))))

                push!(df_solutions_cont, (nf, sol.radius, sol.bias, ρ, γ, k, sol.loss, min_rb_eigenval > 1e-9, min_rb_eigenval))
            end
        end
        return df_solutions_cont
    end
    df_solutions_cont = compute_derived_solutions(nfs_cont, ks; starting_biases = starting_biases, starting_radii = starting_radii)
    filter_valid_solutions!(df_solutions_cont)
end

function make_unique(df; tol=1e-3)
    unique_rows = trues(nrow(df))
    for i in 1:nrow(df)-1
        if !unique_rows[i]
            continue
        end
        for j in i+1:nrow(df)
            if abs(df.radius[i] - df.radius[j]) < tol &&
               abs(df.bias[i] - df.bias[j]) < tol && df.nf[i] == df.nf[j]
                unique_rows[j] = false
            end
        end
    end
    return df[unique_rows, :]
end

begin # Eliminate (group) nearly identical solutions: keep only unique (radius, bias) pairs within tolerance
    df_solutions_unique = make_unique(df_solutions)
    df_cont_solutions_unique = make_unique(df_solutions_cont)
    println("Unique solutions:")
    display_df_colorbg(df_solutions_unique)
end

begin
    function augment_full_loss_eigenvals!(df)
        df.eigenvalues = fill(Vector{Float64}(), nrow(df))
        df.min_eigenval = fill(NaN, nrow(df))
        df.gradient_norm = fill(NaN, nrow(df))

        @showprogress for i in 1:nrow(df)
            isinteger(df.nf[i]) || continue
            println("Evaluating Hessian for row $i with nf=$(df.nf[i]), radius=$(df.radius[i]), bias=$(df.bias[i])")

            nf = round(Int64, df.nf[i])
            r = df.radius[i]
            b = fill(df.bias[i], nf)
            W = polygon_matrix(nf, r)
            params = vcat(vec(W), b)
            obj(p) = gen_l1_loss_der(reshape(p[1:length(vec(W))], size(W)), p[length(vec(W))+1:end])
            gradient_norm = norm(ReverseDiff.gradient(obj, params))
            eigs = eigvals(Symmetric(ForwardDiff.hessian(obj, params)))
            df.eigenvalues[i] = real.(eigs)
            df.min_eigenval[i] = minimum(real.(eigs))
            df.gradient_norm[i] = gradient_norm
        end
        epsilon = 1e-5
        df.unstable = df.min_eigenval .< -epsilon
        return df
    end

    augment_full_loss_eigenvals!(df_solutions_unique)

    # the same for the continuous solutions
    augment_full_loss_eigenvals!(df_cont_solutions_unique)
end

begin
    # calculate the minimum loss solutions for each nf
    # We can do this by grouping by nf and taking the minimum loss
    df_solutions_min_loss = combine(groupby(df_solutions_unique, [:nf])) do sdf
        sdf[argmin(sdf.loss), :]
    end

    display_df_colorbg(df_solutions_min_loss)
end

function colorize(df::DataFrame, col::Symbol, palette)
    #! return the color assignment for the values, for the legend
    
    vals = CategoricalArray(df[!, col])
    n_palette = length(palette)
    if n_palette < length(vals.pool)
        @warn "Not enough colors in palette for unique values in $col. Using modulo to cycle through colors."
    end
    return [palette[1 + (val.ref-1) % n_palette] for val in vals]
end

begin # Plot ρ = r² and negative bias vs nf, grouped by num_neighbors
    # Overlay minimum-loss solutions as a line on both plots

    plot_df = copy(df_solutions_unique)
    # Filter out all trivial solutions: keep only those with γ ≤ 1
    plot_df = filter(row -> row.γ ≤ 1, df_solutions_unique)

    plot_df.group_label = ["""neighbors=$(n), $(u ? "unstable" : "")""" for (n, u) in zip(plot_df.num_neighbors, plot_df.unstable)]
    plot_df.marker_shape = [u ? :x : :circle for u in plot_df.unstable]
    # choose the color based on the number of neighbors
    plot_df.color = colorize(plot_df, :num_neighbors, color_palette)
    

    rho_plot = @df plot_df scatter(:nf, :ρ, xlabel="Number of Features", ylabel="ρ = r²", legend=false, marker=:marker_shape, color=:color, label = nothing)

    bias_plot = @df plot_df scatter(:nf, -:bias, xlabel="Number of Features", ylabel="-β", legend=:topright, color = :color, marker=(:marker_shape), label = nothing)

    # Add legend color entries for discrete number-of-neighbors (k) similar to the continuous plot
    k_vals = unique(sort(plot_df.num_neighbors)) .÷ 2
    for (i, k) in enumerate(k_vals)
        c = color_palette[1 + mod(i-1, length(color_palette))]
        scatter!(bias_plot, [NaN], marker=:circle, label="k=$(k)", color=c, markersize=6)
    end

    # add a plot for the γ — enable legend so neighbor lines can be shown; keep other elements unlabeled
    γ_plot = @df plot_df scatter(:nf, :γ, xlabel="Number of Features", ylabel="γ", legend=:topleft, marker=:marker_shape, color=:color, label = nothing, ylims = (-0.1, 1.2))
    # add the neighbor lines (i.e., where jumps of neighbors happen)
    k_neighbor_line(nf, k) = 1. * cos(k * 2π / nf)
    for k in 1:3
        @df plot_df plot!(γ_plot, :nf, k_neighbor_line.(:nf, k), color=:black, linestyle=:dash, label="Neighbor Line $k")
    end

    # additionally plot the average neighbor line between k = 0, 1 and k = 1,2
    for k in 1:2
        @df plot_df plot!(γ_plot, :nf, (k_neighbor_line.(:nf, k) + k_neighbor_line.(:nf, k+1)) / 2, color=:violet, linestyle=:dash, label="Mid-neighbor line between $k and $(k+1)")
    end

    # manually adding the legend into the bias plot by plotting empty data
    # marker shape: plot :circle for stable, :x for unstable
    scatter!(bias_plot, [NaN], marker=:circle, label="Hessian ≥ 0", color=:white, markersize=4)
    scatter!(bias_plot, [NaN], marker = :x, label="Unstable (Saddle)", color=:black, markersize=4)

    p = plot(rho_plot, bias_plot, layout=(1, 2), size = (1200, 500), bottom_margin = 5mm, left_margin = 5mm)
    display(p)
    save_figure(logger, p, "ρβ_discrete")

    save_figure(logger, γ_plot, "γ_discrete")
    display(γ_plot)

    #! γ plot
    
end
display_df_colorbg(plot_df)

begin # analogous plots of the continuous, derived loss - now we plot over continuous nf

    # Create analogous plots for continuous solutions
    plot_df_cont = copy(df_cont_solutions_unique)
    # Filter out trivial solutions: keep only those with γ ≤ 1 (equivalent to γ ≤ 1)
    plot_df_cont = filter(row -> row.γ ≤ 1 && row.γ > 0, plot_df_cont)

    # Add marker shapes and colors based on k_defined and stability
    plot_df_cont.group_label = ["""k=$(k), $(m ? "min" : "")""" for (k, m) in zip(plot_df_cont.k_defined, plot_df_cont.is_rb_minimum)]
    plot_df_cont.marker_shape = [m ? :circle : :x for m in plot_df_cont.is_rb_minimum]
    # Choose color based on k_defined
    plot_df_cont.color = colorize(plot_df_cont, :k_defined, [:red, :blue, :green, :orange, :purple])

    # also add a column for cos(1 * 2π / nf)
    plot_df_cont.c_1 = cos.(2π ./ plot_df_cont.nf) # note that this is prob. only useful for k = 1

    # Find minimum loss solutions for each nf (continuous)
    plot_df_cont_min_loss = combine(groupby(plot_df_cont, [:nf])) do sdf
        sdf[argmin(sdf.loss), :]
    end

    #! overlay the integer solutions from the continuous dataframe and plot as square if stable, star if unstable 
    

    # Create the three plots
    # We'll draw rho and bias as connected line plots per k_defined. To avoid connecting large jumps
    # we split each k-group into segments where consecutive points differ by more than thresholds.
    nf_gap = 0.9       # gap in nf to break a line (continuous nf steps are ~0.1)
    rho_gap = 0.5      # absolute gap in ρ to break a line
    bias_gap = 0.5     # absolute gap in -β to break a line

    rho_plot_cont = plot(xlabel="Number of Features (continuous)", ylabel="ρ = r²", legend=false, alpha=0.8)
    bias_plot_cont = plot(xlabel="Number of Features (continuous)", ylabel="-β", legend=:topright, alpha=0.8)

    # Choose colors per k_defined consistently with earlier palette
    k_values = unique(sort(plot_df_cont.k_defined))
    base_colors = color_palette
    color_map = Dict(k_values[i] => base_colors[1 + mod(i-1, length(base_colors))] for i in 1:length(k_values))

    for k in k_values
        sub = plot_df_cont[plot_df_cont.k_defined .== k, :]
        if nrow(sub) == 0
            continue
        end
        order = sortperm(sub.nf)
        xs = sub.nf[order]
        ys_rho = sub.ρ[order]
        ys_beta = -sub.bias[order]

        # split into contiguous segments based on gaps
        idxs = collect(1:length(xs))
        seg_start = 1
        for i in 2:length(xs)
            if (xs[i] - xs[i-1] > nf_gap) || (abs(ys_rho[i] - ys_rho[i-1]) > rho_gap) || (abs(ys_beta[i] - ys_beta[i-1]) > bias_gap)
                # plot the previous segment
                seg_range = seg_start:(i-1)
                if length(seg_range) >= 2
                    plot!(rho_plot_cont, xs[seg_range], ys_rho[seg_range], color=color_map[k], lw=2, label=nothing)
                    plot!(bias_plot_cont, xs[seg_range], ys_beta[seg_range], color=color_map[k], lw=2, label=nothing)
                end
                seg_start = i
            end
        end
        # final segment
        seg_range = seg_start:length(xs)
        if length(seg_range) >= 2
            plot!(rho_plot_cont, xs[seg_range], ys_rho[seg_range], color=color_map[k], lw=2, label=nothing)
            plot!(bias_plot_cont, xs[seg_range], ys_beta[seg_range], color=color_map[k], lw=2, label=nothing)
        end
    end

    # Plot γ (γ equivalent) vs nf — start with an empty plot so continuous solutions are drawn as lines only
    gamma_plot_cont = plot(xlabel="Number of Features (continuous)", ylabel="γ", legend=false, ylims=(-0.1, 1.2), alpha=0.7)
    #hline!(gamma_plot_cont, [1.0], color=:black, linestyle=:dash, label="Trivial Solution Line")
    
    # Add neighbor lines for continuous case
    k_neighbor_line_cont(nf, k) = cos(k * 2π / nf) * (cos(k * 2π / nf) > 0)
    nf_range = range(minimum(plot_df_cont.nf), maximum(plot_df_cont.nf), length=100)
    for k in 0:3
        plot!(gamma_plot_cont, nf_range, k_neighbor_line_cont.(nf_range, k), color=:black, linestyle=:dash, label="Neighbor Line $k")
    end
    
    # Add mid-neighbor lines
    for k in 1:2
        plot!(gamma_plot_cont, nf_range, (k_neighbor_line_cont.(nf_range, k + 0.5)), color=:violet, linestyle=:dash, label="Mid-neighbor line between $k and $(k+1)")
    end

    # Draw continuous γ solution lines per k_defined (connect points into segments like rho/bias plots)
    gamma_gap = 0.2
    for k in k_values
        sub = plot_df_cont[plot_df_cont.k_defined .== k, :]
        if nrow(sub) == 0
            continue
        end
        order = sortperm(sub.nf)
        xs = sub.nf[order]
        ys_gamma = sub.γ[order]

        # split into contiguous segments based on gaps (reuse nf_gap defined for rho/bias)
        seg_start = 1
        for i in 2:length(xs)
            if (xs[i] - xs[i-1] > nf_gap) || (abs(ys_gamma[i] - ys_gamma[i-1]) > gamma_gap)
                seg_range = seg_start:(i-1)
                if length(seg_range) >= 2
                    plot!(gamma_plot_cont, xs[seg_range], ys_gamma[seg_range], color=color_map[k], lw=2, label=nothing)
                end
                seg_start = i
            end
        end
        # final segment
        seg_range = seg_start:length(xs)
        if length(seg_range) >= 2
            plot!(gamma_plot_cont, xs[seg_range], ys_gamma[seg_range], color=color_map[k], lw=2, label=nothing)
        end
    end

    # Add legend entries for k values (colors)
    for k in unique(sort(plot_df_cont.k_defined))
        scatter!(bias_plot_cont, [NaN], marker=:circle, label="k=$k", color=color_map[k], markersize=6)
    end

    # Also overlay the discrete (integer) solutions computed earlier (plot_df) so both views are available.
    if @isdefined(plot_df) && nrow(plot_df) > 0
        discrete_df = filter(row -> row.γ ≤ 1 && row.γ ≥ -1e-5, plot_df)
        if nrow(discrete_df) > 0
            @df discrete_df scatter!(rho_plot_cont, :nf, :ρ, marker=:marker_shape, color=:color, ms=6, alpha=0.9, label=nothing)
            @df discrete_df scatter!(bias_plot_cont, :nf, -:bias, marker=:marker_shape, color=:color, ms=6, alpha=0.9, label=nothing)
            @df discrete_df scatter!(gamma_plot_cont, :nf, :γ, marker=:marker_shape, color=:color, ms=6, alpha=1.0, label=nothing)
            # legend entries for discrete marker meaning
            scatter!(bias_plot_cont, [NaN], marker=:circle, label="Hessian PSD", color=:black, markersize=6)
            scatter!(bias_plot_cont, [NaN], marker=:x, label="Hessian non-PSD", color=:black, markersize=6)
        end
    end

    #! now plot the gammas, but with the color based on min eigenvalue (blue for negative red for positive on continuous blue-red scale )
    # Gamma vs nf plot colored by min_rb_eigenval
    vals = plot_df_cont.min_rb_eigenval
    vmin, vmax = minimum(vals), maximum(vals)
    norm_vals = (vals .- vmin) ./ (vmax - vmin + 1e-12)
    # Use a valid blue-red colormap, e.g. :RdBu or :balance
    colors_eig = [get(ColorSchemes.RdBu, nv) for nv in norm_vals]

    gamma_eig_plot = @df plot_df_cont scatter(:nf, :γ,
        xlabel="Number of Features (continuous)",
        ylabel="γ (continuous)",
        legend=false,
        marker=:circle,
        color=colors_eig,
        label=nothing,
        ylims=(-0.1, 1.2),
        alpha=0.8)
    hline!(gamma_eig_plot, [1.0], color=:black, linestyle=:dash, label="Trivial Solution Line")
    for k in 1:3
        plot!(gamma_eig_plot, nf_range, k_neighbor_line_cont.(nf_range, k), color=:black, linestyle=:dash, label="Neighbor Line $k")
    end
    for k in 0:2
        plot!(gamma_eig_plot, nf_range, (k_neighbor_line_cont.(nf_range, k + 0.5)), color=:violet, linestyle=:dash, label="Mid-neighbor line between $k and $(k+1)")
    end
    display(gamma_eig_plot)

    p_ρβγ_cont = plot(rho_plot_cont, bias_plot_cont, gamma_plot_cont, layout=(1, 3), size=(1200, 400), bottom_margin = 5mm, left_margin = 5mm)
    display(p_ρβγ_cont)
    save_figure(logger, p_ρβγ_cont, "ρβγ_continuous_nfs")
    
    # Also create a rho vs beta two-panel plot matching the discrete sizing
    p_ρβ_cont = plot(rho_plot_cont, bias_plot_cont, layout=(1, 2), size=(1200, 500), bottom_margin = 5mm, left_margin = 5mm)
    display(p_ρβ_cont)
    save_figure(logger, p_ρβ_cont, "ρβ_continuous")
end

begin
    # Create analogous plots with c_1 = cos(2π/nf) on x-axis instead of nf
    # This shows how solutions vary with the first neighbor's cosine coefficient
    
    # Create the three plots with c_1 on x-axis
    rho_plot_c1 = @df plot_df_cont scatter(:c_1, :ρ, xlabel="c₁ = cos(2π/nf)", ylabel="ρ = r²", legend=false, marker=:marker_shape, color=:color, label=nothing, alpha=0.7)
    
    bias_plot_c1 = @df plot_df_cont scatter(:c_1, -:bias, xlabel="c₁ = cos(2π/nf)", ylabel="-Bias", legend=:topleft, color=:color, marker=(:marker_shape), label=nothing, alpha=0.7)
    
    # Plot γ vs c_1
    gamma_plot_c1 = @df plot_df_cont scatter(:c_1, :γ, xlabel="c₁ = cos(2π/nf)", ylabel="γ", legend=false, marker=:marker_shape, color=:color, label=nothing, ylims=(-0.1, 1.2), alpha=0.7)
    hline!(gamma_plot_c1, [1.0], color=:black, linestyle=:dash, label="Trivial Solution Line")
    # Add vertical lines at key c_1 values for different k transitions
    # When γ = c_1, we have k transitions
    vline!(gamma_plot_c1, [1.0], color=:black, linestyle=:dash, alpha=0.5, label="k=0 boundary")
    vline!(gamma_plot_c1, [0.5], color=:gray, linestyle=:dash, alpha=0.5, label="k=1 boundary (nf=6)")
    vline!(gamma_plot_c1, [0.0], color=:gray, linestyle=:dash, alpha=0.5, label="k=2 boundary (nf=4)")
    vline!(gamma_plot_c1, [-0.5], color=:gray, linestyle=:dash, alpha=0.5, label="k=3 boundary (nf=3)")
    
    # Add legend to bias plot
    scatter!(bias_plot_c1, [NaN], marker=:circle, label="Is minimum", color=:blue, markersize=4)
    scatter!(bias_plot_c1, [NaN], marker=:x, label="Not minimum", color=:blue, markersize=4)

    # Add k-value legend
    for (i, k) in enumerate(unique(sort(plot_df_cont.k_defined)))
        color = [:red, :blue, :green, :orange, :purple][i]
        scatter!(bias_plot_c1, [NaN], marker=:circle, label="k=$k", color=color, markersize=4)
    end

    p_cont_c1 = plot(rho_plot_c1, bias_plot_c1, gamma_plot_c1, layout=(1, 3), size=(1200, 400), left_margin = 5mm, bottom_margin = 5mm)
    display(p_cont_c1)
    
    println("Continuous solutions vs c₁:")
    display_df_colorbg(plot_df_cont_min_loss)
end

begin # plot as scatter plots in the ρ - γ space
    # Create scatter plot in ρ - γ space
    p_ργ = @df plot_df_cont scatter(:ρ, :γ, xlabel="ρ = r²", ylabel="γ = -β/r²", legend=false, marker=:marker_shape, color=:color, label=nothing, alpha=0.7)
    
    # Add minimum loss solutions as a line
    @df df_solutions_min_loss plot!(p_ργ, :ρ, :γ, lw=3, color=:black, label="Min Loss")
    
    # Add legend for marker shapes and colors
    scatter!(p_ργ, [NaN], marker=:circle, label="Hessian ≥ 0", color=:blue, markersize=4)
    scatter!(p_ργ, [NaN], marker=:x, label="Unstable (Saddle)", color=:red, markersize=4)
    
    display(p_ργ)
    save_figure(logger, p_ργ, "polygon_solution_ρ_γ_space")
end


begin
    # Plot all eigenvalues and min eigenvalue vs nf
    xs = [df_solutions_min_loss.nf[i] for i in 1:nrow(df_solutions_min_loss) for _ in df_solutions_min_loss.eigenvalues[i]]
    ys = [λ for i in 1:nrow(df_solutions_min_loss) for λ in df_solutions_min_loss.eigenvalues[i]]
    p = scatter(xs, ys, xlabel="Number Of Features", ylabel="Eigenvalue",
                markersize=2, color=:blue, label="Eigenvalues", legend=:topright)
    hline!([0], color=:red, linestyle=:dash, label="y=0")
    plot!(df_solutions_min_loss.nf, df_solutions_min_loss.min_eigenval, color=:green, linestyle=:dash, label="Min Eigenvalue")
    display(p)
    save_figure(logger, p, "eigenvalue_distribution_unique")
    CSV.write(resourcepath(logger, "unique_solutions.csv"), df_solutions_min_loss)

    
    # Plot the lowest nonzero eigenvalue |λ| on a log plot; we mark an eigenvalue as red / star marker if negative (if unstable), else blue / circle marker
    # Here, we select all |⋅| > 10^-9 eigenvalues
    eig_val_df = DataFrame(
        nf = df_solutions_min_loss.nf,
        min_nonzero_eigenval = [minimum((λ for λ in eigenvalues if abs(λ) ≥ 10^-9)) for eigenvalues in df_solutions_min_loss.eigenvalues],
        unstable = df_solutions_min_loss.unstable,
    )
    eig_val_df.abs_min_eigenval = abs.(eig_val_df.min_nonzero_eigenval)

    
    # Create the logarithmic plot
    p_eig = plot(xlabel="Number Of Features", ylabel="|Min Eigenvalue|", yscale=:log10, legend=:topleft)
    
    # Plot stable solutions (positive eigenvalues) in blue with circle markers
    stable_mask = eig_val_df.min_nonzero_eigenval .> 0
    if any(stable_mask)
        scatter!(p_eig, eig_val_df.nf[stable_mask], eig_val_df.abs_min_eigenval[stable_mask], 
                color=:blue, marker=:circle, label="Stable (λ > 0)", markersize=4)
    end
    
    # Plot unstable solutions (negative eigenvalues) in red with star markers
    unstable_mask = eig_val_df.unstable
    if any(unstable_mask)
        scatter!(p_eig, eig_val_df.nf[unstable_mask], eig_val_df.abs_min_eigenval[unstable_mask], 
                color=:red, marker=:star, label="Unstable (λ < 0)", markersize=6)
    end
    
    display(p_eig)
    save_figure(logger, p_eig, "min_eigenvalue_log_plot")
    display_df_colorbg(eig_val_df)
end

begin # loss comparison analysis
    # Compare losses of lower-order polygon solutions extended to higher dimensions
    comparison_nfs = 3:min(9, maximum(nfs))  # Lower-order polygons to test

    # Initialize DataFrame with nf column
    loss_df = DataFrame(nf=collect(nfs))

    # Add columns for each polygon type
    for poly_nf in comparison_nfs
        col_name = "loss_poly_$(poly_nf)"
        loss_df[!, col_name] = Vector{Union{Missing,Float64}}(missing, nrow(loss_df))

        # Get the minimum loss solution for this polygon
        poly_solution_row = df_solutions_min_loss[df_solutions_min_loss.nf .== poly_nf, :]
        if nrow(poly_solution_row) > 0
            poly_r = poly_solution_row.radius[1]
            poly_b = poly_solution_row.bias[1]
            
            for (i, target_nf) in enumerate(nfs)
                if poly_nf <= target_nf
                    # Create the polygon solution and extend it
                    W_poly = polygon_matrix(poly_nf, poly_r)
                    b_poly = fill(poly_b, poly_nf)
                    
                    W_extended = zeros(2, target_nf)
                    W_extended[:, 1:poly_nf] = W_poly
                    b_extended = zeros(target_nf)
                    b_extended[1:poly_nf] = b_poly

                    # Calculate loss for extended solution
                    loss_df[i, col_name] = gen_l1_loss_der(W_extended, b_extended)
                end
            end
        end
    end

    # Add native solution losses (minimum loss for each nf)
    loss_df.loss_native = df_solutions_min_loss.loss

    println("Loss Comparison:")
    println(loss_df)

    # Plot loss comparison (absolute losses)
    p_loss = plot(xlabel="Number Of Features", ylabel="Loss")
    # Precompute baseline for the inset (3-gon) if present
    baseline = hasproperty(loss_df, :loss_poly_3) ? loss_df[:, :loss_poly_3] : Vector{Union{Missing,Float64}}(missing, nrow(loss_df))

    for poly_nf in comparison_nfs
        col_name = "loss_poly_$(poly_nf)"
        valid_mask = .!ismissing.(loss_df[!, col_name])
        plot!(p_loss, loss_df.nf[valid_mask], loss_df[valid_mask, col_name],
            marker=:circle, markersize = 2 , label="$(poly_nf)-gon Extended", linestyle = linestyle_palette[1 + mod(poly_nf - 2, length(linestyle_palette))])
    end

    plot!(p_loss, loss_df.nf, loss_df.loss_native,
        marker=:star, markersize=4, linewidth = 3, label="Native Solution", color=:black)

    # Add second plot to the right that shows (x_loss[nf] - 3_poly_loss[nf] ∀nf)
    p_loss_relative = plot(xlabel = "Number Of Features", ylabel = "Loss - 3-gon-loss", legend = false)
    for poly_nf in comparison_nfs
        col_name = "loss_poly_$(poly_nf)"
        valid_mask = .!ismissing.(loss_df[!, col_name])
        plot!(p_loss_relative, loss_df.nf[valid_mask], loss_df[valid_mask, col_name] .- loss_df[valid_mask, "loss_poly_3"],
            marker=:circle, markersize = 2 , label="$(poly_nf)-gon Extended", linestyle = linestyle_palette[1 + mod(poly_nf - 2, length(linestyle_palette))])
    end
    plot!(p_loss_relative, loss_df.nf, loss_df.loss_native .- loss_df.loss_poly_3,
        marker=:star, markersize=4, linewidth = 3, label="Native Solution", color=:black)

    combined_loss_comparison = plot(p_loss, p_loss_relative, layout = (1, 2), size=(800, 400), left_margin = 5mm, bottom_margin = 5mm)

    save_figure(logger, combined_loss_comparison, "loss_comparison")
    display(combined_loss_comparison)
end

begin #! plot the loss landscape for a bunch of nfs.

    # Plot a grid of heatmaps: bias (x), ρ = r² (y), loss (color) for nf=3:8
    nfs_grid = nfs
    
    Zs = []
    eps = 1e-10  # small offset to avoid log(0)

    bias_range = range(-2.0, 0.2, length=60)
    rho_range = range(0.25, 4.0, length=60)  # ρ = r² range
    
    @showprogress for nf in nfs_grid
        Z = [gen_l1_loss_der(polygon_matrix(nf, sqrt(rho)), fill(b, nf)) for rho in rho_range, b in bias_range]
        Z_min = minimum(Z)
        Z_rel = Z .- Z_min .+ eps
        push!(Zs, Z_rel)
    end
end

begin

    # Overlay unique solutions as black stars, but only if within the heatmap's bias and ρ range
    heatmaps = []
    for (i, nf) in enumerate(nfs_grid)
        # Get solutions for this nf
        mask = df_solutions_unique.nf .== nf
        radii = df_solutions_unique.radius[mask]
        biases = df_solutions_unique.bias[mask]
        rhos = radii.^2  # Convert radius to ρ = r²
        # Filter to only those within the displayed range
        in_range = (rho, b) -> (rho_range[1] <= rho <= rho_range[end]) && (bias_range[1] <= b <= bias_range[end])
        in_bounds = [in_range(rho, b) for (rho, b) in zip(rhos, biases)]
        rhos_in = rhos[in_bounds]
        biases_in = biases[in_bounds]
        h = heatmap(bias_range, rho_range, Zs[i],
                   xlabel="Bias", ylabel="ρ = r²", title="nf=$(nf)",
                   color=cgrad([:white, :orange, :red]), colorbar=true, zscale=:log10)
        if !isempty(rhos_in)
            scatter!(h, biases_in, rhos_in, marker=:star5, color=:black, ms=7, label="Solutions")
        end
        push!(heatmaps, h)
    end

    n_heat = length(heatmaps)
    n_rows = ceil(Int, sqrt(n_heat))
    n_cols = ceil(Int, sqrt(n_heat))
    p_grid = plot(heatmaps..., layout = (n_rows, n_cols), size=(400 * n_cols, 400 * n_rows), titlefontsize=8, left_margin = 5mm, bottom_margin = 5mm)
    display(p_grid)
    save_figure(logger, p_grid, "loss_landscape_grid")

end

begin #! look at some unstable example solutions if we retrain them fully using BFGS
    unstable_solutions = df_solutions_unique[df_solutions_unique.unstable .== true, :]
    # take the first 4 unstable solutions
    unstable_solutions = unstable_solutions[1:min(4, nrow(unstable_solutions)), :]
    # now, retrain all parameters using BFGS
    retrained_models = []
    @showprogress for i in 1:nrow(unstable_solutions)
        sol = unstable_solutions[i, :]
        W_init = polygon_matrix(sol.nf, sol.radius)
        ε = 0.01
        W_init_perturbed = W_init .+ ε * randn(size(W_init))
        b_init = fill(sol.bias, sol.nf)
        model_props = train_full_model(W_init_perturbed, b_init; n_steps=1000)
        push!(retrained_models, merge(model_props, (
            perturbation="random",
            nf=sol.nf,
            radius=sol.radius,
            bias=sol.bias
        )))
    end
end
begin # todo
    # Now, add some more retrained solutions to the list
    # Now, perturb only one weight vector in the radial direction; and once in the tangential direction
    unstable_base_sols = unstable_solutions[1:min(2, nrow(unstable_solutions)), :] # take 2 unstable solutions
    retrained_models_directed = []
    ε = 0.01
    @showprogress for i in 1:nrow(unstable_base_sols)
        sol = unstable_base_sols[i, :]
        nf = sol.nf
        W_base = polygon_matrix(nf, sol.radius)
        b_init = fill(sol.bias, nf)
        # Reference: perturb the first weight vector
        w = W_base[:, 1]
        r = norm(w)
        # Radial perturbation: increase radius by ε
        w_radial = w * (1 + ε / r)
        W_radial = copy(W_base)
        W_radial[:, 1] = w_radial
        model_props_radial = train_full_model(W_radial, b_init; n_steps=1000)
        push!(retrained_models_directed, merge(model_props_radial, (
            perturbation="radial",
            nf=nf,
            radius=sol.radius,
            bias=sol.bias
        )))
        # Tangential perturbation: rotate by ε/rad (small angle approx)
        # Find orthogonal vector in plane
        w_tang = [-w[2], w[1]]
        w_tang = w_tang / norm(w_tang) * r # ensure same norm
        θ = ε / r
        w_tangential = cos(θ) * w + sin(θ) * w_tang
        W_tangential = copy(W_base)
        W_tangential[:, 1] = w_tangential
        model_props_tang = train_full_model(W_tangential, b_init; n_steps=1000)
        push!(retrained_models_directed, merge(model_props_tang, (
            perturbation="tangential",
            nf=nf,
            radius=sol.radius,
            bias=sol.bias
        )))
    end
    all_retrained_models = [retrained_models ... , retrained_models_directed...]
end

begin #! opt: show trajectories of the polygon solutions
    # Plot each retrained model in its own subplot, but use only two consistent labels and one legend
    n_plot = length(all_retrained_models)
    plot_list = Vector{Any}(undef, n_plot)
    for (i, retrained) in enumerate(all_retrained_models)
        W_init = retrained.initial_model.W
        W_retrained = retrained.final_model.W

        # find directions of greatest descent; obtain eigenvalues and eigenvectors of the hessian and then
        _hessian = retrained.initial_hessian
        _eigen = eigen(Symmetric(_hessian))
        i_min_eigval = argmin(real(_eigen.values))
        min_eigvector = _eigen.vectors[:, i_min_eigval]
        # take the (W, b) components of the eigenvector
        min_eigvector_W = min_eigvector[1:length(vec(W_init))]
        min_eigvector_b = min_eigvector[length(vec(W_init))+1:end]

        x_init, y_init = line_connected_to_center(eachcol(W_init))
        x_ret, y_ret = line_connected_to_center(eachcol(W_retrained))
        show_legend = (i == 1)
        plt = plot(x_init, y_init, color=:blue, label=show_legend ? "Initial" : "", lw=2, xlims=(-2,2), ylims=(-2,2),
                  xlabel="x", ylabel="y", aspect_ratio=1, legend=show_legend ? :topleft : false, title="nf=$(retrained.nf), $(retrained.perturbation)")
        plot!(plt, x_ret, y_ret, color=:black, label=show_legend ? "Retrained" : "", lw=2, linestyle=:dash)
        scatter!(plt, [0.0], [0.0], color=:black, marker=:star, label=show_legend ? "center" : "")

        # Plot the direction of the minimum eigenvalue as an arrow
        # min_eigvector_W is a vector of length 2*nf, reshape to (2, nf)
        min_eigvector_W_mat = reshape(min_eigvector_W, 2, retrained.nf)
        # Scale for visualization
        arrow_scale = 0.5 * maximum(norm.(eachcol(W_init)))
        # Plot arrows from each vertex in the direction of the eigenvector
        for j in 1:retrained.nf
            v = W_init[:, j]
            dv = min_eigvector_W_mat[:, j]
            arrow_start = v
            arrow_end = v + arrow_scale * dv / (norm(dv) + 1e-12)
            quiver!(plt, [arrow_start[1]], [arrow_start[2]],
                quiver=([arrow_end[1] - arrow_start[1]], [arrow_end[2] - arrow_start[2]]),
                color=:red, label=show_legend && j == 1 ? "Min eigvec dir" : "", arrow=:closed, lw=2)
        end
        plot_list[i] = plt
        
    end
    layout = (ceil(Int, sqrt(n_plot)), ceil(Int, sqrt(n_plot)))
    p = plot(plot_list..., layout=layout, size=(400 * layout[2], 800))
    display(p)
end

let base_df = df_solutions_unique
    # Plot starlike polygons for selected solutions, overlaying minimum descent direction (no perturbation results)
    # Only plot stable solutions for nf=9 and 10

    stable_nfs = [8, 9, 10, 11]
    stable_sols = [let rows = filter(row -> row.nf == nf && !row.unstable, eachrow(base_df)); isempty(rows) ? nothing : first(rows) end for nf in stable_nfs]
    stable_sols = filter(!isnothing, stable_sols)
    # Take first two unstable solutions
    unstable_sols = [row for row in eachrow(base_df) if row.unstable][1:min(2, sum(base_df.unstable))]
    all_sols = vcat(stable_sols, unstable_sols)

    plot_list = Vector{Any}(undef, length(all_sols))
    for (i, sol) in enumerate(all_sols)
        nf = convert(Int, sol.nf)
        r = sol.radius
        b = fill(sol.bias, nf)
        W = polygon_matrix(nf, r)
        # Compute Hessian and its minimum eigenvector
        params = vcat(vec(W), b)
        obj(p) = gen_l1_loss_der(reshape(p[1:length(vec(W))], size(W)), p[length(vec(W))+1:end])
        hess = Symmetric(ForwardDiff.hessian(obj, params))
        eig = eigen(hess)
        i_min = argmin(real(eig.values))
        min_eigvec = eig.vectors[:, i_min]
        min_eigvec_W = min_eigvec[1:length(vec(W))]
        min_eigvec_b = min_eigvec[length(vec(W))+1:end]
        min_eigvec_W_mat = reshape(min_eigvec_W, 2, nf)
        # Plot polygon
        x, y = line_connected_to_center(eachcol(W))
        plt = plot(x, y, color=:blue, lw=2, xlims=(-2,2), ylims=(-2,2),
                  xlabel="x", ylabel="y", aspect_ratio=1, legend=false, title="nf=$(nf), $(sol.unstable ? "unstable" : "stable")")
        scatter!(plt, [0.0], [0.0], color=:black, marker=:star, ms=7, label="center")
        # Overlay minimum descent direction as quiver from each vertex
        arrow_scale = 0.5 * maximum(norm.(eachcol(W)))
        for j in 1:nf
            v = W[:, j]
            dv = min_eigvec_W_mat[:, j]
            arrow_start = v
            arrow_end = v + arrow_scale * dv / (norm(dv) + 1e-12)
            quiver!(plt, [arrow_start[1]], [arrow_start[2]],
                quiver=([arrow_end[1] - arrow_start[1]], [arrow_end[2] - arrow_start[2]]),
                color=:red, label="", arrow=:closed, lw=2)
        end
        plot_list[i] = plt
    end
    layout = (2, 2)
    p = plot(plot_list..., layout=layout, size=(800, 800))
    display(p)
    save_figure(logger, p, "descent_directions")
    # a priori it is unclear if critical points with zero minimum eigenvalue are actually local minima
    # Under small perturbation, however, it seems to be empirically the case that they are.
    # eigenvector for minimal eigenvalue of hessian (min eigenvec): For stable solutions, minimum eigenvalue corresponds to eigenvector

end