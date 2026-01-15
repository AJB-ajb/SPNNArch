using JlCode
using DataFrames
using Statistics
using Random
using LinearAlgebra
using ProgressMeter
using Flux



Execute_Accuracy_Comp = true

# N_nets # sample N_nets networks
# N_vecs # sample N_vecs vectors per network
# n = 2 # We consider the 2-ary UAND first for simplicity

begin # Test the variance and expectation estimates against sampling
    # Setup Logger
    base_path = joinpath(dirname(@__DIR__), "experiments")
    global logger = nothing
    try
        global logger = load_logger("variance_estimates"; prefix="sampling", base_path=base_path)
        println("Loaded existing experiment: $(experimentfolder(logger))")
    catch e
        println("No existing experiment found (Error: $e), creating new one.")
        global logger = ExpLogger("variance_estimates"; prefix="sampling", tmp=false, base_path=base_path)
    end
    println("Experiment Folder: $(experimentfolder(logger))")

    N_vecs = 200; N_nets = 5; n = 2
    p(d, m) = log(m)^2 / sqrt(d)

    if Execute_Accuracy_Comp
        m = 30
        s = 5
        ds = [30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 435, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000]

        global df = DataFrame(Type=String[], d=Int[], m=Int[], sr=Int[], bits=Tuple{Int,Int}[], cnn_Ex=Float64[], cnn_Var=Float64[], ana_Ex = Float64[], ana_Var=Float64[])

        # create an overall progress bar for the different `d` values
        p_outer = Progress(length(ds); desc = "ds")
        for d_val in ds
            sr = s - 2
            
            constructions = Any[]
            p_raw = log(m)^2 / sqrt(d_val)
            if p_raw < 1.0
                push!(constructions, ("Bernoulli", BernoulliUNAnd(; p=p_raw, n=2, m=m, d=d_val)))
            end
            push!(constructions, ("Rademacher", RademacherUNAnd(; n=2, m=m, d=d_val)))
            push!(constructions, ("Gaussian", GaussianUNAnd(; σ=1.0, n=2, m=m, d=d_val)))

            for (name, constr) in constructions
                for (b1_int, b2_int) in [(0,0), (0,1), (1,1)]
                    b1, b2 = Bool(b1_int), Bool(b2_int)
                    cnn_Ex, cnn_Var, _ = estimate_error_stats(constr, N_nets, N_vecs, s, b1, b2)
                    ana_Ex = errorExEstimate(constr, sr) 
                    ana_Var = errorVarEstimate(constr, sr)
                    push!(df, (Type=name, d=d_val, m=m, sr=sr, bits=(b1_int, b2_int), cnn_Ex=cnn_Ex, cnn_Var=cnn_Var, ana_Ex=ana_Ex, ana_Var=ana_Var))
                end
            end
            next!(p_outer)
        end
        save_data(logger, df, "results_df.jls")
    end
end

using Plots, StatsPlots
using Measures
default(fontfamily="Computer Modern", framestyle=:box, grid=true)

begin 
    if Execute_Accuracy_Comp
        # plot; sample expectation as line, estimate as dotted line

        colors = Dict((0,0) => :blue, (0,1) => :green, (1,1) => :orange)
        
        for type in unique(df.Type)
            subdf = filter(row -> row.Type == type, df)
            
            is_unbiased = type in ["Rademacher", "Gaussian"]

            # Standard Deviation Plot
            p_std = plot(xlabel="Width d", ylabel="Std Dev", yaxis=:log, xlims=(0, Inf))
            
            # Expectation Plot
            if is_unbiased
                p_exp = plot(xlabel="Width d", ylabel="Abs Expectation", yaxis=:log, xlims=(0, Inf))
            else
                p_exp = plot(xlabel="Width d", ylabel="Expectation", xlims=(0, Inf))
            end
            
            for bits in unique(subdf.bits)
                bit_df = filter(row -> row.bits == bits, subdf)
                c = get(colors, bits, :black)
                lbl = "CN $(bits)"
                
                # Filter valid variance for std plot
                valid_var_mask = .!(isnan.(bit_df.cnn_Var) .| isinf.(bit_df.cnn_Var) .| (bit_df.cnn_Var .< 0))
                bit_df_std = bit_df[valid_var_mask, :]
                
                plot!(p_std, bit_df_std.d, sqrt.(bit_df_std.cnn_Var), label=lbl, marker=:square, linestyle=:dash, color=c)
                
                y_samp = bit_df.cnn_Ex
                if is_unbiased
                    y_samp = abs.(y_samp)
                end
                plot!(p_exp, bit_df.d, y_samp, label=lbl, marker=:square, linestyle=:dash, color=c)
            end

            # Plot Analytical (only once per type, as it doesn't depend on bits in current code)
            first_bits = (0,0)
            ana_df = filter(row -> row.bits == first_bits, subdf)
            if !isempty(ana_df)
                # Filter valid analytical variance
                valid_ana_mask = .!(isnan.(ana_df.ana_Var) .| isinf.(ana_df.ana_Var) .| (ana_df.ana_Var .< 0))
                ana_df_std = ana_df[valid_ana_mask, :]

                plot!(p_std, ana_df_std.d, sqrt.(ana_df_std.ana_Var), label="AE", marker=:none, linestyle=:dot, color=:black, lw=2)
                
                if !is_unbiased
                    plot!(p_exp, ana_df.d, ana_df.ana_Ex, label="AE", marker=:none, linestyle=:dot, color=:black, lw=2)
                end
            end
            
            p_combined = plot(p_exp, p_std, layout=(1,2), size=(1000, 400), bottom_margin=5mm, left_margin=5mm)
            save_figure(logger, p_combined, "$(type)_error_stats_vs_width")
            display(p_combined)
        end

        display_df_colorbg(df)
    end
end

"""
    chebychev_bound(construction :: Construction, sr :: Integer, prob :: Float64)
Returns an error bound based on the Chebyshev inequality for the given construction, number of spurious features sr, and desired probability prob.
Concretely, the bound is given by:
    E[ε] + (1 - prob)^(-1/2) * sqrt(Var(ε))
where ε is the error when reading off the AND from the UAND and holds with probability at least prob 
"""
function chebychev_bound(construction :: Construction, sr::Integer, prob::Float64)
    var_est = errorVarEstimate(construction, sr)
    σ = sqrt(var_est)
    α = (1.0 - prob)^(-0.5)
    error_bound = α * σ + errorExEstimate(construction, sr)
    return error_bound
end

begin #compare concrete error bounds from each construction
    prob = 0.9
    m = 500
    s = 5
    sr = s - 2
    # Ensure d is large enough for p <= 1 in Bernoulli case
    # p = log(m)^2 / sqrt(d) <= 1 => d >= log(m)^4
    min_d = ceil(Int, log(m)^4)
    ds = range(max(100, min_d), 10000, step=100)
    
    # Bernoulli
    # p = log(m)^2 / sqrt(d) as per Hanni et al.
    p_ber(d) = min(1.0, log(m)^2 / sqrt(d)) 
    ber_bounds = [chebychev_bound(BernoulliUNAnd(p=p_ber(d), n=2, m=m, d=d), sr, prob) for d in ds]
    
    # Rademacher
    rad_bounds = [chebychev_bound(RademacherUNAnd(n=2, m=m, d=d), sr, prob) for d in ds]
    
    # Gaussian
    gauss_bounds = [chebychev_bound(GaussianUNAnd(σ=1.0, n=2, m=m, d=d), sr, prob) for d in ds]
    
    # Plot
    p_comp = plot(ds, ber_bounds, 
        label="Bernoulli", 
        xlabel="Dimension d", 
        ylabel="Error Bound (prob=$(prob))", 
        lw=2,
        legend=:topright,
        xaxis=:log,
        yaxis=:log,
    )
    plot!(p_comp, ds, rad_bounds, label="Rademacher", lw=2)
    plot!(p_comp, ds, gauss_bounds, label="Gaussian", lw=2)
    
    display(p_comp)
end


begin # Compare behavior across different sparsities using sampling
    # 1. Plot: Error levels for fixed d, m, varying sr on x-axis
    # 2. Plot: Error levels for fixed m, different d (as lines), varying sr on x-axis

    if Execute_Accuracy_Comp
        println("Running Sparsity Comparison...")
        m_fixed = 50
        d_fixed = 2000
        ds_list = [500, 1000, 2000, 4000]
        s_values = 2:2:20 # sr = 0 to 18
        
        N_nets = 10
        N_vecs = 200
        
        prob = 0.9
        alpha = (1.0 - prob)^(-0.5)
        
        results_sparsity = DataFrame(Type=String[], d=Int[], m=Int[], s=Int[], sr=Int[], bits=Tuple{Int,Int}[], cnn_Ex=Float64[], cnn_Var=Float64[], ana_Bound=Float64[], samp_Bound=Float64[])
        
        p_func(d, m) = min(1.0, log(m)^2 / sqrt(d))
        
        # Ensure d_fixed is in the list for the first plot
        all_ds = unique([ds_list; d_fixed])
        
        @showprogress desc="Sparsity Exp" for s in s_values
            sr = s - 2
            for d_val in all_ds
                constructions = [
                    ("Bernoulli", BernoulliUNAnd(; p=p_func(d_val, m_fixed), n=2, m=m_fixed, d=d_val)),
                    ("Rademacher", RademacherUNAnd(; n=2, m=m_fixed, d=d_val)),
                    ("Gaussian", GaussianUNAnd(; σ=1.0, n=2, m=m_fixed, d=d_val))
                ]
                
                for (name, constr) in constructions
                    # Analytical Bound
                    ana_Bound = chebychev_bound(constr, sr, prob)

                    for (b1_int, b2_int) in [(0,0), (0,1), (1,1)]
                        b1, b2 = Bool(b1_int), Bool(b2_int)
                        cnn_Ex, cnn_Var, _ = estimate_error_stats(constr, N_nets, N_vecs, s, b1, b2)
                        
                        # Sampled Bound
                        samp_Bound = cnn_Ex + alpha * sqrt(cnn_Var)
                        
                        push!(results_sparsity, (Type=name, d=d_val, m=m_fixed, s=s, sr=sr, bits=(b1_int, b2_int), cnn_Ex=cnn_Ex, cnn_Var=cnn_Var, ana_Bound=ana_Bound, samp_Bound=samp_Bound))
                    end
                end
            end
        end
        
        save_data(logger, results_sparsity, "sparsity_results.jls")
        
        # Plot 1: Error levels for fixed d, m, varying sr on x-axis
        # We plot Error Bound for (0,1) and (1,1)
        for bits in [(0,1), (1,1)]
            subdf = filter(r -> r.d == d_fixed && r.bits == bits, results_sparsity)
            
            p1 = plot(xlabel="Sparsity s", ylabel="Error Bound", yaxis=:linear, xlims=(0, Inf), ylims=(0, Inf))
            
            colors = Dict("Bernoulli" => :blue, "Rademacher" => :green, "Gaussian" => :orange)
            
            for type in unique(subdf.Type)
                ss = filter(r -> r.Type == type, subdf)
                c = get(colors, type, :black)
                plot!(p1, ss.s, ss.samp_Bound, label="$(type) CN", marker=:circle, lw=2, color=c)
                plot!(p1, ss.s, ss.ana_Bound, label="$(type) AE", linestyle=:dash, marker=:none, lw=2, color=c)
            end
            save_figure(logger, p1, "constructions_error_bound_vs_sparsity_d$(d_fixed)_bits$(bits)")
            display(p1)
        end
        
        # Plot 2: Error levels for fixed m, different d (as lines), varying sr on x-axis
        # Focus on Bernoulli
        type_focus = "Bernoulli"
        for bits in [(0,1), (1,1)]
            subdf = filter(r -> r.Type == type_focus && r.bits == bits && r.d in ds_list, results_sparsity)
            
            p2 = plot(xlabel="Sparsity s", ylabel="Error Bound", yaxis=:linear, xlims=(0, Inf), ylims=(0, Inf))
            
            # Get unique d values and sort them
            d_vals = sort(unique(subdf.d))
            # Create a color palette
            palette = cgrad(:viridis, length(d_vals), categorical=true)
            
            for (i, d_val) in enumerate(d_vals)
                ss = filter(r -> r.d == d_val, subdf)
                c = palette[i]
                plot!(p2, ss.s, ss.samp_Bound, label="d=$d_val CN", marker=:circle, lw=2, color=c)
                plot!(p2, ss.s, ss.ana_Bound, label="d=$d_val AE", linestyle=:dash, marker=:none, lw=2, color=c)
            end
            save_figure(logger, p2, "$(type_focus)_error_bound_vs_sparsity_scaling_bits$(bits)")
            display(p2)
        end
    end
end
