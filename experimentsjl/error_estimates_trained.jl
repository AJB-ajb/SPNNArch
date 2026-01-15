using Flux
using Statistics
using Random
using ProgressMeter
using Plots
using Measures
using DataFrames
using JlCode
using Hyperopt
using Serialization
using LinearAlgebra

# Helper to map (i, j) to linear index for upper triangular part
function pair_to_idx(i, j, m)
    if i > j
        i, j = j, i
    end
    # Row-major upper triangle indexing
    # Offset for row i is sum(m-k for k in 1:i-1)
    # sum_{k=1}^{i-1} (m-k) = (i-1)m - (i-1)*i/2
    offset = (i - 1) * m - (i * (i - 1)) ÷ 2
    return offset + (j - i)
end

struct SparseReadOffModel
    input_layer::Dense
    readoff::Embedding
    m::Int
end

Flux.@layer SparseReadOffModel

# Constructor
function SparseReadOffModel(m::Int, d::Int; init=Flux.glorot_uniform, activation=x -> leakyrelu(x, 1e-3))
    num_pairs = (m * (m - 1)) ÷ 2
    return SparseReadOffModel(
        Dense(m => d, activation; init=init),     # Hidden layer
        Embedding(num_pairs => d; init=init), # Read-off vectors stored as embeddings
        m
    )
end

# Flux.@functor SparseReadOffModel # Deprecated, and likely not needed for CPU training with implicit params

function (model::SparseReadOffModel)(x::AbstractMatrix, pairs::Vector{Tuple{Int,Int}})
    # x: Input batch (m x batch_size)
    # pairs: Vector of (i, j) tuples to evaluate

    # 1. Compute hidden representation
    h = model.input_layer(x) # Result: d x batch_size

    # 2. Retrieve read-off vectors for the sampled pairs
    indices = [pair_to_idx(p[1], p[2], model.m) for p in pairs]
    W_out = model.readoff(indices) # Result: d x num_pairs

    # 3. Compute predictions for all (pair, sample) combinations
    # Result: num_pairs x batch_size
    # W_out is d x num_pairs. h is d x batch_size.
    # We want output[k, b] = W_out[:, k] ⋅ h[:, b]
    return W_out' * h
end

function train_loop!(model, opt, losses, steps, m, s, batch_size, pairs_per_step)
    for step in 1:steps
        # 1. Sample batch of inputs
        x_batch = hcat([sampleSparseVec(m, s) for _ in 1:batch_size]...)

        # 2. Identify active pairs in this batch
        active_pairs = Set{Tuple{Int,Int}}()
        for b in 1:batch_size
            inds = findall(x_batch[:, b])
            for i_idx in 1:length(inds)
                for j_idx in i_idx+1:length(inds)
                    u, v = inds[i_idx], inds[j_idx]
                    if u > v
                        u, v = v, u
                    end
                    push!(active_pairs, (u, v))
                end
            end
        end

        # 3. Sample random pairs to ensure we also learn zeros
        # We combine all active pairs found with some random pairs
        pairs_batch = collect(active_pairs)

        for _ in 1:pairs_per_step
            i = rand(1:m)
            j = rand(1:m)
            while i == j
                j = rand(1:m)
            end
            if i > j
                i, j = j, i
            end
            push!(pairs_batch, (i, j))
        end

        pairs_batch = unique(pairs_batch)

        # 3. Compute gradients
        grads = Flux.gradient(model) do m_model
            preds = m_model(x_batch, pairs_batch)

            # Compute targets efficiently
            # targets[k, b] corresponds to pair k and batch sample b
            targets = [Float32(x_batch[p[1], b] * x_batch[p[2], b]) for p in pairs_batch, b in 1:batch_size]

            Flux.mse(preds, targets)
        end

        # Calculate loss for logging (re-computing to be safe, though could use value from gradient)
        # Using the same batch for loss logging
        current_loss = Flux.mse(model(x_batch, pairs_batch), [Float32(x_batch[p[1], b] * x_batch[p[2], b]) for p in pairs_batch, b in 1:batch_size])
        push!(losses, current_loss)

        Flux.update!(opt, model, grads[1])
    end
end

function train_uand_model(;
    m=20, d=100, s=5, steps=5000,
    batch_size=32, pairs_per_step=32, lr=1e-3,
    init=Flux.glorot_uniform, activation=x -> leakyrelu(x, 1e-3),
    finetune_steps=0
)
    model = SparseReadOffModel(m, d; init=init, activation=activation)
    opt = Flux.setup(AdamW(lr), model)

    losses = Float64[]

    # Phase 1: Initial Training
    train_loop!(model, opt, losses, steps, m, s, batch_size, pairs_per_step)

    # Phase 2: Finetuning with ReLU
    if finetune_steps > 0
        println("Switching to ReLU for finetuning...")

        # Create new Dense layer with ReLU, preserving weights
        old_dense = model.input_layer
        new_dense = Dense(old_dense.weight, old_dense.bias, relu)

        # Create new model
        model = SparseReadOffModel(new_dense, model.readoff, model.m)

        # Re-setup optimizer (keeping same LR for now)
        opt = Flux.setup(AdamW(lr), model)

        train_loop!(model, opt, losses, finetune_steps, m, s, batch_size, pairs_per_step)
    end

    return model, losses
end


function evaluate_model(model, m, s; n_samples=1000)
    # Evaluate error statistics on new data
    # We want to estimate E[ε] and Var(ε)
    # ε = prediction - target

    errors = Float64[]

    # Sample inputs
    x_batch = hcat([sampleSparseVec(m, s) for _ in 1:n_samples]...)

    # Evaluate on ALL pairs or a large subset?
    # For m=20, pairs=190. We can evaluate all.
    all_pairs = Tuple{Int,Int}[]
    for i in 1:m
        for j in i+1:m
            push!(all_pairs, (i, j))
        end
    end

    preds = model(x_batch, all_pairs)
    targets = [Float32(x_batch[p[1], b] * x_batch[p[2], b]) for p in all_pairs, b in 1:n_samples]

    all_errors = preds .- targets

    # We are interested in the error distribution
    # Specifically, we want to know if the error is biased and what its variance is.
    # We can aggregate over all pairs and samples.

    return mean(all_errors), var(all_errors)
end

function evaluate_model_stats(model, m, s; n_samples=1000)
    # More detailed evaluation returning errors broken down by bit combination
    x_batch = hcat([sampleSparseVec(m, s) for _ in 1:n_samples]...)

    all_pairs = Tuple{Int,Int}[]
    for i in 1:m
        for j in i+1:m
            push!(all_pairs, (i, j))
        end
    end

    preds = model(x_batch, all_pairs)
    # targets = [Float32(x_batch[p[1], b] * x_batch[p[2], b]) for p in all_pairs, b in 1:n_samples]

    errors_dict = Dict{Tuple{Int,Int},Vector{Float64}}()
    errors_dict[(0, 0)] = Float64[]
    errors_dict[(0, 1)] = Float64[]
    errors_dict[(1, 1)] = Float64[]

    for b in 1:n_samples
        for (k, p) in enumerate(all_pairs)
            val1 = x_batch[p[1], b]
            val2 = x_batch[p[2], b]
            target = Float32(val1 * val2)
            pred = preds[k, b]
            err = pred - target

            key = (Int(val1), Int(val2))
            if key == (1, 0)
                key = (0, 1)
            end
            push!(errors_dict[key], err)
        end
    end

    return errors_dict
end

function count_learned_ands(model, m, s; n_samples=1000, error_threshold=0.2, quantile_threshold=0.9)
    # Count how many ANDs (pairs) are "learned" for each bit combination
    # A pair is learned for a bit combination if for quantile_threshold (e.g. 90%) of samples of that type, 
    # the absolute error is <= error_threshold
    
    x_batch = hcat([sampleSparseVec(m, s) for _ in 1:n_samples]...)
    
    all_pairs = Tuple{Int,Int}[]
    for i in 1:m
        for j in i+1:m
            push!(all_pairs, (i, j))
        end
    end
    
    preds = model(x_batch, all_pairs)
    targets = [Float32(x_batch[p[1], b] * x_batch[p[2], b]) for p in all_pairs, b in 1:n_samples]
    
    abs_errors = abs.(preds .- targets)
    
    counts = Dict((0,0)=>0, (0,1)=>0, (1,1)=>0)
    
    for (k, p) in enumerate(all_pairs)
        errors_by_type = Dict((0,0)=>Float64[], (0,1)=>Float64[], (1,1)=>Float64[])
        
        for b in 1:n_samples
            val1 = x_batch[p[1], b]
            val2 = x_batch[p[2], b]
            key = (Int(val1), Int(val2))
            if key == (1, 0)
                key = (0, 1)
            end
            push!(errors_by_type[key], abs_errors[k, b])
        end
        
        for (key, errs) in errors_by_type
            if !isempty(errs)
                ok_count = count(e -> e <= error_threshold, errs)
                if ok_count / length(errs) >= quantile_threshold
                    counts[key] += 1
                end
            end
        end
    end
    
    return counts
end

function run_hpo()
    # Hyperparameter Optimization
    println("Starting Hyperparameter Optimization...")

    m = 20
    s = 5
    d = 50 # Fix small d for HPO
    steps = 500 # Short runs

    # Define objective function for HPO
    function objective(lr, batch_size, pairs_per_step, init_std, activation_slope)
        # Setup initialization
        init_fn = (dims...) -> randn(Float32, dims...) .* Float32(init_std)
        act_fn = x -> leakyrelu(x, Float32(activation_slope))

        model, losses = train_uand_model(
            m=m, d=d, s=s, steps=steps,
            batch_size=batch_size,
            pairs_per_step=pairs_per_step,
            lr=lr,
            init=init_fn,
            activation=act_fn
        )

        # Evaluate
        # We want to minimize variance of error (and bias, but mainly variance for now as bias can be shifted)
        # Or just MSE on validation set
        mu, sigma2 = evaluate_model(model, m, s; n_samples=500)

        # Metric: MSE = Bias^2 + Variance
        mse = mu^2 + sigma2
        return mse
    end

    # Run HPO
    ho = @hyperopt for i = 50,
        lr = LinRange(1e-4, 1e-2, 100),
        batch_size = [16, 32, 64],
        pairs_per_step = [16, 32, 64],
        init_std = LinRange(0.01, 1.0, 20),
        activation_slope = [0.0, 1e-3, 1e-2, 1e-1]

        val = objective(lr, batch_size, pairs_per_step, init_std, activation_slope)
        # println("Trial $i: val=$val (lr=$lr, bs=$batch_size, pps=$pairs_per_step, std=$init_std, slope=$activation_slope)")
    end

    best_params, min_val = ho.minimizer, ho.minimum
    println("Best Parameters found: $best_params")
    println("Minimum MSE: $min_val")

    return best_params
end

function run_scaling_experiment(;
    lr=0.0064,
    batch_size=64,
    pairs_per_step=32,
    init_std=0.01,
    activation_slope=0.1,
    finetune_steps=2000,
    force_retrain=false
)
    # Setup Logger
    base_path = joinpath(dirname(@__DIR__), "experiments")
    logger = nothing
    try
        logger = load_logger("variance_estimates_trained"; prefix="scaling", base_path=base_path)
        println("Loaded existing experiment: $(experimentfolder(logger))")
    catch e
        println("No existing experiment found (Error: $e), creating new one.")
        logger = ExpLogger("variance_estimates_trained"; prefix="scaling", tmp=false, base_path=base_path)
    end
    println("Experiment Folder: $(experimentfolder(logger))")

    # Run full scaling experiment with best params
    println("Running Scaling Experiment with Parameters:")
    println("  lr=$lr, batch_size=$batch_size, pairs_per_step=$pairs_per_step, init_std=$init_std, activation_slope=$activation_slope, finetune_steps=$finetune_steps")

    m = 30
    s = 5
    ds = [30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 435, 600, 800]

    results = DataFrame(d=Int[], type=String[], bits=Tuple{Int,Int}[], mean_error=Float64[], var_error=Float64[], q90_error=Float64[], mae_error=Float64[])
    learned_ands_results = DataFrame(d=Int[], count_00=Int[], count_01=Int[], count_11=Int[], total_pairs=Int[])

    init_fn = (dims...) -> randn(Float32, dims...) .* Float32(init_std)
    act_fn = x -> leakyrelu(x, Float32(activation_slope))

    # 1. Trained Neural Networks (TNN)
    for d in ds
        model_filename = "model_d$(d).jls"
        model = load_data(logger, model_filename)

        if isnothing(model) || force_retrain
            println("Training for d=$d...")
            model, losses = train_uand_model(
                m=m, d=d, s=s, steps=5000,
                batch_size=batch_size,
                pairs_per_step=pairs_per_step,
                lr=lr,
                init=init_fn,
                activation=act_fn,
                finetune_steps=finetune_steps
            )
            save_data(logger, model, model_filename)
            save_data(logger, losses, "losses_d$(d).jls")

            # Plot loss
            p_loss = plot(losses, xlabel="Step", ylabel="MSE", yaxis=:log)
            save_figure(logger, p_loss, "loss_d$(d)")
        else
            println("Loaded model for d=$d")
        end

        # Evaluate TNN
        errors_dict = evaluate_model_stats(model, m, s; n_samples=1000)

        for (bits, errs) in errors_dict
            mu = mean(errs)
            sigma2 = var(errs)
            q90 = quantile(abs.(errs), 0.9)
            mae = mean(abs.(errs))
            push!(results, (d, "TNN", bits, mu, sigma2, q90, mae))
            println("TNN d=$d, bits=$bits: Mean=$mu, Var=$sigma2, Q90=$q90, MAE=$mae")
        end

        # Histogram of weights
        # Input layer weights
        W_in = model.input_layer.weight
        p_hist = histogram(vec(W_in), label="Input Weights", alpha=0.5, normalize=true)
        # Readoff weights (embeddings)
        W_out = model.readoff.weight
        histogram!(p_hist, vec(W_out), label="Readoff Weights", alpha=0.5, normalize=true)
        save_figure(logger, p_hist, "weights_hist_d$(d)")

        # Count Learned ANDs
        counts = count_learned_ands(model, m, s; n_samples=1000, error_threshold=0.2, quantile_threshold=0.9)
        total_pairs = (m * (m - 1)) ÷ 2
        push!(learned_ands_results, (d, counts[(0,0)], counts[(0,1)], counts[(1,1)], total_pairs))
        println("TNN d=$d: Learned ANDs (0,0)=$(counts[(0,0)]), (0,1)=$(counts[(0,1)]), (1,1)=$(counts[(1,1)]) / $total_pairs")
    end

    # 2. Constructed Neural Networks (CNN) & Analytical
    # We use the constructions from JlCode
    println("Evaluating Constructions...")
    N_nets = 5
    N_vecs = 200

    for d in ds
        # Bernoulli
        p_val = min(1.0, log(m)^2 / sqrt(d))
        constr_ber = BernoulliUNAnd(; p=p_val, n=2, m=m, d=d)

        # Rademacher
        constr_rad = RademacherUNAnd(; n=2, m=m, d=d)

        # Gaussian
        constr_gauss = GaussianUNAnd(; σ=1.0, n=2, m=m, d=d)

        constructions = [
            ("Bernoulli", constr_ber),
            ("Rademacher", constr_rad),
            ("Gaussian", constr_gauss)
        ]

        for (name, constr) in constructions
            for (b1_int, b2_int) in [(0, 0), (0, 1), (1, 1)]
                b1, b2 = Bool(b1_int), Bool(b2_int)
                bits = (b1_int, b2_int)

                # Sampled (CNN)
                mu, sigma2, all_errors = estimate_error_stats(constr, N_nets, N_vecs, s, b1, b2)
                q90 = quantile(abs.(all_errors), 0.9)
                mae = mean(abs.(all_errors))

                push!(results, (d, name, bits, mu, sigma2, q90, mae))

                # Analytical
                # Note: Analytical estimates in CompSuperpos.jl depend on sr (spurious features).
                # estimate_error_stats uses s-2 background features always.
                # So we use sr = s - 2.
                sr = s - 2
                ana_mu = errorExEstimate(constr, sr)
                ana_var = errorVarEstimate(constr, sr)

                push!(results, (d, "$(name)_Ana", bits, ana_mu, ana_var, NaN, NaN))
            end
        end
    end

    save_data(logger, results, "results_df.jls")
    display_df_colorbg(results)

    # 3. Generate Plots

    # Filter data
    # We will make separate plots for each bit combination or combine them?
    # Let's make separate plots for each bit combination to avoid clutter.

    colors = Dict("TNN" => :red, "Bernoulli" => :blue, "Rademacher" => :green, "Gaussian" => :orange)

    q90_plots = []
    mae_plots = []

    for bits in [(0, 0), (0, 1), (1, 1)]
        df_bits = filter(r -> r.bits == bits, results)

        # A. Std Dev Comparison
        p_std = plot(xlabel="Width d", ylabel="Std Dev", yaxis=:log, xlims=(0, Inf))

        # TNN
        df_tnn = filter(r -> r.type == "TNN", df_bits)
        plot!(p_std, df_tnn.d, sqrt.(df_tnn.var_error), label="TN", marker=:circle, lw=2, color=colors["TNN"])

        # Sampled
        for name in ["Bernoulli", "Rademacher", "Gaussian"]
            df_cnn = filter(r -> r.type == name, df_bits)
            c = colors[name]
            plot!(p_std, df_cnn.d, sqrt.(df_cnn.var_error), label="$(name) CN", marker=:square, linestyle=:dash, color=c)
        end

        # Analytical
        for name in ["Bernoulli", "Rademacher", "Gaussian"]
            df_ana = filter(r -> r.type == "$(name)_Ana", df_bits)
            c = colors[name]
            plot!(p_std, df_ana.d, sqrt.(df_ana.var_error), label="$(name) AE", marker=:none, linestyle=:dot, color=c, lw=2)
        end
        save_figure(logger, p_std, "std_dev_comparison_$(bits)")

        # B. Expectation Comparison
        p_exp = plot(xlabel="Width d", ylabel="Expectation", xlims=(0, Inf))

        plot!(p_exp, df_tnn.d, df_tnn.mean_error, label="TN", marker=:circle, lw=2, color=colors["TNN"])

        # Sampled
        for name in ["Bernoulli", "Rademacher", "Gaussian"]
            df_cnn = filter(r -> r.type == name, df_bits)
            c = colors[name]
            plot!(p_exp, df_cnn.d, df_cnn.mean_error, label="$(name) CN", marker=:square, linestyle=:dash, color=c)
        end

        save_figure(logger, p_exp, "expectation_comparison_$(bits)")

        # C. 90% Error Level Comparison
        leg_pos = bits == (1, 1) ? :bottomleft : false
        title_str = "Bits $(bits)"
        p_q90 = plot(xlabel="Width d", ylabel="90% Error", yaxis=:log, legend=leg_pos, xlims=(0, Inf), title=title_str)

        # Mark naive computation threshold
        naive_d = binomial(m, 2)
        vline!(p_q90, [naive_d], label="Naive", color=:black, linestyle=:dash)

        plot!(p_q90, df_tnn.d, df_tnn.q90_error, label="TN", marker=:circle, lw=2, color=colors["TNN"])

        for name in ["Bernoulli", "Rademacher", "Gaussian"]
            df_cnn = filter(r -> r.type == name, df_bits)
            c = colors[name]
            plot!(p_q90, df_cnn.d, df_cnn.q90_error, label="$(name) CN", marker=:square, linestyle=:dash, color=c)
        end
        save_figure(logger, p_q90, "q90_comparison_$(bits)")

        # D. MAE Comparison
        leg_pos = bits == (1, 1) ? :bottomleft : false
        title_str = "Bits $(bits)"
        p_mae = plot(xlabel="Width d", ylabel="MAE", yaxis=:log, legend=leg_pos, xlims=(0, Inf), title=title_str)

        # Mark naive computation threshold
        naive_d = binomial(m, 2)
        vline!(p_mae, [naive_d], label="Naive", color=:black, linestyle=:dash)

        plot!(p_mae, df_tnn.d, df_tnn.mae_error, label="TN", marker=:circle, lw=2, color=colors["TNN"])

        for name in ["Bernoulli", "Rademacher", "Gaussian"]
            df_cnn = filter(r -> r.type == name, df_bits)
            c = colors[name]
            plot!(p_mae, df_cnn.d, df_cnn.mae_error, label="$(name) CN", marker=:square, linestyle=:dash, color=c)
        end
        save_figure(logger, p_mae, "mae_comparison_$(bits)")

        # Combined Plot
        p_combined = plot(p_exp, p_std, p_q90, layout=(1, 3), size=(1500, 400), bottom_margin=10mm, left_margin=10mm)
        save_figure(logger, p_combined, "combined_comparison_$(bits)")
        display(p_combined)
        
        # Remove ylabel for combined plot if not the first one
        if bits != (0, 0)
            plot!(p_q90, ylabel="")
            plot!(p_mae, ylabel="")
        end
        push!(q90_plots, p_q90)
        push!(mae_plots, p_mae)
    end

    # Combined Q90 Plot
    p_q90_all = plot(q90_plots..., layout=(1, 3), size=(1500, 400), bottom_margin=10mm, left_margin=10mm, link=:y)
    save_figure(logger, p_q90_all, "q90_comparison_combined")
    display(p_q90_all)

    # Combined MAE Plot
    p_mae_all = plot(mae_plots..., layout=(1, 3), size=(1500, 400), bottom_margin=10mm, left_margin=10mm, link=:y)
    save_figure(logger, p_mae_all, "mae_comparison_combined")
    display(p_mae_all)

    # D. TNN Bits Comparison (Expectation)
    p_bits = plot(xlabel="Width d", ylabel="Expectation", xlims=(0, Inf))
    
    tnn_results = filter(r -> r.type == "TNN", results)
    colors_bits = Dict((0,0) => :blue, (0,1) => :green, (1,1) => :orange)
    
    for bits in [(0,0), (0,1), (1,1)]
        sub = filter(r -> r.bits == bits, tnn_results)
        c = get(colors_bits, bits, :black)
        plot!(p_bits, sub.d, sub.mean_error, label="TN $(bits)", marker=:circle, lw=2, color=c)
    end
    save_figure(logger, p_bits, "tnn_bits_expectation_comparison")
    display(p_bits)

    # E. Learned ANDs vs Width
    p_learned = plot(xlabel="Width d", ylabel="Learned ANDs Count", xlims=(0, Inf), legend=:bottomright)
    
    plot!(p_learned, learned_ands_results.d, learned_ands_results.count_00, label="Learned (0,0)", marker=:circle, lw=2, color=:blue)
    plot!(p_learned, learned_ands_results.d, learned_ands_results.count_01, label="Learned (0,1)", marker=:square, lw=2, color=:green)
    plot!(p_learned, learned_ands_results.d, learned_ands_results.count_11, label="Learned (1,1)", marker=:diamond, lw=2, color=:orange)
    
    # Add total pairs line
    total_pairs = (m * (m - 1)) ÷ 2
    hline!(p_learned, [total_pairs], label="Total Pairs", linestyle=:dash, color=:black)
    
    save_figure(logger, p_learned, "tnn_learned_ands_vs_width")
    display(p_learned)

    # 4. Sparsity Comparison (New)
    println("Running Sparsity Evaluation...")
    s_evals = [2, 3, 4, 5, 6, 8, 10, 12, 15]
    sparsity_results = DataFrame(d=Int[], s=Int[], mae=Float64[], mae_00=Float64[], mae_01=Float64[], mae_11=Float64[])

    for d in ds
        model_filename = "model_d$(d).jls"
        model = load_data(logger, model_filename)
        if isnothing(model)
            println("Model for d=$d not found, skipping sparsity eval.")
            continue
        end

        for s_val in s_evals
            # Evaluate
            errors_dict = evaluate_model_stats(model, m, s_val; n_samples=500)

            # Combine all errors for overall MAE
            all_errs = vcat(errors_dict[(0, 0)], errors_dict[(0, 1)], errors_dict[(1, 1)])
            mae = mean(abs.(all_errs))

            mae_00 = isempty(errors_dict[(0, 0)]) ? NaN : mean(abs.(errors_dict[(0, 0)]))
            mae_01 = isempty(errors_dict[(0, 1)]) ? NaN : mean(abs.(errors_dict[(0, 1)]))
            mae_11 = isempty(errors_dict[(1, 1)]) ? NaN : mean(abs.(errors_dict[(1, 1)]))

            push!(sparsity_results, (d, s_val, mae, mae_00, mae_01, mae_11))
        end
    end

    save_data(logger, sparsity_results, "sparsity_results.jls")

    # Plot 1: Sparsity on x, MAE on y, lines for d
    p_sparsity_x = plot(xlabel="Sparsity s", ylabel="MAE", yaxis=:log, legend=:bottomright, xlims=(0, Inf))

    # Use a palette
    colors_d = cgrad(:viridis, length(ds), categorical=true)

    for (i, d_val) in enumerate(ds)
        sub = filter(r -> r.d == d_val, sparsity_results)
        plot!(p_sparsity_x, sub.s, sub.mae, label="d=$(d_val)", marker=:circle, color=colors_d[i])
    end
    save_figure(logger, p_sparsity_x, "tnn_mae_vs_sparsity")

    # Plot 2: Width on x, MAE on y, lines for s
    p_width_x = plot(xlabel="Width d", ylabel="MAE", yaxis=:log, xlims=(0, Inf))

    # Mark naive computation threshold
    naive_d = binomial(m, 2)
    vline!(p_width_x, [naive_d], label="Naive", color=:black, linestyle=:dash)

    colors_s = cgrad(:plasma, length(s_evals), categorical=true)

    for (i, s_val) in enumerate(s_evals)
        sub = filter(r -> r.s == s_val, sparsity_results)
        plot!(p_width_x, sub.d, sub.mae, label="s=$(s_val)", marker=:square, color=colors_s[i])
    end
    save_figure(logger, p_width_x, "tnn_mae_vs_width")

    display(p_sparsity_x)
    display(p_width_x)

    # 5. Gaussian Sparsity Evaluation (New)
    println("Running Gaussian Sparsity Evaluation...")
    gaussian_results = DataFrame(d=Int[], s=Int[], mae=Float64[])

    # Fixed calibration sparsity
    s_calib = 5

    for d in ds
        # Sample W for Gaussian construction
        W = randn(Float32, d, m)

        # Pre-compute unnormalized read-off vectors for all pairs
        pairs = [(i, j) for i in 1:m for j in i+1:m]
        r_tilde = zeros(Float32, d, length(pairs))
        for (k, (u, v)) in enumerate(pairs)
            r_tilde[:, k] .= sign.(W[:, u] .* W[:, v])
        end

        # Estimate normalization constant eta for s=5
        # eta = E[r_tilde_uv * relu(Wx)] given u,v active
        n_calib = 1000
        eta_acc = 0.0

        for _ in 1:n_calib
            # Pick random pair (u, v)
            idx = rand(1:length(pairs))
            u, v = pairs[idx]
            r_uv = r_tilde[:, idx]

            # Create x with u, v active + s_calib-2 others
            x = zeros(Float32, m)
            x[u] = 1f0
            x[v] = 1f0

            # Fill remaining s-2
            others = setdiff(1:m, [u, v])
            shuffle!(others)
            x[others[1:(s_calib-2)]] .= 1f0

            # Forward
            h = relu.(W * x)

            eta_acc += dot(r_uv, h)
        end
        eta = eta_acc / n_calib

        # Evaluate across sparsities
        for s_val in s_evals
            n_eval = 500
            err_acc = 0.0

            for _ in 1:n_eval
                x = sampleSparseVec(m, s_val)
                h = relu.(W * x)

                # Predictions: (r_tilde' * h) / eta
                preds = (r_tilde' * h) ./ eta

                # Targets
                targets = [x[p[1]] * x[p[2]] for p in pairs]

                err_acc += mean(abs.(preds .- targets))
            end
            mae = err_acc / n_eval
            push!(gaussian_results, (d, s_val, mae))
        end
    end

    save_data(logger, gaussian_results, "gaussian_sparsity_results.jls")

    # Plot 3: Gaussian MAE vs Sparsity
    p_gauss_sparsity = plot(xlabel="Sparsity s", ylabel="MAE", yaxis=:log, legend=:bottomright, xlims=(0, Inf))
    for (i, d_val) in enumerate(ds)
        sub = filter(r -> r.d == d_val, gaussian_results)
        plot!(p_gauss_sparsity, sub.s, sub.mae, label="d=$(d_val)", marker=:circle, color=colors_d[i])
    end
    save_figure(logger, p_gauss_sparsity, "gaussian_mae_vs_sparsity")

    # Plot 4: Gaussian MAE vs Width
    p_gauss_width = plot(xlabel="Width d", ylabel="MAE", yaxis=:log, xlims=(0, Inf))
    for (i, s_val) in enumerate(s_evals)
        sub = filter(r -> r.s == s_val, gaussian_results)
        plot!(p_gauss_width, sub.d, sub.mae, label="s=$(s_val)", marker=:square, color=colors_s[i])
    end
    save_figure(logger, p_gauss_width, "gaussian_mae_vs_width")

    # Plot 5: Comparison MAE vs Sparsity (Selected Widths)
    # Select a few widths
    sel_ds = [50, 100, 200, 400, 800]
    sel_ds = filter(d -> d in ds, sel_ds)

    p_comp_sparsity = plot(xlabel="Sparsity s", ylabel="MAE", yaxis=:log, legend=:bottomright, xlims=(0, Inf))

    comp_colors = cgrad(:viridis, length(sel_ds), categorical=true)

    for (i, d_val) in enumerate(sel_ds)
        # TNN
        sub_tnn = filter(r -> r.d == d_val, sparsity_results)
        plot!(p_comp_sparsity, sub_tnn.s, sub_tnn.mae, label="TN d=$(d_val)", marker=:circle, color=comp_colors[i], lw=2)

        # Gaussian
        sub_gauss = filter(r -> r.d == d_val, gaussian_results)
        plot!(p_comp_sparsity, sub_gauss.s, sub_gauss.mae, label="Gauss CN d=$(d_val)", marker=:none, linestyle=:dash, color=comp_colors[i], lw=2)
    end
    save_figure(logger, p_comp_sparsity, "tnn_vs_gaussian_mae_vs_sparsity")

    # Plot 6: Comparison MAE vs Width (Selected Sparsities)
    sel_ss = [2, 5, 10, 15]
    sel_ss = filter(s -> s in s_evals, sel_ss)

    p_comp_width = plot(xlabel="Width d", ylabel="MAE", yaxis=:log, xlims=(0, Inf))

    # Mark naive computation threshold
    naive_d = binomial(m, 2)
    vline!(p_comp_width, [naive_d], label="Naive", color=:black, linestyle=:dash)

    comp_colors_s = cgrad(:plasma, length(sel_ss), categorical=true)

    for (i, s_val) in enumerate(sel_ss)
        # TNN
        sub_tnn = filter(r -> r.s == s_val, sparsity_results)
        plot!(p_comp_width, sub_tnn.d, sub_tnn.mae, label="TN s=$(s_val)", marker=:circle, color=comp_colors_s[i], lw=2)

        # Gaussian
        sub_gauss = filter(r -> r.s == s_val, gaussian_results)
        plot!(p_comp_width, sub_gauss.d, sub_gauss.mae, label="Gauss CN s=$(s_val)", marker=:none, linestyle=:dash, color=comp_colors_s[i], lw=2)
    end
    save_figure(logger, p_comp_width, "tnn_vs_gaussian_mae_vs_width")

    display(p_gauss_sparsity)
    display(p_comp_sparsity)
    return logger
end

# Main execution
# run_hpo() # Uncomment to run HPO
logger = run_scaling_experiment(force_retrain=false)

