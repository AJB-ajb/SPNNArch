using Test

include("polygon_defs.jl")

begin
    (@__MODULE__) != Main && return
    (PROGRAM_FILE == @__FILE__) || (contains(PROGRAM_FILE, "terminal")) || return

    @testset "Loss consistency" begin # check if derived loss is consistent with loss calculated via numeric integration
        for nf in 3:20, r in [0.9, 1.0], β in [-0.25, -1.] # 0.1

            W = polygon_matrix(nf, r)
            b_vec = fill(β, nf)
            # Calculate the loss using the derived formula
            ρ = r^2
            γ = -β / r^2

            γ < 0 && continue # invalid

            k = k_of_nf(γ, nf)

            loss_derived = l1_loss_derived(; r, β=β, k, nf)
            loss_numeric = l1_loss(W, b_vec, n_integration=400)

            loss_numeric, (feature_benefit, interference) = l1_loss_extended(W, b_vec, n_integration=400)
            total_interference = 2 * nf * sum(interference_term(ρ, γ, cos(Δi * 2π / nf)) for Δi in 1:k; init=0.)

            loss_gen, (benefit_gen, interference_gen) = gen_l1_loss_extended(W, b_vec)

            if abs(loss_derived - loss_numeric) > 1e-5
                @show nf k ρ γ r β
                @show (loss_derived - loss_numeric)
                @show (total_interference - interference)
                @show (nf * benefit_term(ρ, γ) - feature_benefit)
            end
            if abs(loss_gen - loss_numeric) > 1e-5
                @show nf k ρ γ r β
                @show (loss_gen - loss_numeric)
                @show (benefit_gen - feature_benefit)
                @show (interference_gen - total_interference)
            end

            @test loss_derived ≈ loss_numeric atol = 1e-5

            @test benefit_gen ≈ feature_benefit atol = 1e-5
            @test interference_gen ≈ total_interference atol = 1e-5

            # check numerical benefit and interference terms separately
            @test feature_benefit ≈ nf * benefit_term(ρ, γ) atol = 1e-4
            @test interference ≈ total_interference atol = 1e-4

        end

    end

end

using BenchmarkTools

let benchmark_run = false # benchmark the three methods if run = true
    benchmark_run || return
    (@__MODULE__) != Main && return
    (PROGRAM_FILE == @__FILE__) || (contains(PROGRAM_FILE, "terminal")) || return

    function bench_case(nf, r, β; n_integration=200)
        W = polygon_matrix(nf, r)
        b = fill(β, nf)
        γ = -β / r^2
        k = k_of_nf(γ, nf)

        println("Benchmark case: nf=$nf, r=$r, β=$β, γ=$(round(γ, sigdigits=6)), k=$k, n_integration=$n_integration")

        # warmup to compile
        l1_loss_derived(; r=r, β=β, k=k, nf=nf)
        l1_loss(W, b; n_integration=n_integration)
        gen_l1_loss_der(W, b)

        println("\nTimings (BenchmarkTools):")
        print("- l1_loss_derived: ")
        @btime l1_loss_derived(; r=$r, β=$β, k=$k, nf=$nf)

        print("- l1_loss (numeric): ")
        @btime l1_loss($W, $b; n_integration=$n_integration)

        print("- gen_l1_loss_der: ")
        @btime gen_l1_loss_der($W, $b)

        println("\nValue sanity check:")
        println("  derived = ", l1_loss_derived(; r=r, β=β, k=k, nf=nf))
        println("  numeric  = ", l1_loss(W, b; n_integration=n_integration))
        println("  general  = ", gen_l1_loss_der(W, b))
    end

    # Example cases; adjust or extend as needed
    cases = [(12, 1.0, -0.5), (20, 1.0, -0.5), (50, 1.0, -0.2)]
    for (nf, r, β) in cases
        bench_case(nf, r, β; n_integration=200)
        println("="^60, "\n")
    end
end