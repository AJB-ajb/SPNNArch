using Test
using JlCode
using Plots
using Dates

@testset "ExpLogger Tests" begin

    @testset "Initialization" begin
        # Test default initialization
        logger = ExpLogger("test_exp", prefix="test")
        @test logger.exp_name == "test_exp"
        @test startswith(logger.exp_id, "test_")
        @test !logger.tmp

        # Test tmp mode
        tmp_logger = ExpLogger("tmp_exp", prefix="test", tmp=true)
        @test tmp_logger.tmp
        # Even in tmp mode, folder is created on disk
        @test isdir(experimentfolder(tmp_logger))

        # Test custom exp_id
        custom_logger = ExpLogger("custom_exp", prefix="test", exp_id="custom_id")
        @test custom_logger.exp_id == "custom_id"
    end

    @testset "Data Operations" begin
        logger = ExpLogger("data_test", prefix="test")
        test_data = Dict("key" => "value", "numbers" => [1, 2, 3])

        # Save and load data
        save_data(logger, test_data, "test.jls")
        loaded_data = load_data(logger, "test.jls")
        @test loaded_data == test_data

        # Test nonexistent data
        @test isnothing(load_data(logger, "nonexistent.jls"))

        # Test tmp mode - now saves to disk in tmp folder
        tmp_logger = ExpLogger("tmp_data", prefix="test", tmp=true)
        save_data(tmp_logger, test_data, "test.jls")
        @test load_data(tmp_logger, "test.jls") == test_data
        # Check that file exists on disk
        @test isfile(joinpath(experimentfolder(tmp_logger), "test.jls"))
    end

    @testset "Figure Operations" begin
        logger = ExpLogger("fig_test", prefix="test")

        # Create simple plot
        p = plot([1, 2, 3], [1, 4, 2], title="Test Plot")
        save_figure(logger, p, "test_plot")

        # Check file exists
        fig_path = joinpath(experimentfolder(logger), "test_plot.svg")
        @test isfile(fig_path)

        # Test tmp mode - saves to disk
        tmp_logger = ExpLogger("tmp_fig", prefix="test", tmp=true)
        save_figure(tmp_logger, p, "tmp_plot")
        # Check that file exists on disk
        fig_path = joinpath(experimentfolder(tmp_logger), "tmp_plot.svg")
        @test isfile(fig_path)
    end

    @testset "Load Logger" begin
        # Create logger and save data
        original = ExpLogger("load_test", prefix="original")
        test_data = "test_value"
        save_data(original, test_data, "data.jls")

        # Load latest logger
        loaded = load_logger("load_test")
        @test loaded.exp_name == "load_test"
        @test load_data(loaded, "data.jls") == test_data

        # Test prefix filtering
        sleep(0.1)  # Ensure different timestamp
        second = ExpLogger("load_test", prefix="second")
        save_data(second, "second_value", "data.jls")

        # Load with prefix
        loaded_original = load_logger("load_test", prefix="original")
        @test load_data(loaded_original, "data.jls") == test_data

        loaded_second = load_logger("load_test", prefix="second")
        @test load_data(loaded_second, "data.jls") == "second_value"
    end

    @testset "Utilities" begin
        logger = ExpLogger("util_test", prefix="test")

        # Test path utilities
        expected_path = joinpath(experimentfolder(logger), "test_file.txt")
        @test get_path(logger, "test_file.txt") == expected_path

        # Test print_summary (should not error)
        @test_nowarn print_summary(logger)

        tmp_logger = ExpLogger("tmp_util", prefix="test", tmp=true)
        @test_nowarn print_summary(tmp_logger)
    end

    @testset "Error Handling" begin
        # Test loading nonexistent experiment
        @test_throws ErrorException load_logger("nonexistent_exp")

        # Test loading with nonexistent prefix
        ExpLogger("prefix_test", prefix="existing")
        @test_throws ErrorException load_logger("prefix_test", prefix="nonexistent")
    end
end

# Cleanup: Remove all test experiment folders
println("Cleaning up test folders...")
test_base_path = joinpath(dirname(@__DIR__), "../experiments")
test_experiments = ["test_exp", "tmp_exp", "custom_exp", "data_test", "tmp_data",
    "fig_test", "tmp_fig", "load_test", "util_test", "tmp_util", "prefix_test"]

for exp_name in test_experiments
    exp_path = joinpath(test_base_path, exp_name)
    if isdir(exp_path)
        rm(exp_path; force=true, recursive=true)
        println("  Removed: $exp_path")
    end
end
println("Cleanup complete!")
