using Dates
using Serialization
using Plots

export ExpLogger, 
       experimentfolder, 
       resourcepath,
       save_data, 
       load_data, 
       save_figure, 
       load_logger, 
       get_path, 
       print_summary

"""
Elegant experiment logger for Julia - saves data, figures, and manages experiment runs.
"""
struct ExpLogger
    exp_name::String
    exp_id::String
    base_path::String
    tmp::Bool # temporary ≙ overwrite mode
end

# Constructor with smart defaults
function ExpLogger(exp_name::String; prefix, tmp=false, base_path=nothing, exp_id=nothing)
    if isnothing(base_path)
        base_path = joinpath(dirname(@__DIR__), "../experiments")
    end

    if isnothing(exp_id)
        if tmp
            exp_id = "$(prefix)_tmp"
        else
            timestamp = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
            exp_id = "$(prefix)_$(timestamp)"
        end
    end

    logger = ExpLogger(exp_name, exp_id, base_path, tmp)
    mkpath(experimentfolder(logger))
    return logger
end

# Core functionality
experimentfolder(logger::ExpLogger) = joinpath(logger.base_path, logger.exp_name, logger.exp_id)
resourcepath(logger::ExpLogger, name::String) = joinpath(experimentfolder(logger), name)

function save_data(logger::ExpLogger, data, filename::String)
    filepath = joinpath(experimentfolder(logger), filename)
    serialize(filepath, data)
end

function load_data(logger::ExpLogger, filename::String)
    filepath = joinpath(experimentfolder(logger), filename)
    return isfile(filepath) ? deserialize(filepath) : nothing
end

function save_figure(logger::ExpLogger, figure, figname::String; force_save=false)
    filepath = joinpath(experimentfolder(logger), "$(figname).svg")
    savefig(figure, filepath)
end

# Static loader with prefix filtering
function load_logger(exp_name::String; prefix=nothing, base_path=nothing)
    if isnothing(base_path)
        base_path = joinpath(dirname(@__DIR__), "../experiments")
    end
    
    exp_path = joinpath(base_path, exp_name)
    !isdir(exp_path) && error("Experiment folder not found: $exp_path")
    
    # Find run folders
    run_folders = filter(isdir, readdir(exp_path, join=true))
    isempty(run_folders) && error("No runs found for experiment: $exp_name")
    
    # Filter by prefix if specified
    if !isnothing(prefix)
        run_folders = filter(f -> startswith(basename(f), "$(prefix)_"), run_folders)
        isempty(run_folders) && error("No runs found with prefix '$prefix'")
    end
    
    # Get latest by timestamp
    function extract_timestamp(folder)
        m = match(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$", basename(folder))
        isnothing(m) ? DateTime(0) : DateTime(m[1], "yyyy_mm_dd_HH_MM_SS")
    end
    
    latest_folder = run_folders[argmax(extract_timestamp.(run_folders))]
    exp_id = basename(latest_folder)
    
    # Extract prefix from exp_id (format: prefix_timestamp)
    prefix_match = match(r"^(.+)_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", exp_id)
    extracted_prefix = isnothing(prefix_match) ? "unknown" : prefix_match[1]
    
    return ExpLogger(exp_name, prefix=extracted_prefix, exp_id=exp_id, base_path=base_path)
end

# Utility functions
get_path(logger::ExpLogger, filename::String) = joinpath(experimentfolder(logger), filename)

function print_summary(logger::ExpLogger)
    println("Experiment: $(logger.exp_name)")
    println("ID: $(logger.exp_id)")
    println("Folder: $(experimentfolder(logger))")
    println("Temporary mode: $(logger.tmp)")
    if logger.tmp
        println("TMP mode: writing to disk folder $(experimentfolder(logger))")
    end
end

# Make current run a tmp run: move folder to <prefix>_tmp and update logger
function make_tmp(logger::ExpLogger)
    if logger.tmp
        println("Already in tmp mode.")
        return
    end
    src = experimentfolder(logger)
    tmp_exp_id = "$(split(logger.exp_id, "_")[1])_tmp"
    dst = joinpath(logger.base_path, logger.exp_name, tmp_exp_id)
    if isdir(dst)
        rm(dst; force=true, recursive=true)
    end
    cp(src, dst; force=true)
    rm(src; force=true, recursive=true)
    logger.tmp = true
    logger.exp_id = tmp_exp_id
    println("Moved experiment to tmp folder: $dst")
end

# Make a reference copy: copy folder to <prefix>_ref
function make_ref(logger::ExpLogger)
    src = experimentfolder(logger)
    ref_exp_id = "$(split(logger.exp_id, "_")[1])_ref"
    dst = joinpath(logger.base_path, logger.exp_name, ref_exp_id)
    if isdir(dst)
        println("Reference folder already exists: $dst, overwriting...")
        rm(dst; force=true, recursive=true)
    end
    cp(src, dst; force=true)
    println("Copied experiment to reference folder: $dst")
end
