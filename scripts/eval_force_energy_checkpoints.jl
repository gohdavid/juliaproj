#!/usr/bin/env julia

"""
Evaluate CNF score/force and centered potential metrics for checkpoint samples.

This expects samples produced by sample_eval_structural_checkpoints.jl under
<run>/structural_checkpoint_eval/checkpoint_epoch_XXXX/samples.jls.
"""

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(PROJECT_ROOT, "scripts", "eval_checkpoint_samples.jl"))

function usage()
    println("""
    Usage:
      julia --project=. scripts/eval_force_energy_checkpoints.jl --checkpoint-dir DIR [options]

    Options:
      --checkpoint-dir DIR  Directory containing checkpoint_epoch_*.jls files. Required.
      --samples-root DIR    Directory containing checkpoint_epoch_XXXX/samples.jls.
                          Default: <run>/structural_checkpoint_eval.
      --max-samples N      Number of samples per checkpoint. Default: 128.
      --batch-size N       CNF score batch size. Default: 64.
      --min-epoch N        First checkpoint epoch to evaluate. Default: 2.
      --max-epoch N        Last checkpoint epoch to evaluate. Default: all.
      --force              Recompute even if sample_eval_metrics.txt exists.
      --help               Show this message.
    """)
end

function parse_force_args(args)
    opts = Dict{String,Any}(
        "max_samples" => 128,
        "batch_size" => 64,
        "min_epoch" => 2,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--force"
            opts["force"] = true
            i += 1
        elseif arg in ("--checkpoint-dir", "--samples-root", "--max-samples",
                       "--batch-size", "--min-epoch", "--max-epoch")
            i < length(args) || error("Missing value for $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            opts[key] = key in ("max_samples", "batch_size", "min_epoch", "max_epoch") ?
                        parse(Int, value) : value
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    haskey(opts, "checkpoint_dir") || error("Set --checkpoint-dir DIR")
    return opts
end

force_project_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))

function force_checkpoint_paths(checkpoint_dir::AbstractString)
    paths = sort([
        joinpath(checkpoint_dir, name)
        for name in readdir(checkpoint_dir)
        if startswith(name, "checkpoint_epoch_") && endswith(name, ".jls")
    ])
    isempty(paths) && error("No checkpoint_epoch_*.jls files found in $checkpoint_dir")
    return paths
end

function checkpoint_epoch(path::AbstractString)
    m = match(r"checkpoint_epoch_(\d+)\.jls$", basename(path))
    m === nothing && error("Could not parse checkpoint epoch from $path")
    return parse(Int, m.captures[1])
end

function write_force_summary(path::AbstractString, records)
    open(path, "w") do io
        println(io, "epoch,n_samples,diffusion,score_mse,force_mse,score_cosine,force_cosine,potential_centered_mse,potential_centered_mae,potential_correlation,metrics_path")
        for rec in records
            println(io, join((
                rec["checkpoint_epoch"],
                rec["n_samples"],
                rec["diffusion"],
                rec["score_mse"],
                rec["force_mse"],
                rec["score_cosine"],
                rec["force_cosine"],
                rec["potential_centered_mse"],
                rec["potential_centered_mae"],
                rec["potential_correlation"],
                rec["metrics_path"],
            ), ","))
        end
    end
    return path
end

function run_force_energy_checkpoints(opts)
    checkpoint_dir = force_project_path(String(opts["checkpoint_dir"]))
    run_dir = dirname(checkpoint_dir)
    samples_root = force_project_path(String(get(opts, "samples_root",
                                                joinpath(run_dir, "structural_checkpoint_eval"))))
    min_epoch = Int(opts["min_epoch"])
    max_epoch = get(opts, "max_epoch", typemax(Int))
    max_epoch = Int(max_epoch)
    force = Bool(get(opts, "force", false))

    records = Vector{Dict{String,Any}}()
    for checkpoint_path in force_checkpoint_paths(checkpoint_dir)
        epoch = checkpoint_epoch(checkpoint_path)
        min_epoch <= epoch <= max_epoch || continue
        stem = splitext(basename(checkpoint_path))[1]
        samples_path = joinpath(samples_root, stem, "samples.jls")
        isfile(samples_path) || error("Missing samples for epoch $epoch: $samples_path")
        output_dir = joinpath(samples_root, stem, "force_energy_eval_128")
        metrics_path = joinpath(output_dir, "sample_eval_metrics.txt")
        if isfile(metrics_path) && !force
            @info "force/energy metrics already exist" epoch metrics_path
        else
            @info "evaluating force/energy metrics" epoch samples_path checkpoint_path
            evaluate_samples(Dict{String,Any}(
                "samples" => samples_path,
                "checkpoint" => checkpoint_path,
                "output_dir" => output_dir,
                "batch_size" => Int(opts["batch_size"]),
                "max_samples" => Int(opts["max_samples"]),
                "max_frames" => 0,
            ))
        end
        metrics = deserialize(joinpath(output_dir, "sample_eval_metrics.jls"))
        metrics["metrics_path"] = metrics_path
        push!(records, metrics)
    end

    summary_path = joinpath(samples_root, "force_energy_metrics_128.csv")
    write_force_summary(summary_path, records)
    println("wrote force/energy summary: ", summary_path)
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_force_energy_checkpoints(parse_force_args(ARGS))
end
