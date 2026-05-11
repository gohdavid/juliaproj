#!/usr/bin/env julia

"""
Sample every checkpoint in a run and evaluate generated structures with
label-aligned pairwise-distance energy and MMD.

Usage:
  julia --project=. scripts/sample_eval_structural_checkpoints.jl --checkpoint-dir path/to/checkpoints
"""

using Dates
using Printf
using Random
using Serialization

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/sample_eval_structural_checkpoints.jl --checkpoint-dir DIR [options]

    Options:
      --checkpoint-dir DIR  Directory containing checkpoint_epoch_*.jls files. Required.
      --output-dir DIR      Output directory. Default: <run>/structural_checkpoint_eval.
      --n-samples N         Number of samples per checkpoint. Default: config sampling.n_samples.
      --batch-size N        Sampling batch size. Default: n-samples, one solve per checkpoint.
      --seed N              Base sampling seed. Default: checkpoint config experiment.seed.
      --force               Resample checkpoints even if sample payloads already exist.
      --help                Show this message.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--force"
            opts["force"] = true
            i += 1
        elseif arg in ("--checkpoint-dir", "--output-dir", "--n-samples", "--batch-size", "--seed")
            i < length(args) || error("Missing value for $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            opts[key] = key in ("n_samples", "batch_size", "seed") ? parse(Int, value) : value
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    haskey(opts, "checkpoint_dir") || error("Set --checkpoint-dir DIR")
    return opts
end

project_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))

function checkpoint_paths(checkpoint_dir::AbstractString)
    isdir(checkpoint_dir) || error("checkpoint dir does not exist: $checkpoint_dir")
    paths = sort([
        joinpath(checkpoint_dir, name)
        for name in readdir(checkpoint_dir)
        if startswith(name, "checkpoint_epoch_") && endswith(name, ".jls")
    ])
    isempty(paths) && error("No checkpoint_epoch_*.jls files found in $checkpoint_dir")
    return paths
end

function sample_from_checkpoint(ctx, params, normalizer, n_samples::Int,
                                batch_size::Int, rng::AbstractRNG)
    field = ctx.field
    samples = Array{Float32}(undef, field.dim, field.n_atoms, n_samples)
    cpu = BoltzFlow.DiffEqFlux.Lux.cpu_device()
    cursor = 1
    while cursor <= n_samples
        stop = min(cursor + batch_size - 1, n_samples)
        local_n = stop - cursor + 1
        samples[:, :, cursor:stop] .= cpu(
            BoltzFlow.generate_normalized_cnf_samples(ctx, params, local_n, normalizer; rng)
        )
        cursor = stop + 1
    end
    return samples
end

function write_metrics_text(path::AbstractString, metrics)
    open(path, "w") do io
        for key in sort!(collect(keys(metrics)); by=string)
            println(io, key, " = ", metrics[key])
        end
    end
    return path
end

function write_summary_csv(path::AbstractString, records)
    open(path, "w") do io
        println(io, "epoch,checkpoint_loss,pairwise_distance_energy,pairwise_distance_mmd,samples_path")
        for rec in records
            println(io, join((
                rec["checkpoint_epoch"],
                rec["checkpoint_loss"],
                rec["pairwise_distance_energy"],
                rec["pairwise_distance_mmd"],
                rec["samples_path"],
            ), ","))
        end
    end
    return path
end

function run_checkpoint_structural_eval(opts)
    checkpoint_dir = project_path(String(opts["checkpoint_dir"]))
    checkpoints = checkpoint_paths(checkpoint_dir)
    run_dir = dirname(checkpoint_dir)
    output_dir = project_path(String(get(opts, "output_dir",
                                         joinpath(run_dir, "structural_checkpoint_eval"))))
    mkpath(output_dir)

    first_checkpoint = deserialize(first(checkpoints))
    cfg = first_checkpoint["config"]
    base_seed = Int(get(opts, "seed", BoltzFlow.cfgint(cfg, "experiment.seed", 0)))
    n_samples = Int(get(opts, "n_samples", BoltzFlow.cfgint(cfg, "sampling.n_samples", 1000)))
    batch_size = Int(get(opts, "batch_size", n_samples))
    n_samples > 0 || error("--n-samples must be positive")
    batch_size > 0 || error("--batch-size must be positive")

    data_rng = Xoshiro(BoltzFlow.cfgint(cfg, "experiment.seed", 0))
    data_cfg = BoltzFlow._data_config(cfg)
    train_data_all = Float32.(BoltzFlow.generate_nbody_dataset(data_rng, data_cfg))
    if BoltzFlow.cfgbool(cfg, "data.center", false)
        train_data_all = BoltzFlow.center_positions(train_data_all)
    end
    train_data = train_data_all[:, :, 1:min(n_samples, size(train_data_all, 3))]

    device = BoltzFlow._device_from_config(cfg)
    ctx, _ = BoltzFlow._cnf_context_from_config(cfg, device)
    normalizer = BoltzFlow.cfgget(cfg, "data.normalizer", BoltzFlow.identity_data_normalizer())
    force = Bool(get(opts, "force", false))

    records = Vector{Dict{String,Any}}()
    for checkpoint_path in checkpoints
        checkpoint = deserialize(checkpoint_path)
        epoch = Int(get(checkpoint, "epoch", length(records) + 1))
        stem = splitext(basename(checkpoint_path))[1]
        checkpoint_output_dir = joinpath(output_dir, stem)
        mkpath(checkpoint_output_dir)
        samples_path = joinpath(checkpoint_output_dir, "samples.jls")

        local samples
        if isfile(samples_path) && !force
            @info "loading existing checkpoint samples" epoch samples_path
            samples = Float32.(deserialize(samples_path)["samples"])
        else
            @info "sampling checkpoint" epoch checkpoint_path n_samples batch_size
            params = checkpoint["params"] |> device
            rng = Xoshiro(base_seed + epoch)
            samples = sample_from_checkpoint(ctx, params, normalizer, n_samples, batch_size, rng)
            serialize(samples_path, Dict{String,Any}(
                "samples" => samples,
                "checkpoint_path" => checkpoint_path,
                "checkpoint_epoch" => epoch,
                "checkpoint_loss" => get(checkpoint, "loss", nothing),
                "config" => checkpoint["config"],
                "sampling" => Dict{String,Any}(
                    "n_samples" => n_samples,
                    "batch_size" => batch_size,
                    "seed" => base_seed + epoch,
                ),
                "data_normalizer" => normalizer,
                "created_at" => Dates.now(),
            ))
        end

        metrics = merge(Dict{String,Any}(
            "samples_path" => samples_path,
            "checkpoint_path" => checkpoint_path,
            "checkpoint_epoch" => epoch,
            "checkpoint_loss" => get(checkpoint, "loss", nothing),
            "n_generated_samples" => size(samples, 3),
            "n_data_samples" => size(train_data, 3),
            "feature" => "label-aligned pairwise node distances",
        ), BoltzFlow.pairwise_distance_distribution_metrics(train_data, samples))
        serialize(joinpath(checkpoint_output_dir, "structural_metrics.jls"), metrics)
        write_metrics_text(joinpath(checkpoint_output_dir, "structural_metrics.txt"), metrics)
        push!(records, metrics)
        @printf("epoch %d: energy %.6g, MMD %.6g\n",
                epoch, metrics["pairwise_distance_energy"], metrics["pairwise_distance_mmd"])
    end

    serialize(joinpath(output_dir, "structural_metrics_all.jls"), records)
    write_summary_csv(joinpath(output_dir, "structural_metrics_all.csv"), records)
    println("wrote summary: ", joinpath(output_dir, "structural_metrics_all.csv"))
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_checkpoint_structural_eval(parse_args(ARGS))
end
