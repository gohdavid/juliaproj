#!/usr/bin/env julia

using Dates
using Printf
using Random
using Serialization

const DEFAULT_CHECKPOINT_DIR = "/home/davidgoh/orcd/scratch/juliaproj/runs/nbody_polymer_hairpin_lj_cnf_egnn_rouse_hdf5/c0d11ac2c64e/checkpoints"
const PROJECT_ROOT = dirname(@__DIR__)

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/sample_checkpoint_conformations.jl [options]

    Options:
      --checkpoint PATH       Checkpoint .jls file to sample from.
      --checkpoint-dir DIR    Directory with checkpoint_epoch_*.jls files.
                              Defaults to the finished hairpin LJ run.
      --n-samples N           Number of conformations to generate. Default: 1000.
      --batch-size N          Sampling batch size. Default: 64.
      --seed N                RNG seed. Default: checkpoint experiment seed, or 0.
      --output-dir DIR        Output directory. Default: <run>/samples/<checkpoint-stem>.
      --output-file NAME      Serialized output filename. Default: conformations.jls.
      --help                  Show this message.

    Output:
      A serialized Dict containing "samples" with shape (dim, n_atoms, n_samples),
      plus checkpoint metadata and the effective sampling settings.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}(
        "checkpoint_dir" => DEFAULT_CHECKPOINT_DIR,
        "n_samples" => 1000,
        "batch_size" => 64,
        "output_file" => "conformations.jls",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg in ("--checkpoint", "--checkpoint-dir", "--n-samples",
                       "--batch-size", "--seed", "--output-dir", "--output-file")
            i < length(args) || error("Missing value for $arg")
            value = args[i + 1]
            key = replace(arg[3:end], "-" => "_")
            if key in ("n_samples", "batch_size", "seed")
                opts[key] = parse(Int, value)
            else
                opts[key] = value
            end
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    return opts
end

function latest_checkpoint(checkpoint_dir::AbstractString)
    isdir(checkpoint_dir) || error("checkpoint dir does not exist: $checkpoint_dir")
    candidates = sort([
        joinpath(checkpoint_dir, name)
        for name in readdir(checkpoint_dir)
        if startswith(name, "checkpoint_epoch_") && endswith(name, ".jls")
    ])
    isempty(candidates) && error("No checkpoint_epoch_*.jls files found in $checkpoint_dir")
    return last(candidates)
end

function default_output_dir(checkpoint_path::AbstractString)
    checkpoint_dir = dirname(checkpoint_path)
    run_dir = dirname(checkpoint_dir)
    stem = splitext(basename(checkpoint_path))[1]
    return joinpath(run_dir, "samples", stem)
end

function cfg_value(cfg::AbstractDict, path::AbstractString, default=nothing)
    current = cfg
    for key in split(path, ".")
        if current isa AbstractDict && haskey(current, key)
            current = current[key]
        else
            return default
        end
    end
    return current
end

function sample_checkpoint_conformations(opts)
    checkpoint_path = haskey(opts, "checkpoint") ?
                      String(opts["checkpoint"]) :
                      latest_checkpoint(String(opts["checkpoint_dir"]))
    isfile(checkpoint_path) || error("checkpoint does not exist: $checkpoint_path")

    checkpoint = deserialize(checkpoint_path)
    cfg = checkpoint["config"]
    seed = Int(get(opts, "seed", cfg_value(cfg, "experiment.seed", 0)))
    rng = Xoshiro(seed)

    device = BoltzFlow._device_from_config(cfg)
    ctx, _ = BoltzFlow._cnf_context_from_config(cfg, device)
    params = checkpoint["params"] |> device

    n_samples = Int(opts["n_samples"])
    batch_size = Int(opts["batch_size"])
    n_samples > 0 || error("--n-samples must be positive")
    batch_size > 0 || error("--batch-size must be positive")

    field = ctx.field
    samples = Array{Float32}(undef, field.dim, field.n_atoms, n_samples)
    cpu = BoltzFlow.DiffEqFlux.Lux.cpu_device()

    cursor = 1
    while cursor <= n_samples
        stop = min(cursor + batch_size - 1, n_samples)
        local_n = stop - cursor + 1
        @info "sampling conformations" checkpoint_path epoch=get(checkpoint, "epoch", nothing) range="$(cursor):$(stop)" total=n_samples
        samples[:, :, cursor:stop] .= cpu(
            BoltzFlow.generate_cnf_samples(ctx, params, local_n; rng)
        )
        cursor = stop + 1
    end

    output_dir = String(get(opts, "output_dir", default_output_dir(checkpoint_path)))
    mkpath(output_dir)
    output_path = joinpath(output_dir, String(opts["output_file"]))
    payload = Dict{String,Any}(
        "samples" => samples,
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => get(checkpoint, "epoch", nothing),
        "checkpoint_loss" => get(checkpoint, "loss", nothing),
        "config" => cfg,
        "sampling" => Dict{String,Any}(
            "n_samples" => n_samples,
            "batch_size" => batch_size,
            "seed" => seed,
        ),
        "created_at" => Dates.now(),
    )
    serialize(output_path, payload)

    @info "sampling complete" output_path sample_shape=size(samples)
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    sample_checkpoint_conformations(parse_args(ARGS))
end
