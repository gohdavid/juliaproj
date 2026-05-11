#!/usr/bin/env julia

"""
Evaluate generated samples against their training distribution using only
label-aligned pairwise-distance features.

Usage:
  julia --project=. scripts/eval_structural_samples.jl --samples path/to/samples.jls
"""

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
      julia --project=. scripts/eval_structural_samples.jl --samples SAMPLES.jls [options]

    Options:
      --samples PATH       Samples .jls payload. Required.
      --checkpoint PATH    Override checkpoint path. Default: samples payload checkpoint_path.
      --output-dir DIR     Output directory. Default: <samples-dir>/structural_eval.
      --max-samples N      Number of generated/data samples to compare. Default: all generated samples.
      --help               Show this message.
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
        elseif arg in ("--samples", "--checkpoint", "--output-dir", "--max-samples")
            i < length(args) || error("Missing value for $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            opts[key] = key == "max_samples" ? parse(Int, value) : value
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    haskey(opts, "samples") || error("Set --samples PATH")
    return opts
end

project_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))

function evaluate_structural_samples(opts)
    samples_path = project_path(String(opts["samples"]))
    isfile(samples_path) || error("samples file does not exist: $samples_path")
    payload = deserialize(samples_path)
    samples_all = Float32.(payload["samples"])

    checkpoint_path = haskey(opts, "checkpoint") ?
                      project_path(String(opts["checkpoint"])) :
                      project_path(String(payload["checkpoint_path"]))
    isfile(checkpoint_path) || error("checkpoint file does not exist: $checkpoint_path")
    checkpoint = deserialize(checkpoint_path)
    cfg = checkpoint["config"]

    max_samples = min(Int(get(opts, "max_samples", size(samples_all, 3))),
                      size(samples_all, 3))
    samples = samples_all[:, :, 1:max_samples]

    rng = Xoshiro(BoltzFlow.cfgint(cfg, "experiment.seed", 0))
    data_cfg = BoltzFlow._data_config(cfg)
    train_data_all = Float32.(BoltzFlow.generate_nbody_dataset(rng, data_cfg))
    if BoltzFlow.cfgbool(cfg, "data.center", false)
        train_data_all = BoltzFlow.center_positions(train_data_all)
    end
    train_data = train_data_all[:, :, 1:min(max_samples, size(train_data_all, 3))]

    metrics = merge(Dict{String,Any}(
        "samples_path" => samples_path,
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => get(checkpoint, "epoch", nothing),
        "n_generated_samples" => size(samples, 3),
        "n_data_samples" => size(train_data, 3),
        "feature" => "label-aligned pairwise node distances",
    ), BoltzFlow.pairwise_distance_distribution_metrics(train_data, samples))

    output_dir = project_path(String(get(opts, "output_dir",
                                         joinpath(dirname(samples_path), "structural_eval"))))
    mkpath(output_dir)
    serialize(joinpath(output_dir, "structural_metrics.jls"), metrics)
    open(joinpath(output_dir, "structural_metrics.txt"), "w") do io
        for key in sort!(collect(keys(metrics)); by=string)
            println(io, key, " = ", metrics[key])
        end
    end

    @printf("pairwise distance energy: %.6g\n", metrics["pairwise_distance_energy"])
    @printf("pairwise distance MMD:    %.6g\n", metrics["pairwise_distance_mmd"])
    println("wrote metrics: ", joinpath(output_dir, "structural_metrics.txt"))
    return metrics
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate_structural_samples(parse_args(ARGS))
end
