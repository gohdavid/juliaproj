#!/usr/bin/env julia

"""
Plot median-filtered training curves from BoltzFlow run directories.

By default this scans `runs/` for run directories with `losses.jls` or
`checkpoints/checkpoint_epoch_*.jls`, then overlays all available curves.

Usage:
  julia --project=. scripts/plot_training_curves.jl
  julia --project=. scripts/plot_training_curves.jl --run-dir runs/nbody_polymer_hairpin_lj_cnf_egnn_rouse_hdf5/c0d11ac2c64e
  julia --project=. scripts/plot_training_curves.jl --root runs --window 101 --output runs/training_curves/all_training_curves.png
  julia --project=. scripts/plot_training_curves.jl --show-raw
"""

using CairoMakie
using Printf
using Serialization
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/plot_training_curves.jl [options]

    Options:
      --root DIR          Root to scan for runs. Default: runs.
      --run-dir DIR       Specific run directory. Can be repeated.
      --window N          Median-filter window in training steps. Default: 101.
      --output PATH       Plot output path. Default: runs/training_curves/training_curves.png.
      --show-raw          Also show faint unfiltered curves behind filtered curves.
      --help              Show this message.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}(
        "root" => "runs",
        "run_dirs" => String[],
        "window" => 101,
        "output" => joinpath("runs", "training_curves", "training_curves.png"),
        "show_raw" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--show-raw"
            opts["show_raw"] = true
            i += 1
        elseif arg in ("--root", "--run-dir", "--window", "--output")
            i < length(args) || error("Missing value for $arg")
            value = args[i + 1]
            if arg == "--run-dir"
                push!(opts["run_dirs"], value)
            elseif arg == "--window"
                opts["window"] = parse(Int, value)
            else
                opts[replace(arg[3:end], "-" => "_")] = value
            end
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    return opts
end

project_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))

function discover_run_dirs(root::AbstractString)
    root_path = project_path(root)
    isdir(root_path) || error("root does not exist: $root_path")

    dirs = Set{String}()
    for (dir, subdirs, files) in walkdir(root_path)
        if "losses.jls" in files
            push!(dirs, dir)
        end
        if basename(dir) == "checkpoints" &&
           any(name -> startswith(name, "checkpoint_epoch_") && endswith(name, ".jls"), files)
            push!(dirs, dirname(dir))
        end
        filter!(subdir -> subdir != "samples", subdirs)
    end
    return sort!(collect(dirs))
end

function latest_checkpoint_path(run_dir::AbstractString)
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    isdir(checkpoint_dir) || return nothing
    candidates = sort([
        joinpath(checkpoint_dir, name)
        for name in readdir(checkpoint_dir)
        if startswith(name, "checkpoint_epoch_") && endswith(name, ".jls")
    ])
    return isempty(candidates) ? nothing : last(candidates)
end

function run_label(run_dir::AbstractString, cfg=nothing)
    name = cfg isa AbstractDict ? String(BoltzFlow.cfgget(cfg, "experiment.name", basename(dirname(run_dir)))) :
           basename(dirname(run_dir))
    return "$(name)/$(basename(run_dir))"
end

function loss_curve_from_run(run_dir::AbstractString)
    losses_path = joinpath(run_dir, "losses.jls")
    if isfile(losses_path)
        payload = deserialize(losses_path)
        losses = Float64.(payload["losses"])
        records = get(payload, "records", nothing)
        steps = if records isa AbstractVector && length(records) == length(losses)
            Float64[get(record, "step", i) for (i, record) in enumerate(records)]
        else
            Float64.(1:length(losses))
        end
        epochs = if records isa AbstractVector && length(records) == length(losses)
            Float64[get(record, "epoch", NaN) for record in records]
        else
            fill(NaN, length(losses))
        end
        cfg = get(payload, "config", nothing)
        return (; run_dir, source=losses_path, label=run_label(run_dir, cfg),
                steps, epochs, losses)
    end

    checkpoint_path = latest_checkpoint_path(run_dir)
    checkpoint_path === nothing && error("No losses.jls or checkpoints found in $run_dir")
    checkpoint = deserialize(checkpoint_path)
    losses = Float64.(checkpoint["losses"])
    steps = Float64.(1:length(losses))
    epochs = fill(Float64(get(checkpoint, "epoch", NaN)), length(losses))
    cfg = get(checkpoint, "config", nothing)
    return (; run_dir, source=checkpoint_path, label=run_label(run_dir, cfg),
            steps, epochs, losses)
end

function sliding_median(values::AbstractVector{<:Real}, window::Integer)
    n = length(values)
    n == 0 && return Float64[]
    w = max(Int(window), 1)
    half = w ÷ 2
    filtered = Vector{Float64}(undef, n)
    for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        filtered[i] = median(@view values[lo:hi])
    end
    return filtered
end

function ensure_odd_window(window::Integer)
    window > 0 || error("--window must be positive")
    isodd(window) ? Int(window) : Int(window) + 1
end

function plot_training_curves(opts)
    run_dirs = isempty(opts["run_dirs"]) ?
               discover_run_dirs(String(opts["root"])) :
               [project_path(path) for path in opts["run_dirs"]]
    isempty(run_dirs) && error("No run directories with losses/checkpoints found")

    curves = [loss_curve_from_run(run_dir) for run_dir in run_dirs]
    filter!(curve -> !isempty(curve.losses), curves)
    isempty(curves) && error("No non-empty loss curves found")

    window = ensure_odd_window(Int(opts["window"]))
    output_path = project_path(String(opts["output"]))
    mkpath(dirname(output_path))

    fig = Figure(size=(1100, 680), fontsize=18)
    ax = Axis(fig[1, 1],
              xlabel="training step",
              ylabel="negative log likelihood",
              title="Training curves, median filter window = $window")
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
    hidespines!(ax, :t, :r)

    palette = Makie.wong_colors()
    for (idx, curve) in enumerate(curves)
        color = palette[mod1(idx, length(palette))]
        if Bool(opts["show_raw"])
            lines!(ax, curve.steps, curve.losses; color=(color, 0.18), linewidth=1)
        end
        filtered = sliding_median(curve.losses, window)
        lines!(ax, curve.steps, filtered; color, linewidth=3, label=curve.label)
    end

    axislegend(ax; position=:rt, framevisible=false, nbanks=max(1, cld(length(curves), 8)))
    save(output_path, fig)

    println("wrote ", output_path)
    for curve in curves
        @printf("%-80s points=%d source=%s\n", curve.label, length(curve.losses), curve.source)
    end
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_training_curves(parse_args(ARGS))
end
