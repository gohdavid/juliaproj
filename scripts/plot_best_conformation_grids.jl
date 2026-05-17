#!/usr/bin/env julia

using CairoMakie
using HDF5
using Random
using Serialization
using Statistics

const PROJECT_ROOT = dirname(@__DIR__)

struct GridSpec
    name::String
    dataset_path::String
    samples_path::String
    output_path::String
    draw_ring_bond::Bool
    generated_seed::Union{Nothing,Int}
end

function center_frame(x)
    return x .- mean(x; dims=2)
end

function chain_points(x)
    return Point2f.(Float32.(x[1, :]), Float32.(x[2, :]))
end

function load_dataset_frames(path::AbstractString)
    isfile(path) || error("Missing dataset trajectory: $path")
    h5open(path, "r") do f
        return Float32.(read(f["traj"]))
    end
end

function load_generated_samples(path::AbstractString)
    isfile(path) || error("Missing generated samples: $path")
    payload = deserialize(path)
    haskey(payload, "samples") || error("No \"samples\" entry in $path")
    return Float32.(payload["samples"])
end

function evenly_spaced_indices(n::Integer, k::Integer)
    n >= k || error("Need at least $k frames, found $n")
    return unique(round.(Int, range(1, n; length=k)))
end

function sample_indices(n::Integer, k::Integer, seed::Union{Nothing,Int})
    seed === nothing && return evenly_spaced_indices(n, k)
    n >= k || error("Need at least $k frames, found $n")
    return sort(randperm(MersenneTwister(seed), n)[1:k])
end

function axis_limits(frames; pad=0.16f0)
    xmin, xmax = extrema(Float32.(frames[1, :, :]))
    ymin, ymax = extrema(Float32.(frames[2, :, :]))
    span = max(xmax - xmin, ymax - ymin, 1.0f0)
    xmid = (xmin + xmax) / 2.0f0
    ymid = (ymin + ymax) / 2.0f0
    half_width = (0.5f0 + pad) * span
    return (xmid - half_width, xmid + half_width),
           (ymid - half_width, ymid + half_width)
end

function axis_limits(frame::AbstractMatrix; pad=0.16f0)
    return axis_limits(reshape(frame, size(frame, 1), size(frame, 2), 1); pad)
end

function draw_polymer!(ax, frame; draw_ring_bond::Bool=false)
    pts = chain_points(frame)
    lines!(ax, pts; color=:gray20, linewidth=3.0)
    if draw_ring_bond && length(pts) > 2
        lines!(ax, Point2f[pts[end], pts[1]]; color=:gray55, linewidth=3.0)
    end
    scatter!(ax, pts; color=1:length(pts), colormap=:viridis,
             markersize=11, strokewidth=0)
end

function plot_grid(spec::GridSpec)
    dataset = load_dataset_frames(spec.dataset_path)
    generated = load_generated_samples(spec.samples_path)

    dataset_idx = evenly_spaced_indices(size(dataset, 3), 8)
    generated_idx = sample_indices(size(generated, 3), 8, spec.generated_seed)

    dataset_frames = cat((center_frame(dataset[:, :, i]) for i in dataset_idx)...; dims=3)
    generated_frames = cat((center_frame(generated[:, :, i]) for i in generated_idx)...; dims=3)
    fig = Figure(size=(1160, 1200), backgroundcolor=:white)
    Label(fig[1, 1:4], spec.name; fontsize=28, font=:bold, tellwidth=false)
    Label(fig[2, 1:4], "Dataset"; fontsize=18, tellwidth=false)
    Label(fig[5, 1:4], "Generated"; fontsize=18, tellwidth=false)

    for i in 1:16
        row = i <= 8 ? 3 + ((i - 1) ÷ 4) : 6 + ((i - 9) ÷ 4)
        col = ((i - 1) % 4) + 1
        frame = i <= 8 ? dataset_frames[:, :, i] : generated_frames[:, :, i - 8]
        ax = Axis(fig[row, col]; aspect=DataAspect(), xgridvisible=false,
                  ygridvisible=false, xticksvisible=false, yticksvisible=false,
                  xticklabelsvisible=false, yticklabelsvisible=false)
        hidespines!(ax)
        xlims, ylims = axis_limits(frame)
        xlims!(ax, xlims...)
        ylims!(ax, ylims...)
        draw_polymer!(ax, frame; draw_ring_bond=spec.draw_ring_bond)
    end

    rowsize!(fig.layout, 1, Fixed(42))
    rowsize!(fig.layout, 2, Fixed(28))
    rowsize!(fig.layout, 5, Fixed(28))
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    mkpath(dirname(spec.output_path))
    save(spec.output_path, fig; px_per_unit=2)
    return spec.output_path
end

function main()
    specs = [
        GridSpec(
            "Base: best structural checkpoint epoch 9",
            joinpath(PROJECT_ROOT, "outputs/rouse_base/f5ecf974b2a8/trajectories/traj_seed0.h5"),
            joinpath(PROJECT_ROOT, "runs/nbody_polymer_cnf_egnn_rouse_hdf5/b93840a65754/structural_checkpoint_eval/checkpoint_epoch_000009/samples.jls"),
            joinpath(PROJECT_ROOT, "runs/conformation_grids/base_best_epoch_000009_grid.png"),
            false,
            nothing,
        ),
        GridSpec(
            "Hairpin polymer: best structural checkpoint epoch 7",
            joinpath(PROJECT_ROOT, "outputs/rouse_hairpin_lj/3fd424a98954/trajectories/traj_seed25.h5"),
            joinpath(PROJECT_ROOT, "runs/nbody_polymer_hairpin_lj_cnf_egnn_3layer_rouse_hdf5/fa05e3687f52/structural_checkpoint_eval/checkpoint_epoch_000007/samples.jls"),
            joinpath(PROJECT_ROOT, "runs/conformation_grids/hairpin_polymer_best_epoch_000007_grid.png"),
            false,
            nothing,
        ),
        GridSpec(
            "Ring: best structural checkpoint epoch 11",
            joinpath(PROJECT_ROOT, "outputs/rouse_ring_bond/a7c4e6f5760f/trajectories/traj_seed26.h5"),
            joinpath(PROJECT_ROOT, "runs/nbody_polymer_ring_bond_cnf_egnn_3layer_rouse_hdf5/81b1db52e057/structural_checkpoint_eval/checkpoint_epoch_000011/samples.jls"),
            joinpath(PROJECT_ROOT, "runs/conformation_grids/ring_best_epoch_000011_grid.png"),
            true,
            20260511,
        ),
    ]

    for spec in specs
        println(plot_grid(spec))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
