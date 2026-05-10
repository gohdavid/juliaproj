#!/usr/bin/env julia

"""
Plot Rouse potential/force differences with the same histogram style as
`rouse_score_analytics.jl`.

Usage:
    julia --project=. scripts/rouse_difference_analytics.jl configs/plots/rouse_differences.yaml
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using HDF5
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "config.jl"))
include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))
using .BoltzFlow

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const OKABE_ITO_BLUE = colorant"#0072B2"
const OKABE_ITO_VERMILLION = colorant"#D55E00"

function style_axis!(ax)
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
    hidespines!(ax, :t, :r)
    return ax
end

function project_path(path::AbstractString)
    return isabspath(path) ? path : joinpath(PROJECT_ROOT, path)
end

function seed_filename(filename::AbstractString, seed::Int)
    root, ext = splitext(filename)
    return string(root, "_seed", seed, ext)
end

function frame_hdf5_path(output_path::AbstractString)
    root, ext = splitext(output_path)
    return isempty(ext) ? string(output_path, ".h5") : string(root, ".h5")
end

function output_child_path(cfg, out_dir::AbstractString, path_key::AbstractString,
                           file_key::AbstractString, default_file::AbstractString)
    file = cfgget(cfg, file_key, nothing)
    file !== nothing && return joinpath(out_dir, String(file))

    legacy_path = cfgget(cfg, path_key, nothing)
    legacy_path !== nothing && return joinpath(out_dir, basename(String(legacy_path)))

    return joinpath(out_dir, default_file)
end

function experiment_output_paths_and_dir(exp_cfg, prefer_hdf5::Bool)
    output_dir = cfgget(exp_cfg, "output.dir", nothing)
    output_dir === nothing && error("Experiment config needs output.dir")
    legacy_output = cfgget(exp_cfg, "output.path", nothing)
    default_file = legacy_output === nothing ? "traj.jls" : basename(String(legacy_output))
    output_file = String(cfgget(exp_cfg, "output.file", default_file))
    output_hdf5_file = String(cfgget(exp_cfg, "output.hdf5_file", frame_hdf5_path(output_file)))
    selected_file = prefer_hdf5 ? output_hdf5_file : output_file
    data_dir = project_path(config_output_dir(exp_cfg; default_name="rouse_raw"))

    n_workers = cfgint(exp_cfg, "parallel.workers", 1)
    if n_workers > 1
        seed0 = cfgint(exp_cfg, "seed", 11)
        stride = cfgint(exp_cfg, "parallel.seed_stride", 1)
        paths = [joinpath(data_dir, seed_filename(selected_file, seed0 + (worker_id - 1) * stride))
                 for worker_id in 1:n_workers]
        all(isfile, paths) && return paths, data_dir

        experiment_name = String(cfgget(exp_cfg, "output.experiment_name", "rouse_raw"))
        root = project_path(joinpath(String(output_dir), experiment_name))
        if isdir(root)
            for hash_dir in sort(readdir(root; join=true))
                isdir(hash_dir) || continue
                candidate = [joinpath(hash_dir, seed_filename(selected_file, seed0 + (worker_id - 1) * stride))
                             for worker_id in 1:n_workers]
                all(isfile, candidate) && return candidate, dirname(candidate[1])
            end
        end
        return paths, data_dir
    end
    path = joinpath(data_dir, selected_file)
    isfile(path) && return [path], data_dir

    experiment_name = String(cfgget(exp_cfg, "output.experiment_name", "rouse_raw"))
    root = project_path(joinpath(String(output_dir), experiment_name))
    if isdir(root)
        for hash_dir in sort(readdir(root; join=true))
            isdir(hash_dir) || continue
            candidate = joinpath(hash_dir, selected_file)
            isfile(candidate) && return [candidate], dirname(candidate)
        end
    end
    return [path], data_dir
end

function rouse_relaxation_time(n::Int, k_over_xi::Real)
    lambda1 = 2.0 * (1.0 - cos(pi / Float64(n)))
    return 1.0 / (Float64(k_over_xi) * lambda1)
end

function nonideal_param_vector(sim_cfg)
    nonideal = get(sim_cfg, "nonideal", nothing)
    nonideal === nothing && return Float32[]
    lj = get(nonideal, "lennard_jones", Dict())
    ev = get(nonideal, "excluded_volume", Dict())
    conf = get(nonideal, "confinement", Dict())
    ring_lj = get(nonideal, "ring_lj", Dict())
    ring_bond = get(nonideal, "ring_bond", Dict())
    return Float32[
        0.0f0,
        0.0f0,
        0.0f0,
        get(lj, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(lj, "epsilon", 0.0f0)),
        Float32(get(lj, "sigma", 1.0f0)),
        Float32(get(lj, "softening", 0.0f0)),
        Float32(get(lj, "cutoff", 0.0f0)),
        get(lj, "exclude_bonded", true) ? 1.0f0 : 0.0f0,
        get(lj, "shift", true) ? 1.0f0 : 0.0f0,
        get(ev, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(ev, "epsilon", 0.0f0)),
        Float32(get(ev, "sigma", 1.0f0)),
        Float32(get(ev, "softening", 0.0f0)),
        Float32(get(ev, "power", 12.0f0)),
        Float32(get(ev, "cutoff", 0.0f0)),
        get(ev, "exclude_bonded", true) ? 1.0f0 : 0.0f0,
        get(conf, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(conf, "strength", 0.0f0)),
        get(conf, "centered", true) ? 1.0f0 : 0.0f0,
        get(get(nonideal, "hairpin_lj", Dict()), "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(get(nonideal, "hairpin_lj", Dict()), "epsilon", 0.0f0)),
        Float32(get(get(nonideal, "hairpin_lj", Dict()), "sigma", 1.0f0)),
        Float32(get(get(nonideal, "hairpin_lj", Dict()), "softening", 0.0f0)),
        Float32(get(get(nonideal, "hairpin_lj", Dict()), "cutoff", 0.0f0)),
        get(get(nonideal, "hairpin_lj", Dict()), "shift", true) ? 1.0f0 : 0.0f0,
        Float32(get(get(nonideal, "hairpin_lj", Dict()), "min_separation", 4)),
        0.0f0,
        1.0f0,
        0.0f0,
        get(ring_lj, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(ring_lj, "epsilon", 0.0f0)),
        Float32(get(ring_lj, "sigma", 1.0f0)),
        Float32(get(ring_lj, "softening", 0.0f0)),
        Float32(get(ring_lj, "cutoff", 0.0f0)),
        get(ring_lj, "shift", true) ? 1.0f0 : 0.0f0,
        get(ring_bond, "enabled", false) ? 1.0f0 : 0.0f0,
    ]
end

function sim_config_from_experiment_config(exp_cfg; force_ideal::Bool=false)
    dim = cfgint(exp_cfg, "rouse.dim", 2)
    n_beads = cfgint(exp_cfg, "rouse.n_beads", 32)
    diffusion = cfgfloat32(exp_cfg, "rouse.diffusion", 1.0f0)
    bond_length = cfgfloat32(exp_cfg, "rouse.bond_length", 1.0f0)
    k_over_xi = cfgfloat32(exp_cfg, "rouse.k_over_xi",
                           3.0f0 * diffusion / bond_length^2)
    nonideal = force_ideal ? Dict(
        "lennard_jones" => Dict("enabled" => false),
        "excluded_volume" => Dict("enabled" => false),
        "confinement" => Dict("enabled" => false),
    ) : Dict(
        "lennard_jones" => Dict(
            "enabled" => cfgbool(exp_cfg, "nonideal.lennard_jones.enabled", false),
            "epsilon" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.epsilon", 0.0f0),
            "sigma" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.sigma", 1.0f0),
            "softening" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.softening", 0.0f0),
            "cutoff" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.cutoff", 0.0f0),
            "exclude_bonded" => cfgbool(exp_cfg, "nonideal.lennard_jones.exclude_bonded", true),
            "shift" => cfgbool(exp_cfg, "nonideal.lennard_jones.shift", true),
        ),
        "excluded_volume" => Dict(
            "enabled" => cfgbool(exp_cfg, "nonideal.excluded_volume.enabled", false),
            "epsilon" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.epsilon", 0.0f0),
            "sigma" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.sigma", 1.0f0),
            "softening" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.softening", 0.0f0),
            "power" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.power", 12.0f0),
            "cutoff" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.cutoff", 0.0f0),
            "exclude_bonded" => cfgbool(exp_cfg, "nonideal.excluded_volume.exclude_bonded", true),
        ),
        "confinement" => Dict(
            "enabled" => cfgbool(exp_cfg, "nonideal.confinement.enabled", false),
            "strength" => cfgfloat32(exp_cfg, "nonideal.confinement.strength", 0.0f0),
            "centered" => cfgbool(exp_cfg, "nonideal.confinement.centered", true),
        ),
        "hairpin_lj" => Dict(
            "enabled" => cfgbool(exp_cfg, "nonideal.hairpin_lj.enabled", false),
            "epsilon" => cfgfloat32(exp_cfg, "nonideal.hairpin_lj.epsilon", 0.0f0),
            "sigma" => cfgfloat32(exp_cfg, "nonideal.hairpin_lj.sigma", 1.0f0),
            "softening" => cfgfloat32(exp_cfg, "nonideal.hairpin_lj.softening", 0.0f0),
            "cutoff" => cfgfloat32(exp_cfg, "nonideal.hairpin_lj.cutoff", 0.0f0),
            "shift" => cfgbool(exp_cfg, "nonideal.hairpin_lj.shift", true),
            "min_separation" => cfgint(exp_cfg, "nonideal.hairpin_lj.min_separation", 4),
        ),
        "ring_lj" => Dict(
            "enabled" => cfgbool(exp_cfg, "nonideal.ring_lj.enabled", false),
            "epsilon" => cfgfloat32(exp_cfg, "nonideal.ring_lj.epsilon", 0.0f0),
            "sigma" => cfgfloat32(exp_cfg, "nonideal.ring_lj.sigma", 1.0f0),
            "softening" => cfgfloat32(exp_cfg, "nonideal.ring_lj.softening", 0.0f0),
            "cutoff" => cfgfloat32(exp_cfg, "nonideal.ring_lj.cutoff", 0.0f0),
            "shift" => cfgbool(exp_cfg, "nonideal.ring_lj.shift", true),
        ),
        "ring_bond" => Dict(
            "enabled" => cfgbool(exp_cfg, "nonideal.ring_bond.enabled", false),
        ),
    )
    return Dict(
        "dim" => dim,
        "n_beads" => n_beads,
        "diffusion" => diffusion,
        "bond_length" => bond_length,
        "k_over_xi" => k_over_xi,
        "tau_r" => rouse_relaxation_time(n_beads, k_over_xi),
        "nonideal" => nonideal,
    )
end

function analytic_potential(x, diffusion::Real, k_over_xi::Real, sim_cfg)
    nonideal = BoltzFlow.polymer_nonideal_params(nonideal_param_vector(sim_cfg))
    return BoltzFlow.polymer_langevin_potential(x, diffusion, k_over_xi, nonideal)
end

function analytic_force!(force, x, diffusion::Real, k_over_xi::Real, sim_cfg)
    nonideal = BoltzFlow.polymer_nonideal_params(nonideal_param_vector(sim_cfg))
    BoltzFlow.polymer_langevin_force!(force, x, diffusion, k_over_xi, nonideal)
    return force
end

function load_hdf5_traj(paths, sim_cfg; allow_partial::Bool=false)
    chunks = Array{Float32,3}[]
    loaded_paths = String[]
    for path in paths
        try
            h5open(path, "r") do h5
                written = haskey(h5, "written") ? Int(read(h5["written"])[1]) : size(h5["traj"], 3)
                written > 0 || error("No frames have been written in $path")
                push!(chunks, Float32.(h5["traj"][:, :, 1:written]))
                push!(loaded_paths, path)
            end
        catch err
            if allow_partial
                @warn "Skipping unavailable trajectory output" path exception=(err, catch_backtrace())
            else
                rethrow()
            end
        end
    end
    isempty(chunks) && error("No trajectory outputs were available for analysis")
    return cat(chunks...; dims=3), loaded_paths
end

function difference_values(traj, variant_cfg, base_cfg)
    diffusion = variant_cfg["diffusion"]
    k_over_xi = variant_cfg["k_over_xi"]
    base_diffusion = base_cfg["diffusion"]
    base_k_over_xi = base_cfg["k_over_xi"]
    potentials = Float32[]
    force_norms = Float32[]
    variant_force = zeros(Float32, size(traj, 1), size(traj, 2))
    base_force = similar(variant_force)
    for i in axes(traj, 3)
        x = @view traj[:, :, i]
        u_variant = analytic_potential(x, diffusion, k_over_xi, variant_cfg)
        u_base = analytic_potential(x, base_diffusion, base_k_over_xi, base_cfg)
        push!(potentials, Float32(u_variant - u_base))
        analytic_force!(variant_force, x, diffusion, k_over_xi, variant_cfg)
        analytic_force!(base_force, x, base_diffusion, base_k_over_xi, base_cfg)
        push!(force_norms, sqrt(sum(abs2, variant_force .- base_force)))
    end
    return potentials, force_norms
end

function plot_panel_histograms(series; xlabel, out_path, bins::Int=60)
    n = length(series)
    ncols = 2
    nrows = cld(n, ncols)
    fig = Figure(size=(1000, 360 * nrows), backgroundcolor=:white)
    for (idx, item) in enumerate(series)
        row = fld(idx - 1, ncols) + 1
        col = mod(idx - 1, ncols) + 1
        ax = Axis(fig[row, col],
                  xlabel=xlabel,
                  ylabel="probability density",
                  title=item.label)
        style_axis!(ax)
        hist!(ax, item.values; bins=bins, normalization=:pdf,
              color=(OKABE_ITO_BLUE, 0.55), strokewidth=1,
              strokecolor=:white)
        vlines!(ax, [mean(item.values)], color=OKABE_ITO_VERMILLION,
                linestyle=:dash, linewidth=4,
                label=@sprintf("sample mean = %.3g", mean(item.values)))
        axislegend(ax, position=:rt)
    end
    save(out_path, fig)
    return out_path
end

function plot_output_dir(plot_cfg)
    out_root = project_path(String(cfgget(plot_cfg, "output.dir", "outputs")))
    experiment_name = String(cfgget(plot_cfg, "output.experiment_name", "rouse_differences"))
    return joinpath(out_root, experiment_name, "figures", config_hash(plot_cfg; exclude=("output",)))
end

function main()
    length(ARGS) == 1 || error("Usage: julia --project=. scripts/rouse_difference_analytics.jl <config.yaml>")
    plot_cfg = load_yaml_config(project_path(ARGS[1]))
    prefer_hdf5 = cfgbool(plot_cfg, "data.prefer_hdf5", true)
    allow_partial = cfgbool(plot_cfg, "data.allow_partial", false)
    bins = cfgint(plot_cfg, "histogram.bins", 60)

    base_exp_cfg = load_yaml_config(project_path(String(cfgget(plot_cfg, "reference.config",
                                                               "configs/experiments/rouse_base.yaml"))))
    base_sim_cfg = sim_config_from_experiment_config(base_exp_cfg; force_ideal=true)

    variants = cfgget(plot_cfg, "variants", nothing)
    variants === nothing && error("Plot config needs variants")
    variant_order = split(String(cfgget(plot_cfg, "variant_order", "")), ",")
    variant_order = [strip(name) for name in variant_order if !isempty(strip(name))]
    isempty(variant_order) && (variant_order = sort(collect(keys(variants))))

    potential_series = []
    force_series = []
    for name in variant_order
        haskey(variants, name) || error("Missing variants.$name in plot config")
        variant = variants[name]
        label = String(variant["label"])
        exp_cfg = load_yaml_config(project_path(String(variant["config"])))
        sim_cfg = sim_config_from_experiment_config(exp_cfg)
        paths, _ = experiment_output_paths_and_dir(exp_cfg, prefer_hdf5)
        traj, loaded_paths = load_hdf5_traj(paths, sim_cfg; allow_partial)
        potentials, force_norms = difference_values(traj, sim_cfg, base_sim_cfg)
        push!(potential_series, (; label, values=potentials))
        push!(force_series, (; label, values=force_norms))
        @printf("%-18s files=%d frames=%d mean_dU=%.4f std_dU=%.4f mean_dF=%.4f std_dF=%.4f\n",
                label, length(loaded_paths), size(traj, 3), mean(potentials), std(potentials),
                mean(force_norms), std(force_norms))
    end

    out_dir = plot_output_dir(plot_cfg)
    mkpath(out_dir)
    do_potential = cfgbool(plot_cfg, "analysis.potential_difference", true)
    do_force = cfgbool(plot_cfg, "analysis.force_difference", true)
    if do_potential
        path = output_child_path(plot_cfg, out_dir, "output.potential_path",
                                 "output.potential_file",
                                 "rouse_potential_difference_histograms.png")
        plot_panel_histograms(potential_series;
                              xlabel="U_variant(X) - U_base(X)",
                              out_path=path,
                              bins=bins)
        println("Saved potential difference histograms: $path")
    end
    if do_force
        path = output_child_path(plot_cfg, out_dir, "output.force_path",
                                 "output.force_file",
                                 "rouse_force_difference_histograms.png")
        plot_panel_histograms(force_series;
                              xlabel="||F_variant(X) - F_base(X)||",
                              out_path=path,
                              bins=bins)
        println("Saved force difference histograms:     $path")
    end
end

main()
