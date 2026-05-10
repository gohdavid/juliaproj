"""
Overlay potential and score-norm distributions for Rouse variants.

Usage:
    julia --project=. scripts/rouse_overlay_analytics.jl
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

function experiment_paths(exp_cfg)
    output_dir = cfgget(exp_cfg, "output.dir", nothing)
    output_dir === nothing && error("Expected output.dir in experiment config")

    legacy_output = cfgget(exp_cfg, "output.path", nothing)
    default_file = legacy_output === nothing ? "traj.jls" : basename(String(legacy_output))
    output_file = String(cfgget(exp_cfg, "output.file", default_file))
    output_hdf5_file = String(cfgget(exp_cfg, "output.hdf5_file", frame_hdf5_path(output_file)))
    data_dir = project_path(config_output_dir(exp_cfg; default_name="rouse_raw"))

    n_workers = cfgint(exp_cfg, "parallel.workers", 1)
    seed0 = cfgint(exp_cfg, "seed", 11)
    stride = cfgint(exp_cfg, "parallel.seed_stride", 1)
    expected = [joinpath(data_dir, seed_filename(output_hdf5_file, seed0 + (i - 1) * stride))
                for i in 1:n_workers]
    all(isfile, expected) && return expected, data_dir

    experiment_name = String(cfgget(exp_cfg, "output.experiment_name", "rouse_raw"))
    root = project_path(joinpath(String(output_dir), experiment_name))
    if isdir(root)
        for hash_dir in sort(readdir(root; join=true))
            isdir(hash_dir) || continue
            candidate = [joinpath(hash_dir, seed_filename(output_hdf5_file, seed0 + (i - 1) * stride))
                         for i in 1:n_workers]
            all(isfile, candidate) && return candidate, hash_dir
        end
    end
    return expected, data_dir
end

function nonideal_param_vector(sim_cfg)
    nonideal = get(sim_cfg, "nonideal", Dict{String,Any}())
    lj = get(nonideal, "lennard_jones", Dict{String,Any}())
    ev = get(nonideal, "excluded_volume", Dict{String,Any}())
    conf = get(nonideal, "confinement", Dict{String,Any}())
    hairpin_lj = get(nonideal, "hairpin_lj", Dict{String,Any}())
    ring_lj = get(nonideal, "ring_lj", Dict{String,Any}())
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
        get(hairpin_lj, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(hairpin_lj, "epsilon", 0.0f0)),
        Float32(get(hairpin_lj, "sigma", 1.0f0)),
        Float32(get(hairpin_lj, "softening", 0.0f0)),
        Float32(get(hairpin_lj, "cutoff", 0.0f0)),
        get(hairpin_lj, "shift", true) ? 1.0f0 : 0.0f0,
        Float32(get(hairpin_lj, "min_separation", 4)),
        0.0f0,
        1.0f0,
        0.0f0,
        get(ring_lj, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(ring_lj, "epsilon", 0.0f0)),
        Float32(get(ring_lj, "sigma", 1.0f0)),
        Float32(get(ring_lj, "softening", 0.0f0)),
        Float32(get(ring_lj, "cutoff", 0.0f0)),
        get(ring_lj, "shift", true) ? 1.0f0 : 0.0f0,
    ]
end

function sim_config_from_experiment_config(exp_cfg)
    dim = cfgint(exp_cfg, "rouse.dim", 2)
    n_beads = cfgint(exp_cfg, "rouse.n_beads", 32)
    diffusion = cfgfloat32(exp_cfg, "rouse.diffusion", 1.0f0)
    bond_length = cfgfloat32(exp_cfg, "rouse.bond_length", 1.0f0)
    k_over_xi = cfgfloat32(exp_cfg, "rouse.k_over_xi",
                           3.0f0 * diffusion / bond_length^2)
    return Dict(
        "dim" => dim,
        "n_beads" => n_beads,
        "diffusion" => diffusion,
        "bond_length" => bond_length,
        "k_over_xi" => k_over_xi,
        "nonideal" => Dict(
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
        ),
    )
end

function analytic_potential(x, diffusion::Real, k_over_xi::Real, sim_cfg)
    nonideal = BoltzFlow.polymer_nonideal_params(nonideal_param_vector(sim_cfg))
    return BoltzFlow.polymer_langevin_potential(x, diffusion, k_over_xi, nonideal)
end

function analytic_score_norm2(x, diffusion::Real, k_over_xi::Real, sim_cfg)
    score = zeros(Float32, size(x))
    nonideal = BoltzFlow.polymer_nonideal_params(nonideal_param_vector(sim_cfg))
    BoltzFlow.polymer_langevin_score!(score, x, diffusion, k_over_xi, nonideal)
    return sum(abs2, score)
end

function quantile_sorted(sorted_values, q::Real)
    n = length(sorted_values)
    n == 0 && error("Cannot compute quantile of an empty vector")
    pos = 1.0 + Float64(q) * (n - 1)
    lo = floor(Int, pos)
    hi = ceil(Int, pos)
    lo == hi && return sorted_values[lo]
    weight = pos - lo
    return (1.0 - weight) * sorted_values[lo] + weight * sorted_values[hi]
end

function robust_range(values; lo_q=0.005, hi_q=0.995)
    sorted_values = sort(Float64.(values))
    lo = quantile_sorted(sorted_values, lo_q)
    hi = quantile_sorted(sorted_values, hi_q)
    lo == hi && return lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return max(0.0, lo - pad), hi + pad
end

function silverman_bandwidth(values)
    vals = Float64.(values)
    n = length(vals)
    n > 1 || return 1.0
    s = std(vals)
    sorted_vals = sort(vals)
    iqr = quantile_sorted(sorted_vals, 0.75) - quantile_sorted(sorted_vals, 0.25)
    scale = min(s, iqr / 1.349)
    if !isfinite(scale) || scale <= 0
        scale = s > 0 ? s : max(abs(mean(vals)), 1.0)
    end
    return max(0.9 * scale * n^(-1 / 5), eps(Float64))
end

function kde_density(values, xs; bandwidth=nothing)
    vals = Float64.(values)
    h = bandwidth === nothing ? silverman_bandwidth(vals) : Float64(bandwidth)
    norm = 1.0 / (length(vals) * h * sqrt(2.0 * pi))
    density = zeros(Float64, length(xs))
    inv_h = 1.0 / h
    for v in vals
        @. density += exp(-0.5 * ((xs - v) * inv_h)^2)
    end
    return density .* norm
end

function load_values(label, cfg_path)
    exp_cfg = load_yaml_config(project_path(cfg_path))
    sim_cfg = sim_config_from_experiment_config(exp_cfg)
    paths, _ = experiment_paths(exp_cfg)
    diffusion = sim_cfg["diffusion"]
    k_over_xi = sim_cfg["k_over_xi"]
    potentials = Float32[]
    score_norm2 = Float32[]
    force_norm = Float32[]
    for path in paths
        isfile(path) || error("Missing trajectory file: $path")
        h5open(path, "r") do h5
            written = haskey(h5, "written") ? Int(read(h5["written"])[1]) : size(h5["traj"], 3)
            for i in 1:written
                x = h5["traj"][:, :, i]
                push!(potentials, analytic_potential(x, diffusion, k_over_xi, sim_cfg))
                norm2 = analytic_score_norm2(x, diffusion, k_over_xi, sim_cfg)
                push!(score_norm2, norm2)
                push!(force_norm, Float32(diffusion) * sqrt(norm2))
            end
        end
    end
    @printf("%-18s N=%d mean_U=%.4f std_U=%.4f mean_force=%.4f std_force=%.4f mean_score2=%.4f std_score2=%.4f\n",
            label, length(potentials), mean(potentials), std(potentials),
            mean(force_norm), std(force_norm), mean(score_norm2), std(score_norm2))
    return (; label, potentials, score_norm2, force_norm)
end

function main()
    variants = [
        ("base", "configs/experiments/rouse_base.yaml"),
        ("LJ", "configs/experiments/rouse_lennard_jones.yaml"),
        ("excluded volume", "configs/experiments/rouse_excluded_volume.yaml"),
        ("confinement", "configs/experiments/rouse_confinement.yaml"),
        ("additive", "configs/experiments/rouse_nonideal_additive.yaml"),
        ("hairpin LJ", "configs/experiments/rouse_hairpin_lj.yaml"),
        ("ring LJ", "configs/experiments/rouse_ring_lj.yaml"),
    ]
    data = [load_values(label, cfg) for (label, cfg) in variants]

    potential_all = reduce(vcat, (Float64.(d.potentials) for d in data))
    score_all = reduce(vcat, (Float64.(d.score_norm2) for d in data))
    force_all = reduce(vcat, (Float64.(d.force_norm) for d in data))
    potential_lo, potential_hi = robust_range(potential_all)
    score_lo, score_hi = robust_range(score_all)
    force_lo, force_hi = robust_range(force_all)
    potential_xs = collect(range(potential_lo, potential_hi; length=800))
    score_xs = collect(range(score_lo, score_hi; length=800))
    force_xs = collect(range(force_lo, force_hi; length=800))

    colors = [colorant"#000000", colorant"#0072B2", colorant"#D55E00",
              colorant"#009E73", colorant"#CC79A7", colorant"#F0E442",
              colorant"#56B4E9"]

    fig = Figure(size=(1200, 560), backgroundcolor=:white)
    ax_u = Axis(fig[1, 1], xlabel="U(X)", ylabel="probability density",
                title="Potential KDE")
    ax_s = Axis(fig[1, 2], xlabel="||score(X)||²", ylabel="probability density",
                title="Score-norm² KDE")
    for ax in (ax_u, ax_s)
        ax.xgridvisible[] = false
        ax.ygridvisible[] = false
        hidespines!(ax, :t, :r)
    end

    for (i, d) in enumerate(data)
        yu = kde_density(d.potentials, potential_xs)
        ys = kde_density(d.score_norm2, score_xs)
        lines!(ax_u, potential_xs, yu, color=colors[i], linewidth=3, label=d.label)
        lines!(ax_s, score_xs, ys, color=colors[i], linewidth=3, label=d.label)
        vlines!(ax_u, [mean(d.potentials)], color=colors[i], linestyle=:dash, linewidth=1.6)
        vlines!(ax_s, [mean(d.score_norm2)], color=colors[i], linestyle=:dash, linewidth=1.6)
    end
    axislegend(ax_u, position=:rt)

    out_dir = project_path("outputs/rouse_overlays")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "rouse_potential_score_kde_overlay.png")
    save(out_path, fig)
    println("Saved overlay: $out_path")

    force_fig = Figure(size=(900, 620), backgroundcolor=:white)
    ax_f = Axis(force_fig[1, 1], xlabel="||force(X)||", ylabel="probability density",
                title="Force norm")
    ax_f.xgridvisible[] = false
    ax_f.ygridvisible[] = false
    hidespines!(ax_f, :t, :r)
    for (i, d) in enumerate(data)
        yf = kde_density(d.force_norm, force_xs)
        lines!(ax_f, force_xs, yf, color=colors[i], linewidth=3, label=d.label)
        vlines!(ax_f, [mean(d.force_norm)], color=colors[i], linestyle=:dash, linewidth=1.6)
    end
    axislegend(ax_f, position=:rt)
    force_path = joinpath(out_dir, "rouse_force_norm_kde_overlay.png")
    save(force_path, force_fig)
    println("Saved force overlay: $force_path")
end

main()
