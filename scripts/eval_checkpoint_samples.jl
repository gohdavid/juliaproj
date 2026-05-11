#!/usr/bin/env julia

"""
Evaluate and visualize CNF checkpoint samples.

The learned EGNN-CNF score is computed with `cnf_logp_gradient`, i.e.
`grad_x log p_model(x)`. Ground-truth polymer force is computed analytically
from the checkpoint training config. The script reports both score MSE and
force MSE, where `force_model = diffusion * score_model`.

Usage:
  julia --project=. scripts/eval_checkpoint_samples.jl --samples path/to/checkpoint_epoch_000035_samples.jls
  julia --project=. scripts/eval_checkpoint_samples.jl --samples path/to/samples.jls --max-frames 200 --batch-size 16
"""

using CairoMakie
using Printf
using Random
using Serialization
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_PLOT_CONFIG = joinpath(PROJECT_ROOT, "configs", "plots", "rouse_hairpin_lj_video.yaml")

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/eval_checkpoint_samples.jl --samples SAMPLES.jls [options]

    Options:
      --samples PATH       Samples .jls from sample_checkpoint_conformations.jl. Required.
      --checkpoint PATH    Override checkpoint path. Default: samples payload checkpoint_path.
      --plot-config PATH   Polymer video plot config. Default: configs/plots/rouse_hairpin_lj_video.yaml.
      --output-dir DIR     Output directory. Default: <samples-dir>/eval.
      --batch-size N       CNF score evaluation batch size. Default: 16.
      --max-samples N      Number of samples for force/score MSE. Default: all.
      --max-frames N       Number of conformations to render as PNG frames. Default: 200.
      --help              Show this message.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}(
        "plot_config" => DEFAULT_PLOT_CONFIG,
        "batch_size" => 16,
        "max_frames" => 200,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg in ("--samples", "--checkpoint", "--plot-config", "--output-dir",
                       "--batch-size", "--max-samples", "--max-frames")
            i < length(args) || error("Missing value for $arg")
            value = args[i + 1]
            key = replace(arg[3:end], "-" => "_")
            if key in ("batch_size", "max_samples", "max_frames")
                opts[key] = parse(Int, value)
            else
                opts[key] = value
            end
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

function style_axis!(ax)
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
    hidespines!(ax, :t, :r)
    return ax
end

chain_points(x) = Point2f.(x[1, :], x[2, :])

function draw_chain!(ax, pts, n_beads::Int; draw_ring_bond::Bool=false)
    lines!(ax, pts; color=:gray20, linewidth=4)
    if draw_ring_bond && n_beads > 2
        lines!(ax, Point2f[pts[end], pts[1]]; color=:gray55, linewidth=4)
    end
    scatter!(ax, pts; color=1:n_beads, colormap=:viridis,
             markersize=16, strokewidth=0)
end

axis_limits(traj) = (-10.0f0, 10.0f0), (-10.0f0, 10.0f0)

function output_child_path(cfg, out_dir::AbstractString, file_key::AbstractString,
                           default_file::AbstractString)
    file = BoltzFlow.cfgget(cfg, file_key, nothing)
    file === nothing && return joinpath(out_dir, default_file)
    return joinpath(out_dir, basename(String(file)))
end

function save_chain_frames(traj, frame_dir; draw_ring_bond::Bool=false)
    mkpath(frame_dir)
    xlims, ylims = axis_limits(traj)
    n_frames = size(traj, 3)
    n_beads = size(traj, 2)

    for frame in 1:n_frames
        fig = Figure(size=(700, 700), backgroundcolor=:white)
        ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
        style_axis!(ax)
        xlims!(ax, xlims...)
        ylims!(ax, ylims...)
        draw_chain!(ax, chain_points(traj[:, :, frame]), n_beads; draw_ring_bond)
        text!(ax, xlims[1] + 0.04 * (xlims[2] - xlims[1]),
              ylims[2] - 0.06 * (ylims[2] - ylims[1]),
              text=@sprintf("sample %d", frame), fontsize=24)
        save(joinpath(frame_dir, @sprintf("frame_%05d.png", frame)), fig)
    end
    return frame_dir
end

function polymer_reference(samples, cfg)
    p = Float32.(BoltzFlow.cfgget(cfg, "data.physics_params", Float32[]))
    diffusion = length(p) >= 1 ? p[1] : 1.0f0
    bond_length = length(p) >= 2 ? p[2] : 1.0f0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0f0 * diffusion / bond_length^2
    nonideal = BoltzFlow.polymer_nonideal_params(p)

    potential = Vector{Float32}(undef, size(samples, 3))
    force = similar(samples)
    score = similar(samples)
    for i in axes(samples, 3)
        x = @view(samples[:, :, i])
        potential[i] = Float32(
            BoltzFlow.polymer_langevin_potential(x, diffusion, k_over_xi, nonideal)
        )
        BoltzFlow.polymer_langevin_force!(@view(force[:, :, i]), x,
                                          diffusion, k_over_xi, nonideal)
        score[:, :, i] .= @view(force[:, :, i]) ./ Float32(diffusion)
    end
    return (; potential, force, score,
            diffusion=Float32(diffusion), k_over_xi=Float32(k_over_xi))
end

function cnf_potential_and_scores(checkpoint, checkpoint_path::AbstractString, samples;
                                  batch_size::Int=16, rng::AbstractRNG=Xoshiro(0))
    cfg = checkpoint["config"]
    device = BoltzFlow._device_from_config(cfg)
    ctx, _ = BoltzFlow._cnf_context_from_config(cfg, device)
    params = checkpoint["params"] |> device
    normalizer = BoltzFlow.cfgget(cfg, "data.normalizer", BoltzFlow.identity_data_normalizer())
    cpu = BoltzFlow.DiffEqFlux.Lux.cpu_device()

    potential = Vector{Float32}(undef, size(samples, 3))
    scores = similar(samples)
    n_samples = size(samples, 3)
    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        @info "evaluating CNF potential/score" checkpoint_path range="$(batch_start):$(batch_stop)" total=n_samples
        batch = samples[:, :, batch_idx]
        logp = cpu(BoltzFlow.normalized_cnf_logp(
            ctx, params, batch, normalizer; rng))
        scores[:, :, batch_idx] .= cpu(
            BoltzFlow.normalized_cnf_logp_gradient(
                ctx, params, batch, normalizer; rng)
        )
        potential[batch_idx] .= Float32.(-vec(logp))
    end
    return potential, scores
end

function cosine_stats(pred, target)
    cosine_sum = 0.0
    n_valid = 0
    for i in axes(pred, 3)
        p = @view(pred[:, :, i])
        t = @view(target[:, :, i])
        denom = sqrt(sum(abs2, p)) * sqrt(sum(abs2, t))
        if denom > 0
            cosine_sum += sum(p .* t) / denom
            n_valid += 1
        end
    end
    return n_valid == 0 ? NaN : cosine_sum / n_valid
end

function centered_mse(pred, target)
    pred_centered = pred .- mean(pred)
    target_centered = target .- mean(target)
    return mean(abs2, pred_centered .- target_centered)
end

function centered_mae(pred, target)
    pred_centered = pred .- mean(pred)
    target_centered = target .- mean(target)
    return mean(abs, pred_centered .- target_centered)
end

function pearson_corr(pred, target)
    pred_centered = pred .- mean(pred)
    target_centered = target .- mean(target)
    denom = sqrt(sum(abs2, pred_centered) * sum(abs2, target_centered))
    denom > 0 || return NaN
    return sum(pred_centered .* target_centered) / denom
end

function evaluate_samples(opts)
    samples_path = project_path(String(opts["samples"]))
    isfile(samples_path) || error("samples file does not exist: $samples_path")
    payload = deserialize(samples_path)
    samples_all = Float32.(payload["samples"])

    n_total = size(samples_all, 3)
    max_samples = Int(get(opts, "max_samples", n_total))
    max_samples = min(max_samples, n_total)
    samples = samples_all[:, :, 1:max_samples]

    checkpoint_path = haskey(opts, "checkpoint") ?
                      project_path(String(opts["checkpoint"])) :
                      String(payload["checkpoint_path"])
    isfile(checkpoint_path) || error("checkpoint file does not exist: $checkpoint_path")
    checkpoint = deserialize(checkpoint_path)
    cfg = checkpoint["config"]

    output_dir = project_path(String(get(opts, "output_dir", joinpath(dirname(samples_path), "eval"))))
    mkpath(output_dir)

    reference = polymer_reference(samples, cfg)
    model_potential, model_score = cnf_potential_and_scores(
        checkpoint, checkpoint_path, samples; batch_size=Int(opts["batch_size"]))
    model_force = model_score .* reference.diffusion

    score_mse = mean(abs2, model_score .- reference.score)
    force_mse = mean(abs2, model_force .- reference.force)
    score_cosine = cosine_stats(model_score, reference.score)
    force_cosine = cosine_stats(model_force, reference.force)
    potential_centered_mse = centered_mse(model_potential, reference.potential)
    potential_centered_mae = centered_mae(model_potential, reference.potential)
    potential_correlation = pearson_corr(model_potential, reference.potential)

    metrics = Dict{String,Any}(
        "samples_path" => samples_path,
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => get(checkpoint, "epoch", nothing),
        "n_samples" => max_samples,
        "diffusion" => reference.diffusion,
        "k_over_xi" => reference.k_over_xi,
        "score_definition" => "model_score = grad_x log p_model(x); reference_score = analytic_force / diffusion",
        "force_definition" => "model_force = diffusion * model_score; reference_force = analytic Langevin drift force",
        "potential_definition" => "model_potential = -log p_model(x); reference_potential = analytic dimensionless potential with diffusion scaling; centered metrics remove additive constant",
        "score_mse" => Float32(score_mse),
        "force_mse" => Float32(force_mse),
        "score_cosine" => Float32(score_cosine),
        "force_cosine" => Float32(force_cosine),
        "potential_centered_mse" => Float32(potential_centered_mse),
        "potential_centered_mae" => Float32(potential_centered_mae),
        "potential_correlation" => Float32(potential_correlation),
        "model_potential_mean" => Float32(mean(model_potential)),
        "reference_potential_mean" => Float32(mean(reference.potential)),
    )

    serialize(joinpath(output_dir, "sample_eval_metrics.jls"), metrics)
    open(joinpath(output_dir, "sample_eval_metrics.txt"), "w") do io
        for key in sort!(collect(keys(metrics)); by=string)
            println(io, key, " = ", metrics[key])
        end
    end

    plot_cfg = BoltzFlow.load_yaml_config(project_path(String(get(opts, "plot_config", DEFAULT_PLOT_CONFIG))))
    frame_dir = output_child_path(plot_cfg, output_dir, "output.frame_dirname", "frames")
    draw_ring_bond = length(BoltzFlow.cfgget(cfg, "data.physics_params", Float32[])) >= 37 &&
                     Float32(BoltzFlow.cfgget(cfg, "data.physics_params", Float32[])[37]) > 0.5f0

    n_frames = min(Int(opts["max_frames"]), size(samples_all, 3))
    plot_traj = samples_all[:, :, 1:n_frames]
    save_chain_frames(plot_traj, frame_dir; draw_ring_bond)

    @printf("score MSE: %.6g\n", score_mse)
    @printf("force MSE: %.6g\n", force_mse)
    @printf("score cosine: %.6f\n", score_cosine)
    @printf("force cosine: %.6f\n", force_cosine)
    @printf("potential centered MSE: %.6g\n", potential_centered_mse)
    @printf("potential centered MAE: %.6g\n", potential_centered_mae)
    @printf("potential correlation: %.6f\n", potential_correlation)
    println("wrote metrics: ", joinpath(output_dir, "sample_eval_metrics.txt"))
    println("wrote frames:  ", frame_dir)
    return metrics
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate_samples(parse_args(ARGS))
end
