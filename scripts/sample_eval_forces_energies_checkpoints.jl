#!/usr/bin/env julia

"""
Sample every checkpoint in a run and evaluate force/score/energy diagnostics.

Usage:
  julia --project=. scripts/sample_eval_forces_energies_checkpoints.jl --checkpoint-dir path/to/checkpoints
"""

using CairoMakie
using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/sample_eval_forces_energies_checkpoints.jl --checkpoint-dir DIR [options]

    Options:
      --checkpoint-dir DIR          Directory containing checkpoint_epoch_*.jls files.
      --checkpoint PATH             Evaluate one specific checkpoint_epoch_*.jls file.
      --output-dir DIR              Output directory. Default: <run>/forces_energies_checkpoint_eval.
      --n-samples N                 Number of samples per checkpoint. Default: config sampling.n_samples.
      --sample-batch-size N         Sampling batch size. Default: n-samples.
      --eval-batch-size N           Model score batch size. Default: config training.batch_size or 64.
      --max-data-samples N          Limit training datapoints for evaluation. Default: all training data.
      --distribution-max-samples N  Limit datapoints for MMD/energy metrics. Default: all.
      --scatter-points N            Score entries in each scatterplot. Default: 10000.
      --min-epoch N                 First checkpoint epoch to evaluate. Default: all.
      --last-checkpoint             Evaluate only the last checkpoint in --checkpoint-dir.
      --max-checkpoints N           Evaluate only the first N checkpoints. Default: all.
      --seed N                      Base sampling/eval seed. Default: checkpoint config experiment.seed.
      --force                       Resample checkpoints even if sample payloads already exist.
      --help                        Show this message.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}()
    value_args = Set([
        "--checkpoint-dir",
        "--checkpoint",
        "--output-dir",
        "--n-samples",
        "--sample-batch-size",
        "--eval-batch-size",
        "--max-data-samples",
        "--distribution-max-samples",
        "--scatter-points",
        "--min-epoch",
        "--max-checkpoints",
        "--seed",
    ])
    int_keys = Set([
        "n_samples",
        "sample_batch_size",
        "eval_batch_size",
        "max_data_samples",
        "distribution_max_samples",
        "scatter_points",
        "min_epoch",
        "max_checkpoints",
        "seed",
    ])
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--force"
            opts["force"] = true
            i += 1
        elseif arg == "--last-checkpoint"
            opts["last_checkpoint"] = true
            i += 1
        elseif occursin("=", arg) && startswith(arg, "--")
            raw_key, value = split(arg, "="; limit=2)
            raw_key in value_args || error("Unknown argument: $raw_key")
            key = replace(raw_key[3:end], "-" => "_")
            opts[key] = key in int_keys ? parse(Int, value) : value
            i += 1
        elseif arg in value_args
            i < length(args) || error("Missing value for $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            opts[key] = key in int_keys ? parse(Int, value) : value
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    haskey(opts, "checkpoint_dir") || haskey(opts, "checkpoint") ||
        error("Set --checkpoint-dir DIR or --checkpoint PATH")
    haskey(opts, "checkpoint_dir") && haskey(opts, "checkpoint") &&
        error("Set only one of --checkpoint-dir or --checkpoint")
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

function write_metrics_text(path::AbstractString, metrics)
    open(path, "w") do io
        for key in sort!(collect(keys(metrics)); by=string)
            println(io, key, " = ", metrics[key])
        end
    end
    return path
end

function csv_value(x)
    x === nothing && return "NA"
    x isa AbstractString && return replace(x, "," => ";")
    return string(x)
end

function write_summary_csv(path::AbstractString, records)
    columns = [
        "epoch",
        "checkpoint_loss",
        "train_score_mse",
        "train_score_cosine",
        "train_score_pearson",
        "sample_score_mse",
        "sample_score_cosine",
        "sample_score_pearson",
        "physics_score_norm_train_vs_sample_mmd",
        "physics_score_norm_train_vs_sample_energy",
        "model_score_norm_train_vs_sample_mmd",
        "model_score_norm_train_vs_sample_energy",
        "model_force_norm_sample_vs_physics_force_norm_train_mmd",
        "model_force_vector_sample_vs_physics_force_vector_train_energy",
        "physics_force_vector_train_vs_sample_energy",
        "potential_energy_train_vs_sample_mmd",
        "potential_energy_train_vs_sample_energy",
        "samples_path",
        "metrics_path",
    ]
    open(path, "w") do io
        println(io, join(columns, ","))
        for rec in records
            println(io, join((csv_value(get(rec, col, nothing)) for col in columns), ","))
        end
    end
    return path
end

function diffusion_context_from_config(cfg::Dict{String,Any}, device)
    data_cfg = BoltzFlow._data_config(cfg)
    model = BoltzFlow.build_diffusion_model(
        dim=data_cfg.dim,
        n_atoms=data_cfg.n_atoms,
        hidden_dims=Int.(BoltzFlow.cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
        n_layers=BoltzFlow.cfgint(cfg, "model.n_layers", 4),
        node_embedding_dim=BoltzFlow.cfgint(cfg, "model.node_embedding_dim", min(16, data_cfg.n_atoms)),
    )
    sample_steps = BoltzFlow.cfgint(cfg, "sampling.steps",
                                    BoltzFlow.cfgint(cfg, "diffusion.steps", 1000))
    return BoltzFlow.NBodyDiffusionContext(model, device, sample_steps), data_cfg
end

function model_context_from_config(cfg::Dict{String,Any})
    family = BoltzFlow.cfgsymbol(cfg, "experiment.family", "nbody_cnf")
    device = BoltzFlow._device_from_config(cfg)
    if family == :nbody_cnf
        ctx, data_cfg = BoltzFlow._cnf_context_from_config(cfg, device)
        return (; family, ctx, data_cfg, device)
    elseif family in (:nbody_diffusion, :nbody_equivariant_diffusion)
        ctx, data_cfg = diffusion_context_from_config(cfg, device)
        return (; family, ctx, data_cfg, device)
    end
    error("Unsupported checkpoint family for force/energy evaluation: $family")
end

function sample_from_checkpoint(model_ctx, params, normalizer, n_samples::Int,
                                batch_size::Int, rng::AbstractRNG)
    cpu = BoltzFlow.DiffEqFlux.Lux.cpu_device()
    dim = model_ctx.data_cfg.dim
    n_atoms = model_ctx.data_cfg.n_atoms
    samples = Array{Float32}(undef, dim, n_atoms, n_samples)
    cursor = 1
    while cursor <= n_samples
        stop = min(cursor + batch_size - 1, n_samples)
        local_n = stop - cursor + 1
        if model_ctx.family == :nbody_cnf
            samples[:, :, cursor:stop] .= cpu(
                BoltzFlow.generate_normalized_cnf_samples(
                    model_ctx.ctx, params, local_n, normalizer; rng)
            )
        else
            samples_norm = BoltzFlow.generate_diffusion_samples(
                model_ctx.ctx, params, local_n; rng)
            samples[:, :, cursor:stop] .= cpu(BoltzFlow.center_positions(
                BoltzFlow.invert_data_normalizer(samples_norm, normalizer)
            ))
        end
        cursor = stop + 1
    end
    return samples
end

function load_training_data(cfg::Dict{String,Any}; max_samples=nothing)
    data_rng = Xoshiro(BoltzFlow.cfgint(cfg, "experiment.seed", 0))
    data_cfg = BoltzFlow._data_config(cfg)
    train_data = Float32.(BoltzFlow.generate_nbody_dataset(data_rng, data_cfg))
    if BoltzFlow.cfgbool(cfg, "data.center", true)
        train_data = BoltzFlow.center_positions(train_data)
    end
    if max_samples !== nothing
        n = min(Int(max_samples), size(train_data, 3))
        train_data = train_data[:, :, 1:n]
    end
    return train_data, data_cfg
end

function polymer_reference(samples, data_cfg)
    data_cfg.kind in (:polymer_langevin, :rouse_hdf5) ||
        error("Polymer score/energy evaluation requires polymer_langevin or rouse_hdf5 data, got $(data_cfg.kind)")

    p = data_cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0f0
    bond_length = length(p) >= 2 ? p[2] : 1.0f0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0f0 * diffusion / bond_length^2
    nonideal = BoltzFlow.polymer_nonideal_params(p)

    potentials = Vector{Float32}(undef, size(samples, 3))
    score = similar(samples)
    for i in axes(samples, 3)
        x = @view(samples[:, :, i])
        potentials[i] = Float32(
            BoltzFlow.polymer_langevin_potential(x, diffusion, k_over_xi, nonideal)
        )
        BoltzFlow.polymer_langevin_score!(@view(score[:, :, i]), x,
                                          diffusion, k_over_xi, nonideal)
    end
    return (; potential=potentials,
            score,
            diffusion=Float32(diffusion),
            k_over_xi=Float32(k_over_xi))
end

function model_scores(model_ctx, params, samples, normalizer;
                      batch_size::Int=64, rng::AbstractRNG=Xoshiro(0))
    n_samples = size(samples, 3)
    scores = similar(samples)
    cpu = BoltzFlow.DiffEqFlux.Lux.cpu_device()
    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        @info "evaluating model score" family=model_ctx.family range="$(batch_start):$(batch_stop)" total=n_samples
        batch = samples[:, :, batch_idx]
        if model_ctx.family == :nbody_cnf
            scores[:, :, batch_idx] .= cpu(
                BoltzFlow.normalized_cnf_logp_gradient(
                    model_ctx.ctx, params, batch, normalizer; rng)
            )
        else
            x_norm = BoltzFlow.center_positions(
                BoltzFlow.apply_data_normalizer(batch, normalizer)
            )
            pred = cpu(BoltzFlow.diffusion_logp_gradient(model_ctx.ctx, params, x_norm))
            if Bool(get(normalizer, "enabled", false))
                pred = pred ./ normalizer["scale"]
            end
            scores[:, :, batch_idx] .= BoltzFlow.center_positions(pred)
        end
    end
    return scores
end

function score_norms(scores)
    values = Vector{Float32}(undef, size(scores, 3))
    for i in axes(scores, 3)
        values[i] = Float32(norm(@view(scores[:, :, i])))
    end
    return values
end

finite_values(x) = Float64.(filter(isfinite, vec(collect(x))))

function pearson_corr(pred, target)
    pred_flat = vec(collect(pred))
    target_flat = vec(collect(target))
    n_flat = min(length(pred_flat), length(target_flat))
    finite_idx = findall(i -> isfinite(pred_flat[i]) && isfinite(target_flat[i]),
                         1:n_flat)
    n = length(finite_idx)
    n > 1 || return Float32(NaN)
    p = Float64.(pred_flat[finite_idx])
    t = Float64.(target_flat[finite_idx])
    pc = p .- mean(p)
    tc = t .- mean(t)
    denom = sqrt(sum(abs2, pc) * sum(abs2, tc))
    denom > 0 || return Float32(NaN)
    return Float32(sum(pc .* tc) / denom)
end

function score_cosine(pred, target)
    cosine_sum = 0.0
    n_valid = 0
    for i in axes(pred, 3)
        p = @view(pred[:, :, i])
        t = @view(target[:, :, i])
        denom = norm(p) * norm(t)
        if denom > 0 && isfinite(denom)
            cosine_sum += sum(p .* t) / denom
            n_valid += 1
        end
    end
    return n_valid == 0 ? Float32(NaN) : Float32(cosine_sum / n_valid)
end

function score_agreement_metrics(prefix::AbstractString, pred, target)
    diff = pred .- target
    return Dict{String,Any}(
        "$(prefix)_score_mse" => Float32(mean(abs2, diff)),
        "$(prefix)_score_cosine" => score_cosine(pred, target),
        "$(prefix)_score_pearson" => pearson_corr(pred, target),
    )
end

function _limit_distribution(values, max_samples::Int, rng::AbstractRNG)
    vals = finite_values(values)
    if max_samples > 0 && length(vals) > max_samples
        return vals[randperm(rng, length(vals))[1:max_samples]]
    end
    return vals
end

function flattened_sample_vectors(x)
    n_samples = size(x, 3)
    n_features = div(length(x), n_samples)
    return reshape(Float64.(collect(x)), n_features, n_samples)
end

function _limit_vector_distribution(values, max_samples::Int, rng::AbstractRNG)
    n_samples = size(values, 2)
    if max_samples > 0 && n_samples > max_samples
        idx = randperm(rng, n_samples)[1:max_samples]
        return values[:, idx]
    end
    return values
end

function mean_pairwise_abs(x, y)
    (isempty(x) || isempty(y)) && return NaN
    total = 0.0
    for b in y, a in x
        total += abs(a - b)
    end
    return total / (length(x) * length(y))
end

function mean_pairwise_l2(x, y)
    (size(x, 2) == 0 || size(y, 2) == 0) && return NaN
    n_features = size(x, 1)
    total = 0.0
    for j in axes(y, 2), i in axes(x, 2)
        sqdist = 0.0
        for k in 1:n_features
            delta = x[k, i] - y[k, j]
            sqdist += delta * delta
        end
        total += sqrt(sqdist)
    end
    return total / (size(x, 2) * size(y, 2))
end

function median_pairwise_abs(values; max_points::Int=512)
    n = length(values)
    n_eval = min(n, max_points)
    n_eval >= 2 || return 1.0
    indices = round.(Int, range(1, n; length=n_eval))
    distances = Float64[]
    sizehint!(distances, n_eval * (n_eval - 1) ÷ 2)
    for b in 2:n_eval, a in 1:(b - 1)
        push!(distances, abs(values[indices[a]] - values[indices[b]]))
    end
    med = median!(distances)
    return max(med, eps(Float64))
end

function mean_rbf_kernel(x, y, sigma::Real)
    sigma2 = Float64(sigma)^2
    sigma2 > 0 || return NaN
    total = 0.0
    for b in y, a in x
        delta = a - b
        total += exp(-0.5 * delta * delta / sigma2)
    end
    return total / (length(x) * length(y))
end

function scalar_distribution_energy(x, y)
    (isempty(x) || isempty(y)) && return Float32(NaN)
    value = 2.0 * mean_pairwise_abs(x, y) -
            mean_pairwise_abs(x, x) -
            mean_pairwise_abs(y, y)
    return Float32(max(value, 0.0))
end

function vector_distribution_energy(x, y)
    (size(x, 2) == 0 || size(y, 2) == 0) && return Float32(NaN)
    value = 2.0 * mean_pairwise_l2(x, y) -
            mean_pairwise_l2(x, x) -
            mean_pairwise_l2(y, y)
    return Float32(max(value, 0.0))
end

function scalar_distribution_mmd(x, y; sigmas=nothing)
    (isempty(x) || isempty(y)) && return Float32(NaN)
    if sigmas === nothing
        base_sigma = median_pairwise_abs(vcat(x, y))
        sigmas = [0.5 * base_sigma, base_sigma, 2.0 * base_sigma, 4.0 * base_sigma]
    end
    total = 0.0
    n_sigmas = 0
    for sigma in sigmas
        isfinite(sigma) && sigma > 0 || continue
        value = mean_rbf_kernel(x, x, sigma) +
                mean_rbf_kernel(y, y, sigma) -
                2.0 * mean_rbf_kernel(x, y, sigma)
        total += value
        n_sigmas += 1
    end
    n_sigmas > 0 || return Float32(NaN)
    return Float32(max(total / n_sigmas, 0.0))
end

function distribution_pair_metrics(prefix::AbstractString, x, y)
    return Dict{String,Any}(
        "$(prefix)_mmd" => scalar_distribution_mmd(x, y),
        "$(prefix)_energy" => scalar_distribution_energy(x, y),
    )
end

function all_pair_distribution_metrics(prefix::AbstractString, series::Dict{String,Vector{Float64}})
    metrics = Dict{String,Any}()
    labels = sort!(collect(keys(series)))
    for b in 2:length(labels), a in 1:(b - 1)
        left = labels[a]
        right = labels[b]
        key = "$(prefix)_$(left)_vs_$(right)"
        merge!(metrics, distribution_pair_metrics(key, series[left], series[right]))
    end
    return metrics
end

function robust_range(values)
    vals = sort(finite_values(values))
    isempty(vals) && return (-1.0, 1.0)
    lo = vals[max(1, floor(Int, 0.01 * length(vals)))]
    hi = vals[min(length(vals), ceil(Int, 0.99 * length(vals)))]
    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        lo = minimum(vals)
        hi = maximum(vals)
    end
    if lo == hi
        pad = max(abs(lo), 1.0) * 0.1
        return lo - pad, hi + pad
    end
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad
end

function kde_density(values, xs)
    vals = finite_values(values)
    n = length(vals)
    ys = zeros(Float64, length(xs))
    n > 0 || return ys
    sigma = std(vals)
    span = maximum(vals) - minimum(vals)
    bandwidth = 1.06 * sigma * n^(-0.2)
    if !isfinite(bandwidth) || bandwidth <= 0
        bandwidth = max(span / 50, 1.0e-3)
    end
    inv_norm = 1.0 / (n * bandwidth * sqrt(2π))
    for v in vals
        @. ys += exp(-0.5 * ((xs - v) / bandwidth)^2)
    end
    return ys .* inv_norm
end

function style_axis!(ax)
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
    hidespines!(ax, :t, :r)
    return ax
end

function plot_density_overlay(path::AbstractString, series;
                              xlabel::AbstractString,
                              title::AbstractString)
    all_values = reduce(vcat, (finite_values(s.values) for s in series))
    lo, hi = robust_range(all_values)
    xs = collect(range(lo, hi; length=700))
    colors = [:black, :dodgerblue3, :orangered3, :seagreen4, :orange3, :purple4]

    fig = Figure(size=(900, 560), backgroundcolor=:white)
    ax = Axis(fig[1, 1], xlabel=xlabel, ylabel="density", title=title)
    style_axis!(ax)
    for (i, s) in enumerate(series)
        ys = kde_density(s.values, xs)
        lines!(ax, xs, ys; linewidth=3, color=colors[mod1(i, length(colors))],
               label=s.label)
        vals = finite_values(s.values)
        !isempty(vals) && vlines!(ax, [mean(vals)];
                                  color=colors[mod1(i, length(colors))],
                                  linestyle=:dash, linewidth=1.4)
    end
    axislegend(ax, position=:rt)
    save(path, fig)
    return path
end

function sampled_score_pairs(pred, target, n_points::Int, rng::AbstractRNG)
    p = vec(collect(pred))
    t = vec(collect(target))
    finite_idx = findall(i -> isfinite(p[i]) && isfinite(t[i]), eachindex(p))
    n = min(n_points, length(finite_idx))
    n == 0 && return Float32[], Float32[]
    chosen = length(finite_idx) > n ? finite_idx[randperm(rng, length(finite_idx))[1:n]] :
             finite_idx
    return Float32.(t[chosen]), Float32.(p[chosen])
end

function plot_score_scatter(path::AbstractString, pred, target;
                            n_points::Int, rng::AbstractRNG,
                            title::AbstractString)
    truth, model = sampled_score_pairs(pred, target, n_points, rng)
    fig = Figure(size=(720, 680), backgroundcolor=:white)
    ax = Axis(fig[1, 1], xlabel="ground truth score entry",
              ylabel="model-predicted score entry", title=title)
    style_axis!(ax)
    if !isempty(truth)
        scatter!(ax, truth, model; markersize=3, color=(:black, 0.22),
                 strokewidth=0)
        lo = min(minimum(truth), minimum(model))
        hi = max(maximum(truth), maximum(model))
        lines!(ax, [lo, hi], [lo, hi]; color=:red, linewidth=2)
    end
    save(path, fig)
    return (; path, truth, model)
end

function evaluate_checkpoint(checkpoint, checkpoint_path::AbstractString, model_ctx,
                             train_data, train_reference, opts, output_dir::AbstractString)
    cfg = checkpoint["config"]
    normalizer = BoltzFlow.cfgget(cfg, "data.normalizer", BoltzFlow.identity_data_normalizer())
    epoch = Int(get(checkpoint, "epoch", 0))
    stem = splitext(basename(checkpoint_path))[1]
    checkpoint_output_dir = joinpath(output_dir, stem)
    mkpath(checkpoint_output_dir)

    n_samples = Int(opts["n_samples"])
    sample_batch_size = Int(opts["sample_batch_size"])
    eval_batch_size = Int(opts["eval_batch_size"])
    scatter_points = Int(opts["scatter_points"])
    distribution_max_samples = Int(get(opts, "distribution_max_samples", 0))
    base_seed = Int(opts["seed"])
    force = Bool(get(opts, "force", false))

    samples_path = joinpath(checkpoint_output_dir, "samples.jls")
    local samples
    if isfile(samples_path) && !force
        @info "loading existing checkpoint samples" epoch samples_path
        samples = Float32.(deserialize(samples_path)["samples"])
        if size(samples, 3) > n_samples
            samples = samples[:, :, 1:n_samples]
        end
    else
        @info "sampling checkpoint" epoch checkpoint_path n_samples sample_batch_size
        params = checkpoint["params"] |> model_ctx.device
        samples = sample_from_checkpoint(model_ctx, params, normalizer,
                                         n_samples, sample_batch_size,
                                         Xoshiro(base_seed + epoch))
        serialize(samples_path, Dict{String,Any}(
            "samples" => samples,
            "checkpoint_path" => checkpoint_path,
            "checkpoint_epoch" => epoch,
            "checkpoint_loss" => get(checkpoint, "loss", nothing),
            "config" => checkpoint["config"],
            "sampling" => Dict{String,Any}(
                "n_samples" => n_samples,
                "batch_size" => sample_batch_size,
                "seed" => base_seed + epoch,
            ),
            "data_normalizer" => normalizer,
            "created_at" => Dates.now(),
        ))
    end

    params = checkpoint["params"] |> model_ctx.device
    sample_reference = polymer_reference(samples, model_ctx.data_cfg)
    train_model_score = model_scores(
        model_ctx, params, train_data, normalizer;
        batch_size=eval_batch_size, rng=Xoshiro(base_seed + 100000 + epoch))
    sample_model_score = model_scores(
        model_ctx, params, samples, normalizer;
        batch_size=eval_batch_size, rng=Xoshiro(base_seed + 200000 + epoch))

    train_physics_norm = score_norms(train_reference.score)
    train_model_norm = score_norms(train_model_score)
    sample_physics_norm = score_norms(sample_reference.score)
    sample_model_norm = score_norms(sample_model_score)
    diffusion = train_reference.diffusion
    train_physics_force = train_reference.score .* diffusion
    train_model_force = train_model_score .* diffusion
    sample_physics_force = sample_reference.score .* diffusion
    sample_model_force = sample_model_score .* diffusion
    train_physics_force_norm = score_norms(train_physics_force)
    sample_physics_force_norm = score_norms(sample_physics_force)
    sample_model_force_norm = score_norms(sample_model_force)

    dist_rng = Xoshiro(base_seed + 300000 + epoch)
    score_norm_series = Dict{String,Vector{Float64}}(
        "physics_score_norm_train" => _limit_distribution(train_physics_norm, distribution_max_samples, dist_rng),
        "model_score_norm_train" => _limit_distribution(train_model_norm, distribution_max_samples, dist_rng),
        "physics_score_norm_sample" => _limit_distribution(sample_physics_norm, distribution_max_samples, dist_rng),
        "model_score_norm_sample" => _limit_distribution(sample_model_norm, distribution_max_samples, dist_rng),
    )
    potential_series = Dict{String,Vector{Float64}}(
        "potential_energy_train" => _limit_distribution(train_reference.potential, distribution_max_samples, dist_rng),
        "potential_energy_sample" => _limit_distribution(sample_reference.potential, distribution_max_samples, dist_rng),
    )
    force_vector_series = Dict{String,Matrix{Float64}}(
        "physics_force_train" => _limit_vector_distribution(
            flattened_sample_vectors(train_physics_force), distribution_max_samples, dist_rng),
        "physics_force_sample" => _limit_vector_distribution(
            flattened_sample_vectors(sample_physics_force), distribution_max_samples, dist_rng),
        "model_force_sample" => _limit_vector_distribution(
            flattened_sample_vectors(sample_model_force), distribution_max_samples, dist_rng),
    )

    score_norm_density_path = joinpath(checkpoint_output_dir, "score_norm_density.png")
    potential_density_path = joinpath(checkpoint_output_dir, "potential_energy_density.png")
    plot_density_overlay(score_norm_density_path, [
        (; label="physics train", values=train_physics_norm),
        (; label="model train", values=train_model_norm),
        (; label="physics samples", values=sample_physics_norm),
        (; label="model samples", values=sample_model_norm),
    ]; xlabel="||score(X)||", title="Score-norm distributions")
    plot_density_overlay(potential_density_path, [
        (; label="training data", values=train_reference.potential),
        (; label="generated samples", values=sample_reference.potential),
    ]; xlabel="polymer potential energy", title="Physics potential energy distributions")

    train_scatter = plot_score_scatter(
        joinpath(checkpoint_output_dir, "train_score_scatter.png"),
        train_model_score, train_reference.score;
        n_points=scatter_points, rng=Xoshiro(base_seed + 400000 + epoch),
        title="Training data score agreement")
    sample_scatter = plot_score_scatter(
        joinpath(checkpoint_output_dir, "sample_score_scatter.png"),
        sample_model_score, sample_reference.score;
        n_points=scatter_points, rng=Xoshiro(base_seed + 500000 + epoch),
        title="Generated sample score agreement")

    metrics = Dict{String,Any}(
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => epoch,
        "checkpoint_loss" => get(checkpoint, "loss", nothing),
        "model_family" => String(model_ctx.family),
        "samples_path" => samples_path,
        "n_generated_samples" => size(samples, 3),
        "n_data_samples" => size(train_data, 3),
        "diffusion" => train_reference.diffusion,
        "k_over_xi" => train_reference.k_over_xi,
        "score_definition" => "model_score = grad_x log p_model(x); reference_score = polymer_langevin_score(x)",
        "score_norm_definition" => "per-datapoint Frobenius norm of the D x N score matrix",
        "potential_definition" => "polymer_langevin_potential(x)",
        "score_norm_density_path" => score_norm_density_path,
        "potential_energy_density_path" => potential_density_path,
        "train_score_scatter_path" => train_scatter.path,
        "sample_score_scatter_path" => sample_scatter.path,
        "train_score_norm_physics_mean" => Float32(mean(train_physics_norm)),
        "train_score_norm_model_mean" => Float32(mean(train_model_norm)),
        "sample_score_norm_physics_mean" => Float32(mean(sample_physics_norm)),
        "sample_score_norm_model_mean" => Float32(mean(sample_model_norm)),
        "force_definition" => "force = diffusion * score; force-vector energy uses flattened D x N force matrices per sample",
        "train_potential_energy_mean" => Float32(mean(train_reference.potential)),
        "sample_potential_energy_mean" => Float32(mean(sample_reference.potential)),
    )
    merge!(metrics, score_agreement_metrics("train", train_model_score, train_reference.score))
    merge!(metrics, score_agreement_metrics("sample", sample_model_score, sample_reference.score))
    merge!(metrics, all_pair_distribution_metrics("score_norm", score_norm_series))
    merge!(metrics, distribution_pair_metrics(
        "physics_score_norm_train_vs_sample",
        score_norm_series["physics_score_norm_train"],
        score_norm_series["physics_score_norm_sample"],
    ))
    merge!(metrics, distribution_pair_metrics(
        "model_score_norm_train_vs_sample",
        score_norm_series["model_score_norm_train"],
        score_norm_series["model_score_norm_sample"],
    ))
    merge!(metrics, distribution_pair_metrics(
        "model_force_norm_sample_vs_physics_force_norm_train",
        finite_values(sample_model_force_norm),
        finite_values(train_physics_force_norm),
    ))
    metrics["model_force_vector_sample_vs_physics_force_vector_train_energy"] =
        vector_distribution_energy(
            force_vector_series["model_force_sample"],
            force_vector_series["physics_force_train"],
        )
    metrics["physics_force_vector_train_vs_sample_energy"] =
        vector_distribution_energy(
            force_vector_series["physics_force_train"],
            force_vector_series["physics_force_sample"],
        )
    merge!(metrics, distribution_pair_metrics(
        "potential_energy_train_vs_sample",
        potential_series["potential_energy_train"],
        potential_series["potential_energy_sample"],
    ))

    distributions_path = joinpath(checkpoint_output_dir, "force_energy_distributions.jls")
    serialize(distributions_path, Dict{String,Any}(
        "score_norms" => Dict{String,Any}(
            "physics_train" => train_physics_norm,
            "model_train" => train_model_norm,
            "physics_sample" => sample_physics_norm,
            "model_sample" => sample_model_norm,
        ),
        "potential_energy" => Dict{String,Any}(
            "train" => train_reference.potential,
            "sample" => sample_reference.potential,
        ),
        "force_norms" => Dict{String,Any}(
            "physics_train" => train_physics_force_norm,
            "physics_sample" => sample_physics_force_norm,
            "model_sample" => sample_model_force_norm,
        ),
        "scatter" => Dict{String,Any}(
            "train_truth" => train_scatter.truth,
            "train_model" => train_scatter.model,
            "sample_truth" => sample_scatter.truth,
            "sample_model" => sample_scatter.model,
        ),
    ))
    metrics["distributions_path"] = distributions_path

    metrics_path = joinpath(checkpoint_output_dir, "force_energy_metrics.jls")
    metrics["metrics_path"] = metrics_path
    serialize(metrics_path, metrics)
    write_metrics_text(joinpath(checkpoint_output_dir, "force_energy_metrics.txt"), metrics)

    @printf("epoch %d: train score MSE %.6g, sample score MSE %.6g, potential energy MMD %.6g\n",
            epoch, metrics["train_score_mse"], metrics["sample_score_mse"],
            metrics["potential_energy_train_vs_sample_mmd"])
    return metrics
end

function run_checkpoint_force_energy_eval(opts)
    checkpoints = if haskey(opts, "checkpoint")
        checkpoint_path = project_path(String(opts["checkpoint"]))
        isfile(checkpoint_path) || error("checkpoint file does not exist: $checkpoint_path")
        [checkpoint_path]
    else
        checkpoint_dir = project_path(String(opts["checkpoint_dir"]))
        paths = checkpoint_paths(checkpoint_dir)
        min_epoch = get(opts, "min_epoch", nothing)
        if min_epoch !== nothing
            min_epoch = Int(min_epoch)
            paths = [
                path for path in paths
                if parse(Int, match(r"checkpoint_epoch_(\d+)\.jls$", basename(path)).captures[1]) >= min_epoch
            ]
            isempty(paths) && error("No checkpoints remain after applying --min-epoch $min_epoch")
        end
        if Bool(get(opts, "last_checkpoint", false))
            paths = [last(paths)]
        else
            max_checkpoints = get(opts, "max_checkpoints", nothing)
            if max_checkpoints !== nothing
                n_keep = min(Int(max_checkpoints), length(paths))
                n_keep > 0 || error("--max-checkpoints must be positive")
                paths = paths[1:n_keep]
            end
        end
        paths
    end
    checkpoint_dir = dirname(first(checkpoints))
    run_dir = dirname(checkpoint_dir)
    output_dir = project_path(String(get(opts, "output_dir",
                                         joinpath(run_dir, "forces_energies_checkpoint_eval"))))
    mkpath(output_dir)

    first_checkpoint = deserialize(first(checkpoints))
    cfg = first_checkpoint["config"]
    base_seed = Int(get(opts, "seed", BoltzFlow.cfgint(cfg, "experiment.seed", 0)))
    n_samples = Int(get(opts, "n_samples", BoltzFlow.cfgint(cfg, "sampling.n_samples", 1000)))
    sample_batch_size = Int(get(opts, "sample_batch_size", n_samples))
    eval_batch_size = Int(get(opts, "eval_batch_size",
                              BoltzFlow.cfgint(cfg, "training.batch_size", 64)))
    scatter_points = Int(get(opts, "scatter_points", 10000))
    n_samples > 0 || error("--n-samples must be positive")
    sample_batch_size > 0 || error("--sample-batch-size must be positive")
    eval_batch_size > 0 || error("--eval-batch-size must be positive")
    scatter_points > 0 || error("--scatter-points must be positive")

    opts = copy(opts)
    opts["seed"] = base_seed
    opts["n_samples"] = n_samples
    opts["sample_batch_size"] = sample_batch_size
    opts["eval_batch_size"] = eval_batch_size
    opts["scatter_points"] = scatter_points

    max_data_samples = get(opts, "max_data_samples", nothing)
    train_data, data_cfg = load_training_data(
        cfg; max_samples=max_data_samples === nothing ? nothing : Int(max_data_samples))
    model_ctx = model_context_from_config(cfg)
    train_reference = polymer_reference(train_data, data_cfg)

    serialize(joinpath(output_dir, "training_reference.jls"), Dict{String,Any}(
        "train_data_shape" => size(train_data),
        "potential" => train_reference.potential,
        "score_norm" => score_norms(train_reference.score),
        "diffusion" => train_reference.diffusion,
        "k_over_xi" => train_reference.k_over_xi,
    ))

    records = Vector{Dict{String,Any}}()
    for checkpoint_path in checkpoints
        checkpoint = deserialize(checkpoint_path)
        push!(records, evaluate_checkpoint(
            checkpoint, checkpoint_path, model_ctx, train_data, train_reference,
            opts, output_dir))
    end

    serialize(joinpath(output_dir, "force_energy_metrics_all.jls"), records)
    write_summary_csv(joinpath(output_dir, "force_energy_metrics_all.csv"), records)
    println("wrote summary: ", joinpath(output_dir, "force_energy_metrics_all.csv"))
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_checkpoint_force_energy_eval(parse_args(ARGS))
end
