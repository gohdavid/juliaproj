using Serialization

export InferenceResult, run_inference

struct InferenceResult
    config::Dict{String,Any}
    checkpoint_path::String
    checkpoint_epoch
    samples
    potential::Vector{Float32}
    score
    output_dir::String
    paths::Dict{String,String}
end

function _checkpoint_path_from_config(cfg::Dict{String,Any})
    explicit_path = cfgget(cfg, "checkpoint.path", nothing)
    if explicit_path !== nothing
        path = String(explicit_path)
        isfile(path) || error("checkpoint.path does not exist or is not a file: $path")
        return path
    end

    checkpoint_dir = String(cfgget(cfg, "checkpoint.dir", ""))
    !isempty(checkpoint_dir) || error("Set checkpoint.path or checkpoint.dir in the inference config")
    isdir(checkpoint_dir) || error("checkpoint.dir does not exist or is not a directory: $checkpoint_dir")

    latest_path = joinpath(checkpoint_dir, "latest.jls")
    isfile(latest_path) && return latest_path

    candidates = sort([
        joinpath(checkpoint_dir, name)
        for name in readdir(checkpoint_dir)
        if startswith(name, "checkpoint_epoch_") && endswith(name, ".jls")
    ])
    isempty(candidates) && error("No checkpoint_epoch_*.jls files found in $checkpoint_dir")
    return last(candidates)
end

function _inference_output_dir(cfg::Dict{String,Any}; config_path=nothing)
    default_name = "nbody_inference_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    path = config_output_dir(cfg; default_root="runs/inference", default_name)
    mkpath(path)
    _copy_config_snapshot(config_path, path)
    return path
end

function _checkpoint_training_config(checkpoint, inference_cfg::Dict{String,Any})
    ckpt_cfg = deepcopy(checkpoint["config"])

    if haskey(inference_cfg, "runtime")
        ckpt_cfg["runtime"] = inference_cfg["runtime"]
    end
    if haskey(inference_cfg, "ode")
        ckpt_cfg["ode"] = merge(get(ckpt_cfg, "ode", Dict{String,Any}()),
                                inference_cfg["ode"])
    end
    return ckpt_cfg
end

function _cnf_context_from_config(cfg::Dict{String,Any}, device)
    data_cfg = _data_config(cfg)
    field = build_vector_field(
        cfgsymbol(cfg, "model.kind", "mlp");
        dim=data_cfg.dim,
        n_atoms=data_cfg.n_atoms,
        hidden_dims=Int.(cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
        node_embedding_dim=cfgint(cfg, "model.node_embedding_dim", min(16, data_cfg.n_atoms)),
        n_layers=cfgint(cfg, "model.egnn_layers", 0),
    )
    return _make_context(cfg, field, device), data_cfg
end

function _evaluate_cnf_potential_and_score(ctx::NBodyCNFContext, params, samples;
                                           batch_size::Int=64,
                                           normalizer=identity_data_normalizer(),
                                           rng::AbstractRNG=Random.default_rng())
    n_samples = size(samples, 3)
    potential = Vector{Float32}(undef, n_samples)
    score = similar(samples)
    cpu = DiffEqFlux.Lux.cpu_device()

    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        batch = samples[:, :, batch_idx]
        logp = cpu(normalized_cnf_logp(ctx, params, batch, normalizer; rng))
        grad = cpu(normalized_cnf_logp_gradient(ctx, params, batch, normalizer; rng))
        potential[batch_idx] .= Float32.(-vec(logp))
        score[:, :, batch_idx] .= grad
    end

    return potential, score
end

function _maybe_polymer_reference_payload(samples, data_cfg::NBodyDataConfig)
    data_cfg.kind in (:polymer_langevin, :rouse_hdf5) || return nothing

    p = data_cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0f0
    bond_length = length(p) >= 2 ? p[2] : 1.0f0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0f0 * diffusion / bond_length^2
    nonideal = polymer_nonideal_params(p)

    n_samples = size(samples, 3)
    potentials = Vector{Float32}(undef, n_samples)
    force = similar(samples)
    for i in 1:n_samples
        frame = @view(samples[:, :, i])
        potentials[i] = Float32(polymer_langevin_potential(frame, diffusion, k_over_xi, nonideal))
        polymer_langevin_force!(@view(force[:, :, i]), frame, diffusion, k_over_xi, nonideal)
    end

    return Dict{String,Any}(
        "potential" => potentials,
        "force" => force,
        "score" => force ./ Float32(diffusion),
        "diffusion" => Float32(diffusion),
        "k_over_xi" => Float32(k_over_xi),
    )
end

function _serialize_inference_outputs(cfg, checkpoint_path, checkpoint, samples,
                                      potential, score, output_dir, data_cfg)
    conformations_file = String(cfgget(cfg, "output.conformations_file", "conformations.jls"))
    potential_file = String(cfgget(cfg, "output.potential_file", "potential.jls"))
    score_file = String(cfgget(cfg, "output.score_file", "score.jls"))
    result_file = String(cfgget(cfg, "output.result_file", "inference_result.jls"))

    epoch = get(checkpoint, "epoch", nothing)
    common = Dict{String,Any}(
        "config" => cfg,
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => epoch,
        "created_at" => Dates.now(),
    )

    paths = Dict{String,String}(
        "conformations" => joinpath(output_dir, conformations_file),
        "potential" => joinpath(output_dir, potential_file),
        "score" => joinpath(output_dir, score_file),
        "result" => joinpath(output_dir, result_file),
    )

    serialize(paths["conformations"], merge(common, Dict{String,Any}(
        "samples" => samples,
    )))
    serialize(paths["potential"], merge(common, Dict{String,Any}(
        "potential" => potential,
        "definition" => "-normalized_cnf_logp(samples); includes affine data-normalization log-Jacobian when present",
    )))
    serialize(paths["score"], merge(common, Dict{String,Any}(
        "score" => score,
        "force" => score,
        "definition" => "grad_x normalized_cnf_logp(samples); includes affine data-normalization chain rule when present",
    )))

    reference = _maybe_polymer_reference_payload(samples, data_cfg)
    if reference !== nothing
        paths["polymer_reference"] = joinpath(
            output_dir,
            String(cfgget(cfg, "output.polymer_reference_file", "polymer_reference.jls")),
        )
        serialize(paths["polymer_reference"], merge(common, reference))
    end

    serialize(paths["result"], merge(common, Dict{String,Any}(
        "samples" => samples,
        "potential" => potential,
        "score" => score,
        "force" => score,
        "paths" => paths,
    )))
    return paths
end

function run_inference(config_path::AbstractString)
    cfg = load_yaml_config(config_path)
    return run_inference(cfg; config_path)
end

function run_inference(cfg::Dict{String,Any}; config_path=nothing)
    seed = cfgint(cfg, "inference.seed", cfgint(cfg, "experiment.seed", 0))
    rng = Xoshiro(seed)
    checkpoint_path = _checkpoint_path_from_config(cfg)
    output_dir = _inference_output_dir(cfg; config_path)

    return _with_run_logger(output_dir) do log_path
        @info "starting checkpoint inference" config_path checkpoint_path output_dir log_path
        checkpoint = deserialize(checkpoint_path)
        train_cfg = _checkpoint_training_config(checkpoint, cfg)
        family = cfgsymbol(train_cfg, "experiment.family", "nbody_cnf")
        family == :nbody_cnf || error("Checkpoint inference currently supports nbody_cnf checkpoints, got $family")

        device = _device_from_config(train_cfg)
        ctx, data_cfg = _cnf_context_from_config(train_cfg, device)
        params = checkpoint["params"] |> device
        normalizer = cfgget(train_cfg, "data.normalizer", identity_data_normalizer())

        n_samples = cfgint(cfg, "sampling.n_samples",
                           cfgint(train_cfg, "sampling.n_samples", 1000))
        samples = DiffEqFlux.Lux.cpu_device()(
            generate_normalized_cnf_samples(ctx, params, n_samples, normalizer; rng))
        if cfgbool(cfg, "sampling.center", cfgbool(train_cfg, "data.center", false))
            samples = center_positions(samples)
        end

        eval_batch_size = cfgint(cfg, "evaluation.batch_size",
                                 cfgint(cfg, "sampling.batch_size", 64))
        potential, score = _evaluate_cnf_potential_and_score(
            ctx, params, samples; batch_size=eval_batch_size, normalizer, rng)

        paths = _serialize_inference_outputs(
            cfg, checkpoint_path, checkpoint, samples, potential, score, output_dir, data_cfg)

        @info "checkpoint inference complete" output_dir paths n_samples
        return InferenceResult(cfg, checkpoint_path, get(checkpoint, "epoch", nothing),
                               samples, potential, score, output_dir, paths)
    end
end
