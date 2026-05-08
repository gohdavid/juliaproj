using Serialization

export ExperimentResult, run_nbody_experiment, run_experiment

struct ExperimentResult
    config::Dict{String,Any}
    train_data
    params
    losses::Vector{Float32}
    samples
    output_dir::String
end

function _device_from_config(cfg)
    use_gpu = cfgbool(cfg, "runtime.use_gpu", false)
    if use_gpu && CUDA.functional()
        return CUDA.cu
    end
    return DiffEqFlux.Lux.cpu_device()
end

function _data_config(cfg)
    params = cfgget(cfg, "data.physics_params", Float32[])
    params = params === nothing ? Float32[] : Float32.(params)
    return NBodyDataConfig(
        kind=cfgsymbol(cfg, "data.kind", "static_asymmetric"),
        dim=cfgint(cfg, "data.dim", 2),
        n_atoms=cfgint(cfg, "data.n_atoms", 10),
        n_samples=cfgint(cfg, "data.n_samples", 4000),
        burn_in=cfgint(cfg, "data.burn_in", 0),
        noise_std=cfgfloat32(cfg, "data.noise_std", 0.1),
        total_steps=cfgint(cfg, "data.total_steps", 4000),
        dt=cfgfloat32(cfg, "data.dt", 0.05),
        min_spawn_dist=cfgfloat32(cfg, "data.min_spawn_dist", 1.7),
        physics_params=params,
    )
end

function _make_context(cfg, field, device)
    t0 = cfgfloat32(cfg, "ode.t0", 0.0)
    t1 = cfgfloat32(cfg, "ode.t1", 1.0)
    return NBodyCNFContext(
        field,
        device,
        (t0, t1),
        Tsit5(),
        cfgfloat32(cfg, "ode.abstol", 1f-5),
        cfgfloat32(cfg, "ode.reltol", 1f-5),
        cfgsymbol(cfg, "model.trace", "hutchinson"),
    )
end

function _output_dir(cfg)
    default_name = "nbody_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    path = config_output_dir(cfg; default_root="runs", default_name)
    mkpath(path)
    return path
end

function _sample_metrics(train_data, samples)
    samples_cpu = DiffEqFlux.Lux.cpu_device()(samples)
    return Dict{String,Float32}(
        "pairwise_distance_mae" => pairwise_distance_mae(train_data, samples_cpu),
    )
end

function _save_result(result::ExperimentResult)
    path = joinpath(result.output_dir, "result.jls")
    serialize(path, Dict(
        "config" => result.config,
        "params" => DiffEqFlux.Lux.cpu_device()(result.params),
        "losses" => result.losses,
        "samples" => DiffEqFlux.Lux.cpu_device()(result.samples),
        "metrics" => _sample_metrics(result.train_data, result.samples),
    ))
    return path
end

function _save_result(result::FlowMatchingResult)
    path = joinpath(result.output_dir, "result.jls")
    serialize(path, Dict(
        "config" => result.config,
        "params" => DiffEqFlux.Lux.cpu_device()(result.params),
        "losses" => result.losses,
        "samples" => DiffEqFlux.Lux.cpu_device()(result.samples),
        "metrics" => result.metrics,
    ))
    return path
end

function _save_result(result::DiffusionResult)
    path = joinpath(result.output_dir, "result.jls")
    serialize(path, Dict(
        "config" => result.config,
        "params" => DiffEqFlux.Lux.cpu_device()(result.params),
        "losses" => result.losses,
        "samples" => DiffEqFlux.Lux.cpu_device()(result.samples),
        "metrics" => result.metrics,
    ))
    return path
end

function _maybe_plot_result(result::ExperimentResult)
    cfgbool(result.config, "output.plots", false) || return nothing
    train_data = result.train_data
    samples = DiffEqFlux.Lux.cpu_device()(result.samples)
    path = joinpath(result.output_dir, "samples_vs_train.png")
    plt = scatter(train_data[1, :, :], train_data[2, :, :];
                  title="Generated samples vs training data",
                  xlabel="x", ylabel="y", markersize=2,
                  markercolor=:blue, legend=false, aspect_ratio=:equal)
    scatter!(plt, samples[1, :, :], samples[2, :, :];
             markersize=2, markercolor=:tomato, legend=false)
    savefig(plt, path)
    return path
end

function _maybe_plot_result(result::FlowMatchingResult)
    cfgbool(result.config, "output.plots", false) || return nothing
    train_data = result.train_data
    samples = DiffEqFlux.Lux.cpu_device()(result.samples)
    path = joinpath(result.output_dir, "samples_vs_train.png")
    plt = scatter(train_data[1, :, :], train_data[2, :, :];
                  title="Flow matching samples vs training data",
                  xlabel="x", ylabel="y", markersize=2,
                  markercolor=:blue, legend=false, aspect_ratio=:equal)
    scatter!(plt, samples[1, :, :], samples[2, :, :];
             markersize=2, markercolor=:tomato, legend=false)
    savefig(plt, path)
    return path
end

function _maybe_plot_result(result::DiffusionResult)
    cfgbool(result.config, "output.plots", false) || return nothing
    train_data = result.train_data
    samples = DiffEqFlux.Lux.cpu_device()(result.samples)
    path = joinpath(result.output_dir, "samples_vs_train.png")
    plt = scatter(train_data[1, :, :], train_data[2, :, :];
                  title="Diffusion samples vs training data",
                  xlabel="x", ylabel="y", markersize=2,
                  markercolor=:blue, legend=false, aspect_ratio=:equal)
    scatter!(plt, samples[1, :, :], samples[2, :, :];
             markersize=2, markercolor=:tomato, legend=false)
    savefig(plt, path)
    return path
end

function run_nbody_experiment(cfg::Dict{String,Any})
    seed = cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)
    device = _device_from_config(cfg)

    data_cfg = _data_config(cfg)
    train_data = generate_nbody_dataset(rng, data_cfg)

    field = build_vector_field(
        cfgsymbol(cfg, "model.kind", "mlp");
        dim=data_cfg.dim,
        n_atoms=data_cfg.n_atoms,
        hidden_dims=Int.(cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
    )
    ctx = _make_context(cfg, field, device)
    params = init_cnf_params(rng, field; device)

    params, _, losses = train_cnf_adam(
        ctx, params, train_data;
        epochs=cfgint(cfg, "training.epochs", 10),
        batch_size=cfgint(cfg, "training.batch_size", 64),
        learning_rate=cfgfloat32(cfg, "training.learning_rate", 5f-3),
        rng,
    )

    samples = generate_cnf_samples(
        ctx, params, cfgint(cfg, "sampling.n_samples", 1000); rng)

    result = ExperimentResult(cfg, train_data, params, losses, samples, _output_dir(cfg))
    result_path = _save_result(result)
    plot_path = _maybe_plot_result(result)
    metrics = _sample_metrics(train_data, samples)
    @info "experiment complete" output_dir=result.output_dir result_path plot_path metrics
    return result
end

function run_nbody_flow_matching_experiment(cfg::Dict{String,Any})
    seed = cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)
    device = _device_from_config(cfg)

    data_cfg = _data_config(cfg)
    train_data = generate_nbody_dataset(rng, data_cfg)
    if cfgbool(cfg, "data.center", true)
        train_data = center_positions(train_data)
    end

    field = build_fm_vector_field(
        dim=data_cfg.dim,
        n_atoms=data_cfg.n_atoms,
        hidden_dims=Int.(cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
    )
    ctx = NBodyFlowMatchingContext(
        field,
        device,
        cfgint(cfg, "sampling.steps", 64),
    )
    params = init_fm_params(rng, field; device)

    params, _, losses = train_flow_matching_adam(
        ctx, params, train_data;
        epochs=cfgint(cfg, "training.epochs", 10),
        batch_size=cfgint(cfg, "training.batch_size", 64),
        learning_rate=cfgfloat32(cfg, "training.learning_rate", 1f-3),
        rng,
    )

    samples = generate_flow_matching_samples(
        ctx, params, cfgint(cfg, "sampling.n_samples", 1000); rng)
    metrics = merge(_sample_metrics(train_data, samples), Dict{String,Float32}(
        "final_training_loss" => isempty(losses) ? Float32(NaN) : last(losses),
    ))

    result = FlowMatchingResult(cfg, train_data, params, losses, samples,
                                _output_dir(cfg), metrics)
    result_path = _save_result(result)
    plot_path = _maybe_plot_result(result)
    @info "flow matching experiment complete" output_dir=result.output_dir result_path plot_path metrics
    return result
end

function run_nbody_diffusion_experiment(cfg::Dict{String,Any})
    seed = cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)
    device = _device_from_config(cfg)

    data_cfg = _data_config(cfg)
    train_data = generate_nbody_dataset(rng, data_cfg)
    if cfgbool(cfg, "data.center", true)
        train_data = center_positions(train_data)
    end

    model = build_diffusion_model(
        dim=data_cfg.dim,
        n_atoms=data_cfg.n_atoms,
        hidden_dims=Int.(cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
    )
    n_steps = cfgint(cfg, "diffusion.steps", 100)
    betas, alphas, alpha_bars = _diffusion_schedule(
        n_steps,
        cfgfloat32(cfg, "diffusion.beta_start", 1f-4),
        cfgfloat32(cfg, "diffusion.beta_end", 2f-2),
    )
    ctx = NBodyDiffusionContext(model, device, n_steps, betas, alphas, alpha_bars)
    params = init_diffusion_params(rng, model; device)

    params, _, losses = train_diffusion_adam(
        ctx, params, train_data;
        epochs=cfgint(cfg, "training.epochs", 10),
        batch_size=cfgint(cfg, "training.batch_size", 64),
        learning_rate=cfgfloat32(cfg, "training.learning_rate", 1f-3),
        rng,
    )

    samples = generate_diffusion_samples(
        ctx, params, cfgint(cfg, "sampling.n_samples", 1000); rng)
    metrics = merge(_sample_metrics(train_data, samples), Dict{String,Float32}(
        "final_training_loss" => isempty(losses) ? Float32(NaN) : last(losses),
    ))

    result = DiffusionResult(cfg, train_data, params, losses, samples,
                             _output_dir(cfg), metrics)
    result_path = _save_result(result)
    plot_path = _maybe_plot_result(result)
    @info "diffusion experiment complete" output_dir=result.output_dir result_path plot_path metrics
    return result
end

function run_experiment(config_path::AbstractString)
    cfg = load_yaml_config(config_path)
    return run_experiment(cfg)
end

function run_experiment(cfg::Dict{String,Any})
    family = cfgsymbol(cfg, "experiment.family", "nbody_cnf")
    if family == :nbody_cnf
        return run_nbody_experiment(cfg)
    elseif family in (:nbody_flow_matching, :nbody_fm, :nbody_equivariant_fm)
        return run_nbody_flow_matching_experiment(cfg)
    elseif family in (:nbody_diffusion, :nbody_equivariant_diffusion)
        return run_nbody_diffusion_experiment(cfg)
    end
    error("Unsupported experiment.family: $family")
end
