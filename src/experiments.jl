using Serialization

export ExperimentResult, run_nbody_cnf_experiment, run_nbody_diffusion_experiment, run_experiment

struct ExperimentResult
    config::Dict{String,Any}
    train_data
    params
    losses::Vector{Float32}
    samples
    output_dir::String
    metrics::Dict{String,Float32}
end

_cpu_device() = DiffEqFlux.Lux.cpu_device()

function _gpu_initialization_error_message(err)
    return sprint(showerror, err)
end

function _device_from_config(cfg)
    use_gpu = cfgbool(cfg, "runtime.use_gpu", false)
    use_gpu || return _cpu_device()

    if !CUDA.functional()
        @warn "runtime.use_gpu=true but CUDA is not functional; falling back to CPU"
        return _cpu_device()
    end

    strict_gpu = cfgbool(cfg, "runtime.strict_gpu", false)
    try
        CUDA.zeros(Float32, 1)
        return CUDA.cu
    catch err
        if strict_gpu
            rethrow()
        end
        @warn "CUDA initialization failed; falling back to CPU" error=_gpu_initialization_error_message(err)
        return _cpu_device()
    end
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
        source_path=String(cfgget(cfg, "data.source_path", cfgget(cfg, "data.path", ""))),
        source_paths=String.(cfgget(cfg, "data.source_paths", cfgget(cfg, "data.paths", String[]))),
        source_dir=String(cfgget(cfg, "data.source_dir", cfgget(cfg, "data.dir", ""))),
        source_pattern=String(cfgget(cfg, "data.source_pattern", cfgget(cfg, "data.pattern", "*.h5"))),
        allow_partial=cfgbool(cfg, "data.allow_partial", false),
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

function _copy_config_snapshot(config_path, output_dir::AbstractString)
    config_path === nothing && return nothing
    cp(String(config_path), joinpath(output_dir, "config.yaml"); force=true)
    return nothing
end

function _output_dir(cfg; config_path=nothing)
    default_name = "nbody_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    path = config_output_dir(cfg; default_root="runs", default_name)
    mkpath(path)
    _copy_config_snapshot(config_path, path)
    return path
end

struct _TeeLogger <: AbstractLogger
    console::AbstractLogger
    file::AbstractLogger
end

Logging.min_enabled_level(logger::_TeeLogger) =
    min(Logging.min_enabled_level(logger.console),
        Logging.min_enabled_level(logger.file))

Logging.shouldlog(logger::_TeeLogger, level, _module, group, id) =
    Logging.shouldlog(logger.console, level, _module, group, id) ||
    Logging.shouldlog(logger.file, level, _module, group, id)

Logging.catch_exceptions(logger::_TeeLogger) =
    Logging.catch_exceptions(logger.console) &&
    Logging.catch_exceptions(logger.file)

function Logging.handle_message(logger::_TeeLogger, level, message, _module,
                                group, id, file, line; kwargs...)
    if Logging.shouldlog(logger.console, level, _module, group, id)
        Logging.handle_message(logger.console, level, message, _module,
                               group, id, file, line; kwargs...)
    end
    if Logging.shouldlog(logger.file, level, _module, group, id)
        Logging.handle_message(logger.file, level, message, _module,
                               group, id, file, line; kwargs...)
        hasproperty(logger.file, :stream) && flush(getfield(logger.file, :stream))
    end
    return nothing
end

function _with_run_logger(f, output_dir::AbstractString)
    log_path = joinpath(output_dir, "run.log")
    open(log_path, "a") do io
        logger = _TeeLogger(current_logger(), SimpleLogger(io, Logging.Info))
        with_logger(logger) do
            @info "run logging started" output_dir log_path started_at=Dates.now()
            try
                return f(log_path)
            finally
                @info "run logging finished" output_dir log_path finished_at=Dates.now()
                flush(io)
            end
        end
    end
end

function _latest_checkpoint_path(checkpoint_dir::AbstractString)
    isdir(checkpoint_dir) || return nothing
    latest_path = joinpath(checkpoint_dir, "latest.jls")
    isfile(latest_path) && return latest_path
    candidates = [
        joinpath(checkpoint_dir, name)
        for name in readdir(checkpoint_dir)
        if startswith(name, "checkpoint_epoch_") && endswith(name, ".jls")
    ]
    isempty(candidates) && return nothing
    return last(sort(candidates))
end

function _resume_checkpoint(cfg, output_dir::AbstractString)
    checkpoint_path = cfgget(cfg, "training.resume_checkpoint", nothing)
    if checkpoint_path === nothing
        checkpoint_dir = String(cfgget(cfg, "training.checkpoint_dir",
                                       joinpath(output_dir, "checkpoints")))
        checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
    end
    checkpoint_path === nothing && return nothing
    isfile(String(checkpoint_path)) ||
        error("Resume checkpoint does not exist: $(checkpoint_path)")
    checkpoint = deserialize(String(checkpoint_path))
    @info "resuming checkpoint" checkpoint_path epoch=get(checkpoint, "epoch", nothing)
    return checkpoint
end

function _atomic_serialize(path::AbstractString, value)
    tmp_path = path * ".tmp"
    serialize(tmp_path, value)
    mv(tmp_path, path; force=true)
    return path
end

function _loss_stream_callback(cfg, output_dir::AbstractString, loss_kind::AbstractString)
    path = joinpath(output_dir, "losses.jls")
    records = Vector{Dict{String,Any}}()
    if isfile(path)
        try
            previous = deserialize(path)
            append!(records, get(previous, "records", Vector{Dict{String,Any}}()))
        catch err
            @warn "Could not load previous loss stream for resume" path exception=(err, catch_backtrace())
        end
    end

    return function (; epoch, batch, step, total_steps, loss)
        push!(records, Dict{String,Any}(
            "epoch" => epoch,
            "batch" => batch,
            "step" => step,
            "total_steps" => total_steps,
            "loss" => Float32(loss),
        ))
        _atomic_serialize(path, Dict{String,Any}(
            "config" => cfg,
            "loss_kind" => String(loss_kind),
            "losses" => Float32[record["loss"] for record in records],
            "records" => records,
            "updated_at" => Dates.now(),
        ))
        return path
    end
end

function _checkpoint_callback(cfg, output_dir::AbstractString)
    checkpoint_every = cfgint(cfg, "training.checkpoint_every", 0)
    checkpoint_every > 0 || return nothing

    checkpoint_dir = String(cfgget(cfg, "training.checkpoint_dir",
                                   joinpath(output_dir, "checkpoints")))
    keep_all = cfgbool(cfg, "training.keep_all_checkpoints", true)
    mkpath(checkpoint_dir)

    return function (; epoch, params, opt_state, losses, loss)
        if epoch % checkpoint_every != 0
            return nothing
        end

        checkpoint = Dict(
            "config" => cfg,
            "epoch" => epoch,
            "loss" => loss,
            "losses" => copy(losses),
            "params" => DiffEqFlux.Lux.cpu_device()(params),
            "opt_state" => DiffEqFlux.Lux.cpu_device()(opt_state),
        )
        if keep_all
            path = joinpath(checkpoint_dir, "checkpoint_epoch_$(lpad(epoch, 6, '0')).jls")
            serialize(path, checkpoint)
        else
            path = joinpath(checkpoint_dir, "latest.jls")
            serialize(path, checkpoint)
        end
        @info "checkpoint saved" epoch path
        return path
    end
end

function _sample_metrics(train_data, samples)
    samples_cpu = DiffEqFlux.Lux.cpu_device()(samples)
    return Dict{String,Float32}(
        "pairwise_distance_mae" => pairwise_distance_mae(train_data, samples_cpu),
    )
end

function _polymer_langevin_force_targets(train_data, data_cfg::NBodyDataConfig)
    data_cfg.kind in (:polymer_langevin, :rouse_hdf5) ||
        error("Force evaluation currently requires polymer_langevin or rouse_hdf5 data")
    p = data_cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0f0
    bond_length = length(p) >= 2 ? p[2] : 1.0f0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0f0 * diffusion / bond_length^2
    nonideal = polymer_nonideal_params(p)

    targets = similar(train_data)
    for i in axes(train_data, 3)
        polymer_langevin_force!(@view(targets[:, :, i]), @view(train_data[:, :, i]),
                                diffusion, k_over_xi, nonideal)
    end
    return targets
end

function _cnf_gradlogp_force_mse_metric(ctx::NBodyCNFContext, params, train_data,
                                        force_targets; batch_size::Int=64,
                                        rng::AbstractRNG=Random.default_rng())
    n_samples = size(train_data, 3)
    sqerr = 0.0
    n_values = 0
    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        pred = DiffEqFlux.Lux.cpu_device()(
            cnf_logp_gradient(ctx, params, train_data[:, :, batch_idx]; rng))
        target = @view(force_targets[:, :, batch_idx])
        sqerr += sum(abs2, pred .- target)
        n_values += length(target)
    end
    return Float32(sqerr / n_values)
end

function _diffusion_gradlogp_force_mse_metric(ctx::NBodyDiffusionContext, params,
                                             train_data, force_targets;
                                             batch_size::Int=64)
    n_samples = size(train_data, 3)
    sqerr = 0.0
    n_values = 0
    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        pred = DiffEqFlux.Lux.cpu_device()(
            diffusion_logp_gradient(ctx, params, train_data[:, :, batch_idx]))
        target = @view(force_targets[:, :, batch_idx])
        sqerr += sum(abs2, pred .- target)
        n_values += length(target)
    end
    return Float32(sqerr / n_values)
end

function _gradlogp_force_cosine_stats(pred, target)
    batch_size = size(target, 3)
    cosine_sum = 0.0
    n_valid = 0
    for i in 1:batch_size
        pred_i = @view(pred[:, :, i])
        target_i = @view(target[:, :, i])
        pred_norm2 = sum(abs2, pred_i)
        target_norm2 = sum(abs2, target_i)
        if pred_norm2 > 0.0 && target_norm2 > 0.0
            cosine_sum += sum(pred_i .* target_i) / sqrt(pred_norm2 * target_norm2)
            n_valid += 1
        end
    end
    return cosine_sum, n_valid
end

function _cnf_gradlogp_force_cosine_metric(ctx::NBodyCNFContext, params, train_data,
                                           force_targets; batch_size::Int=64,
                                           rng::AbstractRNG=Random.default_rng())
    n_samples = size(train_data, 3)
    cosine_sum = 0.0
    n_valid = 0
    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        pred = DiffEqFlux.Lux.cpu_device()(
            cnf_logp_gradient(ctx, params, train_data[:, :, batch_idx]; rng))
        target = @view(force_targets[:, :, batch_idx])
        batch_cosine_sum, batch_valid = _gradlogp_force_cosine_stats(pred, target)
        cosine_sum += batch_cosine_sum
        n_valid += batch_valid
    end
    return n_valid == 0 ? Float32(NaN) : Float32(cosine_sum / n_valid)
end

function _diffusion_gradlogp_force_cosine_metric(ctx::NBodyDiffusionContext, params,
                                                 train_data, force_targets;
                                                 batch_size::Int=64)
    n_samples = size(train_data, 3)
    cosine_sum = 0.0
    n_valid = 0
    for batch_start in 1:batch_size:n_samples
        batch_stop = min(batch_start + batch_size - 1, n_samples)
        batch_idx = batch_start:batch_stop
        pred = DiffEqFlux.Lux.cpu_device()(
            diffusion_logp_gradient(ctx, params, train_data[:, :, batch_idx]))
        target = @view(force_targets[:, :, batch_idx])
        batch_cosine_sum, batch_valid = _gradlogp_force_cosine_stats(pred, target)
        cosine_sum += batch_cosine_sum
        n_valid += batch_valid
    end
    return n_valid == 0 ? Float32(NaN) : Float32(cosine_sum / n_valid)
end

function _save_result(result::ExperimentResult)
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
    plt = scatter(samples[1, :, :], samples[2, :, :];
                  title="Diffusion samples vs training data",
                  xlabel="x", ylabel="y", markersize=2,
                  markercolor=:blue, legend=false, aspect_ratio=:equal)
    scatter!(plt, train_data[1, :, :], train_data[2, :, :];
             markersize=2, markercolor=:tomato, legend=false)
    savefig(plt, path)
    return path
end


function center_of_mass(x)
    return mean(x; dims=2)
end

function center_positions(x)
    return x .- center_of_mass(x)
end


function run_nbody_cnf_experiment(cfg::Dict{String,Any}; config_path=nothing)
    seed = cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)
    device = _device_from_config(cfg)
    output_dir = _output_dir(cfg; config_path)

    return _with_run_logger(output_dir) do log_path
        @info "starting nbody CNF experiment" config_path output_dir log_path
        data_cfg = _data_config(cfg)
        train_data = generate_nbody_dataset(rng, data_cfg)
        if cfgbool(cfg, "data.center", false)
            train_data = center_positions(train_data)
        end

        field = build_vector_field(
            cfgsymbol(cfg, "model.kind", "mlp");
            dim=data_cfg.dim,
            n_atoms=data_cfg.n_atoms,
            hidden_dims=Int.(cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
            node_embedding_dim=cfgint(cfg, "model.node_embedding_dim", min(16, data_cfg.n_atoms)),
        )
        ctx = _make_context(cfg, field, device)
        params = init_cnf_params(rng, field; device)
        resume_checkpoint = _resume_checkpoint(cfg, output_dir)
        start_epoch = 0
        opt_state = nothing
        previous_losses = Float32[]
        if resume_checkpoint !== nothing
            start_epoch = Int(get(resume_checkpoint, "epoch", 0))
            target_epochs = cfgint(cfg, "training.epochs", 10)
            start_epoch < target_epochs ||
                error("Resume checkpoint epoch ($start_epoch) is already >= training.epochs ($target_epochs)")
            params = get(resume_checkpoint, "params", params) |> device
            opt_state = get(resume_checkpoint, "opt_state", nothing)
            opt_state = opt_state === nothing ? nothing : opt_state |> device
            previous_losses = Float32.(get(resume_checkpoint, "losses", Float32[]))
        end

        params, _, losses = train_cnf_adam(
            ctx, params, train_data;
            epochs=cfgint(cfg, "training.epochs", 10),
            batch_size=cfgint(cfg, "training.batch_size", 64),
            learning_rate=cfgfloat32(cfg, "training.learning_rate", 5f-3),
            log_every=cfgint(cfg, "training.log_every", 1),
            checkpoint_callback=_checkpoint_callback(cfg, output_dir),
            loss_callback=_loss_stream_callback(cfg, output_dir, "cnf_negative_log_likelihood"),
            opt_state,
            loss_history=previous_losses,
            start_epoch,
            rng,
        )
        samples = generate_cnf_samples(
            ctx, params, cfgint(cfg, "sampling.n_samples", 1000); rng)

        metrics = merge(_sample_metrics(train_data, samples), Dict{String,Float32}(
            "final_training_loss" => isempty(losses) ? Float32(NaN) : last(losses),
        ))
        if data_cfg.kind in (:polymer_langevin, :rouse_hdf5)
            force_targets = _polymer_langevin_force_targets(train_data, data_cfg)
            metrics["gradlogp_force_mse"] = _cnf_gradlogp_force_mse_metric(
                ctx, params, train_data, force_targets;
                batch_size=cfgint(cfg, "training.batch_size", 64),
                rng,
            )
        end

        result = ExperimentResult(cfg, train_data, params, losses, samples,
                                  output_dir, metrics)
        result_path = _save_result(result)
        plot_path = _maybe_plot_result(result)
        @info "experiment complete" output_dir=result.output_dir result_path plot_path metrics
        return result
    end
end

function run_nbody_flow_matching_experiment(cfg::Dict{String,Any}; config_path=nothing)
    seed = cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)
    device = _device_from_config(cfg)
    output_dir = _output_dir(cfg; config_path)

    return _with_run_logger(output_dir) do log_path
        @info "starting nbody flow matching experiment" config_path output_dir log_path
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
            log_every=cfgint(cfg, "training.log_every", 1),
            checkpoint_callback=_checkpoint_callback(cfg, output_dir),
            loss_callback=_loss_stream_callback(cfg, output_dir, "flow_matching_velocity_mse"),
            rng,
        )

        samples = generate_flow_matching_samples(
            ctx, params, cfgint(cfg, "sampling.n_samples", 1000); rng)
        metrics = merge(_sample_metrics(train_data, samples), Dict{String,Float32}(
            "final_training_loss" => isempty(losses) ? Float32(NaN) : last(losses),
        ))

        result = FlowMatchingResult(cfg, train_data, params, losses, samples,
                                    output_dir, metrics)
        result_path = _save_result(result)
        plot_path = _maybe_plot_result(result)
        @info "flow matching experiment complete" output_dir=result.output_dir result_path plot_path metrics
        return result
    end
end

function run_nbody_diffusion_experiment(cfg::Dict{String,Any}; config_path=nothing)
    seed = cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)
    device = _device_from_config(cfg)
    output_dir = _output_dir(cfg; config_path)

    return _with_run_logger(output_dir) do log_path
        @info "starting nbody diffusion experiment" config_path output_dir log_path
        data_cfg = _data_config(cfg)
        train_data = generate_nbody_dataset(rng, data_cfg)
        if cfgbool(cfg, "data.center", true)
            train_data = center_positions(train_data)
        end

        model = build_diffusion_model(
            dim=data_cfg.dim,
            n_atoms=data_cfg.n_atoms,
            hidden_dims=Int.(cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
            n_layers=cfgint(cfg, "model.n_layers", 4),
            node_embedding_dim=cfgint(cfg, "model.node_embedding_dim", min(16, data_cfg.n_atoms)),
        )
        sample_steps = cfgint(cfg, "sampling.steps", cfgint(cfg, "diffusion.steps", 1000))
        ctx = NBodyDiffusionContext(model, device, sample_steps)
        params = init_diffusion_params(rng, model; device)

        params, _, losses = train_diffusion_adam(
            ctx, params, train_data;
            epochs=cfgint(cfg, "training.epochs", 10),
            batch_size=cfgint(cfg, "training.batch_size", 64),
            learning_rate=cfgfloat32(cfg, "training.learning_rate", 1f-3),
            log_every=cfgint(cfg, "training.log_every", 1),
            checkpoint_callback=_checkpoint_callback(cfg, output_dir),
            loss_callback=_loss_stream_callback(cfg, output_dir, "diffusion_denoising_score_mse"),
            rng,
        )

        samples = generate_diffusion_samples(
            ctx, params, cfgint(cfg, "sampling.n_samples", 1000); rng)
        metrics = merge(_sample_metrics(train_data, samples), Dict{String,Float32}(
            "final_training_loss" => isempty(losses) ? Float32(NaN) : last(losses),
        ))
        if data_cfg.kind in (:polymer_langevin, :rouse_hdf5)
            force_targets = _polymer_langevin_force_targets(train_data, data_cfg)
            metrics["gradlogp_force_mse"] = _diffusion_gradlogp_force_mse_metric(
                ctx, params, train_data, force_targets;
                batch_size=cfgint(cfg, "training.batch_size", 64),
            )
        end

        result = DiffusionResult(cfg, train_data, params, losses, samples,
                                 output_dir, metrics)
        result_path = _save_result(result)
        plot_path = _maybe_plot_result(result)
        @info "diffusion experiment complete" output_dir=result.output_dir result_path plot_path metrics
        return result
    end
end

function run_experiment(config_path::AbstractString)
    cfg = load_yaml_config(config_path)
    return run_experiment(cfg; config_path)
end

function run_experiment(cfg::Dict{String,Any}; config_path=nothing)
    family = cfgsymbol(cfg, "experiment.family", "nbody_cnf")
    if family == :nbody_cnf
        return run_nbody_cnf_experiment(cfg; config_path)
    elseif family in (:nbody_diffusion, :nbody_equivariant_diffusion)
        return run_nbody_diffusion_experiment(cfg; config_path)
    end
    error("Unsupported experiment.family: $family")
end
