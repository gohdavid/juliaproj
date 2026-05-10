export EquivariantDiffusionModel, NBodyDiffusionContext, DiffusionResult
export build_diffusion_model, init_diffusion_params, diffusion_loss
export train_diffusion_adam, generate_diffusion_samples, predict_diffusion_score
export diffusion_logp_gradient
export beta, mu, sigma_square, sigma, forward_noise, euler_maruyama_sample

struct EquivariantDiffusionModel
    dim::Int
    n_atoms::Int
    node_embedding_dim::Int
    hidden_dims::Vector{Int}
    n_layers::Int
    normalize_edge_features::Bool
    seq_dist_scale::Float32
    radial_scale::Float32
    coordinate_update_scale::Float32
    edge_nets
    edge_states
    node_nets
    node_states
end

struct NBodyDiffusionContext
    model::EquivariantDiffusionModel
    device
    sample_steps::Int
    sample_t_min::Float32
    sample_t_max::Float32
    max_score_norm::Float32
end

NBodyDiffusionContext(model::EquivariantDiffusionModel, device, sample_steps::Int) =
    NBodyDiffusionContext(model, device, sample_steps,
                          _T_SAMPLE_MIN_F32, _T_SAMPLE_MAX_F32,
                          _MAX_SAMPLE_SCORE_NORM_F32)

struct DiffusionResult
    config::Dict{String,Any}
    train_data
    params
    losses::Vector{Float32}
    samples
    output_dir::String
    metrics::Dict{String,Float32}
end

function _diffusion_silu(x)
    return x / (one(x) + exp(-x))
end

function _diffusion_dense_chain(input_dim::Int, output_dim::Int,
                                hidden_dims::Vector{Int};
                                activation=_diffusion_silu,
                                final_activation=identity)
    dims = [input_dim; hidden_dims; output_dim]
    layers = Any[]
    for i in 1:(length(dims) - 1)
        layer_activation = i == length(dims) - 1 ? final_activation : activation
        push!(layers, DiffEqFlux.Dense(dims[i] => dims[i + 1], layer_activation))
    end
    return DiffEqFlux.Chain(layers...)
end

function build_diffusion_model(; dim::Int=2, n_atoms::Int=10,
                               hidden_dims=[64, 64, 64],
                               n_layers::Int=4,
                               node_embedding_dim::Int=min(16, n_atoms),
                               normalize_edge_features::Bool=true,
                               seq_dist_scale::Real=max(n_atoms - 1, 1),
                               radial_scale::Real=max(n_atoms, 1),
                               coordinate_update_scale::Real=0.1f0,
                               activation=tanh)
    hidden = Int.(hidden_dims)
    edge_input_dim = 2 * node_embedding_dim + 3
    edge_nets = ntuple(_ -> _diffusion_dense_chain(edge_input_dim, 1, hidden; activation),
                       n_layers)
    edge_states = map(net -> DiffEqFlux.Lux.initialstates(Random.default_rng(), net),
                      edge_nets)
    node_nets = ntuple(_ -> _diffusion_dense_chain(node_embedding_dim + 1, node_embedding_dim,
                                                   hidden; activation),
                       n_layers)
    node_states = map(net -> DiffEqFlux.Lux.initialstates(Random.default_rng(), net),
                      node_nets)
    return EquivariantDiffusionModel(dim, n_atoms, node_embedding_dim, hidden, n_layers,
                                     normalize_edge_features,
                                     Float32(seq_dist_scale),
                                     Float32(radial_scale),
                                     Float32(coordinate_update_scale),
                                     edge_nets, edge_states,
                                     node_nets, node_states)
end

function init_diffusion_params(rng::AbstractRNG, model::EquivariantDiffusionModel;
                               device=identity)
    edge_params = map(model.edge_nets) do net
        params, _ = DiffEqFlux.Lux.setup(rng, net)
        params
    end
    node_params = map(model.node_nets) do net
        params, _ = DiffEqFlux.Lux.setup(rng, net)
        params
    end
    node_embed = 0.02f0 .* randn(rng, Float32, model.node_embedding_dim, model.n_atoms)
    return (edge=edge_params,
            node=node_params,
            node_embed=node_embed) |> device
end

const _VP_BETA_MIN_F32 = 0.1f0
const _VP_BETA_MAX_F32 = 20.0f0
const _VP_BETA_DELTA_F32 = _VP_BETA_MAX_F32 - _VP_BETA_MIN_F32
const _T_MIN_F32 = 1.0f-5
const _T_SAMPLE_MIN_F32 = 1.0f-4
const _T_SAMPLE_MAX_F32 = 1.0f0
const _MAX_SAMPLE_SCORE_NORM_F32 = 64.0f0
const _MAX_DIFFUSION_GRAD_VALUE_F32 = 1.0f0
const _MIN_DIFFUSION_SIGMA_SQUARE_F32 =
    Float32(1.0 - exp(-0.5 * Float64(_T_MIN_F32)^2 *
                      Float64(_VP_BETA_DELTA_F32) -
                      Float64(_VP_BETA_MIN_F32) * Float64(_T_MIN_F32)))
const _MIN_DIFFUSION_SIGMA_F32 = sqrt(_MIN_DIFFUSION_SIGMA_SQUARE_F32)

function beta(t)
    return @. _VP_BETA_MIN_F32 + t * _VP_BETA_DELTA_F32
end

function mu(t)
    return @. exp(-0.25f0 * t^2 * _VP_BETA_DELTA_F32 -
                  0.5f0 * _VP_BETA_MIN_F32 * t)
end

function mu(x0, t)
    return x0 .* mu(t)
end

function sigma_square(t)
    return @. 1.0f0 - exp(-0.5f0 * t^2 * _VP_BETA_DELTA_F32 -
                          _VP_BETA_MIN_F32 * t)
end

function sigma(t)
    return @. sqrt(max(sigma_square(t), 0.0f0))
end

function _stable_diffusion_time(t)
    return @. max(t, _T_MIN_F32)
end

function _safe_diffusion_sigma(t)
    raw_sigma = sigma(t)
    return @. ifelse(isfinite(raw_sigma),
                     max(raw_sigma, _MIN_DIFFUSION_SIGMA_F32),
                     _MIN_DIFFUSION_SIGMA_F32)
end

function _as_batch_time(t, batch_size::Int, device)
    if t isa Number
        return fill(Float32(t), 1, 1, batch_size) |> device
    end
    t_arr = Float32.(t)
    if ndims(t_arr) == 1
        length(t_arr) == batch_size ||
            error("Expected $batch_size time values, got $(length(t_arr)).")
        return reshape(t_arr, 1, 1, batch_size) |> device
    end
    return t_arr |> device
end

function forward_noise(x0, t; rng::AbstractRNG=Random.default_rng(), device=identity)
    batch_size = size(x0, ndims(x0))
    t_batch = _as_batch_time(t, batch_size, device)
    eps = center_positions(randn(rng, eltype(x0), size(x0)...)) |> device
    xt = mu(x0, t_batch) .+ sigma(t_batch) .* eps
    xt = center_positions(xt)
    return xt, eps
end

function _diffusion_fixed_inputs(model::EquivariantDiffusionModel, batch_size::Int, device)
    n_atoms = model.n_atoms

    seq_dist_fixed = Float32[i - j for i in 1:n_atoms, j in 1:n_atoms]
    if model.normalize_edge_features
        seq_dist_fixed ./= max(model.seq_dist_scale, eps(Float32))
    end
    seq_dist_fixed = reshape(seq_dist_fixed, 1, n_atoms * n_atoms)
    seq_dist_flat = repeat(seq_dist_fixed, outer=(1, batch_size)) |> device

    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device
    return seq_dist_flat, mask
end

_diffusion_layer_params(layer_params::Tuple, layer_idx::Int) = layer_params[layer_idx]
_diffusion_layer_params(layer_params, ::Int) = layer_params

function _diffusion_edge_message(input, params, model::EquivariantDiffusionModel,
                                 layer_idx::Int)
    edge_params = _diffusion_layer_params(params.edge, layer_idx)
    val, _ = model.edge_nets[layer_idx](
        input, edge_params, model.edge_states[layer_idx])
    return val
end

function _diffusion_node_state(input, params, model::EquivariantDiffusionModel,
                               layer_idx::Int)
    node_params = _diffusion_layer_params(params.node, layer_idx)
    val, _ = model.node_nets[layer_idx](
        input, node_params, model.node_states[layer_idx])
    return val
end

function _diffusion_radial_feature(d_ij, model::EquivariantDiffusionModel)
    if model.normalize_edge_features
        scale = max(model.radial_scale, eps(Float32))
        return @. log1p(max(d_ij, 0.0f0) / scale)
    end
    return d_ij
end

function _diffusion_node_pair_features_primal(h)
    feature_dim, n_atoms, batch_size = size(h)
    h_i = repeat(reshape(h, feature_dim, n_atoms, 1, batch_size),
                 inner=(1, 1, n_atoms, 1))
    h_j = repeat(reshape(h, feature_dim, 1, n_atoms, batch_size),
                 inner=(1, n_atoms, 1, 1))
    return reshape(h_i, feature_dim, n_atoms * n_atoms * batch_size),
           reshape(h_j, feature_dim, n_atoms * n_atoms * batch_size)
end

function _diffusion_node_pair_features(h)
    return _diffusion_node_pair_features_primal(h)
end

function _diffusion_vcat_rows(parts...)
    return vcat(parts...)
end

Zygote.@adjoint function _diffusion_vcat_rows(parts...)
    y = vcat(parts...)
    function pullback(Δ)
        Δ === nothing && return ntuple(_ -> nothing, length(parts))
        Δ = Zygote.unthunk(Δ)

        row_start = 1
        grads = map(parts) do part
            row_stop = row_start + size(part, 1) - 1
            grad = Δ[row_start:row_stop, :]
            row_start = row_stop + 1
            grad
        end
        return grads
    end
    return y, pullback
end

function _diffusion_batch_node_embed(node_embed, batch_size::Int)
    feature_dim, n_atoms = size(node_embed)
    return repeat(reshape(node_embed, feature_dim, n_atoms, 1),
                  outer=(1, 1, batch_size))
end

Zygote.@adjoint function _diffusion_batch_node_embed(node_embed, batch_size::Int)
    y = _diffusion_batch_node_embed(node_embed, batch_size)
    function pullback(Δ)
        Δ === nothing && return (nothing, nothing)
        Δ = Zygote.unthunk(Δ)
        feature_dim, n_atoms = size(node_embed)
        grad = sum(reshape(Δ, feature_dim, n_atoms, batch_size); dims=3)
        return (reshape(grad, feature_dim, n_atoms), nothing)
    end
    return y, pullback
end

Zygote.@adjoint function _diffusion_node_pair_features(h)
    y = _diffusion_node_pair_features_primal(h)
    function pullback(Δ)
        Δ_i, Δ_j = Δ
        feature_dim, n_atoms, batch_size = size(h)
        dh = zero(h)

        if Δ_i !== nothing
            Δ_i = Zygote.unthunk(Δ_i)
            dh_i = sum(reshape(Δ_i, feature_dim, n_atoms, n_atoms, batch_size);
                       dims=3)
            dh = dh .+ reshape(dh_i, feature_dim, n_atoms, batch_size)
        end

        if Δ_j !== nothing
            Δ_j = Zygote.unthunk(Δ_j)
            dh_j = sum(reshape(Δ_j, feature_dim, n_atoms, n_atoms, batch_size);
                       dims=2)
            dh = dh .+ reshape(dh_j, feature_dim, n_atoms, batch_size)
        end

        return (dh,)
    end
    return y, pullback
end

function center_of_mass(x)
    return mean(x; dims=2)
end

function center_positions(x)
    return x .- center_of_mass(x)
end

Zygote.@adjoint function center_positions(x)
    y = x .- center_of_mass(x)
    function pullback(Δ)
        Δ === nothing && return (nothing,)
        return (center_positions(Zygote.unthunk(Δ)),)
    end
    return y, pullback
end

function _equivariant_diffusion_prediction(x, t, params;
                                           model::EquivariantDiffusionModel,
                                           seq_dist_flat, mask)
    dim, n_atoms, batch_size = size(x)
    x_l = x
    h = _diffusion_batch_node_embed(params.node_embed, batch_size)
    mask_const = Zygote.dropgrad(mask)
    t_flat = Zygote.ignore() do
        repeat(reshape(t, 1, batch_size), inner=(1, n_atoms * n_atoms))
    end

    for layer_idx in 1:model.n_layers
        x_i = reshape(x_l, dim, n_atoms, 1, batch_size)
        x_j = reshape(x_l, dim, 1, n_atoms, batch_size)
        r_ij = x_i .- x_j
        d_ij = sum(abs2, r_ij; dims=1)
        d_ij_flat = reshape(_diffusion_radial_feature(d_ij, model),
                            1, n_atoms * n_atoms * batch_size)
        h_i_flat, h_j_flat = _diffusion_node_pair_features(h)
        edge_input = _diffusion_vcat_rows(
            h_i_flat, h_j_flat, seq_dist_flat, d_ij_flat, t_flat)

        m_flat = _diffusion_edge_message(edge_input, params, model, layer_idx)
        scalar_ij = reshape(m_flat, 1, n_atoms, n_atoms, batch_size)

        dx = (model.coordinate_update_scale .*
              sum(r_ij .* (scalar_ij .* mask_const); dims=3)) ./ Float32(n_atoms)
        x_l = x_l .+ reshape(dx, dim, n_atoms, batch_size)

        m_i = sum(scalar_ij .* mask_const; dims=3)
        node_input = _diffusion_vcat_rows(
            reshape(h, model.node_embedding_dim, n_atoms * batch_size),
            reshape(m_i, 1, n_atoms * batch_size))
        h_flat = _diffusion_node_state(node_input, params, model, layer_idx)
        h = h .+ reshape(h_flat, model.node_embedding_dim, n_atoms, batch_size)
    end

    return center_positions(x_l .- x)
end

function _predict_diffusion_score(ctx::NBodyDiffusionContext, params, x_batch, time;
                                  seq_dist_flat=nothing, mask=nothing)
    model = ctx.model
    batch_size = size(x_batch, 3)
    x = center_positions(x_batch) |> ctx.device
    t = _stable_diffusion_time(_as_batch_time(time, batch_size, ctx.device))
    if seq_dist_flat === nothing || mask === nothing
        seq_dist_flat, mask = Zygote.ignore() do
            _diffusion_fixed_inputs(model, batch_size, ctx.device)
        end
    end
    score = _equivariant_diffusion_prediction(
        x, t, params; model, seq_dist_flat, mask)
    return center_positions(score)
end

function predict_diffusion_score(ctx::NBodyDiffusionContext, params, x_batch;
                                 time=_T_MIN_F32)
    return _predict_diffusion_score(ctx, params, x_batch, time)
end

"""
    diffusion_logp_gradient(ctx, params, x_batch; time=_T_MIN_F32)

Return the diffusion score, interpreted as `grad_x logP_t(x)`, for every sample
in `x_batch`. The network output is the score `s(x,t)` directly.
"""
function diffusion_logp_gradient(ctx::NBodyDiffusionContext, params, x_batch;
                                 time=_T_MIN_F32)
    return predict_diffusion_score(ctx, params, x_batch; time)
end

function diffusion_loss(ctx::NBodyDiffusionContext, params, x_batch;
    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    batch_size = size(x_batch, 3)
    x0 = center_positions(x_batch) |> ctx.device
    t = rand(rng, Float32, 1, 1, batch_size) |> ctx.device
    t = _stable_diffusion_time(t)
    xt, eps = Zygote.ignore() do
        forward_noise(x0, t; rng, device=ctx.device)
    end

    seq_dist_flat, mask = Zygote.ignore() do
        _diffusion_fixed_inputs(model, batch_size, ctx.device)
    end
    score_pred = _equivariant_diffusion_prediction(
        xt, t, params; model, seq_dist_flat, mask)
    score_pred = center_positions(score_pred)
    sigma_t = Zygote.ignore() do
        _safe_diffusion_sigma(t)
    end
    score_target = @. -eps / sigma_t
    lambda_t = Zygote.ignore() do
        sigma_square(t)
    end
    return mean(@. lambda_t * abs2(score_pred - score_target))
end

function _clip_diffusion_gradient(g::AbstractArray, max_abs::Float32)
    lo = -max_abs
    hi = max_abs
    z = zero(eltype(g))
    return @. ifelse(isfinite(g), clamp(g, lo, hi), z)
end

_clip_diffusion_gradient(g::Number, max_abs::Float32) =
    isfinite(g) ? clamp(g, -max_abs, max_abs) : zero(g)

_clip_diffusion_gradient(g::NamedTuple, max_abs::Float32) =
    NamedTuple{keys(g)}(map(v -> _clip_diffusion_gradient(v, max_abs), values(g)))

_clip_diffusion_gradient(g::Tuple, max_abs::Float32) =
    map(v -> _clip_diffusion_gradient(v, max_abs), g)

_clip_diffusion_gradient(::Nothing, ::Float32) = nothing

function train_diffusion_adam(ctx::NBodyDiffusionContext, params, train_data;
                              epochs::Int=25, batch_size::Int=64,
                              learning_rate::Real=1f-3,
                              log_every::Int=1,
                              checkpoint_callback=nothing,
                              loss_callback=nothing,
                              rng::AbstractRNG=Xoshiro(42))
    n_samples = size(train_data, 3)
    steps_per_epoch = cld(n_samples, batch_size)
    total_steps = epochs * steps_per_epoch
    step = 0
    opt_state = Optimisers.setup(Optimisers.Adam(Float32(learning_rate)), params)
    loss_history = Float32[]

    for epoch in 1:epochs
        shuffled_idx = randperm(rng, n_samples)
        epoch_loss = 0.0f0
        for (batch_num, batch_start) in enumerate(1:batch_size:n_samples)
            batch_stop = min(batch_start + batch_size - 1, n_samples)
            batch_idx = shuffled_idx[batch_start:batch_stop]
            x_batch = center_positions(train_data[:, :, batch_idx])
            local_batch_size = length(batch_idx)

            loss, grads = Zygote.withgradient(
                p -> diffusion_loss(ctx, p, x_batch; rng), params)
            grad_params = grads[1]
            grad_params === nothing && error("Zygote returned no parameter gradient.")

            loss_f32 = Float32(loss)
            isfinite(loss_f32) ||
                error("Non-finite diffusion loss at epoch=$epoch batch=$batch_num: $loss_f32")
            grad_params = _clip_diffusion_gradient(grad_params,
                                                   _MAX_DIFFUSION_GRAD_VALUE_F32)
            opt_state, params = Optimisers.update(opt_state, params, grad_params)
            step += 1
            epoch_loss += loss_f32 * local_batch_size
            push!(loss_history, loss_f32)
            if loss_callback !== nothing
                loss_callback(;
                    epoch,
                    batch=batch_num,
                    step,
                    total_steps,
                    loss=loss_f32,
                )
            end

            if log_every > 0 && (step == total_steps || step % log_every == 0)
                @info "diffusion_training" epoch batch=batch_num step total_steps steps_per_epoch loss=loss_f32
            end
        end
        mean_loss = Float32(epoch_loss / n_samples)
        if log_every > 0 && (step == total_steps || step % log_every == 0)
            @info "diffusion_training_epoch" epoch step total_steps steps_per_epoch loss=mean_loss
        end
        if checkpoint_callback !== nothing
            checkpoint_callback(;
                epoch,
                params,
                opt_state,
                losses=loss_history,
                loss=mean_loss,
            )
        end
    end

    return params, opt_state, loss_history
end

function _reverse_sde_coefficients(t::Float32)
    beta_t = Float32(beta(t))
    f_t = -0.5f0 * beta_t
    g2_t = beta_t
    g_t = sqrt(g2_t)
    return f_t, g2_t, g_t
end

function _clip_sample_score_norm(score, max_score_norm::Float32)
    isfinite(max_score_norm) || return score
    max_score_norm > 0.0f0 || return score
    norms = sqrt.(sum(abs2, score; dims=(1, 2)))
    scales = @. min(1.0f0, max_score_norm / max(norms, eps(Float32)))
    return score .* scales
end

function euler_maruyama_sample(score_function, shape; num_steps::Int=1000,
                               rng::AbstractRNG=Random.default_rng(),
                               device=identity,
                               t_min::Real=_T_SAMPLE_MIN_F32,
                               t_max::Real=_T_SAMPLE_MAX_F32,
                               max_score_norm::Real=_MAX_SAMPLE_SCORE_NORM_F32)
    x_t = center_positions(randn(rng, Float32, shape...)) |> device
    batch_size = shape[end]
    t0 = Float32(t_min)
    t1 = Float32(t_max)
    0.0f0 <= t0 < t1 <= 1.0f0 ||
        error("Diffusion sampler requires 0 <= t_min < t_max <= 1, got t_min=$t0 t_max=$t1")
    time_steps = range(t1, t0; length=num_steps)
    dt = (t1 - t0) / Float32(num_steps)
    sqrt_dt = sqrt(dt)
    score_clip = Float32(max_score_norm)
    warned_nonfinite = false

    for (step, t_raw) in enumerate(time_steps)
        x_t = center_positions(x_t)
        t = Float32(t_raw)
        f_t, g2_t, g_t = _reverse_sde_coefficients(t)
        t_values = fill(t, batch_size) |> device
        score = center_positions(score_function(x_t, t_values))
        score = _clip_sample_score_norm(score, score_clip)
        z = step == num_steps ?
            zero.(x_t) :
            (center_positions(randn(rng, eltype(x_t), size(x_t)...)) |> device)
        dx = @. (f_t * x_t - g2_t * score) * dt
        noise = @. g_t * sqrt_dt * z
        x_t = @. x_t - dx + noise
        x_t = center_positions(x_t)
        if !warned_nonfinite && !all(isfinite, x_t)
            x_cpu = DiffEqFlux.Lux.cpu_device()(x_t)
            finite_samples = count(i -> all(isfinite, @view(x_cpu[:, :, i])),
                                   axes(x_cpu, 3))
            @warn "non-finite diffusion sample state" step t finite_samples batch_size
            warned_nonfinite = true
        end
    end

    return x_t
end

function euler_maruyama_sample(ctx::NBodyDiffusionContext, params, shape;
                               num_steps::Int=ctx.sample_steps,
                               rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    batch_size = shape[end]
    seq_dist_flat, mask = Zygote.ignore() do
        _diffusion_fixed_inputs(model, batch_size, ctx.device)
    end

    diffusion_score = function (x_t, t_values)
        return _predict_diffusion_score(ctx, params, x_t, t_values;
                                        seq_dist_flat, mask)
    end

    return euler_maruyama_sample(diffusion_score, shape;
                                 num_steps, rng, device=ctx.device,
                                 t_min=ctx.sample_t_min,
                                 t_max=ctx.sample_t_max,
                                 max_score_norm=ctx.max_score_norm)
end

function generate_diffusion_samples(ctx::NBodyDiffusionContext, params,
                                    n_samples::Int;
                                    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    shape = (model.dim, model.n_atoms, n_samples)
    return euler_maruyama_sample(ctx, params, shape;
                                 num_steps=ctx.sample_steps, rng)
end
