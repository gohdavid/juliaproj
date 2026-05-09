export EquivariantDiffusionModel, NBodyDiffusionContext, DiffusionResult
export build_diffusion_model, init_diffusion_params, diffusion_loss
export train_diffusion_adam, generate_diffusion_samples, predict_diffusion_score
export diffusion_logp_gradient
export alpha, beta, forward_noise, euler_maruyama_sample

struct EquivariantDiffusionModel
    dim::Int
    n_atoms::Int
    hidden_dims::Vector{Int}
    n_layers::Int
    edge_net
    edge_state
    coord_net
    coord_state
    node_net
    node_state
end

struct NBodyDiffusionContext
    model::EquivariantDiffusionModel
    device
    sample_steps::Int
end

struct DiffusionResult
    config::Dict{String,Any}
    train_data
    params
    losses::Vector{Float32}
    samples
    output_dir::String
    metrics::Dict{String,Float32}
end

function build_diffusion_model(; dim::Int=2, n_atoms::Int=10,
                               hidden_dims=[64, 64, 64],
                               n_layers::Int=4)
    hidden = Int.(hidden_dims)
    message_dim = dim
    edge_input_dim = 2 * n_atoms + 3
    edge_net = _dense_chain(edge_input_dim, message_dim, hidden)
    edge_state = DiffEqFlux.Lux.initialstates(Random.default_rng(), edge_net)
    coord_net = _dense_chain(message_dim, 1, hidden)
    coord_state = DiffEqFlux.Lux.initialstates(Random.default_rng(), coord_net)
    node_net = _dense_chain(n_atoms + message_dim, n_atoms, hidden)
    node_state = DiffEqFlux.Lux.initialstates(Random.default_rng(), node_net)
    return EquivariantDiffusionModel(dim, n_atoms, hidden, n_layers,
                                     edge_net, edge_state,
                                     coord_net, coord_state,
                                     node_net, node_state)
end

function init_diffusion_params(rng::AbstractRNG, model::EquivariantDiffusionModel;
                               device=identity)
    edge_params, _ = DiffEqFlux.Lux.setup(rng, model.edge_net)
    coord_params, _ = DiffEqFlux.Lux.setup(rng, model.coord_net)
    node_params, _ = DiffEqFlux.Lux.setup(rng, model.node_net)
    return ComponentArray((edge=edge_params,
                           coord=coord_params,
                           node=node_params)) |> device
end

const _HALF_PI_F32 = Float32(pi / 2)
const _T_MIN_F32 = 1.0f-5
const _T_SAMPLE_EPS_F32 = 1.0f-4

function beta(t)
    return @. sin(_HALF_PI_F32 * t)
end

function alpha(t)
    return @. cos(_HALF_PI_F32 * t)
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
    xt = @. alpha(t_batch) * x0 + beta(t_batch) * eps
    xt = center_positions(xt)
    return xt, eps
end

function _diffusion_fixed_inputs(model::EquivariantDiffusionModel, batch_size::Int, device)
    n_atoms = model.n_atoms
    h_fixed = Matrix{Float32}(I, n_atoms, n_atoms)
    h = repeat(reshape(h_fixed, n_atoms, n_atoms, 1), outer=(1, 1, batch_size)) |>
        device

    seq_dist_fixed = Float32[i - j for i in 1:n_atoms, j in 1:n_atoms]
    seq_dist_fixed = reshape(seq_dist_fixed, 1, n_atoms * n_atoms)
    seq_dist_flat = repeat(seq_dist_fixed, outer=(1, batch_size)) |> device

    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device
    return h, seq_dist_flat, mask
end

function _diffusion_edge_message(input, params, model::EquivariantDiffusionModel)
    val, _ = model.edge_net(input, params.edge, model.edge_state)
    return val
end

function _diffusion_coord_weight(input, params, model::EquivariantDiffusionModel)
    val, _ = model.coord_net(input, params.coord, model.coord_state)
    return val
end

function _diffusion_node_state(input, params, model::EquivariantDiffusionModel)
    val, _ = model.node_net(input, params.node, model.node_state)
    return val
end

function _diffusion_node_pair_features(h)
    feature_dim, n_atoms, batch_size = size(h)
    h_i = repeat(reshape(h, feature_dim, n_atoms, 1, batch_size),
                 inner=(1, 1, n_atoms, 1))
    h_j = repeat(reshape(h, feature_dim, 1, n_atoms, batch_size),
                 inner=(1, n_atoms, 1, 1))
    return reshape(h_i, feature_dim, n_atoms * n_atoms * batch_size),
           reshape(h_j, feature_dim, n_atoms * n_atoms * batch_size)
end

function center_of_mass(x)
    return mean(x; dims=2)
end

function center_positions(x)
    return x .- center_of_mass(x)
end

function _equivariant_diffusion_prediction(x, t, params;
                                           model::EquivariantDiffusionModel,
                                           h, seq_dist_flat, mask)
    dim, n_atoms, batch_size = size(x)
    x_l = x
    mask_const = Zygote.dropgrad(mask)
    t_flat = repeat(reshape(t, 1, batch_size), inner=(1, n_atoms * n_atoms))

    for _ in 1:model.n_layers
        x_i = reshape(x_l, dim, n_atoms, 1, batch_size)
        x_j = reshape(x_l, dim, 1, n_atoms, batch_size)
        r_ij = x_i .- x_j
        d_ij = sum(abs2, r_ij; dims=1)
        d_ij_flat = reshape(d_ij, 1, n_atoms * n_atoms * batch_size)
        h_i_flat, h_j_flat = _diffusion_node_pair_features(h)
        edge_input = vcat(h_i_flat, h_j_flat, seq_dist_flat, d_ij_flat, t_flat)

        m_flat = _diffusion_edge_message(edge_input, params, model)
        m_ij = reshape(m_flat, dim, n_atoms, n_atoms, batch_size)
        coord_flat = _diffusion_coord_weight(m_flat, params, model)
        coord_ij = reshape(coord_flat, 1, n_atoms, n_atoms, batch_size)

        dx = sum(r_ij .* (coord_ij .* mask_const); dims=3) ./ Float32(n_atoms)
        x_l = x_l .+ reshape(dx, dim, n_atoms, batch_size)

        m_i = sum(m_ij .* mask_const; dims=3)
        node_input = vcat(reshape(h, n_atoms, n_atoms * batch_size),
                          reshape(m_i, dim, n_atoms * batch_size))
        h_flat = _diffusion_node_state(node_input, params, model)
        h = reshape(h_flat, n_atoms, n_atoms, batch_size)
    end

    return center_positions(x_l .- x)
end

function predict_diffusion_score(ctx::NBodyDiffusionContext, params, x_batch;
                                 time::Real=_T_MIN_F32)
    model = ctx.model
    batch_size = size(x_batch, 3)
    x = center_positions(x_batch) |> ctx.device
    t = fill(Float32(time), 1, 1, batch_size) |> ctx.device
    h, seq_dist_flat, mask = Zygote.ignore() do
        _diffusion_fixed_inputs(model, batch_size, ctx.device)
    end
    score = _equivariant_diffusion_prediction(
        x, t, params; model, h, seq_dist_flat, mask)
    return center_positions(score)
end

"""
    diffusion_logp_gradient(ctx, params, x_batch; time=_T_MIN_F32)

Return the diffusion model score, interpreted as `grad_x logP_t(x)`, for every
sample in `x_batch`. By default this evaluates real inputs at the minimum
sampled training time rather than at exactly zero diffusion time.
"""
function diffusion_logp_gradient(ctx::NBodyDiffusionContext, params, x_batch;
                                 time::Real=_T_MIN_F32)
    return predict_diffusion_score(ctx, params, x_batch; time)
end

function diffusion_loss(ctx::NBodyDiffusionContext, params, x_batch;
    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    batch_size = size(x_batch, 3)
    x0 = center_positions(x_batch) |> ctx.device
    t = (_T_MIN_F32 .+ (1.0f0 - _T_MIN_F32) .*
         rand(rng, Float32, 1, 1, batch_size)) |> ctx.device
    xt, eps = forward_noise(x0, t; rng, device=ctx.device)

    h, seq_dist_flat, mask = Zygote.ignore() do
        _diffusion_fixed_inputs(model, batch_size, ctx.device)
    end
    score = _equivariant_diffusion_prediction(
        xt, t, params; model, h, seq_dist_flat, mask)
    score = center_positions(score)
    return mean(abs2, @. beta(t) * score + Zygote.dropgrad(eps))
end

function train_diffusion_adam(ctx::NBodyDiffusionContext, params, train_data;
                              epochs::Int=25, batch_size::Int=64,
                              learning_rate::Real=1f-3,
                              log_every::Int=1,
                              checkpoint_callback=nothing,
                              rng::AbstractRNG=Xoshiro(42))
    n_samples = size(train_data, 3)
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

            opt_state, params = Optimisers.update(opt_state, params, grad_params)
            loss_f32 = Float32(loss)
            isfinite(loss_f32) ||
                error("Non-finite diffusion loss at epoch=$epoch batch=$batch_num: $loss_f32")
            epoch_loss += loss_f32 * local_batch_size
            push!(loss_history, loss_f32)
        end
        mean_loss = Float32(epoch_loss / n_samples)
        if log_every > 0 && (epoch == 1 || epoch == epochs || epoch % log_every == 0)
            @info "diffusion_training" epoch loss=mean_loss
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
    f_t = Float32(-(pi / 2) * tan((pi / 2) * Float64(t)))
    g2_t = -2.0f0 * f_t
    g_t = sqrt(g2_t)
    return f_t, g2_t, g_t
end

function euler_maruyama_sample(model, shape; num_steps::Int=1000,
                               rng::AbstractRNG=Random.default_rng(),
                               device=identity)
    x_t = center_positions(randn(rng, Float32, shape...)) |> device
    batch_size = shape[end]
    time_steps = range(1.0f0 - _T_SAMPLE_EPS_F32, _T_SAMPLE_EPS_F32;
                       length=num_steps)
    dt = 1.0f0 / Float32(num_steps)
    sqrt_dt = sqrt(dt)

    for (step, t_raw) in enumerate(time_steps)
        x_t = center_positions(x_t)
        t = Float32(t_raw)
        f_t, g2_t, g_t = _reverse_sde_coefficients(t)
        t_values = fill(t, batch_size) |> device
        score = center_positions(model(x_t, t_values))
        z = step == num_steps ?
            zero.(x_t) :
            (center_positions(randn(rng, eltype(x_t), size(x_t)...)) |> device)
        dx = @. (f_t * x_t - g2_t * score) * dt
        noise = @. g_t * sqrt_dt * z
        x_t = @. x_t - dx + noise
        x_t = center_positions(x_t)
    end

    return x_t
end

function euler_maruyama_sample(ctx::NBodyDiffusionContext, params, shape;
                               num_steps::Int=ctx.sample_steps,
                               rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    batch_size = shape[end]
    h, seq_dist_flat, mask = Zygote.ignore() do
        _diffusion_fixed_inputs(model, batch_size, ctx.device)
    end

    diffusion_score = function (x_t, t_values)
        t_batch = reshape(t_values, 1, 1, batch_size)
        return _equivariant_diffusion_prediction(
            x_t, t_batch, params; model, h, seq_dist_flat, mask)
    end

    return euler_maruyama_sample(diffusion_score, shape;
                                 num_steps, rng, device=ctx.device)
end

function generate_diffusion_samples(ctx::NBodyDiffusionContext, params,
                                    n_samples::Int;
                                    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    shape = (model.dim, model.n_atoms, n_samples)
    return euler_maruyama_sample(ctx, params, shape;
                                 num_steps=ctx.sample_steps, rng)
end
