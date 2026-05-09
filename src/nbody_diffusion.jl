export EquivariantDiffusionModel, NBodyDiffusionContext, DiffusionResult
export build_diffusion_model, init_diffusion_params, diffusion_loss
export train_diffusion_adam, generate_diffusion_samples, predict_diffusion_score
export alpha, beta, forward_noise, euler_maruyama_sample

struct EquivariantDiffusionModel
    dim::Int
    n_atoms::Int
    hidden_dims::Vector{Int}
    edge_net
    edge_state
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
                               hidden_dims=[64, 64, 64])
    hidden = Int.(hidden_dims)
    input_dim = 2 * n_atoms + 2
    edge_net = _dense_chain(input_dim, 1, hidden)
    edge_state = DiffEqFlux.Lux.initialstates(Random.default_rng(), edge_net)
    return EquivariantDiffusionModel(dim, n_atoms, hidden, edge_net, edge_state)
end

function init_diffusion_params(rng::AbstractRNG, model::EquivariantDiffusionModel;
                               device=identity)
    params, _ = DiffEqFlux.Lux.setup(rng, model.edge_net)
    return ComponentArray(params) |> device
end

const _HALF_PI_F32 = Float32(pi / 2)
const _T_MIN_F32 = 1.0f-5

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
    eps = randn(rng, eltype(x0), size(x0)...) |> device
    xt = @. alpha(t_batch) * x0 + beta(t_batch) * eps
    return xt, eps
end

function _diffusion_fixed_inputs(model::EquivariantDiffusionModel, batch_size::Int, device)
    n_atoms = model.n_atoms
    h_fixed = Matrix{Float32}(I, n_atoms, n_atoms)
    h_i_flat = repeat(h_fixed, outer=(1, n_atoms * batch_size)) |> device
    h_j_flat = repeat(h_fixed, inner=(1, n_atoms), outer=(1, batch_size)) |> device
    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device
    return h_i_flat, h_j_flat, mask
end

function _diffusion_edge_scalar(input, params, model::EquivariantDiffusionModel)
    val, _ = model.edge_net(input, params, model.edge_state)
    return val
end

function _equivariant_diffusion_prediction(x, t, params;
                                           model::EquivariantDiffusionModel,
                                           h_i_flat, h_j_flat, mask)
    dim, n_atoms, batch_size = size(x)
    x_i = reshape(x, dim, n_atoms, 1, batch_size)
    x_j = reshape(x, dim, 1, n_atoms, batch_size)
    r_ij = x_i .- x_j
    d_ij = sum(abs2, r_ij; dims=1)
    r_ij_const = Zygote.dropgrad(r_ij)
    d_ij_const = Zygote.dropgrad(d_ij)
    mask_const = Zygote.dropgrad(mask)
    d_ij_flat = reshape(d_ij, 1, n_atoms * n_atoms * batch_size)
    t_flat = repeat(reshape(t, 1, batch_size), inner=(1, n_atoms * n_atoms))
    net_input = Zygote.dropgrad(vcat(h_i_flat, h_j_flat, d_ij_flat, t_flat))

    w_flat = _diffusion_edge_scalar(net_input, params, model)
    w_ij = reshape(w_flat, 1, n_atoms, n_atoms, batch_size)
    weights = (w_ij ./ (d_ij_const .+ one(eltype(x)))) .* mask_const
    pred = sum(r_ij_const .* weights; dims=3) ./ Float32(n_atoms)
    return center_positions(reshape(pred, dim, n_atoms, batch_size))
end

function predict_diffusion_score(ctx::NBodyDiffusionContext, params, x_batch;
                                 time::Real=_T_MIN_F32)
    model = ctx.model
    batch_size = size(x_batch, 3)
    x = x_batch |> ctx.device
    t = fill(Float32(time), 1, 1, batch_size) |> ctx.device
    h_i_flat, h_j_flat, mask = _diffusion_fixed_inputs(model, batch_size, ctx.device)
    return _equivariant_diffusion_prediction(
        x, t, params; model, h_i_flat, h_j_flat, mask)
end

function diffusion_loss(ctx::NBodyDiffusionContext, params, x_batch;
    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    batch_size = size(x_batch, 3)
    x0 = x_batch |> ctx.device
    t = (_T_MIN_F32 .+ (1.0f0 - _T_MIN_F32) .*
         rand(rng, Float32, 1, 1, batch_size)) |> ctx.device
    xt, eps = forward_noise(x0, t; rng, device=ctx.device)

    h_i_flat, h_j_flat, mask = _diffusion_fixed_inputs(model, batch_size, ctx.device)
    score = _equivariant_diffusion_prediction(
        xt, t, params; model, h_i_flat, h_j_flat, mask)
    return mean(abs2, @. beta(t) * score + Zygote.dropgrad(eps))
end

function train_diffusion_adam(ctx::NBodyDiffusionContext, params, train_data;
                              epochs::Int=25, batch_size::Int=64,
                              learning_rate::Real=1f-3,
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
            x_batch = train_data[:, :, batch_idx]
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
        @info "diffusion_training" epoch loss=(epoch_loss / n_samples)
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
                               rng::AbstractRNG=Random.default_rng())
    x_t = randn(rng, Float32, shape...)
    batch_size = shape[end]
    time_steps = range(1.0f0, 1.0f-4; length=num_steps)
    dt = 1.0f0 / Float32(num_steps)
    sqrt_dt = sqrt(dt)

    for (step, t_raw) in enumerate(time_steps)
        t = Float32(t_raw)
        f_t, g2_t, g_t = _reverse_sde_coefficients(t)
        score = model(x_t, fill(t, batch_size))
        z = step == num_steps ? zero.(x_t) : randn(rng, eltype(x_t), size(x_t)...)
        dx = @. (f_t * x_t - g2_t * score) * dt
        noise = @. g_t * sqrt_dt * z
        x_t = @. x_t - dx + noise
    end

    return x_t
end

function euler_maruyama_sample(ctx::NBodyDiffusionContext, params, shape;
                               num_steps::Int=ctx.sample_steps,
                               rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    x_t = randn(rng, Float32, shape...) |> ctx.device
    batch_size = shape[end]
    h_i_flat, h_j_flat, mask = _diffusion_fixed_inputs(model, batch_size, ctx.device)
    time_steps = range(1.0f0, 1.0f-4; length=num_steps)
    dt = 1.0f0 / Float32(num_steps)
    sqrt_dt = sqrt(dt)

    for (step, t_raw) in enumerate(time_steps)
        t = Float32(t_raw)
        f_t, g2_t, g_t = _reverse_sde_coefficients(t)
        t_batch = fill(t, 1, 1, batch_size) |> ctx.device
        score = _equivariant_diffusion_prediction(
            x_t, t_batch, params; model, h_i_flat, h_j_flat, mask)
        z = step == num_steps ?
            zero.(x_t) :
            (randn(rng, eltype(x_t), size(x_t)...) |> ctx.device)
        dx = @. (f_t * x_t - g2_t * score) * dt
        noise = @. g_t * sqrt_dt * z
        x_t = @. x_t - dx + noise
    end

    return x_t
end

function generate_diffusion_samples(ctx::NBodyDiffusionContext, params,
                                    n_samples::Int;
                                    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    shape = (model.dim, model.n_atoms, n_samples)
    return euler_maruyama_sample(ctx, params, shape;
                                 num_steps=ctx.sample_steps, rng)
end
