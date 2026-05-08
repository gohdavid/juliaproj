export EquivariantDiffusionModel, NBodyDiffusionContext, DiffusionResult
export build_diffusion_model, init_diffusion_params, diffusion_loss
export train_diffusion_adam, generate_diffusion_samples

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
    n_steps::Int
    betas::Vector{Float32}
    alphas::Vector{Float32}
    alpha_bars::Vector{Float32}
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

function _diffusion_schedule(n_steps::Int, beta_start::Real, beta_end::Real)
    betas = Float32.(range(Float32(beta_start), Float32(beta_end); length=n_steps))
    alphas = 1.0f0 .- betas
    alpha_bars = accumulate(*, alphas)
    return betas, alphas, alpha_bars
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

function diffusion_loss(ctx::NBodyDiffusionContext, params, x_batch;
                        rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    dim, n_atoms, batch_size = size(x_batch)
    x0 = x_batch |> ctx.device
    eps = center_positions(randn(rng, Float32, dim, n_atoms, batch_size)) |> ctx.device
    t_idx = rand(rng, 1:ctx.n_steps, batch_size)
    alpha_bar = reshape(ctx.alpha_bars[t_idx], 1, 1, batch_size) |> ctx.device
    xt = sqrt.(alpha_bar) .* x0 .+ sqrt.(1.0f0 .- alpha_bar) .* eps
    t = reshape(Float32.(t_idx) ./ Float32(ctx.n_steps), 1, 1, batch_size) |>
        ctx.device

    h_i_flat, h_j_flat, mask = _diffusion_fixed_inputs(model, batch_size, ctx.device)
    pred = _equivariant_diffusion_prediction(
        xt, t, params; model, h_i_flat, h_j_flat, mask)
    return sum(abs2, pred .- Zygote.dropgrad(eps)) / Float32(length(pred))
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

function generate_diffusion_samples(ctx::NBodyDiffusionContext, params,
                                    n_samples::Int;
                                    rng::AbstractRNG=Random.default_rng())
    model = ctx.model
    x = center_positions(randn(rng, Float32, model.dim, model.n_atoms, n_samples)) |>
        ctx.device
    h_i_flat, h_j_flat, mask = _diffusion_fixed_inputs(model, n_samples, ctx.device)

    for step in ctx.n_steps:-1:1
        beta = ctx.betas[step]
        alpha = ctx.alphas[step]
        alpha_bar = ctx.alpha_bars[step]
        t = fill(Float32(step) / Float32(ctx.n_steps), 1, 1, n_samples) |>
            ctx.device
        eps_pred = _equivariant_diffusion_prediction(
            x, t, params; model, h_i_flat, h_j_flat, mask)
        x = (x .- (beta / sqrt(1.0f0 - alpha_bar)) .* eps_pred) ./ sqrt(alpha)

        if step > 1
            prev_alpha_bar = ctx.alpha_bars[step - 1]
            posterior_var = beta * (1.0f0 - prev_alpha_bar) / (1.0f0 - alpha_bar)
            z = center_positions(randn(rng, Float32, model.dim, model.n_atoms, n_samples)) |>
                ctx.device
            x = x .+ sqrt(posterior_var) .* z
        end
        x = center_positions(x)
    end

    return x
end
