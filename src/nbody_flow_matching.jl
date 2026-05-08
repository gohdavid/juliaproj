export EquivariantFMVectorField, NBodyFlowMatchingContext, FlowMatchingResult
export build_fm_vector_field, init_fm_params, flow_matching_loss
export train_flow_matching_adam, generate_flow_matching_samples
export center_positions, center_of_mass, pairwise_distance_mae

struct EquivariantFMVectorField
    dim::Int
    n_atoms::Int
    hidden_dims::Vector{Int}
    edge_net
    edge_state
end

struct NBodyFlowMatchingContext
    field::EquivariantFMVectorField
    device
    sample_steps::Int
end

struct FlowMatchingResult
    config::Dict{String,Any}
    train_data
    params
    losses::Vector{Float32}
    samples
    output_dir::String
    metrics::Dict{String,Float32}
end

function build_fm_vector_field(; dim::Int=2, n_atoms::Int=10,
                               hidden_dims=[64, 64, 64])
    hidden = Int.(hidden_dims)
    input_dim = 2 * n_atoms + 2
    edge_net = _dense_chain(input_dim, 1, hidden)
    edge_state = DiffEqFlux.Lux.initialstates(Random.default_rng(), edge_net)
    return EquivariantFMVectorField(dim, n_atoms, hidden, edge_net, edge_state)
end

function init_fm_params(rng::AbstractRNG, field::EquivariantFMVectorField; device=identity)
    params, _ = DiffEqFlux.Lux.setup(rng, field.edge_net)
    return ComponentArray(params) |> device
end

function center_of_mass(x)
    return mean(x; dims=2)
end

function center_positions(x)
    return x .- center_of_mass(x)
end

function _fm_fixed_inputs(field::EquivariantFMVectorField, batch_size::Int, device)
    n_atoms = field.n_atoms
    h_fixed = Matrix{Float32}(I, n_atoms, n_atoms)
    h_i_flat = repeat(h_fixed, outer=(1, n_atoms * batch_size)) |> device
    h_j_flat = repeat(h_fixed, inner=(1, n_atoms), outer=(1, batch_size)) |> device
    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device
    return h_i_flat, h_j_flat, mask
end

function _edge_net_scalar(input, params, field::EquivariantFMVectorField)
    val, _ = field.edge_net(input, params, field.edge_state)
    return val
end

function _equivariant_fm_velocity(x, t, params; field::EquivariantFMVectorField,
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

    w_flat = _edge_net_scalar(net_input, params, field)
    w_ij = reshape(w_flat, 1, n_atoms, n_atoms, batch_size)
    weights = (w_ij ./ (d_ij_const .+ one(eltype(x)))) .* mask_const
    velocity = sum(r_ij_const .* weights; dims=3) ./ Float32(n_atoms)
    return reshape(velocity, dim, n_atoms, batch_size)
end

function flow_matching_loss(ctx::NBodyFlowMatchingContext, params, x_batch;
                            rng::AbstractRNG=Random.default_rng())
    field = ctx.field
    dim, n_atoms, batch_size = size(x_batch)
    x1 = x_batch |> ctx.device
    x0 = center_positions(randn(rng, Float32, dim, n_atoms, batch_size)) |> ctx.device
    t = rand(rng, Float32, 1, 1, batch_size) |> ctx.device
    xt = (1.0f0 .- t) .* x0 .+ t .* x1
    target = Zygote.dropgrad(x1 .- x0)
    h_i_flat, h_j_flat, mask = _fm_fixed_inputs(field, batch_size, ctx.device)
    pred = _equivariant_fm_velocity(
        xt, t, params; field, h_i_flat, h_j_flat, mask)
    return sum(abs2, pred .- target) / Float32(length(pred))
end

function train_flow_matching_adam(ctx::NBodyFlowMatchingContext, params, train_data;
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
                p -> flow_matching_loss(ctx, p, x_batch; rng), params)
            grad_params = grads[1]
            grad_params === nothing && error("Zygote returned no parameter gradient.")

            opt_state, params = Optimisers.update(opt_state, params, grad_params)
            loss_f32 = Float32(loss)
            isfinite(loss_f32) ||
                error("Non-finite flow-matching loss at epoch=$epoch batch=$batch_num: $loss_f32")
            epoch_loss += loss_f32 * local_batch_size
            push!(loss_history, loss_f32)
        end
        @info "flow_matching_training" epoch loss=(epoch_loss / n_samples)
    end

    return params, opt_state, loss_history
end

function generate_flow_matching_samples(ctx::NBodyFlowMatchingContext, params,
                                        n_samples::Int;
                                        rng::AbstractRNG=Random.default_rng())
    field = ctx.field
    x = center_positions(randn(rng, Float32, field.dim, field.n_atoms, n_samples)) |>
        ctx.device
    h_i_flat, h_j_flat, mask = _fm_fixed_inputs(field, n_samples, ctx.device)
    dt = 1.0f0 / Float32(ctx.sample_steps)

    for step in 0:(ctx.sample_steps - 1)
        t_scalar = Float32(step) * dt
        t = fill(t_scalar, 1, 1, n_samples) |> ctx.device
        v = _equivariant_fm_velocity(
            x, t, params; field, h_i_flat, h_j_flat, mask)
        x = x .+ dt .* v
    end

    return x
end

function pairwise_distance_mae(reference, generated)
    ref_dist = pairwise_distances(reference)
    gen_dist = pairwise_distances(generated)
    ref_mean = mean(ref_dist; dims=3)
    gen_mean = mean(gen_dist; dims=3)
    return Float32(mean(abs.(ref_mean .- gen_mean)))
end
