export MLPVectorField, EGNNVectorField, NBodyCNFContext
export build_vector_field, init_cnf_params, cnf_logp, cnf_logp_gradient
export cnf_nll, train_cnf_adam, generate_cnf_samples
export log_standard_normal

abstract type AbstractNBodyVectorField end

struct MLPVectorField <: AbstractNBodyVectorField
    dim::Int
    n_atoms::Int
    hidden_dims::Vector{Int}
    net
end

struct EGNNVectorField <: AbstractNBodyVectorField
    dim::Int
    n_atoms::Int
    node_embedding_dim::Int
    hidden_dims::Vector{Int}
    edge_net
end

struct NBodyCNFContext
    field::AbstractNBodyVectorField
    device
    tspan::Tuple{Float32,Float32}
    solver
    abstol::Float32
    reltol::Float32
    trace_mode::Symbol
end

function _dense_chain(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int}; final_activation=identity)
    dims = [input_dim; hidden_dims; output_dim]
    layers = Any[]
    for i in 1:(length(dims) - 1)
        activation = i == length(dims) - 1 ? final_activation : tanh
        push!(layers, DiffEqFlux.Dense(dims[i] => dims[i + 1], activation))
    end
    return DiffEqFlux.Chain(layers...)
end

function build_vector_field(kind::Symbol; dim::Int=2, n_atoms::Int=10,
                            hidden_dims=[64, 64, 64],
                            node_embedding_dim::Int=min(16, n_atoms))
    hidden = Int.(hidden_dims)
    if kind == :mlp
        input_dim = dim * n_atoms + 1
        output_dim = dim * n_atoms
        return MLPVectorField(dim, n_atoms, hidden,
                              _dense_chain(input_dim, output_dim, hidden))
    elseif kind == :egnn
        input_dim = 2 * node_embedding_dim + 3
        return EGNNVectorField(dim, n_atoms, node_embedding_dim, hidden,
                               _dense_chain(input_dim, 1, hidden))
    else
        error("Unknown vector field kind: $kind")
    end
end

function init_cnf_params(rng::AbstractRNG, field::MLPVectorField; device=identity)
    params, _ = DiffEqFlux.Lux.setup(rng, field.net)
    return ComponentArray(params) |> device
end

function init_cnf_params(rng::AbstractRNG, field::EGNNVectorField; device=identity)
    edge_params, _ = DiffEqFlux.Lux.setup(rng, field.edge_net)
    node_embed = 0.1f0 .* randn(rng, Float32, field.node_embedding_dim, field.n_atoms)
    return ComponentArray(edge_net=edge_params, node_embed=node_embed) |> device
end

function forward_mlp_generic_jvp(input, params, jvp_probe)
    val = input
    deriv = jvp_probe
    layer_keys = collect(keys(params))
    last_key = last(layer_keys)

    for key in layer_keys
        layer_params = params[key]
        w = layer_params.weight
        b = layer_params.bias
        pre_act = w * val .+ b
        pre_deriv = w * deriv

        if key == last_key
            val = pre_act
            deriv = pre_deriv
        else
            val = tanh.(pre_act)
            deriv = (one(eltype(input)) .- val .^ 2) .* pre_deriv
        end
    end
    return val, deriv
end

function log_standard_normal(x)
    dim, n_atoms, batch_size = size(x)
    const_term = 0.5f0 * Float32(dim * n_atoms) * log(2.0f0 * Float32(pi))
    sq_norm = sum(abs2, x; dims=(1, 2))
    return -0.5f0 .* reshape(sq_norm, 1, batch_size) .- const_term
end

function _rademacher(rng::AbstractRNG, dims...)
    return ifelse.(rand(rng, Float32, dims...) .< 0.5f0, -1.0f0, 1.0f0)
end

function _mlp_hutchinson_dynamics(u, params, t; field::MLPVectorField, probe_in, probe_out)
    x = u.x
    dim, n_atoms, batch_size = size(x)
    x_flat = reshape(x, dim * n_atoms, batch_size)
    t_flat = x_flat[1:1, :] .* zero(eltype(x)) .+ eltype(x)(t)
    net_input = vcat(x_flat, t_flat)
    dx_flat, jvp_flat = forward_mlp_generic_jvp(net_input, params, probe_in)
    dx = reshape(dx_flat, dim, n_atoms, batch_size)
    dlogp = -sum(probe_out .* jvp_flat; dims=1)
    return ComponentArray(x=dx, logp=dlogp)
end

function _mlp_exact_dynamics(u, params, t; field::MLPVectorField)
    x = u.x
    dim, n_atoms, batch_size = size(x)
    state_dim = dim * n_atoms
    x_flat = reshape(x, state_dim, batch_size)
    t_flat = x_flat[1:1, :] .* zero(eltype(x)) .+ eltype(x)(t)
    net_input = vcat(x_flat, t_flat)
    trace_accum = x_flat[1:1, :] .* zero(eltype(x))
    local dx_flat

    for i in 1:state_dim
        probe = net_input .* zero(eltype(x))
        probe[i, :] .= one(eltype(x))
        val, jvp = forward_mlp_generic_jvp(net_input, params, probe)
        i == 1 && (dx_flat = val)
        trace_accum = trace_accum .+ jvp[i:i, :]
    end

    return ComponentArray(x=reshape(dx_flat, dim, n_atoms, batch_size),
                          logp=-trace_accum)
end

function _egnn_fixed_inputs(field::EGNNVectorField, batch_size::Int, device)
    n_atoms = field.n_atoms

    edge_i = repeat(collect(1:n_atoms), outer=n_atoms * batch_size) |> device
    edge_j = repeat(collect(1:n_atoms), inner=n_atoms, outer=batch_size) |> device

    seq_dist_fixed = Float32[i - j for i in 1:n_atoms, j in 1:n_atoms]
    seq_dist_fixed = reshape(seq_dist_fixed, 1, n_atoms * n_atoms)
    seq_dist_flat = repeat(seq_dist_fixed, outer=(1, batch_size)) |> device

    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device

    node_embedding_dim = field.node_embedding_dim
    jvp_probe = vcat(zeros(Float32, 2*node_embedding_dim+1, 1),
                     ones(Float32, 1, 1),
                     zeros(Float32, 1, 1)) |> device

    return edge_i, edge_j, seq_dist_flat, mask, jvp_probe
end

function center_of_mass(x)
    return mean(x; dims=2)
end

function center_positions(x)
    return x .- center_of_mass(x)
end

function _egnn_exact_dynamics(u, params, t; field::EGNNVectorField,
                              edge_i, edge_j, seq_dist_flat, mask, jvp_probe)
    x = u.x
    dim, n_atoms, batch_size = size(x)
    x_i = reshape(x, dim, n_atoms, 1, batch_size)
    x_j = reshape(x, dim, 1, n_atoms, batch_size)
    r_ij = x_i .- x_j
    d_ij = sum(abs2, r_ij; dims=1)
    d_ij_flat = reshape(d_ij, 1, n_atoms * n_atoms * batch_size)
    t_flat = d_ij_flat .* zero(eltype(x)) .+ eltype(x)(t)

    h_i_flat = params.node_embed[:, edge_i]
    h_j_flat = params.node_embed[:, edge_j]
    net_input = vcat(h_i_flat, h_j_flat, seq_dist_flat, d_ij_flat, t_flat)
    m_flat, dm_flat = forward_mlp_generic_jvp(net_input, params.edge_net, jvp_probe)
    m_ij = reshape(m_flat, 1, n_atoms, n_atoms, batch_size)
    dm_ij = reshape(dm_flat, 1, n_atoms, n_atoms, batch_size)

    denom = d_ij .+ one(eltype(x))
    w_ij = (m_ij ./ denom) .* mask
    dw_ij = ((dm_ij .* denom .- m_ij) ./ denom .^ 2) .* mask

    dx_terms = w_ij .* r_ij
    dx = sum(dx_terms; dims=3) ./ Float32(n_atoms)
    dx = reshape(dx, dim, n_atoms, batch_size)

    trace_terms = Float32(dim) .* w_ij .+ 2.0f0 .* d_ij .* dw_ij
    trace_sum = sum(trace_terms; dims=(1, 2, 3)) ./ Float32(n_atoms)
    dlogp = -reshape(trace_sum, 1, batch_size)
    return ComponentArray(x=dx, logp=dlogp)
end

function _cnf_problem(ctx::NBodyCNFContext, params, batch_size::Int, tspan; rng)
    field = ctx.field
    if field isa MLPVectorField
        if ctx.trace_mode == :exact
            ode_func = (u, p, t) -> _mlp_exact_dynamics(u, p, t; field)
        else
            state_dim = field.dim * field.n_atoms
            probe = _rademacher(rng, state_dim, batch_size)
            probe_in = vcat(probe, zeros(Float32, 1, batch_size)) |> ctx.device
            probe_out = probe |> ctx.device
            ode_func = (u, p, t) -> _mlp_hutchinson_dynamics(
                u, p, t; field, probe_in, probe_out)
        end
    elseif field isa EGNNVectorField
        edge_i, edge_j, seq_dist_flat, mask, jvp_probe = Zygote.ignore() do
            _egnn_fixed_inputs(field, batch_size, ctx.device)
        end
        ode_func = (u, p, t) -> _egnn_exact_dynamics(
            u, p, t; field, edge_i, edge_j, seq_dist_flat, mask, jvp_probe)
    else
        error("Unsupported vector field $(typeof(field))")
    end

    u0 = ComponentArray(
        x=zeros(Float32, field.dim, field.n_atoms, batch_size),
        logp=zeros(Float32, 1, batch_size)) |> ctx.device
    return ODEProblem(ode_func, u0, tspan, params)
end

function _cnf_logp_device(ctx::NBodyCNFContext, params, x_dev; rng)
    batch_size = size(x_dev, 3)
    u0 = ComponentArray(x=x_dev, logp=zeros(Float32, 1, batch_size) |> ctx.device)
    prob = _cnf_problem(ctx, params, batch_size, ctx.tspan; rng)
    sol = solve(remake(prob; u0, p=params), ctx.solver;
                abstol=ctx.abstol, reltol=ctx.reltol,
                save_everystep=false, save_start=false, save_end=true,
                verbose=false)
    x_noise = sol.u[end].x
    delta_logp = sol.u[end].logp
    return log_standard_normal(x_noise) .- delta_logp
end

function cnf_logp(ctx::NBodyCNFContext, params, x_batch; rng=Random.default_rng())
    x_dev = x_batch |> ctx.device
    return _cnf_logp_device(ctx, params, x_dev; rng)
end

function cnf_nll(ctx::NBodyCNFContext, params, x_batch; rng=Random.default_rng())
    batch_size = size(x_batch, 3)
    log_pdata = cnf_logp(ctx, params, x_batch; rng)
    return sum(-log_pdata) / batch_size
end

"""
    cnf_logp_gradient(ctx, params, x_batch; rng=Random.default_rng())

Return `grad_x logP(x)` for every sample in `x_batch` under the trained CNF
parameters. The output has the same `(dim, n_atoms, batch)` layout as
`x_batch`. When `ctx.trace_mode != :exact`, the gradient is taken through the
same Hutchinson trace estimate used by the likelihood calculation.
"""
function cnf_logp_gradient(ctx::NBodyCNFContext, params, x_batch;
                           rng=Random.default_rng())
    x_dev = x_batch |> ctx.device
    grad = Zygote.gradient(x -> sum(_cnf_logp_device(ctx, params, x; rng)), x_dev)[1]
    grad === nothing && error("Zygote returned no coordinate gradient.")
    return center_positions(grad)
end

function train_cnf_adam(ctx::NBodyCNFContext, params, train_data;
                        epochs::Int=25, batch_size::Int=64,
                        learning_rate::Real=1f-3,
                        log_every::Int=1,
                        checkpoint_callback=nothing,
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

            loss, grads = Zygote.withgradient(p -> cnf_nll(ctx, p, x_batch; rng), params)
            grad_params = grads[1]
            grad_params === nothing && error("Zygote returned no parameter gradient.")

            opt_state, params = Optimisers.update(opt_state, params, grad_params)
            step += 1
            loss_f32 = Float32(loss)
            isfinite(loss_f32) ||
                error("Non-finite loss at epoch=$epoch batch=$batch_num: $loss_f32")
            epoch_loss += loss_f32 * local_batch_size
            push!(loss_history, loss_f32)
        end
        mean_loss = Float32(epoch_loss / n_samples)
        if log_every > 0 && (epoch == 1 || epoch == epochs || epoch % log_every == 0)
            @info "training" epoch step total_steps steps_per_epoch loss=mean_loss
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

function generate_cnf_samples(ctx::NBodyCNFContext, params, n_samples::Int;
                              rng::AbstractRNG=Random.default_rng())
    field = ctx.field
    x0 = randn(rng, Float32, field.dim, field.n_atoms, n_samples) |> ctx.device
    u0 = ComponentArray(x=x0, logp=zeros(Float32, 1, n_samples) |> ctx.device)
    reverse_tspan = (ctx.tspan[2], ctx.tspan[1])
    prob = _cnf_problem(ctx, params, n_samples, reverse_tspan; rng)
    sol = solve(remake(prob; u0, p=params), ctx.solver;
                abstol=ctx.abstol, reltol=ctx.reltol,
                save_everystep=false, save_start=false, save_end=true,
                verbose=false)
    return center_positions(sol.u[end].x)
end
