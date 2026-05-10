export MLPVectorField, EGNNVectorField, NBodyCNFContext
export build_vector_field, init_cnf_params, cnf_logp, cnf_logp_gradient
export cnf_nll, train_cnf_adam, generate_cnf_samples
export identity_data_normalizer, fit_data_normalizer, apply_data_normalizer
export invert_data_normalizer, normalized_cnf_logp, normalized_cnf_logp_gradient
export generate_normalized_cnf_samples
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
    node_net
    n_layers::Int
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

mutable struct CNFSolveStats
    rhs_calls::Base.RefValue{Int}
    forward_rhs_calls::Base.RefValue{Int}
    forward_nf::Base.RefValue{Int}
    forward_naccept::Base.RefValue{Int}
    forward_nreject::Base.RefValue{Int}
end

CNFSolveStats() = CNFSolveStats(Ref(0), Ref(0), Ref(0), Ref(0), Ref(0))

function _reset!(stats::CNFSolveStats)
    stats.rhs_calls[] = 0
    stats.forward_rhs_calls[] = 0
    stats.forward_nf[] = 0
    stats.forward_naccept[] = 0
    stats.forward_nreject[] = 0
    return stats
end

function _count_rhs!(stats)
    stats === nothing && return nothing
    Zygote.ignore() do
        stats.rhs_calls[] += 1
    end
    return nothing
end

function _record_forward_stats!(stats, sol)
    stats === nothing && return nothing
    Zygote.ignore() do
        stats.forward_rhs_calls[] = stats.rhs_calls[]
        stats.forward_nf[] = Int(sol.destats.nf)
        stats.forward_naccept[] = Int(sol.destats.naccept)
        stats.forward_nreject[] = Int(sol.destats.nreject)
    end
    return nothing
end

_backprop_rhs_calls(stats::CNFSolveStats) =
    max(stats.rhs_calls[] - stats.forward_rhs_calls[], 0)

function _gradient_l2_norm(grads)
    grads === nothing && return Float32(NaN)
    return Float32(sqrt(Float64(sum(abs2, grads))))
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
                            node_embedding_dim::Int=min(16, n_atoms),
                            n_layers::Int=0)
    hidden = Int.(hidden_dims)
    if kind == :mlp
        input_dim = dim * n_atoms + 1
        output_dim = dim * n_atoms
        return MLPVectorField(dim, n_atoms, hidden,
                              _dense_chain(input_dim, output_dim, hidden))
    elseif kind == :egnn
        if n_layers <= 0
            input_dim = 2 * n_atoms + 3
            return EGNNVectorField(dim, n_atoms, n_atoms, hidden,
                                   _dense_chain(input_dim, dim, hidden),
                                   nothing,
                                   0)
        end
        edge_input_dim = 2 * node_embedding_dim + 3
        edge_output_dim = node_embedding_dim + 1
        node_input_dim = 2 * node_embedding_dim + 1
        return EGNNVectorField(dim, n_atoms, node_embedding_dim, hidden,
                               _dense_chain(edge_input_dim, edge_output_dim, hidden),
                               _dense_chain(node_input_dim, node_embedding_dim, hidden),
                               n_layers)
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
    if field.n_layers <= 0
        return ComponentArray(edge_net=edge_params) |> device
    end
    node_params, _ = DiffEqFlux.Lux.setup(rng, field.node_net)
    node_embed = 0.1f0 .* randn(rng, Float32, field.node_embedding_dim, field.n_atoms)
    return ComponentArray(edge_net=edge_params, node_net=node_params,
                          node_embed=node_embed) |> device
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

function forward_mlp_generic(input, params)
    val = input
    layer_keys = collect(keys(params))
    last_key = last(layer_keys)

    for key in layer_keys
        layer_params = params[key]
        pre_act = layer_params.weight * val .+ layer_params.bias
        val = key == last_key ? pre_act : tanh.(pre_act)
    end
    return val
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

    seq_dist_fixed = Float32[i - j for i in 1:n_atoms, j in 1:n_atoms]
    seq_dist_fixed = reshape(seq_dist_fixed, 1, n_atoms * n_atoms)
    seq_dist_flat = repeat(seq_dist_fixed, outer=(1, batch_size)) |> device

    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device

    return seq_dist_flat, mask
end

function _egnn_legacy_fixed_inputs(field::EGNNVectorField, batch_size::Int, device)
    n_atoms = field.n_atoms

    seq_dist_fixed = Float32[i - j for i in 1:n_atoms, j in 1:n_atoms]
    seq_dist_fixed = reshape(seq_dist_fixed, 1, n_atoms * n_atoms)
    seq_dist_flat = repeat(seq_dist_fixed, outer=(1, batch_size)) |> device

    mask = reshape(1.0f0 .- Matrix{Float32}(I, n_atoms, n_atoms),
                   1, n_atoms, n_atoms, 1) |> device

    h_fixed = Matrix{Float32}(I, n_atoms, n_atoms)
    h_i_flat = repeat(h_fixed, outer=(1, n_atoms * batch_size)) |> device
    h_j_flat = repeat(h_fixed, inner=(1, n_atoms), outer=(1, batch_size)) |> device

    jvp_probe = vcat(zeros(Float32, 2*n_atoms+1, 1),
                     ones(Float32, 1, 1),
                     zeros(Float32, 1, 1)) |> device

    return h_i_flat, h_j_flat, seq_dist_flat, mask, jvp_probe
end

function center_of_mass(x)
    return mean(x; dims=2)
end

function center_positions(x)
    return x .- center_of_mass(x)
end

function identity_data_normalizer()
    return Dict{String,Any}(
        "enabled" => false,
        "mode" => "identity",
        "mean" => Float32[0.0],
        "scale" => Float32[1.0],
        "eps" => Float32(0.0),
        "log_abs_det" => Float32(0.0),
    )
end

function _normalizer_enabled(normalizer)
    normalizer isa AbstractDict || return false
    return Bool(get(normalizer, "enabled", false))
end

function _normalizer_stat_shape(mode::AbstractString, x)
    dim, n_atoms, _ = size(x)
    if mode == "scalar"
        return (1, 1, 1)
    elseif mode == "per_dim"
        return (dim, 1, 1)
    elseif mode == "per_feature"
        return (dim, n_atoms, 1)
    end
    error("Unsupported data.normalization_mode: $mode")
end

function _normalizer_log_abs_det(scale, dim::Int, n_atoms::Int)
    if length(scale) == 1
        return Float32(dim * n_atoms * log(Float32(scale[1])))
    elseif size(scale, 2) == 1
        return Float32(n_atoms * sum(log.(Float32.(scale))))
    end
    return Float32(sum(log.(Float32.(scale))))
end

function fit_data_normalizer(x; enabled::Bool=false,
                             mode::AbstractString="scalar",
                             eps::Real=1.0f-6)
    enabled || return identity_data_normalizer()
    dim, n_atoms, _ = size(x)
    shape = _normalizer_stat_shape(mode, x)
    reduce_dims = mode == "scalar" ? (1, 2, 3) :
                  mode == "per_dim" ? (2, 3) :
                  (3,)
    mean_x = Float32.(mean(x; dims=reduce_dims))
    std_x = Float32.(sqrt.(mean(abs2, x .- mean_x; dims=reduce_dims)))
    scale = max.(std_x, Float32(eps))
    mean_x = reshape(mean_x, shape)
    scale = reshape(scale, shape)
    return Dict{String,Any}(
        "enabled" => true,
        "mode" => String(mode),
        "mean" => Array{Float32}(mean_x),
        "scale" => Array{Float32}(scale),
        "eps" => Float32(eps),
        "log_abs_det" => _normalizer_log_abs_det(scale, dim, n_atoms),
    )
end

function apply_data_normalizer(x, normalizer)
    _normalizer_enabled(normalizer) || return x
    mean_x = get(normalizer, "mean", Float32[0.0])
    scale = get(normalizer, "scale", Float32[1.0])
    return (x .- mean_x) ./ scale
end

function invert_data_normalizer(x, normalizer)
    _normalizer_enabled(normalizer) || return x
    mean_x = get(normalizer, "mean", Float32[0.0])
    scale = get(normalizer, "scale", Float32[1.0])
    return x .* scale .+ mean_x
end

function _normalizer_log_abs_det(normalizer, x)
    _normalizer_enabled(normalizer) || return Float32(0.0)
    if haskey(normalizer, "log_abs_det")
        return Float32(normalizer["log_abs_det"])
    end
    return _normalizer_log_abs_det(normalizer["scale"], size(x, 1), size(x, 2))
end

function normalized_cnf_logp(ctx::NBodyCNFContext, params, x_batch, normalizer;
                             rng=Random.default_rng(), stats=nothing)
    x_norm = center_positions(apply_data_normalizer(x_batch, normalizer))
    logp_norm = cnf_logp(ctx, params, x_norm; rng, stats)
    return logp_norm .- _normalizer_log_abs_det(normalizer, x_batch)
end

function normalized_cnf_logp_gradient(ctx::NBodyCNFContext, params, x_batch,
                                      normalizer; rng=Random.default_rng())
    x_norm = center_positions(apply_data_normalizer(x_batch, normalizer))
    grad_norm = cnf_logp_gradient(ctx, params, x_norm; rng)
    _normalizer_enabled(normalizer) || return grad_norm
    return grad_norm ./ normalizer["scale"]
end

function _egnn_velocity(field::EGNNVectorField, params, x, t;
                        seq_dist_flat, mask)
    dim, n_atoms, batch_size = size(x)
    h0 = params.node_embed
    batch_zeros = x[1:1, 1:1, :] .* zero(eltype(x))
    h = reshape(h0, field.node_embedding_dim, n_atoms, 1) .+ batch_zeros
    dx_total = x .* zero(eltype(x))

    for _ in 1:field.n_layers
        x_i = reshape(x, dim, n_atoms, 1, batch_size)
        x_j = reshape(x, dim, 1, n_atoms, batch_size)
        r_ij = x_i .- x_j
        d_ij = sum(abs2, r_ij; dims=1)
        d_ij_flat = reshape(d_ij, 1, n_atoms * n_atoms * batch_size)
        t_flat = d_ij_flat .* zero(eltype(x)) .+ eltype(x)(t)
        edge_zeros = d_ij .* zero(eltype(x))

        h_i = reshape(h, field.node_embedding_dim, n_atoms, 1, batch_size)
        h_j = reshape(h, field.node_embedding_dim, 1, n_atoms, batch_size)
        h_i_flat = reshape(h_i .+ edge_zeros,
                           field.node_embedding_dim, n_atoms * n_atoms * batch_size)
        h_j_flat = reshape(h_j .+ edge_zeros,
                           field.node_embedding_dim, n_atoms * n_atoms * batch_size)

        edge_input = vcat(h_i_flat, h_j_flat, seq_dist_flat, d_ij_flat, t_flat)
        edge_out = forward_mlp_generic(edge_input, params.edge_net)
        coord_weight = reshape(edge_out[1:1, :], 1, n_atoms, n_atoms, batch_size)
        messages = reshape(edge_out[2:end, :], field.node_embedding_dim, n_atoms, n_atoms, batch_size)

        layer_dx = sum((coord_weight .* mask) .* r_ij; dims=3) ./ Float32(n_atoms)
        layer_dx = reshape(layer_dx, dim, n_atoms, batch_size)
        dx_total = dx_total .+ layer_dx

        node_agg = reshape(sum(messages .* mask; dims=3) ./ Float32(n_atoms),
                           field.node_embedding_dim, n_atoms, batch_size)
        h_flat = reshape(h, field.node_embedding_dim, n_atoms * batch_size)
        node_agg_flat = reshape(node_agg, field.node_embedding_dim, n_atoms * batch_size)
        node_t = h_flat[1:1, :] .* zero(eltype(x)) .+ eltype(x)(t)
        node_input = vcat(h_flat, node_agg_flat, node_t)
        node_delta = forward_mlp_generic(node_input, params.node_net)
        h = h .+ reshape(node_delta, field.node_embedding_dim, n_atoms, batch_size)
    end

    return dx_total ./ Float32(field.n_layers)
end

function _zygote_divergence(f, x_flat)
    trace = zero(eltype(x_flat))
    for i in eachindex(x_flat)
        grad_i = Zygote.gradient(z -> f(z)[i], x_flat)[1]
        grad_i === nothing && error("Zygote returned no coordinate gradient while computing divergence.")
        trace = trace + grad_i[i]
    end
    return trace
end

function _egnn_exact_trace_dynamics(u, params, t; field::EGNNVectorField,
                                    seq_dist_flat, mask)
    x = u.x
    dim, n_atoms, batch_size = size(x)
    dx = _egnn_velocity(field, params, x, t; seq_dist_flat, mask)
    traces = [
        _zygote_divergence(reshape(x[:, :, b], dim * n_atoms)) do x_flat
            x_single = reshape(x_flat, dim, n_atoms, 1)
            seq_single = @view(seq_dist_flat[:, ((b - 1) * n_atoms * n_atoms + 1):(b * n_atoms * n_atoms)])
            reshape(_egnn_velocity(field, params, x_single, t;
                                   seq_dist_flat=seq_single, mask), dim * n_atoms)
        end
    for b in 1:batch_size
    ]
    dlogp = -reshape(collect(traces), 1, batch_size)
    return ComponentArray(x=dx, logp=dlogp)
end

function _egnn_hutchinson_trace_dynamics(u, params, t; field::EGNNVectorField,
                                         seq_dist_flat, mask, probe)
    x = u.x
    dx = _egnn_velocity(field, params, x, t; seq_dist_flat, mask)
    jtv = Zygote.gradient(z -> sum(_egnn_velocity(field, params, z, t;
                                                  seq_dist_flat, mask) .* probe), x)[1]
    jtv === nothing && error("Zygote returned no coordinate gradient while computing Hutchinson trace.")
    trace_est = sum(probe .* jtv; dims=(1, 2))
    dlogp = -reshape(trace_est, 1, size(x, 3))
    return ComponentArray(x=dx, logp=dlogp)
end

function _egnn_fd_hutchinson_trace_dynamics(u, params, t; field::EGNNVectorField,
                                            seq_dist_flat, mask, probe,
                                            epsilon::Float32=1.0f-3)
    x = u.x
    dx = _egnn_velocity(field, params, x, t; seq_dist_flat, mask)
    eps = eltype(x)(epsilon)
    f_plus = _egnn_velocity(field, params, x .+ eps .* probe, t; seq_dist_flat, mask)
    f_minus = _egnn_velocity(field, params, x .- eps .* probe, t; seq_dist_flat, mask)
    jv = (f_plus .- f_minus) ./ (2eps)
    trace_est = sum(probe .* jv; dims=(1, 2))
    dlogp = -reshape(trace_est, 1, size(x, 3))
    return ComponentArray(x=dx, logp=dlogp)
end

function _egnn_legacy_exact_dynamics(u, params, t; field::EGNNVectorField,
                                     h_i_flat, h_j_flat, seq_dist_flat, mask, jvp_probe)
    x = u.x
    dim, n_atoms, batch_size = size(x)
    x_i = reshape(x, dim, n_atoms, 1, batch_size)
    x_j = reshape(x, dim, 1, n_atoms, batch_size)
    r_ij = x_i .- x_j
    d_ij = sum(abs2, r_ij; dims=1)
    d_ij_flat = reshape(d_ij, 1, n_atoms * n_atoms * batch_size)
    t_flat = d_ij_flat .* zero(eltype(x)) .+ eltype(x)(t)

    net_input = vcat(h_i_flat, h_j_flat, seq_dist_flat, d_ij_flat, t_flat)
    m_flat, dm_flat = forward_mlp_generic_jvp(net_input, params.edge_net, jvp_probe)
    m_ij = reshape(m_flat, dim, n_atoms, n_atoms, batch_size)
    dm_ij = reshape(dm_flat, dim, n_atoms, n_atoms, batch_size)

    denom = d_ij .+ one(eltype(x))
    w_ij = (m_ij ./ denom) .* mask
    dw_ij = ((dm_ij .* denom .- m_ij) ./ denom .^ 2) .* mask

    dx_terms = w_ij .* r_ij
    dx = sum(dx_terms; dims=3) ./ Float32(n_atoms)
    dx = reshape(dx, dim, n_atoms, batch_size)

    trace_terms = w_ij .+ 2.0f0 .* r_ij .^ 2 .* dw_ij
    trace_sum = sum(trace_terms; dims=(1, 2, 3)) ./ Float32(n_atoms)
    dlogp = -reshape(trace_sum, 1, batch_size)
    return ComponentArray(x=dx, logp=dlogp)
end

function _cnf_problem(ctx::NBodyCNFContext, params, batch_size::Int, tspan; rng,
                      stats=nothing)
    field = ctx.field
    if field isa MLPVectorField
        if ctx.trace_mode == :exact
            ode_func = (u, p, t) -> begin
                _count_rhs!(stats)
                _mlp_exact_dynamics(u, p, t; field)
            end
        else
            state_dim = field.dim * field.n_atoms
            probe = _rademacher(rng, state_dim, batch_size)
            probe_in = vcat(probe, zeros(Float32, 1, batch_size)) |> ctx.device
            probe_out = probe |> ctx.device
            ode_func = (u, p, t) -> begin
                _count_rhs!(stats)
                _mlp_hutchinson_dynamics(u, p, t; field, probe_in, probe_out)
            end
        end
    elseif field isa EGNNVectorField
        if field.n_layers <= 0
            h_i_flat, h_j_flat, seq_dist_flat, mask, jvp_probe = Zygote.ignore() do
                _egnn_legacy_fixed_inputs(field, batch_size, ctx.device)
            end
            ode_func = (u, p, t) -> begin
                _count_rhs!(stats)
                _egnn_legacy_exact_dynamics(
                    u, p, t; field, h_i_flat, h_j_flat, seq_dist_flat, mask, jvp_probe)
            end
        else
            seq_dist_flat, mask = Zygote.ignore() do
                _egnn_fixed_inputs(field, batch_size, ctx.device)
            end
            if ctx.trace_mode == :exact
                ode_func = (u, p, t) -> begin
                    _count_rhs!(stats)
                    _egnn_exact_trace_dynamics(u, p, t; field, seq_dist_flat, mask)
                end
            elseif ctx.trace_mode == :hutchinson_ad
                probe = _rademacher(rng, field.dim, field.n_atoms, batch_size) |> ctx.device
                ode_func = (u, p, t) -> begin
                    _count_rhs!(stats)
                    _egnn_hutchinson_trace_dynamics(
                        u, p, t; field, seq_dist_flat, mask, probe)
                end
            else
                probe = _rademacher(rng, field.dim, field.n_atoms, batch_size) |> ctx.device
                ode_func = (u, p, t) -> begin
                    _count_rhs!(stats)
                    _egnn_fd_hutchinson_trace_dynamics(
                        u, p, t; field, seq_dist_flat, mask, probe)
                end
            end
        end
    else
        error("Unsupported vector field $(typeof(field))")
    end

    u0 = ComponentArray(
        x=zeros(Float32, field.dim, field.n_atoms, batch_size),
        logp=zeros(Float32, 1, batch_size)) |> ctx.device
    return ODEProblem(ode_func, u0, tspan, params)
end

function _cnf_logp_device(ctx::NBodyCNFContext, params, x_dev; rng, stats=nothing)
    batch_size = size(x_dev, 3)
    u0 = ComponentArray(x=x_dev, logp=zeros(Float32, 1, batch_size) |> ctx.device)
    stats !== nothing && _reset!(stats)
    prob = _cnf_problem(ctx, params, batch_size, ctx.tspan; rng, stats)
    sol = solve(remake(prob; u0, p=params), ctx.solver;
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                dt=1f-5,
                abstol=ctx.abstol, reltol=ctx.reltol,
                save_everystep=false, save_start=false, save_end=true,
                verbose=false)
    _record_forward_stats!(stats, sol)
    x_noise = sol.u[end].x
    delta_logp = sol.u[end].logp
    return log_standard_normal(x_noise) .- delta_logp
end

function cnf_logp(ctx::NBodyCNFContext, params, x_batch; rng=Random.default_rng(),
                  stats=nothing)
    x_dev = x_batch |> ctx.device
    return _cnf_logp_device(ctx, params, x_dev; rng, stats)
end

function cnf_nll(ctx::NBodyCNFContext, params, x_batch; rng=Random.default_rng(),
                 stats=nothing)
    batch_size = size(x_batch, 3)
    log_pdata = cnf_logp(ctx, params, x_batch; rng, stats)
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
                        loss_callback=nothing,
                        opt_state=nothing,
                        loss_history::Vector{Float32}=Float32[],
                        start_epoch::Int=0,
                        rng::AbstractRNG=Xoshiro(42))
    n_samples = size(train_data, 3)
    steps_per_epoch = cld(n_samples, batch_size)
    total_steps = epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    opt_state = opt_state === nothing ?
        Optimisers.setup(Optimisers.Adam(Float32(learning_rate)), params) : opt_state
    loss_history = copy(loss_history)

    for epoch in (start_epoch + 1):epochs
        shuffled_idx = randperm(rng, n_samples)
        epoch_loss = 0.0f0
        epoch_forward_steps = 0
        epoch_forward_rejects = 0
        epoch_forward_rhs = 0
        epoch_backprop_rhs = 0
        for (batch_num, batch_start) in enumerate(1:batch_size:n_samples)
            batch_stop = min(batch_start + batch_size - 1, n_samples)
            batch_idx = shuffled_idx[batch_start:batch_stop]
            x_batch = center_positions(train_data[:, :, batch_idx])
            local_batch_size = length(batch_idx)

            solve_stats = CNFSolveStats()
            loss, grads = Zygote.withgradient(
                p -> cnf_nll(ctx, p, x_batch; rng, stats=solve_stats), params)
            grad_params = grads[1]
            grad_params === nothing && error("Zygote returned no parameter gradient.")
            gradient_norm = _gradient_l2_norm(grad_params)

            opt_state, params = Optimisers.update(opt_state, params, grad_params)
            step += 1
            loss_f32 = Float32(loss)
            isfinite(loss_f32) ||
                error("Non-finite loss at epoch=$epoch batch=$batch_num: $loss_f32")
            epoch_loss += loss_f32 * local_batch_size
            epoch_forward_steps += solve_stats.forward_naccept[]
            epoch_forward_rejects += solve_stats.forward_nreject[]
            epoch_forward_rhs += solve_stats.forward_nf[]
            batch_backprop_rhs = _backprop_rhs_calls(solve_stats)
            epoch_backprop_rhs += batch_backprop_rhs
            push!(loss_history, loss_f32)
            if loss_callback !== nothing
                loss_callback(;
                    epoch,
                    batch=batch_num,
                    step,
                    total_steps,
                    loss=loss_f32,
                    gradient_norm,
                )
            end

            if log_every > 0 && (step == total_steps || step % log_every == 0)
                @info "training_step" epoch batch=batch_num step total_steps loss=loss_f32 gradient_norm ode_forward_steps=solve_stats.forward_naccept[] ode_forward_rejects=solve_stats.forward_nreject[] ode_forward_rhs_calls=solve_stats.forward_nf[] backprop_rhs_calls=batch_backprop_rhs
            end
        end
        mean_loss = Float32(epoch_loss / n_samples)
        if log_every > 0 && (step == total_steps || step % log_every == 0)
            @info "training_epoch" epoch step total_steps steps_per_epoch loss=mean_loss ode_forward_steps=epoch_forward_steps ode_forward_rejects=epoch_forward_rejects ode_forward_rhs_calls=epoch_forward_rhs backprop_rhs_calls=epoch_backprop_rhs
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

function generate_normalized_cnf_samples(ctx::NBodyCNFContext, params, n_samples::Int,
                                         normalizer;
                                         rng::AbstractRNG=Random.default_rng())
    samples_norm = generate_cnf_samples(ctx, params, n_samples; rng)
    return center_positions(invert_data_normalizer(samples_norm, normalizer))
end
