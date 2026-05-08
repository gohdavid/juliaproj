export NBodyDataConfig, generate_nbody_dataset, sample_training_batch
export nri_spring_physics!, swirling_spring_physics!, lj_physics!
export polymer_langevin_force!, run_polymer_langevin_simulation
export run_polymer_langevin_sde_simulation, polymer_langevin_sde_problem
export pairwise_distances

struct NBodyDataConfig
    kind::Symbol
    dim::Int
    n_atoms::Int
    n_samples::Int
    burn_in::Int
    noise_std::Float32
    total_steps::Int
    dt::Float32
    min_spawn_dist::Float32
    physics_params::Vector{Float32}
end

function NBodyDataConfig(;
    kind::Symbol=:static_asymmetric,
    dim::Int=2,
    n_atoms::Int=10,
    n_samples::Int=4000,
    burn_in::Int=0,
    noise_std::Real=0.1f0,
    total_steps::Int=4000,
    dt::Real=0.05f0,
    min_spawn_dist::Real=1.7f0,
    physics_params=Float32[],
)
    NBodyDataConfig(kind, dim, n_atoms, n_samples, burn_in, Float32(noise_std),
                    total_steps, Float32(dt), Float32(min_spawn_dist),
                    Float32.(physics_params))
end

function nri_spring_physics!(dx, x, p, t)
    k, rest_length = p
    n_atoms = size(x, 2)
    fill!(dx, zero(eltype(dx)))
    @inbounds for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        r_ij = x[:, i] - x[:, j]
        dist = norm(r_ij)
        if dist > 1e-4
            dx[:, i] .+= -k * (one(eltype(x)) - rest_length / dist) .* r_ij
        end
    end
    return nothing
end

function swirling_spring_physics!(dx, x, p, t)
    k, rest_length, swirl_strength = p
    size(x, 1) == 2 || error("swirling_spring is implemented for 2D states.")
    n_atoms = size(x, 2)
    fill!(dx, zero(eltype(dx)))
    @inbounds for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        r_ij = x[:, i] - x[:, j]
        dist = norm(r_ij)
        if dist > 1e-4
            spring_force = -k * (one(eltype(x)) - rest_length / dist) .* r_ij
            swirl_force = swirl_strength .* [-r_ij[2], r_ij[1]] ./ dist
            dx[:, i] .+= spring_force .+ swirl_force
        end
    end
    return nothing
end

function lj_physics!(dx, x, p, t)
    epsilon, sigma = p
    n_atoms = size(x, 2)
    fill!(dx, zero(eltype(dx)))
    @inbounds for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        r_ij = x[:, i] - x[:, j]
        dist_sq = sum(abs2, r_ij)
        if dist_sq > 1e-4
            term = 24.0f0 * epsilon *
                   (2.0f0 * sigma^12 / dist_sq^7 - sigma^6 / dist_sq^4)
            dx[:, i] .+= term .* r_ij
        end
    end
    return nothing
end

function safe_lj_init(rng::AbstractRNG, dim::Int, n_atoms::Int, min_dist::Real)
    x = zeros(Float32, dim, n_atoms)
    for i in 1:n_atoms
        while true
            proposal = randn(rng, Float32, dim) .* 2.0f0
            if i == 1 || minimum(norm(proposal - x[:, j]) for j in 1:(i - 1)) > min_dist
                x[:, i] = proposal
                break
            end
        end
    end
    return x
end

function generate_static_asymmetric_dataset(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("static_asymmetric data is implemented for dim=2.")
    x_coords = Float32.(range(0, step=0.5, length=cfg.n_atoms))
    y_coords = cfg.n_atoms == 10 ?
        Float32[2.0, 1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] :
        Float32.(range(0, stop=3.5, length=cfg.n_atoms))
    base = reshape(vcat(reshape(x_coords, 1, :), reshape(y_coords, 1, :)),
                   cfg.dim, cfg.n_atoms, 1)
    noise = randn(rng, Float32, cfg.dim, cfg.n_atoms, cfg.n_samples) .* cfg.noise_std
    return base .+ noise
end

function center_frame(x)
    return x .- mean(x; dims=2)
end

function init_polymer_chain(rng::AbstractRNG, dim::Int, n_atoms::Int, bond_std::Real)
    x = zeros(Float32, dim, n_atoms)
    scale = Float32(bond_std) / sqrt(Float32(dim))
    for i in 2:n_atoms
        x[:, i] = x[:, i - 1] .+ scale .* randn(rng, Float32, dim)
    end
    return center_frame(x)
end

function polymer_langevin_force!(force, x, k_over_xi::Real)
    n_atoms = size(x, 2)
    fill!(force, zero(eltype(force)))
    @inbounds for i in 2:n_atoms
        r = x[:, i] .- x[:, i - 1]
        drift = Float32(k_over_xi) .* r
        force[:, i] .-= drift
        force[:, i - 1] .+= drift
    end
    return nothing
end

function run_polymer_langevin_simulation(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("polymer_langevin is implemented for dim=2.")
    p = cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0f0
    bond_length = length(p) >= 2 ? p[2] : 1.0f0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0f0 * diffusion / bond_length^2

    x = init_polymer_chain(rng, cfg.dim, cfg.n_atoms, bond_length)
    force = similar(x)
    trajectory = zeros(Float32, cfg.dim, cfg.n_atoms, cfg.total_steps)
    noise_scale = sqrt(2.0f0 * diffusion * cfg.dt)

    for t in 1:cfg.total_steps
        polymer_langevin_force!(force, x, k_over_xi)
        x = x .+ cfg.dt .* force .+
            noise_scale .* randn(rng, Float32, cfg.dim, cfg.n_atoms)
        trajectory[:, :, t] = center_frame(x)
    end
    return trajectory
end

function polymer_langevin_sde_problem(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("polymer_langevin is implemented for dim=2.")
    p = cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0f0
    bond_length = length(p) >= 2 ? p[2] : 1.0f0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0f0 * diffusion / bond_length^2

    x0 = vec(init_polymer_chain(rng, cfg.dim, cfg.n_atoms, bond_length))
    drift = zeros(Float32, cfg.dim, cfg.n_atoms)
    function f!(du, u, params, t)
        x = reshape(u, cfg.dim, cfg.n_atoms)
        du_mat = reshape(du, cfg.dim, cfg.n_atoms)
        polymer_langevin_force!(drift, x, k_over_xi)
        du_mat .= drift
        return nothing
    end

    sigma = sqrt(2.0f0 * diffusion)
    function g!(du, u, params, t)
        fill!(du, 0.0f0)
        @inbounds for i in axes(du, 1)
            du[i, i] = sigma
        end
        return nothing
    end

    tspan = (0.0f0, cfg.total_steps * cfg.dt)
    return SDEProblem(f!, g!, x0, tspan, nothing;
                      noise_rate_prototype=zeros(Float32, length(x0), length(x0)))
end

function run_polymer_langevin_sde_simulation(rng::AbstractRNG, cfg::NBodyDataConfig;
                                             save_stride::Int=1,
                                             center::Bool=true)
    prob = polymer_langevin_sde_problem(rng, cfg)
    saveat = range(0.0f0, cfg.total_steps * cfg.dt;
                   length=(cfg.total_steps ÷ save_stride) + 1)
    sol = solve(prob, EM(); dt=cfg.dt, saveat, adaptive=false)

    trajectory = zeros(Float32, cfg.dim, cfg.n_atoms, length(sol.u))
    for i in eachindex(sol.u)
        frame = reshape(sol.u[i], cfg.dim, cfg.n_atoms)
        trajectory[:, :, i] = center ? center_frame(frame) : frame
    end
    return trajectory
end

function run_md_simulation(rng::AbstractRNG, cfg::NBodyDataConfig)
    tspan = (0.0f0, cfg.total_steps * cfg.dt)
    saveat = range(tspan[1], tspan[2]; length=cfg.total_steps)

    if cfg.kind == :spring
        p = isempty(cfg.physics_params) ? Float32[0.01, 1.0] : cfg.physics_params
        x0 = randn(rng, Float32, cfg.dim, cfg.n_atoms) .* 2.5f0
        prob = ODEProblem(nri_spring_physics!, x0, tspan, p)
    elseif cfg.kind == :swirling_spring
        p = isempty(cfg.physics_params) ? Float32[0.01, 1.0, 0.05] : cfg.physics_params
        x0 = randn(rng, Float32, cfg.dim, cfg.n_atoms) .* 2.5f0
        prob = ODEProblem(swirling_spring_physics!, x0, tspan, p)
    elseif cfg.kind == :lennard_jones
        p = isempty(cfg.physics_params) ? Float32[0.10, 1.0] : cfg.physics_params
        x0 = safe_lj_init(rng, cfg.dim, cfg.n_atoms, cfg.min_spawn_dist)
        prob = ODEProblem(lj_physics!, x0, tspan, p)
    else
        error("Unknown n-body data kind: $(cfg.kind)")
    end

    sol = solve(prob, Tsit5(); saveat, reltol=1f-4, abstol=1f-4)
    trajectory = zeros(Float32, cfg.dim, cfg.n_atoms, cfg.total_steps)
    for t in 1:cfg.total_steps
        trajectory[:, :, t] = sol.u[t]
    end
    return trajectory
end

function generate_nbody_dataset(rng::AbstractRNG, cfg::NBodyDataConfig)
    if cfg.kind == :static_asymmetric
        return generate_static_asymmetric_dataset(rng, cfg)
    elseif cfg.kind == :polymer_langevin
        trajectory = run_polymer_langevin_sde_simulation(rng, cfg; center=true)
        return sample_training_batch(rng, trajectory; batch_size=cfg.n_samples,
                                     burn_in=cfg.burn_in)
    end
    trajectory = run_md_simulation(rng, cfg)
    return sample_training_batch(rng, trajectory; batch_size=cfg.n_samples,
                                 burn_in=cfg.burn_in)
end

function sample_training_batch(rng::AbstractRNG, trajectory; batch_size::Int=32, burn_in::Int=1000)
    total_steps = size(trajectory, 3)
    burn_in < total_steps || error("burn_in ($burn_in) must be less than total steps ($total_steps).")
    valid_indices = (burn_in + 1):total_steps
    return trajectory[:, :, rand(rng, valid_indices, batch_size)]
end

function sample_training_batch(trajectory; batch_size::Int=32, burn_in::Int=1000)
    return sample_training_batch(Random.default_rng(), trajectory; batch_size, burn_in)
end

function pairwise_distances(positions)
    dim, n_atoms, batch_size = size(positions)
    pos_i = reshape(positions, dim, n_atoms, 1, batch_size)
    pos_j = reshape(positions, dim, 1, n_atoms, batch_size)
    d_ij = sqrt.(sum(abs2, pos_i .- pos_j; dims=1))
    return reshape(d_ij, n_atoms, n_atoms, batch_size)
end
