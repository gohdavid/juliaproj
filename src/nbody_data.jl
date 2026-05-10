export NBodyDataConfig, generate_nbody_dataset, sample_training_batch
export nri_spring_physics!, swirling_spring_physics!, lj_physics!
export polymer_langevin_force!, run_polymer_langevin_simulation
export run_polymer_langevin_sde_simulation, polymer_langevin_sde_problem
export polymer_langevin_potential, polymer_langevin_score!, polymer_nonideal_params
export pairwise_distances

struct NBodyDataConfig
    kind::Symbol
    dim::Int
    n_atoms::Int
    n_samples::Int
    burn_in::Int
    noise_std::Float64
    total_steps::Int
    dt::Float64
    min_spawn_dist::Float64
    physics_params::Vector{Float64}
    source_path::String
    source_paths::Vector{String}
    source_dir::String
    source_pattern::String
    allow_partial::Bool
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
    source_path::AbstractString="",
    source_paths=String[],
    source_dir::AbstractString="",
    source_pattern::AbstractString="*.h5",
    allow_partial::Bool=false,
)
    NBodyDataConfig(kind, dim, n_atoms, n_samples, burn_in, Float64(noise_std),
                    total_steps, Float64(dt), Float64(min_spawn_dist),
                    Float64.(physics_params), String(source_path),
                    String.(source_paths), String(source_dir),
                    String(source_pattern), allow_partial)
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

function _notebook_checkmark_base(::Type{T}=Float32) where {T}
    return T[
        0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5;
        2.0  1.0  0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5
    ]
end

function generate_checkmark_synthetic_dataset(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("checkmark_synthetic data is implemented for dim=2.")
    cfg.n_atoms == 10 || error("checkmark_synthetic data follows the notebook and requires n_atoms=10.")
    base = reshape(_notebook_checkmark_base(Float32), cfg.dim, cfg.n_atoms, 1)
    noise = randn(rng, Float32, cfg.dim, cfg.n_atoms, cfg.n_samples) .* cfg.noise_std
    return base .+ noise
end

function generate_static_asymmetric_dataset(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("static_asymmetric data is implemented for dim=2.")
    if cfg.n_atoms == 10
        return generate_checkmark_synthetic_dataset(rng, cfg)
    end
    x_coords = Float32.(range(0, step=0.5, length=cfg.n_atoms))
    y_coords =
        Float32.(range(0, stop=3.5, length=cfg.n_atoms))
    base = reshape(vcat(reshape(x_coords, 1, :), reshape(y_coords, 1, :)),
                   cfg.dim, cfg.n_atoms, 1)
    noise = randn(rng, Float32, cfg.dim, cfg.n_atoms, cfg.n_samples) .* cfg.noise_std
    return base .+ noise
end

function _data_path(path::AbstractString)
    isempty(path) && return ""
    return isabspath(path) ? String(path) : abspath(String(path))
end

function _simple_pattern_match(name::AbstractString, pattern::AbstractString)
    if pattern == "*"
        return true
    elseif !occursin("*", pattern)
        return name == pattern
    end
    parts = split(pattern, "*")
    length(parts) == 2 || error("Only one '*' wildcard is supported in data.source_pattern")
    prefix, suffix = parts
    return startswith(name, prefix) && endswith(name, suffix)
end

function _rouse_hdf5_paths(cfg::NBodyDataConfig)
    paths = String[]
    if !isempty(cfg.source_path)
        push!(paths, _data_path(cfg.source_path))
    end
    append!(paths, _data_path.(cfg.source_paths))
    if !isempty(cfg.source_dir)
        dir = _data_path(cfg.source_dir)
        isdir(dir) || error("data.source_dir does not exist: $dir")
        for name in sort(readdir(dir))
            _simple_pattern_match(name, cfg.source_pattern) &&
                push!(paths, joinpath(dir, name))
        end
    end
    unique!(paths)
    isempty(paths) && error("rouse_hdf5 data needs data.source_path, data.source_paths, or data.source_dir")
    return paths
end

function load_rouse_hdf5_dataset(cfg::NBodyDataConfig)
    paths = _rouse_hdf5_paths(cfg)
    trajectories = Array{Float32,3}[]
    for path in paths
        try
            h5open(path, "r") do h5
                haskey(h5, "traj") || error("Missing HDF5 dataset 'traj' in $path")
                written = haskey(h5, "written") ? Int(read(h5["written"])[1]) :
                          size(h5["traj"], 3)
                written > cfg.burn_in || error("burn_in ($(cfg.burn_in)) must be less than written frames ($written) in $path")
                dim, n_atoms, _ = size(h5["traj"])
                dim == cfg.dim || error("Expected dim=$(cfg.dim), got $dim in $path")
                n_atoms == cfg.n_atoms || error("Expected n_atoms=$(cfg.n_atoms), got $n_atoms in $path")
                push!(trajectories, Float32.(h5["traj"][:, :, (cfg.burn_in + 1):written]))
            end
        catch err
            if cfg.allow_partial
                @warn "Skipping unavailable Rouse HDF5 file" path exception=(err, catch_backtrace())
            else
                rethrow()
            end
        end
    end
    isempty(trajectories) && error("No Rouse HDF5 frames were loaded")
    return cat(trajectories...; dims=3)
end

function center_frame(x)
    return x .- mean(x; dims=2)
end

function init_polymer_chain(rng::AbstractRNG, dim::Int, n_atoms::Int, bond_std::Real)
    x = zeros(Float64, dim, n_atoms)
    scale = Float64(bond_std) / sqrt(Float64(dim))
    for i in 2:n_atoms
        x[:, i] = x[:, i - 1] .+ scale .* randn(rng, Float64, dim)
    end
    return center_frame(x)
end

function init_hairpin_polymer_chain(rng::AbstractRNG, dim::Int, n_atoms::Int,
                                    bond_length::Real, separation::Real,
                                    noise_std::Real)
    dim == 2 || error("hairpin polymer initialization is implemented for dim=2.")
    iseven(n_atoms) || error("hairpin polymer initialization requires an even bead count.")
    half = div(n_atoms, 2)
    x = zeros(Float64, dim, n_atoms)
    sep = Float64(separation)
    step = Float64(bond_length)
    for i in 1:half
        x[1, i] = (i - 1) * step
        x[2, i] = 0.5 * sep
    end
    for i in (half + 1):n_atoms
        partner = n_atoms + 1 - i
        x[1, i] = (partner - 1) * step
        x[2, i] = -0.5 * sep
    end
    if noise_std > 0
        x .+= Float64(noise_std) .* randn(rng, Float64, dim, n_atoms)
    end
    return center_frame(x)
end

function init_ring_polymer_chain(rng::AbstractRNG, dim::Int, n_atoms::Int,
                                 bond_length::Real, noise_std::Real)
    dim == 2 || error("ring polymer initialization is implemented for dim=2.")
    n_atoms >= 3 || error("ring polymer initialization requires at least 3 beads.")
    x = zeros(Float64, dim, n_atoms)
    radius = Float64(bond_length) / (2.0 * sin(pi / Float64(n_atoms)))
    for i in 1:n_atoms
        theta = 2.0 * pi * Float64(i - 1) / Float64(n_atoms)
        x[1, i] = radius * cos(theta)
        x[2, i] = radius * sin(theta)
    end
    if noise_std > 0
        x .+= Float64(noise_std) .* randn(rng, Float64, dim, n_atoms)
    end
    return center_frame(x)
end

function _polymer_initial_state(rng::AbstractRNG, cfg::NBodyDataConfig,
                                bond_length::Real, p)
    init_mode = length(p) >= 28 ? Int(round(p[28])) : 0
    if init_mode == 1
        separation = length(p) >= 29 ? p[29] : bond_length
        noise_std = length(p) >= 30 ? p[30] : 0.0
        return init_hairpin_polymer_chain(rng, cfg.dim, cfg.n_atoms, bond_length,
                                          separation, noise_std)
    elseif init_mode == 2
        noise_std = length(p) >= 30 ? p[30] : 0.0
        return init_ring_polymer_chain(rng, cfg.dim, cfg.n_atoms, bond_length,
                                       noise_std)
    end
    return init_polymer_chain(rng, cfg.dim, cfg.n_atoms, bond_length)
end

const POLYMER_NONIDEAL_OFFSET = 4

function _polymer_nonideal_params(p)
    return (
        lj_enabled = length(p) >= 4 && p[4] > 0.5f0,
        lj_epsilon = length(p) >= 5 ? p[5] : 0.0f0,
        lj_sigma = length(p) >= 6 ? p[6] : 1.0f0,
        lj_softening = length(p) >= 7 ? p[7] : 0.0f0,
        lj_cutoff = length(p) >= 8 ? p[8] : 0.0f0,
        lj_exclude_bonded = length(p) >= 9 ? p[9] > 0.5f0 : true,
        lj_shift = length(p) >= 10 ? p[10] > 0.5f0 : true,
        ev_enabled = length(p) >= 11 && p[11] > 0.5f0,
        ev_epsilon = length(p) >= 12 ? p[12] : 0.0f0,
        ev_sigma = length(p) >= 13 ? p[13] : 1.0f0,
        ev_softening = length(p) >= 14 ? p[14] : 0.0f0,
        ev_power = length(p) >= 15 ? p[15] : 12.0f0,
        ev_cutoff = length(p) >= 16 ? p[16] : 0.0f0,
        ev_exclude_bonded = length(p) >= 17 ? p[17] > 0.5f0 : true,
        conf_enabled = length(p) >= 18 && p[18] > 0.5f0,
        conf_strength = length(p) >= 19 ? p[19] : 0.0f0,
        conf_centered = length(p) >= 20 ? p[20] > 0.5f0 : true,
        hairpin_lj_enabled = length(p) >= 21 && p[21] > 0.5f0,
        hairpin_lj_epsilon = length(p) >= 22 ? p[22] : 0.0f0,
        hairpin_lj_sigma = length(p) >= 23 ? p[23] : 1.0f0,
        hairpin_lj_softening = length(p) >= 24 ? p[24] : 0.0f0,
        hairpin_lj_cutoff = length(p) >= 25 ? p[25] : 0.0f0,
        hairpin_lj_shift = length(p) >= 26 ? p[26] > 0.5f0 : true,
        hairpin_lj_min_separation = length(p) >= 27 ? Int(round(p[27])) : 4,
        ring_lj_enabled = length(p) >= 31 && p[31] > 0.5f0,
        ring_lj_epsilon = length(p) >= 32 ? p[32] : 0.0f0,
        ring_lj_sigma = length(p) >= 33 ? p[33] : 1.0f0,
        ring_lj_softening = length(p) >= 34 ? p[34] : 0.0f0,
        ring_lj_cutoff = length(p) >= 35 ? p[35] : 0.0f0,
        ring_lj_shift = length(p) >= 36 ? p[36] > 0.5f0 : true,
        ring_bond_enabled = length(p) >= 37 && p[37] > 0.5f0,
    )
end

polymer_nonideal_params(p) = _polymer_nonideal_params(Float64.(p))

function _within_cutoff(r2, cutoff)
    cutoff <= 0 && return true
    return r2 <= cutoff * cutoff
end

function _skip_bonded(i::Int, j::Int, n_atoms::Int, exclude_bonded::Bool,
                      ring_bond_enabled::Bool)
    return exclude_bonded &&
           (abs(i - j) == 1 || (ring_bond_enabled && _ring_contact(i, j, n_atoms)))
end

function _hairpin_contact(i::Int, j::Int, n_atoms::Int, min_separation::Int)
    return i + j == n_atoms + 1 && abs(i - j) >= min_separation
end

function _ring_contact(i::Int, j::Int, n_atoms::Int)
    return i == 1 && j == n_atoms
end

function polymer_langevin_score!(score, x, diffusion::Real, k_over_xi::Real,
                                 nonideal)
    dim = size(x, 1)
    n_atoms = size(x, 2)
    fill!(score, zero(eltype(score)))
    inv_d = 1.0 / Float64(diffusion)
    k_score = Float64(k_over_xi) * inv_d

    @inbounds for i in 2:n_atoms
        for d in 1:dim
            s = k_score * (x[d, i] - x[d, i - 1])
            score[d, i] -= s
            score[d, i - 1] += s
        end
    end
    if nonideal.ring_bond_enabled && n_atoms >= 3
        @inbounds for d in 1:dim
            s = k_score * (x[d, 1] - x[d, n_atoms])
            score[d, 1] -= s
            score[d, n_atoms] += s
        end
    end

    min_r2 = 1.0f-12
    @inbounds for i in 1:(n_atoms - 1), j in (i + 1):n_atoms
        dx1 = x[1, i] - x[1, j]
        dx2 = dim >= 2 ? x[2, i] - x[2, j] : 0.0
        raw_r2 = dx1 * dx1 + dx2 * dx2

        coeff = 0.0
        if nonideal.lj_enabled &&
           !_skip_bonded(i, j, n_atoms, nonideal.lj_exclude_bonded,
                         nonideal.ring_bond_enabled) &&
           _within_cutoff(raw_r2, nonideal.lj_cutoff)
            r2 = max(raw_r2 + nonideal.lj_softening^2, min_r2)
            inv_r2 = 1.0f0 / r2
            sig2_over_r2 = (nonideal.lj_sigma * nonideal.lj_sigma) * inv_r2
            sr6 = sig2_over_r2^3
            sr12 = sr6 * sr6
            coeff += 24.0 * nonideal.lj_epsilon * inv_r2 *
                     (2.0 * sr12 - sr6)
        end

        if nonideal.ev_enabled &&
           !_skip_bonded(i, j, n_atoms, nonideal.ev_exclude_bonded,
                         nonideal.ring_bond_enabled) &&
           _within_cutoff(raw_r2, nonideal.ev_cutoff)
            r2 = max(raw_r2 + nonideal.ev_softening^2, min_r2)
            power = nonideal.ev_power
            sigma_power = nonideal.ev_sigma^power
            r_power_plus2 = r2^((power + 2.0) / 2.0)
            coeff += nonideal.ev_epsilon * power * sigma_power / r_power_plus2
        end

        if nonideal.hairpin_lj_enabled &&
           _hairpin_contact(i, j, n_atoms, nonideal.hairpin_lj_min_separation) &&
           _within_cutoff(raw_r2, nonideal.hairpin_lj_cutoff)
            r2 = max(raw_r2 + nonideal.hairpin_lj_softening^2, min_r2)
            inv_r2 = 1.0f0 / r2
            sig2_over_r2 = (nonideal.hairpin_lj_sigma * nonideal.hairpin_lj_sigma) * inv_r2
            sr6 = sig2_over_r2^3
            sr12 = sr6 * sr6
            coeff += 24.0 * nonideal.hairpin_lj_epsilon * inv_r2 *
                     (2.0 * sr12 - sr6)
        end

        if nonideal.ring_lj_enabled &&
           _ring_contact(i, j, n_atoms) &&
           _within_cutoff(raw_r2, nonideal.ring_lj_cutoff)
            r2 = max(raw_r2 + nonideal.ring_lj_softening^2, min_r2)
            inv_r2 = 1.0f0 / r2
            sig2_over_r2 = (nonideal.ring_lj_sigma * nonideal.ring_lj_sigma) * inv_r2
            sr6 = sig2_over_r2^3
            sr12 = sr6 * sr6
            coeff += 24.0 * nonideal.ring_lj_epsilon * inv_r2 *
                     (2.0 * sr12 - sr6)
        end

        if coeff != 0.0
            for d in 1:dim
                dx = x[d, i] - x[d, j]
                s = coeff * dx
                score[d, i] += s
                score[d, j] -= s
            end
        end
    end

    if nonideal.conf_enabled
        strength = Float64(nonideal.conf_strength)
        if nonideal.conf_centered
            for d in 1:dim
                cm = zero(eltype(x))
                for i in 1:n_atoms
                    cm += x[d, i]
                end
                cm /= n_atoms
                for i in 1:n_atoms
                    score[d, i] -= 2.0 * strength * (x[d, i] - cm)
                end
            end
        else
            for i in 1:n_atoms, d in 1:dim
                score[d, i] -= 2.0 * strength * x[d, i]
            end
        end
    end
    return nothing
end

function polymer_langevin_force!(force, x, diffusion::Real, k_over_xi::Real, nonideal)
    polymer_langevin_score!(force, x, diffusion, k_over_xi, nonideal)
    force .*= Float64(diffusion)
    return nothing
end

function polymer_langevin_force!(force, x, k_over_xi::Real)
    nonideal = _polymer_nonideal_params(Float64[])
    return polymer_langevin_force!(force, x, 1.0, k_over_xi, nonideal)
end

function polymer_langevin_potential(x, diffusion::Real, k_over_xi::Real, nonideal)
    dim = size(x, 1)
    n_atoms = size(x, 2)
    u = 0.0
    coeff = Float64(k_over_xi) / (2.0 * Float64(diffusion))
    @inbounds for i in 2:n_atoms, d in 1:dim
        u += coeff * (x[d, i] - x[d, i - 1])^2
    end
    if nonideal.ring_bond_enabled && n_atoms >= 3
        @inbounds for d in 1:dim
            u += coeff * (x[d, 1] - x[d, n_atoms])^2
        end
    end

    min_r2 = 1.0f-12
    @inbounds for i in 1:(n_atoms - 1), j in (i + 1):n_atoms
        dx1 = x[1, i] - x[1, j]
        dx2 = dim >= 2 ? x[2, i] - x[2, j] : 0.0
        raw_r2 = dx1 * dx1 + dx2 * dx2

        if nonideal.lj_enabled &&
           !_skip_bonded(i, j, n_atoms, nonideal.lj_exclude_bonded,
                         nonideal.ring_bond_enabled) &&
           _within_cutoff(raw_r2, nonideal.lj_cutoff)
            r2 = max(raw_r2 + nonideal.lj_softening^2, min_r2)
            sig2_over_r2 = (nonideal.lj_sigma * nonideal.lj_sigma) / r2
            sr6 = sig2_over_r2^3
            sr12 = sr6 * sr6
            val = 4.0 * nonideal.lj_epsilon * (sr12 - sr6)
            if nonideal.lj_shift && nonideal.lj_cutoff > 0.0
                src2 = (nonideal.lj_sigma / nonideal.lj_cutoff)^2
                src6 = src2^3
                val -= 4.0 * nonideal.lj_epsilon * (src6^2 - src6)
            end
            u += val
        end

        if nonideal.ev_enabled &&
           !_skip_bonded(i, j, n_atoms, nonideal.ev_exclude_bonded,
                         nonideal.ring_bond_enabled) &&
           _within_cutoff(raw_r2, nonideal.ev_cutoff)
            r2 = max(raw_r2 + nonideal.ev_softening^2, min_r2)
            r = sqrt(r2)
            u += nonideal.ev_epsilon * (nonideal.ev_sigma / r)^nonideal.ev_power
        end

        if nonideal.hairpin_lj_enabled &&
           _hairpin_contact(i, j, n_atoms, nonideal.hairpin_lj_min_separation) &&
           _within_cutoff(raw_r2, nonideal.hairpin_lj_cutoff)
            r2 = max(raw_r2 + nonideal.hairpin_lj_softening^2, min_r2)
            sig2_over_r2 = (nonideal.hairpin_lj_sigma * nonideal.hairpin_lj_sigma) / r2
            sr6 = sig2_over_r2^3
            sr12 = sr6 * sr6
            val = 4.0 * nonideal.hairpin_lj_epsilon * (sr12 - sr6)
            if nonideal.hairpin_lj_shift && nonideal.hairpin_lj_cutoff > 0.0
                src2 = (nonideal.hairpin_lj_sigma / nonideal.hairpin_lj_cutoff)^2
                src6 = src2^3
                val -= 4.0 * nonideal.hairpin_lj_epsilon * (src6^2 - src6)
            end
            u += val
        end

        if nonideal.ring_lj_enabled &&
           _ring_contact(i, j, n_atoms) &&
           _within_cutoff(raw_r2, nonideal.ring_lj_cutoff)
            r2 = max(raw_r2 + nonideal.ring_lj_softening^2, min_r2)
            sig2_over_r2 = (nonideal.ring_lj_sigma * nonideal.ring_lj_sigma) / r2
            sr6 = sig2_over_r2^3
            sr12 = sr6 * sr6
            val = 4.0 * nonideal.ring_lj_epsilon * (sr12 - sr6)
            if nonideal.ring_lj_shift && nonideal.ring_lj_cutoff > 0.0
                src2 = (nonideal.ring_lj_sigma / nonideal.ring_lj_cutoff)^2
                src6 = src2^3
                val -= 4.0 * nonideal.ring_lj_epsilon * (src6^2 - src6)
            end
            u += val
        end
    end

    if nonideal.conf_enabled
        strength = Float64(nonideal.conf_strength)
        if nonideal.conf_centered
            for d in 1:dim
                cm = zero(eltype(x))
                for i in 1:n_atoms
                    cm += x[d, i]
                end
                cm /= n_atoms
                for i in 1:n_atoms
                    u += strength * (x[d, i] - cm)^2
                end
            end
        else
            u += strength * sum(abs2, x)
        end
    end
    return u
end

function run_polymer_langevin_simulation(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("polymer_langevin is implemented for dim=2.")
    p = cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0
    bond_length = length(p) >= 2 ? p[2] : 1.0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0 * diffusion / bond_length^2
    nonideal = _polymer_nonideal_params(p)

    x = _polymer_initial_state(rng, cfg, bond_length, p)
    force = similar(x)
    trajectory = zeros(Float64, cfg.dim, cfg.n_atoms, cfg.total_steps)
    noise_scale = sqrt(2.0 * diffusion * cfg.dt)

    for t in 1:cfg.total_steps
        polymer_langevin_force!(force, x, diffusion, k_over_xi, nonideal)
        x = x .+ cfg.dt .* force .+
            noise_scale .* randn(rng, Float64, cfg.dim, cfg.n_atoms)
        trajectory[:, :, t] = center_frame(x)
    end
    return trajectory
end

function polymer_langevin_sde_problem(rng::AbstractRNG, cfg::NBodyDataConfig)
    cfg.dim == 2 || error("polymer_langevin is implemented for dim=2.")
    p = cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0
    bond_length = length(p) >= 2 ? p[2] : 1.0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0 * diffusion / bond_length^2
    nonideal = _polymer_nonideal_params(p)

    x0 = vec(_polymer_initial_state(rng, cfg, bond_length, p))
    drift = zeros(Float64, cfg.dim, cfg.n_atoms)
    function f!(du, u, params, t)
        x = reshape(u, cfg.dim, cfg.n_atoms)
        du_mat = reshape(du, cfg.dim, cfg.n_atoms)
        polymer_langevin_force!(drift, x, diffusion, k_over_xi, nonideal)
        du_mat .= drift
        return nothing
    end

    sigma = sqrt(2.0 * diffusion)
    function g!(du, u, params, t)
        fill!(du, sigma)
        return nothing
    end

    tspan = (0.0, Float64(cfg.total_steps) * Float64(cfg.dt))
    return SDEProblem(f!, g!, x0, tspan, nothing)
end

function run_polymer_langevin_sde_simulation(rng::AbstractRNG, cfg::NBodyDataConfig;
                                             save_stride::Int=1,
                                             center::Bool=true)
    prob = polymer_langevin_sde_problem(rng, cfg)
    saveat = range(0.0, cfg.total_steps * cfg.dt;
                   length=(cfg.total_steps ÷ save_stride) + 1)
    sol = solve(prob, EM(); dt=cfg.dt, saveat, adaptive=false)

    trajectory = zeros(Float64, cfg.dim, cfg.n_atoms, length(sol.u))
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
    elseif cfg.kind in (:checkmark_synthetic, :checkmark, :check_mark)
        return generate_checkmark_synthetic_dataset(rng, cfg)
    elseif cfg.kind == :polymer_langevin
        trajectory = run_polymer_langevin_sde_simulation(rng, cfg; center=true)
        return sample_training_batch(rng, trajectory; batch_size=cfg.n_samples,
                                     burn_in=cfg.burn_in)
    elseif cfg.kind == :rouse_hdf5
        trajectory = load_rouse_hdf5_dataset(cfg)
        if cfg.n_samples <= 0 || cfg.n_samples >= size(trajectory, 3)
            return trajectory
        end
        return sample_training_batch(rng, trajectory; batch_size=cfg.n_samples,
                                     burn_in=0)
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
