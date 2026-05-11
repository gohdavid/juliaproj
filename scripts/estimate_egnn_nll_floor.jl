#!/usr/bin/env julia

"""
Estimate the best achievable EGNN-CNF NLL from simulation data.

The EGNN models in this repository center coordinates before evaluating the
CNF likelihood. This script therefore projects centered bead coordinates onto
an orthonormal center-of-mass-free basis and estimates the differential
entropy there. That entropy is the data NLL floor in the same coordinate
system used by the EGNN.

Usage:
  julia --project=. scripts/estimate_egnn_nll_floor.jl --config configs/model/nbody_polymer_hairpin_lj_cnf_egnn_3layer.yaml
  julia --project=. scripts/estimate_egnn_nll_floor.jl --config CONFIG --max-samples 4000 --k 5 --output-dir runs/nll_floor
"""

using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/estimate_egnn_nll_floor.jl --config CONFIG [options]

    Options:
      --config PATH       Model/training YAML config whose data section points at simulation HDF5. Required.
      --max-samples N    Maximum samples used by estimators. Default: 4000.
      --k N              k-th neighbor for Kozachenko-Leonenko entropy. Default: 5.
      --seed N           Subsampling seed. Default: config experiment.seed.
      --output-dir DIR   Directory for nll_floor.jls and nll_floor.txt. Default: none.
      --help             Show this message.

    Reported floors:
      model_space_nll_floor_knn       Empirical kNN entropy after training normalizer + EGNN centering/projection.
      model_space_nll_floor_gaussian  Gaussian covariance entropy in that same space.
      raw_centered_nll_floor_*        Same estimators before data normalization.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}(
        "max_samples" => 4000,
        "k" => 5,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg in ("--config", "--max-samples", "--k", "--seed", "--output-dir")
            i < length(args) || error("Missing value for $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            opts[key] = key in ("max_samples", "k", "seed") ? parse(Int, value) : value
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    haskey(opts, "config") || error("Set --config CONFIG")
    return opts
end

project_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))

function helmert_contrast(n::Int)
    h = zeros(Float64, n - 1, n)
    for k in 1:(n - 1)
        denom = sqrt(Float64(k * (k + 1)))
        h[k, 1:k] .= 1.0 / denom
        h[k, k + 1] = -Float64(k) / denom
    end
    return h
end

function contrast_coordinates(x)
    dim, n_atoms, n_samples = size(x)
    h = helmert_contrast(n_atoms)
    z = Matrix{Float64}(undef, dim * (n_atoms - 1), n_samples)
    for sample in 1:n_samples
        for d in 1:dim
            rows = ((d - 1) * (n_atoms - 1) + 1):(d * (n_atoms - 1))
            z[rows, sample] .= h * Float64.(view(x, d, :, sample))
        end
    end
    return z
end

function digamma_approx(x::Real)
    y = Float64(x)
    result = 0.0
    while y < 8.0
        result -= 1.0 / y
        y += 1.0
    end
    inv = 1.0 / y
    inv2 = inv * inv
    return result + log(y) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
end

function loggamma_lanczos(z::Real)
    x = Float64(z)
    coeffs = (
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    if x < 0.5
        return log(pi) - log(sin(pi * x)) - loggamma_lanczos(1.0 - x)
    end
    x -= 1.0
    a = 0.99999999999980993
    for (i, c) in enumerate(coeffs)
        a += c / (x + i)
    end
    t = x + length(coeffs) - 0.5
    return 0.5 * log(2.0 * pi) + (x + 0.5) * log(t) - t + log(a)
end

log_unit_ball_volume(d::Int) = 0.5 * d * log(pi) - loggamma_lanczos(0.5 * d + 1.0)

function kth_neighbor_radii(data::AbstractMatrix{<:Real}, k::Int)
    d, n = size(data)
    1 <= k < n || error("k must satisfy 1 <= k < number of samples")
    radii = Vector{Float64}(undef, n)
    distances = Vector{Float64}(undef, n - 1)

    for i in 1:n
        cursor = 1
        xi = view(data, :, i)
        for j in 1:n
            i == j && continue
            s = 0.0
            xj = view(data, :, j)
            @inbounds for row in 1:d
                delta = xi[row] - xj[row]
                s += delta * delta
            end
            distances[cursor] = sqrt(s)
            cursor += 1
        end
        radii[i] = max(partialsort!(distances, k), eps(Float64))
    end
    return radii
end

function knn_entropy(data::AbstractMatrix{<:Real}; k::Int=5)
    d, n = size(data)
    radii = kth_neighbor_radii(data, k)
    return digamma_approx(n) - digamma_approx(k) + log_unit_ball_volume(d) +
           d * mean(log.(radii))
end

function gaussian_entropy(data::AbstractMatrix{<:Real}; ridge::Real=1.0e-10)
    d, n = size(data)
    centered = data .- mean(data; dims=2)
    covmat = Symmetric((centered * centered') / max(n - 1, 1))
    vals = eigvals(covmat)
    vals = max.(vals, Float64(ridge))
    return 0.5 * (d * log(2.0 * pi * exp(1.0)) + sum(log.(vals)))
end

function sample_indices(rng::AbstractRNG, n::Int, max_samples::Int)
    if max_samples <= 0 || max_samples >= n
        return collect(1:n)
    end
    return sort(randperm(rng, n)[1:max_samples])
end

function load_training_data(cfg, rng::AbstractRNG)
    data_cfg = BoltzFlow._data_config(cfg)
    data = Float32.(BoltzFlow.generate_nbody_dataset(rng, data_cfg))
    if BoltzFlow.cfgbool(cfg, "data.center", false)
        data = BoltzFlow.center_positions(data)
    end
    stored_normalizer = BoltzFlow.cfgget(cfg, "data.normalizer", nothing)
    normalizer = stored_normalizer === nothing ? BoltzFlow.fit_data_normalizer(
        data;
        enabled=BoltzFlow.cfgbool(cfg, "data.normalize", false),
        mode=String(BoltzFlow.cfgget(cfg, "data.normalization_mode", "scalar")),
        eps=BoltzFlow.cfgfloat32(cfg, "data.normalization_eps", 1.0f-6),
    ) : stored_normalizer
    model_data = BoltzFlow.center_positions(BoltzFlow.apply_data_normalizer(data, normalizer))
    raw_centered = BoltzFlow.center_positions(data)
    return raw_centered, model_data, normalizer, data_cfg
end

function estimate_floor(opts)
    config_path = project_path(String(opts["config"]))
    isfile(config_path) || error("Config does not exist: $config_path")
    cfg = BoltzFlow.load_yaml_config(config_path)

    seed = Int(get(opts, "seed", BoltzFlow.cfgint(cfg, "experiment.seed", 0)))
    rng = Xoshiro(seed)
    raw_data_all, model_data_all, normalizer, data_cfg = load_training_data(cfg, rng)
    idx = sample_indices(rng, size(raw_data_all, 3), Int(opts["max_samples"]))
    raw_data = raw_data_all[:, :, idx]
    model_data = model_data_all[:, :, idx]

    raw_z = contrast_coordinates(raw_data)
    model_z = contrast_coordinates(model_data)
    k = Int(opts["k"])

    @info "estimating EGNN NLL floor" samples=size(model_z, 2) dimension=size(model_z, 1) k
    raw_knn = knn_entropy(raw_z; k)
    model_knn = knn_entropy(model_z; k)
    raw_gaussian = gaussian_entropy(raw_z)
    model_gaussian = gaussian_entropy(model_z)

    result = Dict{String,Any}(
        "config_path" => config_path,
        "seed" => seed,
        "k" => k,
        "n_samples" => size(model_z, 2),
        "dimension" => size(model_z, 1),
        "dim" => data_cfg.dim,
        "n_atoms" => data_cfg.n_atoms,
        "normalizer" => normalizer,
        "model_space_nll_floor_knn" => model_knn,
        "model_space_nll_floor_gaussian" => model_gaussian,
        "raw_centered_nll_floor_knn" => raw_knn,
        "raw_centered_nll_floor_gaussian" => raw_gaussian,
        "created_at" => Dates.now(),
        "notes" => "Floors are differential entropy estimates in the center-of-mass-free coordinates used by EGNN likelihoods.",
    )

    if haskey(opts, "output_dir")
        output_dir = project_path(String(opts["output_dir"]))
        mkpath(output_dir)
        serialize(joinpath(output_dir, "nll_floor.jls"), result)
        open(joinpath(output_dir, "nll_floor.txt"), "w") do io
            for key in sort!(collect(keys(result)); by=string)
                println(io, key, " = ", result[key])
            end
        end
        result["output_dir"] = output_dir
    end

    return result
end

function print_result(result)
    @printf("EGNN NLL floor estimate\n")
    @printf("  samples: %d\n", result["n_samples"])
    @printf("  centered dimension: %d\n", result["dimension"])
    @printf("  kNN k: %d\n", result["k"])
    @printf("  model-space kNN floor: %.6f nats/sample\n", result["model_space_nll_floor_knn"])
    @printf("  model-space Gaussian floor: %.6f nats/sample\n", result["model_space_nll_floor_gaussian"])
    @printf("  raw centered kNN floor: %.6f nats/sample\n", result["raw_centered_nll_floor_knn"])
    @printf("  raw centered Gaussian floor: %.6f nats/sample\n", result["raw_centered_nll_floor_gaussian"])
    if haskey(result, "output_dir")
        @printf("  wrote: %s\n", result["output_dir"])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = estimate_floor(parse_args(ARGS))
    print_result(result)
end
