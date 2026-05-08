"""
BoltzFlow: Inference of Atomistic Protein Dynamics from Equilibrium Conformational
Distributions using Neural ODEs.

Algorithm (from paper):
1. Obtain equilibrium snapshots q_1...q_n
2. Train a CNF to learn P(q) via maximum log-likelihood (adjoint method)
3. Infer trajectory: for each step t,
   a. Integrate CNF forward to get log P(q(t))
   b. Backpropagate via adjoint to get ∇_q log P(q(t))
   c. Use this as force in Hamilton's equations
"""
module BoltzFlow

using LinearAlgebra, Random, Statistics, Printf
using OrdinaryDiffEq
using StochasticDiffEq
using ComponentArrays
using DiffEqFlux
using Optimisers
using Zygote
using CUDA
get!(ENV, "GKSwstype", "100")
using Plots
using HDF5

export SpringSystem, generate_equilibrium_data, maxwell_boltzmann_sample
export boltzmann_logp, boltzmann_score
export MLPNet, count_params, init_params, mlp_forward
export cnf_loglikelihood, score_function
export train_cnf!
export infer_trajectory, leapfrog_step, GenericSystem
export compare_distributions, report_stats
export load_yaml_config, run_experiment, run_nbody_experiment
export run_nbody_flow_matching_experiment, run_nbody_diffusion_experiment
export NBodyDataConfig, generate_nbody_dataset, sample_training_batch
export MLPVectorField, EGNNVectorField, NBodyCNFContext
export build_vector_field, init_cnf_params, train_cnf_adam, generate_cnf_samples
export EquivariantFMVectorField, NBodyFlowMatchingContext, FlowMatchingResult
export build_fm_vector_field, init_fm_params, train_flow_matching_adam
export generate_flow_matching_samples, center_positions, pairwise_distance_mae
export EquivariantDiffusionModel, NBodyDiffusionContext, DiffusionResult
export build_diffusion_model, init_diffusion_params, train_diffusion_adam
export generate_diffusion_samples
export config_hash, config_output_dir
export polymer_langevin_potential, polymer_langevin_score!, polymer_nonideal_params

# ============================================================
# §1  Spring System (toy data)
# ============================================================
# 3D harmonic oscillator with V(q) = (k/2)||q||².
# At equilibrium the Boltzmann distribution is
#   P(q) = N(0, σ²I),  σ² = k_B T / k
# so training data is just Gaussian samples.
# The exact score is:  ∇_q log P(q) = -q / σ²

struct SpringSystem
    k  ::Float64   # spring constant
    m  ::Float64   # particle mass (reduced units)
    T  ::Float64   # temperature
    kB ::Float64   # Boltzmann constant (set 1 in reduced units)
    dim::Int       # spatial dimension
end

SpringSystem(; k=1.0, m=1.0, T=1.0, kB=1.0, dim=3) =
    SpringSystem(k, m, T, kB, dim)

"""Sample n_samples equilibrium snapshots from the Boltzmann distribution."""
function generate_equilibrium_data(sys::SpringSystem, n_samples::Int;
                                   rng=Random.default_rng())
    σ = sqrt(sys.kB * sys.T / sys.k)
    Float32.(σ .* randn(rng, sys.dim, n_samples))
end

"""Sample momenta from the Maxwell-Boltzmann distribution."""
function maxwell_boltzmann_sample(sys::SpringSystem; rng=Random.default_rng())
    σ_p = sqrt(sys.m * sys.kB * sys.T)
    Float32.(σ_p .* randn(rng, sys.dim))
end

"""Exact log P(q) under the Boltzmann distribution (ground truth)."""
function boltzmann_logp(q::AbstractVector, sys::SpringSystem)
    σ² = sys.kB * sys.T / sys.k
    d  = length(q)
    -d/2 * log(2π * σ²) - sum(q.^2) / (2σ²)
end

"""Exact score (gradient of log P(q)) — used to validate CNF score."""
function boltzmann_score(q::AbstractVector, sys::SpringSystem)
    σ² = sys.kB * sys.T / sys.k
    -q ./ σ²
end

# ============================================================
# §2  Neural Network (DiffEqFlux FFJORD CNF)
# ============================================================
# DiffEqFlux.FFJORD wraps a Lux network f(z) as a Continuous Normalizing Flow.
# It handles the augmented ODE and log-density correction internally.

struct MLPNet
    dims::Vector{Int}   # e.g. [3, 64, 64, 3] for 3-D spring
    model               # Lux.Chain defining the FFJORD dynamics f(z)
    input_dims::Tuple{Int}
    tspan::Tuple{Float32, Float32}
    solver
    ad
    default_reltol::Float32
    default_abstol::Float32
    monte_carlo::Bool
end

"""Build an MLP with `n_hidden` hidden layers of size `hidden_dim`."""
function MLPNet(data_dim::Int, hidden_dim::Int; n_hidden::Int=2,
                tspan=(0f0, 1f0), solver=Tsit5(),
                reltol::Real=1f-4, abstol::Real=1f-4,
                ad=DiffEqFlux.AutoZygote(),
                monte_carlo::Bool=false)
    dims = [data_dim;
            fill(hidden_dim, n_hidden);
            data_dim]              # output: same dim as z
    n_layers = length(dims) - 1
    layers = [
        DiffEqFlux.Dense(dims[i] => dims[i+1], i < n_layers ? tanh : identity)
        for i in 1:n_layers
    ]
    MLPNet(dims, DiffEqFlux.Chain(layers...), (data_dim,),
           (Float32(tspan[1]), Float32(tspan[2])), solver, ad,
           Float32(reltol), Float32(abstol), monte_carlo)
end

"""Total number of scalar parameters in the MLP."""
function count_params(net::MLPNet)
    s = 0
    for i in 1:length(net.dims)-1
        s += net.dims[i] * net.dims[i+1] + net.dims[i+1]
    end
    s
end

"""Initialise FFJORD/Lux parameters as a ComponentArray."""
function init_params(net::MLPNet; scale=0.01f0, rng=Random.default_rng())
    ffjord = ffjord_layer(net)
    ps, _ = DiffEqFlux.Lux.setup(rng, ffjord)
    params = ComponentArrays.ComponentArray(ps)
    scale === nothing && return params
    params .* Float32(scale)
end

"""
Forward pass through the MLP.
  z      — state vector (Float32, length data_dim)
  τ      — accepted for compatibility; DiffEqFlux.FFJORD uses autonomous f(z)
  params — Lux/ComponentArray parameter tree
  net    — MLPNet struct with layer sizes
Returns a vector of the same size as z.
"""
function mlp_forward(z::AbstractVector, τ::Real, params, net::MLPNet)
    st = DiffEqFlux.Lux.initialstates(Random.default_rng(), net.model)
    y, _ = net.model(z, params, st)
    return y
end

# ============================================================
# §3  Continuous Normalizing Flow (CNF)
# ============================================================
# DiffEqFlux.FFJORD integrates the augmented CNF ODE:
#   dz/dτ = f(z, θ)
#   d(Δlog P_z)/dτ = -Tr(∂f/∂z)

function ffjord_layer(net::MLPNet; solver=net.solver,
                      reltol::Real=net.default_reltol,
                      abstol::Real=net.default_abstol)
    DiffEqFlux.FFJORD(net.model, net.tspan, net.input_dims, solver;
                      ad=net.ad, reltol=Float32(reltol), abstol=Float32(abstol))
end

function ffjord_state(ffjord, net::MLPNet; rng=Random.default_rng(),
                      regularize::Bool=false,
                      monte_carlo::Bool=net.monte_carlo)
    st = DiffEqFlux.Lux.initialstates(rng, ffjord)
    (; st..., regularize, monte_carlo)
end

"""log P(z) for z ~ N(0, I)."""
function standard_normal_logp(z::AbstractVector)
    d = length(z)
    -d/2 * log(2π) - sum(z.^2) / 2
end

"""
Compute log P(q) for a single data point q using DiffEqFlux.FFJORD.
"""
function cnf_loglikelihood(q::AbstractVector, params, net::MLPNet;
                           solver=Tsit5(), reltol=1f-4, abstol=1f-4,
                           monte_carlo::Bool=net.monte_carlo,
                           regularize::Bool=false,
                           sensealg=nothing)
    first(cnf_batch_loglikelihood(reshape(q, :, 1), params, net;
                                  solver=solver, reltol=reltol, abstol=abstol,
                                  monte_carlo=monte_carlo,
                                  regularize=regularize))
end

"""
Batch version: returns a vector of log-likelihoods, one per column of Q.
"""
function cnf_batch_loglikelihood(Q::AbstractMatrix, params, net::MLPNet;
                                  solver=Tsit5(), reltol=1f-4, abstol=1f-4,
                                  monte_carlo::Bool=net.monte_carlo,
                                  regularize::Bool=false,
                                  sensealg=nothing)
    ffjord = ffjord_layer(net; solver=solver, reltol=reltol, abstol=abstol)
    st = ffjord_state(ffjord, net; regularize=regularize, monte_carlo=monte_carlo)
    model = DiffEqFlux.StatefulLuxLayer{true}(ffjord, nothing, st)
    logpx, _, _ = model(Q, params)
    vec(logpx)
end

# ============================================================
# §4  Score function ∇_q log P(q)
# ============================================================
# This is the key bridge from the CNF to Hamiltonian dynamics:
#   F(q) = -∇V(q) = k_B T · ∇_q log P(q)
# We obtain it by differentiating the CNF log-likelihood w.r.t. q.

"""
Compute ∇_q log P(q) using the adjoint method (Zygote differentiates through
the ODE solver).
"""
function score_function(q::AbstractVector, params, net::MLPNet;
                        reltol=1f-3, abstol=1f-3,
                        monte_carlo::Bool=false,
                        sensealg=nothing)
    grad = Zygote.gradient(
        q_ -> cnf_loglikelihood(q_, params, net;
                                reltol=reltol, abstol=abstol,
                                monte_carlo=monte_carlo, sensealg=sensealg),
        q
    )[1]
    return grad
end

# ============================================================
# §5  Training loop
# ============================================================

"""
Train the CNF via maximum log-likelihood.
  Loss = -E_q[ log P_CNF(q) ]

Returns a vector of per-epoch losses.
"""
function train_cnf!(params, net::MLPNet, data::AbstractMatrix;
                    n_epochs::Int=200,
                    batch_size::Int=32,
                    lr::Real=1f-3,
                    ode_reltol::Float32=1f-3,
                    ode_abstol::Float32=1f-3,
                    monte_carlo::Bool=net.monte_carlo,
                    device=identity,
                    verbose::Bool=true)

    n_data = size(data, 2)
    train_data = device(data)
    train_params = device(params)
    opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), train_params)
    losses = Float32[]

    if verbose
        println("  Training data device: $(DiffEqFlux.Lux.get_device(train_data))")
        println("  Parameter device:     $(DiffEqFlux.Lux.get_device(train_params))")
    end

    for epoch in 1:n_epochs
        idx   = randperm(n_data)[1:min(batch_size, n_data)]
        batch = train_data[:, idx]

        loss, (∂params,) = Zygote.withgradient(train_params) do p
            logps = cnf_batch_loglikelihood(batch, p, net;
                        reltol=ode_reltol, abstol=ode_abstol,
                        monte_carlo=monte_carlo)
            -mean(logps)
        end

        opt_state, updated_params = Optimisers.update!(opt_state, train_params, ∂params)
        updated_params === train_params || (train_params .= updated_params)

        loss_value = loss isa AbstractArray ? only(DiffEqFlux.Lux.cpu_device()(loss)) : loss
        push!(losses, Float32(loss_value))

        if verbose && epoch % 10 == 0
            println("Epoch $(lpad(epoch,4)) | Loss: $(round(loss_value; digits=4))")
        end
    end

    train_params === params || (params .= DiffEqFlux.Lux.cpu_device()(train_params))

    return losses
end

# ============================================================
# §6  Hamiltonian Trajectory Inference
# ============================================================
# Hamilton's equations:
#   dq/dt =  ∂H/∂p  =  p / m
#   dp/dt = -∂H/∂q  =  F(q)  =  k_B T · ∇_q log P(q)
#
# Integrated with the leapfrog (Störmer-Verlet) method, which is
# symplectic and conserves a modified Hamiltonian exactly.

"""Minimal parameter bag for non-spring systems (e.g. proteins)."""
struct GenericSystem
    kB ::Float64
    T  ::Float64
    m  ::Float64
    dim::Int
end

GenericSystem(; kB=1.0, T=1.0, m=1.0, dim) = GenericSystem(kB, T, m, dim)

"""Sample momenta from Maxwell-Boltzmann for a GenericSystem."""
function maxwell_boltzmann_sample(sys::GenericSystem; rng=Random.default_rng())
    σ_p = sqrt(sys.m * sys.kB * sys.T)
    Float32.(σ_p .* randn(rng, sys.dim))
end

"""
Single leapfrog step.
  q, p   — current position and momentum
  F      — force function q → F(q)
  dt     — timestep
  m      — particle mass
"""
function leapfrog_step(q, p, F, dt, m)
    F_cur  = F(q)
    p_half = p  .+ (dt/2) .* F_cur    # half-step p
    q_new  = q  .+ dt     .* p_half ./ m
    F_new  = F(q_new)
    p_new  = p_half .+ (dt/2) .* F_new   # half-step p
    return q_new, p_new
end

"""
Infer a Hamiltonian trajectory of length `n_steps` starting from `q0`.

Steps (from paper §4):
  a. Compute log P(q(t)) via forward CNF integration
  b. Backpropagate (adjoint) to get ∇_q log P(q(t))  →  force F(q)
  c. Leapfrog update of (q, p) under Hamilton's equations

Accepts any system type with fields kB, T, m (SpringSystem or GenericSystem).
Returns (trajectory_q, trajectory_p, KE_trace).
"""
function infer_trajectory(q0::AbstractVector,
                          params,
                          net::MLPNet,
                          sys;                         # SpringSystem or GenericSystem
                          n_steps::Int=500,
                          dt::Float32=0.01f0,
                          score_reltol::Float32=1f-3,
                          score_abstol::Float32=1f-3)

    d   = length(q0)
    q   = Float32.(q0)
    p   = maxwell_boltzmann_sample(sys)      # Maxwell-Boltzmann init

    traj_q = zeros(Float32, d, n_steps+1)
    traj_p = zeros(Float32, d, n_steps+1)
    KE_trace = Float32[]

    traj_q[:, 1] = q
    traj_p[:, 1] = p

    # Force from CNF: F(q) = k_B T · ∇_q log P(q)
    force(q_) = Float32(sys.kB * sys.T) .*
                    score_function(q_, params, net;
                                   reltol=score_reltol, abstol=score_abstol)

    for step in 1:n_steps
        q, p = leapfrog_step(q, p, force, dt, Float32(sys.m))

        traj_q[:, step+1] = q
        traj_p[:, step+1] = p
        push!(KE_trace, sum(p.^2) / (2f0 * Float32(sys.m)))
    end

    return traj_q, traj_p, KE_trace
end

# ============================================================
# §8  Analysis helpers
# ============================================================

"""Print comparison statistics between the training data and inferred trajectory."""
function report_stats(train_data::AbstractMatrix,
                      traj_q::AbstractMatrix,
                      sys::SpringSystem)
    σ_true = sqrt(sys.kB * sys.T / sys.k)
    d      = size(train_data, 1)

    println("\n========= BoltzFlow — Distribution Statistics =========")
    @printf "Expected σ (Boltzmann distribution)  : %.4f\n" σ_true
    @printf "Training data σ  (per-dim mean)      : %.4f\n" mean(std(train_data; dims=2))
    @printf "Trajectory σ     (per-dim mean)      : %.4f\n" mean(std(traj_q;    dims=2))

    mean_KE          = mean(sum(traj_q.^2; dims=1) ./ (2sys.m))
    equipartition_KE = d * sys.kB * sys.T / 2
    @printf "\nMean KE from trajectory              : %.4f\n" mean_KE
    @printf "Expected KE (equipartition theorem)  : %.4f\n"  equipartition_KE
    println("=======================================================\n")
end

include("config.jl")
include("nbody_data.jl")
include("nbody_cnf.jl")
include("nbody_flow_matching.jl")
include("nbody_diffusion.jl")
include("experiments.jl")

end # module BoltzFlow
