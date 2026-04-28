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
using SciMLSensitivity
using Zygote
using ForwardDiff

export SpringSystem, generate_equilibrium_data, maxwell_boltzmann_sample
export boltzmann_logp, boltzmann_score
export MLPNet, count_params, init_params, mlp_forward
export cnf_loglikelihood, score_function
export AdamState, init_adam, update_adam!
export train_cnf!
export infer_trajectory, leapfrog_step, GenericSystem
export compare_distributions, report_stats

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
# §2  Neural Network (flat-parameter MLP)
# ============================================================
# Representing the network parameters as a flat vector makes the
# model fully compatible with OrdinaryDiffEq / SciMLSensitivity.
#
# Architecture:  [z; τ]  →  Linear → tanh → ... → Linear  →  z
# Input dimension = data_dim + 1  (appended pseudo-time τ)

struct MLPNet
    dims::Vector{Int}   # e.g. [4, 64, 64, 3] for 3-D spring
end

"""Build an MLP with `n_hidden` hidden layers of size `hidden_dim`."""
function MLPNet(data_dim::Int, hidden_dim::Int; n_hidden::Int=2)
    dims = [data_dim + 1;          # input: z concatenated with τ
            fill(hidden_dim, n_hidden);
            data_dim]              # output: same dim as z
    MLPNet(dims)
end

"""Total number of scalar parameters in the MLP."""
function count_params(net::MLPNet)
    s = 0
    for i in 1:length(net.dims)-1
        s += net.dims[i] * net.dims[i+1] + net.dims[i+1]
    end
    s
end

"""Initialise parameters with small random values."""
function init_params(net::MLPNet; scale=0.01f0, rng=Random.default_rng())
    scale .* randn(rng, Float32, count_params(net))
end

"""
Forward pass through the MLP.
  z      — state vector (Float32, length data_dim)
  τ      — pseudo-time scalar
  params — flat parameter vector
  net    — MLPNet struct with layer sizes
Returns a vector of the same size as z.
"""
function mlp_forward(z::AbstractVector, τ::Real, params::AbstractVector, net::MLPNet)
    x      = vcat(z, eltype(z)[τ])
    offset = 0
    n_layers = length(net.dims) - 1

    for i in 1:n_layers
        d_in  = net.dims[i]
        d_out = net.dims[i+1]
        n_w   = d_in * d_out

        W = reshape(params[offset+1 : offset+n_w],          d_out, d_in)
        b =         params[offset+n_w+1 : offset+n_w+d_out]
        x = W * x + b

        if i < n_layers          # tanh on hidden layers, linear on output
            x = tanh.(x)
        end
        offset += n_w + d_out
    end
    return x
end

# ============================================================
# §3  Continuous Normalizing Flow (CNF)
# ============================================================
# Augmented ODE state: u = [z (dim d); Δlog P_z (dim 1)]
#
# Dynamics:
#   dz/dτ          =  f(z, τ, θ)
#   d(Δlog P_z)/dτ = -Tr(∂f/∂z)
#
# Trace estimated with Hutchinson's estimator:
#   Tr(J) ≈ εᵀ J ε   where  ε ~ Rademacher{±1}^d
# and  J ε  is the Jacobian-vector product computed via ForwardDiff.

"""
Core CNF dynamics with `net` as an explicit argument (not an ODE parameter).
Used internally to build closures that pass only `params` as the ODE `p`.
"""
function cnf_dynamics_core(u, params::AbstractVector, τ, net::MLPNet)
    d = length(u) - 1
    z = u[1:d]

    # Neural-network velocity field
    fz = mlp_forward(z, τ, params, net)

    # Hutchinson trace estimate of div f  (single Rademacher probe)
    ε   = sign.(randn(eltype(z), d))
    jvp = ForwardDiff.derivative(
        ξ -> mlp_forward(z .+ ξ .* ε, τ, params, net),
        zero(eltype(z))
    )                              # J ε  via forward-mode AD
    trace_est = dot(ε, jvp)       # εᵀ J ε ≈ Tr(J)

    vcat(fz, eltype(z)[-trace_est])
end

"""log P(z) for z ~ N(0, I)."""
function standard_normal_logp(z::AbstractVector)
    d = length(z)
    -d/2 * log(2π) - sum(z.^2) / 2
end

"""
Compute log P(q) for a single data point q by integrating the augmented CNF ODE
from τ = 0 → 1.

The ODE parameter is only `params` (an AbstractVector), satisfying SciMLSensitivity's
requirement. `net` is closed over.
"""
function cnf_loglikelihood(q::AbstractVector, params::AbstractVector, net::MLPNet;
                           solver=Tsit5(), reltol=1f-4, abstol=1f-4,
                           sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    d  = length(q)
    u0 = vcat(q, zero(eltype(q))[])   # [z(0)=q; Δlogp(0)=0]

    # Close over net; only params flows as the differentiable ODE parameter
    dynamics = (u, p, τ) -> cnf_dynamics_core(u, p, τ, net)

    prob = ODEProblem(dynamics, u0, (0f0, 1f0), params)
    sol  = solve(prob, solver;
                 reltol=reltol, abstol=abstol,
                 sensealg=sensealg,
                 save_everystep=false, save_start=false)

    u1    = sol[:, end]
    z1    = u1[1:d]
    Δlogp = u1[d+1]

    # log P(q) = log P(z(1)) − Δlog P_z(1)
    standard_normal_logp(z1) - Δlogp
end

"""
Batch version: returns a vector of log-likelihoods, one per column of Q.
"""
function cnf_batch_loglikelihood(Q::AbstractMatrix, params::AbstractVector,
                                  net::MLPNet; kwargs...)
    [cnf_loglikelihood(Q[:, i], params, net; kwargs...) for i in axes(Q, 2)]
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
function score_function(q::AbstractVector, params::AbstractVector, net::MLPNet;
                        reltol=1f-3, abstol=1f-3,
                        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    grad = Zygote.gradient(
        q_ -> cnf_loglikelihood(q_, params, net;
                                reltol=reltol, abstol=abstol, sensealg=sensealg),
        q
    )[1]
    return grad
end

# ============================================================
# §5  Adam Optimiser (minimal, no external dependency)
# ============================================================

mutable struct AdamState
    m  ::Vector{Float32}   # first moment
    v  ::Vector{Float32}   # second moment
    t  ::Int               # step counter
    lr ::Float32
    β1 ::Float32
    β2 ::Float32
    ε  ::Float32
end

function init_adam(params::AbstractVector, lr::Real=1f-3)
    n = length(params)
    AdamState(zeros(Float32, n), zeros(Float32, n), 0,
              Float32(lr), 0.9f0, 0.999f0, 1f-8)
end

function update_adam!(params::AbstractVector, grads::AbstractVector, st::AdamState)
    st.t += 1
    @. st.m = st.β1 * st.m + (1 - st.β1) * grads
    @. st.v = st.β2 * st.v + (1 - st.β2) * grads^2
    m̂ = st.m ./ (1 - st.β1^st.t)
    v̂ = st.v ./ (1 - st.β2^st.t)
    @. params -= st.lr * m̂ / (sqrt(v̂) + st.ε)
end

# ============================================================
# §6  Training loop
# ============================================================

"""
Train the CNF via maximum log-likelihood.
  Loss = -E_q[ log P_CNF(q) ]

Returns a vector of per-epoch losses.
"""
function train_cnf!(params::AbstractVector, net::MLPNet, data::AbstractMatrix;
                    n_epochs::Int=200,
                    batch_size::Int=32,
                    lr::Real=1f-3,
                    ode_reltol::Float32=1f-3,
                    ode_abstol::Float32=1f-3,
                    verbose::Bool=true)

    n_data = size(data, 2)
    opt    = init_adam(params, lr)
    losses = Float32[]

    for epoch in 1:n_epochs
        idx   = randperm(n_data)[1:min(batch_size, n_data)]
        batch = data[:, idx]

        loss, (∂params,) = Zygote.withgradient(params) do p
            logps = cnf_batch_loglikelihood(batch, p, net;
                        reltol=ode_reltol, abstol=ode_abstol,
                        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
            -mean(logps)
        end

        update_adam!(params, ∂params, opt)
        push!(losses, loss)

        if verbose && epoch % 10 == 0
            println("Epoch $(lpad(epoch,4)) | Loss: $(round(loss; digits=4))")
        end
    end

    return losses
end

# ============================================================
# §7  Hamiltonian Trajectory Inference
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
                          params::AbstractVector,
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

end # module BoltzFlow
