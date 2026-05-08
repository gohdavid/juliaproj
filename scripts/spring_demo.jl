"""
spring_demo.jl  —  End-to-end BoltzFlow demo on a 3-D harmonic oscillator.

Pipeline:
  1. Generate equilibrium snapshots from the Boltzmann distribution
  2. Train a CNF to recover P(q)
  3. Validate: compare learned log P(q) against ground truth
  4. Infer a Hamiltonian trajectory from a single initial state
  5. Compare trajectory statistics against the Boltzmann distribution
"""

# ── project environment ────────────────────────────────────────────────────────
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))
using .BoltzFlow
using Random, Statistics, LinearAlgebra
using Plots
using CUDA

Random.seed!(42)
CUDA.functional() || error("CUDA is not functional; this GPU runtime is required for training.")

# ── §1  System and data ────────────────────────────────────────────────────────
println("=== Step 1: Generating equilibrium data ===")

sys = SpringSystem(k=2.0, m=1.0, T=1.0, kB=1.0, dim=3)

N_train = 2_000
data    = generate_equilibrium_data(sys, N_train)

println("  Spring constant k = $(sys.k),  temperature T = $(sys.T)")
println("  Expected σ = $(round(sqrt(sys.kB * sys.T / sys.k); digits=4))")
println("  Data σ     = $(round(mean(std(data; dims=2)); digits=4))")
println("  Generated $N_train training snapshots in ℝ$(sys.dim)")

# ── §2  Build and initialise the CNF ──────────────────────────────────────────
println("\n=== Step 2: Building CNF model ===")

net    = MLPNet(sys.dim, 64; n_hidden=2)   # architecture: [3, 64, 64, 3]
params = init_params(net)
train_device = CUDA.cu

println("  MLP dims:      $(net.dims)")
println("  Total params:  $(count_params(net))")
println("  Training via:  CUDA.cu")

# ── §3  Train ─────────────────────────────────────────────────────────────────
println("\n=== Step 3: Training CNF ===")

losses = train_cnf!(params, net, data;
    n_epochs   = 200,
    batch_size = 32,
    lr         = 1f-3,
    ode_reltol = 1f-3,
    ode_abstol = 1f-3,
    monte_carlo = true,  # GPU-friendly FFJORD trace estimator
    device     = train_device,
    verbose    = true)

p_loss = plot(losses;
              xlabel="Epoch", ylabel="NLL Loss",
              title="CNF Training Loss (Spring System)",
              legend=false, lw=1.5)
savefig(p_loss, joinpath(@__DIR__, "training_loss.png"))
println("  Loss curve saved to scripts/training_loss.png")

# ── §4  Validate: compare learned vs true log P(q) ───────────────────────────
println("\n=== Step 4: Validating learned log-likelihood ===")

N_val      = 20
val_data   = generate_equilibrium_data(sys, N_val)

learned_logps = [cnf_loglikelihood(val_data[:, i], params, net;
                     reltol=1f-4, abstol=1f-4) for i in 1:N_val]
true_logps    = [boltzmann_logp(val_data[:, i], sys) for i in 1:N_val]

println("  Correlation (learned vs true): ",
        round(cor(learned_logps, true_logps); digits=4))
println("  Mean absolute error:           ",
        round(mean(abs.(learned_logps .- true_logps)); digits=4))

p_ll = scatter(true_logps, learned_logps;
               xlabel="True log P(q)", ylabel="CNF log P(q)",
               title="Log-Likelihood Validation", legend=false)
plot!(p_ll, x -> x, linestyle=:dash, lw=1, color=:red)   # y=x reference
savefig(p_ll, joinpath(@__DIR__, "loglikelihood_validation.png"))
println("  Validation plot saved to scripts/loglikelihood_validation.png")

# ── §5  Score (force) validation ──────────────────────────────────────────────
println("\n=== Step 5: Validating score function (forces) ===")

q_test    = val_data[:, 1]
s_learned = score_function(q_test, params, net)
s_true    = boltzmann_score(q_test, sys)

println("  q_test  : $(round.(q_test; digits=3))")
println("  ∇logP (true)   : $(round.(s_true;    digits=3))")
println("  ∇logP (learned): $(round.(s_learned; digits=3))")
println("  Cosine sim:  ",
        round(dot(s_learned, s_true) / (norm(s_learned)*norm(s_true)); digits=4))

# ── §6  Hamiltonian trajectory inference ──────────────────────────────────────
println("\n=== Step 6: Inferring Hamiltonian trajectory ===")

q_init      = generate_equilibrium_data(sys, 1)[:, 1]
n_traj_steps = 1_000

traj_q, traj_p, KE_trace = infer_trajectory(
    q_init, params, net, sys;
    n_steps     = n_traj_steps,
    dt          = 0.01f0,
    score_reltol = 1f-3,
    score_abstol = 1f-3
)

println("  Trajectory length: $(n_traj_steps) steps (dt=0.01)")

# ── §7  Statistics comparison ──────────────────────────────────────────────────
report_stats(data, traj_q, sys)

# ── §8  Plots ──────────────────────────────────────────────────────────────────
println("=== Step 7: Saving plots ===")

# Trajectory in x-y plane
p_traj = plot(traj_q[1, :], traj_q[2, :];
              xlabel="q_x", ylabel="q_y",
              title="Inferred Hamiltonian Trajectory (x-y)",
              legend=false, alpha=0.6, lw=0.7)
scatter!(p_traj, [q_init[1]], [q_init[2]]; marker=:star5, ms=8,
         label="start", color=:red)
savefig(p_traj, joinpath(@__DIR__, "trajectory_xy.png"))

# KE time series  (should fluctuate around  d/2 · k_B T = 1.5)
p_ke = plot(KE_trace;
            xlabel="Step", ylabel="Kinetic Energy",
            title="KE along Trajectory  (equipartition: $(sys.dim/2 * sys.kB * sys.T))",
            legend=false, lw=0.8)
hline!(p_ke, [sys.dim/2 * sys.kB * sys.T]; linestyle=:dash, color=:red,
       label="equipartition")
savefig(p_ke, joinpath(@__DIR__, "kinetic_energy.png"))

# Per-dimension histograms: training data vs trajectory
p_hist = plot(layout=(1, sys.dim))
for k in 1:sys.dim
    histogram!(p_hist[k], data[k, :]; bins=40, normalize=:pdf,
               alpha=0.5, label="train", title="dim $k")
    histogram!(p_hist[k], traj_q[k, :]; bins=40, normalize=:pdf,
               alpha=0.5, label="traj")
end
savefig(p_hist, joinpath(@__DIR__, "histogram_comparison.png"))

println("  Plots saved to scripts/")
println("\nDone.")
