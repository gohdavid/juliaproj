"""
chignolin_demo.jl — BoltzFlow pipeline on Chignolin pairwise Cα distances.

Coordinate representation
-------------------------
q ∈ ℝ^45  where each element is log(d_ij) for the n_ca*(n_ca-1)/2 = 45 unique
Cα-Cα pairwise distances.  Log distances are:
  - Rotation- and translation-invariant (no alignment needed)
  - Strictly positive → log-transform makes them roughly Gaussian
  - Physically meaningful: folding events are visible as distance changes

After loading, we standardise to zero-mean, unit-variance per dimension so
the CNF base distribution N(0,I) is a reasonable prior.

Trajectory inference
--------------------
Hamilton's equations are integrated directly in the log-distance space.
q(t) is a sequence of log-distance vectors; p(t) are abstract momenta
conjugate to those coordinates.  kB = T = m = 1 in these reduced units,
consistent with the standardised representation.

Comparison
----------
We compare the marginal distribution of each pairwise distance (train vs
BoltzFlow trajectory) and report the mean absolute deviation in log-distance
per pair as a summary metric.

Run from the project root:
  julia --project=. scripts/chignolin_demo.jl
"""

# ── environment ───────────────────────────────────────────────────────────────
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))
using .BoltzFlow
using DelimitedFiles, Random, Statistics, LinearAlgebra, Printf
using Plots

Random.seed!(42)

# ── §1  Load data ─────────────────────────────────────────────────────────────
println("=== Step 1: Loading Chignolin pairwise Cα distances ===")

out_dir  = joinpath(@__DIR__, "..", "outputs", "chignolin_10us")
data_csv = joinpath(out_dir, "ca_distances.csv")

if !isfile(data_csv)
    error("""
    ca_distances.csv not found at: $data_csv
    Run the extraction script first:
        python scripts/extract_chignolin.py --stride 10
    """)
end

raw       = readdlm(data_csv, ',', Float32)   # (n_frames, n_pairs)
data_full = permutedims(raw)                   # (n_pairs, n_frames)
n_dim, n_frames = size(data_full)

# n_pairs = n_ca*(n_ca-1)/2  →  invert to get n_ca
n_ca = Int(round((1 + sqrt(1 + 8 * n_dim)) / 2))

println("  Frames:          $n_frames")
println("  Cα atoms:        $n_ca")
println("  Pairwise distances: $n_dim  ($(n_ca)×($(n_ca)-1)/2)")
@printf "  Log-dist range:  [%.3f, %.3f]  →  [%.2f, %.2f] Å\n" minimum(data_full) maximum(data_full) exp(minimum(data_full)) exp(maximum(data_full))

# ── §2  Train / validation split + standardise ────────────────────────────────
println("\n=== Step 2: Preprocessing ===")

n_train = min(5_000, floor(Int, 0.9 * n_frames))
n_val   = min(500,   n_frames - n_train)

perm      = randperm(n_frames)
train_idx = perm[1:n_train]
val_idx   = perm[n_train+1 : n_train+n_val]

train_raw = data_full[:, train_idx]
val_raw   = data_full[:, val_idx]

# Standardise using training statistics (in log-distance space)
data_mean = mean(train_raw; dims=2)            # (n_dim, 1)
data_std  = std(train_raw;  dims=2) .+ 1f-8    # guard against zero-std dims

train_data = (train_raw .- data_mean) ./ data_std
val_data   = (val_raw   .- data_mean) ./ data_std

println("  Training frames:   $n_train")
println("  Validation frames: $n_val")
@printf "  Standardised range: [%.3f, %.3f]\n" minimum(train_data) maximum(train_data)

# Save standardisation stats for later denormalisation
mkpath(out_dir)
writedlm(joinpath(out_dir, "dist_mean.csv"), vec(data_mean)', ',')
writedlm(joinpath(out_dir, "dist_std.csv"),  vec(data_std)',  ',')
println("  Saved dist_mean.csv and dist_std.csv")

# ── §3  Build and train the CNF ───────────────────────────────────────────────
println("\n=== Step 3: Training CNF on pairwise distances ===")

# 45-D input → hidden layers of 256 → 45-D output
# Wider / deeper than spring demo to handle richer 45-D distribution.
hidden_dim = 256
n_hidden   = 3
net    = MLPNet(n_dim, hidden_dim; n_hidden=n_hidden)
params = init_params(net)

println("  MLP dims:     $(net.dims)")
println("  Total params: $(count_params(net))")

losses = train_cnf!(params, net, train_data;
    n_epochs   = 500,
    batch_size = 64,
    lr         = 1f-3,
    ode_reltol = 1f-3,
    ode_abstol = 1f-3,
    verbose    = true)

# Save trained parameters
params_path = joinpath(out_dir, "cnf_params.csv")
writedlm(params_path, params, ',')
println("\n  Saved CNF parameters → $params_path")

p_loss = plot(losses;
              xlabel="Epoch", ylabel="NLL Loss",
              title="Chignolin CNF Training Loss (pairwise distances)",
              legend=false, lw=1.5)
savefig(p_loss, joinpath(@__DIR__, "chignolin_training_loss.png"))
println("  Loss curve saved.")

# ── §4  Validation log-likelihoods ────────────────────────────────────────────
println("\n=== Step 4: Validation log-likelihoods ===")

n_eval = min(50, n_val)
train_logps = [cnf_loglikelihood(train_data[:, i], params, net;
                   reltol=1f-3, abstol=1f-3) for i in 1:n_eval]
val_logps   = [cnf_loglikelihood(val_data[:, i],   params, net;
                   reltol=1f-3, abstol=1f-3) for i in 1:n_eval]

@printf "  Mean log P — training subset:   %.4f\n" mean(train_logps)
@printf "  Mean log P — validation subset: %.4f\n" mean(val_logps)
@printf "  Generalisation gap:             %.4f\n" (mean(train_logps) - mean(val_logps))

# ── §5  Hamiltonian trajectory inference ─────────────────────────────────────
println("\n=== Step 5: Inferring Hamiltonian trajectory (distance space) ===")

# kB=1, T=1, m=1 in standardised log-distance units.
# Force F(q) = ∇_q log P(q)  (kBT = 1 so the factor drops out).
sys = GenericSystem(kB=1.0, T=1.0, m=1.0, dim=n_dim)

q_init       = val_data[:, 1]    # a real held-out conformation
n_traj_steps = 500

println("  Starting from held-out frame, running $n_traj_steps leapfrog steps ...")

traj_q, traj_p, KE_trace = infer_trajectory(
    q_init, params, net, sys;
    n_steps      = n_traj_steps,
    dt           = 0.005f0,
    score_reltol = 1f-3,
    score_abstol = 1f-3,
)

println("  Trajectory shape: $(size(traj_q))")

# Denormalise: standardised log-dist → log-dist → Angstrom
traj_logdist = traj_q .* data_std .+ data_mean    # (n_dim, n_steps+1) log-Å
traj_dist_ang = exp.(traj_logdist)                 # (n_dim, n_steps+1) Å

# Save (in Angstrom)
writedlm(joinpath(out_dir, "hmd_trajectory_distances.csv"),
         permutedims(traj_dist_ang), ',')
println("  Saved trajectory distances (Å) → hmd_trajectory_distances.csv")

# ── §6  Statistics comparison ──────────────────────────────────────────────────
println("\n=== Step 6: Distribution statistics ===")

# Denormalise training data to Angstrom for comparison
train_dist_ang = exp.(train_raw)    # (n_dim, n_train)

# Mean absolute deviation per pair (in Angstrom)
traj_mean_dist  = mean(traj_dist_ang; dims=2)      # (n_dim, 1)
train_mean_dist = mean(train_dist_ang; dims=2)     # (n_dim, 1)
mad_per_pair    = abs.(traj_mean_dist .- train_mean_dist)   # mean Å deviation per pair
overall_mad     = mean(mad_per_pair)

println("\n========= BoltzFlow — Chignolin Distance Statistics =========")
@printf "Mean absolute distance deviation (traj vs train): %.3f Å\n" overall_mad
@printf "Max absolute distance deviation:                  %.3f Å\n" maximum(mad_per_pair)

mean_KE  = mean(KE_trace)
equip_KE = n_dim * sys.kB * sys.T / 2
@printf "\nMean KE along trajectory:    %.4f\n" mean_KE
@printf "Equipartition expected KE:   %.4f\n"  equip_KE
@printf "Ratio (≈1 if thermalised):   %.4f\n"  (mean_KE / equip_KE)
println("=============================================================\n")

# ── §7  Plots ─────────────────────────────────────────────────────────────────
println("=== Step 7: Saving plots ===")

# KE trace
p_ke = plot(KE_trace;
            xlabel="Leapfrog step", ylabel="KE (reduced units)",
            title="Hamiltonian Trajectory KE  (equipartition ≈ $(round(equip_KE; digits=1)))",
            legend=false, lw=0.8)
hline!(p_ke, [equip_KE]; linestyle=:dash, color=:red)
savefig(p_ke, joinpath(@__DIR__, "chignolin_kinetic_energy.png"))

# Mean distance per pair: training vs trajectory
p_mean = scatter(vec(train_mean_dist), vec(traj_mean_dist);
                 xlabel="Training mean dist (Å)", ylabel="Trajectory mean dist (Å)",
                 title="Per-pair mean distance: train vs trajectory",
                 legend=false, alpha=0.7, ms=4)
lims = extrema(vcat(vec(train_mean_dist), vec(traj_mean_dist)))
plot!(p_mean, collect(lims), collect(lims); linestyle=:dash, color=:red, lw=1)
savefig(p_mean, joinpath(@__DIR__, "chignolin_mean_distance_comparison.png"))

# Histograms for the 4 most variable distances in training data
pair_var  = vec(var(train_dist_ang; dims=2))
top4_idx  = sortperm(pair_var; rev=true)[1:4]

# Load pair labels if available
labels_csv = joinpath(out_dir, "ca_pair_labels.csv")
pair_labels = if isfile(labels_csv)
    raw_labels = readdlm(labels_csv, ',', String; skipstart=1)
    ["$(raw_labels[k,3])-$(raw_labels[k,4])" for k in axes(raw_labels, 1)]
else
    ["pair $k" for k in 1:n_dim]
end

p_hist = plot(layout=(2, 2), size=(900, 700))
for (pi, k) in enumerate(top4_idx)
    histogram!(p_hist[pi], train_dist_ang[k, :];
               bins=40, normalize=:pdf, alpha=0.6, label="MD train",
               title=pair_labels[k])
    histogram!(p_hist[pi], traj_dist_ang[k, :];
               bins=40, normalize=:pdf, alpha=0.6, label="BoltzFlow")
    xlabel!(p_hist[pi], "Distance (Å)")
end
savefig(p_hist, joinpath(@__DIR__, "chignolin_distance_histograms.png"))

println("  Plots saved to scripts/")
println("\nDone.")
