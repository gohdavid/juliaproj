"""Rouse-chain and n-body generative modeling utilities."""
module BoltzFlow

using LinearAlgebra, Random, Statistics
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

export load_yaml_config, run_experiment, run_nbody_experiment
export run_nbody_flow_matching_experiment, run_nbody_diffusion_experiment
export NBodyDataConfig, generate_nbody_dataset, sample_training_batch
export MLPVectorField, EGNNVectorField, NBodyCNFContext
export build_vector_field, init_cnf_params, cnf_logp, cnf_logp_gradient
export train_cnf_adam, generate_cnf_samples
export EquivariantFMVectorField, NBodyFlowMatchingContext, FlowMatchingResult
export build_fm_vector_field, init_fm_params, train_flow_matching_adam
export generate_flow_matching_samples, center_positions, pairwise_distance_mae
export EquivariantDiffusionModel, NBodyDiffusionContext, DiffusionResult
export build_diffusion_model, init_diffusion_params, train_diffusion_adam
export generate_diffusion_samples, diffusion_logp_gradient
export config_hash, config_output_dir
export polymer_langevin_force!, polymer_langevin_potential
export polymer_langevin_score!, polymer_nonideal_params

include("config.jl")
include("nbody_data.jl")
include("nbody_cnf.jl")
include("nbody_flow_matching.jl")
include("nbody_diffusion.jl")
include("experiments.jl")

end # module BoltzFlow
