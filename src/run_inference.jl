#!/usr/bin/env julia

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia --project=. src/run_inference.jl <config.yaml>")

    project_root = dirname(@__DIR__)
    push!(LOAD_PATH, project_root)
    include(joinpath(@__DIR__, "BoltzFlow.jl"))
    using .BoltzFlow

    BoltzFlow.run_inference(ARGS[1])
end
