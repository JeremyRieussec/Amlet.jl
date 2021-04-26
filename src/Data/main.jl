"""
`AbstractData` contains the observation on which the model is based. Should implement `getindex`, `length`
"""
abstract type AbstractData end
abstract type AbstractPanelData <: AbstractData end

include("LinedObs.jl")
