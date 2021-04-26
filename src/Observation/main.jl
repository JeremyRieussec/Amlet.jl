""""
`AbstractObs` contains the information of a single observation. Should have `computeUtilities` that computes the utilities for all the alternatives and the function `choice` that return what was the choice of the observation.
"""
abstract type AbstractObs end
abstract type AbstractPanelObs <: AbstractObs end


include("ObsAsMatrix.jl")
include("ObsAsVector.jl")
include("PanelObs.jl")