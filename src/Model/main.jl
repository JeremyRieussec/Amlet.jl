@enum IsUpdatable Updatable NotUpdatable

abstract type AmletModel{U, D <: AbstractData} <: AbstractNLPModel{Float64, Vector{Float64}} end
function Nobs(mo::AmletModel)
    return length(mo.data)
end

include("Logit/main.jl")

#include("MixedLogit/main.jl")
