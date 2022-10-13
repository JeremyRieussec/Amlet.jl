@enum IsUpdatable Updatable NotUpdatable

abstract type AmletModel{U, D <: AbstractData} <: AbstractStochasticNLPModels{Float64, Vector{Float64}} end

function nobs(mo::AmletModel)
    return length(mo.data)
end

include("Logit/main.jl")

include("MixedLogit/main.jl")
    