
abstract type AbstractLogitUtility{L} <: AbstractUtility{L} end

struct StandardLogitUtility <: AbstractLogitUtility{Linear} end

function u(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector, i::Int)
    return dot(explanatory(obs, i), beta)
end
function NLPModels.grad(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector, i::Int)
    return explanatory(obs, i)
end
function NLPModels.hess(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector{T}, i::Int) where T
    @warn "Hessian of linear utility called"
    lb = length(beta)
    return zeros(T, lb, lb)
end
function hessdotv(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector{T}, i::Int, v::Vector) where T
    @warn "Hessian of linear utility called"
    lb = length(beta)
    return zeros(T, lb)
end


"""
    computeUtilities(x::Vector, obs::ObsAsVector, u::LogitUtility)

Computes utility value for every alternative in a Logit context -> returns an array.
"""
function computeUtilities(::Type{UTI}, obs::ObsAsVector, beta::AbstractArray{T}) where {T, UTI <: AbstractLogitUtility}
    n = nalt(obs)
    ar = Array{T, 1}(undef, n)
    #for some reason, faster than [u(UTI, obs, beta, i) for i in 1:nalt(obs)]???
    for i in 1:n
        ar[i] = dot(explanatory(obs, i), beta)
    end
    return ar
end

function dim(::Type{UTI}, s::AbstractData) where {UTI <: AbstractLogitUtility}
    n = explanatorylength(s)
    return n 
end
