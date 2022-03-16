
abstract type AbstractLogitUtility{L} <: AbstractUtility{L} end

struct StandardLogitUtility <: AbstractLogitUtility{Linear} end

function u(::Type{StandardLogitUtility}, x::AbstractVector, beta::AbstractVector, i::Int)
    return dot(x[access(length(beta), i)], beta)
end
function NLPModels.grad(::Type{StandardLogitUtility}, x::AbstractVector, beta::AbstractVector, i::Int)
    return x[access(length(beta), i)]
end
function hess(::Type{StandardLogitUtility}, x::AbstractVector, beta::AbstractVector{T}, i::Int) where T
    return zeros(T, length(beta), length(beta))
end