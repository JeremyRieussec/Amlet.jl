abstract type isLinear end
struct Linear <: isLinear end
struct NotLinear <: isLinear end

function access(n::Int, m::Int)
    return (m-1)*n+1:n*m
end

abstract type AbstractUtility{T<: isLinear} end


include("LogitUtility.jl")

include("MixedLogitUtility.jl")