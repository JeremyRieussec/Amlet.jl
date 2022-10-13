"""
`struct ObsAsVector` represent data as a `Vector` where attributes of a possible alternative are one after the other. The parameters of the
choice made by the observation are the first ones.

# Fields
- `data::AbstractVector` :  attribute vectors associated with each alternative are concatenated as one vector. First one is the chosen alternative.
- `nalt::Int` : number of alternatives faced by individual.
- `nsim::Int` : number of similar individuals in this configuration.
"""
struct ObsAsVector{V} <: AbstractObs{V}
    data::V
    nalt::Int
    nsim::Int
    function ObsAsVector(data::V, nalt::Int, nsim::Int = 1) where V
        return new{V}(data, nalt, nsim)
    end
    function ObsAsVector{V}(data::V, nalt::Int, nsim::Int = 1) where V
        return new{V}(data, nalt, nsim)
    end
end

"""
    nalt(obs::ObsAsVector)

Returns the number of alternatives faced by individual.
"""
function nalt(obs::ObsAsVector)
    return obs.nalt
end
function dim(obs::ObsAsVector)
    return div(length(obs.data), nalt(obs))
end


#=
"""
    computeUtilities(x::Vector, obs::ObsAsMatrix, u::MixedLogitUtility, gamma::Vector)

Computes utility value for every alternative in a MixedLogit context -> returns an array.
"""
function computeUtilities(beta::AbstractArray{T}, obs::ObsAsVector, u::MixedLogitUtility, gamma::Vector) where T
    # NP = Int(length(obs.data)/obs.nalt) --> Not used
    # utilities = Array{Float64, 1}(undef, nalt(obs))
    # for i in 1:nalt(obs)
    #     utilities[i] = u.u(obs.data, beta, gamma, i)
    # end
    # return utilities
    return T[u.u(obs.data, beta, gamma, i) for i in 1:nalt(obs)]
end
=#
"""
    choice(obs::ObsAsVector)

Returns alternative chosen by individual (by default returs `1` because data is formatted to have choice on first row).
"""
function choice(obs::ObsAsVector)
    return 1
end


"""
    nsim(obs::ObsAsVector)

Returns number of similar individuals with these observations.
"""
function nsim(obs::ObsAsVector)
    return obs.nsim
end

function access(n::Int, m::Int)
    return (m-1)*n+1:n*m
end

function explanatory(obs::ObsAsVector, i::Int)
    data = obs.data
    n = dim(obs)
    @view data[access(n, i)]

end

function explanatorylength(l::ObsAsVector)
    n = length(l.data)
    return div(n, l.nalt)
end