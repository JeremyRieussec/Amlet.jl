"""
`struct ObsAsVector` represent data as a `Vector` where attributes of a possible alternative are one after the other. The parameters of the
choice made by the observation are the first ones.

# Fields
- `data::AbstractVector{Float64}` :  attribute vectors associated with each alternative are concatenated as one vector. First one is the chosen alternative.
- `nalt::Int` : number of alternatives faced by individual.
- `nsim::Int` : number of similar individuals in this configuration.
"""
struct ObsAsVector <: AbstractObs
    data::AbstractVector{Float64}
    nalt::Int
    nsim::Int
    function ObsAsVector(data::AbstractVector{Float64}, nalt::Int, nsim::Int = 1)
        return new(data, nalt, nsim)
    end
end

"""
    nalt(obs::ObsAsVector)

Returns the number of alternatives faced by individual.
"""
function nalt(obs::ObsAsVector)
    return obs.nalt
end


"""
    computeUtilities(x::Vector, obs::ObsAsVector, u::LogitUtility)

Computes utility value for every alternative in a Logit context -> returns an array.
"""
function computeUtilities(beta::AbstractArray{T}, obs::ObsAsVector, u::LogitUtility) where T
    # NP = Int(length(obs.data)/obs.nalt) --> this is not used
    # utilities = Array{Float64, 1}(undef, nalt(obs))
    # for i in 1:nalt(obs)
    #     utilities[i] = u.u(obs.data, beta, i)
    # end
    # return utilities
    return T[u.u(obs.data, beta, i) for i in 1:nalt(obs)]
end

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
