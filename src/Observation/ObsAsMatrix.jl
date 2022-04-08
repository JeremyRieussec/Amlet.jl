"""
`struct ObsAsMatrix`

# Fields
- `data::Array{Float64, 2}` : represent data as a `Matrix` where each row represents a choice, the choice made by individual is assumed to be the first row.
- `nsim::Int` : number of similar individuals
"""
struct ObsAsMatrix <: AbstractObs
    data::Array{Float64, 2}
    nsim::Int
    function ObsAsMatrix(data::Array{Float64, 2}, nsim::Int = 1) where T
        return new(data, nsim)
    end
end

"""
    nalt(obs::ObsAsMatrix)

Returns the number of alternatives faced by individual.
"""
function nalt(obs::ObsAsMatrix)
    return size(obs.data, 1)
end

#=
"""
    computeUtilities(x::Vector, obs::ObsAsMatrix, u::MixedLogitUtility, gamma::Vector)

Computes utility value for every alternative in a MixedLogit context -> returns an array.
"""
function computeUtilities(x::Vector, obs::ObsAsMatrix, u::MixedLogitUtility, gamma::Vector)
    return [u.u(obs_i, x, gamma) for obs_i in eachrow(obs.data)]
end
=#
"""
    choice(obs::ObsAsMatrix)

Returns alternative chosen by individual (by default returs `1` because data is formatted to have choice on first row).
"""
function choice(obs::ObsAsMatrix)
    return 1
end

"""
    nsim(obs::ObsAsMatrix)

Returns number of similar individuals with these observations.
"""
function nsim(obs::ObsAsMatrix)
    return obs.nsim
end
