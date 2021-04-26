"""
`obsAsVector` represent data as a Vector where parameters of a possbile alternative are one after the other. The parameters of the 
choice made by the observation are the first ones.
"""
struct ObsAsVector <: AbstractObs
    data::AbstractVector{Float64}
    nalt::Int 
    nsim::Int
    function ObsAsVector(data::AbstractVector{Float64}, nalt::Int, nsim::Int = 1)
        return new(data, nalt, nsim)
    end
end
function nalt(obs::ObsAsVector)
    return obs.nalt
end
function computeUtilities(x::Vector, obs::ObsAsVector, u::LogitUtility)
    NP = Int(length(obs.data)/obs.nalt)
    utilities = Array{Float64, 1}(undef, nalt(obs))
    for i in 1:nalt(obs)
        utilities[i] = u.u(obs.data, x, i)
    end
    return utilities
end
function computeUtilities(x::Vector, obs::ObsAsVector, u::MixedLogitUtility, gamma::Vector)
    NP = Int(length(obs.data)/obs.nalt)
    utilities = Array{Float64, 1}(undef, nalt(obs))
    for i in 1:nalt(obs)
        utilities[i] = u.u(x.data, x, gamma, i)
    end
    return utilities
end
        
function choice(obs::ObsAsVector)
    return 1
end
function nsim(obs::ObsAsVector)
    return obs.nsim
end
