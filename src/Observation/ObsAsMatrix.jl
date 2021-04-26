"""
`ObsAsMatrix` represent data as a Matrix where each row represent a choice, the choice is assumed to be the first row.
"""

struct ObsAsMatrix <: AbstractObs
    data::Array{Float64, 2}
    nsim::Int
    function ObsAsMatrix(data::Array{Float64, 2}, nsim::Int = 1) where T
        return new(data, nsim)
    end
end
function nalt(obs::ObsAsMatrix)
    return size(obs, 1)
end

function computeUtilities(x::Vector, obs::ObsAsMatrix, u::LogitUtility)
    return [u.u(obs.data, x, i) for i in 1:nalt(obs)]
end
function computeUtilities(x::Vector, obs::ObsAsMatrix, u::MixedLogitUtility, gamma::Vector)
    return [u.u(obs_i, x, gamma) for obs_i in eachrow(obs.data)]
end
function choice(obs::ObsAsMatrix)
    return 1
end
function nsim(obs::ObsAsMatrix)
    return obs.nsim
end
