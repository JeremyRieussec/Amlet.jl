
struct PanelObsAsVector <: AbstractPanelObs
    data::AbstractVector{Float64}
    nalt::Int
    nsim::Int
    nseq::Int
    function PanelObsAsVector(data::AbstractVector{Float64}, nalt::Int, nseq::Int, nsim::Int = 1)
        return new(data, nalt, nsim)
    end
end
function nalt(obs::PanelObsAsVector)
    return obs.nalt
end

function computeUtilities(x::Vector, obs::PanelObsAsVector, u::MixedLogitUtility, gamma::Vector)
    NP = Int(length(obs.data)/obs.nalt)
    utilities = Array{Float64, 2}(undef, nalt(obs), obs.nseq)
    for j in 1:obs.nseq
        tmp = (j - 1)*nalt(obs)
        for i in 1:nalt(obs)
            data = @view obs.data[tmp + (i-1)*NP+1:tmp+i*NP]
            utilities[i, j] = u.u(data, x, gamma, i)
        end
    end
    return utilities
end
        
function choice(obs::PanelObsAsVector)
    return 1
end
function nsim(obs::PanelObsAsVector)
    return obs.nsim
end
