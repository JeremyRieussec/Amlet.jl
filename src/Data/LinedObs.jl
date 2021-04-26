
"""
`LineObs`: All observation is assumed to be unique. All individual have the same number of alternatives.

Data is stored in a `Matrix` where each row represent an observation.
"""
struct LinedObs <: AbstractData
    data::Array{Float64, 2}
    nalt::Int
end
function getindex(lI::LinedObs, index::Int)
    data = @view lI.data[:, index]
    return ObsAsVector(data, lI.nalt)
end
function length(l::LinedObs)
    return size(l.data, 2)
end
function nalt(l::LinedObs)
    return l.nalt
end
function dim(l::LinedObs)
    n = size(l.data, 1)
    return div(n, l.nalt)
end
