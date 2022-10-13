
@doc raw"""
`struct LinedObs`: All observation is assumed to be unique. All individual have the same number of alternatives ``p``.

# Fields
- `data::Array{Float64, 2}` is a matrix where each column represents an individual,
    where the attribute vectors associated to each choice are concatenated.
    Therefore,  the data matrix has ``J \times p`` rows and ``N`` columns which represents the number of individuals.
- `nalt::Int` is the number of alternatives ``J``
"""
struct LinedObs <: AbstractData
    data::Array{Float64, 2}
    nalt::Int
end

"""
    getindex(lI::LinedObs, index::Int)

Returns
"""
function getindex(lI::LinedObs, index::Int)
    data = @view lI.data[:, index]
    return ObsAsVector(data, lI.nalt)
end

"""
    length(l::LinedObs)

Returns munber of individuals.
"""
function length(l::LinedObs)
    return size(l.data, 2)
end
function nobs(l::LinedObs)
    return length(l)
end
function nalt(l::LinedObs)
    return l.nalt
end

function explanatorylength(l::LinedObs)
    n = size(l.data, 1)
    return div(n, l.nalt)
end

@doc raw"""
    dim(l::MatrixObs)

Returns the dimension of the attribute vectors, i.e. ``p``.
"""
function dim(l::LinedObs)
    n = size(l.data, 1)
    return div(n, l.nalt)
end
