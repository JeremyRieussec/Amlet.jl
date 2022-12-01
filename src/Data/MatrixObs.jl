
@doc raw"""
`struct MatrixObs`: All observation is assumed to be unique. All individual have the same number of alternatives ``J``.

# Fields
- `data::Array{Matrix, 1}` is an array of `Matrix`. Every matrix represents the observations
        for one individual, the dimensions are ``J \times p``, where ``p`` is the size of the attribute vectors.

- `nalt::Int` is the number of alternatives ``J``.
"""
struct MatrixObs <: AbstractData
    data::Array{Matrix, 1}
    nalt::Int
end

@doc raw"""
    getindex(lI::MatrixObs, index::Int)

Returns the matrix data for individual `index`. Dimension is ``J \times p``,
    where ``p`` is the size of the attribute vectors.
"""
function getindex(lI::MatrixObs, index::Int)
    return ObsAsMatrix(lI.data[index][:,:])
end

"""
    length(l::MatrixObs)

Returns number of individuals.
"""
function Base.length(l::MatrixObs)
    return size(l.data, 1)
end
function ENLPModels.nobs(l::MatrixObs)
    return length(l.data, 2)
end
"""
    nalt(l::MatrixObs)

Returns number of alternatives.
"""
function nalt(l::MatrixObs)
    return l.nalt
end

@doc raw"""
    dim(l::MatrixObs)

Returns the dimension of the attribute vectors, i.e. ``p``.
"""
function dim(l::MatrixObs)
    n = size(l.data, 1)
    return div(n, l.nalt)
end
