abstract type AbstractStorageEngine{T} end

"""
     StorageEngine{T}

## Fields
- `beta::Vector{T}` is the parameter vector.
- `cv::Matrix{T}` is a matrix where every column represent the alternative probabilities for individual ``j``.
- `updatedInd::BitArray{1}` to keep track of the probabilities up to date with parameter vector.

## Constructor(s)
- StorageEngine(data::AbstractData, T::Type = Float64)
"""
struct StorageEngine{T} <: AbstractStorageEngine{T}
    beta::Vector{T} # parameter vector
    cv::Matrix{T} # probability matrix
    updatedInd::BitArray{1} # array to identify which individual have been used or not

    function StorageEngine(data::AbstractData, T::Type = Float64)
        noa = nalt(data) # number of alternatives
        d = dim(data) # parameter vector dimension
        nind = length(data) # population size

        v = T(1.0/noa) # uniform probability because parameters initialized to 0
        cv = Array{T, 2}(undef, noa, nind)
        fill!(cv, v)

        beta = zeros(T, d)
        return new{T}(beta, cv, trues(nind))
    end
end
