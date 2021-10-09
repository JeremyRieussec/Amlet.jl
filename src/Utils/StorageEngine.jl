struct StorageEngine{T}
    x::Vector{T}
    cv::Matrix{T}
    updatedInd::BitArray{1}
    function StorageEngine(data::AbstractData, T::Type = Float64)
        noa = nalt(data)
        d = dim(data)
        nind = length(data)
        v = T(1.0/noa)
        cv = Array{T, 2}(undef, noa, nind)
        fill!(cv, v)
        x = zeros(T, d)
        return new{T}(x, cv, trues(nind))
    end
end
