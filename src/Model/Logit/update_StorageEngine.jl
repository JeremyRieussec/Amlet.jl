function update!(se::StorageEngine{T}, x::AbstractVector{T}, sampling::AbstractVector{Int}, mo::LogitModel) where T
    if se.x != x
        se.x = x
        se.updatedInd = Int[]
    end
    for i in sampling
        if !(i in se.updatedInd)
            push!(se.updatedInd, i)
            se.cv[:, i] = computePrecomputedVal(x, mo.data[i], mo.u)
        end
    end
    sort!(se.updatedInd)
    mo
end