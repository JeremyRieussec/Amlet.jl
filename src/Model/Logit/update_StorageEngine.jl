
function update!(se::StorageEngine{T}, beta::AbstractVector{T}, sampling::AbstractVector{Int}, mo::LogitModel) where T
    #if we are at a new point, all computed values are wrong
    toupdate = falses(length(se.updatedInd))
    if se.beta != beta
        se.beta[:] = beta
        se.updatedInd[:] .= false
        toupdate[sampling] .= true
    #if we are at the same point as in the storage engine but some observation in the sample are not already updated
    elseif !all(se.updatedInd[sampling])
        #bitsample contains is a bitarray with a 1 if the observation is in the sample and a 0 else
        bitsample = falses(length(se.updatedInd))
        bitsample[sampling] .= true
        notuptodate = .! se.updatedInd
        toupdate[:] = notuptodate .& bitsample
    end
    #toupdateasInt contains the indexes of observation to update, isa vector of Int
    toupdateasInt = findall(toupdate)
    for i in toupdateasInt
        se.cv[:, i] = computePrecomputedVal(beta, mo.data[i], mo.u)
    end
    #tell the storage engine which observations are up to date
    se.updatedInd[sampling] .= true
end
