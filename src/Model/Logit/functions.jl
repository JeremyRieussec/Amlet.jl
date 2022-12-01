
"""
    obj(mo, beta; sample , update)

Function to compute the averaged negative log-likelihood of the Logit model over a sampled population.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit model where data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
- `update::Bool = false` is for use of storage Engine.
"""
function PM.obj(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data), update::Bool = false) where {T, UPD, D, L, UTI}
    update && update!(mo.se, beta, sample, mo)
    ac = zero(T)
    nind = 0
    UPD == Updatable && @assert ((mo.se.beta == beta) && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        ac += ns*logit(UTI, mo.data[i], beta, precomputedValues = cv)
        nind += ns
    end
    return -ac/nind
end

"""
    Fs!(mo, beta, ac; sample , inplace)

Computation of the negative log-likelihoods for all individuals in the selected population.

Modifies the `ac` array in place.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full
- `inplace::Bool = false`
"""
function PM.objs!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data), inplace::Bool = false) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for (index, i) in enumerate(sample)
        # ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        ll = logit(UTI, mo.data[i], beta, precomputedValues = cv)
        inplace ? ac[i] = -ll : ac[index] = -ll
    end
    return ac
end

"""
    Fs(mo, beta; sample)

Computation of the negative log-likelihoods for all individuals in the selected population.

Returns an array comtaining all values.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.objs(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    ac = Array{T, 1}(undef, length(sample))
    objs!(mo, beta, ac; sample = sample, inplace = false)
    return ac
end

"""
    grad!(mo, beta, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.grad!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        ac[:] += ns*gradlogit(UTI, mo.data[i], beta, precomputedValues = cv)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

"""
    grad(mo, beta; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.grad(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
        ac = Array{T, 1}(undef, length(beta))
        PM.grad!(mo, beta, ac, sample = sample)
        return ac
end

"""
    grads!(mo, beta, ac; sample, inplace)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::AbstractArray{T, 2}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
- `inplace::Bool = true`
"""
function PM.grads!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::AbstractArray{T, 2};
        sample = 1:length(mo.data), inplace::Bool = true) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for (index, i) in enumerate(sample)
        # ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        gll = gradlogit(UTI, mo.data[i], beta, precomputedValues = cv)
        inplace ? ac[:, i] = gll : ac[:, index] = gll
    end
    ac[:, :] .*= -1
    return ac
end

"""
    grads!(mo, beta; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.grads(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
    sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    dim = length(beta)
    ac = Array{T, 2}(undef, dim, length(sample))
    grads!(mo, beta, ac, sample = sample, inplace = false)
    return ac
end

"""
    H!(mo, beta, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array{T, 2}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.hess!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        ac[:, :] += ns*Hessianlogit(UTI, mo.data[i], beta, precomputedValues = cv)
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end
function PM.hess(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0) where {T, UPD, D, L, UTI}
    n = dim(UTI, mo.data)
    ac = zeros(T, n, n)
    hess!(mo, beta, ac, sample = sample, obj_weight = obj_weight)
end
"""
    Hdotv!(mo, beta, v, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `v::AbstractVector`
- `ac::Array{T, 1}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.hprod!(mo::LogitModel{UPD, D, L, UTI}, beta::AbstractVector{T}, v::AbstractVector, ac::Array{T, 1};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    #@show size(ac)
    #@show size(v)
    #@show size(beta)
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        ac[:] += ns*Hessianlogitdotv(UTI, mo.data[i], beta, v, precomputedValues = cv)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

"""
    Hdotv(mo, beta, v, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `v::AbstractVector`
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.hprod(mo::LogitModel{UPD, D, L, UTI}, beta::AbstractVector{T}, v::AbstractVector;
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0) where {T, UPD, D, L, UTI}
    ac = zeros(T, length(beta))
    hprod!(mo, beta, v, ac, sample = sample, obj_weight = obj_weight)
    return ac
end

"""
    BHHH!(mo, beta, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array{T, 2}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function PM.bhhh!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    # dim = length(beta)
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        
        gll = gradlogit(UTI, mo.data[i], beta, precomputedValues = cv)
        ac[:, :] += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= nind
    return ac
end
function PM.bhhh(mo::LogitModel, beta::Vector{T};
        sample = 1:length(mo.data)) where T
    ac = Array{T, 2}(undef, length(beta), length(beta))
    bhhh!(mo, beta, ac, sample = sample)
    return ac
end
function PM.bhhhprod!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 1}, v::Vector{T};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    # dim = length(beta)
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(UTI, mo.data[i], beta) : @view mo.se.cv[:, i]
        
        gll = gradlogit(UTI, mo.data[i], beta, precomputedValues = cv)
        ac[:] += ns * dot(gll, v) * gll
        nind += ns
    end
    ac[:] ./= nind
    return ac
end
function PM.bhhhprod(mo::LogitModel, beta::Vector{T}, v::Vector{T};
        sample = 1:length(mo.data)) where T
    ac = Array{T, 1}(undef, length(beta))
    bhhhprod!(mo, beta, ac, v, sample = sample)
    return ac
end
function getchoice(mo::LogitModel{U, D, L, UTI}, beta::Vector; sample = 1:length(mo.data)) where {U, D, L, UTI}
    choices = [argmax(computeUtilities(UTI, mo.data[i], beta)) for i in sample]
end
function ratiorightchoice(mo::LogitModel{U, D, L, UTI}, beta::Vector; sample = 1:length(mo.data)) where {U, D, L, UTI}
    choicesmodel = getchoice(mo, beta; sample = sample)
    truechoice = choice.(mo.data)
    return count(iszero, choicesmodel - truechoice)
end