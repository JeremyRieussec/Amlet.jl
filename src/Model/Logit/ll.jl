
"""
    obj(mo, beta; sample , update)

Function to compute the averaged negative log-likelihood of the Logit model over a sampled population.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit model where data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
- `update::Bool = false` is for use of storage Engine.
"""
function NLPModels.obj(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data), update::Bool = false) where {T, UPD, D, L, UTI}
    update && update!(mo.se, beta, sample, mo)
    ac = zero(T)
    nind = 0
    UPD == Updatable && @assert ((mo.se.beta == beta) && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        ac += ns*logit(beta, mo.data[i], UTI, cv)
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
function objs!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data), inplace::Bool = false) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for (index, i) in enumerate(sample)
        # ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        ll = logit(beta, mo.data[i], UTI, cv)
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
function objs(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    ac = Array{T, 1}(undef, length(sample))
    Fs!(mo, beta, ac; sample = sample, inplace = false)
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
function NLPModels.grad!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        ac[:] += ns*gradlogit(beta, mo.data[i], UTI, cv)
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
function NLPModels.grad(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
        ac = Array{T, 1}(undef, length(beta))
        NLPModels.grad!(mo, beta, ac, sample = sample)
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
function grads!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::AbstractArray{T, 2};
        sample = 1:length(mo.data), inplace::Bool = true) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for (index, i) in enumerate(sample)
        # ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        gll = gradlogit(beta, mo.data[i], UTI, cv)
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
function grads(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
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
function hess!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        ac[:, :] += ns*Hessianlogit(beta, mo.data[i], UTI, cv)
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end
function NLPModels.hess(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    ac = Array{Float64, 2}(undef, dim(mo.data), dim(mo.data))
    hess!(mo, beta, ac, sample = sample)
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
function NLPModels.hprod!(mo::LogitModel{UPD, D, L, UTI}, beta::AbstractVector{T}, v::AbstractVector, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        ac[:] += ns*Hessianlogit_dot_v(beta, mo.data[i], UTI, cv, v)
        nind += ns
    end# NP = Int(length(obs.data)/obs.nalt) --> this is not used
    # utilities = Array{Float64, 1}(undef, nalt(obs))
    # for i in 1:nalt(obs)
    #     utilities[i] = u.u(obs.data, beta, i)
    # end
    # return utilities
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
function NLPModels.hprod(mo::LogitModel{UPD, D, L, UTI}, beta::AbstractVector{T}, v::AbstractVector;
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    ac = Array{T, 1}(undef, length(beta))
    Hdotv!(mo, beta, v, ac, sample = sample)
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
function bhhh!(mo::LogitModel{UPD, D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data)) where {T, UPD, D, L, UTI}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    # dim = length(beta)
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        gll = gradlogit(beta, mo.data[i], UTI, cv)
        ac[:, :] += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= nind
    return ac
end

#=
"""
    BHHHdotv!(mo, beta, v, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector# NP = Int(length(obs.data)/obs.nalt) --> this is not used
# utilities = Array{Float64, 1}(undef, nalt(obs))
# for i in 1:nalt(obs)
#     utilities[i] = u.u(obs.data, beta, i)
# end
# return utilities
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    # dim = length(beta)
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], UTI) : @view mo.se.cv[:, i]
        gll = gradlogit(beta, mo.data[i], UTI, cv)
        ac[:] += ns*dot(gll, v)*gll
        nind += ns
    end
    ac[:] ./= nind
    return ac
end

"""
    BHHHdotv(mo, beta, v; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `v::Vector`
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function bhhhprod(beta::AbstractVector{T}, mo::LogitModel{UPD, D}, v::Vector;
        sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(beta)
    ac = zeros(T, dim)
    bhhhprod!(mo, beta, v, ac; sample = sample)
    return ac
end
=#