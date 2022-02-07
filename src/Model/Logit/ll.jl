
"""
    F(beta, mo; sample , update)

Function to compute the averaged negative log-likelihood of the Logit model over a sampled population.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit model where data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
- `update::Bool = false` is for use of storage Engine.
"""
function Sofia.F(beta::Vector{T}, mo::LogitModel{UPD, D};
        sample = 1:length(mo.data), update::Bool = false) where {T, UPD, D}
    update && update!(mo.se, beta, sample, mo)
    ac = zero(T)
    nind = 0
    UPD == Updatable && @assert ((mo.se.beta == beta) && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac += ns*loglogit(beta, mo.data[i], mo.u, cv)
        nind += ns
    end
    return -ac/nind
end


# Pourquoi avoir inplace = false ? Le ! ne signifie-t-il pas qu' on fait en place ?
# Pour le stockage en accumulateur, on devrait avoir le negative log likelihood
"""
    Fs!(beta, mo, ac; sample , inplace)

Computation of the negative log-likelihoods for all individuals in the selected population.

Modifies the `ac` array in place.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full
- `inplace::Bool = false`
"""
function Sofia.Fs!(beta::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 1};
        sample = 1:length(mo.data), inplace::Bool = false) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for (index, i) in enumerate(sample)
        # ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ll = loglogit(beta, mo.data[i], mo.u, cv)
        inplace ? ac[i] = -ll : ac[index] = -ll
    end
    return ac
end

"""
    Fs(beta, mo; sample)

Computation of the negative log-likelihoods for all individuals in the selected population.

Returns an array comtaining all values.

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.Fs(beta::Vector{T}, mo::LogitModel{UPD, D};
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = Array{T, 1}(undef, length(sample))
    Fs!(beta, mo, ac; sample = sample, inplace = false)
    return ac
end

"""
    grad!(beta, mo, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.grad!(beta::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:] += ns*gradloglogit(beta, mo.data[i], mo.u, cv)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

"""
    grad(beta, mo; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.grad(beta::Vector{T}, mo::LogitModel{UPD, D};
        sample = 1:length(mo.data)) where {T, UPD, D}
        ac = Array{T, 1}(undef, length(beta))
        Sofia.grad(beta, mo, ac, sample = sample)
        return ac
end

"""
    grads!(beta, mo, ac; sample, inplace)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::AbstractArray{T, 2}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
- `inplace::Bool = true`
"""
function Sofia.grads!(beta::Vector{T}, mo::LogitModel{UPD, D}, ac::AbstractArray{T, 2};
        sample = 1:length(mo.data), inplace::Bool = true) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    for (index, i) in enumerate(sample)
        # ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(beta, mo.data[i], mo.u, cv)
        inplace ? ac[:, i] = gll : ac[:, index] = gll
    end
    ac[:, :] .*= -1
    return ac
end

"""
    grads!(beta, mo; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.grads(beta::Vector{T}, mo::LogitModel{UPD, D};
    sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(beta)
    ac = Array{T, 2}(undef, dim, length(sample))
    grads!(beta, mo, ac, sample = sample, inplace = false)
    return ac
end

"""
    H!(beta, mo, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array{T, 2}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.H!(beta::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 2};
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:, :] += ns*Hessianloglogit(beta, mo.data[i], mo.u, cv)
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end

"""
    Hdotv!(beta, mo, v, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `v::AbstractVector`
- `ac::Array{T, 1}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.Hdotv!(beta::AbstractVector{T}, mo::LogitModel{UPD, D}, v::AbstractVector, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:] += ns*Hessianloglogit_dot_v(beta, mo.data[i], mo.u, cv, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

"""
    Hdotv(beta, mo, v, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `v::AbstractVector`
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.Hdotv(beta::AbstractVector{T}, mo::LogitModel{UPD, D}, v::AbstractVector;
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = Array{T, 1}(undef, length(beta))
    Hdotv!(beta, mo, v, ac, sample = sample)
    return ac
end

"""
    BHHH!(beta, mo, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array{T, 2}` is the accumulator vector to modify in place
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.BHHH!(beta::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 2};
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    # dim = length(beta)
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(beta, mo.data[i], mo.u, cv)
        ac[:, :] += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= nind
    return ac
end


"""
    BHHHdotv!(beta, mo, v, ac; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `ac::Array{T, 1}` is the accumulator vector to modify in place
- `v::Vector`
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.BHHHdotv!(beta::Vector{T}, mo::LogitModel{UPD, D}, v::Vector, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.beta == beta && all(mo.se.updatedInd[sample])) "Storage Engine not updated"
    # dim = length(beta)
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(beta, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(beta, mo.data[i], mo.u, cv)
        ac[:] += ns*dot(gll, v)*gll
        nind += ns
    end
    ac[:] ./= nind
    return ac
end

"""
    BHHHdotv(beta, mo, v; sample)

# Arguments
- `beta::Vector{T}` is the parameter vector
- `mo::LogitModel{UPD, D}` is the Logit modelwhere data is stored
- `v::Vector`
- `sample = 1:length(mo.data)` is the sampled population used for computation, by default full.
"""
function Sofia.BHHHdotv(beta::AbstractVector{T}, mo::LogitModel{UPD, D}, v::Vector;
        sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(beta)
    ac = zeros(T, dim)
    BHHHdotv!(beta, mo, v, ac; sample = sample)
    return ac
end
