function SP(x::AbstractVector{T}, ind::AbstractInd, u::MixedLogitUtility, rng::AbstractRNG, R::Int = 100) where T
    ac = zero(T)
    reset_substream!(rng)
    for r in 1:R
        gamma = rand(rng, u.distro)
        cpv[:] = computePrecomputedVal(x, ind, u, gamma)
        ac += logit(x, ind, u, gamma, pv)
    end
    return ac/R
end

function gradSP(x::AbstractVector{T}, ind::AbstractInd, u::MixedLogitUtility, rng::AbstractRNG, R::Int = 100) where T
    dim = length(x)
    ac = zeros(T, dim)
    nalt = length(cpv)
    reset_substream!(rng)
    for r in 1:R
        gamma = rand(rng, u.distro)
        ac[:] += gradlogit(x, ind, u, gamma)
    end
    ac ./= R
end

function HSP(x::AbstractVector{T}, ind::AbstractInd, u::MixedLogitUtility{Linear}, rng::AbstractRNG, R::Int = 100) where T
    dim = length(x)
    ac = Array{T, 2)(undef, dim, dim)
    nalt = length(cpv)
    reset_substream!(rng)
    for r in 1:R
        gamma = rand(rng, u.distro)
        ac[:, :] += Hlogit(x, ind, u, gamma)
    end
    ac[:, :] ./= R
end