
function squareit(s::Vector{T}) where T
    return s*s'
end

function computePrecomputedVal(::Type{UTI}, obs::AbstractObs, theta::Vector, gamma::Vector) where {UTI <: AbstractMixedLogitUtility}
    uti = computeUtilities(UTI, obs, theta, gamma)
    #@show uti
    uti .-= maximum(uti)
    map!(exp, uti , uti)
    s = sum(uti)
    #@show uti / s
    return uti / s
    
end


function L(::Type{UTI}, obs::AbstractObs, theta::Vector, 
        gamma::Vector; pv::Vector = computePrecomputedVal(UTI, obs, theta, gamma)) where {UTI <: AbstractMixedLogitUtility}
    #@show pv
    return pv[choice(obs)]
end
function gradL(::Type{UTI}, obs::AbstractObs, theta::Vector{T}, 
        gamma::Vector; pv::Vector = computePrecomputedVal(UTI, obs, theta, gamma)) where {T, UTI <: AbstractMixedLogitUtility}
    ch = choice(obs)
    s = sum(pv)
    g = pv[ch] * s * grad(UTI, obs, theta, gamma, ch)
    for (k, lk) in enumerate(pv)
        g[:] -= lk * pv[ch] * grad(UTI, obs, theta, gamma, k)
    end
    g
end
function HL(::Type{UTI}, obs::AbstractObs, theta::Vector{T}, 
            gamma::Vector; pv::Vector = computePrecomputedVal(UTI, obs, theta, gamma)) where {T, isLin, UTI <: AbstractMixedLogitUtility{isLin}}
    
    nalt = length(pv)
    dim = length(theta)
    ch = choice(obs)
    lgu = sum(pv[j] * grad(UTI, obs, theta, gamma, j) for j in 1:nalt)
    ac = pv[ch] * (grad(UTI, obs, theta, gamma, ch) - lgu) * (grad(UTI, obs, theta, gamma, ch) - lgu)'
    
    for j in 1:nalt
        ac[:, :] += pv[ch]*pv[j]*(grad(UTI, obs, theta, gamma, j) - lgu) *
             (grad(UTI, obs, theta, gamma, ch) - grad(UTI, obs, theta, gamma, j))'
        
    end
    (isLin == NotLinear) && (ac[:, :] += pv[ch] * sum(pv[j] * (hess(UTI, obs, theta, gamma, ch) - hess(UTI, obs, theta, gamma, j)) for j in 1:nalt))
    return ac
end


function SP(::Type{UTI}, obs::AbstractObs, theta::Vector{T}, rng::AbstractRNG; R::Int = Rbase) where {T,  UTI <: AbstractMixedLogitUtility}
    ac = zero(T)
    reset_substream!(rng)
    for r in 1:R
        n = gammaDim(UTI, obs)
        gamma = getgamma(UTI, rng, n)
        ac += L(UTI, obs, theta, gamma)
    end
    return ac/R
end
function gradSP(::Type{UTI}, obs::AbstractObs, theta::Vector{T}, rng::AbstractRNG; R::Int = Rbase) where {T, UTI <: AbstractMixedLogitUtility}
    dim = length(theta)
    ac = zeros(T, dim)
    reset_substream!(rng)
    for r in 1:R
        n = gammaDim(UTI, obs)
        gamma = getgamma(UTI, rng, n)
        ac[:] += gradL(UTI, obs, theta, gamma)
    end
    ac[:] ./= R
    return ac
end
function HSP(::Type{UTI}, obs::AbstractObs, theta::Vector{T}, rng::AbstractRNG; R::Int = Rbase) where {T, UTI <: AbstractMixedLogitUtility}
    dim = length(theta)
    ac = zeros(T, dim, dim)
    reset_substream!(rng)
    for r in 1:R
        n = gammaDim(UTI, obs)
        gamma = getgamma(UTI, rng, n)
        ac[:, :] += HL(UTI, obs, theta, gamma)
    end
    ac[:, :] ./= R
    return ac
end



function lsp(::Type{UTI}, obs::AbstractObs, theta::Vector, rng::AbstractRNG; R::Int = Rbase) where {UTI <: AbstractMixedLogitUtility}
    s = SP(UTI, obs, theta, rng, R = R)

    #s, gs, HS = computeallsp(UTI, obs, theta, rng, R = R, grad = false, hess = false)
    return log(s)
end
function gradlsp(::Type{UTI}, obs::AbstractObs, theta::Vector, rng::AbstractRNG; R::Int = Rbase) where {UTI <: AbstractMixedLogitUtility}
    s = SP(UTI, obs, theta, rng, R = R)
    gs = gradSP(UTI, obs, theta, rng, R = R)
    #s, gs, HS = computeallsp(UTI, obs, theta, rng, R = R, grad = true, hess = false)
    return (1/s)*gs
end
function Hlsp(::Type{UTI}, obs::AbstractObs, theta::Vector, rng::AbstractRNG; R::Int = Rbase) where {UTI <: AbstractMixedLogitUtility}
    s = SP(UTI, obs, theta, rng, R = R)
    gs = gradSP(UTI, obs, theta, rng, R = R)
    Hs = HSP(UTI, obs, theta, rng, R = R)

    #s, gs, HS = computeallsp(UTI, obs, theta, rng, R = R, grad = true, hess = true)
    return (s*Hs - gs*gs')/(s^2)
end




function ll(mo::MixedLogitModel{D, L, UTI}, theta::Vector{T} ; sample = 1:length(mo.data), R::Int = Rbase) where {D, L, UTI, T}
    ac = zero(T)
    total = 0
    for i in sample
        obs = mo.data[i]
        n = nsim(obs)
        total += n
        ac += n * lsp(UTI, mo.data[i], theta, MRG32k3a(mo.seeds[i]), R = R)
    end
    return ac/total
end
function lls(mo::MixedLogitModel{D, L, UTI}, theta::Vector; sample = 1:length(mo.data), R::Int = Rbase) where {D, L, UTI, T}
    return [lsp(UTI, mo.data[i], theta, MRG32k3a(mo.seeds[i]), R = R) for i in sample]
end
function gradll!(mo::MixedLogitModel{D, L, UTI}, theta::Vector, ac::Vector; sample = 1:length(mo.data), R::Int = Rbase) where {D, L, UTI}
    total = 0
    for i in sample
        obs = mo.data[i]
        n = nsim(obs)
        total += n
        ac[:] += n * gradlsp(UTI, mo.data[i], theta, MRG32k3a(mo.seeds[i]), R = R)
    end
    ac[:] ./= total
    return ac
end
function gradlls!(mo::MixedLogitModel{D, L, UTI}, theta::Vector, ac::Matrix; sample = 1:length(mo.data), R::Int = Rbase) where {D, L, UTI}
    for (index, i) in enumerate(sample)
        obs = mo.data[i]
        ac[index, :] = gradlsp(UTI, mo.data[i], theta, MRG32k3a(mo.seeds[i]), R = R)
    end
    return ac
end
function Hll!(mo::MixedLogitModel{D, L, UTI}, theta::Vector, ac::Matrix; sample = 1:length(mo.data), R::Int = Rbase) where {D, L, UTI}
    total = 0
    for i in sample
        obs = mo.data[i]
        n = nsim(obs)
        total += n
        ac[:, :] += n * Hlsp(UTI, mo.data[i], theta, MRG32k3a(mo.seeds[i]), R = R)
    end
    ac[:, :] ./= total
    return ac
end

#=
    total = 0
    for i in sample
        obs = mo.data[i]
        n = nsim(obs)
        total += n
        ac += n * Hlsp(UTI, mo.data[i], theta, MRG32k3a(mo.seeds[i]), R = R)
    end
    ac[:, :] ./= total
    return ac
end

=#