#precomputedValues contains the probability associate to each alternatives.
function squareit(s::AbstractVector{T}) where T
    return s*s'
end

function computePrecomputedVal(x::AbstractVector{T}, ind::AbstractInd, u::LogitUtility, gamma::Vector)::Array{T, 1} where T
    uti = computeUtilities(x, ind, u, gamma)
    uti .-= maximum(uti)
    map!(exp, uti , uti)
    s = sum(uti)
    return uti / s
    
end

function logit(x::AbstractVector{T}, ind::AbstractInd, 
        u::MixedLogitUtility, gamma::Vector, pv::AbstractVector = computePrecomputedVal(x, ind, u, gamma))
    return pv[choice(ind)]
end
function gradlogit(x::AbstractVector{T}, ind::AbstractInd, 
        u::MixedLogitUtility, gamma::Vector, pv::AbstractVector = computePrecomputedVal(x, ind, u, gamma))
    ch = choice(ind)
    s = sum(pv)
    g = Arrat{T, 1}(undef, length(x))
    g[:] = pv[ch] * s * u.grad(ind.data, x, ch)
    for (k, lk) in enumerate(pv)
        g[:] -= lk * pv[ch] * u.grad(ind.data, x, k, gamma)
    end
    g
end

function Hlogit(x::AbstractVector{T}, ind::AbstractInd, 
            u::MixedLogitUtility{Linear}, gamma::Vector, pv::AbstractVector = computePrecomputedVal(x, ind, u, gamma))
    nalt = length(pv)
    dim = length(x)
    ch = choice(ind)
    lgu = sum(pv[j] * u.grad(ind.data, x, j, gamma) for j in 1:nalt)
    ac = Array{T, 2}(undef, dim, dim)
    
    
    ac[:, :] += pv[ch] * squareit(u.grad(ind.data, x, ch, gamma) - lu)
    
    for j in 1:nalt
        ac[:, :] += pv[ch]*pv[j]*(u.grad(ind.data, x, j, gamma) - lu) *
             (u.grad(ind.data, x, ch, gamma) - u.grad(ind.data, x, j, gamma))
        
    end
    ac
end


function Hlogit(x::AbstractVector{T}, ind::AbstractInd, 
            u::MixedLogitUtility{NotLinear}, gamma::Vector, pv::AbstractVector = computePrecomputedVal(x, ind, u, gamma))
    nalt = length(pv)
    dim = length(x)
    ch = choice(ind)
    lgu = sum(pv[j] * u.grad(ind.data, x, j, gamma) for j in 1:nalt)
    ac = Array{T, 2}(undef, dim, dim)
    
    
    ac[:, :] += pv[ch] * squareit(u.grad(ind.data, x, ch, gamma) - lu)
    
    for j in 1:nalt
        ac[:, :] += pv[ch]*pv[j]*(u.grad(ind.data, x, j, gamma) - lu) *
             (u.grad(ind.data, x, ch, gamma) - u.grad(ind.data, x, j, gamma))
        
    end
    ac[:, :] += pv[ch] * sum(pv[j] * (u.H(ind.data, x, ch, gamma) - u.H(ind.data, x, j, gamma)) for j in 1:nalt)
    
    return ac
        
end
