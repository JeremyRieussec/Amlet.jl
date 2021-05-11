#precomputedValues contains the probability associate to each alternatives.
function squareit(s::AbstractVector{T}) where T
    return s*s'
end

function computePrecomputedVal(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility)::Array{T, 1} where T
    uti = computeUtilities(x, obs, u)
    uti .-= maximum(uti)
    map!(exp, uti , uti)
    s = sum(uti)
    return uti / s

end

function loglogit(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility, precomputedValues::AbstractVector{T})::T where T
    return log(precomputedValues[choice(obs)] / sum(precomputedValues))

end

function gradloglogit(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility,
        precomputedValues::AbstractVector{T})::Vector{T} where T
    g = zeros(T, length(x))
    g[:] += u.grad(obs.data, x, choice(obs))
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * u.grad(obs.data, x, k)

    end
    return g
end

function gradloglogit(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility) where T
    return gradloglogit(x, obs, u, computePrecomputedVal(x, obs, u))
end

function Hessianloglogit(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{NotLinear},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where T
    dim = length(x)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)
    H[:, :] += u.H(obs.data, x, choice(obs))

    gradS = sum(precomputedValues[k] * u.grad(obs.data, x, k) for k in 1:nalt)
    H[:, :]  += gradS * gradS'
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * squareit(u.grad(obs.data, x, k))
        H[:, :] -= precomputedValues[k] * u.H(obs.data, x, k)
    end
    return H
end

function Hessianloglogit(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{Linear},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where T
    dim = length(x)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)

    gradS = sum(precomputedValues[k] * u.grad(obs.data, x, k) for k in 1:nalt)
    H[:, :]  += squareit(gradS)
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * squareit(u.grad(obs.data, x, k))
    end
    return H
end


function Hessianloglogit_dot_v(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{NotLinear},
        precomputedValues::AbstractVector{T}, v::AbstractVector)::AbstractVector{T} where T
    dim = length(x)
    nalt = length(precomputedValues)
    Hv = Array{T, 1}(undef, dim)
    #todo would be more efficient to add the function "u.H_dot_v(obs.data, x, choice(obs), v)"
    #but running out of time
    Hv[:] = u.H(obs.data, x, choice(obs)) * v

    gradS = sum(precomputedValues[k] * u.grad(obs.data, x, k) for k in 1:nalt)
    Hv[:]  += gradS * dot(gradS, v)
    for k in 1:nalt
        gk = u.grad(obs.data, x, k)
        Hv[:] -= precomputedValues[k] * gk * dot(gk, v)
        #see previous todo
        Hv[:, :] -= precomputedValues[k] * u.H(obs.data, x, k)*v
    end
    return Hv
end

function Hessianloglogit_dot_v(x::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{Linear},
        precomputedValues::AbstractVector{T}, v::AbstractVector)::AbstractVector{T} where T
    dim = length(x)
    nalt = length(precomputedValues)
    Hv = Array{T, 1}(undef, dim)

    gradS = sum(precomputedValues[k] * u.grad(obs.data, x, k) for k in 1:nalt)
    Hv[:]  = gradS * dot(gradS, v)
    for k in 1:nalt
        gk = u.grad(obs.data, x, k)
        Hv[:] -= precomputedValues[k] * gk * dot(gk, v)
    end
    return Hv
end
