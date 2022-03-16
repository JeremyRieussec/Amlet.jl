# A function to compute the outer product of a vector.
function squareit(s::AbstractVector{T}) where T
    return s*s'
end

@doc raw"""
    computePrecomputedVal(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI})

Returns an array containing the probabilities associated to every alternative.
"""
function computePrecomputedVal(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI})::Array{T, 1} where {T, UTI}
    # compute utility value for all alternative
    uti = computeUtilities(beta, obs, UTI)
    # re-write in the form utility - utility_max
    uti .-= maximum(uti)
    # compute exp(utilities)
    map!(exp, uti , uti)
    s = sum(uti)
    return uti / s # retrun probability in the form exp(uti)/sum(exp(uti))

end

@doc raw"""
    logit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI}, precomputedValues::AbstractVector{T})

Returns the log-likelihood of the model on this observation.
"""
function logit(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI}, precomputedValues::AbstractVector{T})::T where {T, UTI}
    return log(precomputedValues[choice(obs)])  # / sum(precomputedValues)) -- the sum of the precomputed values is already 1
end

"""
    gradlogit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI}, precomputedValues::AbstractVector{T})


"""
function gradlogit(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI},
        precomputedValues::AbstractVector{T})::Vector{T} where {T, UTI}
    g = zeros(T, length(beta))
    g[:] += grad(UTI, obs.data, beta, choice(obs))
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * grad(UTI, obs.data, beta, k)

    end
    return g
end

function gradlogit(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI}) where {T, UTI}
    return gradlogit(beta, obs, u, computePrecomputedVal(beta, obs, u))
end
#=
function gradproba(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI},
        precomputedValues::AbstractVector{T}, j::Int)::Vector{T} where T
    g = zeros(T, length(beta))
    g[:] += grad(UTI, obs.data, beta, j)
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * grad(UTI, obs.data, beta, k)
    end
    return precomputedValues[j]*g
end

function gradproba(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI}, j::Int)::Vector{T} where T
    return gradproba(beta, obs, u, computePrecomputedVal(beta, obs, u), j)
end

function my_gradlogit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI},
        precomputedValues::AbstractVector{T})::Vector{T} where T
    return (1/precomputedValues[choice(obs)])*gradproba(beta, obs, u, precomputedValues , choice(obs))
end

function my_gradlogit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI})::Vector{T} where T
    return my_gradlogit(beta, obs, u, computePrecomputedVal(beta, obs, u))
end
=#
function Hessianlogit(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where {T, UTI <: AbstractLogitUtility{NotLinear}}
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)
    H[:, :] += hess(UTI, obs.data, beta, choice(obs))

    gradS = sum(precomputedValues[k] * grad(UTI, obs.data, beta, k) for k in 1:nalt)
    H[:, :]  += gradS * gradS'
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * squareit(grad(UTI, obs.data, beta, k))
        H[:, :] -= precomputedValues[k] * hess(UTI, obs.data, beta, k)
    end
    return H
end
function Hessianlogit(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where {T, UTI <: AbstractLogitUtility{Linear}}
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)

    gradS = sum(precomputedValues[k] * grad(UTI, obs.data, beta, k) for k in 1:nalt)
    H[:, :]  += squareit(gradS)
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * squareit(grad(UTI, obs.data, beta, k))
    end
    return H
end
#=
function my_Hessianlogit(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where {T, UTI <: AbstractLogitUtility{Linear}}
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)

    for k in 1:nalt
        H[:, :] -=  grad(UTI, obs.data, beta, k) * gradproba(beta, obs, u, precomputedValues, k)'
    end
    return H
end
=#
function Hessianlogit_dot_v(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI},
        precomputedValues::AbstractVector{T}, v::AbstractVector)::AbstractVector{T} where {T, UTI <: AbstractLogitUtility{NotLinear}}
    dim = length(beta)
    nalt = length(precomputedValues)
    Hv = Array{T, 1}(undef, dim)
    #todo would be more efficient to add the function "u.H_dot_v(obs.data, x, choice(obs), v)"
    #but running out of time
    Hv[:] = hess(UTI, obs.data, beta, choice(obs)) * v

    gradS = sum(precomputedValues[k] * grad(UTI, obs.data, beta, k) for k in 1:nalt)
    Hv[:]  += gradS * dot(gradS, v)
    for k in 1:nalt
        gk = grad(UTI, obs.data, beta, k)
        Hv[:] -= precomputedValues[k] * gk * dot(gk, v)
        #see previous todo
        Hv[:, :] -= precomputedValues[k] * hess(UTI, obs.data, beta, k)*v
    end
    return Hv
end

function Hessianlogit_dot_v(beta::AbstractVector{T}, obs::AbstractObs, ::Type{UTI},
        precomputedValues::AbstractVector{T}, v::AbstractVector)::AbstractVector{T} where {T, UTI <: AbstractLogitUtility{Linear}}
    dim = length(beta)
    nalt = length(precomputedValues)
    Hv = Array{T, 1}(undef, dim)

    gradS = sum(precomputedValues[k] * grad(UTI, obs.data, beta, k) for k in 1:nalt)
    Hv[:]  = gradS * dot(gradS, v)
    for k in 1:nalt
        gk = grad(UTI, obs.data, beta, k)
        Hv[:] -= precomputedValues[k] * gk * dot(gk, v)
    end
    return Hv
end
