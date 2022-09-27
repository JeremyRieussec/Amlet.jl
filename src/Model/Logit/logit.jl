@doc raw"""
    computePrecomputedVal(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI})

Returns an array containing the probabilities associated to every alternative.
"""
function computePrecomputedVal(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T})::Array{T, 1} where {T, UTI}
    # compute utility value for all alternative
    uti = computeUtilities(UTI, obs, beta)
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
function logit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T}; precomputedValues::AbstractVector{T})::T where {T, UTI}
    return log(precomputedValues[choice(obs)])  # / sum(precomputedValues)) -- the sum of the precomputed values is already 1
end

"""
    gradlogit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI}, precomputedValues::AbstractVector{T})


"""
function gradlogit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T};
        precomputedValues::AbstractVector{T} = computePrecomputedVal(UTI, obs, beta))::Vector{T} where {T, UTI}
    n = length(beta)
    g = Array{T, 1}(undef, n)
    g[:] = grad(UTI, obs, beta, choice(obs))
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * grad(UTI, obs, beta, k)

    end
    return g
end

function Hessianlogit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T};
        precomputedValues::AbstractVector{T} = computePrecomputedVal(UTI, obs, beta))::Matrix{T} where {T, L, UTI <: AbstractLogitUtility{L}}
    
    hasnonlinearutility = (Int(L) == 1)
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)
    hasnonlinearutility && (H[:, :] = hess(UTI, obs, beta, choice(obs)))

    gradS = sum(precomputedValues[k] * grad(UTI, obs, beta, k) for k in 1:nalt)
    H[:, :]  += gradS * gradS'
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * grad(UTI, obs, beta, k) * grad(UTI, obs, beta, k)'
        hasnonlinearutility && (H[:, :] -= precomputedValues[k] * hess(UTI, obs, beta, k))
    end
    return H
end

function Hessianlogitdotv(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T}, v::Vector; 
        precomputedValues::AbstractVector{T} = computePrecomputedVal(UTI, obs, beta))::Vector{T} where {T, L, UTI <: AbstractLogitUtility{L}}
    #@show size(beta)
    #@show size(v)
    #@show size(precomputedValues)
    hasnonlinearutility = (Int(L) == 1)
    dim = length(beta)
    nalt = length(precomputedValues)
    Hv = zeros(T, dim)
    hasnonlinearutility && (H[:] = hessdotv(UTI, obs, beta, choice(obs), v))

    gradS = sum(precomputedValues[k] * grad(UTI, obs, beta, k) for k in 1:nalt)
    Hv[:]  += gradS * dot(gradS, v)
    for k in 1:nalt
        Hv[:] -= precomputedValues[k] * grad(UTI, obs, beta, k) * dot(grad(UTI, obs, beta, k), v)
        hasnonlinearutility && (Hv[:] -= precomputedValues[k] * hessdotv(UTI, obs, beta, k, v))
    end
    return Hv
end
