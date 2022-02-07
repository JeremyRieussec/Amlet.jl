
# A function to compute the outer product of a vector.
function squareit(s::AbstractVector{T}) where T
    return s*s'
end

@doc raw"""
    computePrecomputedVal(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility)

Returns an array containing the probabilities associated to every alternative.
"""
# Ok checked
function computePrecomputedVal(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility)::Array{T, 1} where T
    # compute utility value for all alternative
    uti = computeUtilities(beta, obs, u)
    # re-write in the form utility - utility_max
    uti .-= maximum(uti)
    # compute exp(utilities)
    map!(exp, uti , uti)
    s = sum(uti)
    return uti / s # retrun probability in the form exp(uti)/sum(exp(uti))

end

@doc raw"""
    loglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility, precomputedValues::AbstractVector{T})

Returns the log-likelihood of the model on this observation.
"""
function loglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility, precomputedValues::AbstractVector{T})::T where T
    return log(precomputedValues[choice(obs)])  # / sum(precomputedValues)) -- the sum of the precomputed values is already 1
end

"""
    gradloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility, precomputedValues::AbstractVector{T})


"""
function gradloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility,
        precomputedValues::AbstractVector{T})::Vector{T} where T
    g = zeros(T, length(beta))
    g[:] += u.grad(obs.data, beta, choice(obs))
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * u.grad(obs.data, beta, k)

    end
    return g
end

function gradloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility) where T
    return gradloglogit(beta, obs, u, computePrecomputedVal(beta, obs, u))
end

## ------------- temporary

function gradproba(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility,
        precomputedValues::AbstractVector{T}, j::Int)::Vector{T} where T
    g = zeros(T, length(beta))
    g[:] += u.grad(obs.data, beta, j)
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * u.grad(obs.data, beta, k)
    end
    return precomputedValues[j]*g
end

# Check ok with ForwardDiff
function gradproba(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility, j::Int)::Vector{T} where T
    return gradproba(beta, obs, u, computePrecomputedVal(beta, obs, u), j)
end

# Ok, checked with ForwardDiff
function my_gradloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility,
        precomputedValues::AbstractVector{T})::Vector{T} where T
    return (1/precomputedValues[choice(obs)])*gradproba(beta, obs, u, precomputedValues , choice(obs))
end

# Ok, checked with ForwardDiff
function my_gradloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility)::Vector{T} where T
    return my_gradloglogit(beta, obs, u, computePrecomputedVal(beta, obs, u))
end

## End temp


# Checked
function Hessianloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{NotLinear},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where T
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)
    H[:, :] += u.H(obs.data, beta, choice(obs))

    gradS = sum(precomputedValues[k] * u.grad(obs.data, beta, k) for k in 1:nalt)
    H[:, :]  += gradS * gradS'
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * squareit(u.grad(obs.data, beta, k))
        H[:, :] -= precomputedValues[k] * u.H(obs.data, beta, k)
    end
    return H
end

# Ok, checked with ForwardDiff
function Hessianloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{Linear},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where T
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)

    gradS = sum(precomputedValues[k] * u.grad(obs.data, beta, k) for k in 1:nalt)
    H[:, :]  += squareit(gradS)
    for k in 1:nalt
        H[:, :] -= precomputedValues[k] * squareit(u.grad(obs.data, beta, k))
    end
    return H
end

# Ok, checked with ForwardDiff
function my_Hessianloglogit(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{Linear},
        precomputedValues::AbstractVector{T})::AbstractMatrix{T} where T
    dim = length(beta)
    nalt = length(precomputedValues)
    H = zeros(T, dim, dim)

    for k in 1:nalt
        H[:, :] -=  u.grad(obs.data, beta, k) * gradproba(beta, obs, u, precomputedValues, k)'
    end
    return H
end

# Checked
function Hessianloglogit_dot_v(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{NotLinear},
        precomputedValues::AbstractVector{T}, v::AbstractVector)::AbstractVector{T} where T
    dim = length(beta)
    nalt = length(precomputedValues)
    Hv = Array{T, 1}(undef, dim)
    #todo would be more efficient to add the function "u.H_dot_v(obs.data, x, choice(obs), v)"
    #but running out of time
    Hv[:] = u.H(obs.data, beta, choice(obs)) * v

    gradS = sum(precomputedValues[k] * u.grad(obs.data, beta, k) for k in 1:nalt)
    Hv[:]  += gradS * dot(gradS, v)
    for k in 1:nalt
        gk = u.grad(obs.data, beta, k)
        Hv[:] -= precomputedValues[k] * gk * dot(gk, v)
        #see previous todo
        Hv[:, :] -= precomputedValues[k] * u.H(obs.data, beta, k)*v
    end
    return Hv
end

# Ok, checked with ForwardDiff
function Hessianloglogit_dot_v(beta::AbstractVector{T}, obs::AbstractObs, u::LogitUtility{Linear},
        precomputedValues::AbstractVector{T}, v::AbstractVector)::AbstractVector{T} where T
    dim = length(beta)
    nalt = length(precomputedValues)
    Hv = Array{T, 1}(undef, dim)

    gradS = sum(precomputedValues[k] * u.grad(obs.data, beta, k) for k in 1:nalt)
    Hv[:]  = gradS * dot(gradS, v)
    for k in 1:nalt
        gk = u.grad(obs.data, beta, k)
        Hv[:] -= precomputedValues[k] * gk * dot(gk, v)
    end
    return Hv
end
