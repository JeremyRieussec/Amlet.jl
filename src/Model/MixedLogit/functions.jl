function NLPModels.obj(mo::MixedLogitModel, theta::Vector{T};
        sample = 1:length(mo.data), R::Int = Rbase)::T where T
    return -ll(mo, theta, sample = sample, R = R)
end

function objs!(mo::MixedLogitModel, theta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data), inplace::Bool = false, R::Int = Rbase)::Vector{T} where T
    vals = -lls(mo, theta, sample = sample, R = R)
    inplace ? (ac[sample] = vals) : (ac[:] = vals)
    return ac
end
function objs(mo::MixedLogitModel, theta::Vector{T};
        sample = 1:length(mo.data), R::Int = Rbase)::Vector{T} where T
    ac = Array{T, 1}(undef, length(sample))
    objs!(mo, theta, ac, sample = sample, inplace = false, R = R) #the minus is done in objs!
    return ac
end


function NLPModels.grad!(mo::MixedLogitModel, theta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data), R::Int = Rbase)::Vector{T} where T
    gradll!(mo, theta, ac, sample = sample, R = R)
    ac[:] *= -1
    return ac
end
function NLPModels.grad(mo::MixedLogitModel, theta::Vector{T};
        sample = 1:length(mo.data), R::Int = Rbase) ::Vector{T} where T
        n = length(theta)
        ac = Array{T, 1}(undef, n)
        gradll!(mo, theta, ac, sample = sample, R = R) #the minus is done in grad!
    return -ac
end
function grads!(mo::MixedLogitModel, theta::Vector{T}, ac::Matrix{T};
        sample = 1:length(mo.data), inplace::Bool = true, R::Int = Rbase)::Matrix{T} where T
    if !inplace
        gradlls!(mo, theta, ac, sample = sample, R = R)
        ac[sample, :] .*= -1
    else
        actmp = zeros(T, length(theta), length(sample))
        gradlls!(mo, theta, actmp, sample = sample, R = R)
        for (index, i) in enumerate(sample)
            ac[:, i] = -actmp[:, index]
        end
    end
    return ac
end
function grads(mo::MixedLogitModel, theta::Vector{T};
        sample = 1:length(mo.data), R::Int = Rbase)::Matrix{T} where T
    ac = zeros(T, length(theta), length(sample))
    gradlls!(mo, theta, ac, sample = sample, R = R) #the minus is done in grads!
    return ac
end


function hess!(mo::MixedLogitModel, theta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, R::Int = Rbase)::Matrix{T} where T
    Hll!(mo, theta, ac, sample = sample, R = R)
    ac[:, :] *= -1
    return ac
end
function NLPModels.hess(mo::MixedLogitModel, theta::Vector{T};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, R::Int = Rbase)::Matrix{T} where T
    tmp = length(theta)
    ac = zeros(T, tmp, tmp)
    Hll!(mo, theta, ac, sample = sample, R = R)#the minus is done in hess!
    return -ac
end


function NLPModels.hprod!(mo::MixedLogitModel, theta::AbstractVector{T}, v::AbstractVector, ac::Array{T, 1};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, R::Int = Rbase)::Vector{T} where T
    h = hess(mo, theta, sample = sample, R = R)
    ac[:] = -h*v
end
function NLPModels.hprod(mo::MixedLogitModel, theta::AbstractVector{T}, v::AbstractVector;
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, R::Int = Rbase)::Vector{T} where T
    ac = Array{T, 1}(undef, length(theta))
    return hprod!(mo, theta, v, ac, sample = sample, R = R)#the minus is done in hprod!
end