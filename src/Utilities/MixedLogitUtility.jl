
abstract type AbstractMixedLogitUtility{L, GT} <: AbstractUtility{L} end
function computeUtilities(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT) where {L, GT, TYPEU <: AbstractMixedLogitUtility{L, GT}}
    return [u(TYPEU, obs, theta, gamma, i) for i in 1:nalt(obs)]
end
function PM.grad(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Int) where {L, GT, TYPEU <: AbstractMixedLogitUtility{L, GT}}
    return ForwardDiff.gradient(t -> u(TYPEU, obs, t, gamma, i), theta)
end
function PM.hess(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Int) where {L, GT, TYPEU <: AbstractMixedLogitUtility{L, GT}}
    @warn "Hessian of linear utility called"
    lt = length(theta)
    return zeros(Float64, lt, lt)
end
function PM.hess(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Int) where {GT, TYPEU <: AbstractMixedLogitUtility{NotLinear, GT}}
    return ForwardDiff.hessian(t -> u(TYPEU, obs, t, gamma, i), theta)
end


abstract type AbstractLinearParametricMixedLogitUtility{Distro, f, GT} <: AbstractMixedLogitUtility{Linear, GT} end
const ALPMLU{Distro, f, GT} = AbstractLinearParametricMixedLogitUtility{Distro, f, GT}
function linearutilityinmixedlogit(obs::AbstractObs, beta::Vector, i::Int)
    xi = explanatory(obs, i)
    return dot(xi, beta)
end
function u(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Int) where {Distro, f, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    beta = f(theta, gamma)
    linearutilityinmixedlogit(obs, beta, i)
end
function PM.hess(::Type{TYPEU}, obs::AbstractObs, theta::GT, gamma::GT, i::Int) where {Distro, f, T, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    @warn "Hessian of linear utility called"
    lt = length(theta)
    return zeros(Float64, lt, lt)
end
function computeUtilities(::Type{TYPEU}, obs::AbstractObs, theta::GT, gamma::GT) where {Distro, f, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    beta = f(theta, gamma::GT)
    return [linearutilityinmixedlogit(obs, beta, i) for i in 1:nalt(obs)]
end
function getgamma(::Type{TYPEU}, rng::AbstractRNG, n::Int)::GT where {Distro, f, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    d = Distro(n)
    return rand(rng, d)
end




module ModuleNormalTasteUtility
using LinearAlgebra, Distributions, Random

const GT = Vector{Float64}
function fdiag(theta::Vector{Float64}, gamma::GT)
    n = length(gamma)
    mu = @view theta[1:n]
    sig = @view theta[n+1:end]
    return mu + (sig .* gamma)
end
function utilitygraddiagnormal(xi::V, theta::Vector{Float64}, gamma::GT) where {V <: AbstractVector{Float64}}
    return [xi; xi.*gamma]
end
function fuppertriangular(theta::Vector{Float64}, gamma::GT)
    n = length(gamma)
    #this is a copy, SHOULD NOT be a view!!!
    mu = theta[1:n]
    result = mu
    sig = @view theta[n+1:end]
    current = 1
    for i in 1:n
        v1 = @view sig[current:current+n-i] 
        v2 = @view gamma[i:end]
        result[i] += dot(v1, v2)
        current += n-i+1
    end
    return result
end
function utilitygraduppertrignormal(xi::V, theta::Vector{Float64}, gamma::GT) where {V <: AbstractVector{Float64}}
    g = similar(theta)
    n = length(gamma)
    g[1:n] = xi
    current = n+1
    currentlength = n
    for i in 1:n
        v = @view gamma[i:end]
        g[current:current + currentlength - 1] = xi[i] * v
        current += currentlength
        currentlength -= 1
    end
    return g
end

function mvnormal(n::Int)
    return MvNormal(zeros(n), Diagonal(ones(n)))
end
function getgammanormal(rng::AbstractRNG, n::Int)::GT
    return rand(rng, mvnormal(n))
end

end

const NormalDiagUtility = ALPMLU{MvNormal, ModuleNormalTasteUtility.fdiag}
const NDU = NormalDiagUtility
function PM.grad(::Type{NDU}, obs::AbstractObs, theta::Vector, gamma::ModuleNormalTasteUtility.GT, i::Int)
    lb = div(length(theta), 2)
    #a view is slower here.
    xi = explanatory(obs, i)
    return ModuleNormalTasteUtility.utilitygraddiagnormal(xi, theta, gamma)
end
function PM.hess(::Type{NDU}, obs::AbstractObs, theta::Vector, gamma::ModuleNormalTasteUtility.GT, i::Int)
    lt = length(theta)
    return zeros(Float64, lt, lt)
end

const NormalUpperTriangularUtility = ALPMLU{MvNormal, ModuleNormalTasteUtility.fuppertriangular}
const NUTU = NormalUpperTriangularUtility
function PM.grad(::Type{NUTU}, obs::AbstractObs, theta::Vector, gamma::ModuleNormalTasteUtility.GT, i::Int)
    m = length(theta)
    n = div(round(Int, -3 + sqrt(9 + 8*m)), 2)
    xi = explanatory(obs, i)
    return ModuleNormalTasteUtility.utilitygraduppertrignormal(xi, theta, gamma)
end


function getgamma(::Type{T}, rng::AbstractRNG, n::Int) where {T <: Union{NDU, NUTU}}
    return ModuleNormalTasteUtility.getgammanormal(rng, n)
end

function gammaDim(::Type{T}, obs::AbstractObs) where {T <: Union{NDU, NUTU}}
    return explanatorylength(obs)
end

function dim(::Type{NUTU}, s::AbstractData)::Int
    n = explanatorylength(s)
    return n + div(n*(n-1), 2)
end
function dim(::Type{NDU}, s::AbstractData)::Int
    n = explanatorylength(s)
    return 2 * n
end


