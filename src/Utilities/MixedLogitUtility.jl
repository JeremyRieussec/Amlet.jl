
abstract type AbstractMixedLogitUtility{L} <: AbstractUtility{L} end

abstract type AbstractLinearParametricMixedLogitUtility{Distro, f} <: AbstractMixedLogitUtility{Linear} end
const ALPMLU{Distro, f} = AbstractLinearParametricMixedLogitUtility{Distro, f}
function u(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::Vector, i::Int) where {Distro, f, TYPEU <: ALPMLU{Distro, f}}
    #@show f
    beta = f(theta, gamma)
    xi = explanatory(obs, i)
    return dot(xi, beta)
end
function NLPModels.grad(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::Vector, i::Int) where {Distro, f, TYPEU <: ALPMLU{Distro, f}}
    xi = @view x[access(length(beta), i)]
    #language abuse, actually Jac
    return grad(f, theta, gamma)' * xi
end
function NLPModels.hess(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector{T}, gamma::Vector, i::Int) where {Distro, f, T, TYPEU <: ALPMLU{Distro, f}}
    @warn "Hessian of linear utility called"
    lt = length(theta)
    return zeros(Float64, lt, lt)
end
function getgamma(::Type{TYPEU}, rng::AbstractRNG, n::Int) where {Distro, f, TYPEU <: ALPMLU{Distro, f}}
    d = Distro(n)
    return rand(rng, d)
end
function computeUtilities(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::Vector) where {Distro, f, TYPEU <: ALPMLU{Distro, f}}
    beta = f(theta, gamma)
    return [dot(explanatory(obs, i), beta) for i in 1:nalt(obs)]
end
module ModuleNormalTasteUtility
using LinearAlgebra, Distributions, Random
function fdiag(theta::AbstractVector, gamma::Vector)
    n = length(gamma)
    mu = @view theta[1:n]
    sig = @view theta[n+1:end]
    return mu + (sig .* gamma)
end
function utilitygraddiagnormal(xi::AbstractVector, theta::Vector, gamma::AbstractVector)
    return [xi; xi.*gamma]
end
function fuppertriangular(theta::Vector, gamma::Vector)
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
function utilitygraduppertrignormal(xi::AbstractVector, theta::Vector, gamma::AbstractVector)
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
function getgammanormal(rng::AbstractRNG, n::Int)
    return rand(rng, mvnormal(n))
end

end

struct NormalDiagUtility <: ALPMLU{MvNormal, ModuleNormalTasteUtility.fdiag} end
const NDU = NormalDiagUtility
function NLPModels.grad(::Type{NDU}, obs::AbstractObs, theta::AbstractVector, gamma::Vector, i::Int)
    lb = div(length(theta), 2)
    #a view is slower here.
    xi = explanatory(obs, i)
    return ModuleNormalTasteUtility.utilitygraddiagnormal(xi, theta, gamma)
end
function NLPModels.hess(::Type{NDU}, obs::AbstractObs, theta::AbstractVector, gamma::Vector, i::Int)
    lt = length(theta)
    return zeros(Float64, lt, lt)
end

struct NormalUpperTriangularUtility <: ALPMLU{MvNormal, ModuleNormalTasteUtility.fuppertriangular} end
const NUTU = NormalUpperTriangularUtility
function NLPModels.grad(::Type{NUTU}, obs::AbstractObs, theta::AbstractVector, gamma::Vector, i::Int)
    m = length(theta)
    n = div(round(Int, -3 + sqrt(9 + 8*m)), 2)
    xi = explanatory(obs, i)
    return ModuleNormalTasteUtility.utilitygraduppertrignormal(xi, theta, gamma)
end


function  getgamma(::Type{T}, rng::AbstractRNG, n::Int) where {T <: Union{NDU, NUTU}}
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


