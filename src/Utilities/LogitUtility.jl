struct LogitUtility{L <: isLinear} <: AbstractUtility{L}
    u::Function
    grad::Function
    H::Function
    function LogitUtility(u::Function, L::Type = NotLinear)
        function grad(x::Any, beta::AbstractVector, i::Int)
            return ForwardDiff.gradient(t -> u(x, t, i), beta)
        end
        function H(x::Any, beta, i::Int)
            return ForwardDiff.hessian(t -> u(x, t, i), x)
        end
        return new{L}(u, grad, H)
    end
    function LogitUtility(u::Function, grad::Function, hes::Function, L::Type = NotLinear)
        return new{L}(u, grad, hes)
    end
end

module LinearUtilityForLogitModelWithCodeWellEncapsulated
using LinearAlgebra
    #AbstractVector
    function access(n::Int, m::Int)
        return (m-1)*n+1:n*m
    end
    function u(x::AbstractVector, beta::AbstractVector, i::Int)
        return dot(x[access(length(beta), i)], beta)
    end
    function grad(x::AbstractVector, beta::AbstractVector, i::Int)
        return x[access(length(beta), i)]
    end
    function H(x::AbstractVector, beta::AbstractVector, i::Int)
        return Array{Float64, 2}(I, length(beta), length(beta))
    end

    #AbstractMatrix
    function u(x::AbstractMatrix, beta::AbstractVector, i::Int)
        return dot(beta, @view x[i, :])
    end
    function grad(x::AbstractMatrix, beta::AbstractVector, i::Int)
        return x[i, :]
    end
    function H(x::AbstractMatrix, beta::AbstractVector, i::Int)
        return Array{Float64, 2}(I, length(beta), length(beta))
    end
end
LUFLMWCWE = LinearUtilityForLogitModelWithCodeWellEncapsulated
LinUti = LogitUtility(LUFLMWCWE.u, LUFLMWCWE.grad, LUFLMWCWE.H, Linear)

