
struct MixedLogitUtility{L <: isLinear} <: AbstractUtility{L}
    u::Function
    grad::Function
    H::Function
    distro::Distribution
    function MixedLogitUtility(u::Function, distro::Distribution, L::Type = NotLinear)
        function grad(x::AbstractVector, theta::AbstractVector, i::Int, gamma::Vector)
            return ForwardDiff.gradient(t -> u(x, t, i, gamma), theta)
        end
        function H(x::AbstractVector, theta::AbstractVector, gamma::AbstractVector)
            return ForwardDiff.hessian(t -> u(x, t, i, gamma), theta)
        end
        return new{L}(u, grad, H, distro)
    end
    function MixedLogitUtility(u::Function, grad::Function, hes::Function, distro::Distribution, L::Type = NotLinear)
        return new{L}(u, grad, hes, distro)
    end
end
