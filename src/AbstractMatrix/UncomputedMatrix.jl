mutable struct UncomputedMatrix <: AbstractMatrix{Float64}
    f::Function
    function UncomputedMatrix(f::Function = identity)
        return new(f)
    end
end

function *(m::UncomputedMatrix, b::Vector)
    return m.f(b)
end
function Base.size(m::UncomputedMatrix)
    return (1,1)
end
function getindex(m::UncomputedMatrix, args...)
    UndefInitializer()
end