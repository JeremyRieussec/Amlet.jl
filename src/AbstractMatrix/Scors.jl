struct Scors <: AbstractMatrix{Float64}
    vectors::Matrix
    weights::Vector
    function Scors(sz::Tuple{Int, Int})
        vectors = zeros(sz...)
        weights = zeros(sz[1])
        return new(vectors, weights)
    end
end
function *(m::Scors, b::Vector)
    return sum(m.weights[k] * m.vectors[k, :] * (m.vectors[k, :]'*b) for k in 1:length(m.weights))/sum(m.weights)
end
function Base.size(m::Scors)
    return (1,1)
end
function getindex(m::Scors, args...)
    UndefInitializer()
end