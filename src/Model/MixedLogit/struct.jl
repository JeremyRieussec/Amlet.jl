mutable struct MixedLogitModel{U, D <: AbstractData} <: AmletModel{U, D}
    u::AbstractUtility
    data::D
    seeds::Array{Array{Int, 1}, 1}
    R::Int
    function MixedLogitModel(u::MixedLogitUtility, data::D, rng::MRG32k3a, R = 100) where {D <: AbstractData}
        seeds = Array{Int, 1}[]
        for _ in 1:length(data)
            push!(seeds, copy(rng.Bg))
            next_substream!(rng)
        end

        return new{NotUpdatable, D}(u, data, seeds, R)
    end
    function MixedLogitModel(u::MixedLogitUtility, data::D, seed0::Array{Int, 1}, R = 100) where {D <: AbstractData}
        rng = MRG32k3a(seed0)
        return MixedLogit{NotUpdatable, D}(u, data, rng0, R)
    end
end