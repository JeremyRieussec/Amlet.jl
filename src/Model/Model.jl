mutable struct LogitModel{U<:IsUpdatable, D <: AbstractData} <: AmletModel{U, D}
    u::AbstractUtility
    data::D
    se::StorageEngine
    function LogitModel(u::LogitUtility, data::D; T::Type = Float64, upd::Bool = false) where {D <: AbstractData}
        UPD = upd ? Updatable : NotUpdatable
        upd && return new{UPD, D}(u, data, StorageEngine(data, T))
        return new{UPD, D}(u, data)
    end
    function LogitModel(data::D; upd::Bool = false) where {D <: AbstractData}
        UPD = upd ? Updatable : NotUpdatable
        upd && return new{UPD, D}(LinUti, data, StorageEngine(data, T))
        return new{UPD, D}(LinUti, data)
    end
end
mutable struct MixedLogitModel{U<:IsUpdatable, D <: AbstractData} <: AmletModel{U, D}
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
copy(a::AmletModel) = typeof(a)([getfield(a, f) for f in fieldnames(typeof(a))]...)

