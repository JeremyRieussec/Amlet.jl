"""
     LogitModel

## Fields
- `u::AbstractUtility`
- `data::D`
- `se::StorageEngine`
"""
mutable struct LogitModel{U, D <: AbstractData, L, UTI <: AbstractLogitUtility{L}} <: AmletModel{U, D}
    data::D
    se::StorageEngine
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    function LogitModel{UTI}(data::D; T::Type = Float64, upd::Bool = false) where {D <: AbstractData, L, UTI <: AbstractLogitUtility{L}}
        UPD = upd ? Updatable : NotUpdatable
        model = upd ? new{UPD, D, L, UTI}(data, StorageEngine(data, T)) : new{UPD, D, L, UTI}(data)
        model.meta = NLPModelMeta(dim(data))
        model.counters = Counters()
        return model
    end
    function LogitModel(data::D; T::Type = Float64, upd::Bool = false) where {D <: AbstractData}
        UTI = StandardLogitUtility
        L = Linear
        UPD = upd ? Updatable : NotUpdatable
        model = upd ? new{UPD, D, L, UTI}(data, StorageEngine(data, T)) : new{UPD, D, L, UTI}(u, data)
        model.meta = NLPModelMeta(dim(data))
        model.counters = Counters()
        return model
    end
end