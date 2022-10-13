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
    nobs::Int
    function LogitModel{UTI}(data::D; T::Type = Float64, upd::Bool = false) where {D <: AbstractData, L, UTI <: AbstractLogitUtility{L}}
        UPD = upd ? Updatable : NotUpdatable
        model = upd ? new{UPD, D, L, UTI}(data, StorageEngine(data, T)) : new{UPD, D, L, UTI}(data)
        n = dim(UTI, data)
        model.meta = NLPModelMeta(n)
        model.counters = Counters()
        model.nobs = nobs(model)
        return model
    end
    function LogitModel(data::D; T::Type = Float64, upd::Bool = false) where {D <: AbstractData}
        UTI = StandardLogitUtility
        L = Linear
        UPD = upd ? Updatable : NotUpdatable
        model = upd ? new{UPD, D, L, UTI}(data, StorageEngine(data, T)) : new{UPD, D, L, UTI}(data)
        n = dim(UTI, data)
        model.meta = NLPModelMeta(n)
        model.counters = Counters()
        model.nobs = nobs(model)
        return model
    end
end

function ENLPModels.nobs(lm::LogitModel)
    return nobs(lm.data)
end

function dim(lm::LogitModel{U, D, L, UTI}) where {U, D, L, UTI}
    return dim(UTI, lm.data)
end

