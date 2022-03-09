"""
     LogitModel

## Fields
- `u::AbstractUtility`
- `data::D`
- `se::StorageEngine`
"""
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
        T = Float64
        upd && return new{UPD, D}(LinUti, data, StorageEngine(data, T))
        return new{UPD, D}(LinUti, data)
    end
    function LogitModel{isUPD}(data::D) where {D <: AbstractData, isUPD <: IsUpdatable}
        T = Float64
        upd = (isUPD == Updatable)
        upd && return new{isUPD, D}(LinUti, data, StorageEngine(data, T))
        return new{isUPD, D}(LinUti, data)
    end
end