abstract type AmletModel{U <: IsUpdatable, D <: AbstractData} <: AbstractStochasticModel{U} end
function Sofia.Nobs(mo::AmletModel)
    return length(mo.data)
end