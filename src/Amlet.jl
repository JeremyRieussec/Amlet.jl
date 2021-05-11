"""
Another Machine Learning Estimation Tool (`AMLET`).
"""
module Amlet

using Distributions, ForwardDiff, LinearAlgebra, RDST, Random, Statistics, Sofia, OnlineStats

import Base.getindex
import Base.iterate
import Base.length
import Base.copy
import Base.*

export AbstractUtility, AbstractData, AbstractObs, computeUtilities


include("Utilities/main.jl")
include("Observation/main.jl")
include("Data/main.jl")
include("Utils/main.jl")
include("Model/main.jl")
end # module
