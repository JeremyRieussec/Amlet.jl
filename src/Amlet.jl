"""
Another Machine Learning Estimation Tool (`AMLET`).

A few steps to folow:
- create data structure and utility function
- create LogitModel
Then, you will have access to the functions: F, Fs, Fs!,
grad, grad!, grads, grads!,
H, H!, Hdotv, Hdotv!,
BHHH, BHHH!, BHHHdotv, BHHHdotv!.

For further details see: https://github.com/JeremyRieussec/Amlet.jl/docs/build

Good luck!!
"""
module Amlet

using Distributions, ForwardDiff, LinearAlgebra, RDST, Random, NLPModels

import Base.getindex
import Base.iterate
import Base.length
import Base.copy
import Base.*
#=
export AbstractUtility, AbstractData, AbstractObs, computeUtilities,
        F, Fs, Fs!,
        grad, grad!, grads, grads!,
        H, H!, Hdotv, Hdotv!,
        BHHH, BHHH!, BHHHdotv, BHHHdotv!
=#

include("Utilities/main.jl")
include("Observation/main.jl")
include("Data/main.jl")
include("Utils/main.jl")
include("Model/main.jl")
end # module
