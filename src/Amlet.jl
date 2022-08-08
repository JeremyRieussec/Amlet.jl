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

using ForwardDiff, LinearAlgebra, RDST, Random, ENLPModels, Distributions

const PM = ENLPModels
import Base.getindex
import Base.iterate
import Base.length
import Base.copy
import Base.*

include("Observation/main.jl")
include("Data/main.jl")
include("Utilities/main.jl")
include("Utils/main.jl")
include("Model/main.jl")
end # module
