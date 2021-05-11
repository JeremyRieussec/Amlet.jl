
function Sofia.F(x::Vector{T}, mo::LogitModel{NotUpdatable, D}; sample = 1:length(mo.data)) where {T, D}
    ac = zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac += ns*loglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        nind += ns
    end
    return -ac/nind
end

<<<<<<< HEAD

# true var
# function Sofia.F(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}; sample = 1:length(mo.data)) where {T, D}
#     #ac = zero(T)
#     #nind = 0
#     stats_data = Series(Mean(), Variance())
#     f_values = T[]
#     for i in sample
#         ns = nsim(mo.data[i])
#         val = -loglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
#         fit!(stats_data, ones(ns)*val)
#         #ac += ns*loglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
#         #nind += ns
#         push!(f_values, val)
#     end
#     value_f , var_f = OnlineStats.value(stats_data)
#     return value_f , var_f, f_values
# end



function Sofia.grad!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
=======
function Sofia.grad!(x::Vector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
>>>>>>> 31cb4407a07ecd6f50d3dcaa6fbd70cf94d5f7fc
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:] += ns*gradloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.H!(x::Vector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 2}; sample = 1:length(mo.data)) where {T, D}
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:, :] += ns*Hessianloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end

<<<<<<< HEAD
function Sofia.Hdotv!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, v::AbstractVector, ac::Array{T, 1};
=======
function Sofia.Hdotv!(x::Vector{T}, mo::LogitModel{NotUpdatable, D}, v::AbstractVector, ac::Array{T, 1}; 
>>>>>>> 31cb4407a07ecd6f50d3dcaa6fbd70cf94d5f7fc
        sample = 1:length(mo.data)) where {T, D}
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:] += ns*Hessianloglogit_dot_v(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u), v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.Hdotv(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, v::AbstractVector;
        sample = 1:length(mo.data)) where {T, D}
    ac = zeros(T, length(x))
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:] += ns*Hessianloglogit_dot_v(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u), v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end


function Sofia.BHHH!(x::Vector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 2}; sample = 1:length(mo.data)) where {T, D}
    dim = length(x)
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        gll = gradloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        ac[:, :] += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end

function Sofia.BHHHdotv!(x::Vector{T}, mo::LogitModel{NotUpdatable, D}, v::Vector, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
    dim = length(x)
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        gll = gradloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        ac[:] += ns*gll * dot(gll, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.BHHHdotv(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, v::Vector; sample = 1:length(mo.data)) where {T, D}
    dim = length(x)
    ac = zeros(T, dim)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        gll = gradloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        ac[:] += ns*gll * dot(gll, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end


function Sofia.Fs(x::Vector{T}, mo::LogitModel{NotUpdatable, D}; sample = 1:length(mo.data)) where {T, D}
    ac = Array{T, 1}(undef, length(sample))
    #weig = Array{Int, 1}(undef, length(sample))
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        #weig[index] = ns
        ac[index] = ns*loglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
    end
    return -ac#, weig
end

function Sofia.grads!(x::Vector{T}, mo::LogitModel{NotUpdatable, D}, ac::AbstractArray{T, 2}; sample = 1:length(mo.data)) where {T, D}
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        ac[:, index] = gradloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
    end
    ac[:, :] .*= -1
    return ac
end
