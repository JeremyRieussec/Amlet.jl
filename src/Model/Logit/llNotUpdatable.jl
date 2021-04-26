
function Sofia.F(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}; sample = 1:length(mo.data)) where {T, D}
    ac = zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac += ns*loglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
        nind += ns
    end
    return -ac/nind
end

function Sofia.grad!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
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

function Sofia.H!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 2}; sample = 1:length(mo.data)) where {T, D}
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

function Sofia.Hdotv!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, v::AbstractVector, ac::Array{T, 1}; 
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



function Sofia.BHHH!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, ac::Array{T, 2}; sample = 1:length(mo.data)) where {T, D}
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

function Sofia.BHHHdotv!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, v::Vector, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
    dim = length(x)
    ac[:] = zero(T)
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


function Sofia.Fs(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}; sample = 1:length(mo.data)) where {T, D}
    ac = Array{T, 1}(undef, length(sample))
    #weig = Array{Int, 1}(undef, length(sample))
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        #weig[index] = ns
        ac[index] = ns*loglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
    end
    return -ac#, weig
end

function Sofia.grads!(x::AbstractVector{T}, mo::LogitModel{NotUpdatable, D}, ac::AbstractArray{T, 2}; sample = 1:length(mo.data)) where {T, D}
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        ac[:, index] = gradloglogit(x, mo.data[i], mo.u, computePrecomputedVal(x, mo.data[i], mo.u))
    end
    ac[:, :] .*= -1
    return ac
end
