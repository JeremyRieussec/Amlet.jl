function Sofia.F(x::AbstractVector{T}, mo::LogitModel{Updatable, D}; sample = 1:length(mo.data), update::Bool = false) where {T, D}
    update && (update!(mo.se, x, sample, mo) ; return zero(T))
    @assert mo.se.x == x "storage engine not up to date"
    ac = zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        ac += ns*loglogit(x, mo.data[i], mo.u, @view tmp[:, i])
        nind += ns
    end
    return -ac/nind
end

function Sofia.grad!(x::AbstractVector{T}, mo::LogitModel{Updatable, D}, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        ac += ns*gradloglogit(x, mo.data[i], mo.u, @view tmp[:, i])
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.H!(x::AbstractVector{T}, mo::LogitModel{Updatable, D}, ac::Array{T, 2}; sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        ac += ns*Hessianloglogit(x, mo.data[i], mo.u, @view tmp[:, i])
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end

function Sofia.Hdotv!(x::AbstractVector{T}, mo::LogitModel{Updatable, D}, v::AbstractVector, ac::Array{T, 1}; 
        sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        vtmp = @view tmp[:, i]
        ac += ns*Hessianloglogit_dot_v(x, mo.data[i], mo.u, vtmp, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end



function Sofia.BHHH!(x::AbstractVector{T}, mo::LogitModel{Updatable, D}, ac::Array{T, 2}; sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    dim = length(x)
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        gll = gradloglogit(x, mo.data[i], mo.u, @view tmp[:, i])
        ac += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end

function Sofia.BHHHdotv!(x::AbstractVector{T}, mo::LogitModel{Updatable, D}, v::Vector, ac::Array{T, 1}; sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    dim = length(x)
    ac[:] = zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        gll = gradloglogit(x, mo.data[i], mo.u, @view tmp[:, i])
        ac += ns*gll * dot(gll, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.Fs(x::AbstractVector{T}, mo::LogitModel{Updatable, D}; sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    ac = Array{T, 1}(undef, length(sample))
    #weig = Array{Int, 1}(undef, length(sample))
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        #weig[index] = ns
        tmp = mo.se.cv
        ac[index] = ns*loglogit(x, mo.data[i], mo.u, @view tmp[:, i])
    end
    return -ac#, weig
end

function Sofia.grads!(x::AbstractVector{T}, mo::LogitModel{Updatable, D}, ac::AbstractArray{T, 2}; sample = 1:length(mo.data)) where {T, D}
    @assert mo.se.x == x "storage engine not up to date"
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        tmp = mo.se.cv
        ac[:, index] = gradloglogit(x, mo.data[i], mo.u, @view tmp[:, i])
    end
    return -ac
end
