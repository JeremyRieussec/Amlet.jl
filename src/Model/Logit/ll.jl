
function Sofia.F(x::Vector{T}, mo::LogitModel{UPD, D}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac += ns*loglogit(x, mo.data[i], mo.u, cv)
        nind += ns
    end
    return -ac/nind
end

function Sofia.grad!(x::AbstractVector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 1}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:] += ns*gradloglogit(x, mo.data[i], mo.u, cv)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.H!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 2}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:, :] += ns*Hessianloglogit(x, mo.data[i], mo.u, cv)
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end

function Sofia.Hdotv!(x::AbstractVector{T}, mo::LogitModel{UPD, D}, v::AbstractVector, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:] += ns*Hessianloglogit_dot_v(x, mo.data[i], mo.u, cv), v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.Hdotv(x::AbstractVector{T}, mo::LogitModel{UPD, D}, v::AbstractVector;
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = zeros(T, length(x))
    Hdotv!(x, mo, v, ac, sample = sample)
    return ac
end


function Sofia.BHHH!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 2}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(x)
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(x, mo.data[i], mo.u, cv)
        ac[:, :] += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= nind
    return ac
end

function Sofia.BHHHdotv!(x::Vector{T}, mo::LogitModel{UPD, D}, v::Vector, ac::Array{T, 1}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(x)
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(x, mo.data[i], mo.u, cv)
        ac[:] += ns*gll * dot(gll, v)
        nind += ns
    end
    ac[:] ./= nind
    return ac
end

function Sofia.BHHHdotv(x::AbstractVector{T}, mo::LogitModel{UPD, D}, v::Vector; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(x)
    ac = zeros(T, dim)
    BHHHdotv!(x, mo, v, ac; sample = sample)
    return ac
end


function Sofia.Fs!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 1}; 
        sample = 1:length(mo.data), inplace::Bool = false) where {T, UPD, D}
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ll = loglogit(x, mo.data[i], mo.u, cv)
        inplace ? ac[index] = ll : ac[i] = ll
    end
    return -ac
end
function Sofia.Fs(x::Vector{T}, mo::LogitModel{UPD, D}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = Array{T, 1}(undef, length(sample))
    Fs!(x, mo, ac; sample = sample, inplace = false)
    return ac
end
function Sofia.grads!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::AbstractArray{T, 2}; 
        sample = 1:length(mo.data), inplace::Bool = true) where {T, UPD, D}
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(x, mo.data[i], mo.u, cv)
        inplace ? ac[:, index] = gll : ac[:, i] = gll
    end
    ac[:, :] .*= -1
    return ac
end

function Sofia.grads(x::Vector{T}, mo::LogitModel{UPD, D}; 
    sample = 1:length(mo.data)) where {T, UPD, D}
    dim = length(x)
    ac = Array{T, 2}(undef, dim, length(sample))
    grads!(x, mo, ac, sample = sample, inplace = false)
    return ac
end
