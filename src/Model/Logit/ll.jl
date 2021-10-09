
function Sofia.F(x::Vector{T}, mo::LogitModel{UPD, D}; 
        sample = 1:length(mo.data), update::Bool = false) where {T, UPD, D}
    update && update!(mo.se, x, sample, mo)
    ac = zero(T)
    nind = 0
    UPD == Updatable && @assert ((mo.se.x == x) && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac += ns*loglogit(x, mo.data[i], mo.u, cv)
        nind += ns
    end
    return -ac/nind
end

function Sofia.Fs!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 1}; 
        sample = 1:length(mo.data), inplace::Bool = false) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ll = loglogit(x, mo.data[i], mo.u, cv)
        inplace ? ac[i] = ll : ac[index] = ll
    end
    return -ac
end
function Sofia.Fs(x::Vector{T}, mo::LogitModel{UPD, D}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = Array{T, 1}(undef, length(sample))
    Fs!(x, mo, ac; sample = sample, inplace = false)
    return ac
end

function Sofia.grad!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 1}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
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

function Sofia.grad(x::Vector{T}, mo::LogitModel{UPD, D}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
        ac = Array{T, 1}(undef, length(x))
        Sofia.grad(x, mo, ac, sample = sample)
        return ac
end

function Sofia.grads!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::AbstractArray{T, 2}; 
        sample = 1:length(mo.data), inplace::Bool = true) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
    for (index, i) in enumerate(sample)
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        gll = gradloglogit(x, mo.data[i], mo.u, cv)
        inplace ? ac[:, i] = gll : ac[:, index] = gll
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

function Sofia.H!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 2}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
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
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        cv = (UPD == NotUpdatable) ? computePrecomputedVal(x, mo.data[i], mo.u) : @view mo.se.cv[:, i]
        ac[:] += ns*Hessianloglogit_dot_v(x, mo.data[i], mo.u, cv, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end

function Sofia.Hdotv(x::AbstractVector{T}, mo::LogitModel{UPD, D}, v::AbstractVector;
        sample = 1:length(mo.data)) where {T, UPD, D}
    ac = Array{T, 1}(undef, length(x))
    Hdotv!(x, mo, v, ac, sample = sample)
    return ac
end


function Sofia.BHHH!(x::Vector{T}, mo::LogitModel{UPD, D}, ac::Array{T, 2}; 
        sample = 1:length(mo.data)) where {T, UPD, D}
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
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
    UPD == Updatable && @assert (mo.se.x == x && all(mo.se.updatedInd[sample])) "Storage Engine not updated" 
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




