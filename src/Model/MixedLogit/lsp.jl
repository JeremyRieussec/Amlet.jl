function lsp(x::AbstractVector{T}, ind::AbstractInd, u::MixedLogitUtility, rng::AbstractRNG, R::Int = 100) where T
    return log(sp(x, ind, u, rng, R = R))
end

function gradlsp(x::AbstractVector{T}, ind::AbstractInd, u::MixedLogitUtility, rng::AbstractRNG, R::Int = 100) where T
    s = sp(x, ind, u, rng, R = R)
    return (1/s)*gradsp(x, ind, u, rng, R = R)
end

function Hlsp(x::AbstractVector{T}, ind::AbstractInd, u::MixedLogitUtility, rng::AbstractRNG, R::Int = 100) where T
    s = sp(x, ind, u, rng, R = R)
    gs = gradsp(x, ind, u, rng, R = R)
    return (s*Hsp(x, ind, u, rng, R = R) - gs*gs')/(s^2)
end

function ll(x::AbstractVector{T}, mo::Model; sample = 1:length(mo.data)) where T
    return sum(lsp(x, mo.data[i], u, mo.rng, mo.R) for i in sample)/length(sample)
end

function lls(x::AbstractVector{T}, mo::Model; sample = 1:length(mo.data)) where T
    return [lsp(x, mo.data[i], u, mo.rng, mo.R) for i in sample]
end

function gradll(x::AbstractVector{T}, mo::Model; sample = 1:length(mo.data)) where T
    return sum(gradlsp(x, mo.data[i], u, mo.rng, mo.R) for i in sample)/length(sample)
end
function Hll(x::AbstractVector{T}, mo::Model; sample = 1:length(mo.data)) where T
    return sum(Hlsp(x, mo.data[i], u, mo.rng, mo.R) for i in sample)/length(sample)
end