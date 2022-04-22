const Rbase = 100
struct MixedLogitModel{D, L, UTI} <: AmletModel{NotUpdatable, D}
    data::D
    seeds::Array{Array{Int, 1}, 1}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    nobs::Int
    function MixedLogitModel(::Type{UTI}, data::D, rng::MRG32k3a = MRG32k3a()) where {D <: AbstractData, L, UTI <: AbstractMixedLogitUtility{L}}
        state0 = get_state(rng)
        seeds = Array{Int, 1}[]
        for _ in 1:length(data)
            push!(seeds, copy(rng.Bg))
            next_substream!(rng)
        end
        n = dim(UTI, data)
        meta = NLPModelMeta(n)
        counters = Counters()
        #println("model defined")
        return new{D, L, UTI}(data, seeds, meta, counters, nobs(data))
    end
end