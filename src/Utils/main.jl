include("StorageEngine.jl")
function access(n::Int, m::Int)
    return (m-1)*n+1:n*m
end