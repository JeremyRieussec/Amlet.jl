using Test
using Amlet

@testset "Basics" begin
    @test greet() === nothing
end;

### Imprecise tests
# As calculations on floating-point values can be imprecise, you can perform approximate equality checks 
@testset "Imprecision test ----" begin
    @test 1 ≈ 0.999999  rtol=1e-5
end;

### Test sets
@testset "trigonometric identities" begin
    θ = 2/3*π
    @test sin(-θ) ≈ -sin(θ)
    @test cos(-θ) ≈ cos(θ)
    @test sin(2θ) ≈ 2*sin(θ)*cos(θ)
    @test cos(2θ) ≈ cos(θ)^2 - sin(θ)^2
end;