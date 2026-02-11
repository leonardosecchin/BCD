using BCD
using Test
using LinearAlgebra
using SparseArrays

@testset "Simple test" begin
    struct DATA
        A::Vector{SparseMatrixCSC{Float64, Int64}}
        Axb::Vector{Float64}
        Axi::Vector{Vector{Float64}}
        Axtmp::Vector{Float64}
        B::Vector{SparseMatrixCSC{Float64, Int64}}
        b::Vector{Float64}
    end

    A = Symmetric([1.0 0.5 0.1; 0.0 2.0 0.0; 0.0 0.0 3.0])
    b = A * ones(3)

    p = 2.0

    # function to allocate and initialize data
    function data_initialize(x, bs)
        data = DATA(
            SparseMatrixCSC{Float64, Int64}[],
            zeros(length(b)),
            Vector{Float64}[],
            Vector{Float64}(undef, size(A,1)),
            SparseMatrixCSC{Float64, Int64}[],
            b
        )
        for i in 1:length(bs)
            push!(data.A, sparse(A[:,bs[i].idx]))
            push!(data.Axi, data.A[i] * x[bs[i].idx])
            push!(data.B, tril(transpose(data.A[i]) * data.A[i]))
            dropzeros!(data.B[end])
        end
        for j in 1:length(bs)
            data.Axb .+= data.Axi[j]
        end
        data.Axb .-= data.b
        return data
    end

    # f(x) = 1/p |Ax - b|_p^p
    function f(x, bs, i, data)
        @inbounds begin
            # compute Ai * xi
            data.Axtmp .= data.Axi[i]
            @views data.Axi[i] .= data.A[i] * x[bs[i].idx]
            # update A*x - b
            @. data.Axb += data.Axi[i] - data.Axtmp
        end
        return (1/p) * norm(data.Axb, p)^p
    end

    # partial gradient
    function g!(g, x, bs, i, data)
        @inbounds @views g[bs[i].idx] .= data.A[i]' * (abs.(data.Axb).^(p-1) .* sign.(data.Axb))
    end

    # B_i
    function B(x, bs, i, data)
        return (p-1)*transpose(data.A[i]) * spdiagm(min.(abs.(data.Axb).^(p-2), 10^3)) * data.A[i]
    end

    blocks = create_blocks(3, [1;2;3])
    output = bcd(blocks, f, g!, B, data_initialize; verbose = 0)
    @test output.status == 0
end
