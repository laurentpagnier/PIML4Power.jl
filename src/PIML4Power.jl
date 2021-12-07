module PIML4Power

using SparseArrays
using BSON: @load, @save
using Flux
using LinearAlgebra
using HDF5

struct Matrices
    Bm::SparseMatrixCSC{Float64, Int64}
    Bp::SparseMatrixCSC{Float64, Int64}
    Bin::SparseMatrixCSC{Float64, Int64}
    Bout::SparseMatrixCSC{Float64, Int64}
    Bp_ns::SparseMatrixCSC{Float64, Int64}
    Bp_pq::SparseMatrixCSC{Float64, Int64}
    Bm_ns::SparseMatrixCSC{Float64, Int64}
    Bin_ns::SparseMatrixCSC{Float64, Int64}
    Bin_pq::SparseMatrixCSC{Float64, Int64}
    Bout_ns::SparseMatrixCSC{Float64, Int64}
    Bout_pq::SparseMatrixCSC{Float64, Int64}
    Bmt::SparseMatrixCSC{Float64, Int64}
    Bm_nst::SparseMatrixCSC{Float64, Int64}
    Bint::SparseMatrixCSC{Float64, Int64}
    Bin_nst::SparseMatrixCSC{Float64, Int64}
    Bin_pqt::SparseMatrixCSC{Float64, Int64}
    Boutt::SparseMatrixCSC{Float64, Int64}
    Bout_nst::SparseMatrixCSC{Float64, Int64}
    Bout_pqt::SparseMatrixCSC{Float64, Int64}
    pq2ns::SparseMatrixCSC{Float64, Int64}
    pq2nst::SparseMatrixCSC{Float64, Int64}
    pv2full::SparseMatrixCSC{Float64, Int64}
    s2full::SparseMatrixCSC{Float64, Int64}
    ns2full::SparseMatrixCSC{Float64, Int64}
    pq2full::SparseMatrixCSC{Float64, Int64}
    Ins::SparseMatrixCSC{Float64, Int64}
    Ipq::SparseMatrixCSC{Float64, Int64}
end


struct Indices
    slack::Int64
    pv::Vector{Int64}
    pq::Vector{Int64}
    ns::Vector{Int64}
    Nbus::Int64
    epsilon::Matrix{Int64}
end


mutable struct GridModel
    epsilon::Matrix{Int64}
    b::Vector{Float64}
    g::Vector{Float64}
    bsh::Vector{Float64}
    gsh::Vector{Float64}
end


struct Observation
    v::Matrix{Float64}
    th::Matrix{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
end


struct SystemData
    v::Matrix{Float64}
    th::Matrix{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
    gm::Matrix{Int64}
    b::Vector{Float64}
    g::Vector{Float64}
    bsh::Vector{Float64}
    gsh::Vector{Float64}
end

include("kron_reduction.jl")
include("mapping.jl")
include("newton_raphson.jl")
include("params.jl")
include("utils.jl")

end


