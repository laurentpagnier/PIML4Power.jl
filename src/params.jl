# some ways to parametrize grid parmeters
# outputs should be
# b: line susceptances (a vector with an entry per line)
# g: line conductances (a vector with an entry per line)
# bsh: shunt susceptances (a vector with an entry per bus)
# gsh: shunt conductances (a vector with an entry per bus)


function trivial_param(
    p::NTuple{4, Vector{Float64}}
)
    return p[1], p[2], p[3], p[4]
end


function rx_param(
    p::NTuple{4, Vector{Float64}}
)
    b = - p[2] ./ (p[1].^2 + p[2].^2)
    g = p[1] ./ (p[1].^2 + p[2].^2)
    return b, g, p[3], p[4]
end


function rx_phys_legit_param(
    p::NTuple{4, Vector{Float64}}
)
    #b = min.(-p[2] ./ (p[1].^2 + p[2].^2), -1E-3)
    #g = max.(p[1] ./ (p[1].^2 + p[2].^2), 1E-3)
    b = -abs.(p[2] ./ (p[1].^2 + p[2].^2))
    g = abs.(p[1] ./ (p[1].^2 + p[2].^2))
    return b, g, p[3], p[4]
end


function bg_phys_legit_param(
    p::NTuple{4, Vector{Float64}}
)
    b = -abs.(p[1])
    g = abs.(p[2])
    return b, g, p[3], p[4]
end


function exp_param(
    p::NTuple{4, Vector{Float64}}
)
    b = -exp.(p[1])
    g = exp.(p[2])
    return b, g, p[3], p[4]
end

# reduction param functions provide the rule that should be applied to
# parameters if the list of edges is reduced

function red_param!(
    p::Vector{Vector{Float64}},
    iskept::BitVector,
)
    p[1] = p[1][iskept]
    p[2] = p[2][iskept]
    return nothing
end

function convert_param_tuple_into_vector(p)
    v = Vector{Vector{Float64}}([])
    for i = 1:length(p)
        push!(v, p[i])
    end
    return v
end

function convert_param_vector_into_tuple(v)
    return Tuple(copy(v[i]) for i in 1:length(v))
end
