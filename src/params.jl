# some ways to parametrize grid parmeters
# outputs should be
# b: line susceptances (a vector with an entry per line)
# g: line conductances (a vector with an entry per line)
# bsh: shunt susceptances (a vector with an entry per bus)
# gsh: shunt conductances (a vector with an entry per bus)


function trival_param(
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


function exp_param(
    p::NTuple{4, Vector{Float64}}
)
    b = -exp.(p[1])
    g = exp.(p[2])
    return b, g, p[3], p[4]
end
