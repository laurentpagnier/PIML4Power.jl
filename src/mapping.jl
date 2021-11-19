export train_V2S_map!

function preproc_V2S_map(
    th::Matrix{Float64},
    v::Matrix{Float64},
    mat::Matrices,
    id::Indices
)
    dtheta = mat.Bmt * th
    Vij = (mat.Bint * vref) .* (mat.Boutt * v)
    V2cos = Vij .* cos.(dtheta)
    V2sin = Vij .* sin.(dtheta)
    Vii = (v .* v)
    return Vij, V2cos, V2sin, Vii
end


function V2S_map(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    V2cos::Matrix{Float64},
    V2sin::Matrix{Float64},
    Vii::Matrix{Float64},
    mat::Matrices
)
    
    Gsh = mat.Bp * g + gsh
    Bsh = mat.Bp * b + bsh
    
    # compute power injections (ie p and q)
    p = - mat.Bp * (g .* V2cos) + Gsh .* Vii -
        mat.Bm * (b .* V2sin)
    q = - mat.Bm * (g .* V2sin) - Bsh .* Vii +
        mat.Bp * (b .* V2cos)
    return p, q
end


function train_V2S_map!(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Matrix{Float64},
    q::Matrix{Float64},
    vg::Matrix{Float64},
    thref::Matrix{Float64},
    vref::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices;
    Niter = 3::Int64,
    Nepoch = 10::Int64,
)
    Nbatch = size(vg,2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)

    for e in 1:Nepoch
        g = gradient((beta, gamma, bsh, gsh) do
            b = -exp.(beta)
            g = exp.(gamma)_ 
            V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            sum(abs.(p-pref)) + sum(abs.(q-qref))
        end
        Flux.update!(opt, beta, g[1])
        Flux.update!(opt, gamma, g[2])
        Flux.update!(opt, bsh, g[3])
        Flux.update!(opt, gsh, g[4])
        if(mod(e, 5) == 0)
            println([e])
        end
    end  
    return nothing
end
