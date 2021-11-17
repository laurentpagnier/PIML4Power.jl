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
    return dtheta, Vij, V2cos, V2sin, Vii
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


function V2S(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    v::Vector{Float64},
    th::Vector{Float64},
    mat::Matrices,
    id::Indices
)
    # compute the missmatches for NR scheme
    dtheta = mat.Bmt * vec(th)
    costh = cos.(dtheta) 
    sinth = sin.(dtheta) 

    gs = g .* sinth
    gc = g .* costh
    bs = b .* sinth
    bc = b .* costh

    p = v .* ( (mat.Bin * ((-gc - bs) .* mat.Boutt) +
        mat.Bout * ((-gc + bs) .* mat.Bint)) * v  +
        (mat.Bp * g + gsh) .* v)
    
    q = v .* ( (mat.Bin * ((-gs + bc) .* mat.Boutt) +
        mat.Bout * ((gs + bc) .* mat.Bint)) * v -
        (mat.Bp * b + bsh) .* v)
    return p, q
end




function partial_pq_missmatch_loss(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    V2cos::Matrix{Float64},
    V2sin::Matrix{Float64},
    Vii::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64}
)
    p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii)
   
    
    p = 
    q = 
    return sum(abs.(p-pref)) + sum(abs.(q-qref)) + regularization
end


function training_partial_V2S(
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
    theta_slack::Vector{Float64},
    mat::Matrices,
    id::Indices;
    Niter = 3::Int64
)
    Nbatch = size(vg,2)
    grad = (zeros(length(beta)),zeros(length(beta)),zeros(length(gsh)),zeros(length(bsh)))
    
    theta, Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)

V2S_map(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    V2cos::Array{Float64,2},
    V2sin::Array{Float64,2},
    Vii::Array{Float64,2},
    mat::Matrices
)


    for batch in 1:Nbatch
        g = gradient((beta, gamma, bsh, gsh) -> loss(beta, gamma, bsh, gsh, p[:,batch],
        q[:,batch], vg[:,batch], thref[:,batch], vref[:,batch], pref[:,batch], qref[:,batch], th_slack[batch], mat, id, Niter = Niter), beta, gamma, bsh, gsh)

        grad[1] .+= g[1] / Nbatch
        grad[2] .+= g[2] / Nbatch
        grad[3] .+= g[3] / Nbatch
        grad[4] .+= g[4] / Nbatch
    end
    return grad
end
