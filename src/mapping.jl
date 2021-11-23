export train_V2S_map!

#using Zygote

function preproc_V2S_map(
    th::Matrix{Float64},
    v::Matrix{Float64},
    mat::Matrices,
    id::Indices
)
    # some preprocessing for the V2S map that can be done once and for all
    dtheta = mat.Bmt * th
    Vij = (mat.Bint * v) .* (mat.Boutt * v)
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
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    opt;
    Ninter = 5::Int64,
    Nepoch = 10::Int64,
)
    Nbatch = size(v,2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    ps = params(beta, gamma, bsh, gsh)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            return sum(abs.(p-pref)) + sum(abs.(q-qref))
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            error = (sum(abs.(p-pref)) + sum(abs.(q-qref)))  / 2.0 /
                prod(size(pref))
            println([e error])
        end
    end  
    return nothing
end


function train_n_update_V2S_map!(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    epsilon::Matrix{Int64},
    mat::Matrices,
    id::Indices,
    opt;
    Ninter = 5::Int64,
    Nepoch = 10::Int64,
    beta_thres = -3.0::Float64,
)
    Nbatch = size(v,2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    ps = params(beta, gamma, bsh, gsh)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            return sum(abs.(p-pref)) + sum(abs.(q-qref))
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            # if beta is smaller than a threshold, one ass
            id_kept = beta .> beta_thres
            beta = b[id_kept]
            beta = beta[id_kept]
            epsilon = epsilon[id_kept,:]
            mat = create_incidence_matrices(epsilon, id)
            Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            error = (sum(abs.(p-pref)) + sum(abs.(q-qref)))  / 2.0 /
                prod(size(pref))
            println([e error sum(id_kept)])
            ps = params(beta, gamma, bsh, gsh)
        end
    end
    return nothing
end


function train_hybrid_V2S_map!(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    c::Flux.Chain;
    Niter = 3::Int64,
    Nepoch = 10::Int64,
)
    Nbatch = size(v,2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    ps = Params(c, beta, gamma, bsh, gsh)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = c([v;th])
            reg = sum(abs2, c[])
            return sum(abs.(p + x[1:id.Nbus,:] - pref)) +
                sum(abs.(q + x[id.Nbus+1:end,:] - qref))
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = c([v;th])
            error = (sum(abs.(p + x[1:id.Nbus,:] - pref)) +
                sum(abs.(q + x[id.Nbus+1:end,:] - qref))) / 2.0 /
                prod(size(pref))
            println([e error])
        end
    end  
    return nothing
end


function create_simple_full_nn(
    N::Int64
)
    return c = Chain(Dense(2*N,2*Ntanh), Dense(2*N,2*N, tanh),
        Dense(2*N,2*N,tanh), Dense(2*N,2*N))
end
